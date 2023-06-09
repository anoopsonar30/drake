/*
 * Set up and solve a user-supplied trajectory optimization problem.
 *
 * To call cc-opt, the user must specify the following:
 *
 * 1. A list of agents (in `agents.json`), which for each agent includes
 *     - A URDF model of the agent specifying links, joints, and primitive collision geometry
 *       (optionally linking to visualization geometry, which may be 3D model files)
 *     - Any constraints on the agent's behavior, e.g.
 *         - start and end state constraints
 *         - a joint speed limit
 *         - a no-collision constraint with the environment
 *         - a no-collision constraint with other agents
 * 2. An exogenous state plan (in `exogenous.json`), which specifies
 *     - The static obstacle around which we are planning
 *     - A list of "fixed" agents, including URDFs and joint trajectories
 * 3. The total number of timesteps `NT` that MA-Chekhov should plan over, and the amount of
 *    real-world time that represents `T` (in seconds). For instance, if you wish to plan a
 *    trajectory over 10 seconds in 1-second increments, `T=10`, `NT=10`.
 *
 * MA-Chekhov will compute a locally-optimal trajectory and store its results in `plan.json`.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <chrono>

#include <gflags/gflags.h>

#include <nlohmann/json.hpp>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include <drake/geometry/scene_graph_inspector.h>
#include <drake/geometry/geometry_set.h>
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/weld_joint.h"
#include "drake/multibody/inverse_kinematics/minimum_distance_constraint.h"
#include "drake/multibody/inverse_kinematics/orientation_constraint.h"
#include "drake/multibody/inverse_kinematics/position_constraint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include <drake/systems/rendering/multibody_position_to_geometry_pose.h>
#include <drake/common/trajectories/piecewise_polynomial.h>
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/mathematical_program_result.h>
#include "drake/solvers/solve.h"
#include <drake/solvers/solver_options.h>
#include "drake/solvers/snopt_solver.h"

#include "chance_constraints.h"
#include "joint_chance_constraints.h"

namespace ccopt {
namespace runner {

#define MAX_TIME_STEP 1.0e-3

// Define command-line flags for users
DEFINE_double(min_distance, 0.0,
              "Deterministic collision buffer.");
DEFINE_string(agents_json_path, "./agents.json",
              "Path to agents.json, relative to the current directory.");
DEFINE_string(exogenous_json_path, "./exogenous.json",
              "Path to exogenous.json, relative to the current directory.");
DEFINE_string(save_plan_location, "./plan.json",
              "Specifies the location to save the results, relative to the current directory.");
DEFINE_bool(visualize, true,
            "If true, visualize the resulting trajectory.");


// For convenience
using json = nlohmann::json;
using drake::multibody::MultibodyPlant;


/*
 * Load all agents from the provided specification. These can either be conventional planning agents
 * (e.g. in agents.json under the key "agents") or they can be the dynamic obstacles, which we treat
 * as uncontrollable agents (these would be in exogenous.json under the key "dynamic obstacles").
 *
 * Modifies the agents_spec json object to include, for each agent, a "num_positions" field containing
 * the integer number of degrees of freedom for that agent.
 *
 * Returns a vector of drake::multibody::ModelInstanceIndex corresponding to each agent added to the plant
 */
std::vector<drake::multibody::ModelInstanceIndex> load_agents_into_plant(
    MultibodyPlant<double>& plant,
    drake::geometry::SceneGraph<double>& scene_graph,
    json agents_spec,
    std::string key)
{
    // While loading agents, we keep track of all agents that are not supposed to collide with other
    // agents (the "collides_with_agents" field is "false").
    std::vector<drake::geometry::GeometrySet> collision_free_agent_geometry_sets;

    // Also save the model instance indices for each agent that we add
    std::vector<drake::multibody::ModelInstanceIndex> model_indices;

    // Loop through the agents in the specification and load each one
    for (auto agent : agents_spec.at(key)) {
        // Get agent name and the path to its URDF
        std::string agent_name = agent.at("name");
        std::string agent_urdf_path = agent.at("urdf");

        // Load the agent into the plant
        drake::multibody::ModelInstanceIndex agent_index = drake::multibody::Parser(&plant).AddModelFromFile(
            agent_urdf_path, agent_name);
        model_indices.push_back(agent_index);

        // Weld the agent base link to the scene
        const auto& agent_root = plant.GetBodyByName(agent.at("root_link_name"), agent_index);
        drake::math::RigidTransform agent_xform = drake::math::RigidTransform(drake::Isometry3<double>::Identity());
        drake::Vector3<double> agent_offset;
        agent_offset << agent.at("root_link_position").at(0),
                        agent.at("root_link_position").at(1),
                        agent.at("root_link_position").at(2);
        agent_xform.set_translation(agent_offset);
        double agent_orientation_angle = agent.at("root_link_orientation").at(3);
        drake::Vector3<double> agent_orientation_axis;
        agent_orientation_axis << agent.at("root_link_orientation").at(0),
                                  agent.at("root_link_orientation").at(1),
                                  agent.at("root_link_orientation").at(2);
        agent_xform.set_rotation(Eigen::AngleAxisd(agent_orientation_angle, agent_orientation_axis));

        plant.AddJoint<drake::multibody::WeldJoint>("weld_base_" + agent_name, plant.world_body(), {},
                                                    agent_root, {},
                                                    agent_xform);

        // If this agent is not supposed to collide with other agents, save its geometry
        if (!agent.value("collides_with_agents", false)) {  // if field is not specified, assume we can ignore collision
            // Get all the bodies associated with this agent
            std::vector<drake::multibody::BodyIndex> agent_body_indices = plant.GetBodyIndices(agent_index);
            // Track down all the geometry ids associated with the agent and save them in a GeometrySet
            drake::geometry::GeometrySet agent_geometry_ids;
            for (drake::multibody::BodyIndex body_index : agent_body_indices) {
                const std::vector<drake::geometry::GeometryId> body_geometry_ids =
                    plant.GetCollisionGeometriesForBody(plant.get_body(body_index));
                agent_geometry_ids.Add(body_geometry_ids);
            }

            collision_free_agent_geometry_sets.push_back(agent_geometry_ids);
        }
    }

    // Now we've loaded all the agents, but we need to tell the scene graph not to track collisions
    // between pairs of geometry
    int num_excluded_pairs = collision_free_agent_geometry_sets.size();
    for (int i = 0; i < num_excluded_pairs; i++) {
        drake::geometry::GeometrySet setA = collision_free_agent_geometry_sets[i];

        for (int j = 0; j < num_excluded_pairs; j++) {
            // Don't exclude a pair and itself
            if ( i == j) { continue; }

            drake::geometry::GeometrySet setB = collision_free_agent_geometry_sets[j];
            scene_graph.ExcludeCollisionsBetween(setA, setB);
        }
    }

    return model_indices;
}


/*
 * Load all static obstacle geometry from the provided specification
 */
void load_static_obstacles_into_plant(MultibodyPlant<double>& plant, json exogenous_spec) {
    // Get the static obstacles specification
    json static_obstacles_spec = exogenous_spec.at("static_obstacles");
    std::string name = "static_obstacles";

    // Get the path to the static obstacles URDF
    std::string obstacles_urdf_path = static_obstacles_spec.at("urdf");

    // Load the static obstacles into the plant
    drake::multibody::ModelInstanceIndex static_obstacles_index = drake::multibody::Parser(&plant).AddModelFromFile(
        obstacles_urdf_path, name);

    // Weld the base link to the scene
    const auto& obstacles_root = plant.GetBodyByName(static_obstacles_spec.at("root_link_name"), static_obstacles_index);
    drake::math::RigidTransform obstacles_xform = drake::math::RigidTransform(drake::Isometry3<double>::Identity());
    drake::Vector3<double> obstacles_offset;
    obstacles_offset << static_obstacles_spec.at("root_link_position").at(0),
                        static_obstacles_spec.at("root_link_position").at(1),
                        static_obstacles_spec.at("root_link_position").at(2);
    obstacles_xform.set_translation(obstacles_offset);
    double obstacles_orientation_angle = static_obstacles_spec.at("root_link_orientation").at(3);
    drake::Vector3<double> obstacles_orientation_axis;
    obstacles_orientation_axis << static_obstacles_spec.at("root_link_orientation").at(0),
                              static_obstacles_spec.at("root_link_orientation").at(1),
                              static_obstacles_spec.at("root_link_orientation").at(2);
    obstacles_xform.set_rotation(Eigen::AngleAxisd(obstacles_orientation_angle, obstacles_orientation_axis));

    plant.AddJoint<drake::multibody::WeldJoint>("weld_base_" + name, plant.world_body(), {},
                                                obstacles_root, {},
                                                obstacles_xform);
}


/*
 * Given a vector of decision variables q (of length plant.num_positions), set q so that the elements
 * of q that correspond to model_instance match the corresponding decision variable in q_instance
 */
void set_decision_vars_in_array(
    MultibodyPlant<double>& plant,
    drake::multibody::ModelInstanceIndex model_instance,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& q_instance,
    drake::EigenPtr<drake::solvers::VectorXDecisionVariable> q) {

    // We need some way of getting the decision variables for each model instance into the
    // correct order for the full plant.
    
    // Make a dummy array, fill it with placeholder numbers to trace where each decision variable should end up
    Eigen::VectorXd dummy_q_instance = drake::VectorX<double>::Zero(plant.num_positions(model_instance));
    for (int i = 0; i < plant.num_positions(model_instance); i++) {
        dummy_q_instance(i) = i+1;
    }
    Eigen::VectorXd dummy_q = drake::VectorX<double>::Zero(plant.num_positions());
    plant.SetPositionsInArray(model_instance,
                              dummy_q_instance,
                              &dummy_q);
    // Now loop through dummy_q to see where each index ended up, and set the corresponding entry of q
    // if necessary
    for (int i = 0; i < plant.num_positions(); i++) {
        int source_index = int(dummy_q(i));
        if (source_index > 0) {
            (*q)(i) = q_instance(source_index - 1);
        }
    }
}


/*
 * Load a problem specification from the command line arguments, set up a trajectory optimization
 * problem, solve it, and save the results.
 */
void Run() {
    drake::log()->set_level(spdlog::level::warn);
    // ---------------------------------------------------------------------------------------------
    // Load the problem specification (both agents and exogenous)
    // ---------------------------------------------------------------------------------------------
    std::ifstream agents_json_file(FLAGS_agents_json_path);
    if (!agents_json_file.good()) { throw std::runtime_error("No agents.json file found at given location!"); }
    json agents_spec;
    agents_json_file >> agents_spec;

    std::ifstream exogenous_json_file(FLAGS_exogenous_json_path);
    if (!exogenous_json_file.good()) { throw std::runtime_error("No exogenous.json file found at given location!"); }
    json exogenous_spec;
    exogenous_json_file >> exogenous_spec;

    // Get the trajectory parameters
    int NT = agents_spec.at("trajectory_steps");
    float T = agents_spec.at("trajectory_time");

    // ---------------------------------------------------------------------------------------------
    // Create a multibody plant
    // ---------------------------------------------------------------------------------------------
    // Create a builder to build the Drake scene.
    drake::systems::DiagramBuilder<double> builder;

    // Create a scene graph to manage the geometry.
    drake::geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<drake::geometry::SceneGraph>();
    scene_graph.set_name("scene_graph");

    // Create a MultibodyPlant to manage the robot and environment, and add it as a source
    // for the SceneGraph
    MultibodyPlant<double>& plant =
        *builder.AddSystem<MultibodyPlant<double>>(MAX_TIME_STEP);
    plant.set_name("plant");
    drake::geometry::SourceId plant_source_id = plant.RegisterAsSourceForSceneGraph(&scene_graph);

    // ---------------------------------------------------------------------------------------------
    // Load the static obstacles, agents, and dynamic obstacles into the plant
    // ---------------------------------------------------------------------------------------------
    load_static_obstacles_into_plant(plant, exogenous_spec);
    std::vector<drake::multibody::ModelInstanceIndex> agent_model_indices =
        load_agents_into_plant(plant, scene_graph, agents_spec, "agents");
    std::vector<drake::multibody::ModelInstanceIndex> dynamic_obstacle_model_indices =
        load_agents_into_plant(plant, scene_graph, exogenous_spec, "dynamic_obstacles");

    // ---------------------------------------------------------------------------------------------
    // Finalize the plant and create the needed contexts
    // ---------------------------------------------------------------------------------------------
    // Connect the scene graph and multibodyplant
    // First connect the plant geometry to the scene graph
    builder.Connect(plant.get_geometry_poses_output_port(),
                    scene_graph.get_source_pose_port(plant_source_id));
    // Then connect the scene graph query output to the plant
    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port());

    // Now the model is complete, so we finalize the plant
    plant.Finalize();

    // Build the plant
    std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();
    // Create a diagram-level context
    std::unique_ptr<drake::systems::Context<double>> diagram_context = diagram->CreateDefaultContext();

    // Then create a subsystem context for the multibodyplant
    drake::systems::Context<double>* plant_context =
        &diagram->GetMutableSubsystemContext(plant, diagram_context.get());

    // To facilitate later bookkeeping, we need to save the number of positions for each agent in the spec
    int num_agents = agent_model_indices.size();
    for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
        agents_spec.at("agents")[agent_idx]["num_positions"] = plant.num_positions(agent_model_indices[agent_idx]);
    }
    int num_dynamic_obstacles = dynamic_obstacle_model_indices.size();
    for (int obstacle_idx = 0; obstacle_idx < num_dynamic_obstacles; obstacle_idx++) {
        exogenous_spec.at("dynamic_obstacles")[obstacle_idx]["num_positions"] = plant.num_positions(dynamic_obstacle_model_indices[obstacle_idx]);
    }

    // ---------------------------------------------------------------------------------------------
    // Create a MathematicalProgram to hold the optimization problem
    // ---------------------------------------------------------------------------------------------
    std::unique_ptr<drake::solvers::MathematicalProgram> prog = std::make_unique<drake::solvers::MathematicalProgram>();

    // ---------------------------------------------------------------------------------------------
    // Create decision variables for the state of each agent and dynamic obstacle at each timestep
    // ---------------------------------------------------------------------------------------------
    // We need to create decision variables for each agent and each dynamic obstacle. These will be
    // stored in two vectors, each ordered in the same manner as in agents_spec and exogenous_spec,
    // respectively
    std::vector<drake::solvers::MatrixXDecisionVariable> agents_decision_vars;
    std::vector<drake::solvers::MatrixXDecisionVariable> dynamic_obstacles_decision_vars;

    // Create the agent decision variables. I would love to do this using a helper function but we're
    // stuck with a unique_ptr for the program object.

    // Loop through the agents in the specification and load each one
    int total_state_vars = 0;
    for (auto agent : agents_spec.at("agents")) {
        // Create a matrix of decision variables for this agent and save them
        int num_agent_positions = agent.at("num_positions");
        std::string agent_name = agent.at("name");
        agents_decision_vars.push_back(
            prog->NewContinuousVariables(NT, num_agent_positions, "q_" + agent_name)
        );
        total_state_vars += num_agent_positions;
    }
    // Do the same for the dynamic obstacles
    for (auto obstacle : exogenous_spec.at("dynamic_obstacles")) {
        // Create a matrix of decision variables for this obstacle and save them
        int num_obstacle_positions = obstacle.at("num_positions");
        std::string obstacle_name = obstacle.at("name");
        dynamic_obstacles_decision_vars.push_back(
            prog->NewContinuousVariables(NT, num_obstacle_positions, "q_" + obstacle_name)
        );
        total_state_vars += num_obstacle_positions;
    }

    // We'll later want access to a giant matrix of decision variables, where there is a row for each
    // timestep and a column for each state
    drake::solvers::MatrixXDecisionVariable q(NT, total_state_vars);
    for (int t = 0; t < NT; t++) {
        int state_index = 0;
        // Read in all agent decision variables for this timestep
        for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
            drake::solvers::MatrixXDecisionVariable agent_vars = agents_decision_vars[agent_idx];
            for (int agent_state_index = 0; agent_state_index < agent_vars.cols(); agent_state_index++) {
                q(t, state_index + agent_state_index) = agent_vars(t, agent_state_index);
            }
            state_index += agent_vars.cols();
        }
        // Do the same for the dynamic obstacles
        for (int obstacle_idx = 0; obstacle_idx < num_dynamic_obstacles; obstacle_idx++) {
            drake::solvers::MatrixXDecisionVariable obstacle_vars = dynamic_obstacles_decision_vars[obstacle_idx];
            for (int obstacle_state_index = 0; obstacle_state_index < obstacle_vars.cols(); obstacle_state_index++) {
                q(t, state_index + obstacle_state_index) = obstacle_vars(t, obstacle_state_index);
            }
            state_index += obstacle_vars.cols();
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Add a quadratic cost for each agent's sum-squared joint displacements
    // ---------------------------------------------------------------------------------------------
    for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
        // For each agent, sum the square displacement at each timestep and add it to the cost
        for (int t = 1; t < NT; t++) {
            drake::solvers::VectorXDecisionVariable q_t = agents_decision_vars[agent_idx].row(t);
            drake::solvers::VectorXDecisionVariable q_last_t = agents_decision_vars[agent_idx].row(t-1);
            prog->AddQuadraticCost(
                (q_t - q_last_t).dot(q_t - q_last_t)
            );
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Add the waypoint constraints for each agent.
    // ---------------------------------------------------------------------------------------------
    for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
        auto agent_trajectory_spec = agents_spec.at("agents")[agent_idx].at("trajectory");
        int num_waypoints = agent_trajectory_spec.size();
        int num_agent_positions = agents_spec.at("agents")[agent_idx].at("num_positions");

        // Add a constraint at each waypoint
        for (int waypoint_idx = 0; waypoint_idx < num_waypoints; waypoint_idx++) {
            auto waypoint = agent_trajectory_spec[waypoint_idx];

            // Assign the constraint to the nearest time
            float waypoint_t = waypoint.at("t");
            int t = int(waypoint_t * (NT - 1) / T);

            // If the waypoint is a state constraint, add a bounding box constraint on the state
            // at this point
            if (waypoint.contains("state")) {
                Eigen::VectorXd waypoint_state = drake::VectorX<double>::Zero(num_agent_positions);
                auto waypoint_state_spec = waypoint.at("state");
                if (waypoint_state_spec.size() != num_agent_positions) {
                    throw std::runtime_error("Number of states specified for each waypoint must match the number of states in the URDF!");
                }
                for (int q_idx = 0; q_idx < num_agent_positions; q_idx++) {
                    waypoint_state(q_idx) = waypoint_state_spec[q_idx];
                }
                
                Eigen::MatrixXd margin = 0.1 * Eigen::VectorXd::Ones(num_agent_positions);
                prog->AddBoundingBoxConstraint(waypoint_state - margin,
                                               waypoint_state + margin,
                                               agents_decision_vars[agent_idx].row(t));

            // Otherwise, if the waypoint has an angle constraint, add that
            } else if (waypoint.contains("roll")
                       && waypoint.contains("pitch")
                       && waypoint.contains("yaw")
                       && waypoint.contains("frame")) {
                // Get the orientation angles
                double roll = waypoint.at("roll");
                double pitch = waypoint.at("pitch");
                double yaw = waypoint.at("yaw");
                // And convert them to an orientation matrix
                drake::math::RollPitchYaw rpy = drake::math::RollPitchYaw(roll, pitch, yaw);
                drake::math::RotationMatrix desired_orientation = drake::math::RotationMatrix(rpy);
                drake::math::RotationMatrix identity = drake::math::RotationMatrix<double>();

                // Get the end effector frame
                std::string frame_name = waypoint.at("frame");
                const drake::multibody::Frame<double>& constraint_frame = plant.GetFrameByName(frame_name);

                // Make the orientation constraint
                float theta_bound = 0.1;
                if (waypoint.contains("margin")) {
                    theta_bound = waypoint.at("margin");
                }
                auto orientation_constraint = std::make_shared<drake::multibody::OrientationConstraint>(
                    &plant,
                    plant.world_body().body_frame(),
                    desired_orientation,
                    constraint_frame,
                    identity,
                    theta_bound,
                    plant_context
                );

                // We need to bind the constraint to the decision variables, so we need to assemble the
                // relevant decision variables all in one vector
                drake::solvers::VectorXDecisionVariable q_vectorized;
                q_vectorized.resize(plant.num_positions(), 1);

                // Set all the elements in q_vectorized corresponding to agents
                for (int i = 0; i < num_agents; i++) {
                    drake::multibody::ModelInstanceIndex agent_model_index = agent_model_indices[i];
                    drake::solvers::VectorXDecisionVariable q_t = agents_decision_vars[i].row(t);
                    set_decision_vars_in_array(plant, agent_model_index, q_t, &q_vectorized);
                }
                
                // Set all the elements in q_vectorized corresponding to obstacles
                for (int i = 0; i < num_dynamic_obstacles; i++) {
                    drake::multibody::ModelInstanceIndex obstacle_model_index = dynamic_obstacle_model_indices[i];
                    drake::solvers::VectorXDecisionVariable q_t = dynamic_obstacles_decision_vars[i].row(t);
                    set_decision_vars_in_array(plant, obstacle_model_index, q_t, &q_vectorized);
                }

                prog->AddConstraint(orientation_constraint, q_vectorized);

            // We also support angle constraints with axis angles
            } else if (waypoint.contains("axis_angle")
                       && waypoint.contains("frame")) {
                // Get the orientation
                drake::Vector3<double> orientation_axis;
                orientation_axis << waypoint.at("axis_angle").at(0),
                                    waypoint.at("axis_angle").at(1),
                                    waypoint.at("axis_angle").at(2);
                Eigen::AngleAxisd rotation_spec = Eigen::AngleAxisd(waypoint.at("axis_angle").at(3), orientation_axis);
                // And convert them to an orientation matrix
                drake::math::RotationMatrix desired_orientation = drake::math::RotationMatrix(rotation_spec);
                drake::math::RotationMatrix identity = drake::math::RotationMatrix<double>();

                // Get the end effector frame
                std::string frame_name = waypoint.at("frame");
                const drake::multibody::Frame<double>& constraint_frame = plant.GetFrameByName(frame_name);

                // Make the orientation constraint
                float theta_bound = 0.1;
                if (waypoint.contains("margin")) {
                    theta_bound = waypoint.at("margin");
                }
                auto orientation_constraint = std::make_shared<drake::multibody::OrientationConstraint>(
                    &plant,
                    plant.world_body().body_frame(),
                    desired_orientation,
                    constraint_frame,
                    identity,
                    theta_bound,
                    plant_context
                );

                // We need to bind the constraint to the decision variables, so we need to assemble the
                // relevant decision variables all in one vector
                drake::solvers::VectorXDecisionVariable q_vectorized;
                q_vectorized.resize(plant.num_positions(), 1);

                // Set all the elements in q_vectorized corresponding to agents
                for (int i = 0; i < num_agents; i++) {
                    drake::multibody::ModelInstanceIndex agent_model_index = agent_model_indices[i];
                    drake::solvers::VectorXDecisionVariable q_t = agents_decision_vars[i].row(t);
                    set_decision_vars_in_array(plant, agent_model_index, q_t, &q_vectorized);
                }
                
                // Set all the elements in q_vectorized corresponding to obstacles
                for (int i = 0; i < num_dynamic_obstacles; i++) {
                    drake::multibody::ModelInstanceIndex obstacle_model_index = dynamic_obstacle_model_indices[i];
                    drake::solvers::VectorXDecisionVariable q_t = dynamic_obstacles_decision_vars[i].row(t);
                    set_decision_vars_in_array(plant, obstacle_model_index, q_t, &q_vectorized);
                }

                prog->AddConstraint(orientation_constraint, q_vectorized);

            // Otherwise, if the waypoint has a position constraint, add that
            } else if (waypoint.contains("x")
                       && waypoint.contains("y")
                       && waypoint.contains("z")
                       && waypoint.contains("frame")) {
                // Get the orientation angles
                double x = waypoint.at("x");
                double y = waypoint.at("y");
                double z = waypoint.at("z");
                // And convert them to positions
                Eigen::Vector3d margin = 0.1 * Eigen::VectorXd::Ones(3);
                if (waypoint.contains("margin")) {
                    margin = waypoint.at("margin") * Eigen::VectorXd::Ones(3);
                }
                Eigen::Vector3d target_position{x, y, z};
                Eigen::Vector3d lower_bound = target_position - margin;
                Eigen::Vector3d upper_bound = target_position + margin;

                // Get the end effector frame
                std::string frame_name = waypoint.at("frame");
                const drake::multibody::Frame<double>& constraint_frame = plant.GetFrameByName(frame_name);

                // Make the position constraint
                auto position_constraint = std::make_shared<drake::multibody::PositionConstraint>(
                    &plant,
                    plant.world_body().body_frame(),
                    lower_bound,
                    upper_bound,
                    constraint_frame,
                    Eigen::VectorXd::Zero(3),
                    plant_context
                );

                // We need to bind the constraint to the decision variables, so we need to assemble the
                // relevant decision variables all in one vector
                drake::solvers::VectorXDecisionVariable q_vectorized;
                q_vectorized.resize(plant.num_positions(), 1);

                // Set all the elements in q_vectorized corresponding to agents
                for (int i = 0; i < num_agents; i++) {
                    drake::multibody::ModelInstanceIndex agent_model_index = agent_model_indices[i];
                    drake::solvers::VectorXDecisionVariable q_t = agents_decision_vars[i].row(t);
                    set_decision_vars_in_array(plant, agent_model_index, q_t, &q_vectorized);
                }
                
                // Set all the elements in q_vectorized corresponding to obstacles
                for (int i = 0; i < num_dynamic_obstacles; i++) {
                    drake::multibody::ModelInstanceIndex obstacle_model_index = dynamic_obstacle_model_indices[i];
                    drake::solvers::VectorXDecisionVariable q_t = dynamic_obstacles_decision_vars[i].row(t);
                    set_decision_vars_in_array(plant, obstacle_model_index, q_t, &q_vectorized);
                }

                prog->AddConstraint(position_constraint, q_vectorized);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Add the joint speed constraints for each agent
    // ---------------------------------------------------------------------------------------------
    // For each timestep, for each agent, limit the agent's maximum speed if a limit is provided
    for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
        // First check if a speed limit is even provided (skip this if it's not there)
        auto agent_spec = agents_spec.at("agents")[agent_idx];
        if (agent_spec.find("joint_speed_limit") == agent_spec.end()) {
            continue;
        }

        // Otherwise, infer whether the speed limit is a single scalar or a vector
        // Start by creating a vector to hold the speed limits for each joint.
        int num_agent_positions = agents_spec.at("agents")[agent_idx].at("num_positions");
        Eigen::VectorXd speed_limit = Eigen::VectorXd::Zero(num_agent_positions);

        // If the speed limit is just a single scalar, assume all joints are limited by that number
        if (agent_spec.at("joint_speed_limit").is_number()) {
            speed_limit = agent_spec.at("joint_speed_limit") * Eigen::VectorXd::Ones(num_agent_positions);
        } else { // If the speed limit is a vector, then make sure it has a consistent size
            auto speed_limit_spec = agent_spec.at("joint_speed_limit");
            if (speed_limit_spec.size() != num_agent_positions) {
                throw std::runtime_error("Number of joint speed limits specified must match the number of states in the URDF!");
            }
            for (int q_idx = 0; q_idx < num_agent_positions; q_idx++) {
                speed_limit(q_idx) = speed_limit_spec[q_idx];
            }
        }

        // Now enforce the constraint at each timestep
        double timestep = T / NT;
        for (int t = 1; t < NT; t++) {
            drake::solvers::VectorXDecisionVariable q_t = agents_decision_vars[agent_idx].row(t);
            drake::solvers::VectorXDecisionVariable q_last_t = agents_decision_vars[agent_idx].row(t-1);
            prog->AddLinearConstraint((q_t - q_last_t) / timestep <= speed_limit);
            prog->AddLinearConstraint((q_t - q_last_t) / timestep >= -1.0 * speed_limit);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Add the trajectory constraints for each dynamic obstacle using a first-order hold to define
    // the trajectory
    // ---------------------------------------------------------------------------------------------
    // Loop through the dynamic obstacles and define a first-order hold trajectory at the provided
    // knot points, then resample that trajectory according to the knot points of the optimization
    std::vector<drake::trajectories::PiecewisePolynomial<double>> obstacle_trajectories;
    for (int obstacle_idx = 0; obstacle_idx < num_dynamic_obstacles; obstacle_idx++) {
        auto obstacle_trajectory_spec = exogenous_spec.at("dynamic_obstacles")[obstacle_idx].at("trajectory");
        int num_waypoints = obstacle_trajectory_spec.size();
        int num_obstacle_positions = exogenous_spec.at("dynamic_obstacles")[obstacle_idx].at("num_positions");

        // Extract the waypoints from the trajectory specification
        std::vector<double> obstacle_knot_t;
        std::vector<Eigen::MatrixXd> obstacle_knot_q;
        for (int waypoint_idx = 0; waypoint_idx < num_waypoints; waypoint_idx++) {
            auto waypoint = obstacle_trajectory_spec[waypoint_idx];
            obstacle_knot_t.push_back(waypoint.at("t"));

            Eigen::MatrixXd waypoint_state = drake::VectorX<double>::Zero(num_obstacle_positions);
            auto waypoint_state_spec = waypoint.at("state");
            if (waypoint_state_spec.size() != num_obstacle_positions) {
                throw std::runtime_error("Number of states specified for each waypoint must match the number of states in the URDF!");
            }
            for (int q_idx = 0; q_idx < num_obstacle_positions; q_idx++) {
                waypoint_state(q_idx) = waypoint_state_spec[q_idx];
            }
            obstacle_knot_q.push_back(waypoint_state);
        }

        // Use those waypoints to construct the first-order hold trajectory
        drake::trajectories::PiecewisePolynomial<double> obstacle_trajectory = 
            drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(obstacle_knot_t, obstacle_knot_q);
        // Save this trajectory so we can reference it later when constructing the initial guess
        obstacle_trajectories.push_back(obstacle_trajectory);

        // Constrain every step of the obstacle's trajectory to match this trajectory
        for (int t = 0; t < NT; t++) {
            Eigen::VectorXd obstacle_q_values = obstacle_trajectory.value(t * T / NT);
            prog->AddBoundingBoxConstraint(obstacle_q_values,
                                           obstacle_q_values,
                                           dynamic_obstacles_decision_vars[obstacle_idx].row(t));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Add no-collision constraint covering all agents and all obstacles
    // ---------------------------------------------------------------------------------------------
    // To avoid collisions with obstacles, we need to add a constraint ensuring that
    // the distance between the robot and all other geometries remains above some margin
    // Add a no-collision constraint at each timestep
    std::vector<drake::solvers::Binding<drake::solvers::Constraint>> no_collision_constraints;
    for (int t = 0; t < NT; t++) {
        auto no_collision_constraint = std::make_shared<drake::multibody::MinimumDistanceConstraint>(
            &plant,
            FLAGS_min_distance,
            plant_context
        );
        // We need to bind the constraint to the decision variables, so we need to assemble the
        // relevant decision variables all in one vector
        drake::solvers::VectorXDecisionVariable q_vectorized;
        q_vectorized.resize(plant.num_positions(), 1);

        // Set all the elements in q_vectorized corresponding to agents
        for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
            drake::multibody::ModelInstanceIndex agent_model_index = agent_model_indices[agent_idx];
            drake::solvers::VectorXDecisionVariable q_t = agents_decision_vars[agent_idx].row(t);
            set_decision_vars_in_array(plant, agent_model_index, q_t, &q_vectorized);
        }
        
        // Set all the elements in q_vectorized corresponding to obstacles
        for (int obstacle_idx = 0; obstacle_idx < num_dynamic_obstacles; obstacle_idx++) {
            drake::multibody::ModelInstanceIndex obstacle_model_index = dynamic_obstacle_model_indices[obstacle_idx];
            drake::solvers::VectorXDecisionVariable q_t = dynamic_obstacles_decision_vars[obstacle_idx].row(t);
            set_decision_vars_in_array(plant, obstacle_model_index, q_t, &q_vectorized);
        }

        // Add the binding.
        no_collision_constraints.push_back(prog->AddConstraint(no_collision_constraint, q_vectorized));
    }

    // ---------------------------------------------------------------------------------------------
    // If a collision chance constraint is specified, add it.
    // ---------------------------------------------------------------------------------------------
    float Delta;
    drake::solvers::MatrixXDecisionVariable risk_allocations;
    bool use_joint_cc = true;
    std::shared_ptr<drake::solvers::Constraint> chance_constraint;
    // We need to get the robot states in a row-major vector
    Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_row_major(q);
    Eigen::Map<drake::solvers::VectorXDecisionVariable> q_vectorized(q_row_major.data(), q_row_major.size());
    std::cout << q_row_major << std::endl;
    std::cout << q_vectorized.transpose() << std::endl;
    drake::solvers::VectorXDecisionVariable all_decision_variables(q_vectorized.rows() + 2);
    if (agents_spec.contains("collision_chance_constraint")) {
        // Extract the overall chance constraint from the agents specification
        Delta = agents_spec.at("collision_chance_constraint");
        // and make sure that it is valid
        if (Delta <= 0.0 || Delta > 1) {
            throw std::runtime_error("Collision chance constraint must be in (0, 1]");
        }
        // Also extract the precision
        float risk_precision = 0.000001;
        if (agents_spec.contains("risk_precision")) {
            risk_precision = agents_spec.at("risk_precision");
        }

        // If we have a chance-constraint, at least one uncertain obstacle must be specified
        if (!exogenous_spec.contains("uncertain_obstacles") || exogenous_spec.at("uncertain_obstacles").empty()) {
            throw std::runtime_error("Chance constrained exogenous specification must include uncertain_obstacles list");
        }
        // Load the uncertain obstacles
        std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids;
        std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances;
        for (auto& uncertain_obstacle_spec : exogenous_spec.at("uncertain_obstacles")) {
            std::string uncertain_obstacle_name = uncertain_obstacle_spec.at("link_name");
            // Get the geometry IDs corresponding to this obstacle
            std::vector<drake::geometry::GeometryId> obstacle_ids =
                plant.GetCollisionGeometriesForBody(
                    plant.GetBodyByName(uncertain_obstacle_name));
            // Append that to the vector of uncertain_obstacle_ids
            uncertain_obstacle_ids.insert(uncertain_obstacle_ids.end(),
                                          obstacle_ids.begin(),
                                          obstacle_ids.end());

            // Load the covariance
            if (!uncertain_obstacle_spec.contains("location_covariance")) {
                throw std::runtime_error("Uncertain obstacles must include location_covariance");
            }
            Eigen::Matrix3d uncertain_obstacle_covariance;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    uncertain_obstacle_covariance(i, j) = uncertain_obstacle_spec.at("location_covariance")[i][j];
                }
            }
            // Include a copy of the covariance for each obstacle ID associated with this obstacle name
            for (const drake::geometry::GeometryId id : obstacle_ids) {
                uncertain_obstacle_covariances.push_back(uncertain_obstacle_covariance);
            }
        }

        // Create the world manager
        ccopt::BulletWorldManager<double>* world_manager = new ccopt::BulletWorldManager<double>();
        // And include all robot links as robot geometries
        for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
            std::vector<drake::multibody::BodyIndex> body_indices = plant.GetBodyIndices(agent_model_indices[agent_idx]);
            for (drake::multibody::BodyIndex body_index : body_indices) {
                const std::vector<drake::geometry::GeometryId> agent_geometry_ids =
                    plant.GetCollisionGeometriesForBody(plant.get_body(body_index));
                world_manager->AddRobotGeometryIds(agent_geometry_ids);
            }
        }

        // If all agents in the agents specification includes joint covariances, then use a joint collision chance
        // constraint. Otherwise, use an obstacle-only chance constraint.
        for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
            auto agent_spec = agents_spec.at("agents")[agent_idx];
            if (!agent_spec.contains("joint_covariance")) {
                use_joint_cc = false;
            }
        }

        if (use_joint_cc) {
            // And all agents should include a joint covariance
            // Simultaneously perform sanity checks and load the covariances into a large matrix
            Eigen::MatrixXd state_covariance = Eigen::MatrixXd::Zero(plant.num_positions(), plant.num_positions());
            for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
                auto agent_spec = agents_spec.at("agents")[agent_idx];
                if (!agent_spec.contains("joint_covariance")) {
                    throw std::runtime_error("Chance constrained agent specifications must include joint_covariance");
                }

                // And that joint covariance should have consistent dimension
                int num_agent_positions = agent_spec.at("num_positions");
                if (agent_spec.at("joint_covariance").size() != num_agent_positions) {
                    throw std::runtime_error("Agent joint_covariance has inconsistent dimension");
                }
                for (int dimension = 0; dimension < num_agent_positions; dimension++) {
                    if (agent_spec.at("joint_covariance")[dimension].size() != num_agent_positions) {
                        throw std::runtime_error("Agent joint_covariance has inconsistent dimension");
                    }
                }

                // We need to fit the state covariance matrix for this one agent into the much larger
                // matrix containing the covariance for all agents.
                for (int i = 0; i < num_agent_positions; i++) {
                    for (int j = 0; j < num_agent_positions; j++) {
                        int target_i = prog->FindDecisionVariableIndex(agents_decision_vars[agent_idx](i));
                        int target_j = prog->FindDecisionVariableIndex(agents_decision_vars[agent_idx](j));

                        state_covariance(target_i, target_j) = agent_spec.at("joint_covariance")[i][j];
                    }
                }
            }

            // We need two additional decision variables for delta and gamma
            risk_allocations = prog->NewContinuousVariables(2, 1, "risk_allocations");
            // Constrain the sum to be less than the user-specified bound
            prog->AddConstraint(risk_allocations(0) + risk_allocations(1) <= Delta);
            prog->AddConstraint(risk_allocations(0) >= 0.0001);
            prog->AddConstraint(risk_allocations(1) >= 0.0001);
            prog->AddCost(10 * risk_allocations(1));

            all_decision_variables << q_vectorized, risk_allocations;

            // Next add a chance constraint covering the entire trajectory
            std::cout << "making constraint" << std::endl;
            chance_constraint = std::make_shared<ccopt::JointCollisionChanceConstraint>(
                &plant, plant_context, world_manager,
                risk_precision,
                NT,
                uncertain_obstacle_ids, uncertain_obstacle_covariances,
                state_covariance
            );
            std::cout << "binding constraint" << std::endl;
            std::cout << chance_constraint->num_vars() << std::endl;
            std::cout << prog->decision_variables().rows() << std::endl;
            prog->AddConstraint(chance_constraint, all_decision_variables);
            std::cout << "bound constraint" << std::endl;
        } else{
            // Next add a chance constraint covering the entire trajectory
            chance_constraint = std::make_shared<ccopt::CollisionChanceConstraint>(
                &plant, plant_context, world_manager,
                risk_precision,
                Delta,
                NT,
                false,
                uncertain_obstacle_ids, uncertain_obstacle_covariances
            );
            // Add the chance constraint to the program
            std::cout << "binding constraint" << std::endl;
            prog->AddConstraint(chance_constraint, q_vectorized);
            std::cout << "bound constraint" << std::endl;
        }

    }

    // ---------------------------------------------------------------------------------------------
    // Create a seed trajectory. This should include the exact trajectory for all dynamic obstacles,
    // and some initial guess for all agents.
    // ---------------------------------------------------------------------------------------------
    // Construct a vector to hold the guess
    Eigen::VectorXd guess = drake::VectorX<double>::Zero(prog->num_vars());

    // Do the same for obstacles
    for (int obstacle_idx = 0; obstacle_idx < num_dynamic_obstacles; obstacle_idx++) {
        // We've already constructed a trajectory for each obstacle, so we merely need to load it
        // into the guess
        for (int t = 0; t < NT; t++) {
            Eigen::VectorXd q_values = obstacle_trajectories[obstacle_idx].value(t * T / NT);
            prog->SetDecisionVariableValueInVector(
                dynamic_obstacles_decision_vars[obstacle_idx].row(t),
                q_values.transpose(),
                &guess
            );
        }
    }

    // And the same for risk allocations if needed
    if (agents_spec.contains("collision_chance_constraint") && use_joint_cc) {
        prog->SetDecisionVariableValueInVector(
            risk_allocations(0),
            Delta / 2.0,
            &guess
        );
        prog->SetDecisionVariableValueInVector(
            risk_allocations(1),
            Delta / 2.0,
            &guess
        );
    }

    // ---------------------------------------------------------------------------------------------
    // Create the solver and solve the trajectory optimization problem
    // ---------------------------------------------------------------------------------------------
    drake::solvers::SnoptSolver solver;
    drake::solvers::SolverOptions opts;
    // opts.SetOption(solver.solver_id(), "Print file", "ccopt_runner.snopt.out");
    // opts.SetOption(solver.solver_id(), "Verify level", 2);
    opts.SetOption(solver.solver_id(), "Major optimality tolerance", sqrt(0.00001));
    opts.SetOption(solver.solver_id(), "Major feasibility tolerance", sqrt(10*0.00001));
    opts.SetOption(solver.solver_id(), "Function precision", 0.00001);
    std::cout << "Solving" << std::endl;
    drake::solvers::MathematicalProgramResult result = solver.Solve(*prog, guess, opts);
    std::cout << "Solved" << std::endl;

    // ---------------------------------------------------------------------------------------------
    // Save the results to the specified file
    // ---------------------------------------------------------------------------------------------
    // First create a json with the plan result, and then serialize that to a file.
    json response_json;

    // Reference the planning problem specifications
    response_json["agents_spec"] = FLAGS_agents_json_path;
    response_json["exogenous_spec"] = FLAGS_exogenous_json_path;
    
    // Create a sub-json for the plan itself
    json plan;
    plan["optimal_cost"] = double(result.get_optimal_cost());

    // Extract the trajectory for each agent from the result
    std::cout << "Extracting trajectory" << std::endl;
    json agent_trajectories;
    for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
        int num_agent_positions = agents_spec.at("agents")[agent_idx].at("num_positions");
        json agent_trajectory;

        for (int t = 0; t < NT; t++) {
            json waypoint;
            waypoint["t"] = t * T / NT;

            json q_t;
            Eigen::VectorXd q_t_result = result.GetSolution(agents_decision_vars[agent_idx].row(t));
            for (int i = 0; i < q_t_result.size(); i ++) {
                q_t.push_back(q_t_result(i));
            }
            waypoint["state"] = q_t;

            agent_trajectory.push_back(waypoint);
        }

        agent_trajectories.push_back(agent_trajectory);
    }
    plan["agent_trajectories"] = agent_trajectories;
    std::cout << "Done" << std::endl;

    // Save the chance constraint data as well, if a chance constraint was used
    std::cout << "Checking chance constraints" << std::endl;
    if (agents_spec.contains("collision_chance_constraint")) {
        if (use_joint_cc) {
            plan["delta"] = result.GetSolution(risk_allocations(0));
            plan["gamma"] = result.GetSolution(risk_allocations(1));
            plan["chance_constraint_satisfied"] = chance_constraint->CheckSatisfied(result.GetSolution(all_decision_variables));
        } else {
            std::cout << "EvalBinding" << std::endl;
            Eigen::VectorXd cc_value(1);
            chance_constraint->Eval(result.GetSolution(q_vectorized), &cc_value);
            std::cout << cc_value << std::endl;
            std::cout << "Done" << std::endl;
            plan["delta"] = cc_value(0);
            plan["chance_constraint_satisfied"] = chance_constraint->CheckSatisfied(result.GetSolution(q_vectorized));
        }
    }
    std::cout << "Done" << std::endl;

    // Write the plan to the specified file
    std::ofstream o(FLAGS_save_plan_location);
    o << std::setw(4) << plan << std::endl;

    // ---------------------------------------------------------------------------------------------
    // Visualize the trajectory if needed
    // ---------------------------------------------------------------------------------------------
    std::cout << "Visualizing" << std::endl;
    if (FLAGS_visualize) {
        // Make a new diagram builder and scene graph for visualizing
        drake::systems::DiagramBuilder<double> viz_builder;
        drake::geometry::SceneGraph<double>& viz_scene_graph =
          *viz_builder.AddSystem<drake::geometry::SceneGraph>();
        viz_scene_graph.set_name("scene_graph");

        // Also make a new plant (annoying that we have to do this)
        MultibodyPlant<double> viz_plant = MultibodyPlant<double>(MAX_TIME_STEP);
        drake::geometry::SourceId viz_plant_source_id = viz_plant.RegisterAsSourceForSceneGraph(&viz_scene_graph);
        load_static_obstacles_into_plant(viz_plant, exogenous_spec);
        std::vector<drake::multibody::ModelInstanceIndex> agent_model_indices_viz =
            load_agents_into_plant(viz_plant, viz_scene_graph, agents_spec, "agents");
        std::vector<drake::multibody::ModelInstanceIndex> dynamic_obstacles_model_indices_viz =
            load_agents_into_plant(viz_plant, viz_scene_graph, exogenous_spec, "dynamic_obstacles");
        viz_plant.Finalize();

        // Extract the trajectory to visualize
        std::vector<double> t_solution;
        std::vector<Eigen::MatrixXd> q_solution;
        // At each timestep, loop through each agent/obstacle and copy its pose into the visualization trajectory
        for (int t = 0; t < NT; t++) {
            t_solution.push_back(t * T / NT);

            Eigen::VectorXd q_t = drake::VectorX<double>::Zero(viz_plant.num_positions());
            for (int agent_idx = 0; agent_idx < num_agents; agent_idx++) {
                viz_plant.SetPositionsInArray(agent_model_indices_viz[agent_idx],
                                              result.GetSolution(agents_decision_vars[agent_idx].row(t)),
                                              &q_t);
            }
            for (int obstacle_idx = 0; obstacle_idx < num_dynamic_obstacles; obstacle_idx++) {
                viz_plant.SetPositionsInArray(dynamic_obstacles_model_indices_viz[obstacle_idx],
                                              result.GetSolution(dynamic_obstacles_decision_vars[obstacle_idx].row(t)),
                                              &q_t);
            }
            q_solution.push_back(q_t);
        }
        // Copy the last waypoint at the end
        q_solution.push_back(q_solution[NT - 1]);
        t_solution.push_back(T);
        // Make a trajectory from these waypoints (first-order hold)
        drake::trajectories::PiecewisePolynomial<double> trajectory_solution =
            drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(t_solution, q_solution);
        const auto traj_source = viz_builder.AddSystem<drake::systems::TrajectorySource<double>>(
            trajectory_solution);

        // Connect the trajectory source directly to the geometry poses
        auto q_to_pose =
            viz_builder.AddSystem<drake::systems::rendering::MultibodyPositionToGeometryPose<double>>(
                viz_plant);
        viz_builder.Connect(traj_source->get_output_port(),
                            q_to_pose->get_input_port());
        viz_builder.Connect(q_to_pose->get_output_port(),
                            viz_scene_graph.get_source_pose_port(viz_plant_source_id));

        // Create the visualizer
        drake::geometry::ConnectDrakeVisualizer(&viz_builder, viz_scene_graph);
        std::unique_ptr<drake::systems::Diagram<double>> viz_diagram = viz_builder.Build();

        // Set up simulator and run
        drake::systems::Simulator<double> simulator(*viz_diagram);
        simulator.set_publish_every_time_step(true);
        simulator.set_target_realtime_rate(1.0);
        simulator.Initialize();
        simulator.AdvanceTo(T);
    }
    std::cout << "Done" << std::endl;

    std::cout << "cc-opt finished!" << std::endl;
}

} // namespace runner
} // namespace ccopt

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ccopt::runner::Run();
  return 0;
}
