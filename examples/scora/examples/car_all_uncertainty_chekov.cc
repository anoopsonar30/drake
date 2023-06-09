/*
 * Set up, solve, and visualize a trajectory optimization problem involving a no-collision constraint
 * as well as a chance constraint dealing with uncertain obstacles.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <chrono>
#include <math.h>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include <drake/geometry/scene_graph_inspector.h>
#include <drake/geometry/geometry_set.h>
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/weld_joint.h"
#include "drake/multibody/inverse_kinematics/minimum_distance_constraint.h"
#include "drake/multibody/inverse_kinematics/distance_constraint_utilities.h"
#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"
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

// Wheelbase lengths
#define LR 0.2
#define LF 0.2

namespace drake {
namespace examples {
namespace kuka {

using drake::multibody::MultibodyPlant;

DEFINE_double(function_precision, 0.0001,
              "SNOPT option.");

DEFINE_double(delta, 0.1,
              "The maximum acceptable risk of collision over the entire trajectory.");
DEFINE_double(min_distance, 0.05,
              "The minimum allowable distance between collision bodies during the trajectory.");
DEFINE_double(traj_duration, 10,
              "The total duration of the trajectory (in seconds).");
DEFINE_double(max_accel, 0.5,
              "The maximum rate of change in velocity (m/s^2).");
DEFINE_double(max_speed, 0.5,
              "The maximum speed of the car (m/s).");
DEFINE_int32(num_benchmark_runs, 1,
             "The number of times which the optimization problem should be solved to measure its runtime.");
DEFINE_bool(use_max, false,
             "If true, only the maximum waypoint risk over the entire trajectory is constrained.");
DEFINE_int32(T, 16,
             "The number of timesteps used to define the trajectory.");
DEFINE_int32(T_check, 100,
             "The number of timesteps used to check the trajectory.");
DEFINE_int32(N_check, 1000,
             "The number of random trials used to check the trajectory.");
DEFINE_double(simulation_time, 5.0,
              "Desired duration of the simulation in seconds");
DEFINE_double(max_time_step, 1.0e-3,
              "Simulation time step used for integrator.");
DEFINE_double(target_realtime_rate, 2,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");


// Load a plant for this arm planning problem
void load_plant_components(MultibodyPlant<double>& plant, bool fix_obstacles) {
    // Load the obstacle course model into the MultibodyPlant
    std::string obstacle_course_model_file = "./examples/models/car_obstacle_course.urdf";
    multibody::ModelInstanceIndex plant_index =
      multibody::Parser(&plant).AddModelFromFile(obstacle_course_model_file);

    // Weld the floor to the world frame
    const auto& floor_root = plant.GetBodyByName("floor");
    plant.AddJoint<multibody::WeldJoint>("weld_floor", plant.world_body(), {},
                                         floor_root, {},
                                         drake::math::RigidTransform(Isometry3<double>::Identity()));

    // Optionally weld the obstacles to their nominal positions
    if (fix_obstacles) {
        const auto& obstacle_1_root = plant.GetBodyByName("obstacle_1");
        drake::Vector3<double> obstacle_1_pos{0.5, 1.0, 0.3};
        Eigen::AngleAxisd obstacle_1_rot = Eigen::AngleAxisd(0.0*M_PI, Eigen::Vector3d::UnitZ());
        plant.AddJoint<multibody::WeldJoint>("weld_obstacle_1", floor_root, {},
                                             obstacle_1_root, {},
                                             drake::math::RigidTransform(
                                                obstacle_1_rot, obstacle_1_pos));

        const auto& obstacle_2_root = plant.GetBodyByName("obstacle_2");
        drake::Vector3<double> obstacle_2_pos{0.5, -1.0, 0.3};
        Eigen::AngleAxisd obstacle_2_rot = Eigen::AngleAxisd(-0.0*M_PI, Eigen::Vector3d::UnitZ());
        plant.AddJoint<multibody::WeldJoint>("weld_obstacle_2", floor_root, {},
                                             obstacle_2_root, {},
                                             drake::math::RigidTransform(
                                                obstacle_2_rot, obstacle_2_pos));
    }

    // Load the cube into the MultibodyPlant (this cube will act as the robot navigating through space)
    std::string full_name = "./examples/models/car.urdf";
    plant_index = multibody::Parser(&plant).AddModelFromFile(full_name);

    // The cube is free-floating with x, y, z coordinates relative to the world origin, so no
    // further welding is needed
}


void visualize_result(
        drake::solvers::MathematicalProgramResult result,
        drake::solvers::MatrixXDecisionVariable q
    ) {
    // Make a new diagram builder and scene graph for visualizing
    systems::DiagramBuilder<double> viz_builder;
    geometry::SceneGraph<double>& viz_scene_graph =
      *viz_builder.AddSystem<geometry::SceneGraph>();
    viz_scene_graph.set_name("scene_graph");

    // Also make a new plant (annoying that we have to do this)
    MultibodyPlant<double> viz_plant = MultibodyPlant<double>(FLAGS_max_time_step);
    drake::geometry::SourceId viz_plant_source_id = viz_plant.RegisterAsSourceForSceneGraph(&viz_scene_graph);
    load_plant_components(viz_plant, true);
    viz_plant.Finalize();

    // Define the trajectory as a piecewise linear
    std::vector<double> t_solution;
    std::vector<Eigen::MatrixXd> q_solution;
    int T = FLAGS_T;
    double timestep = FLAGS_traj_duration / T;
    for (int t = 0; t < T; t++) {
        t_solution.push_back(t * timestep);
        q_solution.push_back(result.GetSolution(q.row(t).transpose()));
    }
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
    geometry::ConnectDrakeVisualizer(&viz_builder, viz_scene_graph);
    std::unique_ptr<systems::Diagram<double>> viz_diagram = viz_builder.Build();

    drake::log()->debug("Visualizer built");

    // Set up simulator.
    systems::Simulator<double> simulator(*viz_diagram);
    simulator.set_publish_every_time_step(true);
    simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
    simulator.Initialize();
    simulator.AdvanceTo(T * timestep);
}


double validate_result(
        drake::solvers::MathematicalProgramResult result,
        drake::solvers::MatrixXDecisionVariable q,
        std::shared_ptr<ccopt::JointCollisionChanceConstraint> chance_constraint) {
    // Extract the solution as a matrix (columns = timesteps, rows = DOFs)
    Eigen::MatrixXd q_trajectory = result.GetSolution(q).transpose();
    // Flatten it into a vector (Eigen defaults to column major representation, which concatenates
    // columns into one big matrix, so we now get [q1_t1, q2_t1, q3_t1, q1_t2, q2_t2, q3_t2, ...])
    Eigen::Map<const Eigen::VectorXd> q_traj_flat(q_trajectory.data(), q_trajectory.size());

    // Run a bunch of randomized trials
    int n_failures = 0;
    for (int i = 0; i < FLAGS_N_check; i++) {
        // Check whether the trajectory has a collision using the chance constraint's helper function
        bool collision_free = chance_constraint->IsTrajectoryCollisionFreeUnderPerturbation(q_traj_flat, FLAGS_T_check / FLAGS_T);

        // If there was a collision, record it
        if (!collision_free) { n_failures++; }
    }

    // Calculate the failure rate
    double failure_rate = (float)n_failures / (float)FLAGS_N_check;
    drake::log()->warn("FAILURE RATE ({} trials): {}", FLAGS_N_check, failure_rate);
    return failure_rate;
}


float estimate_collision_prob_chekov(Eigen::VectorXd waypoint, Eigen::VectorXd state_covariance,
                                     ccopt::BulletWorldManager<double>* world_manager,
                                     drake::systems::Context<double>* context,
                                     const drake::multibody::MultibodyPlant<double>& plant,
                                     std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids
                                     ) {
    float estimated_risk = 0.0;
    float n_dims = 3.0;

    Eigen::VectorXd state_std_dev = Eigen::VectorXd::Zero(state_covariance.size());
    for (int i = 0; i < state_covariance.size(); i++) {
        state_std_dev[i] = std::sqrt(state_covariance[i]);
    }
    std::vector<Eigen::VectorXd> sample_pts;
    std::vector<float> weights;
    // loop through x
    for (int ix=-1; ix <= 1; ix++) {
        float x_sample = ix * state_std_dev[0] * std::sqrt(6.0) / 2.0;
        float x_weight = (ix == 0) ? (2.0 / 3.0) : (1.0 / 6.0);
        // loop through y
        for (int iy=-1; iy <= 1; iy++) {
            float y_sample = iy * state_std_dev[1] * std::sqrt(6.0) / 2.0;
            float y_weight = (iy == 0) ? (2.0 / 3.0) : (1.0 / 6.0);
            // loop through theta
            for (int it=-1; it <= 1; it++) {
                float theta_sample = it * state_std_dev[2] * std::sqrt(6.0) / 2.0;
                float theta_weight = (it == 0) ? (2.0 / 3.0) : (1.0 / 6.0);
                
                // Add the sample point and weight
                Eigen::Vector3d sample_pt;
                sample_pt << x_sample, y_sample, theta_sample;
                sample_pts.push_back(sample_pt);
                weights.push_back(x_weight * y_weight * theta_weight * std::pow(3.141592, -n_dims / 2.0));
            }
        }
    }

    // Now accumulate the collision probability across the samples
    int num_samples = sample_pts.size();
    for (int i = 0; i < num_samples; i++) {
        Eigen::VectorXd sample_pt = sample_pts[i];
        Eigen::VectorXd test_waypoint = waypoint;
        test_waypoint[0] += sample_pt[0];
        test_waypoint[1] += sample_pt[1];
        test_waypoint[2] += sample_pt[2];
        float weight = weights[i];

        drake::multibody::internal::UpdateContextConfiguration(
            context,
            plant,
            test_waypoint);
        // Refresh the world manager
        world_manager->SynchronizeInternalWorld(plant,
                                                *context,
                                                uncertain_obstacle_ids);
        // check for collision
        if (world_manager->CheckCollision()) {
            estimated_risk += weight;
        }
    }
    drake::log()->debug("Risk estimated");
    return estimated_risk;
}


float estimate_collision_prob_chekov_extended(
    Eigen::VectorXd waypoint, Eigen::VectorXd state_covariance,
    ccopt::BulletWorldManager<double>* world_manager,
    drake::systems::Context<double>* context,
    const drake::multibody::MultibodyPlant<double>& plant,
    std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
    std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances)
{
    float estimated_risk = 0.0;
    float n_dims = 3.0;

    Eigen::VectorXd state_std_dev = Eigen::VectorXd::Zero(state_covariance.size());
    for (int i = 0; i < state_covariance.size(); i++) {
        state_std_dev[i] = std::sqrt(state_covariance[i]);
    }
    std::vector<Eigen::VectorXd> sample_pts;
    std::vector<float> weights;
    // loop through x
    for (int ix=-1; ix <= 1; ix++) {
        float x_sample = ix * state_std_dev[0] * std::sqrt(6.0) / 2.0;
        float x_weight = (ix == 0) ? (2.0 / 3.0) : (1.0 / 6.0);
        // loop through y
        for (int iy=-1; iy <= 1; iy++) {
            float y_sample = iy * state_std_dev[1] * std::sqrt(6.0) / 2.0;
            float y_weight = (iy == 0) ? (2.0 / 3.0) : (1.0 / 6.0);
            // loop through theta
            for (int it=-1; it <= 1; it++) {
                float theta_sample = it * state_std_dev[2] * std::sqrt(6.0) / 2.0;
                float theta_weight = (it == 0) ? (2.0 / 3.0) : (1.0 / 6.0);
                
                // Add the sample point and weight
                Eigen::Vector3d sample_pt;
                sample_pt << x_sample, y_sample, theta_sample;
                sample_pts.push_back(sample_pt);
                weights.push_back(x_weight * y_weight * theta_weight * std::pow(3.141592, -n_dims / 2.0));
            }
        }
    }

    // Now accumulate the collision probability across the samples
    int num_samples = sample_pts.size();
    int N_monte_carlo_checks_per_sample = 100;
    for (int i = 0; i < num_samples; i++) {
        Eigen::VectorXd sample_pt = sample_pts[i];
        Eigen::VectorXd test_waypoint = waypoint;
        test_waypoint[0] += sample_pt[0];
        test_waypoint[1] += sample_pt[1];
        test_waypoint[2] += sample_pt[2];
        float weight = weights[i];

        drake::multibody::internal::UpdateContextConfiguration(
            context,
            plant,
            test_waypoint);

        // check for collision
        for (int j = 0; j < N_monte_carlo_checks_per_sample; j++) {
            // Refresh the world manager
            world_manager->SynchronizeInternalWorld(plant, *context);
            world_manager->PerturbUncertainObstacles(uncertain_obstacle_ids,
                                                     uncertain_obstacle_covariances);
            if (world_manager->CheckCollision()) {
                estimated_risk += weight / N_monte_carlo_checks_per_sample;
            }
        }
    }
    drake::log()->debug("Risk estimated");
    return estimated_risk;
}


bool chekov_violation(Eigen::VectorXd risk_estimates, Eigen::VectorXd risk_limits) {
    return risk_estimates.sum() > risk_limits.sum();
}


Eigen::VectorXd chekov_risk_reallocation(
    Eigen::VectorXd risk_estimates, Eigen::VectorXd risk_limits,
    float adjustment_rate, float tolerance)
{
    int num_steps = risk_estimates.size();
    Eigen::VectorXd risk_limits_new = VectorX<double>::Zero(num_steps);
    std::vector<int> violated_constraints;
    float risk_used = 0.0;
    for (int i = 0; i < num_steps; i++) {
        if (risk_estimates[i] > risk_limits[i]) {
            violated_constraints.push_back(i);
        }

        if (risk_limits[i] - risk_estimates[i] > tolerance) {
            // inactive constraint
            risk_limits_new[i] = adjustment_rate * risk_limits[i] + (1.0 - adjustment_rate) * risk_estimates[i];
        } else {
            // active constraint
            risk_limits_new[i] = risk_limits[i];
        }
        risk_used += risk_limits_new[i];
    }
    
    float residual_risk = FLAGS_delta - risk_used;
    // drake::log()->warn("risk_limits {}", risk_limits.transpose());
    // drake::log()->warn("risk_limits_new {}", risk_limits_new.transpose());
    // drake::log()->warn("residual_risk {}", residual_risk);
    float total_violation = 0.0;
    int num_violated = violated_constraints.size();
    for (int i = 0; i < num_violated; i++) {
        int idx = violated_constraints[i];
        total_violation += risk_estimates[idx] - risk_limits[idx];
    }
    for (int i = 0; i < num_violated; i++) {
        int idx = violated_constraints[i];
        risk_limits_new[idx] = risk_limits[idx] + residual_risk * (risk_estimates[idx] - risk_limits[idx]) / total_violation;
    }

    return risk_limits_new;
}


void DoMain() {
    DRAKE_DEMAND(FLAGS_simulation_time > 0);
    drake::log()->set_level(spdlog::level::warn);

    // Create a builder to build the Drake scene.
    systems::DiagramBuilder<double> builder;

    // Create a scene graph to manage the geometry.
    geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
    scene_graph.set_name("scene_graph");

    // Create a MultibodyPlant to manage the robot and environment, and add it as a source
    // for the SceneGraph
    drake::log()->debug("Loading plant components...");
    MultibodyPlant<double>& plant =
        *builder.AddSystem<MultibodyPlant<double>>(FLAGS_max_time_step);
    plant.set_name("plant");
    drake::geometry::SourceId plant_source_id = plant.RegisterAsSourceForSceneGraph(&scene_graph);
    load_plant_components(plant, true);
    drake::log()->debug("Plant components loaded");

    // Connect the scene graph and multibodyplant
    // First connect the plant geometry to the scene graph
    builder.Connect(plant.get_geometry_poses_output_port(),
                    scene_graph.get_source_pose_port(plant_source_id));
    // Then connect the scene graph query output to the plant
    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port());

    // Now the model is complete, so we finalize the plant
    plant.Finalize();
    drake::log()->debug("Plant fianlized");

    // Build the plant
    std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();
    // Create a diagram-level context
    std::unique_ptr<drake::systems::Context<double>> diagram_context = diagram->CreateDefaultContext();
    drake::log()->debug("Plant built");

    // Then create a subsystem context for the multibodyplant
    drake::systems::Context<double>* plant_context =
        &diagram->GetMutableSubsystemContext(plant, diagram_context.get());

    // The first step is to find a simple collision-free trajectory for the cube through
    // the scene. We'll use a mathematical program for this.
    std::unique_ptr<drake::solvers::MathematicalProgram> prog =
        std::make_unique<drake::solvers::MathematicalProgram>();

    // Let's define some number of timesteps and assume a fixed time step
    int T = FLAGS_T;
    double timestep = FLAGS_traj_duration / T;

    // To define the trajectory, we need a decision variable for the x, y, theta coordinates
    // at each timestep. We'll do a row for each timestep and a column for each joint
    int num_positions = plant.num_positions();
    drake::solvers::MatrixXDecisionVariable q = prog->NewContinuousVariables(
        T, num_positions, "q");
    // The kinematic bicycle model also includes velocity in the state
    drake::solvers::MatrixXDecisionVariable v = prog->NewContinuousVariables(
        T, 1, "v");
    // We also need to define the control inputs: linear acceleration u1 and steering angle change u2
    // We'll have a row for each timestep and a column for each input
    drake::solvers::MatrixXDecisionVariable u = prog->NewContinuousVariables(
        T, 2, "u");

    // We'll add a quadratic cost on the sum of the squared displacements between timesteps
    for (int t = 1; t < T; t++) {
        prog->AddQuadraticCost(0.5 * (q.row(t) - q.row(t-1)).dot(q.row(t) - q.row(t-1)));
    }

    // We'll also add a max acceleration constraint
    Eigen::VectorXd speed_limit = FLAGS_max_accel * Eigen::VectorXd::Ones(2);
    for (int t = 1; t < T; t++) {
        prog->AddLinearConstraint(u(t, 0) <= FLAGS_max_accel);
        prog->AddLinearConstraint(u(t, 0) >= -FLAGS_max_accel);
        prog->AddLinearConstraint(u(t, 1) <= FLAGS_max_accel);
        prog->AddLinearConstraint(u(t, 1) >= -FLAGS_max_accel);

        prog->AddLinearConstraint(v(t, 0) <= FLAGS_max_speed);
        prog->AddLinearConstraint(v(t, 0) >= -FLAGS_max_speed);
    }

    // We want to constrain the start and end positions. The start position should be strict
    // equality, but we can make do with a bounding box on the goal position.
    //
    // We want the cube to start below the obstacle course and navigate to the other side.
    Eigen::MatrixXd start = VectorX<double>::Zero(plant.num_positions());
    start(0) = -1.0;
    start(1) = 1.5;
    start(2) = -1.57;
    // We make this constraint a strict equality since we don't usually have the luxury of
    // changing the start position.
    prog->AddBoundingBoxConstraint(start, start, q.row(0));
    Eigen::MatrixXd end = VectorX<double>::Zero(plant.num_positions());
    end(0) = 0.5;
    end(1) = 0;
    end(2) = 1.57;
    // This defines the tolerance on reaching the goal (via a bounding box around the
    // desired end position). In many situations, we can accept "getting close" to the
    // goal, and this margin can help find a feasible solution.
    Eigen::MatrixXd goal_margin = 0.01 * VectorX<double>::Ones(plant.num_positions());
    prog->AddBoundingBoxConstraint(end - goal_margin, end + goal_margin, q.row(T-1));

    // We need the car to obey kinematic bicycle dynamics
    //
    // x_{t+1} = x_t + v cos(theta + beta)
    // y_{t+1} = y_t + v sin(theta + beta)
    // v_{t+1} = v_t + u1
    // theta_{t+1} = theta_t + v_t / l_r * sin(beta)
    // beta = atan(tan(u2) * l_r / (l_f + l_r))
    for (int t = 1; t < T; t++) {
        // x
        prog->AddConstraint(q(t, 0) == q(t-1, 0) + timestep * v(t, 0) * cos(q(t-1, 2) + atan(tan(u(t, 1)) + LR / (LR + LF))));
        // y
        prog->AddConstraint(q(t, 1) == q(t-1, 1) + timestep * v(t, 0) * sin(q(t-1, 2) + atan(tan(u(t, 1)) + LR / (LR + LF))));
        // v
        prog->AddConstraint(v(t, 0) == v(t-1, 0) + timestep * u(t, 0));
        // theta
        prog->AddConstraint(q(t, 2) == q(t-1, 2) + timestep * v(t, 0) / LR * atan(tan(u(t, 1)) + LR / (LR + LF)));
    }

    // This solver will find only a locally-optimal path, but we can seed it with a linear
    // interpolation to give it a good starting point
    std::vector<double> knot_t{0.0, (T - 1) * timestep};
    // Then define state at the beginning and ending knot points
    std::vector<Eigen::MatrixXd> knot_q(2);
    knot_q[0] = start;
    knot_q[1] = end;
    // Generate a linear interpolation (first-order hold) between the start and the end
    drake::trajectories::PiecewisePolynomial<double> trajectory = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(knot_t, knot_q);

    // Save this seed trajectory in the guess vector
    Eigen::VectorXd guess = VectorX<double>::Zero(prog->num_vars());
    for (int t = 0; t < T; t++) {
        Eigen::VectorXd q_values = trajectory.value(t * timestep);
        prog->SetDecisionVariableValueInVector(
            q.row(t),
            q_values.transpose(),
            &guess
        );
    }

    // To check the probability of collision between the robot and the environment,
    // we need to define the list of bodies that make up the "robot", which we save
    // in the Bullet world manager
    ccopt::BulletWorldManager<double>* world_manager = new ccopt::BulletWorldManager<double>();
    std::vector<std::string> robot_body_names{
        "car_link"
    };
    for (const std::string body_name : robot_body_names) {
        const std::vector<drake::geometry::GeometryId> robot_ids =
            plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name));
        world_manager->AddRobotGeometryIds(robot_ids);
    }

    // We also need to define the bodies that are uncertain.
    std::vector<std::string> uncertain_obstacle_names{
        "obstacle_1",
        "obstacle_2"
    };
    std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids;
    for (const std::string name : uncertain_obstacle_names) {
        // Get the geometry IDs corresponding to this obstacle
        std::vector<drake::geometry::GeometryId> obstacle_ids =
            plant.GetCollisionGeometriesForBody(
                plant.GetBodyByName(name));
        // Append that to the vector of uncertain_obstacle_ids
        uncertain_obstacle_ids.insert(uncertain_obstacle_ids.end(),
                                      obstacle_ids.begin(),
                                      obstacle_ids.end());
    }

    // Let's make both obstacles uncertain uniformly in xyz
    Eigen::Matrix3d uncertain_obstacle_covariance;
    uncertain_obstacle_covariance << 0.01, 0.0, 0.0,
                                     0.0, 0.01, 0.0,
                                     0.0, 0.0, 0.01;
    // Make a vector of n copies of the covariance, where n = the number of uncertain
    // geometry IDs found above
    std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances;
    for (const drake::geometry::GeometryId id : uncertain_obstacle_ids) {
        uncertain_obstacle_covariances.push_back(uncertain_obstacle_covariance);
    }

    Eigen::Vector3d state_covariance;
    state_covariance << 0.01, 0.01, 0.01;

    // That completes our setup for this mathematical program
    drake::log()->debug("program definition complete");

    // We can solve the collision-free problem to provide a seed for the chance-constrained
    // problem.
    drake::solvers::SnoptSolver solver;
    drake::log()->debug("Solving pChekov program...");
    drake::solvers::MathematicalProgramResult chekov_result;
    auto start_time_det = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
        // Solve for initial trajectory with uniform risk allocation
        std::unique_ptr<drake::solvers::MathematicalProgram> prog_clone = prog->Clone();
        drake::log()->debug("Program cloned");
        Eigen::VectorXd risk_limits = FLAGS_delta / T * VectorX<double>::Ones(T);
        Eigen::VectorXd keepaway = VectorX<double>::Zero(T);
        float dstep = 0.02;
        float tolerance = 0.000001;
        float adjustment_rate = 0.9;
        for (int t = 0; t < T; t++) {
            auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
                &plant,
                keepaway[t],
                plant_context
            );
            prog_clone->AddConstraint(no_collision_constraint, q.row(t));
        }
        // drake::log()->debug("No-collision constraint added");
        drake::solvers::MathematicalProgramResult result = solver.Solve(*prog_clone, guess);
        // drake::log()->debug("First optimization complete");

        // Sample to get the risk on this trajectory
        Eigen::VectorXd risk_estimates = VectorX<double>::Zero(T);
        for (int t = 0; t < T; t++) {
            Eigen::VectorXd waypoint = result.GetSolution(q.row(t).transpose());
            risk_estimates[t] = estimate_collision_prob_chekov(
                waypoint, state_covariance,
                world_manager, plant_context,
                plant,
                uncertain_obstacle_ids
            );
        }
        // drake::log()->debug("Risk estimated");

        // Reallocate risk and continue to optimize
        int max_opt_calls = 100;
        int opt_calls = 1;
        while (chekov_violation(risk_estimates, risk_limits) && opt_calls < max_opt_calls) {
            opt_calls++;
            for (int t = 0; t < T; t++) {
                if (risk_estimates[t] > risk_limits[t]) {
                    // increase hit-in distance for waypionts in violation
                    keepaway[t] += dstep;
                }
            }
            risk_limits = chekov_risk_reallocation(risk_estimates, risk_limits, adjustment_rate, tolerance);

            prog_clone = prog->Clone();
            for (int t = 0; t < T; t++) {
                auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
                    &plant,
                    keepaway[t],
                    plant_context
                );
                prog_clone->AddConstraint(no_collision_constraint, q.row(t));
            }
            result = solver.Solve(*prog_clone, guess);
            // drake::log()->warn("Optimization {} complete", opt_calls);

            // Sample to get the risk on this trajectory
            risk_estimates = VectorX<double>::Zero(T);
            for (int t = 0; t < T; t++) {
                Eigen::VectorXd waypoint = result.GetSolution(q.row(t).transpose());
                risk_estimates[t] = estimate_collision_prob_chekov(
                    waypoint, state_covariance,
                    world_manager, plant_context,
                    plant,
                    uncertain_obstacle_ids
                );
            }
        }
        chekov_result = result;
        // drake::log()->warn("Solved pChekov optimization with {} calls to the optimizer", opt_calls);
        // drake::log()->warn("Estimated risk {}", risk_estimates.sum());
    }
    auto stop_time_det = std::chrono::high_resolution_clock::now();
    auto duration_det = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_det - start_time_det);
    drake::log()->info("======================================================================================");
    drake::log()->warn("Solved {} pChekov optimization problems, avg. duration {} ms",
                        FLAGS_num_benchmark_runs,
                        double(duration_det.count()) / FLAGS_num_benchmark_runs);
    drake::log()->debug("pChekov program solved");

    // We'll eventually run a second optimization problem to deal with risk, but let's start
    // with a simple sanity check on the collision-free solution
    visualize_result(chekov_result, q);

    drake::log()->debug("Solving extended pChekov program...");
    drake::solvers::MathematicalProgramResult extended_chekov_result;
    auto start_time_ext = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
        // Solve for initial trajectory with uniform risk allocation
        std::unique_ptr<drake::solvers::MathematicalProgram> prog_clone = prog->Clone();
        // drake::log()->warn("Program cloned");
        Eigen::VectorXd risk_limits = FLAGS_delta / T * VectorX<double>::Ones(T);
        Eigen::VectorXd keepaway = VectorX<double>::Zero(T);
        float dstep = 0.02;
        float tolerance = 0.000001;
        float adjustment_rate = 0.9;
        for (int t = 0; t < T; t++) {
            auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
                &plant,
                keepaway[t],
                plant_context
            );
            prog_clone->AddConstraint(no_collision_constraint, q.row(t));
        }
        // drake::log()->warn("No-collision constraint added");
        drake::solvers::MathematicalProgramResult result = solver.Solve(*prog_clone, guess);
        // drake::log()->warn("First optimization complete");

        // Sample to get the risk on this trajectory
        Eigen::VectorXd risk_estimates = VectorX<double>::Zero(T);
        for (int t = 0; t < T; t++) {
            Eigen::VectorXd waypoint = result.GetSolution(q.row(t).transpose());
            risk_estimates[t] = estimate_collision_prob_chekov_extended(
                waypoint, state_covariance,
                world_manager, plant_context,
                plant,
                uncertain_obstacle_ids,
                uncertain_obstacle_covariances
            );
        }
        // drake::log()->warn("Risk estimated");

        // Reallocate risk and continue to optimize
        int max_opt_calls = 100;
        int opt_calls = 1;
        while (chekov_violation(risk_estimates, risk_limits) && opt_calls < max_opt_calls) {
            opt_calls++;
            // drake::log()->warn("Estimated risk {}", risk_estimates.transpose());
            for (int t = 0; t < T; t++) {
                if (risk_estimates[t] > risk_limits[t]) {
                    // increase hit-in distance for waypionts in violation
                    keepaway[t] += dstep;
                }
            }
            risk_limits = chekov_risk_reallocation(risk_estimates, risk_limits, adjustment_rate, tolerance);

            prog_clone = prog->Clone();
            for (int t = 0; t < T; t++) {
                auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
                    &plant,
                    keepaway[t],
                    plant_context
                );
                prog_clone->AddConstraint(no_collision_constraint, q.row(t));
            }
            result = solver.Solve(*prog_clone, guess);

            // Sample to get the risk on this trajectory
            risk_estimates = VectorX<double>::Zero(T);
            for (int t = 0; t < T; t++) {
                Eigen::VectorXd waypoint = result.GetSolution(q.row(t).transpose());
                risk_estimates[t] = estimate_collision_prob_chekov_extended(
                    waypoint, state_covariance,
                    world_manager, plant_context,
                    plant,
                    uncertain_obstacle_ids,
                    uncertain_obstacle_covariances
                );
            }
            // drake::log()->warn("Optimization {} complete", opt_calls);
            // drake::log()->warn("risk limits {}", risk_limits.transpose());
            // drake::log()->warn("Estimated risk {}", risk_estimates.transpose());
            // drake::log()->warn("Estimated risk {}", risk_estimates.sum());
            // drake::log()->warn("keepaway {}", keepaway.transpose());
        }
        extended_chekov_result = result;
        // drake::log()->warn("Solved pChekov optimization with {} calls to the optimizer", opt_calls);
        // drake::log()->warn("Estimated risk {}", risk_estimates.sum());
    }
    auto stop_time_ext = std::chrono::high_resolution_clock::now();
    auto duration_ext = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_ext - start_time_ext);
    drake::log()->info("======================================================================================");
    drake::log()->warn("Solved {} extended pChekov optimization problems, avg. duration {} ms",
                        FLAGS_num_benchmark_runs,
                        double(duration_ext.count()) / FLAGS_num_benchmark_runs);
    drake::log()->debug("Extended pChekov program solved");
    drake::log()->warn("Optimal cost {}",
                            double(extended_chekov_result.get_optimal_cost()));

    // We'll eventually run a second optimization problem to deal with risk, but let's start
    // with a simple sanity check on the collision-free solution
    visualize_result(extended_chekov_result, q);

    std::vector<Eigen::Vector3d> uncertain_obstacle_nominal_positions;
    drake::Vector3<double> obstacle_1_pos{0.5, 1.0, 0.3};
    drake::Vector3<double> obstacle_2_pos{0.5, -1.0, 0.3};
    uncertain_obstacle_nominal_positions.push_back(obstacle_1_pos);
    uncertain_obstacle_nominal_positions.push_back(obstacle_2_pos);
    drake::log()->warn("Validating via Monte Carlo");
    // Next add a chance constraint covering the entire trajectory
    Eigen::Matrix3d state_covariance_mat;
    state_covariance_mat << state_covariance[0], 0.0, 0.0,
                            0.0, state_covariance[1], 0.0,
                            0.0, 0.0, state_covariance[2];
    float risk_precision = 0.000001;
    auto joint_chance_constraint = std::make_shared<ccopt::JointCollisionChanceConstraint>(
        &plant, plant_context, world_manager,
        risk_precision,
        FLAGS_T,
        uncertain_obstacle_ids, uncertain_obstacle_covariances,
        state_covariance_mat
    );
    drake::log()->warn("\t pChekov:");
    validate_result(chekov_result, q,
                    joint_chance_constraint);
    drake::log()->warn("\t pChekov extended:");
    validate_result(extended_chekov_result, q,
                    joint_chance_constraint);
}

}  // namespace kuka
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::kuka::DoMain();
  return 0;
}