/*
 * Set up, solve, and visualize a trajectory optimization problem involving only no-collision constraints.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <chrono>

#include <gflags/gflags.h>

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

namespace drake {
namespace examples {
namespace kuka {

using drake::multibody::MultibodyPlant;

DEFINE_double(min_distance, 0.01,
              "The minimum allowable distance between collision bodies during the trajectory.");
DEFINE_int32(num_benchmark_runs, 1,
             "The number of times which the optimization problem should be solved to measure its runtime.");
DEFINE_int32(T, 12,
             "The number of timesteps used to define the trajectory.");
DEFINE_double(simulation_time, 4.0,
              "Desired duration of the simulation in seconds");
DEFINE_double(max_time_step, 1.0e-3,
              "Simulation time step used for integrator.");
DEFINE_double(target_realtime_rate, 1,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");


// Load a plant for this arm planning problem
std::vector<multibody::ModelInstanceIndex> load_plant_components(MultibodyPlant<double>& plant) {
    std::vector<multibody::ModelInstanceIndex> plant_indices; 

    // Add a moving obstacle
    std::string obstacle_file_name = "./examples/models/movable_box.urdf";
    multibody::ModelInstanceIndex plant_index_obstacle = multibody::Parser(&plant).AddModelFromFile(obstacle_file_name, "obstacle");
    plant_indices.push_back(plant_index_obstacle);

    // Load the robot arm into the MultibodyPlant
    std::string robot_file_name = "./examples/models/iiwa14_primitive_collision.urdf";
    multibody::ModelInstanceIndex plant_index1 = multibody::Parser(&plant).AddModelFromFile(robot_file_name, "robot1");
    multibody::ModelInstanceIndex plant_index2 = multibody::Parser(&plant).AddModelFromFile(robot_file_name, "robot2");
    multibody::ModelInstanceIndex plant_index3 = multibody::Parser(&plant).AddModelFromFile(robot_file_name, "robot3");
    plant_indices.push_back(plant_index1);
    plant_indices.push_back(plant_index2);
    plant_indices.push_back(plant_index3);

    // Weld the robot arm to the origin.
    const auto& joint_arm_root1 = plant.GetBodyByName("base", plant_index1);
    drake::math::RigidTransform bot_xform1 = drake::math::RigidTransform(Isometry3<double>::Identity());
    Vector3<double> bot_offset1;
    bot_offset1 << 0.0, -0.15, 0.0;
    bot_xform1.set_translation(bot_offset1);
    plant.AddJoint<multibody::WeldJoint>("weld_arm", plant.world_body(), {},
                                         joint_arm_root1, {},
                                         bot_xform1);

    const auto& joint_arm_root2 = plant.GetBodyByName("base", plant_index2);
    drake::math::RigidTransform bot_xform2 = drake::math::RigidTransform(Isometry3<double>::Identity());
    Vector3<double> bot_offset2;
    bot_offset2 << 1.0, 0.15, 0.0;
    bot_xform2.set_translation(bot_offset2);
    plant.AddJoint<multibody::WeldJoint>("weld_arm", plant.world_body(), {},
                                         joint_arm_root2, {},
                                         bot_xform2);

    const auto& joint_arm_root3 = plant.GetBodyByName("base", plant_index3);
    drake::math::RigidTransform bot_xform3 = drake::math::RigidTransform(Isometry3<double>::Identity());
    Vector3<double> bot_offset3;
    bot_offset3 << 0.0, 0.45, 0.0;
    bot_xform3.set_translation(bot_offset3);
    plant.AddJoint<multibody::WeldJoint>("weld_arm", plant.world_body(), {},
                                         joint_arm_root3, {},
                                         bot_xform3);
    
    return plant_indices;
}


void set_decision_vars_in_array(
    MultibodyPlant<double>& plant,
    multibody::ModelInstanceIndex model_instance,
    const Eigen::Ref<const drake::solvers::VectorXDecisionVariable>& q_instance,
    drake::EigenPtr<drake::solvers::VectorXDecisionVariable> q) {

    // We need some way of getting the decision variables for each model instance into the
    // correct order for the full plant.
    
    // Make a dummy array, fill it with placeholder numbers to trace where each decision variable should end up
    Eigen::VectorXd dummy_q_instance = VectorX<double>::Zero(plant.num_positions(model_instance));
    for (int i = 0; i < plant.num_positions(model_instance); i++) {
        dummy_q_instance(i) = i+1;
    }
    Eigen::VectorXd dummy_q = VectorX<double>::Zero(plant.num_positions());
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


void DoMain() {
    DRAKE_DEMAND(FLAGS_simulation_time > 0);
    drake::log()->set_level(spdlog::level::debug);

    // Create a builder to build the Drake scene.
    systems::DiagramBuilder<double> builder;

    // Create a scene graph to manage the geometry.
    geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
    scene_graph.set_name("scene_graph");

    // Create a MultibodyPlant to manage the robot and environment, and add it as a source
    // for the SceneGraph
    MultibodyPlant<double>& plant =
        *builder.AddSystem<MultibodyPlant<double>>(FLAGS_max_time_step);
    plant.set_name("plant");
    drake::geometry::SourceId plant_source_id = plant.RegisterAsSourceForSceneGraph(&scene_graph);
    std::vector<multibody::ModelInstanceIndex> model_indices = load_plant_components(plant);

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

    // The first step is to find a simple collision-free trajectory for the robot through
    // the scene. We'll use a mathematical program for this.
    std::unique_ptr<drake::solvers::MathematicalProgram> prog =
        std::make_unique<drake::solvers::MathematicalProgram>();

    // Let's define some number of timesteps and assume a fixed time step
    int T = FLAGS_T;
    double timestep = 10.0 / T;

    // To define the trajectory, we need a decision variable for each joint position
    // at each timestep. We'll do a row for each timestep and a column for each joint, for each agent
    multibody::ModelInstanceIndex robot1_model_instance = plant.GetModelInstanceByName("robot1");
    multibody::ModelInstanceIndex robot2_model_instance = plant.GetModelInstanceByName("robot2");
    multibody::ModelInstanceIndex robot3_model_instance = plant.GetModelInstanceByName("robot3");
    multibody::ModelInstanceIndex obstacle_model_instance = plant.GetModelInstanceByName("obstacle");
    int num_positions = plant.num_positions();
    int num_robot1_positions = plant.num_positions(robot1_model_instance);
    int num_robot2_positions = plant.num_positions(robot2_model_instance);
    int num_robot3_positions = plant.num_positions(robot3_model_instance);
    int num_obstacle_positions = plant.num_positions(obstacle_model_instance);
    
    drake::solvers::MatrixXDecisionVariable q_robot1 = prog->NewContinuousVariables(
        T, num_robot1_positions, "q_r1");
    drake::solvers::MatrixXDecisionVariable q_robot2 = prog->NewContinuousVariables(
        T, num_robot2_positions, "q_r2");
    drake::solvers::MatrixXDecisionVariable q_robot3 = prog->NewContinuousVariables(
        T, num_robot3_positions, "q_r3");
    drake::solvers::MatrixXDecisionVariable q_obstacle = prog->NewContinuousVariables(
        T, num_obstacle_positions, "q_o");

    // We'll add a quadratic cost on the sum of the squared displacements between timesteps for each robot
    for (int t = 1; t < T; t++) {
        drake::solvers::VectorXDecisionVariable q_t = q_robot1.row(t);
        drake::solvers::VectorXDecisionVariable q_last_t = q_robot1.row(t-1);
        prog->AddQuadraticCost((q_t - q_last_t).dot(q_t - q_last_t) / timestep);
    }
    for (int t = 1; t < T; t++) {
        drake::solvers::VectorXDecisionVariable q_t = q_robot2.row(t);
        drake::solvers::VectorXDecisionVariable q_last_t = q_robot2.row(t-1);
        prog->AddQuadraticCost((q_t - q_last_t).dot(q_t - q_last_t) / timestep);
    }
    for (int t = 1; t < T; t++) {
        drake::solvers::VectorXDecisionVariable q_t = q_robot3.row(t);
        drake::solvers::VectorXDecisionVariable q_last_t = q_robot3.row(t-1);
        prog->AddQuadraticCost((q_t - q_last_t).dot(q_t - q_last_t) / timestep);
    }

    // We want to constrain the start and end positions. The start position should be strict
    // equality, but we can make do with a bounding box on the goal position. Do this for both robots
    Eigen::MatrixXd start1 = VectorX<double>::Zero(num_robot1_positions);
    Eigen::MatrixXd start2 = VectorX<double>::Zero(num_robot2_positions);
    Eigen::MatrixXd start3 = VectorX<double>::Zero(num_robot3_positions);
    prog->AddBoundingBoxConstraint(start1, start1, q_robot1.row(0));
    prog->AddBoundingBoxConstraint(start2, start2, q_robot2.row(0));
    prog->AddBoundingBoxConstraint(start3, start3, q_robot3.row(0));

    Eigen::MatrixXd end1 = VectorX<double>::Zero(num_robot1_positions);
    Eigen::MatrixXd end2 = VectorX<double>::Zero(num_robot2_positions);
    Eigen::MatrixXd end3 = VectorX<double>::Zero(num_robot3_positions);
    end1(1) = 2.0;
    end2(1) = -2.0;
    end3(1) = 2.0;
    Eigen::MatrixXd goal_margin = 0.01 * VectorX<double>::Ones(num_robot1_positions);
    prog->AddBoundingBoxConstraint(end1 - goal_margin, end1 + goal_margin, q_robot1.row(T-1));
    prog->AddBoundingBoxConstraint(end2 - goal_margin, end2 + goal_margin, q_robot2.row(T-1));
    prog->AddBoundingBoxConstraint(end3 - goal_margin, end3 + goal_margin, q_robot3.row(T-1));

    // If we had dynamics constraints, we could add those here. For now, we'll just
    // focus on the kinematic planning problem.

    // We want the obstacle to move linearly between two positions, so we fully constrain its trajectory
    std::vector<double> obstacle_knot_t{0.0, (T - 1) * timestep};
    // Then define state at the beginning and ending knot points
    std::vector<Eigen::MatrixXd> obstacle_knot_q(2);
    obstacle_knot_q[0] = Eigen::Vector3d{0.5, -0.5, 0.5};
    obstacle_knot_q[1] = Eigen::Vector3d{0.5, 0.75, 0.5};
    // Generate a linear interpolation (first-order hold) between the start and the end
    drake::trajectories::PiecewisePolynomial<double> obstacle_trajectory = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(obstacle_knot_t, obstacle_knot_q);
    // Constrain every step of the trajectory to match this trajectory
    for (int t = 0; t < T; t++) {
        Eigen::VectorXd obstacle_q_values = obstacle_trajectory.value(t * timestep);
        prog->AddBoundingBoxConstraint(obstacle_q_values, obstacle_q_values, q_obstacle.row(t));
    }

    // To avoid collisions with obstacles, we need to add a constraint ensuring that
    // the distance between the robot and all other geometries remains above some margin
    // Add a no-collision constraint at each timestep EXCEPT the start and end, since the
    // start and end configurations are too close to the table
    std::vector<drake::solvers::Binding<drake::solvers::Constraint>> no_collision_constraints;
    for (int t = 1; t < T-1; t++) {
        auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
            &plant,
            FLAGS_min_distance,
            plant_context
        );
        drake::solvers::VectorXDecisionVariable q_vectorized;
        q_vectorized.resize(num_positions, 1);
        set_decision_vars_in_array(plant, obstacle_model_instance, q_obstacle.row(t), &q_vectorized);
        set_decision_vars_in_array(plant, robot1_model_instance, q_robot1.row(t), &q_vectorized);
        set_decision_vars_in_array(plant, robot2_model_instance, q_robot2.row(t), &q_vectorized);
        set_decision_vars_in_array(plant, robot3_model_instance, q_robot3.row(t), &q_vectorized);
        no_collision_constraints.push_back(prog->AddConstraint(no_collision_constraint, q_vectorized));
    }

    // Before solving, it's best to provide an initial guess. Construct a vector
    // to hold that guess
    Eigen::VectorXd guess = VectorX<double>::Zero(prog->num_vars());

    // We'll seed the nonlinear solver with a linear interpolation between the  start and end positions.
    // Start by defining the knot points at the start and end times
    std::vector<double> knot_t{0.0, (T - 1) * timestep};
    // Then define state at the beginning and ending knot points
    std::vector<Eigen::MatrixXd> knot_q_robot1(2);
    std::vector<Eigen::MatrixXd> knot_q_robot2(2);
    std::vector<Eigen::MatrixXd> knot_q_robot3(2);
    knot_q_robot1[0] = start1;
    knot_q_robot1[1] = end1;
    knot_q_robot2[0] = start2;
    knot_q_robot2[1] = end2;
    knot_q_robot3[0] = start3;
    knot_q_robot3[1] = end3;
    // Generate a linear interpolation (first-order hold) between the start and the end
    drake::trajectories::PiecewisePolynomial<double> trajectory1 = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(knot_t, knot_q_robot1);
    drake::trajectories::PiecewisePolynomial<double> trajectory2 = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(knot_t, knot_q_robot2);
    drake::trajectories::PiecewisePolynomial<double> trajectory3 = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(knot_t, knot_q_robot3);

    // Save this seed trajectory in the guess vector
    for (int t = 0; t < T; t++) {
        Eigen::VectorXd q1_values = trajectory1.value(t * timestep);
        Eigen::VectorXd q2_values = trajectory2.value(t * timestep);
        Eigen::VectorXd q3_values = trajectory3.value(t * timestep);
        Eigen::VectorXd obstacle_q_values = obstacle_trajectory.value(t * timestep);
        prog->SetDecisionVariableValueInVector(
            q_robot1.row(t),
            q1_values.transpose(),
            &guess
        );
        prog->SetDecisionVariableValueInVector(
            q_robot2.row(t),
            q2_values.transpose(),
            &guess
        );
        prog->SetDecisionVariableValueInVector(
            q_robot3.row(t),
            q3_values.transpose(),
            &guess
        );
        prog->SetDecisionVariableValueInVector(
            q_obstacle.row(t),
            obstacle_q_values.transpose(),
            &guess
        );
    }

    // Now the fun part: we can finally solve the problem! (don't forget to measure runtime)
    // Turns out the guess doesn't help, so we don't use it (implicitly guess all zeros)
    drake::solvers::SnoptSolver solver;
    drake::solvers::SolverOptions opts;
    drake::log()->debug("Starting solve...");
    auto start_time = std::chrono::high_resolution_clock::now();
    drake::solvers::MathematicalProgramResult result = solver.Solve(*prog, guess, opts);
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
    drake::log()->debug("Solved in {} ms",
                        double(duration.count()) / FLAGS_num_benchmark_runs);
    drake::log()->debug("Success? {}", result.is_success());

    // Make a new diagram builder and scene graph for visualizing
    systems::DiagramBuilder<double> viz_builder;
    geometry::SceneGraph<double>& viz_scene_graph =
      *viz_builder.AddSystem<geometry::SceneGraph>();
    viz_scene_graph.set_name("scene_graph");

    // Also make a new plant (annoying that we have to do this)
    drake::log()->debug("Loading visualization plant...");
    MultibodyPlant<double> viz_plant = MultibodyPlant<double>(FLAGS_max_time_step);
    drake::geometry::SourceId viz_plant_source_id = viz_plant.RegisterAsSourceForSceneGraph(&viz_scene_graph);
    load_plant_components(viz_plant);
    viz_plant.Finalize();
    multibody::ModelInstanceIndex robot1_model_instance_viz = viz_plant.GetModelInstanceByName("robot1");
    multibody::ModelInstanceIndex robot2_model_instance_viz = viz_plant.GetModelInstanceByName("robot2");
    multibody::ModelInstanceIndex robot3_model_instance_viz = viz_plant.GetModelInstanceByName("robot3");
    multibody::ModelInstanceIndex obstacle_model_instance_viz = viz_plant.GetModelInstanceByName("obstacle");
    drake::log()->debug("\tDone.");

    // Define the trajectory as a piecewise linear
    drake::log()->debug("Constructing visualization solution...");
    std::vector<double> t_solution;
    std::vector<Eigen::MatrixXd> q_solution;
    for (int t = 0; t < T; t++) {
        t_solution.push_back(t * timestep);

        Eigen::VectorXd q_t = VectorX<double>::Zero(num_positions);
        viz_plant.SetPositionsInArray(obstacle_model_instance_viz,
                                      result.GetSolution(q_obstacle.row(t)),
                                      &q_t);
        viz_plant.SetPositionsInArray(robot1_model_instance_viz,
                                      result.GetSolution(q_robot1.row(t)),
                                      &q_t);
        viz_plant.SetPositionsInArray(robot2_model_instance_viz,
                                      result.GetSolution(q_robot2.row(t)),
                                      &q_t);
        viz_plant.SetPositionsInArray(robot3_model_instance_viz,
                                      result.GetSolution(q_robot3.row(t)),
                                      &q_t);
        q_solution.push_back(q_t);
    }
    drake::trajectories::PiecewisePolynomial<double> trajectory_solution = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(t_solution, q_solution);
    drake::log()->debug("\tDone");

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

    drake::log()->debug("Built");

    // Set up simulator.
    systems::Simulator<double> simulator(*viz_diagram);
    simulator.set_publish_every_time_step(true);
    simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
    simulator.Initialize();
    simulator.AdvanceTo(T * timestep);
}

}  // namespace kuka
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::kuka::DoMain();
  return 0;
}