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

#define PI 3.141592

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
void load_plant_components(MultibodyPlant<double>& plant) {
    // Add a moving obstacle
    std::string obstacle_file_name = "./examples/models/movable_box.urdf";
    multibody::ModelInstanceIndex plant_index2 = multibody::Parser(&plant).AddModelFromFile(obstacle_file_name);

    // Load the robot arm into the MultibodyPlant
    std::string robot_file_name = "./examples/models/rs010n.urdf";
    multibody::ModelInstanceIndex plant_index1 = multibody::Parser(&plant).AddModelFromFile(robot_file_name);

    // Weld the robot arm to the origin.
    const auto& joint_arm_root = plant.GetBodyByName("base_link");
    drake::math::RigidTransform bot_xform = drake::math::RigidTransform(Isometry3<double>::Identity());
    Vector3<double> bot_offset;
    bot_offset << 0.0, 0.0, 0.0;
    bot_xform.set_translation(bot_offset);
    bot_xform.set_rotation(Eigen::AngleAxisd(-0.25 * PI, Eigen::Vector3d::UnitZ()));
    plant.AddJoint<multibody::WeldJoint>("weld_arm", plant.world_body(), {},
                                         joint_arm_root, {},
                                         bot_xform);
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
    load_plant_components(plant);

    // Connect the scene graph and multibodyplant
    // First connect the plant geometry to the scene graph
    builder.Connect(plant.get_geometry_poses_output_port(),
                    scene_graph.get_source_pose_port(plant_source_id));
    // Then connect the scene graph query output to the plant
    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port());

    // Now the model is complete, so we finalize the plant
    plant.Finalize();

    // // Make sure that the scene graph ignores collisions between the robot and the box
    // // that it's holding, as well as self-collisions.
    // geometry::GeometrySet robot_link_ids;
    // const std::vector<drake::geometry::GeometryId> end_link_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_7"));
    // robot_link_ids.Add(end_link_ids);
    // const std::vector<drake::geometry::GeometryId> link_6_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_6"));
    // robot_link_ids.Add(link_6_ids);
    // const std::vector<drake::geometry::GeometryId> link_5_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_5"));
    // robot_link_ids.Add(link_5_ids);
    // const std::vector<drake::geometry::GeometryId> link_4_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_4"));
    // robot_link_ids.Add(link_4_ids);
    // const std::vector<drake::geometry::GeometryId> link_3_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_3"));
    // robot_link_ids.Add(link_3_ids);
    // const std::vector<drake::geometry::GeometryId> link_2_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_2"));
    // robot_link_ids.Add(link_2_ids);
    // const std::vector<drake::geometry::GeometryId> link_1_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_1"));
    // robot_link_ids.Add(link_1_ids);
    // const std::vector<drake::geometry::GeometryId> link_0_ids =
    //     plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_0"));
    // robot_link_ids.Add(link_0_ids);
    // scene_graph.ExcludeCollisionsWithin(robot_link_ids);

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
    // at each timestep. We'll do a row for each timestep and a column for each joint
    int num_positions = plant.num_positions(); // 7 robot coordinates + 3 box coordinates
    drake::solvers::MatrixXDecisionVariable q = prog->NewContinuousVariables(
        T, num_positions, "q");

    // We'll add a quadratic cost on the sum of the squared displacements between timesteps
    // We only care about the robot coordinates (the first 7)
    int num_robot_positions = num_positions - 3;
    for (int t = 1; t < T; t++) {
        drake::solvers::VectorXDecisionVariable q_t = q.row(t);
        drake::solvers::VectorXDecisionVariable q_last_t = q.row(t-1);
        prog->AddQuadraticCost(
            (q_t.tail(num_robot_positions) - q_last_t.tail(num_robot_positions)).dot(
                q_t.tail(num_robot_positions) - q_last_t.tail(num_robot_positions)) / timestep);
    }

    // We want to constrain the start and end positions. The start position should be strict
    // equality, but we can make do with a bounding box on the goal position.
    Eigen::MatrixXd start = VectorX<double>::Zero(num_robot_positions);
    // Because we need the start position to match the current position of the robot, we
    // make this constraint a strict equality
    drake::solvers::VectorXDecisionVariable q_0 = q.row(0);
    prog->AddBoundingBoxConstraint(start, start, q_0.tail(num_robot_positions));

    Eigen::MatrixXd end = VectorX<double>::Zero(num_robot_positions);
    end(0) = 0.3;
    end(1) = 0.6;
    end(2) = -2.0;
    // This defines the tolerance on reaching the goal (via a bounding box around the
    // desired end position). In many situations, we can accept "getting close" to the
    // goal, and this margin can help find a feasible solution.
    Eigen::MatrixXd goal_margin = 0.05 * VectorX<double>::Ones(num_robot_positions);
    drake::solvers::VectorXDecisionVariable q_end = q.row(T-1);
    prog->AddBoundingBoxConstraint(end - goal_margin, end + goal_margin, q_end.tail(num_robot_positions));

    // If we had dynamics constraints, we could add those here. For now, we'll just
    // focus on the kinematic planning problem.

    // We want the obstacle to move linearly between two positions, so we fully constrain its trajectory
    std::vector<double> obstacle_knot_t{0.0, (T - 1) * timestep};
    // Then define state at the beginning and ending knot points
    std::vector<Eigen::MatrixXd> obstacle_knot_q(2);
    // obstacle_knot_q[0] = Eigen::Vector3d{0.5, 1, 0.5};
    // obstacle_knot_q[1] = Eigen::Vector3d{0.5, -1, 0.5};
    obstacle_knot_q[0] = Eigen::Vector3d{0.5, -0.5, 0.5};
    obstacle_knot_q[1] = Eigen::Vector3d{0.5, 0.5, 0.5};
    // Generate a linear interpolation (first-order hold) between the start and the end
    drake::trajectories::PiecewisePolynomial<double> obstacle_trajectory = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(obstacle_knot_t, obstacle_knot_q);
    // Constrain every step of the trajectory to match this trajectory
    for (int t = 0; t < T; t++) {
        Eigen::VectorXd obstacle_q_values = obstacle_trajectory.value(t * timestep);
        drake::solvers::VectorXDecisionVariable q_t = q.row(t);
        prog->AddBoundingBoxConstraint(obstacle_q_values, obstacle_q_values,
                                       q_t.head(3));
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
        no_collision_constraints.push_back(prog->AddConstraint(no_collision_constraint, q.row(t)));
    }

    // Before solving, it's best to provide an initial guess. Construct a vector
    // to hold that guess
    Eigen::VectorXd guess = VectorX<double>::Zero(prog->num_vars());

    // We'll seed the nonlinear solver with a linear interpolation between the  start and end positions.
    // Start by defining the knot points at the start and end times
    std::vector<double> knot_t{0.0, (T - 1) * timestep};
    // Then define state at the beginning and ending knot points
    std::vector<Eigen::MatrixXd> knot_q(2);
    knot_q[0] = start;
    knot_q[1] = end;
    // Generate a linear interpolation (first-order hold) between the start and the end
    drake::trajectories::PiecewisePolynomial<double> trajectory = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(knot_t, knot_q);

    // Save this seed trajectory in the guess vector
    for (int t = 0; t < T; t++) {
        Eigen::VectorXd q_values = trajectory.value(t * timestep);
        drake::solvers::VectorXDecisionVariable q_t = q.row(t);
        prog->SetDecisionVariableValueInVector(
            q_t.tail(num_robot_positions),
            q_values,
            &guess
        );
    }

    // Now the fun part: we can finally solve the problem! (don't forget to measure runtime)
    // Turns out the guess doesn't help, so we don't use it (implicitly guess all zeros)
    drake::solvers::SnoptSolver solver;
    drake::solvers::SolverOptions opts;
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
    MultibodyPlant<double> viz_plant = MultibodyPlant<double>(FLAGS_max_time_step);
    drake::geometry::SourceId viz_plant_source_id = viz_plant.RegisterAsSourceForSceneGraph(&viz_scene_graph);
    load_plant_components(viz_plant);
    viz_plant.Finalize();

    // Define the trajectory as a piecewise linear
    std::vector<double> t_solution;
    std::vector<Eigen::MatrixXd> q_solution;
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