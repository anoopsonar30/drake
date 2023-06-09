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
#include "drake/solvers/binding.h"
#include <drake/solvers/solver_options.h>
#include "drake/solvers/snopt_solver.h"

#include "chance_constraints.h"

namespace drake {
namespace examples {
namespace kuka {

using drake::multibody::MultibodyPlant;

DEFINE_double(function_precision, 0.0001,
              "SNOPT option.");

DEFINE_double(delta, 0.2,
              "The maximum acceptable risk of collision over the entire trajectory.");
DEFINE_double(min_distance, 0.01,
              "The minimum allowable distance between collision bodies during the trajectory.");
DEFINE_double(traj_duration, 5,
              "The total duration of the trajectory (in seconds).");
DEFINE_double(max_speed, 2,
              "The maximum rate of change in any joint angle (rad/s).");
DEFINE_int32(num_benchmark_runs, 1,
             "The number of times which the optimization problem should be solved to measure its runtime.");
DEFINE_bool(use_max, false,
             "If true, only the maximum waypoint risk over the entire trajectory is constrained.");
DEFINE_int32(T, 10,
             "The number of timesteps used to define the trajectory.");
DEFINE_double(simulation_time, 5.0,
              "Desired duration of the simulation in seconds");
DEFINE_double(max_time_step, 1.0e-3,
              "Simulation time step used for integrator.");
DEFINE_double(target_realtime_rate, 1,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");


// Load a plant for this arm planning problem
void load_plant_components(MultibodyPlant<double>& plant) {
    // Load the workstation model into the MultibodyPlant
    std::string workstation_model_file = "./examples/models/workstation.urdf";
    multibody::ModelInstanceIndex plant_index =
      multibody::Parser(&plant).AddModelFromFile(workstation_model_file);

    // Weld the workstation table to the world frame (everything in the workstation
    // is fixed relative to the table)
    const auto& table_root = plant.GetBodyByName("table");
    plant.AddJoint<multibody::WeldJoint>("weld_table", plant.world_body(), {},
                                       table_root, {},
                                       drake::math::RigidTransform(Isometry3<double>::Identity()));

    // Load the robot arm into the MultibodyPlant
    std::string full_name = "./examples/models/iiwa14_primitive_collision.urdf";
    plant_index = multibody::Parser(&plant).AddModelFromFile(full_name);

    // Weld the robot arm to the designated pad in the workstation.
    const auto& joint_arm_root = plant.GetBodyByName("base");
    const auto& robot_pad_body = plant.GetBodyByName("robot_pad");
    drake::math::RigidTransform bot_xform = drake::math::RigidTransform(Isometry3<double>::Identity());
    Vector3<double> bot_offset;
    bot_offset << 0.0, 0.0, 0.005;
    bot_xform.set_translation(bot_offset);
    plant.AddJoint<multibody::WeldJoint>("weld_arm", robot_pad_body, {},
                                         joint_arm_root, {},
                                         bot_xform);

    // Load the box that we'll be moving around
    std::string box_model_file = "./examples/models/box.urdf";
    plant_index = multibody::Parser(&plant).AddModelFromFile(box_model_file);

    // We weld the box to the end effector with a bit of an offset
    const auto& joint_arm_eef_frame = plant.GetFrameByName("iiwa_link_ee_kuka");
    const auto& box_frame = plant.GetFrameByName("box_link");
    drake::math::RigidTransform box_xform = drake::math::RigidTransform(Isometry3<double>::Identity());
    Vector3<double> box_offset;
    box_offset << 0.0, 0.0, 0.05;
    box_xform.set_translation(box_offset);
    plant.WeldFrames(
      joint_arm_eef_frame,
      box_frame,
      box_xform);
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

    // Make sure that the scene graph ignores collisions between the robot and the box
    // that it's holding.
    geometry::GeometrySet box_and_end_link;
    const std::vector<drake::geometry::GeometryId> box_ids =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("box_link"));
    box_and_end_link.Add(box_ids);
    const std::vector<drake::geometry::GeometryId> end_link_ids =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_7"));
    box_and_end_link.Add(end_link_ids);
    const std::vector<drake::geometry::GeometryId> near_end_link_ids =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_6"));
    box_and_end_link.Add(near_end_link_ids);
    scene_graph.ExcludeCollisionsWithin(box_and_end_link);

    // Also exclude collisions between the base link of the robot and the table
    geometry::GeometrySet table_and_base_link;
    const std::vector<drake::geometry::GeometryId> table_ids =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("table"));
    table_and_base_link.Add(table_ids);
    const std::vector<drake::geometry::GeometryId> pad_ids =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("robot_pad"));
    table_and_base_link.Add(pad_ids);
    const std::vector<drake::geometry::GeometryId> base_link_ids =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("iiwa_link_0"));
    table_and_base_link.Add(base_link_ids);
    scene_graph.ExcludeCollisionsWithin(table_and_base_link);

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

    // In this simple example, we lock all joints except for the shoulder, then try to
    // maximize the shoulder angle subject to a chance constraint.

    // To define the trajectory, we need a decision variable for each joint position
    // at each timestep. We'll do a row for each timestep and a column for each joint
    int num_positions = plant.num_positions();
    drake::solvers::MatrixXDecisionVariable q = prog->NewContinuousVariables(
        num_positions, "q");

    // We'll add a cost to encourage maximizing the shoulder
    int shoulder = 1;
    prog->AddLinearCost(-1.0*q(shoulder));

    // Lock all the joints except the shoulder
    std::vector<int> other_joints{0, 2, 3, 4, 5, 6};
    for (const int joint : other_joints) {
        prog->AddLinearConstraint(q(joint) == 0.0);
    }

    // Also constrain the shoulder angle in magnitude, to avoid spinning around
    prog->AddLinearConstraint(q(shoulder) <= 1.5);
    prog->AddLinearConstraint(q(shoulder) >= -1.5);

    // // To avoid collisions with obstacles, we need to add a constraint ensuring that
    // // the distance between the robot and all other geometries remains above some margin
    // // Add a no-collision constraint at each timestep EXCEPT the start and end, since the
    // // start and end configurations are too close to the table
    // for (int t = 1; t < T-1; t++) {
    //     auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
    //         &plant,
    //         FLAGS_min_distance,
    //         plant_context
    //     );
    //     prog->AddConstraint(no_collision_constraint, q.row(t));
    // }

    // To check the probability of collision between the robot and the environment,
    // we need to define the list of bodies that make up the "robot", which we save
    // in the Bullet world manager
    ccopt::BulletWorldManager<double>* world_manager = new ccopt::BulletWorldManager<double>();
    std::vector<std::string> robot_body_names{
        // "iiwa_link_0",
        // "iiwa_link_1",
        // "iiwa_link_2",
        // "iiwa_link_3",
        // "iiwa_link_4",
        // "iiwa_link_5",
        // "iiwa_link_6",
        // "iiwa_link_7",
        "box_link"
    };
    for (const std::string body_name : robot_body_names) {
      const std::vector<drake::geometry::GeometryId> robot_ids =
          plant.GetCollisionGeometriesForBody(
              plant.GetBodyByName(body_name));
      world_manager->AddRobotGeometryIds(robot_ids);
    }

    // We also need to define the bodies that are uncertain.
    std::vector<std::string> uncertain_obstacle_names{
        "human_legs",
        "human_torso",
        "human_r_arm",
        "human_l_arm",
        "human_head"
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

    // Since the uncertain obstacles are all part of the human, we give them all
    // the same covariance matrix (representing relative certainty in z, mild
    // uncertainty in x, and moderate uncertainty in y).
    Eigen::Matrix3d uncertain_obstacle_covariance;
    uncertain_obstacle_covariance << 0.01, 0.0, 0.0,
                                     0.0, 0.05, 0.0,
                                     0.0, 0.0, 0.0001;
    // Make a vector of n copies of the covariance, where n = the number of uncertain
    // geometry IDs found above
    std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances;
    for (const drake::geometry::GeometryId id : uncertain_obstacle_ids) {
        uncertain_obstacle_covariances.push_back(uncertain_obstacle_covariance);
    }

    // Now we can add the chance constraint
    // First define the precision we want from our risk estimates
    double risk_precision = 0.0000001; // 10^-7

    // Next add a chance constraint covering the entire trajectory
    auto collision_chance_constraint = std::make_shared<ccopt::CollisionChanceConstraint>(
        &plant, plant_context, world_manager,
        risk_precision,
        FLAGS_delta,
        1,
        FLAGS_use_max,
        uncertain_obstacle_ids, uncertain_obstacle_covariances
    );
    // Add the chance constraint to the program
    Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_row_major(q);
    Eigen::Map<drake::solvers::VectorXDecisionVariable> q_vectorized(q_row_major.data(), q_row_major.size());
    drake::solvers::Binding<drake::solvers::Constraint> chance_constraint_bound =
        prog->AddConstraint(collision_chance_constraint, q_vectorized);

    // Now the fun part: we can finally solve the problem! (don't forget to measure runtime)
    drake::solvers::SnoptSolver solver;
    drake::solvers::SolverOptions opts;
    opts.SetOption(solver.solver_id(), "Print file", "simple_arm_snopt.out");
    opts.SetOption(solver.solver_id(), "Verify level", 2);
    // opts.SetOption(solver.solver_id(), "Major optimality tolerance", sqrt(FLAGS_function_precision));
    // // opts.SetOption(solver.solver_id(), "Major feasibility tolerance", sqrt(10*risk_precision));
    // opts.SetOption(solver.solver_id(), "Function precision", FLAGS_function_precision);
    auto start_time = std::chrono::high_resolution_clock::now();
    drake::solvers::MathematicalProgramResult result;
    for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
        result = solver.Solve(*prog, std::nullopt, opts);
    }
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
    drake::log()->info("Ran {} optimization problems, avg. duration {} ms",
                        FLAGS_num_benchmark_runs,
                        double(duration.count()) / FLAGS_num_benchmark_runs);
    drake::log()->info("Success? {}", result.is_success());

    // Print the collision-free risk for comparison
    drake::log()->set_level(spdlog::level::debug);
    // std::cout << "Risk on collision-free seed: " << collision_free_result.EvalBinding(chance_constraint_bound) << std::endl;
    std::cout << "Risk on chance-constrained trajectory:" << std::endl;
    drake::AutoDiffVecXd x_ad = drake::math::initializeAutoDiff(result.GetSolution(q));
    drake::AutoDiffVecXd result_ad;
    collision_chance_constraint->Eval(x_ad, &result_ad);
    std::cout << result_ad(0).value() << std::endl;
    std::cout << result_ad(0).derivatives().transpose()<< std::endl;
    std::cout << result.GetSolution(q).transpose()<< std::endl;
    drake::log()->set_level(spdlog::level::off);

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
    t_solution.push_back(0.0);
    q_solution.push_back(Eigen::VectorXd::Zero(num_positions));
    t_solution.push_back(FLAGS_simulation_time);
    q_solution.push_back(result.GetSolution(q));
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
    simulator.AdvanceTo(FLAGS_simulation_time);
}

}  // namespace kuka
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::kuka::DoMain();
  return 0;
}