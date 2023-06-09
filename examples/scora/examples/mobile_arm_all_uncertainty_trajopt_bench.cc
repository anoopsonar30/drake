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
#include <drake/solvers/solver_options.h>
#include "drake/solvers/snopt_solver.h"

#include "chance_constraints.h"
#include "joint_chance_constraints.h"


namespace drake {
namespace examples {
namespace kuka {

using drake::multibody::MultibodyPlant;

DEFINE_double(function_precision, 0.0001,
              "SNOPT option.");

DEFINE_double(delta, 0.05,
              "The maximum acceptable risk of collision over the entire trajectory.");
DEFINE_double(min_distance, 0.05,
              "The minimum allowable distance between collision bodies during the trajectory.");
DEFINE_double(traj_duration, 10,
              "The total duration of the trajectory (in seconds).");
DEFINE_double(max_accel, 0.5,
              "The maximum rate of change in velocity (m/s^2).");
DEFINE_double(max_speed, 10,
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
    std::string obstacle_course_model_file = "./examples/models/movable_base_obstacle_course.urdf";
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
        drake::Vector3<double> obstacle_1_pos{0.0, 0.5, 0.5};
        plant.AddJoint<multibody::WeldJoint>("weld_obstacle_1", floor_root, {},
                                             obstacle_1_root, {},
                                             drake::math::RigidTransform(obstacle_1_pos));

        // const auto& obstacle_2_root = plant.GetBodyByName("obstacle_2");
        // drake::Vector3<double> obstacle_2_pos{0.75, 0.5, 0.75};
        // plant.AddJoint<multibody::WeldJoint>("weld_obstacle_2", floor_root, {},
        //                                      obstacle_2_root, {},
        //                                      drake::math::RigidTransform(obstacle_2_pos));
    }

    // Load the cube into the MultibodyPlant (this cube will act as the robot navigating through space)
    std::string full_name = "./examples/models/moving_base_iiwa14_primitive_collision.urdf";
    plant_index = multibody::Parser(&plant).AddModelFromFile(full_name);

    // The cube is free-floating with x, y, z coordinates relative to the world origin, so no
    // further welding is needed
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

    // Let's define some number of timesteps and assume a fixed time step
    std::vector<int> Ts{15, 17, 19, 21, 23, 25};
    // std::vector<int> Ts{22};
    std::cout << "num_vars,time_ms" << std::endl;
    for (int T : Ts) {

        // The first step is to find a simple collision-free trajectory for the cube through
        // the scene. We'll use a mathematical program for this.
        std::unique_ptr<drake::solvers::MathematicalProgram> prog =
        std::make_unique<drake::solvers::MathematicalProgram>();

        double timestep = FLAGS_traj_duration / T;

        int num_positions = plant.num_positions();
        drake::solvers::MatrixXDecisionVariable q = prog->NewContinuousVariables(
            T, num_positions, "q");
        drake::solvers::MatrixXDecisionVariable u = prog->NewContinuousVariables(
            T, 1, "u");

        // We'll add a quadratic cost on the sum of the squared displacements between timesteps
        for (int t = 1; t < T; t++) {
            prog->AddQuadraticCost(0.5 * (q.row(t) - q.row(t-1)).dot(q.row(t) - q.row(t-1)));
        }

        // We want to constrain the start and end positions. The start position should be strict
        // equality, but we can make do with a bounding box on the goal position.
        Eigen::MatrixXd start = VectorX<double>::Zero(plant.num_positions());
        start(0) = -2.0;
        start(1) = 1.0;
        start(2) = -1.57;
        start(4) = 0.5;
        start(6) = -1.0;
        // We make this constraint a strict equality since we don't usually have the luxury of
        // changing the start position.
        prog->AddBoundingBoxConstraint(start, start, q.row(0));
        Eigen::MatrixXd end = VectorX<double>::Zero(plant.num_positions());
        end(0) = 2.0;
        end(1) = 1.0;
        end(2) = -1.57;
        end(4) = 0.5;
        end(6) = -1.0;
        // This defines the tolerance on reaching the goal (via a bounding box around the
        // desired end position). In many situations, we can accept "getting close" to the
        // goal, and this margin can help find a feasible solution.
        Eigen::MatrixXd goal_margin = 0.2 * VectorX<double>::Ones(plant.num_positions());
        prog->AddBoundingBoxConstraint(end - goal_margin, end + goal_margin, q.row(T-1));

        // Unicycle dynamics on the base
        for (int t = 1; t < T; t++) {
            // x
            prog->AddConstraint(q(t, 0) == q(t-1, 0) + timestep * u(t, 0) * cos(q(t-1, 2)));
            // y
            prog->AddConstraint(q(t, 1) == q(t-1, 1) + timestep * u(t, 0) * sin(q(t-1, 2)));
            prog->AddLinearConstraint(u(t, 0) <= FLAGS_max_speed);
            prog->AddLinearConstraint(-u(t, 0) <= FLAGS_max_speed);
        }

        // To avoid collisions with obstacles, we need to add a constraint ensuring that
        // the distance between the cube and all other geometries remains above some margin
        // Add a no-collision constraint at each timestep
        for (int t = 0; t < T; t++) {
            auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
                &plant,
                FLAGS_min_distance,
                plant_context
            );
            prog->AddConstraint(no_collision_constraint, q.row(t));
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

        // That completes our setup for this mathematical program
        drake::log()->debug("Deterministic program definition complete");

        // We can solve the collision-free problem to provide a seed for the chance-constrained
        // problem.
        drake::solvers::SnoptSolver solver;
        drake::solvers::SolverOptions opts;
        drake::log()->debug("Solving deterministic program...");
        drake::solvers::MathematicalProgramResult collision_free_result;
        auto start_time_det = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) {
            collision_free_result = solver.Solve(*prog, guess);
        }
        auto stop_time_det = std::chrono::high_resolution_clock::now();
        auto duration_det = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_det - start_time_det);
        drake::log()->info("======================================================================================");
        drake::log()->info("Solved {} deterministic optimization problems, avg. duration {} ms",
                            FLAGS_num_benchmark_runs,
                            double(duration_det.count()) / FLAGS_num_benchmark_runs);
        double avg_det_duration = double(duration_det.count()) / FLAGS_num_benchmark_runs;
        // drake::solvers::MathematicalProgramResult collision_free_result = solver.Solve(*prog, guess);
        drake::log()->debug("Deterministic program solved");

        // To check the probability of collision between the robot and the environment,
        // we need to define the list of bodies that make up the "robot", which we save
        // in the Bullet world manager
        ccopt::BulletWorldManager<double>* world_manager = new ccopt::BulletWorldManager<double>();
        std::vector<std::string> robot_body_names{
            "mobile_base_link",
            "iiwa_link_0",
            "iiwa_link_1",
            "iiwa_link_2",
            "iiwa_link_3",
            "iiwa_link_4",
            "iiwa_link_5",
            "iiwa_link_6",
            "iiwa_link_7",
        };
        for (const std::string body_name : robot_body_names) {
            const std::vector<drake::geometry::GeometryId> robot_ids =
                plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name));
            world_manager->AddRobotGeometryIds(robot_ids);
        }

        // We also need to define the bodies that are uncertain.
        std::vector<std::string> uncertain_obstacle_names{
            // "obstacle_2",
            "obstacle_1"
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

        Eigen::Matrix3d uncertain_obstacle_covariance;
        uncertain_obstacle_covariance << 0.06, 0.05, 0.0,
                                         0.05, 0.06, 0.0,
                                         0.0, 0.0, 0.01;
        // Make a vector of n copies of the covariance, where n = the number of uncertain
        // geometry IDs found above
        std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances;
        for (const drake::geometry::GeometryId id : uncertain_obstacle_ids) {
            uncertain_obstacle_covariances.push_back(uncertain_obstacle_covariance);
        }

        // Now we can add the chance constraint
        // First define the precision we want from our risk estimates
        double risk_precision = 0.0000001; // 10^-7

        // Make a clone of the program to use later
        std::unique_ptr<drake::solvers::MathematicalProgram> joint_prog = prog->Clone();

        // Now upgrade to chance constraints that consider both joint uncertainty and obstacle uncertainty
        Eigen::MatrixXd state_covariance(10, 10);
        state_covariance << 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005;

        // We need two additional decision variables for delta and gamma
        drake::solvers::MatrixXDecisionVariable risk_allocations = joint_prog->NewContinuousVariables(
            2, 1, "risk_allocations");
        // Constrain the sum to be less than the user-specified bound
        joint_prog->AddConstraint(risk_allocations(0) + risk_allocations(1) <= FLAGS_delta);
        joint_prog->AddConstraint(risk_allocations(0) >= 0.0);
        joint_prog->AddConstraint(risk_allocations(1) >= 0.0);


        // Next add a chance constraint covering the entire trajectory
        auto joint_chance_constraint = std::make_shared<ccopt::JointCollisionChanceConstraint>(
            &plant, plant_context, world_manager,
            risk_precision,
            T,
            uncertain_obstacle_ids, uncertain_obstacle_covariances,
            state_covariance
        );

        // Add the joint chance constraint to the program
        Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_row_major(q);
        Eigen::Map<drake::solvers::VectorXDecisionVariable> q_vectorized(q_row_major.data(), q_row_major.size());
        drake::solvers::VectorXDecisionVariable all_decision_variables(q_vectorized.rows() + 2);
        all_decision_variables << q_vectorized, risk_allocations;
        drake::solvers::Binding<drake::solvers::Constraint> joint_chance_constraint_bound =
            joint_prog->AddConstraint(joint_chance_constraint, all_decision_variables);

        // We'll seed the nonlinear solver with the solution from the collision-free problem
        guess = VectorX<double>::Zero(joint_prog->num_vars());
        joint_prog->SetDecisionVariableValueInVector(
            q,
            collision_free_result.GetSolution(q),
            &guess
        );

        // That completes our setup for the chance-constrained mathematical program
        drake::log()->debug("Chance-constrained (joint) program definition complete");

        // Now we get to solve it!
        drake::log()->debug("Solving chance-constrained program (joint)...");
        drake::solvers::MathematicalProgramResult joint_result;
        auto start_time_joint = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
            auto start_time_joint_i = std::chrono::high_resolution_clock::now();
            joint_result = solver.Solve(*joint_prog, guess, opts);
            auto stop_time_joint_i = std::chrono::high_resolution_clock::now();
            auto duration_joint_i = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_joint_i - start_time_joint_i);

            std::cout << 2 + 10 * T << "," << double(duration_joint_i.count()) + avg_det_duration << std::endl;
        }
        auto stop_time_joint = std::chrono::high_resolution_clock::now();
        auto duration_joint = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_joint - start_time_joint);
        drake::log()->info("======================================================================================");
        drake::log()->info("Solved {} chance-constrained optimization problems (joint), avg. duration {} ms",
                            FLAGS_num_benchmark_runs,
                            double(duration_joint.count()) / FLAGS_num_benchmark_runs);
        // drake::solvers::MathematicalProgramResult joint_result = solver.Solve(*joint_prog, guess, opts);
        drake::log()->info("Success? {}", joint_result.is_success());
        drake::log()->info("Risk allocation:\n\tdelta = {}\n\tgamma = {}",
            joint_result.GetSolution(risk_allocations(0)),
            joint_result.GetSolution(risk_allocations(1)));
    }
}

}  // namespace kuka
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::kuka::DoMain();
  return 0;
}