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

DEFINE_double(delta, 0.1,
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
    drake::log()->debug("Solving deterministic program...");
    drake::solvers::MathematicalProgramResult collision_free_result;
    auto start_time_det = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
        collision_free_result = solver.Solve(*prog, guess);
    }
    auto stop_time_det = std::chrono::high_resolution_clock::now();
    auto duration_det = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_det - start_time_det);
    drake::log()->info("======================================================================================");
    drake::log()->warn("Solved {} deterministic optimization problems, avg. duration {} ms",
                        FLAGS_num_benchmark_runs,
                        double(duration_det.count()) / FLAGS_num_benchmark_runs);
    // drake::solvers::MathematicalProgramResult collision_free_result = solver.Solve(*prog, guess);
    drake::log()->debug("Deterministic program solved");

    // We'll eventually run a second optimization problem to deal with risk, but let's start
    // with a simple sanity check on the collision-free solution
    visualize_result(collision_free_result, q);

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

    // Next add a chance constraint covering the entire trajectory
    auto collision_chance_constraint = std::make_shared<ccopt::CollisionChanceConstraint>(
        &plant, plant_context, world_manager,
        risk_precision,
        FLAGS_delta,
        FLAGS_T,
        FLAGS_use_max,
        uncertain_obstacle_ids, uncertain_obstacle_covariances
    );
    // Add the chance constraint to the program
    Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_row_major(q);
    Eigen::Map<drake::solvers::VectorXDecisionVariable> q_vectorized(q_row_major.data(), q_row_major.size());
    drake::solvers::Binding<drake::solvers::Constraint> chance_constraint_bound =
        prog->AddConstraint(collision_chance_constraint, q_vectorized);

    // We'll seed the nonlinear solver with the solution from the collision-free problem
    guess = VectorX<double>::Zero(prog->num_vars());
    prog->SetDecisionVariableValueInVector(
        q,
        collision_free_result.GetSolution(q),
        &guess
    );

    // That completes our setup for the chance-constrained mathematical program
    drake::log()->debug("Chance-constrained (obstacles only) program definition complete");

    // Before solving, we need to set SNOPT solver options
    drake::solvers::SolverOptions opts;
    // opts.SetOption(solver.solver_id(), "Print file", "ufo_snopt.out");
    // opts.SetOption(solver.solver_id(), "Verify level", 0);
    // opts.SetOption(solver.solver_id(), "Major optimality tolerance", sqrt(FLAGS_function_precision));
    // opts.SetOption(solver.solver_id(), "Major feasibility tolerance", sqrt(10*risk_precision));
    // opts.SetOption(solver.solver_id(), "Function precision", FLAGS_function_precision);

    // Now the fun part: we can finally solve the problem! (don't forget to measure runtime)
    drake::log()->debug("Solving chance-constrained program (obstacles only)...");
    drake::solvers::MathematicalProgramResult obstacle_only_result;
    auto start_time_obs = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
        obstacle_only_result = solver.Solve(*prog, guess, opts);
    }
    auto stop_time_obs = std::chrono::high_resolution_clock::now();
    auto duration_obs = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_obs - start_time_obs);
    drake::log()->info("======================================================================================");
    drake::log()->warn("Solved {} chance-constrained optimization problems (obstacles only), avg. duration {} ms",
                        FLAGS_num_benchmark_runs,
                        double(duration_obs.count()) / FLAGS_num_benchmark_runs);
    // drake::solvers::MathematicalProgramResult obstacle_only_result = solver.Solve(*prog, guess, opts);
    drake::log()->info("Success? {}", obstacle_only_result.is_success());
    // Visualize the results of the chance-constrained optimization, and report the risk incurred
    visualize_result(obstacle_only_result, q);

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
        FLAGS_T,
        uncertain_obstacle_ids, uncertain_obstacle_covariances,
        state_covariance
    );

    // Add the joint chance constraint to the program
    drake::solvers::VectorXDecisionVariable all_decision_variables(q_vectorized.rows() + 2);
    all_decision_variables << q_vectorized, risk_allocations;
    drake::solvers::Binding<drake::solvers::Constraint> joint_chance_constraint_bound =
        joint_prog->AddConstraint(joint_chance_constraint, all_decision_variables);

    // We'll seed the nonlinear solver with the solution from the collision-free problem
    guess = VectorX<double>::Zero(joint_prog->num_vars());
    joint_prog->SetDecisionVariableValueInVector(
        q,
        obstacle_only_result.GetSolution(q),
        &guess
    );

    // That completes our setup for the chance-constrained mathematical program
    drake::log()->debug("Chance-constrained (joint) program definition complete");

    // Now we get to solve it!
    drake::log()->debug("Solving chance-constrained program (joint)...");
    drake::solvers::MathematicalProgramResult joint_result;
    auto start_time_joint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
        joint_result = solver.Solve(*joint_prog, guess, opts);
    }
    auto stop_time_joint = std::chrono::high_resolution_clock::now();
    auto duration_joint = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_joint - start_time_joint);
    drake::log()->info("======================================================================================");
    drake::log()->warn("Solved {} chance-constrained optimization problems (joint), avg. duration {} ms",
                        FLAGS_num_benchmark_runs,
                        double(duration_joint.count()) / FLAGS_num_benchmark_runs);
    // drake::solvers::MathematicalProgramResult joint_result = solver.Solve(*joint_prog, guess, opts);
    drake::log()->info("Success? {}", joint_result.is_success());
    drake::log()->warn("Risk allocation:\n\tdelta = {}\n\tgamma = {}",
        joint_result.GetSolution(risk_allocations(0)),
        joint_result.GetSolution(risk_allocations(1)));

    // Visualize the results of the chance-constrained optimization, and report the risk incurred
    visualize_result(joint_result, q);

    drake::log()->warn("Risk bounds on collision-free seed: {}",
        collision_free_result.EvalBinding(chance_constraint_bound));
    drake::log()->warn("Risk bounds on chance-constrained trajectory (obstacles only): {}",
        obstacle_only_result.EvalBinding(chance_constraint_bound));
    drake::log()->warn("Risk bounds on chance-constrained trajectory (joint): {}",
        joint_result.EvalBinding(joint_chance_constraint_bound));

    drake::log()->warn("Objective for collision-free seed: {}",
        collision_free_result.get_optimal_cost());
    drake::log()->warn("Objective for chance-constrained trajectory (obstacles only): {}",
        obstacle_only_result.get_optimal_cost());
    drake::log()->warn("Objective for chance-constrained trajectory (joint): {}",
        joint_result.get_optimal_cost());

    // Validate with random trials
    std::vector<Eigen::Vector3d> uncertain_obstacle_nominal_positions;
    drake::Vector3<double> obstacle_1_pos{0.0, 0.5, 0.5};
    // drake::Vector3<double> obstacle_2_pos{0.5, -1.0, 0.3};
    uncertain_obstacle_nominal_positions.push_back(obstacle_1_pos);
    // uncertain_obstacle_nominal_positions.push_back(obstacle_2_pos);
    drake::log()->warn("Validating via Monte Carlo");
    drake::log()->warn("\t collision free seed:");
    validate_result(collision_free_result, q,
                    joint_chance_constraint);
    drake::log()->warn("\t obstacle chance-constrained:");
    validate_result(obstacle_only_result, q,
                    joint_chance_constraint);
    drake::log()->warn("\t joint chance-constrained:");
    validate_result(joint_result, q,
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