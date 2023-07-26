/*
 * Set up, solve, and visualize a trajectory optimization problem involving a no-collision constraint
 * as well as a chance constraint dealing with uncertain obstacles.
 */

#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <math.h>

#include "drake/common/proto/call_python.h"
#include "drake/common/pointer_cast.h"
#include "drake/common/scope_exit.h"
#include "drake/common/symbolic/decompose.h"

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include <drake/geometry/scene_graph_inspector.h>
#include <drake/geometry/geometry_set.h>
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/weld_joint.h"
#include "drake/multibody/inverse_kinematics/minimum_distance_constraint.h"

#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

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
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/decision_variable.h"

#include "drake/common/trajectories/bezier_curve.h"
#include "drake/geometry/optimization/graph_of_convex_sets.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/point.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/planning/trajectory_optimization/gcs_trajectory_optimization.h"

#include "drake/examples/scora/ccopt/chance_constraints.h"
#include "drake/examples/scora/ccopt/bezier_min_dist_constraint.h"

namespace drake {
namespace examples {
namespace scora {

using drake::multibody::MultibodyPlant;
using solvers::MathematicalProgram;
using symbolic::Expression;
using symbolic::MakeMatrixContinuousVariable;
using symbolic::MakeVectorContinuousVariable;
using solvers::VectorXDecisionVariable;
using trajectories::BezierCurve;
using trajectories::CompositeTrajectory;
using trajectories::Trajectory;

using Eigen::Vector2d;
using geometry::optimization::ConvexSets;
using geometry::optimization::GraphOfConvexSetsOptions;
using geometry::optimization::HPolyhedron;
using geometry::optimization::MakeConvexSets;
using geometry::optimization::Point;
using geometry::optimization::VPolytope;
using planning::trajectory_optimization::GcsTrajectoryOptimization;

DEFINE_double(function_precision, 0.001,
              "SNOPT option.");
DEFINE_double(delta, 0.2,
              "The maximum acceptable risk of collision over the entire trajectory.");
DEFINE_double(min_distance, 0.5,
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


void PlotResults(const Eigen::Matrix<double, 1, Eigen::Dynamic> xData, const Eigen::Matrix<double, 1, Eigen::Dynamic> yData) {
  using common::CallPython;
  using common::ToPythonTuple;

  Eigen::Matrix<int, 1, 4> diamondX {1, 0, -1, 0};
  Eigen::Matrix<int, 1, 4> diamondY {0, 1, 0, -1};
  Vector2d start(0.0, -2.0), goal(0.0, 2.0);
  
  CallPython("figure", 1);
  CallPython("clf");

  CallPython("plot", start[0], start[1], "b*");
  CallPython("plot", goal[0], goal[1], "g*");

  CallPython("plot", xData.transpose(), yData.transpose(), "b");
  CallPython("fill", diamondX.transpose(), diamondY.transpose(), "r");
  CallPython("axis", "equal");

  // Give time for Python to plot.
  std::this_thread::sleep_for(std::chrono::seconds(1));
}


bool GurobiOrMosekSolverAvailable() {
  return (solvers::MosekSolver::is_available() &&
          solvers::MosekSolver::is_enabled()) ||
         (solvers::GurobiSolver::is_available() &&
          solvers::GurobiSolver::is_enabled());
}

// Load a plant for this arm planning problem
void load_plant_components(MultibodyPlant<double>& plant, bool fix_obstacles) {
    // Load the obstacle course model into the MultibodyPlant
    std::string obstacle_course_model_file = "./examples/scora/models/obstacle_course_simplified.urdf";
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
        drake::Vector3<double> obstacle_1_pos{0, 0, 1.0};
        plant.AddJoint<multibody::WeldJoint>("weld_obstacle_1", floor_root, {},
                                             obstacle_1_root, {},
                                             drake::math::RigidTransform(obstacle_1_pos));

        // const auto& obstacle_2_root = plant.GetBodyByName("obstacle_2");
        // drake::Vector3<double> obstacle_2_pos{0.5, 0.75, 1.0};
        // plant.AddJoint<multibody::WeldJoint>("weld_obstacle_2", floor_root, {},
        //                                      obstacle_2_root, {},
        //                                      drake::math::RigidTransform(obstacle_2_pos));
    }

    // Load the cube into the MultibodyPlant (this cube will act as the robot navigating through space)
    std::string full_name = "./examples/scora/models/movable_box_point.urdf";
    plant_index = multibody::Parser(&plant).AddModelFromFile(full_name);

    // The cube is free-floating with x, y, z coordinates relative to the world origin, so no
    // further welding is needed
}


void visualize_result(
        drake::solvers::MathematicalProgramResult result, CompositeTrajectory<double> traj) {

    if (!result.is_success()){

        std::cout << "Solver failed..." << std::endl;
        return;
    }

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

    int T = 20;
    double timestep = FLAGS_traj_duration / T;
    int numberOfSegments = traj.get_number_of_segments();
    double multiplier = 5.0 / (1 * numberOfSegments);

    for (int segmentID = 0; segmentID < numberOfSegments; segmentID++)
    {
        for (int t = 0; t < T; t++)
        {
            double tStep= segmentID * 1 + static_cast<double>(t) / T;
            auto coords = traj.segment(segmentID).value(tStep);
            std::cout << "X : " << coords(0) << ", Y : " << coords(1) << " Z : " << coords(2) << std::endl;
            
            t_solution.push_back(tStep * multiplier);
            q_solution.push_back(traj.segment(segmentID).value(tStep));
        }
    }
    // Add the last point.
    t_solution.push_back(numberOfSegments * multiplier);
    q_solution.push_back(traj.segment(numberOfSegments - 1).value(numberOfSegments));

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
    auto lcm = viz_builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
    geometry::DrakeVisualizerd::AddToBuilder(&viz_builder, viz_scene_graph, lcm);
    std::unique_ptr<systems::Diagram<double>> viz_diagram = viz_builder.Build();

    drake::log()->debug("Visualizer built");

    // Set up simulator.
    systems::Simulator<double> simulator(*viz_diagram);
    simulator.set_publish_every_time_step(true);
    simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
    simulator.Initialize();
    simulator.AdvanceTo(T * timestep);
}

void visualize_result(
        drake::solvers::MathematicalProgramResult result, std::vector<MatrixX<symbolic::Variable>> segment_controls) {

    if (!result.is_success()){
        std::cout << "Solver failed..." << std::endl;
        return;
    }

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

    for (int i = 0; i < static_cast<int>(segment_controls.size()); i++)
    {
        auto optimal_coeffs = result.GetSolution(segment_controls[i]);
        auto optimal_trajectory = BezierCurve<double>(0, 1, optimal_coeffs);

        for (int t = 0; t < T; t++) {
            double tStep= static_cast<double>(t) / T;

            std::cout << "t: " << (i * T + t) << ", point: " << optimal_trajectory.value(tStep).transpose() << std::endl;

            t_solution.push_back((i + tStep) * 5 / segment_controls.size());
            q_solution.push_back(optimal_trajectory.value(tStep));
        }
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
    auto lcm = viz_builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
    geometry::DrakeVisualizerd::AddToBuilder(&viz_builder, viz_scene_graph, lcm);
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
        int num_positions,
        drake::solvers::MathematicalProgramResult result,
        std::vector<MatrixX<symbolic::Variable>> segment_controls,
        std::shared_ptr<ccopt::CollisionChanceConstraint> chance_constraint) {

    int T = FLAGS_T;
    Eigen::MatrixXd q_trajectory(num_positions, T * segment_controls.size());
    for (int i = 0; i < static_cast<int>(segment_controls.size()); i++)
    {
        auto optimal_coeffs = result.GetSolution(segment_controls[i]);
        auto optimal_trajectory = BezierCurve<double>(0, 1, optimal_coeffs);

        for (int t = 0; t < T; t++) {
            double tStep= static_cast<double>(t) / T;
            q_trajectory.col(i * T + t) = optimal_trajectory.value(tStep);
        }
    }

    // Extract the solution as a matrix (columns = timesteps, rows = DOFs)
    // Eigen::MatrixXd q_trajectory = q_solution;
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
    double failure_rate = static_cast<float>(n_failures) / static_cast<float>(FLAGS_N_check);
    drake::log()->info("FAILURE RATE: {}", failure_rate);
    return failure_rate;
}

bool VerifyTrajectoryInConvexSetsConvexHull(drake::solvers::MathematicalProgramResult result, std::vector<MatrixX<symbolic::Variable>> segment_controls, ConvexSets regions_){
    
    if (!result.is_success()){
        return false;
    }
    
    bool trajectoryInConvexSets = true;
    // Iterate over each segment of the trajectory
    for (int i = 0; i < static_cast<int>(segment_controls.size()); i++){
        // auto optimalCoeffs = traj_control_points[i];
        // auto optimalCoeffs = collision_free_result.GetSolution(segment_controls[i]);
        auto optimalCoeffs = result.GetSolution(segment_controls[i]);
        
        bool segmentInSet = true;
        // Iterate over each control point.
        for (int j = 0; j < segment_controls[i].cols(); j++){
            // Iterate over all convex sets
            bool pointIsInSet = false;
            int regionNumber = 0;
            for (auto set : regions_)
            {
                bool check = set->PointInSet(optimalCoeffs.col(j), 1e-6);
                // std::cout << "i: " << i << ", j: " << j << " region : " << regionNumber << ", check : " << check << std::endl;
                regionNumber++;
                if (check)
                {
                    pointIsInSet = true;
                    break;
                }
            }

            if (!pointIsInSet){
                segmentInSet = false;
                break;
            }
        }

        if (!segmentInSet){
            trajectoryInConvexSets = false;
            break;
        }
    }

    return trajectoryInConvexSets;
}

bool VerifyTrajectoryInConvexSetsSampling(drake::solvers::MathematicalProgramResult result, std::vector<MatrixX<symbolic::Variable>> segment_controls, ConvexSets regions_){
    if (!result.is_success()){
        return false;
    }

    bool trajectoryInConvexSets = true;
    // Iterate over each segment of the trajectory
    for (int i = 0; i < static_cast<int>(segment_controls.size()); i++){
        // auto optimalCoeffs = traj_control_points[i];
        auto optimalCoeffs = result.GetSolution(segment_controls[i]);
        auto optimalTrajectory = BezierCurve<double>(0, 1, optimalCoeffs);

        bool segmentInSet = true;
        // Iterate over FLAGS_T_check points for each segment
        for (int j = 0; j <= FLAGS_T_check; j++){
            // Iterate over all convex sets
            double tStep = static_cast<double>(j) / FLAGS_T_check;
            bool pointIsInSet = false;
            int regionNumber = 0;
            for (auto set : regions_)
            {
                bool check = set->PointInSet(optimalTrajectory.value(tStep), 1e-6);
                // std::cout << "i: " << i << ", j: " << j << " region : " << regionNumber << ", check : " << check << std::endl;
                regionNumber++;
                if (check)
                {
                    pointIsInSet = true;
                    break;
                }
            }

            if (!pointIsInSet){
                segmentInSet = false;
                break;
            }
        }

        if (!segmentInSet){
            trajectoryInConvexSets = false;
            break;
        }
    }

    return trajectoryInConvexSets;
}

void DoMain() {

    if(!GurobiOrMosekSolverAvailable()){
        std::cout << "Cannot find Gurobi or Mosek, terminating..." << std::endl;
        return;
    }

    DRAKE_DEMAND(FLAGS_simulation_time > 0);
    drake::log()->set_level(spdlog::level::info);

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
    drake::log()->debug("Plant finalized");

    // Build the plant
    std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();
    // Create a diagram-level context
    std::unique_ptr<drake::systems::Context<double>> diagram_context = diagram->CreateDefaultContext();
    drake::log()->debug("Plant built");

    // Then create a subsystem context for the multibodyplant
    drake::systems::Context<double>* plant_context =
        &diagram->GetMutableSubsystemContext(plant, diagram_context.get());
        
    const int kDimension = 3;
    const double kMinimumDuration = 1.0;
    GcsTrajectoryOptimization gcs(kDimension);

    Eigen::Vector3d start(0.0, -2.0, 1.0), goal(0.0, 2.0, 1.0);

    Eigen::Matrix<double, 3, 3> A_bl;
    A_bl << -1, 0, 0, 0, -1, 0, 1, 1, 0;
    Eigen::Matrix<double, 3, 3> A_br;
    A_br << 1, 0, 0, 0, -1, 0, -1, 1, 0;
    Eigen::Matrix<double, 3, 3> A_tl;
    A_tl << -1, 0, 0, 0, 1, 0, 1, -1, 0;
    Eigen::Matrix<double, 3, 3> A_tr;
    A_tr << 1, 0, 0, 0, 1, 0, -1, -1, 0;
    Eigen::Vector3d b(3, 3, -1);

    HPolyhedron region_1(A_bl, b);
    HPolyhedron region_2(A_br, b);
    HPolyhedron region_3(A_tl, b);
    HPolyhedron region_4(A_tr, b);

    ConvexSets regions_ = MakeConvexSets(region_1, region_2, region_3, region_4);
    // ConvexSets regions_ = MakeConvexSets(region_1, region_3);

    auto& regions = gcs.AddRegions(regions_, 1, kMinimumDuration);
    auto& source = gcs.AddRegions(MakeConvexSets(Point(start)), 0);
    auto& target = gcs.AddRegions(MakeConvexSets(Point(goal)), 0);

    gcs.AddEdges(source, regions);
    gcs.AddEdges(regions, target);

    gcs.AddPathLengthCost(1.0);
    gcs.AddTimeCost(1.0);

    GraphOfConvexSetsOptions options;
    options.max_rounded_paths = 3;
    // drake::solvers::SnoptSolver sn_solver;
    // options.solver = &sn_solver;

    // double risk_precision = 0.0000001;
    // double function_precision = 0.0001;
    // drake::solvers::SolverOptions solver_options_;
    // solver_options_.SetOption(drake::solvers::SnoptSolver::id(), "Major iterations limit", 1000000);
    // solver_options_.SetOption(drake::solvers::SnoptSolver::id(), "Major optimality tolerance", sqrt(function_precision));
    // solver_options_.SetOption(drake::solvers::SnoptSolver::id(), "Major feasibility tolerance", sqrt(10 * risk_precision));
    // solver_options_.SetOption(drake::solvers::SnoptSolver::id(), "Function precision", function_precision);
    // options.solver_options = solver_options_;

    // gcs.AddMinDistanceConstraint(plant, plant_context, FLAGS_min_distance, FLAGS_T);

    auto [traj, traj_control_points, result] = gcs.SolvePath(source, target, options);

    // std::cout << "Number of segments : " << traj.get_number_of_segments() << std::endl;

    //////////////////////////////////////////////////////
    /////// PLOTTING CODE 
    //////////////////////////////////////////////////////

    // Remove the first and last segments, i.e. the source and target segment since those are trivial.
    int numberOfSegments = traj.get_number_of_segments();

    Eigen::Matrix<double, 1, Eigen::Dynamic> x_data;
    Eigen::Matrix<double, 1, Eigen::Dynamic> y_data;

    int numPoints = FLAGS_T;
    int numIntervals = numPoints - 1;

    x_data.resize(1, numPoints * numberOfSegments);
    y_data.resize(1, numPoints * numberOfSegments);

    std::cout << "START : " << traj.start_time() << ", END : " << traj.end_time() << std::endl;

    for (int segmentID = 0; segmentID < numberOfSegments; segmentID++)
    {
        double start_time = traj.segment(segmentID).start_time();
        double end_time = traj.segment(segmentID).end_time();
        double timeStepSize = (end_time - start_time) / numIntervals;

        // std::cout << "Start time : " << start_time << std::endl;
        // std::cout << "End time : " << end_time << std::endl;

        for (int i = 0; i < numPoints; i++)
        {
        double timeStep = start_time + timeStepSize * i;
        auto coords = traj.segment(segmentID).value(timeStep);

        // std::cout << "X : " << coords(0) << ", Y : " << coords(1) << " Z : " << coords(2) << std::endl;

        x_data(1, segmentID * numPoints + i) = coords(0);
        y_data(1, segmentID * numPoints + i) = coords(1);
        }
    }

    (void) plant_context;
    // PlotResults(x_data, y_data);
    visualize_result(result, traj);

    // std::cout << "Size : " << traj_control_points[0].rows() << " x " << traj_control_points[0].cols() << std::endl;
    // Now that we have a GCS trajectory, use this as a starting point and add
    // a minimum distance constraint to the trajectory.    
    std::unique_ptr<drake::solvers::MathematicalProgram> prog =
        std::make_unique<drake::solvers::MathematicalProgram>();

    const int order = 1;
    const int numSegments = traj_control_points.size();
    int num_positions = plant.num_positions();
    // Let's define some number of timesteps and assume a fixed time step
    int T = FLAGS_T;

    // To define the trajectory, we need a decision variable for the x, y, z coordinates
    // at each timestep. We'll do a row for each timestep and a column for each joint
    
    std::vector<MatrixX<symbolic::Variable>> segment_controls(numSegments);
    std::vector<drake::trajectories::BezierCurve<drake::symbolic::Expression>> segment_trajectories(numSegments);
    for (int i = 0; i < numSegments; i++)
    {
        segment_controls[i] = prog->NewContinuousVariables(num_positions, order + 1, "xu_" + std::to_string(i));
        segment_trajectories[i] = BezierCurve<Expression>(0, 1, segment_controls[i].cast<drake::symbolic::Expression>());
    }

    for (int i = 0; i < (numSegments - 1); i++)
    {
        prog->AddLinearEqualityConstraint(segment_controls[i].col(order) == segment_controls[i + 1].col(0));
    }

    // We want to constrain the start and end positions. The start position should be strict
    // equality, but we can make do with a bounding box on the goal position.
    //
    // We want the cube to start below the obstacle course and navigate to the other side.
    Eigen::MatrixXd start_ = VectorX<double>::Zero(num_positions);
    start_(0) = 0.0;
    start_(1) = -2.0;
    start_(2) = 1.0;
    // We make this constraint a strict equality since we don't usually have the luxury of
    // changing the start position.
    
    Eigen::MatrixXd end_ = VectorX<double>::Zero(num_positions);
    end_(0) = 0.0;
    end_(1) = 2.0;
    end_(2) = 1.0;

    Eigen::MatrixXd goal_margin = 0.01 * VectorX<double>::Ones(num_positions);
    auto xvars = prog->NewContinuousVariables(T * numSegments, num_positions, "xvars");
    
    prog->AddBoundingBoxConstraint(start_.transpose(), start_.transpose(), xvars.row(0));
    prog->AddBoundingBoxConstraint((end_ - goal_margin).transpose(), (end_ + goal_margin).transpose(), xvars.row(T * numSegments -1));

    for (int t = 1; t < (T * numSegments); t++) {
        prog->AddQuadraticCost(0.5 * (xvars.row(t) - xvars.row(t-1)).dot(xvars.row(t) - xvars.row(t-1)));
    }

    for (int t = 0; t < (T * numSegments); t++) {
        prog->AddBoundingBoxConstraint(start(2), start(2), xvars(t, 2));
    }

    for (int segNum = 0; segNum < numSegments; segNum++)
    {
        for (int t = 0; t < T; t++) {
            auto no_collision_constraint = std::make_shared<multibody::MinimumDistanceConstraint>(
                &plant,
                FLAGS_min_distance,
                plant_context
            );

            double tStep= static_cast<double>(t) / T;
            prog->AddLinearEqualityConstraint(xvars.row(segNum * T + t) == segment_trajectories[segNum].value(tStep).transpose());
            prog->AddConstraint(no_collision_constraint, xvars.row(segNum * T + t));
        }
    }

    for (int i = 0; i < numSegments; i++)
    {
        prog->SetInitialGuess(segment_controls[i], traj_control_points[i]);
    }
    // // That completes our setup for this mathematical program
    drake::log()->debug("Deterministic program definition complete");

    // // We can solve the collision-free problem to provide a seed for the chance-constrained
    // // problem.
    drake::solvers::SnoptSolver solver;
    drake::solvers::SolverOptions options_;
    prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Major iterations limit", 1000000);

    drake::log()->debug("Solving deterministic program...");
    drake::solvers::MathematicalProgramResult collision_free_result = solver.Solve(*prog);
    drake::log()->debug("Deterministic program solved");

    // We'll eventually run a second optimization problem to deal with risk, but let's start
    // with a simple sanity check on the collision-free solution
    visualize_result(collision_free_result, segment_controls);

    // To check the probability of collision between the robot and the environment,
    // we need to define the list of bodies that make up the "robot", which we save
    // in the Bullet world manager
    ccopt::BulletWorldManager<double>* world_manager = new ccopt::BulletWorldManager<double>();
    std::vector<std::string> robot_body_names{
        "box_link_1"
    };
    for (const std::string body_name : robot_body_names) {
        const std::vector<drake::geometry::GeometryId> robot_ids =
            plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name));
        world_manager->AddRobotGeometryIds(robot_ids);
        std::cout << "=====================\n" << body_name << std::endl;
        for (auto id : robot_ids) {
            drake::log()->info("ID: {}", id);
        }
    }

    const std::vector<drake::geometry::GeometryId> floor_ids =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName("floor"));
    std::cout << "=====================\n" << "floor" << std::endl;
    for (auto id : floor_ids) {
        drake::log()->info("ID: {}", id);
    }

    // // We also need to define the bodies that are uncertain.
    std::vector<std::string> uncertain_obstacle_names{
        "obstacle_1",
        // "obstacle_2"
    };
    std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids;
    for (const std::string name : uncertain_obstacle_names) {
        std::cout << "=====================\n" << name << std::endl;
        // Get the geometry IDs corresponding to this obstacle
        std::vector<drake::geometry::GeometryId> obstacle_ids =
            plant.GetCollisionGeometriesForBody(
                plant.GetBodyByName(name));
        for (auto id : obstacle_ids) {
            drake::log()->info("ID: {}", id);
        }
        // Append that to the vector of uncertain_obstacle_ids
        uncertain_obstacle_ids.insert(uncertain_obstacle_ids.end(),
                                      obstacle_ids.begin(),
                                      obstacle_ids.end());
    }

    // // Let's make both pillars uncertain in the x direction.
    Eigen::Matrix3d uncertain_obstacle_covariance;
    uncertain_obstacle_covariance << 0.1, 0.0, 0.0,
                                     0.0, 0.1, 0.0,
                                     0.0, 0.0, 0.1;
    // // Make a vector of n copies of the covariance, where n = the number of uncertain
    // // geometry IDs found above
    std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances;

    for (int i = 0; i < static_cast<int>(uncertain_obstacle_ids.size()); i++){
        uncertain_obstacle_covariances.push_back(uncertain_obstacle_covariance);
    }

    // // Now we can add the chance constraint
    // // First define the precision we want from our risk estimates
    double risk_precision = 0.0000001; // 10^-7

    // // Next add a chance constraint covering the entire trajectory
    auto collision_chance_constraint = std::make_shared<ccopt::CollisionChanceConstraint>(
        &plant, plant_context, world_manager,
        risk_precision,
        FLAGS_delta,
        numSegments * FLAGS_T,
        FLAGS_use_max,
        uncertain_obstacle_ids, uncertain_obstacle_covariances
    );
    // // Add the chance constraint to the program
    Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xvars_row_major(xvars);
    Eigen::Map<drake::solvers::VectorXDecisionVariable> xvars_vectorized(xvars_row_major.data(), xvars_row_major.size());
    drake::solvers::Binding<drake::solvers::Constraint> chance_constraint_bound =
        prog->AddConstraint(collision_chance_constraint, xvars_vectorized);

    // // We'll seed the nonlinear solver with the solution from the minimum distance constraint problem
    for (int i = 0; i < numSegments; i++)
        prog->SetInitialGuess(segment_controls[i], collision_free_result.GetSolution(segment_controls[i]));

    // // That completes our setup for the chance-constrained mathematical program
    drake::log()->debug("Chance-constrained program definition complete");

    // // Before solving, we need to set SNOPT solver options
    // drake::solvers::SolverOptions opts;
    // opts.SetOption(solver.solver_id(), "Print file", "ufo_snopt.out");
    // opts.SetOption(solver.solver_id(), "Verify level", 0);
    // opts.SetOption(solver.solver_id(), "Major optimality tolerance", sqrt(FLAGS_function_precision));
    // opts.SetOption(solver.solver_id(), "Major feasibility tolerance", sqrt(10*risk_precision));
    // opts.SetOption(solver.solver_id(), "Function precision", FLAGS_function_precision);

    prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Print file", "ufo_snopt.out");
    prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Verify level", 0);
    prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Major optimality tolerance", sqrt(FLAGS_function_precision));
    prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Major feasibility tolerance", sqrt(10*risk_precision));
    prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Function precision", FLAGS_function_precision);

    // // Now the fun part: we can finally solve the problem! (don't forget to measure runtime)
    drake::log()->debug("Solving chance-constrained program...");
    auto start_time = std::chrono::high_resolution_clock::now();
    drake::solvers::MathematicalProgramResult chance_constrained_result;
    for (int i = 0; i < FLAGS_num_benchmark_runs; i++) {
        // result = solver.Solve(*prog, guess, opts);
        chance_constrained_result = solver.Solve(*prog);
    }

    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
    drake::log()->info("======================================================================================");
    drake::log()->info("Solved {} chance-constrained optimization problems, avg. duration {} ms",
                        FLAGS_num_benchmark_runs,
                        double(duration.count()) / FLAGS_num_benchmark_runs);
    drake::log()->info("Success? {}", chance_constrained_result.is_success());

    // Visualize the results of the chance-constrained optimization, and report the risk incurred
    visualize_result(chance_constrained_result, segment_controls);

    // Validate with random trials
    // std::vector<Eigen::Vector3d> uncertain_obstacle_nominal_positions;
    // drake::Vector3<double> obstacle_1_pos{0, 0, 1.0};
    // // // drake::Vector3<double> obstacle_2_pos{0.5, 0.75, 1.0};
    // uncertain_obstacle_nominal_positions.push_back(obstacle_1_pos);
    // // // uncertain_obstacle_nominal_positions.push_back(obstacle_2_pos);
    // validate_result(num_positions, chance_constrained_result, segment_controls,
    //                 collision_chance_constraint);
    // drake::log()->info("Risk on collision-free seed: {}",
    //     collision_free_result.EvalBinding(chance_constraint_bound));
    // drake::log()->info("Risk on chance-constrained trajectory: {}",
    //     chance_constrained_result.EvalBinding(chance_constraint_bound));


    // Verify whether the trajectory is within the convex regions.
    // First method using the convex hull property of bezier curves
    bool trajectoryInConvexSets = VerifyTrajectoryInConvexSetsConvexHull(chance_constrained_result, segment_controls, regions_);
    std::cout << "Trajectory in convex sets : " << trajectoryInConvexSets << std::endl;

    // Second method by sampling a bunch of points and checking if each point lies in the convex regions.
    trajectoryInConvexSets = VerifyTrajectoryInConvexSetsSampling(chance_constrained_result, segment_controls, regions_);
    std::cout << "Trajectory in convex sets : " << trajectoryInConvexSets << std::endl;

    // This is slightly more work so I'll implement it later.
    // trajectoryInConvexSets = VerifyTrajectoryInConvexSetsSOSConstraint(collision_free_result, segment_controls, regions_);
    // std::cout << "Trajectory in convex sets : " << trajectoryInConvexSets << std::endl;
}

}  // namespace scora
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::scora::DoMain();
  return 0;
}