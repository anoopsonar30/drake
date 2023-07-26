/// @file
///
/// Implements a simulation of the KUKA iiwa arm.  Like the driver for the
/// physical arm, this simulation communicates over LCM using lcmt_iiwa_status
/// and lcmt_iiwa_command messages. It is intended to be a be a direct
/// replacement for the KUKA iiwa driver and the actual robot hardware.

#include <memory>
#include <iostream>
#include <cmath>
#include <map>
#include <unordered_map>
#include <omp.h>
#include <vector>
#include <thread>
#include <chrono>
#include <future>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "drake/examples/gcs/manipulation/ThreadPool.h"

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/gcs/manipulation/iiwa_common.h"
#include "drake/examples/gcs/manipulation/iiwa_lcm.h"
#include "drake/examples/gcs/manipulation/kuka_torque_controller.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/multibody/inverse_kinematics/minimum_distance_constraint.h"

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/parsing/process_model_directives.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/controllers/state_feedback_controller_interface.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/systems/primitives/constant_value_source.h"

#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

#include "drake/geometry/optimization/iris.h"
#include "drake/geometry/optimization/hpolyhedron.h"

#include <drake/solvers/mathematical_program_result.h>
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

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/copyable_unique_ptr.h"

#include "drake/examples/scora/ccopt/chance_constraints.h"

DEFINE_double(simulation_sec, std::numeric_limits<double>::infinity(),
              "Number of seconds to simulate.");
DEFINE_double(traj_duration, 2,
              "The total duration of the trajectory (in seconds).");
DEFINE_string(urdf, "", "Name of urdf to load");
DEFINE_double(target_realtime_rate, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_bool(torque_control, false, "Simulate using torque control mode.");
DEFINE_double(sim_dt, 3e-3,
              "The time step to use for MultibodyPlant model "
              "discretization.");
DEFINE_int32(T, 10,
             "The number of timesteps used to define the trajectory.");
DEFINE_double(min_distance, 0.05,
              "The minimum allowable distance between collision bodies during the trajectory.");
DEFINE_double(function_precision, 0.0001,
              "SNOPT option.");
DEFINE_int32(num_benchmark_runs, 1,
             "The number of times which the optimization problem should be solved to measure its runtime.");
DEFINE_double(delta, 0.2,
              "The maximum acceptable risk of collision over the entire trajectory.");
DEFINE_bool(use_max, false,
             "If true, only the maximum waypoint risk over the entire trajectory is constrained.");

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace gcs{
using multibody::MultibodyPlant;
using systems::Context;
using systems::StateInterpolatorWithDiscreteDerivative;
using systems::Simulator;
using systems::controllers::InverseDynamicsController;
using systems::controllers::StateFeedbackControllerInterface;
using multibody::parsing::LoadModelDirectives;
using multibody::parsing::ModelDirectives;
using multibody::parsing::ModelInstanceInfo;
using geometry::SceneGraph;
using multibody::MultibodyPlant;
using multibody::Parser;
using solvers::Solve;
using multibody::InverseKinematics;

using geometry::MeshcatVisualizerParams;
using geometry::MeshcatVisualizer;

using geometry::optimization::IrisOptions;
using geometry::optimization::IrisInConfigurationSpace;

using solvers::MathematicalProgram;
using symbolic::Expression;
using symbolic::MakeMatrixContinuousVariable;
using symbolic::MakeVectorContinuousVariable;
using solvers::VectorXDecisionVariable;
using trajectories::BezierCurve;
using trajectories::CompositeTrajectory;
using trajectories::Trajectory;

using Eigen::Vector2d;
using geometry::optimization::ConvexSet;
using geometry::optimization::ConvexSets;
using geometry::optimization::GraphOfConvexSetsOptions;
using geometry::optimization::HPolyhedron;
using geometry::optimization::MakeConvexSets;
using geometry::optimization::Point;
using geometry::optimization::VPolytope;
using planning::trajectory_optimization::GcsTrajectoryOptimization;

// Save the regions to a binary file
void saveRegions(const std::unordered_map<std::string, HPolyhedron>& regions) {
    for (const auto& entry : regions) {
        std::string seed_name = entry.first;
        const HPolyhedron region = entry.second;

        // Replaces spaces with underscores
        std::replace(seed_name.begin(), seed_name.end(), ' ', '_');

        std::filesystem::path filePath = std::filesystem::path("/home/drparadox30/petersen_home") / (seed_name + ".yaml");

        std::ofstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + seed_name);
        }

        drake::yaml::SaveYamlFile(filePath.string(), region);
        std::cout << "Saving seed : " << filePath.string() << std::endl;
    }
}

// Load the regions from a binary file
std::unordered_map<std::string, HPolyhedron> loadRegions(const std::string& folderPath) {
    std::unordered_map<std::string, HPolyhedron> regions;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        const std::filesystem::path& filePath = entry.path();
        if (filePath.extension() == ".yaml") {
            std::ifstream file(filePath.string(), std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file for reading: " + filePath.string());
            }

            HPolyhedron region = drake::yaml::LoadYamlFile<HPolyhedron>(filePath.string());
            std::string seedName = filePath.stem().string();
            std::cout << "Loaded : " << seedName << std::endl;
            regions.emplace(std::move(seedName), std::move(region));            
        }
    }

    return regions;
}

Eigen::VectorXd doInverseKinematics(const Eigen::VectorXd& q0, const Eigen::Vector3d& translation, const Eigen::Vector3d& rpy) {

  systems::DiagramBuilder<double> builder;

  // Adds a plant.
  auto [plant, scene_graph] = multibody::AddMultibodyPlantSceneGraph(
      &builder, FLAGS_sim_dt);
  
  auto parser = multibody::Parser(&plant, &scene_graph);
  const std::string directives_file = FindResourceOrThrow("drake/examples/gcs/manipulation/models/iiwa14_spheres_collision_welded_gripper.yaml");
  const ModelDirectives directives = LoadModelDirectives(directives_file);
  const std::vector<ModelInstanceInfo> models = ProcessModelDirectives(directives, &parser);

  plant.Finalize();

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto plant_context = &plant.GetMyMutableContextFromRoot(context.get());

  const auto& gripper_frame = plant.GetBodyByName("body").body_frame();
  InverseKinematics ik(plant, plant_context);

  ik.AddPositionConstraint(gripper_frame, Eigen::Vector3d::Zero(), plant.world_frame(), translation, translation);
  
  drake::math::RotationMatrix<double> identity_rotation = drake::math::RotationMatrix<double>::Identity();
  drake::math::RollPitchYaw<double> rpy_angles(rpy);
  drake::math::RotationMatrix<double> target_rotation(rpy_angles);

  ik.AddOrientationConstraint(gripper_frame, identity_rotation, plant.world_frame(), target_rotation, 0.001);

  auto prog = ik.get_mutable_prog();
  auto q = ik.q();

  prog->AddQuadraticErrorCost(Eigen::MatrixXd::Identity(q.size(), q.size()), q0, q);
  prog->SetInitialGuess(q, q0);
  auto result = Solve(*prog);

  if (!result.is_success()) {
    std::cout << "IK failed" << std::endl;
    return Eigen::VectorXd();
  }
  
  Eigen::VectorXd q1 = result.GetSolution(q);
  return q1;
}

// Function to calculate the region for a single seed
HPolyhedron calcRegion(Eigen::VectorXd& seed, bool verbose, const drake::systems::Diagram<double>& diagram,
                       const drake::multibody::MultibodyPlant<double>& plant,
                       const IrisOptions& iris_options) {
    auto context = diagram.CreateDefaultContext();
    auto plant_context = &plant.GetMyMutableContextFromRoot(context.get());
    plant.SetPositions(plant_context, seed);
    
    auto start_time = std::chrono::steady_clock::now();
    auto hpoly = IrisInConfigurationSpace(plant, *plant_context, iris_options);
    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time);
    
    if (verbose) {
        std::cout << "Seed: " << seed << "\tTime: " << elapsed_time.count() << " s" << " Faces : " << hpoly.b().size() << std::endl;
    }
    return hpoly;
}

// Function to generate regions for multiple seeds
std::unordered_map<std::string, HPolyhedron> generateRegions(
        const std::unordered_map<std::string, Eigen::VectorXd>& seed_points, const IrisOptions& iris_options,
        bool verbose = true) {
    std::vector<Eigen::VectorXd> seeds;
    seeds.reserve(seed_points.size());
    std::vector<HPolyhedron> regions(seed_points.size());
    std::transform(seed_points.begin(), seed_points.end(), std::back_inserter(seeds),
                   [](const auto& entry) { return entry.second; });

    // Create a DiagramBuilder for building the system
    drake::systems::DiagramBuilder<double> builder;
    // Add a MultibodyPlant and SceneGraph to the diagram
    auto [plant, scene_graph] = drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
    // Create a Parser and load the model directives
    Parser parser(&plant, &scene_graph);
    const std::string directives_file = FindResourceOrThrow(
        "drake/examples/gcs/manipulation/models/iiwa14_spheres_collision_welded_gripper.yaml");
    const ModelDirectives directives = LoadModelDirectives(directives_file);
    const std::vector<ModelInstanceInfo> models =
        ProcessModelDirectives(directives, &parser);

    plant.Finalize();

    // Build the diagram and get the context
    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();
    // auto plant_context = &plant.GetMyContextFromRoot(*context.get());

    auto loop_start_time = std::chrono::steady_clock::now();

    const unsigned int NUM_THREADS = std::thread::hardware_concurrency() * 0.75;
    std::cout << "Number of cores : " << NUM_THREADS << std::endl;
    // Generate regions for each seed in parallel
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < int(seeds.size()); ++i) {
        regions[i] = calcRegion(seeds[i], verbose, *diagram, plant, iris_options);
    }

    auto loop_end_time = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::seconds>(
        loop_end_time - loop_start_time).count();
    if (verbose) {
        std::cout << "Loop time: " << loop_time << " s" << std::endl;
    }

    // Combine the results into a map
    std::unordered_map<std::string, HPolyhedron> result;
    for (const auto& entry : seed_points) {
        result[entry.first] = regions[std::distance(seeds.begin(), std::find(seeds.begin(), seeds.end(), entry.second))];
      }
    
    return result;
}

void VisualizeTrajectory(CompositeTrajectory<double> traj, solvers::MathematicalProgramResult result, Eigen::VectorXd start_state){

  if (!result.is_success()){

      std::cout << "Solver failed..." << std::endl;
      return;
  }
  else
  {
    std::cout << "Program solved successfully!" << std::endl;
  }

  std::cout << "Number of segments : " << traj.get_number_of_segments() << std::endl;

  std::vector<double> t_solution;
  std::vector<Eigen::MatrixXd> q_solution;

  int T = 50;
  double timestep = FLAGS_traj_duration / T;
  int numberOfSegments = traj.get_number_of_segments();
  double multiplier = FLAGS_traj_duration / (1 * numberOfSegments);

  for (int segmentID = 0; segmentID < numberOfSegments; segmentID++)
  {
      for (int t = 0; t < T; t++)
      {
          double tStep= segmentID * 1 + static_cast<double>(t) / T;
          auto coords = traj.segment(segmentID).value(tStep);
          // std::cout << coords.transpose() << std::endl;
          t_solution.push_back(tStep * multiplier);
          q_solution.push_back(traj.segment(segmentID).value(tStep));
      }
  }

  // Add the last point.
  t_solution.push_back(numberOfSegments * multiplier);
  q_solution.push_back(traj.segment(numberOfSegments - 1).value(numberOfSegments));
  // std::cout << traj.segment(numberOfSegments - 1).value(numberOfSegments).transpose() << std::endl;

  for (auto elem : t_solution)
    std::cout << elem << " ";
  std::cout << std::endl;
 
  for (auto elem : q_solution) 
    std::cout << elem.transpose() << " " << std::endl;
  std::cout << std::endl;

  drake::trajectories::PiecewisePolynomial<double> trajectory_solution = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(t_solution, q_solution);

  systems::DiagramBuilder<double> builder;
  // Adds a plant.
  auto [plant, scene_graph] = multibody::AddMultibodyPlantSceneGraph(
      &builder, FLAGS_sim_dt);
  
  auto parser = multibody::Parser(&plant, &scene_graph);
  const std::string directives_file = FindResourceOrThrow("drake/examples/gcs/manipulation/models/iiwa14_spheres_collision_welded_gripper.yaml");
  const ModelDirectives directives = LoadModelDirectives(directives_file);
  const std::vector<ModelInstanceInfo> models = ProcessModelDirectives(directives, &parser);

  // const auto [iiwa_instance, wsg, shelf, binR, binL, table] = std::make_tuple(models[0].model_instance,
  //                                                                   models[1].model_instance,
  //                                                                   models[2].model_instance,
  //                                                                   models[3].model_instance,
  //                                                                   models[4].model_instance,
  //                                                                   models[5].model_instance);

  plant.Finalize();

  int num_iiwa = 1;
  int num_joints = 7;
  StateFeedbackControllerInterface<double>* controller = nullptr;
  VectorX<double> iiwa_kp, iiwa_kd, iiwa_ki;
  SetPositionControlledIiwaGains(&iiwa_kp, &iiwa_ki, &iiwa_kd);
  iiwa_kp = iiwa_kp.replicate(num_iiwa, 1).eval();
  iiwa_kd = iiwa_kd.replicate(num_iiwa, 1).eval();
  iiwa_ki = iiwa_ki.replicate(num_iiwa, 1).eval();
  controller = builder.AddSystem<InverseDynamicsController<double>>(
      plant, iiwa_kp, iiwa_ki, iiwa_kd,
      false /* without feedforward acceleration */);

  auto desired_state_from_position = builder.AddSystem<
      StateInterpolatorWithDiscreteDerivative>(
          num_joints, kIiwaLcmStatusPeriod,
          true /* suppress_initial_transient */);

  auto trajectory_source = builder.AddSystem<drake::systems::TrajectorySource<double>>(trajectory_solution);
  
  builder.Connect(trajectory_source->get_output_port(), desired_state_from_position->get_input_port());
  builder.Connect(desired_state_from_position->get_output_port(), controller->get_input_port_desired_state());
  builder.Connect(plant.get_state_output_port(), controller->get_input_port_estimated_state());
  builder.Connect(controller->get_output_port_control(), plant.get_actuation_input_port());

  // Create the visualizer
  auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, lcm);

  // Build the diagram.
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto plant_context = &plant.GetMyMutableContextFromRoot(context.get());

  plant.SetPositions(plant_context, start_state);

  Simulator<double> simulator(*diagram, std::move(context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(T * timestep);
}

void VisualizeTrajectory(std::vector<MatrixX<symbolic::Variable>> segment_controls, solvers::MathematicalProgramResult result, Eigen::VectorXd start_state){

  if (!result.is_success()){

      std::cout << "Solver failed..." << std::endl;
      return;
  }
  else{
    std::cout << "Program solved successfully!" << std::endl;
  }

  std::cout << "Number of segments : " << segment_controls.size() << std::endl;

  std::vector<double> t_solution;
  std::vector<Eigen::MatrixXd> q_solution;

  int T = 50;
  double timestep = FLAGS_traj_duration / T;
  int numberOfSegments = segment_controls.size() ;
  double multiplier = FLAGS_traj_duration / (1 * numberOfSegments);

  for (int segmentID = 0; segmentID < numberOfSegments; segmentID++)
  {
      auto optimal_coeffs = result.GetSolution(segment_controls[segmentID]);
      auto optimal_trajectory = BezierCurve<double>(0, 1, optimal_coeffs);

      for (int t = 0; t < T; t++)
      {
          double tStep= static_cast<double>(t) / T;
          // std::cout << optimal_trajectory.value(tStep).transpose() << std::endl;

          t_solution.push_back((segmentID * 1 + tStep) * multiplier);
          q_solution.push_back(optimal_trajectory.value(tStep));
      }

      if (segmentID == (numberOfSegments - 1)){
        t_solution.push_back(numberOfSegments * multiplier);
        q_solution.push_back(optimal_trajectory.value(1));
      }
  }

  for (auto elem : t_solution)
    std::cout << elem << " ";
  std::cout << std::endl;
 
  for (auto elem : q_solution) 
    std::cout << elem.transpose() << " " << std::endl;
  std::cout << std::endl;

  drake::trajectories::PiecewisePolynomial<double> trajectory_solution = 
        drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(t_solution, q_solution);

  systems::DiagramBuilder<double> builder;
  // Adds a plant.
  auto [plant, scene_graph] = multibody::AddMultibodyPlantSceneGraph(
      &builder, FLAGS_sim_dt);
  
  auto parser = multibody::Parser(&plant, &scene_graph);
  const std::string directives_file = FindResourceOrThrow("drake/examples/gcs/manipulation/models/iiwa14_spheres_collision_welded_gripper.yaml");
  const ModelDirectives directives = LoadModelDirectives(directives_file);
  const std::vector<ModelInstanceInfo> models = ProcessModelDirectives(directives, &parser);

  // const auto [iiwa_instance, wsg, shelf, binR, binL, table] = std::make_tuple(models[0].model_instance,
  //                                                                   models[1].model_instance,
  //                                                                   models[2].model_instance,
  //                                                                   models[3].model_instance,
  //                                                                   models[4].model_instance,
  //                                                                   models[5].model_instance);

  plant.Finalize();

  int num_iiwa = 1;
  int num_joints = 7;
  StateFeedbackControllerInterface<double>* controller = nullptr;
  VectorX<double> iiwa_kp, iiwa_kd, iiwa_ki;
  SetPositionControlledIiwaGains(&iiwa_kp, &iiwa_ki, &iiwa_kd);
  iiwa_kp = iiwa_kp.replicate(num_iiwa, 1).eval();
  iiwa_kd = iiwa_kd.replicate(num_iiwa, 1).eval();
  iiwa_ki = iiwa_ki.replicate(num_iiwa, 1).eval();
  controller = builder.AddSystem<InverseDynamicsController<double>>(
      plant, iiwa_kp, iiwa_ki, iiwa_kd,
      false /* without feedforward acceleration */);

  auto desired_state_from_position = builder.AddSystem<
      StateInterpolatorWithDiscreteDerivative>(
          num_joints, kIiwaLcmStatusPeriod,
          true /* suppress_initial_transient */);

  auto trajectory_source = builder.AddSystem<drake::systems::TrajectorySource<double>>(trajectory_solution);
  
  builder.Connect(trajectory_source->get_output_port(), desired_state_from_position->get_input_port());
  builder.Connect(desired_state_from_position->get_output_port(), controller->get_input_port_desired_state());
  builder.Connect(plant.get_state_output_port(), controller->get_input_port_estimated_state());
  builder.Connect(controller->get_output_port_control(), plant.get_actuation_input_port());

  // Create the visualizer
  auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, lcm);

  // Build the diagram.
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto plant_context = &plant.GetMyMutableContextFromRoot(context.get());

  plant.SetPositions(plant_context, start_state);

  Simulator<double> simulator(*diagram, std::move(context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(T * timestep);
}

int RunExample() {

  const int order = 3;
  Eigen::VectorXd q0(7);
  q0 << 0, 0.3, 0, -1.8, 0, 1, 1.57;

  std::map<std::string, std::vector<Eigen::Vector3d>> milestones = {
        {"Above Shelve", {{0.75, 0, 0.9}, {0, -M_PI, -M_PI / 2}}},
        {"Top Rack", {{0.75, 0, 0.67}, {0, -M_PI, -M_PI / 2}}},
        {"Middle Rack", {{0.75, 0, 0.41}, {0, -M_PI, -M_PI / 2}}},
        {"Left Bin", {{0.0, 0.6, 0.22}, {M_PI / 2, M_PI, 0}}},
        {"Right Bin", {{0.0, -0.6, 0.22}, {M_PI / 2, M_PI, M_PI}}}
        };

  std::map<std::string, Eigen::VectorXd> additional_seed_points = {
      {"Front to Shelve", Eigen::VectorXd::Zero(7)},
      {"Left to Shelve", Eigen::VectorXd::Zero(7)},
      {"Right to Shelve", Eigen::VectorXd::Zero(7)}};

  additional_seed_points["Front to Shelve"] << 0, 0.2, 0, -2.09, 0, -0.3, M_PI / 2;
  additional_seed_points["Left to Shelve"] << 0.8, 0.7, 0, -1.6, 0, 0, M_PI / 2;
  additional_seed_points["Right to Shelve"] << -0.8, 0.7, 0, -1.6, 0, 0, M_PI / 2;

  // Compute milestone configurations
  std::unordered_map<std::string, Eigen::VectorXd> milestone_configurations;
  for (const auto& milestone : milestones) {
    const std::string& name = milestone.first;
    const Eigen::Vector3d& trans = milestone.second[0];
    const Eigen::Vector3d& rot = milestone.second[1];
    milestone_configurations[name] = drake::examples::kuka_iiwa_arm::gcs::doInverseKinematics(q0, trans, rot);
  }

  std::unordered_map<std::string, Eigen::VectorXd> seed_points;
  seed_points.insert(milestone_configurations.begin(), milestone_configurations.end());
  seed_points.insert(additional_seed_points.begin(), additional_seed_points.end());

  IrisOptions iris_options;
  iris_options.require_sample_point_is_contained = true;
  iris_options.iteration_limit = 20;
  iris_options.termination_threshold = -1;
  iris_options.configuration_space_margin = 1e-02;
  iris_options.relative_termination_threshold = 0.005;

  bool use_pregenerated_regions = true;
  std::unordered_map<std::string, HPolyhedron> loaded_regions;
  if (use_pregenerated_regions)
  {
      loaded_regions = loadRegions("/home/drparadox30/petersen_home");
  }
  else
  {
    loaded_regions = generateRegions(seed_points, iris_options);
    saveRegions(loaded_regions);
  }
  
  // Run GCS
  std::unordered_map<std::string, std::vector<Eigen::Vector3d>> demonstration = {
      {"Above Shelve", {{0.75, -0.12, 0.9}, {0, -M_PI, -M_PI/2}}},
      {"Top Rack", {{0.75, 0.12, 0.67}, {0, -M_PI, -M_PI/2}}},
      {"Middle Rack", {{0.75, 0.12, 0.41}, {0, -M_PI, -M_PI/2}}},
      {"Left Bin", {{0.08, 0.6, 0.22}, {M_PI/2, M_PI, 0}}},
      {"Right Bin", {{-0.08, -0.6, 0.22}, {M_PI/2, M_PI, M_PI}}}
  };

  std::unordered_map<std::string, Eigen::VectorXd> demonstration_configurations;
  for (const auto& entry : demonstration)
  {
      const std::string& name = entry.first;
      const std::vector<Eigen::Vector3d>& config = entry.second;
      const Eigen::Vector3d& trans = config[0];
      const Eigen::Vector3d& rot = config[1];
      Eigen::VectorXd result = doInverseKinematics(q0, trans, rot);
      demonstration_configurations[name] = result;
  }

  std::vector<Eigen::VectorXd> demo_a = {
      demonstration_configurations["Above Shelve"],
      demonstration_configurations["Left Bin"]
  };

  std::vector<Eigen::VectorXd> execute_demo = demo_a;

  std::cout << "Start : " << execute_demo[0].transpose() << std::endl;
  std::cout << "End : " << execute_demo[1].transpose() << std::endl;

  const int kDimension = 7;
  const double kMinimumDuration = 1.0;
  GcsTrajectoryOptimization gcs(kDimension);

  std::vector<HPolyhedron> regionsArray;
  for (auto& entry : loaded_regions){
    std::string name = entry.first;
    regionsArray.push_back(entry.second);
  }

  // Assuming ConvexSets is defined as a vector of unique pointers to ConvexSet objects
  std::vector<copyable_unique_ptr<ConvexSet>> regions_;

  // Then, when assigning the regionsArray to regions_:
  regions_.clear();
  regions_.reserve(regionsArray.size());
  for (const auto& polyhedron : regionsArray) {
    regions_.emplace_back(std::make_unique<HPolyhedron>(polyhedron));
  }

  auto& regions = gcs.AddRegions(regions_, order, kMinimumDuration);

  auto& source = gcs.AddRegions(MakeConvexSets(Point(demo_a[0])), 0);
  auto& target = gcs.AddRegions(MakeConvexSets(Point(demo_a[1])), 0);

  gcs.AddEdges(source, regions);
  gcs.AddEdges(regions, target);

  gcs.AddPathLengthCost(1.0);
  gcs.AddTimeCost(1.0);

  GraphOfConvexSetsOptions options;
  options.max_rounded_paths = 10;
  options.max_rounding_trials = 100;

  auto [traj, traj_control_points, result] = gcs.SolvePath(source, target, options);

  VisualizeTrajectory(traj, result, execute_demo[0]);

  // Add a minimum distance constraint to the GCS output

  std::cout << "APPLY THE MINIMUM DISTANCE CONSTRAINT!" << std::endl;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = multibody::AddMultibodyPlantSceneGraph(
      &builder, FLAGS_sim_dt);
  
  auto parser = multibody::Parser(&plant, &scene_graph);
  const std::string directives_file = FindResourceOrThrow("drake/examples/gcs/manipulation/models/iiwa14_spheres_collision_welded_gripper.yaml");
  const ModelDirectives directives = LoadModelDirectives(directives_file);
  const std::vector<ModelInstanceInfo> models = ProcessModelDirectives(directives, &parser);

  // const auto [iiwa_instance, wsg, shelf, binR, binL, table] = std::make_tuple(models[0].model_instance,
  //                                                                   models[1].model_instance,
  //                                                                   models[2].model_instance,
  //                                                                   models[3].model_instance,
  //                                                                   models[4].model_instance,
  //                                                                   models[5].model_instance);

  plant.Finalize();

  // Build the plant
  std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();
  // Create a diagram-level context
  std::unique_ptr<drake::systems::Context<double>> diagram_context = diagram->CreateDefaultContext();
  drake::log()->debug("Plant built");

  // Then create a subsystem context for the multibodyplant
  drake::systems::Context<double>* plant_context =
      &diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  std::unique_ptr<drake::solvers::MathematicalProgram> prog =
      std::make_unique<drake::solvers::MathematicalProgram>();

  const int numSegments = traj_control_points.size();
  int num_positions = plant.num_positions();
  int T = FLAGS_T;
  
  std::cout << order << " " << numSegments << " " << num_positions << " " << T << std::endl;

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

  Eigen::MatrixXd start_ = VectorX<double>::Zero(num_positions);
  start_ = execute_demo[0];
  Eigen::MatrixXd end_ = VectorX<double>::Zero(num_positions);
  end_ = execute_demo[1];

  Eigen::MatrixXd goal_margin = 0.01 * VectorX<double>::Ones(num_positions);
  auto xvars = prog->NewContinuousVariables(T * numSegments, num_positions, "xvars");

  prog->AddBoundingBoxConstraint(start_.transpose(), start_.transpose(), xvars.row(0));
  prog->AddBoundingBoxConstraint((end_ - goal_margin).transpose(), (end_ + goal_margin).transpose(), xvars.row(T * numSegments -1));

  for (int t = 1; t < (T * numSegments); t++) {
      prog->AddQuadraticCost(0.5 * (xvars.row(t) - xvars.row(t-1)).dot(xvars.row(t) - xvars.row(t-1)));
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

  drake::log()->debug("Deterministic program definition complete");

  double risk_precision = 0.0001; // 10^-7

  drake::solvers::SnoptSolver solver;
  drake::solvers::SolverOptions options_;
  prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Major iterations limit", 10000000);
  prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Print file", "ufo_snopt.out");
  prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Verify level", 0);
  prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Major optimality tolerance", sqrt(FLAGS_function_precision));
  prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Major feasibility tolerance", sqrt(10 * risk_precision));
  prog->SetSolverOption(drake::solvers::SnoptSolver::id(), "Function precision", FLAGS_function_precision);

  drake::log()->debug("Solving deterministic program...");
  drake::solvers::MathematicalProgramResult collision_free_result = solver.Solve(*prog);
  drake::log()->debug("Deterministic program solved");

  VisualizeTrajectory(segment_controls, collision_free_result, execute_demo[0]);

  // To check the probability of collision between the robot and the environment,
  // we need to define the list of bodies that make up the "robot", which we save
  // in the Bullet world manager
  ccopt::BulletWorldManager<double>* world_manager = new ccopt::BulletWorldManager<double>();
  std::vector<std::string> robot_body_names{
      "base"
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
      plant.GetCollisionGeometriesForBody(plant.GetBodyByName("table_body"));
  std::cout << "=====================\n" << "floor" << std::endl;
  for (auto id : floor_ids) {
      drake::log()->info("ID: {}", id);
  }

  // We also need to define the bodies that are uncertain.
  std::vector<std::string> uncertain_obstacle_names{
      "table_body",
      "shelves_body",
      "top_and_bottom",
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

  // We also need to define the bodies that are uncertain.
  std::vector<std::string> uncertain_bin_names{
      "binR",
      "binL",
  };

  for (const std::string name : uncertain_bin_names) {
      std::cout << "=====================\n" << name << std::endl;
      // Get the geometry IDs corresponding to this obstacle
      std::vector<drake::geometry::GeometryId> obstacle_ids =
          plant.GetCollisionGeometriesForBody(
              plant.GetBodyByName("bin_base", plant.GetModelInstanceByName(name)));
      for (auto id : obstacle_ids) {
          drake::log()->info("ID: {}", id);
      }
      // Append that to the vector of uncertain_obstacle_ids
      uncertain_obstacle_ids.insert(uncertain_obstacle_ids.end(),
                                    obstacle_ids.begin(),
                                    obstacle_ids.end());
  }

  // Let's make both pillars uncertain in the x direction.
  Eigen::Matrix3d uncertain_obstacle_covariance;
  uncertain_obstacle_covariance << 0.1, 0.0, 0.0,
                                    0.0, 0.1, 0.0,
                                    0.0, 0.0, 0.1;
  // Make a vector of n copies of the covariance, where n = the number of uncertain
  // geometry IDs found above
  std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances;

  for (int i = 0; i < static_cast<int>(uncertain_obstacle_ids.size()); i++){
      uncertain_obstacle_covariances.push_back(uncertain_obstacle_covariance);
  }

  // what():  GetBodyByName(): There is no Body named 'iiwa' anywhere in the model 
  // (valid names are: base, bin_base, body, iiwa_link_0, iiwa_link_1, iiwa_link_2,
  //  iiwa_link_3, iiwa_link_4, iiwa_link_5, iiwa_link_6, iiwa_link_7, iiwa_link_ee, 
  // iiwa_link_ee_kuka, left_finger, right_finger, shelves_body, table_body, 
  // top_and_bottom, world)

  // Next add a chance constraint covering the entire trajectory
  auto collision_chance_constraint = std::make_shared<ccopt::CollisionChanceConstraint>(
      &plant, plant_context, world_manager,
      risk_precision,
      FLAGS_delta,
      numSegments * FLAGS_T,
      FLAGS_use_max,
      uncertain_obstacle_ids, uncertain_obstacle_covariances
  );
  // Add the chance constraint to the program
  Eigen::Matrix<drake::symbolic::Variable, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xvars_row_major(xvars);
  Eigen::Map<drake::solvers::VectorXDecisionVariable> xvars_vectorized(xvars_row_major.data(), xvars_row_major.size());
  drake::solvers::Binding<drake::solvers::Constraint> chance_constraint_bound =
      prog->AddConstraint(collision_chance_constraint, xvars_vectorized);

  // We'll seed the nonlinear solver with the solution from the minimum distance constraint problem
  for (int i = 0; i < numSegments; i++)
      prog->SetInitialGuess(segment_controls[i], collision_free_result.GetSolution(segment_controls[i]));

  // That completes our setup for the chance-constrained mathematical program
  drake::log()->debug("Chance-constrained program definition complete");

  // Now the fun part: we can finally solve the problem! (don't forget to measure runtime)
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
  VisualizeTrajectory(segment_controls, chance_constrained_result, execute_demo[0]);

  return 0;
}

}  // namespace gcs
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::kuka_iiwa_arm::gcs::RunExample();
}
