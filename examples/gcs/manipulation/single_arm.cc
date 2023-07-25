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

DEFINE_double(simulation_sec, std::numeric_limits<double>::infinity(),
              "Number of seconds to simulate.");
DEFINE_double(traj_duration, 5,
              "The total duration of the trajectory (in seconds).");
DEFINE_string(urdf, "", "Name of urdf to load");
DEFINE_double(target_realtime_rate, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_bool(torque_control, false, "Simulate using torque control mode.");
DEFINE_double(sim_dt, 3e-3,
              "The time step to use for MultibodyPlant model "
              "discretization.");

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




int RunExample() {

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
      demonstration_configurations["Right Bin"],
      demonstration_configurations["Above Shelve"]
  };

  std::vector<Eigen::VectorXd> execute_demo = demo_a;

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

  auto& regions = gcs.AddRegions(regions_, 1, kMinimumDuration);

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

  if (!result.is_success()){

      std::cout << "Solver failed..." << std::endl;
      return 0;
  }

  std::cout << "Number of segments : " << traj.get_number_of_segments() << std::endl;

  std::vector<double> t_solution;
  std::vector<Eigen::MatrixXd> q_solution;

  int T = 2;
  double timestep = FLAGS_traj_duration / T;
  int numberOfSegments = traj.get_number_of_segments();
  double multiplier = 5.0 / (1 * numberOfSegments);

  for (int segmentID = 0; segmentID < numberOfSegments; segmentID++)
  {
      for (int t = 0; t < T; t++)
      {
          double tStep= segmentID * 1 + static_cast<double>(t) / T;
          auto coords = traj.segment(segmentID).value(tStep);
          std::cout << coords << std::endl;
          std::cout << std::endl;
          
          t_solution.push_back(tStep * multiplier);
          q_solution.push_back(traj.segment(segmentID).value(tStep));
      }
  }
  // Add the last point.
  t_solution.push_back(numberOfSegments * multiplier);
  q_solution.push_back(traj.segment(numberOfSegments - 1).value(numberOfSegments));

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

  auto trajectory_source = builder.AddSystem<drake::systems::TrajectorySource<double>>(trajectory_solution);
  builder.Connect(trajectory_source->get_output_port(), plant.get_actuation_input_port());

  // Create the visualizer
  auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, lcm);

  // Build the diagram.
  auto diagram = builder.Build();

  drake::log()->debug("Visualizer built");

  Simulator<double> simulator(*diagram);
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();

  std::cout << "Reached the end..." << std::endl;

  simulator.AdvanceTo(T * timestep);

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
