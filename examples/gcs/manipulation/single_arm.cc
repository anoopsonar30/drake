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

#include "drake/examples/gcs/manipulation/ThreadPool.h"

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/gcs/manipulation/iiwa_common.h"
#include "drake/examples/gcs/manipulation/iiwa_lcm.h"
#include "drake/examples/gcs/manipulation/kuka_torque_controller.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/parsing/process_model_directives.h"
// #include "drake/multibody/parsing/model_directives.h"
// #include "drake/multibody/parsing/detail_urdf_parser.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/controllers/state_feedback_controller_interface.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/systems/primitives/constant_value_source.h"

#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

#include "drake/geometry/optimization/iris.h"
#include "drake/geometry/optimization/hpolyhedron.h"

DEFINE_double(simulation_sec, std::numeric_limits<double>::infinity(),
              "Number of seconds to simulate.");
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
using systems::Demultiplexer;
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

using geometry::optimization::IrisOptions;
using geometry::optimization::IrisInConfigurationSpace;
using geometry::optimization::HPolyhedron;

// Save the regions to a binary file
void saveRegions(const std::string& filename,
                 const std::unordered_map<std::string, HPolyhedron>& regions) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    // Serialize each region as a binary string and write it to the file
    for (const auto& entry : regions) {
        const std::string& seed_name = entry.first;
        const HPolyhedron& region = entry.second;
        std::string region_data(region.SerializeAsString());
        uint32_t name_size = seed_name.size();
        uint32_t data_size = region_data.size();
        file.write(reinterpret_cast<const char*>(&name_size), sizeof(name_size));
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(seed_name.c_str(), seed_name.size());
        file.write(region_data.c_str(), region_data.size());
    }
}

// Load the regions from a binary file
std::unordered_map<std::string, HPolyhedron> loadRegions(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    std::unordered_map<std::string, HPolyhedron> regions;
    drake::math::DeserializeFromFile(file, &regions);
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
  systems::DiagramBuilder<double> builder;

  // Adds a plant.
  auto [plant, scene_graph] = multibody::AddMultibodyPlantSceneGraph(
      &builder, FLAGS_sim_dt);
  
  auto parser = multibody::Parser(&plant, &scene_graph);
  const std::string directives_file = FindResourceOrThrow("drake/examples/gcs/manipulation/models/iiwa14_spheres_collision_welded_gripper.yaml");
  const ModelDirectives directives = LoadModelDirectives(directives_file);
  const std::vector<ModelInstanceInfo> models = ProcessModelDirectives(directives, &parser);

  const auto [iiwa_instance, wsg, shelf, binR, binL, table] = std::make_tuple(models[0].model_instance,
                                                                    models[1].model_instance,
                                                                    models[2].model_instance,
                                                                    models[3].model_instance,
                                                                    models[4].model_instance,
                                                                    models[5].model_instance);

  plant.Finalize();

  // // Creates and adds LCM publisher for visualization.
  auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, lcm);

  // // Since we welded the model to the world above, the only remaining joints
  // // should be those in the arm.
  const int num_joints = plant.num_positions();
  std::cout << "Number of joints : " << num_joints << std::endl;
  DRAKE_DEMAND(num_joints % kIiwaArmNumJoints == 0);
  const int num_iiwa = num_joints / kIiwaArmNumJoints;

  // // Adds a iiwa controller.
  StateFeedbackControllerInterface<double>* controller = nullptr;
  if (FLAGS_torque_control) {
    VectorX<double> stiffness, damping_ratio;
    SetTorqueControlledIiwaGains(&stiffness, &damping_ratio);
    stiffness = stiffness.replicate(num_iiwa, 1).eval();
    damping_ratio = damping_ratio.replicate(num_iiwa, 1).eval();
    controller = builder.AddSystem<KukaTorqueController<double>>(
        plant, stiffness, damping_ratio);
  } else {
    VectorX<double> iiwa_kp, iiwa_kd, iiwa_ki;
    SetPositionControlledIiwaGains(&iiwa_kp, &iiwa_ki, &iiwa_kd);
    iiwa_kp = iiwa_kp.replicate(num_iiwa, 1).eval();
    iiwa_kd = iiwa_kd.replicate(num_iiwa, 1).eval();
    iiwa_ki = iiwa_ki.replicate(num_iiwa, 1).eval();
    controller = builder.AddSystem<InverseDynamicsController<double>>(
        plant, iiwa_kp, iiwa_ki, iiwa_kd,
        false /* without feedforward acceleration */);
  }

  // // Create the command subscriber and status publisher.
  auto command_sub = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_command>(
          "IIWA_COMMAND", lcm));
  command_sub->set_name("command_subscriber");
  auto command_receiver =
      builder.AddSystem<IiwaCommandReceiver>(num_joints);
  command_receiver->set_name("command_receiver");
  auto plant_state_demux = builder.AddSystem<Demultiplexer>(
      2 * num_joints, num_joints);
  plant_state_demux->set_name("plant_state_demux");
  auto desired_state_from_position = builder.AddSystem<
      StateInterpolatorWithDiscreteDerivative>(
          num_joints, kIiwaLcmStatusPeriod,
          true /* suppress_initial_transient */);
  desired_state_from_position->set_name("desired_state_from_position");
  auto status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_iiwa_status>(
          "IIWA_STATUS", lcm, kIiwaLcmStatusPeriod /* publish period */));
  status_pub->set_name("status_publisher");
  auto status_sender = builder.AddSystem<IiwaStatusSender>(num_joints);
  status_sender->set_name("status_sender");

  builder.Connect(command_sub->get_output_port(),
                  command_receiver->get_message_input_port());
  builder.Connect(plant_state_demux->get_output_port(0),
                  command_receiver->get_position_measured_input_port());
  builder.Connect(command_receiver->get_commanded_position_output_port(),
                  desired_state_from_position->get_input_port());
  builder.Connect(desired_state_from_position->get_output_port(),
                  controller->get_input_port_desired_state());
  builder.Connect(plant.get_state_output_port(iiwa_instance),
                  plant_state_demux->get_input_port(0));
  builder.Connect(plant_state_demux->get_output_port(0),
                  status_sender->get_position_measured_input_port());
  builder.Connect(plant_state_demux->get_output_port(1),
                  status_sender->get_velocity_estimated_input_port());
  builder.Connect(command_receiver->get_commanded_position_output_port(),
                  status_sender->get_position_commanded_input_port());
  builder.Connect(plant.get_state_output_port(),
                  controller->get_input_port_estimated_state());
  builder.Connect(controller->get_output_port_control(),
                  plant.get_actuation_input_port(iiwa_instance));
  builder.Connect(controller->get_output_port_control(),
                  status_sender->get_torque_commanded_input_port());
  builder.Connect(controller->get_output_port_control(),
                  status_sender->get_torque_measured_input_port());
  // TODO(sammy-tri) Add a low-pass filter for simulated external torques.
  // This would slow the simulation significantly, however.  (see #12631)
  builder.Connect(
      plant.get_generalized_contact_forces_output_port(iiwa_instance),
      status_sender->get_torque_external_input_port());
  builder.Connect(status_sender->get_output_port(),
                  status_pub->get_input_port());
  // Connect the torque input in torque control
  if (FLAGS_torque_control) {
    KukaTorqueController<double>* torque_controller =
        dynamic_cast<KukaTorqueController<double>*>(controller);
    DRAKE_DEMAND(torque_controller != nullptr);
    builder.Connect(command_receiver->get_commanded_torque_output_port(),
                    torque_controller->get_input_port_commanded_torque());
  }

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  auto plant_context = &plant.GetMyMutableContextFromRoot(context.get());
  
  Eigen::VectorXd current_positions = plant.GetPositions(*plant_context);
  std::cout << "Current positions: " << current_positions.transpose() << std::endl;
  
  Eigen::VectorXd q0(7);
  q0 << 0, 0.3, 0, -1.8, 0, 1, 1.57;
  plant.SetPositions(plant_context, q0);
  current_positions = plant.GetPositions(*plant_context);
  std::cout << "Current positions: " << current_positions.transpose() << std::endl;

  Simulator<double> simulator(*diagram);

  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();

  // Simulate for a very long time.
  // simulator.AdvanceTo(FLAGS_simulation_sec);

  std::map<std::string, std::vector<Eigen::Vector3d>> milestones = {
        {"Above Shelve", {{0.75, 0, 0.9}, {0, -M_PI, -M_PI / 2}}},
        {"Top Rack", {{0.75, 0, 0.67}, {0, -M_PI, -M_PI / 2}}},
        {"Middle Rack", {{0.75, 0, 0.41}, {0, -M_PI, -M_PI / 2}}},
        {"Left Bin", {{0.0, 0.6, 0.22}, {M_PI / 2, M_PI, 0}}},
        {"Right Bin", {{0.0, -0.6, 0.22}, {M_PI / 2, M_PI, M_PI}}}};

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

  std::unordered_map<std::string, HPolyhedron> regions = generateRegions(seed_points, iris_options);

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
