#include <unistd.h>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging.h"
#include "drake/common/type_safe_index.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include <drake/geometry/scene_graph_inspector.h>
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/weld_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include <drake/systems/rendering/multibody_position_to_geometry_pose.h>
#include <drake/common/trajectories/piecewise_polynomial.h>

namespace drake {
namespace examples {
namespace kuka {

using drake::multibody::MultibodyPlant;

DEFINE_double(simulation_time, 5.0,
              "Desired duration of the simulation in seconds");
DEFINE_double(max_time_step, 1.0e-3,
              "Simulation time step used for integrator.");
DEFINE_double(target_realtime_rate, 1,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

DEFINE_bool(add_gravity, true,
            "Indicator for whether terrestrial gravity"
            " (9.81 m/s²) is included or not.");

void DoMain() {
  DRAKE_DEMAND(FLAGS_simulation_time > 0);

  systems::DiagramBuilder<double> builder;

  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  MultibodyPlant<double> plant = MultibodyPlant<double>(FLAGS_max_time_step);
  drake::geometry::SourceId plant_source_id = plant.RegisterAsSourceForSceneGraph(&scene_graph);
  std::string full_name = "./examples/models/rs010n.urdf";

  multibody::ModelInstanceIndex plant_index =
      multibody::Parser(&plant).AddModelFromFile(full_name);

  // Weld the arm to the world frame.
  const auto& joint_arm_root = plant.GetBodyByName("base_link");
  plant.AddJoint<multibody::WeldJoint>("weld_arm", plant.world_body(), {},
                                       joint_arm_root, {},
                                       drake::math::RigidTransform(Isometry3<double>::Identity()));

  if (!FLAGS_add_gravity) {
    plant.mutable_gravity_field().set_gravity_vector(Eigen::Vector3d::Zero());
  }

  // Now the model is complete.
  plant.Finalize();

  // Define the trajectory as a piecewise linear
  std::vector<double> knot_t{0.0, 1.0};
  Eigen::MatrixXd start = VectorX<double>::Zero(plant.num_positions());
  start(1) = 0.0;
  Eigen::MatrixXd end = VectorX<double>::Zero(plant.num_positions());
  end(1) = 0.0;
  std::vector<Eigen::MatrixXd> knot_q(2);
  knot_q[0] = start;
  knot_q[1] = end;
  drake::trajectories::PiecewisePolynomial<double> trajectory = 
    drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(knot_t, knot_q);

  const auto traj_source = builder.AddSystem<drake::systems::TrajectorySource<double>>(
    trajectory
  );

  // Connect the trajectory source directly to the geometry poses
  auto q_to_pose = 
  builder.AddSystem<drake::systems::rendering::MultibodyPositionToGeometryPose<double>>(
    plant
  );
  builder.Connect(traj_source->get_output_port(),
                  q_to_pose->get_input_port());
  builder.Connect(q_to_pose->get_output_port(),
                  scene_graph.get_source_pose_port(plant_source_id));

  // Create the visualizer
  geometry::ConnectDrakeVisualizer(&builder, scene_graph);
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();

  std::vector<drake::geometry::GeometryId> geom_ids = scene_graph.model_inspector().GetAllGeometryIds();
  for (int i = 0; i < geom_ids.size(); i++) {
    std::cout << geom_ids[i] << std::endl;
  }

  // Set up simulator.
  systems::Simulator<double> simulator(*diagram);
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);
}

}  // namespace kuka
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "bazel run"
      "//examples/kuka_iiwa_arm_idc:run_kuka_idc");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::kuka::DoMain();
  return 0;
}