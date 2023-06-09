/*
 * Interfaces with Drake SceneGraph to synchronize the position of geometry
 * in the Bullet collision world, and exposes queries on signed distance and
 * collision probability.
 *
 * Written by Charles Dawson (cbd@mit.edu) on April 30, 2020
 */
#pragma once

#include <vector>
#include <string>
#include <map>

#include <drake/common/default_scalars.h>
#include <drake/geometry/scene_graph.h>
#include <drake/geometry/geometry_set.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/systems/framework/context.h>

#include "btBulletCollisionCommon.h"

#include <Eigen/Dense>

#include "drake/examples/scora/ProximityAlert/proximityAlert.h"

namespace ccopt {

    /*
     * BulletWorldManager maintains an internal btCollisionWorld for performing
     * collision-probability queries, ensuring that the internal
     * btCollisionWorld stays synchronized with the Drake SceneGraph.
     *
     * Since SceneGraphs can be either double or drake::AutoDiffXd, we need
     * to define a template that supports either.
     */
    template <typename T>
    class BulletWorldManager : public drake::systems::LeafSystem<T>{

        // This is the internal Bullet btCollisionWorld, which we keep synchronized
        // with the Drake collision world
        btCollisionConfiguration* m_bt_collision_configuration;
        btCollisionDispatcher* m_bt_dispatcher;
        btBroadphaseInterface* m_bt_broadphase;
        btCollisionWorld* m_collision_world;

        // This maps from Drake geometry names to synced Bullet btCollisionObject
        std::map<drake::geometry::GeometryId, std::shared_ptr<btCollisionObject>>
          m_collision_objects;
        // We also save the collision geometry, to make sure the shared_ptr
        // sticks around long enough
        std::map<drake::geometry::GeometryId, std::shared_ptr<btConvexShape>>
          m_collision_shapes;

        // We need some way of differentiating robot geometry from obstacle geometry
        // that's flexible enough to allow something like moving a box from "obstacle"
        // to "robot" when the robot picks it up. I think we can do that best with a set
        // of GeometryIds indicating the set of "robot" geometries
        drake::geometry::GeometrySet m_robot_geometry_set;

    public:
        BulletWorldManager();

        ~BulletWorldManager() {
            delete m_collision_world;
            delete m_bt_broadphase;
            delete m_bt_dispatcher;
            delete m_bt_collision_configuration;
            m_collision_objects.clear();
        }

        // The main function of the world manager is to support queries on
        // the probability of collision between the robot and any number of
        // obstacles with uncertain position. We overload this function to support
        // both double and AutoDiffXd queries (the latter additionally returns
        // information about the gradient, while the former returns the risk only).
        void ComputeCollisionProbability(
                                      const drake::multibody::MultibodyPlant<T>& plant,
                                      const drake::systems::Context<T>& context,
                                      const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
                                      const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
                                      double tolerance,
                                      double &result);
        void ComputeCollisionProbability(
                                      const drake::multibody::MultibodyPlant<T>& plant,
                                      const drake::systems::Context<T>& context,
                                      const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
                                      const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
                                      double tolerance,
                                      drake::AutoDiffXd& result);

        // To support this, we need some helper functions to synchronize the
        // internal Bullet btCollisionWorld
        void SynchronizeInternalWorld(const drake::multibody::MultibodyPlant<T>& plant,
                                      const drake::systems::Context<T>& context);
        void SynchronizeInternalWorld(const drake::multibody::MultibodyPlant<T>& plant,
                                      const drake::systems::Context<T>& context,
                                      const std::vector<drake::geometry::GeometryId> frozen_geometry_ids);

        // These functions allow the user to update the list of geometries considered as "robot"
        // geometries (and thus checked for collision with uncertain obstacles)
        void ClearRobotGeometryIds();
        void AddRobotGeometryIds(const std::vector<drake::geometry::GeometryId> to_add);
        int num_robot_geometry_ids();

        // These helper functions allow us to check for collisions with perturbed obstacles, to validate
        // the safety of trajectories.
        void PerturbUncertainObstacles(const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
                                       const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances);
        bool CheckCollision();

    private:
        Eigen::VectorXd CalcGradientFromProximityAlertResult(
                                      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant,
                                      const drake::systems::Context<drake::AutoDiffXd>& context,
                                      const ProximityAlert::BoundWithGradient proximity_alert_result,
                                      const drake::geometry::GeometryId uncertain_obstacle_id);
        Eigen::VectorXd CalcGradientFromProximityAlertResult(
                                      const drake::multibody::MultibodyPlant<double>& plant,
                                      const drake::systems::Context<double>& context,
                                      const ProximityAlert::BoundWithGradient proximity_alert_result,
                                      const drake::geometry::GeometryId uncertain_obstacle_id);
    };

}

// Drake SceneGraphs can only be double or drake::AutoDiffXd, so our
// BulletWorldManager only needs to handle those types as well
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
        class ::ccopt::BulletWorldManager)