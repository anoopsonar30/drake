/*
 * Implements the BulletWorldManager class.
 *
 * Written by Charles Dawson (cbd@mit.edu) on April 30, 2020
 */
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include <drake/geometry/shape_specification.h>
#include <drake/geometry/scene_graph_inspector.h>
#include <drake/math/autodiff.h>
#include <drake/math/autodiff_gradient.h>
#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/systems/framework/input_port.h>
#include <drake/systems/framework/output_port.h>

#include "geometry_set_tester.h"
#include "bullet_world_manager.h"

#include "utils.h"

#define ROBOT_FILTER_GROUP 1
#define OBSTACLE_FILTER_GROUP 2

namespace ccopt {
    /*
     * Instantiate a BulletWorldManager.
     */
    template <typename T>
    BulletWorldManager<T>::BulletWorldManager() {
        // Initiallize the required bullet overhead
        double scene_size = 10;
        unsigned int max_objects = 100;

        m_bt_collision_configuration = new btDefaultCollisionConfiguration();
        m_bt_dispatcher = new btCollisionDispatcher(m_bt_collision_configuration);

        btScalar sscene_size = static_cast<btScalar>(scene_size);
        btVector3 worldAabbMin(-sscene_size, -sscene_size, -sscene_size);
        btVector3 worldAabbMax(sscene_size, sscene_size, sscene_size);
        //This is one type of broadphase, bullet has others that might be faster depending on the application
        m_bt_broadphase = new bt32BitAxisSweep3(worldAabbMin, worldAabbMax, max_objects, 0, false); 

        m_collision_world = new btCollisionWorld(m_bt_dispatcher, m_bt_broadphase, m_bt_collision_configuration);
        // Bullet overhead done!
    }


    // Via templating, we support both double and AutoDiffXd data types. Both
    // of these data types query the collision probability, but the AutoDiffXd
    // variant additionally queries the gradient of collision risk.
    //
    // To deal with this variation, we'll overload the function for each data
    // type, where the double specialization just computes the collision risk, and
    // the AutoDiffXd specialization also computes the gradient.

    /*
     * Computes an upper bound on the risk of collision between the robot and
     * the uncertain obstacles. This overload (for doubles) returns only
     * the collision risk, not the gradient of the collision risk.
     *
     * Before calling this function, you MUST call SynchronizeInternalWorld.
     *
     * @param uncertain_obstacle_ids: the drake::geometry::GeometryIds of the uncertain obstacles
     * @param uncertain_obstacle_covariances: a vector of covariance matrices for each uncertain obstacle
     * @param tolerance: the computed upper bound is guaranteed to be within tolerance/2 of a true upper bound
     * @param result: the double in which the risk bound will be stored.
     */
    template <typename T>
    void BulletWorldManager<T>::ComputeCollisionProbability(
            const drake::multibody::MultibodyPlant<T>& plant,
            const drake::systems::Context<T>& context,
            const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
            const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
            double tolerance, double& result) {
        // First of all, we need to do some sanity checks.
        // Tolerance needs to be positive and non-zero
        if (tolerance <= 0.0) {
            throw std::invalid_argument("Tolerance must be positive and non-zero");
        }
        // Also, we need to have the same number of uncertain obstacles as covariance matrices
        if (uncertain_obstacle_ids.size() != uncertain_obstacle_covariances.size()) {
            throw std::invalid_argument(
                "Number of provided IDs must match the number of provided covariance matrices");
        }

        // Create a variable to accumulate the risk from each obstacle
        double accumulated_risk = 0.0;
        // Iterate through the uncertain obstacles
        for (int i = 0; i < static_cast<int>(uncertain_obstacle_ids.size()); i++) {
            // Get the ID and covariance for each obstacle
            drake::geometry::GeometryId uncertain_obstacle_id =
                uncertain_obstacle_ids[i];
            Eigen::Matrix3d uncertain_obstacle_covariance =
                uncertain_obstacle_covariances[i];

            // Make sure the ID is in the map of collision objects
            if (m_collision_objects.find(uncertain_obstacle_id) == m_collision_objects.end()) {
                throw std::invalid_argument("Unknown uncertain_obstacle_id. Did you SynchronizeInternalWorld?");
            }
            // Use the ID to get the collision object from the map
            std::shared_ptr<btCollisionObject> collision_object =
                m_collision_objects[uncertain_obstacle_id];

            // Call ProximityAlert to get the risk of collision
            // Get the SceneGraphInspector for this plant's associated SceneGraph
            const auto& query_port = plant.get_geometry_query_input_port();
            const auto& query_object =
                query_port.template Eval<drake::geometry::QueryObject<T>>(context);
            const auto& inspector = query_object.inspector();
            drake::log()->info("Computing collision probability bound for obstacle (id {}): {}",
                                uncertain_obstacle_id, inspector.GetName(uncertain_obstacle_id));
            drake::log()->debug("This obstacle has position: [{}, {}, {}]",
                collision_object->getWorldTransform().getOrigin().getX(),
                collision_object->getWorldTransform().getOrigin().getY(),
                collision_object->getWorldTransform().getOrigin().getZ());
            double risk_estimate = ProximityAlert::compute_collision_probability_bound(
                collision_object.get(),
                uncertain_obstacle_covariance,
                tolerance,
                m_collision_world,
                ROBOT_FILTER_GROUP);

            // Accumulate the risk from each obstacle
            accumulated_risk += risk_estimate;
        } 

        // Return the accumulated estimate in the designated reference
        result = accumulated_risk;
    }

    /*
     * Computes the gradient of collision risk from a ProximityAlert one-shot or two-shot
     * result. Overloaded for drake::AutoDiffXd.
     *
     * @param plant: the MultibodyPlant representing the robot
     * @param context: the context for plant
     * @param proximity_alert_result: a ProximityAlert::BoundWithGradient containing the
     *                                result of a collision probability query
     *
     * @returns an Eigen::VectorXd containing the gradient
     */
    template <typename T>
    Eigen::VectorXd BulletWorldManager<T>::CalcGradientFromProximityAlertResult(
          const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant,
          const drake::systems::Context<drake::AutoDiffXd>& context,
          const ProximityAlert::BoundWithGradient proximity_alert_result,
          const drake::geometry::GeometryId uncertain_obstacle_id) {
        // Compute the gradient of the risk from this obstacle.
        // This takes a bit of work, and has to be done for the one-shot and two-shot
        // results. In fact, it's involved enough that I need to write myself some notes here:
        // (for more details, see Dawson, Jasour, Hofmann, & Williams 2020)
        //
        // For each \epsilon (from either the one- or the two-shot), the gradient is given by:
        //     \nabla_q \epsilon = - X^2_3(\phi^{-1}(1-\epsilon)) * 2 x^T \Sigma^{-1} J
        // where
        //  - X^2_3 is the chi-squared PDF with 3 degrees of freedom (3 for 3D)
        //  - \phi^{-1} is the chi-squared inverse CDF with 3 degrees of freedom
        //  - x is the vector from the center of the colliding \epsilon-shadow ellipsoid to the robot,
        //    expressed in the world frame.
        //  - \Sigma^{-1} is the inverse of the covariance matrix
        //  - J is the Jacobian of the point of contact with the \epsilon-shadow with respect to the
        //    robot state (q), expressed in the world frame
        //
        // Fortunately, ProximityAlert already computes and returns:
        //     \nabla_x \epsilon = - X^2_3(\phi^{-1}(1-\epsilon)) * 2 x^T \Sigma^{-1},
        // so all we have to do is get the Jacobian of x w.r.t. q.
        //
        // Also fortunately, ProximityAlert also returns x. Less fortunately, it returns it as
        // a btVector3, so we will need to convert it.
        //
        // In summary, the game plan is this:
        //  1.) Check if the expansion hit part of the robot.
        //      1.a.) If it did, then get the body on the robot with which it makes contact.
        //          1.a.i.)  Use that body, the plant (and plant context), and x to get the Jacobian
        //                   of x with respect to robot state.
        //          1.a.ii.) Use the Jacobian to compute the gradient and store it
        //      1.b.) Otherwise, the risk is zero and the gradient is also zero

        (void)uncertain_obstacle_id;

        // Initialize the gradient to zeros.
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(plant.num_positions());

        // Convert the contact point to an Eigen datatype
        Eigen::Vector3d contact_pt =
            toEigen(proximity_alert_result.contact_point_on_robot);

        // Find the body on the robot that the one-shot expansion hit. Do this by
        // matching the pointer to a collision object returned in the result struct to
        // a GeometryId in m_collision_objects
        drake::geometry::GeometryId collider_id;
        bool collider_found = false;
        std::map<drake::geometry::GeometryId, std::shared_ptr<btCollisionObject>>::iterator it;
        for (it = m_collision_objects.begin(); it != m_collision_objects.end(); it++) {
            // Check each object in m_collision_objects. If one matches the collider...
            if (it->second.get() == proximity_alert_result.collider) {
                // Save the GeometryId
                collider_id = it->first;
                collider_found = true;
                break;
            }
        }
        // If we found a match, continue to calculate the gradient; otherwise, leave the
        // gradient as zero and move on
        if (collider_found) {
            // Get the SceneGraphInspector for this plant's associated SceneGraph
            const auto& query_port = plant.get_geometry_query_input_port();
            const auto& query_object =
                query_port.template Eval<drake::geometry::QueryObject<drake::AutoDiffXd>>(context);
            const auto& inspector = query_object.inspector();

            drake::log()->debug("Found epsilon-shadow collider on robot: {}",
                                inspector.GetName(collider_id));

            // Get the the frame to which the collider geometry is attached
            const drake::geometry::FrameId frame_id = inspector.GetFrameId(collider_id);
            const drake::multibody::Body<drake::AutoDiffXd>* body = plant.GetBodyFromFrameId(frame_id);
            if (body == nullptr) {
                throw std::invalid_argument(
                    "Could not find body for collider frame ID!");
            }
            const drake::multibody::Frame<drake::AutoDiffXd>& collision_frame = body->body_frame();

            drake::log()->debug("\tattached to frame: {}",
                                collision_frame.name());

            // Also get the world frame, with respect to which we will calculate the Jacobian
            const drake::multibody::Frame<drake::AutoDiffXd>& world_frame = plant.world_frame();

            // contact_pt is currently expressed in the world frame, but we need to convert
            // it to the collision_frame before getting the Jacobian.
            // First, get the transform of the collision_frame in the world
            //  (i.e. takes points in F to W)
            const drake::math::RigidTransform<drake::AutoDiffXd> X_WF = collision_frame.CalcPoseInWorld(context);

            // Figure out the size to reshape the contact point derivatives to (we don't actually need these derivatives
            // since we don't compute second derivatives)
            int num_derivatives = X_WF.translation()(0).derivatives().size();

            // Also cast the contact point to an AutoDiff type
            drake::AutoDiffVecXd contact_pt_cast = drake::math::InitializeAutoDiff(contact_pt);
            for (int i = 0; i < contact_pt_cast.size(); i++) {
                auto& derivs = contact_pt_cast(i).derivatives();
                derivs.resize(num_derivatives);
                derivs.setZero();
            }

            // Then use the inverse to take contact_pt from W to F
            drake::log()->debug("Contact pt in world_frame: {}", contact_pt_cast.transpose());
            drake::log()->debug("X_WF translation: {}", drake::math::ExtractValue(X_WF.translation().transpose()));
            drake::log()->debug("X_WF inverse translation: {}", drake::math::ExtractValue(X_WF.inverse().translation().transpose()));
            drake::log()->debug("X_WF inverse rotation:\n{}", drake::math::ExtractValue(X_WF.inverse().rotation().matrix()));
            drake::log()->debug("X_WF inverse rotation derivs:\n{}", X_WF.inverse().rotation().matrix()(0, 0).derivatives());
            drake::log()->debug("X_WF inverse rotation rows: {}", X_WF.inverse().rotation().matrix().rows());
            drake::log()->debug("X_WF inverse rotation cols: {}", X_WF.inverse().rotation().matrix().cols());
            drake::log()->debug("contact_pt_cast rows: {}", contact_pt_cast.rows());
            drake::log()->debug("contact_pt_cast cols: {}", contact_pt_cast.cols());
            drake::log()->debug("Rotated pt: {}", X_WF.inverse().rotation() * contact_pt_cast);
            drake::log()->debug("Rotated pt derivs: {}", (X_WF.inverse().rotation() * contact_pt_cast)(0).derivatives());
            drake::log()->debug("Translation: {}", X_WF.inverse().translation());
            drake::log()->debug("Translation derivs: {}", X_WF.inverse().translation()(0).derivatives());
            drake::log()->debug("Rotated translated pt: {}", X_WF.inverse().translation() + X_WF.inverse().rotation() * contact_pt_cast);
            drake::log()->debug("Contact pt in collision frame: {}", X_WF.inverse() * contact_pt_cast);
            contact_pt_cast = X_WF.inverse() * contact_pt_cast;
            drake::log()->debug("Contact pt in collision_frame: {}", contact_pt_cast.transpose());

            // Resize contact_pt_cast to have the right number of derivatives
            int num_generalized_positions = plant.num_positions();
            for (int i = 0; i < contact_pt_cast.size(); i++) {
                auto& derivs = contact_pt_cast(i).derivatives();
                derivs.conservativeResize(num_generalized_positions);
            }

            // Using this frame, compute the Jacobian of the contact point with respect to robot state
            // We'll use plant.CalcJacobianTranslationalVelocity(), but be careful of AutoDiffXd in 
            // the Jacobian! (I'll probably have to convert from an AutoDiffXd matrix to a double
            // matrix, then use that).
            drake::MatrixX<drake::AutoDiffXd> jacobian_uncast(3, num_generalized_positions);
            plant.CalcJacobianTranslationalVelocity(
                context,
                drake::multibody::JacobianWrtVariable::kQDot,
                collision_frame,
                contact_pt_cast,
                world_frame,
                world_frame,
                &jacobian_uncast);

            // Convert the Jacobian to double, since the plant is in AutoDiffXd
            drake::log()->debug("Jacobian uncast: {}", jacobian_uncast);
            drake::MatrixX<double> jacobian =
                drake::math::DiscardGradient(jacobian_uncast);

            drake::log()->debug("Jacobian: {}", jacobian);

            // Compute the gradient from the Jacobian!
            gradient = proximity_alert_result.d_epsilon_d_x * jacobian;
            drake::log()->debug("gradient: {}", gradient.transpose());
        }

        return gradient;
    }

    /*
     * Computes the gradient of collision risk from a ProximityAlert one-shot or two-shot
     * result. Overloaded for doubles.
     *
     * @param plant: the MultibodyPlant representing the robot
     * @param context: the context for plant
     * @param proximity_alert_result: a ProximityAlert::BoundWithGradient containing the
     *                                result of a collision probability query
     *
     * @returns an Eigen::VectorXd containing the gradient
     */
    template <typename T>
    Eigen::VectorXd BulletWorldManager<T>::CalcGradientFromProximityAlertResult(
          const drake::multibody::MultibodyPlant<double>& plant,
          const drake::systems::Context<double>& context,
          const ProximityAlert::BoundWithGradient proximity_alert_result,
          const drake::geometry::GeometryId uncertain_obstacle_id) {
        // Compute the gradient of the risk from this obstacle.
        // This takes a bit of work, and has to be done for the one-shot and two-shot
        // results. In fact, it's involved enough that I need to write myself some notes here:
        // (for more details, see Dawson, Jasour, Hofmann, & Williams 2020)
        //
        // For each \epsilon (from either the one- or the two-shot), the gradient is given by:
        //     \nabla_q \epsilon = - X^2_3(\phi^{-1}(1-\epsilon)) * 2 x^T \Sigma^{-1} J
        // where
        //  - X^2_3 is the chi-squared PDF with 3 degrees of freedom (3 for 3D)
        //  - \phi^{-1} is the chi-squared inverse CDF with 3 degrees of freedom
        //  - x is the vector from the center of the colliding \epsilon-shadow ellipsoid to the robot,
        //    expressed in the world frame.
        //  - \Sigma^{-1} is the inverse of the covariance matrix
        //  - J is the Jacobian of the point of contact with the \epsilon-shadow with respect to the
        //    robot state (q), expressed in the world frame
        //
        // Fortunately, ProximityAlert already computes and returns:
        //     \nabla_x \epsilon = - X^2_3(\phi^{-1}(1-\epsilon)) * 2 x^T \Sigma^{-1},
        // so all we have to do is get the Jacobian of x w.r.t. q.
        //
        // Also fortunately, ProximityAlert also returns x. Less fortunately, it returns it as
        // a btVector3, so we will need to convert it.
        //
        // In summary, the game plan is this:
        //  1.) Check if the expansion hit part of the robot.
        //      1.a.) If it did, then get the body on the robot with which it makes contact.
        //          1.a.i.)  Use that body, the plant (and plant context), and x to get the Jacobian
        //                   of x with respect to robot state.
        //          1.a.ii.) Use the Jacobian to compute the gradient and store it
        //      1.b.) Otherwise, the risk is zero and the gradient is also zero

        // Initialize the gradient to zeros.
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(plant.num_positions());

        // Convert the contact point to an Eigen datatype
        Eigen::Vector3d contact_pt =
            toEigen(proximity_alert_result.contact_point_on_robot);

        // Find the body on the robot that the one-shot expansion hit. Do this by
        // matching the pointer to a collision object returned in the result struct to
        // a GeometryId in m_collision_objects
        drake::geometry::GeometryId collider_id;
        bool collider_found = false;
        std::map<drake::geometry::GeometryId, std::shared_ptr<btCollisionObject>>::iterator it;
        for (it = m_collision_objects.begin(); it != m_collision_objects.end(); it++) {
            // Check each object in m_collision_objects. If one matches the collider...
            if (it->second.get() == proximity_alert_result.collider) {
                // Save the GeometryId
                collider_id = it->first;
                collider_found = true;
                break;
            }
        }
        // If we found a match, continue to calculate the gradient; otherwise, leave the
        // gradient as zero and move on
        if (collider_found) {
            // Get the SceneGraphInspector for this plant's associated SceneGraph
            const auto& query_port = plant.get_geometry_query_input_port();
            const auto& query_object =
                query_port.template Eval<drake::geometry::QueryObject<double>>(context);
            const auto& inspector = query_object.inspector();

            drake::log()->debug("Found epsilon-shadow collider on robot: {}",
                                inspector.GetName(collider_id));

            // Get the the frame to which the collider geometry is attached
            const drake::geometry::FrameId frame_id = inspector.GetFrameId(collider_id);
            const drake::multibody::Body<double>* body = plant.GetBodyFromFrameId(frame_id);
            if (body == nullptr) {
                throw std::invalid_argument(
                    "Could not find body for collider frame ID!");
            }
            const drake::multibody::Frame<double>& collision_frame = body->body_frame();

            drake::log()->debug("\tattached to frame: {}",
                                collision_frame.name());

            // Also get the world frame, with respect to which we will calculate the Jacobian
            // const drake::multibody::Frame<double>& world_frame = plant.world_frame();

            // contact_pt is currently expressed in the world frame, but we need to convert
            // it to the collision_frame before getting the Jacobian.
            // First, get the transform of the collision_frame in the world
            //  (i.e. takes points in F to W)
            const drake::math::RigidTransform<double> X_WF = collision_frame.CalcPoseInWorld(context);
            // Then use the inverse to take contact_pt from W to F
            drake::log()->debug("Collision frame location:\n{}", X_WF.translation());
            drake::log()->debug("Contact pt in world: {}", contact_pt.transpose());
            drake::log()->debug("Contact normal in world (into robot): {}",
                                proximity_alert_result.contact_normal_into_robot.transpose());
            contact_pt = X_WF.inverse() * contact_pt;
            drake::log()->debug("Contact pt in collision_frame: {}", contact_pt.transpose());

            // Using this frame, compute the Jacobian of the contact point with respect to robot state
            // We'll use plant.CalcJacobianTranslationalVelocity(), but be careful of AutoDiffXd in 
            // the Jacobian! (I'll probably have to convert from an AutoDiffXd matrix to a double
            // matrix, then use that).
            // We also have to make sure to compute the Jacobian in the frame of the uncertain obstacle
            const drake::geometry::FrameId obstacle_frame_id = inspector.GetFrameId(uncertain_obstacle_id);
            const drake::multibody::Body<double>* obstacle_body = plant.GetBodyFromFrameId(obstacle_frame_id);
            if (body == nullptr) {
                throw std::invalid_argument(
                    "Could not find body for uncertain obstacle frame ID!");
            }
            const drake::multibody::Frame<double>& obstacle_frame = obstacle_body->body_frame();
            int num_generalized_positions = plant.num_positions();
            drake::MatrixX<double> jacobian(3, num_generalized_positions);
            plant.CalcJacobianTranslationalVelocity(
                context,
                drake::multibody::JacobianWrtVariable::kQDot,
                collision_frame,
                contact_pt,
                obstacle_frame,
                obstacle_frame,
                &jacobian);

            drake::log()->debug("Jacobian of collision_frame:\n{}", jacobian);
            drake::log()->debug("d_epsilon_d_x:\n{}", proximity_alert_result.d_epsilon_d_x);

            // Compute the gradient from the Jacobian!
            gradient = proximity_alert_result.d_epsilon_d_x * jacobian;
        }

        return gradient;
    }

    /*
     * Computes an upper bound on the risk of collision between the robot and
     * the uncertain obstacles. This overload (for AutoDiffXd's) returns both
     * the collision risk and the gradient of the collision risk.
     *
     * Before calling this function, you MUST call SynchronizeInternalWorld.
     *
     * @param plant: the MultibodyPlant representing the robot
     * @param context: the context specifying the state for the robot/environment
     * @param uncertain_obstacle_ids: the drake::geometry::GeometryIds of the uncertain obstacles
     * @param uncertain_obstacle_covariances: a vector of covariance matrices for each uncertain obstacle
     * @param tolerance: the computed upper bound is guaranteed to be within tolerance/2 of a true upper bound
     * @param result: the AutoDiffXd in which the risk bound and gradient will be stored.
     */
    template <typename T>
    void BulletWorldManager<T>::ComputeCollisionProbability(
            const drake::multibody::MultibodyPlant<T>& plant,
            const drake::systems::Context<T>& context,
            const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
            const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
            double tolerance, drake::AutoDiffXd& result) {
        // First of all, we need to do some sanity checks.
        // Tolerance needs to be positive and non-zero
        if (tolerance <= 0.0) {
            throw std::invalid_argument("Tolerance must be positive and non-zero");
        }
        // Also, we need to have the same number of uncertain obstacles as covariance matrices
        if (uncertain_obstacle_ids.size() != uncertain_obstacle_covariances.size()) {
            throw std::invalid_argument(
                "Number of provided IDs must match the number of provided covariance matrices");
        }

        // Get the SceneGraphInspector for this plant's associated SceneGraph
        const auto& query_port = plant.get_geometry_query_input_port();
        const auto& query_object =
            query_port.template Eval<drake::geometry::QueryObject<T>>(context);
        const auto& inspector = query_object.inspector();

        // Create a variable to accumulate the risk from each obstacle
        // Since we assume that accumulated_risk = \Sum risks from each obstacle,
        // then the gradient \nabla accumulated_risk = \Sum \nabla risks for each obstacle.
        double accumulated_risk = 0.0;
        Eigen::VectorXd accumulated_risk_gradient = Eigen::VectorXd::Zero(plant.num_positions());
        // Iterate through the uncertain obstacles
        for (int i = 0; i < static_cast<int>(uncertain_obstacle_ids.size()); i++) {
            // Get the ID and covariance for each obstacle
            drake::geometry::GeometryId uncertain_obstacle_id =
                uncertain_obstacle_ids[i];
            Eigen::Matrix3d uncertain_obstacle_covariance =
                uncertain_obstacle_covariances[i];

            // Make sure the ID is in the map of collision objects
            if (m_collision_objects.find(uncertain_obstacle_id) == m_collision_objects.end()) {
                throw std::invalid_argument("Unknown uncertain_obstacle_id. Did you SynchronizeInternalWorld?");
            }
            // Use the ID to get the collision object from the map
            std::shared_ptr<btCollisionObject> collision_object =
                m_collision_objects[uncertain_obstacle_id];

            drake::log()->debug("Checking risk of collision with obstacle: {}",
                            inspector.GetName(uncertain_obstacle_id));

            // Call ProximityAlert to get the risk of collision and gradient information
            ProximityAlert::CombinedBoundWithGradient proximity_alert_result;
            proximity_alert_result = ProximityAlert::compute_collision_probability_bound_and_gradient(
                collision_object.get(),
                uncertain_obstacle_covariance,
                tolerance,
                m_collision_world,
                ROBOT_FILTER_GROUP);

            // Accumulate the risk from each obstacle
            accumulated_risk += proximity_alert_result.epsilon;
            
            // Initialize vectors to store the gradients (filled with zeros)
            Eigen::VectorXd one_shot_gradient = Eigen::VectorXd::Zero(plant.num_positions());
            Eigen::VectorXd two_shot_gradient = Eigen::VectorXd::Zero(plant.num_positions());

            // If the one-shot expansion hit part of the robot, then it will result in
            // an epsilon greater than zero (within tolerance)
            if (proximity_alert_result.one_shot_result.epsilon > tolerance) {
                drake::log()->debug("One shot result: eps = {}", proximity_alert_result.one_shot_result.epsilon);
                one_shot_gradient = CalcGradientFromProximityAlertResult(
                    plant,
                    context,
                    proximity_alert_result.one_shot_result,
                    uncertain_obstacle_id);
            }

            // If the two-shot expansion hit part of the robot, then it will likewise
            // result in an epsilon greater than zero (within tolerance)
            if (proximity_alert_result.two_shot_result.epsilon > tolerance) {
                drake::log()->debug("Two shot result: eps = {}", proximity_alert_result.one_shot_result.epsilon);
                two_shot_gradient = CalcGradientFromProximityAlertResult(
                    plant,
                    context,
                    proximity_alert_result.two_shot_result,
                    uncertain_obstacle_id);
            }

            // Combine the one- and two-shot gradients, and accumulate them
            accumulated_risk_gradient += 0.5 * one_shot_gradient + 0.5 * two_shot_gradient;
        } 

        // Return the accumulated estimate in the designated reference
        drake::AutoDiffXd accumulated_risk_ad =
            drake::AutoDiffXd(accumulated_risk, accumulated_risk_gradient);
        result = accumulated_risk_ad;
    }

    /*
     * Given a context specifying the state of the robot and environment,
     * synchronize the internal btCollisionWorld so that it contains all the right
     * collision bodies in all the right places.
     *
     * @param plant: the MultibodyPlant representing the robot
     * @param context: the context for that MultibodyPlant
     */
    template <typename T>
    void BulletWorldManager<T>::SynchronizeInternalWorld(const drake::multibody::MultibodyPlant<T>& plant,
                                                         const drake::systems::Context<T>& context) {
        // Create an empty set of frozen geometries
        std::vector<drake::geometry::GeometryId> frozen_geometry_ids{};
        SynchronizeInternalWorld(plant, context, frozen_geometry_ids);
    }

    /*
     * Given a context specifying the state of the robot and environment,
     * synchronize the internal btCollisionWorld so that it contains all the right
     * collision bodies in all the right places.
     *
     * @param plant: the MultibodyPlant representing the robot
     * @param context: the context for that MultibodyPlant
     * @param frozen_geometry_ids: a vector of geometry ids that should not be updated (mainly used during testing)
     */
    template <typename T>
    void BulletWorldManager<T>::SynchronizeInternalWorld(const drake::multibody::MultibodyPlant<T>& plant,
                                                         const drake::systems::Context<T>& context,
                                                         const std::vector<drake::geometry::GeometryId> frozen_geometry_ids) {
        // Get the geometry query port
        const auto& query_port = plant.get_geometry_query_input_port();

        // Query that port to get the Geometry QueryObject
        const auto& query_object =
            query_port.template Eval<drake::geometry::QueryObject<T>>(context);
        // Also get the SceneGraphInspector while we're at it
        const auto& inspector = query_object.inspector();

        // Now we want to update our internal collision world to match the
        // geometries in the current context.

        // Get a vector of all geometry IDs
        const std::vector<drake::geometry::GeometryId> geometry_ids =
            inspector.GetAllGeometryIds();

        // The first step is to clear the collision world of any objects that
        // are no longer in the scene and update the positions of objects
        // that have already been added to the collision world
        drake::log()->debug("Updating collision world....");
        // int num_cleared = 0;
        for (drake::geometry::GeometryId id : geometry_ids) {
            // Get the name of the geometry
            std::string name = inspector.GetName(id);
            drake::log()->debug("\t Drake geometry: {}", inspector.GetName(id));

            // Skip it if it's in the set of frozen geometries
            if (std::find(frozen_geometry_ids.begin(), frozen_geometry_ids.end(), id) != frozen_geometry_ids.end()) {
                drake::log()->debug("\t\t Geometry is frozen; skipping.");
                continue;
            }

            // And check if it's in the map of objects in the collision world
            if (m_collision_objects.find(id) != m_collision_objects.end()) {
                drake::log()->debug("\t\t Already present in btCollisionWorld. Updating pose");
                // If the object is already there, we just need to make sure its
                // pose matches. To do this, we need the pose of the geometry
                // in the world
                drake::math::RigidTransform<T> world_pose = query_object.GetPoseInWorld(id);

                // We technically support either double or AutoDiffXd as the type T,
                // but in either case we need to give Bullet a double
                drake::math::RigidTransform<double> world_pose_d = toDouble(world_pose);
                // Extract the position and rotation from the pose
                Eigen::Matrix3d rotation_matrix = world_pose_d.rotation().matrix();
                Eigen::Vector3d translation = world_pose_d.translation();
                // Convert the rotation and position to Bullet data types
                btVector3 bt_translation = toBt(translation);
                btMatrix3x3 bt_rotation_matrix = toBt(rotation_matrix);

                // Get the btCollisionObject from the map
                std::shared_ptr<btCollisionObject> collision_object =
                    m_collision_objects[id];
                // Set the transform of that object in the btCollisionWorld
                btTransform new_object_pose = collision_object->getWorldTransform();
                new_object_pose.setOrigin(bt_translation);
                new_object_pose.setBasis(bt_rotation_matrix);
                collision_object->setWorldTransform(new_object_pose);

                // Once we've updated the position of this collision object, we need to update
                // the broadphase collision checker as well, so we don't cache the old position.
                m_collision_world->updateSingleAabb(collision_object.get());
            } else {
                // If we reach this point, the Drake geometry is not
                // yet represented in the btCollisionWorld.
                drake::log()->debug("\t\t Not present in btCollisionWorld.");

                // We only have to add the Drake geometry if it has Proximity properties
                if (!inspector.GetProximityProperties(id)) {
                    // This geometry has no proximity properties, so we can ignore it
                    // (all collision geometries should have these properties)
                    drake::log()->debug("\t\t Not collision geometry. Skipping.");
                    continue;
                }

                // If we reach this point, the geometry *is* a collision geometry,
                // so we can add it to the btCollisionWorld
                drake::log()->debug("\t\t Adding btCollisionObject for {}", name);
                
                // Get the type of shape, and make sure it's allowable
                const drake::geometry::Shape& shape = inspector.GetShape(id);
                std::string shape_name = drake::geometry::ShapeName(shape).name();
                drake::log()->debug("\t\t Drake shape is type {}", shape_name);

                // This case statement will create a btConvexShape based on
                // the drake geometry, or raise an error if the shape is not
                // an allowable shape type
                std::shared_ptr<btConvexShape> bt_shape;
                if (shape_name == "Sphere") {
                    // Cast to a sphere and get its radius
                    double sphere_radius = (static_cast<const drake::geometry::Sphere&>(shape)).radius();

                    // Make a btSphereShape to match
                    bt_shape = std::make_shared<btSphereShape>(static_cast<btScalar>(sphere_radius));
                } else if (shape_name == "Cylinder") {
                    // Cast to a cylinder and get its radius and length
                    double cyl_length = (static_cast<const drake::geometry::Cylinder&>(shape)).length();
                    double cyl_radius = (static_cast<const drake::geometry::Cylinder&>(shape)).radius();

                    // Make a btCylinderShapeZ, since Drake cylinders are implicitly
                    // oriented along the z axis.
                    bt_shape = std::make_shared<btCylinderShapeZ>(btVector3(
                        static_cast<btScalar>(cyl_radius), static_cast<btScalar>(cyl_radius),
                        static_cast<btScalar>(cyl_length) / 2.0 // Need "half-extents" of cylinder
                    ));
                } else if (shape_name == "Box") {
                    // Cast to a box and get its size, making sure to convert to
                    // half-extents rather than full-extents
                    Eigen::Vector3d box_half_extents = 0.5 * (static_cast<const drake::geometry::Box&>(shape)).size();
                    
                    // Make a btBoxShape using the resulting half-extents
                    bt_shape = std::make_shared<btBoxShape>(toBt(box_half_extents));
                } else if (shape_name == "Capsule") {
                    // Cast to a capsule and get its radius and length
                    double cap_length = (static_cast<const drake::geometry::Capsule&>(shape)).length();
                    double cap_radius = (static_cast<const drake::geometry::Capsule&>(shape)).radius();

                    // Make a btCapsuleShapeZ, since Drake capsules are implicitly
                    // oriented along the z axis.
                    bt_shape = std::make_shared<btCapsuleShapeZ>(
                        static_cast<btScalar>(cap_radius), static_cast<btScalar>(cap_length)
                    );
                } else if (shape_name == "Ellipsoid") {
                    // Cast to an ellipsoid and extract the axis half-lengths
                    double a = (static_cast<const drake::geometry::Ellipsoid&>(shape)).a();
                    double b = (static_cast<const drake::geometry::Ellipsoid&>(shape)).b();
                    double c = (static_cast<const drake::geometry::Ellipsoid&>(shape)).c();

                    // To make an ellipsoid in Bullet, we need to make a
                    // btMultiSphereShape, then scale that, since simple spheres
                    // don't support scaling. To do this, we make a unit sphere
                    // at the origin and then scale it.
                    btVector3 sphere_origin_bt(0.0, 0.0, 0.0);
                    btScalar sphere_radius_bt(1.0);
                    bt_shape = std::make_shared<btMultiSphereShape>(&sphere_origin_bt, &sphere_radius_bt, 1);
                    bt_shape->setLocalScaling(btVector3(
                        static_cast<btScalar>(a), static_cast<btScalar>(b), static_cast<btScalar>(c)
                    ));
                } else {
                    throw std::invalid_argument(
                        "BulletWorldManager does not support shapes with type " + shape_name);
                }

                // Now that we have the shape, make the btCollisionObject to hold it
                std::shared_ptr<btCollisionObject> collision_object = std::make_shared<btCollisionObject>();
                // Add the collision shape to the collision object
                collision_object->setCollisionShape(bt_shape.get());

                // Set the pose of the btCollisionObject to match the original
                // To do this, we need the pose of the geometry in the world
                drake::math::RigidTransform<T> world_pose = query_object.GetPoseInWorld(id);

                // We technically support either double or AutoDiffXd as the type T,
                // but in either case we need to give Bullet a double
                drake::math::RigidTransform<double> world_pose_d = toDouble(world_pose);
                // Extract the position and rotation from the pose
                Eigen::Matrix3d rotation_matrix = world_pose_d.rotation().matrix();
                Eigen::Vector3d translation = world_pose_d.translation();

                // Convert the rotation and position to Bullet data types
                btVector3 bt_translation = toBt(translation);
                btMatrix3x3 bt_rotation_matrix = toBt(rotation_matrix);

                // Set the transform of that object in the btCollisionWorld
                btTransform new_object_pose = collision_object->getWorldTransform();
                new_object_pose.setOrigin(bt_translation);
                new_object_pose.setBasis(bt_rotation_matrix);
                collision_object->setWorldTransform(new_object_pose);

                // Add the btCollisionObject to the btCollisionWorld.
                // Make sure to add the right filter group, which we check by testing
                // whether this geometry ID is in the set of robot geometry IDs
                drake::geometry::GeometrySetTester tester(&m_robot_geometry_set);
                if (tester.contains(id)) {
                    // If the geometry belongs to the robot, give it
                    // the ROBOT_FILTER_GROUP filter ID
                    drake::log()->debug("\t\t Shape is part of robot");
                    m_collision_world->addCollisionObject(
                        collision_object.get(),
                        ROBOT_FILTER_GROUP
                    );
                } else {
                    // Otherwise, we assume the geometry is an obstacle
                    drake::log()->debug("\t\t Shape is part of obstacle");
                    m_collision_world->addCollisionObject(
                        collision_object.get(),
                        OBSTACLE_FILTER_GROUP
                    );
                }

                // Also make sure to save a pointer to the newly created
                // btCollisionObject so we can get it later
                m_collision_objects[id] = collision_object;
                m_collision_shapes[id] = bt_shape;
            }
        }

        // Now all geometry in the Drake scene should be accurately represented
        // in the btCollisionWorld, but we still have to remove anything in the
        // btCollisionWorld that is no longer present in the Drake scene
        drake::log()->debug("Removing stranded geometry....");
        std::map<drake::geometry::GeometryId, std::shared_ptr<btCollisionObject>>::iterator it;
        for (it = m_collision_objects.begin(); it != m_collision_objects.end(); it++) {
            drake::geometry::GeometryId id_in_bt = it->first;
            std::shared_ptr<btCollisionObject> collision_object_ptr = it->second;

            // Check if this name is in the Drake scene
            bool present_in_drake = false;
            for (drake::geometry::GeometryId id_in_drake : geometry_ids) {
                if (id_in_bt == id_in_drake) {
                    present_in_drake = true;
                }
            }
            // If the name was not found in the Drake scene, remove this collision
            // object from the btCollisionWorld and from the map
            if (!present_in_drake) {
                drake::log()->debug("\t Removing object stored under ID {}", id_in_bt);
                m_collision_world->removeCollisionObject(collision_object_ptr.get());
                m_collision_objects.erase(id_in_bt);
                m_collision_shapes.erase(id_in_bt);
            }
        }

        // Now we're done updating the btCollisionWorld!
        drake::log()->debug("Done updating! Internal btCollisionWorld has {} collision objects.",
                            m_collision_world->getNumCollisionObjects());
        // No return value
    }

    /*
     * Clear the internal list of GeometryIds considered to be part of the robot.
     */
    template <typename T>
    void BulletWorldManager<T>::ClearRobotGeometryIds() {
        // Make a new, empty GeometrySet to replace the current one
        drake::geometry::GeometrySet new_set;
        m_robot_geometry_set = new_set;
    }

    /*
     * Adds the given GeometryIds to the internal list determining which geometries
     * are considered to be part of the robot.
     *
     * @param to_add: the vector of geometry IDs to add.
     */
    template <typename T>
    void BulletWorldManager<T>::AddRobotGeometryIds(const std::vector<drake::geometry::GeometryId> to_add) {
        m_robot_geometry_set.Add(to_add);
    }

    /*
     * Returns the number of geometries currently stored in the internal list of geometries
     * considered to be part of the robot.
     */
    template <typename T>
    int BulletWorldManager<T>::num_robot_geometry_ids() {
        drake::geometry::GeometrySetTester tester(&m_robot_geometry_set);
        return tester.num_geometries();
    }

    /*
     * Adds a random perturbation to the positions of all uncertain obstacles.
     *
     * @param uncertain_obstacle_ids: a vector of geometry IDs representing uncertain obstacles
     * @param uncertain_obstacle_covariances: a vector of covariance matrices for the uncertain obstacles
     */
    template <typename T>
    void BulletWorldManager<T>::PerturbUncertainObstacles(
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances
    ){
        // Loop through the uncertain obstacles, adding a random perturbation to each one.
        for (int i = 0; i < static_cast<int>(uncertain_obstacle_ids.size()); i++){
            // Get the btCollisionObject corresponding to this ID
            drake::geometry::GeometryId uncertain_obstacle_id = uncertain_obstacle_ids[i];
            std::shared_ptr<btCollisionObject> uncertain_obstacle = m_collision_objects[uncertain_obstacle_id];

            // Extract its world transform, and use that as the mean for the random position
            btTransform object_pose = uncertain_obstacle->getWorldTransform();
            Eigen::Vector3d nominal_position = toEigen(object_pose.getOrigin());
            Eigen::Matrix3d covariance = uncertain_obstacle_covariances[i];
            Eigen::Vector3d random_position = ProximityAlert::sample_from_multivariate_normal(nominal_position, covariance);

            // Update the object's position with the new origin
            object_pose.setOrigin(toBt(random_position));
            uncertain_obstacle->setWorldTransform(object_pose);

            // Once we've updated the position of this collision object, we need to update
            // the broadphase collision checker as well, so we don't cache the old position.
            m_collision_world->updateSingleAabb(uncertain_obstacle.get());

            drake::log()->debug("Perturbing id {} to position: {}",
                uncertain_obstacle_id, random_position.transpose());
        }
    }

    /*
     * Returns true if there is a collision between the robot and its environment in the current
     * state of the collision world
     *
     * @returns true if a collision exists in the current state, false otherwise
     */
    template <typename T>
    bool BulletWorldManager<T>::CheckCollision() {
        // To check for collision in the current state, we just have to check that each part of
        // the robot is collision-free (respecting filtering)

        // Start by making a callback for holding the result of the collision query
        ProximityAlert::UncertainContactResultCallback* collision_callback = new ProximityAlert::UncertainContactResultCallback(OBSTACLE_FILTER_GROUP);

        // Loop through all geometries
        drake::geometry::GeometrySetTester tester(&m_robot_geometry_set);
        for (auto const& collision_object_map_entry : m_collision_objects){
            // Only check if the object is part of the robot
            if (tester.contains(collision_object_map_entry.first)) {
                m_collision_world->contactTest(collision_object_map_entry.second.get(), *collision_callback);
                drake::log()->debug("Collision check on {} yields {}",
                                   collision_object_map_entry.first, collision_callback->bCollision);
            }
        }

        // Extract the result of the collision query and clean up before
        bool collision_occurred = collision_callback->bCollision;
        delete collision_callback;
        return collision_occurred;
    }
}

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::ccopt::BulletWorldManager)
