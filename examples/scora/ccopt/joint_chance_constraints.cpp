/*
 * Implements the JointCollisionChanceConstraint class.
 *
 * Written by Charles Dawson (cbd@mit.edu) on Nov 10, 2020
 */
#include <stdexcept>
#include <iostream>
#include <limits>
#include <math.h>

#include <boost/math/distributions/normal.hpp> // for normal_distribution

#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include <drake/math/autodiff.h>
#include "drake/multibody/inverse_kinematics/distance_constraint_utilities.h"
#include "drake/multibody/inverse_kinematics/kinematic_evaluator_utilities.h"

#include "joint_chance_constraints.h"
#include "drake/examples/scora/ProximityAlert/proximityAlert.h"

namespace ccopt {

    /*
     * Instantiate a JointCollisionChanceConstraint. Considers only one timestep
     *
     * @param plant: the Drake MultibodyPlant representing the robot
     * @param plant_context: the context for plant
     * @param risk_precision: a double specifying the precision to which risk estimates
                              should be computed.
     * @param T: the number of timesteps
     * @param uncertain_obstacle_ids: a vector of drake::GeometryId speciying the uncertain obstacles
     * @param uncertain_obstacle_covariances: a vector of 3x3 double matrices representing the position
     *                                        covariance of each uncertain obstacle.
     * @param state_covariance: a NxN double matrix representing the covariance in the state of the robot
     *                          (N = # of DOFs in plant). The state is assumed to be Gaussian with this
     *                          covariance.
     */
    JointCollisionChanceConstraint::JointCollisionChanceConstraint(
        const drake::multibody::MultibodyPlant<double>* const plant,
        drake::systems::Context<double>* plant_context,
        BulletWorldManager<double>* world_manager,
        double risk_precision,
        int T,
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
        const Eigen::MatrixXd state_covariance)
        : drake::solvers::Constraint(
            2,
            drake::multibody::internal::RefFromPtrOrThrow(plant).num_positions()*T + 2,
            Eigen::Vector2d(risk_precision, 1),
            Eigen::Vector2d(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())),
          m_plant_double{plant},
          m_plant_context_double{plant_context},
          m_world_manager_double{world_manager},
          m_plant_autodiff{nullptr},
          m_plant_context_autodiff{nullptr},
          m_world_manager_autodiff{nullptr},
          m_risk_precision{risk_precision},
          m_T{T},
          m_uncertain_obstacle_ids{uncertain_obstacle_ids},
          m_uncertain_obstacle_covariances{uncertain_obstacle_covariances}
    {
        // Make sure the plant has been wired to a Drake SceneGraph
        drake::multibody::internal::CheckPlantIsConnectedToSceneGraph(*m_plant_double, *m_plant_context_double);

        // Assume the state covariance is constant
        // Assemble joint state covariance matrix
        int num_DOFs = drake::multibody::internal::RefFromPtrOrThrow(plant).num_positions();
        m_state_covariance = Eigen::MatrixXd::Zero(T*num_DOFs, T*num_DOFs);
        for (int t = 0; t < T; t++) {
            m_state_covariance.block(t*num_DOFs, t*num_DOFs,
                                     num_DOFs, num_DOFs) = state_covariance;
        }
    }


    /*
     * Overload the constructor to allow for AutoDiffXd plants
     */
    JointCollisionChanceConstraint::JointCollisionChanceConstraint(
        const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* const plant,
        drake::systems::Context<drake::AutoDiffXd>* plant_context,
        BulletWorldManager<drake::AutoDiffXd>* world_manager,
        double risk_precision,
        int T,
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
        const Eigen::MatrixXd state_covariance)
        : drake::solvers::Constraint(
            2,
            drake::multibody::internal::RefFromPtrOrThrow(plant).num_positions()*T + 2,
            Eigen::Vector2d(risk_precision, 1),
            Eigen::Vector2d(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())),
          m_plant_double{nullptr},
          m_plant_context_double{nullptr},
          m_world_manager_double{nullptr},
          m_plant_autodiff{plant},
          m_plant_context_autodiff{plant_context},
          m_world_manager_autodiff{world_manager},
          m_risk_precision{risk_precision},
          m_T{T},
          m_uncertain_obstacle_ids{uncertain_obstacle_ids},
          m_uncertain_obstacle_covariances{uncertain_obstacle_covariances}
    {
        // Make sure the plant has been wired to a Drake SceneGraph
        drake::multibody::internal::CheckPlantIsConnectedToSceneGraph(*m_plant_autodiff, *m_plant_context_autodiff);

        // Assume the state covariance is constant
        // Assemble joint state covariance matrix
        int num_DOFs = drake::multibody::internal::RefFromPtrOrThrow(plant).num_positions();
        m_state_covariance = Eigen::MatrixXd::Zero(T*num_DOFs, T*num_DOFs);
        for (int t = 0; t < T; t++) {
            m_state_covariance.block(t*num_DOFs, t*num_DOFs,
                                     num_DOFs, num_DOFs) = state_covariance;
        }
    }

    // To correctly handle AutoDiffs (and getting all derivatives where they need to go),
    // we need to provide overloaded functions for initializing and accumulating risk
    // static double initialize_risk(const Eigen::Ref<const drake::VectorX<double>>& x,
    //                               int num_positions) {
    //     (void)x;
    //     (void)num_positions;
        
    //     return 0.0;
    // }
    static drake::AutoDiffXd initialize_risk(const Eigen::Ref<const drake::AutoDiffVecXd>& x,
                                             int num_positions) {
        
        (void)x;
        (void)num_positions;

        // The total risk tracker should have one derivative per position per timestep
        return drake::AutoDiffXd(0.0, Eigen::VectorXd::Zero(num_positions));
    }

    // static double accumulate_risk(double accumulated_risk, double risk,
    //                               int t, int num_timesteps, int num_positions) {
        
    //     (void)t;
    //     (void)num_timesteps;
    //     (void)num_positions;
        
    //     return accumulated_risk + risk;
    // }
    static drake::AutoDiffXd accumulate_risk(drake::AutoDiffXd accumulated_risk, drake::AutoDiffXd risk,
                                             int t, int num_timesteps, int num_positions) {

        (void)num_timesteps;
        // We need to make sure the derivatives match up, since the derivatives in risk
        // refer only to q[t*num_positions : (t+1)*num_positions], while those in accumulated_risk
        // refer to all of q.
        //
        // We can do this by making a new AutoDiffXd that has the same value of risk,
        // but the derivatives are in the right place
        Eigen::VectorXd expanded_derivatives = Eigen::VectorXd::Zero(accumulated_risk.derivatives().size());
        expanded_derivatives.segment(t * num_positions, num_positions) = risk.derivatives();
        drake::AutoDiffXd new_risk = drake::AutoDiffXd(risk.value(), expanded_derivatives);

        // Now that we have everything sorted out, we can do the addition.
        drake::log()->debug("\t\trisk derivative: {}", risk.derivatives().transpose());
        return accumulated_risk + new_risk;
    }

    /*
     * Template function for evaluating the chance constraint. Called by DoEval
     */
    template <typename T>
    static void DoEvalGeneric(
        const drake::multibody::MultibodyPlant<T>& plant,
        drake::systems::Context<T>* context,
        BulletWorldManager<T>* world_manager,
        double risk_precision,
        int num_timesteps,
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
        const Eigen::MatrixXd state_covariance,
        const Eigen::Ref<const drake::VectorX<drake::AutoDiffXd>>& x,
        drake::VectorX<drake::AutoDiffXd>* y)
    {
        // Resize y to contain two scalars (one will enforce delta-safety, the other gamma-robustness)
        y->resize(2);

        // Make sure x is the correct size
        if (x.size() != plant.num_positions() * num_timesteps + 2) {
            throw std::logic_error("x must contain plant->num_positions() * num_timesteps + 2 elements!");
        }

        drake::log()->info("Evaluating collision chance constraint...");

        // Extract decision variables for delta and gamma (by convention, the second to last and last
        // decision variables respectively)
        drake::AutoDiffXd delta = x[x.size() - 2];
        drake::AutoDiffXd gamma = x[x.size() - 1];

        // Evaluate the chance constraint. This joint chance constraint only applies to one obstacle
        // at one timestep
        drake::log()->info("\tState overall: {}", x);
        drake::AutoDiffXd accumulated_risk = initialize_risk(x, num_timesteps * plant.num_positions() + 2);
        for (int t = 0; t < num_timesteps; t++) {
            // Extract the various decision variables from x
            // (i.e. the DOF values for this waypoint)
            drake::VectorX<drake::AutoDiffXd> plant_coordinates = x.segment(t * plant.num_positions(),
                                                                            plant.num_positions());

            // Update the plant context to reflect the plant_coordinates in x
            drake::multibody::internal::UpdateContextConfiguration(
                context,
                plant,
                plant_coordinates);

            // Refresh the world manager
            world_manager->SynchronizeInternalWorld(plant,
                                                    *context);

            // Compute the risk estimate using epsilon_shadows. What we're really interested in here
            // is the contact point between the shadow and the robot, and the Jacobian of that point
            drake::AutoDiffXd risk;
            world_manager->ComputeCollisionProbability(
                plant,
                *context,
                uncertain_obstacle_ids,
                uncertain_obstacle_covariances,
                risk_precision,
                risk
            );

            drake::log()->info("\tRisk at time {}: {}", t, risk);
            drake::log()->info("\tState at time {}: {}", t, plant_coordinates.transpose());
            accumulated_risk = accumulate_risk(accumulated_risk, risk,
                                               t, num_timesteps, plant.num_positions());
        }

        // We have two things to constrain. First, we want to enforce delta-safety, i.e.:
        //
        //      P(collision w/ obstacles | trajectory) <= delta
        //      delta - accumulated_risk >= 0
        (*y)(0) = delta - accumulated_risk;
        drake::log()->info("\tTotal risk: {}", accumulated_risk);
        drake::log()->info("\tc(1): {}", delta - accumulated_risk);

        // Next, we need to constrain
        //
        //      P(robot is delta-safe | uncertainty in state) >= 1 - gamma
        //      gamma + CDF(delta - accumulated_risk, sigma_z^2) >= 1
        //
        // where CDF is the cumulative distribution function of the Gaussian and
        // sigma_z^2 = grad_accumulated_risk^T state_covariance grad_accumulated_risk represents
        // the standard deviation of the marginal risk due to state uncertainty

        // We only want the derivatives with respect to q

        Eigen::VectorXd truncated_derivatives = accumulated_risk.derivatives().head(num_timesteps * plant.num_positions());
        drake::log()->debug("accumulated_risk derivs:\n{}", truncated_derivatives);
        drake::log()->debug("state_covar:\n{}", state_covariance);
        drake::log()->debug("state_covar * accumulated_risk derivs:\n{}", state_covariance * truncated_derivatives);
        drake::log()->debug("accumulated_risk derivs T:\n{}", truncated_derivatives.transpose());
        double z_variance = truncated_derivatives.dot(state_covariance * truncated_derivatives);
        drake::log()->info("\tzvar: {}", z_variance);
        if (z_variance > risk_precision) {
            // Build the CDF from a normal distribution with mean 0 and standard deviation sqrt(z_variance)
            boost::math::normal normal_distribution(0.0, sqrt(z_variance));
            // Define the second constrained quantity using the cdf
            drake::AutoDiffXd cdf_input = delta - accumulated_risk;
            drake::AutoDiffXd cdf_output = drake::AutoDiffXd(
                boost::math::cdf(normal_distribution, cdf_input.value()),
                cdf_input.derivatives() * boost::math::pdf(normal_distribution, cdf_input.value()));
            (*y)(1) = gamma + cdf_output;
        } else {
            // If z_variance is too small, then we can model the CDF as a step function
            drake::AutoDiffXd cdf_input = delta - accumulated_risk;
            drake::AutoDiffXd cdf_output;
            if (cdf_input.value() >= 0) {
                cdf_output = drake::AutoDiffXd(1, cdf_input.derivatives() * 0.0);
            } else {
                cdf_output = drake::AutoDiffXd(0, cdf_input.derivatives() * 0.0);
            }
            (*y)(1) = gamma + cdf_output;
        }
        drake::log()->info("\tc(2): {}", (*y)(1));
    }

    /*
     * Evaluate the chance constraint constraint for a given set of decision variables
     * x, where there are T * num_DOFs variables, and storing the (double) result in y.
     *
     * @param x: a vector of decision variables. There should be #DOF * #timestep decision variables
     *           representing the trajectory.
     * @param y: a pointer to a vector in which we'll store the result of evaluating this constraint:
     *              y(1) = delta - P(collision | configuration)
     */
    void JointCollisionChanceConstraint::DoEval(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        Eigen::VectorXd* y) const
    {
        // Evaluating this constraint requires gradient information, even if the caller is not asking
        // for it. As a result, if DoEval is called just with doubles then we need to convert to
        // AutoDiff and back once we're done with the computation.
        drake::AutoDiffVecXd y_ad;
        Eval(drake::math::InitializeAutoDiff(x), &y_ad);
        *y = drake::math::ExtractValue(y_ad);
    };

    /*
     * Evaluate the chance constraint constraint for a given set of (T * # DOFs),
     * decision variables x and storing the (AutoDiff) result in y.
     *
     * @param x: a vector of decision variables. There should be #DOF * #timestep decision variables
     *           representing the trajectory.
     * @param y: a pointer to a vector in which we'll store the result of evaluating this constraint:
     *              y(1) = delta - P(collision | configuration)
     */
    void JointCollisionChanceConstraint::DoEval(
        const Eigen::Ref<const drake::AutoDiffVecXd>& x,
        drake::AutoDiffVecXd* y) const
    {
        if (use_autodiff()) {
            DoEvalGeneric(*m_plant_autodiff, m_plant_context_autodiff, m_world_manager_autodiff,
                          m_risk_precision, m_T, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances,
                          m_state_covariance,
                          x, y);
        } else {
            DoEvalGeneric(*m_plant_double, m_plant_context_double, m_world_manager_double,
                          m_risk_precision, m_T, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances,
                          m_state_covariance,
                          x, y);
            drake::log()->debug("\tTotal risk derivative: {}", (*y)(0).derivatives().transpose());
        }
    };

    /*
     * Helper function to check a trajectory for collision after randomly perturbing the locations
     * of uncertain obstacles in the environment. Templated by plant type T and trajectory type S,
     *
     * @param plant: the MultiBodyPlant representing the robot
     * @param context: the context for plant
     * @param world_manager: a BulletWorldManager tasked with managing the uncertain obstacles
     * @param num_timesteps: the number of timesteps in the trajectory
     * @param subsamples: the number of samples to check for collision in between timesteps
     * @param uncertain_obstacle_ids: a vector of geometry IDs identifying all uncertain obstacles
     * @param uncertain_obstacle_covariances: a vector of covariance matrices for all uncertain obstacles
     * @param traj_x: a vector of T * #DOF doubles representing the (already randomly perturbed) trajectory of the robot.
     *
     * @returns true if there was no collision after perturbing obstacles, false otherwise
     */
    template <typename T, typename S>
    static bool IsTrajectoryCollisionFreeUnderPerturbationGeneric(
        const drake::multibody::MultibodyPlant<T>& plant,
        drake::systems::Context<T>* context,
        BulletWorldManager<T>* world_manager,
        int num_timesteps,
        int subsamples,
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
        const Eigen::MatrixXd state_covariance,
        const Eigen::Ref<const drake::VectorX<S>>& x
    ){
        // Randomly perturb the obstacles
        world_manager->SynchronizeInternalWorld(plant, *context);
        world_manager->PerturbUncertainObstacles(uncertain_obstacle_ids,
                                                 uncertain_obstacle_covariances);

        // Randomly perturb each waypoint
        int n_dims = plant.num_positions();
        drake::VectorX<S> x_perturbed = x;
        Eigen::VectorXd perturbation = ProximityAlert::sample_from_multivariate_normal(
            Eigen::VectorXd::Zero(x.size()), state_covariance);
        x_perturbed = x + perturbation;

        // Loop through the timesteps. At each step, check for collision.
        // Stop when we find a collision
        bool collision_free = true;
        for (int t = 0; t < num_timesteps - 1; t++) {
            // Extract the DOF values for this point along the trajectory and the next point,
            // so we can interpolate
            drake::VectorX<S> q_t = x_perturbed.segment(t * n_dims, n_dims);
            drake::VectorX<S> q_t_next = x_perturbed.segment((t+1) * n_dims, n_dims);

            // Loop through subsamples within the interval
            for (int i = 0; i < subsamples; i++) {
                drake::VectorX<S> current_q = q_t + static_cast<double>(i) / static_cast<double>(subsamples) * (q_t_next - q_t);

                // Update the plant context to reflect the plant_coordinates in x
                drake::multibody::internal::UpdateContextConfiguration(
                    context,
                    plant,
                    current_q);

                // Refresh the world manager
                world_manager->SynchronizeInternalWorld(plant,
                                                        *context,
                                                        uncertain_obstacle_ids);

                // Check if there's any collision
                world_manager->CheckCollision();
                if (world_manager->CheckCollision()) {
                    collision_free = false;
                    break;
                }
            }

            // Break if there's been a collision
            if (!collision_free) { break; }
        }
        return collision_free;
    };

    /*
     * Helper function to check a trajectory for collision after randomly perturbing the locations
     * of uncertain obstacles in the environment and randomly upsetting the trajectory.
     *
     * @param traj_x: a vector of T * #DOF doubles representing the trajectory of the robot.
     * @param subsamples: the number of samples to check in between each waypoint
     *
     * @returns true if there was no collision after perturbing obstacles, false otherwise
     */
    bool JointCollisionChanceConstraint::IsTrajectoryCollisionFreeUnderPerturbation(
        const Eigen::Ref<const Eigen::VectorXd>& traj_x,
        int subsamples
    ){
        if (use_autodiff()) {
            // Fall over to autodiff version
            return IsTrajectoryCollisionFreeUnderPerturbation(
                drake::math::InitializeAutoDiff(traj_x), subsamples);
        } else {
            return IsTrajectoryCollisionFreeUnderPerturbationGeneric(
                *m_plant_double, m_plant_context_double, m_world_manager_double,
                m_T, subsamples, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, m_state_covariance,
                traj_x);
        }
    };

    /*
     * Helper function to check a trajectory for collision after randomly perturbing the locations
     * of uncertain obstacles in the environment and randomly upsetting the trajectory.
     *
     * @param traj_x: a vector of T * #DOF doubles representing the trajectory of the robot.
     * @param subsamples: the number of samples to check in between each waypoint
     *
     * @returns true if there was no collision after perturbing obstacles, false otherwise
     */
    bool JointCollisionChanceConstraint::IsTrajectoryCollisionFreeUnderPerturbation(
        const Eigen::Ref<const drake::AutoDiffVecXd>& traj_x,
        int subsamples
    ){
        if (use_autodiff()) {
            return IsTrajectoryCollisionFreeUnderPerturbationGeneric(
                *m_plant_autodiff, m_plant_context_autodiff, m_world_manager_autodiff,
                m_T, subsamples, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, m_state_covariance,
                traj_x);
        } else {
            return IsTrajectoryCollisionFreeUnderPerturbationGeneric(
                *m_plant_double, m_plant_context_double, m_world_manager_double,
                m_T, subsamples, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, m_state_covariance,
                traj_x);
        }
    };

}