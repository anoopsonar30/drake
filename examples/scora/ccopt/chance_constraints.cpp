/*
 * Implements the Chance Constraint classes.
 *
 * Written by Charles Dawson (cbd@mit.edu) on May 14, 2020
 */
#include <stdexcept>
#include <iostream>
#include <limits>

#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include <drake/math/autodiff.h>
#include "drake/multibody/inverse_kinematics/distance_constraint_utilities.h"
#include "drake/multibody/inverse_kinematics/kinematic_evaluator_utilities.h"

#include "chance_constraints.h"

namespace ccopt {

    /*
     * Instantiate a CollisionChanceConstraint.
     *
     * @param plant: the Drake MultibodyPlant representing the robot
     * @param plant_context: the context for plant
     * @param risk_precision: a double specifying the precision to which risk estimates
                              should be computed.
     * @param total_acceptable_risk: the total risk acceptable across all waypoints.
     * @param T: the number of timesteps in the trajectory.
     * @param use_max: true if we constrain only the maximum risk on the trajectory,
     *                 false if we should accumulate risk using the inclusion-exclusion principle.
     */
    CollisionChanceConstraint::CollisionChanceConstraint(
        const drake::multibody::MultibodyPlant<double>* const plant,
        drake::systems::Context<double>* plant_context,
        BulletWorldManager<double>* world_manager,
        double risk_precision,
        double total_acceptable_risk,
        int T,
        bool use_max,
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances)
        : drake::solvers::Constraint(
            1,
            drake::multibody::internal::RefFromPtrOrThrow(plant).num_positions() * T,
            drake::Vector1d(0),  // probability cannot be less than 0
            drake::Vector1d(total_acceptable_risk)),
          m_plant_double{plant},
          m_plant_context_double{plant_context},
          m_world_manager_double{world_manager},
          m_plant_autodiff{nullptr},
          m_plant_context_autodiff{nullptr},
          m_world_manager_autodiff{nullptr},
          m_risk_precision{risk_precision},
          m_total_acceptable_risk{total_acceptable_risk},
          m_T{T},
          m_use_max{use_max},
          m_uncertain_obstacle_ids{uncertain_obstacle_ids},
          m_uncertain_obstacle_covariances{uncertain_obstacle_covariances}
    {
        // Make sure the plant has been wired to a Drake SceneGraph
        drake::multibody::internal::CheckPlantIsConnectedToSceneGraph(*m_plant_double, *m_plant_context_double);
    }


    /*
     * Overload the constructor to allow for AutoDiffXd plants
     */
    CollisionChanceConstraint::CollisionChanceConstraint(
        const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* const plant,
        drake::systems::Context<drake::AutoDiffXd>* plant_context,
        BulletWorldManager<drake::AutoDiffXd>* world_manager,
        double risk_precision,
        double total_acceptable_risk,
        int T,
        bool use_max,
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances)
        : drake::solvers::Constraint(
            1,
            drake::multibody::internal::RefFromPtrOrThrow(plant).num_positions() * T,
            drake::Vector1d(0),
            drake::Vector1d(total_acceptable_risk)),
          m_plant_double{nullptr},
          m_plant_context_double{nullptr},
          m_world_manager_double{nullptr},
          m_plant_autodiff{plant},
          m_plant_context_autodiff{plant_context},
          m_world_manager_autodiff{world_manager},
          m_risk_precision{risk_precision},
          m_total_acceptable_risk{total_acceptable_risk},
          m_T{T},
          m_use_max{use_max},
          m_uncertain_obstacle_ids{uncertain_obstacle_ids},
          m_uncertain_obstacle_covariances{uncertain_obstacle_covariances}
    {
        // Make sure the plant has been wired to a Drake SceneGraph
        drake::multibody::internal::CheckPlantIsConnectedToSceneGraph(*m_plant_autodiff, *m_plant_context_autodiff);
    }

    // To correctly handle AutoDiffs (and getting all derivatives where they need to go),
    // we need to provide overloaded functions for initializing and accumulating risk
    static double initialize_risk(const Eigen::Ref<const drake::VectorX<double>>& x,
                                  int num_timesteps,
                                  int num_positions) {
        (void)x;
        (void)num_timesteps;
        (void)num_positions;
                                    
        return 0.0;
    }
    static drake::AutoDiffXd initialize_risk(const Eigen::Ref<const drake::AutoDiffVecXd>& x,
                                             int num_timesteps,
                                             int num_positions) {
        
        (void)x;
        // The total risk tracker should have one derivative per position per timestep
        return drake::AutoDiffXd(0.0, Eigen::VectorXd::Zero(num_timesteps * num_positions));
    }

    static double accumulate_risk(double accumulated_risk, double risk,
                                  int t, int num_timesteps, int num_positions, bool use_max) {
        
        (void)t;
        (void)num_timesteps;
        (void)num_positions;

        if (use_max) {
            if (risk > accumulated_risk) {
                return risk;
            } else {
                return accumulated_risk;
            }
        } else {
            return accumulated_risk + risk;
        }
    }
    static drake::AutoDiffXd accumulate_risk(drake::AutoDiffXd accumulated_risk, drake::AutoDiffXd risk,
                                             int t, int num_timesteps, int num_positions, bool use_max) {
        // We need to make sure the derivatives match up, since the derivatives in risk
        // refer only to q[t*num_positions : (t+1)*num_positions], while those in accumulated_risk
        // refer to all of q.
        //
        // We can do this by making a new AutoDiffXd that has the same value of risk,
        // but the derivatives are in the right place
        Eigen::VectorXd expanded_derivatives = Eigen::VectorXd::Zero(num_timesteps * num_positions);
        expanded_derivatives.segment(t * num_positions, num_positions) = risk.derivatives();
        drake::AutoDiffXd new_risk = drake::AutoDiffXd(risk.value(), expanded_derivatives);

        // Now that we have everything sorted out, we can do the addition.
        if (use_max) {
            if (new_risk > accumulated_risk) {
                return new_risk;
            } else {
                return accumulated_risk;
            }
        } else {
            drake::log()->debug("\t\trisk derivative: {}", risk.derivatives().transpose());
            return accumulated_risk + new_risk;
        }
    }

    /*
     * Template function for evaluating the chance constraint. Called by DoEval
     */
    template <typename T, typename S>
    static void DoEvalGeneric(
        const drake::multibody::MultibodyPlant<T>& plant,
        drake::systems::Context<T>* context,
        BulletWorldManager<T>* world_manager,
        double risk_precision,
        int num_timesteps,
        const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
        const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
        bool use_max,
        const Eigen::Ref<const drake::VectorX<S>>& x,
        drake::VectorX<S>* y)
    {
        // Resize y to contain only a single scalar
        y->resize(1);

        // Make sure x is the correct size
        if (x.size() != num_timesteps * plant.num_positions()) {
            throw std::logic_error("x must contain plant->num_positions() element for each of an integer number of timesteps!");
        }

        drake::log()->info("Evaluating collision chance constraint...");

        // For each timestep, evaluate the chance constraint and accumulate the risks between timesteps.
        // This applies the union bound to find a conservative upper bound on risk along the trajectory.
        S accumulated_risk = initialize_risk(x, num_timesteps, plant.num_positions());
        for (int t = 0; t < num_timesteps; t++) {
            // Extract the various decision variables from x
            // (i.e. the DOF values for this waypoint)
            drake::VectorX<S> plant_coordinates = x.segment(t * plant.num_positions(),
                                                            plant.num_positions());

            // Update the plant context to reflect the plant_coordinates in x
            drake::multibody::internal::UpdateContextConfiguration(
                context,
                plant,
                plant_coordinates);

            // Refresh the world manager
            world_manager->SynchronizeInternalWorld(plant,
                                                    *context);

            // Compute the risk estimate
            S risk;
            world_manager->ComputeCollisionProbability(
                plant,
                *context,
                uncertain_obstacle_ids,
                uncertain_obstacle_covariances,
                risk_precision,
                risk
            );

            drake::log()->info("\tRisk at time {}: {}", t, risk);
            accumulated_risk = accumulate_risk(accumulated_risk, risk,
                                               t, num_timesteps, plant.num_positions(),
                                               use_max);
        }

        (*y)(0) = accumulated_risk;
        drake::log()->info("\tTotal risk: {}", accumulated_risk);
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
    void CollisionChanceConstraint::DoEval(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        Eigen::VectorXd* y) const
    {
        // If DoEval gets called with doubles, but we only have an autodiff plant,
        // we should wrap the doubles with AutoDiffs and pass to the auto-diff DoEval.
        // This ensures that DoEvalGeneric is only called with types that match the plant
        if (use_autodiff()) {
            drake::AutoDiffVecXd y_ad;
            Eval(drake::math::InitializeAutoDiff(x), &y_ad);
            *y = drake::math::ExtractValue(y_ad);
        } else {
            DoEvalGeneric(*m_plant_double, m_plant_context_double, m_world_manager_double,
                          m_risk_precision, m_T, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, m_use_max,
                          x, y);
        }
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
    void CollisionChanceConstraint::DoEval(
        const Eigen::Ref<const drake::AutoDiffVecXd>& x,
        drake::AutoDiffVecXd* y) const
    {
        if (use_autodiff()) {
            DoEvalGeneric(*m_plant_autodiff, m_plant_context_autodiff, m_world_manager_autodiff,
                          m_risk_precision, m_T, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, m_use_max,
                          x, y);
        } else {
            DoEvalGeneric(*m_plant_double, m_plant_context_double, m_world_manager_double,
                          m_risk_precision, m_T, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, m_use_max,
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
     * @param traj_x: a vector of T * #DOF doubles representing the trajectory of the robot.
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
        const Eigen::Ref<const drake::VectorX<S>>& x
    ){
        // Randomly perturb the obstacles
        world_manager->SynchronizeInternalWorld(plant, *context);
        world_manager->PerturbUncertainObstacles(uncertain_obstacle_ids,
                                                 uncertain_obstacle_covariances);

        // Loop through the timesteps. At each step, check for collision.
        // Stop when we find a collision
        bool collision_free = true;
        for (int t = 0; t < num_timesteps - 1; t++) {
            // Extract the DOF values for this point along the trajectory and the next point,
            // so we can interpolate
            drake::VectorX<S> q_t = x.segment(t * plant.num_positions(),
                                              plant.num_positions());
            drake::VectorX<S> q_t_next = x.segment((t+1) * plant.num_positions(),
                                                   plant.num_positions());
            
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
     * of uncertain obstacles in the environment.
     *
     * @param traj_x: a vector of T * #DOF doubles representing the trajectory of the robot.
     * @param subsamples: the number of samples to check in between each waypoint
     *
     * @returns true if there was no collision after perturbing obstacles, false otherwise
     */
    bool CollisionChanceConstraint::IsTrajectoryCollisionFreeUnderPerturbation(
        const Eigen::Ref<const Eigen::VectorXd>& traj_x,
        int subsamples
    ){
        if (use_autodiff()) {
            // Fall over to autodiff version
            return IsTrajectoryCollisionFreeUnderPerturbation(drake::math::InitializeAutoDiff(traj_x), subsamples);
        } else {
            return IsTrajectoryCollisionFreeUnderPerturbationGeneric(
                *m_plant_double, m_plant_context_double, m_world_manager_double,
                m_T, subsamples, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, traj_x);
        }
    };

    /*
     * Helper function to check a trajectory for collision after randomly perturbing the locations
     * of uncertain obstacles in the environment.
     *
     * @param traj_x: a vector of T * #DOF doubles representing the trajectory of the robot.
     * @param subsamples: the number of samples to check in between each waypoint
     *
     * @returns true if there was no collision after perturbing obstacles, false otherwise
     */
    bool CollisionChanceConstraint::IsTrajectoryCollisionFreeUnderPerturbation(
        const Eigen::Ref<const drake::AutoDiffVecXd>& traj_x,
        int subsamples
    ){
        if (use_autodiff()) {
            return IsTrajectoryCollisionFreeUnderPerturbationGeneric(
                *m_plant_autodiff, m_plant_context_autodiff, m_world_manager_autodiff,
                m_T, subsamples, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, traj_x);
        } else {
            return IsTrajectoryCollisionFreeUnderPerturbationGeneric(
                *m_plant_double, m_plant_context_double, m_world_manager_double,
                m_T, subsamples, m_uncertain_obstacle_ids, m_uncertain_obstacle_covariances, traj_x);
        }
    };

}