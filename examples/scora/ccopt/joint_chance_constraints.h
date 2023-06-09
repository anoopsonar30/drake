/*
 * Wraps the BulletWorldManager to provide a Drake-compatible optimization
 * constraint limiting the risk of collision with uncertain obstacles when
 * robot state is also uncertain. (In contrast, the standard `CollisionChanceConstraint`
 * considers only obstacle uncertainty, not actuation uncertainty as well).
 *
 * Written by Charles Dawson (cbd@mit.edu) on Nov 9, 2020
 */
#pragma once

#define EIGEN_NO_DEBUG

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/framework/context.h"

#include <Eigen/Dense>

#include "bullet_world_manager.h"
#include "chance_constraints.h"


namespace ccopt {

    /*
     * JointCollisionChanceConstraint constrains the probability of the robot colliding
     * with any uncertain obstacle at a given configuration, given uncertainty in state.
     * The way it does this requires some explanation. Let's say that a robot is "delta-safe"
     * with respect to some obstacle in some configuration if the risk of colliding with that
     * obstacle is less than delta, as measured by the epsilon shadows technique; i.e. a robot
     * is delta-safe in a configuration if
     *
     *  P(collision w/ obstacle i | configuration) <= delta
     *
     * This notion of safety captures the uncertainty from obstacle i; however, if the configuration
     * is itself a random variable, then the event "P(collision w/ obstacle i | configuration) <= delta,"
     * i.e. the event that "the robot is delta-safe w.r.t. obstacle i," is also subject to some probability.
     * We want to constrain that the robot is delta-safe with at least some probability, i.e.:
     *
     *  P(robot is delta-safe w.r.t. obstacle i) >= 1 - gamma
     *
     * Since delta-safety is based on the epsilon-shadow risk measure epsilon being less than delta,
     * we can also express this probability as
     *
     *  P(epsilon_i(q) <= delta) >= 1 - gamma
     *
     * We can approximate this probability using a linearization.
     *
     * In particular, we simultaneously enforce delta-safety for all obstacles at all timesteps:
     *
     * delta - sum_t sum_i epsilon_i(q_t) = delta - R(q) >= 0
     *
     * and gamma-robustness
     *
     * gamma + CDF(delta - R(q)) >= 1
     *
     */
    class JointCollisionChanceConstraint : public drake::solvers::Constraint {
    public:
        DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(JointCollisionChanceConstraint)

        // Construct a collision chance constraint for the given plant and plant context
        JointCollisionChanceConstraint(const drake::multibody::MultibodyPlant<double>* const plant,
                                       drake::systems::Context<double>* plant_context,
                                       BulletWorldManager<double>* world_manager,
                                       double risk_precision,
                                       int T,
                                       const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
                                       const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
                                       const Eigen::MatrixXd state_covariance);
        // Overload the constructor to allow for AutoDiffXd plants
        JointCollisionChanceConstraint(const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* const plant,
                                       drake::systems::Context<drake::AutoDiffXd>* plant_context,
                                       BulletWorldManager<drake::AutoDiffXd>* world_manager,
                                       double risk_precision,
                                       int T,
                                       const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
                                       const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances,
                                       const Eigen::MatrixXd state_covariance);

        // Destructor
        ~JointCollisionChanceConstraint() override {};

        // Helper functions to validate trajectories by randomly perturbing them, overloaded for different
        // scalar types.
        bool IsTrajectoryCollisionFreeUnderPerturbation(
            const Eigen::Ref<const Eigen::VectorXd>& traj_x, int subsamples);
        bool IsTrajectoryCollisionFreeUnderPerturbation(
            const Eigen::Ref<const drake::AutoDiffVecXd>& traj_x, int subsamples);

    private:
        // To make a custom drake::solvers::Constraint, we need to support three
        // different signatures for DoEval, which evaluate the constraint for doubles,
        // AutoDiffXds, and Symbolic variables (even though we don't support symbolic variables).
        void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                    Eigen::VectorXd* y) const override;

        void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd>& x,
                    drake::AutoDiffVecXd* y) const override;

        void DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>>&,
                    drake::VectorX<drake::symbolic::Expression>*) const override {
        throw std::logic_error(
            "JointCollisionChanceConstraint::DoEval does not support symbolic variables.");
        }

        bool use_autodiff() const { return m_plant_autodiff; }

        // Following the implementation of drake::multibody::DistanceConstraint,
        // we store pointers to the plant and context (in both double and AutoDiff form) as
        // class members. Also create members for the BulletWorldManager
        const drake::multibody::MultibodyPlant<double>* const m_plant_double;
        drake::systems::Context<double>* m_plant_context_double;
        BulletWorldManager<double>* m_world_manager_double;

        const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* const m_plant_autodiff;
        drake::systems::Context<drake::AutoDiffXd>* m_plant_context_autodiff;
        BulletWorldManager<drake::AutoDiffXd>* m_world_manager_autodiff;

        // These member variables store the parameters needed to compute collision risk
        double m_risk_precision;
        double m_total_acceptable_risk;
        int m_T;
        const std::vector<drake::geometry::GeometryId> m_uncertain_obstacle_ids;
        const std::vector<Eigen::Matrix3d> m_uncertain_obstacle_covariances;
        Eigen::MatrixXd m_state_covariance;
    };

}
