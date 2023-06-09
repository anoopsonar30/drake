/*
 * Wraps the BulletWorldManager to provide a Drake-compatible optimization
 * constraint limiting the risk of collision with uncertain obstacles.
 *
 * Written by Charles Dawson (cbd@mit.edu) on May 14, 2020
 */
#pragma once

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/framework/context.h"

#include <Eigen/Dense>

#include "bullet_world_manager.h"


namespace ccopt {

    /*
     * CollisionChanceConstraint constrains the probability of the robot colliding
     * with any uncertain obstacle at a given configuration:
     *
     *  P(collision | configuration) <= delta
     *
     * we do this by constraining:
     *
     *  delta - P(collision | configuration) \in [0, \infty)
     *
     * Credit where credit's due: the implementation of this class is based on the
     * example of drake::multibody::DistanceConstraint
     */
    class CollisionChanceConstraint : public drake::solvers::Constraint {
    public:
        DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CollisionChanceConstraint)

        // Construct a collision chance constraint for the given plant and plant context
        CollisionChanceConstraint(const drake::multibody::MultibodyPlant<double>* const plant,
                                  drake::systems::Context<double>* plant_context,
                                  BulletWorldManager<double>* world_manager,
                                  double risk_precision,
                                  double total_acceptable_risk,
                                  int T,
                                  bool use_max,
                                  const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
                                  const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances);
        // Overload the constructor to allow for AutoDiffXd plants
        CollisionChanceConstraint(const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* const plant,
                                  drake::systems::Context<drake::AutoDiffXd>* plant_context,
                                  BulletWorldManager<drake::AutoDiffXd>* world_manager,
                                  double risk_precision,
                                  double total_acceptable_risk,
                                  int T,
                                  bool use_max,
                                  const std::vector<drake::geometry::GeometryId> uncertain_obstacle_ids,
                                  const std::vector<Eigen::Matrix3d> uncertain_obstacle_covariances);

        // Destructor
        ~CollisionChanceConstraint() override {};

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
            "CollisionChanceConstraint::DoEval does not support symbolic variables.");
        }

        bool use_autodiff() const { return m_plant_autodiff; }

        // Following the implementation of drake::multibody::DistanceConstraint,
        // we store the plant and context (in both double and AutoDiff form) as
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
        bool m_use_max;
        const std::vector<drake::geometry::GeometryId> m_uncertain_obstacle_ids;
        const std::vector<Eigen::Matrix3d> m_uncertain_obstacle_covariances;
    };

}
