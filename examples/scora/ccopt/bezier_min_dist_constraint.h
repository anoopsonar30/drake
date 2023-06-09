/*
 * ADD COMMENT
 *
 * Written by Anoopkumar Sonar (with help from Hongkai Dai) on June 7, 2023
 */
#pragma once

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/framework/context.h"
#include "drake/multibody/inverse_kinematics/minimum_distance_constraint.h"
#include "drake/common/trajectories/bezier_curve.h"
#include <Eigen/Dense>

namespace ccopt {
    using drake::multibody::MinimumDistanceConstraint;
    using drake::trajectories::BezierCurve;

    class BezierCurveMinimalDistanceConstraint : public drake::solvers::Constraint {
    public:
        DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BezierCurveMinimalDistanceConstraint)

        // Construct a collision chance constraint for the given plant and plant context
        BezierCurveMinimalDistanceConstraint(BezierCurve<double> curve, double num_bezier_curve_parameter, double waypoint, double minimum_distance, const drake::multibody::MultibodyPlant<double>* const plant, drake::systems::Context<double>* plant_context);
        // Overload the constructor to allow for AutoDiffXd plants
        BezierCurveMinimalDistanceConstraint(BezierCurve<double> curve, double num_bezier_curve_parameter, double waypoint, double minimum_distance, const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* const plant, drake::systems::Context<drake::AutoDiffXd>* plant_context);

        // Destructor
        ~BezierCurveMinimalDistanceConstraint() override {};

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
            "BezierCurveMinimalDistanceConstraint::DoEval does not support symbolic variables.");
        }

        // Following the implementation of drake::multibody::DistanceConstraint,
        // we store the plant and context (in both double and AutoDiff form) as
        // class members. Also create members for the BulletWorldManager
        MinimumDistanceConstraint m_distance_constraint;
        
        const drake::multibody::MultibodyPlant<double>* const m_plant_double;
        drake::systems::Context<double>* m_plant_context_double;

        const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* const m_plant_autodiff;
        drake::systems::Context<drake::AutoDiffXd>* m_plant_context_autodiff; 

        BezierCurve<double> m_curve;
        double m_waypoint;
        double m_min_dist;
    };
}
