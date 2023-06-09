/*
 * ADD COMMENT
 *
 * Written by Anoopkumar Sonar (with help from Hongkai Dai) on June 7, 2023
 */

#include <stdexcept>
#include <iostream>
#include <limits>

#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include <drake/math/autodiff.h>

#include "drake/common/trajectories/bezier_curve.h"

#include "bezier_min_dist_constraint.h"

namespace ccopt {
    using drake::systems::Context;
    using drake::trajectories::BezierCurve;
    using drake::multibody::MultibodyPlant;
    using Vector1d = Eigen::Matrix<double, 1, 1>;

    BezierCurveMinimalDistanceConstraint::BezierCurveMinimalDistanceConstraint(
        BezierCurve<double> curve,
        double num_bezier_curve_parameter,
        double waypoint, 
        double minimum_distance, 
        const MultibodyPlant<double>* plant, 
        Context<double>* context)
        : Constraint(1, num_bezier_curve_parameter + 1, Vector1d(0), Vector1d(0)),
          m_distance_constraint(
            plant, 
            minimum_distance, 
            context),
          m_plant_double{plant},
          m_plant_context_double{context},
          m_plant_autodiff(nullptr),  
          m_plant_context_autodiff(nullptr), 
          m_curve{curve},
          m_waypoint{waypoint},
          m_min_dist{minimum_distance}
        {
            return;
        }

    void BezierCurveMinimalDistanceConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const {
        // First compute q from the curve and the waypoint
        (void) x;
        
        auto q = m_curve.value(m_waypoint);
        m_distance_constraint.Eval(q, y);
    }

}