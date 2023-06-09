/*
 * Defines a number of convenient types.
 *
 * Created on May 1, 2020 by Charles Dawson (cbd@mit.edu)
 */
#pragma once
#include <Eigen/Dense>

#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>

#include "btBulletCollisionCommon.h"

namespace ccopt {

    // Utility functions to convert from Eigen data types to Bullet data types
    btVector3 toBt(Eigen::Vector3d vector);
    btMatrix3x3 toBt(Eigen::Matrix3d matrix);
    // And vice-versa
    Eigen::Vector3d toEigen(btVector3 vector);

    // Utility function to convert from Drake RigidTransform to double template
    drake::math::RigidTransform<double> toDouble(drake::math::RigidTransform<double> xform);
    drake::math::RigidTransform<double> toDouble(drake::math::RigidTransform<drake::AutoDiffXd> xform);

}
