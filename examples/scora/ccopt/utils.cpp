/*
 * Defines a number of convenient types.
 *
 * Created on May 1, 2020 by Charles Dawson (cbd@mit.edu)
 */
#include "utils.h"

#include <drake/math/autodiff.h>

namespace ccopt {

    // Utility functions to convert from Eigen data types to Bullet data types
    btVector3 toBt(Eigen::Vector3d vector) {
        btVector3 vec(
            // (btScalar) vector[0],
            // (btScalar) vector[1],
            // (btScalar) vector[2]
            static_cast<btScalar>(vector[0]),
            static_cast<btScalar>(vector[1]),
            static_cast<btScalar>(vector[2])
        );
        return vec;
    }

    btMatrix3x3 toBt(Eigen::Matrix3d matrix) {
        btMatrix3x3 mat(
            // (btScalar) matrix(0, 0), (btScalar) matrix(0, 1), (btScalar) matrix(0, 2),
            // (btScalar) matrix(1, 0), (btScalar) matrix(1, 1), (btScalar) matrix(1, 2),
            // (btScalar) matrix(2, 0), (btScalar) matrix(2, 1), (btScalar) matrix(2, 2)
            static_cast<btScalar>(matrix(0, 0)), static_cast<btScalar>(matrix(0, 1)), static_cast<btScalar>(matrix(0, 2)),
            static_cast<btScalar>(matrix(1, 0)), static_cast<btScalar>(matrix(1, 1)), static_cast<btScalar>(matrix(1, 2)),
            static_cast<btScalar>(matrix(2, 0)), static_cast<btScalar>(matrix(2, 1)), static_cast<btScalar>(matrix(2, 2))
        );
        return mat;
    }

    // Vice-versa (Bullet to Eigen)
    Eigen::Vector3d toEigen(btVector3 vector_bt) {
        drake::Vector3<double> vector;
        vector << vector_bt.getX(), vector_bt.getY(), vector_bt.getZ();
        return vector;
    }

    // Utility function to convert from Drake RigidTransform to double template
    drake::math::RigidTransform<double> toDouble(drake::math::RigidTransform<double> xform) {
        // If we're given a double-templatized transform, just return it
        return xform;
    }
    drake::math::RigidTransform<double> toDouble(drake::math::RigidTransform<drake::AutoDiffXd> xform) {
        // If we're given an AutoDiffXd-templatized transform, we need to extract its
        // position and rotation, convert those to doubles, and return a new transform
        // containing the double values

        // Extract AutoDiffXd translation and rotation
        drake::Matrix3<drake::AutoDiffXd> rotation_matrix = xform.rotation().matrix();
        drake::Vector3<drake::AutoDiffXd> translation = xform.translation();

        // Define new translation and rotation with double template
        drake::Matrix3<double> rotation_matrix_d = drake::math::DiscardGradient(rotation_matrix);
        drake::Vector3<double> translation_d = drake::math::DiscardGradient(translation);

        // Construct a new transform from the double versions
        return drake::math::RigidTransform<double>(
            drake::math::RotationMatrix<double>(rotation_matrix_d),
            translation_d
        );
    }
    

}
