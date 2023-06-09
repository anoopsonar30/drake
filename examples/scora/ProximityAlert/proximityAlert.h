/*
*  This file implements uncertainty-aware collision checking using the BulletCollision library
*
*  Author: Charles B. Dawson (cbd@mit.edu)
*  Written: Nov 15, 2019
*
*  All rights reserved. For now.
*/
#pragma once

#include "btBulletCollisionCommon.h"
#include "BulletCollision/CollisionShapes/btMinkowskiSumShape.h"

// linear algebra library
#include <Eigen/Dense>

// Probability distribution library
#include <boost/math/distributions/chi_squared.hpp>

// For square root
#include <math.h>

#include <iostream>

// For the cutoff sphere shape
#include "cutoffSphere.h"

// For various utility functions
#include "proximityAlertUtils.h"

// For logging
#include "drake/common/text_logging.h"

namespace ProximityAlert
{
    // Custom class for collecting the result of pairwise contact tests
    struct UncertainContactResultCallback : public btCollisionWorld::ContactResultCallback
    {
        // flag will be set to true if a collision is detected
        bool bCollision;

        // This vector stores the contact normal (pointing in towards the uncertain object)
        // if a collision has indeed occurred.
        btVector3 m_contact_normal;

        // This vector stores the contact point on the robot (in the world frame)
        // if a collision has indeed occurred.
        btVector3 m_contact_pt_on_robot;

        // This object will store the object the uncertain obstacle collides with, so that it can be referenced
        const btCollisionObject * m_collider;

        // This filter group can be used to optionally only check objects matching this mask
        int m_filter_group;

        UncertainContactResultCallback() :
            bCollision(false),
            m_contact_normal(0.0, 0.0, 0.0),
            m_contact_pt_on_robot(0.0, 0.0, 0.0),
            m_filter_group(-1)
        {}

        // Overload constructor to optionally accept a broadphase filter group
        UncertainContactResultCallback(int broadphase_filter_group) :
            bCollision(false),
            m_contact_normal(0.0, 0.0, 0.0),
            m_contact_pt_on_robot(0.0, 0.0, 0.0),
            m_filter_group(broadphase_filter_group)
        {}

        // We are going to override the needsCollision method from the superclass
        // Our version will do the same thing as the superclass method but also check
        // the broadphase ID
        bool needsCollision(btBroadphaseProxy* proxy0) const override
        {
            bool collides = btCollisionWorld::ContactResultCallback::needsCollision(proxy0);
            collides = (collides) &&
                        (proxy0->m_collisionFilterGroup & m_filter_group) &&
                        (m_filter_group & proxy0->m_collisionFilterGroup);
            return collides;
        }


        // This method is called when a collision is possibly detected (but it could be a
        // false positive, so we have to check)
        btScalar addSingleResult(btManifoldPoint& pt,
                                 const btCollisionObjectWrapper* col_obj0_wrap,
                                 int part_id0,
                                 int index0,
                                 const btCollisionObjectWrapper* col_obj1_wrap,
                                 int part_id1,
                                 int index1)
        {
            // TODO : Added to remove unused variable warning 
            (void)part_id0;
            (void)index0;
            (void)part_id1;
            (void)index1;

            // Collision will have occured if the distance is zero or negative
            float dist = pt.getDistance();

            // Figure out which collision object is the uncertain one and extract
            // the point on the uncertain object where the collision occurs, in the uncertain object frame
            btVector3 collision_point_on_uncertain_object;
            // Also save the point on the robot object where the collision occurs, in the world object frame
            btVector3 collision_point_on_robot_world_frame;
            // Also save the normal pointing in towards the uncertain object, in the uncertain object frame
            btVector3 contact_normal;
            // Also save the object we collide with
            const btCollisionObject * collider = nullptr;

            // We can tell which object is the uncertain one because it will not have a broadphase handle
            // (since the epsilon shadow is never added to the collision world).
            const btBroadphaseProxy* obj0_broadphase_proxy = col_obj0_wrap->getCollisionObject()->getBroadphaseHandle();
            const btBroadphaseProxy* obj1_broadphase_proxy = col_obj1_wrap->getCollisionObject()->getBroadphaseHandle();

            if (!obj0_broadphase_proxy && !obj1_broadphase_proxy) {
                // both objects are uncertain. This is unsupported so we'll just ignore it for now
                return 0;
            } else if (!obj0_broadphase_proxy) {
                // col_obj0_wrap is uncertain!

                // The world transform takes points in the object frame to the world frame, so to go from
                // world to object we need the inverse transform;
                collision_point_on_uncertain_object = col_obj0_wrap->getCollisionObject()->getWorldTransform().invXform(pt.getPositionWorldOnA());
                collision_point_on_robot_world_frame = pt.getPositionWorldOnB();

                contact_normal = pt.m_normalWorldOnB;

                // also save collider
                collider = col_obj1_wrap->getCollisionObject();
            } else if (!obj1_broadphase_proxy) {
                // col_obj1_wrap is uncertain!

                // The world transform takes points in the object frame to the world frame, so to go from
                // world to object we need the inverse transform;
                collision_point_on_uncertain_object = col_obj1_wrap->getCollisionObject()->getWorldTransform().invXform(pt.getPositionWorldOnA());
                collision_point_on_robot_world_frame = pt.getPositionWorldOnA();

                // Multiply by -1 because B is the uncertain object, and we want this to point in towards B
                contact_normal = (static_cast<btScalar>(-1))*pt.m_normalWorldOnB;

                // Save collider
                collider = col_obj0_wrap->getCollisionObject();
            } 
            else {
                // Neither object is uncertain. This is unsupported so we'll just default to standard collision
                // checking behavior for now.
            }

            // We only accept the collision if the distance is negative (not a false positive)
            if (dist <= 0.0) {
                bCollision = true;

                // save the collision normal pointing in towards the uncertain object
                m_contact_normal = contact_normal;
                m_contact_pt_on_robot = collision_point_on_robot_world_frame;

                if (collider == nullptr) {
                    // Handle the error, e.g., log it and return
                    drake::log()->debug("Error: collider is uninitialized.");
                    return 0.0;  // return a default btScalar value
                }

                // and the colliding object
                m_collider = collider;
            }

            return 0.0;
        }
    };


    // Custom struct for returning gradient information along with the probability estimate
    struct BoundWithGradient {
        float epsilon;

        /* To get the derivative of epsilon with respect to joint angles, you need to add
        * the Jacobian of the contact point with respect to joint angles, which is beyond
        * the scope of this code (which has zero knowledge of kinematics):
        *
        *   d_epsilon_d_theta = d_epsilon_d_x * Jacobian_{contact_point_on_robot}(theta)
        */ 
        Eigen::RowVector3d d_epsilon_d_x;
        Eigen::Vector3d contact_normal_into_robot;
        btVector3 contact_point_on_robot;
        const btCollisionObject* collider;
    };
    // Custom struct for returning two-shot gradient information along with the probability estimate
    struct CombinedBoundWithGradient {
        float epsilon;

        BoundWithGradient one_shot_result;
        BoundWithGradient two_shot_result;
    };


    /**
     * Return the probability of collision between an uncertain object and all other objects
     * in the collision world.
     *
     * Given an object with known orientation but uncertain 3D position (all other objects
     * are assumed to have known orientation and position), returns an upper bound epsilon
     * on the probability of collision between the two objects. Formally:
     *
     *      P(obj_A collides with any object in collision world) <= epsilon
     *
     * This bound is computed using the concept of an epsilon-shadow. An epsilon shadow of
     * some uncertain object can be thought of as an enlarged version of that object such
     * that the probability that the object is contained entirely within the shadow is
     * 1-epsilon. It follows that the probability of the object protruding beyond its
     * epsilon-shadow is epsilon. Thus, the probability that another object does collide
     * with the uncertain object can be no greater than epsilon if that object does not
     * intersect the epsilon-shadow. We use bisection line search to find a maximal epsilon
     * such that the epsilon-shadow of obj_A does not intersect with any other object in the
     * scene, which places an upper bound on the probability of collision between the two
     * objects.
     *
     * @param obj_A: a btCollisionObject with a btConvexShape collision shape.
     * @param sigma: an Eigen::Matrix3d (3x3 matrix of doubles) representing the covariance
     *               matrix for the position of obj_A. Note that the position of obj_A is
     *               assumed to be drawn from a multivariate Gaussian distribution with
     *               a mean equal to the location of obj_A. Also note that sigma must be
     *               symmetric and positive semidefinite. Additionally, for numerical reasons,
     *               sigma must be non-singular.
     * @param epsilon_tolerance: a float indicating the tolerance for error in the estimate
     *                           of the maximal epsilon.
     * @param collision_world: a btCollisionWorld object. Collision will be checked between
     *                         obj_A and all objects in collision_world. Note that obj_A
     *                         should *not* be included in collision_world.
     * @param filter_group: an optional integer masking the allowed filter groups.
     *
     * @return result: a CombinedBoundWithGradient struct containing the estimated risk
     *                      bound along with the information needed to compute the gradient.
     *
     */
    inline CombinedBoundWithGradient compute_collision_probability_bound_and_gradient(
        btCollisionObject *obj_A,
        const Eigen::Matrix3d sigma,
        float epsilon_tolerance,
        btCollisionWorld *collision_world,
        int filter_group = -1)
    {
        // We need to do some preliminary calculations to get the inverse of the covariance
        // matrix sigma. The eigenvalues and eigenvectors of this matrix define the shape and
        // orientation of the confidence region ellipsoid for points in obj_A (i.e. an
        // ellipsoid such that the probability of sampling a point with zero mean and sigma
        // covariance will have probability 1-epsilon of being contained within the ellipsoid).
        //
        // Eigen::EigenSolver has the nice property that it returns normalized eigenvectors.
        // Due to the principal axis theorem, since sigma is symmetric, its eigenvectors will
        // be orthogonal, meaning that they form an orthonormal basis for R^3.
        Eigen::Matrix3d sigma_inv = sigma.inverse();
        Eigen::EigenSolver<Eigen::Matrix3d> sigma_inv_eigensolver(sigma_inv);
        Eigen::Vector3d sigma_inv_eigenvalues = sigma_inv_eigensolver.eigenvalues().real();
        Eigen::Matrix3d sigma_inv_eigenvectors = sigma_inv_eigensolver.eigenvectors().real();

        // Now we need to convert those eigenvalues and eigenvectors into the btMatrix3x3 and
        // btVector3 data types that BulletCollision understands. This is also a good opportunity
        // to choose an ordering for the orthonormal eigenvalue basis. We would like an ordering
        // that yields a change-of-basis matrix from the standard basis to the eigenbasis with
        // determinant +1 (i.e. a proper rotation). The basis strategy for this is simple: if at
        // first we don't succeed: try, try again. If we find that we have an improper rotation
        // (i.e. a rotation + reflection) from the standard basis to our eigenbasis, simply swapping
        // the order of two eigenvalues will flip the sign of the determinant, yielding a proper
        // rotation. Note that we also need to track the new ordering.
        // btMatrix3x3 eigenbasis_bt(
        //     (btScalar) sigma_inv_eigenvectors(0, 0), (btScalar) sigma_inv_eigenvectors(0, 1), (btScalar) sigma_inv_eigenvectors(0, 2),
        //     (btScalar) sigma_inv_eigenvectors(1, 0), (btScalar) sigma_inv_eigenvectors(1, 1), (btScalar) sigma_inv_eigenvectors(1, 2),
        //     (btScalar) sigma_inv_eigenvectors(2, 0), (btScalar) sigma_inv_eigenvectors(2, 1), (btScalar) sigma_inv_eigenvectors(2, 2)
        // );
        btMatrix3x3 eigenbasis_bt(
                static_cast<btScalar>(sigma_inv_eigenvectors(0, 0)), static_cast<btScalar>(sigma_inv_eigenvectors(0, 1)), static_cast<btScalar>(sigma_inv_eigenvectors(0, 2)),
                static_cast<btScalar>(sigma_inv_eigenvectors(1, 0)), static_cast<btScalar>(sigma_inv_eigenvectors(1, 1)), static_cast<btScalar>(sigma_inv_eigenvectors(1, 2)),
                static_cast<btScalar>(sigma_inv_eigenvectors(2, 0)), static_cast<btScalar>(sigma_inv_eigenvectors(2, 1)), static_cast<btScalar>(sigma_inv_eigenvectors(2, 2))
            );
        int basis_ordering[] = {0, 1, 2};

        // If determinant = -1, we have a rotation with reflection, so we need to swap two basis vectors
        if (eigenbasis_bt.determinant() < 0) {
            // swap first and second basis vectors
            eigenbasis_bt = btMatrix3x3(
                static_cast<btScalar>(sigma_inv_eigenvectors(0, 1)), static_cast<btScalar>(sigma_inv_eigenvectors(0, 0)), static_cast<btScalar>(sigma_inv_eigenvectors(0, 2)),
                static_cast<btScalar>(sigma_inv_eigenvectors(1, 1)), static_cast<btScalar>(sigma_inv_eigenvectors(1, 0)), static_cast<btScalar>(sigma_inv_eigenvectors(1, 2)),
                static_cast<btScalar>(sigma_inv_eigenvectors(2, 1)), static_cast<btScalar>(sigma_inv_eigenvectors(2, 0)), static_cast<btScalar>(sigma_inv_eigenvectors(2, 2))
            );
            basis_ordering[0] = 1;
            basis_ordering[1] = 0;
        }

        // For efficiency, we need to initialize some objects before starting the line search loop.

        // This vector will store the lengths of the principal axes of the confidence region ellipsoid.
        btVector3 ellipsoid_axis_lengths_bt(static_cast<btScalar>(0.0), static_cast<btScalar>(0.0), static_cast<btScalar>(0.0));

        // Initialize a chi-squared distribution with 3 degrees of freedom
        boost::math::chi_squared chi_squared_dist_3dof = boost::math::chi_squared(3);

        // Initialize a unit sphere that we will rotate and scale to produce the confidence-region ellipsoid
        // We need to use a MultiSphereShape to avoid the issues arising from BulletCollision's implicit
        // representation of spheres.
        btVector3 sphere_origin_bt(0.0, 0.0, 0.0);
        btScalar sphere_radius_bt(1.0);
        btMultiSphereShape * confidence_interval_ellipsoid = new btMultiSphereShape(&sphere_origin_bt, &sphere_radius_bt, 1);
        
        // we then construct a rotation to bring the principal axes of this unit sphere into alignment with
        // the axes of the confidence region ellipsoid.
        btTransform shadow_transform(
            eigenbasis_bt,
            btVector3(static_cast<btScalar>(0.0), static_cast<btScalar>(0.0), static_cast<btScalar>(0.0))
        );

        // We can also pre-allocate the collision object for the shadow (we can change its shape as we go)
        btCollisionObject* epsilon_shadow_collision_obj = new btCollisionObject();
        // Put the shadow in the same location as obj_A
        epsilon_shadow_collision_obj->setWorldTransform(obj_A->getWorldTransform());

        // We can also save some computation by precalculating the reciprocal of the square root of each eigenvalue
        Eigen::Vector3d sqrt_eigenvalue_reciprocal = sigma_inv_eigenvalues.cwiseInverse();
        sqrt_eigenvalue_reciprocal = sqrt_eigenvalue_reciprocal.cwiseSqrt();

        // This variable will store the contact normal between obj_A and the closest object in the scene once
        // we're done with the line search. See the block comment after the line search while loop for an
        // explanation of why we need this
        btVector3 contact_normal(0.0, 0.0, 0.0);

        // To allow us to calculate gradient information later, we need to store the point on the robot
        // at which collision with the maximal epsilon shadow occurs
        btVector3 contact_point_on_robot;
        // as well as the actual collision object on the robot we collided with
        const btCollisionObject* collider;

        // Now we need to do a line search to find an (approximate) maximal epsilon such that the epsilon shadow
        // of obj_A does not collide with any other object. This line search will maintain an upper and lower bound
        // on epsilon, then repeatedly test collision between the shadow of obj_A and all other objects using the
        // midpoint of the epsilon range.
        //      If the midpoint is not in collision, we set epsilon_ub = midpoint (lowering risk dilates shadow).
        //      If the midpoint is in collision, we set epsilon_lb = midpoint (increasing risk shrinks shadow).
        //      When the size of the range is less than the specified tolerance, terminate
        float epsilon_ub = 1.0;
        float epsilon_lb = 0.0;
        float epsilon = (epsilon_ub + epsilon_lb) / 2.0;
        // the length of the interval halves at each step, so we need order -log2(tolerance) steps to meet the tolerance
        float last_epsilon = epsilon_lb;
        last_epsilon += 0; // TODO : Added to remove unused variable warning 
        int iter_count = 0; // for diagnostics
        float quantile;
        while (epsilon_ub - epsilon_lb > epsilon_tolerance) {
            iter_count++;
            // Start by creating the shadow
            // To do that, we need to get the constant that determines the size of the ellipsoid so that it
            // contains 1-epsilon of the probability mass of the multivariate normal distribution, which is
            // the inverse of the chi-squared cumulative distribution function with 3 degrees of freedom.
            quantile = boost::math::quantile(chi_squared_dist_3dof, 1.0-epsilon);

            // This relates to the length of each principle axis of the confidence region ellipsoid by
            // the formula length_i = sqrt(quantile) / sqrt(eigenvalue_i)
            Eigen::Vector3d ellipsoid_scale_vector = sqrt(quantile) * sqrt_eigenvalue_reciprocal;
            // Store result for each axis in a btVector3, respecting the ordering of the eigenbasis
            ellipsoid_axis_lengths_bt.setX(ellipsoid_scale_vector(basis_ordering[0]));
            ellipsoid_axis_lengths_bt.setY(ellipsoid_scale_vector(basis_ordering[1]));
            ellipsoid_axis_lengths_bt.setZ(ellipsoid_scale_vector(basis_ordering[2]));

            // Now we can scale the unit sphere by these factors, yielding the ellipsoid we want.
            // Each successive call to setLocalScaling overwrites any previous scaling, so we don't
            // have to worry about that here.
            confidence_interval_ellipsoid->setLocalScaling(ellipsoid_axis_lengths_bt);

            // Now we can create the epsilon shadow of obj_A using a Minkowski sum of the original
            // shape of object A and the confidence region
            btMinkowskiSumShape * epsilon_shadow = new btMinkowskiSumShape(static_cast<btConvexShape*>(obj_A->getCollisionShape()),
                                                                           confidence_interval_ellipsoid);
            // Make sure to rotate the ellipsoid to align with the eigenbasis
            epsilon_shadow->setTransformB(shadow_transform);
            // And assign the shadow to the wrapping collision object
            epsilon_shadow_collision_obj->setCollisionShape(epsilon_shadow);


            // Now we're finally ready to do a collision check
            // Next we instantiate a pairwise collision callback
            UncertainContactResultCallback* collision_callback = new UncertainContactResultCallback(filter_group);
            // Then we perform the check
            collision_world->contactTest(epsilon_shadow_collision_obj, *collision_callback);
            
            // Then we look to see if the check found a collision
            // If there was a collision, we increase the risk (shrinking the shadow) by raising the lower bound.
            // Otherwise, decrease the risk (expand the shadow) by lowering the upper bound
            if (collision_callback->bCollision) {
                epsilon_lb = epsilon;

                // if there was a collision, we save the contact normal and the object we collided with
                contact_normal = collision_callback->m_contact_normal;
                contact_point_on_robot = collision_callback->m_contact_pt_on_robot;
                collider = collision_callback->m_collider;
            } else {
                epsilon_ub = epsilon;
            }

            // Update epsilon as the midpoint of the new interval
            epsilon = (epsilon_ub + epsilon_lb) / 2.0;

            // Clean up the objects allocated during this loop
            delete collision_callback;
            delete epsilon_shadow;
        }
        // normalize contact normal (if it's not zero, it should be a unit vector)
        if (contact_normal.length() > 0.00001) {
            contact_normal = contact_normal.normalized();
        }

        // Make a struct to store the result and gradient info from this first search
        BoundWithGradient one_shot_result;
        one_shot_result.epsilon = epsilon;
        // contact_normal points into the UNCERTAIN object, so we negate it.
        one_shot_result.contact_normal_into_robot << (static_cast<btScalar>(-1.0)) * contact_normal.getX(),
                                                     (static_cast<btScalar>(-1.0)) * contact_normal.getY(),
                                                     (static_cast<btScalar>(-1.0)) * contact_normal.getZ();
        one_shot_result.contact_point_on_robot = contact_point_on_robot;
        one_shot_result.collider = collider;

        drake::log()->debug("First-shot epsilon: {}", epsilon);
        if (collider) {
            drake::log()->debug("\tcollider location: {} {} {}", collider->getWorldTransform().getOrigin().getX(),
                                                                 collider->getWorldTransform().getOrigin().getY(),
                                                                 collider->getWorldTransform().getOrigin().getZ());
        } else {
            drake::log()->debug("\tNo collider");
        }
        drake::log()->debug("\tcontact normal: {} {} {}", contact_normal.getX(),
                                                          contact_normal.getY(),
                                                          contact_normal.getZ());
        drake::log()->debug("\tcontact pt (on robot): {} {} {}", contact_point_on_robot.getX(),
                                                                 contact_point_on_robot.getY(),
                                                                 contact_point_on_robot.getZ());

        // Calculating the gradient is a bit involved. See write-up for derivation
        // Start with the derivative of epsilon w.r.t. r where r = x^T Sigma^-1 x (the "radius" of the ellipsoid)
        float d_epsilon_d_r = (-1.0)*boost::math::pdf(chi_squared_dist_3dof, quantile);
        // Get contact point in confidence interval ellipsoid frame
        btTransform epsilon_shadow_object_rotation = btTransform(epsilon_shadow_collision_obj->getWorldTransform().getBasis());

        btVector3 contact_normal_into_robot_shadow_frame =
            -1.0 * shadow_transform.invXform(epsilon_shadow_object_rotation.invXform(contact_normal));
        drake::log()->debug("\tcontact normal (shadow frame): {} {} {}", contact_normal_into_robot_shadow_frame.getX(),
                                                                         contact_normal_into_robot_shadow_frame.getY(),
                                                                         contact_normal_into_robot_shadow_frame.getZ());

        // We want to get the support of the shadow ellipse in the obstacle (obj_A) frame,
        // but it starts in the world frame, so we need to convert first from world to obstacle,
        // then obstacle to ellipse.
        btVector3 x = confidence_interval_ellipsoid->localGetSupportingVertex(
            -1.0 * shadow_transform.invXform(
                epsilon_shadow_object_rotation.invXform(contact_normal)));
        drake::log()->debug("\tlocal support: {} {} {}", x.getX(),
                                                         x.getY(),
                                                         x.getZ());
        // However, we then need to convert back to the obstacle frame, then convert to Eigen format to return.
        x = (shadow_transform * x);
        Eigen::Vector3d x_eigen;
        x_eigen << x.getX(), x.getY(), x.getZ();

        // Now derivative of epsilon w.r.t. x
        one_shot_result.d_epsilon_d_x = d_epsilon_d_r * 2.0 * x_eigen.transpose() * sigma_inv;

        drake::log()->debug("\tcoll object xform angle: {}", epsilon_shadow_collision_obj->getWorldTransform().getRotation().getAngle());
        drake::log()->debug("\tx: {}", x_eigen.transpose());
        drake::log()->debug("\tdeps_dx: {}", one_shot_result.d_epsilon_d_x);

        // At this point, epsilon contains an upper bound on the collision probability. However, it is
        // very conservative, since it sees any event where the uncertain object protrudes beyond the
        // confidence region ellipsoid as a collision risk (even if the uncertain object protrudes in
        // a direction *away* from the object the shadow collides with above). To tighten this bound,
        // we can use just the half of the epsilon-shadow facing towards the nearest collision, then
        // construct a larger epsilon-shadow facing away from the nearest collision. A one dimensional
        // ascii explanation is below:
        //
        // Let O be the uncertain object and X be other objects in the scene, and let (---O---) be the
        // epsilon=1/3 shadow of O (1/(3 dashes)). After the first line search, we get:
        //
        //       X   (---O---)X         : risk of collision is overestimates
        //
        // If we observe that the half-ellipsoid O---) has half the probability mass of (---O---), then
        // we can combine two half-ellipsoids: (------O and O---) to get:
        //
        //       X(------O---)X         : more accurate risk of collision
        //
        // To find this larger half-ellipsoid, we save the contact normal from the smaller ellipsoid,
        // then use it to construct a new confidence interval ellipse that is intersected with the half-space
        // facing away from the collision, and then use that new confidence interval to construct shadows.
        epsilon_ub = epsilon;
        epsilon_lb = 0.0;
        float epsilon_small = epsilon;
        float epsilon_large = (epsilon_ub + epsilon_lb) / 2.0;
        // the length of the interval halves at each step, so we need order -log2(tolerance) steps to meet the tolerance
        last_epsilon = epsilon_lb;

        // Reset the collider and contact normal for the second search
        collider = 0;
        btVector3 two_shot_contact_normal(0.0, 0.0, 0.0);

        // Construct a new confidence interval ellipsoid that is intersected with the half-space
        // facing away from collision
        delete confidence_interval_ellipsoid;
        // We want the contact normal pointing into the epsilon shadow now, so we flip it
        CutoffSphere * cutoff_confidence_interval_ellipsoid = new CutoffSphere(&sphere_origin_bt,
                                                                               &sphere_radius_bt,
                                                                               contact_normal_into_robot_shadow_frame);


        while (epsilon_ub - epsilon_lb > epsilon_tolerance) {
            iter_count++;
            // Start by creating the shadow
            // To do that, we need to get the constant that determines the size of the ellipsoid so that it
            // contains 1-epsilon_large of the probability mass of the multivariate normal distribution, which is
            // the inverse of the chi-squared cumulative distribution function with 3 degrees of freedom.
            quantile = boost::math::quantile(chi_squared_dist_3dof, 1.0-epsilon_large);

            // This relates to the length of each principle axis of the confidence region ellipsoid by
            // the formula length_i = sqrt(quantile) / sqrt(eigenvalue_i)
            Eigen::Vector3d ellipsoid_scale_vector = sqrt(quantile) * sqrt_eigenvalue_reciprocal;
            // Store result for each axis in a btVector3, respecting the ordering of the eigenbasis
            ellipsoid_axis_lengths_bt.setX(ellipsoid_scale_vector(basis_ordering[0]));
            ellipsoid_axis_lengths_bt.setY(ellipsoid_scale_vector(basis_ordering[1]));
            ellipsoid_axis_lengths_bt.setZ(ellipsoid_scale_vector(basis_ordering[2]));

            // Now we can scale the unit sphere by these factors, yielding the ellipsoid we want.
            // Each successive call to setLocalScaling overwrites any previous scaling, so we don't
            // have to worry about that here.
            cutoff_confidence_interval_ellipsoid->setLocalScaling(ellipsoid_axis_lengths_bt);

            // Now we can create the epsilon shadow of obj_A using a Minkowski sum of the original
            // shape of object A and the confidence region
            btMinkowskiSumShape * epsilon_shadow = new btMinkowskiSumShape(static_cast<btConvexShape*>(obj_A->getCollisionShape()),
                                                                           cutoff_confidence_interval_ellipsoid);
            // Make sure to rotate the ellipsoid to align with the eigenbasis
            epsilon_shadow->setTransformB(shadow_transform);
            // And assign the shadow to the wrapping collision object
            epsilon_shadow_collision_obj->setCollisionShape(epsilon_shadow);


            // Now we're finally ready to do a collision check
            // Next we instantiate a pairwise collision callback
            UncertainContactResultCallback* collision_callback = new UncertainContactResultCallback(filter_group);

            // Then we perform the check
            collision_world->contactTest(epsilon_shadow_collision_obj, *collision_callback);

            // Then we look to see if the check found a collision
            // If there was a collision, we increase the risk (shrinking the shadow) by raising the lower bound.
            // Otherwise, decrease the risk (expand the shadow) by lowering the upper bound
            if (collision_callback->bCollision) {
                epsilon_lb = epsilon_large;

                // if there was a collision, we save the contact normal and the object we collided with
                two_shot_contact_normal = collision_callback->m_contact_normal;
                contact_point_on_robot = collision_callback->m_contact_pt_on_robot;
                collider = collision_callback->m_collider;
            } else {
                epsilon_ub = epsilon_large;
            }

            // Update epsilon_large as the midpoint of the new interval
            epsilon_large = (epsilon_ub + epsilon_lb) / 2.0;

            // Clean up the objects allocated during this loop
            delete collision_callback;
            delete epsilon_shadow;
        }
        // normalize contact normal if not zero (it's a unit vector otherwise)
        if (two_shot_contact_normal.length() >= 0.0001) {
            two_shot_contact_normal = two_shot_contact_normal.normalized();
        }

        // Make a struct to store the result and gradient info from the second search
        BoundWithGradient two_shot_result;
        two_shot_result.epsilon = epsilon_large;
        two_shot_result.contact_normal_into_robot << (static_cast<btScalar>(-1.0)) * two_shot_contact_normal.getX(),
                                                     (static_cast<btScalar>(-1.0)) * two_shot_contact_normal.getY(),
                                                     (static_cast<btScalar>(-1.0)) * two_shot_contact_normal.getZ();
        two_shot_result.contact_point_on_robot = contact_point_on_robot;
        two_shot_result.collider = collider;

        // Calculating the gradient is a bit involved. See write-up for derivation
        // Start with the derivative of epsilon w.r.t. r where r = x^T Sigma^-1 x (the "radius" of the ellipsoid)
        d_epsilon_d_r = (-1.0)*boost::math::pdf(chi_squared_dist_3dof, quantile);
        // Get contact point in confidence interval ellipsoid frame, then convert back to obstacle frame
        x = cutoff_confidence_interval_ellipsoid->localGetSupportingVertex(
            -1.0 * shadow_transform.invXform(
                epsilon_shadow_object_rotation.invXform(two_shot_contact_normal)));
        x = (shadow_transform * x);
        x_eigen << x.getX(), x.getY(), x.getZ();
        // Now derivative of epsilon w.r.t. x
        two_shot_result.d_epsilon_d_x = d_epsilon_d_r * 2.0 * x_eigen.transpose() * sigma_inv;

        // Now we can compute the estimated upper bound on collision probability
        // This upper bound will be the average of the large and small epsilons found above,
        // since the final shadow is the union of the large and small half-ellipsoids.
        CombinedBoundWithGradient combined_result;
        combined_result.epsilon = 0.5*(epsilon_large + epsilon_small);
        combined_result.one_shot_result = one_shot_result;
        combined_result.two_shot_result = two_shot_result;

        // Clean up our memory usage.
        delete cutoff_confidence_interval_ellipsoid;
        delete epsilon_shadow_collision_obj;

        return combined_result;
    }

    /**
     * Return the probability of collision between an uncertain object and all other objects
     * in the collision world.
     *
     * Given an object with known orientation but uncertain 3D position (all other objects
     * are assumed to have known orientation and position), returns an upper bound epsilon
     * on the probability of collision between the two objects. Formally:
     *
     *      P(obj_A collides with any object in collision world) <= epsilon
     *
     * This bound is computed using the concept of an epsilon-shadow. An epsilon shadow of
     * some uncertain object can be thought of as an enlarged version of that object such
     * that the probability that the object is contained entirely within the shadow is
     * 1-epsilon. It follows that the probability of the object protruding beyond its
     * epsilon-shadow is epsilon. Thus, the probability that another object does collide
     * with the uncertain object can be no greater than epsilon if that object does not
     * intersect the epsilon-shadow. We use bisection line search to find a maximal epsilon
     * such that the epsilon-shadow of obj_A does not intersect with any other object in the
     * scene, which places an upper bound on the probability of collision between the two
     * objects.
     *
     * @param obj_A: a btCollisionObject with a btConvexShape collision shape.
     * @param sigma: an Eigen::Matrix3d (3x3 matrix of floats) representing the covariance
     *               matrix for the position of obj_A. Note that the position of obj_A is
     *               assumed to be drawn from a multivariate Gaussian distribution with
     *               a mean equal to the location of obj_A. Also note that sigma must be
     *               symmetric and positive semidefinite. Additionally, for numerical reasons,
     *               sigma must be non-singular.
     * @param epsilon_tolerance: a float indicating the tolerance for error in the estimate
     *                           of the maximal epsilon.
     * @param collision_world: a btCollisionWorld object. Collision will be checked between
     *                         obj_A and all objects in collision_world. Note that obj_A
     *                         should *not* be included in collision_world.
     * @param filter_group: an optional integer masking the allowed filter groups.
     *
     * @return epsilon: the estimated upper bound on the probability of collision
     *                  between obj_A and all objects in the collision world.
     *
     */
    inline float compute_collision_probability_bound(btCollisionObject *obj_A,
                                                     const Eigen::Matrix3d sigma,
                                                     float epsilon_tolerance,
                                                     btCollisionWorld *collision_world,
                                                     int filter_group = -1)
    {
        CombinedBoundWithGradient result = compute_collision_probability_bound_and_gradient(
            obj_A,
            sigma,
            epsilon_tolerance,
            collision_world,
            filter_group);
        return result.epsilon;
    }
}
