/*
*  This file implements uncertainty-aware collision checking using the BulletCollision library
*
*  Author: Charles B. Dawson (cbd@mit.edu)
*  Written: Nov 15, 2019
*/
#pragma once

#include "btBulletCollisionCommon.h"

// linear algebra library
#include <Eigen/Dense>


namespace ProximityAlert
{
    // Custom class for an sphere that has been cut off by a plane
    struct CutoffSphere : public btMultiSphereShape
    {

        // This is the normal vector to the plane that cuts the sphere
        // in two. Assumed to pass through the origin of the sphere,
        // and point into the interior of the shape.
        btVector3 m_normal;

        // Construct a new cutoff sphere with the given radius
        // and cutoff plane (passing through the center of the
        // sphere normal to the given vector, which points towards
        // the interior of the shape)
        CutoffSphere(const btVector3* origin,
                     const btScalar* radius,
                     const btVector3& planeNormal) :
            btMultiSphereShape(origin,
                               radius,
                               1)
        {
            m_normal = planeNormal;
            // make sure to normalize!
            btScalar recip_length = btScalar(1.0) / planeNormal.length();
            m_normal *= recip_length;
        }

        // Return the supporting vertex in the direction specified by
        // vec0.
        btVector3 localGetSupportingVertexWithoutMargin(const btVector3& vec0) const {
            // Our strategy for this is as follows.
            // First, we check if the query vector vec0 points into the cutoff region.
            // If it does *not*, then we just return the supporting vector of the
            // underlying sphere.
            //
            // If it does, then we calculate the vector that lies in the cutoff plane
            // and is also coplanar with the query vector and return the support vector
            // of the underlying sphere in that direction. This essentially forces the
            // query vector to lie in the cutoff plane.
            float vec0_normal_scale = m_normal.dot(vec0);
            if (vec0_normal_scale >= 0) {
                // if the query vector is *not* pointing into the cutoff region
                // then we can act as if nothing special is going on here
                return btMultiSphereShape::localGetSupportingVertexWithoutMargin(vec0);
            } else {
                // Otherwise, we need to force the query vector down into the cutoff plane
                
                // We do this by first calculating the component of the query vector parallel
                // to the plane normal, then subtracting that from the query vector to
                // get the in-plane component
                btVector3 vec0_normal = m_normal * vec0_normal_scale;
                btVector3 vec0_planar = vec0 - vec0_normal;
                return btMultiSphereShape::localGetSupportingVertexWithoutMargin(vec0_planar);
            }
        }
    };
}
