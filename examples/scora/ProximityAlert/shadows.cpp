///-----includes_start-----
#include "btBulletCollisionCommon.h"
#include <stdio.h>

// linear algebra library
#include <Eigen/Dense>

#include "proximityAlert.h"
#include "proximityAlertUtils.h"

#define PI 3.141592
#define INV_SQRT2 0.707106781186548
#define SQRT2 1.4142135623730950

#define ROBOT_FILTER_GROUP 1
#define OBSTACLE_FILTER_GROUP 2

struct PairwiseContactResultCallback : public btCollisionWorld::ContactResultCallback
{
    // flag will be set to true if a collision is detected
    bool bCollision;
    float x;

    PairwiseContactResultCallback(float x_init) : 
        bCollision(false),
        x(x_init)
    {}

    // This method is called when a collision is possibly detected (could be false positive, so we have to check)
    btScalar addSingleResult(btManifoldPoint& pt,
                             const btCollisionObjectWrapper* colObj0Wrap,
                             int partId0,
                             int index0,
                             const btCollisionObjectWrapper* colObj1Wrap,
                             int partId1,
                             int index1)
    {
        // Collision will have occured if the distance is zero or negative
        double ptdist = pt.getDistance();

        if (ptdist <= 0.0) {
            bCollision = true;
        }
    }
};

int main(int argc, char** argv)
{
    // Initiallize the required bullet overhead
    btCollisionConfiguration* bt_collision_configuration;
    btCollisionDispatcher* bt_dispatcher;
    btBroadphaseInterface* bt_broadphase;
    btCollisionWorld* bt_collision_world;

    double scene_size = 10;
    unsigned int max_objects = 100;

    bt_collision_configuration = new btDefaultCollisionConfiguration();
    bt_dispatcher = new btCollisionDispatcher(bt_collision_configuration);

    btScalar sscene_size = (btScalar) scene_size;
    btVector3 worldAabbMin(-sscene_size, -sscene_size, -sscene_size);
    btVector3 worldAabbMax(sscene_size, sscene_size, sscene_size);
    //This is one type of broadphase, bullet has others that might be faster depending on the application
    bt_broadphase = new bt32BitAxisSweep3(worldAabbMin, worldAabbMax, max_objects, 0, false); 

    bt_collision_world = new btCollisionWorld(bt_dispatcher, bt_broadphase, bt_collision_configuration);
    

    // Start my code. Overhead is done

    //Create objects for the arm link
    btCollisionObject* arm_link_1 = new btCollisionObject();
    btCollisionObject* arm_link_2 = new btCollisionObject();
    btCollisionObject* end_effector = new btCollisionObject();

    // Create objects for the obstacles
    btCollisionObject* obstacle_1 = new btCollisionObject();
    btCollisionObject* obstacle_2 = new btCollisionObject();
    btCollisionObject* obstacle_3 = new btCollisionObject();
    
    // Create boxes for the arm links
    btBoxShape * arm_link_1_shape = new btBoxShape(btVector3((btScalar) 1.0, (btScalar) 0.1, (btScalar) 0.1));
    btBoxShape * arm_link_2_shape = new btBoxShape(btVector3((btScalar) 0.7, (btScalar) 0.1, (btScalar) 0.1));
    btBoxShape * end_effector_shape = new btBoxShape(btVector3((btScalar) 0.1, (btScalar) 0.2, (btScalar) 0.2));
    //Set the shape of each arm link
    arm_link_1->setCollisionShape(arm_link_1_shape);
    arm_link_2->setCollisionShape(arm_link_2_shape);
    end_effector->setCollisionShape(end_effector_shape);
    
    // Move each to a specific location to make the shape of an arm
    arm_link_1->getWorldTransform().setOrigin(btVector3((btScalar) SQRT2/2.0, (btScalar) 0, (btScalar) 0));
    arm_link_2->getWorldTransform().setOrigin(btVector3((btScalar) SQRT2*1.5, (btScalar) 0, (btScalar) -0.1));
    end_effector->getWorldTransform().setOrigin(btVector3((btScalar) SQRT2*1.5+0.85*INV_SQRT2, (btScalar) 0, (btScalar) -0.25+0.7*INV_SQRT2));
    // rotate a bit about y axis
    btVector3 yaxis = btVector3(0.0, 1.0, 0.0);
    btQuaternion arm_link_1_rotation(yaxis, PI/6.0);
    arm_link_1->getWorldTransform().setRotation(arm_link_1_rotation);
    btQuaternion arm_link_2_rotation(yaxis, -PI/6.0);
    arm_link_2->getWorldTransform().setRotation(arm_link_2_rotation);
    btQuaternion end_effector_rotation(yaxis, -PI/6.0);
    end_effector->getWorldTransform().setRotation(end_effector_rotation);
    
    // Add the arm to the scene
    bt_collision_world->addCollisionObject(arm_link_1, ROBOT_FILTER_GROUP);
    bt_collision_world->addCollisionObject(arm_link_2, ROBOT_FILTER_GROUP);
    bt_collision_world->addCollisionObject(end_effector, ROBOT_FILTER_GROUP);

    // Create collision shapes
    btCylinderShape * obstacle_shape = new btCylinderShape(btVector3(0.25, 0.25, 0.5));
    btBoxShape * obstacle_box_shape = new btBoxShape(btVector3(0.25, 0.25, 0.5));
    obstacle_1->setCollisionShape(obstacle_shape);
    obstacle_2->setCollisionShape(obstacle_shape);
    obstacle_3->setCollisionShape(obstacle_box_shape);

    // Position obstacles
    obstacle_1->getWorldTransform().setOrigin(btVector3((btScalar) 1, (btScalar) 0, (btScalar) 0.5));
    obstacle_2->getWorldTransform().setOrigin(btVector3((btScalar) 2, (btScalar) 0, (btScalar) -1));
    obstacle_3->getWorldTransform().setOrigin(btVector3((btScalar) 3.25, (btScalar) 0, (btScalar) 0.75));
    obstacle_3->getWorldTransform().setRotation(end_effector_rotation);

    // Add obstacles to the scene (marked as obstacles)
    bt_collision_world->addCollisionObject(obstacle_1, OBSTACLE_FILTER_GROUP);
    bt_collision_world->addCollisionObject(obstacle_2, OBSTACLE_FILTER_GROUP);
    bt_collision_world->addCollisionObject(obstacle_3, OBSTACLE_FILTER_GROUP);

    // Create the covariances and means for these obstacles
    Eigen::Matrix3d sigma_1;
    sigma_1 << 0.01, 0.0, 0.0,
               0.0, 0.01, 0.0,
               0.0, 0.0, 0.01;
    Eigen::Vector3d mean_1;
    mean_1 << 1.0, 0.0, 0.5;
    
    Eigen::Matrix3d sigma_2;
    sigma_2 << 0.05, 0.07, 0.0,
               0.07, 0.1, 0.0,
               0.0, 0.0, 0.01;
    Eigen::Vector3d mean_2;
    mean_2 << 2.0, 0.0, -1.0;

    Eigen::Matrix3d sigma_3;
    sigma_3 << 0.001, 0.0, 0.0,
               0.0, 0.001, 0.0,
               0.0, 0.0, 0.05;
    Eigen::Vector3d mean_3;
    mean_3 << 3.25, 0.0, 0.75;

    // First we get the estimated collision risk using the shadows method

    float epsilon_tolerance = 0.000001; // tolerance 10^-6

    // float epsilon1_estimated;
    // float epsilon2_estimated;
    // float epsilon3_estimated;
    // // Loop in case we want to try multiple queries for benchmarking
    // for (int i = 1; i <= 1; i++) {
    //     epsilon1_estimated = ProximityAlert::compute_collision_probability_bound(obstacle_1, sigma_1, epsilon_tolerance, bt_collision_world, ROBOT_FILTER_GROUP);
    //     epsilon2_estimated = ProximityAlert::compute_collision_probability_bound(obstacle_2, sigma_2, epsilon_tolerance, bt_collision_world, ROBOT_FILTER_GROUP);
    //     epsilon3_estimated = ProximityAlert::compute_collision_probability_bound(obstacle_3, sigma_3, epsilon_tolerance, bt_collision_world, ROBOT_FILTER_GROUP);
    // }
    // printf("Epsilon tolerance: %f\n", epsilon_tolerance);
    // printf("Estimated upper bound on individual collision probabilities:\n\teps1: %.9f\n\teps2: %.9f\n\teps3: %.9f\n", epsilon1_estimated, epsilon2_estimated, epsilon3_estimated);
    // printf("Estimated overall upper bound on collision probability:\n\teps: %.6f\n", epsilon1_estimated + epsilon2_estimated + epsilon3_estimated);
    // // printf("%.6f\n", epsilon1_estimated + epsilon2_estimated + epsilon3_estimated);

    int num_collisions;
    int N = 10000;
    for (int trial = 0; trial < 100; trial++){
        num_collisions = 0;
        for (int i = 0; i < N; i++) {
            // Get random location for each obstacle:
            Eigen::VectorXd x1 = ProximityAlert::sample_from_multivariate_normal(mean_1, sigma_1);
            Eigen::VectorXd x2 = ProximityAlert::sample_from_multivariate_normal(mean_2, sigma_2);
            Eigen::VectorXd x3 = ProximityAlert::sample_from_multivariate_normal(mean_3, sigma_3);
            obstacle_1->getWorldTransform().setOrigin(btVector3((btScalar) x1(0), (btScalar) x1(1), (btScalar) x1(2)));
            obstacle_2->getWorldTransform().setOrigin(btVector3((btScalar) x2(0), (btScalar) x2(1), (btScalar) x2(2)));
            obstacle_3->getWorldTransform().setOrigin(btVector3((btScalar) x3(0), (btScalar) x3(1), (btScalar) x3(2)));

            // instantiate pairwise collision callback
            PairwiseContactResultCallback* collision_callback = new PairwiseContactResultCallback(0.0);
            // perform the checks
            bt_collision_world->contactTest(obstacle_1, *collision_callback);
            bt_collision_world->contactTest(obstacle_2, *collision_callback);
            bt_collision_world->contactTest(obstacle_3, *collision_callback);

            // Check if we got a collision
            if (collision_callback->bCollision) {
                num_collisions++;
            }

            // Clean up the callback
            delete collision_callback;
        }
        printf("Simulated collision probability (%d samples): %.9f\n", N, (float)num_collisions/N);
    }
    // printf("Simulated collision probability (%d samples): %.9f\n", N, (float)num_collisions/N);
    // printf("Colliding samples: %d\n", num_collisions);
}
