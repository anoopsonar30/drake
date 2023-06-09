/*
*  This file implements sampling from a multivariate normal distribution
*
*  Author: Charles B. Dawson (cbd@mit.edu)
*  Code based on that available at https://forum.kde.org/viewtopic.php?f=74&t=95260
* 
*  Written: Nov 15, 2019
*/
#pragma once
#include <random>
#include <Eigen/Dense>

#include <iostream>

namespace ProximityAlert
{

    /*
    *  Generate one sample from a multivariate normal distribution with specified mean and covariance.
    *
    *  @param mean: an Eigen::VectorXd column vector specifying the mean of the distribution
    *  @param covariance: an Eigen::MatrixXd specifying the covariance matrix of the distribution.
    *                     This matrix must be positive semidefinite and symmetric.
    *
    *  @return sample: an Eigen::VectorXd sampled from the specified normal distribution
    *
    */
    inline Eigen::VectorXd sample_from_multivariate_normal(const Eigen::VectorXd mean,
                                                           const Eigen::MatrixXd covariance)
    {
        // Our general strategy is to generate a vector sampled from a zero-mean,
        // identity-covariance distribution, then transform it to align with the
        // specified mean and covariance.

        // Start by extracting the dimension we have
        int n_dim = mean.size();

        // Create normal distribution sample generator
        // zero mean, standard deviation 1
        auto normal_dist_sampler = std::bind(std::normal_distribution<double>{0.0, 1.0},
                                             std::mt19937(std::random_device{}()));

        // Create random vector by sampling from that distribution
        Eigen::VectorXd sample(n_dim);
        for (int i = 0; i < n_dim; i++) {
            sample(i) = normal_dist_sampler();
        }

        // Compute the eigenvalues and eigenvectors of the covariance matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covariance_eigensolver(covariance);

        // Now we take the random vector, scale by the square root of the eigenvalues,
        // transform by the eigenvectors, then shift by the mean to get our desired sample
        sample = mean + covariance_eigensolver.eigenvectors() *
                            covariance_eigensolver.eigenvalues().cwiseSqrt().asDiagonal() *
                            sample;
        return sample;
    }

} // namespace ProximityAlert

