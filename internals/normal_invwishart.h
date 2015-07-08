#ifndef NORMAL_INVWISHART_H
#define NORMAL_INVWISHART_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <boost/math/special_functions/digamma.hpp>
#include "np_types.h"
#include "eigen_types.h"


namespace niw {
    using namespace std;
    using namespace boost;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

    static const constexpr double log2   = 0.69314718055994530941723212145817656;
    static const constexpr double log2pi = 1.83787706640934533908193770912475883;

    template <typename Type>
    void log_likelihood(int N, int D, Type* obs, Type* mu, Type* sigma, Type* lliks) {
        NPArray<Type> e_obs = NPArray<Type>(obs, N, D);
        NPArray<Type> e_lliks = NPArray<Type>(lliks, N, 1);

        NPVector<Type> e_mu = NPVector<Type>(mu, D);
        NPMatrix<Type> e_sigma = NPMatrix<Type>(sigma, D, D);

        MatrixXt<Type> sigma_inv = e_sigma.inverse().eval();
        Type log_sigma_det = log(e_sigma.determinant());

        // -0.5*(D*log2pi - log_sigma_det) doesn't seem to work
        Type base = -0.5*D*log2pi - 0.5*log_sigma_det;
        auto diff = VectorXt<Type>(D);
        Type descriptive_stat;
        for (int i = 0; i < N; ++i) {
            diff = (e_obs.row(i) - e_mu.transpose().array()).matrix();
            descriptive_stat = diff.transpose()*sigma_inv*diff;
            e_lliks(i) = base - 0.5*descriptive_stat;
        }
    }

    template <typename Type>
    Type _log_lambda_tilde(const MatrixXt<Type>& sigma_0_inv, const Type nu_0) {
        Type D = sigma_0_inv.rows();
        Type log_sigma_0_det = log(sigma_0_inv.determinant());

        Type sum = 0.0;
        for (int i = 0; i < D; ++i) sum += math::digamma((nu_0 + 1 - i)/2);

        return sum + D*log2 + log_sigma_0_det;
    }

    /*
     * Calculates the expected log likelihood (or responsibilities) for 
     * variational normal-inverse-Wishart distribution.
     *
     * N - Number of observations.
     * D - Dimension of data.
     * obs - The observations.
     *
     * The follow are hyperparameters for a normal-inverse-Wishart distribution.
     * mu_0_    - The mean hyperparameter.
     * sigma_0_ - The covariance hyperparameter.
     * kappa_0_ - psuedo counts for the mean.
     * nu_0_    - psuedo counts for the covariance.
     *
     * rs - This holds the responsibilities.
     *
     * Notes: See Bishop chapter 10.2.
     */
    template <typename Type>
    void expected_log_likelihood(int N, int D, Type* obs, Type* mu_0_, 
                                 Type* sigma_0_, Type kappa_0_, Type nu_0_,
                                 Type* rs) {
        NPArray<Type> e_obs = NPArray<Type>(obs, N, D);
        NPArray<Type> e_rs  = NPArray<Type>(rs, N, 1);

        NPVector<Type> mu_0 = NPVector<Type>(mu_0_, D);
        NPMatrix<Type> sigma_0 = NPMatrix<Type>(sigma_0_, D, D);
        Type& kappa_0 = kappa_0_;
        Type& nu_0 = nu_0_;

        MatrixXt<Type> sigma_0_inv = sigma_0.inverse().eval();
        Type log_lambda_tilde = _log_lambda_tilde(sigma_0_inv, nu_0);

        Type base = 0.5*(log_lambda_tilde - D*(1/kappa_0) - D*log2pi);
        auto diff = VectorXt<Type>(D);
        Type descriptive_stat;
        for (int i = 0; i < N; ++i) {
            diff = (e_obs.row(i) - mu_0.transpose().array()).matrix();
            descriptive_stat = diff.transpose()*sigma_0_inv*diff;
            e_rs(i) = base - 0.5*nu_0*descriptive_stat;
        }
    }

    template <typename Type>
    void sufficient_statistics(int N, int D, Type* obs, Type* expected_states,
                               Type* s1_, Type* s2_, Type* s3_) {
        NPArray<Type> e_obs = NPArray<Type>(obs, N, D);
        NPArray<Type> e_es  = NPArray<Type>(expected_states, N, 1);

        Type& s1 = s1_;
        NPArray<Type>  s2 = NPArray<Type>(s2_, D, 1);
        NPMatrix<Type> s3 = NPMatrix<Type>(s3_, D, D);
    }


    /*
     * Computes a variational meanfield update in natural parameter form.
     *
     * D - Dimension of data.
     *
     * The n's are the hyperparameters for Gaussian using a normal-inverse-
     * Wishart distribution with parameters mu_0, sigma_0, kappa_0 and nu_0.
     * These are updated inplace.
     *
     * n1 - kappa_0*mu_0
     * n2 - kappa_0
     * n3 - sigma_0 + kappa_0*mu_0*mu_0^T
     * n4 - nu_0 + 2 + p
     *
     * Weighted Sufficient Statistics.
     *
     * s1 - \sum_{i=1}^N w_i
     * s2 - \sum_{i=1}^N w_i*x_i
     * s3 - \sum_{i=1}^N w_i(x_i*x_i^T)
     */
    template  <typename Type>
    void meanfield_update(int D, Type* nat_params, Type s1_, Type* s2_, Type* s3_) {
        NPMatrix<Type> n3 = NPMatrix<Type>(nat_params, D, D);
        NPVector<Type> n1 = NPVector<Type>(&nat_params[D*D], D, 1);
        Type& n2 = nat_params[D*(D+1)];
        Type& n4 = nat_params[D*(D+2)];

        Type& s1 = s1_;
        NPMatrix<Type> s2 = NPMatrix<Type>(s2_, D, 1);
        NPVector<Type> s3 = NPVector<Type>(s3_, D, D);
        
        n1 += s2;
        n2 += s1;
        n3 += s3;
        n4 += s1;
    }
}

#endif
