#ifndef NORMAL_INVWISHART_H
#define NORMAL_INVWISHART_H

#include <Eigen/Core>
#include <boost/math/special_functions/digamma.hpp>
#include "np_types.h"
#include "eigen_types.h"


namespace niw {
    using namespace boost;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

    static const constexpr double log2   = 0.69314718055994530941723212145817656;
    static const constexpr double log2pi = 1.83787706640934533908193770912475883;

    template <typename T>
    struct map_mo_params {
        int D;
        T* sigma;
        T* mu;
        T* kappa;
        T* nu;
    };

    template <typename T>
    struct mo_params {
        MatrixXt<T> sigma;
        VectorXt<T> mu;
        T kappa;
        T nu;
    };

    template <typename T>
    struct nat_params {
        VectorXt<T> n1;
        T n2;
        MatrixXt<T> n3;
        T n4;
    };

    template <typename T>
    struct e_suff_stats {
        T s1;
        ArrayXt<T> s2;
        MatrixXt<T> s3;
    };

    template <typename T>
    e_suff_stats<T> create_zero_ss(int D) {
        T s1 = 0.0;
        ArrayXt<T> s2 = ArrayXt<T>::Zero(D, 1);
        MatrixXt<T> s3 = MatrixXt<T>::Zero(D, D);

        e_suff_stats<T> ess = {s1, s2, s3};
        return ess;
    }

    template <typename T>
    map_mo_params<T> convert_to_struct(T* map_mo_params_raw, int D, int s) {
        int offset = (D*D + 3*D);
        return map_mo_params<T>{D,
                                &map_mo_params_raw[s*offset],
                                &map_mo_params_raw[s*offset + D*D],
                                &map_mo_params_raw[s*offset + D*D + D],
                                &map_mo_params_raw[s*offset + D*D + 2*D]};
    }

    template <typename T>
    nat_params<T> convert_mo_to_nat(const map_mo_params<T>& params) {
        int D = params.D;
        NPMatrix<T> sigma(params.sigma, D, D);
        NPVector<T> mu(params.mu, D, 1);
        T& kappa = *(params.kappa);
        T& nu = *(params.nu);

        VectorXt<T> n1 = kappa*mu;
        T n2 = kappa;
        MatrixXt<T> n3 = sigma + kappa*(mu*mu.transpose());
        T n4 = nu + 2 + mu.size();

        return nat_params<T>{n1, n2, n3, n4};
    }

    template <typename T>
    mo_params<T> convert_nat_to_mo(const nat_params<T>& params) {
        T kappa = params.n2;
        VectorXt<T> mu = params.n1/params.n2;
        MatrixXt<T> sigma = params.n3 - kappa*(mu*mu.transpose());
        T nu = params.n4 - 2 - mu.size();
        return mo_params<T>{sigma, mu, kappa, nu};
    }

    template <typename Type>
    Type _log_lambda_tilde(const MatrixXt<Type>& sigma_inv, const Type nu) {
        Type D = sigma_inv.rows();
        Type log_sigma_det = log(sigma_inv.determinant());

        Type sum = 0.0;
        for (int i = 1; i < D + 1; ++i) sum += math::digamma((nu + 1 - i)/2);

        return sum + D*log2 + log_sigma_det;
    }

    /*
     * Calculates the expected log likelihood (or responsibilities) for 
     * variational normal-inverse-Wishart distribution.
     *
     * obs    - The observations.
     * params - The moment parameters.
     *
     * Notes: See Bishop chapter 10.2.
     */
    template <typename Type>
    ArrayXt<Type> expected_log_likelihood(const ArrayXt<Type>& obs,
                                          const map_mo_params<Type>& params) {
        ArrayXt<Type> rs = ArrayXt<Type>::Zero(obs.rows(), 1);

        int D = params.D;
        NPMatrix<Type> sigma_N(params.sigma, D, D);
        NPVector<Type> mu_N(params.mu, D, 1);
        Type& kappa_N = *(params.kappa);
        Type& nu_N = *(params.nu);

        MatrixXt<Type> sigma_inv = sigma_N.inverse().eval();
        Type log_lambda_tilde = _log_lambda_tilde<Type>(sigma_inv, nu_N);

        Type base = 0.5*(log_lambda_tilde - D*(1/kappa_N) - D*log2pi);
        auto diff = VectorXt<Type>(D);
        Type descriptive_stat;
        for (int i = 0; i < obs.rows(); ++i) {
            diff = (obs.row(i) - mu_N.transpose().array()).matrix();
            descriptive_stat = diff.transpose()*sigma_inv*diff;
            rs(i) = base - 0.5*nu_N*descriptive_stat;
        }

        return rs;
    }

    /*
     * Calculates the sufficient statistics for a normal-inverse-Wishart
     * distribution.
     *
     * obs - The observations.
     * es  - An N x 1 array representing the expected value for a given state.
     *
     * Weighted sufficient statistics.
     *
     * s1 - \sum_{i=1}^N w_i
     * s2 - \sum_{i=1}^N w_i*x_i
     * s3 - \sum_{i=1}^N w_i(x_i*x_i^T)
     */
    template <typename Type>
    e_suff_stats<Type> expected_sufficient_statistics(const ArrayXt<Type>& obs,
                                                      const ArrayXt<Type>& es,
                                                      int s) {
        Type s1           = 0.0;
        ArrayXt<Type>  s2 = ArrayXt<Type>::Zero(obs.cols(), 1);
        MatrixXt<Type> s3 = MatrixXt<Type>::Zero(obs.cols(), obs.cols());

        for (int i = 0; i < obs.rows(); ++i) {
            s1 += es.coeff(i, s);
            s2 += obs.row(i)*es.coeff(i, s);
            s3 += obs.row(i).matrix().transpose()*obs.row(i).matrix()*es.coeff(i, s);
        }

        e_suff_stats<Type> ess = {s1, s2, s3};
        return ess;
    }

    /*
     * Computes a variational meanfield update in natural parameter form
     * inplace.
     *
     * lrate     - The learning rate.
     * bfactor   - The batch factor.
     * emit_mo_0 - The prior moment parameters.
     * emit_mo_N - The moment parameters.
     * ess       - The expected sufficient statistics.
     *
     * Weighted sufficient statistics.
     *
     * s1 - \sum_{i=1}^N w_i
     * s2 - \sum_{i=1}^N w_i*x_i
     * s3 - \sum_{i=1}^N w_i(x_i*x_i^T)
     */
    template <typename Type>
    void meanfield_sgd_update(Type lrate, Type bfactor,
                              const map_mo_params<Type>& emit_mo_0,
                              const map_mo_params<Type>& emit_mo_N,
                              const e_suff_stats<Type>& ess) {
        nat_params<Type> emit_nat_0 = convert_mo_to_nat<Type>(emit_mo_0);
        nat_params<Type> emit_nat_N = convert_mo_to_nat<Type>(emit_mo_N);

        emit_nat_N.n1 = (1 - lrate)*emit_nat_N.n1 + \
                        lrate*(emit_nat_0.n1 + bfactor*ess.s2.matrix());
        emit_nat_N.n2 = (1 - lrate)*emit_nat_N.n2 + \
                        lrate*(emit_nat_0.n2 + bfactor*ess.s1);
        emit_nat_N.n3 = (1 - lrate)*emit_nat_N.n3 + \
                        lrate*(emit_nat_0.n3 + bfactor*ess.s3);
        emit_nat_N.n4 = (1 - lrate)*emit_nat_N.n4 + \
                        lrate*(emit_nat_0.n4 + bfactor*ess.s1);

        mo_params<Type> emit_mo_N_up = convert_nat_to_mo<Type>(emit_nat_N);
        // inplace updates
        int D = emit_mo_0.D;
        std::memcpy(emit_mo_N.sigma,  emit_mo_N_up.sigma.data(), sizeof(Type[D*D]));
        std::memcpy(emit_mo_N.mu,     emit_mo_N_up.mu.data(),    sizeof(Type[D]));
        std::memcpy(emit_mo_N.kappa, &emit_mo_N_up.kappa,        sizeof(Type));
        std::memcpy(emit_mo_N.nu,    &emit_mo_N_up.nu,           sizeof(Type));
    }
}


#endif
