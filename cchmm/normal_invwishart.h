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
    struct mo_params {
        int D;
        T* sigma_N;
        T* mu_N;
        T* kappa_N;
        T* nu_N;
    };

    template <typename T>
    struct e_suff_stats {
        T s1;
        ArrayXt<T> s2;
        MatrixXt<T> s3;
    };

    template <typename T>
    mo_params<T> convert_to_struct(T* mo_params_raw, int D, int s) {
        int offset = (D*D + 3*D);
        return mo_params<T>{D,
                            &mo_params_raw[s*offset],
                            &mo_params_raw[s*offset + D*D],
                            &mo_params_raw[s*offset + D*D + D],
                            &mo_params_raw[s*offset + D*D + 2*D]};
    }

    template <typename Type>
    Type _log_lambda_tilde(const MatrixXt<Type>& sigma_inv, const Type nu) {
        Type D = sigma_inv.rows();
        Type log_sigma_det = log(sigma_inv.determinant());

        Type sum = 0.0;
        for (int i = 1; i < D + 1; ++i) sum += math::digamma((nu + 1 - i)/2);

        return sum + D*log2 + log_sigma_det;
    }

    template <typename Type>
    ArrayXt<Type> expected_log_likelihood(const ArrayXt<Type>& obs,
                                          const mo_params<Type>& params) {
        ArrayXt<Type> rs = ArrayXt<Type>::Zero(obs.rows(), 1);

        int D = params.D;
        NPMatrix<Type> sigma_N(params.sigma_N, D, D);
        NPVector<Type> mu_N(params.mu_N, D, 1);
        Type& kappa_N = *(params.kappa_N);
        Type& nu_N = *(params.nu_N);

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

    template <typename Type>
    e_suff_stats<Type> expected_sufficient_statistics(const ArrayXt<Type>& obs,
                                                      const ArrayXt<Type>& es) {
        Type s1           = 0.0;
        ArrayXt<Type> s2  = ArrayXt<Type>::Zero(obs.cols(), 1);
        MatrixXt<Type> s3 = MatrixXt<Type>::Zero(obs.cols(), obs.cols());

        for (int i = 0; i < obs.rows(); ++i) {
            s1 += es.coeff(i);
            s2 += obs.row(i)*es.coeff(i);
            s3 += obs.row(i).matrix().transpose()*obs.row(i).matrix()*es.coeff(i);
        }

        e_suff_stats<Type> ess = {s1, s2, s3};
        return ess;
    }
}


#endif
