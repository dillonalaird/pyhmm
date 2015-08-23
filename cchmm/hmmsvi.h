#ifndef HMMSVI_h
#define HMMSVI_h

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/math/special_functions/digamma.hpp>
#include "np_types.h"
#include "eigen_types.h"
#include "metaobs.h"
#include "forward_backward_msgs.h"
#include "normal_invwishart.h"
#include "dirichlet.h"


namespace hmmsvi {
    using namespace boost;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

    static const constexpr double eps = 0.000000001;

    template <typename Type>
    VectorXt<Type> _calc_pi(const NPMatrix<Type>& A_nat_N) {
        // can't use NPMatrix<Type> as type
        EigenSolver<MatrixXt<Type> > es(A_nat_N);
        VectorXcd evals = es.eigenvalues();

        int argmax = 0;
        for (int i = 0; i < evals.size(); ++i)
            if (evals(i).real() > evals(argmax).real()) argmax = i;

        return es.eigenvectors().col(argmax).real().cwiseAbs();
    }

    template <typename Type>
    void global_update(int T, int L, Type lrate,
                       const MatrixXt<Type>& A_0,
                       const MatrixXt<Type>& A_N,
                       const MatrixXt<Type>& A_inter,
                       const std::vector<niw::e_suff_stats<Type> >& emits_inter) {
    }

    template <typename Type>
    ArrayXt<Type> local_update(const ArrayXt<Type>& obs,
                               const VectorXt<Type>& pi,
                               const NPMatrix<Type>& A_nat_N,
                               const std::vector<niw::map_mo_params<Type> >& emits_N) {
        MatrixXt<Type> A = A_nat_N + MatrixXt<Type>::Ones(A_nat_N.rows(), A_nat_N.cols());
        MatrixXt<Type> A_mod = dir::expected_sufficient_statistics<Type>(A);
        A_mod = A_mod.array().exp().matrix();

        VectorXt<Type> pi_mod = VectorXt<Type>::Zero(pi.size());
        Type sum = math::digamma(pi_mod.sum() + eps);
        for (int i = 0; i < pi.size(); ++i)
            pi_mod(i) = math::digamma(pi(i) + eps) - sum;
        pi_mod = pi_mod.array().exp().matrix();

        ArrayXt<Type> elliks = ArrayXt<Type>::Zero(obs.rows(), emits_N.size());
        for (int i = 0; i < emits_N.size(); ++i)
            elliks.block(0, i, elliks.rows(), 1) = niw::expected_log_likelihood(obs, emits_N[i]);

        ArrayXt<Type> lalpha = fb::forward_msgs(elliks, pi_mod, A_mod);
        ArrayXt<Type> lbeta  = fb::backward_msgs(elliks, A_mod);

        ArrayXt<Type> es = fb::expected_states(lalpha, lbeta);
        es = es.exp();
        ArrayXt<Type> row_sums = es.rowwise().sum();
        for (int i = 0; i < es.rows(); ++i)
            es.row(i) = es.row(i)/row_sums.row(i).coeff(0);

        return es;
    }

    template <typename Type>
    void infer(int D, int S, int T, Type* obs, Type* A_0, Type* A_N,
               Type* emits_0, Type* emits_N, Type tau, Type kappa, int L, int n,
               int itr) {
        // pass in init and nat params?
        NPArray<Type> e_obs(obs, T, S);
        NPMatrix<Type> A_nat_0(A_0, S, S);
        A_nat_0 -= MatrixXt<Type>::Ones(S, S);
        NPMatrix<Type> A_nat_N(A_N, S, S);
        A_nat_N -= MatrixXt<Type>::Ones(S, S);

        std::vector<niw::map_mo_params<Type> > emits_mo_0;
        for (int s = 0; s < S; ++s)
            emits_mo_0.push_back(niw::convert_to_struct(emits_0, D, s));
        std::vector<niw::map_mo_params<Type> > emits_mo_N;
        for (int s = 0; s < S; ++s)
            emits_mo_N.push_back(niw::convert_to_struct(emits_N, D, s));

        Type lrate = 0.0;
        for (int it = 0; it < itr; ++it) {
            lrate = pow(it + tau, -1*kappa);

            MatrixXt<Type> A_inter = MatrixXt<Type>::Zero(S, S);
            std::vector<niw::e_suff_stats<Type> > emits_inter;
            for (int s = 0; s < S; ++s)
                emits_inter.push_back(niw::create_zero_ss<Type>(D));

            for (int mit = 0; mit < n; ++mit) {
                mo::metaobs m = mo::metaobs_unif(T, L);
                VectorXt<Type> pi = _calc_pi<Type>(A_nat_N);

                ArrayXt<Type> obs_sub = e_obs.block(m.i1, 0, m.i2, 2);
                ArrayXt<Type> var_x = local_update(obs_sub, pi, A_nat_N, emits_mo_N);

                // intermediate parameters
                MatrixXt<Type> A_i = dir::sufficient_statistics(var_x);
                A_i -= MatrixXt<Type>::Ones(S, S);
                A_inter += A_i;

                for (int s = 0; s < S; ++s) {
                    niw::e_suff_stats<Type> emit_i =  \
                        niw::expected_sufficient_statistics(obs_sub, var_x);
                    emits_inter[s].s1 += emit_i.s1;
                    emits_inter[s].s2 += emit_i.s2;
                    emits_inter[s].s3 += emit_i.s3;
                }
            }

            // global update
            int B = 2*L + 1;
            Type A_bfactor = (T - 2*L - 1)/(2*L*B);
            A_nat_N = dir::meanfield_sgd_update(lrate, A_bfactor, A_nat_0,
                                                A_nat_N, A_inter);

            Type e_bfactor = (T - 2*L - 1)/((2*L + 1)*B);
            for (int s = 0; s < S; ++S) {
                niw::meanfield_sgd_update(lrate, e_bfactor, emits_mo_0[s],
                                          emits_mo_N[s], emits_inter[s]);
            }
        }
    }
}


#endif
