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

    /*
     * Computers the stationary distribution of the mean of the distribution for
     * the current transition matrix.
     *
     * A_nat_N - The hyperparameters of the transition matrix in natural
     *     parameter form.
     */
    template <typename Type>
    VectorXt<Type> _calc_pi(const NPMatrix<Type>& A_nat_N) {
        // can't use NPMatrix<Type> as type
        MatrixXt<Type> A_mean = MatrixXt<Type>::Zero(A_nat_N.rows(), A_nat_N.cols());
        ArrayXt<Type> A_row_sums = A_nat_N.rowwise().sum();
        for (int i = 0; i < A_nat_N.rows(); ++i)
            for (int j = 0; j < A_nat_N.cols(); ++j)
                A_mean(i,j) = A_nat_N(i,j)/A_row_sums(i);

        EigenSolver<MatrixXt<Type> > es(A_mean);
        VectorXcd evals = es.eigenvalues();

        int argmax = 0;
        for (int i = 0; i < evals.size(); ++i)
            if (evals(i).real() > evals(argmax).real()) argmax = i;

        return es.eigenvectors().col(argmax).real().cwiseAbs();
    }

    /*
     * Cacluates the local update.
     *
     * obs     - The observations.
     * pi      - The initial distribution.
     * A_nat_N - The dirichlet hyperparameters of the transition matrix in
     *     natural parameter form.
     * emits_N - The emission hyperparameters in moment form.
     */
    template <typename Type>
    ArrayXt<Type> local_update(const ArrayXt<Type>& obs,
                               const VectorXt<Type>& pi,
                               const NPMatrix<Type>& A_nat_N,
                               const std::vector<niw::map_mo_params<Type> >& emits_N) {
        MatrixXt<Type> A = A_nat_N + MatrixXt<Type>::Ones(A_nat_N.rows(), A_nat_N.cols());
        MatrixXt<Type> A_mod = dir::expected_sufficient_statistics<Type>(A);
        A_mod = A_mod.array().exp().matrix();

        VectorXt<Type> pi_mod = VectorXt<Type>::Zero(pi.size());
        Type sum = math::digamma(pi.sum() + eps);
        for (int i = 0; i < pi.size(); ++i)
            pi_mod(i) = math::digamma(pi(i) + eps) - sum;

        pi_mod = pi_mod.array().exp().matrix();

        ArrayXt<Type> elliks = ArrayXt<Type>::Zero(obs.rows(), emits_N.size());
        for (int i = 0; i < emits_N.size(); ++i)
            elliks.block(0, i, elliks.rows(), 1) = niw::expected_log_likelihood(obs, emits_N[i]);

        ArrayXt<Type> lalpha = fb::forward_msgs(elliks, pi_mod, A_mod);
        ArrayXt<Type> lbeta  = fb::backward_msgs(elliks, A_mod);
        ArrayXt<Type> es     = fb::expected_states(lalpha, lbeta);

        es = es.exp();
        ArrayXt<Type> row_sums = es.rowwise().sum();
        for (int i = 0; i < es.rows(); ++i)
            es.row(i) = es.row(i)/row_sums.row(i).coeff(0);

        return es;
    }

    /*
     * Does stochastic variational inference on the given data.
     *
     * D       - The dimension of the observations.
     * S       - The number of states.
     * T       - The number of observations.
     * obs     - The observations (row major).
     * A_0     - The prior dirichlet parameters for the transition matrix.
     * A_N     - The dirichlet parameters for the transition matrix.
     * emits_0 - The prior hyperparameters for the emission distribution in
     *     moment form.
     * emits_N - The hyperparameters for the emission distribution in moment
     *     form.
     * tau     - Delay for learning rate, \tau >= 0.
     * kappa   - Forgetting factor for learning rate, \kappa \in (0.5, 1].
     * L       - Meta-observations will be of size 2*L + 1, L > 0.
     * n       - The number of meta-oberservations in a minibatch.
     * itr     - The number of iterations.
     */
    template <typename Type>
    void infer(int D, int S, int T, Type* obs, Type* A_0, Type* A_N,
               Type* emits_0, Type* emits_N, Type tau, Type kappa, int L, int n,
               int itr) {
        NPArray<Type> e_obs(obs, T, S);
        NPMatrix<Type> A_nat_0(A_0, S, S);
        A_nat_0 -= MatrixXt<Type>::Ones(S, S);
        // for numerical stability
        A_nat_0 += eps*MatrixXt<Type>::Ones(S, S);
        NPMatrix<Type> A_nat_N(A_N, S, S);
        A_nat_N -= MatrixXt<Type>::Ones(S, S);
        // for numerical stability
        A_nat_N += eps*MatrixXt<Type>::Ones(S, S);

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

                ArrayXt<Type> obs_sub = e_obs.block(m.i1, 0, (m.i2 - m.i1) + 1, 2);
                ArrayXt<Type> var_x = local_update(obs_sub, pi, A_nat_N, emits_mo_N);

                // intermediate parameters
                MatrixXt<Type> A_i = A_nat_0 + MatrixXt<Type>::Ones(S, S);
                A_i += dir::sufficient_statistics(var_x);
                A_i -= MatrixXt<Type>::Ones(S, S);
                A_inter += A_i;

                for (int s = 0; s < S; ++s) {
                    niw::e_suff_stats<Type> emit_i =  \
                        niw::expected_sufficient_statistics(obs_sub, var_x, s);
                    emits_inter[s].s1 += emit_i.s1;
                    emits_inter[s].s2 += emit_i.s2;
                    emits_inter[s].s3 += emit_i.s3;
                }
            }

            // global update
            int B = 2.0*L + 1.0;
            Type A_bfactor = (T - 2.0*L - 1.0)/(2.0*L*B);
            dir::meanfield_sgd_update(lrate, A_bfactor, A_nat_0, A_nat_N, A_inter);

            Type e_bfactor = (T - 2.0*L - 1.0)/((2.0*L + 1.0)*B);
            for (int s = 0; s < S; ++s) 
                niw::meanfield_sgd_update(lrate, e_bfactor, emits_mo_0[s],
                                          emits_mo_N[s], emits_inter[s]);
        }

        // convert natural parameters back to moment form before return
        A_nat_N += MatrixXt<Type>::Ones(S, S);
    }
}


#endif
