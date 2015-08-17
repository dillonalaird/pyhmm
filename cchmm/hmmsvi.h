#ifndef HMMSVI_h
#define HMMSVI_h

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "np_types.h"
#include "eigen_types.h"
#include "metaobs.h"
#include "normal_invwishart.h"
#include "dirichlet.h"


namespace hmmsvi {
    using namespace mo;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

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
    void local_update(const ArrayXt<Type>& obs, const VectorXt<Type>& pi,
                      const NPMatrix<Type>& A_nat_N,
                      const std::vector<niw::nat_params<Type> >& emits_N) {
        MatrixXt<Type> A = A_nat_N + MatrixXt<Type>::Ones(A_nat_N.size(), A_nat_N.size());
        MatrixXt<Type> A_mod = dir::expected_sufficient_statistics<Type>(A);
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

        std::vector<niw::nat_params<Type> > nat_params_0;
        for (int s = 0; s < S; ++s)
            nat_params_0.push_back(niw::convert_to_struct(emits_0, D, s));
        std::vector<niw::nat_params<Type> > nat_params_N;
        for (int s = 0; s < S; ++S)
            nat_params_N.push_back(niw::convert_to_struct(emits_N, D, s));

        Type lrate = 0.0;
        for (int it = 0; it < itr; ++it) {
            lrate = pow(it + tau, -1*kappa);

            auto A_inter = MatrixXt<Type>::Zero(S, S);
            Type emits_inter_buff[S*(D*D + 3*D)] __attribute__((aligned(16))) = {};
            std::vector<niw::nat_params<Type> > emit_inter;
            for (int s = 0; s < S; ++s)
                emit_inter.push_back(niw::convert_to_struct(emits_inter_buff, D, s));

            for (int mit = 0; mit < n; ++mit) {
                metaobs m = metaobs_unif(T, L);
                VectorXt<Type> pi = _calc_pi<Type>(A_nat_N);

                ArrayXt<Type> obs_sub = e_obs.block(m.i1, 0, m.i2, 2);
                local_update(obs_sub, pi, A_nat_N, nat_params_N);
            }
        }
    }

    void global_update() { }

    void intermediate_pars() { }
}


#endif
