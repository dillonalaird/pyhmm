#ifndef HMMSVI_h
#define HMMSVI_h

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "np_types.h"
#include "eigen_types.h"
#include "metaobs.h"
#include "normal_invwishart.h"


namespace hmmsvi {
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

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
                mo::metaobs m = mo::metaobs_unif(T, L);
            }
        }
    }

    void local_update() { }

    void global_update() { }

    void intermediate_pars() { }

    template <typename Type>
    VectorXt<Type> _calc_pi(NPMatrix<Type> A_nat_N) {
        return NULL;
    }
}


#endif
