#ifndef DIRICHLET_H
#define DIRICHLET_H

#include <iostream>
#include <Eigen/Core>
#include <boost/math/special_functions/digamma.hpp>
#include "np_types.h"
#include "eigen_types.h"


namespace dir {
    using namespace std;
    using namespace boost;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

    template <typename Type>
    void expected_sufficient_statistics(int S, Type* alphas, Type* ess) {
        NPMatrix<Type> e_alphas = NPMatrix<Type>(alphas, S, S);
        NPMatrix<Type> e_ess    = NPMatrix<Type>(ess, S, S);

        VectorXt<Type> row_sums = e_alphas.rowwise().sum();
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                e_ess(i,j) = math::digamma(e_alphas.coeff(i,j)) - \
                             math::digamma(row_sums.coeff(i));
            }
        }
    }

    template <typename Type>
    void sufficient_statistics(int S, int N, Type* expected_states, Type* ss) {
        NPArray<Type>  e_es = NPArray<Type>(expected_states, N, S);
        NPMatrix<Type> e_ss = NPMatrix<Type>(ss, S, S);

        for (int i = 1; i < N; ++i) {
            e_ss += e_es.row(i-1).matrix().transpose()*e_es.row(i).matrix();
        }
    }

    template <typename Type>
    void meanfield_update(int S, Type* nat_params, Type* ss) {
        NPMatrix<Type> e_ns = NPMatrix<Type>(nat_params, S, S);
        NPMatrix<Type> e_ss = NPMatrix<Type>(ss, S, S);

        e_ns += e_ss;
    }
}

#endif
