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

    /*
     * Calculates the expected sufficient statistics.
     *
     * S      - The number of dirichlet parameters.
     * alphas - The dirichlet parameters.
     *
     * Container for the expected sufficient statistics.
     *
     * ess - \psi(\alpha_{i,j}) - \psi(\sum_{j=1}^S \alpha_{i,j})
     */
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

    /*
     * Calculates the sufficient statistics for the dirichlet distribution.
     *
     * S - The number of dirichlet parameters.
     * N - The number of observations.
     *
     * expected_states - The expected states.
     *
     * Container for the sufficient statistics.
     *
     * ss - \sum_{i=2}^N expected_states_{i-1}^T*expected_states_i
     */
    template <typename Type>
    void sufficient_statistics(int S, int N, Type* expected_states, Type* ss) {
        NPArray<Type>  e_es = NPArray<Type>(expected_states, N, S);
        NPMatrix<Type> e_ss = NPMatrix<Type>(ss, S, S);

        for (int i = 1; i < N; ++i) {
            e_ss += e_es.row(i-1).matrix().transpose()*e_es.row(i).matrix();
        }
    }

    /*
     * Calculates the meanfield update.
     *
     * S - The number of dirichlet parameters.
     * 
     * nat_params - The natural parameters.
     * ss         - The sufficient statistics.
     */
    template <typename Type>
    void meanfield_update(int S, Type* nat_params_0, Type* ss) {
        NPMatrix<Type> e_ns = NPMatrix<Type>(nat_params_0, S, S);
        NPMatrix<Type> e_ss = NPMatrix<Type>(ss, S, S);

        e_ns += e_ss;
    }

    /*
     * Calculates the stochastic gradient descent meanfield update.
     *
     * S       - The number of dirichlet parameters.
     * lrate   - The learning rate.
     * bfactor - The batch factor.
     *
     * nat_params_0 - The prior natural parameters.
     * nat_params_N - The natural parameters.
     * ess          - The expected sufficient statistics.
     */
    template <typename Type>
    void meanfield_sgd_update(int S, Type lrate, Type bfactor,
                              Type* nat_params_0, Type* nat_params_N, 
                              Type* ess) {
        NPMatrix<Type> e_ns_0 = NPMatrix<Type>(nat_params_0, S, S);
        NPMatrix<Type> e_ns_N = NPMatrix<Type>(nat_params_N, S, S);
        NPMatrix<Type> e_ss   = NPMatrix<Type>(ess, S, S);

        e_ns_N = (1 - lrate)*e_ns_N + lrate*(e_ns_0 + bfactor*e_ss);
    }
}

#endif
