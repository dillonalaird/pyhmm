#ifndef DIRICHLET_H
#define DIRICHLET_H

#include <Eigen/Core>
#include <boost/math/special_functions/digamma.hpp>
#include "np_types.h"
#include "eigen_types.h"


namespace dir {
    using namespace boost;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

    static const constexpr double eps = 0.000000001;

    /*
     * Calculates the expected sufficient statistics.
     *
     * A - The dirichlet parameters.
     *
     * ess - \psi(\alpha_{i,j}) - \psi(\sum_{j=1}^S \alpha_{i,j})
     */
    template <typename Type>
    MatrixXt<Type> expected_sufficient_statistics(const MatrixXt<Type>& A) {
        MatrixXt<Type> ess = MatrixXt<Type>::Zero(A.rows(), A.cols());

        VectorXt<Type> row_sums = A.rowwise().sum();
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j)
                ess(i,j) = math::digamma(A.coeff(i,j) + eps) - \
                           math::digamma(row_sums.coeff(i) + eps);

        return ess;
    }

    /*
     * Calculates the sufficient statistics for the dirichlet distribution.
     *
     * es - The expected states.
     *
     * Container for the sufficient statistics.
     *
     * ss - \sum_{i=2}^N expected_states_{i-1}^T*expected_states_i
     */
    template <typename Type>
    MatrixXt<Type> sufficient_statistics(const ArrayXt<Type>& es) {
        MatrixXt<Type> ss = MatrixXt<Type>::Zero(es.cols(), es.cols());

        for (int i = 1; i < es.rows(); ++i)
            ss += es.row(i-1).matrix().transpose()*es.row(i).matrix();

        return ss;
    }

    /*
     * Calculates the stochastic gradient descent meanfield update inplace.
     *
     * lrate   - The learning rate.
     * bfactor - The batch factor.
     *
     * A_nat_0 - The prior natural parameters.
     * A_nat_N - The natural parameters.
     * ess     - The expected sufficient statistics.
     */
    template <typename Type>
    void meanfield_sgd_update(Type lrate, Type bfactor,
                              const NPMatrix<Type>& A_nat_0,
                              NPMatrix<Type>& A_nat_N,
                              const MatrixXt<Type>& ess) {
        A_nat_N = (1 - lrate)*A_nat_N + lrate*(A_nat_0 + bfactor*ess);
    }
}


#endif
