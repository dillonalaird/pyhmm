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

    template <typename Type>
    MatrixXt<Type> expected_sufficient_statistics(const MatrixXt<Type>& A) {
        MatrixXt<Type> ess = MatrixXt<Type>::Zero(A.size(), A.size());

        VectorXt<Type> row_sums = A.rowwise().sum();
        for (int i = 0; i < A.size(); ++i)
            for (int j = 0; j < A.size(); ++j)
                ess(i,j) = math::digamma(A.coeff(i,j) + eps) - \
                           math::digamma(row_sums.coeff(i) + eps);

        return ess;
    }
}


#endif
