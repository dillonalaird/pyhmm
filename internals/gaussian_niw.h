#ifndef GAUSSIAN_NIW_H
#define GAUSSIAN_NIW_H

#include <Eigen/Core>
#include "np_types.h"
#include "eigen_types.h"


namespace gaussian_niw {
    using namespace std;
    using namespace Eigen;
    using namespace eigentypes;


    /*
     * D - Dimension of data.
     *
     * The u's are the hyperparameters for Gaussian using a normal-inverse-
     * Wishart distribution with parameters mu_0, sigma_0, kappa_0 and nu_0.
     * These are updated inplace.
     *
     * u1 - kappa_0*mu_0
     * u2 - kappa_0
     * u3 - sigma_0 + kappa_0*mu_0*mu_0^T
     * u4 - nu_0 + 2 + p
     *
     * Weighted Sufficient Statistics.
     *
     * s1 - \sum_{i=1}^N w_i(x_i*x_i^T)
     * s2 - \sum_{i=1}^N w_i*x_i
     * s3 - \sum_{i=1}^N w_i
     */
    template  <typename Type>
    void meanfield_update(int D, Type* nat_params, Type* s1, Type* s2, Type s3) { }
}

#endif
