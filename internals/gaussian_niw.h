#ifndef GAUSSIAN_NIW_H
#define GAUSSIAN_NIW_H

#include <iostream>
#include <Eigen/Core>
#include "np_types.h"
#include "eigen_types.h"


namespace gaussian_niw {
    using namespace std;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;


    /*
     * D - Dimension of data.
     *
     * The u's are the hyperparameters for Gaussian using a normal-inverse-
     * Wishart distribution with parameters mu_0, sigma_0, kappa_0 and nu_0.
     * These are updated inplace.
     *
     * n1 - kappa_0*mu_0
     * n2 - kappa_0
     * n3 - sigma_0 + kappa_0*mu_0*mu_0^T
     * n4 - nu_0 + 2 + p
     *
     * Weighted Sufficient Statistics.
     *
     * s1 - \sum_{i=1}^N w_i
     * s2 - \sum_{i=1}^N w_i*x_i
     * s3 - \sum_{i=1}^N w_i(x_i*x_i^T)
     */
    template  <typename Type>
    void meanfield_update(int D, Type* nat_params, Type s1_, Type* s2_, Type* s3_) {
        NPMatrix<Type> n3 = NPMatrix<Type>(nat_params, D, D);
        NPVector<Type> n1 = NPVector<Type>(&nat_params[D*D], D, 1);
        Type& n2 = nat_params[D*(D+1)];
        Type& n4 = nat_params[D*(D+2)];

        Type& s1 = s1_;
        NPMatrix<Type> s2 = NPMatrix<Type>(s2_, D, 1);
        NPVector<Type> s3 = NPVector<Type>(s3_, D, D);
        
        n1 += s2;
        n2 += s1;
        n3 += s3;
        n4 += s1;
    }
}

#endif
