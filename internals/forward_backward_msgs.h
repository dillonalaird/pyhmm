#ifndef FORWARD_BACKWARD_DSGS_H
#define FORWARD_BACKWARD_DSGS_H

#include <iostream>
#include <Eigen/Core>
#include <limits>

#include "util.h"
#include "np_types.h"


namespace fb {
  using namespace std;
  using namespace Eigen;
  using namespace nptypes;

  template <typename Type>
  void forward_msgs(int D, int T, Type* pi, Type* A, Type* lliks, Type* lalpha) {
    NPArray<Type>  e_pi(pi, 1, D);
    NPMatrix<Type> e_A(A, D, D);
    NPArray<Type>  e_lliks(lliks, T, D);
    NPArray<Type>  e_lalpha(lalpha, T, D);

    Type cmax;
    e_lalpha.row(0) = e_pi.log() + e_lliks.row(0);

    for (int t = 0; t < T - 1; t++) {
      cmax = e_lalpha.row(t).maxCoeff();
      e_lalpha.row(t + 1) = ((e_lalpha.row(t) - cmax).exp().matrix() * 
          e_A).array().log() + cmax + e_lliks.row(t + 1);
    }
  }

  template <typename Type>
  void backward_msgs(int D, int T, Type* A, Type* lliks, Type* lbeta) {
    NPMatrix<Type>  e_A(A, D, D);
    NPMatrix<Type>  e_lliks(lliks, T, D);
    NPMatrix<Type>  e_lbeta(lbeta, T, D);

    Type thesum_buf[D] __attribute__((aligned(16)));
    NPVector<Type> thesum(thesum_buf, D);
    Type cmax;

    e_lbeta.row(T - 1).setZero();
    for (int t = T - 2; t >= 0; t--) {
      thesum = (e_lliks.row(t + 1) + e_lbeta.row(t + 1)).transpose();
      cmax = thesum.maxCoeff();
      e_lbeta.row(t) = (e_A * (thesum.array() - 
            cmax).exp().matrix()).array().log() + cmax;
    }
  }
}


#endif
