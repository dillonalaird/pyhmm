#ifndef FORWARD_BACKWARD_DSGS_H
#define FORWARD_BACKWARD_DSGS_H

#include <iostream>
#include <Eigen/Core>
#include <limits>

#include "np_types.h"


#define likely(x) __builtin_expect(!!(x), true)
#define unlikely(x) __builtin_expect(!!(x), false)


namespace fb {
  using namespace std;
  using namespace Eigen;
  using namespace nptypes;

  template<typename T> bool is_not_nan(const T& x) { return x == x; }
  template<typename T> bool is_finite(const T& x) { return is_not_nan(x - x); }

  template<typename Type>
  void forward_msgs(int D, int T, Type* pi, Type* ltran, Type* lliks, Type* lalpha) {
    NPArray<Type>  e_pi(pi, 1, D);
    NPMatrix<Type> e_ltran(ltran, D, D);
    NPArray<Type>  e_lliks(lliks, T, D);
    NPArray<Type>  e_lalpha(lalpha, T, D);

    Type cmax;
    e_lalpha.row(0) = e_pi.log() + e_lliks.row(0);

    for (int t = 0; t < T - 1; ++t) {
      cmax = e_lalpha.row(t).maxCoeff();
      if (likely(is_finite(cmax))) {
        e_lalpha.row(t + 1) = ((e_lalpha.row(t) - cmax).exp().matrix() * 
            e_ltran).array().log() + e_lliks.row(t + 1) + cmax;
      } else {
        e_lalpha.block(t+1, 0, T - (t + 1), D).setConstant(
            -numeric_limits<Type>::infinity());
        return;
      }
    }
  }

  template<typename Type>
  void backward_msgs(int D, int T, Type* ltran, Type* lliks, Type* lbeta) {
    NPMatrix<Type> e_ltran(ltran, D, D);
    NPArray<Type>  e_lliks(lliks, T, D);
    NPArray<Type>  e_lbeta(lbeta, T, D);

    Type thesum_buf[D] __attribute__((aligned(16)));
    NPVector<Type> thesum(thesum_buf, D);
    Type cmax;

    e_lbeta.row(T - 1).setZero();
    for (int t = 0; t < T - 2; ++t) {
      thesum = (e_lliks.row(t + 1) + e_lbeta.row(t + 1)).transpose();
      cmax = thesum.maxCoeff();
      e_lbeta.row(t) = (e_ltran * (thesum.array() - 
            cmax).exp().matrix()).array().log() + cmax;
    }
  }
}


#endif
