#ifndef FORWARD_BACKWARD_MSGS_H
#define FORWARD_BACKWARD_MSGS_H

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
  void forward_msgs(int M, int T, Type* pi, Type* ltran, Type* lliks, Type* lalpha) {
    NPArray<Type>  e_pi(pi, 1, M);
    NPMatrix<Type> e_ltran(ltran, M, M);
    NPArray<Type>  e_lliks(lliks, T, M);
    NPArray<Type>  e_lalpha(lalpha, T, M);

    Type cmax;
    e_lalpha.row(0) = e_pi.log() + e_lliks.row(0);

    for (int t = 0; t < T - 1; ++t) {
      cmax = e_lalpha.row(t).maxCoeff();
      if (likely(is_finite(cmax))) {
        e_lalpha.row(t + 1) = ((e_lalpha.row(t) - cmax).exp().matrix() * 
            e_ltran).array().log() + e_lliks.row(t + 1) + cmax;
      } else {
        e_lalpha.block(t+1, 0, T - (t + 1), M).setConstant(
            -numeric_limits<Type>::infinity());
        return;
      }
    }
  }
}


#endif
