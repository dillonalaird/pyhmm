#ifndef FORWARM_BACKWARM_MSGS_H
#define FORWARM_BACKWARM_MSGS_H

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
  void forward_msgs(int M, int T, Type* pi, Type* A, Type* lliks, 
                    Type* lalpha) {
    NPArray<Type>  e_pi(pi, 1, M);
    NPMatrix<Type> e_A(A, M, M);
    NPArray<Type>  e_lliks(lliks, T, M);
    NPArray<Type>  e_lalpha(lalpha, T, M);

    Type cmax;
    e_lalpha.row(0) = e_pi.log() + e_lliks.row(0);

    for (int t = 0; t < T-1; ++t) {
      cmax = e_lalpha.row(t).maxCoeff();
      e_lalpha.row(t+1) = ((e_lalpha.row(t) - cmax).exp().matrix() * 
          e_A).array().log() + cmax + e_lliks.row(t+1);
    }
  }

  template <typename Type>
  void backward_msgs(int M, int T, Type* A, Type* lliks, Type* lbeta) {
    NPMatrix<Type>  e_A(A, M, M);
    NPMatrix<Type>  e_lliks(lliks, T, M);
    NPMatrix<Type>  e_lbeta(lbeta, T, M);

    Type thesum_buf[M] __attribute__((aligned(16)));
    NPVector<Type> thesum(thesum_buf, M);
    Type cmax;

    e_lbeta.row(T-1).setZero();
    for (int t = T-2; t >= 0; --t) {
      thesum = (e_lliks.row(t+1) + e_lbeta.row(t+1)).transpose();
      cmax = thesum.maxCoeff();
      e_lbeta.row(t) = (e_A*(thesum.array() - 
            cmax).exp().matrix()).array().log() + cmax;
    }
  }

  template <typename Type>
  void expected_statistics(int M, int T, Type* pi, Type* A, Type* lliks,
                           Type* lalpha, Type* lbeta,
                           Type* lexpected_states, 
                           Type* expected_transcounts) {
    NPArray<Type> e_lA(A, M, M);
    e_lA = e_lA.log();
    NPArray<Type> e_lliks(lliks, T, M);
    NPArray<Type> e_lalpha(lalpha, T, M);
    NPArray<Type> e_lbeta(lbeta, T, M);
    NPArray<Type> e_lexpected_states(lexpected_states, T, M);
    NPArray<Type> e_expected_transcounts(expected_transcounts, M, M);

    Type pair_buf[M*M] __attribute__((aligned(16)));
    NPArray<Type> pair(pair_buf, M, M); 

    Type cmax;
    Type cmax_holder_buf[4] __attribute__((aligned(16)));
    NPArray<Type> cmax_holder(cmax_holder_buf, 4, 1);

    for (int t = 0; t < T-1; ++t) {
      pair = e_lA;
      pair.colwise() += e_lalpha.row(t).transpose().array();
      pair.rowwise() += e_lbeta.row(t+1) + e_lliks.row(t+1);

      cmax_holder << e_lA.maxCoeff(), e_lalpha.row(t).maxCoeff(), 
           e_lbeta.row(t+1).maxCoeff(), e_lliks.row(t+1).maxCoeff();
      cmax = cmax_holder.maxCoeff();
      pair -= log((pair - cmax).exp().sum()) + cmax;

      e_expected_transcounts += pair.exp();
      e_lexpected_states.row(t) = (e_lalpha.row(t) + e_lbeta.row(t));
    }

    // don't need to add e_lbeta.row(T-1) here because it's 0
    e_lexpected_states.row(T-1) = e_lalpha.row(T-1);
  }
}


#endif
