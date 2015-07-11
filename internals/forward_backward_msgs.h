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
  void forward_msgs(int S, int T, Type* pi, Type* A, Type* lliks, 
                    Type* lalpha) {
    NPArray<Type>  e_pi(pi, 1, S);
    NPMatrix<Type> e_A(A, S, S);
    NPArray<Type>  e_lliks(lliks, T, S);
    NPArray<Type>  e_lalpha(lalpha, T, S);

    Type cmax;
    e_lalpha.row(0) = e_pi.log() + e_lliks.row(0);

    for (int t = 0; t < T-1; ++t) {
      cmax = e_lalpha.row(t).maxCoeff();
      e_lalpha.row(t+1) = ((e_lalpha.row(t) - cmax).exp().matrix() * 
          e_A).array().log() + cmax + e_lliks.row(t+1);
    }
  }

  template <typename Type>
  void backward_msgs(int S, int T, Type* A, Type* lliks, Type* lbeta) {
    NPMatrix<Type>  e_A(A, S, S);
    NPMatrix<Type>  e_lliks(lliks, T, S);
    NPMatrix<Type>  e_lbeta(lbeta, T, S);

    Type thesum_buf[S] __attribute__((aligned(16)));
    NPVector<Type> thesum(thesum_buf, S);
    Type cmax;

    e_lbeta.row(T-1).setZero();
    for (int t = T-2; t >= 0; --t) {
      thesum = (e_lliks.row(t+1) + e_lbeta.row(t+1)).transpose();
      cmax = thesum.maxCoeff();
      e_lbeta.row(t) = (e_A*(thesum.array() - 
            cmax).exp().matrix()).array().log() + cmax;
    }
  }

  /*
   * Notes:
   * These are log weights for states. These don't include log likelihoods.
   */
  template <typename Type>
  void log_weights(int S, int T, Type* lalpha, Type* lbeta, Type* lweights) {
    NPArray<Type> e_lalpha(lalpha, T, S);
    NPArray<Type> e_lbeta(lbeta, T, S);
    NPArray<Type> e_lweights(lweights, T, S);

    Type thesum_buf[S] __attribute__((aligned(16)));
    NPVector<Type> thesum(thesum_buf, S);
    Type abmax;

    for (int t = 0; t < T; ++t) {
      thesum = e_lalpha.row(t) + e_lbeta.row(t);
      abmax = thesum.maxCoeff();
      e_lweights.row(t) = e_lalpha.row(t) + e_lbeta.row(t) - 
        (log((e_lalpha.row(t) + e_lbeta.row(t) - abmax).exp().sum()) + abmax);
    }
  }

  template <typename Type>
  void expected_statistics(int S, int T, Type* pi, Type* A, Type* lliks,
                           Type* lalpha, Type* lbeta,
                           Type* lexpected_states, 
                           Type* expected_transcounts) {
    NPArray<Type> e_lA(A, S, S);
    e_lA = e_lA.log();
    NPArray<Type> e_lliks(lliks, T, S);
    NPArray<Type> e_lalpha(lalpha, T, S);
    NPArray<Type> e_lbeta(lbeta, T, S);
    NPArray<Type> e_lexpected_states(lexpected_states, T, S);
    NPArray<Type> e_expected_transcounts(expected_transcounts, S, S);

    Type pair_buf[S*S] __attribute__((aligned(16)));
    NPArray<Type> pair(pair_buf, S, S); 

    Type cmax;

    for (int t = 0; t < T-1; ++t) {
      pair = e_lA;
      pair.colwise() += e_lalpha.row(t).transpose().array();
      pair.rowwise() += e_lbeta.row(t+1) + e_lliks.row(t+1);

      cmax = pair.maxCoeff();
      pair -= cmax;

      e_expected_transcounts += pair.exp();
      cmax = (e_lalpha.row(t) + e_lbeta.row(t)).maxCoeff();
      e_lexpected_states.row(t) = (e_lalpha.row(t) + e_lbeta.row(t)) - cmax;
    }

    // don't need to add e_lbeta.row(T-1) here because it's 0
    cmax = e_lalpha.row(T-1).maxCoeff();
    e_lexpected_states.row(T-1) = e_lalpha.row(T-1) - cmax;
  }
}


#endif
