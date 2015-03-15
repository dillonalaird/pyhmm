#ifndef GMM_H
#define GMM_H

#include <vector>
#include <iostream>
#include "np_types.h"


namespace gmm {
  using namespace std;
  using namespace Eigen;
  using namespace nptypes;

  static const constexpr double log2pi = 1.83787706640934533908193770912475883;

  /* Multivariate Gaussian Distribution.
   */
  template <typename Type>
  Type _log_probability_m(NPVector<Type> x, NPVector<Type> mu, 
                          NPMatrix<Type> sigma_inv, Type lsigma_det) {

    NPArray<Type> diff = x - mu;
    Type descriptive_stat = diff.transpose()*sigma_inv*diff;
    return -0.5*(lsigma_det - descriptive_stat - mu.rows()*log2pi);
  }

  /* Multivariate Gaussian Mixture Model.
   */
  template <typename Type>
  Type _log_probabilities_mm(int L, NPVector<Type> x,
                             NPArray<NPVector<Type> > mus, 
                             NPArray<NPMatrix<Type> > sigma_invs,
                             NPArray<Type> lsigma_dets) {
    NPArray<Type> probs(L, 1);
    for (int i = 0; i < L; ++i) {
      probs[i] = _log_probability_m(x, mus[i], sigma_invs[i], lsigma_dets[i]);
    }
  }

  template <typename Type>
  void log_likelihood() { }

  template <typename Type>
  void weighted_sufficient_statistics() { }

  /*
   * cs     - S x L
   * mus    - S x L x D
   * sigmas - S x L x D x D
   */
  template <typename Type>
  void update_parameters(int S, int T, int D, int L, Type* cs, Type* mus, 
                         Type* sigmas, Type* obs, Type* lweights) {
    int i1;
    int i2;
    int i3;

    vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > e_cs;
    for (i1 = 0, i2 = 0; i1 < S, i2 < S*L; ++i1, i2 += L)
      e_cs[i1] = NPArray<Type>(&cs[i2], L, 1);

    vector<vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > > e_mus;
    for (i1 = 0, i2 = 0, i3 = 0; i1 < S, i2 < L, i3 < S*L*D; ++i1, ++i2, i3 += D)
      e_mus[i1][i2] = NPArray<Type>(&cs[i3], D, 1);

    vector<vector<NPMatrix<Type>, aligned_allocator<NPArray<Type> > > > e_sigmas;
    for (i1 = 0, i2 = 0, i3 = 0; i1 < S, i2 < L, i3 < S*L*D*D; ++i1, ++i2, i3 += D*D)
      e_sigmas[i1][i2] = NPMatrix<Type>(&sigmas[i3], D, D);

    NPArray<Type> e_obs(obs, T, D);
    NPArray<Type> e_lweights(lweights, T, S);
  }
}


#endif
