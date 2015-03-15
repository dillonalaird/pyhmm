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
    vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > e_cs;
    vector<vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > > e_mus;
    vector<vector<NPMatrix<Type>, aligned_allocator<NPArray<Type> > > > e_sigmas;
    NPArray<Type> e_obs(obs, T, D);
    NPArray<Type> e_lweights(lweights, T, S);
  }

  /*
  template <typename Type>
  void update_parameters(int M, int T, int L, Type* cs, Type* mus, Type* sigmas,
                         Type* obs, Type* lweights) {
    NPArray<Type> e_cs(cs, L, 1);
    NPArray<Type> e_mus(mus, L, D);

    // Can't find better way to do this. Also need special allocator for stl
    // containers holding Eigen structures.
    vector<NPMatrix<Type>, aligned_allocator<NPMatrix<Type> > > e_sigmas;
    int j;
    int i;
    for (i = 0, j = 0; i < L, j < D*D; ++i, j += D*D) 
      e_sigmas[i] = NPMatrix<Type>(&sigmas[j], D, D);

    NPArray<Type> e_lweights(lweights, T, D);

    for (auto it = e_sigmas.begin(); it != e_sigmas.end(); ++it)
      cout << *it << endl;
  }
  */
}


#endif
