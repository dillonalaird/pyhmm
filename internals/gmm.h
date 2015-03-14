#ifndef GMM_H
#define GMM_H

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

  template <typename Type>
  void update_parameters(int D, int T, int L, Type* cs, Type* mus, Type* sigmas,
                         Type* obs, Type* lweights) {
    NPArray<Type> e_cs(cs, L, 1);
    NPArray<Type> e_mus(mus, L, D);
    // use std::vector here instead?
    NPArray<Type> e_sigmas(sigmas, L*D, D);
    /*
     * for i in e_sigmas:
     *  e_sigmas[i] = NPMatrix()
     */
    NPArray<Type> e_lweights(lweights, T, D);
  }
}


#endif
