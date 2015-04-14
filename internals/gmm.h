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

  /* 
   * Multivariate Gaussian Distribution.
   */
  template <typename Type>
  Type _log_probability_m(NPVector<Type> x, NPVector<Type> e_mu, 
                          NPMatrix<Type> sigma_inv, Type lsigma_det) {

    NPArray<Type> diff = x - e_mu;
    Type descriptive_stat = diff.transpose()*sigma_inv*diff;
    return -0.5*(lsigma_det + descriptive_stat + e_mu.rows()*log2pi);
  }

  /* 
   * Multivariate Gaussian Mixture Model.
   *
   * Notes:
   * Once this is tested pass in log sigma inverse and log sigma determinant so
   * they only need to be calculated once.
   *
   * Taking Eigen types as arguments.
   * eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
   */
  template <typename Type, typename Derived>
  NPArray<Type> _log_probabilities_mm(int L, const ArrayBase<Derived>& x, 
      NPArray<Type> e_cs, 
      vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > e_mus, 
      vector<NPMatrix<Type>, aligned_allocator<NPMatrix<Type> > > e_sigma) {

    Array<Type, Dynamic, Dynamic, RowMajor> lprobs(L, 1);
    /*
    for (int l = 0; l < L; ++l) {
      // precalculate these
      NPMatrix<Type> lsigma_inv = e_sigma[l].inverse().log();
      Type lsigma_det = e_sigma[l].determinant().log();

      lprobs[l] = e_cs[l].log() + _log_probability_m(x, e_mus[l], lsigma_inv,
          lsigma_det);
    }
    */

    return lprobs;
  }

  /*
   * S - The number of states.
   * T - The number of observations.
   * D - The dimension of the observations.
   * L - The number of mixtures.
   *
   * e_cs       - An S x L vector of NPArray's representing the mixture 
   *            coefficients.
   * e_mus      - An S x L x D vector of vectors of NPArray's representing the 
   *            means of the mixtures.
   * e_sigmas   - An S x L x D x D vector of vectors of NPMatrix's representing 
   *            the covariances of the mixtures.
   * e_obs      - A T x D NPArray representing the observations.
   * e_lweights - A T x S NPArray representing the state weights.
   *
   * expected_x2     - An S x L x D x D vector of vectors of NPMatrix's that 
   *                 will contain the expected observations squared.
   * expected_x      - An S x L x D vector of vectors of NPArray's that will
   *                 contain the expected observations.
   * expected_counts - An S x L NPMatrix containing the expected counts.
   *
   * Notes:
   * expected_x2 = \sum_{t=1}^T \gamma_{s,l}(t)*o_t*o_t^T
   * expectex_x = \sum_{t=1}^T \gamma_{s,l}*o_t
   * expected_counts = \sum_{t=1}^T \gamma_{s,l}
   *
   * This might be more of an internal function since the parameters are all
   * Eigen containers (you can't call this from Python). This is so we don't
   * have to refill the Eigen containers again from update_parameters.
   */
  template <typename Type>
  void _weighted_sufficient_statistics(int S, int T, int D, int L, 
      vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > e_cs,
      vector<vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > > e_mus,
      vector<vector<NPMatrix<Type>, aligned_allocator<NPMatrix<Type> > > > e_sigmas,
      NPArray<Type> e_obs, NPArray<Type> e_lweights,

      vector<vector<NPMatrix<Type>, aligned_allocator<NPMatrix<Type> > > > expected_x2,
      vector<vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > > expected_x,
      NPMatrix<Type> expected_count) {

    Type lprobs_buff[L] __attribute__((aligned(16)));
    NPArray<Type> lprobs(lprobs_buff, L, 1);

    Type x2_buff[D*D] __attribute__((aligned(16)));
    NPMatrix<Type> x2(x2_buff, D, D);

    for (int t = 0; t < T; ++t) {
      x2 = e_obs.row(t)*e_obs.row(t);
      for (int s = 0; s < S; ++s) {
        lprobs = _log_probabilities_mm<Type>(L, e_obs.row(t), e_cs[s],
            e_mus[s], e_sigmas[s]);
        //expected_count.row(s) += e_lweights.coeff(t, s) + lprobs;
        for (int l = 0; l < L; ++l) {
          //expected_x2[s][l] += (e_lweights.coeff(t, s) + lprobs[l]).exp()*x2;
          //expected_x[s][l] += (e_lweights.coeff(t, s) + lprobs[l]).exp()*e_obs.row(t);
        }
      }
    }
  }

  /*
   * S - The number of states.
   * T - The number of observations.
   * D - The dimension of the observations.
   * L - The number of mixtures.
   *
   * cs     - An S x L array representing the mixutre coefficients.
   * mus    - An S x L x D array representing the means of the mixtures.
   * sigmas - An S x L x D x D array representing the covariances of the 
   *          mixtures.
   *
   * Notes:
   * 16-byte-aligned allocator must be used when putting Eigen types in STL
   * containers.
   *
   * eigen.tuxfamily.org/dox-devel/gropu__TopicStlContainers.html
   */
  template <typename Type>
  void update_parameters(int S, int T, int D, int L, Type* cs, Type* mus, 
                         Type* sigmas, Type* obs, Type* lweights) {
    // populate Eigen containers
    int i1;
    int i2;
    int i3;

    vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > e_cs;
    for (i1 = 0, i2 = 0; i1 < S, i2 < S*L; ++i1, i2 += L)
      e_cs[i1] = NPArray<Type>(&cs[i2], L, 1);

    vector<vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > > e_mus;
    for (i1 = 0, i2 = 0, i3 = 0; i1 < S, i2 < L, i3 < S*L*D; ++i1, ++i2, i3 += D)
      e_mus[i1][i2] = NPArray<Type>(&cs[i3], D, 1);

    vector<vector<NPMatrix<Type>, aligned_allocator<NPMatrix<Type> > > > e_sigmas;
    for (i1 = 0, i2 = 0, i3 = 0; i1 < S, i2 < L, i3 < S*L*D*D; ++i1, ++i2, i3 += D*D)
      e_sigmas[i1][i2] = NPMatrix<Type>(&sigmas[i3], D, D);

    NPArray<Type> e_obs(obs, T, D);
    NPArray<Type> e_lweights(lweights, T, S);

    // containers for weighted sufficient statistics
    vector<vector<NPMatrix<Type>, aligned_allocator<NPMatrix<Type> > > > expected_x2;
    for (i1 = 0, i2 = 0; i1 < S, i2 < L; ++i1, ++i2) {
      Type expected_x2_buff[D*D] __attribute__((aligned(16)));
      expected_x2[i1][i2] = NPMatrix<Type>(expected_x2_buff, D, D);
    }

    vector<vector<NPArray<Type>, aligned_allocator<NPArray<Type> > > > expected_x;
    for (i1 = 0, i2 = 0; i1 < S, i2 < L; ++i1, ++i2) {
      Type expected_x_buff[D] __attribute__((aligned(16)));
      expected_x[i1][i2] = NPArray<Type>(expected_x_buff, D, 1);
    }

    Type expected_counts_buff[S*L] __attribute__((aligned(16)));
    NPMatrix<Type> expected_counts(expected_counts_buff, S, L);

    _weighted_sufficient_statistics(S, T, D, L, e_cs, e_mus, e_sigmas, e_obs, 
        e_lweights, expected_x2, expected_x, expected_counts);

    for (int s = 0; s < S; ++s)
      for (int l = 0; l < L; ++l) {
      }
  }
}


#endif
