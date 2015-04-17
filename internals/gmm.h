#ifndef GMM_H
#define GMM_H

#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include "np_types.h"
#include "eigen_types.h"


namespace gmm {
  using namespace std;
  using namespace Eigen;
  using namespace nptypes;
  using namespace eigentypes;

  static const constexpr double log2pi = 1.83787706640934533908193770912475883;

  /*
   * Multivariate Gaussian Distribution.
   *
   * x
   * e_mu
   * sigma_inv
   * lsigma_det
   */
  template <typename Type, typename Derived>
  Type _log_probability_m(const ArrayBase<Derived>& x, const ArrayXt<Type> e_mu,
      const MatrixXt<Type> sigma_inv, const Type lsigma_det) {
    auto diff = (x - e_mu).matrix();
    auto descriptive_stat = diff*sigma_inv*diff.transpose();
    // NOTE: e_mu = [x_1, ..., x_n]
    return -0.5*(lsigma_det + descriptive_stat + e_mu.cols()*log2pi);
  }

  /* 
   * Multivariate Gaussian Mixture Model.
   *
   * L
   * x
   * e_mus
   * e_sigmas
   *
   * Notes:
   * Once this is tested pass in log sigma inverse and log sigma determinant so
   * they only need to be calculated once.
   *
   * Taking Eigen types as arguments.
   * eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
   */
  template <typename Type, typename Derived>
  ArrayXt<Type> _log_probabilities_mm(int L,
      const ArrayBase<Derived>& x, const ArrayXt<Type>& e_c,
      const vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > >& e_mus,
      const vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > >& e_sigmas) {

    auto lprobs = ArrayXt<Type>(1, L);
    for (int l = 0; l < L; ++l) {
      auto sigma_inv = e_sigmas[l].inverse();
      auto lsigma_det = log(e_sigmas[l].determinant());
      lprobs(1, l) = log(e_c.coeff(l)) + _log_probability_m<Type>(x, e_mus[l], sigma_inv, lsigma_det);
    }
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
   * expected_x2     = \sum_{t=1}^T \gamma_{s,l}(t)*o_t*o_t^T
   * expectex_x      = \sum_{t=1}^T \gamma_{s,l}*o_t
   * expected_counts = \sum_{t=1}^T \gamma_{s,l}
   */
  template <typename Type>
  void _weighted_sufficient_statistics(int S, int T, int D, int L,
      const vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > >& e_cs,
      const vector<vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > > >& e_mus,
      const vector<vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > > >& e_sigmas,
      const NPArray<Type>& e_obs, const NPArray<Type>& e_lweights,

      vector<vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > > >& expected_x2,
      vector<vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > > >& expected_x,
      MatrixXt<Type>& expected_count) {

    auto lprobs = ArrayXt<Type>(1, L);
    auto component_weight = ArrayXt<Type>(1, L);
    auto x2 = MatrixXt<Type>(D, D);

    for (int t = 0; t < T; ++t) {
      x2 = e_obs.row(t)*e_obs.row(t).transpose();
      for (int s = 0; s < S; ++s) {
        lprobs = _log_probabilities_mm<Type>(L, e_obs.row(t), e_cs[s], e_mus[s], e_sigmas[s]);
        component_weight = (exp(e_lweights.coeff(t, s))*lprobs.exp())/lprobs.exp().sum();

        expected_count.row(s) += component_weight.matrix();
        for (int l = 0; l < L; ++l) {
          expected_x2[s][l] += component_weight.coeff(l)*x2;
          expected_x[s][l] += component_weight.coeff(l)*e_obs.row(t);
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

    NPArray<Type> e_obs(obs, T, D);
    NPArray<Type> e_lweights(lweights, T, S);

    vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > > e_cs;
    for (i1 = 0; i1 < S; ++i1)
      for (i2 = 0; i2 < S*L; i2 += L)
        e_cs[i1] = NPArray<Type>(&cs[i2], L, 1);

    vector<vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > > > e_mus;
    for (i1 = 0; i1 < S; ++i1)
      for (i2 = 0; i2 < L; ++i2)
        e_mus[i1][i2] = NPArray<Type>(&mus[(i1*L+i2)*D], D, 1);

    vector<vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > > > e_sigmas;
    for (i1 = 0; i1 < S; ++i1)
      for (i2 = 0; i2 < L; ++i2)
        e_sigmas[i1][i2] = NPMatrix<Type>(&sigmas[(i1*L+i2)*D*D], D, D);

    vector<vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > > > expected_x2;
    for (i1 = 0; i1 < S; ++i1)
      for (i2 = 0; i2 < L; ++i2)
        expected_x2[i1][i2] = MatrixXt<Type>(D, D);

    vector<vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > > > expected_x;
    for (i1 = 0; i1 < S; ++i1)
      for (i2 = 0; i2 < L; ++i2)
        expected_x[i1][i2] = ArrayXt<Type>(1, D);

    auto expected_counts = MatrixXt<Type>(S, L);

    _weighted_sufficient_statistics(S, T, D, L, e_cs, e_mus, e_sigmas, e_obs,
        e_lweights, expected_x2, expected_x, expected_counts);
  }
}


#endif
