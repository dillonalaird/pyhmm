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
  Type _log_probability_m(const ArrayBase<Derived>& x, const VectorXt<Type> e_mu,
      const MatrixXt<Type> sigma_inv, const Type lsigma_det) {
    auto diff = (x - e_mu.array().transpose()).matrix().transpose();
    auto descriptive_stat = (diff.transpose()*sigma_inv*diff).coeff(0);
    return -0.5*(lsigma_det + descriptive_stat + e_mu.rows()*log2pi);
  }

  /* 
   * Multivariate Gaussian Mixture Model.
   *
   * L
   * x
   * e_c
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
      const vector<VectorXt<Type>, aligned_allocator<VectorXt<Type> > >& e_mus,
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
   * Multivariate Gaussian Mixture Model. This is the same as 
   * _log_probabilities_mm but takes in the inverse covariance's and the log
   * determinant's of the covariances so they don't have to be re-calculated.
   *
   * L
   * x
   * e_c
   * e_mus
   * sigma_invs
   * lsigma_dets
   *
   * Notes:
   *
   * Taking Eigen types as arguments.
   * eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
   */
  template <typename Type, typename Derived>
  ArrayXt<Type> _log_probabilities_mm_fast(int L,
      const ArrayBase<Derived>& x, const ArrayXt<Type>& e_c,
      const vector<VectorXt<Type>, aligned_allocator<VectorXt<Type> > >& e_mus,
      const vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > >& sigma_invs,
      const vector<Type> lsigma_dets) {

    auto lprobs = ArrayXt<Type>(1, L);
    for (int l = 0; l < L; ++l)
      lprobs(1, l) = log(e_c.coeff(l)) + _log_probability_m<Type>(x, e_mus[l], sigma_invs[l], lsigma_dets[l]);

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
      const vector<vector<VectorXt<Type>, aligned_allocator<VectorXt<Type> > > >& e_mus,
      const vector<vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > > >& e_sigmas,
      const NPArray<Type>& e_obs, const NPArray<Type>& e_lweights,

      vector<vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > > >& expected_x2,
      vector<vector<ArrayXt<Type>, aligned_allocator<ArrayXt<Type> > > >& expected_x,
      MatrixXt<Type>& expected_count) {

    // precalculate inverses and log determinants
    vector<vector<MatrixXt<Type>, aligned_allocator<MatrixXt<Type> > > > sigma_invs;
    vector<vector<Type> > lsigma_dets;
    for (int s = 0; s < S; ++s)
      for (int l = 0; l < L; ++l) {
        sigma_invs[s][l] = e_sigmas[s][l].inverse();
        lsigma_dets[s][l] = log(e_sigmas[s][l].determinant());
      }

    auto lprobs = ArrayXt<Type>(1, L);
    auto component_weight = ArrayXt<Type>(1, L);
    auto x2 = MatrixXt<Type>(D, D);

    for (int t = 0; t < T; ++t) {
      x2 = e_obs.row(t)*e_obs.row(t).transpose();
      for (int s = 0; s < S; ++s) {
        //lprobs = _log_probabilities_mm<Type>(L, e_obs.row(t), e_cs[s], e_mus[s], e_sigmas[s]);
        lprobs = _log_probabilities_mm_fast<Type>(L, e_obs.row(t), e_cs[s], e_mus[s], 
            sigma_invs[s], lsigma_dets[s]);
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
   * eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html
   * eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
   */
  template <typename Type>
  void update_parameters(int S, int T, int D, int L, Type* cs, Type* mus, 
                         Type* sigmas, Type* obs, Type* lweights) {
    // populate Eigen containers
    NPArray<Type> e_obs(obs, T, D);
    NPArray<Type> e_lweights(lweights, T, S);

    for (int i = 0; i < S*L; ++i) {
      std::cout << i << " cs[i] = " << cs[i] << std::endl;
      //std::cout << i << " &cs[i] = " << &cs[i] << std::endl;
    }
    
    int i = 0;
    vector<NPArray<Type> > e_cs;

    // segfault
    //e_cs.push_back(NPArray<Type>(&cs[L], 1, L));
    //std::cout << e_cs[0] << std::endl;
    
    /*
    auto tmp0 = NPArray<Type>(&cs[0*L], 1, L);
    std::cout << tmp0 << std::endl;

    auto tmp2 = NPArray<Type>(&cs[2*L], 1, L);
    std::cout << tmp2 << std::endl;
    */

    auto tmpX = NPArray<Type>(&cs[2*L-2], 1, L);
    std::cout << tmpX << std::endl;

    auto tmpY = NPArray<Type>(&cs[0*L+2], 1, L);
    std::cout << tmpY << std::endl;

    std::cout << "coefficients" << std::endl;
    auto tmp1 = NPArray<Type>(&cs[1*L], 1, L);
    std::cout << tmp1.coeff(0,0) << std::endl;
    std::cout << tmp1.coeff(0,1) << std::endl;
    std::cout << tmp1.coeff(0,2) << std::endl;

    // segfault
    std::cout << tmp1 << std::endl;

    for (int i1 = 0; i1 < S; ++i1) {
      std::cout << "push back " << i1*L << std::endl;
      e_cs.push_back(NPArray<Type>(&cs[i1*L], 1, L));
      std::cout << cs[i1*L] << std::endl;

      //for (int i2 = i1*L; i2 < i1*L+L; ++i2)
      //  std::cout << i2 << " cs[i2] = " << cs[i2] << std::endl;

      //std::cout << i << " e_cs[i] = " << e_cs[i] << std::endl;
      ++i;
    }

    /*
    vector<vector<NPVector<Type> > > e_mus;
    for (int i1 = 0; i1 < S; ++i1) {
      e_mus.push_back(vector<NPVector<Type> >());
      for (int i2 = 0; i2 < L; ++i2)
        e_mus[i1].push_back(NPVector<Type>(&mus[(i1*L+i2)*D], D));
    }
    */

    /*
    vector<vector<NPMatrix<Type> > > e_sigmas;
    for (int i1 = 0; i1 < S; ++i1)
      for (int i2 = 0; i2 < L; ++i2) {
        e_sigmas.push_back(vector<NPMatrix<Type> >());
        e_sigmas[i1].push_back(NPMatrix<Type>(&sigmas[(i1*L+i2)*D*D], D, D));
      }

    vector<vector<MatrixXt<Type> > > expected_x2;
    for (int i1 = 0; i1 < S; ++i1)
      for (int i2 = 0; i2 < L; ++i2) {
        expected_x2.push_back(vector<MatrixXt<Type> >());
        expected_x2[i1].push_back(MatrixXt<Type>(D, D));
      }

    vector<vector<ArrayXt<Type> > > expected_x;
    for (int i1 = 0; i1 < S; ++i1)
      for (int i2 = 0; i2 < L; ++i2) {
        expected_x.push_back(vector<ArrayXt<Type> >());
        expected_x[i1].push_back(ArrayXt<Type>(1, D));
      }

    auto expected_counts = MatrixXt<Type>(S, L);
    */

    //DEBUG
    std::cout << "0: " << e_cs[0] << std::endl;
    std::cout << "2: " << e_cs[2] << std::endl;
    std::cout << "4: " << e_cs[4] << std::endl;
    // segfault?
    std::cout << "3: " << e_cs[3] << std::endl;
    std::cout << "1: " << e_cs[1] << std::endl;

    std::cout << "e_cs =" << std::endl;
    for (int i1 = 0; i1 < S; ++i1) {
      std::cout << i1 << std::endl;
      std::cout << e_cs[i1] << std::endl;
    }
    
    /*
    std::cout << "e_mus =" << std::endl;
    for (int i1 = 0; i1 < S; ++i1)
      for (int i2 = 0; i2 < L; ++i2)
        std::cout << e_mus[i1][i2] << std::endl;
    */

    /*
    std::cout << "e_sigmas =" << std::endl;
    for (int i1 = 0; i1 < S; ++i1)
      for (int i2 = 0; i2 < L; ++i2)
        std::cout << e_sigmas[i1][i2] << std::endl;
    */

    /*
    _weighted_sufficient_statistics(S, T, D, L, e_cs, e_mus, e_sigmas, e_obs,
        e_lweights, expected_x2, expected_x, expected_counts);

    // update parameters
    for (int s = 0; s < S; ++s) {
      e_cs[s] = expected_counts.row(s)/expected_counts.row(s).sum();
      for (int l = 0; l < L; ++l) {
        e_mus[s][l] = expected_x[s][l]/expected_counts.coeff(s, l);
        e_sigmas[s][l] = (expected_x2[s][l] - expected_counts.coeff(s, l)*
            (e_mus[s][l]*e_mus[s][l].transpose()).matrix())/expected_counts.coeff(s, l);
      }
    }
    */
  }
}


#endif
