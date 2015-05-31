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
  template <typename Type>
  Type _log_probability_m(const VectorXt<Type>& x, const VectorXt<Type>& mu,
      const MatrixXt<Type>& sigma_inv, const Type lsigma_det) {
    VectorXt<Type> diff = (x - mu);
    Type descriptive_stat = (diff.transpose()*sigma_inv*diff);
    return -0.5*(lsigma_det + descriptive_stat + mu.rows()*log2pi);
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
      const ArrayBase<Derived>& x, const NPArray<Type>& e_c,
      const vector<NPVector<Type> >& e_mus,
      const vector<internal::inverse_impl<NPMatrix<Type> > >& e_sigmas) {

    auto lprobs = ArrayXt<Type>(1, L);
    for (int l = 0; l < L; ++l) {
      auto sigma_inv = e_sigmas[l].inverse();
      auto lsigma_det = log(e_sigmas[l].determinant());
      lprobs(1, l) = log(e_c.coeff(l)) + _log_probability_m<Type>(x, e_mus[l], 
          sigma_inv, lsigma_det);
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
  // TODO: change these to take in Base types, more generic
  template <typename Type>
  ArrayXt<Type> _log_probabilities_mm_fast(int L,
      const VectorXt<Type>& x, const NPArray<Type>& e_c,
      const vector<NPVector<Type> >& e_mus,
      const vector<MatrixXt<Type> >& sigma_invs,
      const vector<Type>& lsigma_dets) {

    auto lprobs = ArrayXt<Type>(1, L);
    for (int l = 0; l < L; ++l)
      lprobs(0, l) = log(e_c.coeff(l)) + _log_probability_m<Type>(x, e_mus[l], 
          sigma_invs[l], lsigma_dets[l]);

    return lprobs;
  }

  /*
   * Contant Paramters:
   *
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
   * Modified Paramters:
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
      const vector<NPArray<Type> >& e_cs,
      const vector<vector<NPVector<Type> > >& e_mus,
      const vector<vector<NPMatrix<Type> > >& e_sigmas,
      const NPArray<Type>& e_obs, const NPArray<Type>& e_lweights,

      vector<vector<MatrixXt<Type> > >& expected_x2,
      vector<vector<ArrayXt<Type> > >& expected_x,
      MatrixXt<Type>& expected_counts) {

    // precalculate inverses and log determinants
    vector<vector<MatrixXt<Type> > > sigma_invs;
    vector<vector<Type> > lsigma_dets;
    for (int s = 0; s < S; ++s) {
      sigma_invs.push_back(vector<MatrixXt<Type> >());
      lsigma_dets.push_back(vector<Type>());  
      for (int l = 0; l < L; ++l) {
        // TODO: better way to calculate inverse
        Type det = log(e_sigmas[s][l].determinant());
        MatrixXt<Type> inv = e_sigmas[s][l].inverse().eval();

        lsigma_dets[s].push_back(det);
        sigma_invs[s].push_back(inv);
      }
    }

    auto lprobs = ArrayXt<Type>(1, L).setZero();
    Type pmax;
    auto component_weight = ArrayXt<Type>(1, L).setZero();
    auto x2 = MatrixXt<Type>(D, D).setZero();

    for (int t = 0; t < T; ++t) {
      VectorXt<Type> ob = e_obs.row(t).matrix();
      x2 = ob*ob.transpose();
      for (int s = 0; s < S; ++s) {
        lprobs = _log_probabilities_mm_fast<Type>(L, ob, e_cs[s], e_mus[s], 
            sigma_invs[s], lsigma_dets[s]);

        pmax = lprobs.maxCoeff();
        component_weight = ((e_lweights.coeff(t, s) + lprobs) -
            (log((lprobs - pmax).exp().sum()) + pmax)).exp();

        expected_counts.row(s) += component_weight.matrix();
        for (int l = 0; l < L; ++l) {
          expected_x2[s][l] += component_weight.coeff(l)*x2;
          expected_x[s][l] += component_weight.coeff(l)*ob.array().transpose();
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
   * eigen.tuxfamily.org/dox/gropu__TutorialMapClass.html
   * eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html
   */
  template <typename Type>
  void update_parameters(int S, int T, int D, int L, Type* cs, Type* mus, 
                         Type* sigmas, Type* obs, Type* lweights) {
    // populate Eigen containers
    NPArray<Type> e_obs(obs, T, D);
    NPArray<Type> e_lweights(lweights, T, S);

    vector<NPArray<Type> > e_cs;
    for (int i1 = 0; i1 < S; ++i1)
      e_cs.push_back(NPArray<Type>(&cs[i1*L], 1, L));


    vector<vector<NPVector<Type> > > e_mus;
    for (int i1 = 0; i1 < S; ++i1) {
      e_mus.push_back(vector<NPVector<Type> >());
      for (int i2 = 0; i2 < L; ++i2)
        e_mus[i1].push_back(NPVector<Type>(&mus[(i1*L+i2)*D], D));
    }

    vector<vector<NPMatrix<Type> > > e_sigmas;
    for (int i1 = 0; i1 < S; ++i1) {
      e_sigmas.push_back(vector<NPMatrix<Type> >());
      for (int i2 = 0; i2 < L; ++i2)
        e_sigmas[i1].push_back(NPMatrix<Type>(&sigmas[(i1*L+i2)*D*D], D, D));
    }

    vector<vector<MatrixXt<Type> > > expected_x2;
    for (int i1 = 0; i1 < S; ++i1) {
      expected_x2.push_back(vector<MatrixXt<Type> >());
      for (int i2 = 0; i2 < L; ++i2)
        expected_x2[i1].push_back(MatrixXt<Type>(D, D).setZero());
    }

    vector<vector<ArrayXt<Type> > > expected_x;
    for (int i1 = 0; i1 < S; ++i1) {
      expected_x.push_back(vector<ArrayXt<Type> >());
      for (int i2 = 0; i2 < L; ++i2)
        expected_x[i1].push_back(ArrayXt<Type>(1, D).setZero());
    }

    auto expected_counts = MatrixXt<Type>(S, L).setZero();

    // DEBUG
    /*
    cout << "e_cs = " << endl;
    for (int i1 = 0; i1 < S; ++i1) 
      cout << e_cs[i1] << endl;
    
    cout << "e_mus =" << endl;
    for (int i1 = 0; i1 < S; ++i1)
      for (int i2 = 0; i2 < L; ++i2)
        cout << e_mus[i1][i2] << endl;

    std::cout << "e_sigmas =" << std::endl;
    for (int i1 = 0; i1 < S; ++i1)
      for (int i2 = 0; i2 < L; ++i2)
        cout << e_sigmas[i1][i2] << endl;
    */

    _weighted_sufficient_statistics(S, T, D, L, e_cs, e_mus, e_sigmas, e_obs,
        e_lweights, expected_x2, expected_x, expected_counts);

    /*
    cout << "expected_counts = " << endl;
    cout << expected_counts      << endl;

    cout << "expected_x2 = "     << endl;
    for (auto it = expected_x2.begin(); it != expected_x2.end(); ++it)
      for (auto itt = it->begin(); itt != it->end(); ++itt)
        cout << *itt << endl;

    cout << "expected_x = "      << endl;
    for (auto it = expected_x.begin(); it != expected_x.end(); ++it)
      for (auto itt = it->begin(); itt != it->end(); ++itt)
        cout << *itt << endl;
    */

    // update parameters
    for (int s = 0; s < S; ++s) {
      e_cs[s] = expected_counts.row(s)/expected_counts.row(s).sum();
      cout << "expected_counts = " << endl;
      cout << expected_counts << endl;
      for (int l = 0; l < L; ++l) {
        e_mus[s][l] = expected_x[s][l].transpose()/expected_counts.coeff(s, l);
        cout << "expected_x[" << s << "][" << l << "] = " << endl;
        cout << expected_x[s][l] << endl;
        e_sigmas[s][l] = (expected_x2[s][l] - expected_counts.coeff(s, l)*
            (e_mus[s][l]*e_mus[s][l].transpose()).matrix())/expected_counts.coeff(s, l);
      }
    }
  }

  /*
   * Calculates log likelihood for each GMM for each state.
   *
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
   * obs   - A T x D array representing the observations.
   * lliks - A T x S array that will hold the log likelihoods of each GMM for 
   *       each state
   *
   */
  template <typename Type>
  void log_likelihood(int S, int T, int D, int L, Type* cs, Type* mus, 
                      Type* sigmas, Type* obs, Type* lliks) {
    // duplicate code {
    // populate Eigen containers
    NPArray<Type> e_obs(obs, T, D);
    NPArray<Type> e_lliks(lliks, T, S);

    vector<NPArray<Type> > e_cs;
    for (int i1 = 0; i1 < S; ++i1)
      e_cs.push_back(NPArray<Type>(&cs[i1*L], 1, L));


    vector<vector<NPVector<Type> > > e_mus;
    for (int i1 = 0; i1 < S; ++i1) {
      e_mus.push_back(vector<NPVector<Type> >());
      for (int i2 = 0; i2 < L; ++i2)
        e_mus[i1].push_back(NPVector<Type>(&mus[(i1*L+i2)*D], D));
    }

    vector<vector<NPMatrix<Type> > > e_sigmas;
    for (int i1 = 0; i1 < S; ++i1) {
      e_sigmas.push_back(vector<NPMatrix<Type> >());
      for (int i2 = 0; i2 < L; ++i2)
        e_sigmas[i1].push_back(NPMatrix<Type>(&sigmas[(i1*L+i2)*D*D], D, D));
    }
    // }

    // duplicate code {
    vector<vector<MatrixXt<Type> > > sigma_invs;
    vector<vector<Type> > lsigma_dets;
    for (int s = 0; s < S; ++s) {
      sigma_invs.push_back(vector<MatrixXt<Type> >());
      lsigma_dets.push_back(vector<Type>());
      for (int l = 0; l < L; ++l) {
        // TODO: better way to calculate invervse
        Type det = log(e_sigmas[s][l].determinant());
        MatrixXt<Type> inv = e_sigmas[s][l].inverse().eval();

        lsigma_dets[s].push_back(det);
        sigma_invs[s].push_back(inv);
      }
    }
    // }

    auto lprobs = ArrayXt<Type>(1, L).setZero();

    for (int t = 0; t < T; ++t) {
      VectorXt<Type> ob = e_obs.row(t).matrix();
      for (int s = 0; s < S; ++s) {
        lprobs = _log_probabilities_mm_fast<Type>(L, ob, e_cs[s], e_mus[s],
            sigma_invs[s], lsigma_dets[s]);
        e_lliks(t, s) = lprobs.sum();
      }
    }
  }
}


#endif
