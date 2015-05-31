#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>

#include "np_types.h"
#include "eigen_types.h"


using namespace std;
using namespace nptypes;
using namespace eigentypes;

static const constexpr double log2pi = 1.83787706640934533908193770912475883;

template <typename Type>
Type _log_probability_m(const VectorXt<Type>& x, const VectorXt<Type>& mu,
    const MatrixXt<Type>& sigma_inv, const Type lsigma_det) {
  VectorXt<Type> diff = (x - mu);
  Type descriptive_stat = (diff.transpose()*sigma_inv*diff);
  return -0.5*(lsigma_det + descriptive_stat + mu.rows()*log2pi);
}


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
  auto component_weight = ArrayXt<Type>(1, L).setZero();
  auto x2 = MatrixXt<Type>(D, D).setZero();

  for (int t = 0; t < T; ++t) {
    VectorXt<Type> ob = e_obs.row(t).matrix();
    x2 = ob*ob.transpose();
    for (int s = 0; s < S; ++s) {
      lprobs = _log_probabilities_mm_fast<Type>(L, ob, e_cs[s], e_mus[s], 
          sigma_invs[s], lsigma_dets[s]);
      cout << "lprobs = " << lprobs << endl;
      component_weight = (exp(e_lweights.coeff(t, s))*lprobs.exp())/lprobs.exp().sum();
      cout << "component_weight = " << component_weight << endl;

      expected_counts.row(s) += component_weight.matrix();
      for (int l = 0; l < L; ++l) {
        expected_x2[s][l] += component_weight.coeff(l)*x2;
        expected_x[s][l] += component_weight.coeff(l)*ob.array().transpose();
      }
    }
  }
}


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
  cout << "before cs = " << endl;
  for (auto it = e_cs.begin(); it != e_cs.end(); ++it)
    cout << *it << endl;

  cout << "before mus = " << endl;
  for (auto it1 = e_mus.begin(); it1 != e_mus.end(); ++it1)
    for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
      cout << (*it2).transpose() << endl;

  cout << "before sigmas = " << endl;
  for (auto it1 = e_sigmas.begin(); it1 != e_sigmas.end(); ++it1)
    for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
      cout << *it2 << endl;

  _weighted_sufficient_statistics(S, T, D, L, e_cs, e_mus, e_sigmas, e_obs,
      e_lweights, expected_x2, expected_x, expected_counts);

  for (int s = 0; s < S; ++s) {
    e_cs[s] = expected_counts.row(s)/expected_counts.row(s).sum();
    cout << "expected_counts = " << expected_counts << endl;
    for (int l = 0; l < L; ++l) {
      cout << "expected_x2[s][l] = " << expected_x2[s][l] << endl;
      cout << "expected_x[s][l] = " << expected_x[s][l] << endl;
      e_mus[s][l] = expected_x[s][l].transpose()/expected_counts.coeff(s, l);
      e_sigmas[s][l] = (expected_x2[s][l] - expected_counts.coeff(s, l)*
          (e_mus[s][l]*e_mus[s][l].transpose()))/expected_counts.coeff(s, l);
    }
  }
}


int main() {
  int T = 10;
  int S = 2;
  int D = 3;
  int L = 2;
  double obs[10*3] = {1, 1, 1,  1, 1, 1,
                      1, 1, 1,  1, 1, 1,
                      1, 1, 1,  1, 1, 1,
                      1, 1, 1,  1, 1, 1,
                      1, 1, 1,  1, 1, 1};

  double log_hlf = log(0.5);
  double lweights[10*2] = {log_hlf, log_hlf,  log_hlf, log_hlf,
                           log_hlf, log_hlf,  log_hlf, log_hlf,
                           log_hlf, log_hlf,  log_hlf, log_hlf,
                           log_hlf, log_hlf,  log_hlf, log_hlf,
                           log_hlf, log_hlf,  log_hlf, log_hlf};

  double cs[2*2] = {0.5, 0.5,
                    0.5, 0.5};

  double mus[2*2*3] = {1, 1, 1,  1, 1, 1,
                       1, 1, 1,  1, 1, 1};

  double sigmas[2*2*3*3] = {1, 0, 0,
                            0, 1, 0,
                            0, 0, 1,

                            1, 0, 0,
                            0, 1, 0,
                            0, 0, 1,

                            1, 0, 0,
                            0, 1, 0,
                            0, 0, 1,

                            1, 0, 0,
                            0, 1, 0,
                            0, 0, 1};
                    
  update_parameters<double>(S, T, D, L, cs, mus, sigmas, obs, lweights);

  vector<NPArray<double> > e_cs;
  for (int i1 = 0; i1 < S; ++i1)
    e_cs.push_back(NPArray<double>(&cs[i1*L], 1, L));


  vector<vector<NPVector<double> > > e_mus;
  for (int i1 = 0; i1 < S; ++i1) {
    e_mus.push_back(vector<NPVector<double> >());
    for (int i2 = 0; i2 < L; ++i2)
      e_mus[i1].push_back(NPVector<double>(&mus[(i1*L+i2)*D], D));
  }

  vector<vector<NPMatrix<double> > > e_sigmas;
  for (int i1 = 0; i1 < S; ++i1) {
    e_sigmas.push_back(vector<NPMatrix<double> >());
    for (int i2 = 0; i2 < L; ++i2)
      e_sigmas[i1].push_back(NPMatrix<double>(&sigmas[(i1*L+i2)*D*D], D, D));
  }

  cout << "after cs = " << endl;
  for (auto it = e_cs.begin(); it != e_cs.end(); ++it)
    cout << *it << endl;

  cout << "after mus = " << endl;
  for (auto it1 = e_mus.begin(); it1 != e_mus.end(); ++it1)
    for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
      cout << (*it2).transpose() << endl;

  cout << "after sigmas = " << endl;
  for (auto it1 = e_sigmas.begin(); it1 != e_sigmas.end(); ++it1)
    for (auto it2 = it1->begin(); it2 != it1->end(); ++it2)
      cout << *it2 << endl;
}
