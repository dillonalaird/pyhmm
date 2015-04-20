//#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "np_types.h"
#include "eigen_types.h"

using namespace std;
using namespace Eigen;
using namespace nptypes;
using namespace eigentypes;


double log2pi = 1.83787706640934533908193770912475883;


template <typename Type, typename Derived>
void test(const ArrayBase<Derived>& x, const VectorXt<Type> e_mu,
    const MatrixXt<Type> sigma_inv, const Type lsigma_det) {

  std::cout << "x =" << std::endl;
  std::cout << x << std::endl;

  std::cout << "e_mu =" << std::endl;
  std::cout << e_mu.array().transpose() << std::endl;

  std::cout << "diff =" << std::endl;
  auto diff = (x - e_mu.array().transpose()).matrix().transpose();
  std::cout << diff << std::endl;

  std::cout << "descriptive_stat =" << std::endl;
  auto descriptive_stat = (diff.transpose()*sigma_inv*diff).coeff(0);
  std::cout << descriptive_stat << std::endl;

  std::cout << "final probability" << std::endl;
  auto prob = -0.5*(lsigma_det + descriptive_stat + e_mu.rows()*log2pi);
  std::cout << prob << std::endl;
}


template <typename Type>
void test2(int S, int L, Type* cs) {
  auto tmp0 = NPArray<Type>(&cs[0*L], 1, L);
  std::cout << tmp0 << std::endl;

  auto tmp2 = NPArray<Type>(&cs[2*L], 1, L);
  std::cout << tmp2 << std::endl;

  auto tmp1 = NPArray<Type>(&cs[1*L], 1, L);
  std::cout << tmp1.coeff(0,0) << std::endl;
  std::cout << tmp1.coeff(0,1) << std::endl;
  std::cout << tmp1.coeff(0,2) << std::endl;

  std::cout << tmp1 << std::endl;
}


int main() {
  /*
   * x = {1,2,3}
   * e_mu = {3,3,3}
   *
   * sigma = {{1,0,0},
   *          {0,1,0},
   *          {0,0,1}}
   * 
   * sigma_inv = {{1,0,0},
   *              {0,1,0},
   *              {0,0,1}}
   *
   * sigma_det = 1
   * lsigma_det = 0
   */

  /*
  int D = 3;
  auto x = ArrayXt<double>(1, D);
  x << 1,2,3;
  auto e_mu = VectorXt<double>(D);
  e_mu << 3,3,3;
  auto e_sigma = MatrixXt<double>(D, D);
  e_sigma << 1,0,0,
             0,1,0,
             0,0,1;

  std::cout << "e_sigma_inv ="  << std::endl;
  auto e_sigma_inv = e_sigma.inverse();
  std::cout << e_sigma_inv << std::endl;

  std::cout << "sigma_det =" << std::endl;
  auto sigma_det   = e_sigma.determinant();
  std::cout << sigma_det << std::endl;

  std::cout << "lsigma_det =" << std::endl;
  auto lsigma_det  = log(sigma_det);
  std::cout << lsigma_det << std::endl;

  test<double>(x, e_mu, e_sigma_inv, lsigma_det);
  */


  int S = 3;
  int L = 3;
  double cs[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  test2<double>(S, L, cs);
}
