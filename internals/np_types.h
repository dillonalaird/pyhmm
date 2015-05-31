#ifndef NP_TYPES_H
#define NP_TYPES_H

#ifndef EIGEN_H
#include <Eigen/Core>
#endif


namespace nptypes {
  using namespace Eigen;

  template <typename T>
  using NPMatrix = Map<Matrix<T, Dynamic, Dynamic, RowMajor> >;

  template <typename T>
  using NPArray = Map<Array<T, Dynamic, Dynamic, RowMajor> >;

  template <typename T>
  using NPVector = Map<Matrix<T, Dynamic, 1> >;
}


#endif
