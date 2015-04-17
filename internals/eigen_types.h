#ifndef EIGEN_TYPES_H
#define EIGEN_TYPES_H

#ifndef EIGEN_H
#include <Eigen/Core>
#endif

namespace eigentypes {
  using namespace Eigen;

  template <typename T>
  using MatrixXt = Matrix<T, Dynamic, Dynamic, RowMajor>;

  template <typename T>
  using ArrayXt = Array<T, Dynamic, Dynamic, RowMajor>;

  template <typename T>
  using VectorXt = Matrix<T, Dynamic, 1>;
}

#endif
