#ifndef FORWARD_BACKWARD_MSGS_H
#define FORWARD_BACKWARD_MSGS_H

#include <iostream>
#include <Eigen/Core>
#include <limits>

#include "np_types.h"


namespace fb {
  using namespace std;
  using namespace Eigen;
  using namespace nptypes;

  template <typename Type>
  void forward_msgs(int M, int T, Type* pi, Type* ltran, Type* lliks, Type* lalpha) {
  }
}


#endif
