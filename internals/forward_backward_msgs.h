#ifndef FORWARD_BACKWARD_MSGS_H
#define FORWARD_BACKWARD_MSGS_H

#include <Eigen/Core>
#include <stdint.h>
#include <limits>

//#include "nptypes.h"

namespace hmm {
    using namespace std;
    using namespace Eigen;
    //using namespace nptypes;

    //template <typename Type>
    void forward_msgs() {}
}

//template <typename FloatType, typename IntType = int32_t>
class hmmcc {
    public:
        static void forward_msgs() { hmm::forward_msgs(); }
};

#endif
