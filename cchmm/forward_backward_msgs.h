#ifndef FORWARD_BACKWARD_MSGS_H
#define FORWARD_BACKWARD_MSGS_H

#include <Eigen/Core>
#include "np_types.h"
#include "eigen_types.h"


namespace fb {
    using namespace std;
    using namespace Eigen;
    using namespace nptypes;
    using namespace eigentypes;

    template <typename Type>
    ArrayXt<Type> forward_msgs(const ArrayXt<Type>& lliks,
                               const VectorXt<Type>& pi,
                               const MatrixXt<Type>& A) {
        ArrayXt<Type> lalpha = ArrayXt<Type>::Zero(lliks.rows(), lliks.cols());
        Type cmax;
        lalpha.row(0) = pi.array().log() + lliks.row(0);

        for (int t = 0; t < lliks.rows() - 1; ++t) {
            cmax = lalpha.row(t).maxCoeff();
            lalpha.row(t+1) = ((lalpha.row(t) - cmax).exp().matrix()*
                    A).array().log() + cmax + lliks.row(t+1);
        }

        return lalpha;
    }

    template <typename Type>
    ArrayXt<Type> backward_msgs(const ArrayXt<Type>& lliks,
                               const MatrixXt<Type>& A) {
        ArrayXt<Type> lbeta = ArrayXt<Type>::Zero(lliks.rows(), lliks.cols());

        Type thesum_buf[lliks.cols()] __attribute__((aligned(16)));
        NPVector<Type> thesum(thesum_buf, lliks.cols());
        Type cmax;
        for (int t = lliks.rows() - 2; t >= 0; --t) {
            thesum = (lliks.row(t+1) + lbeta.row(t+1)).transpose();
            cmax = thesum.maxCoeff();
            lbeta.row(t) = (A*(thesum.array() - 
                    cmax).exp().matrix()).array().log() + cmax;
        }

        return lbeta;
    }
}


#endif
