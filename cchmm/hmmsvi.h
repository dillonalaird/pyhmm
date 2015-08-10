#ifndef hmmsvi_h
#define hmmsvi_h

#include <Eigen/Core>
#include "np_types.h"
#include "normal_invwishart.h"


namespace hmmsvi {
    using namespace Eigen;
    using namespace nptypes;

    template <typename Type>
    void infer(int D, int S, int T, Type* obs, Type* A_0, Type* emits,
               Type tau, Type kappa, int L, int n, int itr) {
        NPArray<Type> e_obs(obs, T, S);
        NPMatrix<Type> A_nat_0(A_0, S, S);
        //A_nat_0 -= 1;

        std::vector<niw::nat_params<Type> > nat_params;
        for (int s = 0; s < S; ++s)
            nat_params.push_back(niw::convert_to_struct(emits, D, s));
    }

    void local_update() { }

    void global_update() { }

    void intermediate_pars() { }

    void _calc_pi() { }
}


#endif
