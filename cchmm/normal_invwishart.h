#ifndef normal_invwishart_h
#define normal_invwishart_h

#include <Eigen/Core>


namespace niw {
    template <typename T>
    struct nat_params {
        int D;
        T* sigma_N;
        T* mu_N;
        T* kappa_N;
        T* nu_N;
    };

    template <typename T>
    nat_params<T> convert_to_struct(T* nat_params_t, int D, int s) {
        int offset = (D*D + 3*D);
        return nat_params<T>{D,
                             &nat_params_t[s*offset],
                             &nat_params_t[s*offset + D*D],
                             &nat_params_t[s*offset + D*D + D],
                             &nat_params_t[s*offset + D*D + 2*D]};
    }

}


#endif
