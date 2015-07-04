cdef extern from "normal_invwishart.h" namespace "niw" nogil:
    void meanfield_update[Type](int D, Type* nat_params, Type s1, Type* s2, Type* s3)
    void expected_log_likelihood[Type](int N, int D, Type* obs, Type* mu_0, Type* sigma_0, Type kappa_0, Type nu_0, Type* rs)
    void log_likelihood[Type](int N, int D, Type* obs, Type* mu, Type* sigma, Type* lliks)
