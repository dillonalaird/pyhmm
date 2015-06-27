cdef extern from "gaussian_wishart.h" namespace "gaussian_wishart" nogil:
    void meanfield_update[Type](int D, Type* nat_params, Type s1, Type* s2, Type* s3)
    void responsibilities[Type](Type* obs, Type* mu_0, Type* sigma_0, Type kappa_0, Type nu_0, Type* rs)
