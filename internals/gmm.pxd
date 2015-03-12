cdef extern from "gmm.h" namespace "gmm" nogil:
    void update_parameters[Type](int D, int T, int L, Type* cs, Type* mus, Type* sigmas, Type expected_obs2, Type expected_obs, Type* lweights)
