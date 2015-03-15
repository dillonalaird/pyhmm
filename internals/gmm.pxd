cdef extern from "gmm.h" namespace "gmm" nogil:
    void update_parameters[Type](int S, int T, int D, int L, Type* cs, Type* mus, Type* sigmas, Type* obs, Type* lweights)
