cdef extern from "gmm.h" namespace "gmm" nogil:
    void update_parameters[Type](Type* cs, Type* mus, Type* sigmas, Type* lexpected_states)
