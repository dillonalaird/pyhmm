cdef extern from "dirichlet.h" namespace "dir" nogil:
    void expected_sufficient_statistics[Type](int S, Type* alphas, Type* ess)
    void meanfield_update[Type](int S, Type* nat_params, Type* ss)
