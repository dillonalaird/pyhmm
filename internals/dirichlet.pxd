cdef extern from "dirichlet.h" namespace "dir" nogil:
    void expected_sufficient_statistics[Type](int S, Type* alphas, Type* ess)
