cdef extern from "dirichlet.h" namespace "dir" nogil:
    void expected_sufficient_statistics[Type](int S, Type* alphas, Type* ess)
    void meanfield_update[Type](int S, Type* nat_params_0, Type* ss)
    void meanfield_sgd_update[Type](int S, Type lrate, Type bfactor, Type* nat_params_0, Type* nat_params_N, Type* ess)
    void sufficient_statistics[Type](int S, int N, Type* expected_states, Type* ss)
