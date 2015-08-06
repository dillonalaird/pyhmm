cdef extern from "hmmsvi.h" namespace "hmmsvi" nogil:
    void infer[Type](int D, int S, int T, Type* obs, Type* A_0, Type* emits, Type tau, Type kappa, int L, int n, int itr)
