import numpy as np
cimport numpy as np

cimport cchmm.hmmsvi as hmmsvi


def convert_emits(S, D, emits):
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] _emits = np.zeros(S*(D*D + 3*D))

    cdef int s = 0
    cdef int offset1 = D*D + 3*D
    cdef int offset2 = D + 3
    for s in range(S):
        _emits[(s*offset1):(s*offset1 + D*D)] = \
                emits[(s*offset2):(s*offset2 + D)].flatten().copy(order='C')
        _emits[(s*offset1 + D*D):(s*offset1 + D*D + D)] = \
                emits[(s*offset2 + D),:].flatten().copy(order='C')
        _emits[(s*offset1 + D*D + D)] = emits[(s*offset2 + D + 1),0]
        _emits[(s*offset1 + D*D + 2*D)] = emits[(s*offset2 + D + 2),0]

    return _emits

def infer(np.ndarray[np.double_t, ndim=2, mode='c'] obs,
          np.ndarray[np.double_t, ndim=2, mode='c'] A_0,
          np.ndarray[np.double_t, ndim=2, mode='c'] emits_0,
          np.double_t tau, np.double_t kappa,
          int L, int n, int itr):
    cdef int D = obs.shape[1]
    cdef int T = obs.shape[0]
    cdef int S = A_0.shape[0]

    cdef np.ndarray[np.double_t, ndim=2, mode='c'] A_N = A_0.copy(order='C')
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] _emits_0 = \
            convert_emits(S, D, emits_0)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] _emits_N = \
            convert_emits(S, D, emits_0)

    hmmsvi.infer[np.double_t](D, S, T, &obs[0,0], &A_0[0,0], &A_N[0,0],
                              &_emits_0[0], &_emits_N[0], tau, kappa, L, n, itr)
