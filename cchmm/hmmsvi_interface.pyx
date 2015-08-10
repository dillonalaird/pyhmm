import numpy as np
cimport numpy as np

cimport cchmm.hmmsvi as hmmsvi

def infer(np.ndarray[np.double_t, ndim=2, mode='c'] obs,
          np.ndarray[np.double_t, ndim=2, mode='c'] A_0,
          np.ndarray[np.double_t, ndim=2, mode='c'] emits,
          np.double_t tau, np.double_t kappa,
          int L, int n, int itr):
    cdef int D = obs.shape[1]
    cdef int T = obs.shape[0]
    cdef int S = A_0.shape[0]
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] _emits = np.zeros(S*(D*D + 3*D))

    cdef int s = 0
    cdef int offset = D*D + 3*D
    for s in range(S):
        _emits[(s*offset):(s*offset + D*D)] = emits[:D,:D].flatten().copy(order='C')
        _emits[(s*offset + D*D):(s*offset + D*D + D)] = emits[D,:].flatten().copy(order='C')
        _emits[(s*offset + D*D + 2*D)] = emits[D+1,0]
        _emits[(s*offset + D*D + 3*D)] = emits[D+2,0]
    hmmsvi.infer[np.double_t](D, S, T, &obs[0,0], &A_0[0,0], &emits[0,0],
                              tau, kappa, L, n, itr)
