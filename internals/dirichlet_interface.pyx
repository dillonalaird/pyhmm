import numpy as np
cimport numpy as np

cimport internals.dirichlet as dir


def expected_sufficient_statistics(np.ndarray[np.double_t, ndim=2, mode='c'] alphas not None):
    cdef int S = alphas.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] ess = np.zeros_like(alphas)
    dir.expected_sufficient_statistics[np.double_t](S, &alphas[0,0], &ess[0,0])
    return ess

def meanfield_update(np.ndarray[np.double_t, ndim=2, mode='c'] nat_params not None,
                     np.ndarray[np.double_t, ndim=2, mode='c'] ss not None):
    cdef int S = nat_params.shape[0]
    dir.meanfield_update(S, &nat_params[0,0], &ss[0,0])
    return nat_params;
