import numpy as np
cimport numpy as np

cimport internals.dirichlet as dir


def expected_sufficient_statistics(np.ndarray[np.double_t, ndim=2, mode='c'] alphas not None):
    cdef int S = alphas.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] ess = np.zeros_like(alphas)
    dir.expected_sufficient_statistics[np.double_t](S, &alphas[0,0], &ess[0,0])
    return ess
