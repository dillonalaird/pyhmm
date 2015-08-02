import numpy as np
cimport numpy as np

cimport internals.dirichlet as dir


def expected_sufficient_statistics(np.ndarray[np.double_t, ndim=2, mode='c'] alphas not None):
    cdef int S = alphas.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] ess = np.zeros_like(alphas)
    dir.expected_sufficient_statistics[np.double_t](S, &alphas[0,0], &ess[0,0])
    return ess


def sufficient_statistics(np.ndarray[np.double_t, ndim=2, mode='c'] expected_states not None):
    cdef int N = expected_states.shape[0]
    cdef int S = expected_states.shape[1]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] ss = np.zeros((S, S))
    dir.sufficient_statistics[np.double_t](S, N, &expected_states[0,0],
                                           &ss[0,0])
    return ss


def meanfield_update(np.ndarray[np.double_t, ndim=2, mode='c'] nat_params not None,
                     np.ndarray[np.double_t, ndim=2, mode='c'] ss not None):
    cdef int S = nat_params.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] _nat_params = nat_params.copy()
    dir.meanfield_update[np.double_t](S, &_nat_params[0,0], &ss[0,0])
    return _nat_params


def meanfield_sgd_update(np.ndarray[np.double_t, ndim=2, mode='c'] nat_params_0 not None,
                         np.ndarray[np.double_t, ndim=2, mode='c'] nat_params_N not None,
                         np.ndarray[np.double_t, ndim=2, mode='c'] ess not None,
                         np.double_t lrate, np.double_t bfactor):
    cdef int S = nat_params_0.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] _nat_params_N = nat_params_N.copy()
    dir.meanfield_sgd_update[np.double_t](S, lrate, bfactor, &nat_params_0[0,0],
                                          &_nat_params_N[0,0], &ess[0,0])
    return _nat_params_N
