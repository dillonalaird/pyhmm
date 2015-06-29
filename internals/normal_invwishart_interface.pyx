import numpy as np
cimport numpy as np

cimport internals.normal_invwishart as niw

def meanfield_update(np.ndarray[np.double_t, ndim=1, mode='c'] n1 not None,
                     np.double_t n2,
                     np.ndarray[np.double_t, ndim=2, mode='c'] n3 not None,
                     np.double_t n4,
                     np.double_t s1,
                     np.ndarray[np.double_t, ndim=1, mode='c'] s2 not None,
                     np.ndarray[np.double_t, ndim=2, mode='c'] s3 not None):
    cdef int D = n1.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] nat_params = np.zeros((D+3,D))
    nat_params[:D,:D] = n3.copy()
    nat_params[D,:] = n1.copy()
    nat_params[D+1,0] = n2
    nat_params[D+2,0] = n4
    niw.meanfield_update[np.double_t](D, &nat_params[0,0], s1, &s2[0], &s3[0,0])
    return nat_params[D,:], nat_params[D+1,0], nat_params[:D,:D], nat_params[D+2,0]

def responsibilities(np.ndarray[np.double_t, ndim=2, mode='c'] obs not None,
                     np.ndarray[np.double_t, ndim=1, mode='c'] mu_0 not None,
                     np.ndarray[np.double_t, ndim=2, mode='c'] sigma_0 not None,
                     np.double_t kappa_0,
                     np.double_t nu_0):
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] rs = np.zeros_like(obs)
    niw.responsibilities[np.double_t](obs.shape[0], obs.shape[1], &obs[0,0],
                                      &mu_0[0], &sigma_0[0,0], kappa_0, nu_0,
                                      &rs[0,0])
    return rs

