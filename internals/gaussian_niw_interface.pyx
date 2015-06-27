import numpy as np
cimport numpy as np

cimport internals.gaussian_niw as gaussian_niw

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
    gaussian_niw.meanfield_update[np.double_t](D, &nat_params[0,0],
                                               s1, &s2[0], &s3[0,0])
    return nat_params[D,:], nat_params[D+1,0], nat_params[:D,:D], nat_params[D+2,0]
