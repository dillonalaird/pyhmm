import numpy as np
cimport numpy as np

cimport internals.gaussian_niw as gaussian_niw

def meanfield_update(np.ndarray[np.double_t, ndim=1, mode='c'] u1 not None,
                     np.double_t u2,
                     np.ndarray[np.double_t, ndim=2, mode='c'] u3 not None,
                     np.double_t u4,
                     np.ndarray[np.double_t, ndim=2, mode='c'] s1 not None,
                     np.ndarray[np.double_t, ndim=1, mode='c'] s2 not None,
                     np.double_t s3):
    cdef int D = u1.shape[0]
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] nat_params = np.zeros((D+2,D+2))
    nat_params[:D,:D] = u3.copy()
    nat_params[:D,-2] = u1.copy()
    nat_params[-2,-2] = u2.copy()
    nat_params[-1,-1] = u4.copy()
    gaussian_niw.meanfield_update[np.double_t](u1.shape[0], &nat_params[0,0],
            &s1[0,0], &s2[0], s3)
    return nat_params[:D,:D], nat_params[:D,-2], nat_params[-2,-2], nat_params[-1,-1]
