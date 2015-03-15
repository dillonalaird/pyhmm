import numpy as np
cimport numpy as np

cimport internals.gmm as gmm

def update_parameters_gmm(np.ndarray[np.double_t, ndim=2, mode='c'] cs,
                          np.ndarray[np.double_t, ndim=3, mode='c'] mus,
                          np.ndarray[np.double_t, ndim=4, mode='c'] sigmas,
                          np.ndarray[np.double_t, ndim=2, mode='c'] obs,
                          np.ndarray[np.double_t, ndim=2, mode='c'] lweights):
    gmm.update_parameters[np.double_t](cs.shape[0], obs.shape[0], 
            mus[0,0].shape[0], mus[0].shape[0], &cs[0,0], &mus[0,0,0],
            &sigmas[0,0,0,0], &obs[0,0], &lweights[0,0])
    return (cs, mus, sigmas)