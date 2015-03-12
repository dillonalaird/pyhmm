import numpy as np
cimport numpy as np

cimport internals.gmm as gmm

def update_parameters_gmm(np.ndarray[np.double_t, ndim=1, mode='c'] cs,
                          np.ndarray[np.double_t, ndim=2, mode='c'] mus,
                          np.ndarray[np.double_t, ndim=2, mode='c'] sigmas,
                          np.double_t expected_obs2,
                          np.double_t expected_obs,
                          np.ndarray[np.double_t, ndim=2, mode='c'] lweights):
    gmm.update_parameters[np.double_t](mus[0].shape[0], lweights.shape[0],
            cs.shape[0], &cs[0], &mus[0,0], &sigmas[0,0], expected_obs2,
            expected_obs, &lweights[0,0])
    return (cs, mus, sigmas)
