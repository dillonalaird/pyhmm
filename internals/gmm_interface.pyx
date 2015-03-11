import numpy as np
cimport numpy as np

cimport internals.gmm as gmm

def update_parameters_gmm(np.ndarray[np.double_t, ndim=1, mode='c'] cs,
                          np.ndarray[np.double_t, ndim=2, mode='c'] mus,
                          np.ndarray[np.double_t, ndim=2, mode='c'] sigmas,
                          np.ndarray[np.double_t, ndim=2, mode='c'] lexpected_states):
    gmm.update_parameters[np.double_t](&cs[0], &mus[0,0], &sigmas[0,0], 
            &lexpected_states[0,0])
    return (cs, mus, sigmas)
