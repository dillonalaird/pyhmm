import numpy as np
cimport numpy as np

cimport internals.forward_backward_msgs as fb_msgs


# can't figure out how to use cython floating here
def forward_msgs(np.ndarray[np.double_t, ndim=1, mode='c'] pi not None,
                 np.ndarray[np.double_t, ndim=2, mode='c'] A not None,
                 np.ndarray[np.double_t, ndim=2, mode='c'] lliks not None):
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] lalpha = np.zeros_like(lliks)
    fb_msgs.forward_msgs[np.double_t](A.shape[0], lliks.shape[0], &pi[0],
            &A[0,0], &lliks[0,0], &lalpha[0,0])
    return lalpha


def backward_msgs(np.ndarray[np.double_t, ndim=2, mode='c'] A not None,
                  np.ndarray[np.double_t, ndim=2, mode='c'] lliks not None):
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] lbeta = np.zeros_like(lliks)
    fb_msgs.backward_msgs[np.double_t](A.shape[0], lliks.shape[0], 
            &A[0,0], &lliks[0,0], &lbeta[0,0])
    return lbeta
