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

def log_weights(np.ndarray[np.double_t, ndim=2, mode='c'] lalpha not None,
                np.ndarray[np.double_t, ndim=2, mode='c'] lbeta not None):
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] lweights = np.zeros_like(lalpha)
    fb_msgs.log_weights[np.double_t](lalpha.shape[1], lalpha.shape[0], &lalpha[0,0],
            &lbeta[0,0], &lweights[0,0])
    return lweights

def expected_statistics(np.ndarray[np.double_t, ndim=1, mode='c'] pi not None,
                        np.ndarray[np.double_t, ndim=2, mode='c'] A not None,
                        np.ndarray[np.double_t, ndim=2, mode='c'] lliks not None,
                        np.ndarray[np.double_t, ndim=2, mode='c'] lalpha not None,
                        np.ndarray[np.double_t, ndim=2, mode='c'] lbeta not None):
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] \
            lexpected_states = np.zeros_like(lliks)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] \
            expected_transcounts = np.zeros_like(A)
    fb_msgs.expected_statistics[np.double_t](A.shape[0], lliks.shape[0], &pi[0],
            &A[0,0], &lliks[0,0], &lalpha[0,0], &lbeta[0,0], 
            &lexpected_states[0,0], &expected_transcounts[0,0])
    return lexpected_states, expected_transcounts
