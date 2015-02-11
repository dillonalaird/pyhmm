import numpy as np
cimport numpy as np

#from cython cimport floating

cdef extern from "forward_backward_msgs.h":
    cdef cppclass hmmcc:
        hmmcc()
        void forward_msgs()

def forward_msgs():
    cdef hmmcc ref
    ref.forward_msgs()
