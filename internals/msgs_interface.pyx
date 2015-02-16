# distutils: name = internals.msgs_interface
# distutils: language = c++
# distutils: extra_compile_args = -Ofast -std=c++11
# cython: boundscheck = False

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

