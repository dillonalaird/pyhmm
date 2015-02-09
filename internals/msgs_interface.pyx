import numpy as np
cimport numpy as np

from cython cimport floating

cdef extern from "forward_backward_msgs.h":
    cdef cppclass hmmc[Type]:
        hmmc()
        void forward_msgs() nogil

def forward_msgs():
    cdef hmmc[floating] ref
    ref.forward_msgs()
