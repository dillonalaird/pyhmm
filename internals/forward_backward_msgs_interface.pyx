import numpy as np
cimport numpy as np

cimport internals.forward_backward_msgs as fb_msgs


# can't figure out how to use cython floating here
def forward_msgs():
    fb_msgs.forward_msgs[np.double_t]()
