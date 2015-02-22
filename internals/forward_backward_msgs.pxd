cdef extern from "forward_backward_msgs.h" namespace "fb" nogil:
    void forward_msgs[Type](int D, int T, Type* pi, Type* ltran, Type* lliks, Type* lalpha)
    void backward_msgs[Type](int D, int T, Type* ltran, Type* lliks, Type* lbeta)
