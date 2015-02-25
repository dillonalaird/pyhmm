cdef extern from "forward_backward_msgs.h" namespace "fb" nogil:
    void forward_msgs[Type](int D, int T, Type* pi, Type* A, Type* lliks, Type* lalpha)
    void backward_msgs[Type](int D, int T, Type* A, Type* lliks, Type* lbeta)
