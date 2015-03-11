cdef extern from "forward_backward_msgs.h" namespace "fb" nogil:
    void forward_msgs[Type](int M, int T, Type* pi, Type* A, Type* lliks, Type* lalpha)
    void backward_msgs[Type](int M, int T, Type* A, Type* lliks, Type* lbeta)
    void expected_statistics[Type](int M, int T, Type* pi, Type* A, Type* lliks, Type* lalpha, Type* lbeta, Type* lexpected_states, Type* expected_transcounts)
