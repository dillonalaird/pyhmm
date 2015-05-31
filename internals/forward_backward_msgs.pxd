cdef extern from "forward_backward_msgs.h" namespace "fb" nogil:
    void forward_msgs[Type](int S, int T, Type* pi, Type* A, Type* lliks, Type* lalpha)
    void backward_msgs[Type](int S, int T, Type* A, Type* lliks, Type* lbeta)
    void log_weights[Type](int S, int T, Type* lalpha, Type* lbeta, Type* lweights)
    void expected_statistics[Type](int S, int T, Type* pi, Type* A, Type* lliks, Type* lalpha, Type* lbeta, Type* lexpected_states, Type* expected_transcounts)
