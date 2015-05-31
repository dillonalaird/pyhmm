cdef extern from "gaussian_niw.h" namespace "gaussian_niw" nogil:
    void meanfield_update[Type](int D, Type* nat_params, Type* s2, Type* s2, Type s3)
