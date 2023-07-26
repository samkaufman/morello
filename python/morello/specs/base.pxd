from .tensorspec cimport TensorSpec

cdef class Spec:

    cpdef Spec replace_io(
        self,
        tuple inputs,
        TensorSpec output,
        serial_only=*
    )
