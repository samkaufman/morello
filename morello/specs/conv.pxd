from . cimport base
from .tensorspec cimport TensorSpec

cdef class Convolution(base.Spec):
    cdef TensorSpec lhs
    cdef TensorSpec rhs
    cdef TensorSpec _output
    cdef bint _serial_only
