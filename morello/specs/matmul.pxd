from . cimport base
from .tensorspec cimport TensorSpec

cdef class Matmul(base.Spec):
    cdef readonly TensorSpec lhs
    cdef readonly TensorSpec rhs
    cdef readonly TensorSpec _output
    cdef readonly bint _serial_only
