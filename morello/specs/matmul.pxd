from . cimport base
from .tensorspec cimport TensorSpec

cdef class MatmulBase(base.Spec):
    cdef readonly TensorSpec lhs
    cdef readonly TensorSpec rhs
    cdef readonly TensorSpec _output
    cdef readonly bint _serial_only

cdef class Matmul(Matmul):
    pass

cdef class MatmulAccum(Matmul):
    pass