import cython

from . cimport base
from .tensorspec cimport TensorSpec

cdef class ReduceSum(base.Spec):
    cdef readonly TensorSpec source
    cdef readonly TensorSpec _output
    cdef bint _serial_only
