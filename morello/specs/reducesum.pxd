import cython

from . cimport base
from .tensorspec cimport TensorSpec

@cython.dataclasses.dataclass(frozen=True)
cdef class ReduceSum(base.Spec):
    cdef TensorSpec source
    cdef TensorSpec output
    cdef bint serial_only
