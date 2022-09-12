import cython

from . cimport base
from .tensorspec cimport TensorSpec

cdef class ReduceSumBase(base.Spec):
    cdef readonly TensorSpec source
    cdef readonly TensorSpec _output
    cdef bint _serial_only

cdef class ReduceSum(ReduceSumBase):
    pass

cdef class ReduceSumAccum(ReduceSumBase):
    pass
