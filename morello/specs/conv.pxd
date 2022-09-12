from . cimport base
from .tensorspec cimport TensorSpec

cdef class ConvolutionBase(base.Spec):
    cdef TensorSpec lhs
    cdef TensorSpec rhs
    cdef TensorSpec _output
    cdef bint _serial_only

cdef class Convolution(ConvolutionBase):
    pass

cdef class ConvolutionAccum(ConvolutionBase):
    pass