from . cimport base
from .tensorspec cimport TensorSpec

cdef class Compose(base.Spec):
    cdef readonly tuple subspec_classes
    cdef readonly tuple _inputs
    cdef readonly TensorSpec _output
    cdef readonly tuple intermediate_dtypes
    cdef readonly bint _serial_only
