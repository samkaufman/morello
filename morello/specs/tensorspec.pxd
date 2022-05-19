from .. cimport dtypes
from .. cimport layouts

cimport cython

cdef class TensorSpec:
    cdef readonly tuple dim_sizes
    cdef readonly dtypes.Dtype dtype
    cdef readonly str bank
    cdef readonly layouts.Layout layout
