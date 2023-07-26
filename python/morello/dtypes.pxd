import cython

@cython.dataclasses.dataclass(frozen=True)
cdef class Dtype:
    cdef readonly int size
    cdef readonly str c_type
    cdef readonly str int_fmt_macro
    cdef readonly object np_type
    cdef readonly str short_name
