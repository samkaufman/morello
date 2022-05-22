from libc cimport limits

cdef extern from "stdbool.h":
    ctypedef bint bool

ctypedef long MainCost

cdef enum:
    _MAX_COST = limits.LONG_MAX # 2147483647  # Maximum long (MainCost)

cpdef MainCost move_cost(object src, object dest_layout, bool prefetching) except -1;
cpdef MainCost compute_cost(object op) except -1;

cdef extern bool __builtin_saddl_overflow(const long x, const long y, long *sum) nogil;
cdef extern bool __builtin_ssubl_overflow(const long x, const long y, long *diff) nogil;
cdef extern bool __builtin_smull_overflow(const long x, const long y, long *prod) nogil;

# @cython.total_ordering
cdef class ExtendedCost:
    cdef public MainCost main
    cdef long[:] peaks
    cdef int depth
