import cython

cpdef tuple schedule_key(schedule: object)

@cython.dataclasses.dataclass(frozen=False)
cdef class SearchStats:
    cdef int expansions
