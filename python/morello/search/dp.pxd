import cython

cdef extern from *:
    ctypedef int int128 "__int128_t"

@cython.dataclasses.dataclass(frozen=True)
cdef class SearchResult:
    cdef readonly object impls
    cdef readonly int128 dependent_paths