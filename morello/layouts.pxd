cdef class Layout:
    cdef object buffer_indexing_expr(self, concrete_shape: Sequence[int])