import abc
from typing import Sequence

import sympy

from .codegen.expr_utils import FloorDiv


class Layout(abc.ABC):
    """The layout of a tensor."""

    @abc.abstractmethod
    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        pass

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(other) == type(self)


class RowMajor(Layout):
    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        return _rm_cm_buffer_indexing_expr(
            _tensor_row_major_indexing_expr, concrete_shape
        )

    def __str__(self) -> str:
        return "RM"


ROW_MAJOR = RowMajor()  # singleton


class ColMajor(Layout):
    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        return _rm_cm_buffer_indexing_expr(
            _tensor_col_major_indexing_expr, concrete_shape
        )

    def __str__(self) -> str:
        return "CM"


COL_MAJOR = ColMajor()  # singleton


class HexagonTranspacked(Layout):
    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        # This layout is only used for Uint8, so the following will index
        # 128-bit blocks (vectors in HVX VMEM).
        orig_rows, orig_cols = concrete_shape
        padded_rows = orig_rows - orig_rows % -32
        p0, p1 = sympy.symbols("p0 p1")
        # Logical sizes for the 128-byte vectors.
        row_block = FloorDiv(p0, 4)
        col_block = FloorDiv(p1, 32)
        # The blocks in each dimension might differ because of padding, and this
        # can affect the offsets (e.g., by added intervening all-zero vectors).
        inner_offset = 4 * sympy.UnevaluatedExpr(p1 % 32) + sympy.UnevaluatedExpr(
            p0 % 4
        )
        block_rows = padded_rows // 4
        block_offset = (128 * block_rows * col_block) + row_block  # block offset
        return block_offset + inner_offset

    def __str__(self) -> str:
        return "TP"


HEXAGON_TRANSPACKED = HexagonTranspacked()  # singleton


def _rm_cm_buffer_indexing_expr(iefn, concrete_shape) -> sympy.Expr:
    substitutions = {}
    for idx, dim in enumerate(concrete_shape):
        substitutions[sympy.symbols(f"s{idx}")] = dim
        if dim == 1:
            substitutions[sympy.symbols(f"p{idx}")] = 0
    index_expr = iefn(len(concrete_shape))
    index_expr = index_expr.subs(substitutions, simultaneous=True)
    assert isinstance(index_expr, sympy.Expr)
    return index_expr


def _tensor_row_major_indexing_expr(rank: int) -> sympy.Expr:
    assert rank > 0
    if rank == 1:
        return sympy.symbols("p0")
    else:
        s, p = sympy.symbols(f"s{rank - 1}, p{rank - 1}")
        return _tensor_row_major_indexing_expr(rank - 1) * s + p


def _tensor_col_major_indexing_expr(rank: int) -> sympy.Expr:
    if rank == 2:
        p0, p1, s0 = sympy.symbols("p0 p1 s0")
        return (p1 * s0) + p0
    elif rank > 2:
        s, p = sympy.symbols(f"s{rank - 1}, p{rank - 1}")
        return _tensor_col_major_indexing_expr(rank - 1) * s + p
    else:
        raise ValueError("rank must be at least 2, but was " + str(rank))
