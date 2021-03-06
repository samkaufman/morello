from typing import Optional

import sympy

from .. import specs
from ..tensor import ConvolutionImageTile, SimpleTile, TensorBase, Tile
from .expr_utils import FloorDiv


def buffer_indexing_expr(
    tensor: TensorBase, concrete_shape: Optional[tuple[int, ...]] = None
) -> sympy.Expr:
    """Returns a sympy.Expr mapping logical Tensor coordinates to buffer offset."""
    if concrete_shape is None:
        concrete_shape = tensor.dim_sizes
    assert len(concrete_shape) == len(tensor.dim_sizes)
    if tensor.spec.layout in (layouts.ROW_MAJOR, layouts.COL_MAJOR):
        substitutions = {}
        for idx, dim in enumerate(concrete_shape):
            substitutions[sympy.symbols(f"s{idx}")] = dim
            if dim == 1:
                substitutions[sympy.symbols(f"p{idx}")] = 0
        if tensor.layout == layouts.ROW_MAJOR:
            index_expr = _tensor_row_major_indexing_expr(len(concrete_shape))
        elif tensor.layout == layouts.COL_MAJOR:
            index_expr = _tensor_col_major_indexing_expr(len(concrete_shape))
        else:
            raise NotImplementedError(f"Unsupported layout: {tensor.layout}")
        index_expr = index_expr.subs(substitutions, simultaneous=True)
        assert isinstance(index_expr, sympy.Expr)
        return index_expr
    elif tensor.spec.layout == layouts.HEXAGON_TRANSPACKED:
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
    else:
        raise NotImplementedError(
            f"Indexing expression not defined for {tensor.spec.layout}"
        )


def logical_indexing_expr(source: Tile, dim: int) -> sympy.Expr:
    """Returns an Expr mapping tile coordinates to source coordinates.

    The expression may contain the following symbols:
        * p0, p1, ... pn: the point in `source`
        * i0, i1, ... in: the index of a concrete tile in the given dimension

    :param dim: The dimension in `source`'s origin into which the returned Expr result
        translates. Normally, this function is called once for each dimension in the
        origin.
    """
    idx, pt = sympy.symbols(f"i{dim} p{dim}")

    # TODO: Reintroduce the following optimization.
    # If we know there is only one tile along this dimension, just set idx
    # to the constant 0
    #
    # if source.steps_dim(dim) == 1:
    #     idx = sympy.core.numbers.Zero()

    if isinstance(source, SimpleTile) or (
        isinstance(source, ConvolutionImageTile) and dim == 0
    ):
        w = source.dim_sizes[dim]
        # If w == 1, pt must be 0, so we can simplify.
        if w == 1:
            return idx
        return (w * idx) + pt
    elif isinstance(source, ConvolutionImageTile):
        assert dim > 0, f"dim ({dim}) was expected to be handled by prior case"
        c = 1 + source.dim_sizes[dim] - source.filter_shape[dim - 1]
        return c * idx + pt
    else:
        raise NotImplementedError(f"Not defined for {type(source).__name__}")