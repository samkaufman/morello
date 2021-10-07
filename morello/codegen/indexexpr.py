from typing import Optional

import sympy

from ..specs import Layout
from ..tensor import ConvolutionImageTile, SimpleTile, TensorBase, Tile


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


def buffer_indexing_expr(
    tensor: TensorBase, concrete_shape: Optional[tuple[int, ...]] = None
) -> sympy.Expr:
    """Returns a sympy.Expr mapping logical Tensor coordinates to buffer offset."""
    if concrete_shape is None:
        concrete_shape = tensor.dim_sizes
    assert len(concrete_shape) == len(tensor.dim_sizes)
    if tensor.spec.layout in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
        substitutions = {}
        for idx, dim in enumerate(concrete_shape):
            substitutions[sympy.symbols(f"s{idx}")] = dim
            if dim == 1:
                substitutions[sympy.symbols(f"p{idx}")] = 0
        if tensor.layout == Layout.ROW_MAJOR:
            index_expr = _tensor_row_major_indexing_expr(len(concrete_shape))
        else:
            index_expr = _tensor_col_major_indexing_expr(len(concrete_shape))
        index_expr = index_expr.subs(substitutions, simultaneous=True)
        assert isinstance(index_expr, sympy.Expr)
        return index_expr
    else:
        raise NotImplementedError(
            f"Indexing expression not defined for {tensor.spec.layout}"
        )


def logical_indexing_expr(source: Tile, dim: int) -> sympy.Expr:
    """Returns an Expr mapping tile coordinates to source coordinates.

    The expression may contain the following symbols:
        * p0, p1, ... pn: the point in `source`
        * i0, i1, ... in: the index of a concrete tile in the given dimension

    :param dim: The dimension in `source.origin` into which the returned Expr result
        translates. Normally, this function is called once for each dimension in the
        origin.
    """
    idx, pt = sympy.symbols(f"i{dim} p{dim}")
    # If we know there is only one tile along this dimension, just set idx
    # to the constant 0
    if source.steps_dim(dim) == 1:
        idx = sympy.core.numbers.Zero()
    if isinstance(source, SimpleTile):
        w = source.dim_sizes[dim]
        # If w == 1, pt must be 0 an w == 1, so we can simplify.
        if w == 1:
            return idx
        else:
            return (w * idx) + pt
    elif isinstance(source, ConvolutionImageTile):
        c = 1 + source.dim_sizes[dim] - source.filter_shape[dim]
        return c * idx + pt
    else:
        raise NotImplementedError(f"Not defined for {type(source).__name__}")
