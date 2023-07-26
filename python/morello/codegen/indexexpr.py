import sympy

from ..tensor import (
    ConvolutionImageTile,
    InnerContigFlatteningTile,
    SimpleTile,
    SqueezingTile,
    Tensor,
    TensorLike,
    TransposingTile,
)


def vsub(expr, *args, **kwargs) -> sympy.Expr:
    kwargs = dict(kwargs)
    if "simultaneous" not in kwargs:
        kwargs["simultaneous"] = True
    result = expr.subs(*args, **kwargs)
    assert isinstance(result, sympy.Expr)
    return result


def logical_indexing_expr(source: TensorLike, dim: int) -> sympy.Expr:
    """Returns an Expr mapping tile coordinates to source coordinates.

    If given a Tensor, will return the identity translation.

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
    elif isinstance(source, SqueezingTile):
        # TODO: Move this logic into the Tiles themselves so that we don't need
        #  to call into private methods.
        source_expr = logical_indexing_expr(source.inner, dim)
        exploded = source._squeezed_to_exploded_dims()
        substitutions = []
        for s, e in exploded.items():
            substitutions.append((sympy.symbols(f"p{e}"), sympy.symbols(f"p{s}")))
        return vsub(source_expr, substitutions, simultaneous=True)
    elif isinstance(source, TransposingTile):
        i, j = source.swap_dims
        new_dim = dim
        # if dim == i:
        #     new_dim = j
        # elif dim == j:
        #     new_dim = i
        source_expr = logical_indexing_expr(source.inner, new_dim)
        return vsub(
            source_expr, [(f"p{i}", f"p{j}"), (f"p{j}", f"p{i}")], simultaneous=True
        )
        # return source_expr
    elif isinstance(source, InnerContigFlatteningTile):
        assert dim < len(source.dim_sizes)
        raise NotImplementedError()
    elif isinstance(source, Tensor):
        w = source.dim_sizes[dim]
        return sympy.core.numbers.Zero() if w == 1 else pt
    else:
        raise NotImplementedError(f"Not defined for {type(source).__name__}")
