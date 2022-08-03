import sympy
from typing import Optional, Sequence, Iterable

from morello import tensor
from morello.codegen import indexexpr


def compose_indexing_exprs(
    stack, concrete_tile_idxs: Iterable[Sequence[int]]
) -> sympy.Expr:
    """Compose indexing expressions.

    This yields a mapping from the final tile's coordinates all
    the way back to its root tensor.
    """

    stack = list(stack)
    concrete_tile_idxs = list(concrete_tile_idxs)
    assert len(stack) == len(concrete_tile_idxs) + 1

    operand: Optional[tensor.TensorLike] = None
    last_spec = stack[0].spec
    expr = stack.pop(0).spec.layout.buffer_indexing_expr(last_spec.dim_sizes)
    while stack:
        operand = stack[0]
        assert isinstance(operand, tensor.Tile)
        del stack[0]
        all_substitutions = {}
        tile_it_vars = concrete_tile_idxs[-1]
        del concrete_tile_idxs[-1]
        for dim_idx, it_var in zip(range(len(operand.dim_sizes)), tile_it_vars):
            e = indexexpr.logical_indexing_expr(operand, dim_idx)
            e = e.subs(f"i{dim_idx}", it_var)
            all_substitutions[f"p{dim_idx}"] = e
        expr = expr.subs(all_substitutions)
        last_spec = operand.spec
    assert operand is not None
    return expr