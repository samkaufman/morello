import sys
from typing import NewType, Sequence

import cython

if cython.compiled:
    from cython.cimports.libc import math
else:
    import math

from . import specs
from .impl import (
    Add,
    Block,
    BroadcastVecMult,
    ComposeHole,
    Impl,
    Loop,
    MatmulAccumHole,
    MatmulHole,
    MemsetZero,
    MoveLet,
    Mult,
    Pipeline,
    SlidingWindowLoop,
    SpecCast,
    ValueAssign,
    VectorZero,
    VectorAssign,
)
from .system_config import current_system

COST_ATTR = "_cost"
INST_COST = 1000
ASSIGN_INST_COST = 10

if not cython.compiled:
    _MAX_COST = sys.maxsize
    MainCost = NewType("MainCost", int)


@cython.exceptval(-1)
def move_cost(
    src: specs.TensorSpec, dest: specs.TensorSpec, prefetching: bool
) -> MainCost:  # type: ignore
    """Estimate a cost of moving all data in `src` to a new matrix with `dest_layout`.

    Notice that the destination level is not considered. The cache_hit_cost for a
    level should reflect the cost of movement from that level to the next highest.

    Doesn't handle the case where a single cache line crosses major panels very well.
    """

    # "Real" inputs:
    #   - src.root.level's cache_hit_cost
    #   - src.contiguous
    #   - whether src.layout matches dest_layout:
    #     * src.layout
    #     * dest_layout

    src_hit_cost = current_system().banks[src.bank].cache_hit_cost
    dest_hit_cost = current_system().banks[dest.bank].cache_hit_cost

    # Cost is the sum of the cost of touching both memories.
    cost = (
        10
        * src_hit_cost
        * src.layout.estimate_cache_lines(src.dim_sizes, src.dtype, src.contiguous)
    )
    cost += (
        10
        * dest_hit_cost
        * dest.layout.estimate_cache_lines(dest.dim_sizes, dest.dtype, dest.contiguous)
    )

    # Remove half the cost for prefetched moves. This is essentially a
    # tie-breaking hack to get around the fact that we are not modeling
    # pipelining.
    if prefetching:
        cost //= 2

    # Add a 10% to penalize a lack of hardware prefetching. (This is target-
    # specific!)
    # TODO: Revise.
    if not src.contiguous or src.layout != dest.layout:
        cost = int(2 * cost)
    return cost


def detailed_analytical_cost(op: Impl, depth: int = 0) -> dict[Impl, tuple[int, str]]:
    """Compute costs for a given Impl and its children.

    Returns a dict mapping each Impl to a 2-tuple of its cost and a
    string describing how that cost was computed (used for pretty-printing).

    :param op: The root of the schedule for which to calculate a cost.
    :param depth: The amount of whitespace to prefix logs.
    """
    d = {}
    _detailed_analytical_cost_inner(d, op, depth=depth)
    return d


def _detailed_analytical_cost_inner(
    out_dict: dict[Impl, tuple[int, str]], op: Impl, depth: int
) -> None:
    for child in op.children:
        _detailed_analytical_cost_inner(out_dict, child, depth=depth + 1)
    child_costs: list[MainCost] = [out_dict[c][0] for c in op.children]
    c = compute_cost_node(op, child_costs)
    new_str = _detailed_node_str(op, c, child_costs, depth=depth)
    out_dict[op] = (c, new_str)


def _detailed_node_str(
    op: Impl, op_cost: MainCost, child_costs: Sequence[MainCost], depth: int
) -> str:
    if isinstance(op, (Block, Pipeline)):
        return f"{op_cost} (sum of {len(op.children)})"
    elif isinstance(op, Loop):
        factor = op.steps
        if op.parallel:
            main_steps = op.full_steps
            factor = cython.cast(
                cython.int, math.ceil(main_steps / current_system().processors)
            )
            factor += op.steps - main_steps
        return f"{op_cost:5d} = {factor} * _"
    elif isinstance(op, SlidingWindowLoop):
        # The moves are implicit in SlidingWindowLoop, so we'll construct
        # Tiles to serve as operands to `move_cost`.
        whole_window_tile = op.operands[op.live_tensor_idx].simple_tile(
            op.live_tensor.dim_sizes
        )
        frontier_tile = op.operands[op.live_tensor_idx].simple_tile(op.frontier_shape)
        # TODO: Should support prefetching for sliding windows.
        whole_load_cost = move_cost(whole_window_tile.spec, op.live_tensor.spec, False)
        update_cost = move_cost(frontier_tile.spec, op.live_tensor.spec, False)
        new_cost = (
            (op.whole_loads * whole_load_cost)
            + (op.update_loads * update_cost)
            + (op.steps * child_costs[0])
        )
        return (
            f"{new_cost:5d} = {op.whole_loads}({whole_load_cost}) + "
            f"{op.update_loads}({update_cost}) + {op.steps}(_)"
        )
    elif isinstance(op, (Add, Mult, BroadcastVecMult)):
        # Tensor multiplication is free but its operands must be in memory.
        # (This cost model is only interested in the cost of moving data.)
        return f"{INST_COST:5}"
    elif isinstance(op, (ValueAssign, VectorAssign, MemsetZero, VectorZero)):
        return f"{ASSIGN_INST_COST:5}"
    elif isinstance(op, MoveLet):
        # This is the core of the cost model; the cost of a schedule is derived
        # entirely from its moves, which are done by MoveLet operations.
        mcost: int = move_cost(
            op.spec.operands[op.source_idx], op.destination.spec, op.prefetching
        )
        return f"{op_cost:5d} = {mcost} + _"
    elif isinstance(op, SpecCast):
        return "_"
    else:
        raise TypeError(f"Unsupported op. {type(op)}")


# TODO: Reduce code duplication with detailed_analytical_cost
@cython.exceptval(-1)
@cython.cdivision(True)
def compute_cost(op: Impl) -> MainCost:
    # Return if already computed
    try:
        return cython.cast(MainCost, getattr(op, COST_ATTR), typecheck=True)
    except AttributeError:
        pass
    return compute_cost_node(op, [compute_cost(child) for child in op.children])


# TODO: Reduce code duplication with detailed_analytical_cost
@cython.exceptval(-1)
@cython.cdivision(True)
def compute_cost_node(op: Impl, child_costs: list[MainCost]) -> MainCost:
    if isinstance(op, (Block, Pipeline)):
        sum_cost: MainCost = 0
        for child_cost in child_costs:
            sum_cost = _clip_add(sum_cost, child_cost)
        return _assign_cost(op, sum_cost)
    elif isinstance(op, SpecCast):
        return _assign_cost(op, child_costs[0])
    elif isinstance(op, Loop):
        if not op.parallel:
            factor: MainCost = op.steps
        else:
            system_processors: cython.int = current_system().processors
            main_steps: cython.int = op.full_steps
            op_steps: cython.int = op.steps
            with cython.nogil:
                factor = cython.cast(
                    cython.int, (main_steps + system_processors - 1) / system_processors
                )
                factor += _clip_sub(op_steps, main_steps)
        assert isinstance(
            child_costs[0], int
        ), f"{child_costs[0]} was {type(child_costs[0])}"
        return _assign_cost(op, _clip_mul(factor, child_costs[0]))
    elif isinstance(op, SlidingWindowLoop):
        raise NotImplementedError()
    elif isinstance(op, (Add, Mult, BroadcastVecMult)):
        return _assign_cost(op, INST_COST)
    elif isinstance(op, (ValueAssign, VectorAssign, MemsetZero, VectorZero)):
        return _assign_cost(op, ASSIGN_INST_COST)
    elif isinstance(op, MoveLet):
        mcost: MainCost = move_cost(  # type: ignore
            op.spec.operands[op.source_idx], op.destination.spec, op.prefetching
        )
        # TODO: Replace following with child costs
        v: MainCost = mcost
        for child_cost in child_costs:
            v = _clip_add(v, child_cost)
        return _assign_cost(op, v)
    elif isinstance(op, (ComposeHole, MatmulHole, MatmulAccumHole)):
        return _assign_cost(op, 0)
    else:
        raise TypeError(f"Unsupported op. {type(op)}")


@cython.cfunc
@cython.nogil
def _clip_add(a: MainCost, b: MainCost) -> MainCost:
    if not cython.compiled:
        # This is a functional difference. No clipping happens if interpreted.
        return a + b
    result: MainCost
    overflowed = __builtin_saddl_overflow(a, b, cython.address(result))
    if overflowed:
        return _MAX_COST
    return result


@cython.cfunc
@cython.exceptval(_MAX_COST, check=True)
@cython.nogil
def _clip_sub(a: MainCost, b: MainCost) -> MainCost:
    if not cython.compiled:
        # This is a functional difference. No clipping happens if interpreted.
        return a - b
    result: MainCost
    underflowed = __builtin_ssubl_overflow(a, b, cython.address(result))
    if underflowed:
        with cython.gil:
            raise OverflowError(f"Subtracting {a} - {b} underflowed")
    return result


@cython.cfunc
@cython.nogil
def _clip_mul(a: MainCost, b: MainCost) -> MainCost:
    if not cython.compiled:
        # This is a functional difference. No clipping happens if interpreted.
        return a * b
    result: MainCost
    overflowed = __builtin_smull_overflow(a, b, cython.address(result))
    if overflowed:
        return _MAX_COST
    return result


@cython.cfunc
@cython.inline
def _assign_cost(impl: Impl, val: MainCost) -> MainCost:
    assert val >= 0, f"val was unexpectedly negative: {val}"
    object.__setattr__(impl, COST_ATTR, val)
    return val
