import sys
from typing import NewType, Union

import cython

if cython.compiled:
    from cython.cimports.libc import math
else:
    import math

from . import specs
from .impl import ComposeHole, DirectConv, Impl, Loop, MatmulHole, MoveLet, ReduceSum
from .impl.compose import Pipeline
from .impl.loops import SlidingWindowLoop
from .impl.matmuls import BroadcastVecMult, HvxVrmpyaccVuwVubRub, Mult
from .system_config import current_system
from .tensor import Tensor, Tile

COST_ATTR = "_cost"

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


def detailed_analytical_cost(
    op: Impl,
    depth: int = 0,
    env: dict[Union[Tensor, Tile], str] = None,
    holes_ok=False,
) -> dict[Impl, tuple[int, str]]:
    """Compute costs for a given Impl and its children.

    Returns a dict mapping each Impl to a 2-tuple of its cost and a
    string describing how that cost was computed (used for pretty-printing).

    :param op: The root of the schedule for which to calculate a cost.
    :param depth: The amount of whitespace to prefix logs.
    :param env: Names for matrices and views. Used for prettier printing.
    :param holes_ok: If `True`, cost holes as zero. Otherwise, raise an exception.
    """
    if env is None:
        env = {}

    if isinstance(op, Pipeline):
        cost_dict: dict[Impl, tuple[int, str]] = {}
        sum_cost = 0
        for stage in op.stages:
            sub_cd = detailed_analytical_cost(
                stage, depth=depth + 1, env=env, holes_ok=holes_ok,
            )
            cost_dict.update(sub_cd)
            sum_cost += sub_cd[stage][0]
        assert compute_cost(op) == sum_cost
        cost_dict[op] = (sum_cost, f"{sum_cost} (sum of {len(op.stages)})")
        return cost_dict
    elif isinstance(op, Loop):
        # Non-sliding loops are just the inner cost times the number of iterations.
        cost_dict = detailed_analytical_cost(
            op.inner, depth=depth + 1, env=env, holes_ok=holes_ok
        )

        if not op.parallel:
            factor = op.steps
        else:
            main_steps = op.full_steps
            factor = cython.cast(
                cython.int, math.ceil(main_steps / current_system().processors)
            )
            factor += op.steps - main_steps

        new_cost = factor * cost_dict[op.inner][0]
        cost_expl = f"{new_cost:5d} = {factor} * _"
        # assert compute_cost(op) == new_cost, f"{compute_cost(op)} != {new_cost}"
        cost_dict[op] = (new_cost, cost_expl)
        return cost_dict
    elif isinstance(op, SlidingWindowLoop):
        cost_dict = detailed_analytical_cost(
            op.inner, depth=depth + 1, env=env, holes_ok=holes_ok,
        )
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
            + (op.steps * cost_dict[op.inner][0])
        )
        cost_expl = (
            f"{new_cost:5d} = {op.whole_loads}({whole_load_cost}) + "
            f"{op.update_loads}({update_cost}) + {op.steps}(_)"
        )
        cost_dict[op] = (new_cost, cost_expl)
        assert compute_cost(op) == new_cost
        return cost_dict
    elif isinstance(op, (DirectConv, ReduceSum)) and (op.is_scheduled or holes_ok):
        assert compute_cost(op) == 4999
        return {op: (4999, "    4999")}
    elif isinstance(op, (Mult, BroadcastVecMult, HvxVrmpyaccVuwVubRub)):
        # Tensor multiplication is free but its operands must be in memory.
        # (This cost model is only interested in the cost of moving data.)
        assert compute_cost(op) == 1
        return {op: (1, "    1")}
    elif isinstance(op, MoveLet):
        # This is the core of the cost model; the cost of a schedule is derived
        # entirely from its moves, which are done by MoveLet operations.
        mcost = move_cost(
            op.spec.operands[op.source_idx], op.destination.spec, op.prefetching
        )
        cost_dict = detailed_analytical_cost(
            op.inner, depth=depth + 1, env=env, holes_ok=holes_ok,
        )
        assert isinstance(mcost, int)
        assert isinstance(cost_dict[op.inner][0], int)
        new_cost = mcost + cost_dict[op.inner][0]
        cost_expl = f"{new_cost:5d} = {mcost} + _"
        assert compute_cost(op) == new_cost
        cost_dict[op] = (new_cost, cost_expl)
        return cost_dict
    elif holes_ok and isinstance(op, (ComposeHole, MatmulHole)):
        assert compute_cost(op) == 0
        return {op: (0, "")}
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

    if isinstance(op, Pipeline):
        sum_cost: MainCost = 0
        for s in op.stages:
            sum_cost = _clip_add(sum_cost, compute_cost(s))
        return _assign_cost(op, sum_cost)
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
        return _assign_cost(op, _clip_mul(factor, compute_cost(op.inner)))
    elif isinstance(op, SlidingWindowLoop):
        raise NotImplementedError()
    elif isinstance(op, (DirectConv, ReduceSum)):
        # Reminder: these types can be either holes or scheduled
        return _assign_cost(op, 4999)
    elif isinstance(op, (Mult, BroadcastVecMult, HvxVrmpyaccVuwVubRub)):
        return _assign_cost(op, 1)
    elif isinstance(op, MoveLet):
        mcost: MainCost = move_cost(  # type: ignore
            op.spec.operands[op.source_idx], op.destination.spec, op.prefetching
        )
        return _assign_cost(op, _clip_add(mcost, compute_cost(op.inner)))
    elif isinstance(op, (ComposeHole, MatmulHole)):
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
