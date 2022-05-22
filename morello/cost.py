import functools
from operator import mul
from typing import NewType, Union

import cython

if cython.compiled:
    from cython.cimports.libc import limits, math
else:
    import math

from . import layouts, specs, utils
from .impl import ComposeHole, DirectConv, Impl, Loop, MatmulHole, MoveLet, ReduceSum
from .impl.compose import Pipeline
from .impl.loops import SlidingWindowLoop
from .impl.matmuls import HvxVrmpyaccVuwVubRub, Mult
from .system_config import current_system
from .tensor import Tensor, Tile

COST_ATTR = "_cost"

if not cython.compiled:
    MainCost = NewType("MainCost", int)


@cython.exceptval(-1)
def move_cost(
    src: specs.TensorSpec, dest_layout: layouts.Layout, prefetching: bool
) -> MainCost:
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

    hit_cost = current_system().banks[src.bank].cache_hit_cost

    lodims = utils.layout_ordered_dims(src)
    meaningful_layout_difference = (
        src.layout != dest_layout and lodims[0] != 1 and lodims[1] != 1
    )

    # Make real_dims, which is all non-1 dimensions
    real_dims = [d for d in lodims if d > 1]
    if not real_dims:
        real_dims = [1]

    # Main cost formula
    cost = (
        10
        * hit_cost
        * functools.reduce(mul, real_dims[:-1], 1)
        * cython.cast(
            cython.int,
            math.ceil(real_dims[-1] / current_system().line_size),
            typecheck=True,
        )
    )

    # Remove half the cost for prefetched moves. This is essentially a
    # tie-breaking hack to get around the fact that we are not modeling both
    # compute and memory cost.
    if prefetching:
        cost //= 2

    # Add a 10% to penalize a lack of hardware prefetching
    # TODO: Add contiguousness to specs; check `not src.contiguous`.
    if meaningful_layout_difference:
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
                stage,
                depth=depth + 1,
                env=env,
                holes_ok=holes_ok,
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
            op.inner,
            depth=depth + 1,
            env=env,
            holes_ok=holes_ok,
        )
        # The moves are implicit in SlidingWindowLoop, so we'll construct
        # Tiles to serve as operands to `move_cost`.
        whole_window_tile = op.live_tensor.origin.simple_tile(op.live_tensor.dim_sizes)
        frontier_tile = op.live_tensor.origin.simple_tile(op.frontier_shape)
        # TODO: Should support prefetching for sliding windows.
        whole_load_cost = move_cost(whole_window_tile, op.live_tensor.layout, False)
        update_cost = move_cost(frontier_tile, op.live_tensor.layout, False)
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
        assert compute_cost(op) == 0
        return {op: (0, "    0")}
    elif isinstance(op, (Mult, HvxVrmpyaccVuwVubRub)):
        # Tensor multiplication is free but its operands must be in memory.
        # (This cost model is only interested in the cost of moving data.)
        assert compute_cost(op) == 0
        return {op: (0, "    0")}
    elif isinstance(op, MoveLet):
        # This is the core of the cost model; the cost of a schedule is derived
        # entirely from its moves, which are done by MoveLet operations.
        mcost = move_cost(
            op.spec.operands[op.source_idx], op.destination.layout, op.prefetching
        )
        cost_dict = detailed_analytical_cost(
            op.inner,
            depth=depth + 1,
            env=env,
            holes_ok=holes_ok,
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
                    cython.int, math.ceil(main_steps / system_processors)
                )
                factor += _clip_sub(op_steps, main_steps)
        return _assign_cost(op, _clip_mul(factor, compute_cost(op.inner)))
    elif isinstance(op, SlidingWindowLoop):
        raise NotImplementedError()
    elif isinstance(op, (DirectConv, ReduceSum)):
        # Reminder: these types can be either holes or scheduled
        return _assign_cost(op, 0)
    elif isinstance(op, (Mult, HvxVrmpyaccVuwVubRub)):
        return _assign_cost(op, 0)
    elif isinstance(op, MoveLet):
        mcost: MainCost = move_cost(
            op.spec.operands[op.source_idx], op.destination.layout, op.prefetching
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
