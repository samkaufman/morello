import functools
import math
from operator import mul
from typing import Union

from . import specs, utils
from .ops import (
    DirectConv,
    HvxVrmpyaccVuwVubRub,
    Loop,
    MatmulSplitLoop,
    MoveLet,
    Mult,
    Pipeline,
    ReduceSum,
    Schedule,
    SlidingWindowLoop,
)
from .system_config import current_system
from .tensor import Tensor, Tile


def move_cost(
    src: Union[Tensor, Tile], dest_layout: specs.Layout, prefetching: bool
) -> int:
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
        * math.ceil(real_dims[-1] / current_system().line_size)
    )

    # Remove half the cost for prefetched moves. This is essentially a
    # tie-breaking hack to get around the fact that we are not modeling both
    # compute and memory cost.
    if prefetching:
        cost //= 2

    # Add a 10% to penalize a lack of hardware prefetching
    if not src.contiguous or meaningful_layout_difference:
        cost = int(1.1 * cost)
    return cost


def detailed_analytical_cost(
    op: Schedule,
    depth: int = 0,
    env: dict[Union[Tensor, Tile], str] = None,
) -> dict[Schedule, tuple[int, str]]:
    """Compute costs for a given Schedule and its children.

    Returns a dict mapping each Schedule to a 2-tuple of its cost and a
    string describing how that cost was computed (used for pretty-printing).

    :param op: The root of the schedule for which to calculate a cost.
    :param depth: The amount of whitespace to prefix logs.
    :param env: Names for matrices and views. Used for prettier printing.
    """
    if env is None:
        env = {}

    if isinstance(op, Pipeline):
        cost_dict: dict[Schedule, tuple[int, str]] = {}
        sum_cost = 0
        for stage in op.stages:
            sub_cd = detailed_analytical_cost(
                stage,
                depth=depth + 1,
                env=env,
            )
            cost_dict.update(sub_cd)
            sum_cost += sub_cd[stage][0]
        cost_dict[op] = (sum_cost, f"{sum_cost} (sum of {len(op.stages)})")
        return cost_dict
    elif isinstance(op, (MatmulSplitLoop, Loop)):
        # Non-sliding loops are just the inner cost times the number of iterations.
        cost_dict = detailed_analytical_cost(
            op.inner,
            depth=depth + 1,
            env=env,
        )

        factor = op.steps
        if op.parallel:
            factor = math.ceil(op.steps / current_system().processors)

        new_cost = factor * cost_dict[op.inner][0]
        cost_expl = f"{new_cost:5d} = {factor} * _"
        cost_dict[op] = (new_cost, cost_expl)
        return cost_dict
    elif isinstance(op, SlidingWindowLoop):
        cost_dict = detailed_analytical_cost(
            op.inner,
            depth=depth + 1,
            env=env,
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
        return cost_dict
    elif isinstance(op, (DirectConv, Mult, HvxVrmpyaccVuwVubRub, ReduceSum)):
        # Tensor multiplication is free but its operands must be in memory.
        # (This cost model is only interested in the cost of moving data.)
        return {op: (0, "    0")}
    elif isinstance(op, MoveLet):
        # This is the core of the cost model; the cost of a schedule is derived
        # entirely from its moves, which are done by MoveLet operations.
        mcost = move_cost(op.source, op.destination.layout, op.prefetching)
        cost_dict = detailed_analytical_cost(
            op.inner,
            depth=depth + 1,
            env=env,
        )
        new_cost = mcost + cost_dict[op.inner][0]
        cost_expl = f"{new_cost:5d} = {mcost} + _"
        cost_dict[op] = (new_cost, cost_expl)
        return cost_dict
    else:
        raise ValueError(f"Unsupported op. {type(op)}")


def analytical_cost(op: Schedule, *args, **kwargs) -> int:
    return detailed_analytical_cost(op, *args, **kwargs)[op][0]
