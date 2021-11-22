import contextlib
import dataclasses
import logging
import sys
import os
from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import tqdm

from . import common, random
from .. import cost, ops, pruning, specs

HEURISTIC_SAMPLES_PER_SPEC = 10
HEURISTIC_MAX_RESTARTS = int(os.getenv("HEURISTIC_MAX_RESTARTS", 100))
STOCHASTIC = False

logger = logging.getLogger(__name__)


def _cost(
    schedule: ops.Schedule, limits: Sequence[pruning.MemoryLimits]
) -> Union[int, float]:
    """Returns a cost to use as a heuristic."""
    return cost.analytical_cost(schedule, holes_ok=True)


def sampling_heuristic(
    schedule: ops.Schedule, limits: Sequence[pruning.MemoryLimits]
) -> Union[int, float]:
    best_cost = sys.maxsize
    for _ in range(HEURISTIC_SAMPLES_PER_SPEC):
        # We're not adding memory limits restrictions here, which suggests the
        # simulated model for our heuristic has not learned to encode that
        # information.
        sampled_impl = None
        while not sampled_impl:
            try:
                sampled_impl = random.randomly_schedule_impl(
                    schedule,
                    budget=None,
                    memory_limits=limits,
                    max_restarts=HEURISTIC_MAX_RESTARTS,
                    skip_sliding=True,
                )[0]
            except random.MaxRestartsExceeded:
                logger.info("MaxRestartsExceeded; switching to no memory limits")
                sampled_impl = random.randomly_schedule_impl(
                    schedule,
                    budget=None,
                    skip_sliding=True,
                )[0]
        assert isinstance(
            sampled_impl, ops.Schedule
        ), f"Schedule was unexpectedly {sampled_impl}"
        sampled_cost = cost.analytical_cost(sampled_impl, holes_ok=True)
        best_cost = min(best_cost, sampled_cost)
    return best_cost


@dataclasses.dataclass(frozen=True)
class _State:
    schedule: ops.Schedule
    child_limits: tuple[pruning.MemoryLimits, ...]
    cost: Union[int, float]


def beam_schedule_search(
    spec: specs.Spec,
    inputs: tuple,
    output,
    k: int = 10000,
    budget: Optional[int] = None,
    stats: Optional[common.SearchStats] = None,
    cost_fn: Callable[
        [ops.Schedule, Sequence[pruning.MemoryLimits]], Union[int, float]
    ] = None,
    progress_bar: bool = False,
    return_run_costs: bool = False,
) -> Optional[Union[ops.Schedule, tuple[ops.Schedule, Sequence[Union[int, float]]]]]:
    if cost_fn is None:
        cost_fn = _cost

    select_fn = _select_top_k_states
    if STOCHASTIC:
        select_fn = _select_new_states

    if common.prune_column_major.get():
        if any(inp.layout == specs.Layout.COL_MAJOR for inp in spec.inputs):
            return None
        if spec.output.layout == specs.Layout.COL_MAJOR:
            return None

    pb_ctx_a = contextlib.nullcontext(None)
    pb_ctx_b = contextlib.nullcontext(None)
    if progress_bar:
        pb_ctx_a = tqdm.tqdm(unit="state")
        pb_ctx_b = tqdm.tqdm(unit="action")

    # Initialize the beam search with a single state: an Impl hole for the query Spec.
    best_found: tuple[Optional[ops.Schedule], Union[int, float]] = None, sys.maxsize
    first_impl = ops.spec_to_hole(spec, inputs, output)
    fresh_limits = (pruning.StandardMemoryLimits(),)
    states = [_State(first_impl, fresh_limits, cost_fn(first_impl, fresh_limits))]

    all_run_costs = [states[0].cost]

    with pb_ctx_a as pb_a, pb_ctx_b as pb_b:
        while True:
            if pb_a is not None:
                pb_a.reset(total=len(states))

            # Only retain scheduled implementations through to the next step.
            new_states: list[_State] = []
            for state in states:
                hole_actions = list(
                    common.hole_actions(state.schedule, include_inner_impl=True)
                )
                if pb_b is not None:
                    pb_b.reset(total=len(hole_actions))
                for act, child_idx in hole_actions:
                    limits_for_action_child = state.child_limits[child_idx]
                    new_impl, introduced_inner = act()
                    introduced_limits = limits_for_action_child.transition(
                        introduced_inner
                    )

                    # Ignore the action if it exceeds our memory limits.
                    if introduced_limits is not None:
                        updated_limits = (
                            state.child_limits[:child_idx]
                            + tuple(introduced_limits)
                            + state.child_limits[child_idx + 1 :]
                        )

                        introduced_cost = cost_fn(new_impl, updated_limits)
                        new_states.append(
                            _State(new_impl, updated_limits, introduced_cost)
                        )
                        all_run_costs.append(introduced_cost)

                        if (
                            new_states[-1].schedule.is_scheduled
                            and introduced_cost < best_found[1]
                        ):
                            best_found = new_impl, introduced_cost

                    # Decrement budget even if the action exceeds memory limits.
                    if budget is not None:
                        if budget == 0:
                            assert isinstance(best_found[0], ops.Schedule)
                            if return_run_costs and best_found[0] is not None:
                                return best_found[0], all_run_costs
                            return best_found[0]
                        budget -= 1
                    stats.expansions += 1

                    if pb_b is not None:
                        pb_b.update(1)

                if pb_a is not None:
                    pb_a.update(1)

            if not new_states:
                if return_run_costs and best_found[0] is not None:
                    return best_found[0], all_run_costs
                return best_found[0]
            new_states = select_fn(new_states, k)
            if new_states == states:
                if return_run_costs and best_found[0] is not None:
                    return best_found[0], all_run_costs
                return best_found[0]
            states = new_states


def _select_top_k_states(states: Sequence[_State], k: int) -> list[_State]:
    """Select the `k` lowest-cost states from those given."""
    sorted_states = sorted(states, key=lambda s: s.cost)
    assert not sorted_states or sorted_states[0].cost <= sorted_states[-1].cost
    return sorted_states[:k]


def _select_new_states(states: Sequence[_State], k: int) -> list[_State]:
    """Stochastically select `k` states from possibilities."""
    costs = np.array([s.cost for s in states])

    p = costs / (costs.sum() + 1e-6)
    if np.allclose(p, 0.0):
        p = np.repeat(1.0 / len(p), len(p))
    else:
        p = p / p.sum()

    idxs = np.random.choice(np.arange(len(states)), k, replace=True, p=p)
    return [states[i] for i in idxs]
