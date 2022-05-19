import io
import logging
import os
import sys
from collections.abc import Sequence
from typing import Iterable, NamedTuple, Optional, Union

import cython

import numpy as np

from .. import cost, layouts, op_pprint, pruning, search_cache, specs
from ..impl.base import Impl, spec_to_hole
from . import common, dp, random

HEURISTIC_SAMPLES_PER_SPEC = 10
HEURISTIC_MAX_RESTARTS = int(os.getenv("HEURISTIC_MAX_RESTARTS", 100))
STOCHASTIC = False
NOISE_STD_DEV = 0.01

logger = logging.getLogger(__name__)


def _cost(
    schedule: Impl, limits: Sequence[Optional[pruning.MemoryLimits]]
) -> tuple[Union[int, float], str]:
    """Returns a cost to use as a heuristic."""
    return cost.compute_cost(schedule), op_pprint.pformat(schedule)


def _iter_empty(it) -> bool:
    for _ in it:
        return False
    return True


def _dp_schedule_leaves(
    schedule: Impl,
    limits_queue: list[pruning.MemoryLimits],
    cache: search_cache.ScheduleCache,
) -> object:
    if not len(schedule.children):
        sublimits = limits_queue.pop(0)
        if not _iter_empty(schedule.actions()):
            return dp.schedule_search(
                schedule.spec,
                schedule.inputs,
                schedule.output,
                memory_limits=sublimits,
                cache=cache,
            )
        return schedule

    new_children = []
    for child in schedule.children:
        child_impl = _dp_schedule_leaves(child, limits_queue, cache)
        if child_impl is None:
            return None
        new_children.append(child_impl)
    return schedule.replace_children(new_children)


def noisy_optimal_heuristic(
    schedule: Impl,
    limits: Sequence[pruning.MemoryLimits],
    cache: search_cache.ScheduleCache,
) -> tuple[Union[int, float], str]:
    limits_queue = list(limits)
    optimal_completion = _dp_schedule_leaves(schedule, limits_queue, cache)
    if optimal_completion is None:
        return sys.maxsize, ""
    assert not len(limits_queue)
    return (
        cost.compute_cost(optimal_completion) * np.random.normal(1, NOISE_STD_DEV),
        "OPTIMAL COMPLETION\n" + op_pprint.pformat(optimal_completion),
    )


def random_sampling_heuristic(
    schedule: Impl, limits: Sequence[pruning.MemoryLimits]
) -> tuple[Union[int, float], str]:
    best_cost = sys.maxsize
    formatted_samples = []
    for _ in range(HEURISTIC_SAMPLES_PER_SPEC):
        try:
            sampled_impl = random.randomly_schedule_impl(
                schedule,
                budget=None,
                memory_limits=limits,
                max_restarts=HEURISTIC_MAX_RESTARTS,
                skip_sliding=True,
            )[0]
        except random.MaxRestartsExceeded:
            # If we keep restarting (this can happen if random exploration has
            # trouble completing the schedule), then drop this one sample out of
            # HEURISTIC_SAMPLES_PER_SPEC.
            continue
        else:
            assert isinstance(
                sampled_impl, Impl
            ), f"Impl was unexpectedly {sampled_impl}"
            sampled_cost = cost.compute_cost(sampled_impl)
            best_cost = min(best_cost, sampled_cost)
            formatted_samples.append((sampled_cost, op_pprint.pformat(sampled_impl)))
    formatted_samples.sort(key=lambda x: x[0])
    return best_cost, "RANDOM SAMPLING HEURISTIC" + "\n".join(
        t[1] for t in formatted_samples
    )


@cython.dataclasses.dataclass(frozen=True)
@cython.cclass
class _State:
    schedule: Impl
    child_limits: tuple[pruning.MemoryLimits, ...]
    estimated_cost: Union[int, float]
    cost_log_str: str


class BeamScheduleSearchResult(NamedTuple):
    best_impl: Impl
    best_impl_cost: int
    all_estimated_costs: Sequence[Union[int, float]]


def beam_schedule_search(
    spec: specs.Spec,
    inputs: tuple,
    output,
    k: int = 10000,
    budget=None,
    stats=None,
    cost_fn=None,
) -> tuple[Optional[BeamScheduleSearchResult], str]:
    if cost_fn is None:
        cost_fn = _cost

    select_fn = _select_top_k_states
    if STOCHASTIC:
        select_fn = _select_new_states

    if common.prune_column_major.get():
        if any(isinstance(inp.layout, layouts.ColMajor) for inp in spec.inputs):
            return None, ""
        if isinstance(spec.output.layout, layouts.ColMajor):
            return None, ""

    # Initialize the beam search with a single state: an Impl hole for the query Spec.
    best_found: tuple[Optional[Impl], Union[int, float]] = (
        None,
        sys.maxsize,
    )
    first_impl = spec_to_hole(spec, inputs, output)
    fresh_limits = (pruning.StandardMemoryLimits(),)
    states = [_State(first_impl, fresh_limits, *cost_fn(first_impl, fresh_limits))]
    all_run_cost_estimates = [states[0].estimated_cost]

    log_io = io.StringIO()

    step_idx = 0
    while True:
        # Only retain scheduled implementations through to the next step.
        new_states: list[_State] = []
        for state in states:
            hole_actions = list(
                common.leaf_actions(state.schedule, include_inner_impl=True)[0]
            )
            for act, child_idx in hole_actions:
                limits_for_action_child = state.child_limits[child_idx]
                new_impl, introduced_inner = act()
                introduced_limits = limits_for_action_child.transition(introduced_inner)

                # Ignore the action if it exceeds our memory limits.
                if introduced_limits is not None:
                    if not len(introduced_limits):
                        introduced_limits = [None]

                    updated_limits = (
                        state.child_limits[:child_idx]
                        + tuple(introduced_limits)
                        + state.child_limits[child_idx + 1 :]
                    )

                    estimated_cost, heuristic_logs = cost_fn(new_impl, updated_limits)
                    heuristic_logs = _compose_impl_log(
                        new_impl, updated_limits, heuristic_logs
                    )
                    new_states.append(
                        _State(new_impl, updated_limits, estimated_cost, heuristic_logs)
                    )
                    all_run_cost_estimates.append(estimated_cost)

                    if (
                        new_states[-1].schedule.is_scheduled
                        and estimated_cost < best_found[1]
                    ):
                        best_found = new_impl, estimated_cost

                # Decrement budget even if the action exceeds memory limits.
                if budget is not None:
                    if budget == 0:
                        if best_found[0] is None:
                            return None, log_io.getvalue()
                        return (
                            BeamScheduleSearchResult(
                                best_found[0],
                                cost.compute_cost(best_found[0]),
                                all_run_cost_estimates,
                            ),
                            log_io.getvalue(),
                        )
                    budget -= 1
                stats.expansions += 1

        if not new_states:
            if best_found[0] is None:
                return None, log_io.getvalue()
            return (
                BeamScheduleSearchResult(
                    best_found[0],
                    cost.compute_cost(best_found[0]),
                    all_run_cost_estimates,
                ),
                log_io.getvalue(),
            )
        new_states = select_fn(new_states, k)
        if new_states == states:
            if best_found[0] is None:
                return None, log_io.getvalue()
            return (
                BeamScheduleSearchResult(
                    best_found[0],
                    cost.compute_cost(best_found[0]),
                    all_run_cost_estimates,
                ),
                log_io.getvalue(),
            )
        states = new_states
        _write_states_log(log_io, step_idx, states)
        step_idx += 1


def _select_top_k_states(states: Sequence[_State], k: int) -> list[_State]:
    """Select the `k` lowest-cost states from those given."""
    sorted_states = sorted(states, key=lambda s: s.estimated_cost)
    assert (
        not sorted_states
        or sorted_states[0].estimated_cost <= sorted_states[-1].estimated_cost
    )
    return sorted_states[:k]


def _select_new_states(states: Sequence[_State], k: int) -> list[_State]:
    """Stochastically select `k` states from possibilities."""
    costs = np.array([s.estimated_cost for s in states])

    p = costs / (costs.sum() + 1e-6)
    if np.allclose(p, 0.0):
        p = np.repeat(1.0 / len(p), len(p))
    else:
        p = p / p.sum()

    idxs = np.random.choice(np.arange(len(states)), k, replace=True, p=p)
    return [states[i] for i in idxs]


def _compose_impl_log(new_impl: Impl, updated_limits, cost_logs: str) -> str:
    return "\n".join("  " + line for line in cost_logs.split("\n"))


def _write_states_log(dest: io.StringIO, step_idx: int, states: Iterable[_State]):
    state_logs = []
    for i, s in enumerate(states):
        cost_log_str_indented = "\n".join(
            f"  {line}" for line in s.cost_log_str.split("\n")
        )
        state_logs.append(
            f"BEAM STATE {i:03d}\n\n"
            + op_pprint.pformat(s.schedule, show_cost=False)
            + "\n\n"
            + cost_log_str_indented
        )
    dest.write(f"\n\n--------\nSTEP {step_idx:03d}\n--------\n\n")
    dest.write("\n\n".join(state_logs))
