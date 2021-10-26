import dataclasses
import sys
from collections import Sequence, Iterable
from typing import Optional, Union, Callable

import numpy as np

from . import common
from .. import cost, specs, pruning, ops


def _cost(schedule: ops.Schedule) -> Union[int, float]:
    return cost.analytical_cost(schedule, holes_ok=True)


@dataclasses.dataclass(frozen=True)
class _State:
    schedule: ops.Schedule
    child_limits: tuple[pruning.MemoryLimits]


def beam_schedule_search(
    spec: specs.Spec,
    inputs: tuple,
    output,
    k: int = 10000,
    budget: Optional[int] = None,
    stats: Optional[common.SearchStats] = None,
) -> Optional[ops.Schedule]:
    if common.prune_column_major.get():
        if any(inp.layout == specs.Layout.COL_MAJOR for inp in spec.inputs):
            return None
        if spec.output.layout == specs.Layout.COL_MAJOR:
            return None

    # Initialize the beam search with a single state: an Impl hole for the query Spec.
    best_found: tuple[Optional[ops.Schedule], Union[int, float]] = None, sys.maxsize
    states: list[_State] = [
        _State(
            ops.spec_to_hole(spec, inputs, output), (pruning.StandardMemoryLimits(),)
        )
    ]
    while True:
        # Only retain scheduled implementations through to the next step.
        new_states: list[_State] = []
        for state in states:
            for act, child_idx in _leaf_actions(state.schedule):
                child_limits = state.child_limits[child_idx]
                introduced = act()
                introduced_limits = child_limits.transition(introduced)

                # Ignore the action if it exceeds our memory limits.
                if introduced_limits is not None:
                    introduced_cost = _cost(introduced)
                    new_states.append(_State(introduced, introduced_limits))
                    if (
                        new_states[-1].schedule.is_scheduled
                        and introduced_cost < best_found[1]
                    ):
                        best_found = introduced, introduced_cost

                # Decrement budget even if the action exceeds memory limits.
                if budget is not None:
                    if budget == 0:
                        return best_found[0]
                    budget -= 1
                stats.expansions += 1

        if not new_states:
            return best_found[0]
        new_states = _select_new_states(new_states, k)
        if new_states == states:
            return best_found[0]
        states = new_states


def _leaf_actions(root: ops.Schedule) -> Iterable[tuple[Callable, int]]:
    """Yields callables for each action possible in root's leaves.

    More specifically, this yields from a concatenation of all leaves' actions wrapped
    in functions that yield, not the result of the action itself, but `root` with the
    result substituted for the leaf.
    """
    if not root.children:
        return root.actions()
    for idx, c in enumerate(root.children):
        for action in _leaf_actions(c):
            yield lambda i=idx, a=action: root.replace_child(i, a())


def _select_new_states(states: Sequence[_State], k: int) -> list[_State]:
    """Stochastically select `k` states from possibilities."""
    costs = np.array([_cost(s.schedule) for s in states])

    # TODO: Exponentiate and smooth the costs?
    p = costs / costs.sum()
    p = p / p.sum()

    idxs = np.random.choice(np.arange(len(states)), k, replace=True, p=p)
    return [states[i] for i in idxs]
