import copy
import logging
import random
from collections.abc import Sequence
from typing import Callable, Optional

import morello.impl.base
from . import common
from .. import op_pprint, pruning

logger = logging.getLogger(__name__)


class MaxRestartsExceeded(Exception):
    pass


def randomly_schedule_impl(
    root_impl: morello.impl.base.Impl,
    budget: Optional[int],
    memory_limits: Optional[Sequence[pruning.MemoryLimits]] = None,
    max_restarts: Optional[int] = None,
    skip_sliding=False,
) -> tuple[Optional[morello.impl.base.Impl], int]:
    """Randomly schedule an Impl.

    Schedules the Impl by repeatedly choosing actions at random from the union
    of possible actions for all leaves. If the given Impl---or the Impl at any
    point during the search---is scheduled, returning that Impl is included as
    an action.

    :returns: A tuple of: either the scheduled Impl or None if the budget was exhausted
      and the number of steps taken.
    """
    # The default available memory is the capacities of current_system().
    if memory_limits is None:
        memory_limits = [pruning.StandardMemoryLimits()]
    else:
        memory_limits = list(memory_limits)

    child_limits = copy.deepcopy(memory_limits)

    steps_taken = 0
    impl = root_impl
    restarts = 0
    edge_case_restarts = 0
    while max_restarts is None or restarts < max_restarts:
        if budget is not None and budget <= steps_taken:
            assert steps_taken == budget
            return None, steps_taken

        acts_idxs: list[Optional[tuple[Callable, int]]] = list(
            common.hole_actions(
                impl, include_inner_impl=True, skip_sliding=skip_sliding
            )
        )
        if impl.is_scheduled:
            acts_idxs.append(None)

        if not acts_idxs:
            impl, child_limits = root_impl, copy.deepcopy(memory_limits)
            restarts += 1
            continue

        chosen_act_idx = random.choice(acts_idxs)
        if chosen_act_idx is None:
            return impl, steps_taken
        chosen_action, chosen_child_idx = chosen_act_idx
        impl, inner_impl = chosen_action()

        # Workaround for a rare edge case that only matters for the beam
        # heuristic. Doesn't affect results; we can just retry.
        # TODO: Fix root cause.
        if chosen_child_idx >= len(child_limits):
            logger.warning(
                "Hit a heuristic edge case; restarting with subtracting from "
                "max restarts\n"
                f"{chosen_child_idx} >= {len(child_limits)} for action (index: "
                f"{chosen_child_idx}): {str(chosen_action)}\nwhich produced Impl:\n"
                f"{str(op_pprint.pformat(impl, show_cost=False))}"
            )
            edge_case_restarts += 1
            if edge_case_restarts > 10:
                logger.warning("Saw %s edge case restarts", edge_case_restarts)
            continue

        new_child_limits = child_limits[chosen_child_idx].transition(inner_impl)
        if new_child_limits is None:
            impl, child_limits = root_impl, copy.deepcopy(memory_limits)
            restarts += 1
            continue

        child_limits = (
            child_limits[:chosen_child_idx]
            + new_child_limits
            + child_limits[chosen_child_idx + 1 :]
        )

        steps_taken += 1

    assert restarts == max_restarts
    raise MaxRestartsExceeded(f"Exceeded max restarts: {max_restarts}")
