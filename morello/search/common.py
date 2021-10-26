import contextvars
import sys
from collections import Sequence
from typing import Any
import dataclasses

from .. import cost, ops, system_config

prune_column_major: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "prune_column_major", default=False
)


class ActionFailedException(Exception):
    def __init__(self, act) -> None:
        super().__init__(f"Failed to call action: {act}")


@dataclasses.dataclass(frozen=False)
class SearchStats:
    expansions: int = 0


def schedule_key(schedule: ops.Schedule) -> tuple[int, Sequence[int], Any]:
    """Returns a key for ordering schedules during search.

    The returned key is a tuple of the schedule cost, peak memory usage, and
    schedule depth. In Python, tuples are compared by their first non-equal
    term, so this key can be used to select a schedule with the lowest cost
    with peak memory and then syntactic depth as tie-breakers.
    """
    system = system_config.current_system()
    base_cost = sys.maxsize
    if schedule.is_scheduled:
        base_cost = cost.analytical_cost(schedule)
    peaks = [schedule.peak_memory[b] for b in system.ordered_banks]
    return base_cost, tuple(peaks), schedule.depth
