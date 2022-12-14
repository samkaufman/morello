import contextvars
import dataclasses
import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Tuple

import cython

if cython.compiled:
    from cython.cimports.libc import limits
    from cython.cimports.morello import cost
else:
    import sys


if TYPE_CHECKING:
    from morello.impl.base import Impl

from .. import cost, pruning, specs, system_config
from ..impl import actions


class ActionFailedException(Exception):
    def __init__(self, act) -> None:
        super().__init__(f"Failed to call action: {act}")


class SearchCallbacks:
    def enter_unseen(self, spec: specs.Spec, limits: pruning.MemoryLimits) -> None:
        """Called when scheduling an uncached spec."""
        pass

    def expanded_hole(self, impl: "Impl") -> None:
        pass

    def visit_impl(self, spec: specs.Spec, imp: "Impl", impl_cost: int) -> None:
        pass

    def exit(self, spec: specs.Spec, best) -> None:
        """Called when finished scheduling an uncached spec."""
        pass


@cython.dataclasses.dataclass(frozen=False)
@cython.cclass
class SearchStats:
    expansions: int = 0


@cython.ccall
def schedule_key(schedule: "Impl") -> Tuple[int, Sequence[int], Any]:
    """Returns a key for ordering schedules during search.

    The returned key is a tuple of the schedule cost, peak memory usage, and
    schedule depth. In Python, tuples are compared by their first non-equal
    term, so this key can be used to select a schedule with the lowest cost
    with peak memory and then syntactic depth as tie-breakers.
    """
    system = system_config.current_system()
    base_cost = limits.LONG_MAX if cython.compiled else sys.maxsize
    if schedule.is_scheduled:
        base_cost = cost.compute_cost(schedule)
    peaks = tuple([schedule.peak_memory[b] for b in system.ordered_banks])
    return base_cost, peaks, schedule.depth


# TODO: Consider making this a default implementation of `actions` itself.
# TODO: Remove skip_sliding in favor of a global ContextVar for actions
def leaf_actions(
    root: "Impl", include_inner_impl=False, skip_sliding=False
) -> tuple[Sequence[tuple[Callable, int]], int]:
    """Yields callables for each action possible in root's leaves.

    More specifically, this yields from a concatenation of all leaves' actions wrapped
    in functions that yield, not the result of the action itself, but `root` with the
    result substituted for the leaf.
    """
    if not root.children:
        to_return = []
        for act in root.actions():
            if skip_sliding and isinstance(act, actions.SlidingTileOutAction):
                continue
            if include_inner_impl:
                to_return.append((functools.partial(_double_result, act), 0))
            else:
                to_return.append((act, 0))
        return to_return, 1

    to_return = []
    idx_offset = 0
    for child_idx, child in enumerate(root.children):
        inner, inner_leaf_cnt = leaf_actions(
            child, include_inner_impl=include_inner_impl, skip_sliding=skip_sliding
        )
        for action, sub_idx in inner:
            to_return.append(
                (
                    _WrappedRecurAction(root, include_inner_impl, child_idx, action),
                    sub_idx + idx_offset,
                )
            )
        idx_offset += inner_leaf_cnt
    return to_return, idx_offset


def _double_result(action):
    a = action()
    return a, a


@dataclasses.dataclass(frozen=True)
class _WrappedRecurAction:
    # This class could just be a partially applied (nullary) function, but
    # debugging is a lot easier with a custom __str__ implementation.
    root: "Impl"
    include_inner_impl: bool
    idx: int
    action: Callable

    def __call__(self) -> Any:
        if self.include_inner_impl:
            new_impl, inner_impl = self.action()
            children = list(self.root.children)
            children[self.idx] = new_impl
            r = self.root.replace_children(children)
            return r, inner_impl
        else:
            children = list(self.root.children)
            children[self.idx] = self.action()
            return self.root.replace_children(children)

    def __str__(self) -> str:
        return f"R[{self.action}]"
