import dataclasses
import functools
from collections.abc import Sequence
from typing import Any, Callable

import cython

if cython.compiled:
    from cython.cimports.morello import cost

from .. import cost, system_config, utils
from ..impl import Impl, actions


class ActionFailedException(Exception):
    def __init__(self, act) -> None:
        super().__init__(f"Failed to call action: {act}")


class SearchCallbacks:
    def expanded_hole(self, impl: "Impl") -> None:
        pass


@cython.dataclasses.dataclass(frozen=False)
@cython.cclass
class SearchStats:
    expansions: int = 0


# TODO: Just merge this into the cost model. Key and cost distinction is useless.
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class ScheduleKey:
    """A key for ordering schedules during search.

    Can be seen as the "true" cost of an Impl.

    When ordering ScheduleKeys, the `MainCost` will be compared first, with
    ties broken by minimizing memory peaks and then Impl depth.
    """

    main_cost: cost.MainCost
    peaks: tuple[int, ...]
    depth: int

    @staticmethod
    def from_complete_impl(imp: "Impl", cache: bool = False) -> "ScheduleKey":
        cached_key = getattr(imp, "_schedule_key_cached", None)
        if cached_key:
            return cached_key

        if not imp.is_scheduled:
            raise ValueError("Impl must be complete")
        if not imp.children:
            r = ScheduleKey.from_child_keys(imp, [])
        else:
            r = ScheduleKey.from_child_keys(
                imp,
                [ScheduleKey.from_complete_impl(c, cache=cache) for c in imp.children],
            )
        if cache:
            object.__setattr__(imp, "_schedule_key_cached", r)
        return r

    # TODO: Make callers check memory correctness when calling this.
    @staticmethod
    def from_child_keys(
        imp: "Impl", child_keys: Sequence["ScheduleKey"]
    ) -> "ScheduleKey":
        banks = system_config.current_system().ordered_banks
        main_cost = cost.compute_cost_node(imp, [k.main_cost for k in child_keys])
        # TODO: Handle other kinds of memory, not just standard/TinyMap peaks.
        raised_peaks = Impl.peak_memory_from_child_peaks(
            imp.memory_allocated, [utils.TinyMap(banks, k.peaks) for k in child_keys]
        )
        raised_peaks = utils.snap_availables_up(raised_peaks)
        return ScheduleKey(
            main_cost=main_cost,
            peaks=raised_peaks.raw_values,
            depth=1 + max((k.depth for k in child_keys), default=0),
        )


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
