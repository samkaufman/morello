import contextvars
import functools
import sys
import warnings
from typing import Any, Callable, Generator, Iterable, Optional

from . import cost, ops, pruning, replace, specs, tiling
from .ops import Schedule
from .search_cache import CachedSchedule, ScheduleCache

prune_column_major: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "prune_column_major", default=False
)


class ActionFailedException(Exception):
    def __init__(self, act) -> None:
        super().__init__(f"Failed to call action: {act}")


# TODO: Remove this once grid_search.py no longer needs it
def apply_or_max(fn: Callable[..., int], schedule: Schedule, *args, **kwargs) -> int:
    """Applies function, or returns maxsize if `schedule` unscheduled or over memory.

    This function is useful for searches to assign an effectively infinite cost to
    un-executable implementations.
    """
    if not schedule.is_scheduled:
        return sys.maxsize
    return fn(schedule, *args, **kwargs)


def _schedule_key(schedule: Schedule) -> tuple[int, Any, Any]:
    """Returns a key for ordering schedules during search.

    The returned key is a tuple of the schedule cost, peak memory usage, and
    schedule depth. In Python, tuples are compared by their first non-equal
    term, so this key can be used to select a schedule with the lowest cost
    with peak memory and then syntactic depth as tie-breakers.
    """
    base_cost = sys.maxsize
    if schedule.is_scheduled:
        base_cost = cost.analytical_cost(schedule)
    return base_cost, tuple(schedule.peak_memory), schedule.depth


def _best_schedule(
    it: Iterable[Schedule],
) -> Optional[tuple[Schedule, tuple[int, Any, Any, Any]]]:
    """Returns the best schedule if `it` is non-empty; `None` if it is.

    This uses _schedule_key, so it will return the lowest cost schedule,
    breaking ties as described in _schedule_key's docstring.
    """
    it = ((s, _schedule_key(s)) for s in it)
    return min(it, key=lambda x: x[1], default=None)


# TODO: Don't just use a global variable. Use a thread local or contextvars.
_specs_on_stack = []


def _assert_no_cycles_in_stack(func):
    """Decorates schedule_search to ensure no spec ever occurs twice on the stack."""
    global _specs_on_stack

    if not __debug__:
        return func

    # noinspection PyUnreachableCode
    @functools.wraps(func)
    def assert_no_cycles_in_stack_inner(*args, **kwargs):
        spec = args[0]
        assert isinstance(spec, specs.Spec)
        if spec in _specs_on_stack:
            raise Exception(
                f"Spec {spec} was in the stack:"
                + "".join(f"\n - {s}" for s in _specs_on_stack)
            )
        _specs_on_stack.append(spec)
        try:
            return func(*args, **kwargs)
        finally:
            popped = _specs_on_stack.pop()
            assert popped == spec, "spec on _specs_on_stack doesn't match real stack"

    return assert_no_cycles_in_stack_inner


@_assert_no_cycles_in_stack
def schedule_search(
    spec: specs.Spec,
    inputs: tuple,
    output,
    memory_limits: Optional[pruning.MemoryLimits] = None,
    cache: Optional[ScheduleCache] = None,
    parent_summary: Optional[ops.ParentSummary] = None,
) -> Optional[Schedule]:
    """Find a cool schedule for the given spec.

    :param search_depth: The search depth. A returned schedule might be deeper if a
        hole is expanded via `complete` or a cache lookup.
    """
    if prune_column_major.get():
        if any(inp.layout == specs.Layout.COL_MAJOR for inp in spec.inputs):
            return None
        if spec.output.layout == specs.Layout.COL_MAJOR:
            return None

    # If no cache is provided, initialize a new cache
    if cache is None:
        cache = ScheduleCache()

    # The default available memory is the capacities of current_system().
    # These capacities are measured in cache lines, not words, so: multiply by
    # line size when initializing the limit.
    if memory_limits is None:
        memory_limits = pruning.StandardMemoryLimits()

    # Do a cache lookup
    cached_schedule: Optional[Schedule] = None
    try:
        wrapped_cached_schedule = cache.get(spec, memory_limits)
        if wrapped_cached_schedule is None:
            return None
    except KeyError:
        pass
    else:
        # Substitute in the correct operands.
        cached_schedule = wrapped_cached_schedule.schedule
        assert len(inputs) == len(cached_schedule.inputs)
        operand_replacements = dict(zip(cached_schedule.inputs, inputs))
        operand_replacements[cached_schedule.output] = output
        cached_schedule = replace.replace(cached_schedule, operand_replacements)
        assert cached_schedule.spec == spec, (
            "spec doesn't match query after operand replacement; expected "
            f"{spec} but was {cached_schedule.spec}"
        )
        return cached_schedule

    # Create a an Impl hole corresponding to the query spec
    leaf = ops.spec_to_hole(spec, inputs, output)
    assert leaf.depth == 1, f"Expected hole to have depth 1; had {leaf.depth}"

    # Return the best option
    def yield_options() -> Generator[Schedule, None, None]:
        """Returns Impls which implement the query spec."""

        # If the leaf is itself scheduled, yield it (i.e. no action) as an option.
        if all(m >= 0 for m in memory_limits.available.values()) and leaf.is_scheduled:
            yield leaf

        # Yield all the complete expansions of the hole by expanding once into
        # an Impl which may or may not have its own holes. If it does have its
        # own holes, fill them by recursively calling into schedule_search.
        for act in leaf.actions(parent_summary=parent_summary):
            try:
                new_tree = act()
            except tiling.UnimplementedCompositionError as e:
                # This is a temporary workaround until I can implement Convolution
                # and PartialConvolutionImageTile
                warnings.warn("Skipping a composed tile_out: " + str(e))
                continue
            except Exception as e:
                # Re-raise the exception with a little more detail about act.
                raise ActionFailedException(act) from e
            assert new_tree.spec == spec, f"{str(new_tree.spec)} != {str(spec)}"
            if new_tree == leaf:
                warnings.warn(
                    f"Action returned self: {new_tree}; spec = {str(new_tree.spec)}; action = {act}"
                )
                continue

            # Ignore the action if it uses more memory than is available for any
            # hole.
            new_child_memory_limits = memory_limits.transition(new_tree)
            if new_child_memory_limits is None:
                continue

            # Repeatedly expand all holes in new_tree until no holes remain.
            subsearch_results = [
                schedule_search(
                    child.spec,
                    inputs=child.inputs,
                    output=child.output,
                    cache=cache,
                    memory_limits=mem,
                    parent_summary=ops.ParentSummary.update(
                        parent_summary, parent=new_tree
                    ),
                )
                for child, mem in zip(new_tree.children, new_child_memory_limits)
            ]

            # If any hole could not be filled--this can happen, for instance, if
            # every possible action uses too much memory.
            if any(r is None for r in subsearch_results):
                continue

            # Fill the holes and yield as a possible Impl
            completed = new_tree.replace_children(subsearch_results)
            assert (
                completed.spec == new_tree.spec
            ), f"{str(completed.spec)} != {str(new_tree.spec)}"
            yield completed

    best_result = _best_schedule(yield_options())
    if best_result is not None:
        schedule_to_return, (cost_ret, _, _) = best_result
        assert (
            schedule_to_return.spec == spec
        ), f"{str(schedule_to_return.spec)} != {str(spec)}"
        cache.put(
            spec, CachedSchedule(schedule_to_return, cost=cost_ret), memory_limits
        )
        return best_result[0]
    else:
        cache.put(spec, None, memory_limits)
        return None
