import functools
from typing import Any, Generator, Iterable, Optional

from morello import system_config

from . import common
from .. import ops, pruning, replace, specs
from ..ops import Schedule
from ..search_cache import CachedSchedule, ScheduleCache


def _best_schedule(
    it: Iterable[Schedule],
) -> Optional[tuple[Schedule, tuple[int, Any, Any, Any]]]:
    """Returns the best schedule if `it` is non-empty; `None` if it is.

    This uses schedule_key, so it will return the lowest cost schedule,
    breaking ties as described in schedule_key's docstring.
    """
    it = ((s, common.schedule_key(s)) for s in it)
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
    inputs: Optional[tuple] = None,
    output: Optional[Any] = None,
    memory_limits: Optional[pruning.MemoryLimits] = None,
    cache: Optional[ScheduleCache] = None,
    parent_summary: Optional[ops.ParentSummary] = None,
    stats: Optional[common.SearchStats] = None,
    callbacks: Optional[common.SearchCallbacks] = None,
) -> Optional[Schedule]:
    """Returns the best Impl for a given Spec and memory limits.

    May return `None` if no Impl satisfies the given Spec and memory limits.
    """
    if inputs is None:
        target = system_config.current_target()
        inputs = tuple(target.tensor(s) for s in spec.inputs)
    if output is None:
        output = system_config.current_target().tensor(spec.output)

    if common.prune_column_major.get():
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

    if stats:
        stats.expansions += 1

    # Do a cache lookup. There are three outcomes: (a) a cached schedule is present and
    # can be returned, (b) the cache knows that no Impl satisfying the given limits
    # exists and we can propagate that fact to the caller, and (c) the cache has no
    # information for us and search can proceed.
    try:
        wrapped_cached_schedule = cache.get(spec, memory_limits)
        if wrapped_cached_schedule is None:
            return None
    except KeyError:
        pass
    else:
        # Substitute in the correct operands. This is needed because schedules are
        # stored in the cache along with their concrete operands. While these operands
        # have the same TensorSpecs as those of our query Spec, they aren't the same
        # objects, so: swap in the query operands before returning the cached schedule.
        return _subs_query_operands(spec, inputs, output, wrapped_cached_schedule)

    if callbacks:
        callbacks.enter_unseen(spec, memory_limits)

    # Create a an Impl hole corresponding to the query spec
    leaf = ops.spec_to_hole(spec, inputs, output)
    assert leaf.depth == 1, f"Expected hole to have depth 1; had {leaf.depth}"

    # A generator of expansions of `leaf`. This will be wrapped with `_best_schedule`.
    def yield_options() -> Generator[Schedule, None, None]:
        """Yields best Impls after taking any of leaf's actions (or no action)."""
        # If the leaf is itself scheduled, yield it (i.e. no action) as an option.
        if all(m >= 0 for m in memory_limits.available.values()) and leaf.is_scheduled:
            yield leaf

        # Yield all the complete expansions of the hole by expanding once into
        # an Impl which may or may not have its own holes. If it does have its
        # own holes, fill them by recursively calling into schedule_search.
        for act in leaf.actions(parent_summary=parent_summary):
            try:
                new_tree = act()
            except ops.ActionOutOfDomain:
                continue
            except Exception as e:
                # Re-raise the exception with a little more detail about act.
                raise common.ActionFailedException(act) from e
            assert new_tree.spec == spec, f"{str(new_tree.spec)} != {str(spec)}"
            assert new_tree != leaf, (
                f"Action returned self: {new_tree}; spec = {str(new_tree.spec)}; "
                f"action = {act}"
            )

            if callbacks:
                callbacks.applied_action(act, new_tree)

            new_parent_summary = ops.ParentSummary.update(
                parent_summary, parent=new_tree
            )

            # Ignore the action if it uses more memory than is available for any hole.
            new_child_memory_limits = memory_limits.transition(new_tree)
            if new_child_memory_limits is None:
                continue

            # Recurse for all holes (nested Specs) in the new Impl. If any hole cannot
            # be filled, short-circuit because this action is a dead end.
            subsearch_results = []
            for child, mem in zip(new_tree.children, new_child_memory_limits):
                child_result = schedule_search(
                    child.spec,
                    inputs=child.inputs,
                    output=child.output,
                    cache=cache,
                    memory_limits=mem,
                    parent_summary=new_parent_summary,
                    stats=stats,
                    callbacks=callbacks,
                )
                # If any hole could not be filled--this can happen, for instance, if
                # every possible action uses too much memory--then exit the outer loop.
                if child_result is None:
                    break
                subsearch_results.append(child_result)
            if len(subsearch_results) < len(new_tree.children):
                continue

            # Fill the holes and yield as a possible Impl
            completed = new_tree.replace_children(subsearch_results)
            assert (
                completed.spec == new_tree.spec
            ), f"{str(completed.spec)} != {str(new_tree.spec)}"

            yield completed

    best_result = _best_schedule(yield_options())
    if callbacks:
        callbacks.exit(spec, best_result[1][0] if best_result else None)

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


def _subs_query_operands(spec, inputs, output, wrapped_cached_schedule):
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
