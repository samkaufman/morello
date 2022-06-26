import dataclasses
import itertools
from typing import Any, List, Optional, Tuple

import cython

try:
    from cython.cimports.morello import specs

    from ..cython.cimports import common
except ImportError:
    pass

from .. import impl, layouts, pruning, specs
from ..impl import Impl
from ..search_cache import CachedScheduleSet, ScheduleCache
from . import common


def schedule_search(
    spec: specs.Spec,
    memory_limits=None,
    cache=None,
    top_k=None,
    parent_summary=None,
    stats=None,
    callbacks=None,
    return_extra: bool = False,
):
    """Returns the best Impl for a given Spec and memory limits.

    May return `None` if no Impl satisfies the given Spec and memory limits.
    """
    # If no cache is provided, initialize a new cache
    if cache is None:
        cache = ScheduleCache()

    # The default available memory is the capacities of current_system().
    # These capacities are measured in cache lines, not words, so: multiply by
    # line size when initializing the limit.
    if memory_limits is None:
        memory_limits = pruning.StandardMemoryLimits()

    inner_result = _inner_schedule_search(
        spec,
        memory_limits,
        cache,
        top_k=(top_k if top_k is not None else 1),
        parent_summary=parent_summary,
        stats=stats,
        callbacks=callbacks,
    )

    if return_extra:
        return inner_result
    if top_k is not None:
        return inner_result.impls
    if len(inner_result.impls):
        return inner_result.impls[0]
    return None


@cython.dataclasses.dataclass(frozen=True)
@cython.cclass
class SearchResult:
    impls: List[impl.Impl]
    dependent_paths: int


@cython.cfunc
def _inner_schedule_search(
    spec: specs.Spec,
    memory_limits: pruning.MemoryLimits,
    cache: ScheduleCache,
    top_k: int,
    parent_summary=None,
    stats: Optional[common.SearchStats] = None,
    callbacks=None,
) -> SearchResult:
    """Implements most of the logic of schedule_search.

    Returns a list of Impls which satisfy the given Spec and memory limits,
    sorted in order of increasing cost, up to `top_k` results. This is the empty
    list if no Impls satisfy the given Spec and memory bounds.
    """

    if stats is not None:
        stats.expansions += 1

    # Do a cache lookup. There are three outcomes: (a) a cached schedule is present and
    # can be returned, (b) the cache knows that no Impl satisfying the given limits
    # exists and we can propagate that fact to the caller, and (c) the cache has no
    # information for us and search can proceed.
    try:
        cache_result = cache.get(spec, memory_limits)
    except KeyError:
        pass
    else:
        assert cache_result is not None  # TODO: Remove
        return SearchResult(
            [im for im, _ in cache_result.contents],
            cache_result.dependent_paths
        )

    if callbacks is not None:
        callbacks.enter_unseen(spec, memory_limits)

    # Create a an Impl hole corresponding to the query spec
    leaf = impl.spec_to_hole(spec)
    assert leaf.depth == 1, f"Expected hole to have depth 1; had {leaf.depth}"

    # A generator of expansions of `leaf`. This will be wrapped with `_best_schedule`.
    best_results, specs_explored_by_options = _best_options(
        spec,
        leaf,
        memory_limits,
        cache,
        top_k,
        parent_summary=parent_summary,
        stats=stats,
        callbacks=callbacks,
    )
    specs_explored = specs_explored_by_options + 1

    if callbacks is not None:
        if best_results:
            callbacks.exit(spec, [(r[0], r[1][0]) for r in best_results])
        else:
            callbacks.exit(spec, None)

    cache.put(
        spec,
        CachedScheduleSet(tuple((im, c) for im, (c, _, _) in best_results), specs_explored),
        memory_limits,
    )
    return SearchResult(
        [im for im, _ in best_results], specs_explored
    )


@cython.cfunc
def _best_options(
    spec: specs.Spec,
    leaf: impl.Impl,
    memory_limits: pruning.MemoryLimits,
    cache: ScheduleCache,
    top_k: int,
    parent_summary=None,
    stats=None,
    callbacks=None,
) -> Tuple[List[Tuple[Impl, tuple]], int]:
    """Returns top-k best Impls after taking any of leaf's actions (if any).
    
    Also returns the number of unique Specs explored.
    """
    unique_specs_visited = 0
    best_results: list[tuple[Impl, Any]] = []

    # If the leaf is itself scheduled, yield it (i.e. no action) as an option.
    if all(m >= 0 for m in memory_limits.available.values()) and leaf.is_scheduled:
        _update_best_results(best_results, leaf, spec, callbacks)

    # Yield all the complete expansions of the hole by expanding once into
    # an Impl which may or may not have its own holes. If it does have its
    # own holes, fill them by recursively calling into schedule_search.
    for act in leaf.actions(parent_summary=parent_summary):
        try:
            new_tree = act()
        except impl.ActionOutOfDomain:
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

        new_parent_summary = impl.ParentSummary.update(parent_summary, parent=new_tree)

        # Ignore the action if it uses more memory than is available for any hole.
        new_child_memory_limits = memory_limits.transition(new_tree)
        if new_child_memory_limits is None:
            continue

        # Recurse for all holes (nested Specs) in the new Impl. If any hole cannot
        # be filled, short-circuit because this action is a dead end.
        subsearch_results: list[list[Impl]] = []
        for child, mem in zip(new_tree.children, new_child_memory_limits):
            child_result = _inner_schedule_search(
                child.spec,
                cache=cache,
                top_k=top_k,
                memory_limits=mem,
                parent_summary=new_parent_summary,
                stats=stats,
                callbacks=callbacks,
            )
            unique_specs_visited += child_result.dependent_paths
            # If any hole could not be filled--this can happen, for instance, if
            # every possible action uses too much memory--then exit the outer loop.
            if not child_result.impls:
                break
            subsearch_results.append(child_result.impls)
        if len(subsearch_results) < len(new_tree.children):
            continue

        # Yield the product of all possible child Impls as options.
        # TODO: There's almost certainly a smarter way to enumerate these.
        for selected_children in itertools.product(*subsearch_results):
            completed = new_tree.replace_children(selected_children)
            assert (
                completed.spec == new_tree.spec
            ), f"{str(completed.spec)} != {str(new_tree.spec)}"
            _update_best_results(best_results, completed, spec, callbacks)

    return _finalize_best_results(best_results, top_k), unique_specs_visited


@cython.cfunc
def _update_best_results(
    results: list[tuple[Impl, Any]], new_impl: impl.Impl, spec, callbacks
):
    # TODO: Actually necessary to pass spec *and* new_impl?
    cost_tuple = common.schedule_key(new_impl)
    if callbacks:
        callbacks.visit_impl(spec, new_impl, cost_tuple[0])
    results.append((new_impl, cost_tuple))


@cython.cfunc
def _finalize_best_results(
    results: list[tuple[Impl, Any]], top_k: int
) -> List[Tuple[Impl, Any]]:
    # Using sorted here for stability.
    return sorted(results, key=lambda x: x[1])[:top_k]
    # return heapq.nsmallest(top_k, results, key=lambda x: x[1])
