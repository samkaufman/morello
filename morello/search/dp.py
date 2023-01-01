import asyncio
import itertools
import typing
from typing import Any, Generator, Iterable, List, Optional, Sequence

import cython

try:
    from ..cython.cimports import common, specs  # type: ignore
except ImportError:
    pass

from .. import impl, pruning, specs
from ..impl import Impl
from ..search_cache import (
    CachedScheduleSet,
    InMemoryScheduleCache,
    ScheduleCache,
    assert_access_on_log_boundaries,
)
from ..utils import snap_availables_down
from . import common


def schedule_search(
    spec: specs.Spec,
    memory_limits=None,
    cache=None,
    top_k=1,
    parent_summary=None,
    callbacks=None,
) -> list[Impl]:
    """Returns the best Impl for a given Spec and memory limits.

    May return `None` if no Impl satisfies the given Spec and memory limits.
    """
    return asyncio.run(
        Search(top_k, callbacks=callbacks)(spec, memory_limits, cache, parent_summary)
    )


@cython.dataclasses.dataclass(frozen=True)
@cython.cclass
class SearchResult:
    impls: List[impl.Impl]
    dependent_paths: int


class SearchMessage(typing.NamedTuple):
    needed: Sequence[tuple[specs.Spec, pruning.MemoryLimits]]
    computed: Sequence[tuple[pruning.MemoryLimits, CachedScheduleSet]]


SearchResponse = Sequence[Optional[SearchResult]]


class Search:
    def __init__(self, top_k=1, callbacks=None) -> None:
        # If no cache is provided, initialize a new cache
        self.top_k = top_k
        self.callbacks = callbacks

    @typing.final
    async def __call__(
        self,
        spec: specs.Spec,
        memory_limits=None,
        parent_summary=None,
        cache: Optional[ScheduleCache] = None,
        stats=None,
        complete: bool = True,
    ) -> list[impl.Impl]:
        """Returns the best Impl for a given Spec and memory limits.

        May return `None` if no Impl satisfies the given Spec and memory limits.
        """
        if complete:
            raise NotImplementedError("complete=True not implemented")

        if cache is None:
            cache = InMemoryScheduleCache()
        else:
            cache = cache

        # The default available memory is the capacities of current_system().
        # These capacities are measured in cache lines, not words, so: multiply
        # by line size when initializing the limit.
        if memory_limits is None:
            memory_limits = pruning.StandardMemoryLimits()

        assert (
            not assert_access_on_log_boundaries.get()
            or snap_availables_down(memory_limits.available) == memory_limits.available
        )

        hole = impl.spec_to_hole(spec)
        assert hole.spec == spec
        search_gen = self._search(
            hole,
            memory_limits=memory_limits,
            parent_summary=parent_summary,
            stats=stats,
        )
        msg = next(search_gen)
        try:
            cache_response = await cache.get_many(msg.needed)
            for mlims, entry in msg.computed:
                await cache.put(entry, mlims)
            msg = search_gen.send(list(cache_response))
        except StopIteration as e:
            return e.value.impls
        assert False, "Should not reach here"

    # TODO: Make a public method.
    # TODO: Don't need a return type, just send and yield.
    # TODO: Do we need the Impls, or just the costs, provided?
    def _search(
        self,
        hole: impl.Impl,  # TODO: Take a Spec, not a hole
        memory_limits: pruning.MemoryLimits,
        parent_summary: Optional[impl.ParentSummary] = None,
        stats: Optional[common.SearchStats] = None,
    ) -> Generator[SearchMessage, SearchResponse, SearchResult]:
        """Implements most of the logic of schedule_search.

        Returns a list of Impls which satisfy the Spec of given `hole` and memory
        limits, sorted in order of increasing cost, up to `top_k` results. This is the
        empty list if no Impls satisfy the given Spec and memory bounds.
        """
        assert hole.depth == 1, f"Expected hole to have depth 1; had {hole.depth}"

        if stats is not None:
            stats.expansions += 1

        unique_specs_visited = 1
        reducer = _ImplReducer(self.top_k)

        # If the leaf is itself scheduled, yield it (i.e. no action) as an option.
        if all(m >= 0 for m in memory_limits.available.values()) and hole.is_scheduled:
            reducer(hole, hole.spec, self.callbacks)

        # First give the caller the opportunity to provide a cached Impl.  There are
        # three outcomes: (a) a cached schedule is present and can be returned, (b) the
        # cache knows that no Impl satisfying the given limits exists and we can
        # propagate that fact to the caller, and (c) the cache has no information for us
        # and search can proceed.
        # TODO: Push cache query down and call `get_many`.
        caller_response = yield SearchMessage(((hole.spec, memory_limits),), [])
        cache_result = caller_response[0]
        if cache_result is not None:
            # TODO: Should we send the computed Impl *back* to the caller?
            return SearchResult(
                [im for im, _ in cache_result.contents], cache_result.dependent_paths
            )

        # Yield all the complete expansions of the hole by expanding once into
        # an Impl which may or may not have its own holes. If it does have its
        # own holes, fill them by recursively calling into schedule_search.
        for new_tree in self._iter_expansions(hole, parent_summary):
            if self.callbacks:
                self.callbacks.expanded_hole(new_tree)

            # Skip actions using more memory than is available for any hole.
            new_child_memory_limits = memory_limits.transition(new_tree)
            if new_child_memory_limits is None:
                continue

            # Recurse for all holes (nested Specs) in the new Impl. If any hole
            # cannot be filled, short-circuit because this action is a dead end.
            subsearch_results: list[list[Impl]] = []
            for child, mem in zip(new_tree.children, new_child_memory_limits):
                child_result = yield from self._search(
                    child,
                    memory_limits=mem,
                    parent_summary=impl.ParentSummary.update(
                        parent_summary, parent=new_tree
                    ),
                    stats=stats,
                )
                unique_specs_visited += child_result.dependent_paths
                # If any hole could not be filled--this can happen, for
                # instance, if every possible action uses too much memory--then
                # exit the outer loop.
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
                reducer(completed, hole.spec, self.callbacks)

        best_results = reducer.finalize()
        assert len(best_results) <= self.top_k

        yield SearchMessage(
            [],
            [
                (
                    memory_limits,
                    CachedScheduleSet(
                        hole.spec,
                        tuple((im, c) for im, (c, _, _) in best_results),
                        unique_specs_visited,
                    ),
                )
            ],
        )
        return SearchResult([im for im, _ in best_results], unique_specs_visited)

    def _iter_expansions(
        self, leaf: impl.Impl, parent_summary: Optional[impl.ParentSummary]
    ) -> Iterable[Impl]:
        """Iterate pairs of action callables and their resulting expanded Impls."""
        for act in leaf.actions(parent_summary=parent_summary):
            try:
                new_tree = act()
            except impl.ActionOutOfDomain:
                continue
            except Exception as e:
                # Re-raise the exception with a little more detail about act.
                raise common.ActionFailedException(act) from e
            assert (
                new_tree.spec == leaf.spec
            ), f"{str(new_tree.spec)} != {str(leaf.spec)}"
            assert new_tree != leaf, (
                f"Action returned self: {new_tree}; spec = {str(new_tree.spec)}; "
                f"action = {act}"
            )
            yield new_tree


class _ImplReducer:
    __slots__ = ["results", "top_k"]

    results: list[tuple[Impl, Any]]
    top_k: int

    def __init__(self, top_k: int):
        self.results = []
        self.top_k = top_k

    def __call__(self, new_impl: impl.Impl, spec: specs.Spec, callbacks):
        # TODO: Actually necessary to pass spec *and* new_impl?
        assert new_impl.spec == spec, f"{str(new_impl.spec)} != {str(spec)}"
        self.results.append((new_impl, common.schedule_key(new_impl)))

    def finalize(self) -> list[tuple[Impl, Any]]:
        # Using sorted here for stability.
        return sorted(self.results, key=lambda x: x[1])[: self.top_k]
        # return heapq.nsmallest(top_k, results, key=lambda x: x[1])
