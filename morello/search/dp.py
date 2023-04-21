import asyncio
import collections
import itertools
import typing
from typing import Any, Generator, Iterable, List, Optional, Sequence, TYPE_CHECKING

import cython

try:
    from ..cython.cimports import common, specs  # type: ignore
except ImportError:
    pass

from .. import impl, pruning, specs
from ..impl import Impl
from ..search_cache import (
    CachedScheduleSet,
    ScheduleCache,
    assert_access_on_log_boundaries,
)
from ..utils import snap_availables_down
from . import common

if TYPE_CHECKING:
    from .. import cost


async def schedule_search(
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
    return await Search(top_k, callbacks=callbacks)(
        spec, memory_limits, parent_summary=parent_summary, cache=cache
    )


@cython.dataclasses.dataclass(frozen=True)
@cython.cclass
class SearchResult:
    impl_tuples: List[tuple[impl.Impl, common.ScheduleKey]]
    dependent_paths: int

    @property
    def impls(self) -> list[impl.Impl]:
        return [imp for imp, _ in self.impl_tuples]

    @property
    def costs(self) -> list["cost.MainCost"]:
        return [k.main_cost for _, k in self.impl_tuples]


class SearchMessage(typing.NamedTuple):
    needed: Sequence[tuple[specs.Spec, pruning.MemoryLimits]]
    computed: Sequence[tuple[pruning.MemoryLimits, CachedScheduleSet]]


SearchResponse = Sequence[Optional[CachedScheduleSet]]


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
    ) -> list[impl.Impl]:
        """Returns the best Impl for a given Spec and memory limits.

        May return `None` if no Impl satisfies the given Spec and memory limits.
        """
        if self.top_k > 1:
            raise NotImplementedError("Search for top_k > 1 not yet implemented.")

        if cache is None:
            cache = ScheduleCache()
        if memory_limits is None:
            memory_limits = pruning.StandardMemoryLimits()

        assert (
            not assert_access_on_log_boundaries.get()
            or snap_availables_down(memory_limits.available) == memory_limits.available
        )

        # TODO: The following does nothing to avoid multiple queries.
        partial_impl = impl.spec_to_hole(spec)
        search_gens = [
            self.interactive_search(
                partial_impl, memory_limits, parent_summary=parent_summary, stats=stats
            )
        ]
        msgs = [next(g) for g in search_gens]

        while search_gens:
            finished, msgs = await _step_search_generators(cache, search_gens, msgs)

            # Short-circuit if any subproblem couldn't be implemented.
            if any(not len(imps) for _, imps in finished):
                return []

            # We replace holes, not leaves, because we expect the search generators to
            # be zippable with holes, not leaves. (Schedule leaves have no generator.)
            partial_impl, new_leaves = partial_impl.replace_holes_by_index(
                [(i, l[0]) for i, l in finished]
            )

            # The following is slow. It traverses `partial_impl`.
            # TODO: Speed it up.
            new_leaf_mlims = list(
                _compute_memory_limits_for_leaves(partial_impl, memory_limits)
            )

            # Start generators for any newly inserted sub-Specs.
            # Iterate in reversed order so that we can modify search_gens and msgs
            # with stable indices.
            gen_offset = 0
            for (finished_idx, imps), new_leaves_idxs in zip(finished, new_leaves):
                assert len(imps) == 1

                new_gens = []
                for new_leaf_idx, new_leaf in zip(new_leaves_idxs, imps[0].leaves):
                    if new_leaf.is_scheduled:
                        continue
                    new_gens.append(
                        # TODO: Fill in parent summaries!
                        self.interactive_search(
                            new_leaf,
                            new_leaf_mlims[new_leaf_idx],
                            parent_summary=None,
                            stats=stats,
                        )
                    )

                search_gens[
                    finished_idx + gen_offset : finished_idx + gen_offset + 1
                ] = new_gens
                add_pt = finished_idx + gen_offset
                msgs[add_pt:add_pt] = [next(g) for g in new_gens]
                gen_offset += len(new_gens) - 1

        return [partial_impl]

    # TODO: Don't need a return type, just send and yield.
    # TODO: Do we need the Impls, or just the costs, provided?
    def interactive_search(
        self,
        hole: impl.Impl,  # TODO: Take a Spec, not a hole
        memory_limits: pruning.MemoryLimits,
        parent_summary: Optional[impl.ParentSummary] = None,
        stats: Optional[common.SearchStats] = None,
    ) -> Generator[SearchMessage, SearchResponse, SearchResult]:
        """Returns a search generator, with memoization provided by the caller.

        The generator returns a list of Impls which satisfy the Spec of given `hole` and
        memory limits, sorted in order of increasing cost, up to `top_k` results. This
        is the empty list if no Impls satisfy the given Spec and memory bounds.
        """
        assert hole.depth == 1, f"Expected hole to have depth 1; had {hole.depth}"

        if stats is not None:
            stats.expansions += 1

        unique_specs_visited = 1
        reducer = _ImplReducer(self.top_k)

        # First give the caller the opportunity to provide a cached Impl.  There are
        # three outcomes: (a) a cached schedule is present and can be returned, (b) the
        # cache knows that no Impl satisfying the given limits exists and we can
        # propagate that fact to the caller, and (c) the cache has no information for us
        # and search can proceed.
        # TODO: Push cache query down and call `get_many`.
        cache_results_all = yield SearchMessage(((hole.spec, memory_limits),), [])
        assert len(cache_results_all) == 1, f"{len(cache_results_all)} != 1"
        cache_result = cache_results_all[0]
        if cache_result is not None:
            # TODO: Document: we allow the caller to give an incomplete Impl, where
            #  there's no guarantee the result of interactive_search is a complete Impl.
            #  This can save compute and the requirements for the caller are clear:
            #  complete your cached Impls if that's what you want.
            #  This requires the caller to also provide costs for provided Impls.
            return SearchResult(cache_result.contents, cache_result.dependent_paths)

        # Collect all the sub-problem dependencies for a batched query.
        viable_subproblems, deps = self._collect_subproblems(
            hole, memory_limits, parent_summary
        )

        results: SearchResponse = yield SearchMessage(deps, [])
        assert len(results) == len(deps), f"{len(results)} != {len(deps)}"
        results_consumed = 0

        # Yield all the complete expansions of the hole by expanding once into
        # an Impl which may or may not have its own holes. If it does have its
        # own holes, fill them by recursively calling into schedule_search.
        for new_tree, new_child_memory_limits in viable_subproblems:
            if self.callbacks:
                self.callbacks.expanded_hole(new_tree)

            child_results = results[
                results_consumed : results_consumed + len(new_tree.children)
            ]
            results_consumed += len(new_tree.children)

            # Recurse for all holes (nested Specs) in the new Impl. If any hole
            # cannot be filled, short-circuit because this action is a dead end.
            subsearch_results: list[list[tuple[Impl, common.ScheduleKey]]] = []
            for child_cache_result, child, mem in zip(
                child_results, new_tree.children, new_child_memory_limits, strict=True
            ):
                if child_cache_result is not None:
                    child_impl_tuples = list(child_cache_result.contents)
                else:
                    child_search_result = yield from self.interactive_search(
                        child,
                        memory_limits=mem,
                        parent_summary=impl.ParentSummary.update(
                            parent_summary, parent=new_tree
                        ),
                        stats=stats,
                    )
                    child_impl_tuples = child_search_result.impl_tuples
                    unique_specs_visited += child_search_result.dependent_paths
                # If any hole could not be filled--this can happen, for
                # instance, if every possible action uses too much memory--then
                # exit the outer loop.
                if not child_impl_tuples:
                    break

                subsearch_results.append(child_impl_tuples)
            if len(subsearch_results) < len(new_tree.children):
                continue

            # Yield the product of all possible child Impls as options.
            # TODO: There's almost certainly a smarter way to enumerate these.
            for selected_children in itertools.product(*subsearch_results):
                completed = new_tree.replace_children((i for i, _ in selected_children))
                assert completed.spec == new_tree.spec
                completed_key = common.ScheduleKey.from_child_keys(
                    completed, [k for _, k in selected_children]
                )
                reducer(completed, hole.spec, completed_key)

        best_results = reducer.finalize()
        assert len(best_results) <= self.top_k

        yield SearchMessage(
            [],
            [
                (
                    memory_limits,
                    CachedScheduleSet(
                        hole.spec,
                        tuple((im, k) for im, k in best_results),
                        unique_specs_visited,
                    ),
                )
            ],
        )
        return SearchResult(best_results, unique_specs_visited)

    def _collect_subproblems(
        self, hole, memory_limits, parent_summary
    ) -> tuple[
        list[tuple[Impl, Sequence[pruning.MemoryLimits]]],
        list[tuple[specs.Spec, pruning.MemoryLimits]],
    ]:
        viable_subproblems = []
        deps = []
        for new_tree in self._iter_expansions(hole, parent_summary):
            # Skip actions using more memory than is available for any hole.
            new_child_memory_limits = memory_limits.transition(new_tree)
            if new_child_memory_limits is None:
                continue
            viable_subproblems.append((new_tree, new_child_memory_limits))
            for c, l in zip(new_tree.children, new_child_memory_limits):
                deps.append((c.spec, l))
        return viable_subproblems, deps

    def _iter_expansions(
        self, leaf: impl.Impl, parent_summary: Optional[impl.ParentSummary]
    ) -> Iterable[Impl]:
        """Iterate pairs of action callables and their resulting expanded Impls.

        Essentially walks `actions()`, calling each, filtering those which raise
        ActionOutOfDomain.
        """
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


async def _step_search_generators(
    cache: ScheduleCache,
    search_gens: Sequence[Generator[SearchMessage, SearchResponse, SearchResult]],
    msgs: Sequence[SearchMessage],
) -> tuple[list[tuple[int, list[Impl]]], list[SearchMessage]]:
    # Service lookups.
    cache_response = list(await cache.get_many([n for msg in msgs for n in msg.needed]))

    # Service puts.
    for msg in msgs:
        for mlims, entry in msg.computed:
            await cache.put(entry, mlims)

    # Communicate with the search generators.
    next_msgs = []
    finished: list[tuple[int, list[Impl]]] = []
    consumed = 0
    assert len(set(map(id, search_gens))) == len(search_gens)
    for gen_idx, (search_gen, msg) in enumerate(zip(search_gens, msgs, strict=True)):
        subresponse = cache_response[consumed : consumed + len(msg.needed)]
        assert all(
            css is None or css.spec == n[0]
            for css, n in zip(subresponse, msg.needed, strict=True)
        )
        try:
            next_msgs.append(search_gen.send(subresponse))
        except StopIteration as e:
            finished.append((gen_idx, e.value.impls))
        consumed += len(msg.needed)
    assert consumed == len(cache_response)
    assert len(finished) + len(next_msgs) == len(search_gens)
    assert all(a < b for a, b in zip(finished, finished[1:])), "finished isn't sorted"
    return finished, next_msgs


def _compute_memory_limits_for_leaves(
    imp: Impl, top_limits
) -> Iterable[pruning.MemoryLimits]:
    if not len(imp.children):
        yield top_limits
        return

    child_limits = top_limits.transition(imp)
    if child_limits is None:
        raise ValueError("Limits violated by sub-Impl")
    for child, lims in zip(imp.children, child_limits):
        yield from _compute_memory_limits_for_leaves(child, lims)


class _ImplReducer:
    __slots__ = ["results", "top_k"]

    results: list[tuple[Impl, common.ScheduleKey]]
    top_k: int

    def __init__(self, top_k: int):
        self.results = []
        self.top_k = top_k

    def __call__(
        self, new_impl: impl.Impl, spec: specs.Spec, imp_key: common.ScheduleKey
    ):
        # TODO: Actually necessary to pass spec *and* new_impl?
        assert new_impl.spec == spec, f"{str(new_impl.spec)} != {str(spec)}"
        self.results.append((new_impl, imp_key))

    def finalize(self) -> list[tuple[Impl, common.ScheduleKey]]:
        # Using sorted here for stability.
        return sorted(self.results, key=lambda x: x[1])[: self.top_k]
        # return heapq.nsmallest(top_k, results, key=lambda x: x[1])
