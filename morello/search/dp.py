import asyncio
import itertools
import typing
from typing import Generator, Iterable, List, Optional, Sequence, TYPE_CHECKING

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
        Search(top_k, callbacks=callbacks)(
            spec, memory_limits, parent_summary=parent_summary, cache=cache
        )
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
    ) -> list[impl.Impl]:
        """Returns the best Impl for a given Spec and memory limits.

        May return `None` if no Impl satisfies the given Spec and memory limits.
        """
        if self.top_k > 1:
            raise NotImplementedError("Search for top_k > 1 not yet implemented.")

        if cache is None:
            cache = ScheduleCache()

        # The default available memory is the capacities of current_system().
        # These capacities are measured in cache lines, not words, so: multiply
        # by line size when initializing the limit.
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
            if not finished:
                continue

            # Short-circuit if any subproblem couldn't be implemented.
            if not all(len(imps) for _, imps in finished):
                return []

            hole_leaf_idxs = [
                i for i, leaf in enumerate(partial_impl.leaves) if not leaf.is_scheduled
            ]
            assert len(hole_leaf_idxs) == len(search_gens)

            # Sub. in new leaves, tracking what was inserted for the next step.
            new_leaves: list[range] = []
            leaf_replacements = list(partial_impl.leaves)
            leaf_offset = 0
            for search_gen_idx, imps in finished:
                leaf_idx = hole_leaf_idxs[search_gen_idx]
                leaf_replacements[leaf_idx] = imps[0]
                inserted_leaf_count = sum(1 for _ in imps[0].leaves)
                new_leaves.append(
                    range(
                        leaf_idx + leaf_offset,
                        leaf_idx + leaf_offset + inserted_leaf_count,
                    )
                )
                leaf_offset += inserted_leaf_count - 1

                # TODO: Remove the set intersection
                assert len(set.union(*map(set, new_leaves))) == sum(
                    map(len, new_leaves)
                ), "Expected no overlap in new_leaves."

            partial_impl = partial_impl.replace_leaves(leaf_replacements)
            assert max(r[-1] for r in new_leaves) < sum(1 for _ in partial_impl.leaves)
            del leaf_offset
            del leaf_replacements
            del hole_leaf_idxs  # No longer valid after `partial_impl.replace_leaves`.

            # The following is slow. It traverses `partial_impl`.
            new_leaf_mlims = list(
                _compute_memory_limits_for_leaves(partial_impl, memory_limits)
            )

            # Start generators for any newly inserted sub-Specs.
            #
            # Iterate in reversed order so that we can modify search_gens and msgs
            # with stable indices.
            gens_added = 0
            gen_offset = 0
            for (finished_gen_idx, imps), new_leaf_rng in zip(finished, new_leaves):
                new_gens = []
                for new_leaf_idx, new_leaf in zip(new_leaf_rng, imps[0].leaves):
                    if not new_leaf.is_scheduled:
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
                    finished_gen_idx + gen_offset : finished_gen_idx + gen_offset + 1
                ] = new_gens
                add_pt = finished_gen_idx + gens_added
                msgs[add_pt:add_pt] = [next(g) for g in new_gens]
                gens_added += len(new_gens)
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
        caller_response = yield SearchMessage(((hole.spec, memory_limits),), [])
        assert len(caller_response) == 1
        cache_result = caller_response[0]
        if cache_result is not None:
            # TODO: Document: we allow the caller to give an incomplete Impl, where
            #  there's no guarantee the result of interactive_search is a complete Impl.
            #  This can save compute and the requirements for the caller are clear:
            #  complete your cached Impls if that's what you want.
            #  This requires the caller to also provide costs for provided Impls.
            return SearchResult(cache_result.contents, cache_result.dependent_paths)

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
            subsearch_results: list[list[tuple[Impl, "cost.MainCost"]]] = []
            for child, mem in zip(new_tree.children, new_child_memory_limits):
                child_result = yield from self.interactive_search(
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
                if not child_result.impl_tuples:
                    break
                subsearch_results.append(child_result.impl_tuples)
            if len(subsearch_results) < len(new_tree.children):
                continue

            # Yield the product of all possible child Impls as options.
            # TODO: There's almost certainly a smarter way to enumerate these.
            for selected_children in itertools.product(*subsearch_results):
                completed = new_tree.replace_children(
                    (im for im, _ in selected_children)
                )
                assert (
                    completed.spec == new_tree.spec
                ), f"{str(completed.spec)} != {str(new_tree.spec)}"
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


async def _step_search_generators(
    cache, search_gens, msgs
) -> tuple[list[tuple[int, list[Impl]]], list[SearchMessage]]:
    assert len(search_gens) == len(msgs)
    cache_response = list(await cache.get_many([n for msg in msgs for n in msg.needed]))
    for msg in msgs:
        for mlims, entry in msg.computed:
            await cache.put(entry, mlims)

    next_msgs = []
    finished: list[tuple[int, list[Impl]]] = []
    consumed = 0
    for gen_idx, (search_gen, msg) in enumerate(zip(search_gens, msgs)):
        subresponse = cache_response[consumed : consumed + len(msg.needed)]
        try:
            next_msgs.append(search_gen.send(subresponse))
        except StopIteration as e:
            finished.append((gen_idx, e.value.impls))
        consumed += len(msg.needed)
    assert consumed == len(cache_response)
    assert len(finished) + len(next_msgs) == len(search_gens)
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
