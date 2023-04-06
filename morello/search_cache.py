import asyncio
import contextlib
import contextvars
import dataclasses
import heapq
import logging
import pathlib
import pickle
import sys
import typing
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Generic,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import atomicwrites
import numpy as np
import redis.asyncio as redis

from . import bcarray, pruning
from .impl import Impl, spec_to_hole
from .specs import Load, Spec, Store, Zero, Matmul, MatmulAccum, Spec
from .system_config import current_system
from .utils import TinyMap, snap_availables_down, snap_availables_up, zip_dict

if TYPE_CHECKING:
    from .search.common import ScheduleKey

assert sys.version_info >= (3, 7), "Use Python 3.7 or newer"

RAISE_CAPS_ON_OVERLAP = True
PICKLE_PROTOCOL = 5
REDIS_MIN_DIM = 8
REDIS_ALLOWED_SPECS = (Load, Store, Zero)

T = typing.TypeVar("T")
U = typing.TypeVar("U")

assert_access_on_log_boundaries = contextvars.ContextVar(
    "assert_access_on_log_boundaries", default=True
)

logger = logging.getLogger(__name__)


def _leaf_tuples(
    root_impl: Impl, root_limits: pruning.MemoryLimits
) -> Iterable[tuple[Impl, pruning.MemoryLimits]]:
    if len(root_impl.children) == 0:
        yield root_impl, root_limits
    else:
        child_mlims = root_limits.transition(root_impl)
        assert child_mlims
        assert len(root_impl.children) == len(child_mlims)
        for child, m in zip(root_impl.children, child_mlims):
            yield from _leaf_tuples(child, m)


# TODO: Add __slots__. (Will break compatibility old pickles.)
class CachedScheduleSet:
    """Wraps Impls with their cost and other metadata."""

    spec: Spec
    contents: tuple[tuple[Impl, "ScheduleKey"], ...]
    peak_memory: Optional[TinyMap[str, int]]
    dependent_paths: int

    def __init__(
        self,
        spec: Spec,
        contents: tuple[tuple[Impl, "ScheduleKey"], ...],
        specs_visited: int,  # TODO: Rename to dependent_paths
        peak_memory: Optional[TinyMap[str, int]] = None,
    ):
        if contents and any(spec != imp.spec for imp, _ in contents):
            raise ValueError("All Impls must have the given Spec")

        self.spec = spec
        self.contents = contents
        self.dependent_paths = specs_visited
        self.peak_memory = peak_memory
        if contents and not peak_memory:
            banks = current_system().ordered_banks
            peak_vals: list[int] = [0] * len(contents[0][1].peaks)
            for i in range(len(peak_vals)):
                peak_vals[i] = max(c[1].peaks[i] for c in contents)
            self.peak_memory = TinyMap(banks, tuple(peak_vals))

        if (
            __debug__
            and self.contents
            and self.peak_memory
            and assert_access_on_log_boundaries.get()
        ):
            assert (
                snap_availables_up(self.peak_memory) == self.peak_memory
            ), f"{snap_availables_up(self.peak_memory)} != {self.peak_memory}"

    @property
    def peak_or_zero(self) -> TinyMap[str, int]:
        if self.peak_memory is not None:
            return self.peak_memory
        return TinyMap(
            current_system().ordered_banks, (0,) * len(current_system().banks)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CachedScheduleSet):
            return NotImplemented
        if self.dependent_paths != other.dependent_paths:
            return False
        if self.contents != other.contents:
            return False
        return True

    def __hash__(self) -> int:
        return hash(self.contents)

    def __str__(self) -> str:
        return f"CachedScheduleSet({self.spec}, {self.contents}, {self.dependent_paths}, peak={self.peak_memory})"

    def __repr__(self) -> str:
        return f"CachedScheduleSet({self.spec!r}, {self.contents!r}, {self.dependent_paths!r}, {self.peak_memory!r})"


@dataclasses.dataclass(frozen=True)
class _TableEntry:
    """Stores the best Impl for a region from its used memory to `caps`.

    A _TableEntry is a record of the best Impls that exist up to `caps`.  That schedule
    is no longer the best as soon as any memory capacity is above its corresponding
    level in `caps` or below the memory used at that level by the schedule.
    """

    spec: Spec
    schedules: CachedScheduleSet
    caps: TinyMap[str, int]

    @property
    def peak_memory(self) -> Optional[TinyMap[str, int]]:
        return self.schedules.peak_memory

    def __str__(self) -> str:
        return f"_TableEntry({self.spec}, {self.schedules}, caps={self.caps})"


class ScheduleCache:
    def __init__(
        self, use_redis: Optional[tuple[Any, str]] = None, max_dim: Optional[int] = None
    ):
        # TODO: `_rects` is unneeded tech. debt. They're identical except for a prefix.
        self._rects: dict[Spec, bcarray.BlockCompressedArray] = {}
        self._dirty_rects: set[Spec] = set()
        if use_redis and isinstance(use_redis[0], str):
            use_redis = (redis.Redis.from_url(use_redis[0]),) + use_redis[1:]
        self._use_redis = use_redis
        self._shared_local_get_cache = {}
        self._max_dim = max_dim

    @typing.final
    async def get(
        self, spec: Spec, memory_limits: pruning.MemoryLimits, complete: bool = True
    ) -> CachedScheduleSet:
        """Returns cached Impls. May contain nothing if query has no Impls.

        Raises KeyError if no Impl meeting the given limits exists and it is unknown
        whether such an Impl exists (i.e., search should be done).

        :param complete: If `True`, the Impl's nested Specs will also be resolved
          from the cache. Otherwise, the result, may be a partial Impl (an Impl which
          may contain Spec holes as children).
        :param recurse_fn: If non-`None` and `complete` is `True`, this function will be
          called to look up nested Specs instead of `get`.
        """
        result: Optional[CachedScheduleSet] = next(
            iter(await self.get_many([(spec, memory_limits)]))
        )
        if result is None:
            raise KeyError()
        if not complete or not result.contents:
            return result
        while True:
            new_impl, old_cost = result.contents[0]
            assert len(result.contents) == 1, "No support for multiple Impls."

            leaf_tuples = list(_leaf_tuples(new_impl, memory_limits))
            incomplete_leaf_idxs = [
                i for i, (leaf, _) in enumerate(leaf_tuples) if not leaf.is_scheduled
            ]
            if not incomplete_leaf_idxs:
                return result

            # Query the next row of leaves.
            resolved = list(
                await self.get_many(
                    [
                        (leaf_tuples[i][0].spec, leaf_tuples[i][1])
                        for i in incomplete_leaf_idxs
                    ]
                )
            )
            assert all(r is not None for r in resolved)
            assert len(resolved) == len(incomplete_leaf_idxs)

            new_leaves: list[Impl] = []
            for leaf, _ in leaf_tuples:
                if leaf.is_scheduled:
                    new_leaves.append(leaf)
                else:
                    replacement_entry = resolved.pop(0)
                    assert replacement_entry and len(replacement_entry.contents) == 1
                    new_leaves.append(replacement_entry.contents[0][0])
            assert not resolved

            new_impl = new_impl.replace_leaves(new_leaves)
            result = CachedScheduleSet(
                spec,
                ((new_impl, old_cost),),
                result.dependent_paths,
                result.peak_memory,
            )

    async def get_many(
        self, subproblems: Sequence[tuple[Spec, pruning.MemoryLimits]]
    ) -> Iterable[Optional[CachedScheduleSet]]:
        stage1_queries = []
        unskipped: list[int] = []
        for idx, (spec, memory_limits) in enumerate(subproblems):
            if self._should_ignore_problem(spec, memory_limits):
                continue
            unskipped.append(idx)

            assert (
                not assert_access_on_log_boundaries.get()
                or snap_availables_down(memory_limits.available)
                == memory_limits.available
            )

            rects = self._get_rects(spec)
            snapped = pruning.StandardMemoryLimits(
                snap_availables_up(memory_limits.available)
            )
            stage1_queries.append((rects, snapped))

        # Run stage1_queries and fill in stage1_results
        stage1_results = await self._get_table_entries(stage1_queries)
        assert len(stage1_results) == len(stage1_queries)  # TODO: Remove this line
        stage1_consumed = 0

        results: list[Optional[CachedScheduleSet]] = [None] * len(subproblems)
        for idx in unskipped:
            spec, memory_limits = subproblems[idx]

            initial_result, (rects, snapped) = (
                stage1_results[stage1_consumed],
                stage1_queries[stage1_consumed],
            )
            if initial_result is None:
                continue
            stage1_consumed += 1
            best = initial_result

            if snapped != memory_limits:
                # If we're not on a snapped/log boundary, then we'll check an uper entry (the
                # entry where all levels are higher than the given caps) and see if, in
                # actuality, that Impl is below the requested limits. If it's not, we'll just
                # grab the Impl from the conservative entry.
                best = -1
                if initial_result is not None:
                    upper_impl = initial_result
                    assert (
                        upper_impl is None
                        or upper_impl.peak_memory.raw_keys
                        == memory_limits.available.raw_keys
                    )
                    if upper_impl is None or all(
                        a <= b
                        for a, b in zip(
                            upper_impl.peak_memory.raw_values,
                            memory_limits.available.raw_values,
                        )
                    ):
                        best = upper_impl

                if best == -1:
                    snapped_down = pruning.StandardMemoryLimits(
                        snap_availables_down(memory_limits.available)
                    )
                    # TODO: Vectorize the following too
                    best = await self._get_table_entry(rects, snapped_down)

            results[idx] = best.schedules

        return results

    async def put(
        self, schedules_to_put: CachedScheduleSet, memory_limits: pruning.MemoryLimits
    ) -> None:
        if self._should_ignore_problem(schedules_to_put.spec, memory_limits):
            return

        # Raise exception if a put Impl exceeds the memory bound.
        mlims = memory_limits.available
        for imp, _ in schedules_to_put.contents:
            if any(imp.peak_memory[b] > mlims[b] for b in imp.peak_memory):
                raise ValueError(
                    f"Impl {imp} has peak memory {imp.peak_memory} which exceeds"
                    f" available memory {mlims}"
                )

        assert (
            not assert_access_on_log_boundaries.get()
            or snap_availables_down(mlims) == memory_limits.available
        )

        spec = schedules_to_put.spec
        self._dirty_rects.add(spec)

        schedules_to_put = self._specify_schedule_set(schedules_to_put)

        # If we haven't returned at this point, then we didn't find a _TableEntry to
        # update, so add one.
        r = self._get_rects(spec)
        entry = _TableEntry(spec, schedules_to_put, mlims)

        # compute bottom from entry
        banks = current_system().ordered_banks
        if entry.peak_memory is None:
            assert not len(entry.schedules.contents)
            bot = (0,) * len(banks)
        else:
            assert entry.peak_memory.raw_keys == banks
            bot = entry.peak_memory.raw_values

        top = snap_availables_up(entry.caps.raw_values, always=True)
        bottom_coord = tuple(v.bit_length() for v in bot)
        top_coord = tuple(v.bit_length() for v in top)
        assert all(b <= t for b, t in zip(bottom_coord, top_coord)), (
            f"Bottom coord {bottom_coord} is not less than or equal to "
            f"top coord {top_coord}; bot = {bot}; top = {top}"
        )
        assert not isinstance(entry, np.ndarray)
        await r.fill_range(bottom_coord, tuple(u + 1 for u in top_coord), entry)

    def _should_ignore_problem(
        self, spec: Spec, memory_limits: pruning.MemoryLimits
    ) -> bool:
        # Don't look up tiny-dim. Specs.
        if self._max_dim is not None:
            if any(d > self._max_dim for o in spec.operands for d in o.dim_sizes):
                return True
        if self._use_redis and not isinstance(spec, REDIS_ALLOWED_SPECS):
            return True
        if self._use_redis and all(
            d <= REDIS_MIN_DIM for o in spec.operands for d in o.dim_sizes
        ):
            return True
        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Queries with"
                " other MemoryLimits implementations always miss."
            )
            return True
        return False

    def _get_rects(self, spec: Spec) -> bcarray.BlockCompressedArray:
        """Lazily initializes and returns an entry in `_rects`."""
        assert isinstance(spec, Spec)  # TODO: Remove
        try:
            rects = self._rects[spec]
        except KeyError:
            table_dims = tuple(
                current_system().banks[b].capacity
                for b in current_system().ordered_banks
            )
            storage_dims = _logify(snap_availables_up(table_dims, always=True))
            storage_dims = tuple(d + 1 for d in storage_dims)

            bs = None
            if isinstance(spec, (Load, Store, Zero)):
                bs = 64  # Should cover whole space, using a single block.
            if not bs:
                bs = tuple(6 for _ in storage_dims)
            if isinstance(bs, int):
                bs = (bs,) * len(storage_dims)

            redis_param = None
            if self._use_redis:
                redis_param = (self._use_redis[0], f"{self._use_redis[1]}-{spec}")

            if not redis_param:
                grid = bcarray.NumpyStore(
                    storage_dims, tuple(bs), bcarray.BCA_DEFAULT_VALUE
                )
            else:
                redis_client, prefix = redis_param
                grid = bcarray.BCARedisStore(
                    storage_dims,
                    tuple(bs),
                    redis_client,
                    prefix,
                    bcarray.BCA_DEFAULT_VALUE,
                    local_cache=self._shared_local_get_cache,
                )

            rects = bcarray.BlockCompressedArray(storage_dims, tuple(bs), grid)
            self._rects[spec] = rects
        return rects

    async def flush(self) -> None:
        for spec in self._dirty_rects:
            await self._get_rects(spec).flush()
        self._dirty_rects.clear()

    async def specs(self) -> Iterable[Spec]:
        if not self._use_redis:
            return self._rects.keys()
        else:
            # TODO: Speed this up by grouping and returning keys in Lua.
            redis_db, prefix = self._use_redis
            db_keys = await redis_db.keys(f"{prefix}-*")
            spec_strs = {
                k[len(prefix) + 1 :].decode().split(":BCA")[0] for k in db_keys
            }
            print(f"specs: {spec_strs}")
            raise NotImplementedError()

    async def count_specs(self) -> int:
        return sum(1 for _ in await self.specs())

    async def __aiter__(
        self,
    ) -> AsyncIterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        for spec in await self.specs():
            seen = set()
            rects = self._get_rects(spec)
            async for entry in rects.iter_values():
                if entry is None or entry in seen:
                    continue
                assert isinstance(entry, _TableEntry)
                assert entry.spec == spec
                yield spec, entry.schedules, pruning.StandardMemoryLimits(entry.caps)
                seen.add(entry)

    @staticmethod
    def _specify_schedule_set(schedules_to_put: CachedScheduleSet) -> CachedScheduleSet:
        return CachedScheduleSet(
            schedules_to_put.spec,
            tuple(
                (ScheduleCache._specify_impl(imp), cost)
                for imp, cost in schedules_to_put.contents
            ),
            schedules_to_put.dependent_paths,
            peak_memory=schedules_to_put.peak_memory,
        )

    @staticmethod
    async def _get_table_entry(
        bca: bcarray.BlockCompressedArray, caps: pruning.StandardMemoryLimits
    ) -> _TableEntry:
        """Get an entry for a particular memory limit.

        Will raise an KeyError is any of the limits are out-of-bounds.
        """
        r = (await ScheduleCache._get_table_entries([(bca, caps)]))[0]
        if r is None:
            raise KeyError()
        return r

    @staticmethod
    async def _get_table_entries(
        queries: Sequence[
            tuple[bcarray.BlockCompressedArray, pruning.StandardMemoryLimits]
        ],
    ) -> Sequence[Optional[_TableEntry]]:
        """Get entries for a collection of memory limits across BCAs.

        Will raise an IndexError is any of the limits are out-of-bounds.
        """

        # TODO: This is an awful, hacky implementation. Instead, we should
        #  either flatten or abstract batching queries to the same underlying
        #  database.

        results: list[Optional[_TableEntry]] = [None] * len(queries)

        # Group according to Redis client instance
        redis_groups: dict[Any, tuple[redis.Redis, list[tuple[int, Generator]]]] = {}
        pending_responses = {}
        for i, (bca, caps) in enumerate(queries):
            grid = bca.grid
            if isinstance(grid, bcarray.BCARedisStore):
                red = grid.redis_client
                get_gen = bca.interactive_get_many([_logify(caps.available.raw_values)])
                redis_groups.setdefault(id(red), (red, []))[1].append((i, get_gen))
                assert id(get_gen) not in pending_responses
                pending_responses[id(get_gen)] = None
            else:
                # TODO: Call get_many for NumpyStores too!
                results[i] = await bca.get(_logify(caps.available.raw_values))
            assert results[i] is None or isinstance(results[i], _TableEntry)

        # Drive generators in query-batching steps.
        while redis_groups:
            groups_completed = []

            for group_key, (redis_client, generator_group) in redis_groups.items():
                idxs_to_del = []
                keys_accum: list[str] = []
                key_runs: list[int] = []

                # Send any pending responses and get new requests
                for i, get_gen in generator_group:
                    try:
                        keys = get_gen.send(pending_responses[id(get_gen)])
                    except StopIteration as e:
                        assert len(e.value) == 1
                        results[i] = e.value[0]
                        idxs_to_del.append(i)
                    else:
                        keys_accum.extend(keys)
                        key_runs.append(len(keys))

                # Remove completed generators
                for i in reversed(idxs_to_del):
                    del generator_group[i]
                if not generator_group:
                    groups_completed.append(group_key)
                    continue

                # Do the per-grid queries
                flattened_results = await redis_client.mget(keys_accum)

                # Distribute the query results to the correct generators
                taken = 0
                for (_, get_gen), request_count in zip(generator_group, key_runs):
                    pending_responses[id(get_gen)] = flattened_results[
                        taken : taken + request_count
                    ]
                    taken += request_count

            for k in groups_completed:
                del redis_groups[k]
        return results

    @staticmethod
    def _specify_impl(imp: Impl) -> Impl:
        if not len(imp.children):
            return imp
        return imp.replace_children((spec_to_hole(c.spec) for c in imp.children))

    async def _despecify_impl(
        self, imp: Impl, limits: pruning.StandardMemoryLimits, get_fn
    ) -> Impl:
        all_child_limits = limits.transition(imp)
        assert (
            all_child_limits is not None
        ), f"Limits violated while transitioning from {limits} via {imp}"
        assert len(all_child_limits) == len(imp.children)

        new_children: list[Impl] = []
        for spec_child, child_limits in zip(imp.children, all_child_limits):
            assert not spec_child.is_scheduled
            try:
                child_results = await get_fn(spec_child.spec, child_limits)
            except KeyError:
                raise Exception(
                    f"Unexpectedly got KeyError while reconstructing "
                    f"({spec_child.spec}, {child_limits})"
                )
            assert len(
                child_results.contents
            ), f"Child Spec {spec_child.spec} was not present in cache"
            if len(child_results.contents) > 1:
                raise NotImplementedError(
                    "Need to test proper reductions with top_k > 1. "
                    f"Got {len(child_results.contents)} results for {spec_child.spec}."
                )
            new_children.append(child_results.contents[0][0])

        return imp.replace_children(new_children)


class _PendingEntry(Generic[T]):
    __slots__ = ("event", "result")

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.result: Optional[T] = None

    def set(self, result: T) -> None:
        self.result = result
        self.event.set()


class _PriorityCache(Generic[T, U]):
    """A cache which keeps the top-k entries according to an arbitrary priority."""

    def __init__(self, max_size: int = 160):
        self._data: dict[T, U] = {}
        self._heap = []
        self._index = 0
        self.max_size = max_size

    def _push(self, priority, item):
        heapq.heappush(self._heap, (priority, self._index, item))
        self._index += 1

    def __getitem__(self, key: T) -> U:
        return self._data[key]

    def set(self, key: T, value: U, priority: Union[int, float]) -> None:
        self._data[key] = value
        self._push(priority, key)
        if len(self._heap) > self.max_size:
            v = heapq.heappop(self._heap)[-1]
            del self._data[v]
            assert len(self._heap) == len(self._data)
            logger.debug(
                f"Dropping {v} from cache. Top priorities are now: {self._heap[:5]}"
            )
        logger.debug("Cache size: %d", len(self._heap))

    def __len__(self) -> int:
        return len(self._heap)


class ChainCache(ScheduleCache):
    def __init__(self, caches: Iterable[ScheduleCache], put_all: bool = False) -> None:
        self._inner_caches = list(caches)
        self.put_all = put_all

    async def get_many(
        self, subproblems: Sequence[tuple[Spec, pruning.MemoryLimits]]
    ) -> Iterable[Optional[CachedScheduleSet]]:
        remaining_idxs: list[int] = list(range(len(subproblems)))
        results: list[Optional[CachedScheduleSet]] = [None] * len(remaining_idxs)
        for cache in self._inner_caches:
            cache_results = await cache.get_many(
                [subproblems[i] for i in remaining_idxs]
            )
            to_del = set()  # TODO: Don't use a set
            for subproblem_idx, entry in zip(remaining_idxs, cache_results):
                if entry is not None:
                    to_del.add(subproblem_idx)
                    results[subproblem_idx] = entry
            remaining_idxs = [i for i in remaining_idxs if i not in to_del]
        return results

    async def put(self, *args, **kwargs) -> None:
        if self.put_all:
            for cache in self._inner_caches:
                await cache.put(*args, **kwargs)
        else:
            await self._inner_caches[0].put(*args, **kwargs)

    async def flush(self) -> None:
        for c in self._inner_caches:
            await c.flush()

    async def specs(self) -> Iterable[Spec]:
        result = set()
        for cache in self._inner_caches:
            result.update(await cache.specs())
        return result

    async def __aiter__(
        self,
    ) -> AsyncIterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


@contextlib.contextmanager
def persistent_cache(
    path: Optional[Union[str, pathlib.Path]],
    redis: Optional[tuple[Any, str]] = None,
    save: bool = True,
):
    with _local_persistent_cache(path, save=save) as local_cache:
        if not redis:
            yield local_cache
        else:
            redis_cache = ScheduleCache(use_redis=redis)
            yield ChainCache([local_cache, redis_cache])


@contextlib.contextmanager
def _local_persistent_cache(
    path: Optional[Union[str, pathlib.Path]], save: bool = True
):
    if path is None:
        yield ScheduleCache()
        return

    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.exists():
        if not path.is_file():
            raise ValueError(f"Path was not a file: {str(path)}")
        with path.open(mode="rb") as fo:
            cache = pickle.load(fo)
    else:
        # If we're going to save the cache, make any parent directories
        if save:
            path.parent.mkdir(parents=True, exist_ok=True)
        cache = ScheduleCache()

    try:
        yield cache
    finally:
        if save:
            with atomicwrites.atomic_write(path, mode="wb", overwrite=True) as fo:
                pickle.dump(cache, fo, protocol=PICKLE_PROTOCOL)


def _logify(tup: Sequence[int]) -> tuple[int, ...]:
    return tuple((0 if x == 0 else (x - 1).bit_length() + 1) for x in tup)
