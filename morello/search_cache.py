import abc
import asyncio
import contextlib
import contextvars
import dataclasses
import itertools
import pathlib
import pickle
import typing
import warnings
from typing import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import atomicwrites
import lz4.frame
import numpy as np
import redis.asyncio
import redis.asyncio.lock

from . import bcarray, pruning
from .impl import Impl, spec_to_hole
from .specs import Spec
from .system_config import current_system
from .utils import TinyMap, snap_availables_down, snap_availables_up, zip_dict

TABLE_STYLE: Literal["dense", "list", "dict", "bca"] = "bca"
RAISE_CAPS_ON_OVERLAP = True
PICKLE_PROTOCOL = 5

T = typing.TypeVar("T")

assert_access_on_log_boundaries = contextvars.ContextVar(
    "assert_access_on_log_boundaries", default=True
)


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


class CachedScheduleSet:
    """Wraps Impls with their cost and other metadata."""

    contents: tuple[tuple[Impl, int], ...]
    peak_memory: Optional[TinyMap[str, int]]
    dependent_paths: int

    def __init__(
        self,
        spec: Spec,
        contents: tuple[tuple[Impl, int], ...],
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
            self.peak_memory = snap_availables_up(
                TinyMap(
                    self.contents[0][0].peak_memory.raw_keys,
                    tuple(
                        max(zvals)
                        for zvals in zip(
                            *(c[0].peak_memory.raw_values for c in self.contents)
                        )
                    ),
                )
            )
        assert not self.contents or (
            self.peak_memory is not None
            and (
                not assert_access_on_log_boundaries.get()
                or snap_availables_up(self.peak_memory) == self.peak_memory
            )
        )

    @property
    def bottom_peak(self) -> TinyMap[str, int]:
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


class ScheduleCache(abc.ABC):
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

    @abc.abstractmethod
    async def get_many(
        self, subproblems: Iterable[tuple[Spec, pruning.MemoryLimits]]
    ) -> Iterable[Optional[CachedScheduleSet]]:
        pass

    # TODO: Raise exception if sub-Specs aren't in the cache or document behavior.
    @abc.abstractmethod
    async def put(
        self, schedules_to_put: CachedScheduleSet, memory_limits: pruning.MemoryLimits
    ) -> None:
        pass

    @abc.abstractmethod
    def specs(self) -> Iterable[Spec]:
        pass

    @abc.abstractmethod
    async def count_impls(self) -> int:
        pass

    @abc.abstractmethod
    async def __aiter__(
        self,
    ) -> AsyncIterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        """Iterate over subproblems (Specs and MemoryLimits) and associated Impls.

        There is no guarantee that each step of the iterator corresponds to a single
        `put` entry.
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class InMemoryScheduleCache(ScheduleCache):
    _rects: dict[Spec, "_RectTable"]

    def __init__(self):
        self._rects = {}

    def count_values(self):
        return sum(table.storage.count_values() for table in self._rects.values())

    async def get_many(
        self, subproblems: Iterable[tuple[Spec, pruning.MemoryLimits]]
    ) -> Iterable[Optional[CachedScheduleSet]]:
        results = []

        for spec, memory_limits in subproblems:
            if not isinstance(memory_limits, pruning.StandardMemoryLimits):
                # TODO: Add support for PipelineChildMemoryLimits
                warnings.warn(
                    "ScheduleCache only supports StandardMemoryLimits. Queries with"
                    " other MemoryLimits implementations always miss."
                )
                results.append(None)
                continue

            assert (
                not assert_access_on_log_boundaries.get()
                or snap_availables_down(memory_limits.available)
                == memory_limits.available
            )

            try:
                rects = self._rects[spec]
                best = rects.best_for_cap(memory_limits)
            except KeyError:
                results.append(None)
                continue

            results.append(best.schedules)

        return results

    async def put(
        self, schedules_to_put: CachedScheduleSet, memory_limits: pruning.MemoryLimits
    ) -> None:
        spec = schedules_to_put.spec

        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Puts with"
                " other MemoryLimits implementations always miss."
            )
            return

        for imp, _ in schedules_to_put.contents:
            for b in imp.peak_memory:
                if imp.peak_memory[b] > memory_limits.available[b]:
                    raise ValueError(
                        f"Impl {imp} has peak memory {imp.peak_memory} which exceeds"
                        f" available memory {memory_limits.available}"
                    )

        mlims = memory_limits.available

        assert (
            not assert_access_on_log_boundaries.get()
            or snap_availables_down(mlims) == memory_limits.available
        )

        for im, _ in schedules_to_put.contents:
            for _, (m, c) in zip_dict(im.peak_memory, mlims, same_keys=True):
                if m > c:
                    raise ValueError(
                        f"Impl peak memory {im.peak_memory} not bounded by {mlims}"
                    )

        try:
            rects = self._rects[spec]
        except KeyError:
            if TABLE_STYLE in ("dense", "bca"):
                table_dims = tuple(
                    current_system().banks[b].capacity for b in mlims.raw_keys
                )
                if TABLE_STYLE == "dense":
                    rects = _DenseNumpyTable(mlims.raw_keys, table_dims)
                else:
                    rects = _BlockCompressedTable(mlims.raw_keys, table_dims)
            elif TABLE_STYLE == "list":
                rects = _ListTable(mlims.raw_keys)
            else:
                rects = _DictTable(mlims.raw_keys)
            self._rects[spec] = rects

        schedules_to_put = self._specify_schedule_set(schedules_to_put)

        # If we haven't returned at this point, then we didn't find a _TableEntry to
        # update, so add one.
        rects.add(_TableEntry(spec, schedules_to_put, mlims))

    async def update(self, other: "ScheduleCache") -> None:
        """Update with the contents of another cache.

        Equivalent to `put`ing every entry from the given cache.
        """
        for other_spec, other_table in other._rects.items():
            if other_spec not in self._rects:
                # TODO: We really want a copy-on-write here. This sharing is surprising.
                self._rects[other_spec] = other_table
                continue
            rects_seen = set()
            for rect in other_table:
                if rect in rects_seen:
                    continue
                assert rect.spec == other_spec
                rects_seen.add(rect)
                await self.put(rect.schedules, pruning.StandardMemoryLimits(rect.caps))

    def specs(self) -> Iterable[Spec]:
        yield from self._rects.keys()

    async def count_impls(self) -> int:
        return sum(len(rect) for rect in self._rects.values())

    def __len__(self) -> int:
        return len(self._rects)

    async def __aiter__(
        self,
    ) -> AsyncIterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        for spec, rects in self._rects.items():
            for rect in rects:
                assert rect.spec == spec
                yield spec, rect.schedules, pruning.StandardMemoryLimits(rect.caps)

    def _specify_schedule_set(
        self, schedules_to_put: CachedScheduleSet
    ) -> CachedScheduleSet:
        return CachedScheduleSet(
            schedules_to_put.spec,
            tuple(
                (self._specify_impl(imp), cost)
                for imp, cost in schedules_to_put.contents
            ),
            schedules_to_put.dependent_paths,
            peak_memory=schedules_to_put.peak_memory,
        )

    def _specify_impl(self, imp: Impl) -> Impl:
        if not len(imp.children):
            return imp
        return imp.replace_children((spec_to_hole(c.spec) for c in imp.children))

    async def _despecify_schedule_set(
        self, cached_set: CachedScheduleSet, caps: TinyMap[str, int], get_fn
    ) -> CachedScheduleSet:
        new_impls = []
        if cached_set.contents:
            limits = pruning.StandardMemoryLimits(caps)
            for imp, cost in cached_set.contents:
                despecified = await self._despecify_impl(imp, limits, get_fn)
                new_impls.append((despecified, cost))

        return CachedScheduleSet(
            cached_set.spec,
            tuple(new_impls),
            cached_set.dependent_paths,
            cached_set.peak_memory,
        )

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


class RedisCache(ScheduleCache):
    """A cache which streams blocks to and from a Redis backend.

    Gets are always serviced with a Redis query, so it can be useful to batch many
    queries into large `get_many` calls.

    Puts are sent to an in-memory, "overlay" cache. That cache is sent to Redis and
    and deleted when `flush` is called. Gets are serviced first by the overlay cache
    and, if the get misses, then by querying the Redis cache.
    """

    def __init__(
        self,
        redis_connection: redis.asyncio.Redis,
        namespace: str,
        subproblem_to_block_coordinate_fn: Callable[[Spec, pruning.MemoryLimits], str],
        keep_local: Optional[Callable[[str], bool]] = None,
        autoflush: bool = True,
        updating: bool = True,
    ) -> None:
        """Create a new RedisCache using the given `redis_connection`.

        Args:
            redis_connection: The Redis connection to use.
            namespace: A string which will be included in all Redis keys.
            subproblem_to_block_coordinate_fn: A function which maps a subproblem
              to a string identifying a particular block.
            keep_local: A function which takes a block-identifying string and returns
              `True` is that block should be permanently kept in the local cache after
              its first get.
            autoflush: If `True`, the cache will automatically flush its in-memory data
              when `put` is called on a new block.
            updating: If `True`, puts will update blocks rather than assume they are not
              present in the Redis database. On the first put to a new block, a
              distributed lock will be acquired and any existing block will be
              downloaded. That and subsequent puts will update that block in memory.
              On flush, the updated block will be stored in Redis and the lock released.
        """
        # TODO: Document that this class takes "ownership" of the redis_connection
        super().__init__()
        self.redis_connection = redis_connection
        self.namespace = namespace
        self.overlay = InMemoryScheduleCache()
        self.autoflush = autoflush
        self.updating = updating
        self._keep_local = keep_local
        self._local_cache = {}
        self._dirty_block: Optional[str] = None
        self._dirty_block_culprit: Optional[tuple[Spec, pruning.MemoryLimits]] = None
        self._dirty_block_lock: Optional[redis.asyncio.lock.Lock] = None
        self._given_block_key = subproblem_to_block_coordinate_fn
        self._running_block_task: Optional[
            tuple[str, asyncio.Future[InMemoryScheduleCache]]
        ] = None
        self._pending: dict[str, _PendingEntry[Optional[InMemoryScheduleCache]]] = {}
        self._get_block_lock = asyncio.Lock()

    async def get_many(
        self, subproblems: Iterable[tuple["Spec", pruning.MemoryLimits]]
    ) -> Iterable[Optional[CachedScheduleSet]]:
        subproblems = list(subproblems)  # TODO: Just require Sequence in signature.
        results: list[Optional[CachedScheduleSet]] = list(
            await self.overlay.get_many(subproblems)
        )

        none_idxs = [i for i, r in enumerate(results) if r is None]

        # Map needed block keys to subproblems indices.
        blocks_needed: dict[str, list[int]] = {}
        for idx in none_idxs:
            blocks_needed.setdefault(
                self.redis_block_key(*subproblems[idx]), []
            ).append(idx)

        # Send requests to the Redis server (or wait if one is already running).
        blocks = await asyncio.gather(*[self._get_block(k) for k in blocks_needed])

        for subproblem_idxs, block in zip(blocks_needed.values(), blocks):
            if not block:
                continue
            for i in subproblem_idxs:
                assert results[i] is None
                results[i] = next(iter(await block.get_many([subproblems[i]])))

        return results

    async def _get_block(self, block_key) -> Optional[InMemoryScheduleCache]:
        try:
            pending_entry: _PendingEntry[
                Optional[InMemoryScheduleCache]
            ] = self._pending[block_key]
        except KeyError:
            # No ongoing request for the block, so start one.
            pending_entry = _PendingEntry()
            self._pending[block_key] = pending_entry
            # Await the task instead of the event to propogate exceptions.
            await asyncio.create_task(
                self._download_and_broadcast_block(block_key),
                name=f"RedisGetBlock-{block_key}",
            )
        else:
            await pending_entry.event.wait()
        return pending_entry.result

    async def put(
        self, schedules_to_put: CachedScheduleSet, memory_limits: pruning.MemoryLimits
    ) -> None:
        spec = schedules_to_put.spec
        block_key = self.redis_block_key(spec, memory_limits)

        should_load = False
        if self._dirty_block and self._dirty_block != block_key:
            assert (
                self._dirty_block_culprit
            ), "_dirty_block_culprit should be set is _dirty_block is set"
            if not self.autoflush:
                c_spec, c_memory_limits = self._dirty_block_culprit
                raise Exception(
                    "Putting into a block other than the current dirty "
                    "block. Call `flush` before `put`ing to a new block. "
                    f"Current dirty block is {self._dirty_block}, which was "
                    f"set while updating {c_spec} at {c_memory_limits}. However, "
                    f"this put was to {block_key}, for {spec} at {memory_limits}."
                )
            await self.flush()
            # TODO: Load new overlay here?
            should_load = True
        elif not self._dirty_block:
            should_load = True

        if should_load and self.updating:
            loaded_block = await self._get_block(block_key)
            if loaded_block:
                self.overlay = loaded_block

        self._dirty_block = block_key
        self._dirty_block_culprit = (spec, memory_limits)
        if self.updating:
            self._dirty_block_lock = self.redis_connection.lock(
                f"Lock-{self.namespace}-{block_key}",
                timeout=2 * 60.0,  # TODO: Need to extend periodically.
            )
            await self._dirty_block_lock.acquire()
        await self.overlay.put(schedules_to_put, memory_limits)

    async def flush(self):
        if not self._dirty_block:
            return

        # TODO: There should never be a block, so just check that this is the case,
        #   instead of getting it.
        payload = lz4.frame.compress(
            pickle.dumps(self.overlay, protocol=PICKLE_PROTOCOL)
        )
        if self.updating:
            await self.redis_connection.set(self._dirty_block, payload)
        else:
            set_response = await self.redis_connection.setnx(self._dirty_block, payload)
            if not set_response:
                raise Exception(f"Block already exists: {self._dirty_block}")

        self.overlay = InMemoryScheduleCache()
        self._dirty_block = None
        self._dirty_block_culprit = None
        if self.updating:
            assert self._dirty_block_lock
            await self._dirty_block_lock.release()
            self._dirty_block_lock = None

    def specs(self) -> Iterable[Spec]:
        raise NotImplementedError()

    async def count_impls(self) -> int:
        total_count = 0
        async for block in self._iter_blocks():
            total_count += await block.count_impls()
        return total_count

    async def __aiter__(
        self,
    ) -> AsyncIterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        async for block in self._iter_blocks():
            async for x in block:
                yield x

    async def _iter_blocks(self) -> AsyncGenerator[InMemoryScheduleCache, None]:
        for key in await self._all_block_keys():
            if key == self._dirty_block:
                continue
            block_cache = await self._download_block(key)
            assert block_cache is not None
            yield block_cache
        if self._dirty_block:
            yield self.overlay

    def __len__(self) -> int:
        assert "*" not in self.namespace
        raise NotImplementedError()

    async def _all_block_keys(self) -> Iterable[str]:
        return await self.redis_connection.keys(self.namespace + "*")

    async def _download_and_broadcast_block(self, key: str) -> None:
        self._pending.pop(key).set(await self._download_block(key))

    async def _download_block(self, key: str) -> Optional[InMemoryScheduleCache]:
        # TODO: Though we only have one GET out at once, there's no guarantee we'll wait
        #   for it to be processed by all waiting searches before we start the next GET.
        #   Fix this.

        should_keep_local = self._keep_local and self._keep_local(key)
        if should_keep_local:
            try:
                return self._local_cache[key]
            except KeyError:
                pass

        async with self._get_block_lock:
            blob = await asyncio.wait_for(self.redis_connection.get(key), timeout=900)
            if blob is None:
                return None
            assert isinstance(
                blob, bytes
            ), f"blob was not bytes, was {type(blob).__name__}"
            decoded = lz4.frame.decompress(blob)
            decoded = pickle.loads(decoded)
            assert isinstance(
                decoded, InMemoryScheduleCache
            ), f"decoded was not InMemoryScheduleCache; was {type(decoded).__name__}"
            if should_keep_local:
                self._local_cache[key] = decoded
            return decoded

    def redis_block_key(self, *args, **kwargs):
        return f"{self.namespace}-{self._given_block_key(*args, **kwargs)}"


class _RectTable(abc.ABC):
    @abc.abstractmethod
    def add(self, entry: _TableEntry):
        pass

    @abc.abstractmethod
    def best_for_cap(self, caps: pruning.StandardMemoryLimits) -> _TableEntry:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[_TableEntry]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class _SnappingRectTable(_RectTable):
    def best_for_cap(self, caps: pruning.StandardMemoryLimits) -> _TableEntry:
        snapped = pruning.StandardMemoryLimits(snap_availables_up(caps.available))
        if snapped == caps:
            return self._get_entry(caps)

        # If we're not on a snapped/log boundary, then we'll check an uper entry (the
        # entry where all levels are higher than the given caps) and see if, in
        # actuality, that Impl is below the requested limits. If it's not, we'll just
        # grab the Impl from the conservative entry.
        try:
            upper_impl = self._get_entry(snapped)
        except KeyError:
            pass
        else:
            assert (
                upper_impl is None
                or upper_impl.peak_memory.raw_keys == caps.available.raw_keys
            )
            if upper_impl is None or all(
                a <= b
                for a, b in zip(
                    upper_impl.peak_memory.raw_values, caps.available.raw_values
                )
            ):
                return upper_impl
        snapped_down = pruning.StandardMemoryLimits(
            snap_availables_down(caps.available)
        )
        return self._get_entry(snapped_down)

    @abc.abstractmethod
    def _get_entry(self, caps: pruning.StandardMemoryLimits) -> _TableEntry:
        pass


class _SharedTable(_SnappingRectTable):
    def add(self, entry: _TableEntry):
        bottom_coord = tuple(v.bit_length() for v in self._bottom_from_entry(entry))
        top_coord = tuple(
            v.bit_length()
            for v in snap_availables_up(entry.caps.raw_values, always=True)
        )
        self._fill_storage(bottom_coord, top_coord, entry)

    def _get_entry(self, caps: pruning.StandardMemoryLimits) -> _TableEntry:
        avail = caps.available
        coord = self._logify(avail.raw_values)
        assert isinstance(coord, tuple)
        return self._get_log_scaled_pt(coord)

    @abc.abstractmethod
    def _banks(self) -> tuple[str, ...]:
        pass

    @abc.abstractmethod
    def _get_log_scaled_pt(self, pt) -> _TableEntry:
        pass

    @abc.abstractmethod
    def _fill_storage(self, lower, upper, value):
        """Fills storage.

        Lower and upper points are inclusive.
        """
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[_TableEntry]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @staticmethod
    def _logify(tup: Sequence[int]) -> tuple[int, ...]:
        return tuple((0 if x == 0 else (x - 1).bit_length() + 1) for x in tup)

    def _bottom_from_entry(self, entry: _TableEntry) -> tuple[int, ...]:
        if entry.peak_memory is None:
            assert not len(entry.schedules.contents)
            return (0,) * len(self._banks())
        else:
            return entry.peak_memory.raw_values


@dataclasses.dataclass(frozen=False)
class _DictTable(_SharedTable):
    banks: tuple[str, ...]
    storage: dict[tuple[int, ...], _TableEntry] = dataclasses.field(
        default_factory=dict
    )

    def _banks(self) -> tuple[str, ...]:
        return self.banks

    def _fill_storage(self, lower, upper, value):
        for pt in itertools.product(*[range(l, u + 1) for l, u in zip(lower, upper)]):
            self.storage[pt] = value

    def _get_log_scaled_pt(self, pt) -> _TableEntry:
        return self.storage[pt]

    def __iter__(self) -> Iterator[_TableEntry]:
        for entry in set(self.storage.values()):
            assert isinstance(entry, _TableEntry)
            yield entry

    def __len__(self) -> int:
        return len(set(self.storage.values()))


@dataclasses.dataclass(frozen=False)
class _ListTable(_SnappingRectTable):
    banks: tuple[str, ...]
    storage: list[_TableEntry] = dataclasses.field(default_factory=list)

    @staticmethod
    def _raise_entry(r, entry):
        if RAISE_CAPS_ON_OVERLAP:
            raised_caps = TinyMap(
                entry.caps.raw_keys,
                tuple(
                    max(a, b)
                    for a, b in zip(
                        entry.caps.raw_values,
                        r.caps.raw_values,
                    )
                ),
            )
            return _TableEntry(entry.spec, entry.schedules, raised_caps)
        else:
            return entry

    def add(self, entry: _TableEntry):
        # First, check for a _TableEntry to update.
        for rect_idx in range(len(self.storage)):
            # If we're putting no Impls and there exists an entry already for
            # the no-Impl case, raise its memory caps to whatever we've explored.
            r = self.storage[rect_idx]
            if not entry.schedules.contents:
                if r.schedules.contents:
                    continue
                if _mem_dicts_ordered(r.caps, entry.caps):
                    self.storage[rect_idx] = self._raise_entry(r, entry)
                    return
            else:
                if not r.schedules.contents:
                    continue
                assert r.peak_memory is None or isinstance(r.peak_memory, TinyMap)
                if _mem_dicts_ordered(
                    r.peak_memory, entry.schedules.peak_memory, r.caps, entry.caps
                ):
                    self.storage[rect_idx] = self._raise_entry(r, entry)
                    return
            # TODO: Assert that there is at most one intersection

        # If we haven't returned at this point, then we didn't find a _TableEntry to
        # update, so add one.
        self.storage.append(entry)

    def _get_entry(self, caps: pruning.StandardMemoryLimits) -> _TableEntry:
        for entry in self:
            if _mem_dicts_ordered(entry.peak_memory, caps.available, entry.caps):
                return entry
        raise KeyError()

    def _get_log_scaled_pt(self, pt) -> _TableEntry:
        for entry in self.storage:
            if _mem_dicts_ordered(entry.peak_memory, pt, entry.caps):
                return entry
        raise KeyError()

    def __iter__(self) -> Iterator[_TableEntry]:
        yield from iter(self.storage)

    def __len__(self) -> int:
        return len(self.storage)


@dataclasses.dataclass(frozen=False)
class _DenseNumpyTable(_SharedTable):
    banks: tuple[str, ...]
    storage: np.ndarray

    def __init__(self, banks: tuple[str, ...], table_dims: tuple[int, ...]):
        storage_dims = self._logify(snap_availables_up(table_dims, always=True))
        storage_dims = tuple(d + 1 for d in storage_dims)
        object.__setattr__(self, "banks", banks)
        object.__setattr__(self, "storage", np.empty(storage_dims, dtype=object))
        self.storage.fill(None)

    def _banks(self) -> tuple[str, ...]:
        return self.banks

    def _fill_storage(self, lower, upper, value):
        slicer = tuple(slice(l, u + 1) for l, u in zip(lower, upper))
        self.storage[slicer].fill(value)

    def __iter__(self) -> Iterator[_TableEntry]:
        flat = self.storage.flatten()
        for entry in set(flat[flat != None]):
            assert isinstance(entry, _TableEntry)
            yield entry

    def __len__(self) -> int:
        flat = self.storage.flatten()
        return len(set(flat[flat != None]))

    def _get_log_scaled_pt(self, pt) -> _TableEntry:
        try:
            result = self.storage[pt]
            if result is None:
                raise KeyError()
            return result
        except IndexError:
            raise KeyError()

    def _bottom_from_entry(self, entry: _TableEntry) -> tuple[int, ...]:
        if entry.peak_memory is None:
            assert not len(entry.schedules.contents)
            return (0,) * len(self.banks)
        else:
            assert entry.peak_memory.raw_keys == self.banks
            return entry.peak_memory.raw_values


@dataclasses.dataclass(frozen=False)
class _BlockCompressedTable(_SharedTable):
    banks: tuple[str, ...]
    storage: bcarray.BlockCompressedArray

    def __init__(self, banks: tuple[str, ...], table_dims: tuple[int, ...]):
        storage_dims = self._logify(snap_availables_up(table_dims, always=True))
        storage_dims = tuple(d + 1 for d in storage_dims)
        object.__setattr__(self, "banks", banks)
        object.__setattr__(
            self,
            "storage",
            bcarray.BlockCompressedArray(
                storage_dims, block_shape=tuple(4 for _ in storage_dims)
            ),
        )

    def _banks(self) -> tuple[str, ...]:
        return self.banks

    def _fill_storage(self, lower, upper, value):
        assert not isinstance(value, np.ndarray)
        self.storage.fill_range(lower, tuple(u + 1 for u in upper), value)

    def __iter__(self) -> Iterator[_TableEntry]:
        for entry in set(self.storage.iter_values()):
            if entry is None:
                continue
            assert isinstance(entry, _TableEntry)
            yield entry

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def _get_log_scaled_pt(self, pt) -> _TableEntry:
        try:
            result = self.storage[pt]
            if result is None:
                raise KeyError()
            return result
        except IndexError:
            raise KeyError()

    def _bottom_from_entry(self, entry: _TableEntry) -> tuple[int, ...]:
        if entry.peak_memory is None:
            assert not len(entry.schedules.contents)
            return (0,) * len(self.banks)
        else:
            assert entry.peak_memory.raw_keys == self.banks
            return entry.peak_memory.raw_values


@contextlib.contextmanager
def persistent_cache(path: Optional[Union[str, pathlib.Path]], save: bool = True):
    if path is None:
        yield InMemoryScheduleCache()
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
        cache = InMemoryScheduleCache()

    try:
        yield cache
    finally:
        if save:
            with atomicwrites.atomic_write(path, mode="wb", overwrite=True) as fo:
                pickle.dump(cache, fo, protocol=PICKLE_PROTOCOL)


def _mem_dicts_ordered(*dicts: Optional[TinyMap[str, int]]) -> bool:
    # This function is pretty verbose. It avoids `range`, `any`, and other
    # generators for speed.
    if len(dicts) == 0:
        return True

    idx: int = 0
    while dicts[idx] is None:
        idx += 1
        if idx == len(dicts):
            return True
    head: TinyMap[str, int] = cast(TinyMap[str, int], dicts[idx])
    val_len = len(head.raw_values)

    idx += 1
    while idx < len(dicts):
        cur = dicts[idx]
        if cur is None:
            idx += 1
            continue
        assert cur.raw_keys == head.raw_keys
        j = 0
        while j < val_len:
            if head.raw_values[j] > cur.raw_values[j]:
                return False
            j += 1
        head = cur
        idx += 1
    return True
