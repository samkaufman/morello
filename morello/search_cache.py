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
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import atomicwrites
import numpy as np

from . import bcarray, pruning
from .impl import Impl, spec_to_hole
from .specs import Spec, Load, Store, Zero
from .system_config import current_system
from .utils import TinyMap, snap_availables_down, snap_availables_up, zip_dict

if TYPE_CHECKING:
    from .search.common import ScheduleKey

assert sys.version_info >= (3, 7), "Use Python 3.7 or newer"

RAISE_CAPS_ON_OVERLAP = True
PICKLE_PROTOCOL = 5

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
    def __init__(self, use_redis: Optional[tuple[Any, str]] = None):
        # TODO: `_rects` is unneeded tech. debt. They're identical except for a prefix.
        self._rects: dict[Spec, "_BlockCompressedTable"] = {}
        self._dirty_rects: set[Spec] = set()
        self._use_redis = use_redis

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
        results: list[Optional[CachedScheduleSet]] = [None] * len(subproblems)
        for idx, (spec, memory_limits) in enumerate(subproblems):
            if not isinstance(memory_limits, pruning.StandardMemoryLimits):
                # TODO: Add support for PipelineChildMemoryLimits
                warnings.warn(
                    "ScheduleCache only supports StandardMemoryLimits. Queries with"
                    " other MemoryLimits implementations always miss."
                )
                continue

            assert (
                not assert_access_on_log_boundaries.get()
                or snap_availables_down(memory_limits.available)
                == memory_limits.available
            )

            rects = self._get_rects(spec)
            try:
                best = await rects.best_for_cap(memory_limits)
            except KeyError:
                continue
            results[idx] = best.schedules
        return results

    async def put(
        self, schedules_to_put: CachedScheduleSet, memory_limits: pruning.MemoryLimits
    ) -> None:
        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Puts with"
                " other MemoryLimits implementations always miss."
            )
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
        await self._get_rects(spec).add(_TableEntry(spec, schedules_to_put, mlims))

    def _get_rects(self, spec):
        """Lazily initializes and returns an entry in `_rects`."""
        banks = current_system().ordered_banks
        try:
            rects = self._rects[spec]
        except KeyError:
            table_dims = tuple(current_system().banks[b].capacity for b in banks)
            redis_param = None
            if self._use_redis:
                redis_param = (self._use_redis[0], f"{self._use_redis[1]}-{spec}")
            bs = None
            if isinstance(spec, (Load, Store, Zero)):
                bs = 64  # Should cover whole space, using a single block.
            rects = _BlockCompressedTable(
                banks, table_dims, use_redis=redis_param, block_shape=bs
            )
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
            rects = self._get_rects(spec)
            async for rect in rects:
                assert rect.spec == spec
                yield spec, rect.schedules, pruning.StandardMemoryLimits(rect.caps)

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


class _MemAwareCache(Generic[T, U]):
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


@dataclasses.dataclass(frozen=False)
class _BlockCompressedTable:
    banks: tuple[str, ...]
    storage: bcarray.BlockCompressedArray

    def __init__(
        self,
        banks: tuple[str, ...],
        table_dims: tuple[int, ...],
        use_redis: Optional[tuple[Any, str]] = None,
        block_shape: Optional[Union[int, Sequence[int]]] = None,
    ):
        storage_dims = self._logify(snap_availables_up(table_dims, always=True))
        storage_dims = tuple(d + 1 for d in storage_dims)
        if not block_shape:
            block_shape = tuple(6 for _ in storage_dims)
        if isinstance(block_shape, int):
            block_shape = tuple(block_shape for _ in storage_dims)
        object.__setattr__(self, "banks", banks)
        object.__setattr__(
            self,
            "storage",
            bcarray.BlockCompressedArray(
                storage_dims, block_shape=tuple(block_shape), use_redis=use_redis
            ),
        )

    async def best_for_cap(self, caps: pruning.StandardMemoryLimits) -> _TableEntry:
        snapped = pruning.StandardMemoryLimits(snap_availables_up(caps.available))
        if snapped == caps:
            return await self._get_entry(caps)

        # If we're not on a snapped/log boundary, then we'll check an uper entry (the
        # entry where all levels are higher than the given caps) and see if, in
        # actuality, that Impl is below the requested limits. If it's not, we'll just
        # grab the Impl from the conservative entry.
        try:
            upper_impl = await self._get_entry(snapped)
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
        return await self._get_entry(snapped_down)

    async def add(self, entry: _TableEntry):
        bot = self._bottom_from_entry(entry)
        top = snap_availables_up(entry.caps.raw_values, always=True)
        bottom_coord = tuple(v.bit_length() for v in bot)
        top_coord = tuple(v.bit_length() for v in top)
        assert all(b <= t for b, t in zip(bottom_coord, top_coord)), (
            f"Bottom coord {bottom_coord} is not less than or equal to "
            f"top coord {top_coord}; bot = {bot}; top = {top}"
        )
        await self._fill_storage(bottom_coord, top_coord, entry)

    async def flush(self) -> None:
        await self.storage.flush()

    def _banks(self) -> tuple[str, ...]:
        return self.banks

    async def _fill_storage(self, lower, upper, value):
        assert not isinstance(value, np.ndarray)
        await self.storage.fill_range(lower, tuple(u + 1 for u in upper), value)

    async def __aiter__(self) -> AsyncIterator[_TableEntry]:
        seen = set()
        async for entry in self.storage.iter_values():
            if entry is None or entry in seen:
                continue
            assert isinstance(entry, _TableEntry)
            yield entry
            seen.add(entry)

    async def _get_log_scaled_pt(self, pt) -> _TableEntry:
        try:
            result = await self.storage.get(pt)
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

    async def _get_entry(self, caps: pruning.StandardMemoryLimits) -> _TableEntry:
        avail = caps.available
        coord = self._logify(avail.raw_values)
        assert isinstance(coord, tuple)
        return await self._get_log_scaled_pt(coord)

    @staticmethod
    def _logify(tup: Sequence[int]) -> tuple[int, ...]:
        return tuple((0 if x == 0 else (x - 1).bit_length() + 1) for x in tup)


class ChainCache(ScheduleCache):
    def __init__(self, caches: Iterable[ScheduleCache]) -> None:
        self._inner_caches = list(caches)

    async def get_many(
        self, subproblems: Sequence[tuple[Spec, pruning.MemoryLimits]]
    ) -> Iterable[Optional[CachedScheduleSet]]:
        remaining_idxs: list[int] = list(range(len(subproblems)))
        results: list[Optional[CachedScheduleSet]] = [None] * len(remaining_idxs)
        for cache in self._inner_caches:
            one_result = await cache.get_many([subproblems[i] for i in remaining_idxs])
            to_del = set()  # TODO: Don't use a set
            for subproblem_idx, entry in zip(remaining_idxs, one_result):
                if entry is not None:
                    to_del.add(subproblem_idx)
                    results[subproblem_idx] = entry
            remaining_idxs = [i for i in remaining_idxs if i not in to_del]
        return results

    async def put(self, *args, **kwargs) -> None:
        return await self._inner_caches[0].put(*args, **kwargs)

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
    redis: Optional[tuple[str, str]] = None,
    save: bool = True,
):
    with _local_persistent_cache(path, save=save) as local_cache:
        if not redis:
            yield local_cache
        else:
            redis_cache = _ReadonlyCache(use_redis=redis)
            yield ChainCache([local_cache, redis_cache])


class _ReadonlyCache(ScheduleCache):
    def put(self, *args, **kwargs):
        raise Exception("Should never put. This is read-only.")


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
