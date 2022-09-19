import abc
import contextlib
import itertools
import pathlib
import pickle
import warnings
from typing import (
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import atomicwrites

from . import pruning
from .impl import Impl
from .specs import Spec
from .utils import TinyMap, zip_dict


class CachedScheduleSet:
    """Wraps Impls with their cost and other metadata."""

    contents: tuple[tuple[Impl, int], ...]
    peak_memory: Optional[TinyMap[str, int]]
    dependent_paths: int

    def __init__(self, contents: tuple[tuple[Impl, int]], specs_visited: int):
        self.contents = contents
        self.dependent_paths = specs_visited
        self.peak_memory = None
        if contents:
            self.peak_memory = TinyMap(
                self.contents[0][0].peak_memory.raw_keys,
                tuple(
                    max(zvals)
                    for zvals in zip(
                        *(c[0].peak_memory.raw_values for c in self.contents)
                    )
                ),
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CachedScheduleSet):
            return NotImplemented
        if self.contents != other.contents:
            return False
        if self.dependent_paths != other.dependent_paths:
            return False
        return True

    def __hash__(self) -> int:
        return hash(self.contents)

    def __str__(self) -> str:
        return f"CachedScheduleSet({self.contents}, {self.dependent_paths})"

    def __repr__(self) -> str:
        return f"CachedScheduleSet({repr(self.contents)}, {repr(self.dependent_paths)})"


class _TableEntry(NamedTuple):
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


class ScheduleCache(abc.ABC):
    @abc.abstractmethod
    def get(self, spec: Spec, memory_limits: pruning.MemoryLimits) -> CachedScheduleSet:
        pass

    @abc.abstractmethod
    def put(
        self,
        spec: Spec,
        schedules_to_put: CachedScheduleSet,
        memory_limits: pruning.MemoryLimits,
    ) -> None:
        pass

    @abc.abstractmethod
    def specs(self) -> Iterable[Spec]:
        pass

    @abc.abstractmethod
    def __iter__(
        self,
    ) -> Iterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        pass


class InMemoryScheduleCache(ScheduleCache):
    _rects: dict[Spec, list[_TableEntry]]

    def __init__(self):
        self._rects = {}

    def get(self, spec: Spec, memory_limits: pruning.MemoryLimits) -> CachedScheduleSet:
        """Returns cached Impls. May contain nothing if query has no Impls.

        Raises KeyError if no Impl meeting the given limits exists and it is unknown
        whether or not such an Impl exists (i.e., search should be done).
        """
        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Queries with"
                " other MemoryLimits implementations always miss."
            )
            raise KeyError()

        memory_caps = memory_limits.available
        for entry in self._rects[spec]:
            if _mem_dicts_ordered(entry.peak_memory, memory_caps, entry.caps):
                return entry.schedules
        raise KeyError()

    def put(
        self,
        spec: Spec,
        schedules_to_put: CachedScheduleSet,
        memory_limits: pruning.MemoryLimits,
    ) -> None:
        assert all(spec == imp.spec for imp, _ in schedules_to_put.contents)

        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Puts with"
                " other MemoryLimits implementations always miss."
            )
            return

        memory_caps_to_put = memory_limits.available
        assert all(
            m <= c
            for im, _ in schedules_to_put.contents
            for _, (m, c) in zip_dict(
                im.peak_memory, memory_caps_to_put, same_keys=True
            )
        )
        rects = self._rects.setdefault(spec, [])

        # First, check for a TableEntry to update.
        for rect_idx in range(len(rects)):
            raised_caps = TinyMap(
                memory_caps_to_put.raw_keys,
                tuple(
                    max(a, b)
                    for a, b in zip(
                        memory_caps_to_put.raw_values, rects[rect_idx].caps.raw_values
                    )
                ),
            )

            # If we're putting no Impls and there exists an entry already for
            # the no-Impl case, raise its memory caps to whatever we've explored.
            if not schedules_to_put.contents:
                if not rects[rect_idx].schedules.contents:
                    rects[rect_idx] = _TableEntry(spec, schedules_to_put, raised_caps)
                    return
            else:
                r = rects[rect_idx]
                if not r.schedules.contents:
                    continue
                assert r.peak_memory is None or isinstance(r.peak_memory, TinyMap)
                if _mem_dicts_ordered(
                    r.peak_memory, schedules_to_put.peak_memory, r.caps
                ):
                    rects[rect_idx] = _TableEntry(spec, schedules_to_put, raised_caps)
                    return
            # TODO: Assert that there is at most one intersection

        # If we haven't returned at this point, then we didn't find a _TableEntry to
        # update, so add one.
        rects.append(_TableEntry(spec, schedules_to_put, memory_caps_to_put))

    def update(self, other: "ScheduleCache") -> None:
        """Update with the contents of another cache.

        Equivalent to `put`ing every entry from the given cache.
        """
        for spec, rect_schedule, limits in other:
            self.put(spec, rect_schedule, limits)

    def specs(self) -> Iterable[Spec]:
        yield from self._rects.keys()

    def __iter__(
        self,
    ) -> Iterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        for spec, rects in self._rects.items():
            for rect in rects:
                assert rect.spec == spec
                yield spec, rect.schedules, pruning.StandardMemoryLimits(rect.caps)


class CacheChain(ScheduleCache):
    def __init__(self, caches: Sequence[ScheduleCache]):
        if len(caches) == 0:
            raise ValueError("`caches` must be non-empty")
        self.caches = caches

    def get(self, spec: Spec, memory_limits: pruning.MemoryLimits) -> CachedScheduleSet:
        for cache in self.caches:
            try:
                return cache.get(spec, memory_limits)
            except KeyError:
                pass
        raise KeyError()

    def put(
        self,
        spec: Spec,
        schedules_to_put: CachedScheduleSet,
        memory_limits: pruning.MemoryLimits,
    ) -> None:
        self.caches[0].put(spec, schedules_to_put, memory_limits)

    @property
    def specs(self) -> Iterable[Spec]:
        seen = set()
        for cache in self.caches:
            for spec in cache.specs():
                if spec not in seen:
                    yield spec
                    seen.add(spec)

    def __iter__(
        self,
    ) -> Iterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        yield from itertools.chain(*self.caches)


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
                pickle.dump(cache, fo)


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
