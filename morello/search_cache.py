import abc
import numpy as np
import dataclasses
import contextlib
import itertools
import pathlib
import pickle
import warnings
from typing import (
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import atomicwrites

from . import pruning
from .impl import Impl, spec_to_hole
from .specs import Spec
from .utils import TinyMap, zip_dict, snap_availables_up
from .system_config import current_system

TABLE_STYLE: Literal["dense", "list", "dict"] = "list"
RAISE_CAPS = True


class CachedScheduleSet:
    """Wraps Impls with their cost and other metadata."""

    contents: tuple[tuple[Impl, int], ...]
    peak_memory: Optional[TinyMap[str, int]]
    dependent_paths: int

    def __init__(
        self,
        contents: tuple[tuple[Impl, int]],
        specs_visited: int,
        peak_memory: Optional[TinyMap[str, int]] = None,
    ):
        self.contents = contents
        self.dependent_paths = specs_visited
        self.peak_memory = peak_memory
        if contents and not peak_memory:
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

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class InMemoryScheduleCache(ScheduleCache):
    _rects: dict[Spec, Union["_ListTable", "_SharedTable"]]

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

        rects = self._rects[spec]
        best = rects.best_for_cap(memory_limits)
        if best is None:
            raise KeyError(f"({str(spec)}, {str(memory_limits)})")
        return self._despecify_schedule_set(best.schedules, memory_limits)

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

        mlims = memory_limits.available

        for im, _ in schedules_to_put.contents:
            for _, (m, c) in zip_dict(im.peak_memory, mlims, same_keys=True):
                if m > c:
                    raise ValueError(
                        f"Impl peak memory {im.peak_memory} not bounded by {mlims}"
                    )

        try:
            rects = self._rects[spec]
        except KeyError:
            if TABLE_STYLE == "dense":
                table_dims = tuple(
                    current_system().banks[b].capacity for b in mlims.raw_keys
                )
                rects = _LogScaledTable(mlims.raw_keys, table_dims)
            elif TABLE_STYLE == "list":
                rects = _ListTable(mlims.raw_keys)
            else:
                rects = _DictTable(mlims.raw_keys)
            self._rects[spec] = rects

        schedules_to_put = self._specify_schedule_set(schedules_to_put)

        # If we haven't returned at this point, then we didn't find a _TableEntry to
        # update, so add one.
        rects.add(_TableEntry(spec, schedules_to_put, mlims))

    def update(self, other: "ScheduleCache") -> None:
        """Update with the contents of another cache.

        Equivalent to `put`ing every entry from the given cache.
        """
        for spec, rect_schedule, limits in other:
            self.put(spec, rect_schedule, limits)

    def specs(self) -> Iterable[Spec]:
        yield from self._rects.keys()

    def count_impls(self) -> int:
        return sum(len(rect) for rect in self._rects.values())

    def __len__(self) -> int:
        return len(self._rects)

    def __iter__(
        self,
    ) -> Iterator[Tuple[Spec, CachedScheduleSet, pruning.MemoryLimits]]:
        for spec, rects in self._rects.items():
            for rect in rects:
                assert rect.spec == spec
                yield spec, rect.schedules, pruning.StandardMemoryLimits(rect.caps)

    def _specify_schedule_set(
        self, schedules_to_put: CachedScheduleSet
    ) -> CachedScheduleSet:
        return CachedScheduleSet(
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

    def _despecify_schedule_set(
        self, cached_set: CachedScheduleSet, limits
    ) -> CachedScheduleSet:
        new_impls = []
        for imp, cost in cached_set.contents:
            new_impls.append((self._despecify_impl(imp, limits), cost))

        return CachedScheduleSet(
            tuple(new_impls), cached_set.dependent_paths, cached_set.peak_memory
        )

    def _despecify_impl(self, imp: Impl, limits) -> Impl:
        all_child_limits = limits.transition(imp)
        assert (
            all_child_limits is not None
        ), f"Limits violated while transition from {limits} via {imp}"
        assert len(all_child_limits) == len(imp.children)

        new_children: list[Impl] = []
        for spec_child, child_limits in zip(imp.children, all_child_limits):
            assert not spec_child.is_scheduled
            child_results = self.get(spec_child.spec, child_limits)
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
        raise KeyError(f"({str(spec)}, {str(memory_limits)})")

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

    def __len__(self) -> int:
        return sum(len(cache) for cache in self.caches)


class _SharedTable(abc.ABC):
    def add(self, entry: _TableEntry):
        bottom_coord = self._logify(self._bottom_from_entry(entry))
        top_coord = self._logify(snap_availables_up(entry.caps.raw_values, always=True))
        self._fill_storage(bottom_coord, top_coord, entry)

    def best_for_cap(self, caps: pruning.StandardMemoryLimits) -> Optional[_TableEntry]:
        avail = caps.available
        coord = self._logify(avail.raw_values)
        assert isinstance(coord, tuple)
        return self._get_log_scaled_pt(coord)

    @abc.abstractmethod
    def _banks(self) -> tuple[str, ...]:
        pass

    @abc.abstractmethod
    def _get_log_scaled_pt(self, pt) -> Optional[_TableEntry]:
        pass

    @abc.abstractmethod
    def _fill_storage(self, lower, upper, value):
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

    def _get_log_scaled_pt(self, pt) -> Optional[_TableEntry]:
        return self.storage.get(pt, None)

    def __iter__(self) -> Iterator[_TableEntry]:
        for entry in set(self.storage.values()):
            assert isinstance(entry, _TableEntry)
            yield entry

    def __len__(self) -> int:
        return len(set(self.storage.values()))


@dataclasses.dataclass(frozen=False)
class _ListTable:
    banks: tuple[str, ...]
    storage: list[_TableEntry] = dataclasses.field(default_factory=list)

    @staticmethod
    def _raise_entry(r, entry):
        if RAISE_CAPS:
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
            raised_caps = None
            if RAISE_CAPS:
                raised_caps = TinyMap(
                    entry.caps.raw_keys,
                    tuple(
                        max(a, b)
                        for a, b in zip(
                            entry.caps.raw_values,
                            self.storage[rect_idx].caps.raw_values,
                        )
                    ),
                )

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

    def best_for_cap(self, caps: pruning.StandardMemoryLimits) -> Optional[_TableEntry]:
        for entry in self:
            if _mem_dicts_ordered(entry.peak_memory, caps.available, entry.caps):
                return entry
        return None

    def _get_log_scaled_pt(self, pt) -> Optional[_TableEntry]:
        for entry in self.storage:
            if _mem_dicts_ordered(entry.peak_memory, pt, entry.caps):
                return entry
        return None

    def __iter__(self) -> Iterator[_TableEntry]:
        yield from iter(self.storage)

    def __len__(self) -> int:
        return len(self.storage)


@dataclasses.dataclass(frozen=False)
class _LogScaledTable(_SharedTable):
    banks: tuple[str, ...]
    storage: np.ndarray

    def __init__(self, banks: tuple[str, ...], table_dims: Sequence[int]):
        storage_dims = self._logify(snap_availables_up(table_dims, always=True))
        storage_dims = tuple(d + 1 for d in storage_dims)
        object.__setattr__(self, "banks", banks)
        object.__setattr__(self, "storage", np.empty(storage_dims, dtype=object))
        self.storage.fill(None)

    def _banks(self) -> tuple[str, ...]:
        return self.banks

    def _fill_storage(self, lower, upper, value):
        """Fills storage (inclusive)."""
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

    def _get_log_scaled_pt(self, pt) -> Optional[_TableEntry]:
        try:
            return self.storage[pt]
        except IndexError:
            return None

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
