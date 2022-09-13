import contextlib
import pathlib
import pickle
import warnings
from typing import Iterable, Iterator, Mapping, NamedTuple, Optional, Tuple, Union

import atomicwrites
from frozendict import frozendict

from . import pruning
from .impl import Impl
from .specs import Spec
from .utils import zip_dict


class CachedScheduleSet:
    """A container for schedules stored in the cache.

    Stores, along with schedules themselves, costs of the schedules.
    """

    contents: tuple[tuple[Impl, int], ...]
    dependent_paths: int

    def __init__(self, contents: tuple[tuple[Impl, int]], specs_visited: int):
        self.contents = contents
        self.dependent_paths = specs_visited
        self.peak_memory = None
        if contents:
            self.peak_memory = frozendict(
                (bank, max(all_bytes))
                for bank, all_bytes in zip_dict(
                    *(imp.peak_memory for imp, _ in self.contents), same_keys=True
                ).items()
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
    """Stores the best schedule for a region from its used memory to `caps`.

    A _TableEntry is a record of the best schedule that exists up to `caps`.
    That schedule is no longer the best as soon as any memory capacity is above
    its corresponding level in `caps` or below the memory used at that level by
    the schedule.
    """

    spec: Spec
    schedules: CachedScheduleSet
    caps: Mapping[str, int]

    @property
    def peak_memory(self) -> Optional[Mapping[str, int]]:
        return self.schedules.peak_memory


class ScheduleCache:
    _rects: dict[Spec, list[_TableEntry]]

    def __init__(self):
        self._rects = {}

    def get(
        self, spec: Spec, memory_limits: pruning.MemoryLimits
    ) -> CachedScheduleSet:
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
            for m, c in zip_dict(im.peak_memory, memory_caps_to_put, same_keys=True).values()
        )
        rects = self._rects.setdefault(spec, [])

        # First, check for a TableEntry to update.
        for rect_idx in range(len(rects)):
            raised_caps = frozendict(
                {
                    k: max(a, b)
                    for k, (a, b) in zip_dict(
                        memory_caps_to_put, rects[rect_idx].caps, same_keys=True
                    ).items()
                }
            )

            # If we're putting no Impls and there exists an entry already for
            # the no-Impl case, raise its memory caps to whatever we've explored.
            if not schedules_to_put.contents:
                if not rects[rect_idx].schedules.contents:
                    rects[rect_idx] = _TableEntry(spec, schedules_to_put, raised_caps)
                    return
            else:
                if not rects[rect_idx].schedules.contents:
                    continue
                if _mem_dicts_ordered(rects[rect_idx].peak_memory, schedules_to_put.peak_memory, rects[rect_idx].caps):
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


@contextlib.contextmanager
def persistent_cache(path: Optional[Union[str, pathlib.Path]], save: bool = True):
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
                pickle.dump(cache, fo)


def _mem_dicts_ordered(*dicts: Optional[Mapping[str, int]]) -> bool:
    filtered_dicts = [d for d in dicts if d is not None]
    while len(filtered_dicts) >= 2:
        head = filtered_dicts.pop(0)
        for a, b in zip_dict(head, filtered_dicts[0], same_keys=True).values():
            if a > b:
                return False
    return True