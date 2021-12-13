import contextlib
import math
import pathlib
import pickle
import warnings
from typing import Iterable, Iterator, NamedTuple, Optional, Tuple, Union

import atomicwrites
from frozendict import frozendict

from . import pruning
from .impl import Impl
from .specs import Spec
from .utils import next_power_of_two, zip_dict


class CachedSchedule(NamedTuple):
    """A container for schedules stored in the cache.

    Stores, along with the schedule itself, the cost of the schedule.
    """

    schedule: Impl
    cost: int


class _TableEntry(NamedTuple):
    """Stores the best schedule for a region from its used memory to `caps`.

    A _TabelEntry is a record of the best schedule that exists up to `caps`.
    That schedule is no longer the best as soon as any memory capacity is above
    its corresponding level in `caps` or below the memory used at that level by
    the schedule.
    """

    spec: Spec
    schedule: Optional[CachedSchedule]
    caps: frozendict[str, int]

    @property
    def peak_memory(self) -> frozendict[str, int]:
        """Returns peak memory used by the nested schedule.

        This is just a convenience accessor.
        """
        assert self.schedule is not None
        peaks = self.schedule.schedule.peak_memory
        # If it's a frozendict, we don't need the conversion
        assert not isinstance(peaks, frozendict)
        return frozendict(peaks)


class ScheduleCache:
    _rects: dict[Spec, list[_TableEntry]]

    def __init__(self):
        self._rects = {}

    def get(
        self, spec: Spec, memory_limits: pruning.MemoryLimits
    ) -> Optional[CachedSchedule]:
        """Returns a cached schedule or `None` if no Impl satisfies the given limits.

        Raises KeyError if no Impl meeting the given limits exists and it is unknown
        whether or not such an Impl exists (i.e., search should be done).
        """
        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Queries with"
                " other MemoryLimits implementations always miss."
            )
            raise KeyError(f"'{str(spec)}'")

        memory_caps = memory_limits.available
        for entry in self._rects[spec]:
            if entry.schedule is None:  # No Impl exists for this entry
                if all(
                    q <= b
                    for q, b in zip_dict(
                        memory_caps, entry.caps, same_keys=True
                    ).values()
                ):
                    return None
            else:
                if all(
                    a <= q <= b
                    for a, q, b in zip_dict(
                        entry.peak_memory, memory_caps, entry.caps, same_keys=True
                    ).values()
                ):
                    return entry.schedule
        raise KeyError(f"'{str(spec)}'")

    def put(
        self,
        spec: Spec,
        schedule: Optional[CachedSchedule],
        memory_limits: pruning.MemoryLimits,
    ) -> None:
        assert schedule is None or spec == schedule.schedule.spec

        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Puts with"
                " other MemoryLimits implementations always miss."
            )
            return

        memory_caps = memory_limits.available
        assert schedule is None or all(
            m <= c
            for m, c in zip_dict(
                schedule.schedule.peak_memory, memory_caps, same_keys=True
            ).values()
        )
        rects = self._rects.setdefault(spec, [])

        for idx in range(len(rects)):
            if schedule is None:
                if rects[idx].schedule is None:
                    rects[idx] = _TableEntry(
                        spec,
                        None,
                        frozendict(
                            {
                                k: max(a, b)
                                for k, (a, b) in zip_dict(
                                    memory_caps, rects[idx].caps, same_keys=True
                                ).items()
                            }
                        ),
                    )
                    return
            else:
                if rects[idx].schedule is None:
                    continue
                if all(
                    a <= b <= c
                    for a, b, c in zip_dict(
                        rects[idx].peak_memory,
                        schedule.schedule.peak_memory,
                        rects[idx].caps,
                        same_keys=True,
                    ).values()
                ):
                    rects[idx] = _TableEntry(
                        spec,
                        schedule,
                        frozendict(
                            {
                                k: max(a, b)
                                for k, (a, b) in zip_dict(
                                    memory_caps, rects[idx].caps, same_keys=True
                                ).items()
                            }
                        ),
                    )
                    return
            # TODO: Assert that there is at most one intersection

        # If we haven't returned at this point, then we didn't find a _TableEntry to
        # update, so add one.
        rects.append(_TableEntry(spec, schedule, memory_caps))

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
    ) -> Iterator[Tuple[Spec, Optional[CachedSchedule], pruning.MemoryLimits]]:
        for spec, rects in self._rects.items():
            for rect in rects:
                assert rect.spec == spec
                yield spec, rect.schedule, pruning.StandardMemoryLimits(rect.caps)


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
        cache = ScheduleCache()

    try:
        yield cache
    finally:
        if save:
            with atomicwrites.atomic_write(path, mode="wb", overwrite=True) as fo:
                pickle.dump(cache, fo)


def _validate_cache(cache) -> None:
    if not isinstance(cache, ScheduleCache):
        raise TypeError(f"Expected ScheduleCache, got {type(cache)}")
    # TODO: Make sure all the important contextvars match
    raise NotImplementedError()
