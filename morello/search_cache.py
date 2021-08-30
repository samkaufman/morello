import contextlib
import pathlib
import pickle
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, \
    Union
import warnings

import atomicwrites

from . import pruning
from .ops import Schedule
from .specs import Spec


class CachedSchedule(NamedTuple):
    """A container for schedules stored in the cache.

    Stores, along with the schedule itself, the cost of the schedule.
    """

    schedule: Schedule
    cost: int


class _Rect(NamedTuple):
    """Stores the best schedule for a region from its used memory to `caps`.

    Interpret a _Rect as a record of the best schedule that exists up to `caps`.
    That schedule is no longer the best as soon as any memory capacity is above
    its corresponding level in `caps` or below the memory used at that level by
    the schedule.
    """

    spec: Spec
    schedule: Optional[CachedSchedule]
    caps: Tuple[int, ...]

    @property
    def peak_memory(self) -> Iterable[int]:
        """Returns an iterable of peak memory used by the nested schedule.

        This is just a convenience accessor.
        """
        assert self.schedule is not None
        return iter(self.schedule.schedule.peak_memory)


class ScheduleCache:
    _rects: Dict[Spec, List[_Rect]]

    def __init__(self):
        self._rects = {}

    def get(
        self, spec: Spec, memory_limits: pruning.MemoryLimits
    ) -> Optional[CachedSchedule]:
        """Returns a CachedSchedule or None if no schedule exists below given caps.

        Raises KeyError if neither a schedule nor the fact that no schedule
        exists below that cap exists.
        """
        if not isinstance(memory_limits, pruning.StandardMemoryLimits):
            # TODO: Add support for PipelineChildMemoryLimits
            warnings.warn(
                "ScheduleCache only supports StandardMemoryLimits. Queries with"
                " other MemoryLimits implementations always miss."
            )
            raise KeyError(f"'{str(spec)}'")

        memory_caps = memory_limits.available
        for rect in self._rects[spec]:
            if rect.schedule is None:
                if all(q <= b for q, b in zip(memory_caps, rect.caps)):
                    return None
            else:
                if all(
                    a <= q <= b
                    for a, q, b in zip(rect.peak_memory, memory_caps, rect.caps)
                ):
                    return rect.schedule
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
            m <= c for m, c in zip(schedule.schedule.peak_memory, memory_caps)
        )
        rects = self._rects.setdefault(spec, [])

        for idx in range(len(rects)):
            if schedule is None:
                if rects[idx].schedule is None:
                    rects[idx] = _Rect(
                        spec,
                        None,
                        tuple(max(a, b) for a, b in zip(memory_caps, rects[idx].caps)),
                    )
                    return
            else:
                if rects[idx].schedule is None:
                    continue
                if all(
                    a <= b <= c
                    for a, b, c in zip(
                        rects[idx].peak_memory,
                        schedule.schedule.peak_memory,
                        rects[idx].caps,
                    )
                ):
                    rects[idx] = _Rect(
                        spec,
                        schedule,
                        tuple(max(a, b) for a, b in zip(memory_caps, rects[idx].caps)),
                    )
                    return
            # TODO: Assert that there is at most one intersection

        # If we haven't returned at this point, then we didn't find a _Rect to
        # update, so add one.
        rects.append(_Rect(spec, schedule, memory_caps))

    def update(self, other: "ScheduleCache") -> None:
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
            # TODO: Is ValueError the correct exception here?
            raise ValueError(f"Expected a path to a file; got: {str(path)}")
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
