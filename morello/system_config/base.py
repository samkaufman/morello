import abc
import dataclasses
import logging
import sys
from typing import List, NamedTuple

logger = logging.getLogger(__name__)


class RunResult(NamedTuple):
    stdout: str
    stderr: str


class Target(abc.ABC):
    @property
    @abc.abstractmethod
    def system(self) -> "SystemDescription":
        raise NotImplementedError()

    def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        extra_clang_args=None,
        values=None,
    ) -> RunResult:
        raise NotImplementedError()

    def time_impl(self, impl) -> float:
        raise NotImplementedError()


# TODO: Re-freeze
@dataclasses.dataclass(frozen=False)
class SystemDescription:
    """Describes hardware simulated by a SimpleSystem."""

    line_size: int
    level_configs: List["MemoryLevelConfig"]
    processors: int
    has_hvx: bool

    def __post_init__(self):
        assert self.level_configs[-1].capacity == sys.maxsize
        for a, b in zip(self.level_configs[:-1], self.level_configs[1:]):
            assert a.cache_hit_cost < b.cache_hit_cost
            assert a.capacity < b.capacity
        assert self.processors >= 1


# TODO: Re-freeze
@dataclasses.dataclass(frozen=False)
class MemoryLevelConfig:
    cache_hit_cost: int
    capacity: int
