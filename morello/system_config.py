import dataclasses
import sys
from typing import List

import numpy as np


# TODO: Re-freeze
@dataclasses.dataclass(frozen=False)
class MemoryLevelConfig:
    cache_hit_cost: int
    capacity: int


# TODO: Re-freeze
@dataclasses.dataclass(frozen=False)
class SimpleSystemConfig:
    """Describes hardware simulated by a SimpleSystem."""

    line_size: int
    level_configs: List[MemoryLevelConfig]
    processors: int
    dtype: "_Dtype"

    def __post_init__(self):
        assert self.level_configs[-1].capacity == sys.maxsize
        for a, b in zip(self.level_configs[:-1], self.level_configs[1:]):
            assert a.cache_hit_cost < b.cache_hit_cost
            assert a.capacity < b.capacity
        assert self.processors >= 1


@dataclasses.dataclass(frozen=True)
class _Dtype:
    size: int  # in bytes
    c_type: str
    np_type: np.dtype


DEFAULT_SYSTEM_CONFIG = SimpleSystemConfig(
    line_size=64,
    level_configs=[
        MemoryLevelConfig(cache_hit_cost=0, capacity=100),
        MemoryLevelConfig(cache_hit_cost=10, capacity=sys.maxsize),
    ],
    processors=4,
    dtype=_Dtype(1, "uint8_t", np.uint8),
)
