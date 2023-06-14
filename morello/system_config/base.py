import abc
import dataclasses
import functools
import logging
import math
import typing
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

if TYPE_CHECKING:
    from .. import dtypes
    from ..layouts import Layout
    from ..specs import TensorSpec
    from ..tensor import TensorBase

MIN_SAMPLES = 3
MIN_TRIAL_TIME_SECS = 2.5

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunResult:
    stdout: str
    stderr: str


@dataclasses.dataclass
class RobustTimingResult:
    result: float
    outer_loop_samples: Sequence[float]
    inner_loop_iterations: int
    artifact: "BuiltArtifact"


class Target:
    def tensor(
        self, spec: "TensorSpec", name: Optional[str] = None, **kwargs
    ) -> "TensorBase":
        raise NotImplementedError()

    def tensor_spec(
        self,
        dim_sizes: tuple[int, ...],
        dtype: "dtypes.Dtype",
        contiguous_abs=None,
        bank: Optional[str] = None,
        layout: Optional["Layout"] = None,
        **kwargs,
    ) -> "TensorSpec":
        raise NotImplementedError()

    @property
    def system(self) -> "SystemDescription":
        raise NotImplementedError()

    def all_layouts_for_shape(self, shape: Sequence[int]) -> Iterable["Layout"]:
        from ..layouts import COL_MAJOR, NHWC, row_major

        possible_layouts = [row_major(len(shape)), COL_MAJOR, NHWC]
        for layout in possible_layouts:
            if layout.applies_to_shape(shape):
                yield layout

    async def build_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        values=None,
        extra_clang_args: Optional[Iterable[str]] = None,
        benchmark_samples: Optional[int] = None,
    ) -> "BuiltArtifact":
        raise NotImplementedError()

    async def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        values=None,
        check_flakiness: int = 1,
        extra_clang_args: Optional[Iterable[str]] = None,
    ) -> RunResult:
        raise NotImplementedError()

    @typing.final
    async def time_impl_robustly(self, impl, repeat=10) -> RobustTimingResult:
        """Benchmark several times, returning the minimum of inner loop means.

        This will first estimate a good number of inner loop iterations, then
        build an executable which loops that number of times, returning the mean.
        The final `result` computed is the minimum of the means after running
        that executable `repeat` times.
        """
        # Collect a single rough sample.
        time_check_artifact = await self.build_impl(impl, benchmark_samples=1)
        rough_secs = await time_check_artifact.measure_time()

        # Choose a good number of iterations for benchmarks' inner loop.
        inner_iters = max(MIN_SAMPLES, int(math.ceil(MIN_TRIAL_TIME_SECS / rough_secs)))
        logger.debug("Goal iterations: %d", inner_iters)

        # Run main benchmark loop.
        artifact = await self.build_impl(impl, benchmark_samples=inner_iters)
        means = []
        for _ in range(repeat):
            secs = await artifact.measure_time()
            logger.debug(f"Sample runtime result {secs}s:")
            means.append(secs)

        return RobustTimingResult(min(means), means, inner_iters, artifact)


class BuiltArtifact(abc.ABC):
    @abc.abstractmethod
    async def run(self, check_flakiness: int = 1) -> RunResult:
        pass

    @abc.abstractmethod
    async def measure_time(self) -> float:
        """Executes and benchmarks an Impl on the local machine.

        Returns the mean of the times in seconds.
        """
        pass

    @abc.abstractmethod
    def delete(self):
        pass


# TODO: Re-freeze. (Need a way to cache properties.)
@dataclasses.dataclass(frozen=False, eq=False)
class SystemDescription:
    """Describes hardware simulated by a SimpleSystem."""

    line_size: int
    banks: dict[str, "MemoryBankConfig"]
    default_bank: str
    processors: int
    faster_destination_banks: Callable[[str], set[str]]
    next_general_bank: Callable[[str], Optional[str]]
    ordered_banks: tuple[str, ...]
    addressed_banks: frozenset[str]  # TODO: Replace w/ lack of Alloc Specs

    def __post_init__(self):
        assert self.processors >= 1
        closure = self.destination_banks_closure(self.default_bank)
        assert set(self.ordered_banks) == closure

    @functools.cache
    def destination_banks_closure(self, bank: str) -> set[str]:
        closure = {bank}
        last_size = -1
        while last_size != len(closure):
            last_size = len(closure)
            for b in set(closure):
                closure.update(self.faster_destination_banks(b))
        return closure

    @property
    def default_fast_bank(self) -> str:
        bank = self.default_bank
        while True:
            next_bank = self.next_general_bank(bank)
            if next_bank is None:
                return bank
            bank = next_bank


# TODO: Re-freeze
@dataclasses.dataclass(frozen=False)
class MemoryBankConfig:
    cache_hit_cost: int
    capacity: int  # in bytes
    vector_bytes: Optional[int] = None

    @property
    def vector_rf(self) -> bool:
        return self.vector_bytes is not None
