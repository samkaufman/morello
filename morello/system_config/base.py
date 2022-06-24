import abc
import dataclasses
import functools
import logging
import typing
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

if TYPE_CHECKING:
    from .. import dtypes
    from ..layouts import Layout
    from ..tensor import TensorBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunResult:
    stdout: str
    stderr: str


class Target(abc.ABC):
    @abc.abstractmethod
    def tensor(
        self,
        spec: "TensorSpec",
        name: Optional[str] = None,
        **kwargs,
    ) -> "TensorBase":
        raise NotImplementedError()

    @abc.abstractmethod
    def tensor_spec(
        self,
        dim_sizes: tuple[int, ...],
        dtype: "dtypes.Dtype",
        contiguous: bool = True,
        bank: Optional[str] = None,
        layout: Optional["Layout"] = None,
        **kwargs,
    ) -> "TensorSpec":
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def system(self) -> "SystemDescription":
        raise NotImplementedError()

    @property
    def all_layouts(self) -> Iterable["Layout"]:
        from ..layouts import ROW_MAJOR, NCHWc4, NCHWc32, NCHWc64

        return [ROW_MAJOR, NCHWc4, NCHWc32, NCHWc64]

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

    async def time_impl(self, impl) -> float:
        """Executes and benchmarks an Impl.

        Returns a measurement of time in arbitrary units.
        """
        raise NotImplementedError()

    @typing.final
    async def time_impl_robustly(self, impl, repeat=10) -> float:
        means = []
        for _ in range(repeat):
            means.append(await self.time_impl(impl))
        return min(means)


# TODO: Re-freeze. (Need a way to cache properties.)
@dataclasses.dataclass(frozen=False, eq=False)
class SystemDescription:
    """Describes hardware simulated by a SimpleSystem."""

    line_size: int
    banks: dict[str, "MemoryBankConfig"]
    default_bank: str
    processors: int
    has_hvx: bool
    faster_destination_banks: Callable[[str], set[str]]
    next_general_bank: Callable[[str], Optional[str]]
    ordered_banks: list[str]
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
