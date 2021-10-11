import abc
import dataclasses
import logging
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Union

if TYPE_CHECKING:
    from .. import dtypes
    from ..specs import TensorSpec
    from ..tensor import TensorBase, Tensor, Tile

logger = logging.getLogger(__name__)


class RunResult(NamedTuple):
    stdout: str
    stderr: str


class Target(abc.ABC):
    @abc.abstractmethod
    def tensor(
        self,
        spec: "TensorSpec",
        name: Optional[str],
        origin: Optional[Union["Tensor", "Tile"]] = None,
        **kwargs,
    ) -> "TensorBase":
        raise NotImplementedError()

    @abc.abstractmethod
    def tensor_spec(
        self,
        dim_sizes: tuple[int, ...],
        dtype: "dtypes.Dtype",
        bank: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "TensorSpec":
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def system(self) -> "SystemDescription":
        raise NotImplementedError()

    def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
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
    banks: dict[str, "MemoryBankConfig"]
    default_bank: str
    processors: int
    has_hvx: bool
    faster_destination_banks: Callable[[str], set[str]]
    next_general_bank: Callable[[str], Optional[str]]
    ordered_banks: list[str]

    def __post_init__(self):
        assert self.processors >= 1
        closure = self.destination_banks_closure(self.default_bank)
        assert set(self.ordered_banks) == closure

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
