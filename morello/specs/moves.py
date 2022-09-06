import dataclasses
from typing import Iterable, Sequence

from . import base
from .tensorspec import TensorSpec


@dataclasses.dataclass(frozen=True)
class _MoveBase(base.Spec):
    source: TensorSpec
    destination: TensorSpec
    serial_only: bool

    def __post_init__(self):
        assert self.source.dim_sizes == self.destination.dim_sizes, (
            f"Expected source and destination tensors to have the same shape,"
            f"but were {self.source.dim_sizes} and {self.destination.dim_sizes}"
        )
        assert self.source.dtype == self.destination.dtype, (
            f"Expected source and destination tensors to have the same dtype,"
            f"but were {self.source.dtype} and {self.destination.dtype}"
        )
        # TODO: Check that destination bank is the same or "deeper"

    @property
    def _is_store(self) -> bool:
        if self.source.bank == "RF":
            assert self.destination.bank == "GL"
            return True
        elif self.source.bank == "GL":
            assert self.destination.bank == "RF"
            return False
        raise NotImplementedError()

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return (self.source,)

    @property
    def output(self) -> TensorSpec:
        return self.destination

    @classmethod
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        return next(iter(input_shapes))

    @classmethod
    def inputs_count(cls) -> int:
        return 1

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[tuple[int, ...]], output_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        return (output_shape,)

    @classmethod
    def operands_dim_subscripts_cls(
        cls, operand_ranks: Sequence[int]
    ) -> Sequence[tuple[int, ...]]:
        inp_rank, out_rank = operand_ranks
        assert inp_rank == out_rank
        t = tuple(range(inp_rank))
        return t, t

    def __str__(self) -> str:
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"{type(self).__name__}({self.source}, {self.destination}{epi})"


class Load(_MoveBase):
    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "_MoveBase":
        return Load(inputs[0], output, serial_only=serial_only)


class Store(_MoveBase):
    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "_MoveBase":
        return Store(inputs[0], output, serial_only=serial_only)
