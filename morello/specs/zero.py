import dataclasses
from typing import Iterable, Sequence

from . import base
from .tensorspec import TensorSpec


@dataclasses.dataclass(frozen=True, slots=True)
class Zero(base.Spec):
    destination: TensorSpec
    serial_only: bool

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return tuple()

    @property
    def output(self) -> TensorSpec:
        return self.destination

    @classmethod
    def inputs_count(cls) -> int:
        return 0

    @classmethod
    def output_is_read_cls(cls) -> bool:
        return False

    @classmethod
    def from_io(
        cls, inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> base.Spec:
        return Zero(output, serial_only=serial_only)

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[tuple[int, ...]], output_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        return tuple()

    @classmethod
    def operands_dim_subscripts_cls(
        cls, operand_ranks: Sequence[int]
    ) -> Sequence[tuple[int, ...]]:
        assert len(operand_ranks) == 1
        out_rank = operand_ranks[0]
        return (tuple(range(out_rank)),)

    def __str__(self) -> str:
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"{type(self).__name__}({self.destination}{epi})"
