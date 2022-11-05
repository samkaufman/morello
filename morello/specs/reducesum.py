from typing import Iterable, Optional, Sequence, cast

import cython

try:
    from ..cython.cimports import base
except ImportError:
    pass

from . import base
from .tensorspec import TensorSpec


@cython.dataclasses.dataclass(unsafe_hash=True)
@cython.cclass
class ReduceSumBase(base.Spec):
    """Sums along the the innermost dimension of a tensor.

    Not defined for rank-1 tensors. The actual reduction function (sum, max, etc.) is
    missing because it isn't relevant to schedule search.
    """

    source: TensorSpec
    _output: TensorSpec
    _serial_only: bool

    def __init__(self, source, output, serial_only) -> None:
        super().__init__()
        self.source = source
        self._output = output
        self._serial_only = serial_only

        assert len(self.source.dim_sizes) >= 2
        assert len(self.output.dim_sizes) == len(self.source.dim_sizes) - 1, (
            "Expected output shape to have one fewer dimensions than source; "
            f"source and output shapes were: {self.source.dim_sizes} and "
            f"{self.output.dim_sizes}"
        )
        assert self.output.dim_sizes == self.output_shape(self.source.dim_sizes), (
            f"Given output shape was {self.output.dim_sizes} but the computed "
            f"output shape is {self.output_shape(self.source.dim_sizes)} "
            f"(input shape: {self.source.dim_sizes})"
        )

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return (self.source,)

    @property
    def output(self) -> TensorSpec:
        return self._output

    @property
    def serial_only(self) -> bool:
        return self._serial_only

    @classmethod
    def from_io(
        cls, inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "ReduceSumBase":
        if len(inputs) != 1:
            raise ValueError("Expected 1 input; got {len(inputs)}")
        return cls(inputs[0], output, serial_only=serial_only)  # type: ignore

    @staticmethod
    def output_shape(source_shape: tuple[int, ...]) -> tuple[int, ...]:
        return source_shape[:-1]

    @classmethod
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        input_shapes = tuple(input_shapes)
        assert len(input_shapes) == 1, f"Expected one input; got {len(input_shapes)}"
        return input_shapes[0][:-1]

    @classmethod
    def inputs_count(cls) -> int:
        return 1

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[tuple[int, ...]], output_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        input_shapes = list(input_shapes)
        assert len(input_shapes) == 1, f"input_shapes was not size 1: {input_shapes}"
        assert len(output_shape) + 1 == len(input_shapes[0])
        return (output_shape + (input_shapes[0][-1],),)

    @classmethod
    def operands_dim_subscripts_cls(
        cls, operand_ranks: Sequence[int]
    ) -> Sequence[tuple[int, ...]]:
        inp_rank, out_rank = operand_ranks
        assert inp_rank == out_rank + 1
        return tuple(range(inp_rank)), tuple(range(out_rank))

    def __str__(self):
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"{type(self).__name__}({self.source}, {self.output}{epi})"


@cython.cclass
class ReduceSum(ReduceSumBase):
    @property
    def output_is_read(self) -> bool:
        return False


@cython.cclass
class ReduceSumAccum(ReduceSumBase):
    @property
    def output_is_read(self) -> bool:
        return True
