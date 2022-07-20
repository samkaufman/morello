from typing import Iterable, Optional, Sequence, cast

import cython

try:
    from ..cython.cimports import base
except ImportError:
    pass

from . import base
from .tensorspec import TensorSpec


@cython.cclass
class ReduceSum(base.Spec):
    """Sums along the the innermost dimension of a tensor.

    Not defined for rank-1 tensors. The actual reduction function (sum, max, etc.) is
    missing because it isn't relevant to schedule search.
    """

    source: TensorSpec
    _output: TensorSpec
    _serial_only: bool

    def __init__(self, source: TensorSpec, output: TensorSpec, serial_only: bool):
        self.source = source
        self._output = output
        self._serial_only = serial_only

        assert len(source.dim_sizes) >= 2
        assert len(output.dim_sizes) == len(source.dim_sizes) - 1, (
            "Expected output shape to have one fewer dimensions than source; "
            f"source and output shapes were: {source.dim_sizes} and "
            f"{output.dim_sizes}"
        )
        assert output.dim_sizes == self.output_shape(source.dim_sizes), (
            f"Given output shape was {output.dim_sizes} but the computed "
            f"output shape is {self.output_shape(self.source.dim_sizes)} "
            f"(input shape: {source.dim_sizes})"
        )

    def __eq__(self, other):
        if type(other) != ReduceSum:
            return NotImplemented
        return (
            self.source == other.source
            and self.output == other.output
            and self.serial_only == other.serial_only
        )

    def __hash__(self) -> int:
        return hash((self.source, self.output, self.serial_only))

    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "ReduceSum":
        if len(inputs) != 1:
            raise ValueError("Expected 1 input; got {len(inputs)}")
        return ReduceSum(inputs[0], output, serial_only=serial_only)

    @property
    def output(self) -> TensorSpec:
        return self._output

    @property
    def serial_only(self) -> bool:
        return self._serial_only

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return (self.source,)

    @staticmethod
    def output_shape(source_shape: tuple[int, ...]) -> tuple[int, ...]:
        return source_shape[:-1]

    def shrink_for_tile_out(
        self,
        output_shape: tuple[int, ...],
        serial_only=None,
    ) -> "ReduceSum":
        if len(output_shape) != len(self.output.dim_sizes):
            raise ValueError(
                f"Expected {len(self.output.dim_sizes)} dimensions; got {len(output_shape)}"
            )
        for dim, dim_size in enumerate(output_shape):
            if dim_size <= 0:
                raise ValueError("All dimensions must be size 1 or greater")
            elif dim_size > self.output.dim_sizes[dim]:
                raise ValueError(
                    f"Dimensions {dim} was larger than "
                    f"{self.output.dim_sizes[dim]} ({dim_size} > "
                    f"{self.output.dim_sizes[dim]})"
                )
        return cast(ReduceSum, super().shrink_for_tile_out(output_shape, serial_only))

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
    def operands_dim_subscripts_cls(cls, operand_ranks: Sequence[int]) -> Sequence[tuple[int, ...]]:
        inp_rank, out_rank = operand_ranks
        assert inp_rank == out_rank + 1
        return tuple(range(inp_rank)), tuple(range(out_rank))

    def __str__(self):
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"ReduceSum({self.source}, {self.output}{epi})"
