from typing import Iterable, Sequence

import cython

from .base import Spec
from .tensorspec import TensorSpec


@cython.dataclasses.dataclass(unsafe_hash=True)
@cython.cclass
class Matmul(Spec):
    """A matrix multiplication.

    Both lhs and rhs operands must be rank-2 TensorSpecs (matrices).
    """

    lhs: TensorSpec
    rhs: TensorSpec
    _output: TensorSpec
    _serial_only: bool

    def __init__(self, lhs, rhs, output, serial_only) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self._output = output
        self._serial_only = serial_only
        if self.output.dim_sizes != (self.lhs.dim_sizes[0], self.rhs.dim_sizes[1]):
            raise ValueError(f"Incorrect shape for matmul output: {self.output}")

    @property
    def output(self) -> TensorSpec:
        return self._output

    @property
    def serial_only(self) -> bool:
        return self._serial_only

    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "Matmul":
        lhs, rhs = inputs
        return Matmul(lhs, rhs, output, serial_only=serial_only)

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return self.lhs, self.rhs

    @classmethod
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        input_shapes = tuple(input_shapes)
        assert len(input_shapes) == 2
        input_shapes = list(input_shapes)
        return input_shapes[0][0], input_shapes[1][1]

    @classmethod
    def inputs_count(cls) -> int:
        return 2

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[tuple[int, ...]], output_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        if len(output_shape) != 2:
            raise ValueError(f"Expected rank-2 output; got: {output_shape}")
        m, n = output_shape
        k = list(input_shapes)[0][1]
        return ((m, k), (k, n))

    @classmethod
    def operands_dim_subscripts(cls) -> Sequence[tuple[int, ...]]:
        return ((0, 2), (2, 1), (0, 1))

    def __str__(self):
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"Matmul({self.lhs}, {self.rhs}, {self.output}{epi})"
