from typing import Iterable, Sequence

import cython

try:
    from ..cython.cimports import base
except ImportError:
    pass

from . import base
from .tensorspec import TensorSpec


@cython.dataclasses.dataclass(unsafe_hash=True)
@cython.cclass
class ConvolutionBase(base.Spec):
    """A batched, any-dimensional convolution.

    The lhs operand is an image of shape (batch, channels, spatial dims...).
    The rhs operand is filters of shape: (filters, channels, spatial dims...).

    Stride is 1. Padding is 0.
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

    def __post_init__(self):
        # The following is important enough that we don't want it disabled with
        # the interpreter's -O flag, so raise a ValueError rather than assert.
        expected_output_shape = self.output_shape(
            self.lhs.dim_sizes, self.rhs.dim_sizes
        )
        if self.output.dim_sizes != expected_output_shape:
            raise ValueError(
                f"Expected output tensor with shape {expected_output_shape}; "
                f"got {self.output.dim_sizes}"
            )

    @property
    def output(self) -> TensorSpec:
        return self._output

    @property
    def serial_only(self) -> bool:
        return self._serial_only

    @classmethod
    def from_io(
        cls, inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "ConvolutionBase":
        lhs, rhs = inputs
        return cls(lhs, rhs, output, serial_only=serial_only)

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return self.lhs, self.rhs

    @staticmethod
    def output_shape(
        image_shape: tuple[int, ...], filters_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        batch_cnt, channels = image_shape[:2]
        filter_cnt = filters_shape[0]
        assert channels == filters_shape[1], (
            f"Image had {channels} channels but filters had {filters_shape[1]} "
            f"channels"
        )
        output_spatials = tuple(
            (
                (img_dim - filt_dim + 1)
                for img_dim, filt_dim in zip(image_shape[2:], filters_shape[2:])
            )
        )
        return (batch_cnt, filter_cnt) + output_spatials

    @classmethod
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        return cls.output_shape(*input_shapes)

    @classmethod
    def inputs_count(cls) -> int:
        return 2

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[tuple[int, ...]], output_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        if len(output_shape) < 3:
            raise ValueError(
                f"Expected output shape to have at least 3 dimensions: {output_shape}"
            )

        input_img_shape, input_filter_shape = input_shapes
        batch_cnt, filter_cnt = output_shape[:2]
        smaller_lhs_dims = (batch_cnt, input_img_shape[1]) + tuple(
            i + f - 1 for i, f in zip(output_shape[2:], input_filter_shape[2:])
        )
        smaller_rhs_dims = (filter_cnt,) + input_filter_shape[1:]
        return (smaller_lhs_dims, smaller_rhs_dims)

    @classmethod
    def operands_dim_subscripts_cls(
        cls, operand_ranks: Sequence[int]
    ) -> Sequence[tuple[int, ...]]:
        # Currently, this supports just 2 dimensions.
        # TODO: Extend this to arbitrary number of spatial dimensions.
        b, f, c, h, w, fh, fw = 0, 1, 2, 3, 4, 5, 6
        img = (b, c, h, w)
        filt = (f, c, fh, fw)
        out = (b, f, h, w)
        return (img, filt, out)

    def __str__(self):
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"Conv({self.lhs}, {self.rhs}, {self.output}{epi})"


@cython.cclass
class Convolution(ConvolutionBase):
    @classmethod
    def output_is_read_cls(cls) -> bool:
        return False

    @classmethod
    def short_name(cls):
        return "Conv"


@cython.cclass
class ConvolutionAccum(ConvolutionBase):
    @classmethod
    def output_is_read_cls(cls) -> bool:
        return True

    @classmethod
    def short_name(cls):
        return "ConvAccum"
