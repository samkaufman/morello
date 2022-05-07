"""specs.py contains the Spec language.

In Morello, a program's algorithm is independent of its schedule. A schedule's
logical semantics are described by a Spec.
"""
import abc
import dataclasses
import enum
import functools
import operator
import typing
import warnings
from collections.abc import Sequence
from typing import Callable, Iterable, Optional, Tuple, TypeVar, cast

import dataclass_abc

from . import layouts
from .codegen.expr_utils import FloorDiv
from .dtypes import Dtype, Uint8
from .system_config.state import current_system

T = TypeVar("T")


@dataclasses.dataclass(frozen=True, init=False)
class TensorSpec:
    """A TensorSpec describes an operand to a Spec.

    This class is distinct from impl.Tensor and impl.Tile, which describe operands in
    the scheduling language.
    """

    dim_sizes: Tuple[int, ...]
    dtype: Dtype
    bank: str
    layout: layouts.Layout

    def __init__(
        self,
        dim_sizes: tuple[int, ...],
        dtype: Dtype,
        bank: Optional[str] = None,
        layout: layouts.Layout = layouts.ROW_MAJOR,
    ):
        object.__setattr__(self, "dim_sizes", dim_sizes)
        object.__setattr__(self, "dtype", dtype)
        if bank is None:
            object.__setattr__(
                self,
                "bank",
                current_system().default_bank,
            )
        else:
            object.__setattr__(self, "bank", bank)
        object.__setattr__(self, "layout", layout)
        if not len(self.dim_sizes):
            raise ValueError("dim_sizes cannot be empty")
        if all(d == 1 for d in self.dim_sizes) and self.layout != layouts.ROW_MAJOR:
            raise ValueError("If all dimensions are 1, layout must be row-major")

        if isinstance(self.layout, layouts.HexagonTranspacked):
            if self.dtype != Uint8:
                raise ValueError(
                    f"Cannot create transpacked tensor with type {self.dtype}"
                )
            if len(self.dim_sizes) != 2:
                raise ValueError(
                    f"Cannot create transpacked tensor with rank {len(self.dim_sizes)}"
                )
            if self.dim_sizes[0] % 4 != 0 or self.dim_sizes[1] % 32 != 0:
                raise ValueError(
                    f"Cannot create transpacked tensor with shape "
                    f"{self.dim_sizes}. Must be multiple of 4×32."
                )

    def shrink(self, new_dim_sizes: Tuple[int, ...]) -> "TensorSpec":
        """Returns a clone with new dimensions.

        If new_dim_sizes is all ones, the layout may be changed to row-major.
        """
        new_layout = self.layout
        if all(d == 1 for d in new_dim_sizes):
            new_layout = layouts.ROW_MAJOR
        return TensorSpec(
            new_dim_sizes, dtype=self.dtype, bank=self.bank, layout=new_layout
        )

    def is_valid_tile_shape(self, shape: tuple[int, ...]) -> bool:
        """Returns True if self can be tiled in this shape."""
        if len(shape) != len(self.dim_sizes):
            return False
        if not all(i <= o for (i, o) in zip(shape, self.dim_sizes)):
            return False
        if isinstance(self.layout, layouts.HexagonTranspacked):
            if self.dtype != Uint8:
                return False
            if shape[0] % 4 != 0 or shape[1] % 32 != 0:
                return False
        return True

    def __str__(self):
        layout_epi = ""
        bank_epi = ""
        if not isinstance(self.layout, layouts.RowMajor):
            layout_epi = f", {self.layout}"
        if self.bank != current_system().default_bank:
            bank_epi = f", {self.bank}"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        return f"({dims_part}, {self.dtype}{bank_epi}{layout_epi})"


@dataclasses.dataclass(frozen=True, init=False)
class HvxVmemTensorSpec(TensorSpec):
    vector_shape: tuple[int, ...]

    def __init__(self, *args, vector_shape: tuple[int, ...], **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "vector_shape", vector_shape)
        if any(s < vs for s, vs in zip(self.dim_sizes, self.vector_shape)):
            raise ValueError(
                f"Shape {self.dim_sizes} is smaller in some dimensions than vector shape {vector_shape}"
            )

    def is_valid_tile_shape(self, shape: tuple[int, ...]) -> bool:
        if not super().is_valid_tile_shape(shape):
            return False
        if any(i > v for (i, v) in zip(shape, self.vector_shape)):
            return False
        if functools.reduce(operator.mul, shape, 1) % 128 != 0:
            return False
        return True

    def __str__(self):
        base_str = super().__str__()[:-1]
        vs_dims_part = "×".join(str(s) for s in self.vector_shape)
        base_str = f"{base_str}, {vs_dims_part})"
        return base_str


class Spec(abc.ABC):
    """The abstract root class for program specifications."""

    def replace_io(
        self,
        inputs: tuple[TensorSpec, ...],
        output: TensorSpec,
        serial_only: Optional[bool] = None,
    ) -> "Spec":
        # This method is similar to the static `from_io` except that it will
        # preserve properties other than the inputs and outputs. This is important for
        # Compose, which carries subspec kinds, not just inputs and arrays.
        if serial_only is None:
            serial_only = self.serial_only
        return self.from_io(inputs, output, serial_only=serial_only)

    @staticmethod
    def from_io(
        inputs: Tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "Spec":
        raise NotImplementedError()

    def dim_size_for_subscript(self, subscript) -> int:
        result = None
        for op_idx, operand_sub in enumerate(self.operands_dim_subscripts()):
            for sub_idx, sub in enumerate(operand_sub):
                if sub == subscript:
                    if (
                        result is not None
                        and result != self.operands[op_idx].dim_sizes[sub_idx]
                    ):
                        warnings.warn(
                            "Using dim_size_for_subscript, which assumes dims "
                            "with matching subscripts have the same size, for a "
                            "Spec where that is not the case (subscript: "
                            f"{subscript})"
                        )
                    # Could return here immediately if we didn't want to assert
                    # the invariant that all possible sizes at that subscript
                    # have the same size.
                    result = self.operands[op_idx].dim_sizes[sub_idx]
        if result is None:
            raise KeyError("No subscript: " + str(subscript))
        return result

    @property
    @abc.abstractmethod
    def inputs(self) -> Tuple[TensorSpec, ...]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def output(self) -> TensorSpec:
        raise NotImplementedError()

    @typing.final
    @property
    def operands(self) -> tuple[TensorSpec, ...]:
        return self.inputs + (self.output,)

    @property
    @abc.abstractmethod
    def serial_only(self) -> bool:
        raise NotImplementedError()

    def shrink_for_tile_out(
        self, output_shape: Tuple[int, ...], serial_only: Optional[bool] = None
    ) -> "Spec":
        """Reduces the Spec to dimensions needed to compute the given output tile.

        The default implementation relies on `shrink_inputs_for_output_shape`.

        :returns: A copy of the callee with modified input and output dimensions.
        """
        if serial_only is None:
            serial_only = self.serial_only

        input_shapes = tuple(inp.dim_sizes for inp in self.inputs)
        new_inp_shapes = self.shrink_inputs_for_output_shape(input_shapes, output_shape)

        new_inputs = tuple(
            inp.shrink(new_shape) for inp, new_shape in zip(self.inputs, new_inp_shapes)
        )
        new_output = self.output.shrink(output_shape)

        return self.replace_io(new_inputs, new_output, serial_only=serial_only)

    def calculate_output_shape(
        self, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        return self.calculate_output_shape_cls(input_shapes)

    @classmethod
    @abc.abstractmethod
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[Tuple[int, ...]]
    ) -> Tuple[int, ...]:
        raise NotImplementedError()

    @classmethod
    @property
    def inputs_count(cls) -> int:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[Tuple[int, ...]], output_shape: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def operands_dim_subscripts(cls) -> Sequence[tuple[int, ...]]:
        """Returns which dimensions are the same size between operands.

        More specifically, this returns an iterable of the same length as the
        operands (len(self.inputs) + 1). Each element is a tuple of integers
        where the same integer shared between operands means those dimensions
        must match. An integer appearing once has no match requirements.

        For example, Matmul might be an iterable of `(0, 1)`, `(1, 2)` and
        `(0, 2)`.
        """
        raise NotImplementedError()


@dataclasses.dataclass
class _ComposeSubspec:
    kls: Callable[..., Spec]
    inputs: tuple[tuple[int, ...]]
    output: tuple[int, ...]


@dataclass_abc.dataclass_abc(frozen=True)
class Compose(Spec):
    """Multiple specs where the first operand of each spec is the result of the next."""

    subspec_classes: tuple[Callable[..., Spec], ...]
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    intermediate_dtypes: tuple[Dtype]
    serial_only: bool

    def __post_init__(self):
        assert all(
            s != Compose for s in self.subspec_classes
        ), "Compose should not contain a nested Compose"
        assert isinstance(
            self.inputs, tuple
        ), f"Given non-tuple inputs: {repr(self.inputs)}"
        assert len(self.inputs) == self.calculate_inputs_count(self.subspec_classes)
        assert len(self.intermediate_dtypes) + 1 == len(self.subspec_classes)

    def replace_io(
        self,
        inputs: Tuple[TensorSpec, ...],
        output: TensorSpec,
        serial_only: Optional[bool] = None,
    ) -> "Spec":
        if serial_only is None:
            serial_only = self.serial_only
        return Compose(
            subspec_classes=self.subspec_classes,
            inputs=inputs,
            output=output,
            intermediate_dtypes=self.intermediate_dtypes,
            serial_only=serial_only,
        )

    @property
    def subspec_outputs(self) -> tuple[tuple[int, ...], ...]:
        """The shapes of each output, including intermediate outputs.

        This is in Compose order, which is the inverse of evaluation order. For example,
        `compose.subspec_outputs[0] == compose.output.shape`.
        """
        return tuple(s.output for s in self._list_subspecs())

    @classmethod
    def calculate_output(
        cls,
        subspec_classes: Tuple[Callable[..., Spec], ...],
        inputs_shapes: Iterable[Tuple[int, ...]],
    ) -> tuple[int, ...]:
        return cls.calculate_subspec_outputs(subspec_classes, inputs_shapes)[0]

    @classmethod
    def calculate_subspec_outputs(
        cls,
        subspec_classes: tuple[Callable[..., Spec], ...],
        inputs_shapes: Iterable[tuple[int, ...]],
    ) -> tuple[tuple[int, ...], ...]:
        # This implementation has a lot in common with _list_subspecs. It exists
        # so that callers---notable codegen handling a tile boundary---can pass
        # in explicit inputs_shapes.
        accum = []
        inputs_shapes = list(inputs_shapes)
        for kls in reversed(subspec_classes):
            inps = []
            if accum:
                inps = [accum[-1]]
            to_grab = kls.inputs_count - len(inps)
            if to_grab:
                inps += inputs_shapes[-to_grab:]
                inputs_shapes = inputs_shapes[:-to_grab]
            accum.append(kls.calculate_output_shape_cls(inps))
        return tuple(reversed(accum))

    @staticmethod
    def calculate_inputs_count(
        subspec_classes: Tuple[Callable[..., Spec], ...],
    ) -> int:
        return 1 + sum(c.inputs_count for c in subspec_classes) - len(subspec_classes)

    def shrink_for_tile_out(
        self, output_shape: Tuple[int, ...], serial_only: Optional[bool] = None
    ) -> "Spec":
        # Forward pass to compute the initial input and output shapes for every subspec.
        # The initial input shapes  are used to resolve ambiguities in determining input
        # shapes from the new output.
        orig_input_shapes = self._expand_inputs()

        new_outermost_inp_shps = self.subspec_classes[0].shrink_inputs_for_output_shape(
            orig_input_shapes[0], output_shape
        )

        # The accumulator for the concatenated inputs
        new_input_shapes: Tuple[Tuple[int, ...]] = new_outermost_inp_shps[1:]
        last_input_shapes: Tuple[Tuple[int, ...]] = new_outermost_inp_shps

        for kls_idx in range(1, len(self.subspec_classes)):
            kls = self.subspec_classes[kls_idx]
            last_input_shapes = kls.shrink_inputs_for_output_shape(
                orig_input_shapes[kls_idx], last_input_shapes[0]
            )
            if kls_idx == len(self.subspec_classes) - 1:
                new_input_shapes += last_input_shapes
            else:
                new_input_shapes += last_input_shapes[1:]

        if serial_only is None:
            serial_only = self.serial_only
        return Compose(
            self.subspec_classes,
            tuple(
                dataclasses.replace(inp_spec, dim_sizes=shp)
                for inp_spec, shp in zip(self.inputs, new_input_shapes)
            ),
            self.output.shrink(output_shape),
            intermediate_dtypes=self.intermediate_dtypes,
            serial_only=serial_only,
        )

    # TODO: Rename & document _list_subspecs
    def _list_subspecs(self) -> list[_ComposeSubspec]:
        # Initialize with the first/innermost function's inputs. We'll lift
        # TensorSpecs into _ComposeTensorSpecs, which drops layout, bank, and
        # dtype information, because we can't know those in general for
        # intermediate outputs.
        innermost_spec_inputs = tuple(
            s.dim_sizes for s in self.inputs[-self.subspec_classes[-1].inputs_count :]
        )
        prev_output = self.subspec_classes[-1].calculate_output_shape_cls(
            innermost_spec_inputs
        )
        accum = [
            _ComposeSubspec(
                kls=self.subspec_classes[-1],
                inputs=innermost_spec_inputs,
                output=prev_output,
            )
        ]
        partials_gathered = len(innermost_spec_inputs)

        # Add the inputs for all following subspecs
        for kls_idx in range(len(self.subspec_classes) - 2, -1, -1):
            kls = self.subspec_classes[kls_idx]
            assert kls.inputs_count >= 1, "Compose not defined on nullary ops"
            new_inputs: tuple[int, ...] = (prev_output,) + tuple(
                s.dim_sizes
                for s in self.inputs[
                    1 - kls.inputs_count - partials_gathered : -partials_gathered
                ]
            )
            prev_output = kls.calculate_output_shape_cls(new_inputs)
            accum.append(_ComposeSubspec(kls, new_inputs, prev_output))
        accum.reverse()

        assert len(accum) == len(self.subspec_classes)
        return accum

    def _expand_inputs(self) -> list[tuple[tuple[int, ...], ...]]:
        return [tuple(i.dim_sizes for i in s.inputs) for s in self._list_subspecs()]

    def calculate_output_shape(
        self, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        return self.calculate_output(self.subspec_classes, input_shapes)

    @classmethod
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        raise NotImplementedError("Use calculate_output_shape instead")

    @classmethod
    @property
    def inputs_count(cls) -> int:
        raise NotImplementedError("Use calculate_inputs_count instead")

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[Tuple[int, ...]], output_shape: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], ...]:
        # TODO: Implement this by pulling from logic from ComposeHole's tile_out
        raise NotImplementedError()

    @staticmethod
    def _increment_dims_subscripts(
        subs: Sequence[Sequence[int]], inc: int
    ) -> Sequence[tuple[int, ...]]:
        result = []
        for dims in subs:
            subresult = []
            for d in dims:
                subresult.append(d + inc)
            result.append(tuple(subresult))
        return result

    @staticmethod
    def _sub_subscript(
        source: Sequence[Sequence[int]], substitutions: dict[int, int]
    ) -> Sequence[tuple[int, ...]]:
        result = []
        for dims in source:
            subresult = []
            for d in dims:
                subresult.append(substitutions.get(d, d))
            result.append(tuple(subresult))
        return result

    @functools.lru_cache(maxsize=1)
    def operands_dim_subscripts(self) -> Sequence[tuple[int, ...]]:
        # <subspec idx, subscript> -> new subscript
        max_seen = 0
        accum: list[tuple[int, ...]] = []
        last_out_subs: Optional[tuple[int, ...]] = None
        for kls in reversed(self.subspec_classes):  # start from innermost/first
            # Increment subscripts immediately so that we can replace without
            # worrying about conflicts
            kls_subscripts = Compose._increment_dims_subscripts(
                kls.operands_dim_subscripts(), max_seen
            )
            if not accum:
                accum += kls_subscripts[:-1]  # Drop the output only
                last_out_subs = kls_subscripts[-1]
            else:
                assert last_out_subs is not None
                assert len(last_out_subs) == len(kls_subscripts[0])
                kls_subscripts = Compose._sub_subscript(
                    kls_subscripts, dict(zip(kls_subscripts[0], last_out_subs))
                )
                last_out_subs = kls_subscripts[-1]
                accum = kls_subscripts[1:-1] + accum

            max_seen = max(d for t in kls_subscripts for d in t)

        # Add the Compose' output
        assert last_out_subs is not None
        accum.append(last_out_subs)

        assert len(accum) == len(self.inputs) + 1
        return accum

    def __str__(self):
        inner = "·".join(
            getattr(s, "short_name", lambda: s.__name__)() for s in self.subspec_classes
        )
        inps_str = ", ".join(map(str, self.inputs))
        dtype_str = ", ".join(map(str, self.intermediate_dtypes))
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"Compose({inner}, {inps_str}, out={self.output}, [{dtype_str}]{epi})"


@dataclass_abc.dataclass_abc(frozen=True)
class Matmul(Spec):
    """A matrix multiplication.

    Both lhs and rhs operands must be rank-2 TensorSpecs (matrices).
    """

    lhs: TensorSpec
    rhs: TensorSpec
    output: TensorSpec
    serial_only: bool

    def __post_init__(self):
        if self.output.dim_sizes != (self.lhs.dim_sizes[0], self.rhs.dim_sizes[1]):
            raise ValueError(f"Incorrect shape for matmul output: {self.output}")

    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "Matmul":
        lhs, rhs = inputs
        return Matmul(lhs, rhs, output, serial_only=serial_only)

    @property
    def inputs(self) -> Tuple[TensorSpec, ...]:
        return self.lhs, self.rhs

    @classmethod
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[Tuple[int, ...]]
    ) -> Tuple[int, ...]:
        input_shapes = tuple(input_shapes)
        assert len(input_shapes) == 2
        input_shapes = list(input_shapes)
        return input_shapes[0][0], input_shapes[1][1]

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def inputs_count(cls) -> int:
        return 2

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[Tuple[int, ...]], output_shape: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], ...]:
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


@dataclass_abc.dataclass_abc(frozen=True)
class Convolution(Spec):
    """A batched, any-dimensional convolution.

    The lhs operand is an image of shape (batch, channels, spatial dims...).
    The rhs operand is filters of shape: (filters, channels, spatial dims...).

    Stride is 1. Padding is 0.
    """

    lhs: TensorSpec
    rhs: TensorSpec
    output: TensorSpec
    serial_only: bool

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

    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "Convolution":
        lhs, rhs = inputs
        return Convolution(lhs, rhs, output, serial_only=serial_only)

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
        return Convolution.output_shape(*input_shapes)

    @classmethod
    @property
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
    def operands_dim_subscripts(cls) -> Sequence[tuple[int, ...]]:
        # Currently, this supports just 2 dimensions.
        # TODO: Extend this to arbitrary number of spatial dimensions.
        b, f, c, h, w, fh, fw = 0, 1, 2, 3, 4, 5, 6
        img = (b, c, h, w)
        filt = (f, c, fh, fw)
        out = (b, f, h, w)
        return (img, filt, out)

    @classmethod
    def short_name(cls):
        return "Conv"

    def __str__(self):
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"Conv({self.lhs}, {self.rhs}, {self.output}{epi})"


@dataclass_abc.dataclass_abc(frozen=True)
class ReduceSum(Spec):
    """Sums along the the innermost dimension of a tensor.

    Not defined for rank-1 tensors. The actual reduction function (sum, max, etc.) is
    missing because it isn't relevant to schedule search.
    """

    source: TensorSpec
    output: TensorSpec
    serial_only: bool

    def __post_init__(self):
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

    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
    ) -> "ReduceSum":
        if len(inputs) != 1:
            raise ValueError("Expected 1 input; got {len(inputs)}")
        return ReduceSum(inputs[0], output, serial_only=serial_only)

    @property
    def inputs(self) -> Tuple[TensorSpec, ...]:
        return (self.source,)

    @staticmethod
    def output_shape(source_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return source_shape[:-1]

    def shrink_for_tile_out(
        self, output_shape: tuple[int, ...], serial_only: Optional[bool] = None
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
        cls, input_shapes: Iterable[Tuple[int, ...]]
    ) -> Tuple[int, ...]:
        input_shapes = tuple(input_shapes)
        assert len(input_shapes) == 1, f"Expected one input; got {len(input_shapes)}"
        return input_shapes[0][:-1]

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def inputs_count(cls) -> int:
        return 1

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[Tuple[int, ...]], output_shape: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], ...]:
        input_shapes = list(input_shapes)
        assert len(input_shapes) == 1, f"input_shapes was not size 1: {input_shapes}"
        assert len(output_shape) + 1 == len(input_shapes[0])
        return (output_shape + (input_shapes[0][-1],),)

    @classmethod
    def operands_dim_subscripts(cls) -> Sequence[tuple[int, ...]]:
        return ((0, 1, 2, 3), (0, 1, 2))

    def __str__(self):
        epi = ""
        if self.serial_only:
            epi = ", serial"
        return f"ReduceSum({self.source}, {self.output}{epi})"
