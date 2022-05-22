import warnings
from typing import Iterable, Sequence

import cython

from .tensorspec import TensorSpec


@cython.cclass
class Spec:
    """The abstract root class for program specifications."""

    def replace_operand(self, operand_idx: int, new_operand: TensorSpec) -> "Spec":
        raise NotImplementedError()

    def replace_io(
        self,
        inputs: tuple[TensorSpec, ...],
        output: TensorSpec,
        serial_only=None,
    ) -> "Spec":
        # This method is similar to the static `from_io` except that it will
        # preserve properties other than the inputs and outputs. This is important for
        # Compose, which carries subspec kinds, not just inputs and arrays.
        if serial_only is None:
            serial_only = self.serial_only
        return self.from_io(inputs, output, serial_only=serial_only)

    @staticmethod
    def from_io(
        inputs: tuple[TensorSpec, ...], output: TensorSpec, *, serial_only: bool
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
    def inputs(self) -> tuple[TensorSpec, ...]:
        raise NotImplementedError()

    @property
    def output(self) -> TensorSpec:
        raise NotImplementedError()

    @property
    def operands(self) -> tuple[TensorSpec, ...]:
        return self.inputs + (self.output,)

    @property
    def serial_only(self) -> bool:
        raise NotImplementedError(f"serial_only not implemented for {type(self)}")

    def shrink_for_tile_out(
        self, output_shape: tuple[int, ...], serial_only=None
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
    def calculate_output_shape_cls(
        cls, input_shapes: Iterable[tuple[int, ...]]
    ) -> tuple[int, ...]:
        raise NotImplementedError()

    @classmethod
    def inputs_count(cls) -> int:
        raise NotImplementedError()

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[tuple[int, ...]], output_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        raise NotImplementedError()

    @classmethod
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
