import functools
from typing import Callable, Iterable, Sequence

import cython

try:
    from ..cython.cimports import base
except ImportError:
    pass

from . import base
from ..dtypes import Dtype
from .tensorspec import TensorSpec


@cython.dataclasses.dataclass
@cython.cclass
class _ComposeSubspec:
    kls: Callable[..., base.Spec]
    inputs: tuple[tuple[int, ...]]
    output: tuple[int, ...]


@cython.cclass
class Compose(base.Spec):
    """Multiple specs where the first operand of each spec is the result of the next."""

    subspec_classes: tuple[Callable[..., base.Spec], ...]
    _inputs: tuple[TensorSpec, ...]
    _output: TensorSpec
    intermediate_dtypes: tuple[Dtype, ...]
    _serial_only: bool

    def __init__(
        self, subspec_classes, inputs, output, intermediate_dtypes, serial_only
    ):
        assert all(
            s != Compose for s in subspec_classes
        ), "Compose should not contain a nested Compose"
        assert isinstance(inputs, tuple), f"Given non-tuple inputs: {repr(inputs)}"
        assert len(inputs) == self.calculate_inputs_count(subspec_classes)
        assert len(intermediate_dtypes) + 1 == len(subspec_classes)

        self.subspec_classes = subspec_classes
        self._inputs = inputs
        self._output = output
        self.intermediate_dtypes = intermediate_dtypes
        self._serial_only = serial_only

    def __hash__(self):
        return hash((self.subspec_classes, self._output, self.intermediate_dtypes))

    def __eq__(self, other):
        if not isinstance(other, Compose):
            return False
        if self.subspec_classes != other.subspec_classes:
            return False
        if self.inputs != other.inputs:
            return False
        if self._output != other._output:
            return False
        if self.intermediate_dtypes != other.intermediate_dtypes:
            return False
        if self.serial_only != other.serial_only:
            return False
        return True

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return self._inputs

    @property
    def output(self) -> TensorSpec:
        return self._output

    @property
    def serial_only(self) -> bool:
        return self._serial_only

    def replace_io(
        self,
        inputs: tuple[TensorSpec, ...],
        output: TensorSpec,
        serial_only=None,
    ) -> base.Spec:
        if serial_only is None:
            serial_only = self.serial_only
        return Compose(
            self.subspec_classes, inputs, output, self.intermediate_dtypes, serial_only
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
        subspec_classes: tuple[Callable[..., base.Spec], ...],
        inputs_shapes: Iterable[tuple[int, ...]],
    ) -> tuple[int, ...]:
        return cls.calculate_subspec_outputs(subspec_classes, inputs_shapes)[0]

    @classmethod
    def calculate_subspec_outputs(
        cls,
        subspec_classes: tuple[Callable[..., base.Spec], ...],
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
            to_grab = kls.inputs_count() - len(inps)
            if to_grab:
                inps += inputs_shapes[-to_grab:]
                inputs_shapes = inputs_shapes[:-to_grab]
            accum.append(kls.calculate_output_shape_cls(inps))
        return tuple(reversed(accum))

    @staticmethod
    def calculate_inputs_count(
        subspec_classes: tuple[Callable[..., base.Spec], ...],
    ) -> int:
        return 1 + sum(c.inputs_count() for c in subspec_classes) - len(subspec_classes)

    def shrink_for_tile_out(
        self, output_shape: tuple[int, ...], serial_only=None
    ) -> base.Spec:
        # Forward pass to compute the initial input and output shapes for every subspec.
        # The initial input shapes  are used to resolve ambiguities in determining input
        # shapes from the new output.
        orig_input_shapes = self._expand_inputs()

        new_outermost_inp_shps = self.subspec_classes[0].shrink_inputs_for_output_shape(
            orig_input_shapes[0], output_shape
        )

        # The accumulator for the concatenated inputs
        new_input_shapes: tuple[tuple[int, ...]] = new_outermost_inp_shps[1:]
        last_input_shapes: tuple[tuple[int, ...]] = new_outermost_inp_shps

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
                inp_spec.shrink(inp_spec)
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
            s.dim_sizes for s in self.inputs[-self.subspec_classes[-1].inputs_count() :]
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
            assert kls.inputs_count() >= 1, "Compose not defined on nullary ops"
            new_inputs: tuple[int, ...] = (prev_output,) + tuple(
                s.dim_sizes
                for s in self.inputs[
                    1 - kls.inputs_count() - partials_gathered : -partials_gathered
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
    def inputs_count(cls) -> int:
        raise NotImplementedError("Use calculate_inputs_count instead")

    @classmethod
    def shrink_inputs_for_output_shape(
        cls, input_shapes: Iterable[tuple[int, ...]], output_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
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
        last_out_subs = None

        listed_subspecs = list(self._list_subspecs())
        listed_subspecs.reverse()
        for compose_subspec in listed_subspecs:  # start from innermost/first:
            kls = compose_subspec.kls
            # Increment subscripts immediately so that we can replace without
            # worrying about conflicts
            kls_ranks = [len(inp) for inp in compose_subspec.inputs] + [
                len(compose_subspec.output)
            ]
            kls_subscripts = Compose._increment_dims_subscripts(
                kls.operands_dim_subscripts_cls(kls_ranks), max_seen
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