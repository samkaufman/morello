import dataclasses
import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import dataclass_abc
import termcolor
from torch import inner

from .. import layouts, specs, system_config
from ..layouts import Layout
from ..system_config import current_target
from ..tensor import Tensor, TensorLike, Tile
from . import MoveAction
from .base import AppliedImpl, Impl, make_applied_impl
from .utils import assert_stable_spec, gen_vector_shapes


class _OperandWrapper(Impl):
    # @property
    # def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
    #     new_inputs = []
    #     for inner_inp in self.inner.inputs:
    #         assert inner_inp is not self.source
    #         if inner_inp is self.destination:
    #             new_inputs.append(self.source)
    #         else:
    #             new_inputs.append(inner_inp)
    #     return tuple(new_inputs)

    # @property
    # def output(self):
    #     # A MoveLet can move any operand. This returns the source of the move if output
    #     # is the operand being moved; the inner output otherwise.
    #     if self.inner.output is self.destination:
    #         return self.source
    #     return self.inner.output

    @property
    def children(self) -> tuple[Impl, ...]:
        return (self.inner,)  # type: ignore

    @property
    def is_scheduled(self) -> bool:
        return self.inner.is_scheduled  # type: ignore

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        replacements = list(replacements)
        if len(replacements) != 1:
            raise ValueError(f"One replacement child expected; got {len(replacements)}")
        inner_spec: specs.Spec = self.inner.spec  # type: ignore
        if replacements[0].spec != inner_spec:
            raise ValueError(
                f"Expected a replacement with spec {inner_spec}, "
                f"but received {replacements[0].spec}"
            )
        return dataclasses.replace(self, inner=replacements[0])

    def move_input(self, *args, **kwargs) -> "Impl":
        # Pass move_input through to the inner schedule
        return self.replace_children((self.children[0].move_input(*args, **kwargs),))

    def move_output(self, *args, **kwargs) -> "Impl":
        # Pass move_output through to the inner schedule
        return self.replace_children((self.children[0].move_output(*args, **kwargs),))

    def pad_transpack(self, *args, **kwargs) -> "Impl":
        # Pass pad_transpack through to the inner schedule
        return self.replace_children((self.children[0].pad_transpack(*args, **kwargs),))

    @assert_stable_spec
    def split(self, size: int) -> "Impl":
        return self.replace_children((self.children[0].split(*args, **kwargs),))

    @assert_stable_spec
    def complete(self) -> Impl:
        return self.replace_children((self.children[0].complete(),))

    def _spec_with_replaced_operand(
        self, to_replace: TensorLike, replacement: TensorLike
    ) -> specs.Spec:
        # The spec of a MoveLet can be calculated by taking the spec of the inner
        # schedule and replacing the right inputs/output of the spec with the source
        # tensor's spec.
        new_input_specs = []
        for inp in self.inner.inputs:
            if inp is to_replace:
                new_input_specs.append(replacement.spec)
            else:
                new_input_specs.append(inp.spec)

        new_output_spec = self.inner.output.spec
        if self.inner.output is to_replace:
            new_output_spec = replacement.spec

        return self.inner.spec.replace_io(tuple(new_input_specs), new_output_spec)


@dataclass_abc.dataclass_abc(frozen=True)
class MoveLet(_OperandWrapper):
    """A Move operation composed with some subsequent Impl."""

    spec: specs.Spec
    source_idx: int
    destination: Tensor
    input_idx: Optional[int]
    prefetching: bool
    inner: Impl

    def __post_init__(self):
        assert (
            self.inner.spec.operands[(-1 if self.input_idx is None else self.input_idx)]
            == self.destination.spec
        )
        assert self.destination.layout != layouts.HEXAGON_TRANSPACKED
        # TODO: Assert that the inner Spec is expected..

    @property
    def is_store(self) -> bool:
        return self.source_idx == len(self.spec.inputs)

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        mem = {k: 0 for k in system_config.current_system().banks}
        additional = self.destination.spec.bytes_used
        if self.prefetching:
            additional *= 2
        mem[self.destination.bank] = additional
        return [mem]

    @property
    def peak_memory(self) -> dict[str, int]:
        mem = self.inner.peak_memory
        additional = self.destination.spec.bytes_used
        if self.prefetching:
            additional *= 2
        mem[self.destination.bank] += additional
        return mem

    def apply(self, operands: Sequence[TensorLike]) -> AppliedImpl:
        inner_operands = list(operands)
        if self.input_idx is None:
            inner_operands[-1] = self.destination
        else:
            inner_operands[self.input_idx] = self.destination
        return make_applied_impl(
            self.replace_children([self.inner.apply(inner_operands)]), operands
        )  # type: ignore


@dataclass_abc.dataclass_abc(frozen=True)
class PadTranspack(_OperandWrapper):
    """Impl that pads and transpacks an input tensor.

    Output transpacking not supported.
    """

    # TODO: With padding, Morello is perfectly capable of generating this
    #  without the call to the one-off transpack method. Add padding and remove
    #  this instruction.

    spec: specs.Spec
    source_idx: int
    destination: Tensor
    input_idx: int
    inner: Impl

    def __post_init__(self):
        source_spec = self.spec.operands[self.source_idx]
        if self.source is self.destination:
            raise ValueError("Source and destination cannot be the same tensor")
        if source_spec.dim_sizes != self.destination.dim_sizes:
            raise ValueError("Source and dest. must have matching shapes")
        if source_spec.bank != "GL":
            raise ValueError(f"Source must be in GL, but is in {source_spec.bank}")
        if self.destination.bank != "GL":
            raise ValueError(f"Dest. must be in GL, but is in {self.destination.bank}")
        if source_spec.layout != layouts.ROW_MAJOR:
            raise ValueError("Source must have a row-major layout")
        if self.destination.layout != layouts.HEXAGON_TRANSPACKED:
            raise ValueError("Destination must be HEXAGON_TRANSPACKED")

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        # TODO: Include memory used between pad and transpack.
        return [{k: 0 for k in system_config.current_system().banks}]

    @property
    def peak_memory(self) -> dict[str, int]:
        # TODO: Include memory used between pad and transpack.
        return self.inner.peak_memory


def _move_arguments(
    operand: specs.TensorSpec,
) -> Iterable[tuple[str, Layout, dict[str, Any]]]:
    target = system_config.current_target()

    # If the tensor has only one element, row-major is the only available
    # layout. Otherwise, all layouts are available.
    allowable_layouts = [layouts.ROW_MAJOR]
    if any(d > 1 for d in operand.dim_sizes):
        allowable_layouts = list(target.all_layouts)

    # TODO: Moves into HEXAGON_TRANSPACKED are handled by pad_transpack, not
    #   move_{input, output} at the moment.
    allowable_layouts = [
        l for l in allowable_layouts if not isinstance(l, layouts.HexagonTranspacked)
    ]

    # Yield actions for movement with register file destination, which
    # includes relayouts in registers and movements from level 1 to RF.
    for layout in allowable_layouts:
        for bank in target.system.faster_destination_banks(operand.bank):
            # TODO: Hacky check for VMEM.
            if bank == "VMEM":
                for vector_shape in gen_vector_shapes(operand.dim_sizes, operand.dtype):
                    yield bank, layout, {"vector_shape": vector_shape}
            else:
                yield bank, layout, {}


def common_operand_move_actions(impl: "Impl") -> Iterable[MoveAction]:
    spec = impl.spec

    def inner(inp_idx, operand: specs.TensorSpec) -> Iterable[MoveAction]:
        assert isinstance(operand, specs.TensorSpec)  # TODO: Remove
        move_fn, can_move_fn = impl.move_output, operand.can_move_to
        if inp_idx is not None:
            move_fn = functools.partial(impl.move_input, inp_idx)
            can_move_fn = operand.can_move_to

        for bank, layout, kws in _move_arguments(operand):
            for prf in [True, False]:
                if can_move_fn(bank, layout):
                    yield MoveAction(move_fn, operand, inp_idx, prf, bank, layout, kws)

    for i, inp in enumerate(spec.inputs):
        yield from inner(i, inp)
    yield from inner(None, spec.output)


# TODO: Use this everywhere sliding_tile_out actions are produced
@assert_stable_spec
def common_move(
    op: Impl,
    operand_idx: int,
    bank: Optional[str],
    layout: Optional[Layout],
    prefetching: bool,
    **kwargs,
) -> "MoveLet":
    """Wraps a dataclass-based Impl in a MoveLet moving one of its operands.

    This is the logic underpinning some ops' move_input actions.

    :param attr_name: The name of the field holding the operand to move.
    :param bank: The bank to which the operand should be moved, if not None.
    :param kwargs: Extra keyword arguments are forwarded the current target's
      `tensor_spec` method while constructing the destination tensor.
    """
    operand: specs.TensorSpec = op.spec.operands[operand_idx]
    if bank is None:
        bank = operand.spec.bank
    if layout is None:
        layout = operand.spec.layout
    if bank == operand.bank and layout == operand.layout:
        raise ValueError("Either bank or layout must differ from current")
    new_mat = current_target().tensor(
        spec=current_target().tensor_spec(
            dim_sizes=operand.dim_sizes,
            dtype=operand.dtype,
            layout=layout,
            bank=bank,
            **kwargs,
        ),
        name=None,
    )

    # Figure out the input index, if it's an input
    input_idx = operand_idx if operand_idx < len(op.spec.inputs) else None
    assert input_idx is not None or operand == op.spec.output

    return MoveLet(
        spec=op.spec,
        source_idx=operand_idx,
        destination=new_mat,
        input_idx=input_idx,
        prefetching=prefetching,
        inner=op.replace_spec(op.spec.replace_operand(operand_idx, new_mat.spec)),
    )
