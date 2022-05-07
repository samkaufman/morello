import dataclasses
import functools
from typing import Any, Callable, Iterable, Optional, Union

import dataclass_abc
import termcolor

from .. import specs, system_config
from ..specs import Layout
from ..system_config import current_target
from ..tensor import Tensor, TensorLike, Tile
from . import MoveAction
from .base import Impl
from .utils import assert_stable_spec, gen_vector_shapes


class _OperandWrapper(Impl):
    @property
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        new_inputs = []
        for inner_inp in self.inner.inputs:
            assert inner_inp is not self.source
            if inner_inp is self.destination:
                new_inputs.append(self.source)
            else:
                new_inputs.append(inner_inp)
        return tuple(new_inputs)

    @property
    def output(self):
        # A MoveLet can move any operand. This returns the source of the move if output
        # is the operand being moved; the inner output otherwise.
        if self.inner.output is self.destination:
            return self.source
        return self.inner.output

    @property
    def children(self) -> tuple[Impl, ...]:
        return (self.inner,)

    @property
    def is_scheduled(self) -> bool:
        return self.inner.is_scheduled

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        replacements = list(replacements)
        if len(replacements) != 1:
            raise ValueError(f"One replacement child expected; got {len(replacements)}")
        if replacements[0].spec != self.inner.spec:
            raise ValueError(f"Expected a replacement with spec {self.inner.spec}")
        return dataclasses.replace(self, inner=replacements[0])

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        raise NotImplementedError(f"Not implemented for {type(self).__name__}")

    @functools.cached_property
    def spec(self) -> specs.Spec:
        return self._spec_with_replaced_operand(self.destination, self.source)

    def move_input(self, *args, **kwargs) -> "Impl":
        # Pass move_input through to the inner schedule
        return dataclasses.replace(self, inner=self.inner.move_input(*args, **kwargs))

    def move_output(self, *args, **kwargs) -> "Impl":
        # Pass move_output through to the inner schedule
        return dataclasses.replace(self, inner=self.inner.move_output(*args, **kwargs))

    def pad_transpack(self, *args, **kwargs) -> "Impl":
        # Pass pad_transpack through to the inner schedule
        return dataclasses.replace(
            self, inner=self.inner.pad_transpack(*args, **kwargs)
        )

    @assert_stable_spec
    def split(self, size: int) -> "Impl":
        return dataclasses.replace(self, inner=self.inner.split(size))

    @assert_stable_spec
    def complete(self) -> Impl:
        return dataclasses.replace(self, inner=self.inner.complete())

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

    source: Union[Tensor, Tile]
    destination: Tensor
    input_idx: Optional[int]
    prefetching: bool
    inner: Impl

    def __post_init__(self):
        assert self.destination.origin is self.source, (
            f"Destination's origin {self.destination.origin} was not source"
            f" {self.source}"
        )
        assert (
            self.operands[(-1 if self.input_idx is None else self.input_idx)]
            is self.source
        )
        assert (
            self.inner.operands[(-1 if self.input_idx is None else self.input_idx)]
            is self.destination
        )
        assert self.destination.layout != specs.HEXAGON_TRANSPACKED

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        keyword = "move"
        if self.prefetching:
            keyword += "*"
        if self.prefetching:
            keyword += "[p]"

        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])
        return (
            f"{keyword}[{self.destination.bank}]"
            f" {name_tensor_fn(self.destination)}"
            f" {arrow} {name_tensor_fn(self.source)}"
        )

    def store_env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        fancy: bool = False,
    ):
        keyword = "store"
        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])
        return (
            f"{keyword} {name_tensor_fn(self.source)}"
            f" {arrow} {name_tensor_fn(self.destination)}"
        )

    @property
    def is_store(self) -> bool:
        return self.source == self.output

    @property
    def lhs(self):
        # A MoveLet can move any operand. This returns the source of the move if output
        # is the operand being moved; the inner output otherwise.
        inner_lhs, _ = self.inner.inputs
        if inner_lhs is self.destination:
            return self.source
        return inner_lhs

    @property
    def rhs(self):
        # A MoveLet can move any operand. This returns the source of the move if output
        # is the operand being moved; the inner output otherwise.
        _, inner_rhs = self.inner.inputs
        if inner_rhs is self.destination:
            return self.source
        return inner_rhs

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        mem = {k: 0 for k in system_config.current_system().banks}
        additional = self.destination.bytes_used
        if self.prefetching:
            additional *= 2
        mem[self.destination.bank] = additional
        return [mem]

    @property
    def peak_memory(self) -> dict[str, int]:
        mem = self.inner.peak_memory
        additional = self.destination.bytes_used
        if self.prefetching:
            additional *= 2
        mem[self.destination.bank] += additional
        return mem

    def __hash__(self):
        # A slightly faster hash
        return hash((self.source, self.destination))


@dataclass_abc.dataclass_abc(frozen=True)
class PadTranspack(_OperandWrapper):
    """Impl that pads and transpacks an input tensor.

    Output transpacking not supported.
    """

    # TODO: With padding, Morello is perfectly capable of generating this
    #  without the call to the one-off transpack method. Add padding and remove
    #  this instruction.

    source: TensorLike
    destination: Tensor
    input_idx: int
    inner: Impl

    def __post_init__(self):
        if self.source is self.destination:
            raise ValueError("Source and destination cannot be the same tensor")
        if self.source.dim_sizes != self.destination.dim_sizes:
            raise ValueError("Source and dest. must have matching shapes")
        if self.source.bank != "GL":
            raise ValueError(f"Source must be in GL, but is in {self.source.bank}")
        if self.destination.bank != "GL":
            raise ValueError(f"Dest. must be in GL, but is in {self.destination.bank}")
        if self.source.layout != specs.ROW_MAJOR:
            raise ValueError("Source must have a row-major layout")
        if self.destination.layout != specs.HEXAGON_TRANSPACKED:
            raise ValueError("Destination must be HEXAGON_TRANSPACKED")

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        return (
            f"{type(self).__name__}({name_tensor_fn(self.destination)} <- "
            f"{name_tensor_fn(self.source)})"
        )

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        # TODO: Include memory used between pad and transpack.
        return [{k: 0 for k in system_config.current_system().banks}]

    @property
    def peak_memory(self) -> dict[str, int]:
        # TODO: Include memory used between pad and transpack.
        return self.inner.peak_memory


def _move_arguments(
    operand: Union[Tile, Tensor]
) -> Iterable[tuple[str, Layout, dict[str, Any]]]:
    target = system_config.current_target()

    # If the tensor has only one element, row-major is the only available
    # layout. Otherwise, all layouts are available.
    allowable_layouts = [specs.ROW_MAJOR]
    if any(d > 1 for d in operand.dim_sizes):
        allowable_layouts = list(target.all_layouts)

    # TODO: Moves into HEXAGON_TRANSPACKED are handled by pad_transpack, not
    #   move_{input, output} at the moment.
    allowable_layouts = [
        l for l in allowable_layouts if not isinstance(l, specs.HexagonTranspacked)
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
    def inner(inp_idx, operand):
        move_fn, can_move_fn = impl.move_output, impl.output.can_move_to
        if inp_idx is not None:
            move_fn = functools.partial(impl.move_input, inp_idx)
            can_move_fn = impl.inputs[inp_idx].can_move_to

        for bank, layout, kws in _move_arguments(operand):
            for prf in [True, False]:
                if can_move_fn(bank, layout):
                    yield MoveAction(move_fn, operand, inp_idx, prf, bank, layout, kws)

    for i, inp in enumerate(impl.inputs):
        yield from inner(i, inp)
    yield from inner(None, impl.output)


# TODO: Use this everywhere sliding_tile_out actions are produced
@assert_stable_spec
def common_move(
    op,
    attr_name: str,
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
    operand: Union[Tensor, Tile] = getattr(op, attr_name)
    if bank is None:
        bank = operand.spec.bank
    if layout is None:
        layout = operand.spec.layout
    if bank == operand.root.bank and layout == operand.layout:
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
        origin=operand,
    )

    # Figure out the input index, if it's an input
    # TODO: Faster to have the caller pass the index.
    input_idx = None
    try:
        input_idx = op.inputs.index(operand)
    except ValueError:
        pass
    assert input_idx is not None or operand == op.output

    return MoveLet(
        source=operand,
        destination=new_mat,
        input_idx=input_idx,
        prefetching=prefetching,
        inner=dataclasses.replace(op, **{attr_name: new_mat}),
    )
