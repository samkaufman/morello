import dataclasses
import operator
import functools
from typing import Any, Iterable, Literal, Optional, Callable, Sequence, Union, cast

from .. import layouts, specs, system_config
from .pruning import (
    ParentSummary,
    break_matmul_split_symmetries,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from ..layouts import Layout
from .actions import TileOutAction
from ..system_config import current_system, current_target
from ..tensor import TensorBase, Tensor, TensorLike
from ..utils import TinyMap
from . import MoveAction
from .base import AppliedImpl, Impl, Leaf, NonAllocatingLeaf, make_applied_impl
from .utils import assert_stable_spec, gen_vector_shapes, gen_tile_sizes


class _OperandWrapper(Impl):
    @property
    def children(self) -> tuple[Impl, ...]:
        return (self.inner,)  # type: ignore

    @property
    def is_scheduled(self) -> bool:
        return all(c.is_scheduled for c in self.children)

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
        return self.replace_children((self.children[0].split(size),))

    def spatial_split(self, *args, **kwargs) -> "Impl":
        return self.replace_children((self.children[0].spatial_split(*args, **kwargs),))

    @assert_stable_spec
    def complete(self) -> Impl:
        return self.replace_children((self.children[0].complete(),))


@dataclasses.dataclass(frozen=True)
class MoveLet(Impl):
    """A Move operation composed with some subsequent Impl.

    This Impl corresponds roughly to the following pseudocode:

    ```
    (..., source, ...) => {
        let n = prologue(source);
        inner(..., n, ...);
        epilogue(source, n);
    })
    ```

    Non-`source` operands are passed through to inner, and `n` is passed to
    `inner` at `source_idx`.

    For loads, prologue is generally the load and the epilogue is a no-op
    (`None` here). For stores, the prologue is a load or allocation and the
    epilogue writes data back to the source.
    """

    spec: specs.Spec
    source_idx: int
    destination: TensorBase
    prefetching: bool
    prologue: Optional[Impl]
    body: Impl
    epilogue: Optional[Impl]

    def __post_init__(self):
        assert self.source_idx >= 0
        assert self.destination.layout != layouts.HEXAGON_TRANSPACKED

    @property
    def is_store(self) -> bool:
        # TODO: This is a heuristic. Shouldn't be needed.
        return self.epilogue is not None

    @property
    def children(self) -> tuple[Impl, ...]:
        return tuple(getattr(self, name) for name in self._filled_children_names())

    @property
    def is_scheduled(self) -> bool:
        return all(c.is_scheduled for c in self.children)

    @assert_stable_spec
    def subschedule(self, idx: int, fn: Callable[[Impl], Impl]) -> "Pipeline":
        new_children = list(self.children)
        new_children[idx] = fn(new_children[idx])
        return self.replace_children(new_children)

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        filled_children = list(self._filled_children_names())
        replacements = list(replacements)
        if len(replacements) != len(filled_children):
            raise ValueError(
                f"{len(filled_children)} children expected; got " f"{len(replacements)}"
            )
        return dataclasses.replace(self, **dict(zip(filled_children, replacements)))

    def move_input(self, *args, **kwargs):
        return dataclasses.replace(self, body=self.body.move_input(*args, **kwargs))

    def move_output(self, *args, **kwargs):
        return dataclasses.replace(self, body=self.body.move_output(*args, **kwargs))

    @assert_stable_spec
    def complete(self) -> Impl:
        return self.replace_children((c.complete() for c in self.children))

    @property
    def additional_memories(self) -> list[TinyMap[str, int]]:
        additional = self.destination.spec.bytes_used
        if self.prefetching:
            additional *= 2
        
        banks = system_config.current_system().ordered_banks

        zeros = TinyMap(banks, (0,)* len(banks))
        dest_bank_idx = banks.index(self.destination.bank)
        mem = TinyMap(banks, tuple(additional if i == dest_bank_idx else 0 for i in range(len(banks))))

        to_return = [mem]
        if self.prologue is not None:
            to_return.insert(0, zeros)
        if self.epilogue is not None:
            to_return.append(zeros)
        return to_return

    @property
    def peak_memory(self) -> TinyMap[str, int]:
        mem = self.body.peak_memory
        additional = self.destination.spec.bytes_used
        if self.prefetching:
            additional *= 2
        dest_idx = mem.raw_keys.index(self.destination.bank)
        return TinyMap(
            mem.raw_keys,
            tuple(v + additional if dest_idx == i else v for i, v in enumerate(mem.raw_values))
        )

    def apply(self, operands: Sequence[TensorLike]) -> AppliedImpl:
        move_op_operands = [operands[self.source_idx], self.destination]
        body_operands = list(operands)
        body_operands[self.source_idx] = self.destination
        applied_children = [self.body.apply(body_operands)]
        if self.prologue is not None:
            applied_children.insert(0, self.prologue.apply(move_op_operands))
        if self.epilogue is not None:
            applied_children.append(self.epilogue.apply(move_op_operands))
        return make_applied_impl(
            self.replace_children(applied_children), operands
        )  # type: ignore

    def _filled_children_names(
        self,
    ) -> Iterable[Literal["prologue", "body", "epilogue"]]:
        if self.prologue:
            yield "prologue"
        yield "body"
        if self.epilogue:
            yield "epilogue"


class _BaseMoveHole(Leaf):
    spec: Union[specs.Load, specs.Store]

    @property
    def is_scheduled(self) -> bool:
        return False

    def replace_spec(self, new_spec: specs.Spec) -> "Impl":
        if not isinstance(new_spec, (specs.Load, specs.Store)):
            raise TypeError(f"Spec had unexpected type: {type(new_spec).__name__}")
        return type(self)(new_spec)  # type: ignore

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        # Search only over full line sizes
        source = self.spec.source
        for tile_shape in gen_tile_sizes(source.dim_sizes, filter=self._can_tile_out):
            yield TileOutAction(self, tile_shape, parallel=False)
            if not self.spec.serial_only:
                yield TileOutAction(self, tile_shape, parallel=True)

        if ValueAssign.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, ValueAssign)

        if VectorAssign.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, VectorAssign)

        if CacheAccess.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, CacheAccess)

    def complete(self) -> Impl:
        if any(d > 1 for d in self.spec.source.dim_sizes):
            ones = (1,) * len(self.spec.source.dim_sizes)
            return self.tile_out(ones).complete()
        return self.place(ValueAssign)

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        return make_applied_impl(self, operands)

    @property
    def additional_memories(self) -> list[TinyMap[str, int]]:
        banks = system_config.current_system().ordered_banks
        return [TinyMap(banks, (0,) * len(banks))]

    @property
    def peak_memory(self) -> TinyMap[str, int]:
        banks = system_config.current_system().ordered_banks
        return TinyMap(banks, (0,) * len(banks))


@dataclasses.dataclass(frozen=True)
class LoadHole(_BaseMoveHole):
    spec: specs.Load


@dataclasses.dataclass(frozen=True)
class StoreHole(_BaseMoveHole):
    spec: specs.Store


@dataclasses.dataclass(frozen=True)
class ValueAssign(NonAllocatingLeaf):  # "Allocation" happens in enclosing MoveLet.
    spec: Union[specs.Load, specs.Store]

    def __post_init__(self):
        check_result = self._check_operands(self.spec.operands)
        if check_result:
            raise ValueError(check_result)

    @property
    def is_store(self) -> bool:
        return isinstance(self.spec, specs.Store)

    @property
    def is_scheduled(self) -> bool:
        return True

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from []

    def complete(self, *args, **kwargs):
        return self

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return ValueAssign._check_operands(operands) is None

    @staticmethod
    def _check_operands(operands: Sequence[specs.TensorSpec]) -> Optional[str]:
        source, destination = operands
        if source.bank == destination.bank:
            return "Source and destination are in the same bank"
        if source.bank not in ("RF", "GL"):
            return f"Source is in {source.bank} bank"
        if destination.bank not in ("RF", "GL"):
            return f"Destination is in {destination.bank} bank"
        if any(d != 1 for o in operands for d in o.dim_sizes):
            return "Non-value operand; had operands: " + ", ".join(map(str, operands))
        return None


@dataclasses.dataclass(frozen=True)
class VectorAssign(NonAllocatingLeaf):
    spec: Union[specs.Load, specs.Store]

    def __post_init__(self):
        check_result = VectorAssign._check_operands(self.spec.operands)
        if check_result:
            raise ValueError(check_result)

    @property
    def is_store(self) -> bool:
        return isinstance(self.spec, specs.Store)

    @property
    def is_scheduled(self) -> bool:
        return True

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from []

    def complete(self, *args, **kwargs):
        return self

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return VectorAssign._check_operands(operands) is None

    @staticmethod
    def _check_operands(operands: Sequence[specs.TensorSpec]) -> Optional[str]:
        lhs, rhs = operands
        if not (lhs.contiguous and rhs.contiguous):
            return "Operands must be contiguous, but were: " + str((lhs, rhs))
        if lhs.dtype != rhs.dtype:
            return "Operand value types must match, but were: " + str((lhs, rhs))
        if lhs.dim_sizes != rhs.dim_sizes:
            return "Operand shapes must match, but were: " + str((lhs, rhs))
        if lhs.layout != rhs.layout:
            return "Layouts must match, but were: " + str((lhs, rhs))

        # Check that we're moving an AVX2 vector-sized tensor.
        vol_bytes = functools.reduce(operator.mul, lhs.dim_sizes, 1) * lhs.dtype.size
        if vol_bytes != 32:
            return f"Expected operands to be 32 bytes, but were {vol_bytes} bytes"

        return None


@dataclasses.dataclass(frozen=True)
class CacheAccess(NonAllocatingLeaf):  # "Allocation" happens in enclosing MoveLet.
    spec: specs.Spec

    def __post_init__(self):
        raise NotImplementedError("Current CPU model doesn't model caches")

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
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
        if not source_spec.layout.is_row_major:
            raise ValueError("Source must have a row-major layout")
        if self.destination.layout != layouts.HEXAGON_TRANSPACKED:
            raise ValueError("Destination must be HEXAGON_TRANSPACKED")

    @property
    def additional_memories(self) -> list[TinyMap[str, int]]:
        # TODO: Include memory used between pad and transpack.
        banks = system_config.current_system().ordered_banks
        return [TinyMap(banks, (0,) * len(banks))]

    @property
    def peak_memory(self) -> TinyMap[str, int]:
        # TODO: Include memory used between pad and transpack.
        return self.inner.peak_memory


class Moveable:
    """A mixin providing the most common `move_input` and `move_output` actions."""

    def move_input(
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        assert isinstance(self, Impl)
        if input_idx >= self.spec.inputs_count():
            raise ValueError(f"Input index {input_idx} out of range")
        return common_move(self, input_idx, bank, layout, prefetching, **kwargs)

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        assert isinstance(self, Impl)
        return common_move(self, -1, bank, layout, prefetching, **kwargs)


def _move_arguments(
    operand: specs.TensorSpec,
) -> Iterable[tuple[str, Layout, dict[str, Any]]]:
    target = system_config.current_target()

    # If the tensor has only one element, row-major is the only available
    # layout. Otherwise, all layouts are available.
    allowable_layouts = [layouts.row_major(len(operand.dim_sizes))]
    if any(d > 1 for d in operand.dim_sizes):
        allowable_layouts = [
            l
            for l in target.all_layouts_for_shape(operand.dim_sizes)
            if l.applies_to_shape(operand.dim_sizes, operand.dtype)
        ]

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
    if operand_idx == -1:
        operand_idx = len(op.spec.inputs)
    elif operand_idx < -1:
        raise ValueError("operand_idx must be in [-1, number of operands)")

    operand: specs.TensorSpec = op.spec.operands[operand_idx]
    if bank is None:
        bank = operand.bank
    if layout is None:
        layout = operand.layout
    assert layout is not None
    if bank == operand.bank and layout == operand.layout:
        raise ValueError("Either bank or layout must differ from current")

    # When moving into an addressed bank, we'll generate an aligned destination.
    # If it's into a cache level, alignment won't change.
    aligned = True
    if bank not in current_system().addressed_banks:
        aligned = operand.aligned

    new_mat = current_target().tensor(
        spec=current_target().tensor_spec(
            dim_sizes=operand.dim_sizes,
            dtype=operand.dtype,
            contiguous_abs=transition_contiguous(bank, layout, operand),
            aligned=aligned,
            layout=layout,
            bank=bank,
            **kwargs,
        ),
        name=None,
    )

    new_operands = list(op.spec.operands)
    new_operands[operand_idx] = new_mat.spec
    new_inner_spec = op.spec.replace_io(
        tuple(new_operands[:-1]), new_operands[-1], op.spec.serial_only
    )

    prologue, epilogue = None, None
    if bank in current_system().addressed_banks:
        prologue = LoadHole(
            specs.Load(
                source=op.spec.operands[operand_idx],
                destination=new_mat.spec,
                serial_only=op.spec.serial_only,
            )
        )

        # Add an epilogue if this is an output
        if operand_idx == len(op.spec.operands) - 1:
            epilogue = StoreHole(
                specs.Store(
                    # TODO: Source and destination are confusing here. Reversed.
                    source=op.spec.operands[operand_idx],
                    destination=new_mat.spec,
                    serial_only=op.spec.serial_only,
                )
            )

    return MoveLet(
        spec=op.spec,
        source_idx=operand_idx,
        destination=new_mat,
        prefetching=prefetching,
        prologue=prologue,
        body=op.replace_spec(new_inner_spec),
        epilogue=epilogue,
    )


def transition_contiguous(bank, layout, operand):
    # Will the result be contiguous? If the move is into a cache, it might be.
    # If it's into memory bank with its own address space, then yes.
    if bank not in current_system().addressed_banks:
        return operand.contiguous_abs
    return cast(Layout, layout).check_tile_contiguity(
        operand.dim_sizes, operand.dim_sizes, operand.contiguous_abs
    )
