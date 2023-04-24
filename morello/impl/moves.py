import dataclasses
import functools
import typing
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Union

from .. import layouts, specs, system_config
from ..layouts import Layout
from ..system_config import current_system, current_target
from ..tensor import (
    InnerContigFlatteningTile,
    OperandIdx,
    Tensor,
    TensorBase,
    TensorLike,
)
from ..utils import TinyMap
from . import MoveAction, settings, speccast
from .actions import TileOutAction
from .base import AppliedImpl, Impl, NonAllocatingLeaf, make_applied_impl
from .pruning import (
    ParentSummary,
    break_matmul_split_symmetries,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from .utils import assert_stable_spec, gen_tile_sizes, gen_vector_shapes


@dataclasses.dataclass(frozen=True, slots=True)
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
        assert not self.prefetching or settings.enable_prefetching_moves.get()

        system = system_config.current_system()
        source_bank = self.spec.operands[self.source_idx].bank
        if self.destination.bank not in system.faster_destination_banks(source_bank):
            raise ValueError(
                f"Cannot move from {source_bank} to {self.destination.bank}"
            )

        # The following check is needed because, as the moment, we don't nest any
        # Load or Store Impl under MoveLet when not addressed. An alternative would
        # be to allow this requirement to emerge from the lack of any Load/Store Impl
        # which is willing to implement this impossible behavior.
        if (
            self.destination.bank not in system.addressed_banks
            and self.destination.layout != self.spec.operands[self.source_idx].layout
        ):
            raise ValueError(
                f"Cannot change layout when {self.destination.bank} is not an "
                "addressed bank"
            )

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
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        filled_children = list(self._filled_children_names())
        replacements = list(replacements)
        if len(replacements) != len(filled_children):
            raise ValueError(
                f"{len(filled_children)} children expected; got " f"{len(replacements)}"
            )
        return dataclasses.replace(self, **dict(zip(filled_children, replacements)))

    def move(self, *args, **kwargs):
        return dataclasses.replace(self, body=self.body.move(*args, **kwargs))

    def to_accum(self) -> Impl:
        return dataclasses.replace(self, body=self.body.to_accum())

    @assert_stable_spec
    def complete(self) -> Impl:
        return self.replace_children((c.complete() for c in self.children))

    @property
    def memory_allocated(self) -> tuple[TinyMap[str, int], list[TinyMap[str, int]]]:
        additional = self.destination.spec.bytes_used
        if self.prefetching:
            additional *= 2

        banks = system_config.current_system().ordered_banks

        zeros = TinyMap(banks, (0,) * len(banks))
        dest_bank_idx = banks.index(self.destination.bank)
        mem = TinyMap(
            banks,
            tuple(additional if i == dest_bank_idx else 0 for i in range(len(banks))),
        )

        to_return = [mem]
        if self.prologue is not None:
            to_return.insert(0, zeros)
        if self.epilogue is not None:
            to_return.append(zeros)
        return zeros, to_return

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


class _BaseMoveHole(NonAllocatingLeaf):
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
        # Rank 3+ Zeros can be reinterpreted as 2 dimensional.
        if self.can_flatten:
            yield self.flatten
            return

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

    def flatten(self) -> "Impl":
        if not self.can_flatten:
            return self

        lhs, rhs = self.spec.operands
        cabs = lhs.layout.contiguous_lub(
            rhs.layout, lhs.contiguous_abs, rhs.contiguous_abs
        )

        # TODO: Avoid the need for these pass-through tiles.
        flattened_tiles = []
        for i, o in enumerate(self.spec.operands):
            inner = o.simple_tile(OperandIdx(i), o.dim_sizes)
            flattened_tiles.append(
                InnerContigFlatteningTile(
                    OperandIdx(i),
                    inner,
                    flattening_contiguous_abs=cabs,
                )
            )
        return speccast.SpecCast(
            spec=self.spec,
            inner=type(self)(
                type(self.spec)(
                    *[t.spec for t in flattened_tiles],
                    serial_only=self.spec.serial_only,
                )
            ),
            inner_args=frozenset(flattened_tiles),
        )

    @property
    def can_flatten(self) -> bool:
        return False

        lhs, rhs = self.spec.operands

        # TODO: Add support for vector banks
        if lhs.vector_shape or rhs.vector_shape:
            return False

        # TODO: Push this down into the layouts.
        if type(lhs.layout) != type(rhs.layout):
            return False
        elif type(lhs.layout) == layouts.StandardLayout:
            if lhs.layout != rhs.layout:
                return False

        cabs = lhs.layout.contiguous_lub(
            rhs.layout, lhs.contiguous_abs, rhs.contiguous_abs
        )

        flatteneds = [
            o.layout.flatten_inner_contiguous_dimensions(o.dim_sizes, cabs)
            for o in (lhs, rhs)
        ]
        new_shape = None
        for f, operand in zip(flatteneds, self.spec.operands):
            if f is None:
                return False
            prefix, _, volume = f
            if len(prefix) + 1 >= len(operand.dim_sizes):
                return False

            # We only support flattening in the result would have the same logical
            # shape. A example of where this would not happen is:
            # Load((1×7×1×1, u32, L1, c2, ua), (1×7×1×1, u32, RF, NHWC), serial)
            # which would (without this check) flatten to
            # Load((1×7×1, u32, L1, c1, ua), (1×1×7, u32, RF, c1), serial).
            if new_shape is None:
                new_shape = prefix + (volume,)
            elif new_shape != prefix + (volume,):
                return False

        return True

    def complete(self) -> Impl:
        if any(d > 1 for d in self.spec.source.dim_sizes):
            ones = (1,) * len(self.spec.source.dim_sizes)
            return self.tile_out(ones).complete()
        return self.place(ValueAssign)

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        return make_applied_impl(self, operands)


@dataclasses.dataclass(frozen=True, slots=True)
class LoadHole(_BaseMoveHole):
    spec: specs.Load


@dataclasses.dataclass(frozen=True, slots=True)
class StoreHole(_BaseMoveHole):
    spec: specs.Store


@dataclasses.dataclass(frozen=True, slots=True)
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
        if any(d != 1 for o in operands for d in o.dim_sizes):
            return "Non-value operand; had operands: " + ", ".join(map(str, operands))
        return None


@dataclasses.dataclass(frozen=True, slots=True)
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

        system = current_system()
        has_vrf = False
        for o in operands:
            if system.banks[o.bank].vector_rf:
                has_vrf = True
                if o.dim_sizes != o.vector_shape:
                    return "VectorAssign's vector operand was not a single register"
        if not has_vrf:
            return "Neither operand is in a vector register file"

        return None


@dataclasses.dataclass(frozen=True, slots=True)
class CacheAccess(NonAllocatingLeaf):  # "Allocation" happens in enclosing MoveLet.
    spec: specs.Spec

    def __post_init__(self):
        raise NotImplementedError("Current CPU model doesn't model caches")

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return False


class Moveable:
    """A mixin providing the most common `move` and `move_output` actions."""

    def move(
        self,
        operand_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        assert isinstance(self, Impl)
        return common_move(self, operand_idx, bank, layout, prefetching, **kwargs)

    @typing.final
    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        assert isinstance(self, Impl)
        return self.move(self.operand_count - 1, bank, layout, prefetching, **kwargs)


def _move_arguments(
    operand: specs.TensorSpec,
) -> Iterable[tuple[str, Layout, dict[str, Any]]]:
    target = system_config.current_target()

    allowable_layouts = list(target.all_layouts_for_shape(operand.dim_sizes))

    # Yield actions for movement with register file destination, which
    # includes relayouts in registers and movements from level 1 to RF.
    for layout in allowable_layouts:
        for bank in target.system.faster_destination_banks(operand.bank):
            vector_bytes: Optional[int] = target.system.banks[bank].vector_bytes
            if vector_bytes:
                for vector_shape in gen_vector_shapes(
                    operand.dim_sizes, operand.dtype, vector_bytes
                ):
                    yield bank, layout, {"vector_shape": vector_shape}
            else:
                yield bank, layout, {}


def common_operand_move_actions(impl: "Impl") -> Iterable[MoveAction]:
    spec = impl.spec

    if settings.enable_prefetching_moves.get():
        prf_options = [True, False]
    else:
        prf_options = [False]

    def inner(operand_idx, operand: specs.TensorSpec) -> Iterable[MoveAction]:
        move_fn = functools.partial(impl.move, operand_idx)
        for bank, layout, kws in _move_arguments(operand):
            for prf in prf_options:
                if operand.can_move_to(bank, layout):
                    yield MoveAction(
                        move_fn, operand, operand_idx, prf, bank, layout, kws
                    )

    for i, inp in enumerate(spec.operands):
        yield from inner(i, inp)


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

    This is the logic underpinning some ops' move actions.

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
        is_output = operand_idx == len(op.spec.operands) - 1

        if not is_output or op.spec.output_is_read():
            prologue = LoadHole(
                specs.Load(
                    source=op.spec.operands[operand_idx],
                    destination=new_mat.spec,
                    serial_only=op.spec.serial_only,
                )
            )

        if is_output:
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
    return layout.contiguous_full()
