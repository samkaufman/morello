import dataclasses
import functools
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

from .. import specs, system_config, tensor
from ..impl import actions, base, speccast
from ..impl.utils import gen_tile_sizes
from ..layouts import Layout
from .moves import (
    MoveLet,
    StoreHole,
    common_operand_move_actions,
    transition_contiguous,
)
from .pruning import (
    ParentSummary,
    break_matmul_split_symmetries,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)

if TYPE_CHECKING:
    from .. import impl


@dataclasses.dataclass(frozen=True, slots=True)
class ZeroHole(base.NonAllocatingLeaf):
    spec: specs.Zero

    @property
    def is_scheduled(self) -> bool:
        return False

    def replace_spec(self, new_spec: specs.Spec) -> "impl.Impl":
        if not isinstance(new_spec, specs.Zero):
            raise TypeError(f"Spec had unexpected type: {type(new_spec).__name__}")
        return type(self)(new_spec)  # type: ignore

    def move(
        self,
        operand_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        if operand_idx != 0:
            raise ValueError("operand_idx must be 0")

        target = system_config.current_target()
        output = self.spec.output

        if bank is None:
            bank = output.bank
        if layout is None:
            layout = output.layout

        # When moving into an addressed bank, we'll generate an aligned destination.
        # If it's into a cache level, alignment won't change.
        aligned = True
        if bank not in target.system.addressed_banks:
            aligned = output.aligned

        faster = target.tensor(
            spec=target.tensor_spec(
                dim_sizes=output.dim_sizes,
                dtype=output.dtype,
                contiguous_abs=transition_contiguous(bank, layout, output),
                aligned=aligned,
                layout=layout,
                bank=bank,
                **kwargs,
            ),
            name=None,
        )
        epilogue = StoreHole(
            specs.Store(
                source=output,
                destination=faster.spec,
                serial_only=self.spec.serial_only,
            )
        )
        return MoveLet(
            spec=self.spec,
            source_idx=0,
            destination=faster,
            prefetching=prefetching,
            prologue=None,
            body=self.replace_spec(self.spec.replace_io(self.spec.inputs, faster.spec)),
            epilogue=epilogue,
        )

    def flatten(self) -> "impl.Impl":
        if not self.can_flatten:
            return self

        # TODO: Avoid the need for this pass-through tile.
        inner = self.spec.destination.simple_tile(
            tensor.OperandIdx(0), self.spec.destination.dim_sizes
        )
        new_destination = tensor.InnerContigFlatteningTile(
            tensor.OperandIdx(0),
            inner,
            flattening_contiguous_abs=inner.spec.contiguous_abs,
        )
        return speccast.SpecCast(
            spec=self.spec,
            inner=ZeroHole(
                specs.Zero(new_destination.spec, serial_only=self.spec.serial_only)
            ),
            inner_args=frozenset([new_destination]),
        )

    @property
    def can_flatten(self) -> bool:
        return False

        destination: specs.TensorSpec = self.spec.destination

        # TODO: Add support for vector banks
        if destination.vector_shape:
            return False

        flattened = destination.layout.flatten_inner_contiguous_dimensions(
            destination.dim_sizes, destination.contiguous_abs
        )
        if flattened is None:
            return False
        prefix, _, _ = flattened
        if len(prefix) + 1 >= len(destination.dim_sizes):
            return False
        return True

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], "impl.Impl"]]:
        # Rank 3+ Zeros can be reinterpreted as 2 dimensional.
        if self.can_flatten:
            yield self.flatten
            return

        # Search only over full line sizes
        t = self.spec.destination
        for tile_shape in gen_tile_sizes(t.dim_sizes, filter=self._can_tile_out):
            yield actions.TileOutAction(self, tile_shape, parallel=False)
            if not self.spec.serial_only:
                yield actions.TileOutAction(self, tile_shape, parallel=True)

        yield from common_operand_move_actions(self)

        if MemsetZero.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, MemsetZero)

        if VectorZero.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, VectorZero)

    def complete(self) -> "impl.Impl":
        system = system_config.current_system()
        one = tuple(1 for _ in self.spec.output.dim_sizes)
        if self.spec.output.dim_sizes != one:
            return self.tile_out(one).complete()
        next_general = system.next_general_bank(self.spec.output.bank)
        if next_general:
            return self.move_output(bank=next_general).complete()
        return self.place(MemsetZero)


@dataclasses.dataclass(frozen=True, slots=True)
class MemsetZero(base.NonAllocatingLeaf):
    spec: specs.Zero

    def __post_init__(self):
        check_result = MemsetZero._check_operands(self.spec.operands)
        if check_result:
            raise ValueError(check_result)

    @property
    def is_scheduled(self) -> bool:
        return True

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], "impl.Impl"]]:
        yield from []

    def complete(self, *args, **kwargs):
        return self

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return MemsetZero._check_operands(operands) is None

    @staticmethod
    def _check_operands(operands: Sequence[specs.TensorSpec]) -> Optional[str]:
        if not operands[0].contiguous:
            return f"TensorSpec {operands[0]} must be contiguous"
        if operands[0].bank != "RF":
            return f"TensorSpec {operands[0]} must be in RF"
        return None


@dataclasses.dataclass(frozen=True, slots=True)
class VectorZero(base.NonAllocatingLeaf):
    spec: specs.Zero

    def __post_init__(self):
        check_result = VectorZero._check_operands(self.spec.operands)
        if check_result:
            raise ValueError(check_result)

    @property
    def is_scheduled(self) -> bool:
        return True

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], "impl.Impl"]]:
        yield from []

    def complete(self, *args, **kwargs):
        return self

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return VectorZero._check_operands(operands) is None

    @staticmethod
    def _check_operands(operands: Sequence[specs.TensorSpec]) -> Optional[str]:
        if not operands[0].contiguous:
            return f"TensorSpec {operands[0]} must be contiguous"
        if operands[0].bank != "VRF":
            return f"TensorSpec {operands[0]} must be in VRF"
        if operands[0].dim_sizes != operands[0].vector_shape:
            return "VectorZero applies only to individual vector registers"
        return None
