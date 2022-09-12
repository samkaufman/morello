import dataclasses
import functools
from typing import TYPE_CHECKING, Iterable, Callable, Sequence, Optional

from .moves import (
    transition_contiguous,
    MoveLet,
    StoreHole,
    common_operand_move_actions,
)
from .. import specs, system_config
from ..impl import actions, base
from ..impl.utils import gen_tile_sizes
from .pruning import (
    ParentSummary,
    break_matmul_split_symmetries,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from ..layouts import Layout

if TYPE_CHECKING:
    from .. import impl


@dataclasses.dataclass(frozen=True)
class ZeroHole(base.NonAllocatingLeaf):
    spec: specs.Zero

    @property
    def is_scheduled(self) -> bool:
        return False

    def replace_spec(self, new_spec: specs.Spec) -> "impl.Impl":
        if not isinstance(new_spec, specs.Zero):
            raise TypeError(f"Spec had unexpected type: {type(new_spec).__name__}")
        return type(self)(new_spec)  # type: ignore

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
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

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], "impl.Impl"]]:
        # Search only over full line sizes
        t = self.spec.destination
        for tile_shape in gen_tile_sizes(t.dim_sizes, filter=self._can_tile_out):
            yield actions.TileOutAction(self, tile_shape, parallel=False)
            if not self.spec.serial_only:
                yield actions.TileOutAction(self, tile_shape, parallel=True)

        yield from common_operand_move_actions(self)

        if MemsetZero.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, MemsetZero)

    def complete(self) -> "impl.Impl":
        system = system_config.current_system()
        one = tuple(1 for _ in self.spec.output.dim_sizes)
        if self.spec.output.dim_sizes != one:
            return self.tile_out(one).complete()
        next_general = system.next_general_bank(self.spec.output.bank)
        if next_general:
            return self.move_output(bank=next_general).complete()
        return self.place(MemsetZero)


@dataclasses.dataclass(frozen=True)
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
