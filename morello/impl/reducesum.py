import functools
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

from .. import specs, system_config
from ..layouts import Layout
from ..tensor import OperandIdx
from .actions import TileOutAction
from .base import Impl, NonAllocatingLeaf
from .loops import Loop
from .moves import Moveable, common_operand_move_actions
from .pruning import (
    ParentSummary,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from .settings import allow_reduce_splits
from .utils import assert_stable_spec, dim_range, gen_tile_sizes

if TYPE_CHECKING:
    from .moves import MoveLet


class ReduceSum(NonAllocatingLeaf, Moveable):
    def __init__(self, spec: specs.ReduceSum):
        super().__init__()
        self._spec = spec

    @property
    def spec(self) -> specs.Spec:
        return self._spec

    def __eq__(self, other):
        if type(other) == ReduceSum:
            return NotImplemented
        return self.spec == other.spec

    def __hash__(self) -> int:
        return hash(self.spec)

    @property
    def is_scheduled(self) -> bool:
        # TODO: Drop these hard-coded bank literals. Instead, use target-specific Impls.
        return all(o.bank in ("RF", "HexagonRF") for o in self.spec.operands) and all(
            d == 1 for d in self.spec.inputs[0].dim_sizes
        )

    def replace_spec(self, new_spec: specs.Spec) -> "Impl":
        assert type(self) is ReduceSum
        if not isinstance(new_spec, specs.ReduceSum):
            raise ValueError(f"Expected Spec to be ReduceSum; was: {type(new_spec)}")
        return ReduceSum(new_spec)

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        # TODO: Lots of code duplication in this method
        # Search only over full line sizes
        for ds in gen_tile_sizes(self.spec.output.dim_sizes, filter=self._can_tile_out):
            if not self.spec.output.is_valid_tile_shape(ds):
                continue
            for parallel in [False] if self.spec.serial_only else [True, False]:
                yield TileOutAction(self, ds, parallel)

        # Split over the reduction dimension
        if allow_reduce_splits.get():
            for k in dim_range(self.spec.inputs[0].dim_sizes[-1], include_end=False):
                if k != self.spec.inputs[0].dim_sizes[-1]:
                    if self._split_valid(k):
                        yield functools.partial(self.split, k)

        yield from common_operand_move_actions(self)

    def _split_valid(self, k: int) -> bool:
        orig_shape = self.spec.inputs[0].dim_sizes
        if k > orig_shape[-1]:
            return False
        if not self.spec.inputs[0].is_valid_tile_shape(orig_shape[:-1] + (k,)):
            return False
        return True

    @assert_stable_spec
    def split(self, k: int) -> Union["ReduceSum", "Loop"]:
        if k == self.spec.inputs[0].dim_sizes[-1]:
            return self
        source_tile = self.spec.inputs[0].simple_tile(
            OperandIdx(0), self.spec.inputs[0].dim_sizes[:-1] + (k,)
        )
        driving_subscript = self.spec.operands_dim_subscripts()[0][-1]
        return Loop(
            spec=self.spec,
            subscripts=(driving_subscript,),
            tiles=frozenset([source_tile]),
            inner=ReduceSum(
                specs.ReduceSum(
                    source=source_tile.spec,
                    output=self.spec.output,
                    serial_only=self.spec.serial_only,
                )
            ),
            parallel=False,
        )

    @assert_stable_spec
    def complete(self) -> Impl:
        if any(d > 1 for d in self.output.dim_sizes):
            return self.tile_out(tuple(1 for _ in self.output.dim_sizes)).complete()
        if self.spec.inputs[0].dim_sizes[-1] > 1:
            return self.split(1).complete()

        system = system_config.current_system()
        next_general_source = system.next_general_bank(self.spec.inputs[0].bank)
        if next_general_source:
            return self.move_input(0, bank=next_general_source).complete()
        next_general_out = system.next_general_bank(self.output.bank)
        if next_general_out:
            return self.move_output(bank=next_general_out).complete()

        return self
