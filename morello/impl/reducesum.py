import dataclasses
import functools
from typing import Callable, Iterable, Optional, Sequence, Union

from .. import specs, system_config
from ..tensor import OperandIdx
from .actions import TileOutAction
from .base import Impl, NonAllocatingLeaf
from .block import Block
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
from .zero import ZeroHole


@dataclasses.dataclass(frozen=True, slots=True)
class ReduceSumHoleBase(NonAllocatingLeaf, Moveable):
    spec: specs.Spec

    @property
    def is_scheduled(self) -> bool:
        return False

    def replace_spec(self, new_spec: specs.Spec) -> "Impl":
        return type(self)(new_spec)

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

        yield from common_operand_move_actions(self)


class ReduceSumHole(ReduceSumHoleBase):
    def __post_init__(self):
        assert type(self.spec) == specs.ReduceSum  # TODO: Remove

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from super().actions(parent_summary)
        yield self.to_accum

    def to_accum(self) -> Impl:
        s = self.spec
        assert isinstance(
            s, specs.ReduceSum
        ), f"Spec was expected to be ReduceSum, but was {type(s).__name__}"
        zero_hole = ZeroHole(specs.Zero(s.output, serial_only=s.serial_only))
        accum_spec = specs.ReduceSumAccum(s.source, s.output, s.serial_only)
        accum_hole = ReduceSumAccumHole(accum_spec)

        out_idx = len(s.inputs)
        return Block(
            s, (zero_hole, accum_hole), ((out_idx,), tuple(range(len(s.operands))))
        )

    @assert_stable_spec
    def complete(self) -> Impl:
        return self.to_accum().complete()


class ReduceSumAccumHole(ReduceSumHoleBase):
    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from super().actions(parent_summary)

        # Split over the reduction dimension
        if allow_reduce_splits.get():
            for k in dim_range(self.spec.inputs[0].dim_sizes[-1], include_end=False):
                if k != self.spec.inputs[0].dim_sizes[-1]:
                    if self._split_valid(k):
                        yield functools.partial(self.split, k)

        if Add.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, Add)

    @assert_stable_spec
    def split(self, k: int) -> Union["ReduceSumAccumHole", "Loop"]:
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
            inner=ReduceSumAccumHole(
                specs.ReduceSumAccum(
                    source=source_tile.spec,
                    output=self.spec.output,
                    serial_only=self.spec.serial_only,
                )
            ),
            parallel=False,
        )

    def _split_valid(self, k: int) -> bool:
        orig_shape = self.spec.inputs[0].dim_sizes
        if k > orig_shape[-1]:
            return False
        if not self.spec.inputs[0].is_valid_tile_shape(orig_shape[:-1] + (k,)):
            return False
        return True

    @assert_stable_spec
    def complete(self) -> Impl:
        if any(d > 1 for d in self.spec.output.dim_sizes):
            return self.tile_out(
                tuple(1 for _ in self.spec.output.dim_sizes)
            ).complete()
        if self.spec.inputs[0].dim_sizes[-1] > 1:
            return self.split(1).complete()

        system = system_config.current_system()
        next_general_source = system.next_general_bank(self.spec.inputs[0].bank)
        if next_general_source:
            return self.move_input(0, bank=next_general_source).complete()
        next_general_out = system.next_general_bank(self.spec.output.bank)
        if next_general_out:
            return self.move_output(bank=next_general_out).complete()

        return self


@dataclasses.dataclass(frozen=True)
class Add(NonAllocatingLeaf, Moveable):
    """Implements `output += source;` in the target language."""

    spec: specs.Spec

    def __post_init__(self):
        assert self.applies_to_operands(self.spec.operands)

    @property
    def is_scheduled(self) -> bool:
        return True

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return all(
            o.bank == "RF" and all(d == 1 for d in o.dim_sizes) for o in operands
        )

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from []

    def complete(self, *args, **kwargs):
        return self
