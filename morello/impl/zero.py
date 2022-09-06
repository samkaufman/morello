import dataclasses
import functools
from typing import TYPE_CHECKING, Iterable, Callable, Sequence, Optional

from .. import specs
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

        if MemsetZero.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, MemsetZero)

    def complete(self) -> "impl.Impl":
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
            return "Tensor-to-zero must be contiguous"
        if operands[0].bank != "RF":
            return "Tensor-to-zero must be in RF"
        return None
