import functools
from typing import Union, Tuple, Callable, Optional, Iterable

import dataclass_abc

from .actions import TileOutAction
from .base import Impl
from .loops import Loop
from .moves import common_operand_move_actions, common_move
from .pruning import (
    ParentSummary,
    prune_relayout_cycles,
    break_moves_symmetries,
    break_tile_out_symmetries,
)
from .settings import allow_reduce_splits
from .utils import gen_tile_sizes, assert_stable_spec, dim_range
from .. import specs, system_config
from ..specs import Layout
from ..tensor import Tensor, Tile


@dataclass_abc.dataclass_abc(frozen=True)
class ReduceSum(Impl):
    source: Union[Tensor, Tile]
    "The tensor to reduce."
    output: Union[Tensor, Tile]
    serial_only: bool

    @functools.cached_property
    def spec(self) -> specs.ReduceSum:
        return specs.ReduceSum(self.source.spec, self.output.spec, self.serial_only)

    @property
    def inputs(self):
        return (self.source,)

    @property
    def children(self) -> Tuple["Impl", ...]:
        return tuple()

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        return (
            f"ReduceSum({name_tensor_fn(self.source)}, {name_tensor_fn(self.output)})"
        )

    @property
    def is_scheduled(self) -> bool:
        # TODO: Drop these RF constants. Instead, use target-specific Impls.
        return (
            self.source.bank in ("RF", "HexagonRF")
            and self.output.bank in ("RF", "HexagonRF")
            and all(d == 1 for d in self.source.dim_sizes)
        )

    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        # TODO: Lots of code duplication in this method
        # Search only over full line sizes
        for ds in gen_tile_sizes(self.output.dim_sizes, filter=self._can_tile_out):
            if not self.output.spec.is_valid_tile_shape(ds):
                continue
            for parallel in [False] if self.serial_only else [True, False]:
                yield TileOutAction(self, ds, parallel)

        # Split over the reduction dimension
        if allow_reduce_splits.get():
            for k in dim_range(self.source.dim_sizes[-1], include_end=False):
                if k != self.source.dim_sizes[-1]:
                    if self._split_valid(k):
                        yield functools.partial(self.split, k)

        yield from common_operand_move_actions(self)

    def _split_valid(self, k: int) -> bool:
        orig_shape = self.source.dim_sizes
        if k > orig_shape[-1]:
            return False
        if not self.source.spec.is_valid_tile_shape(orig_shape[:-1] + (k,)):
            return False
        return True

    def move_input(
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "impl.MoveLet":
        if input_idx == 0:
            return common_move(self, "source", bank, layout, prefetching, **kwargs)
        else:
            raise ValueError("input_idx must be 0 ")

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> Impl:
        return common_move(self, "output", bank, layout, prefetching, **kwargs)

    @assert_stable_spec
    def split(self, k: int) -> Union["ReduceSum", "Loop"]:
        if k == self.source.dim_sizes[-1]:
            return self
        source_tile = self.source.simple_tile(self.source.dim_sizes[:-1] + (k,))
        return Loop(
            driving_tile=source_tile,
            dependent_tiles=frozenset(),
            inner=ReduceSum(
                source=source_tile, output=self.output, serial_only=self.serial_only
            ),
            parallel=False,
        )

    @assert_stable_spec
    def complete(self) -> Impl:
        if any(d > 1 for d in self.output.dim_sizes):
            return self.tile_out(tuple(1 for _ in self.output.dim_sizes)).complete()
        if self.source.dim_sizes[-1] > 1:
            return self.split(1).complete()

        system = system_config.current_system()
        next_general_source = system.next_general_bank(self.source.bank)
        if next_general_source:
            return self.move_input(0, bank=next_general_source).complete()
        next_general_out = system.next_general_bank(self.output.bank)
        if next_general_out:
            return self.move_output(bank=next_general_out).complete()

        return self

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        if replacements:
            raise Exception("Reduce has no children to replace")
        return self

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        inputs = list(inputs)
        if len(inputs) != 1:
            raise ValueError("Expected 1 input")
        return ReduceSum(inputs[0], output, serial_only=self.serial_only)

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return [{k: 0 for k in system_config.current_system().banks}]

    @property
    def peak_memory(self) -> dict[str, int]:
        return {k: 0 for k in system_config.current_system().banks}
