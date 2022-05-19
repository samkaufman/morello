import functools
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

from .. import specs, system_config
from ..layouts import Layout
from ..tensor import Tensor, Tile
from .actions import TileOutAction
from .base import Impl, NonAllocatingLeaf
from .loops import Loop
from .moves import common_move, common_operand_move_actions
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


class ReduceSum(NonAllocatingLeaf):
    _source: Union[Tensor, Tile]
    _output: Union[Tensor, Tile]
    _serial_only: bool

    def __init__(
        self,
        source: Union[Tensor, Tile],
        output: Union[Tensor, Tile],
        serial_only: bool,
    ):
        super().__init__()
        self._source = source
        self._output = output
        self._serial_only = serial_only

    def __eq__(self, other):
        if type(other) == ReduceSum:
            return NotImplemented
        return (
            self.source == other.source
            and self.output == other.output
            and self.serial_only == other.serial_only
        )

    def __hash__(self) -> int:
        return hash((self.source, self.output, self.serial_only))

    @functools.cached_property
    def spec(self) -> specs.ReduceSum:
        return specs.ReduceSum(self.source.spec, self.output.spec, self.serial_only)

    @property
    def inputs(self):
        return (self.source,)

    @property
    def source(self) -> Union[Tensor, Tile]:
        return self._source

    @property
    def output(self) -> Union[Tensor, Tile]:
        return self._output

    @property
    def serial_only(self) -> bool:
        return self._serial_only

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

    @prune_nested_parallel_loops
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
    ) -> "MoveLet":
        if input_idx != 0:
            raise ValueError("input_idx must be 0 ")
        return common_move(self, 0, bank, layout, prefetching, **kwargs)

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> Impl:
        return common_move(self, -1, bank, layout, prefetching, **kwargs)

    @assert_stable_spec
    def split(self, k: int) -> Union["ReduceSum", "Loop"]:
        if k == self.source.dim_sizes[-1]:
            return self
        source_tile = self.source.simple_tile(self.source.dim_sizes[:-1] + (k,))
        driving_subscript = self.spec.operands_dim_subscripts()[0][-1]
        return Loop(
            subscripts=(driving_subscript,),
            tiles=frozenset([source_tile]),
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

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        inputs = list(inputs)
        if len(inputs) != 1:
            raise ValueError("Expected 1 input")
        return ReduceSum(inputs[0], output, serial_only=self.serial_only)
