import itertools
from typing import Callable, Iterable, Optional

import dataclass_abc

from .. import specs, system_config
from ..layouts import Layout
from ..tensor import Tensor, Tile
from .actions import SlidingTileOutAction, TileOutAction
from .base import Impl, NonAllocatingLeaf
from .moves import MoveLet, _move_arguments, common_move, common_operand_move_actions
from .pruning import (
    ParentSummary,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from .settings import allow_sliding_windows
from .utils import assert_stable_spec, dim_range, gen_tile_sizes

_directconv_tile_out_params_cache = {}
_directconv_sliding_tile_out_params_cache = {}


@dataclass_abc.dataclass_abc(frozen=True)
class DirectConv(NonAllocatingLeaf):
    """A native implementation of a convolution.

    Stride is 1. No padding.
    """

    spec: specs.Spec

    def replace_spec(self, new_spec: specs.Spec) -> "Impl":
        assert type(self) is DirectConv
        return DirectConv(new_spec)

    @property
    def is_scheduled(self) -> bool:
        # TODO: Drop these RF constants. Instead, use target-specific impls.
        if not all(op.bank in ("RF", "HexagonRF") for op in self.spec.operands):
            return False
        if any(d > 1 for d in self.spec.output.dim_sizes):
            return False
        return True

    # TODO: Remove split_filters
    @assert_stable_spec
    def split_filters(self, k: int) -> "Impl":
        return self.tile_out(self.spec.output.dim_sizes[:-1] + (k,))

    def move_input(
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        if input_idx not in (0, 1):
            raise ValueError("input_idx must be 0 or 1")
        return common_move(self, input_idx, bank, layout, prefetching, **kwargs)

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        return common_move(self, -1, bank, layout, prefetching, **kwargs)

    @assert_stable_spec
    def split(self, size: int) -> "Impl":
        raise NotImplementedError("Split not implemented for DirectConv")

    @assert_stable_spec
    def complete(self) -> Impl:
        if any(d > 1 for d in self.spec.output.dim_sizes):
            return self.tile_out((1, 1, 1)).complete()

        system = system_config.current_system()

        lhs, rhs, out = self.spec.operands
        next_general_lhs = system.next_general_bank(lhs.bank)
        if next_general_lhs:
            return self.move_input(0, bank=next_general_lhs).complete()
        next_general_rhs = system.next_general_bank(rhs.bank)
        if next_general_rhs:
            return self.move_input(1, bank=next_general_rhs).complete()
        next_general_out = system.next_general_bank(out.bank)
        if next_general_out:
            return self.move_output(bank=next_general_out).complete()

        return self

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        # Yield .tile_outs.
        #
        # Because this was a performance problem during beam search trails with the
        # random search heuristic, we can cache the possible tile sizes according to
        # DirectConv's spec. With some refactoring, we would be able to make
        # _can_tile_out a function of the Spec alone, then cache tile size results on
        # Specs and hash-cons the Specs. This isn't super important right now, though,
        # so we just do it here, where the problem is, and because we know it's safe
        # to do with a non-composed Spec and for DirectConv, which has no subclasses.
        try:
            gen_tile_sizes_results = _directconv_tile_out_params_cache[
                (self.spec.output.dim_sizes, self.spec)
            ]
        except KeyError:
            gen_tile_sizes_results = []
            for shape in gen_tile_sizes(
                self.spec.output.dim_sizes, filter=self._can_tile_out
            ):
                for parallel in [False] if self.spec.serial_only else [True, False]:
                    gen_tile_sizes_results.append((shape, parallel))
            _directconv_tile_out_params_cache[
                (self.spec.output.dim_sizes, self.spec)
            ] = gen_tile_sizes_results
        yield from (TileOutAction(self, *a) for a in gen_tile_sizes_results)

        # Yield .sliding_tile_outs.
        # We only need the levels for the left-hand side (images), because that
        # is the only operand over which one can slide.
        if allow_sliding_windows.get():
            try:
                sliding_tile_out_results = _directconv_sliding_tile_out_params_cache[
                    (self.spec.output.dim_sizes, self.spec)
                ]
            except KeyError:
                sliding_tile_out_results = []
                for bank, sliding_dim in itertools.product(
                    set(b for b, _, _ in _move_arguments(self.spec.inputs[0])), [0, 1]
                ):
                    for slide_size in dim_range(
                        self.spec.output.dim_sizes[sliding_dim], include_end=False
                    ):
                        if self._can_sliding_tile_out(sliding_dim, slide_size, bank):
                            sliding_tile_out_results.append(
                                (sliding_dim, slide_size, bank)
                            )
                _directconv_sliding_tile_out_params_cache[
                    (self.spec.output.dim_sizes, self.spec)
                ] = sliding_tile_out_results
            yield from (
                SlidingTileOutAction(self.sliding_tile_out, *a)
                for a in sliding_tile_out_results
            )

        # Search over all possible filters splits
        # TODO: We don't need a sep. split_filters. Should be just a dim for tile_out!
        # if self.rhs.dim_sizes[-1] > 1:
        #     for k in range(1, self.rhs.dim_sizes[-1]):
        #         yield functools.partial(self.split_filters, k)

        yield from common_operand_move_actions(self)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.spec})"