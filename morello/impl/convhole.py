import dataclasses
import itertools
from typing import Callable, Iterable, Optional

from .. import specs, system_config
from .matmuls import MatmulAccumHole
from ..tensor import OperandIdx, SqueezingTile, TransposingTile
from .block import Block
from .actions import SlidingTileOutAction, TileOutAction
from .base import Impl, NonAllocatingLeaf
from .loops import Loop
from .moves import Moveable, _move_arguments, common_operand_move_actions
from .pruning import (
    ParentSummary,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from .settings import allow_sliding_windows
from .utils import assert_stable_spec, dim_range, gen_tile_sizes
from .zero import ZeroHole

_convhole_tile_out_params_cache = {}
_convhole_sliding_tile_out_params_cache = {}


@dataclasses.dataclass(frozen=True, slots=True)
class ConvHoleBase(NonAllocatingLeaf, Moveable):
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
        # Yield .tile_outs.
        #
        # Because this was a performance problem during beam search trails with the
        # random search heuristic, we can cache the possible tile sizes according to
        # ConvHole's spec. With some refactoring, we would be able to make
        # _can_tile_out a function of the Spec alone, then cache tile size results on
        # Specs and hash-cons the Specs. This isn't super important right now, though,
        # so we just do it here, where the problem is, and because we know it's safe
        # to do with a non-composed Spec and for ConvHole, which has no subclasses.
        try:
            gen_tile_sizes_results = _convhole_tile_out_params_cache[
                (self.spec.output.dim_sizes, self.spec)
            ]
        except KeyError:
            gen_tile_sizes_results = []
            for shape in gen_tile_sizes(
                self.spec.output.dim_sizes, filter=self._can_tile_out
            ):
                for parallel in [False] if self.spec.serial_only else [True, False]:
                    gen_tile_sizes_results.append((shape, parallel))
            _convhole_tile_out_params_cache[
                (self.spec.output.dim_sizes, self.spec)
            ] = gen_tile_sizes_results
        yield from (TileOutAction(self, *a) for a in gen_tile_sizes_results)

        # Yield .sliding_tile_outs.
        # We only need the levels for the left-hand side (images), because that
        # is the only operand over which one can slide.
        if allow_sliding_windows.get():
            try:
                sliding_tile_out_results = _convhole_sliding_tile_out_params_cache[
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
                _convhole_sliding_tile_out_params_cache[
                    (self.spec.output.dim_sizes, self.spec)
                ] = sliding_tile_out_results
            yield from (
                SlidingTileOutAction(self.sliding_tile_out, *a)
                for a in sliding_tile_out_results
            )

        yield from common_operand_move_actions(self)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.spec})"


class ConvHole(ConvHoleBase):
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
        zero_hole = ZeroHole(specs.Zero(s.output, serial_only=s.serial_only))
        accum_spec = specs.ConvolutionAccum(*s.operands, s.serial_only)
        accum_hole = ConvAccumHole(accum_spec)

        out_idx = len(s.inputs)
        return Block(
            s, (zero_hole, accum_hole), ((out_idx,), tuple(range(len(s.operands))))
        )

    @assert_stable_spec
    def complete(self) -> Impl:
        if any(d > 1 for d in self.spec.output.dim_sizes):
            return self.tile_out((1, 1, 1, 1)).complete()

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

        return self.to_accum().complete()


class ConvAccumHole(ConvHoleBase):
    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from super().actions(parent_summary)

        if all(d == 1 for d in self.spec.output.dim_sizes[2:]) and all(
            all(v == 1 for v in o.vector_shape[2:])
            for o in self.spec.operands
            if o.vector_shape
        ):
            yield self.spatial_split

    def spatial_split(self) -> "Impl":
        if self.spec.inputs[0].dim_sizes[2:] != self.spec.inputs[1].dim_sizes[2:]:
            raise ValueError(
                f"spatial_split can only be applied when image patch and filter "
                f"spatial dimensions match, but dimensions were "
                f"{self.spec.inputs[0].dim_sizes} and "
                f"{self.spec.inputs[1].dim_sizes}"
            )

        for operand in self.spec.operands:
            if not operand.vector_shape:
                continue
            if any(d != 1 for d in operand.vector_shape[2:]):
                raise ValueError(
                    f"spatial_split can only be applied when spatial vector dimensions "
                    f"are 1, but vector dimensions were {operand.vector_shape}"
                )

        # TODO: This doesn't range over the filters' spatial dims.!
        spatial_subscripts = self.spec.operands_dim_subscripts()[0][2:]

        spatial_rank = len(spatial_subscripts)
        dropped_dims = frozenset(range(2, len(self.spec.inputs[0].dim_sizes)))
        img_view = self.spec.inputs[0].simple_tile(
            OperandIdx(0),
            self.spec.inputs[0].dim_sizes[:2] + tuple(1 for _ in range(spatial_rank)),
        )
        reinterpreted_img_view = SqueezingTile(OperandIdx(0), img_view, dropped_dims)
        filters_view = self.spec.inputs[1].simple_tile(
            OperandIdx(1),
            self.spec.inputs[1].dim_sizes[:2] + tuple(1 for _ in range(spatial_rank)),
        )
        reinterpreted_filters_view = SqueezingTile(
            OperandIdx(1), filters_view, dropped_dims
        )
        if any(d > 1 for d in reinterpreted_filters_view.dim_sizes):
            reinterpreted_filters_view = TransposingTile(
                OperandIdx(1), reinterpreted_filters_view, (0, 1)
            )

        # Building a pass-through tile here is inelegant. Can we improve?
        output_view = self.spec.output.simple_tile(
            OperandIdx(2), self.spec.output.dim_sizes
        )
        reinterpreted_output_view = SqueezingTile(
            OperandIdx(2), output_view, dropped_dims
        )

        # Inner spec applied to produce a simple pixel (cross-batch and channel)
        matmul_spec = specs.MatmulAccum(
            reinterpreted_img_view.spec,
            reinterpreted_filters_view.spec,
            reinterpreted_output_view.spec,
            serial_only=self.spec.serial_only,
        )

        new_operands_subscripts = [
            [900, 901] + list(range(100, 100 + spatial_rank)),
            [902, 901] + list(range(100, 100 + spatial_rank)),
            [900, 902] + ([950] * spatial_rank),
        ]

        return Loop(
            spec=self.spec,
            subscripts=range(100, 100 + spatial_rank),
            operands_subscripts=tuple(map(tuple, new_operands_subscripts)),
            tiles=frozenset([img_view, filters_view]),
            inner_args=(
                reinterpreted_img_view,
                reinterpreted_filters_view,
                reinterpreted_output_view,
            ),
            inner=MatmulAccumHole(matmul_spec),
            parallel=False,
        )

    @assert_stable_spec
    def complete(self) -> Impl:
        if any(d > 1 for d in self.spec.output.dim_sizes):
            return self.tile_out((1, 1, 1, 1)).complete()

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

        if any(d > 1 for d in self.spec.inputs[0].dim_sizes):
            return self.spatial_split().complete()

        return self
