import functools
import itertools
from typing import Union, Tuple, Callable, Optional, Iterable

import dataclass_abc

from .actions import TileOutAction, SlidingTileOutAction
from .base import Impl
from .moves import _move_arguments, common_operand_move_actions, common_move
from .pruning import (
    ParentSummary,
    prune_relayout_cycles,
    break_moves_symmetries,
    break_tile_out_symmetries,
)
from .settings import allow_sliding_windows
from .utils import assert_stable_spec, gen_tile_sizes, dim_range
from .. import specs, system_config
from ..specs import Layout
from ..tensor import Tensor, Tile

_directconv_tile_out_params_cache = {}
_directconv_sliding_tile_out_params_cache = {}


@dataclass_abc.dataclass_abc(frozen=True)
class DirectConv(Impl):
    """A stride-1, no-padding convolution over a single-channel 2-D image."""

    # TODO: Add support for multi-channel images

    lhs: Union[Tensor, Tile]
    "An image over which to convolve."
    rhs: Union[Tensor, Tile]
    "An (m, n, k) tensor of m-by-n filters."
    output: Union[Tensor, Tile]
    "The (_, _, k) output tensor."
    serial_only: bool

    def __post_init__(self):
        if len(self.lhs.dim_sizes) != 2:
            raise ValueError("lhs is not a matrix")
        if len(self.rhs.dim_sizes) != 3:
            raise ValueError("rhs is not a rank-3 tensor")
        if (
            self.image_width < self.kernel_width
            or self.image_height < self.kernel_height
        ):
            raise Exception("Image too small to apply a filter without padding")
        # Check output shape
        assert self.output.dim_sizes == (
            1 + self.lhs.dim_sizes[0] - self.rhs.dim_sizes[0],
            1 + self.lhs.dim_sizes[1] - self.rhs.dim_sizes[1],
            self.rhs.dim_sizes[2],
        )

    @property
    def children(self) -> Tuple["Impl", ...]:
        return tuple()

    @property
    def inputs(self):
        return self.lhs, self.rhs

    @functools.cached_property
    def spec(self) -> specs.Convolution:
        return specs.Convolution(
            self.lhs.spec, self.rhs.spec, self.output.spec, self.serial_only
        )

    @property
    def image_height(self):
        return self.lhs.dim_sizes[0]

    @property
    def image_width(self):
        return self.lhs.dim_sizes[1]

    @property
    def kernel_height(self):
        return self.rhs.dim_sizes[0]

    @property
    def kernel_width(self):
        return self.rhs.dim_sizes[1]

    @property
    def kernel_count(self):
        return self.rhs.dim_sizes[2]

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        return (
            f"DirectConv({name_tensor_fn(self.lhs)}, "
            f"{name_tensor_fn(self.rhs)}, "
            f"{name_tensor_fn(self.output)})"
        )

    @property
    def is_scheduled(self) -> bool:
        # TODO: Drop these RF constants. Instead, use target-specific impls.
        if not all(op.bank in ("RF", "HexagonRF") for op in self.operands):
            return False
        if any(d > 1 for d in self.output.dim_sizes):
            return False
        return True

    @property
    def innermost(self) -> "Impl":
        return self

    # @assert_stable_spec
    # def tile_out(self, output_shape: Tuple[int, ...]) -> "Impl":
    #     # TODO: DirectConv acts as though it has a rank-2 output because of
    #     #   split_filters. Fix this.
    #     return super().tile_out(output_shape + (self.rhs.dim_sizes[-1],))

    # TODO: Remove split_filters
    @assert_stable_spec
    def split_filters(self, k: int) -> "Impl":
        return self.tile_out(self.output.dim_sizes[:-1] + (k,))

    def move_input(
        self,
        input_idx: int,
        bank: Optional[int] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        if input_idx == 0:
            return common_move(self, "lhs", bank, layout, prefetching, **kwargs)
        elif input_idx == 1:
            return common_move(self, "rhs", bank, layout, prefetching, **kwargs)
        else:
            raise ValueError("input_idx must be 0 or 1")

    def move_output(
        self,
        bank: Optional[int] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        return common_move(self, "output", bank, layout, prefetching, **kwargs)

    @assert_stable_spec
    def split(self, size: int) -> "Impl":
        raise NotImplementedError("Split not implemented for DirectConv")

    @assert_stable_spec
    def complete(self) -> Impl:
        if any(d > 1 for d in self.output.dim_sizes):
            return self.tile_out((1, 1, 1)).complete()

        system = system_config.current_system()

        next_general_lhs = system.next_general_bank(self.lhs.bank)
        if next_general_lhs:
            return self.move_input(0, bank=next_general_lhs).complete()
        next_general_rhs = system.next_general_bank(self.rhs.bank)
        if next_general_rhs:
            return self.move_input(1, bank=next_general_rhs).complete()
        next_general_out = system.next_general_bank(self.output.bank)
        if next_general_out:
            return self.move_output(bank=next_general_out).complete()

        return self

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
                (self.output.dim_sizes, self.spec)
            ]
        except KeyError:
            gen_tile_sizes_results = []
            for shape in gen_tile_sizes(
                self.output.dim_sizes, filter=self._can_tile_out
            ):
                for parallel in [False] if self.serial_only else [True, False]:
                    gen_tile_sizes_results.append((shape, parallel))
            _directconv_tile_out_params_cache[
                (self.output.dim_sizes, self.spec)
            ] = gen_tile_sizes_results
        yield from (TileOutAction(self, *a) for a in gen_tile_sizes_results)

        # Yield .sliding_tile_outs.
        # We only need the levels for the left-hand side (images), because that
        # is the only operand over which one can slide.
        if allow_sliding_windows.get():
            try:
                sliding_tile_out_results = _directconv_sliding_tile_out_params_cache[
                    (self.output.dim_sizes, self.spec)
                ]
            except KeyError:
                sliding_tile_out_results = []
                for bank, sliding_dim in itertools.product(
                    set(b for b, _, _ in _move_arguments(self.lhs)), [0, 1]
                ):
                    for slide_size in dim_range(
                        self.output.dim_sizes[sliding_dim], include_end=False
                    ):
                        if self._can_sliding_tile_out(sliding_dim, slide_size, bank):
                            sliding_tile_out_results.append(
                                (sliding_dim, slide_size, bank)
                            )
                _directconv_sliding_tile_out_params_cache[
                    (self.output.dim_sizes, self.spec)
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

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        r = list(replacements)
        if r:
            raise ValueError(
                f"DirectConv has no children, but given {len(r)} replacements"
            )
        return self

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        lhs, rhs = inputs
        return DirectConv(lhs, rhs, output, serial_only=self.serial_only)

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return [{k: 0 for k in system_config.current_system().banks}]

    @property
    def peak_memory(self) -> dict[str, int]:
        return {k: 0 for k in system_config.current_system().banks}

    def __str__(self) -> str:
        epi = ", serial" if self.serial_only else ""
        return (
            f"{type(self).__name__}({self.lhs}, {self.rhs}, " f"out={self.output}{epi})"
        )
