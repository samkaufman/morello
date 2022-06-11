import dataclasses
import functools
import math
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import dataclass_abc

from .. import specs, system_config
from ..tensor import OperandIdx, Tensor, TensorLike, Tile
from .base import AppliedImpl, Impl, make_applied_impl
from .utils import assert_stable_spec


class _TilingMixin:
    @property
    def _introduced_env_srt_sorted(
        self,
    ) -> Iterable[frozenset[tuple[Union[Tensor, Tile], Union[Tensor, Tile]]]]:
        return sorted(self.introduced, key=str)

    @property
    def _env_str_extra(self) -> Optional[str]:
        return None

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return [{b: 0 for b in system_config.current_system().banks}]

    @property
    def peak_memory(self) -> dict[str, int]:
        return self.inner.peak_memory

    @property
    def children(self) -> Tuple[Impl, ...]:
        return (self.inner,)

    def move_input(self, *args, **kwargs) -> "Impl":
        # Pass move_input through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_input(*args, **kwargs))

    def move_output(self, *args, **kwargs) -> "Impl":
        # Pass move_output through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_output(*args, **kwargs))

    def pad_transpack(self, *args, **kwargs) -> "Impl":
        return dataclasses.replace(
            self, inner=self.inner.pad_transpack(*args, **kwargs)
        )

    @assert_stable_spec
    def split(self, size: int) -> "Impl":
        return dataclasses.replace(self, inner=self.inner.split(size))

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        replacements = tuple(replacements)
        if len(replacements) != 1:
            raise ValueError(f"One replacement child expected; got {len(replacements)}")
        if replacements[0].spec != self.inner.spec:
            raise ValueError(f"Expected a replacement with spec {self.inner.spec}")
        return dataclasses.replace(self, inner=replacements[0])

    @assert_stable_spec
    def subschedule(self, fn: Callable[["Impl"], "Impl"]) -> "Impl":
        return self.replace_children([fn(self.inner)])


# TODO: Re-freeze.
@dataclass_abc.dataclass_abc(frozen=False, unsafe_hash=True, eq=True)
class Loop(Impl):
    """Iterate over subscripts of the inner Impl's operand tiles."""

    spec: specs.Spec
    subscripts: tuple[int, ...]
    tiles: frozenset[Tile]
    inner: Impl
    parallel: bool

    def __init__(
        self,
        spec: specs.Spec,
        subscripts: Iterable[int],
        tiles: Iterable[Tile],
        inner: Impl,
        parallel: bool,
    ):
        self.spec = spec
        self.subscripts = tuple(subscripts)
        self.tiles = frozenset(tiles)
        self.inner = inner
        self.parallel = parallel
        if self.parallel and not self.inner.spec.serial_only:
            raise ValueError("Parallel loop's child must be serial only")

    @property
    def steps(self) -> int:
        val = 1
        for s in self.subscripts:
            val *= self.steps_subscript(s)
        return val

    @property
    def full_steps(self) -> int:
        val = 1
        for s in self.subscripts:
            all_steps = self.steps_subscript(s)
            if self.boundary_size(s):
                val *= all_steps - 1
            else:
                val *= all_steps
        return val

    def steps_subscript(self, subscript) -> int:
        return self._apply_to_subscripts(subscript, lambda t: t.steps_dim)

    def boundary_size(self, subscript) -> int:
        return self._apply_to_subscripts(subscript, lambda t: t.boundary_size)

    def _apply_to_subscripts(self, subscript, fn, *args, **kwargs) -> int:
        """Apply `fn` to a dimension matching the given subscript.

        `fn` will be called once on the tile and the return value will be called
        with the dimension and any provided additional arguments.

        It may be called multiple times on multiple tiles and/or multiple
        dimensions to check that the results match.
        """
        # TODO: Raise a warning if the given subscript is not one over which
        #  this loop iterates.
        subscripts = self.spec.operands_dim_subscripts()
        for tile in self.tiles:
            subs = subscripts[tile.source]
            for dim, sub in enumerate(subs):
                if sub == subscript:
                    osize = self.spec.operands[tile.source].dim_sizes[dim]
                    return fn(tile)(dim, *args, origin_size=osize, **kwargs)
        raise ValueError(f"No subscript {subscript} found among tiles")

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return [{k: 0 for k in system_config.current_system().banks}]

    @property
    def peak_memory(self) -> dict[str, int]:
        return self.inner.peak_memory

    @property
    def children(self) -> Tuple[Impl, ...]:
        return (self.inner,)

    def move_input(self, *args, **kwargs) -> "Loop":
        # Pass move_input through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_input(*args, **kwargs))

    def move_output(self, *args, **kwargs) -> "Loop":
        # Pass move_output through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_output(*args, **kwargs))

    def pad_transpack(self, *args, **kwargs) -> "Loop":
        # Pass pad_transpack through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(
            self, inner=self.inner.pad_transpack(*args, **kwargs)
        )

    @assert_stable_spec
    def split(self, size: int) -> "Loop":
        # Pass split through to the inner schedule. This method is
        # sugar for calling subschedule.
        assert hasattr(self.inner, "split")
        return self.subschedule(lambda i: i.split(size))

    @assert_stable_spec
    def split_filters(self, size: int) -> "Loop":
        # Pass split_filters through to the inner schedule.
        assert hasattr(self.inner, "split")
        return self.subschedule(lambda i: i.split_filters(size))

    @assert_stable_spec
    def peel(self, *args, **kwargs) -> "Loop":
        # Pass split through to the inner schedule. This method is
        # sugar for calling subschedule.
        assert hasattr(self.inner, "peel")
        return dataclasses.replace(self, inner=self.inner.peel(*args, **kwargs))

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        replacements = tuple(replacements)
        if len(replacements) != 1:
            raise ValueError(f"One replacement child expected; got {len(replacements)}")
        if replacements[0].spec != self.inner.spec:
            raise ValueError(f"Expected a replacement with spec {self.inner.spec}")
        return dataclasses.replace(self, inner=replacements[0])

    @assert_stable_spec
    def subschedule(self, fn: Callable[["Impl"], "Impl"]) -> "Impl":
        return self.replace_children([fn(self.inner)])

    def apply(self, operands: Sequence[TensorLike]) -> AppliedImpl:
        assert [o.spec for o in operands] == list(self.spec.operands)
        inner_operands = list(operands)
        for tile in self.tiles:
            inner_operands[tile.source] = tile
        applied_body = self.inner.apply(inner_operands)
        return make_applied_impl(self.replace_children([applied_body]), operands)  # type: ignore


@dataclass_abc.dataclass_abc(frozen=True)
class SlidingWindowLoop(_TilingMixin, Impl):
    """A sliding window iterator.

    A sliding window iterator is an iterator over an output tile and a corresponding
    input which, after the first step, only loads non-overlapping portions of that
    input. It contains an Impl than computes the entire output tile, just like any
    other output tiling iterator.

    When printed, the non-overlapping size of the input is shown in square brackets.
    Note that, with convolutions of stride 1, which is all that is implemented in
    Morello at the moment, this size is equal to the output size. This isn't true in
    general.
    """

    live_tensor: Tensor  # Tensor taken as input by inner. Its origin should be the source.
    live_tensor_idx: OperandIdx

    # sliding_dim is the only dim between live_tensor and live_tensor's origin
    # that is non-equal.

    frontier_size: int
    other_tiles: tuple[Tile, ...]
    spec: specs.Spec
    inner: Impl

    @functools.cached_property
    def _sliding_dim(self) -> int:
        live_tensor_origin = self.spec.operands[self.live_tensor_idx]
        for dim in range(len(self.live_tensor.dim_sizes)):
            if self.live_tensor.dim_sizes[dim] != live_tensor_origin.dim_sizes[dim]:
                return dim
        raise Exception("All dimensions were equal")

    @functools.cached_property
    def frontier_shape(self) -> tuple[int, ...]:
        shape = list(self.live_tensor.dim_sizes)
        shape[self._sliding_dim] = self.frontier_size
        return tuple(shape)

    @assert_stable_spec
    def peel(self, *args, **kwargs) -> "SlidingWindowLoop":
        # Pass split through to the inner schedule. This method is
        # sugar for calling subschedule.
        #
        # Note that there is no opportunity here to introduce sliding
        # windows to avoid recomputation of intermediate values.
        assert hasattr(self.inner, "peel")
        return dataclasses.replace(self, inner=self.inner.peel(*args, **kwargs))

    @property
    def _env_str_keyword(self) -> str:
        return "slide"

    @property
    def _introduced_env_srt_sorted(
        self,
    ) -> Iterable[tuple[Union[Tensor, Tile], specs.TensorSpec]]:
        live_tensor_origin = self.spec.operands[self.live_tensor_idx]
        return [(self.live_tensor, live_tensor_origin)] + [
            (t, self.spec.operands[t.source]) for t in self.other_tiles
        ]

    @property
    def _env_str_extra(self) -> Optional[str]:
        return str(self.frontier_size)

    @property
    def whole_loads(self) -> int:
        return 1

    @property
    def update_loads(self) -> int:
        live_tensor_origin = self.spec.operands[self.live_tensor_idx]
        size = live_tensor_origin.dim_sizes[self._sliding_dim]
        size -= self.live_tensor.dim_sizes[self._sliding_dim]
        return math.ceil(size / self.frontier_size)

    @property
    def steps(self) -> int:
        return self.whole_loads + self.update_loads

    @functools.cached_property
    def introduced(self) -> frozenset[tuple[Union[Tensor, Tile], specs.TensorSpec]]:
        live_tensor_origin = self.spec.operands[self.live_tensor_idx]
        return frozenset([(self.live_tensor, live_tensor_origin)]) | frozenset(
            (t, self.spec.operands[t.source]) for t in self.other_tiles
        )

    @property
    def peak_memory(self) -> dict[str, int]:
        inner_peak_memory = dict(self.inner.peak_memory)
        inner_peak_memory[self.live_tensor.bank] += self.live_tensor.spec.bytes_used
        return inner_peak_memory

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        mem = {k: 0 for k in system_config.current_system().banks}
        mem[self.live_tensor.bank] = self.live_tensor.spec.bytes_used
        return [mem]

    def apply(self, operands: Sequence[TensorLike]) -> AppliedImpl:
        inner_operands = list(operands)
        for tile in self.other_tiles:
            inner_operands[tile.source] = tile
        applied_body = self.inner.apply(inner_operands)
        return make_applied_impl(self.replace_children([applied_body]), operands)  # type: ignore
