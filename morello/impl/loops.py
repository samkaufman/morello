import dataclasses
import functools
import math
from typing import Callable, Iterable, Optional, Tuple, Union

import dataclass_abc
import termcolor

from .. import specs, system_config
from ..tensor import ConvolutionImageTile, Tensor, Tile
from .base import Impl
from .utils import assert_stable_spec


# noinspection PyTypeChecker, PyArgumentList, PyUnresolvedReferences
class _TilingMixin:
    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ) -> str:
        istr = ")"
        if not underscore_inner:
            istr = f", {self.inner.env_str(name_tensor_fn, fancy=fancy)})"

        left_strs, right_strs = [], []
        # TODO: Come up with a more readable ordering over tiles
        for tile, source in self._introduced_env_srt_sorted:
            left_strs.append(
                _loop_operand_str(tile, name_tensor_fn=name_tensor_fn, fancy=fancy)
            )
            right_strs.append(name_tensor_fn(source))
        assert left_strs and right_strs

        keyword = self._env_str_keyword
        extra = self._env_str_extra
        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])
        if extra:
            keyword += f"[{extra}]"

        left_concat = ", ".join(left_strs)
        right_concat = ", ".join(right_strs)
        return f"{keyword} ({left_concat}) {arrow} ({right_concat}{istr}"

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


@dataclass_abc.dataclass_abc(frozen=False, unsafe_hash=True, eq=True)
class Loop(Impl):
    """Iterate over subscripts of the inner Impl's operand tiles."""

    subscripts: tuple[int]
    tiles: frozenset[Tile]
    inner: Impl
    parallel: bool

    def __post_init__(self):
        if self.parallel and not self.inner.spec.serial_only:
            raise ValueError("Parallel loop's child must be serial only")

    @functools.cached_property
    def spec(self) -> specs.Spec:
        serial_only = self.inner.spec.serial_only
        if self.parallel:
            serial_only = False
        return self.inner.spec.replace_io(
            inputs=tuple(inp.spec for inp in self.inputs),
            output=self.output.spec,
            serial_only=serial_only,
        )

    @functools.cached_property
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        # The inputs of a Loop are inner's inputs, with the introduced
        # tiles substituted for their origins
        new_inputs = []
        for inner_inp in self.inner.inputs:
            input_to_add = inner_inp
            if inner_inp in self.tiles:
                input_to_add = inner_inp.origin
            assert input_to_add is not None
            new_inputs.append(input_to_add)
        return tuple(new_inputs)

    @property
    def output(self) -> Union[Tensor, Tile]:
        # The output of a Loop is inner's output or, if that output is
        # a tile introduced in this Loop as an tile, the tile itself.
        o: Union[Tensor, Tile] = self.inner.output
        if o in self.tiles:
            o = o.origin
        return o

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ) -> str:
        istr = ")"
        if not underscore_inner:
            istr = f", {self.inner.env_str(name_tensor_fn, fancy=fancy)})"

        left_strs, right_strs = [], []
        for it in sorted(self.tiles, key=str):
            left_strs.append(
                _loop_operand_str(it, name_tensor_fn=name_tensor_fn, fancy=fancy)
            )
            right_strs.append(name_tensor_fn(it.origin))
        assert left_strs and right_strs

        keyword = "tile"
        if self.parallel:
            keyword = "par " + keyword
        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])

        left_concat = ", ".join(left_strs)
        right_concat = ", ".join(right_strs)
        return f"{keyword} ({left_concat}) {arrow} ({right_concat}{istr}"

    @property
    def steps(self) -> int:
        val = 1
        for s in self.subscripts:
            val *= self.steps_subscript(s)
        return val

    def steps_subscript(
        self, subscript, concrete_outer_size: Optional[int] = None
    ) -> int:
        return self._apply_to_subscripts(
            lambda t: t.steps_dim, subscript, concrete_outer_size
        )

    def boundary_size(
        self, subscript, concrete_outer_size: Optional[int] = None
    ) -> int:
        return self._apply_to_subscripts(
            lambda t: t.boundary_size, subscript, concrete_outer_size
        )

    def _apply_to_subscripts(self, fn, subscript, *args) -> int:
        """Apply `fn` to a dimension matching the given subscript.

        `fn` will be called once on the tile and the return value will be called
        with the dimension and any provided additional arguments.

        It may be called multiple times on multiple tiles and/or multiple
        dimensions to check that the results match.
        """
        # TODO: Raise a warning if the given subscript is not one over which
        #  this loop iterates.

        value: Optional[int] = None
        for tile, subs in zip(self.inner.operands, self.spec.operands_dim_subscripts()):
            if not isinstance(tile, Tile):
                continue
            for dim, sub in enumerate(subs):
                if sub == subscript:
                    if value is None:
                        value = fn(tile)(dim, *args)
                    assert value == fn(tile)(dim, *args)

        if value is None:
            raise ValueError(f"No subscript {subscript} found among tiles")

        return value

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

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        return type(self)(
            driving_tile=self.driving_tile,
            dependent_tiles=self.dependent_tiles,
            inner=self.inner,
            parallel=self.parallel,
        )

    @assert_stable_spec
    def subschedule(self, fn: Callable[["Impl"], "Impl"]) -> "Impl":
        return self.replace_children([fn(self.inner)])


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

    inputs: Tuple[Union[Tile, Tensor]]
    output: Union[Tile, Tensor]

    # TODO: Replace live_tensor with an index into inner's inputs?
    live_tensor: Tensor  # Tensor taken as input by inner. Its origin should be the source.

    # sliding_dim is the only dim between live_tensor and live_tensor.origin
    # that is non-equal.

    frontier_size: int
    other_tiles: tuple[Tile, ...]
    # TODO: replace spec w/ property
    spec: specs.Spec
    inner: Impl

    def __post_init__(self):
        assert (
            self.live_tensor.origin in self.inputs
            or self.live_tensor.origin == self.output
        ), f"{str(self.live_tensor.origin)} not in {tuple(str(i) for i in self.inputs)} or {str(self.output)}"
        assert all(
            self.live_tensor.dim_sizes[d] == self.live_tensor.origin.dim_sizes[d]
            for d in range(len(self.live_tensor.dim_sizes))
            if d != self._sliding_dim
        ), f"sliding_dim was {self._sliding_dim} but shapes were {self.live_tensor.dim_sizes} and {self.live_tensor.origin.dim_sizes}"
        assert self.spec == self._compute_spec()

        # Check that substitution over the self.inner.inputs equals self.inputs
        assert self.inputs == tuple(
            self._sub_origin(inp) for inp in self.inner.inputs
        ), f"{[str(i) for i in self.inputs]} != {[str(self._sub_origin(inp)) for inp in self.inner.inputs]}"
        assert self.output == self._sub_origin(
            self.inner.output
        ), f"{self.output} != {self._sub_origin(self.inner.output)}"

    def _compute_spec(self) -> specs.Spec:
        return self.inner.spec.replace_io(
            tuple(inp.spec for inp in self.inputs), self.output.spec
        )

    def _sub_origin(self, t):
        if self.live_tensor == t:
            return t.origin
        elif t in self.other_tiles:
            return t.origin
        return t

    @functools.cached_property
    def _sliding_dim(self) -> int:
        for dim in range(len(self.live_tensor.dim_sizes)):
            if (
                self.live_tensor.dim_sizes[dim]
                != self.live_tensor.origin.dim_sizes[dim]
            ):
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
    ) -> Iterable[tuple[Union[Tensor, Tile], Union[Tensor, Tile]]]:
        return [(self.live_tensor, self.live_tensor.origin)] + [
            (t, t.origin) for t in self.other_tiles
        ]

    @property
    def _env_str_extra(self) -> Optional[str]:
        return str(self.frontier_size)

    @property
    def whole_loads(self) -> int:
        return 1

    @property
    def update_loads(self) -> int:
        size = self.live_tensor.origin.dim_sizes[self._sliding_dim]
        size -= self.live_tensor.dim_sizes[self._sliding_dim]
        return math.ceil(size / self.frontier_size)

    @property
    def steps(self) -> int:
        return self.whole_loads + self.update_loads

    @functools.cached_property
    def introduced(self) -> frozenset[tuple[Union[Tensor, Tile], Union[Tensor, Tile]]]:
        return frozenset([(self.live_tensor, self.live_tensor.origin)]) | frozenset(
            (t, t.origin) for t in self.other_tiles
        )

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        raise NotImplementedError()

    @property
    def peak_memory(self) -> dict[str, int]:
        inner_peak_memory = dict(self.inner.peak_memory)
        inner_peak_memory[self.live_tensor.bank] += self.live_tensor.bytes_used
        return inner_peak_memory

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        mem = {k: 0 for k in system_config.current_system().banks}
        mem[self.live_tensor.bank] = self.live_tensor.bytes_used
        return [mem]


# TODO: Roll MatmulSplitLoop functionality into Loop
@dataclass_abc.dataclass_abc(frozen=True)
class MatmulSplitLoop(_TilingMixin, Impl):
    """A Impl representing a blocked loop over the k dimension of the matmul.

    The `lhs` and `rhs` members are the outer operands.
    """

    lhs: Union[Tensor, Tile]
    rhs: Union[Tensor, Tile]
    output: Union[Tensor, Tile]
    inner: Impl

    def __post_init__(self):
        assert self.output.dim_sizes == (
            self.lhs.dim_sizes[0],
            self.rhs.dim_sizes[1],
        ), f"Expected output to have shape {self.lhs.dim_sizes[0]}×{self.rhs.dim_sizes[1]}"

    # TODO: Needing this for split loops is a design flaw. Should just be
    #   able to specify dimensions being iterated over.
    @property
    def driving_tile(self):
        return self.inner.inputs[0]

    @property
    def dependent_tiles(self):
        return (self.inner.inputs[1],)

    @property
    def parallel(self) -> bool:
        return False

    def steps_subscript(
        self, subscript, concrete_outer_size: Optional[int] = None
    ) -> int:
        expected_subscript = self.spec.operands_dim_subscripts()[0][-1]
        if subscript != expected_subscript:
            raise ValueError(f"subscript must be {expected_subscript}")
        if concrete_outer_size is None:
            concrete_outer_size = self.inputs[0].dim_sizes[1]
        return math.ceil(concrete_outer_size / self.inner.inputs[0].dim_sizes[1])

    def boundary_size(
        self, subscript, concrete_outer_size: Optional[int] = None
    ) -> int:
        expected_subscript = self.spec.operands_dim_subscripts()[0][-1]
        if subscript != expected_subscript:
            raise ValueError(f"subscript must be {expected_subscript}")
        if concrete_outer_size is None:
            concrete_outer_size = self.inputs[0].dim_sizes[1]
        return concrete_outer_size % self.inner.inputs[0].dim_sizes[1]

    @property
    def introduced(self) -> frozenset[tuple[Union[Tensor, Tile], Union[Tensor, Tile]]]:
        operands = list(self.inputs) + [self.output]
        inner_operands = list(self.inner.inputs) + [self.inner.output]
        assert len(operands) == len(inner_operands)
        return frozenset(
            (dest, src)
            for dest, src in zip(inner_operands, operands)
            if dest is not src
        )

    @property
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        return self.lhs, self.rhs

    @functools.cached_property
    def spec(self) -> specs.Spec:
        return specs.Matmul(
            self.lhs.spec, self.rhs.spec, self.output.spec, self.inner.spec.serial_only
        )

    @property
    def _env_str_keyword(self):
        return "split"

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        l, r = inputs
        return MatmulSplitLoop(lhs=l, rhs=r, output=output, inner=self.inner)

    @property
    def steps(self) -> int:
        # We expect the inner schedule here to have two inputs (the two matrices).
        inner_lhs, inner_rhs = self.inner.inputs

        result = None
        for dest, src in [(inner_lhs, self.lhs), (inner_rhs, self.rhs)]:
            dh, dw = dest.dim_sizes[0], dest.dim_sizes[1]
            sh, sw = src.dim_sizes[0], src.dim_sizes[1]
            r = math.ceil(sh / dh) * math.ceil(sw / dw)
            if result is None:
                result = r
            else:
                assert r == result
        return result


def _loop_operand_str(
    t, *, name_tensor_fn: Callable[[Union[Tensor, Tile]], str], fancy: bool
):
    if isinstance(t, ConvolutionImageTile):
        prefix = "conv"
        if fancy:
            prefix = termcolor.colored(prefix, attrs=["bold"])
        desc_part = prefix + " " + "×".join(str(s) for s in t.dim_sizes)
    elif isinstance(t, Tile):
        desc_part = "×".join(str(s) for s in t.dim_sizes)
    else:
        desc_part = str(t)
    return f"{name_tensor_fn(t)}: {desc_part}"
