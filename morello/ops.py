import abc
import contextvars
import dataclasses
import enum
import functools
import itertools
import math
import sys
import warnings
from operator import mul
from typing import (
    Callable,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import dataclass_abc
import termcolor

from . import specs, system_config, tiling, utils
from .specs import Layout
from .tensor import ConvolutionImageTile, SimpleTile, Tensor, Tile


class TileSizeMode(enum.Enum):
    ALL = enum.auto()
    CACHE_LINE_MULTIPLES = enum.auto()
    POWERS_OF_TWO = enum.auto()


tile_size_mode: contextvars.ContextVar[TileSizeMode] = contextvars.ContextVar(
    "tile_size_mode", default=TileSizeMode.POWERS_OF_TWO
)
allow_sliding_windows: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "allow_sliding_windows", default=True
)
allow_reduce_splits: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "allow_reduce_splits", default=True
)

ALLOW_ABOVE_REGISTERS_RELAYOUT = True
PRUNE_RELAYOUT_CYCLES = True
BREAK_MOVE_SYMMETRIES = True
BREAK_SEQUENTIAL_TILES = False

T = TypeVar("T")


def _assert_stable_spec(func):
    """Assert that a method returns a Schedule with the same spec as its first input."""

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        orig_spec = args[0].spec
        value = func(*args, **kwargs)
        assert (
            value.spec == orig_spec
        ), f"Spec {orig_spec} became {value.spec} while executing {func.__name__}"
        return value

    return wrapper_decorator


def _zipply(fn, *args: List[T]) -> List[T]:
    return [fn(z) for z in zip(*args)]


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


def dim_range(dim: int, include_end: bool = True) -> Iterable[int]:
    """Returns possible dimension sizes up to `dim`.

    If `tile_size_mode` is set to `CACHE_LINE_MULTIPLES`, returned sizes
    will be evenly divisible by the system's cache line size.

    If `tile_size_mode` is set to `POWERS_OF_TWO`, returned sizes
    will be powers of two.

    :param include_end: If False, results will exclude `dim` itself.
    """
    assert dim >= 0
    if dim == 0:
        return
    line_size = system_config.DEFAULT_SYSTEM_CONFIG.line_size
    if tile_size_mode.get() == TileSizeMode.CACHE_LINE_MULTIPLES:
        it = range(line_size, dim, line_size)
        if dim > 1:
            it = itertools.chain([1], it)
        if include_end:
            it = itertools.chain(it, [dim])
        yield from it
    elif tile_size_mode.get() == TileSizeMode.POWERS_OF_TWO:
        assert dim >= 0
        if dim == 0:
            return
        power = 0
        while True:
            if 2 ** power >= dim:
                break
            yield 2 ** power
            power += 1
        if include_end:
            yield dim
    elif tile_size_mode.get() == TileSizeMode.ALL:
        end = dim
        if include_end:
            end += 1
        yield from range(1, end)
    else:
        raise NotImplementedError(f"Unsupported tile size mode: {tile_size_mode.get()}")


class SplitNotSupportedByHeadError(NotImplementedError):
    pass


@dataclasses.dataclass
class ParentSummary:
    parent: "Schedule"
    movements: FrozenSet[Tuple[Union[Tensor, Tile], int]]

    @staticmethod
    def update(
        original: Optional["ParentSummary"], parent: "Schedule"
    ) -> "ParentSummary":
        if original is None:
            movements = set()
        else:
            movements = set(original.movements)

        if isinstance(parent, MoveLet):
            movements.add((parent.destination, parent.destination.level))

        return ParentSummary(parent=parent, movements=frozenset(movements))


@dataclasses.dataclass(frozen=True)
class MoveAction:
    """Wraps a function which wraps a Schedule in a MoveLet.

    This is used instead of a functools.partial to expose fields to
    break_moves_symmetries.
    """

    func: Callable[[Optional[int], Optional[Layout]], "Schedule"]
    source: Union[Tensor, Tile]
    input_idx: Optional[int]
    level: Optional[int] = None
    layout: Optional[specs.Layout] = None

    def __post_init__(self):
        assert (
            any(d > 1 for d in self.source.dim_sizes) or self.layout == Layout.ROW_MAJOR
        ), f"Layout was {self.layout} for dims. {self.source.dim_sizes}"

    def __call__(self):
        return self.func(self.level, self.layout)

    def __str__(self):
        return (
            f"MoveAction(input_idx={self.input_idx}, source={str(self.source)}, "
            f"level={self.level}, layout={str(self.layout)})"
        )


@dataclasses.dataclass(frozen=True)
class PeelAction:
    func: Callable[[Optional[int], Optional[Layout]], "Schedule"]
    level: Optional[int] = None
    layout: Optional[specs.Layout] = None

    def __call__(self):
        return self.func(self.level, self.layout)


@dataclasses.dataclass(frozen=True)
class TileOutAction:
    func: Callable[..., "Schedule"]
    shape: Tuple[int, ...]
    parallel: bool

    def __call__(self):
        return self.func(self.shape, parallel=self.parallel)


@dataclasses.dataclass(frozen=True)
class SlidingTileOutAction:
    func: Callable[[int, int, int], "Schedule"]
    sliding_dim: int
    output_size: int
    level: int

    def __call__(self):
        return self.func(self.sliding_dim, self.output_size, self.level)


@dataclasses.dataclass(frozen=True)
class MatmulSplitAction:
    func: Callable[[int], "Schedule"]
    k: int

    def __call__(self):
        return self.func(self.k)


def prune_relayout_cycles(func):
    if not PRUNE_RELAYOUT_CYCLES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        for action in func(*args, **kwargs):
            if not isinstance(action, MoveAction):
                yield action
                continue
            if (action.source, action.level) in parent_summary.movements:
                continue
            yield action

    return wrapper_decorator


def break_moves_symmetries(func):
    """Wraps a function which yields scheduling actions to filter symmetric moves.

    This places a total ordering over moves' source tensors, requiring that all moves
    be from a tensor greater than or equal to its parent move.

    Pruning using this filtering decorator, unfortunately, means that `func` still
    enumerates unneeded actions, but doesn't expand them, which is the important thing.
    Pruning was implemented as a decorator to separate concerns and make it simple to
    disable for testing.
    """
    if not BREAK_MOVE_SYMMETRIES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        if not isinstance(parent_summary.parent, MoveLet):
            yield from func(*args, **kwargs)
            return
        for action in func(*args, **kwargs):
            if not isinstance(action, MoveAction):
                yield action
                continue

            # Assign each operand to an integer. Input index if it applies, otherwise:
            # arbitrarily large number
            action_operand_idx = action.input_idx
            parent_schedule_idx = parent_summary.parent.input_idx
            if action_operand_idx is None:
                action_operand_idx = sys.maxsize
            if parent_schedule_idx is None:
                parent_schedule_idx = sys.maxsize

            # Assert lexicographic order
            if action_operand_idx < parent_schedule_idx:
                continue

            # Assert that there is no interleaving of destination levels between moves.
            if action.level > parent_summary.parent.destination.root.level:
                continue

            # Avoid relayout once in registers if the operand previously had the
            #   opportunity to be put into that layout as moved into registers.
            #   This applies if a single level 0-to-0 relayout happened as well,
            #   preventing multiple relayouts in registers.
            if action.level == 0 and (action.source, 0) in parent_summary.movements:
                continue

            yield action

    return wrapper_decorator


def break_tile_out_symmetries(func):
    """Wraps an actions method to never return sequential .tile_outs.

    This is a no-op if BREAK_SEQUENTIAL_TILES is false.
    """
    if not BREAK_SEQUENTIAL_TILES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        # If no ParentSummary is given, we're at the root, and done.
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        parent_schedule = parent_summary.parent
        # Check if the parent is a loop that tiles output. If not, we're done.
        if (not isinstance(parent_schedule, Loop)) or (
            args[0].output not in [dest for dest, _ in parent_schedule.introduced]
        ):
            yield from func(*args, **kwargs)
            return
        # Filter all .tile_outs with the same parallel flag at the parent.
        for action in func(*args, **kwargs):
            if (
                not isinstance(action, TileOutAction)
                or action.parallel != parent_schedule.parallel
            ):
                yield action

    return wrapper_decorator


def break_matmul_split_symmetries(func):
    if not BREAK_SEQUENTIAL_TILES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        # If no ParentSummary is given, we're at the root, and done.
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        parent_schedule = parent_summary.parent
        # If the parent isn't a matmul split, we're done.
        if not isinstance(parent_schedule, MatmulSplitAction):
            yield from func(*args, **kwargs)
            return
        # Filter out all splits after this point. These splits immediately
        # follow another split.
        for action in func(*args, **kwargs):
            if not isinstance(action, MatmulSplitAction):
                yield action

    return wrapper_decorator


@_assert_stable_spec
def _common_move(
    op, attr_name: str, level: Optional[int], layout: Optional[Layout]
) -> "MoveLet":
    """Wraps a dataclass-based Schedule in a MoveLet moving one of its operands.

    This is the logic underpinning some ops' move_input actions.

    :param attr_name: The name of the field holding the operand to move.
    :param level: The level to which the operand should be moved, if not None.
    """
    operand: Union[Tensor, Tile] = getattr(op, attr_name)
    if level is None:
        level = operand.spec.level
    if layout is None:
        layout = operand.spec.layout
    assert level <= operand.root.level
    if level == operand.root.level and layout == operand.layout:
        raise ValueError("Either level or layout must differ from current")
    if operand.root.level == 0 and all(d == 1 for d in operand.dim_sizes):
        warnings.warn("There is no reason to move a single element in registers")
    new_mat = Tensor(
        spec=specs.TensorSpec(dim_sizes=operand.dim_sizes, layout=layout, level=level),
        name=None,
        origin=operand,
    )

    # Figure out the input index, if it's an input
    # TODO: Faster to have the caller pass the index.
    input_idx = None
    try:
        input_idx = op.inputs.index(operand)
    except ValueError:
        pass
    assert input_idx is not None or operand == op.output

    return MoveLet(
        source=operand,
        destination=new_mat,
        input_idx=input_idx,
        inner=dataclasses.replace(op, **{attr_name: new_mat}),
    )


def _move_arguments(operand: Union[Tile, Tensor]) -> Iterable[tuple[int, Layout]]:
    if operand.root.level > 1:
        raise NotImplementedError("Only two levels of memory tested")

    # There is no reason to move a single-element tensor already in registers.
    if operand.root.level == 0 and all(d == 1 for d in operand.dim_sizes):
        return

    # If the tensor has only one element, row-major is the only available
    # layout. Otherwise, all layouts are available.
    allowable_layouts = [specs.Layout.ROW_MAJOR]
    if any(d > 1 for d in operand.dim_sizes):
        allowable_layouts = list(specs.Layout)

    # If enabled and at level 1, allow relayouts in level 1.
    if ALLOW_ABOVE_REGISTERS_RELAYOUT and operand.root.level > 0:
        for layout in allowable_layouts:
            if layout != operand.layout:
                yield 1, layout

    # If the tensor is already in registers, it makes no sense to do a move
    # (really: a relayout) to its current layout.
    if operand.root.level == 0:
        try:
            allowable_layouts.remove(operand.layout)
        except ValueError:
            # operand.layout wasn't in allowable_layouts
            pass

    # Yield actions for movement with register file destination, which
    # includes relayouts in registers and movements from level 1 to RF.
    for layout in allowable_layouts:
        yield 0, layout


def _common_operand_move_actions(op_move_tuples):
    for inp_idx, operand, move_fn in op_move_tuples:
        for level, layout in _move_arguments(operand):
            yield MoveAction(move_fn, operand, inp_idx, level, layout)


def spec_to_hole(spec: specs.Spec, inputs: Tuple, output) -> "Schedule":
    """Returns a default, incomplete schedule for a Spec which consume given inputs.

    Output tensors will be constructed using the spec's output shape property.
    """
    if isinstance(spec, specs.Convolution):
        assert len(inputs) == 2, f"Expected 2 Tensor/Tile operands; got {len(inputs)}"
        return DirectConv(
            lhs=inputs[0], rhs=inputs[1], output=output, serial_only=spec.serial_only
        )
    elif isinstance(spec, specs.Matmul):
        assert len(inputs) == 2, f"Expected 2 Tensor/Tile operands; got {len(inputs)}"
        return Matmul(
            lhs=inputs[0], rhs=inputs[1], output=output, serial_only=spec.serial_only
        )
    elif isinstance(spec, specs.ReduceSum):
        assert len(inputs) == 1, f"Expected 1 Tensor/Tile operands; got {len(inputs)}"
        return ReduceSum(source=inputs[0], output=output, serial_only=spec.serial_only)
    elif isinstance(spec, specs.Compose):
        return ComposeHole(spec, inputs=inputs, output=output)
    else:
        raise NotImplementedError()


class Schedule(abc.ABC):
    @property
    @abc.abstractmethod
    def spec(self) -> specs.Spec:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def children(self) -> Tuple["Schedule", ...]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def output(self) -> Union[Tensor, Tile]:
        raise NotImplementedError()

    @property
    def operands(self) -> tuple[Union[Tensor, Tile], ...]:
        return self.inputs + (self.output,)

    @property
    def depth(self) -> int:
        # TODO: Relying on reflection is pretty brittle
        inners = []
        if hasattr(self, "inner"):
            inners.append(getattr(self, "inner"))
        return 1 + sum(s.depth for s in inners)

    @abc.abstractmethod
    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        raise NotImplementedError()

    @property
    def innermost(self) -> "Schedule":
        """Returns the innermost Schedule.

        This should be a Matmul. This property will not longer make sense once we've
        implemented schedules with multiple leaves.
        """
        if hasattr(self, "inner"):
            return getattr(self, "inner").innermost
        raise NotImplementedError()

    @property
    def is_scheduled(self) -> bool:
        if hasattr(self, "inner"):
            return getattr(self, "inner").is_scheduled
        raise NotImplementedError(f"No is_scheduled implementation for {type(self)}")

    @_assert_stable_spec
    def replace_leaf(self, replacement: "Schedule") -> "Schedule":
        if self.innermost == self:
            return replacement
        return dataclasses.replace(self, inner=self.inner.replace_leaf(replacement))

    @abc.abstractmethod
    def replace_children(self, replacements: Iterable["Schedule"]) -> "Schedule":
        raise NotImplementedError()

    @abc.abstractmethod
    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        raise NotImplementedError()

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], "Schedule"]]:
        raise NotImplementedError()

    @_assert_stable_spec
    def tile_out(self, output_shape: Tuple[int, ...], parallel=False) -> "Schedule":
        if parallel and self.spec.serial_only:
            raise ValueError("Serial-only Spec prevents parallel tiling")

        # If this is an Impl with a single child, just forward.
        if len(self.children) == 1:
            return self.replace_children(
                [self.children[0].tile_out(output_shape, parallel=parallel)]
            )

        # A no-op if the given shape is already the output shape.
        if self.output.dim_sizes == output_shape:
            return self

        # Tile the output first
        smaller_output = self.output.simple_tile(output_shape)
        assert smaller_output != self.output
        assert isinstance(smaller_output, SimpleTile)

        # Tile the corresponding inputs.
        smaller_inputs = tuple(
            partial_inp.tile(inp)
            for inp, partial_inp in zip(
                self.inputs,
                tiling.tile_out(
                    type(self.spec),
                    [inp.dim_sizes for inp in self.inputs],
                    tiling.tile_to_partial(smaller_output),
                ),
            )
        )

        # Make an inner hole for the now-smaller Spec.
        inner_serial = None
        if parallel:
            inner_serial = True
        inner = spec_to_hole(
            self.spec.shrink_for_tile_out(output_shape, serial_only=inner_serial),
            inputs=smaller_inputs,
            output=smaller_output,
        )

        # Construct the list of tiles, which is every tile_out result that just
        # returned itself because the tile would have been the same shape as the
        # original. (Note that the output, unlike the inputs, is known to have
        # been tiled or this method would have short-circuited.)
        # TODO: Can we share this code with same block in ComposeHole.tile_out?
        unchanged_input_tiles = set()
        for original_input, tiled_input in zip(self.inputs, smaller_inputs):
            if original_input != tiled_input:
                unchanged_input_tiles.add(tiled_input)
        unchanged_input_tiles = frozenset(unchanged_input_tiles)

        return Loop(
            driving_tile=smaller_output,
            dependent_tiles=frozenset(unchanged_input_tiles),
            inner=inner,
            parallel=parallel,
        )

    def split(self, k: int) -> "Schedule":
        raise NotImplementedError()

    @_assert_stable_spec
    def sliding_tile_out(
        self, sliding_dim: int, output_size: int, level: int
    ) -> "Schedule":
        """Like tile_out, without reloading overlapping image regions in one dimension."""
        # If this is an Impl with a single child, just forward.
        if len(self.children) == 1:
            return self.replace_children(
                [self.children[0].sliding_tile_out(sliding_dim, output_size, level)]
            )

        # A no-op if the given shape is already the output shape.
        output_shape = (
            self.output.dim_sizes[:sliding_dim]
            + (output_size,)
            + self.output.dim_sizes[sliding_dim + 1 :]
        )
        if output_shape == self.output.dim_sizes:
            return self

        # We produce a SlidingWindowLoop by first doing a normal tile_out,
        # then swapping an introduced input tile with a non-zero frontier
        # (i.e. a ConvolutionImageTile) for a sliding Tensor managed by the
        # resulting SlidingWindowLoop. This method doesn't yet support the
        # case that multiple introduced inputs have non-zero frontiers.
        impl = cast(Loop, self.tile_out(output_shape))
        assert isinstance(impl, Loop)
        assert impl.spec == self.spec
        assert impl.inputs == self.inputs
        assert impl.output == self.output

        # Sort the tiles into those to preserve and the one over which
        # to slide
        tile_to_convert = None
        other_tiles = []
        for tile in impl.tiles:
            front = tile.frontiers[sliding_dim]
            if front > 0:
                if tile_to_convert is not None:
                    raise Exception("Choice of input to slide is ambiguous")
                tile_to_convert = tile
            else:
                other_tiles.append(tile)
        assert isinstance(
            tile_to_convert, Tile
        ), f"tile_to_convert was unexpectedly a {type(tile_to_convert)}"

        # Turn tile_to_convert into a Tensor under management of the SlidingWindowLoop
        # and a Tile for calculating the update costs
        live_tensor = Tensor(
            specs.TensorSpec(
                tile_to_convert.dim_sizes, level=level, layout=tile_to_convert.layout
            ),
            name=None,
            origin=tile_to_convert.origin,
        )

        assert len(impl.inner.children) == 0, "Expected inner to be a leaf"
        inner = impl.inner.replace_io(
            inputs=(
                live_tensor if inp == tile_to_convert else inp
                for inp in impl.inner.inputs
            ),
            output=(
                live_tensor
                if impl.inner.output == tile_to_convert
                else impl.inner.output
            ),
        )

        return SlidingWindowLoop(
            inputs=self.inputs,
            output=self.output,
            live_tensor=live_tensor,
            frontier_size=tile_to_convert.frontiers[sliding_dim],
            other_tiles=tuple(other_tiles),
            spec=self.spec,
            inner=inner,
        )

    @_assert_stable_spec
    def split_filters(self, k: int) -> "Schedule":
        return dataclasses.replace(self, inner=self.inner.split_filters(k))

    @abc.abstractmethod
    def move_input(
        self,
        input_idx: int,
        level: Optional[int] = None,
        layout: Optional[Layout] = None,
    ) -> "Schedule":
        raise NotImplementedError()

    @abc.abstractmethod
    def move_output(
        self,
        level: Optional[int] = None,
        layout: Optional[Layout] = None,
    ) -> "Schedule":
        raise NotImplementedError()

    @_assert_stable_spec
    def subschedule(self, *args, **kwargs) -> "Schedule":
        if len(self.children) == 1:
            child = self.children[0]
            return self.replace_children([child.subschedule(*args, **kwargs)])
        raise NotImplementedError()

    @_assert_stable_spec
    def complete(self) -> "Schedule":
        return dataclasses.replace(self, inner=self.inner.complete())

    @property
    @abc.abstractmethod
    def additional_memories(self) -> List[List[int]]:
        """Memory costs of self when the corresponding child is executed.

        :returns: A list of amounts of memory to remove from that available. The
          outermost list has the same length as the  number of children in this
          Impl.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def peak_memory(self) -> List[int]:
        raise NotImplementedError()


# TODO: Use this everywhere tile_out actions are produced
def gen_tile_sizes(tensor_shape: Tuple[int, ...]) -> Iterable[Tuple[int, ...]]:
    """Returns tile shapes to explore for a given tensor shape."""
    if len(tensor_shape) == 0:
        return
    elif len(tensor_shape) == 1:
        for d in dim_range(tensor_shape[0]):
            yield (d,)
    else:
        for rest in gen_tile_sizes(tensor_shape[1:]):
            for d in dim_range(tensor_shape[0]):
                yield (d,) + rest


@dataclass_abc.dataclass_abc(frozen=True)
class ComposeHole(Schedule):
    spec: specs.Compose
    inputs: Tuple[Union[Tensor, Tile], ...]
    output: Union[Tensor, Tile]

    def __post_init__(self):
        assert isinstance(self.spec, specs.Compose)
        assert self.spec.inputs == tuple(inp.spec for inp in self.inputs)
        assert self.spec.output == self.output.spec

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        if fancy:
            return termcolor.colored("??compose", attrs=["bold"])
        else:
            return "??compose"

    @property
    def children(self) -> Tuple["Schedule", ...]:
        return tuple()

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        inputs = tuple(inputs)
        new_spec = self.spec.replace_io(tuple(inp.spec for inp in inputs), output.spec)
        return ComposeHole(new_spec, inputs, output)

    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Schedule]]:
        # TODO: Remove this symmetry: lots of ways to iteratively split the pipeline
        # TODO: Reintroduce splitting on non-index 0
        for level in range(len(system_config.DEFAULT_SYSTEM_CONFIG.level_configs)):
            for layout in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
                yield PeelAction(self.peel, level=level, layout=layout)

        yield from _common_operand_move_actions(
            [
                (i, inp, functools.partial(self.move_input, i))
                for i, inp in enumerate(self.inputs)
            ]
            + [(None, self.output, self.move_output)]
        )

        for tile in gen_tile_sizes(self.output.dim_sizes):
            if tile != self.output.dim_sizes:
                for parallel in [False] if self.spec.serial_only else [True, False]:
                    yield TileOutAction(self.tile_out, tile, parallel)

        # TODO: Don't just use the first input. This isn't general to arbitrary
        # slideable ops in a Pipeline, or even, for instance, if we swapped the
        # operand order of Convolution/DirectConv. Besides, the dimensions
        # passed to sliding_tile_out are over the *output* Tile, not either
        # input; this only works because they're one-to-one for DirectConv.
        if allow_sliding_windows.get():
            first_head_input = self.inputs[
                -self.spec.subspec_classes[-1].inputs_count()
            ]
            for sliding_dim in range(len(first_head_input.dim_sizes)):
                for slide_size in dim_range(
                    self.output.dim_sizes[sliding_dim], include_end=False
                ):
                    # TODO: Handle choices of level in a way that generalizes to more than 2 levels
                    yield SlidingTileOutAction(
                        self.sliding_tile_out, sliding_dim, slide_size, level=0
                    )

        # TODO: This is awful. Produce a real interface for both deferring to
        #   inner split and for gathering the right split.
        if allow_reduce_splits.get():
            for k in self.split_sizes():
                for parallel in [False] if self.spec.serial_only else [True, False]:
                    yield functools.partial(self.split, k, parallel=parallel)

    @_assert_stable_spec
    def move_input(
        self,
        input_idx: int,
        level: Optional[int] = None,
        layout: Optional[Layout] = None,
    ):
        operand = self.inputs[input_idx]
        if level is None:
            level = operand.root.level
        if layout is None:
            layout = operand.layout
        assert level >= 0
        if level == operand.root.level and layout == operand.layout:
            raise ValueError("Either level or layout must differ from current")
        new_mat = Tensor(
            spec=specs.TensorSpec(
                dim_sizes=operand.dim_sizes, layout=layout, level=level
            ),
            name=None,
            origin=operand,
        )

        new_inputs = self.inputs[:input_idx] + (new_mat,) + self.inputs[input_idx + 1 :]
        new_inner_spec = dataclasses.replace(
            self.spec, inputs=tuple(inp.spec for inp in new_inputs)
        )
        return MoveLet(
            source=operand,
            destination=new_mat,
            input_idx=input_idx,
            inner=dataclasses.replace(self, spec=new_inner_spec, inputs=new_inputs),
        )

    @_assert_stable_spec
    def move_output(
        self, level: Optional[int] = None, layout: Optional[Layout] = None
    ) -> "Schedule":
        operand = self.output
        if level is None:
            level = operand.root.level
        if layout is None:
            layout = operand.layout
        assert level >= 0
        if level == operand.root.level and layout == operand.layout:
            raise ValueError("Either level or layout must differ from current")
        new_mat = Tensor(
            spec=specs.TensorSpec(
                dim_sizes=operand.dim_sizes, layout=layout, level=level
            ),
            name=None,
            origin=operand,
        )

        new_inner_spec = dataclasses.replace(self.spec, output=new_mat.spec)
        return MoveLet(
            source=operand,
            destination=new_mat,
            input_idx=None,
            inner=dataclasses.replace(self, spec=new_inner_spec, output=new_mat),
        )

    @_assert_stable_spec
    def peel(
        self, level: Optional[int] = None, layout: Optional[Layout] = None
    ) -> Schedule:
        if level is None or layout is None:
            raise NotImplementedError("Auto-selecting level and layout unimplemented")

        # TODO: Using ALPHABET_PRODUCT here will fail for long programs
        intermediate_tensor_layout = layout
        if all(d == 1 for d in self.spec.intermediate_shapes[0]):
            intermediate_tensor_layout = Layout.ROW_MAJOR
        intermediate_tensor = Tensor(
            specs.TensorSpec(
                dim_sizes=self.spec.intermediate_shapes[0],
                level=level,
                layout=intermediate_tensor_layout,
            ),
            name="buf"
            + utils.ALPHABET_PRODUCT[len(self.spec.subspec_classes) - 2].upper(),
        )

        # The head of a Compose corresponds to the last function evaluated
        head_inps = (intermediate_tensor,)
        hi = self.spec.subspec_classes[0].inputs_count() - 1
        if hi:
            head_inps += self.inputs[:hi]
        head_hole = spec_to_hole(
            self.spec.subspec_classes[0].from_io(
                tuple(t.spec for t in head_inps),
                self.output.spec,
                serial_only=self.spec.serial_only,
            ),
            inputs=head_inps,
            output=self.output,
        )

        if len(self.spec.subspec_classes) == 2:
            remainder = spec_to_hole(
                self.spec.subspec_classes[1].from_io(
                    tuple(t.spec for t in self.inputs[hi:]),
                    intermediate_tensor.spec,
                    serial_only=self.spec.serial_only,
                ),
                inputs=self.inputs[hi:],
                output=intermediate_tensor,
            )
        else:
            remainder = ComposeHole(
                specs.Compose(
                    subspec_classes=self.spec.subspec_classes[1:],
                    inputs=tuple(t.spec for t in self.inputs[hi:]),
                    output=intermediate_tensor.spec,
                    serial_only=self.spec.serial_only,
                ),
                inputs=self.inputs[hi:],
                # output=self.inputs[1][-1],
                output=intermediate_tensor,
            )
        return Pipeline((remainder, head_hole))

    @_assert_stable_spec
    def tile_out(self, output_shape: tuple[int, ...], parallel=False) -> Schedule:
        if parallel and self.spec.serial_only:
            raise ValueError("Serial-only Spec prevents parallel tiling")

        # A no-op if the given shape is already the output shape.
        if self.output.dim_sizes == output_shape:
            return self

        # First, tile self.output.
        shrunken_output_tile = self.output.simple_tile(output_shape)
        assert isinstance(shrunken_output_tile, SimpleTile)
        assert shrunken_output_tile != self.output

        # Compute new, reified Tiles for the shrunken ComposeHole. Works by computing
        # PartialTiles for the smaller ComposeHole, then applying those to self.inputs,
        # starting with the new output tile.
        reified_inputs = tuple(
            partial_inp.tile(inp)
            for inp, partial_inp in zip(
                self.inputs, self._compute_partial_inputs(shrunken_output_tile)
            )
        )

        # Construct the spec for the smaller ComposeHole
        return Loop(
            driving_tile=shrunken_output_tile,
            dependent_tiles=frozenset(self._filter_unchanged_inputs(reified_inputs)),
            inner=ComposeHole(
                specs.Compose(
                    subspec_classes=self.spec.subspec_classes,
                    inputs=tuple(inp.spec for inp in reified_inputs),
                    output=shrunken_output_tile.spec,
                    serial_only=(parallel or self.spec.serial_only),
                ),
                inputs=reified_inputs,
                output=shrunken_output_tile,
            ),
            parallel=parallel,
        )

    @_assert_stable_spec
    def split(self, k: int, parallel=False) -> Schedule:
        # TODO: Can we abstract over both Matmul and Reduce' splits
        if self.spec.subspec_classes[0] == specs.ReduceSum:
            return self._split_reduce_head(k, parallel=parallel)
        else:
            raise SplitNotSupportedByHeadError()

    def split_sizes(self) -> Iterable[int]:
        if self.spec.subspec_classes[0] == specs.ReduceSum:
            # TODO: This should defer to the inner op
            for k in dim_range(self.spec.intermediate_shapes[0][-1], include_end=False):
                if k != self.spec.intermediate_shapes[0][-1]:
                    yield k
        else:
            return

    def _split_reduce_head(self, k: int, parallel: bool) -> Schedule:
        assert self.spec.subspec_classes[0] == specs.ReduceSum

        if parallel and self.spec.serial_only:
            raise ValueError("Serial-only Spec prevents parallel tiling")

        # A no-op if `k` is already the max size.
        orig_reduce_input_shape: tuple[int, ...] = self.spec.intermediate_shapes[0]
        if k == orig_reduce_input_shape[-1]:
            return self

        # Make a PartialTile corresponding to the output of the tail's head (i.e.
        # the composed input to the Reduce). Use it to produce Tiles corresponding to
        # our new dependencies on only parts of the input.
        smaller_partial_input_tile = tiling.PartialSimpleTile(
            dim_sizes=orig_reduce_input_shape[:-1] + (k,)
        )
        reified_inputs = tuple(
            partial_inp.tile(inp)
            for inp, partial_inp in zip(
                self.inputs,
                self._compute_partial_inputs(smaller_partial_input_tile, skip_first=1),
            )
        )

        filtered_reified_inputs = list(self._filter_unchanged_inputs(reified_inputs))

        # Select the driving tile by just selecting the inputs with the expected number
        # of steps.
        # TODO: This is extremely ad-hoc. We need a solution for arbitrary accumulating
        #  loops. Fix this.
        expected_steps = math.ceil(orig_reduce_input_shape[-1] / k)
        driving_tile = None
        for inp in filtered_reified_inputs:
            if inp.steps == expected_steps:
                driving_tile = inp
                break
        assert driving_tile, f"No tile had expected number of steps: {expected_steps}"

        # Build the loop
        new_inner = ComposeHole(
            specs.Compose(
                subspec_classes=self.spec.subspec_classes,
                inputs=tuple(inp.spec for inp in reified_inputs),
                output=self.output.spec,
                serial_only=(parallel or self.spec.serial_only),
            ),
            inputs=reified_inputs,
            output=self.output,
        )
        return Loop(
            driving_tile=driving_tile,
            dependent_tiles=frozenset(
                t for t in filtered_reified_inputs if t is not driving_tile
            ),
            inner=new_inner,
            parallel=parallel,
        )

    def _filter_unchanged_inputs(
        self, source: Iterable[Union[Tensor, Tile]]
    ) -> Iterable[Union[Tensor, Tile]]:
        for original_input, tiled_input in zip(self.inputs, source):
            if original_input != tiled_input:
                yield tiled_input

    def _compute_partial_inputs(
        self,
        output_tile: Union[SimpleTile, tiling.PartialSimpleTile],
        skip_first: int = 0,
    ) -> list[tiling.PartialTile]:
        """Returns PartialTiles for this ComposeHole's inputs for an output tiling.

        Accepts either a PartialTile or a Tile which will be converted into a PartialTile.
        """
        subspec_classes = list(self.spec.subspec_classes)
        intermediate_shapes = list(self.spec.intermediate_shapes)
        inputs = list(self.inputs)
        while skip_first > 0:
            popped_cls = subspec_classes.pop(0)
            intermediate_shapes.pop(0)
            if subspec_classes:
                inputs = inputs[popped_cls.inputs_count() - 1 :]
            else:
                inputs = inputs[popped_cls.inputs_count() :]
            skip_first -= 1
        return ComposeHole._compute_partial_inputs_inner(
            tuple(subspec_classes),
            tuple(intermediate_shapes),
            tuple(inputs),
            output_tile,
        )

    @staticmethod
    def _compute_partial_inputs_inner(
        subspec_classes: tuple,
        intermediate_shapes: Iterable[tuple[int, ...]],
        inputs: tuple[Union[Tensor, Tile], ...],
        output_tile: Union[SimpleTile, tiling.PartialSimpleTile],
    ) -> list[tiling.PartialTile]:
        if isinstance(output_tile, Tile):
            output_tile = tiling.tile_to_partial(output_tile)
        assert isinstance(output_tile, tiling.PartialSimpleTile)

        # We would normally need to do a forward pass first to produce the
        # output shapes so we know what the non-final subspecs' first operand
        # shapes are, but this is already implemented by the subspec_outputs
        # property of Compose.
        subspec_output_shapes = list(intermediate_shapes)

        input_tiles: Optional[tuple[tiling.PartialTile, ...]] = None
        all_input_tiles: list[tiling.PartialTile] = []
        flattened_inputs_shapes = tuple(inp.dim_sizes for inp in inputs)
        for idx, subspec_cls in enumerate(subspec_classes):
            inputs_shapes = ()
            if subspec_output_shapes:
                inputs_shapes = (subspec_output_shapes.pop(0),)
            take = subspec_cls.inputs_count() - len(inputs_shapes)
            inputs_shapes += flattened_inputs_shapes[:take]
            flattened_inputs_shapes = flattened_inputs_shapes[take:]
            # We're tracing the type and shape of each subspec's first tile up through the
            # pipeline of composed functions, so store the first into output_tile, which will
            # be an input the next time around the loop. At the end, we'll want input_tiles.
            input_tiles = tiling.tile_out(subspec_cls, inputs_shapes, output_tile)
            if idx == len(subspec_classes) - 1:
                all_input_tiles.extend(input_tiles)
            else:
                all_input_tiles.extend(input_tiles[1:])
            # Because Compose applies the output of a stage to the following stage's
            # first argument, we carry the first input tile into the next iteration.
            output_tile = input_tiles[0]
        assert input_tiles is not None and len(all_input_tiles) == len(inputs)

        return all_input_tiles

    @_assert_stable_spec
    def complete(self) -> "Schedule":
        level_count = len(system_config.DEFAULT_SYSTEM_CONFIG.level_configs)
        return self.peel(level=level_count - 1, layout=Layout.ROW_MAJOR).complete()

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        if replacements:
            raise Exception("Holes have no children to replace")
        return self

    @property
    def additional_memories(self) -> List[List[int]]:
        return [[0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]]

    @property
    def peak_memory(self) -> List[int]:
        return [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]

    @property
    def is_scheduled(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True, init=False)
class Pipeline(Schedule):
    """A sequence of schedules, called stages, which are executed in order.

    The output of the pipeline is the output of the final stage.
    """

    stages: Tuple[Schedule, ...]

    def __init__(self, stages: Tuple[Schedule, ...]):
        assert len(stages) >= 2
        # TODO: Reintroduce check for operand agreement
        # for before, after in zip(self.stages[:-1], self.stages[1:]):
        #     assert (
        #         before.output == after.operands[0]
        #     ), f"Output of {before} didn't match first operand of {after}"

        # Flatten any immediate Pipeline children
        flattened_stages: List[Schedule] = []
        for stage in stages:
            if isinstance(stage, Pipeline):
                flattened_stages.extend(stage.stages)
            else:
                flattened_stages.append(stage)
        object.__setattr__(self, "stages", tuple(flattened_stages))

        assert all(
            self.stages[0].spec.serial_only == s.spec.serial_only for s in self.stages
        ), "All Pipeline stages should have the same serial_only flag"

    @functools.cached_property
    def spec(self) -> specs.Compose:
        subspec_classes = []
        inputs = tuple()
        for i, stage in enumerate(self.stages):
            if isinstance(stage.spec, specs.Compose):
                subspec_classes = list(stage.spec.subspec_classes) + subspec_classes
            else:
                subspec_classes.insert(0, type(stage.spec))
            inputs = tuple(stage.spec.inputs) + inputs
            if i > 0:
                inputs = inputs[1:]
        output = self.stages[-1].spec.output
        return specs.Compose(
            tuple(subspec_classes),
            inputs=inputs,
            output=output,
            serial_only=self.serial_only,
        )

    @property
    def serial_only(self) -> bool:
        # The following is correct because all stages have the same
        # serial_only flag
        return self.stages[0].spec.serial_only

    @property
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        flattened_inputs = list(self.stages[0].inputs)
        for stage in self.stages[1:]:
            flattened_inputs = list(stage.inputs[1:]) + flattened_inputs
        assert len(flattened_inputs) == len(self.spec.inputs)
        return tuple(flattened_inputs)

    @property
    def output(self) -> Union[Tensor, Tile]:
        return self.stages[-1].output

    @property
    def children(self) -> Tuple["Schedule", ...]:
        return self.stages

    @_assert_stable_spec
    def subschedule(self, idx: int, fn: Callable[[Schedule], Schedule]) -> "Pipeline":
        """Transform the stage at a given index by applying the given function."""
        new_stages = list(self.stages)
        new_stages[idx] = fn(new_stages[idx])
        return dataclasses.replace(self, stages=tuple(new_stages))

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ) -> str:
        keyword = "pipeline"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])

        introduced_strs = []
        for stage in self.stages[:-1]:
            introduced_strs.append(f"{name_tensor_fn(stage.output)}: {stage.output}")
        return f"{keyword} ({', '.join(introduced_strs)})"

    @property
    def depth(self) -> int:
        return 1 + max(stage.depth for stage in self.stages)

    def move_input(
        self, input_idx: int, level: Optional[int], layout: Optional[Layout]
    ) -> "Schedule":
        raise NotImplementedError(
            "move_input should usually be called on ComposeHole, not Pipeline"
        )

    def move_output(self, level: Optional[int], layout: Optional[Layout]) -> "Schedule":
        raise NotImplementedError(
            "move_output should usually be called on ComposeHole, not Pipeline"
        )

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Schedule]]:
        raise NotImplementedError("Pipeline has no actions because it is never a leaf")

    @_assert_stable_spec
    def complete(self) -> Schedule:
        return dataclasses.replace(
            self, stages=tuple(s.complete() for s in self.stages)
        )

    @_assert_stable_spec
    def replace_leaf(self, replacement: Schedule) -> Schedule:
        raise NotImplementedError("replace_leaf ambiguous for Pipeline")

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        replacements = tuple(replacements)
        if len(replacements) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} replacement children, but "
                f"got {len(replacements)}"
            )
        for original, replacement in zip(self.stages, replacements):
            if original.spec != replacement.spec:
                raise ValueError(
                    f"Cannot replace {original.spec} with {replacement.spec}; "
                    "specs differ"
                )
        return dataclasses.replace(self, stages=replacements)

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        raise NotImplementedError()

    @property
    def additional_memories(self) -> List[List[int]]:
        # Initialize peaks to dependencies of the first stage, which is just its
        # output
        first_peak = [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]
        first_peak[self.stages[0].output.level] = self.stages[0].output.volume

        middle_peaks: List[List[int]] = []
        for stage_idx in range(1, len(self.stages) - 1):
            before = self.stages[stage_idx - 1].output
            after = self.stages[stage_idx].output
            stage_mem = [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]
            stage_mem[before.level] += before.volume
            stage_mem[after.level] += after.volume
            middle_peaks.append(stage_mem)

        last_peak = [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]
        last_peak[self.stages[-2].output.level] = self.stages[-2].output.volume

        return [first_peak] + middle_peaks + [last_peak]

    @property
    def peak_memory(self) -> List[int]:
        # Pipeline currently adds an intermediate tensor between each stage, so
        # intermediates is just the output of everything but the last stage
        intermediates = [o.output for o in self.stages[:-1]]
        intermed_utils: List[List[int]] = []
        for tensor in intermediates:
            assert isinstance(tensor, Tensor)
            new_mem = [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]
            new_mem[tensor.level] += tensor.volume
            intermed_utils.append(new_mem)

        # Utilization is the memory used by an operand and, where present, input and
        # output intermediate buffers
        mem = _zipply(sum, self.stages[0].peak_memory, intermed_utils[0])
        for stage_idx in range(1, len(self.stages) - 1):
            mem = _zipply(
                max,
                mem,
                _zipply(
                    sum,
                    intermed_utils[stage_idx - 1],
                    self.stages[stage_idx].peak_memory,
                    intermed_utils[stage_idx],
                ),
            )
        mem = _zipply(
            max, mem, _zipply(sum, self.stages[-1].peak_memory, intermed_utils[-1])
        )
        return mem

    @property
    def is_scheduled(self) -> bool:
        return all(op.is_scheduled for op in self.stages)


@dataclass_abc.dataclass_abc(frozen=True)
class Matmul(Schedule):

    lhs: Union[Tensor, Tile]  # n-by-m
    rhs: Union[Tensor, Tile]  # m-by-p
    output: Union[Tensor, Tile]  # m-by-n
    serial_only: bool

    def __post_init__(self):
        lw, rh = self.lhs.width, self.rhs.height
        assert lw == rh, f"Inner dims. of Matmul operands don't match: {lw} and {rh}"

    @property
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        return self.lhs, self.rhs

    @functools.cached_property
    def spec(self) -> specs.Matmul:
        return specs.Matmul(
            self.lhs.spec, self.rhs.spec, self.output.spec, serial_only=self.serial_only
        )

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        return (
            f"Matmul({name_tensor_fn(self.lhs)}, "
            f"{name_tensor_fn(self.rhs)}, "
            f"{name_tensor_fn(self.output)})"
        )

    @property
    def children(self) -> Tuple["Schedule", ...]:
        return tuple()

    @property
    def innermost(self) -> "Schedule":
        return self

    @property
    def is_scheduled(self) -> bool:
        return all(
            o.root.in_registers and functools.reduce(mul, o.dim_sizes, 1) == 1
            for o in [self.lhs, self.rhs, self.output]
        )

    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Schedule]]:
        # Search only over full line sizes
        for h, w in itertools.product(
            dim_range(self.output.dim_sizes[0]), dim_range(self.output.dim_sizes[1])
        ):
            if self.output.dim_sizes != (h, w):
                for parallel in [False] if self.serial_only else [True, False]:
                    yield TileOutAction(self.tile_out, (h, w), parallel)

        if self.lhs.dim_sizes[1] > 1:
            for k in dim_range(self.lhs.dim_sizes[1], include_end=False):
                yield MatmulSplitAction(self.split, k=k)

        yield from _common_operand_move_actions(
            [
                (i, inp, functools.partial(self.move_input, i))
                for i, inp in enumerate(self.inputs)
            ]
            + [(None, self.output, self.move_output)]
        )

    def move_input(
        self,
        input_idx: int,
        level: Optional[int] = None,
        layout: Optional[Layout] = None,
    ) -> "MoveLet":
        if input_idx == 0:
            return _common_move(self, "lhs", level, layout)
        elif input_idx == 1:
            return _common_move(self, "rhs", level, layout)
        else:
            raise ValueError("input_idx must be 0 or 1")

    def move_output(
        self, level: Optional[int] = None, layout: Optional[Layout] = None
    ) -> "MoveLet":
        return _common_move(self, "output", level, layout)

    @_assert_stable_spec
    def split(self, size: int) -> "Schedule":
        assert size > 0
        if size > self.lhs.dim_sizes[1]:
            raise ValueError(
                f"Cannot split {size} with inner dim. {self.lhs.dim_sizes[1]}"
            )
        if size == self.lhs.dim_sizes[1]:
            return self
        left_view = self.lhs.simple_tile((self.lhs.height, size))
        right_view = self.rhs.simple_tile((size, self.rhs.width))
        warnings.warn("Not yet specializing spec for split Matmuls")
        return MatmulSplitLoop(
            lhs=self.lhs,
            rhs=self.rhs,
            output=self.output,
            inner=Matmul(left_view, right_view, self.output, self.serial_only),
        )

    @_assert_stable_spec
    def complete(self) -> Schedule:
        if self.lhs.height > 1 or self.rhs.width > 1:
            return self.tile_out((1, 1)).complete()
        if self.lhs.width > 1:
            return self.split(1).complete()
        if not self.lhs.root.in_registers:
            return self.move_input(0, level=self.lhs.root.level - 1).complete()
        if not self.rhs.root.in_registers:
            return self.move_input(1, level=self.rhs.root.level - 1).complete()
        if not self.output.root.in_registers:
            return self.move_output(level=self.output.root.level - 1).complete()
        return self

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        if replacements:
            raise Exception("Matmul has no children to replace")
        return self

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        lhs, rhs = inputs
        return Matmul(lhs, rhs, output)

    @property
    def additional_memories(self) -> List[List[int]]:
        return [[0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]]

    @property
    def peak_memory(self) -> List[int]:
        return [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]


# noinspection PyTypeChecker, PyArgumentList, PyUnresolvedReferences
@dataclass_abc.dataclass_abc(frozen=True)
class DirectConv(Schedule):
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
            1 + self.lhs.height - self.rhs.dim_sizes[0],
            1 + self.lhs.width - self.rhs.dim_sizes[1],
            self.rhs.dim_sizes[2],
        )

    @property
    def children(self) -> Tuple["Schedule", ...]:
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
        return self.lhs.height

    @property
    def image_width(self):
        return self.lhs.width

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
        if not all(op.root.in_registers for op in self.operands):
            return False
        if any(d > 1 for d in self.output.dim_sizes):
            return False
        return True

    @property
    def innermost(self) -> "Schedule":
        return self

    # @_assert_stable_spec
    # def tile_out(self, output_shape: Tuple[int, ...]) -> "Schedule":
    #     # TODO: DirectConv acts as though it has a rank-2 output because of
    #     #   split_filters. Fix this.
    #     return super().tile_out(output_shape + (self.rhs.dim_sizes[-1],))

    # TODO: Remove split_filters
    @_assert_stable_spec
    def split_filters(self, k: int) -> "Schedule":
        return self.tile_out(self.output.dim_sizes[:-1] + (k,))

    def move_input(
        self,
        input_idx: int,
        level: Optional[int] = None,
        layout: Optional[Layout] = None,
    ) -> "MoveLet":
        if input_idx == 0:
            return _common_move(self, "lhs", level, layout)
        elif input_idx == 1:
            return _common_move(self, "rhs", level, layout)
        else:
            raise ValueError("input_idx must be 0 or 1")

    def move_output(
        self, level: Optional[int] = None, layout: Optional[Layout] = None
    ) -> "MoveLet":
        return _common_move(self, "output", level, layout)

    @_assert_stable_spec
    def split(self, size: int) -> "Schedule":
        raise NotImplementedError("Split not implemented for DirectConv")

    @_assert_stable_spec
    def complete(self) -> Schedule:
        if any(d > 1 for d in self.output.dim_sizes):
            return self.tile_out((1, 1, 1)).complete()
        if not self.lhs.root.in_registers:
            return self.move_input(0, level=self.lhs.root.level - 1).complete()
        if not self.rhs.root.in_registers:
            return self.move_input(1, level=self.rhs.root.level - 1).complete()
        if not self.output.root.in_registers:
            return self.move_output(level=self.output.root.level - 1).complete()
        return self

    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Schedule]]:
        # Yield .tile_outs and .sliding_tile_outs
        for h, w, c, parallel in itertools.product(
            dim_range(self.output.dim_sizes[0]),
            dim_range(self.output.dim_sizes[1]),
            dim_range(self.output.dim_sizes[2]),
            [False] if self.serial_only else [True, False],
        ):
            if self.output.dim_sizes != (h, w, c):
                yield TileOutAction(self.tile_out, (h, w, c), parallel)

        # We only need the levels for the left-hand side (images), because that
        # is the only operand over which one can slide.
        if allow_sliding_windows.get():
            for level in set(l for l, _ in _move_arguments(self.lhs)):
                for sliding_dim in [0, 1]:
                    for slide_size in dim_range(
                        self.output.dim_sizes[sliding_dim], include_end=False
                    ):
                        yield SlidingTileOutAction(
                            self.sliding_tile_out, sliding_dim, slide_size, level
                        )

        # Search over all possible filters splits
        # TODO: We don't need a sep. split_filters. Should be just a dim for tile_out!
        # if self.rhs.dim_sizes[-1] > 1:
        #     for k in range(1, self.rhs.dim_sizes[-1]):
        #         yield functools.partial(self.split_filters, k)

        yield from _common_operand_move_actions(
            [
                (i, inp, functools.partial(self.move_input, i))
                for i, inp in enumerate(self.inputs)
            ]
            + [(None, self.output, self.move_output)]
        )

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        if list(replacements):
            raise ValueError("DirectConv has no children to replace")
        return self

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        lhs, rhs = inputs
        return DirectConv(lhs, rhs, output, serial_only=self.serial_only)

    @property
    def additional_memories(self) -> List[List[int]]:
        return [[0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]]

    @property
    def peak_memory(self) -> List[int]:
        return [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]


@dataclass_abc.dataclass_abc(frozen=True)
class ReduceSum(Schedule):

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
    def children(self) -> Tuple["Schedule", ...]:
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
    def innermost(self) -> Schedule:
        return self

    @property
    def is_scheduled(self) -> bool:
        return (
            self.source.root.in_registers
            and self.output.root.in_registers
            and all(d == 1 for d in self.source.dim_sizes)
        )

    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Schedule]]:
        # TODO: Lots of code duplication in this method
        # Search only over full line sizes
        for ds in itertools.product(*[dim_range(d) for d in self.output.dim_sizes]):
            if self.output.dim_sizes != ds:
                for parallel in [False] if self.serial_only else [True, False]:
                    yield TileOutAction(self.tile_out, ds, parallel)

        # Split over the reduction dimension
        if allow_reduce_splits.get():
            for k in dim_range(self.source.dim_sizes[-1], include_end=False):
                if k != self.source.dim_sizes[-1]:
                    yield functools.partial(self.split, k)

        yield from _common_operand_move_actions(
            [
                (0, self.source, functools.partial(self.move_input, 0)),
                (None, self.output, self.move_output),
            ]
        )

    def move_input(
        self,
        input_idx: int,
        level: Optional[int] = None,
        layout: Optional[Layout] = None,
    ) -> "MoveLet":
        if input_idx == 0:
            return _common_move(self, "source", level, layout)
        else:
            raise ValueError("input_idx must be 0 ")

    def move_output(
        self, level: Optional[int] = None, layout: Optional[Layout] = None
    ) -> Schedule:
        return _common_move(self, "output", level, layout)

    @_assert_stable_spec
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

    @_assert_stable_spec
    def complete(self) -> Schedule:
        if any(d > 1 for d in self.output.dim_sizes):
            return self.tile_out(tuple(1 for _ in self.output.dim_sizes)).complete()
        if self.source.dim_sizes[-1] > 1:
            return self.split(1).complete()
        if not self.source.root.in_registers:
            return self.move_input(0, level=self.source.root.level - 1).complete()
        if not self.output.root.in_registers:
            return self.move_output(level=self.output.root.level - 1).complete()
        return self

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        if replacements:
            raise Exception("Reduce has no children to replace")
        return self

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        inputs = list(inputs)
        if len(inputs) != 1:
            raise ValueError("Expected 1 input")
        return ReduceSum(inputs[0], output, serial_only=self.serial_only)

    @property
    def additional_memories(self) -> List[List[int]]:
        return [[0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]]

    @property
    def peak_memory(self) -> List[int]:
        return [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]


@dataclass_abc.dataclass_abc(frozen=True)
class Loop(Schedule):
    driving_tile: Tile
    dependent_tiles: frozenset[Tile]
    inner: Schedule
    parallel: bool

    def __post_init__(self):
        assert isinstance(self.driving_tile, Tile)
        assert all(isinstance(t, Tile) for t in self.dependent_tiles)
        if not self.tiles:
            raise ValueError("tiles is empty")
        # TODO: Assert that all tiles are used by inner's inputs or output
        if self.parallel and not self.inner.spec.serial_only:
            raise ValueError("Parallel loop's child must be serial only")

    @property
    def tiles(self) -> frozenset[Tile]:
        return frozenset([self.driving_tile]) | self.dependent_tiles

    @property
    def spec(self) -> specs.Spec:
        serial_only = self.inner.spec.serial_only
        if self.parallel:
            serial_only = False
        return self.inner.spec.replace_io(
            inputs=tuple(inp.spec for inp in self.inputs),
            output=self.output.spec,
            serial_only=serial_only,
        )

    @property
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
        for tile_set in [self.dependent_tiles, {self.driving_tile}]:
            for it in sorted(tile_set, key=str):
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
        return self.driving_tile.steps

    def steps_subscript(
        self, subscript, concrete_outer_size: Optional[int] = None
    ) -> int:
        driving_tile_idx = self.inner.operands.index(self.driving_tile)
        dim = self.spec.operands_dim_subscripts()[driving_tile_idx].index(subscript)
        return self.driving_tile.steps_dim(dim, concrete_outer_size)

    def boundary_size(
        self, subscript, concrete_outer_size: Optional[int] = None
    ) -> int:
        driving_tile_idx = self.inner.operands.index(self.driving_tile)
        dim = self.spec.operands_dim_subscripts()[driving_tile_idx].index(subscript)
        return self.driving_tile.boundary_size(dim, concrete_outer_size)

    @property
    def additional_memories(self) -> List[List[int]]:
        return [[0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]]

    @property
    def peak_memory(self) -> List[int]:
        return self.inner.peak_memory

    @property
    def children(self) -> Tuple[Schedule, ...]:
        return (self.inner,)

    def move_input(self, *args, **kwargs) -> "Loop":
        # Pass move_input through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_input(*args, **kwargs))

    def move_output(self, *args, **kwargs) -> "Loop":
        # Pass move_output through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_output(*args, **kwargs))

    @_assert_stable_spec
    def split(self, size: int) -> "Loop":
        # Pass split through to the inner schedule. This method is
        # sugar for calling subschedule.
        assert hasattr(self.inner, "split")
        return self.subschedule(lambda i: i.split(size))

    @_assert_stable_spec
    def split_filters(self, size: int) -> "Loop":
        # Pass split_filters through to the inner schedule.
        assert hasattr(self.inner, "split")
        return self.subschedule(lambda i: i.split_filters(size))

    @_assert_stable_spec
    def peel(self, *args, **kwargs) -> "Loop":
        # Pass split through to the inner schedule. This method is
        # sugar for calling subschedule.
        assert hasattr(self.inner, "peel")
        return dataclasses.replace(self, inner=self.inner.peel(*args, **kwargs))

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        replacements = tuple(replacements)
        if len(replacements) != 1:
            raise ValueError(f"One replacement child expected; got {len(replacements)}")
        if replacements[0].spec != self.inner.spec:
            raise ValueError(f"Expected a replacement with spec {self.inner.spec}")
        return dataclasses.replace(self, inner=replacements[0])

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        raise NotImplementedError()

    @_assert_stable_spec
    def subschedule(self, fn: Callable[["Schedule"], "Schedule"]) -> "Schedule":
        return self.replace_children([fn(self.inner)])


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
    def additional_memories(self) -> List[List[int]]:
        return [[0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]]

    @property
    def peak_memory(self) -> List[int]:
        return self.inner.peak_memory

    @property
    def children(self) -> Tuple[Schedule, ...]:
        return (self.inner,)

    def move_input(self, *args, **kwargs) -> "Schedule":
        # Pass move_input through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_input(*args, **kwargs))

    def move_output(self, *args, **kwargs) -> "Schedule":
        # Pass move_output through to the inner schedule. This method is
        # basically sugar for calling subschedule.
        return dataclasses.replace(self, inner=self.inner.move_output(*args, **kwargs))

    @_assert_stable_spec
    def split(self, size: int) -> "Schedule":
        return dataclasses.replace(self, inner=self.inner.split(size))

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        replacements = tuple(replacements)
        if len(replacements) != 1:
            raise ValueError(f"One replacement child expected; got {len(replacements)}")
        if replacements[0].spec != self.inner.spec:
            raise ValueError(f"Expected a replacement with spec {self.inner.spec}")
        return dataclasses.replace(self, inner=replacements[0])

    @_assert_stable_spec
    def subschedule(self, fn: Callable[["Schedule"], "Schedule"]) -> "Schedule":
        return self.replace_children([fn(self.inner)])


@dataclass_abc.dataclass_abc(frozen=True)
class SlidingWindowLoop(_TilingMixin, Schedule):
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
    inner: Schedule

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

    @_assert_stable_spec
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
    ) -> "Schedule":
        raise NotImplementedError()

    @property
    def peak_memory(self) -> List[int]:
        inner_peak_memory = list(self.inner.peak_memory)
        inner_peak_memory[self.live_tensor.level] += self.live_tensor.volume
        return inner_peak_memory

    @property
    def additional_memories(self) -> List[List[int]]:
        mem = [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]
        mem[self.live_tensor.root.level] = self.live_tensor.volume
        return [mem]


# TODO: Roll MatmulSplitLoop functionality into Loop
@dataclass_abc.dataclass_abc(frozen=True)
class MatmulSplitLoop(_TilingMixin, Schedule):
    """A Schedule representing a blocked loop over the k dimension of the matmul.

    The `lhs` and `rhs` members are the outer operands.
    """

    lhs: Union[Tensor, Tile]
    rhs: Union[Tensor, Tile]
    output: Union[Tensor, Tile]
    inner: Schedule

    def __post_init__(self):
        assert isinstance(self.innermost.spec, specs.Matmul)
        assert self.output.dim_sizes == (
            self.lhs.height,
            self.rhs.width,
        ), f"Expected output to have shape {self.lhs.height}×{self.rhs.width}"

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
    ) -> "Schedule":
        l, r = inputs
        return MatmulSplitLoop(lhs=l, rhs=r, out=output, inner=self.inner)

    @property
    def steps(self) -> int:
        # We expect the inner schedule here to have two inputs (the two matrices).
        inner_lhs, inner_rhs = self.inner.inputs

        result = None
        for dest, src in [(inner_lhs, self.lhs), (inner_rhs, self.rhs)]:
            dh, dw = dest.height, dest.width
            sh, sw = src.height, src.width
            r = math.ceil(sh / dh) * math.ceil(sw / dw)
            if result is None:
                result = r
            else:
                assert r == result
        return result


@dataclass_abc.dataclass_abc(frozen=True)
class MoveLet(Schedule):
    """A Move operation composed with some subsequent Schedule."""

    source: Union[Tensor, Tile]
    destination: Tensor
    input_idx: Optional[int]
    inner: Schedule

    def __post_init__(self):
        assert self.destination.origin is self.source, (
            f"Destination's origin {self.destination.origin} was not source"
            f" {self.source}"
        )

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        keyword = "move"
        if self.is_store:
            keyword = "move*"

        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])
        return (
            f"{keyword}[lvl={self.destination.root.level}]"
            f" {name_tensor_fn(self.destination)}"
            f" {arrow} {name_tensor_fn(self.source)}"
        )

    def store_env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        fancy: bool = False,
    ):
        keyword = "store"
        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])
        return (
            f"{keyword} {name_tensor_fn(self.source)}"
            f" {arrow} {name_tensor_fn(self.destination)}"
        )

    @property
    def is_store(self) -> bool:
        return self.source == self.output

    @property
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        new_inputs = []
        for inner_inp in self.inner.inputs:
            assert inner_inp is not self.source
            if inner_inp is self.destination:
                new_inputs.append(self.source)
            else:
                new_inputs.append(inner_inp)
        return tuple(new_inputs)

    @property
    def children(self) -> Tuple[Schedule, ...]:
        return (self.inner,)

    @functools.cached_property
    def spec(self) -> specs.Spec:
        # The spec of a MoveLet can be calculated by taking the spec of the inner
        # schedule and replacing the right inputs/output of the spec with the source
        # tensor's spec.
        new_input_specs = []
        for inp in self.inner.inputs:
            assert (
                inp is not self.source
            ), "There is no reason for an inner schedule to reference the moved tensor"
            if inp is self.destination:
                new_input_specs.append(self.source.spec)
            else:
                new_input_specs.append(inp.spec)

        new_output_spec = self.inner.output.spec
        assert new_output_spec is not self.source
        if self.inner.output is self.destination:
            new_output_spec = self.source.spec

        return self.inner.spec.replace_io(tuple(new_input_specs), new_output_spec)

    @property
    def lhs(self):
        # A MoveLet can move any operand. This returns the source of the move if output
        # is the operand being moved; the inner output otherwise.
        inner_lhs, _ = self.inner.inputs
        if inner_lhs is self.destination:
            return self.source
        return inner_lhs

    @property
    def rhs(self):
        # A MoveLet can move any operand. This returns the source of the move if output
        # is the operand being moved; the inner output otherwise.
        _, inner_rhs = self.inner.inputs
        if inner_rhs is self.destination:
            return self.source
        return inner_rhs

    @property
    def output(self):
        # A MoveLet can move any operand. This returns the source of the move if output
        # is the operand being moved; the inner output otherwise.
        if self.inner.output is self.destination:
            return self.source
        return self.inner.output

    @_assert_stable_spec
    def split(self, size: int) -> "Schedule":
        return dataclasses.replace(self, inner=self.inner.split(size))

    @_assert_stable_spec
    def complete(self) -> Schedule:
        return dataclasses.replace(self, inner=self.inner.complete())

    def move_input(self, *args, **kwargs) -> "Schedule":
        # Pass move_input through to the inner schedule
        return dataclasses.replace(self, inner=self.inner.move_input(*args, **kwargs))

    def move_output(self, *args, **kwargs) -> "Schedule":
        # Pass move_output through to the inner schedule
        return dataclasses.replace(self, inner=self.inner.move_output(*args, **kwargs))

    @_assert_stable_spec
    def replace_children(self, replacements: Iterable[Schedule]) -> Schedule:
        replacements = list(replacements)
        if len(replacements) != 1:
            raise ValueError(f"One replacement child expected; got {len(replacements)}")
        if replacements[0].spec != self.inner.spec:
            raise ValueError(f"Expected a replacement with spec {self.inner.spec}")
        return dataclasses.replace(self, inner=replacements[0])

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Schedule":
        raise NotImplementedError()

    @property
    def additional_memories(self) -> List[List[int]]:
        mem = [0 for _ in system_config.DEFAULT_SYSTEM_CONFIG.level_configs]
        mem[self.destination.level] = self.destination.volume
        return [mem]

    @property
    def peak_memory(self) -> List[int]:
        mem = self.inner.peak_memory
        mem[self.destination.level] += functools.reduce(
            mul, self.destination.dim_sizes, 1
        )
        return mem

    @property
    def is_scheduled(self) -> bool:
        return self.inner.is_scheduled

    def __hash__(self):
        # A slightly faster hash
        return hash((self.source, self.destination))
