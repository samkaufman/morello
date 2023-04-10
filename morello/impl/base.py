import functools
import typing
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, cast

from .. import specs, tiling
from ..system_config import current_system, current_target
from ..tensor import OperandIdx, TensorLike, Tile
from ..utils import TinyMap
from .pruning import ParentSummary
from .utils import assert_stable_spec


class Impl:
    spec: specs.Spec

    @property
    def children(self) -> tuple["Impl", ...]:
        raise NotImplementedError()

    @property
    @typing.final
    def operand_count(self) -> int:
        return len(self.spec.inputs) + 1

    @property
    def leaves(self) -> Iterable["Impl"]:
        if len(self.children) == 0:
            yield self
        else:
            for child in self.children:
                yield from child.leaves

    @property
    def depth(self) -> int:
        cached_value = getattr(self, "_cached_depth", None)
        if cached_value is not None:
            return cached_value
        d = 1 + max((c.depth for c in self.children), default=0)
        object.__setattr__(self, "_cached_depth", d)
        return d

    @property
    def is_scheduled(self) -> bool:
        if hasattr(self, "inner"):
            return getattr(self, "inner").is_scheduled
        raise NotImplementedError(f"No is_scheduled implementation for {type(self)}")

    def replace_spec(self, new_spec: specs.Spec) -> "Impl":
        raise NotImplementedError(
            f"Not implemented for {type(self).__name__}. (This method should "
            "be implemented for holes.)"
        )

    def replace_children(self, replacements: Iterable["Impl"]) -> "Impl":
        raise NotImplementedError()

    @typing.final
    def replace_child(self, idx: int, replacement: "Impl") -> "Impl":
        new_children = list(self.children)
        new_children[idx] = replacement
        return self.replace_children(new_children)

    @typing.final
    def replace_leaves(self, replacements: Iterable["Impl"]) -> "Impl":
        replacements = list(replacements)
        if not replacements:
            raise ValueError("Must provide at least one replacement")
        result = self._replace_leaves_inner(replacements)
        if replacements:
            raise ValueError("Too many replacements provided")
        return result

    @typing.final
    def _replace_leaves_inner(self, replacements: list["Impl"]) -> "Impl":
        if len(self.children) == 0:
            return replacements.pop(0)
        new_children = []
        for child in self.children:
            new_children.append(child._replace_leaves_inner(replacements))
        return self.replace_children(new_children)

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], "Impl"]]:
        raise NotImplementedError()

    @assert_stable_spec
    def tile_out(self, output_shape: Tuple[int, ...], parallel=False) -> "Impl":
        # Import Loop here to avoid cyclic import
        from .loops import Loop

        if parallel and self.spec.serial_only:
            raise ValueError("Serial-only Spec prevents parallel tiling")

        # If this is an Impl with a single child, just forward.
        if len(self.children) == 1:
            return self.replace_children(
                [self.children[0].tile_out(output_shape, parallel=parallel)]
            )

        if len(output_shape) != len(self.spec.output.dim_sizes):
            raise ValueError(
                f"Expected {len(self.spec.output.dim_sizes)} dimensions; got {len(output_shape)}"
            )
        for dim, dim_size in enumerate(output_shape):
            if dim_size <= 0:
                raise ValueError(
                    "All dimensions must be size 1 or greater, but "
                    f"given output shape {output_shape}"
                )
            elif dim_size > self.spec.output.dim_sizes[dim]:
                raise ValueError(
                    f"Dimensions {dim} was larger than "
                    f"{self.spec.output.dim_sizes[dim]} ({dim_size} > "
                    f"{self.spec.output.dim_sizes[dim]})"
                )

        # A no-op if the given shape is already the output shape.
        if self.spec.output.dim_sizes == output_shape:
            return self

        # Tile the output and inputs.
        smaller_output = self.spec.output.simple_tile(
            OperandIdx(len(self.spec.inputs)), output_shape
        )
        smaller_inputs = [
            partial_tile.tile(OperandIdx(input_idx), inp)
            for input_idx, (inp, partial_tile) in enumerate(
                zip(
                    self.spec.inputs,
                    self._calculate_partial_inputs_for_tile_out(
                        tiling.tile_to_partial(smaller_output)
                    ),
                )
            )
        ]

        # Make an inner hole for the now-smaller Spec.
        inner_serial = None
        if parallel:
            inner_serial = True
        inner = spec_to_hole(
            self.spec.replace_io(
                tuple(x.spec for x in smaller_inputs),
                smaller_output.spec,
                serial_only=inner_serial,
            )
        )

        # Construct the list of tiles, which is every tile_out result that just
        # returned itself because the tile would have been the same shape as the
        # original. (Note that the output, unlike the inputs, is known to have
        # been tiled or this method would have short-circuited.)
        # TODO: Can we share this code with same block in ComposeHole.tile_out?
        changed_input_tiles = set()
        for original_input, tiled_input in zip(self.spec.inputs, smaller_inputs):
            if original_input != tiled_input.spec:
                changed_input_tiles.add(tiled_input)
        changed_input_tiles = frozenset(changed_input_tiles)

        return Loop(
            spec=self.spec,
            subscripts=self.spec.operands_dim_subscripts()[-1],
            tiles=frozenset([smaller_output]) | frozenset(changed_input_tiles),
            inner=inner,
            parallel=parallel,
        )

    @typing.final
    def _can_tile_out(self, output_shape: Sequence[int]) -> bool:
        """Returns True if the Impl can be tiled to a given output shape.

        This is true if the output can be tiled to given shape and all input operands
        can be tiled to the corresponding input shapes.
        """
        output_shape = tuple(output_shape)
        if not self.spec.output.is_valid_tile_shape(output_shape):
            return False

        smaller_partial_out = tiling.PartialSimpleTile(output_shape)
        for inp, partial_input in zip(
            self.spec.inputs,
            self._calculate_partial_inputs_for_tile_out(smaller_partial_out),
        ):
            if not inp.is_valid_tile_shape(partial_input.dim_sizes):
                return False
        return True

    def _can_sliding_tile_out(
        self, sliding_dim: int, output_size: int, bank: str
    ) -> bool:
        # Import here to avoid a cyclic import
        from .loops import Loop

        # If this is an Impl with a single child, just forward.
        if len(self.children) == 1:
            return self.children[0]._can_sliding_tile_out(
                sliding_dim, output_size, bank
            )

        # If the given shape is already the output shape, this is trivially permitted.
        output_shape = (
            self.spec.output.dim_sizes[:sliding_dim]
            + (output_size,)
            + self.spec.output.dim_sizes[sliding_dim + 1 :]
        )
        if output_shape == self.spec.output.dim_sizes:
            return True

        if not self._can_tile_out(output_shape):
            return False
        impl = cast(Loop, self.tile_out(output_shape))

        # We cannot introduce a sliding tile if there is no overlap in the corresponding
        # input dimension.
        if not any(t.frontiers[sliding_dim] for t in impl.tiles):
            return False

        return True

    def _calculate_partial_inputs_for_tile_out(
        self, output_tile: tiling.PartialTile
    ) -> list[tiling.PartialTile]:
        return list(
            tiling.tile_out(
                type(self.spec),
                [inp.dim_sizes for inp in self.spec.inputs],
                output_tile,
            )
        )

    def split(self, k: int) -> "Impl":
        raise NotImplementedError(f"Not implemented for {type(self).__name__}")

    @assert_stable_spec
    def sliding_tile_out(self, sliding_dim: int, output_size: int, bank: str) -> "Impl":
        """Like tile_out, without reloading overlapping image regions in one dimension."""
        # Import here to avoid cyclic import
        from .loops import Loop, SlidingWindowLoop

        # If this is an Impl with a single child, just forward.
        if len(self.children) == 1:
            return self.replace_children(
                [self.children[0].sliding_tile_out(sliding_dim, output_size, bank)]
            )

        # A no-op if the given shape is already the output shape.
        output_shape = (
            self.spec.output.dim_sizes[:sliding_dim]
            + (output_size,)
            + self.spec.output.dim_sizes[sliding_dim + 1 :]
        )
        if output_shape == self.spec.output.dim_sizes:
            return self

        # We produce a SlidingWindowLoop by first doing a normal tile_out,
        # then swapping an introduced input tile with a non-zero frontier
        # (i.e. a ConvolutionImageTile) for a sliding Tensor managed by the
        # resulting SlidingWindowLoop. This method doesn't yet support the
        # case that multiple introduced inputs have non-zero frontiers.
        impl = self.tile_out(output_shape)
        assert isinstance(impl, Loop)
        assert len(impl.inner.children) == 0, "Expected inner to be a leaf"

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
        if tile_to_convert is None:
            raise ValueError(f"There is no overlap in dimension {sliding_dim}")
        assert isinstance(
            tile_to_convert, Tile
        ), f"tile_to_convert was unexpectedly a {type(tile_to_convert)}"

        # Turn tile_to_convert into a Tensor under management of the SlidingWindowLoop
        # and a Tile for calculating the update costs
        live_tensor = current_target().tensor(
            spec=current_target().tensor_spec(
                tile_to_convert.dim_sizes,
                tile_to_convert.dtype,
                bank=bank,
                layout=tile_to_convert.layout,
            ),
            name=None,
        )
        return SlidingWindowLoop(
            live_tensor=live_tensor,
            live_tensor_idx=tile_to_convert.source,
            frontier_size=tile_to_convert.frontiers[sliding_dim],
            other_tiles=tuple(other_tiles),
            spec=self.spec,
            inner=impl.inner,
        )

    @assert_stable_spec
    def place(self, leaf_cls, *args, **kwargs):
        if len(self.children) == 0:
            return leaf_cls(self.spec, *args, **kwargs)
        elif len(self.children) == 1:
            return self.replace_children(
                [next(iter(self.children)).place(leaf_cls, *args, **kwargs)]
            )
        else:
            raise ValueError("Multiple children. Leaf-to-replace is ambiguous.")

    @typing.final
    @assert_stable_spec
    def subschedule(
        self, path: Union[list[int], tuple[int, ...]], fn: Callable[["Impl"], "Impl"]
    ) -> "Impl":
        if not path:
            raise ValueError("path must not be empty")

        new_children = list(self.children)
        p = path[0]
        if len(path) == 1:
            new_children[p] = fn(new_children[p])
        else:
            new_children[p] = self.children[p].subschedule(path[1:], fn)
        return self.replace_children(new_children)

    @typing.final
    def enter(self, path: Union[int, Sequence[int]]) -> "Subscheduler":
        if isinstance(path, int):
            path = (path,)

        # Subscheduler accepts the full path, including single-child sub-Impls, but it's
        # more ergonomic to require the caller to only pass in child indices where there
        # is a real choice between multiple children, so let's convert.
        full_path = []
        remaining_choices = list(path)
        cur = self
        while remaining_choices:
            if len(cur.children) > 1:
                full_path.append(remaining_choices.pop(0))
                cur = cur.children[full_path[-1]]
            else:
                full_path.append(0)
                cur = next(iter(cur.children))

        return Subscheduler(self, full_path)

    @assert_stable_spec
    def complete(self) -> "Impl":
        return self.replace_children(c.complete() for c in self.children)

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        raise NotImplementedError()

    def to_applied(self) -> "AppliedImpl":
        return self.apply([current_target().tensor(o) for o in self.spec.operands])

    @property
    def memory_allocated(self) -> tuple[TinyMap[str, int], list[TinyMap[str, int]]]:
        """Returns the amount of memory allocated by this Impl.

        Returns a tuple of, first, the amount of memory allocated by this Impl which is
        live for the duration of its execution, and second, the amount of memory
        allocated which will be live during the execution of each child Impl.

        The default implementation of this method returns zero for all.
        """
        banks = current_system().ordered_banks
        z = TinyMap(banks, (0,) * len(banks))
        return z, [z] * len(self.children)

    @typing.final
    @property
    def peak_memory(self) -> TinyMap[str, int]:
        return self.peak_memory_from_child_peaks(
            self.memory_allocated, [c.peak_memory for c in self.children]
        )

    @staticmethod
    def peak_memory_from_child_peaks(
        imp_memory_allocated, child_peaks: Sequence[TinyMap[str, int]]
    ) -> TinyMap[str, int]:
        banks = current_system().ordered_banks

        base_adds, per_child_adds = imp_memory_allocated
        assert base_adds.raw_keys == banks

        # The max of each per-child, per-bank bytes allocated, including any additional
        # from this Impl.
        vals = [0] * len(banks)
        for child_adds, child_peak in zip(per_child_adds, child_peaks):
            assert child_peak.raw_keys == banks
            assert child_adds.raw_keys == banks
            for i in range(len(banks)):
                vals[i] = max(
                    vals[i], child_peak.raw_values[i] + child_adds.raw_values[i]
                )

        # Add the base memory allocated. This is the memory live for the entire
        # execution of this Impl.
        vals = [v + b for v, b in zip(vals, base_adds.raw_values)]

        return TinyMap(banks, tuple(vals))

    @property
    def operands_subscripts(self) -> Sequence[tuple[int, ...]]:
        return self.spec.operands_dim_subscripts()


class Leaf(Impl):
    """A helper base class for Impls that have no children (i.e., are leaves)."""

    @property
    def children(self) -> tuple[Impl, ...]:
        return tuple()

    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        try:
            next(iter(replacements))
        except StopIteration:
            return self
        else:
            raise NotImplementedError(
                f"{type(self).__name__} does not have children, "
                "but was given a non-empty iterable"
            )

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        return make_applied_impl(self, operands)


# TODO: Remove this now-useless class.
class NonAllocatingLeaf(Leaf):
    """A helper base class for leaf Impls that do not allocate memory."""

    pass


class AppliedImpl(Impl):
    unapplied: Impl
    operands: tuple[TensorLike, ...]

    def __init__(self, unapplied, operands):
        assert isinstance(unapplied, Impl)

        object.__setattr__(self, "unapplied", unapplied)
        object.__setattr__(self, "operands", tuple(operands))

        assert not hasattr(self.unapplied, "operands")
        assert not hasattr(self.unapplied, "inputs")
        assert not hasattr(self.unapplied, "output")
        assert len(self.operands) == self.unapplied.operand_count, (
            f"Expected {self.unapplied.operand_count} operands, but was given: "
            f"{self.operands}"
        )
        assert all(
            a.spec == b for a, b in zip(self.operands, self.unapplied.spec.inputs)
        ), (
            f"Operand specs didn't match: {', '.join(str(a.spec) for a in self.operands[:-1])} != "
            f"{', '.join(map(str, self.unapplied.spec.inputs))}"
        )
        assert (
            self.operands[-1].spec == self.unapplied.spec.output
        ), f"{self.operands[-1].spec} != {self.unapplied.spec.output}"
        assert all(isinstance(c, AppliedImpl) for c in self.unapplied.children)

    def __eq__(self, other):
        if not isinstance(other, AppliedImpl):
            return NotImplemented
        return self.unapplied == other.unapplied and self.operands == other.operands

    def __hash__(self) -> int:
        return hash((self.unapplied, self.operands))

    @property
    def inputs(self):
        return self.operands[:-1]

    @property
    def output(self):
        return self.operands[-1]

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        if len(operands):
            raise ValueError("Cannot apply again.")
        return self

    def to_applied(self) -> "AppliedImpl":
        return self

    def __getattr__(self, name):
        return getattr(self.unapplied, name)

    def __setattr__(self, name, value):
        setattr(self.unapplied, name, value)


class Subscheduler:
    def __init__(self, root: Union[Impl, "Subscheduler"], full_path: Sequence[int]):
        if not full_path:
            raise ValueError("full_path must not be empty")

        self.root = root
        self.full_path = list(full_path)

        child = root
        for child_idx in full_path:
            child = child.children[child_idx]  # type: ignore
        self.child = child

        # TODO: Assert that child is a leaf

    def enter(self, *args, **kwargs):
        raise NotImplementedError("Recursive `enter` not yet implemented.")

    def exit(self) -> Union[Impl, "Subscheduler"]:
        # TODO: Forward exit to child if it exists

        # Recursively replace children after following the path of child indices.
        stack: list[Union[Impl, "Subscheduler"]] = [self.root]
        for child_idx in self.full_path[:-1]:
            stack.append(stack[-1].children[child_idx])  # type: ignore
        last: Impl = self.child
        for p in reversed(self.full_path):
            last = stack.pop().replace_child(p, last)
        assert last.spec == self.root.spec
        return last

    # Forward properties and methods to the sub-Impl at `path`.
    def __getattr__(self, name):
        child_attr = getattr(self.child, name)
        if not callable(child_attr):
            return child_attr

        def result_captured(*args, **kwargs):
            # Unlike normal scheduling, this will capture results until exit,
            # returning the Subscheduler itself.
            r = child_attr(*args, **kwargs)
            assert isinstance(r, Impl), f"Expected Impl, but got {r}"
            self.child = r
            return self

        return result_captured


@functools.lru_cache(maxsize=None)
def _make_applied_impl_cls(delegate_cls) -> type:
    name = f"Applied{delegate_cls.__name__}"
    new_cls = type(name, (AppliedImpl, delegate_cls), {})
    return new_cls


def make_applied_impl(unapplied: Impl, operands: Sequence[TensorLike]) -> AppliedImpl:
    return _make_applied_impl_cls(type(unapplied))(unapplied, operands)


def spec_to_hole(spec: specs.Spec) -> "Impl":
    """Returns a default, incomplete schedule for a Spec which consume given inputs.

    If either `inputs` or `output` is None, default Tensors from the corresponding
    TensorSpecs will be constructed (using the current target).
    """
    # Import some Impls here to avoid import cycle
    # TODO: Can we move this to its own file instead?
    from .compose import ComposeHole
    from .convhole import ConvAccumHole, ConvHole
    from .matmuls import MatmulAccumHole, MatmulHole
    from .moves import LoadHole, StoreHole
    from .reducesum import ReduceSumAccumHole, ReduceSumHole
    from .zero import ZeroHole

    if isinstance(spec, specs.Convolution):
        return ConvHole(spec)
    elif isinstance(spec, specs.ConvolutionAccum):
        return ConvAccumHole(spec)
    elif isinstance(spec, specs.Matmul):
        return MatmulHole(spec)
    elif isinstance(spec, specs.MatmulAccum):
        return MatmulAccumHole(spec)
    elif isinstance(spec, specs.ReduceSum):
        return ReduceSumHole(spec)
    elif isinstance(spec, specs.ReduceSumAccum):
        return ReduceSumAccumHole(spec)
    elif isinstance(spec, specs.Compose):
        return ComposeHole(spec)
    elif isinstance(spec, specs.Load):
        return LoadHole(spec)
    elif isinstance(spec, specs.Store):
        return StoreHole(spec)
    elif isinstance(spec, specs.Zero):
        return ZeroHole(spec)
    else:
        raise NotImplementedError(f"No hole type for {type(spec).__name__}")
