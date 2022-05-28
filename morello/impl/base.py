import abc
import dataclasses
import typing
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, cast

from morello import specs, tiling

from ..layouts import Layout
from ..system_config import current_system, current_target
from ..tensor import SimpleTile, Tensor, TensorLike, Tile
from .pruning import ParentSummary
from .utils import assert_stable_spec


class Impl(abc.ABC):
    @property
    @abc.abstractmethod
    def spec(self) -> specs.Spec:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def children(self) -> tuple["Impl", ...]:
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
    def leaves(self) -> Iterable["Impl"]:
        if len(self.children) == 0:
            yield self
        else:
            for child in self.children:
                yield from child.leaves

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
    def is_scheduled(self) -> bool:
        if hasattr(self, "inner"):
            return getattr(self, "inner").is_scheduled
        raise NotImplementedError(f"No is_scheduled implementation for {type(self)}")

    @typing.final
    def replace_child(self, child_idx: int, new_child: "Impl") -> "Impl":
        replacements = list(self.children)
        replacements[child_idx] = new_child
        return self.replace_children(replacements)

    @abc.abstractmethod
    def replace_children(self, replacements: Iterable["Impl"]) -> "Impl":
        raise NotImplementedError()

    @abc.abstractmethod
    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        raise NotImplementedError()

    @typing.final
    def replace_operand(self, operand_idx: int, new_operand: TensorLike) -> "Impl":
        ops = list(self.operands)
        ops[operand_idx] = new_operand
        return self.replace_io(ops[:-1], ops[-1])

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

        # A no-op if the given shape is already the output shape.
        if self.output.dim_sizes == output_shape:
            return self

        # Tile the output first
        smaller_output = self.output.simple_tile(output_shape)
        assert smaller_output != self.output
        assert isinstance(smaller_output, SimpleTile)

        # Tile the corresponding inputs.
        smaller_inputs = self._calculate_inputs_for_tile_out(
            tiling.tile_to_partial(smaller_output)
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
            subscripts=self.spec.operands_dim_subscripts()[-1],
            tiles=frozenset([smaller_output]) | frozenset(unchanged_input_tiles),
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
        if not self.output.spec.is_valid_tile_shape(output_shape):
            return False
        # TODO: It'd be nice to not need to tile the output to get its PartialTile
        smaller_partial_out = tiling.tile_to_partial(
            self.output.simple_tile(output_shape)
        )
        for inp, partial_input in zip(
            self.inputs,
            self._calculate_partial_inputs_for_tile_out(smaller_partial_out),
        ):
            if not inp.spec.is_valid_tile_shape(partial_input.dim_sizes):
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
            self.output.dim_sizes[:sliding_dim]
            + (output_size,)
            + self.output.dim_sizes[sliding_dim + 1 :]
        )
        if output_shape == self.output.dim_sizes:
            return True

        # We cannot introduce a sliding tile if there is no overlap in the corresponding
        # input dimension.
        impl = cast(Loop, self.tile_out(output_shape))
        if not any(t.frontiers[sliding_dim] for t in impl.tiles):
            return False

        return True

    @typing.final
    def _calculate_inputs_for_tile_out(
        self, output_tile: Union[SimpleTile, tiling.PartialTile]
    ) -> list[TensorLike]:
        return [
            partial_tile.tile(inp)
            for inp, partial_tile in zip(
                self.inputs, self._calculate_partial_inputs_for_tile_out(output_tile)
            )
        ]

    def _calculate_partial_inputs_for_tile_out(
        self, output_tile: Union[SimpleTile, tiling.PartialTile]
    ) -> list[tiling.PartialTile]:
        return list(
            tiling.tile_out(
                type(self.spec),
                [inp.dim_sizes for inp in self.inputs],
                output_tile,
            )
        )

    def split(self, k: int) -> "Impl":
        raise NotImplementedError()

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

    @assert_stable_spec
    def split_filters(self, k: int) -> "Impl":
        return dataclasses.replace(self, inner=self.inner.split_filters(k))

    @assert_stable_spec
    def place_mult(self, *args, **kwargs):
        if len(self.children) != 1:
            raise NotImplementedError()
        return self.replace_children(
            [next(iter(self.children)).place_mult(*args, **kwargs)]
        )

    @assert_stable_spec
    def place_broadcastvecmult(self, *args, **kwargs):
        if len(self.children) != 1:
            raise NotImplementedError()
        return self.replace_children(
            [next(iter(self.children)).place_broadcastvecmult(*args, **kwargs)]
        )

    @assert_stable_spec
    def place_hvx_vrmpyacc(self, *args, **kwargs):
        if len(self.children) != 1:
            raise NotImplementedError()
        return self.replace_children(
            [next(iter(self.children)).place_hvx_vrmpyacc(*args, **kwargs)]
        )

    @assert_stable_spec
    def place_hvx_gemvmpebbw(self, *args, **kwargs):
        if len(self.children) != 1:
            raise NotImplementedError()
        return self.replace_children(
            [next(iter(self.children)).place_hvx_gemvmpebbw(*args, **kwargs)]
        )

    @abc.abstractmethod
    def move_input(
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "Impl":
        raise NotImplementedError()

    @abc.abstractmethod
    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "Impl":
        raise NotImplementedError()

    def pad_transpack(self, input_idx: int) -> "Impl":
        raise NotImplementedError(f"Unimplemented for {type(self).__name__}")

    @assert_stable_spec
    def subschedule(self, *args, **kwargs) -> "Impl":
        if len(self.children) == 1:
            child = self.children[0]
            return self.replace_children([child.subschedule(*args, **kwargs)])
        raise NotImplementedError()

    @assert_stable_spec
    def complete(self) -> "Impl":
        return dataclasses.replace(self, inner=self.inner.complete())

    @property
    @abc.abstractmethod
    def additional_memories(self) -> list[dict[str, int]]:
        """Memory costs of self when the corresponding child is executed.

        :returns: A list of amounts of memory to remove from that available. The
          outermost list has the same length as the number of children in this
          Impl.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def peak_memory(self) -> dict[str, int]:
        raise NotImplementedError()


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


class NonAllocatingLeaf(Leaf):
    """A helper base class for leaf Impls that do not allocate memory."""

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return []

    @property
    def peak_memory(self) -> dict[str, int]:
        return {k: 0 for k in current_system().banks}


def spec_to_hole(
    spec: specs.Spec, inputs: Optional[Iterable] = None, output: Optional[Any] = None
) -> "Impl":
    """Returns a default, incomplete schedule for a Spec which consume given inputs.

    If either `inputs` or `output` is None, default Tensors from the corresponding
    TensorSpecs will be constructed (using the current target).
    """
    # Import some Impls here to avoid import cycle
    # TODO: Can we move this to its own file instead?
    from .compose import ComposeHole
    from .directconv import DirectConv
    from .matmuls import MatmulHole
    from .reducesum import ReduceSum

    if inputs is None:
        target = current_target()
        inputs = tuple(target.tensor(s) for s in spec.inputs)
    if output is None:
        output = current_target().tensor(spec.output)

    inputs = tuple(inputs)
    if isinstance(spec, specs.Convolution):
        assert len(inputs) == 2, f"Expected 2 Tensor/Tile operands; got {len(inputs)}"
        return DirectConv(
            lhs=inputs[0], rhs=inputs[1], output=output, serial_only=spec.serial_only
        )
    elif isinstance(spec, specs.Matmul):
        assert len(inputs) == 2, f"Expected 2 Tensor/Tile operands; got {len(inputs)}"
        return MatmulHole(
            lhs=inputs[0], rhs=inputs[1], output=output, serial_only=spec.serial_only
        )
    elif isinstance(spec, specs.ReduceSum):
        assert len(inputs) == 1, f"Expected 1 Tensor/Tile operands; got {len(inputs)}"
        return ReduceSum(source=inputs[0], output=output, serial_only=spec.serial_only)
    elif isinstance(spec, specs.Compose):
        return ComposeHole(spec, inputs=inputs, output=output)
    else:
        raise NotImplementedError()
