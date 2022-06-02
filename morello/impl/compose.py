import dataclasses
import functools
import itertools
import math
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import dataclass_abc
import termcolor

from .. import layouts, specs, system_config, tiling, utils
from ..layouts import Layout
from ..system_config import current_system, current_target
from ..tensor import SimpleTile, Tensor, TensorLike, Tile
from .actions import PeelAction, SlidingTileOutAction, TileOutAction
from .base import Impl, spec_to_hole
from .loops import Loop
from .moves import MoveLet, common_operand_move_actions
from .pruning import (
    ParentSummary,
    break_matmul_split_symmetries,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from .settings import allow_reduce_splits, allow_sliding_windows
from .utils import assert_stable_spec, dim_range, gen_tile_sizes, gen_vector_shapes

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class SplitNotSupportedByHeadError(NotImplementedError):
    pass


@dataclass_abc.dataclass_abc(frozen=True)
class ComposeHole(Impl):
    spec: specs.Compose
    inputs: tuple[Union[Tensor, Tile], ...]
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
    def children(self) -> Tuple["Impl", ...]:
        return tuple()

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        inputs = tuple(inputs)
        new_spec = self.spec.replace_io(tuple(inp.spec for inp in inputs), output.spec)
        return ComposeHole(new_spec, inputs, output)

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        target = system_config.current_target()
        system = target.system

        # TODO: Remove this symmetry: lots of ways to iteratively split the pipeline
        # TODO: Reintroduce splitting on non-index 0
        for bank, layout in itertools.product(system.addressed_banks, target.all_layouts):
            # TODO: Remove following check once cost.move_cost handles it correctly.
            if not system.has_hvx and layout == layouts.HEXAGON_TRANSPACKED:
                continue

            # TODO: The below special-casing for VMEM stinks. If we had partial
            #  finite functions/lenses, we could just constrain a few properties and
            #  enumerate the rest. It would also be possible to generate
            #  Tensor-making callables for the peel to populate in a generic way.
            peel_kwargs = [{}]
            if bank == "VMEM":
                peel_kwargs = (
                    {"vector_shape": shape}
                    for shape in gen_vector_shapes(
                        self.spec.subspec_outputs[1],
                        dtype=self.spec.intermediate_dtypes[0],
                    )
                )
            for kws in peel_kwargs:
                if self._can_peel(bank=bank, layout=layout, **kws):
                    yield PeelAction(self, bank=bank, layout=layout, kwargs=kws)

        yield from common_operand_move_actions(self)

        for tile_shape in gen_tile_sizes(
            self.output.dim_sizes, filter=self._can_tile_out
        ):
            for parallel in [False] if self.spec.serial_only else [True, False]:
                yield TileOutAction(self, tile_shape, parallel)

        # TODO: Don't just use the first input. This isn't general to arbitrary
        # slideable ops in a Pipeline, or even, for instance, if we swapped the
        # operand order of Convolution/DirectConv. Besides, the dimensions
        # passed to sliding_tile_out are over the *output* Tile, not either
        # input; this only works because they're one-to-one for DirectConv.
        if allow_sliding_windows.get():
            for slide_dim in self._get_output_dimensions_matching_first_input():
                for slide_size in dim_range(
                    self.output.dim_sizes[slide_dim], include_end=False
                ):
                    # TODO: Range over multiple choices of bank
                    if self._can_sliding_tile_out(
                        slide_dim, slide_size, system.default_fast_bank
                    ):
                        yield SlidingTileOutAction(
                            self.sliding_tile_out,
                            slide_dim,
                            slide_size,
                            system.default_fast_bank,
                        )

        # TODO: This is awful. Produce a real interface for both deferring to
        #   inner split and for gathering the right split.
        if allow_reduce_splits.get():
            for k in self.split_sizes():
                for parallel in [False] if self.spec.serial_only else [True, False]:
                    yield functools.partial(self.split, k, parallel=parallel)

    def _get_output_dimensions_matching_first_input(self) -> Iterable[int]:
        """Returns the dimensions matching subscripts of the first head input."""
        first_head_idx = -self.spec.subspec_classes[-1].inputs_count()
        first_head_subs = self.spec.operands_dim_subscripts()[first_head_idx - 1]
        output_subs = self.spec.operands_dim_subscripts()[-1]
        for sub in first_head_subs:
            try:
                yield output_subs.index(sub)
            except ValueError:
                pass

    @assert_stable_spec
    def move_input(
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ):
        operand = self.inputs[input_idx]
        if bank is None:
            bank = operand.root.bank
        if layout is None:
            layout = operand.layout
        if bank == operand.root.bank and layout == operand.layout:
            raise ValueError("Either bank or layout must differ from current")

        # TODO: Share this block with common_move?
        # Will the result be contiguous? If the move is into "non-addressed"
        # memory, then no. If it is, then it might be.
        contiguous = False
        if bank in current_system().addressed_banks:
            contiguous = utils.contiguous((operand.dim_sizes, layout), operand.spec)

        new_mat = current_target().tensor(
            spec=current_target().tensor_spec(
                operand.dim_sizes,
                dtype=operand.dtype,
                contiguous=contiguous,
                layout=layout,
                bank=bank,
                **kwargs,
            ),
            name=None,
            origin=operand,
        )

        new_inputs = self.inputs[:input_idx] + (new_mat,) + self.inputs[input_idx + 1 :]
        new_inner_spec = self.spec.replace_io(
            tuple(inp.spec for inp in new_inputs), self.spec.output
        )
        return MoveLet(
            source=operand,
            destination=new_mat,
            input_idx=input_idx,
            prefetching=prefetching,
            inner=dataclasses.replace(self, spec=new_inner_spec, inputs=new_inputs),
        )

    @assert_stable_spec
    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "Impl":
        operand = self.output
        if bank is None:
            bank = operand.root.bank
        if layout is None:
            layout = operand.layout
        if bank == operand.root.bank and layout == operand.layout:
            raise ValueError("Either bank or layout must differ from current")

        # TODO: Share this block with common_move?
        # Will the result be contiguous? If the move is into "non-addressed"
        # memory, then no. If it is, then it might be.
        contiguous = False
        if bank in current_system().addressed_banks:
            contiguous = utils.contiguous((operand.dim_sizes, layout), operand.spec)

        new_mat = current_target().tensor(
            spec=current_target().tensor_spec(
                operand.dim_sizes,
                dtype=operand.dtype,
                contiguous=contiguous,
                layout=layout,
                bank=bank,
                **kwargs,
            ),
            name=None,
            origin=operand,
        )

        new_inner_spec = self.spec.replace_io(self.spec.inputs, new_mat.spec)
        return MoveLet(
            source=operand,
            destination=new_mat,
            input_idx=None,
            prefetching=prefetching,
            inner=dataclasses.replace(self, spec=new_inner_spec, output=new_mat),
        )

    def _can_peel(self, bank: str, layout: Layout, **kwargs) -> bool:
        # Check if we can peel by just trying to make the intermediate tensor that peel
        # would make and seeing if we get a ValueError. This isn't a great solution:
        # catching all ValueErrors might become overbroad as the code evolves, and the
        # object construction is inefficient and unneeded. However, it'll work for now.
        intermediate_tensor_layout = layout
        if all(d == 1 for d in self.spec.subspec_outputs[1]):
            intermediate_tensor_layout = layouts.ROW_MAJOR
        try:
            current_target().tensor(
                current_target().tensor_spec(
                    dim_sizes=self.spec.subspec_outputs[1],
                    dtype=self.spec.intermediate_dtypes[0],
                    contiguous=True,
                    bank=bank,
                    layout=intermediate_tensor_layout,
                    **kwargs,
                ),
            )
            return True
        except ValueError:
            return False

    @assert_stable_spec
    def peel(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        **kwargs,
    ) -> Impl:
        if bank is None or layout is None:
            # TODO: Just require them as arguments. Can relax the signature later.
            raise NotImplementedError("Auto-selecting bank or layout unimplemented")

        # TODO: Using ALPHABET_PRODUCT here will fail for long programs
        intermediate_tensor_layout = layout
        if all(d == 1 for d in self.spec.subspec_outputs[1]):
            intermediate_tensor_layout = layouts.ROW_MAJOR
        intermediate_tensor = current_target().tensor(
            current_target().tensor_spec(
                dim_sizes=self.spec.subspec_outputs[1],
                dtype=self.spec.intermediate_dtypes[0],
                contiguous=True,
                bank=bank,
                layout=intermediate_tensor_layout,
                **kwargs,
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
                    intermediate_dtypes=self.spec.intermediate_dtypes[1:],
                    serial_only=self.spec.serial_only,
                ),
                inputs=self.inputs[hi:],
                # output=self.inputs[1][-1],
                output=intermediate_tensor,
            )
        return Pipeline((remainder, head_hole))

    @assert_stable_spec
    def tile_out(self, output_shape: tuple[int, ...], parallel=False) -> Impl:
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
                self.inputs,
                self._calculate_partial_inputs_for_tile_out(shrunken_output_tile),
            )
        )

        # Construct the spec for the smaller ComposeHole
        return Loop(
            subscripts=self.spec.operands_dim_subscripts()[-1],
            tiles=frozenset([shrunken_output_tile])
            | frozenset(t for _, t in self._filter_unchanged_inputs(reified_inputs)),
            inner=ComposeHole(
                specs.Compose(
                    subspec_classes=self.spec.subspec_classes,
                    inputs=tuple(inp.spec for inp in reified_inputs),
                    output=shrunken_output_tile.spec,
                    intermediate_dtypes=self.spec.intermediate_dtypes,
                    serial_only=(parallel or self.spec.serial_only),
                ),
                inputs=reified_inputs,
                output=shrunken_output_tile,
            ),
            parallel=parallel,
        )

    @assert_stable_spec
    def split(self, k: int, parallel=False) -> Impl:
        # TODO: Can we abstract over both Matmul and Reduce' splits
        if self.spec.subspec_classes[0] == specs.ReduceSum:
            return self._split_reduce_head(k, parallel=parallel)
        else:
            raise SplitNotSupportedByHeadError()

    def split_sizes(self) -> Iterable[int]:
        if self.spec.subspec_classes[0] == specs.ReduceSum:
            # TODO: This should defer to the inner op
            for k in dim_range(self.spec.subspec_outputs[1][-1], include_end=False):
                if k != self.spec.subspec_outputs[1][-1]:
                    yield k
        else:
            return

    def _split_reduce_head(self, k: int, parallel: bool) -> Impl:
        assert self.spec.subspec_classes[0] == specs.ReduceSum

        if parallel and self.spec.serial_only:
            raise ValueError("Serial-only Spec prevents parallel tiling")

        # A no-op if `k` is already the max size.
        orig_reduce_input_shape: tuple[int, ...] = self.spec.subspec_outputs[1]
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
                self._calculate_partial_inputs_for_tile_out(
                    smaller_partial_input_tile, skip_first=1
                ),
            )
        )

        filtered_reified_inputs = list(self._filter_unchanged_inputs(reified_inputs))

        # Select the tile from which we'll grab the subscripts (i.e., the
        # driving tile) by just selecting the inputs with the expected number
        # of steps.
        # TODO: This is extremely ad-hoc. We need a solution for arbitrary
        #  accumulating loops. Fix this.
        expected_steps = math.ceil(orig_reduce_input_shape[-1] / k)
        driving_subs = None
        for inp_idx, inp in filtered_reified_inputs:
            if inp.steps == expected_steps:
                driving_subs = self.spec.operands_dim_subscripts()[inp_idx]
                break
        assert driving_subs, f"No tile had expected number of steps: {expected_steps}"

        # Build the loop
        new_inner = ComposeHole(
            specs.Compose(
                subspec_classes=self.spec.subspec_classes,
                inputs=tuple(inp.spec for inp in reified_inputs),
                output=self.output.spec,
                intermediate_dtypes=self.spec.intermediate_dtypes,
                serial_only=(parallel or self.spec.serial_only),
            ),
            inputs=reified_inputs,
            output=self.output,
        )
        return Loop(
            subscripts=driving_subs,
            tiles=frozenset(t for _, t in filtered_reified_inputs),
            inner=new_inner,
            parallel=parallel,
        )

    def _filter_unchanged_inputs(
        self, source: Iterable[TensorLike]
    ) -> Iterable[tuple[int, TensorLike]]:
        """Return given tensors different from self's corresponding inputs."""
        for idx, (original_input, tiled_input) in enumerate(zip(self.inputs, source)):
            if original_input != tiled_input:
                yield idx, tiled_input

    def _calculate_partial_inputs_for_tile_out(
        self,
        output_tile: Union[SimpleTile, tiling.PartialTile],
        skip_first: int = 0,
    ) -> list[tiling.PartialTile]:
        """Returns PartialTiles for this ComposeHole's inputs for an output tiling.

        Accepts either a PartialTile or a Tile which will be converted into a PartialTile.
        """
        subspec_classes = list(self.spec.subspec_classes)
        intermediate_shapes = list(self.spec.subspec_outputs[1:])
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

    @assert_stable_spec
    def complete(self) -> "Impl":
        next_bank = system_config.current_system().default_bank
        return self.peel(bank=next_bank, layout=layouts.ROW_MAJOR).complete()

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        if replacements:
            raise Exception("Holes have no children to replace")
        return self

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return [{b: 0 for b in system_config.current_system().banks}]

    @property
    def peak_memory(self) -> dict[str, int]:
        return {b: 0 for b in system_config.current_system().banks}

    @property
    def is_scheduled(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True, init=False)
class Pipeline(Impl):
    """A sequence of schedules, called stages, which are executed in order.

    The output of the pipeline is the output of the final stage.
    """

    stages: tuple[Impl, ...]

    def __init__(self, stages: tuple[Impl, ...]):
        assert len(stages) >= 2
        # TODO: Reintroduce check for operand agreement
        # for before, after in zip(self.stages[:-1], self.stages[1:]):
        #     assert (
        #         before.output == after.operands[0]
        #     ), f"Output of {before} didn't match first operand of {after}"

        # Flatten any immediate Pipeline children
        flattened_stages: List[Impl] = []
        for stage in stages:
            if isinstance(stage, Pipeline):
                flattened_stages.extend(stage.stages)
            else:
                flattened_stages.append(stage)
        object.__setattr__(self, "stages", tuple(flattened_stages))

    @functools.cached_property
    def spec(self) -> specs.Compose:
        subspec_classes = []
        intermed_dtypes = []
        inputs = tuple()
        for i, stage in enumerate(self.stages):
            if isinstance(stage.spec, specs.Compose):
                subspec_classes = list(stage.spec.subspec_classes) + subspec_classes
                intermed_dtypes = list(stage.spec.intermediate_dtypes) + intermed_dtypes
            else:
                subspec_classes.insert(0, type(stage.spec))
            inputs = tuple(stage.spec.inputs) + inputs
            if i > 0:
                inputs = inputs[1:]
            intermed_dtypes.insert(0, stage.spec.output.dtype)
        del intermed_dtypes[0]  # The head dtype will the output dtype, so drop it
        output = self.stages[-1].spec.output
        return specs.Compose(
            tuple(subspec_classes),
            inputs=inputs,
            output=output,
            intermediate_dtypes=tuple(intermed_dtypes),
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
    def children(self) -> Tuple["Impl", ...]:
        return self.stages

    @assert_stable_spec
    def subschedule(self, idx: int, fn: Callable[[Impl], Impl]) -> "Pipeline":
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
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "Impl":
        raise NotImplementedError(
            "move_input should usually be called on ComposeHole, not Pipeline"
        )

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "Impl":
        raise NotImplementedError(
            "move_output should usually be called on ComposeHole, not Pipeline"
        )

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        raise NotImplementedError("Pipeline has no actions because it is never a leaf")

    @assert_stable_spec
    def complete(self) -> Impl:
        return dataclasses.replace(
            self, stages=tuple(s.complete() for s in self.stages)
        )

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
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
    ) -> "Impl":
        raise NotImplementedError()

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        # Initialize peaks to dependencies of the first stage, which is just its
        # output
        first_peak = {k: 0 for k in system_config.current_system().banks}
        first_peak[self.stages[0].output.bank] = self.stages[0].output.bytes_used

        middle_peaks: list[dict[str, int]] = []
        for stage_idx in range(1, len(self.stages) - 1):
            before = self.stages[stage_idx - 1].output
            after = self.stages[stage_idx].output
            stage_mem = {b: 0 for b in system_config.current_system().banks}
            stage_mem[before.bank] += before.bytes_used
            stage_mem[after.bank] += after.bytes_used
            middle_peaks.append(stage_mem)

        last_peak = {k: 0 for k in system_config.current_system().banks}
        last_peak[self.stages[-2].output.bank] = self.stages[-2].output.bytes_used

        return [first_peak] + middle_peaks + [last_peak]

    @property
    def peak_memory(self) -> dict[str, int]:
        # Pipeline currently adds an intermediate tensor between each stage, so
        # intermediates is just the output of everything but the last stage
        intermediates = [o.output for o in self.stages[:-1]]
        intermed_utils: list[dict[str, int]] = []
        for tensor in intermediates:
            assert isinstance(tensor, Tensor)
            new_mem = {k: 0 for k in system_config.current_system().banks}
            new_mem[tensor.bank] += tensor.bytes_used
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


def _zipply(fn: Callable[[tuple[U]], V], *args: dict[T, U]) -> dict[T, V]:
    if not args:
        return {}
    return {
        k: fn(v) for k, v in utils.zip_dict(args[0], *args[1:], same_keys=True).items()
    }