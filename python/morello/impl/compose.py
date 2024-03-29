import dataclasses
import functools
import itertools
import math
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from .. import layouts, specs, system_config, tiling, utils
from ..layouts import Layout
from ..utils import TinyMap, snap_availables_up
from ..system_config import current_system, current_target
from ..tensor import OperandIdx, SimpleTile, TensorBase, TensorLike, Tile
from .actions import PeelAction, SlidingTileOutAction, TileOutAction
from .base import AppliedImpl, Impl, make_applied_impl, spec_to_hole
from .loops import Loop
from .moves import common_move, common_operand_move_actions
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
Tt = TypeVar("Tt", bound=TensorLike)


class SplitNotSupportedByHeadError(NotImplementedError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class ComposeHole(Impl):
    spec: specs.Compose

    def __post_init__(self):
        assert isinstance(self.spec, specs.Compose)

    @property
    def children(self) -> Tuple[Impl, ...]:
        return tuple()

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
        peeling_shape = self.spec.subspec_outputs[1]
        for bank, layout in itertools.product(
            system.addressed_banks, target.all_layouts_for_shape(peeling_shape)
        ):
            # TODO: The below special-casing for VMEM stinks. If we had partial
            #  finite functions/lenses, we could just constrain a few properties and
            #  enumerate the rest. It would also be possible to generate
            #  Tensor-making callables for the peel to populate in a generic way.
            peel_kwargs = [{}]
            bank_vec_bytes = system.banks[bank].vector_bytes
            if bank_vec_bytes:
                peel_kwargs = (
                    {"vector_shape": shape}
                    for shape in gen_vector_shapes(
                        self.spec.subspec_outputs[1],
                        self.spec.intermediate_dtypes[0],
                        bank_vec_bytes,
                    )
                )
            for kws in peel_kwargs:
                if self._can_peel(bank=bank, layout=layout, **kws):
                    yield PeelAction(self, bank=bank, layout=layout, kwargs=kws)

        yield from common_operand_move_actions(self)

        for tile_shape in gen_tile_sizes(
            self.spec.output.dim_sizes, filter=self._can_tile_out
        ):
            for parallel in [False] if self.spec.serial_only else [True, False]:
                yield TileOutAction(self, tile_shape, parallel)

        # TODO: Don't just use the first input. This isn't general to arbitrary
        # slideable ops in a Pipeline, or even, for instance, if we swapped the
        # operand order of Convolution/ConvHole. Besides, the dimensions
        # passed to sliding_tile_out are over the *output* Tile, not either
        # input; this only works because they're one-to-one for ConvHole.
        if allow_sliding_windows.get():
            for slide_dim in self._get_output_dimensions_matching_first_input():
                for slide_size in dim_range(
                    self.spec.output.dim_sizes[slide_dim], include_end=False
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
                yield functools.partial(self.split, k, parallel=False)
                if not self.spec.serial_only:
                    yield functools.partial(self.split, k, parallel=True)

    def _get_output_dimensions_matching_first_input(self) -> Iterable[int]:
        """Returns the output dimensions zipped with the first head input."""
        # Get the index of the Compose input which matches the first input to
        # the innermost (first executed) Impl.
        compose_subscripts = self.spec.operands_dim_subscripts()
        first_head_idx = -self.spec.subspec_classes[-1].inputs_count() - 1
        first_head_subs = compose_subscripts[first_head_idx]
        output_subs = compose_subscripts[-1]
        for sub in first_head_subs:
            try:
                yield output_subs.index(sub)
            except ValueError:
                pass

    @assert_stable_spec
    def move(
        self,
        operand_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ):
        return common_move(self, operand_idx, bank, layout, prefetching, **kwargs)

    def _can_peel(self, bank: str, layout: Layout, **kwargs) -> bool:
        # Check if we can peel by just trying to make the intermediate tensor that peel
        # would make and seeing if we get a ValueError. This isn't a great solution:
        # catching all ValueErrors might become overbroad as the code evolves, and the
        # object construction is inefficient and unneeded. However, it'll work for now.
        if bank not in current_system().addressed_banks:
            return False
        peeling_shape = self.spec.subspec_outputs[1]
        intermediate_tensor_layout = layout
        if all(d == 1 for d in peeling_shape):
            intermediate_tensor_layout = layouts.row_major(len(peeling_shape))
        try:
            current_target().tensor(
                current_target().tensor_spec(
                    dim_sizes=peeling_shape,
                    dtype=self.spec.intermediate_dtypes[0],
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

        if bank not in current_system().addressed_banks:
            raise ValueError(f"Cannot peel into non-addressed bank {bank}")

        # TODO: Using ALPHABET_PRODUCT here will fail for long programs
        peeling_shape = self.spec.subspec_outputs[1]
        intermediate_tensor_layout = layout
        if all(d == 1 for d in peeling_shape):
            intermediate_tensor_layout = layouts.row_major(len(peeling_shape))
        intermediate_tensor = current_target().tensor(
            current_target().tensor_spec(
                dim_sizes=self.spec.subspec_outputs[1],
                dtype=self.spec.intermediate_dtypes[0],
                bank=bank,
                layout=intermediate_tensor_layout,
                **kwargs,
            ),
            name="buf"
            + utils.ALPHABET_PRODUCT[len(self.spec.subspec_classes) - 2].upper(),
        )

        # The head of a Compose corresponds to the last function evaluated
        head_inps: tuple[specs.TensorSpec, ...] = (intermediate_tensor.spec,)
        hi = self.spec.subspec_classes[0].inputs_count() - 1
        if hi:
            head_inps += self.spec.inputs[:hi]
        head_hole = spec_to_hole(
            self.spec.subspec_classes[0].from_io(
                head_inps, self.spec.output, serial_only=self.spec.serial_only
            )
        )

        if len(self.spec.subspec_classes) == 2:
            remainder = spec_to_hole(
                self.spec.subspec_classes[1].from_io(
                    self.spec.inputs[hi:],
                    intermediate_tensor.spec,
                    serial_only=self.spec.serial_only,
                )
            )
        else:
            remainder = ComposeHole(
                specs.Compose(
                    subspec_classes=self.spec.subspec_classes[1:],
                    inputs=self.spec.inputs[hi:],
                    output=intermediate_tensor.spec,
                    intermediate_dtypes=self.spec.intermediate_dtypes[1:],
                    serial_only=self.spec.serial_only,
                )
            )
        return Pipeline((remainder, head_hole), intermediates=(intermediate_tensor,))

    @assert_stable_spec
    def tile_out(self, output_shape: tuple[int, ...], parallel=False) -> Impl:
        if parallel and self.spec.serial_only:
            raise ValueError("Serial-only Spec prevents parallel tiling")

        # A no-op if the given shape is already the output shape.
        if self.spec.output.dim_sizes == output_shape:
            return self

        # First, tile self.output.
        shrunken_output_tile = self.spec.output.simple_tile(
            OperandIdx(len(self.spec.inputs)), output_shape
        )
        assert isinstance(shrunken_output_tile, SimpleTile)
        # Compute new, reified Tiles for the shrunken ComposeHole. Works by computing
        # PartialTiles for the smaller ComposeHole, then applying those to self.inputs,
        # starting with the new output tile.
        reified_inputs = tuple(
            partial_inp.tile(OperandIdx(op_idx), inp)
            for op_idx, (inp, partial_inp) in enumerate(
                zip(
                    self.spec.inputs,
                    self._calculate_partial_inputs_for_tile_out(shrunken_output_tile),
                )
            )
        )

        # Construct the spec for the smaller ComposeHole
        return Loop(
            spec=self.spec,
            subscripts=self.spec.operands_dim_subscripts()[-1],
            tiles=frozenset([shrunken_output_tile])
            | frozenset(
                cast(Tile, t) for _, t in self._filter_unchanged_inputs(reified_inputs)
            ),
            inner=ComposeHole(
                specs.Compose(
                    subspec_classes=self.spec.subspec_classes,
                    inputs=tuple(inp.spec for inp in reified_inputs),
                    output=shrunken_output_tile.spec,
                    intermediate_dtypes=self.spec.intermediate_dtypes,
                    serial_only=(parallel or self.spec.serial_only),
                )
            ),
            parallel=parallel,
        )

    @assert_stable_spec
    def split(self, k: int, parallel=False) -> Impl:
        # TODO: Can we abstract over both Matmul and Reduce' splits?
        if self.spec.subspec_classes[0] == specs.ReduceSumAccum:
            return self._split_reduce_head(k, parallel=parallel)
        raise SplitNotSupportedByHeadError()

    def split_sizes(self) -> Iterable[int]:
        if self.spec.subspec_classes[0] in (specs.ReduceSum, specs.ReduceSumAccum):
            # TODO: This should defer to the inner op
            for k in dim_range(self.spec.subspec_outputs[1][-1], include_end=False):
                if k != self.spec.subspec_outputs[1][-1]:
                    yield k

    def _split_reduce_head(self, k: int, parallel: bool) -> Impl:
        assert self.spec.subspec_classes[0] in (specs.ReduceSum, specs.ReduceSumAccum)

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
            cast(Tile, partial_inp.tile(OperandIdx(op_idx), inp))
            for op_idx, (inp, partial_inp) in enumerate(
                zip(
                    self.spec.inputs,
                    self._calculate_partial_inputs_for_tile_out(
                        smaller_partial_input_tile, skip_first=1
                    ),
                )
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
            if inp.steps(self.spec.operands[inp_idx].dim_sizes) == expected_steps:
                driving_subs = self.spec.operands_dim_subscripts()[inp_idx]
                break
        assert driving_subs, (
            f"No tile had expected number of steps ({expected_steps}); "
            f" steps were {[i.steps for _, i in filtered_reified_inputs]}"
        )

        # Build the loop
        new_inner = ComposeHole(
            specs.Compose(
                subspec_classes=self.spec.subspec_classes,
                inputs=tuple(inp.spec for inp in reified_inputs),
                output=self.spec.output,
                intermediate_dtypes=self.spec.intermediate_dtypes,
                serial_only=(parallel or self.spec.serial_only),
            )
        )
        return Loop(
            spec=self.spec,
            subscripts=driving_subs,
            tiles=frozenset(t for _, t in filtered_reified_inputs),
            inner=new_inner,
            parallel=parallel,
        )

    def _filter_unchanged_inputs(
        self, source: Iterable[Tt]
    ) -> Iterable[tuple[int, Tt]]:
        """Return given tensors different from self's corresponding inputs."""
        for idx, (orig_spec, tiled_input) in enumerate(zip(self.spec.inputs, source)):
            if orig_spec != tiled_input.spec:
                yield idx, tiled_input

    def _calculate_partial_inputs_for_tile_out(
        self, output_tile: Union[SimpleTile, tiling.PartialTile], skip_first: int = 0
    ) -> list[tiling.PartialTile]:
        """Returns PartialTiles for this ComposeHole's inputs for an output tiling.

        Accepts either a PartialTile or a Tile which will be converted into a PartialTile.
        """
        subspec_classes = list(self.spec.subspec_classes)
        intermediate_shapes = list(self.spec.subspec_outputs[1:])
        inputs = list(self.spec.inputs)
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
            tuple(inp.dim_sizes for inp in inputs),
            output_tile,
        )

    @staticmethod
    def _compute_partial_inputs_inner(
        subspec_classes: tuple,
        intermediate_shapes: Iterable[tuple[int, ...]],
        flattened_inputs_shapes: tuple[tuple[int, ...], ...],
        output_tile: Union[SimpleTile, tiling.PartialSimpleTile],
    ) -> list[tiling.PartialTile]:
        if isinstance(output_tile, Tile):
            partial_output_tile = tiling.tile_to_partial(output_tile)
            assert isinstance(partial_output_tile, tiling.PartialSimpleTile)
        else:
            partial_output_tile = output_tile

        # We would normally need to do a forward pass first to produce the
        # output shapes so we know what the non-final subspecs' first operand
        # shapes are, but this is already implemented by the subspec_outputs
        # property of Compose.
        subspec_output_shapes = list(intermediate_shapes)

        input_tiles: Optional[tuple[tiling.PartialTile, ...]] = None
        all_input_tiles: list[tiling.PartialTile] = []
        for idx, subspec_cls in enumerate(subspec_classes):
            inputs_shapes = ()
            if subspec_output_shapes:
                inputs_shapes = (subspec_output_shapes.pop(0),)
            take = subspec_cls.inputs_count() - len(inputs_shapes)
            inputs_shapes += flattened_inputs_shapes[:take]
            flattened_inputs_shapes = flattened_inputs_shapes[take:]
            # We're tracing the type and shape of each subspec's first tile up through
            # the pipeline of composed functions, so store the first into
            # partial_output_tile, which will be an input the next time around the loop.
            # At the end, we'll want input_tiles.
            input_tiles = tiling.tile_out(
                subspec_cls, inputs_shapes, partial_output_tile
            )
            if idx == len(subspec_classes) - 1:
                all_input_tiles.extend(input_tiles)
            else:
                all_input_tiles.extend(input_tiles[1:])
            # Because Compose applies the output of a stage to the following stage's
            # first argument, we carry the first input tile into the next iteration.
            partial_output_tile = input_tiles[0]

        return all_input_tiles

    @assert_stable_spec
    def complete(self) -> "Impl":
        next_bank = system_config.current_system().default_bank
        shape_to_peel = self.spec.subspec_outputs[1]
        l = layouts.row_major(len(shape_to_peel))
        return self.peel(bank=next_bank, layout=l).complete()

    def replace_spec(self, new_spec: specs.Spec) -> "Impl":
        assert type(self) is ComposeHole
        return ComposeHole(new_spec)

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        if replacements:
            raise Exception("Holes have no children to replace")
        return self

    @property
    def is_scheduled(self) -> bool:
        return False

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        return make_applied_impl(self, operands)


@dataclasses.dataclass(frozen=True, init=False)
class Pipeline(Impl):
    """A sequence of schedules, called stages, which are executed in order.

    The output of the pipeline is the output of the final stage.
    """

    stages: tuple[Impl, ...]
    intermediates: tuple[TensorBase, ...]

    def __init__(self, stages: tuple[Impl, ...], intermediates: tuple[TensorBase, ...]):
        assert len(stages) >= 2
        assert len(stages) == len(intermediates) + 1
        # TODO: Reintroduce check for operand agreement
        # for before, after in zip(self.stages[:-1], self.stages[1:]):
        #     assert (
        #         before.output == after.operands[0]
        #     ), f"Output of {before} didn't match first operand of {after}"

        # Flatten any immediate Pipeline children and intermediate tensors
        flattened_stages = []
        flattened_intermediates = []
        for stage, intermed in zip(stages, intermediates):
            if isinstance(stage, Pipeline):
                flattened_stages.extend(stage.stages)
                flattened_intermediates.extend(stage.intermediates)
            else:
                flattened_stages.append(stage)
            flattened_intermediates.append(intermed)
        if isinstance(stages[-1], Pipeline):
            flattened_stages.extend(cast(Pipeline, stages[-1]).stages)
            flattened_intermediates.extend(cast(Pipeline, stages[-1]).intermediates)
        else:
            flattened_stages.append(stages[-1])
        object.__setattr__(self, "stages", tuple(flattened_stages))
        object.__setattr__(self, "intermediates", tuple(flattened_intermediates))

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
    def children(self) -> Tuple[Impl, ...]:
        return self.stages

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

    @property
    def memory_allocated(self) -> tuple[TinyMap[str, int], list[TinyMap[str, int]]]:
        stages = self.stages
        banks = system_config.current_system().ordered_banks

        # Initialize peaks to dependencies of the first stage, which is just its
        # output
        output_lims: list[int] = [0] * len(banks)
        output_lims[banks.index(stages[0].spec.output.bank)] = stages[
            0
        ].spec.output.bytes_used

        middle_peaks: list[list[int]] = []
        for stage_idx in range(1, len(self.stages) - 1):
            before = self.stages[stage_idx - 1].spec.output
            after = self.stages[stage_idx].spec.output
            stage_mem: list[int] = [0] * len(banks)
            stage_mem[banks.index(before.bank)] += before.bytes_used
            stage_mem[banks.index(after.bank)] += after.bytes_used
            middle_peaks.append(stage_mem)

        last_peak: list[int] = [0] * len(banks)
        last_peak[banks.index(self.stages[-2].spec.output.bank)] = self.stages[
            -2
        ].spec.output.bytes_used

        peaks = [output_lims] + middle_peaks + [last_peak]
        z = TinyMap(banks, (0,) * len(banks))
        return z, [TinyMap(banks, tuple(p)) for p in peaks]

    @property
    def is_scheduled(self) -> bool:
        return all(op.is_scheduled for op in self.stages)

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        inputs, output = tuple(operands[:-1]), operands[-1]

        applied_stages = []
        applied_stages.append(
            self.stages[0].apply(
                inputs[-len(self.stages[0].spec.inputs) :] + (self.intermediates[0],)
            )
        )
        inputs = inputs[: -len(self.stages[0].spec.inputs)]
        for stage_idx in range(1, len(self.stages) - 1):
            applied_stages.append(
                self.stages[-1].apply(
                    (self.intermediates[stage_idx],)
                    + inputs[1 + -len(self.stages[-1].spec.inputs) :]
                    + (output,)
                )
            )
            inputs = inputs[: 1 + -len(self.stages[-1].spec.inputs)]
        applied_stages.append(
            self.stages[-1].apply((self.intermediates[-1],) + inputs + (output,))
        )

        return make_applied_impl(
            Pipeline(stages=applied_stages, intermediates=self.intermediates), operands
        )


def _zipply(fn: Callable[[tuple[U, ...]], V], *args: Mapping[T, U]) -> dict[T, V]:
    if not args:
        return {}
    return {k: fn(v) for k, v in utils.zip_dict(args[0], *args[1:], same_keys=True)}
