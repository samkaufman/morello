import dataclasses
import contextvars
import itertools
import re
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import sympy

from ..tensor import Tensor, TensorLike, Tile
from . import common, indexexpr
from .common import OperandDetails
from .indexexpr import set_subgroup, unset_subgroup, vsub

_IRE = re.compile(r"i(\d+)")

BOUNDARY_ANCESTORS: contextvars.ContextVar[int] = contextvars.ContextVar(
    "BOUNDARY_ANCESTORS", default=0
)


@dataclasses.dataclass(frozen=True)
class OperandDetailsLoopExt(OperandDetails):
    inner_subscripts: tuple[int, ...]
    tiled_operand: TensorLike


@dataclasses.dataclass(frozen=True)
class _LoopNestDescription:
    """Structures returned by _compute_tile_out_loop_nest."""

    subscripts_to_steps: Sequence[tuple[int, int]]  # in emit order
    body_index_exprs: Sequence[sympy.Expr]
    body_shapes: Sequence[tuple[int, ...]]
    is_boundary: bool


def emit_tile_out_loop_nest(
    all_subscripts: set[int],
    remaining_subscripts: list[int],  # Reduces each step; base case = empty
    outer_operands: Sequence[TensorLike],
    inner_operands: Sequence[TensorLike],
    op_details: Sequence[OperandDetailsLoopExt],
    parallel: bool,
    inner_codegen: Callable[[Sequence[OperandDetails]], None],
) -> None:
    namer, writer = common.namer.get(), common.writer.get()

    # Generate new names for loop iterators.
    it_var_names: Optional[defaultdict[int, str]] = None
    if not common.unroll.get():
        it_var_names = defaultdict(lambda: namer.fresh_name("t"))

    # Emit loops.
    for loop_plan in _compute_tile_out_loop_nest(
        all_subscripts,
        remaining_subscripts,
        it_var_names,
        outer_operands,
        inner_operands,
        op_details,
    ):
        assert it_var_names is not None

        if len(loop_plan.subscripts_to_steps):
            if parallel and not loop_plan.is_boundary:
                writer.writeline(
                    "#pragma omp parallel for "
                    f"collapse({len(loop_plan.subscripts_to_steps)}) "
                    "schedule(static)"
                )
            for sub, steps in loop_plan.subscripts_to_steps:
                assert steps > 0
                it_var = it_var_names[sub]
                writer.writeline(
                    f"for (int {it_var} = 0; {it_var} < {steps}; {it_var}++) {{"
                )
            writer.indent()

        BOUNDARY_ANCESTORS.set(BOUNDARY_ANCESTORS.get() + 1)
        try:
            inner_codegen(
                [
                    OperandDetails(
                        d.c_tensor, e, tuple(s), d.previously_transformed_tiles
                    )
                    for d, e, s in zip(
                        op_details, loop_plan.body_index_exprs, loop_plan.body_shapes
                    )
                ]
            )
        finally:
            BOUNDARY_ANCESTORS.set(BOUNDARY_ANCESTORS.get() - 1)
        if len(loop_plan.subscripts_to_steps):
            writer.dedent()
            for _ in loop_plan.subscripts_to_steps:
                writer.writeline("}")


def _compute_tile_out_loop_nest(
    all_subscripts: set[int],
    remaining_subscripts: list[int],
    it_var_names: Optional[Mapping[int, str]],
    outer_operands: Sequence[TensorLike],
    applied_operands: Sequence[TensorLike],
    op_details: Sequence[OperandDetailsLoopExt],
) -> Iterable[_LoopNestDescription]:
    """Yields loop nest plans, including main and boundary cases.

    This function doesn't emit any code, but does decide which loop nests need
    to be produced, over which subscripts, and in what order. It also produces
    updated indexing expressions for boundary cases.
    """
    # The following would be much faster if we sharing prefixes and didn't
    # enumerate combos where boundaries are impossible.
    if common.unroll.get():
        raise NotImplementedError()
    assert it_var_names is not None

    # Compute steps and drop subscripts that have no full steps.
    # We do this first so that we know how many loops we're going to
    # emit.

    # Determine the number of loop steps for all subscripts, and build
    # `emitting_subscripts` for anything with more than one full step (these are
    # the subscripts for which we'll write code in the loop nests).
    #
    # `emitting_subscripts` is built ahead of time so that we know how many loop
    # nests, each corresponding to a boundary configuration, we'll emit.
    emitting_subscripts: list[tuple[int, int, int]] = []
    for subscript in remaining_subscripts:
        full_steps, has_boundary = _calc_steps(subscript, op_details)
        if full_steps == 0:
            continue
        # if full_steps == 1 and not has_boundary:
        #     continue
        assert full_steps > 0, f"full_steps cannot be {full_steps} in an emitting loop"
        emitting_subscripts.append((subscript, full_steps, has_boundary))

    # Range over the loop nests to emit. Each entry in boundary_config
    # corresponds to a unique subscript (group of zipped dimensions) and whether
    # or not we're considering a loop which implements that subscript's
    # boundary case.
    for boundary_config in itertools.product(
        [False, True], repeat=len(emitting_subscripts)
    ):
        # Ignore any combo. where the config. is set to emit a boundary case for
        # a subscript that doesn't have a boundary.
        if any(
            b and not hb for b, (_, _, hb) in zip(boundary_config, emitting_subscripts)
        ):
            continue

        subscripts_to_loop_over = [
            (sub, full_steps)
            for subscript_is_boundary, (sub, full_steps, _) in zip(
                boundary_config, emitting_subscripts
            )
            if not subscript_is_boundary
        ]

        all_subscript_details = {}
        for (s, full_steps, _), in_boundary in zip(
            emitting_subscripts, boundary_config
        ):
            if in_boundary:
                all_subscript_details[s] = _SubscriptDetails(full_steps)
            elif full_steps == 1:
                all_subscript_details[s] = _SubscriptDetails(0)
            else:
                all_subscript_details[s] = _SubscriptDetails(it_var_names[s])

        new_index_exprs = []
        for deets, outer_operand, applied_operand in zip(
            op_details, outer_operands, applied_operands
        ):
            # The following check is important to avoid updating indexing
            # expressions twice for the same operand, such as when an operand is
            # passed through to a child unchanged.
            if outer_operand == applied_operand:
                new_index_exprs.append(deets.index_expr)
            else:
                new_index_exprs.append(
                    _update_index_expr(deets, applied_operand, all_subscript_details)
                )

        concrete_shapes = [list(d.concrete_origin_shape) for d in op_details]
        for subscript_is_boundary, (sub, full_steps, _) in zip(
            boundary_config, emitting_subscripts
        ):
            set_subgroup("   sub: " + str(sub))

            # Edit `concrete_shape` to reflect whichever subscripts are currently
            # boundaries.
            for op_idx, deets in enumerate(op_details):
                for sidx, s in enumerate(deets.inner_subscripts):
                    if s == sub:
                        if not isinstance(deets.tiled_operand, Tile):
                            new_size = deets.tiled_operand.dim_sizes[sidx]
                        else:
                            new_size = (
                                deets.tiled_operand.boundary_size(
                                    sidx, deets.concrete_origin_shape[sidx]
                                )
                                if subscript_is_boundary
                                else deets.tiled_operand.dim_sizes[sidx]
                            )
                        concrete_shapes[op_idx][sidx] = min(
                            concrete_shapes[op_idx][sidx], new_size
                        )
            unset_subgroup()

        # TODO: Remove the following checks
        # if all(len(a) == 4 for a in concrete_shapes):
        #     assert concrete_shapes[0][0] == concrete_shapes[2][0]
        #     assert concrete_shapes[1][0] == concrete_shapes[2][1]
        #     assert concrete_shapes[0][1] == concrete_shapes[1][1]

        yield _LoopNestDescription(
            subscripts_to_loop_over,
            new_index_exprs,
            list(map(tuple, concrete_shapes)),
            is_boundary=any(boundary_config),
        )


def _calc_steps(
    it_subscript: int,
    op_details: Sequence[OperandDetailsLoopExt],
    pass_through_tensors: bool = False,
) -> tuple[int, bool]:
    # The following gathers the number of steps (both full and boundary) as well
    # as whether or not a boundary tile exists. While boundary tile sizes
    # differ, presence of one should be consistent across all operands.
    # As a bit of defensive programming, the following also checks that this
    # is consistent across all operands and matching subscripts.
    partial_steps = None
    has_boundary = None
    for details in op_details:
        if not isinstance(details.tiled_operand, Tile) and not pass_through_tensors:
            continue
        for dim, sub in enumerate(details.inner_subscripts):
            if sub != it_subscript:
                continue
            if not isinstance(details.tiled_operand, Tile) and pass_through_tensors:
                new_partial_steps = 1
                new_has_boundary = False
            else:
                new_partial_steps = details.tiled_operand.steps_dim(
                    dim, details.concrete_origin_shape[dim]
                )
                new_has_boundary = bool(
                    details.tiled_operand.boundary_size(
                        dim, details.concrete_origin_shape[dim]
                    )
                )
            if partial_steps is None:
                partial_steps = new_partial_steps
                has_boundary = new_has_boundary
            assert partial_steps == new_partial_steps, (
                f"Inconsistent partial steps for {it_subscript}. Found both "
                f"{partial_steps} and {new_partial_steps}."
            )
            assert new_has_boundary == has_boundary
    assert isinstance(
        partial_steps, int
    ), f"partial_steps was not an int; was: {partial_steps}"
    assert isinstance(has_boundary, bool)

    full_steps = partial_steps - 1 if has_boundary else partial_steps

    return full_steps, has_boundary


@dataclasses.dataclass(frozen=True)
class _SubscriptDetails:
    it_term: Union[int, str]


def _update_index_expr(
    deets: OperandDetailsLoopExt,
    applied_operand: TensorLike,
    subscripts: Mapping[int, _SubscriptDetails],  # TODO: Just map to replacement
) -> sympy.Expr:
    assert len(deets.inner_subscripts) == len(deets.concrete_origin_shape)

    # Prepare tile name-binding substitutions for each operand. These will
    # replace each i_ symbol in the new logical expressions with the target name
    # or a constant.
    #
    # TODO: Document: Does this need to be all dimensions, or just the emitted?
    binding_subs = []
    for dim in range(len(deets.concrete_origin_shape)):
        sub = deets.inner_subscripts[dim]
        if sub in subscripts:
            replacement = subscripts[sub].it_term
            if isinstance(replacement, str):
                replacement = "_" + replacement
        else:
            replacement = 0
        binding_subs.append((f"i{dim}", replacement))

    # Replace the point ('p_') symbols in the operand's current indexing
    # expressions with the updated logical expressions which have tiles bound
    # to target loop variable names. Do this for all dims, whether or not they
    # are being iterated over.
    expr_subs = {}
    for dim in range(len(deets.concrete_origin_shape)):
        # TODO: Do we need the following short-circuit?
        # if not any(s.name == f"p{dim}" for s in deets.index_expr.free_symbols):
        #     continue
        new_logical = indexexpr.logical_indexing_expr(applied_operand, dim)
        expr_subs[f"p{dim}"] = vsub(new_logical, binding_subs)
        assert not any(_IRE.match(s.name) for s in expr_subs[f"p{dim}"].free_symbols)
    return vsub(deets.index_expr, expr_subs)