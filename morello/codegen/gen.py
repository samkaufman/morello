import contextlib
import dataclasses
import functools
import operator
import string
from collections.abc import Sequence
from typing import Callable, Literal, Optional, Union, cast

import sympy

from .. import impl, utils
from ..dtypes import Dtype
from ..system_config.cpu import X86Target, ArmTarget
from ..tensor import Tensor
from ..system_config.state import current_target
from . import common, expr_utils
from .common import OperandDetails
from .ctensors import (
    GCC_VEC_TYPES,
    CHeapArray,
    CPtr,
    CStackArray,
    CTensor,
    CValueVar,
    CVecVars,
)
from .indexexpr import vsub
from .loops import BOUNDARY_ANCESTORS, OperandDetailsLoopExt, emit_tile_out_loop_nest

STACK_CUTOFF = 256
BENCH_ITERS = 10


def _emit_tensor_print(
    buffer_name: str,
    buffer_ref_fn: Callable[[Union[sympy.Expr, int]], str],
    tensor_shape: Union[tuple[int, int], tuple[int, int, int]],
    dtype: Dtype,
    index_expr: sympy.Expr,
    writer: "common.Writer",
    write_name=True,
) -> None:
    rank = len(tensor_shape)
    assert all(s.name[0] == "p" for s in index_expr.free_symbols)

    writer.writeline("// Print " + buffer_name)
    writer.writeline("{")
    with writer.indent_block():
        if write_name:
            writer.writeline('printf("\\n");')
            writer.writeline(f'printf("{buffer_name}\\n");')

        shape_str = "x".join(str(v) for v in tensor_shape)
        writer.writeline(f'printf("{shape_str}\\n");')

        for idx in range(rank):
            sym = sympy.symbols(f"p{idx}")
            idx = int(sym.name[1:])
            it_name = string.ascii_letters[idx]
            size = tensor_shape[idx]
            writer.writeline(
                f"for (size_t {it_name} = 0; {it_name} < {size}; {it_name}++) {{"
            )
            index_expr = vsub(index_expr, sym, f"_{it_name}")

        with writer.indent_block():
            writer.writeline(
                f'printf("%" {dtype.int_fmt_macro} " ", {buffer_ref_fn(index_expr)});'
            )

        if rank:
            writer.writeline("}")
        for idx in range(rank - 1):
            writer.writeline(f'printf("\\n");')
            writer.writeline("}")
    writer.writeline("}")


def _make_buffer(
    shape: tuple[int, ...],
    vector_shape: Optional[tuple[int, ...]],
    dtype: Dtype,
    bank: str,
) -> CTensor:
    if bank == "VRF" and not BOUNDARY_ANCESTORS.get():
        assert vector_shape
        return CVecVars(common.namer.get(), shape, vector_shape, dtype)

    name = common.namer.get().fresh_name("buf")
    size = functools.reduce(operator.mul, shape, 1)
    if (size * dtype.size) > STACK_CUTOFF:
        return CHeapArray(name, size, dtype)
    elif size > 1:
        return CStackArray(name, size, dtype)
    else:
        return CValueVar(name, dtype)


@contextlib.contextmanager
def _emit_loop_nest_for_shape(shape: Sequence[int]):
    namer, writer = common.namer.get(), common.writer.get()
    subs = []
    for dim_size in shape:
        n = namer.fresh_name("i")
        if dim_size > 1:
            writer.writeline(f"for (int {n} = 0; {n} < {dim_size}; {n}++) {{")
            writer.indent()
            subs.append(n)
        else:
            subs.append(0)
    yield subs
    for dim_size in shape:
        if dim_size > 1:
            writer.dedent()
            writer.writeline("}")


def _inner_generate_c(
    imp: impl.AppliedImpl, op_details: Sequence[OperandDetails], allow_holes: bool
):
    assert len(op_details) == len(
        imp.operands
    ), f"Expected {len(imp.operands)} OperandDetails, got {len(op_details)}"

    namer, writer = common.namer.get(), common.writer.get()

    writer.set_impl(imp)

    # Transform all concrete shapes if we encounter a new tile.
    # (This feels hacky, but goes away if we add boundary Impls.)
    transformed_op_details = []
    for o, d in zip(imp.operands, op_details):
        if o in d.previously_transformed_tiles:
            transformed_op_details.append(d)
            continue
        transformed_op_details.append(
            dataclasses.replace(
                d,
                index_expr=o.transform_index_expr(d.index_expr),
                concrete_origin_shape=o.transform_origin_shape(d.concrete_origin_shape),
                previously_transformed_tiles=d.previously_transformed_tiles | {o},
            )
        )
    op_details = transformed_op_details

    if isinstance(imp, impl.Loop):
        # Collect because these don't necessarily match `imp.inner.operands`.
        # The latter might, for instance, transpose or squeeze the tiles.
        tiled_operands = list(imp.operands)
        for tile in imp.tiles:
            tiled_operands[tile.source] = tile

        emit_tile_out_loop_nest(
            {s for os in imp.operands_subscripts for s in os},
            list(imp.subscripts),
            imp.operands,
            imp.inner.operands,
            [
                OperandDetailsLoopExt(
                    o.c_tensor,
                    o.index_expr,
                    o.concrete_origin_shape,
                    o.previously_transformed_tiles,
                    op_subs,
                    inner_op,
                )
                for o, op_subs, inner_op in zip(
                    op_details,
                    imp.operands_subscripts,
                    tiled_operands,
                )
            ],
            imp.parallel,
            inner_codegen=lambda details: _inner_generate_c(
                imp.inner, details, allow_holes
            ),
        )
    elif isinstance(imp, impl.Block):
        for step, op_idxs in zip(imp.steps, imp.op_idxs):
            assert isinstance(step, impl.AppliedImpl)
            _inner_generate_c(step, [op_details[i] for i in op_idxs], allow_holes)
    elif isinstance(imp, impl.SpecCast):
        _inner_generate_c(imp.inner, op_details, allow_holes)
    elif isinstance(imp, impl.Pipeline):
        inps_op_details = op_details[:-1]
        output_op_details = op_details[-1]

        # First stage
        last_c_buf = _make_buffer(
            imp.stages[0].spec.output.dim_sizes,
            imp.stages[0].spec.output.vector_shape,
            imp.stages[0].spec.output.dtype,
            imp.stages[0].spec.output.bank,
        ).emit(zero_init=False)
        cur_slice = slice(-len(imp.stages[0].spec.inputs), len(inps_op_details))
        cur_out = _pipeline_emit_stage(
            imp.stages[0],
            inps_op_details[cur_slice],
            last_c_buf,
            None,
            None,
            allow_holes,
        )
        cur_slice = slice(
            1 + cur_slice.start - len(imp.stages[1].spec.inputs), cur_slice.start
        )

        # Intermediate stages
        for stage, next_stage in zip(imp.stages[1:], imp.stages[2:]):
            new_c_buf = _make_buffer(
                stage.spec.output.dim_sizes,
                stage.spec.output.vector_shape,
                stage.spec.output.dtype,
                stage.spec.output.bank,
            ).emit(zero_init=False)
            cur_out = _pipeline_emit_stage(
                stage, inps_op_details[cur_slice], new_c_buf, cur_out, None, allow_holes
            )
            last_c_buf.emit_free()
            last_c_buf = new_c_buf
            cur_slice = slice(
                1 + cur_slice.start - len(next_stage.spec.inputs), cur_slice.start
            )

        # Last stage
        cur_out = _pipeline_emit_stage(
            imp.stages[-1],
            inps_op_details[cur_slice],
            output_op_details.c_tensor,
            cur_out,
            output_op_details,
            allow_holes,
        )
        last_c_buf.emit_free()
        assert (
            cur_out.concrete_origin_shape == output_op_details.concrete_origin_shape
        ), "Final stage output shape didn't match Pipeline output shape"
    elif isinstance(imp, impl.VectorZero):
        if BOUNDARY_ANCESTORS.get() > 0:
            _naive_memset(op_details[0], "VectorZero", writer)
        else:
            expr = expr_utils.zero_points(op_details[0].index_expr)
            ten = op_details[0].c_tensor
            assert isinstance(ten, CVecVars), f"ten was not a CVecVars; was: {ten}"
            out_shape = op_details[0].concrete_origin_shape
            # No assignment syntax for Clang vector extensions yet, so we'll
            # multiply by zero, which should produce a setzero or be optimized away.
            writer.writeline(f"{ten.c_index_vec(expr)} *= 0;  /* VectorZero */")
    elif isinstance(imp, impl.MemsetZero):
        if BOUNDARY_ANCESTORS.get() > 0:
            _naive_memset(op_details[0], "MemsetZero", writer)
        else:
            expr = expr_utils.zero_points(op_details[0].index_expr)
            ten = op_details[0].c_tensor
            ref = ten.c_index_ptr
            vol = functools.reduce(operator.mul, op_details[0].concrete_origin_shape, 1)
            vol *= imp.spec.destination.dtype.size
            writer.writeline(
                f"memset((void *)({ref(expr)}), 0, {vol});  /* MemsetZero */"
            )
    elif isinstance(imp, impl.Mult):
        l_ref, r_ref, o_ref = (d.c_tensor.c_index for d in op_details)
        l, r, o = (d.index_expr for d in op_details)
        l = expr_utils.zero_points(l)
        r = expr_utils.zero_points(r)
        o = expr_utils.zero_points(o)
        writer.writeline(f"{o_ref(o)} += {l_ref(l)} * {r_ref(r)};  /* Mult */")
    elif isinstance(imp, impl.BroadcastVecMult):
        l, r, o = (d.index_expr for d in op_details)
        r = expr_utils.zero_points(r)
        o = expr_utils.zero_points(o)
        out_shape = op_details[2].concrete_origin_shape
        rhs_volume = functools.reduce(
            operator.mul, op_details[1].concrete_origin_shape, 1
        )
        if BOUNDARY_ANCESTORS.get() > 0:
            # In the boundary case, we can't safely apply the vectorized op. Tensors
            # might be misaligned or non-contiguous. So: insert a simple multiplication
            # loop instead.
            with _emit_loop_nest_for_shape(out_shape) as it_names:
                substitutions = {
                    f"p{dim}": (s if isinstance(s, int) else f"_{s}")
                    for dim, s in enumerate(it_names)
                }
                l, r, o = [vsub(d.index_expr, substitutions) for d in op_details]
                l_ref, r_ref, o_ref = (d.c_tensor.c_index for d in op_details)
                writer.writeline(
                    f"{o_ref(o)} += {l_ref(l)} * {r_ref(r)};"
                    "  /* Mult (vec boundary) */"
                )
        else:
            vtype = GCC_VEC_TYPES[(imp.spec.operands[2].dtype, rhs_volume)][1]
            writer.writeline(
                f"*({vtype} *)({op_details[-1].c_tensor.c_index_ptr(o)}) "
                "+= "
                f"{op_details[0].c_tensor.c_index(l)} * "
                f"(*({vtype} *)({op_details[1].c_tensor.c_index_ptr(r)}));"
                " /* BroadcastVecMult */"
            )
    elif isinstance(imp, impl.Add):
        assert all(d == 1 for d in imp.output.dim_sizes)
        i_ref, o_ref = [d.c_tensor.c_index for d in op_details]
        i, o = [d.index_expr for d in op_details]
        writer.writeline(f"{o_ref(o)} += {i_ref(i)};")
    elif isinstance(imp, impl.MoveLet):
        source_idx = imp.source_idx
        assert (
            imp.body.operands[source_idx] is imp.destination
        ), "MoveLet's body Impl does not use destination tensor"

        # TODO: Remove the following "destructuring" of OperandDetails
        operand_index_exprs = [d.index_expr for d in op_details]
        concrete_shapes = [d.concrete_origin_shape for d in op_details]

        concrete_shape = op_details[source_idx].concrete_origin_shape

        # Code is only generated (after the previous cases) for MoveLet if there is a
        # layout or contiguity change.  Otherwise, we just recurse.
        if imp.destination.bank in ("RF", "VRF"):
            _move_registers(
                imp,
                source_idx,
                [d.c_tensor for d in op_details],
                operand_index_exprs,
                concrete_shapes,
                [d.previously_transformed_tiles for d in op_details],
                allow_holes,
            )
        else:
            _inner_generate_c(
                cast(impl.AppliedImpl, cast(impl.MoveLet, imp).body),
                op_details,
                allow_holes,
            )
    elif isinstance(imp, impl.ValueAssign):
        l_ref, o_ref = (d.c_tensor.c_index for d in op_details)
        l, o = (d.index_expr for d in op_details)
        l = expr_utils.zero_points(l)
        o = expr_utils.zero_points(o)
        if imp.is_store:
            writer.writeline(f"{l_ref(l)} = {o_ref(o)};  // ValueAssign (store)")
        else:
            writer.writeline(f"{o_ref(o)} = {l_ref(l)};  // ValueAssign")
    elif isinstance(imp, impl.VectorAssign):
        shape = op_details[0].concrete_origin_shape
        assert shape == op_details[1].concrete_origin_shape
        if BOUNDARY_ANCESTORS.get() > 0:
            with _emit_loop_nest_for_shape(shape) as it_names:
                substitutions = {
                    f"p{dim}": (s if isinstance(s, int) else f"_{s}")
                    for dim, s in enumerate(it_names)
                }
                l, r = [vsub(d.index_expr, substitutions) for d in op_details]
                lhs_txt = op_details[0].c_tensor.c_index(l)
                rhs_txt = op_details[1].c_tensor.c_index(r)
                if not imp.is_store:
                    lhs_txt, rhs_txt = rhs_txt, lhs_txt
                writer.writeline(f"{lhs_txt} = {rhs_txt};  // VectorAssign boundary")
        else:
            vol = functools.reduce(operator.mul, shape, 1)
            l, r = (expr_utils.zero_points(d.index_expr) for d in op_details)
            lhs_txt = op_details[0].c_tensor.c_index_ptr(l)
            rhs_txt = op_details[1].c_tensor.c_index_ptr(r)
            if all(t.aligned for t in imp.spec.operands):
                vtype = GCC_VEC_TYPES[(imp.spec.operands[0].dtype, vol)][1]
                if not imp.is_store:
                    lhs_txt, rhs_txt = rhs_txt, lhs_txt
                writer.writeline(
                    f"*({vtype} *)({lhs_txt}) = "
                    f"(*({vtype} *)({rhs_txt}));  // VectorAssign"
                )
            else:
                itype, (load_inst, store_inst) = GCC_VEC_TYPES[
                    (imp.spec.operands[0].dtype, vol)
                ][2:]
                a, b = rhs_txt, lhs_txt
                if imp.is_store:
                    a, b = lhs_txt, rhs_txt
                writer.writeline(
                    f"{store_inst}(({itype} *)({a}), {load_inst}(({itype} *)({b})));  // VectorAssign"
                )
    elif not imp.is_scheduled and allow_holes:
        writer.writeline(f"/* HOLE */")
    else:
        raise NotImplementedError(f"Not implemented for {type(imp).__name__}")


def _pipeline_emit_stage(
    stage: impl.Impl,
    input_operand_details: Sequence[OperandDetails],
    output_c_tensor: CTensor,
    previous_output: Optional[OperandDetails],  # None only on first stage.
    final_output: Optional[OperandDetails],  # Non-None on last stage.
    allow_holes: bool,
) -> OperandDetails:
    """Emits code for one stage in a Pipeline.

    :param stage: The Impl for the stage.
    :param input_operand_details: OperandDetails for the Pipeline inputs consumed by
      this stage.
    :param output_c_tensor: The _CTensor into which this stage will write output.
    :param previous_output: OperandDetails describing the previous stage's output, if
      any. This should be the return value of the previous stage's _pipeline_emit_stage
      call; a caller should just forward it.
    :param final_output: The final output of the Pipeline, or `None` if the given stage
      is not the final stage in the pipeline.
    :return: An OperandDetails describing the output of this stage. This should be
      given to the next stage's _pipeline_emit_stage call as `previous_output`.
    """
    assert not final_output or final_output.c_tensor is output_c_tensor, (
        "final_output provided, which happens on last stage of a Pipeline, "
        "but output_c_tensor was not the same as final_output.c_tensor"
    )

    # Previous stage's output is the new stage's first input. Update operand
    # details to reflect that. This functions `sl` argument corresponds to the
    # slice of Pipeline inputs that are fed to this stage, excluding the input
    # from any previous stage. (On the first stage, `prev_output is None`.)
    cur_c_tensors = [d.c_tensor for d in input_operand_details] + [output_c_tensor]
    cur_index_exprs = [d.index_expr for d in input_operand_details]
    cur_concrete_shapes = [d.concrete_origin_shape for d in input_operand_details]
    cur_prev_transformeds = [
        d.previously_transformed_tiles for d in input_operand_details
    ]
    if previous_output:
        cur_c_tensors.insert(0, previous_output.c_tensor)
        cur_index_exprs.insert(0, previous_output.index_expr)
        cur_concrete_shapes.insert(0, previous_output.concrete_origin_shape)
        cur_prev_transformeds.insert(0, previous_output.previously_transformed_tiles)
    cur_concrete_shapes.append(stage.spec.calculate_output_shape(cur_concrete_shapes))

    # Complete cur_index_exprs with the output indexing expression. In the last stage of
    # a Pipeline, this final indexing expression is just passed in from the caller as
    # `output_index_expr`. In every other stage, this function computes it itself.
    if final_output:
        output_index_expr = final_output.index_expr
    else:
        assert isinstance(stage.output, Tensor)
        output_index_expr = stage.output.layout.buffer_indexing_expr(
            cur_concrete_shapes[-1]
        )
    cur_index_exprs.append(output_index_expr)

    # Also complete cur_prev_transformeds with the output.
    if final_output:
        cur_prev_transformeds.append(final_output.previously_transformed_tiles)
    else:
        cur_prev_transformeds.append(frozenset())

    _inner_generate_c(
        stage,
        [
            OperandDetails(ct, ie, shp, p)
            for ct, ie, shp, p in zip(
                cur_c_tensors,
                cur_index_exprs,
                cur_concrete_shapes,
                cur_prev_transformeds,
            )
        ],
        allow_holes,
    )
    return OperandDetails(
        cur_c_tensors[-1],
        cur_index_exprs[-1],
        cur_concrete_shapes[-1],
        cur_prev_transformeds[-1],
    )


def _move_registers(
    imp: impl.MoveLet,
    source_idx: int,
    c_tensors: Sequence[CTensor],
    operand_index_exprs,
    concrete_shapes,
    previously_transformeds,
    allow_holes: bool,
):
    dest_c_buf = _make_buffer(
        concrete_shapes[source_idx],
        imp.destination.vector_shape,
        imp.destination.dtype,
        imp.destination.bank,
    ).emit(zero_init=False)

    destination_index_expr = imp.destination.layout.buffer_indexing_expr(
        concrete_shapes[source_idx]
    )

    body_operand_details = []
    for i in range(len(imp.body.operands)):
        body_operand_details.append(
            OperandDetails(
                dest_c_buf if i == source_idx else c_tensors[i],
                destination_index_expr if i == source_idx else operand_index_exprs[i],
                concrete_shapes[i],
                previously_transformeds[i],
            )
        )

    move_operand_details = [
        OperandDetails(
            c_tensors[source_idx],
            operand_index_exprs[source_idx],
            concrete_shapes[source_idx],
            previously_transformeds[source_idx],
        ),
        OperandDetails(
            dest_c_buf,
            destination_index_expr,
            concrete_shapes[source_idx],
            previously_transformeds[source_idx],
        ),
    ]

    if imp.prologue:
        # Prologue may or may not take an input (e.g., no inputs for Zero).
        if imp.prologue.spec.inputs:
            _inner_generate_c(imp.prologue, move_operand_details, allow_holes)
        else:
            _inner_generate_c(imp.prologue, move_operand_details[-1:], allow_holes)
    _inner_generate_c(imp.body, body_operand_details, allow_holes)
    if imp.epilogue:
        _inner_generate_c(imp.epilogue, move_operand_details, allow_holes)

    dest_c_buf.emit_free()


def _naive_memset(op, outer_name, writer):
    out_shape = op.concrete_origin_shape
    o_ref = op.c_tensor.c_index
    with _emit_loop_nest_for_shape(out_shape) as it_names:
        substitutions = {
            f"p{dim}": (s if isinstance(s, int) else f"_{s}")
            for dim, s in enumerate(it_names)
        }
        o = vsub(op.index_expr, substitutions)
        writer.writeline(f"{o_ref(o)} = 0;  /* {outer_name} (vec boundary) */")


def generate_c(
    mode: Literal["kernel_only", "benchmark", "print_output"],
    imp: impl.Impl,
    out_fo,
    values=None,
    allow_holes: bool = False,
) -> None:
    imp = imp.to_applied()

    if values is None:
        values = [None] * (imp.operand_count - 1)
    values.append(None)  # for output, which is never initialized by caller

    namer = common.Namer()
    writer = common.Writer(out_fo)

    namer_token = common.namer.set(namer)
    writer_token = common.writer.set(writer)

    writer.writeline("#include <inttypes.h>")
    writer.writeline("#include <stdlib.h>")
    writer.writeline("#include <stdint.h>")
    writer.writeline("#include <stdio.h>")
    writer.writeline("#include <string.h>")
    writer.writeline("#include <time.h>")

    cpu_target = current_target()
    if isinstance(cpu_target, X86Target):
        writer.writeline("#include <immintrin.h>")
    elif isinstance(cpu_target, ArmTarget):
        writer.writeline("#include <arm_neon.h>")

    writer.writeline("#define is_aligned(POINTER, BYTE_COUNT) \\")
    writer.writeline("  (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)")

    for (dt, _), (vec_bytes, name, _, _) in GCC_VEC_TYPES.items():
        # Declare a vector of {vec_bytes} bytes, divided into {dt.c_type}
        # values. (vec_bytes must be a multiple of the c_type size.)
        writer.writeline(
            f"typedef {dt.c_type} {name} __attribute__ "
            f"((vector_size ({vec_bytes})));"
        )

    if mode == "benchmark":
        writer.writeline(
            "struct timespec ts_diff(struct timespec start, struct timespec end) {"
        )
        with writer.indent_block():
            writer.writeline("struct timespec temp;")
            writer.writeline("if ((end.tv_nsec-start.tv_nsec)<0) {")
            with writer.indent_block():
                writer.writeline("temp.tv_sec = end.tv_sec-start.tv_sec-1;")
                writer.writeline("temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;")
            writer.writeline("} else {")
            with writer.indent_block():
                writer.writeline("temp.tv_sec = end.tv_sec-start.tv_sec;")
                writer.writeline("temp.tv_nsec = end.tv_nsec-start.tv_nsec;")
            writer.writeline("}")
            writer.writeline("return temp;")
        writer.writeline("}")
        writer.writeline("")

    # Construct the program inputs, but don't emit anything yet.
    # NOTE: Names are assigned to each buffer here, and that name is used
    #   both as the parameter name for the kernel function and as the name
    #   inside main where it is allocated. (There's no treatment of aliasing.)
    tensor_names = []
    c_tensors = []
    index_exprs = []
    for operand, initial_value in zip(imp.spec.operands, values):
        c_buf = _make_buffer(
            operand.dim_sizes, operand.vector_shape, operand.dtype, operand.bank
        )
        index_exprs.append(operand.layout.buffer_indexing_expr(operand.dim_sizes))
        tensor_names.append(c_buf.name)  # type: ignore
        c_tensors.append(c_buf)

    # Emit the kernel function
    writer.writeline("__attribute__((noinline))")
    writer.writeline("void kernel(")
    for operand_idx, operand in enumerate(imp.spec.operands):
        c_buf = c_tensors[operand_idx]
        term = ", " if operand_idx + 1 < len(imp.spec.operands) else ")"
        writer.writeline(f"  {operand.dtype.c_type} *restrict {c_buf.name}{term}")
    writer.writeline("{")
    with writer.indent_block():
        operand_details = [
            OperandDetails(CPtr(c_buf.name, c_buf), index_expr, shape, frozenset())
            for c_buf, index_expr, shape in zip(
                c_tensors,
                index_exprs,
                (op.dim_sizes for op in imp.spec.operands),
            )
        ]
        _inner_generate_c(imp, operand_details, allow_holes)
    writer.writeline("}")
    writer.writeline("")

    # Emit the main function
    writer.writeline("int main() {")
    with writer.indent_block():
        for operand, c_buf, initial_value in zip(imp.spec.operands, c_tensors, values):
            c_buf.emit(zero_init=False)
            if initial_value is not None:
                if not operand.layout.is_row_major:
                    raise NotImplementedError(
                        "Initializing non-row-major tensors not yet implemented"
                    )
                for idx, el in enumerate(utils.flatten(initial_value)):
                    writer.writeline(
                        f"{c_buf.c_index(idx)} = ({operand.dtype.c_type})({el});"
                    )
            elif mode == "benchmark":
                writer.writeline(
                    f"for (unsigned long idx = 0; idx < {c_buf.size}; ++idx)"
                )
                writer.writeline(
                    f"  {c_buf.c_index(sympy.symbols('_idx'))} = ({operand.dtype.c_type})rand();"
                )

        def kernel():
            nonlocal c_tensors
            call_args_str = ", ".join(c_buf.c_index_ptr(0) for c_buf in c_tensors)
            writer.writeline(f"kernel({call_args_str});")

        if mode == "kernel_only":
            kernel()
        elif mode == "benchmark":
            writer.writeline("// Inlined kernel follows. This is for warm-up.")
            kernel()

            # NOTE: This benchmark does not zero output memory after each iteration.
            #   As a result, the result may be incorrect, though the times are right.
            writer.writeline("")
            writer.writeline("struct timespec start, end;")
            writer.writeline("clock_gettime(CLOCK_MONOTONIC, &start);")
            writer.writeline("#pragma clang loop unroll(disable)")
            writer.writeline(
                f"for (unsigned long benchiter = 0; benchiter < {BENCH_ITERS}; ++benchiter) {{"
            )

            with writer.indent_block():
                kernel()

            writer.writeline("}")
            writer.writeline("clock_gettime(CLOCK_MONOTONIC, &end);")
            writer.writeline("struct timespec delta = ts_diff(start, end);")
            writer.writeline(
                'printf("cpu: %llds %lldns\\n", (long long)delta.tv_sec, (long long)delta.tv_nsec);'
            )
        elif mode == "print_output":
            kernel()
            _emit_tensor_print(
                tensor_names[-1],
                c_tensors[-1].c_index,
                cast(
                    Union[tuple[int, int], tuple[int, int, int]], imp.output.dim_sizes
                ),
                imp.output.dtype,
                index_exprs[-1],
                writer,
                write_name=False,
            )
        else:
            raise ValueError("Unknown mode: " + mode)

        for c_tensor in reversed(c_tensors):
            c_tensor.emit_free()
        writer.writeline("return 0;")
    writer.writeline("}")

    common.namer.reset(namer_token)
    common.writer.reset(writer_token)
