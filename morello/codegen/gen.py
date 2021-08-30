import contextlib
import contextvars
import functools
import operator
import string
import warnings
from collections.abc import Sequence
from typing import Callable, Iterable, Literal, NamedTuple, Optional, Union, cast

import sympy

from .. import ops, specs, tensor
from ..tensor import Tensor, Tile
from . import indexexpr

_namer: contextvars.ContextVar["_Namer"] = contextvars.ContextVar("_namer")
_writer: contextvars.ContextVar["_Writer"] = contextvars.ContextVar("_writer")


# TODO: Choose a more principled STACK_CUTOFF.
STACK_CUTOFF = 256
FLOAT_SIZE = 4  # bytes


class _Namer:
    def __init__(self):
        self.counts = {}

    def fresh_name(self, prefix: str = "v") -> str:
        cnt = self.counts.setdefault(prefix, 0)
        new_name = prefix + str(cnt)
        self.counts[prefix] += 1
        return new_name


class _Writer:
    def __init__(self, fo):
        self._fo = fo
        self._prefix = ""

    def indent(self):
        self._prefix += " " * 2

    def dedent(self):
        self._prefix = self._prefix[:-2]

    def writeline(self, line: str):
        print(self._prefix + line, file=self._fo)

    @contextlib.contextmanager
    def indent_block(self):
        self.indent()
        try:
            yield
        finally:
            self.dedent()


def _emit_tensor_print(
    buffer_name: str,
    buffer_ref_fn: Callable[[Union[sympy.Expr, int]], str],
    tensor_shape: Union[tuple[int, int], tuple[int, int, int]],
    index_expr: sympy.Expr,
    writer: _Writer,
    write_name=True,
) -> None:
    rank = len(tensor_shape)
    assert rank in (2, 3)
    assert all(s.name[0] == "p" for s in index_expr.free_symbols)

    writer.writeline("// Print " + buffer_name)
    writer.writeline("{")
    with writer.indent_block():
        if write_name:
            writer.writeline('printf("\\n");')
            writer.writeline(f'printf("{buffer_name}\\n");')

        for idx in range(rank):
            sym = sympy.symbols(f"p{idx}")
            idx = int(sym.name[1:])
            it_name = string.ascii_letters[idx]
            size = tensor_shape[idx]
            writer.writeline(
                f"for (size_t {it_name} = 0; {it_name} < {size}; {it_name}++) {{"
            )
            index_expr = index_expr.subs(sym, it_name)

        with writer.indent_block():
            writer.writeline(f'printf("%.4f ", {buffer_ref_fn(index_expr)});')

        if rank:
            writer.writeline("}")
        for idx in range(rank - 1):
            writer.writeline('printf("\\n");')
            writer.writeline("}")
    writer.writeline("}")


class _OperandDetails(NamedTuple):
    # inner tensor/tile.
    operand: Union[Tensor, Tile]
    subscripts: tuple[int, ...]
    index_expr: sympy.Expr
    # concrete_origin_shape is usually greater than a tile size in each
    # corresponding dimensions, but might be smaller in boundary cases
    # (where the tile is effectively truncated).
    concrete_origin_shape: tuple[int, ...]


def _emit_tile_out_loop_nest(
    remaining_subscripts: list[int],  # Reduces each step; base case = empty
    op_details: Sequence[_OperandDetails],
    driving_tile_idx: int,
    inner_codegen: Callable[
        [Sequence[sympy.Expr], Sequence[tuple[int, ...]]],
        None,
    ],
) -> None:
    driving_tile = op_details[driving_tile_idx].operand
    assert isinstance(driving_tile, Tile), f"driving_tile was {type(driving_tile)}"

    namer, writer = _namer.get(), _writer.get()

    if not remaining_subscripts:
        inner_codegen(
            [d.index_expr for d in op_details],
            [d.concrete_origin_shape for d in op_details],
        )
        return

    it_subscript = remaining_subscripts[0]
    tile_dim = op_details[driving_tile_idx].subscripts.index(it_subscript)
    concrete_dim_size = op_details[driving_tile_idx].concrete_origin_shape[tile_dim]

    partial_steps = driving_tile.steps_dim(tile_dim, concrete_dim_size)
    driving_boundary_size = driving_tile.boundary_size(tile_dim, concrete_dim_size)
    full_steps = partial_steps - 1 if driving_boundary_size else partial_steps

    if full_steps:
        # Print the loop header
        it_var = 0
        if full_steps > 1:
            it_var = namer.fresh_name("t")
            writer.writeline(
                f"for (int {it_var} = 0; {it_var} < {full_steps}; {it_var}++) {{"
            )
            writer.indent()

        # Update all indexing expressions where dimensions share a subscript with
        # the one we're modifying.
        full_new_index_exprs = _update_index_exprs(it_var, it_subscript, op_details)

        main_concrete_sizes = []
        for tile, subscripts, _, concrete_shape in op_details:
            new_shape = []
            for tile_size, concrete_outer_size, subscript in zip(
                tile.dim_sizes, concrete_shape, subscripts
            ):
                if subscript == it_subscript:
                    new_shape.append(min(tile_size, concrete_outer_size))
                else:
                    new_shape.append(concrete_outer_size)
            main_concrete_sizes.append(tuple(new_shape))

        # Recurse to process the remaining iterators
        _emit_tile_out_loop_nest(
            remaining_subscripts[1:],
            [
                _OperandDetails(d.operand, d.subscripts, e, s)
                for d, e, s in zip(
                    op_details, full_new_index_exprs, main_concrete_sizes
                )
            ],
            driving_tile_idx,
            inner_codegen,
        )

        if full_steps > 1:
            writer.dedent()
            writer.writeline("}")

    # Generate boundary epilogue here
    if driving_boundary_size:
        boundary_new_index_exprs = _update_index_exprs(
            full_steps, it_subscript, op_details
        )

        boundary_concrete_shapes = []
        for operand, subscripts, _, concrete_shape in op_details:
            if it_subscript not in subscripts:
                # No boundary if the subscript isn't present. Just forward the
                # concrete shape.
                boundary_concrete_shapes.append(concrete_shape)
                continue

            # NOTE: The following assumes that at most one subscript for an operand
            # matches it_subscript. If multiple matched, that would mean we're
            # iterating diagonally.
            i = subscripts.index(it_subscript)
            orig_concrete_shape = list(concrete_shape)
            if isinstance(operand, Tile):
                orig_concrete_shape[i] = operand.boundary_size(i, concrete_shape[i])
            boundary_concrete_shapes.append(tuple(orig_concrete_shape))

        _emit_tile_out_loop_nest(
            remaining_subscripts[1:],
            [
                _OperandDetails(d.operand, d.subscripts, e, s)
                for d, e, s in zip(
                    op_details, boundary_new_index_exprs, boundary_concrete_shapes
                )
            ],
            driving_tile_idx,
            inner_codegen,
        )


def _update_index_exprs(
    it_var: Union[str, int],
    it_subscript: int,
    op_details: Iterable[_OperandDetails],
) -> Sequence[sympy.Expr]:
    """Update operand indexing expressions for an introduced C loop.

    :param it_var: Either the name of the iteration variable or an integer constant.
    """
    # Prefix with an underscore if this is a string (symbol name). This is the naming
    # scheme codegen uses for symbols corresponding to names in the target language.
    if isinstance(it_var, str):
        it_var = "_" + it_var

    new_index_exprs = []
    for operand, dims_map, orig_index_expr, _ in op_details:
        # No logical indexing expressions defined for Tensors. Forward their
        # (buffer) indexing expressions.
        if isinstance(operand, Tensor):
            new_index_exprs.append(orig_index_expr)
            continue
        assert isinstance(operand, Tile)
        # For each subscript-matching dimension in this operand, update
        # the operand's corresponding indexing expression with the newly
        # introduced loop iterator variable name and the Expr mapping points
        # in the tile's coordinate space to that of its origin.
        all_substitutions = {}
        for idx, sub in enumerate(dims_map):
            if sub != it_subscript:
                continue
            new_expr = indexexpr.logical_indexing_expr(operand, idx)
            new_expr = new_expr.subs(sympy.symbols(f"i{idx}"), it_var)
            all_substitutions[sympy.symbols(f"p{idx}")] = new_expr
        new_index_exprs.append(orig_index_expr.subs(all_substitutions))
    return new_index_exprs


def _emit_buffer_alloc(
    name: str, size: int
) -> tuple[Callable[[Union[sympy.Expr, int]], str], Callable[[], None]]:
    def emit_free(name, writer):
        writer.writeline(f"free({name});")

    def c_index(name, expr):
        return f"{name}[{_expr_to_c(expr)}]"

    writer = _writer.get()
    if (size * FLOAT_SIZE) > STACK_CUTOFF:
        writer.writeline(f"float *{name};")
        writer.writeline(
            f"posix_memalign((void **)&{name}, 64, {size}*sizeof(float));  // TODO: check return"
        )
        writer.writeline(f"memset({name}, 0, {size}*sizeof(float));")

        return functools.partial(c_index, name), functools.partial(
            emit_free, name, writer
        )
    elif size > 1:
        writer.writeline(f"float {name}[{size}] = {{0}};")
        return functools.partial(c_index, name), lambda: None
    else:
        writer.writeline(f"float {name} = 0.0f;")
        return lambda _: name, lambda: None


@contextlib.contextmanager
def _emit_copy(
    source: Union[Tensor, Tile],
    destination: Tensor,
    source_index_expr: sympy.Expr,
    concrete_shape: tuple[int, ...],
    source_ref_fn: Callable[[Union[sympy.Expr, int]], str],
    new_name: str,
    is_input: bool,
    is_output: bool,
):
    assert is_input or is_output
    assert len(source.dim_sizes) == len(destination.dim_sizes)

    writer = _writer.get()

    destination_index_expr = indexexpr.buffer_indexing_expr(destination, concrete_shape)

    ref_fn, free_fn = _emit_buffer_alloc(
        new_name, size=functools.reduce(operator.mul, concrete_shape, 1)
    )

    def inner_emit(for_output: bool):
        nonlocal source_index_expr, destination_index_expr, source_ref_fn
        with _emit_loop_nest_for_shape(concrete_shape) as loop_subs:
            # Apply the loop substitutions
            substitutions = {
                f"p{dim}": (s if isinstance(s, int) else f"_{s}")
                for dim, s in enumerate(loop_subs)
            }
            subbed_source_index_expr = cast(
                sympy.Expr, source_index_expr.subs(substitutions)
            )
            subbed_destination_index_expr = cast(
                sympy.Expr, destination_index_expr.subs(substitutions)
            )

            left = ref_fn(subbed_destination_index_expr)
            right = source_ref_fn(subbed_source_index_expr)
            if for_output:
                left, right = right, left
            writer.writeline(f"{left} = {right};")

    inner_emit(False)
    yield ref_fn
    if is_output:
        inner_emit(True)
    free_fn()


@contextlib.contextmanager
def _emit_loop_nest_for_shape(shape: tuple[int, ...]):
    namer, writer = _namer.get(), _writer.get()
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
    impl: ops.Schedule,
    tensor_ref_fns: Sequence[Callable[[Union[sympy.Expr, int]], str]],
    operand_index_exprs: Sequence[sympy.Expr],
    concrete_shapes: Sequence[tuple[int, ...]],
) -> None:
    assert operand_index_exprs, "no operand_index_exprs; from " + str(impl.spec)
    assert len(tensor_ref_fns) == len(impl.inputs) + 1

    namer, writer = _namer.get(), _writer.get()

    if isinstance(impl, (ops.Loop, ops.MatmulSplitLoop)):
        if impl.parallel:
            warnings.warn("Parallel loops not implemented")

        def continuation(
            iexprs: Sequence[sympy.Expr],
            shapes: Sequence[tuple[int, ...]],
        ) -> None:
            _inner_generate_c(impl.inner, tensor_ref_fns, iexprs, shapes)

        driving_tile_idx = impl.inner.operands.index(impl.driving_tile)
        _emit_tile_out_loop_nest(
            list(impl.spec.operands_dim_subscripts()[driving_tile_idx]),
            [
                _OperandDetails(op, su, e, sh)
                for op, su, e, sh in zip(
                    impl.inner.operands,
                    impl.spec.operands_dim_subscripts(),
                    operand_index_exprs,
                    concrete_shapes,
                )
            ],
            impl.inner.operands.index(impl.driving_tile),
            inner_codegen=continuation,
        )
    elif isinstance(impl, ops.Pipeline):
        inps_refs_fns = tensor_ref_fns[:-1]
        output_ref_fn = tensor_ref_fns[-1]
        inps_index_exprs = operand_index_exprs[:-1]
        output_index_expr = operand_index_exprs[-1]
        inps_concrete_shapes = concrete_shapes[:-1]

        def emit_stage(
            stage: ops.Schedule,
            sl: slice,
            prev_output: Optional[
                tuple[
                    Callable[[Union[sympy.Expr, int]], str], sympy.Expr, tuple[int, ...]
                ]
            ],
            output_ref_fn: Callable[[Union[sympy.Expr, int]], str],
            oie: Optional[sympy.Expr] = None,
        ) -> tuple[
            Callable[[Union[sympy.Expr, int]], str], sympy.Expr, tuple[int, ...]
        ]:
            cur_tensor_ref_fns = list(inps_refs_fns[sl]) + [output_ref_fn]
            cur_index_exprs = list(inps_index_exprs[sl])
            cur_concrete_shapes = list(inps_concrete_shapes[sl])
            if prev_output is not None:
                cur_tensor_ref_fns.insert(0, prev_output[0])
                cur_index_exprs.insert(0, prev_output[1])
                cur_concrete_shapes.insert(0, prev_output[2])
            cur_concrete_shapes.append(
                stage.spec.calculate_output_shape(cur_concrete_shapes)
            )
            if oie is None:
                assert isinstance(stage.output, Tensor)
                oie = indexexpr.buffer_indexing_expr(
                    stage.output, cur_concrete_shapes[-1]
                )
            cur_index_exprs.append(oie)
            _inner_generate_c(
                stage, cur_tensor_ref_fns, cur_index_exprs, cur_concrete_shapes
            )
            return cur_tensor_ref_fns[-1], cur_index_exprs[-1], cur_concrete_shapes[-1]

        # First stage
        out_name = namer.fresh_name("buf")
        assert isinstance(impl.stages[0].output, Tensor)
        last_ref_fn, last_free_fn = _emit_buffer_alloc(
            out_name, impl.stages[0].output.volume
        )
        cur_slice = slice(-len(impl.stages[0].inputs), len(inps_index_exprs))
        cur_out = emit_stage(impl.stages[0], cur_slice, None, last_ref_fn, None)
        cur_slice = slice(
            1 + cur_slice.start - len(impl.stages[1].inputs), cur_slice.start
        )

        # Intermediate stages
        for stage, next_stage in zip(impl.stages[1:], impl.stages[2:]):
            assert isinstance(stage.output, Tensor)
            out_name = namer.fresh_name("buf")
            ref_fn, free_fn = _emit_buffer_alloc(out_name, stage.output.volume)
            cur_out = emit_stage(stage, cur_slice, cur_out, ref_fn, None)
            last_free_fn()
            last_free_fn = free_fn
            cur_slice = slice(
                1 + cur_slice.start - len(next_stage.inputs), cur_slice.start
            )

        # Last stage
        cur_out = emit_stage(
            impl.stages[-1], cur_slice, cur_out, output_ref_fn, output_index_expr
        )
        last_free_fn()
        assert (
            cur_out[2] == concrete_shapes[-1]
        ), "Final stage output shape didn't match Pipeline output shape"

    elif isinstance(impl, ops.Matmul):
        if not all(d == 1 for d in impl.output.dim_sizes):
            assert not impl.is_scheduled
            raise Exception("Only 1x1x1 Matmuls supported")
        assert impl.is_scheduled
        l, r, o = operand_index_exprs
        writer.writeline(
            f"{tensor_ref_fns[2](o)} += {tensor_ref_fns[0](l)} * {tensor_ref_fns[1](r)};"
        )
    elif isinstance(impl, ops.DirectConv):
        if not all(d == 1 for d in impl.output.dim_sizes):
            raise Exception("Only 1x1x1 output shape DirectConvs supported")
        assert impl.is_scheduled
        img, _, _ = impl.operands
        i_name = namer.fresh_name("pt")
        j_name = namer.fresh_name("pt")
        writer.writeline(
            f"for (int {i_name} = 0; {i_name} < {img.dim_sizes[0]}; {i_name}++) {{"
        )
        writer.writeline(
            f"for (int {j_name} = 0; {j_name} < {img.dim_sizes[1]}; {j_name}++) {{"
        )
        in_i = operand_index_exprs[0].subs([("p0", "_" + i_name), ("p1", "_" + j_name)])
        in_f = operand_index_exprs[1].subs([("p0", "_" + i_name), ("p1", "_" + j_name)])
        o = operand_index_exprs[2]
        with writer.indent_block():
            writer.writeline(
                f"{tensor_ref_fns[2](o)} += {tensor_ref_fns[0](in_i)} * {tensor_ref_fns[1](in_f)};"
            )
        writer.writeline("}")
        writer.writeline("}")
    elif isinstance(impl, ops.ReduceSum):
        if not all(d == 1 for d in impl.output.dim_sizes):
            assert not impl.is_scheduled
            raise Exception("Only 1x1x1 ReduceSums supported")
        assert impl.is_scheduled
        i, o = operand_index_exprs
        writer.writeline(f"{tensor_ref_fns[1](o)} += {tensor_ref_fns[0](i)};")
    elif isinstance(impl, ops.MoveLet):
        source_idx = impl.operands.index(impl.source)
        assert (
            impl.inner.operands[source_idx] is impl.destination
        ), "MoveLet source and destination indices differ"

        # Code is only generated for MoveLet if there is a layout or
        # contiguity change. Otherwise, we just recurse.
        concrete_shape = concrete_shapes[source_idx]
        if not impl.source.contiguous or (
            impl.source.layout != impl.destination.layout
            and functools.reduce(operator.mul, concrete_shape, 1) != 1
        ):
            source_ref_fn = tensor_ref_fns[source_idx]
            source_index_expr = operand_index_exprs[source_idx]
            destination_name = namer.fresh_name("mo")

            new_tensor_ref_fns = list(tensor_ref_fns)
            new_operand_index_exprs = list(operand_index_exprs)
            new_concrete_shapes = list(concrete_shapes)
            # TODO: Do we need to call this both here and in _emit_copy?
            new_operand_index_exprs[source_idx] = indexexpr.buffer_indexing_expr(
                impl.destination, concrete_shape
            )
            new_concrete_shapes[source_idx] = concrete_shape

            with _emit_copy(
                impl.source,
                impl.destination,
                source_index_expr,
                concrete_shape,
                source_ref_fn,
                destination_name,
                is_input=(source_idx < len(impl.operands) - 1),
                is_output=(source_idx >= len(impl.operands) - 1),
            ) as destination_ref_fn:
                new_tensor_ref_fns[source_idx] = destination_ref_fn
                _inner_generate_c(
                    impl.inner,
                    new_tensor_ref_fns,
                    new_operand_index_exprs,
                    new_concrete_shapes,
                )
        else:
            _inner_generate_c(
                impl.inner,
                tensor_ref_fns,
                operand_index_exprs,
                concrete_shapes,
            )

    else:
        raise NotImplementedError(f"Not implemented for {type(impl).__name__}")


def _expr_to_c(expr: Union[sympy.Expr, int]) -> str:
    if isinstance(expr, int):
        return str(expr)
    substitutions = {}
    for sym in expr.free_symbols:
        if sym.name.startswith("_"):
            substitutions[sym] = sym.name[1:]
    return str(expr.subs(substitutions))


def _flatten(src):
    if hasattr(src, "tolist"):
        src = src.tolist()
    for el in src:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el


def generate_c(
    mode: Union[Literal["benchmark"], Literal["print_output"]],
    impl: ops.Schedule,
    out_fo,
    values=None,
) -> None:
    if values is None:
        values = [None] * len(impl.inputs)
    values.append(None)  # for output, which is never initialized by caller

    namer = _Namer()
    writer = _Writer(out_fo)

    namer_token = _namer.set(namer)
    writer_token = _writer.set(writer)

    writer.writeline("#include <stdlib.h>")
    writer.writeline("#include <stdio.h>")
    writer.writeline("#include <string.h>")
    if mode == "benchmark":
        writer.writeline("#include <time.h>")
    writer.writeline("")

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

    writer.writeline("int main() {")
    with writer.indent_block():
        tensor_names = []
        tensor_ref_fns = []
        tensor_free_fns = []
        index_exprs = []
        for operand, initial_value in zip(impl.operands, values):
            assert isinstance(operand, tensor.Tensor)
            buffer_name = namer.fresh_name("a")
            ref_fn, free_fn = _emit_buffer_alloc(buffer_name, operand.volume)
            index_exprs.append(indexexpr.buffer_indexing_expr(operand))
            tensor_names.append(buffer_name)
            tensor_ref_fns.append(ref_fn)
            tensor_free_fns.append(free_fn)
            if initial_value is not None:
                if operand.layout != specs.Layout.ROW_MAJOR:
                    raise NotImplementedError(
                        "Initializing non-row-major tensors not yet implemented"
                    )
                for idx, el in enumerate(_flatten(initial_value)):
                    writer.writeline(f"{ref_fn(idx)} = {el};")

        def kernel():
            nonlocal impl, tensor_ref_fns, index_exprs
            _inner_generate_c(
                impl,
                tensor_ref_fns,
                index_exprs,
                [op.dim_sizes for op in impl.operands],
            )

        if mode == "benchmark":
            writer.writeline("// Inlined kernel follows. This is for warm-up.")
            kernel()

            # NOTE: This benchmark does not zero output memory after each iteration.
            #   As a result, the result may be incorrect.
            writer.writeline("")
            writer.writeline("struct timespec start, end;")
            writer.writeline("clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);")
            writer.writeline(
                "for (unsigned long benchiter = 0; benchiter < 10; ++benchiter) {"
            )
            with writer.indent_block():
                kernel()
            writer.writeline("}")
            writer.writeline("clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);")
            writer.writeline("struct timespec delta = ts_diff(start, end);")
            writer.writeline(
                'printf("cpu: %llds %lldns\\n", (long long)delta.tv_sec, (long long)delta.tv_nsec);'
            )
        elif mode == "print_output":
            kernel()
            _emit_tensor_print(
                tensor_names[-1],
                tensor_ref_fns[-1],
                cast(
                    Union[tuple[int, int], tuple[int, int, int]], impl.output.dim_sizes
                ),
                index_exprs[-1],
                writer,
                write_name=False,
            )
        else:
            raise ValueError("Unknown mode: " + mode)

        for free_fn in reversed(tensor_free_fns):
            free_fn()
        writer.writeline("return 0;")
    writer.writeline("}")

    _namer.reset(namer_token)
    _writer.reset(writer_token)
