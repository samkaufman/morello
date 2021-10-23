import abc
import contextlib
import contextvars
import dataclasses
import functools
import itertools
import operator
import string
import warnings
from collections.abc import Sequence
from typing import Callable, Iterable, Literal, Optional, Union, cast

import sympy

from .. import ops, specs, tensor, utils
from ..dtypes import Dtype
from ..system_config import hexagon
from ..system_config.state import current_system
from ..tensor import Tensor, TensorBase, TensorLike, Tile
from . import expr_utils, indexexpr

_namer: contextvars.ContextVar["_Namer"] = contextvars.ContextVar("_namer")
_writer: contextvars.ContextVar["_Writer"] = contextvars.ContextVar("_writer")
_unroll: contextvars.ContextVar["bool"] = contextvars.ContextVar(
    "_unroll", default=False
)

_DCFETCH_EMIT_STRATEGY = "first-pt"


# TODO: Choose a more principled STACK_CUTOFF.
STACK_CUTOFF = 256
BENCH_ITERS = 10


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
        self._pending_rhs_comment = None

    def indent(self):
        self._prefix += " " * 2

    def dedent(self):
        self._prefix = self._prefix[:-2]

    def set_impl(self, impl: ops.Schedule):
        self._pending_rhs_comment = str(impl.spec)

    def writeline(self, line: str):
        if self._pending_rhs_comment:
            print(
                self._prefix + line + "        // " + self._pending_rhs_comment,
                file=self._fo,
            )
            self._pending_rhs_comment = None
        else:
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
    dtype: Dtype,
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
            index_expr = index_expr.subs(sym, f"_{it_name}")

        with writer.indent_block():
            writer.writeline(
                f'printf("%" {dtype.int_fmt_macro} " ", {buffer_ref_fn(index_expr)});'
            )

        if rank:
            writer.writeline("}")
        for idx in range(rank - 1):
            writer.writeline('printf("\\n");')
            writer.writeline("}")
    writer.writeline("}")


@dataclasses.dataclass(frozen=True)
class _OperandDetails:
    """Data about an Impl operand commonly moved together during codegen.

    This tuple contains a reference to the Tensor/Tile itself, the current index
    expression mapping those dimensions to the underlying buffer, and the
    concrete shape of the *origin* of the tensor.
    """

    c_tensor: "_CTensor"
    index_expr: sympy.Expr
    # concrete_origin_shape is usually greater than a tile size in each
    # corresponding dimension, but might be smaller in boundary cases
    # (where the tile is effectively truncated).
    concrete_origin_shape: tuple[int, ...]


# TODO: Do we want this? Only used for _emit_tile_out_loop_nest.
@dataclasses.dataclass(frozen=True)
class _OperandDetailsExt(_OperandDetails):
    subscripts: tuple[int, ...]
    operand: TensorLike


def _emit_tile_out_loop_nest(
    remaining_subscripts: list[int],  # Reduces each step; base case = empty
    op_details: Sequence[_OperandDetailsExt],
    driving_tile_idx: int,
    inner_codegen: Callable[[Sequence[_OperandDetails]], None],
) -> None:
    driving_tile = op_details[driving_tile_idx].operand
    assert isinstance(driving_tile, Tile), f"driving_tile was {type(driving_tile)}"

    namer, writer = _namer.get(), _writer.get()

    # If given no subscripts, jump to generating the body of the loop. This is the base
    # case for recursive calls below.
    if not remaining_subscripts:
        inner_codegen(op_details)
        return

    it_subscript = remaining_subscripts[0]
    tile_dim = op_details[driving_tile_idx].subscripts.index(it_subscript)
    concrete_dim_size = op_details[driving_tile_idx].concrete_origin_shape[tile_dim]

    partial_steps = driving_tile.steps_dim(tile_dim, concrete_dim_size)
    driving_boundary_size = driving_tile.boundary_size(tile_dim, concrete_dim_size)
    full_steps = partial_steps - 1 if driving_boundary_size else partial_steps

    if full_steps:
        main_concrete_sizes: list[tuple[int, ...]] = [
            tuple(
                (min(ts, cos) if s == it_subscript else cos)
                for ts, cos, s in zip(
                    d.operand.dim_sizes, d.concrete_origin_shape, d.subscripts
                )
            )
            for d in op_details
        ]

        if not _unroll.get():
            # Emit the loop opener.
            it_var = 0
            if full_steps > 1:
                it_var = namer.fresh_name("t")
                writer.writeline(
                    f"for (int {it_var} = 0; {it_var} < {full_steps}; {it_var}++) {{"
                )
                writer.indent()

            # Update all indexing expressions where dimensions share a subscript with
            # the one we're modifying.
            full_new_index_exprs = _update_index_exprs(it_subscript, it_var, op_details)

            # Recurse to process the remaining iterators
            _emit_tile_out_loop_nest(
                remaining_subscripts[1:],
                [
                    _OperandDetailsExt(d.c_tensor, e, s, d.subscripts, d.operand)
                    for d, e, s in zip(
                        op_details, full_new_index_exprs, main_concrete_sizes
                    )
                ],
                driving_tile_idx,
                inner_codegen,
            )

            # Emit loop closer.
            if full_steps > 1:
                writer.dedent()
                writer.writeline("}")
        else:
            for it_var_idx in range(full_steps):
                # Sub. in the concrete step index instead of a reference to a generated
                # iterator name.
                full_new_index_exprs = _update_index_exprs(
                    it_subscript, it_var_idx, op_details
                )

                # Recurse to process the remaining iterators
                _emit_tile_out_loop_nest(
                    remaining_subscripts[1:],
                    [
                        _OperandDetailsExt(d.c_tensor, e, s, d.subscripts, d.operand)
                        for d, e, s in zip(
                            op_details, full_new_index_exprs, main_concrete_sizes
                        )
                    ],
                    driving_tile_idx,
                    inner_codegen,
                )

    # Generate boundary epilogue here
    if driving_boundary_size:
        # We don't check _unroll in the boundary case because it is already effectively
        # "unrolled" (it doesn't generate a loop).
        boundary_new_index_exprs = _update_index_exprs(
            it_subscript, full_steps, op_details
        )

        boundary_concrete_shapes = []
        # for operand, subscripts, _, concrete_shape in op_details:
        for o in op_details:
            if it_subscript not in o.subscripts:
                # No boundary if the subscript isn't present. Just forward the
                # concrete shape.
                boundary_concrete_shapes.append(o.concrete_origin_shape)
                continue

            # NOTE: The following assumes that at most one subscript for an operand
            # matches it_subscript. If multiple matched, that would mean we're
            # iterating diagonally.
            i = o.subscripts.index(it_subscript)
            orig_concrete_shape = list(o.concrete_origin_shape)
            if isinstance(o.operand, Tile):
                orig_concrete_shape[i] = o.operand.boundary_size(
                    i, o.concrete_origin_shape[i]
                )
            boundary_concrete_shapes.append(tuple(orig_concrete_shape))

        _emit_tile_out_loop_nest(
            remaining_subscripts[1:],
            [
                _OperandDetailsExt(d.c_tensor, e, s, d.subscripts, d.operand)
                for d, e, s in zip(
                    op_details, boundary_new_index_exprs, boundary_concrete_shapes
                )
            ],
            driving_tile_idx,
            inner_codegen,
        )


def _update_index_exprs(
    it_subscript: int, it_var: Union[str, int], op_details: Iterable[_OperandDetailsExt]
) -> Sequence[sympy.Expr]:
    """Update operand indexing expressions for an introduced C loop.

    Specifically, this function returns indexing expressions---one for each given
    _OperandDetails---which have had dimensions corresponding to it_subscript replaced
    by the operand's logical indexing expression and the given it_var.

    :param it_var: Either the name of the iteration variable or an integer constant.
    """
    # Prefix with an underscore if this is a string (symbol name). This is the naming
    # scheme codegen uses for symbols corresponding to names in the target language.
    if isinstance(it_var, str):
        it_var = "_" + it_var

    new_index_exprs = []
    for d in op_details:
        # No logical indexing expressions defined for Tensors. Forward their
        # (buffer) indexing expressions.
        if isinstance(d.operand, Tensor):
            new_index_exprs.append(d.index_expr)
            continue
        assert isinstance(d.operand, Tile)
        # For each subscript-matching dimension in this operand, update
        # the operand's corresponding indexing expression with the newly
        # introduced loop iterator variable name and the Expr mapping points
        # in the tile's coordinate space to that of its origin.
        all_substitutions = {}
        for idx, subscript in enumerate(d.subscripts):
            if subscript != it_subscript:
                continue
            new_expr = indexexpr.logical_indexing_expr(d.operand, idx)
            new_expr = new_expr.subs(sympy.symbols(f"i{idx}"), it_var)
            all_substitutions[sympy.symbols(f"p{idx}")] = new_expr
        new_index_exprs.append(d.index_expr.subs(all_substitutions))
    return new_index_exprs


class _CTensor(abc.ABC):
    @abc.abstractmethod
    def c_index(self, expr):
        raise NotImplementedError()

    def c_index_ptr(self, expr):
        ptr_str = f"&{self.c_index(expr)}"
        if ptr_str.endswith("[0]"):
            ptr_str = ptr_str[:-3]
        return ptr_str

    @abc.abstractmethod
    def emit_free(self):
        raise NotImplementedError()


class _CNameTensor(_CTensor):
    name: str


@dataclasses.dataclass(frozen=True)
class _CHeapArray(_CNameTensor):
    name: str
    size: int
    dtype: Dtype

    def emit(self) -> "_CHeapArray":
        writer = _writer.get()
        writer.writeline(f"{self.dtype.c_type} *restrict {self.name};")
        writer.writeline(
            f"posix_memalign((void **)&{self.name}, 128, {self.size}*sizeof({self.dtype.c_type}));  // TODO: Handle return"
        )
        writer.writeline(
            f"memset({self.name}, 0, {self.size}*sizeof({self.dtype.c_type}));"
        )
        return self

    def c_index(self, expr):
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr):
        return f"{self.name} + {_expr_to_c(self.dtype.size * expr)}"

    def emit_free(self):
        _writer.get().writeline(f"free({self.name});")


@dataclasses.dataclass(frozen=True)
class _CStackArray(_CNameTensor):
    name: str
    size: int
    dtype: Dtype

    def c_index(self, expr):
        return f"{self.name}[{_expr_to_c(expr)}]"

    def emit_free(self):
        pass

    def emit(self) -> "_CStackArray":
        _writer.get().writeline(
            f"{self.dtype.c_type} {self.name}[{self.size}] __attribute__((aligned (128))) = {{0}};"
        )
        return self


@dataclasses.dataclass(frozen=True)
class _CValueVar(_CNameTensor):
    name: str
    dtype: Dtype

    def c_index(self, expr):
        # TODO: Assert that expr evaluates to 0
        return self.name

    def emit_free(self):
        pass

    def emit(self) -> "_CValueVar":
        _writer.get().writeline(f"{self.dtype.c_type} {self.name} = 0;")
        return self


def _emit_buffer_alloc(size: int, dtype: Dtype) -> _CNameTensor:
    namer, writer = _namer.get(), _writer.get()

    name = namer.fresh_name("buf")
    if (size * dtype.size) > STACK_CUTOFF:
        return _CHeapArray(name, size, dtype).emit()
    elif size > 1:
        return _CStackArray(name, size, dtype).emit()
    else:
        return _CValueVar(name, dtype).emit()


@dataclasses.dataclass(frozen=True)
class _CHvxVectors(_CTensor):
    names: list[str]
    dtype_bytes: int

    # TODO: Use this static emit pattern for all _CTensor types
    @staticmethod
    def emit(tensor: hexagon.HvxVmemTensor) -> "_CHvxVectors":
        namer, writer = _namer.get(), _writer.get()
        assert (tensor.volume * tensor.dtype.size) % 128 == 0
        names = []
        for _ in range(tensor.vector_count):
            new_name = namer.fresh_name("vv")
            writer.writeline(f"HVX_Vector {new_name};")
            names.append(new_name)
        return _CHvxVectors(names=names, dtype_bytes=tensor.dtype.size)

    def c_index(self, expr):
        offset = int(expr)
        assert 128 % self.dtype_bytes == 0
        increment = 128 // self.dtype_bytes

        if offset % increment != 0:
            raise ValueError(
                f"Unexpected expression: {expr}. HVX vectors can only be indexed by "
                f"the first coordinate in their corresponding vector tile."
            )
        return self.names[offset // increment]

    def emit_free(self):
        pass


@contextlib.contextmanager
def _emit_assignment_copy(
    source: TensorLike,
    destination: TensorBase,
    source_index_expr: sympy.Expr,
    destination_index_expr: sympy.Expr,
    concrete_shape: tuple[int, ...],
    source_ref_fn: Callable[[Union[sympy.Expr, int]], str],
    dest_ref_fn: Callable[[Union[sympy.Expr, int]], str],
    dest_free_fn: Callable[[], None],
    is_input: bool,
    is_output: bool,
):
    assert is_input or is_output
    assert len(source.dim_sizes) == len(destination.dim_sizes)

    writer = _writer.get()

    def inner_emit(for_output: bool):
        nonlocal source_index_expr, destination_index_expr, source_ref_fn
        with _emit_loop_nest_for_shape(concrete_shape) as loop_subs:
            # Substitute the loop iterator names into the source and destination
            # index expressions.
            substitutions = {
                f"p{dim}": (s if isinstance(s, int) else f"_{s}")
                for dim, s in enumerate(loop_subs)
            }
            subbed_source_index_expr = source_index_expr.subs(substitutions)
            subbed_destination_index_expr = destination_index_expr.subs(substitutions)

            left = dest_ref_fn(subbed_destination_index_expr)
            right = source_ref_fn(subbed_source_index_expr)
            if for_output:
                left, right = right, left
            writer.writeline(f"{left} = {right};")

    inner_emit(False)
    yield
    if is_output:
        inner_emit(True)
    dest_free_fn()


@contextlib.contextmanager
def _emit_loop_nest_for_shape(shape: Sequence[int]):
    if _unroll.get():
        warnings.warn(
            "Unrolling not implemented for _emit_loop_nest_for_shape; will be ignored"
        )
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


def _inner_generate_c(impl: ops.Schedule, op_details: Sequence[_OperandDetails]):
    assert impl.is_scheduled
    assert len(op_details) == len(impl.inputs) + 1

    namer, writer = _namer.get(), _writer.get()

    writer.set_impl(impl)

    if isinstance(impl, (ops.Loop, ops.MatmulSplitLoop)):
        if impl.parallel:
            warnings.warn("Parallel loops not implemented")

        # TODO: Add a comment about why the following is `impl.inner`, not `impl`.
        driving_tile_idx = impl.inner.operands.index(impl.driving_tile)
        _emit_tile_out_loop_nest(
            list(impl.spec.operands_dim_subscripts()[driving_tile_idx]),
            [
                _OperandDetailsExt(
                    o.c_tensor, o.index_expr, o.concrete_origin_shape, subs, operand
                )
                for o, subs, operand in zip(
                    op_details,
                    impl.inner.spec.operands_dim_subscripts(),
                    impl.inner.operands,
                )
            ],
            impl.inner.operands.index(impl.driving_tile),
            inner_codegen=lambda details: _inner_generate_c(impl.inner, details),
        )
    elif isinstance(impl, ops.Pipeline):
        inps_op_details = op_details[:-1]
        output_op_details = op_details[-1]

        # First stage
        assert isinstance(impl.stages[0].output, Tensor)

        last_c_buf = _emit_buffer_alloc(
            impl.stages[0].output.volume, impl.stages[0].output.dtype
        )
        cur_slice = slice(-len(impl.stages[0].inputs), len(inps_op_details))
        cur_out = _pipeline_emit_stage(
            impl.stages[0], inps_op_details[cur_slice], last_c_buf, None, None
        )
        cur_slice = slice(
            1 + cur_slice.start - len(impl.stages[1].inputs), cur_slice.start
        )

        # Intermediate stages
        for stage, next_stage in zip(impl.stages[1:], impl.stages[2:]):
            assert isinstance(stage.output, Tensor)

            new_c_buf = _emit_buffer_alloc(stage.output.volume, stage.output.dtype)
            cur_out = _pipeline_emit_stage(
                stage, inps_op_details[cur_slice], new_c_buf, cur_out, None
            )
            last_c_buf.emit_free()
            last_c_buf = new_c_buf
            cur_slice = slice(
                1 + cur_slice.start - len(next_stage.inputs), cur_slice.start
            )

        # Last stage
        cur_out = _pipeline_emit_stage(
            impl.stages[-1],
            inps_op_details[cur_slice],
            output_op_details.c_tensor,
            cur_out,
            output_op_details,
        )
        last_c_buf.emit_free()
        assert (
            cur_out.concrete_origin_shape == output_op_details.concrete_origin_shape
        ), "Final stage output shape didn't match Pipeline output shape"
    elif isinstance(impl, ops.Mult):
        l_ref, r_ref, o_ref = (d.c_tensor.c_index for d in op_details)
        l, r, o = (d.index_expr for d in op_details)
        writer.writeline(f"{o_ref(o)} += {l_ref(l)} * {r_ref(r)};")
    elif isinstance(impl, ops.HvxVrmpyaccVuwVubRub):
        lhs, rhs, out = op_details
        lhs_ref_fn, rhs_ref_fn, out_ref_fn = (
            d.c_tensor.c_index_ptr for d in op_details
        )

        assert impl.lhs.contiguous
        assert impl.rhs.contiguous
        assert impl.output.contiguous

        # Rewrite index exprs. to refer to first element.
        lhs_index_expr = expr_utils.zero_points(lhs.index_expr)
        rhs_index_expr = expr_utils.zero_points(rhs.index_expr)
        out_index_expr = expr_utils.zero_points(out.index_expr)

        out_val = f"*(HVX_Vector *)({out_ref_fn(out_index_expr)})"
        writer.writeline(f"{out_val} = Q6_Vuw_vrmpyacc_VuwVubRub(")
        writer.writeline(f"  {out_val},")
        writer.writeline(f"  *(HVX_Vector *)({lhs_ref_fn(lhs_index_expr)}),")
        writer.writeline(f"  *(uint32_t *)({rhs_ref_fn(rhs_index_expr)})")
        writer.writeline(f");")
    elif isinstance(impl, ops.DirectConv):
        if not all(d == 1 for d in impl.output.dim_sizes):
            raise Exception("Only 1x1x1 output shape DirectConvs supported")
        # TODO: Remove the following _OperandDetails "destructuring"
        operand_index_exprs = [d.index_expr for d in op_details]
        tensor_ref_fns = [d.c_tensor.c_index for d in op_details]

        img, _, _ = impl.operands
        i_name = namer.fresh_name("pt")
        j_name = namer.fresh_name("pt")
        if _unroll.get():
            raise NotImplementedError("unrolling not implemented for DirectConv")
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
            raise Exception("Only 1x1x1 ReduceSums supported")
        # TODO: Remove the following _OperandDetails "destructuring"
        operand_index_exprs = [d.index_expr for d in op_details]
        tensor_ref_fns = [d.c_tensor.c_index for d in op_details]
        assert impl.is_scheduled
        i, o = operand_index_exprs
        writer.writeline(f"{tensor_ref_fns[1](o)} += {tensor_ref_fns[0](i)};")
    elif isinstance(impl, ops.MoveLet):
        source_idx = impl.operands.index(impl.source)
        assert (
            impl.inner.operands[source_idx] is impl.destination
        ), "MoveLet's inner Impl does not use destination tensor"

        # TODO: Remove the following "destructuring" of _OperandDetails
        operand_index_exprs = [d.index_expr for d in op_details]
        concrete_shapes = [d.concrete_origin_shape for d in op_details]

        concrete_shape = op_details[source_idx].concrete_origin_shape

        is_store = impl.input_idx is None

        # On the Hexagon target:
        if current_system().has_hvx:
            if not impl.is_store and impl.destination.bank == "L2":
                assert impl.source.bank == "GL"
                _emit_hvx_l2fetch(impl, is_store, op_details[source_idx])
                _inner_generate_c(impl.inner, op_details)
            # HVX scalar L1/dc-to-register case
            elif not impl.is_store and impl.destination.bank == "L1":
                assert impl.source.bank == "L2"
                _emit_hvx_dcfetch(
                    impl, impl.operands[source_idx], op_details[source_idx]
                )
                _inner_generate_c(impl.inner, op_details)
            elif impl.is_store and impl.destination.bank == "L2":
                # Generate no code for moves from L2 to global.
                assert impl.source.bank == "GL"
                _inner_generate_c(impl.inner, op_details)
            elif impl.is_store and impl.destination.bank == "L1":
                # Generate no code for writing from L1 back to L2
                assert impl.source.bank == "L2"
                _inner_generate_c(impl.inner, op_details)
            elif impl.destination.bank == "VMEM":
                assert impl.source.bank == "L2"
                _move_hvx_vmem(
                    impl,
                    source_idx,
                    [d.c_tensor for d in op_details],
                    operand_index_exprs,
                    concrete_shapes,
                )
            elif impl.destination.bank == "HexagonRF":
                _move_registers(
                    impl,
                    source_idx,
                    [d.c_tensor for d in op_details],
                    operand_index_exprs,
                    concrete_shapes,
                )
            else:
                word = "store" if impl.is_store else "load"
                raise Exception(f"Unexpected {word} case: {impl.destination.bank}")
        # On CPU and Hexagon targets: code is only generated (after the
        # previous cases) for MoveLet if there is a layout or contiguity change.
        # Otherwise, we just recurse.
        elif impl.destination.bank == "RF" and (
            not impl.source.contiguous
            or (
                impl.source.layout != impl.destination.layout
                and functools.reduce(operator.mul, concrete_shape, 1) != 1
            )
        ):
            _move_registers(
                impl,
                source_idx,
                [d.c_tensor for d in op_details],
                operand_index_exprs,
                concrete_shapes,
            )
        else:
            _inner_generate_c(impl.inner, op_details)

    else:
        raise NotImplementedError(f"Not implemented for {type(impl).__name__}")


def _pipeline_emit_stage(
    stage: ops.Schedule,
    input_operand_details: Sequence[_OperandDetails],
    output_c_tensor: _CTensor,
    previous_output: Optional[_OperandDetails],  # None only on first stage.
    final_output: Optional[_OperandDetails],  # Non-None on last stage.
) -> _OperandDetails:
    """Emits code for one stage in a Pipeline.

    :param stage: The Impl for the stage.
    :param input_operand_details: _OperandDetails for the Pipeline inputs consumed by
      this stage.
    :param output_c_tensor: The _CTensor into which this stage will write output.
    :param previous_output: _OperandDetails describing the previous stage's output, if
      any. This should be the return value of the previous stage's _pipeline_emit_stage
      call; a caller should just forward it.
    :param final_output: The final output of the Pipeline, or `None` if the given stage
      is not the final stage in the pipeline.
    :return: An _OperandDetails describing the output of this stage. This should be
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
    if previous_output:
        cur_c_tensors.insert(0, previous_output.c_tensor)
        cur_index_exprs.insert(0, previous_output.index_expr)
        cur_concrete_shapes.insert(0, previous_output.concrete_origin_shape)
    cur_concrete_shapes.append(stage.spec.calculate_output_shape(cur_concrete_shapes))

    # Complete cur_index_exprs with the output indexing expression. In the last stage of
    # a Pipeline, this final indexing expression is just passed in from the caller as
    # `output_index_expr`. In every other stage, this function computes it itself.
    if final_output:
        output_index_expr = final_output.index_expr
    else:
        assert isinstance(stage.output, Tensor)
        output_index_expr = indexexpr.buffer_indexing_expr(
            stage.output, cur_concrete_shapes[-1]
        )
    cur_index_exprs.append(output_index_expr)

    _inner_generate_c(
        stage,
        [
            _OperandDetails(ct, ie, shp)
            for ct, ie, shp in zip(cur_c_tensors, cur_index_exprs, cur_concrete_shapes)
        ],
    )
    return _OperandDetails(
        cur_c_tensors[-1], cur_index_exprs[-1], cur_concrete_shapes[-1]
    )


def _emit_hvx_l2fetch(
    impl: ops.Schedule, is_store: bool, source_operand: _OperandDetails
) -> None:
    writer = _writer.get()

    # TODO: Assert we're *not* in a boundary loop
    assert isinstance(impl, ops.MoveLet)
    assert not impl.is_store
    if not impl.prefetching:
        warnings.warn("l2fetch prefetching not implemented")

    # Swap w and h if column-major.
    # TODO: Add test for following parameter choices.
    lod = utils.layout_ordered_dims(impl.destination)
    head, w = lod[:-1], lod[-1]
    h = functools.reduce(operator.mul, head, 1)
    assert w < 256, f"Maximum size of l2fetch is 255; tile width is: {w}"
    assert h < 256, f"Maximum size of l2fetch is 255; tile height is: {h}"

    outer_w = utils.layout_ordered_dims(impl.source.address_root)[1]
    stride = outer_w - w
    assert stride < 65536

    source_ref_ptr_fn = source_operand.c_tensor.c_index_ptr
    source_index_expr = expr_utils.zero_points(source_operand.index_expr)
    if not is_store:
        writer.writeline(
            f"l2fetch({source_ref_ptr_fn(source_index_expr)}, {stride}, {w}, {h});"
        )


def _emit_hvx_dcfetch(
    impl: ops.MoveLet, source: TensorLike, source_operand: _OperandDetails
) -> None:
    writer = _writer.get()

    if not impl.prefetching:
        warnings.warn("dcfetch prefetching not implemented")
    # TODO: Assert we're *not* in a boundary loop

    # TODO: Determine cache line size and implement a correct strategy.
    if _DCFETCH_EMIT_STRATEGY == "first-pt":
        source_ref_ptr_fn = source_operand.c_tensor.c_index_ptr
        source_index_expr = expr_utils.zero_points(source_operand.index_expr)
        writer.writeline(f"Q6_dcfetch_A({source_ref_ptr_fn(source_index_expr)});")
    elif _DCFETCH_EMIT_STRATEGY == "every-pt":
        # Just dcfetch every point not on the innermost dimension. We do this
        # because we don't know the cache line size.
        sizes_to_scan = source.dim_sizes[:-1]
        for dims in itertools.product(
            *[range(0, dim_max + 1) for dim_max in sizes_to_scan]
        ):
            enumerated = list(enumerate(dims))
            if source.layout == specs.Layout.COL_MAJOR and len(enumerated) > 1:
                enumerated = [enumerated[1], enumerated[0]] + enumerated[2:]
            subs = {f"p{i}": d for i, d in enumerated}
            for i in range(len(source.dim_sizes)):
                subs.setdefault(f"p{i}", 0)
            new_index_expr = source_operand.index_expr.subs(subs) + 512
            writer.writeline(f"Q6_dcfetch_A(&{source_ref_fn(new_index_expr)});")
    else:
        raise Exception("Unknown emit strategy: " + _DCFETCH_EMIT_STRATEGY)


def _move_hvx_vmem(
    impl: ops.MoveLet,
    source_idx,
    c_tensors: Sequence[_CTensor],
    operand_index_exprs,
    concrete_shapes,
):
    writer = _writer.get()

    # TODO: If source is contiguous, just assign. Else, add move loop.

    assert impl.destination.bank == "VMEM"
    assert isinstance(impl.destination, hexagon.HvxVmemTensor)
    if impl.prefetching:
        raise NotImplementedError()

    assert (
        impl.destination.dim_sizes == concrete_shapes[source_idx]
    ), "Shapes don't match. This may be a loop boundary."

    source_c_tensor = c_tensors[source_idx]
    source_index_expr = operand_index_exprs[source_idx]

    vectors = _CHvxVectors.emit(impl.destination)

    new_c_tensors = list(c_tensors)
    new_c_tensors[source_idx] = vectors

    new_operand_index_exprs = list(operand_index_exprs)
    # TODO: Do we need to call this both here and in _emit_assignment_copy?
    new_operand_index_exprs[source_idx] = indexexpr.buffer_indexing_expr(
        impl.destination, concrete_shapes[source_idx]
    )

    slice_idx_exprs, slices_contig = _iter_vectors(impl.destination, source_index_expr)
    if slices_contig:
        # source_index_expr = source_index_expr.subs("p0", 0)
        for destination_name, slice_index_expr in zip(vectors.names, slice_idx_exprs):
            slice_index_expr = expr_utils.zero_points(slice_index_expr)
            writer.writeline(
                f"{destination_name} = *(HVX_Vector *)({source_c_tensor.c_index_ptr(slice_index_expr)});"
            )
        unroll_token = _unroll.set(True)
        _inner_generate_c(
            impl.inner,
            [
                _OperandDetails(*t)
                for t in zip(new_c_tensors, new_operand_index_exprs, concrete_shapes)
            ],
        )
        _unroll.reset(unroll_token)
        if impl.is_store:
            for destination_name, slice_index_expr in zip(
                vectors.names, slice_idx_exprs
            ):
                slice_index_expr = expr_utils.zero_points(slice_index_expr)
                writer.writeline(
                    f"*(HVX_Vector *)({source_c_tensor.c_index_ptr(slice_index_expr)}) = {destination_name};"
                )
    else:
        raise NotImplementedError(
            "The below needs to copy for *all* vectors in this tensor."
        )
        for destination_name, slice_index_expr in zip(vectors.names, slice_idx_exprs):
            # TODO: Can we use vgather here?
            # We don't assign with Q6_V_vzero() because this is a load, so it'll be
            # immediately filled.
            destination_index_expr = indexexpr.buffer_indexing_expr(
                impl.destination, concrete_shape
            )

            with _emit_assignment_copy(
                impl.source,
                impl.destination,
                slice_index_expr,
                destination_index_expr,
                concrete_shapes[source_idx],
                source_ref_fn,
                dest_ref_fn=vectors.c_index,
                dest_free_fn=vectors.emit_free,
                is_input=(not impl.is_store),
                is_output=impl.is_store,
            ):
                unroll_token = _unroll.set(True)
                _inner_generate_c(
                    impl.inner,
                    new_tensor_ref_fns,
                    new_operand_index_exprs,
                    new_concrete_shapes,
                )
                _unroll.reset(unroll_token)


def _iter_vectors(
    destination: hexagon.HvxVmemTensor, source_index_expr: sympy.Expr
) -> tuple[Iterable[sympy.Expr], bool]:
    """Compute source slices for HVX vectors in `destination`.

    This doesn't accept a concrete shapes parameter because it is intended to only be
    used in non-boundary cases. (In that case, the destination and origin should already
    have the correct concrete shape.)

    :param destination:
    :param source_index_expr:
    :return: Indexing expressions for concrete source tensors corresponding to each
             concrete vector, as well as whether or not source tiles are contiguous.
    """
    vector_tiling = destination.simple_tile(destination.vector_shape)
    steps_dim: Callable[[int], int] = getattr(vector_tiling, "steps_dim", lambda _: 1)

    substitutions = {}
    for dim in range(len(vector_tiling.dim_sizes)):
        assert isinstance(vector_tiling, hexagon.HvxVmemSimpleTile)
        substitutions[f"p{dim}"] = indexexpr.logical_indexing_expr(vector_tiling, dim)
    source_index_expr = source_index_expr.subs(substitutions)

    exprs = []
    for step_idxs in itertools.product(  # Loop over each concrete vector tile
        *[range(steps_dim(i)) for i in range(len(vector_tiling.dim_sizes))]
    ):
        subs = {}
        for dim_idx, step in enumerate(step_idxs):
            subs[f"i{dim_idx}"] = step
        exprs.append(source_index_expr.subs(subs))
    assert len(exprs) == destination.vector_count

    # Calculate whether or not the tiles are contiguous in the backing address space.
    contiguous = utils.contiguous(
        (destination.vector_shape, destination.layout), destination.address_root
    )

    return exprs, contiguous


def _move_registers(impl, source_idx, c_tensors, operand_index_exprs, concrete_shapes):
    assert (source_idx < len(impl.operands) - 1) == (not impl.is_store)
    assert (source_idx >= len(impl.operands) - 1) == impl.is_store

    c_buf = _emit_buffer_alloc(
        functools.reduce(operator.mul, concrete_shapes[source_idx], 1),
        impl.destination.dtype,
    )

    # TODO: All of the below could be one _OperandDetails update

    new_operand_index_exprs = list(operand_index_exprs)
    # TODO: Do we need to call this both here and in _emit_assignment_copy?
    new_operand_index_exprs[source_idx] = indexexpr.buffer_indexing_expr(
        impl.destination, concrete_shapes[source_idx]
    )

    new_concrete_shapes = list(concrete_shapes)
    new_concrete_shapes[source_idx] = concrete_shapes[source_idx]

    destination_index_expr = indexexpr.buffer_indexing_expr(
        impl.destination, concrete_shapes[source_idx]
    )

    with _emit_assignment_copy(
        impl.source,
        impl.destination,
        operand_index_exprs[source_idx],
        destination_index_expr,
        concrete_shapes[source_idx],
        c_tensors[source_idx].c_index,
        c_buf.c_index,
        c_buf.emit_free,
        is_input=(not impl.is_store),
        is_output=impl.is_store,
    ):
        new_c_tensors = list(c_tensors)
        new_c_tensors[source_idx] = c_buf
        _inner_generate_c(
            impl.inner,
            [
                _OperandDetails(c_tensor, idx_expr, shape)
                for c_tensor, idx_expr, shape in zip(
                    new_c_tensors,
                    new_operand_index_exprs,
                    new_concrete_shapes,
                )
            ],
        )


def _expr_to_c(expr: Union[sympy.Expr, int]) -> str:
    if isinstance(expr, int):
        return str(expr)
    substitutions = {}
    for sym in expr.free_symbols:
        if sym.name.startswith("_"):
            substitutions[sym] = sym.name[1:]
        else:
            raise ValueError(f"Found unbound symbols in expression: {expr}")
    return str(expr.subs(substitutions))


def generate_c(
    mode: Literal["kernel_only", "benchmark", "print_output"],
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

    writer.writeline("#include <inttypes.h>")
    writer.writeline("#include <stdlib.h>")
    writer.writeline("#include <stdint.h>")
    writer.writeline("#include <stdio.h>")
    writer.writeline("#include <string.h>")
    writer.writeline("#include <time.h>")
    if current_system().has_hvx:
        writer.writeline("#include <hexagon_types.h>")
        writer.writeline("#include <hexagon_protos.h>")
        writer.writeline("#include <hvx_inlines.h>")

    # From nn_asm_ops.h
    writer.writeline("")
    writer.writeline(
        "static inline void __attribute__((always_inline)) l2pref(const void *p, uint32_t height, uint32_t width, uint32_t stride) {"
    )
    writer.writeline("#if defined(__hexagon__)")
    writer.writeline(
        "  uint64_t control = Q6_P_combine_RR(stride,Q6_R_combine_RlRl(width,height));"
    )
    writer.writeline('  asm volatile (" l2fetch(%0,%1) " : :"r"(p),"r"(control));')
    writer.writeline("#endif")
    writer.writeline("}")
    writer.writeline("")
    writer.writeline(
        "static inline void __attribute__((always_inline)) l2fetch(const void *p, uint32_t stride, uint32_t width, uint32_t height) {"
    )
    writer.writeline("  return l2pref(p,height,width,stride);")
    writer.writeline("}")
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
        c_tensors = []
        index_exprs = []
        for operand, initial_value in zip(impl.operands, values):
            assert isinstance(operand, tensor.Tensor)
            c_buf = _emit_buffer_alloc(operand.volume, operand.dtype)
            ref_fn, free_fn = c_buf.c_index, c_buf.emit_free
            index_exprs.append(indexexpr.buffer_indexing_expr(operand))
            tensor_names.append(c_buf.name)
            c_tensors.append(c_buf)
            if initial_value is not None:
                if operand.layout != specs.Layout.ROW_MAJOR:
                    raise NotImplementedError(
                        "Initializing non-row-major tensors not yet implemented"
                    )
                for idx, el in enumerate(utils.flatten(initial_value)):
                    writer.writeline(f"{ref_fn(idx)} = {el};")

        def kernel():
            nonlocal impl, c_tensors, index_exprs
            operand_details = [
                _OperandDetails(c_tensor, index_expr, shape)
                for c_tensor, index_expr, shape in zip(
                    c_tensors,
                    index_exprs,
                    (op.dim_sizes for op in impl.operands),
                )
            ]
            _inner_generate_c(impl, operand_details)

        if mode == "kernel_only":
            kernel()
        elif mode == "benchmark":
            writer.writeline("// Inlined kernel follows. This is for warm-up.")
            kernel()

            # NOTE: This benchmark does not zero output memory after each iteration.
            #   As a result, the result may be incorrect.
            writer.writeline("")
            writer.writeline("struct timespec start, end;")
            writer.writeline("clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);")
            writer.writeline(
                f"for (unsigned long benchiter = 0; benchiter < {BENCH_ITERS}; ++benchiter) {{"
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
                c_tensors[-1].c_index,
                cast(
                    Union[tuple[int, int], tuple[int, int, int]], impl.output.dim_sizes
                ),
                impl.output.dtype,
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

    _namer.reset(namer_token)
    _writer.reset(writer_token)
