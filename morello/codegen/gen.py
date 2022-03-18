import abc
import contextlib
import contextvars
import dataclasses
import functools
import itertools
import operator
import string
import warnings
from collections.abc import Mapping, Sequence
from os import name
from typing import Callable, Iterable, Literal, Optional, Union, cast

import sympy

from .. import impl, specs, tensor, utils
from ..dtypes import Dtype, Uint8, Uint32
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

    def set_impl(self, imp: impl.Impl):
        self._pending_rhs_comment = str(imp.spec)

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
            index_expr = index_expr.subs(sym, f"_{it_name}")

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


@dataclasses.dataclass(frozen=True)
class _OperandDetails:
    """Data about an Impl operand commonly moved together during codegen.

    This tuple contains a reference to the Tensor/Tile itself, the current index
    expression mapping those dimensions to the underlying buffer, and the
    concrete shape of the *origin* of the tensor.
    """

    def __post_init__(self):
        assert all(d > 0 for d in self.concrete_origin_shape)

    c_tensor: "_CTensor"
    index_expr: sympy.Expr
    # concrete_origin_shape is usually greater than a tile size in each
    # corresponding dimension, but might be smaller in boundary cases
    # (where the tile is effectively truncated).
    concrete_origin_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _OperandDetailsExt(_OperandDetails):
    subscripts: tuple[int, ...]
    operand: TensorLike


@dataclasses.dataclass(frozen=True)
class _LoopNestDescription:
    """Structures returned by _compute_tile_out_loop_nest."""

    subscripts_to_steps: Sequence[tuple[int, int]]
    body_index_exprs: Sequence[sympy.Expr]
    body_shapes: Sequence[tuple[int, ...]]
    is_boundary: bool


def _emit_tile_out_loop_nest(
    remaining_subscripts: list[int],  # Reduces each step; base case = empty
    op_details: Sequence[_OperandDetailsExt],
    parallel: bool,
    inner_codegen: Callable[[Sequence[_OperandDetails]], None],
) -> None:
    namer, writer = _namer.get(), _writer.get()

    # Compute steps and drop subscripts that have no full steps.
    # We do this first so that we know how many loops we're going to
    # emit.
    #
    # TODO: Move emitting_steps into compute.
    emitting_steps: list[tuple[int, int, int]] = []
    for subscript in remaining_subscripts:
        full_steps, has_boundary = _calc_steps(subscript, op_details)
        if full_steps != 0:
            emitting_steps.append((subscript, full_steps, has_boundary))

    # Generate new names for loop iterators.
    it_var_names = None
    if not _unroll.get():
        it_var_names = {s: namer.fresh_name("t") for s, _, _ in emitting_steps}

    # Emit loops.
    for loop_plan in _compute_tile_out_loop_nest(
        emitting_steps, it_var_names, op_details
    ):
        assert it_var_names is not None

        if not loop_plan.is_boundary and parallel:
            assert len(loop_plan.subscripts_to_steps)
            writer.writeline(
                "#pragma omp parallel for "
                f"collapse({len(loop_plan.subscripts_to_steps)}) "
                "schedule(static)"
            )
        for sub, steps in loop_plan.subscripts_to_steps:
            it_var = it_var_names[sub]
            writer.writeline(
                f"for (int {it_var} = 0; {it_var} < {steps}; {it_var}++) {{"
            )

        if len(loop_plan.subscripts_to_steps):
            writer.indent()
        inner_codegen(
            [
                _OperandDetailsExt(d.c_tensor, e, tuple(s), d.subscripts, d.operand)
                for d, e, s in zip(
                    op_details, loop_plan.body_index_exprs, loop_plan.body_shapes
                )
            ]
        )
        if len(loop_plan.subscripts_to_steps):
            writer.dedent()
        for _ in loop_plan.subscripts_to_steps:
            writer.writeline("}")


def _compute_tile_out_loop_nest(
    emitting_steps: Sequence[tuple[int, int, int]],
    it_var_names: Optional[Mapping[int, str]],
    op_details: Sequence[_OperandDetailsExt],
) -> Iterable[_LoopNestDescription]:
    # The following would be much faster if we sharing prefixes and didn't
    # enumerate combos where boundaries are impossible.
    if _unroll.get():
        raise NotImplementedError()
    assert it_var_names is not None

    for combo in itertools.product([False, True], repeat=len(emitting_steps)):
        # Ignore any combo. where at least one selected boundary dim. is empty.
        if any(b and not hb for b, (_, _, hb) in zip(combo, emitting_steps)):
            continue

        subscripts_to_loop_over: list[tuple[int, int]] = []
        new_index_exprs = [d.index_expr for d in op_details]
        concrete_shapes = [list(d.concrete_origin_shape) for d in op_details]
        for bit, (sub, full_steps, _) in zip(combo, emitting_steps):
            if bit:
                tile_idx_symbol = full_steps
            else:
                tile_idx_symbol = it_var_names[sub]
                subscripts_to_loop_over.append((sub, full_steps))

            new_index_exprs = _update_index_exprs(
                new_index_exprs, sub, tile_idx_symbol, op_details
            )
            for bcs, o in zip(concrete_shapes, op_details):
                if not isinstance(o.operand, Tile):
                    continue
                for sidx, s in enumerate(o.subscripts):
                    if s == sub:
                        if bit:
                            bcs[sidx] = o.operand.boundary_size(
                                sidx, o.concrete_origin_shape[sidx]
                            )
                        else:
                            bcs[sidx] = o.operand.dim_sizes[sidx]
        yield _LoopNestDescription(
            subscripts_to_loop_over,
            new_index_exprs,
            list(map(tuple, concrete_shapes)),
            is_boundary=any(combo),
        )


def _calc_steps(
    it_subscript: int, op_details: Sequence[_OperandDetailsExt]
) -> tuple[int, bool]:
    # The following gathers the number of steps (both full and boundary) as well
    # as whether or not a boundary tile exists. While boundary tile sizes
    # differ, presence of one should be consistent across all operands.
    # As a bit of defensive programming, the following also checks that this
    # is consistent across all operands and matching subscripts.
    partial_steps = None
    has_boundary = None
    for details in op_details:
        if not isinstance(details.operand, Tile):
            continue
        for dim, sub in enumerate(details.subscripts):
            if sub != it_subscript:
                continue
            new_partial_steps = details.operand.steps_dim(
                dim, details.concrete_origin_shape[dim]
            )
            new_has_boundary = bool(
                details.operand.boundary_size(dim, details.concrete_origin_shape[dim])
            )
            if partial_steps is None:
                partial_steps = new_partial_steps
                has_boundary = new_has_boundary
            assert new_partial_steps == partial_steps
            assert new_has_boundary == has_boundary
    assert isinstance(partial_steps, int) and isinstance(has_boundary, bool)

    full_steps = partial_steps - 1 if has_boundary else partial_steps

    return full_steps, has_boundary


def _update_index_exprs(
    orig_index_exprs: Iterable[sympy.Expr],
    it_subscript: int,
    it_var: Union[str, int],
    op_details: Iterable[_OperandDetailsExt],
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
    for d, orig_idx_expr in zip(op_details, orig_index_exprs):
        # No logical indexing expressions defined for Tensors. Forward their
        # (buffer) indexing expressions.
        if isinstance(d.operand, Tensor):
            new_index_exprs.append(orig_idx_expr)
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
        new_index_exprs.append(orig_idx_expr.subs(all_substitutions))
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
    def emit(self, zero_init=True) -> "_CNameTensor":
        raise NotImplementedError()

    @abc.abstractmethod
    def emit_free(self):
        raise NotImplementedError()


class _CNameTensor(_CTensor):
    name: str


@dataclasses.dataclass(frozen=True)
class _CPtr(_CNameTensor):
    name: str
    backing_tensor: _CTensor

    @property
    def dtype(self) -> Dtype:
        return self.backing_tensor.dtype

    def c_index(self, expr):
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr):
        return f"{self.name} + {_expr_to_c(expr)}"

    def emit_free(self):
        raise NotImplementedError()

    def emit(self):
        # return self
        raise NotImplementedError()


# TODO: Merge with _CHeapArray.
@dataclasses.dataclass(frozen=True)
class _CUnsizedHeapArray(_CNameTensor):
    name: str
    dtype: Dtype

    def c_index(self, expr):
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr):
        return f"{self.name} + {_expr_to_c(expr)}"

    def emit_free(self):
        _writer.get().writeline(f"free({self.name});")


@dataclasses.dataclass(frozen=True)
class _CHeapArray(_CNameTensor):
    name: str
    size: int
    dtype: Dtype

    def emit(self, zero_init=True) -> "_CHeapArray":
        writer = _writer.get()
        writer.writeline(f"{self.dtype.c_type} *restrict {self.name};")
        writer.writeline(
            f"posix_memalign((void **)&{self.name}, 128, {self.size}*sizeof({self.dtype.c_type}));  // TODO: Handle return"
        )
        if zero_init:
            writer.writeline(
                f"memset({self.name}, 0, {self.size}*sizeof({self.dtype.c_type}));"
            )
        return self

    def c_index(self, expr):
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr):
        return f"{self.name} + {_expr_to_c(expr)}"

    def emit_free(self):
        _writer.get().writeline(f"free({self.name});")


@dataclasses.dataclass(frozen=True)
class _CStackArray(_CNameTensor):
    name: str
    size: int
    dtype: Dtype

    def c_index(self, expr):
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr):
        return "&" + self.c_index(expr)

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


def _make_buffer(size: int, dtype: Dtype) -> _CNameTensor:
    name = _namer.get().fresh_name("buf")
    if (size * dtype.size) > STACK_CUTOFF:
        return _CHeapArray(name, size, dtype)
    elif size > 1:
        return _CStackArray(name, size, dtype)
    else:
        return _CValueVar(name, dtype)


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


def _inner_generate_c(imp: impl.Impl, op_details: Sequence[_OperandDetails]):
    assert imp.is_scheduled
    assert len(op_details) == len(imp.inputs) + 1

    namer, writer = _namer.get(), _writer.get()

    writer.set_impl(imp)

    if isinstance(imp, impl.Loop):
        # TODO: Add a comment about why the following is `imp.inner`, not `imp`.
        _emit_tile_out_loop_nest(
            list(imp.subscripts),
            [
                _OperandDetailsExt(
                    o.c_tensor, o.index_expr, o.concrete_origin_shape, subs, operand
                )
                for o, subs, operand in zip(
                    op_details,
                    imp.inner.spec.operands_dim_subscripts(),
                    imp.inner.operands,
                )
            ],
            imp.parallel,
            inner_codegen=lambda details: _inner_generate_c(imp.inner, details),
        )
    elif isinstance(imp, impl.Pipeline):
        inps_op_details = op_details[:-1]
        output_op_details = op_details[-1]

        # First stage
        assert isinstance(imp.stages[0].output, Tensor)

        last_c_buf = _make_buffer(
            imp.stages[0].output.volume, imp.stages[0].output.dtype
        ).emit()
        cur_slice = slice(-len(imp.stages[0].inputs), len(inps_op_details))
        cur_out = _pipeline_emit_stage(
            imp.stages[0], inps_op_details[cur_slice], last_c_buf, None, None
        )
        cur_slice = slice(
            1 + cur_slice.start - len(imp.stages[1].inputs), cur_slice.start
        )

        # Intermediate stages
        for stage, next_stage in zip(imp.stages[1:], imp.stages[2:]):
            assert isinstance(stage.output, Tensor)

            new_c_buf = _make_buffer(stage.output.volume, stage.output.dtype).emit()
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
            imp.stages[-1],
            inps_op_details[cur_slice],
            output_op_details.c_tensor,
            cur_out,
            output_op_details,
        )
        last_c_buf.emit_free()
        assert (
            cur_out.concrete_origin_shape == output_op_details.concrete_origin_shape
        ), "Final stage output shape didn't match Pipeline output shape"
    elif isinstance(imp, impl.Mult):
        l_ref, r_ref, o_ref = (d.c_tensor.c_index for d in op_details)
        l, r, o = (d.index_expr for d in op_details)
        writer.writeline(f"{o_ref(o)} += {l_ref(l)} * {r_ref(r)};")
    elif isinstance(imp, impl.HvxVrmpyaccVuwVubRub):
        lhs, rhs, out = op_details
        lhs_ref_fn, rhs_ref_fn, out_ref_fn = (
            d.c_tensor.c_index_ptr for d in op_details
        )

        assert imp.lhs.contiguous
        assert imp.rhs.contiguous
        assert imp.output.contiguous

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
    elif isinstance(imp, impl.DirectConv):
        if not all(d == 1 for d in imp.output.dim_sizes):
            raise Exception("Only 1x1x1 output shape DirectConvs supported")
        # TODO: Remove the following _OperandDetails "destructuring"
        operand_index_exprs = [d.index_expr for d in op_details]
        tensor_ref_fns = [d.c_tensor.c_index for d in op_details]

        img, _, _ = imp.operands
        if _unroll.get():
            raise NotImplementedError("unrolling not implemented for DirectConv")

        with _emit_loop_nest_for_shape(img.dim_sizes[1:]) as it_names:
            # Add 1 to dim because we're slicing off the first (batch) dim.
            substitutions = {
                f"p{dim + 1}": (s if isinstance(s, int) else f"_{s}")
                for dim, s in enumerate(it_names)
            }
            new_op_idx_exprs = [ie.subs(substitutions) for ie in operand_index_exprs]
            with writer.indent_block():
                in_i, in_f, o = new_op_idx_exprs
                writer.writeline(
                    f"{tensor_ref_fns[2](o)} += {tensor_ref_fns[0](in_i)} * {tensor_ref_fns[1](in_f)};"
                )
    elif isinstance(imp, impl.ReduceSum):
        if not all(d == 1 for d in imp.output.dim_sizes):
            raise Exception("Only 1x1x1 ReduceSums supported")
        # TODO: Remove the following _OperandDetails "destructuring"
        operand_index_exprs = [d.index_expr for d in op_details]
        tensor_ref_fns = [d.c_tensor.c_index for d in op_details]
        assert imp.is_scheduled
        i, o = operand_index_exprs
        writer.writeline(f"{tensor_ref_fns[1](o)} += {tensor_ref_fns[0](i)};")
    elif isinstance(imp, impl.MoveLet):
        source_idx = imp.operands.index(imp.source)
        assert (
            imp.inner.operands[source_idx] is imp.destination
        ), "MoveLet's inner Impl does not use destination tensor"

        # TODO: Remove the following "destructuring" of _OperandDetails
        operand_index_exprs = [d.index_expr for d in op_details]
        concrete_shapes = [d.concrete_origin_shape for d in op_details]

        concrete_shape = op_details[source_idx].concrete_origin_shape

        is_store = imp.input_idx is None  # TODO: Can remove?

        # On the Hexagon target:
        if current_system().has_hvx:
            if not imp.is_store and imp.destination.bank == "L2":
                assert imp.source.bank == "GL"
                _emit_hvx_l2fetch(imp, imp.is_store, op_details[source_idx])
                _inner_generate_c(imp.inner, op_details)
            # HVX scalar L1/dc-to-register case
            elif not imp.is_store and imp.destination.bank == "L1":
                assert imp.source.bank == "L2"
                _emit_hvx_dcfetch(imp, imp.operands[source_idx], op_details[source_idx])
                _inner_generate_c(imp.inner, op_details)
            elif imp.is_store and imp.destination.bank == "L2":
                # Generate no code for moves from L2 to global.
                assert imp.source.bank == "GL"
                _inner_generate_c(imp.inner, op_details)
            elif imp.is_store and imp.destination.bank == "L1":
                # Generate no code for writing from L1 back to L2
                assert imp.source.bank == "L2"
                _inner_generate_c(imp.inner, op_details)
            elif imp.destination.bank == "VMEM":
                assert imp.source.bank == "L2"
                _move_hvx_vmem(
                    imp,
                    source_idx,
                    [d.c_tensor for d in op_details],
                    operand_index_exprs,
                    concrete_shapes,
                )
            elif imp.destination.bank == "HexagonRF":
                _move_registers(
                    imp,
                    source_idx,
                    [d.c_tensor for d in op_details],
                    operand_index_exprs,
                    concrete_shapes,
                )
            else:
                word = "store" if imp.is_store else "load"
                raise Exception(f"Unexpected {word} case: {imp.destination.bank}")
        # On CPU and Hexagon targets: code is only generated (after the
        # previous cases) for MoveLet if there is a layout or contiguity change.
        # Otherwise, we just recurse.
        elif imp.destination.bank == "RF":
            _move_registers(
                imp,
                source_idx,
                [d.c_tensor for d in op_details],
                operand_index_exprs,
                concrete_shapes,
            )
        else:
            _inner_generate_c(imp.inner, op_details)
    elif isinstance(imp, impl.HvxGemvmpybbwAsm):
        lhs, rhs, out = op_details
        lhs_ref_fn, rhs_ref_fn, out_ref_fn = (
            d.c_tensor.c_index_ptr for d in op_details
        )

        # Rewrite index exprs. to refer to first element.
        lhs_index_expr = expr_utils.zero_points(lhs.index_expr)
        rhs_index_expr = expr_utils.zero_points(rhs.index_expr)
        out_index_expr = expr_utils.zero_points(out.index_expr)

        k, n = rhs.concrete_origin_shape

        # Allocate an output buffer which is guaranteed to be aligned. This is
        # not an especially good solution as it stands: it consumes more memory
        # and potentially does a lot of unnecessary allocation. Ideally, the
        # TensorSpec should carry alignment guarantees when they exist and, even
        # if they don't, we could be able to: (a) introduce an aligned buffer
        # with the scheduling language rather than silently inside this Impl
        # leaf, and (b) bypass it when the output is, coincidentally, aligned.
        # TODO: Make these improvements.
        kout_name = namer.fresh_name("kout")
        misalign_name = namer.fresh_name("misalign")
        writer.writeline(
            f"const int8_t {misalign_name} = !is_aligned(({out_ref_fn(out_index_expr)}), 128);"
        )
        writer.writeline(f"int *restrict {kout_name};")
        writer.writeline(f"if ({misalign_name}) {{")
        with writer.indent_block():
            aligned_output = _CHeapArray(namer.fresh_name("ab"), n, Uint32)
            aligned_output.emit(zero_init=False)
            writer.writeline(f"{kout_name} = {aligned_output.c_index_ptr(0)};")
        writer.writeline("} else {")
        with writer.indent_block():
            writer.writeline(f"{kout_name} = {out_ref_fn(out_index_expr)};")
        writer.writeline("}")

        writer.writeline(f"gemvmpybbw_asm(")
        writer.writeline(f"  {lhs_ref_fn(lhs_index_expr)},")
        writer.writeline(f"  0,")
        writer.writeline(f"  {rhs_ref_fn(rhs_index_expr)},")
        writer.writeline(f"  0,")
        writer.writeline(f"  {kout_name},")
        writer.writeline(f"  {n},")
        writer.writeline(f"  {k}")
        writer.writeline(f");")

        writer.writeline(f"if ({misalign_name}) {{")
        with writer.indent_block():
            # Copy from the buffer that's guaranteed to be aligned to destination.
            writer.writeline("vmemcpy_asm(")
            writer.writeline(f"  (void *)({out_ref_fn(out_index_expr)}),")
            writer.writeline(f"  (void *){kout_name},")
            writer.writeline(f"  4*{n}")
            writer.writeline(");")
            writer.writeline(f"free({kout_name});")
        writer.writeline("}")
    elif isinstance(imp, impl.PadTranspack):
        # Make a _CHeapArray for the result of the pad2d_and_transpack call.
        source_op_details = op_details[imp.input_idx]
        concrete_shape = source_op_details.concrete_origin_shape
        assert len(concrete_shape) == 2
        result = _CUnsizedHeapArray(namer.fresh_name("tp"), Uint8)
        result_index_expr = indexexpr.buffer_indexing_expr(
            imp.destination, concrete_shape
        )

        new_op_details = list(op_details)
        new_op_details[imp.input_idx] = _OperandDetails(
            result, result_index_expr, concrete_origin_shape=concrete_shape
        )

        op_txt = op_details[imp.input_idx].c_tensor.c_index_ptr(
            expr_utils.zero_points(op_details[imp.input_idx].index_expr)
        )

        struct_name = namer.fresh_name("tst")
        writer.writeline(
            f"struct tensor *{struct_name} = malloc(sizeof(struct tensor));"
        )
        writer.writeline(f"{struct_name}->shape.batches = 1;")
        writer.writeline(f"{struct_name}->shape.height = 1;")
        writer.writeline(f"{struct_name}->shape.width = {concrete_shape[0]};")
        writer.writeline(f"{struct_name}->shape.depth = {concrete_shape[1]};")
        writer.writeline(f"{struct_name}->data = (void *){op_txt};")
        writer.writeline(
            f"uint8_t *{result.name} = pad2d_and_transpack({struct_name});"
        )
        _inner_generate_c(imp.inner, new_op_details)
        writer.writeline(f"free({result.name});")
        writer.writeline(f"free({struct_name});")
    else:
        raise NotImplementedError(f"Not implemented for {type(imp).__name__}")


def _pipeline_emit_stage(
    stage: impl.Impl,
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
    imp: impl.MoveLet, is_store: bool, source_operand: _OperandDetails
) -> None:
    assert isinstance(imp, impl.MoveLet)

    writer = _writer.get()

    # TODO: Assert we're *not* in a boundary loop
    assert not imp.is_store
    if not imp.prefetching:
        warnings.warn("l2fetch prefetching not implemented")

    if imp.destination.layout == specs.Layout.HEXAGON_TRANSPACKED:
        if len(imp.destination.dim_sizes) != 2:
            warnings.warn("Not emitting l2fetch for transpacked, non-rank-2 tensor")
            return
        h, w = imp.destination.dim_sizes
        assert h % 4 == 0 and w % 32 == 0, f"Unexpected shape: {h}-by-{w}"

        # Compute the *packed* logical width
        outer_w = imp.source.address_root.dim_sizes[1]
        outer_w = outer_w % -16

        # Set `w`, `h`, and `outer_w` to correspond to underlying memory layout.
        # (i.e., row-major)
        h = h // 4
        w = w * 4
        outer_w = outer_w * 4
    else:
        # Swap w and h if column-major.
        # TODO: Add test for following parameter choices.
        lod = utils.layout_ordered_dims(imp.destination)
        head, w = lod[:-1], lod[-1]
        h = functools.reduce(operator.mul, head, 1)
        assert w < 256, f"Maximum size of l2fetch is 255; tile width is: {w}"
        assert h < 256, f"Maximum size of l2fetch is 255; tile height is: {h}"
        outer_w = utils.layout_ordered_dims(imp.source.address_root)[1]

    stride = outer_w
    assert stride < 65536

    source_ref_ptr_fn = source_operand.c_tensor.c_index_ptr
    source_index_expr = expr_utils.zero_points(source_operand.index_expr)
    if not is_store:
        writer.writeline(
            f"l2fetch({source_ref_ptr_fn(source_index_expr)}, {stride}, {w}, {h});"
        )


def _emit_hvx_dcfetch(
    imp: impl.MoveLet, source: TensorLike, source_operand: _OperandDetails
) -> None:
    writer = _writer.get()

    if source.layout == specs.Layout.HEXAGON_TRANSPACKED:
        # TODO: Add support for HEXAGON_TRANSPACKED.
        raise NotImplementedError("dcfetch doesn't support HEXAGON_TRANSPACKED")

    if not imp.prefetching:
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
    imp: impl.MoveLet,
    source_idx,
    c_tensors: Sequence[_CTensor],
    operand_index_exprs,
    concrete_shapes,
):
    writer = _writer.get()

    # TODO: If source is contiguous, just assign. Else, add move loop.

    assert imp.destination.bank == "VMEM"
    assert isinstance(imp.destination, hexagon.HvxVmemTensor)
    if imp.prefetching:
        raise NotImplementedError()

    assert (
        imp.destination.dim_sizes == concrete_shapes[source_idx]
    ), "Shapes don't match. This may be a loop boundary."

    source_c_tensor = c_tensors[source_idx]
    source_index_expr = operand_index_exprs[source_idx]

    vectors = _CHvxVectors.emit(imp.destination)

    new_c_tensors = list(c_tensors)
    new_c_tensors[source_idx] = vectors

    new_operand_index_exprs = list(operand_index_exprs)
    # TODO: Do we need to call this both here and in _emit_assignment_copy?
    new_operand_index_exprs[source_idx] = indexexpr.buffer_indexing_expr(
        imp.destination, concrete_shapes[source_idx]
    )

    slice_idx_exprs, slices_contig = _iter_vectors(imp.destination, source_index_expr)
    if slices_contig:
        # source_index_expr = source_index_expr.subs("p0", 0)
        for destination_name, slice_index_expr in zip(vectors.names, slice_idx_exprs):
            slice_index_expr = expr_utils.zero_points(slice_index_expr)
            writer.writeline(
                f"{destination_name} = *(HVX_Vector *)({source_c_tensor.c_index_ptr(slice_index_expr)});"
            )
        unroll_token = _unroll.set(True)
        _inner_generate_c(
            imp.inner,
            [
                _OperandDetails(*t)
                for t in zip(new_c_tensors, new_operand_index_exprs, concrete_shapes)
            ],
        )
        _unroll.reset(unroll_token)
        if imp.is_store:
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
                imp.destination, concrete_shape
            )

            with _emit_assignment_copy(
                imp.source,
                imp.destination,
                slice_index_expr,
                destination_index_expr,
                concrete_shapes[source_idx],
                source_ref_fn,
                dest_ref_fn=vectors.c_index,
                dest_free_fn=vectors.emit_free,
                is_input=(not imp.is_store),
                is_output=imp.is_store,
            ):
                unroll_token = _unroll.set(True)
                _inner_generate_c(
                    imp.inner,
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

    c_buf = _make_buffer(
        functools.reduce(operator.mul, concrete_shapes[source_idx], 1),
        impl.destination.dtype,
    ).emit()

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

    def _inner_expr_to_c(expr):
        if isinstance(expr, sympy.Add):
            return "(" + " + ".join(_inner_expr_to_c(a) for a in expr.args) + ")"
        elif isinstance(expr, sympy.Mul):
            return "(" + " * ".join(_inner_expr_to_c(a) for a in expr.args) + ")"
        elif isinstance(expr, sympy.Mod):
            assert len(expr.args) == 2
            return "(" + " % ".join(_inner_expr_to_c(a) for a in expr.args) + ")"
        elif isinstance(expr, expr_utils.FloorDiv):
            assert len(expr.args) == 2
            return (
                f"({_inner_expr_to_c(expr.args[0])} / {_inner_expr_to_c(expr.args[1])})"
            )
        elif isinstance(expr, (sympy.Symbol, sympy.core.numbers.Integer)):
            return str(expr)
        elif isinstance(expr, sympy.UnevaluatedExpr):
            assert len(expr.args) == 1
            return _inner_expr_to_c(expr.args[0])
        else:
            raise ValueError(f"Cannot convert {type(expr)} to C")

    return _inner_expr_to_c(expr.subs(substitutions))


def generate_c(
    mode: Literal["kernel_only", "benchmark", "print_output"],
    imp: impl.Impl,
    out_fo,
    values=None,
) -> None:
    if values is None:
        values = [None] * len(imp.inputs)
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
        writer.writeline("#include <hexagon_sim_timer.h>")

    writer.writeline("#define is_aligned(POINTER, BYTE_COUNT) \\")
    writer.writeline("  (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)")

    # TODO: The following prelude is a mess. Include decls/defs on demand, and
    #  don't embed them here, in Python.

    if current_system().has_hvx:
        writer.writeline("struct shape {")
        writer.writeline("	union {")
        writer.writeline("		struct {")
        writer.writeline("			uint32_t batches;")
        writer.writeline("			uint32_t height;")
        writer.writeline("			uint32_t width;")
        writer.writeline("			uint32_t depth;")
        writer.writeline("		};")
        writer.writeline("		struct {")
        writer.writeline("			uint32_t filt_height;")
        writer.writeline("			uint32_t filt_width;")
        writer.writeline("			uint32_t filt_depth;")
        writer.writeline("			uint32_t filt_batches;")
        writer.writeline("		};")
        writer.writeline("		struct {")
        writer.writeline("			uint64_t batches_height;")
        writer.writeline("			uint64_t width_depth;")
        writer.writeline("		};")
        writer.writeline("		uint32_t dimension[4];")
        writer.writeline("	};")
        writer.writeline("};")
        writer.writeline("")
        writer.writeline("struct tensor {")
        writer.writeline("	struct shape shape;")
        writer.writeline("	void *data;")
        writer.writeline("};")
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
        writer.writeline("void vmemset_32_2d_general_asm(")
        writer.writeline("    void *dst,")
        writer.writeline("    int val,")
        writer.writeline("    int width,")
        writer.writeline("    int height,")
        writer.writeline("    int stride);")
        writer.writeline("")
        writer.writeline("#if defined(__hexagon__)")
        writer.writeline("void vmemcpy_asm(void *dst, const void *src, int len);")
        writer.writeline("#endif")
        writer.writeline("")
        writer.writeline("void vmemcpy_2d_asm(")
        writer.writeline("    unsigned wid,   ")
        writer.writeline("    unsigned ht,    ")
        writer.writeline("    void *dst,      ")
        writer.writeline("    int dst_pitch,  ")
        writer.writeline("    void const *src,")
        writer.writeline("    int src_pitch); ")
        writer.writeline("")
        writer.writeline("void vmemcpy_2d_general_asm(")
        writer.writeline("    unsigned wid,   ")
        writer.writeline("    unsigned ht,    ")
        writer.writeline("    void *dst,      ")
        writer.writeline("    int dst_pitch,  ")
        writer.writeline("    void const *src,")
        writer.writeline("    int src_pitch); ")
        writer.writeline("")
        writer.writeline("static void pad2d_generic(")
        writer.writeline("	void const *input_data, //  { inh,  inw ,  (elbytes)}")
        writer.writeline("	int input_height,")
        writer.writeline("	int input_width,")
        writer.writeline("	void *output_data, //  { outh,  outw ,  (elbytes)}")
        writer.writeline("	int output_height,")
        writer.writeline("	int output_width,")
        writer.writeline("	int pad_value,")
        writer.writeline("	int elbytes) // may be 1,2 or 4 (or any, if pad_value=0)")
        writer.writeline("{")
        writer.writeline("	if (elbytes == 1)")
        writer.writeline("		pad_value = Q6_R_vsplatb_R(pad_value);")
        writer.writeline("	else if (elbytes == 2)")
        writer.writeline("		pad_value = Q6_R_combine_RlRl(pad_value, pad_value);")
        writer.writeline("")
        writer.writeline("	const uint8_t *ptr_in = input_data;")
        writer.writeline("	uint8_t *ptr_out = output_data;")
        writer.writeline("	int pad_x = output_width - input_width;")
        writer.writeline("	int pad_y = output_height - input_height;")
        writer.writeline("	if (pad_x > 0)")
        writer.writeline("	{")
        writer.writeline("		vmemcpy_2d_general_asm(")
        writer.writeline(
            "			input_width * elbytes, input_height, // rect width, height"
        )
        writer.writeline("			ptr_out, output_width * elbytes,	 // dst address, stride")
        writer.writeline("			ptr_in, input_width * elbytes);")
        writer.writeline("		vmemset_32_2d_general_asm(")
        writer.writeline("			ptr_out + input_width * elbytes, // location")
        writer.writeline("			pad_value,						 // pad value (32 bits)")
        writer.writeline("			pad_x * elbytes, input_height,	 // w,h of region")
        writer.writeline("			output_width * elbytes			 // stride")
        writer.writeline("		);")
        writer.writeline("	}")
        writer.writeline("	else")
        writer.writeline("	{")
        writer.writeline(
            "		vmemcpy_asm(ptr_out, ptr_in, input_height * output_width * elbytes);"
        )
        writer.writeline("	}")
        writer.writeline("	if (pad_y > 0)")
        writer.writeline("	{")
        writer.writeline("		ptr_out += input_height * output_width * elbytes;")
        writer.writeline("		// fill as 'single row'")
        writer.writeline("		vmemset_32_2d_general_asm(")
        writer.writeline("			ptr_out,")
        writer.writeline("			pad_value,")
        writer.writeline("			pad_y * output_width * elbytes, 1, // width, height")
        writer.writeline("			0);")
        writer.writeline("	}")
        writer.writeline("}")
        writer.writeline("")
        writer.writeline("void pad2d(")
        writer.writeline(
            "	const uint8_t *input_data, int input_height, int input_width,"
        )
        writer.writeline(
            "	uint8_t *output_data, int output_height, int output_width, int pad_value)"
        )
        writer.writeline("{")
        writer.writeline("	pad2d_generic(input_data, input_height, input_width,")
        writer.writeline(
            "				  output_data, output_height, output_width, pad_value, sizeof(uint8_t));"
        )
        writer.writeline("}")
        writer.writeline("")
        writer.writeline("void transpack(")
        writer.writeline("	const uint8_t *in_data, int K, int M, uint8_t *out_data)")
        writer.writeline("{")
        writer.writeline("	int x, y, z;")
        writer.writeline("")
        writer.writeline("	//out_width = 32*K;")
        writer.writeline("	//out_height = M/32;")
        writer.writeline("")
        writer.writeline("	for (x = 0; x < M; x += 32)")
        writer.writeline("	{")
        writer.writeline("		for (y = 0; y < K; y += 4)")
        writer.writeline("			for (z = 0; z < 32; z += 1)")
        writer.writeline("			{")
        writer.writeline(
            "				out_data[32 * y + K * x + 4 * z + 0] = in_data[M * (y + 0) + x + z];"
        )
        writer.writeline(
            "				out_data[32 * y + K * x + 4 * z + 1] = in_data[M * (y + 1) + x + z];"
        )
        writer.writeline(
            "				out_data[32 * y + K * x + 4 * z + 2] = in_data[M * (y + 2) + x + z];"
        )
        writer.writeline(
            "				out_data[32 * y + K * x + 4 * z + 3] = in_data[M * (y + 3) + x + z];"
        )
        writer.writeline("			}")
        writer.writeline("	}")
        writer.writeline("	return;")
        writer.writeline("}")
        writer.writeline("")
        writer.writeline("/**")
        writer.writeline(" * Pads and transpacks the rhs operand.")
        writer.writeline(" */")
        writer.writeline(
            "uint8_t *pad2d_and_transpack(const struct tensor *const filt_tensor)"
        )
        writer.writeline("{")
        writer.writeline("	uint32_t filt_batches = filt_tensor->shape.filt_batches;")
        writer.writeline("	uint32_t filt_depth = filt_tensor->shape.filt_depth;")
        writer.writeline("	uint32_t out_depth = filt_batches;")
        writer.writeline("	uint8_t *filt = filt_tensor->data;")
        writer.writeline("#define BPAD 32")
        writer.writeline("#define APAD 16")
        writer.writeline("#define ALIGN_SIZE 128")
        writer.writeline(
            "	uint32_t filt_elements_pad = (filt_depth + APAD - 1) & (~(APAD - 1));"
        )
        writer.writeline("	int out_depth_pad = (out_depth + BPAD - 1) & ~(BPAD - 1);")
        writer.writeline("	uint32_t consts_size;")
        writer.writeline(
            "	filt_elements_pad = (filt_elements_pad < 32) ? 32 : filt_elements_pad;"
        )
        writer.writeline("	consts_size = filt_elements_pad * out_depth_pad;")
        writer.writeline("")
        writer.writeline("	// Allocate our result buffer")
        writer.writeline("	uint8_t *const opaque;")
        writer.writeline(
            "	if (posix_memalign((void **)&opaque, ALIGN_SIZE, consts_size) != 0)"
        )
        writer.writeline("	{")
        writer.writeline(
            'fprintf(stderr, "couldn\'t allocate buffer for const rearrangement\\n");'
        )
        writer.writeline("		exit(102);")
        writer.writeline("	}")
        writer.writeline("")
        writer.writeline("	// Allocate a temporary buffer for the output of pad2d.")
        writer.writeline("	uint8_t *const pad_output;")
        writer.writeline(
            "	if (posix_memalign((void **)&pad_output, ALIGN_SIZE, filt_elements_pad * out_depth_pad + 256) != 0)"
        )
        writer.writeline("	{")
        writer.writeline(
            '		fprintf(stderr, "couldn\'t allocate buffer pad2d output\\n");'
        )
        writer.writeline("		exit(103);")
        writer.writeline("	}")
        writer.writeline("")
        writer.writeline("	// Pad, transpose, and pack.")
        writer.writeline(
            "	pad2d(filt, filt_depth, out_depth, pad_output, filt_elements_pad, out_depth_pad, 0);"
        )
        writer.writeline(
            "	transpack(pad_output, filt_elements_pad, out_depth_pad, opaque);"
        )
        writer.writeline("")
        writer.writeline("	free(pad_output);")
        writer.writeline("	return opaque;")
        writer.writeline("}")
        writer.writeline("")
        writer.writeline("void gemvmpybbw_asm(")
        writer.writeline("    const uint8_t *x,")
        writer.writeline("    int x_offset,")
        writer.writeline("    const uint8_t *y,")
        writer.writeline("    int y_offset,")
        writer.writeline("    int *z,")
        writer.writeline("    int MSTEP,")
        writer.writeline("    int K);")
        writer.writeline("")

    if mode == "benchmark":
        # TODO: Don't branch on `has_hvx`. Abstract over Targets instead.
        if not current_system().has_hvx:
            writer.writeline(
                "struct timespec ts_diff(struct timespec start, struct timespec end) {"
            )
            with writer.indent_block():
                writer.writeline("struct timespec temp;")
                writer.writeline("if ((end.tv_nsec-start.tv_nsec)<0) {")
                with writer.indent_block():
                    writer.writeline("temp.tv_sec = end.tv_sec-start.tv_sec-1;")
                    writer.writeline(
                        "temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;"
                    )
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
    for operand, initial_value in zip(imp.operands, values):
        assert isinstance(operand, tensor.Tensor)
        c_buf = _make_buffer(operand.volume, operand.dtype)
        index_exprs.append(indexexpr.buffer_indexing_expr(operand))
        tensor_names.append(c_buf.name)
        c_tensors.append(c_buf)

    # Emit the kernel function
    writer.writeline("__attribute__((noinline))")
    writer.writeline("void kernel(")
    for operand_idx in range(len(imp.operands)):
        operand = imp.operands[operand_idx]
        c_buf = c_tensors[operand_idx]
        term = ", " if operand_idx + 1 < len(imp.operands) else ")"
        writer.writeline(f"  {operand.dtype.c_type} *restrict {c_buf.name}{term}")
    writer.writeline("{")
    with writer.indent_block():
        operand_details = [
            _OperandDetails(_CPtr(c_buf.name, c_buf), index_expr, shape)
            for c_buf, index_expr, shape in zip(
                c_tensors,
                index_exprs,
                (op.dim_sizes for op in imp.operands),
            )
        ]
        _inner_generate_c(imp, operand_details)
    writer.writeline("}")
    writer.writeline("")

    # Emit the main function
    writer.writeline("int main() {")
    with writer.indent_block():
        for operand, c_buf, initial_value in zip(imp.operands, c_tensors, values):
            c_buf.emit()
            if initial_value is not None:
                if operand.layout != specs.Layout.ROW_MAJOR:
                    raise NotImplementedError(
                        "Initializing non-row-major tensors not yet implemented"
                    )
                for idx, el in enumerate(utils.flatten(initial_value)):
                    writer.writeline(
                        f"{c_buf.c_index(idx)} = ({operand.dtype.c_type})({el});"
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
            if current_system().has_hvx:
                writer.writeline(
                    "unsigned long long start = hexagon_sim_read_pcycles();"
                )
            else:
                writer.writeline("")
                writer.writeline("struct timespec start, end;")
                writer.writeline("clock_gettime(CLOCK_MONOTONIC, &start);")
                writer.writeline("#pragma clang loop unroll(disable)")
                writer.writeline(
                    f"for (unsigned long benchiter = 0; benchiter < {BENCH_ITERS}; ++benchiter) {{"
                )

            with writer.indent_block():
                kernel()
            if current_system().has_hvx:
                writer.writeline("unsigned long long end = hexagon_sim_read_pcycles();")
                writer.writeline('printf("pcycles: %llu\\n", end - start);')
            else:
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

    _namer.reset(namer_token)
    _writer.reset(writer_token)
