import contextvars
import dataclasses
import functools
import itertools
import operator
from typing import Iterable, Optional, Sequence, Union

import sympy

from ..dtypes import Dtype, Uint8, Uint32
from . import common, expr_utils
from .indexexpr import vsub

GCC_VEC_TYPES: dict[tuple[Dtype, int], tuple[int, str, str, str]] = {
    (Uint32, 8): (32, "vui8", "__m256i", "si256"),
    (Uint32, 4): (16, "vui4", "__m128i", "si128"),
    (Uint8, 32): (32, "vub32", "__m256i", "si256"),
    (Uint8, 16): (16, "vub16", "__m128i", "si128"),
}

ONES_FOR_NON_ZERO_INIT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ONES_FOR_NON_ZERO_INIT", default=False
)


def _expr_to_c(expr: Union[sympy.Expr, int]) -> str:
    if isinstance(expr, int):
        return str(expr)
    substitutions = {}
    for sym in expr.free_symbols:
        if not sym.name.startswith("_"):
            raise ValueError(f"Found unbound symbols in expression: {expr}")
        substitutions[sym] = sym.name[1:]

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

    return _inner_expr_to_c(vsub(expr, substitutions))


class CTensor:
    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        """Return a C expression referring to the value at a given expression.

        Additionally, `reinterpret` may be provided to introduce a type cast.
        This is useful for interpreting a (partial) buffer as a vector type.
        """
        raise NotImplementedError()

    def c_index_ptr(self, expr, reinterpret: Optional[str] = None):
        if reinterpret:
            raise NotImplementedError()
        ptr_str = f"&{self.c_index(expr)}"
        if ptr_str.endswith("[0]"):
            ptr_str = ptr_str[:-3]
        return ptr_str

    def c_index_vec(self, expr, reinterpret: Optional[str] = None) -> str:
        raise NotImplementedError()

    def emit(self, zero_init=True) -> "CNameTensor":
        raise NotImplementedError()

    def emit_free(self):
        raise NotImplementedError()

    @property
    def declared_type(self) -> str:
        raise NotImplementedError()

    @property
    def should_unroll(self) -> bool:
        return False


class CNameTensor(CTensor):
    name: str


@dataclasses.dataclass(frozen=True)
class CPtr(CNameTensor):
    name: str
    backing_tensor: CTensor

    @property
    def dtype(self) -> Dtype:
        return self.backing_tensor.dtype

    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        s = f"{self.name}[{_expr_to_c(expr)}]"
        if reinterpret:
            s = f"*({reinterpret} *)(&{s})"
        return s

    def _bad_new_c_index(self, expr, run=1) -> str:
        # TODO: How's this prototype implementation where we just substitute the name and dereference?
        new_backing_tensor = dataclasses.replace(self.backing_tensor, name=self.name)
        # extra = " /* run={run} */" if run > 1 else ""
        extra = ""
        s = f"*({new_backing_tensor.c_index(expr, run=run)}{extra})"
        o = self._orig_c_index(expr, run=run)
        # assert s == o, f"{s} != {o}"
        # if s != o:
        #     o = f"({o} /* alt={s} */)"
        return o

    def _orig_c_index(self, expr, run=1) -> str:
        # TODO: Specialize below.
        if run != 1:
            return f"({self.name}[{_expr_to_c(expr)}])"
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr, reinterpret: Optional[str] = None):
        if reinterpret:
            raise NotImplementedError()
        return f"{self.name} + {_expr_to_c(expr)}"

    def emit_free(self):
        raise NotImplementedError()

    def emit(self, zero_init=True):
        raise NotImplementedError()


# TODO: Merge with _CHeapArray.
@dataclasses.dataclass(frozen=True)
class CUnsizedHeapArray(CNameTensor):
    name: str
    dtype: Dtype

    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        if reinterpret:
            raise NotImplementedError()
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr, reinterpret: Optional[str] = None):
        if reinterpret:
            raise NotImplementedError()
        return f"{self.name} + {_expr_to_c(expr)}"

    def emit_free(self):
        common.writer.get().writeline(f"free({self.name});")

    @property
    def declared_type(self) -> str:
        return self.dtype.c_type


@dataclasses.dataclass(frozen=True)
class CHeapArray(CNameTensor):
    name: str
    size: int
    dtype: Dtype

    def emit(self, zero_init=True) -> "CHeapArray":
        writer = common.writer.get()
        writer.writeline(f"{self.dtype.c_type} *restrict {self.name};")
        writer.writeline(
            f"posix_memalign((void **)&{self.name}, 128, {self.size}*sizeof({self.dtype.c_type}));  // TODO: Handle return"
        )
        if zero_init:
            writer.writeline(
                f"memset({self.name}, 0, {self.size}*sizeof({self.dtype.c_type}));"
            )
        elif ONES_FOR_NON_ZERO_INIT.get():
            writer.writeline(
                f"memset({self.name}, 1, {self.size}*sizeof({self.dtype.c_type}));"
            )
        return self

    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        if reinterpret:
            raise NotImplementedError()
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr, reinterpret: Optional[str] = None):
        if reinterpret:
            raise NotImplementedError()
        return f"({self.name} + {_expr_to_c(expr)})"

    def emit_free(self):
        common.writer.get().writeline(f"free({self.name});")

    @property
    def declared_type(self) -> str:
        return self.dtype.c_type


@dataclasses.dataclass(frozen=True)
class CStackArray(CNameTensor):
    name: str
    size: int
    dtype: Dtype

    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        if reinterpret:
            raise NotImplementedError()
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr, reinterpret: Optional[str] = None):
        if reinterpret:
            raise NotImplementedError()
        return "&" + self.c_index(expr)

    def emit_free(self):
        pass

    def emit(self, zero_init=True) -> "CStackArray":
        epi = ""
        if zero_init:
            epi = " = {{0}}"
        common.writer.get().writeline(
            f"{self.dtype.c_type} {self.name}[{self.size}] __attribute__((aligned (128))){epi};"
        )
        if not zero_init and ONES_FOR_NON_ZERO_INIT.get():
            common.writer.get().writeline(f"for (uint32_t i = 0; i < {self.size}; ++i)")
            with common.writer.get().indent_block():
                common.writer.get().writeline(f"{self.name}[i] = 1;")
        return self


class CVecVars(CTensor):
    tensor_shape: tuple[int, ...]
    vector_shape: tuple[int, ...]
    dtype: Dtype
    _inner_vecs: list[CNameTensor]

    def __init__(
        self,
        namer: common.Namer,
        tile_shape: tuple[int, ...],
        vector_shape: tuple[int, ...],
        dtype: Dtype,
    ) -> None:
        super().__init__()
        self.tensor_shape = tile_shape
        self.vector_shape = vector_shape
        self.dtype = dtype

        # vector_shape is essentially just a SimpleTile over tile_shape.
        self._vector_size = functools.reduce(operator.mul, vector_shape, 1)

        self._tensor_step_sizes = self.compute_step_sizes(self.tensor_shape)
        self._vectors_in_tensor_step_sizes = self.compute_step_sizes(
            [((t + (v - 1)) // v) for t, v in zip(self.tensor_shape, self.vector_shape)]
        )
        self._vector_step_sizes = self.compute_step_sizes(self.vector_shape)

        self._inner_vecs = []
        for pair in itertools.product(
            *[self._range_vectors_single_dim(d) for d in range(len(tile_shape))]
        ):
            if all(b is None for _, b in pair):
                self._inner_vecs.append(
                    _CSingleVecVar(namer.fresh_name("vbuf"), self._vector_size, dtype)
                )
                continue

            boundary_vol = 1
            for dim_idx, (_, size) in enumerate(pair):
                if not size:
                    size = tile_shape[dim_idx]
                boundary_vol *= size
            self._inner_vecs.append(
                CHeapArray(namer.fresh_name("vbound"), boundary_vol, dtype)
            )

    @property
    def should_unroll(self) -> bool:
        return True

    def compute_step_sizes(self, shape: Sequence[int]) -> list[int]:
        step_sizes = []
        for dim_idx in range(len(shape)):
            step_sizes.append(functools.reduce(operator.mul, shape[dim_idx + 1 :], 1))
        return step_sizes

    @property
    def vec_names(self) -> Iterable[str]:
        return (iv.name for iv in self._inner_vecs if isinstance(iv, _CSingleVecVar))

    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        return self._delegate_c_index_call("c_index", expr, reinterpret)

    def c_index_ptr(self, expr, reinterpret: Optional[str] = None):
        return self._delegate_c_index_call("c_index_ptr", expr, reinterpret)

    def c_index_vec(self, expr, reinterpret: Optional[str] = None):
        return self._delegate_c_index_call("c_index_vec", expr, reinterpret)

    def _delegate_c_index_call(self, func_name, expr, reinterpret):
        inner_vec, vec_offset = self._inner_vec_from_expr(expr)
        meth = getattr(inner_vec, func_name)
        return meth(vec_offset, reinterpret=reinterpret)

    def emit_free(self):
        for inner_vec in self._inner_vecs:
            inner_vec.emit_free()

    def emit(self, zero_init=True) -> "CVecVars":
        for inner_vec in self._inner_vecs:
            inner_vec.emit(zero_init=zero_init)
        return self

    @property
    def declared_type(self) -> str:
        return GCC_VEC_TYPES[(self.dtype, self._vector_size)][1]

    def _range_vectors_single_dim(
        self, dim_idx: int
    ) -> Iterable[tuple[int, Optional[int]]]:
        full_steps = self.tensor_shape[dim_idx] // self.vector_shape[dim_idx]
        boundary_size = self.tensor_shape[dim_idx] % self.vector_shape[dim_idx]
        for step in range(full_steps):
            yield step, None
        if boundary_size:
            yield full_steps, boundary_size

    def _inner_vec_from_expr(self, expr) -> tuple["CNameTensor", int]:
        assert isinstance(expr, sympy.Expr)

        # Extract the constant from the expression, which we expect to be a sum of
        # _i0, _i1, ... and some constant.
        expr_constant, linear_terms = expr.as_coeff_add()
        expr_constant = int(expr_constant)

        # CVecVars reinterprets the (linearized) offset given by `expr` as though it
        # were an offset into a row-major tensor with a simple tiling applied. We
        # convert expr into a row-major tensor coordinate, then choose the vector (tile)
        # into which the coordinate falls assuming all symbols are zero (i.e. based on
        # the constant alone), then apply a row-major layout to the vector "tile" to
        # determine the individual vector offset expression.

        tensor_coord = self._offset_to_rm_tensor_coordinate(expr_constant)

        vector_coord, _ = self._tile_to_vector_coordinate(expr, tensor_coord)
        # TODO: Inline below
        idx = self._coord_to_rm_offset(vector_coord, self._vectors_in_tensor_step_sizes)
        inner_vec = self._inner_vecs[idx]

        inside_vec_coord = [t % v for t, v in zip(tensor_coord, self.vector_shape)]
        inside_vec_offset = self._coord_to_rm_offset(
            inside_vec_coord, self._vector_step_sizes
        )
        inside_vec_expr = inside_vec_offset
        for term in linear_terms:
            inside_vec_expr += term

        return inner_vec, inside_vec_expr

    # TODO: Factor this out somewhere. It happens a lot.
    @staticmethod
    def _coord_to_rm_offset(pt: Sequence[int], inner_step_sizes) -> int:
        return sum(pt * inner_volume for pt, inner_volume in zip(pt, inner_step_sizes))

    def _offset_to_rm_tensor_coordinate(self, offset: int) -> list[int]:
        remaining_offset = offset
        result = []
        for step_size in self._tensor_step_sizes:
            result.append(remaining_offset // step_size)
            remaining_offset -= result[-1] * step_size
        assert len(result) == len(self.tensor_shape)
        assert remaining_offset == 0
        return result

    def _tile_to_vector_coordinate(
        self, expr, tile_coord: Sequence[int]
    ) -> tuple[Sequence[int], int]:
        vec_coord = [t // v for t, v in zip(tile_coord, self.vector_shape)]
        vec_origin = [c * v for c, v in zip(vec_coord, self.vector_shape)]
        vec_origin_offset = self._coord_to_rm_offset(
            vec_origin, self._tensor_step_sizes
        )
        remainder = expr - vec_origin_offset
        return vec_coord, remainder

    def _singlevec_from_expr(self, expr) -> "_CSingleVecVar":
        if not (isinstance(expr, sympy.Expr) and expr.is_constant()):
            raise ValueError(f"expr was not a constant Expr, was: {expr}")
        expr = int(expr)
        if expr % self._vector_size != 0:
            raise ValueError(
                f"Index expression {expr} did not point to the first element of "
                "a vector"
            )
        single_vec = self._inner_vecs[expr // self._vector_size]
        assert isinstance(single_vec, _CSingleVecVar)
        return single_vec


@dataclasses.dataclass(frozen=True)
class _CSingleVecVar(CNameTensor):
    name: str
    size: int
    dtype: Dtype

    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        if reinterpret:
            assert expr == sympy.core.numbers.Zero()
            return f"*({reinterpret} *)(&{self.name})"
        return f"{self.name}[{_expr_to_c(expr)}]"

    def c_index_ptr(self, expr, reinterpret: Optional[str] = None):
        if reinterpret:
            raise NotImplementedError()
        if expr == 0:
            return "&" + self.name
        return "&" + self.c_index(expr)

    def c_index_vec(self, expr, reinterpret: Optional[str] = None) -> str:
        if expr != 0:
            raise ValueError("expr must be 0, but was: " + str(expr))
        if reinterpret:
            raise NotImplementedError()
        return f"{self.name}"

    def emit_free(self):
        pass

    def emit(self, zero_init=True) -> "_CSingleVecVar":
        # Allocate and zero the register.
        epi = ""
        if zero_init:
            epi = " = {{0}}"
        elif ONES_FOR_NON_ZERO_INIT.get():
            epi = f" = 1 - ({self.declared_type}){{}}"
        common.writer.get().writeline(f"{self.declared_type} {self.name}{epi};")
        return self

    @property
    def declared_type(self) -> str:
        return GCC_VEC_TYPES[(self.dtype, self.size)][1]


@dataclasses.dataclass(frozen=True)
class CValueVar(CNameTensor):
    name: str
    dtype: Dtype

    def c_index(self, expr, reinterpret: Optional[str] = None) -> str:
        if reinterpret:
            raise NotImplementedError()
        # TODO: Check that expr evaluates to 0
        return self.name

    def emit_free(self):
        pass

    def emit(self, zero_init=True) -> "CValueVar":
        epi = ""
        if zero_init:
            epi = " = 0"
        elif ONES_FOR_NON_ZERO_INIT.get():
            epi = " = 1"
        common.writer.get().writeline(f"{self.dtype.c_type} {self.name}{epi};")
        return self
