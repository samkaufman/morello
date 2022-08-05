import abc
import dataclasses
from typing import Optional, Union

import sympy

from ..dtypes import Dtype, Uint32, Uint8
from . import common, expr_utils
from .indexexpr import vsub

GCC_VEC_TYPES: dict[tuple[Dtype, int], tuple[int, str, str]] = {
    # TODO: Fix _BROADCAST_VEC_MULT_WIDTH
    (Uint32, 8): (32, "vui8", "__m256i"),
    (Uint8, 32): (32, "vub32", "__m256i"),
}


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

    def emit(self, zero_init=True) -> "CNameTensor":
        raise NotImplementedError()

    def emit_free(self):
        raise NotImplementedError()

    @property
    def declared_type(self) -> str:
        raise NotImplementedError()


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

    def emit(self):
        # return self
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

    def emit(self) -> "CStackArray":
        common.writer.get().writeline(
            f"{self.dtype.c_type} {self.name}[{self.size}] __attribute__((aligned (128))) = {{0}};"
        )
        return self


@dataclasses.dataclass(frozen=True)
class CVecVar(CNameTensor):
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
        if expr == sympy.core.numbers.Zero():
            return "&" + self.name
        return "&" + self.c_index(expr)

    def emit_free(self):
        pass

    def emit(self) -> "CVecVar":
        # Allocate and zero the register.
        common.writer.get().writeline(f"{self.declared_type} {self.name} = {{0}};")
        return self

    def vec(self) -> str:
        return self.c_index(0, reinterpret=self.declared_type)

    @property
    def declared_type(self) -> str:
        return GCC_VEC_TYPES[(self.dtype, self.size)][1]

    @staticmethod
    def accepts(dtype: Dtype, size: int) -> bool:
        """Returns True if there exists a vector type for dtype and byte count."""
        return (dtype, size) in GCC_VEC_TYPES


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

    def emit(self) -> "CValueVar":
        common.writer.get().writeline(f"{self.dtype.c_type} {self.name} = 0;")
        return self

