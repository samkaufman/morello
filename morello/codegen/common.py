import contextlib
import contextvars
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy

    from .. import impl
    from ..tensor import TensorLike
    from .ctensors import CTensor


namer: contextvars.ContextVar["Namer"] = contextvars.ContextVar("_namer")
writer: contextvars.ContextVar["Writer"] = contextvars.ContextVar("_writer")
unroll: contextvars.ContextVar["bool"] = contextvars.ContextVar(
    "_unroll", default=False
)

@dataclasses.dataclass(frozen=True)
class OperandDetails:
    """Data about an Impl operand commonly moved together during codegen."""

    def __post_init__(self):
        assert all(d > 0 for d in self.concrete_origin_shape)

    c_tensor: "CTensor"
    index_expr: "sympy.Expr"
    # concrete_origin_shape is usually greater than a tile size in each
    # corresponding dimension, but might be smaller in boundary cases
    # (where the tile is effectively truncated).
    concrete_origin_shape: tuple[int, ...]
    previously_transformed_tiles: frozenset["TensorLike"]


class Namer:
    def __init__(self):
        self.counts = {}

    def fresh_name(self, prefix: str = "v") -> str:
        cnt = self.counts.setdefault(prefix, 0)
        new_name = prefix + str(cnt)
        self.counts[prefix] += 1
        return new_name


class Writer:
    def __init__(self, fo):
        self._fo = fo
        self._prefix = ""
        self._pending_rhs_comment = None

    def indent(self):
        self._prefix += " " * 2

    def dedent(self):
        self._prefix = self._prefix[:-2]

    def set_impl(self, imp: "impl.Impl"):
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

