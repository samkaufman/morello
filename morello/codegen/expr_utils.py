import re

import sympy

_POINT_SYMBOL_RE = re.compile(r"^p(\d+)$")


class FloorDiv(sympy.Function):
    nargs = 2

    @classmethod
    def eval(cls, n, d):
        if n.is_Number and d.is_Number:
            return sympy.Integer(n // d)
        return None


def zero_points(expr: sympy.Expr) -> sympy.Expr:
    substitutions = {}
    for sym in expr.free_symbols:
        if _POINT_SYMBOL_RE.match(sym.name):
            substitutions[sym] = 0
    return expr.subs(substitutions)