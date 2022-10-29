import dataclasses
import functools
import math
import operator
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

import sympy

from . import dtypes, system_config
from .codegen.expr_utils import FloorDiv

if TYPE_CHECKING:
    from . import dtypes


class Layout:
    """The layout of a tensor."""

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        raise NotImplementedError()

    def contiguous_top(self) -> Any:
        raise NotImplementedError()

    def all_contiguous_abs_for_shape(self) -> Iterable[Any]:
        raise NotImplementedError()

    def tile_is_contiguous(self, contiguous_abs) -> bool:
        raise NotImplementedError()

    def check_tile_contiguity(
        self, tile_shape: Sequence[int], parent_shape: Sequence[int], parent_contiguous
    ) -> Any:
        """Test whether a tile of a particular shape and layout is contiguous.

        This returns `True` if a tile of a given shape and `self` layout, when
        coordinates are enumerated from the innermost `pn` to outermost `p0`
        logical dimension map to sequential addresses in memory.

        Notice: no information about tiling type is given, so the returned value
        must be true of any tile of any tensor of this layout.

        Implementations may under-approximate but will never return `True` if
        the tile is not contiguous.
        """
        raise NotImplementedError()

    def estimate_cache_lines(
        self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool
    ) -> int:
        """Count the cache lines used to store a tensor of a given TensorSpec.

        It may overestimate the true number.
        """
        raise NotImplementedError()

    def applies_to_shape(self, shape: Sequence[int]) -> bool:
        if all(d == 1 for d in shape) and not self.is_row_major:
            return False
        return True

    @property
    def is_row_major(self) -> bool:
        return False

    def dim_drop(
        self, dropped_dims: Iterable[int], contiguous_abs
    ) -> tuple["Layout", Any]:
        raise NotImplementedError()

    def transpose(
        self, swap_dims: tuple[int, int], contiguous_abs
    ) -> tuple["Layout", Any]:
        raise NotImplementedError()

    def flatten_inner_contiguous_dimensions(
        self, shape: Sequence[int], contiguous_abs
    ) -> Optional[tuple[tuple[int, ...], set[int], int]]:
        """Flatten physically innermost dimensions.

        Returns `None` if there are insufficient contiguous physically innermost
        dimensions to flatten.
        """
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class StandardLayout(Layout):

    dim_order: tuple[int, ...]

    def __post_init__(self):
        assert all(d >= 0 for d in self.dim_order)
        assert len(set(self.dim_order)) == len(self.dim_order)

    def contiguous_top(self) -> Any:
        return len(self.dim_order)

    @staticmethod
    def contiguous_one() -> Any:
        """Returns contig. abstractions for just the physically innermost dimension."""
        return 1

    def all_contiguous_abs_for_shape(self) -> Iterable[Any]:
        yield from range(len(self.dim_order) + 1)

    def tile_is_contiguous(self, contiguous_abs) -> bool:
        assert contiguous_abs >= 0 and contiguous_abs <= len(self.dim_order)
        return contiguous_abs == len(self.dim_order)

    def check_tile_contiguity(
        self, tile_shape: Sequence[int], parent_shape: Sequence[int], parent_contiguous
    ) -> Any:
        if all(d == 1 for d in tile_shape):
            return self.contiguous_top()

        cnt = 1  # Skip first.

        def inner_loop(offset: int, comp):
            nonlocal cnt
            while cnt < len(tile_shape):
                phys_idx = self.dim_order[-cnt + offset]
                if tile_shape[phys_idx] != comp(phys_idx):
                    break
                cnt += 1

        inner_loop(0, lambda x: parent_shape[x])
        cnt = min(cnt, parent_contiguous)
        inner_loop(-1, lambda _: 1)
        return cnt

    def estimate_cache_lines(
        self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool
    ) -> int:
        line_size = system_config.current_system().line_size
        if contiguous:
            return math.ceil(
                (functools.reduce(operator.mul, shape, 1) * dtype.size) / line_size
            )
        else:
            lodims = self._layout_ordered_dims(shape)
            real_dims = [d for d in lodims if d > 1]
            if not real_dims:
                real_dims = [1]
            return functools.reduce(operator.mul, real_dims[:-1], 1) * math.ceil(
                (real_dims[-1] * dtype.size) / line_size
            )

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        assert len(concrete_shape) == len(self.dim_order)
        r = _general_index_expr(self.dim_order, concrete_shape)
        if isinstance(r, int):
            return sympy.Integer(r)
        return r

    @property
    def is_row_major(self) -> bool:
        if self.dim_order == tuple(range(len(self.dim_order))):
            return True
        return False

    def applies_to_shape(self, shape: Sequence[int]) -> bool:
        if not super().applies_to_shape(shape):
            return False
        if len(shape) != len(self.dim_order):
            return False
        return True

    def dim_drop(
        self, dropped_dims: Iterable[int], contiguous_abs
    ) -> tuple["Layout", Any]:
        dropped_dims = set(dropped_dims)
        if not dropped_dims:
            return self, contiguous_abs

        new_dim_order = []
        for logical_dim in self.dim_order:  # Iterate toward physically inner
            if logical_dim not in dropped_dims:
                offset = sum(1 for d in dropped_dims if d < logical_dim)
                new_dim_order.append(logical_dim - offset)

        new_contiguous = contiguous_abs
        if contiguous_abs != 0:
            for logical_dim_inside_contig in self.dim_order[-contiguous_abs:]:
                if logical_dim_inside_contig in dropped_dims:
                    new_contiguous -= 1

        return StandardLayout(tuple(new_dim_order)), new_contiguous

    def transpose(
        self, swap_dims: tuple[int, int], contiguous_abs
    ) -> tuple["Layout", Any]:
        if swap_dims[0] >= swap_dims[1]:
            raise ValueError("Dims. must be ordered, but given: {swap_dims}")
        new_dim_order = []
        for orig_dim in self.dim_order:
            if orig_dim == swap_dims[0]:
                new_dim_order.append(swap_dims[1])
            elif orig_dim == swap_dims[1]:
                new_dim_order.append(swap_dims[0])
            else:
                new_dim_order.append(orig_dim)
        return StandardLayout(tuple(new_dim_order)), contiguous_abs

    def flatten_inner_contiguous_dimensions(
        self, shape: Sequence[int], contiguous_abs
    ) -> Optional[tuple[tuple[int, ...], set[int], int]]:
        if len(shape) != len(self.dim_order):
            raise ValueError(
                f"Shape {shape} has the wrong number of dimensions; expected "
                f"{len(self.dim_order)}"
            )
        if contiguous_abs <= 1:
            return None
        prefix = tuple(shape[l] for l in self.dim_order[:-contiguous_abs])
        flat_dims = set(self.dim_order[-contiguous_abs:])
        inner_vol = functools.reduce(operator.mul, (shape[l] for l in flat_dims), 1)
        return prefix, flat_dims, inner_vol

    def _layout_ordered_dims(self, dim_sizes: Sequence[int]) -> tuple[int, ...]:
        assert len(dim_sizes) == len(
            self.dim_order
        ), f"Expected {len(self.dim_order)} dimensions, but given: {dim_sizes}"
        return tuple(dim_sizes[d] for d in self.dim_order)

    def __str__(self) -> str:
        if self.is_row_major:
            return "RM"
        if self.dim_order == (0, 2, 3, 1):
            return "NHWC"
        return f"<{','.join(map(str, self.dim_order))}>"


@dataclasses.dataclass(frozen=True)
class PackedLayout(Layout):
    dim_count: int
    strip_dim: int
    strip_size: int

    def __post_init__(self):
        if self.strip_dim >= self.dim_count:
            raise ValueError(
                f"PackedLayout has {self.dim_count} dimensions, but strip_dim "
                f"is {self.strip_dim}"
            )
        if self.strip_dim == self.dim_count - 1:
            raise ValueError(
                f"Strip dim. {self.strip_dim} cannot be the innermost logical "
                "dimension; that is equivalent to a StandardLayout"
            )

    def contiguous_top(self) -> Any:
        return self.dim_count + 1

    def all_contiguous_abs_for_shape(self) -> Iterable[Any]:
        yield from range(self.dim_count + 2)

    def tile_is_contiguous(self, contiguous_abs) -> bool:
        assert contiguous_abs >= 0 and contiguous_abs <= self.dim_count + 1
        return contiguous_abs == self.dim_count + 1

    # TODO: Prefix calls with layout-checking assertions
    def check_tile_contiguity(
        self, tile_shape: Sequence[int], parent_shape: Sequence[int], parent_contiguous
    ) -> Any:
        if len(parent_shape) != self.dim_count:
            raise ValueError(f"Expected rank-{self.dim_count} outer shape")
        if len(tile_shape) != self.dim_count:
            raise ValueError(f"Expected rank-{self.dim_count} tile")
        expanded_parent_shape = self.expand_shape(parent_shape)
        expanded_tile_shape = self.expand_shape(tile_shape)
        return row_major(len(expanded_parent_shape)).check_tile_contiguity(
            expanded_tile_shape, expanded_parent_shape, parent_contiguous
        )

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        if len(concrete_shape) != self.dim_count:
            raise ValueError(f"Expected rank-{self.dim_count} shape")

        if self._should_fall_back_to_row_major(concrete_shape):
            return row_major(self.dim_count).buffer_indexing_expr(concrete_shape)

        packing_p, last_p = sympy.symbols(f"p{self.strip_dim} p{len(concrete_shape)}")
        expanded = self.expand_shape(concrete_shape)
        idx_expr = row_major(len(expanded)).buffer_indexing_expr(expanded)
        idx_expr = idx_expr.subs(
            [
                (packing_p, FloorDiv(packing_p, self.strip_size)),
                (last_p, packing_p % self.strip_size),
            ],
            simultaneous=True,
        )
        return idx_expr

    def estimate_cache_lines(
        self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool
    ) -> int:
        # Estimate as if the tensor_spec were in row-major but had an extra dim.
        rm_like_shape = self.expand_shape(shape)
        new_contiguous = False
        return row_major(len(rm_like_shape)).estimate_cache_lines(
            rm_like_shape, dtype, new_contiguous
        )

    def applies_to_shape(self, shape: Sequence[int]) -> bool:
        if not super().applies_to_shape(shape):
            return False
        if self.dim_count != len(shape):
            return False
        # Only applies when the strip dimension is a multiple of the strip size.
        # TODO: Relax this.
        if shape[self.strip_dim] % self.strip_size != 0:
            return False
        return True

    def dim_drop(
        self, dropped_dims: Iterable[int], contiguous_abs
    ) -> tuple[Layout, Any]:
        dropped_dims = set(dropped_dims)
        if not dropped_dims:
            return self, contiguous_abs

        if self.strip_dim in dropped_dims:
            rm_contig = max(0, contiguous_abs - 1)
            return row_major(self.dim_count).dim_drop(dropped_dims, rm_contig)

        after_strip_dims = set(range(self.strip_dim + 1, self.dim_count))
        assert after_strip_dims, (
            "There must be dimensions after the strip dim., otherwise this is "
            "really a StandardLayout"
        )
        if dropped_dims.issuperset(after_strip_dims):
            return row_major(self.strip_dim + 1).dim_drop(
                dropped_dims - after_strip_dims,
                max(0, contiguous_abs - len(after_strip_dims) - 1),
            )

        fifth_dim_contig = min(1, contiguous_abs)  # 1 or 0
        standard_contig = max(0, contiguous_abs - 1)
        contig_dropped = sum(
            1 for d in dropped_dims if self.dim_count - d <= standard_contig
        )
        return (
            PackedLayout(
                dim_count=self.dim_count - len(dropped_dims),
                strip_dim=self.strip_dim,
                strip_size=self.strip_size,
            ),
            fifth_dim_contig + standard_contig - contig_dropped,
        )

    def expand_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        """Factor a shape's strip dimension into a new innermost dimension.

        For example:
        >>> l = PackedLayout(dim_count=3, strip_dim=1, strip_size=4)
        >>> l.expand_shape((2, 8, 2))
        (2, 2, 2, 4)
        """
        # TODO: The above doctest should be under pytest, but pytest has
        # difficulty resolving paths with Cython-built extension modules.

        new_shape = list(shape)
        new_shape[self.strip_dim] = math.ceil(
            new_shape[self.strip_dim] / self.strip_size
        )
        new_shape.append(self.strip_size)
        assert all(d > 0 for d in new_shape)
        return tuple(new_shape)

    def flatten_inner_contiguous_dimensions(
        self, shape: Sequence[int], contiguous_abs
    ) -> Optional[tuple[tuple[int, ...], set[int], int]]:
        if contiguous_abs == 0:
            return None
        expanded = self.expand_shape(shape)
        prefix = expanded[:-contiguous_abs]
        contiguous_inner_shape = expanded[-contiguous_abs:]

        dropped_dimensions = {self.strip_dim}
        packing_p = sympy.symbols(f"p{self.strip_dim}")
        v = functools.reduce(operator.mul, contiguous_inner_shape, 1)
        return prefix, dropped_dimensions, v

    def _should_fall_back_to_row_major(self, shape: Sequence[int]) -> bool:
        if shape[self.strip_dim] % self.strip_size != 0:
            return True
        return False

    def __str__(self):
        if self == NCHWc4:
            return "NCHWc4"
        elif self == NCHWc32:
            return "NCHWc32"
        elif self == NCHWc64:
            return "NCHWc64"
        return f"pack({self.dim_count}, {self.strip_dim}, {self.strip_size})"


class HexagonTranspacked(Layout):
    def applies_to_shape(self, shape: Sequence[int]) -> bool:
        from .dtypes import Uint8

        if not super().applies_to_shape(shape):
            return False

        raise NotImplementedError("Should check that this only applies to uint8 tensor")

        if len(shape) != 2:
            return False
        if shape[0] % 4 != 0 or shape[1] % 32 != 0:
            return False
        return True

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        # This layout is only used for Uint8, so the following will index
        # 128-bit blocks (vectors in HVX VMEM).
        orig_rows, orig_cols = concrete_shape
        padded_rows = orig_rows - orig_rows % -32
        p0, p1 = sympy.symbols("p0 p1")
        # Logical sizes for the 128-byte vectors.
        row_block = FloorDiv(p0, 4)
        col_block = FloorDiv(p1, 32)
        # The blocks in each dimension might differ because of padding, and this
        # can affect the offsets (e.g., by added intervening all-zero vectors).
        inner_offset = 4 * sympy.UnevaluatedExpr(p1 % 32) + sympy.UnevaluatedExpr(
            p0 % 4
        )
        block_rows = padded_rows // 4
        block_offset = (128 * block_rows * col_block) + row_block  # block offset
        return block_offset + inner_offset

    def __str__(self) -> str:
        return "TP"


NHWC = StandardLayout((0, 2, 3, 1))
NCHWc4 = PackedLayout(4, 1, 4)
NCHWc32 = PackedLayout(4, 1, 32)
NCHWc64 = PackedLayout(4, 1, 64)
HEXAGON_TRANSPACKED = HexagonTranspacked()  # singleton


def row_major(rank: int) -> StandardLayout:
    return StandardLayout(tuple(range(rank)))


def _general_index_expr(
    logical_dims: Sequence[int], shape: Sequence[int]
) -> Union[sympy.Expr, int]:
    assert logical_dims
    t, remaining_dims = logical_dims[-1], logical_dims[:-1]
    s = shape[t]
    p = sympy.symbols(f"p{t}") if s > 1 else 0
    if not remaining_dims:
        return p
    return _general_index_expr(remaining_dims, shape) * s + p
