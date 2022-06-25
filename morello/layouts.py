import dataclasses
import functools
import itertools
import math
import operator
from typing import TYPE_CHECKING, Sequence, Union

import sympy
from torch import normal

from . import dtypes, system_config
from .codegen.expr_utils import FloorDiv

if TYPE_CHECKING:
    from . import dtypes


class Layout:
    """The layout of a tensor."""

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        raise NotImplementedError()

    def check_tile_contiguity(
        self, outer_shape: Sequence[int], tile_shape: Sequence[int]
    ) -> bool:
        raise NotImplementedError()

    def estimate_cache_lines(self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool) -> int:
        """Count the cache lines used to store a tensor of a given TensorSpec.

        It may overestimate the true number.
        """
        raise NotImplementedError()

    def applies_to_shape(self, shape: Sequence[int], dtype: "dtypes.Dtype") -> bool:
        return True
    
    def normalize(self) -> "Layout":
        return self
    
    @property
    def is_row_major(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class DimDropLayout(Layout):
    inner: Layout
    dropped_dims: frozenset[int]

    def __post_init__(self):
        assert len(self.dropped_dims)

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        e = self.inner.buffer_indexing_expr(self._explode_shape(concrete_shape))
        # TODO: Assert that dropped dims aren't present in returned expression
        return e

    def check_tile_contiguity(
        self, outer_shape: Sequence[int], tile_shape: Sequence[int]
    ) -> bool:
        return self.inner.check_tile_contiguity(
            self._explode_shape(outer_shape), self._explode_shape(tile_shape)
        )

    def estimate_cache_lines(self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool) -> int:
        return self.inner.estimate_cache_lines(self._explode_shape(shape), dtype, contiguous)

    def normalize(self) -> Layout:
        normalized = self.inner.normalize()
        if isinstance(normalized, StandardLayout):
            removed = 0
            new_dim_order = []
            for orig_dim in normalized.dim_order:
                if orig_dim in self.dropped_dims:
                    removed += 1
                else:
                    new_dim_order.append(orig_dim - removed)
            return StandardLayout(tuple(new_dim_order))

        if isinstance(normalized, PackedLayout):
            if normalized.strip_dim in self.dropped_dims:
                return DimDropLayout(
                    inner=row_major(normalized.dim_count),
                    dropped_dims=self.dropped_dims,
                ).normalize()
            
            after_strip_dim = frozenset(range(normalized.strip_dim + 1, normalized.dim_count))
            if after_strip_dim:
                if self.dropped_dims == after_strip_dim:
                    return row_major(normalized.strip_dim + 1).normalize()
                elif self.dropped_dims.issuperset(after_strip_dim):
                    return DimDropLayout(
                        inner=row_major(normalized.strip_dim + 1),
                        dropped_dims=self.dropped_dims - after_strip_dim,
                    ).normalize()

        return normalized

    def _explode_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        inner_shape = list(shape)
        for d in sorted(self.dropped_dims, reverse=True):
            inner_shape.insert(d, 1)
        return tuple(inner_shape)
    
    def __str__(self):
        return str(self.inner) + ".drop"


@dataclasses.dataclass(frozen=True)
class TransposeLayout(Layout):
    inner: Layout
    swap_dims: tuple[int, int]

    def __post_init__(self):
        assert self.swap_dims[0] < self.swap_dims[1]

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        e = self.inner.buffer_indexing_expr(self._transpose_shape(concrete_shape))
        e = e.subs(
            [
                (f"p{self.swap_dims[0]}", f"p{self.swap_dims[1]}"),
                (f"p{self.swap_dims[1]}", f"p{self.swap_dims[0]}"),
            ],
            simultaneous=True,
        )
        return e

    def check_tile_contiguity(
        self, outer_shape: Sequence[int], tile_shape: Sequence[int]
    ) -> bool:
        transposed_outer_shape = self._transpose_shape(outer_shape)
        transposed_tile_shape = self._transpose_shape(tile_shape)
        return self.inner.check_tile_contiguity(
            transposed_outer_shape, transposed_tile_shape
        )

    def estimate_cache_lines(self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool) -> int:
        transposed_shape = self._transpose_shape(shape)
        return self.inner.estimate_cache_lines(transposed_shape, dtype, contiguous)
    
    def normalize(self) -> Layout:
        normalized = self.inner.normalize()
        if isinstance(normalized, StandardLayout):
            new_dim_order = []
            for orig_dim in normalized.dim_order:
                if orig_dim == self.swap_dims[0]:
                    new_dim_order.append(self.swap_dims[1])
                elif orig_dim == self.swap_dims[1]:
                    new_dim_order.append(self.swap_dims[0])
                else:
                    new_dim_order.append(orig_dim)
            return StandardLayout(tuple(new_dim_order))
        elif isinstance(normalized, PackedLayout):
            raise NotImplementedError()
        else:
            return normalized


    def __str__(self):
        return str(self.inner) + ".T"

    def _transpose_shape(self, shape):
        transposed_shape = list(shape)
        i, j = self.swap_dims
        transposed_shape[i], transposed_shape[j] = (
            transposed_shape[j],
            transposed_shape[i],
        )
        return tuple(transposed_shape)


@dataclasses.dataclass(frozen=True)
class StandardLayout(Layout):

    dim_order: tuple[int, ...]

    def check_tile_contiguity(
        self, outer_shape: Sequence[int], tile_shape: Sequence[int]
    ) -> bool:
        self_ordered_dims = self._layout_ordered_dims(tile_shape)
        address_root_ordered_dims = self._layout_ordered_dims(outer_shape)

        pairs = zip(self_ordered_dims, address_root_ordered_dims)
        pairs = itertools.dropwhile(lambda x: x[0] == 1, pairs)
        pairs = itertools.islice(pairs, 1, None)
        for tile_dim_size, root_dim_size in pairs:
            # The following includes the case where an underlying dimension is 1.
            if tile_dim_size != root_dim_size:
                return False
        return True

    def estimate_cache_lines(self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool) -> int:
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

    def _layout_ordered_dims(self, dim_sizes: Sequence[int]) -> tuple[int, ...]:
        assert len(dim_sizes) == len(self.dim_order), (
            f"Expected {len(self.dim_order)} dimensions, but given: {dim_sizes}"
        )
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
        # TODO: Instead of asserting below, add a test that this is equivalent
        assert self.strip_dim + 1 < self.dim_count

    # def applies_to(self, shape: Sequence[int]) -> bool:
    #     if len(shape) != self.dim_count:
    #         return False
    #     if shape[self.strip_dim] % self.strip_size != 0:
    #         return False
    #     return True

    # TODO: Prefix calls with layout-checking assertions
    def check_tile_contiguity(
        self, outer_shape: Sequence[int], tile_shape: Sequence[int]
    ) -> bool:
        if len(outer_shape) != self.dim_count:
            raise ValueError(f"Expected rank-{self.dim_count} outer shape")
        if len(tile_shape) != self.dim_count:
            raise ValueError(f"Expected rank-{self.dim_count} tile")
        expanded_outer = self._expand_shape(outer_shape)
        expanded_tile = self._expand_shape(tile_shape)
        return row_major(len(expanded_outer)).check_tile_contiguity(
            expanded_outer, expanded_tile
        )

    def buffer_indexing_expr(self, concrete_shape: Sequence[int]) -> sympy.Expr:
        if len(concrete_shape) != self.dim_count:
            raise ValueError(f"Expected rank-{self.dim_count} shape")
        
        if self._should_fall_back_to_row_major(concrete_shape):
            return row_major(self.dim_count).buffer_indexing_expr(concrete_shape)
        
        packing_p, last_p = sympy.symbols(f"p{self.strip_dim} p{len(concrete_shape)}")
        expanded = self._expand_shape(concrete_shape)
        idx_expr = row_major(len(expanded)).buffer_indexing_expr(expanded)
        idx_expr = idx_expr.subs(
            [
                (packing_p, FloorDiv(packing_p, self.strip_size)),
                (last_p, packing_p % self.strip_size),
            ],
            simultaneous=True,
        )
        return idx_expr

    def estimate_cache_lines(self, shape: Sequence[int], dtype: dtypes.Dtype, contiguous: bool) -> int:
        # Estimate as if the tensor_spec were in row-major but had an extra dim.
        rm_like_shape = self._expand_shape(shape)

        # TODO: Make this more precise. It's always False right now.
        new_contiguous = False

        return row_major(len(rm_like_shape)).estimate_cache_lines(rm_like_shape, dtype, new_contiguous)

    def applies_to_shape(self, shape: Sequence[int], dtype: "dtypes.Dtype") -> bool:
        if self.dim_count != len(shape):
            return False
        # Only applies when the strip dimension is a multiple of the strip size. 
        # TODO: Relax this.
        if shape[self.strip_dim] % self.strip_size != 0:
            return False
        return True

    def _expand_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        new_shape = list(shape)
        new_shape[self.strip_dim] = math.ceil(new_shape[self.strip_dim] / self.strip_size)
        new_shape.append(self.strip_size)
        assert all(d > 0 for d in new_shape)
        return tuple(new_shape)
    
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
    def applies_to_shape(self, shape: Sequence[int], dtype: "dtypes.Dtype") -> bool:
        from .dtypes import Uint8

        if not super().applies_to_shape(shape, dtype):
            return False

        if dtype != Uint8:
            return False
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
    assert len(shape) == len(logical_dims)
    t, head = logical_dims[-1], logical_dims[:-1]
    s, thead = shape[-1], shape[:-1]
    p = sympy.symbols(f"p{t}") if s > 1 else 0
    if not head:
        return p
    return _general_index_expr(head, thead) * s + p


def _tensor_col_major_indexing_expr(rank: int) -> sympy.Expr:
    if rank == 2:
        p0, p1, s0 = sympy.symbols("p0 p1 s0")
        return (p1 * s0) + p0
    elif rank > 2:
        s, p = sympy.symbols(f"s{rank - 1}, p{rank - 1}")
        return _tensor_col_major_indexing_expr(rank - 1) * s + p
    else:
        raise ValueError("rank must be at least 2, but was " + str(rank))
