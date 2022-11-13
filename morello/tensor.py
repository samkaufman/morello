import dataclasses
import math
import re
import typing
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import sympy

from . import dtypes, layouts, system_config

if TYPE_CHECKING:
    from . import specs

_PRE = re.compile(r"p(\d+)")

OperandIdx = typing.NewType("OperandIdx", int)


class DisallowedTileShapeError(ValueError):
    pass


class TensorLike:
    spec: "specs.TensorSpec"

    @property
    @typing.final
    def vector_shape(self) -> Optional[tuple[int, ...]]:
        return self.spec.vector_shape

    @property
    @typing.final
    def dim_sizes(self) -> tuple[int, ...]:
        return self.spec.dim_sizes

    @property
    @typing.final
    def layout(self) -> layouts.Layout:
        return self.spec.layout

    @property
    @typing.final
    def dtype(self) -> dtypes.Dtype:
        return self.spec.dtype

    @property
    @typing.final
    def bank(self) -> str:
        return self.spec.bank

    def replace_tensors(
        self, replacements: Mapping["TensorLike", "TensorLike"]
    ) -> "Tile":
        raise NotImplementedError("replace_tensors will be removed")

    @typing.final
    def __eq__(self, other):
        return self is other

    @typing.final
    def __hash__(self):
        return hash(id(self))

    @typing.final
    def steps(self, origin_shape: Sequence[int]) -> int:
        if len(origin_shape) != len(self.spec.dim_sizes):
            raise ValueError("origin_shape rank did not match Tile rank")
        s = 1
        for i, origin_dim_size in enumerate(origin_shape):
            s *= self.steps_dim(i, origin_dim_size)
        return s

    def steps_dim(self, dim: int, origin_size: int) -> int:
        raise NotImplementedError()

    def boundary_size(self, dim: int, origin_size: int) -> int:
        raise NotImplementedError()

    @property
    def frontiers(self) -> tuple[int, ...]:
        """The sizes of non-overlapping regions between consecutive tiles in each dimension."""
        raise NotImplementedError()

    def transform_index_expr(self, orig_index_expr: sympy.Expr) -> sympy.Expr:
        return orig_index_expr

    def transform_origin_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return tuple(shape)


class TensorBase(TensorLike):
    spec: "specs.TensorSpec"

    @property
    def dim_sizes(self) -> tuple[int, ...]:
        return self.spec.dim_sizes

    @property
    def root(self) -> "TensorBase":
        return self

    @property
    def bank(self) -> str:
        return self.spec.bank


@dataclasses.dataclass(frozen=True, eq=False)
class Tensor(TensorBase):
    """An n-dimensional array."""

    spec: "specs.TensorSpec"
    name: Optional[str]

    def __str__(self):
        layout_epi = ""
        if not self.layout.is_row_major:
            layout_epi = f", {self.layout}"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        return f"{type(self).__name__}({dims_part}{layout_epi}, {self.bank})"

    def steps_dim(self, dim: int, origin_size: int) -> int:
        return 1

    def boundary_size(self, dim: int, origin_size: int) -> int:
        return 0

    @property
    def frontiers(self) -> tuple[int, ...]:
        return tuple(0 for _ in self.dim_sizes)


@dataclasses.dataclass(frozen=True, eq=False)
class Tile(TensorLike):
    pass


@dataclasses.dataclass(frozen=True, eq=False)
class InnerContigFlatteningTile(Tile):
    source: OperandIdx
    inner: TensorLike
    flattening_contiguous_abs: Any

    @property
    def spec(self) -> "specs.TensorSpec":
        from . import specs

        ispec: "specs.TensorSpec" = self.inner.spec

        if ispec.vector_shape is not None:
            raise NotImplementedError()

        new_dim_sizes = self._flatten_shape()
        new_layout = layouts.row_major(len(new_dim_sizes))
        return specs.TensorSpec(
            dim_sizes=new_dim_sizes,
            dtype=ispec.dtype,
            contiguous_abs=new_layout.contiguous_one(),
            aligned=ispec.aligned,
            bank=ispec.bank,
            layout=new_layout,
        )

    @property
    def name(self) -> str:
        return self.inner.name

    def steps_dim(self, dim: int, origin_size: int) -> int:
        raise NotImplementedError()

    def boundary_size(self, dim: int, origin_size: int) -> int:
        raise NotImplementedError()

    @property
    def frontiers(self) -> tuple[int, ...]:
        raise NotImplementedError()

    def transform_index_expr(self, orig_index_expr: sympy.Expr) -> sympy.Expr:
        expr = orig_index_expr
        # expr = self.inner.transform_index_expr(expr)

        dim_sizes = self._flatten_shape()

        ispec = self.inner.spec
        flattened = ispec.layout.flatten_inner_contiguous_dimensions(
            ispec.dim_sizes, self.flattening_contiguous_abs
        )
        assert flattened
        _, dropped_dims, _ = flattened

        shift = 0
        drop_subs = {}
        for i in sorted(range(len(ispec.dim_sizes))):
            if i in dropped_dims:
                shift += 1
                drop_subs[f"p{i}"] = 0
            else:
                drop_subs[f"p{i}"] = sympy.symbols(f"p{i - shift}")
        expr = expr.subs(drop_subs, simultaneous=True)

        # Add a simple offset since the flattened dimensions are contiguous.
        return expr + sympy.symbols(f"p{len(dim_sizes) - 1}")

    def transform_origin_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        shape = self.inner.transform_origin_shape(shape)

        ispec: "specs.TensorSpec" = self.inner.spec
        flattened = ispec.layout.flatten_inner_contiguous_dimensions(
            shape, self.flattening_contiguous_abs
        )
        assert flattened
        prefix, _, inner_vol = flattened
        return prefix + (inner_vol,)

    def __str__(self):
        return f"{self.inner}.inner_flatten"

    def _flatten_shape(self) -> tuple[int, ...]:
        return self.transform_origin_shape(self.inner.dim_sizes)


@dataclasses.dataclass(frozen=True, eq=False)
class SqueezingTile(Tile):
    source: OperandIdx
    inner: TensorLike
    dropped_dims: frozenset[int]

    @property
    def spec(self) -> "specs.TensorSpec":
        from . import specs

        ispec: "specs.TensorSpec" = self.inner.spec
        new_dim_sizes = list(ispec.dim_sizes)
        for dim in sorted(self.dropped_dims, reverse=True):
            assert new_dim_sizes[dim] == 1
            del new_dim_sizes[dim]
        new_dim_sizes = tuple(new_dim_sizes)

        new_vector_shape = None
        if ispec.vector_shape:
            new_vector_shape = list(ispec.vector_shape)
            for dim in sorted(self.dropped_dims, reverse=True):
                assert (
                    new_vector_shape[dim] == 1
                ), f"Cannot drop dimension {dim} of vector shape {ispec.vector_shape}"
                del new_vector_shape[dim]
            new_vector_shape = tuple(new_vector_shape)

        if all(d == 1 for d in new_dim_sizes):
            new_layout = layouts.row_major(len(new_dim_sizes))
            new_contig = True
        else:
            new_layout, new_contig = ispec.layout.dim_drop(
                self.dropped_dims, ispec.contiguous_abs
            )

        return specs.TensorSpec(
            dim_sizes=new_dim_sizes,
            dtype=ispec.dtype,
            contiguous_abs=new_contig,
            aligned=ispec.aligned,
            bank=ispec.bank,
            layout=new_layout,
            vector_shape=new_vector_shape,
        )

    @property
    def name(self) -> str:
        return self.inner.name

    def steps_dim(self, dim: int, origin_size: int) -> int:
        exploded = self._squeezed_to_exploded_dims()
        return self.inner.steps_dim(exploded[dim], origin_size)

    def boundary_size(self, dim: int, origin_size: int) -> int:
        exploded = self._squeezed_to_exploded_dims()
        return self.inner.boundary_size(exploded[dim], origin_size)

    @property
    def frontiers(self) -> tuple[int, ...]:
        mapping = self._squeezed_to_exploded_dims()
        inner_frontier = self.inner.frontiers
        return tuple(
            inner_frontier[mapping[idx]] for idx in range(len(self.spec.dim_sizes))
        )

    def _squeezed_to_exploded_dims(self) -> Mapping[int, int]:
        to_return = {}
        skipped = 0
        for dim_idx in range(len(self.spec.dim_sizes) + len(self.dropped_dims)):
            if dim_idx in self.dropped_dims:
                skipped += 1
            else:
                to_return[dim_idx - skipped] = dim_idx
        return to_return

    def transform_origin_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        shape = self.inner.transform_origin_shape(shape)
        return tuple(d for i, d in enumerate(shape) if i not in self.dropped_dims)

    def __str__(self):
        return f"{self.inner}.squeeze"


@dataclasses.dataclass(frozen=True, eq=False)
class TransposingTile(Tile):
    source: OperandIdx
    inner: TensorLike
    swap_dims: tuple[int, int]

    def __post_init__(self):
        assert self.swap_dims[0] < self.swap_dims[1]
        assert any(d > 1 for d in self.inner.dim_sizes)

        # TODO: Remove below
        self.spec

    @property
    def spec(self) -> "specs.TensorSpec":
        from . import specs

        ispec: "specs.TensorSpec" = self.inner.spec
        new_dim_sizes = list(ispec.dim_sizes)
        i, j = self.swap_dims
        new_dim_sizes[i], new_dim_sizes[j] = new_dim_sizes[j], new_dim_sizes[i]
        new_layout, new_contig = ispec.layout.transpose(
            self.swap_dims, ispec.contiguous_abs
        )
        return specs.TensorSpec(
            dim_sizes=tuple(new_dim_sizes),
            dtype=ispec.dtype,
            contiguous_abs=new_contig,
            aligned=ispec.aligned,
            bank=ispec.bank,
            layout=new_layout,
            vector_shape=ispec.vector_shape,
        )

    @property
    def name(self) -> str:
        return self.inner.name

    def steps_dim(self, dim: int, origin_size: int) -> int:
        return self.inner.steps_dim(self._flip_dim(dim), origin_size)

    def boundary_size(self, dim: int, origin_size: int) -> int:
        return self.inner.boundary_size(self._flip_dim(dim), origin_size)

    @property
    def frontiers(self) -> tuple[int, ...]:
        f = list(self.inner.frontiers)
        i, j = self.swap_dims
        f[i], f[j] = f[j], f[i]
        return tuple(f)

    def transform_origin_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        shape = self.inner.transform_origin_shape(shape)
        shape = list(shape)
        i, j = self.swap_dims
        shape[i], shape[j] = shape[j], shape[i]
        return tuple(shape)

    def __str__(self):
        return f"{self.inner}.T"

    def _flip_dim(self, dim: int) -> int:
        d = dim
        if d == self.swap_dims[0]:
            d = self.swap_dims[1]
        elif d == self.swap_dims[1]:
            d = self.swap_dims[0]
        return d


@dataclasses.dataclass(frozen=True, eq=False)
class CommonTileBase(Tile):
    source: OperandIdx
    spec: "specs.TensorSpec"
    name: Optional[str]

    def __str__(self):
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        layout_epi = ""
        bank_epi = ""
        if not self.spec.layout.is_row_major:
            layout_epi = f", {self.spec.layout}"
        if self.bank != system_config.current_system().default_bank:
            bank_epi = f", {self.bank}"
        return f"{type(self).__name__}({dims_part}{layout_epi}{bank_epi})"


@dataclasses.dataclass(frozen=True, eq=False)
class SimpleTile(CommonTileBase):
    def steps_dim(self, dim: int, origin_size: int) -> int:
        return math.ceil(origin_size / self.dim_sizes[dim])

    def boundary_size(self, dim: int, origin_size: int) -> int:
        return origin_size % self.dim_sizes[dim]

    @property
    def frontiers(self) -> tuple[int, ...]:
        return tuple(0 for _ in self.dim_sizes)


@dataclasses.dataclass(frozen=True, eq=False)
class ConvolutionImageTile(CommonTileBase):

    filter_shape: tuple[int, ...]

    def __post_init__(self):
        assert len(self.dim_sizes) >= self.minimum_image_rank()
        assert len(self.filter_shape) + 1 == len(self.dim_sizes), (
            f"Incompatible ranks; filters was {self.filter_shape} and image "
            f"was {self.dim_sizes}"
        )

    @staticmethod
    def minimum_image_rank() -> int:
        return 3

    def steps_dim(self, dim: int, origin_size: int) -> int:
        # Batch should be a normal tiling.
        if dim == 0:
            return math.ceil(origin_size / self.dim_sizes[dim])

        inner = self.dim_sizes[dim]
        f = self.filter_shape[dim - 1]
        return int(math.ceil(_s(origin_size, f) / _s(inner, f)))

    def boundary_size(self, dim: int, origin_size: int) -> int:
        # Non-spatial dimensions (batch) should be simple tilings.
        if dim == 0:
            return origin_size % self.dim_sizes[dim]

        filt = self.filter_shape[dim - 1]
        total_filter_applications = 1 + origin_size - filt
        tile_filter_applications = 1 + self.dim_sizes[dim] - filt
        boundary_applications = total_filter_applications % tile_filter_applications
        if boundary_applications == 0:
            return 0
        return boundary_applications + filt - 1

    @property
    def frontiers(self) -> tuple[int, ...]:
        # Step size is one, so the frontier in each dimension equals the output
        # size in that dimension.
        return (0, 0) + tuple(
            _s(w, f) for w, f in zip(self.dim_sizes[1:], self.filter_shape)
        )


def _s(img_size: int, filter_size: int) -> int:
    """Calculates the number of output pixels in a single dimension."""
    return 1 + img_size - filter_size
