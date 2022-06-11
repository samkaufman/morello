import abc
import dataclasses
import math
import typing
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from . import dtypes, layouts, system_config

if TYPE_CHECKING:
    from . import specs

OperandIdx = typing.NewType("OperandIdx", int)


class DisallowedTileShapeError(ValueError):
    pass


class TensorLike(abc.ABC):
    spec: "specs.TensorSpec"
    dim_sizes: tuple[int, ...]

    def _common_init_checks(self):
        for dim_size in self.dim_sizes:
            if dim_size <= 0:
                raise ValueError("Invalid dimensions: " + str(self.dim_sizes))

    @property
    @typing.final
    def layout(self) -> layouts.Layout:
        return self.spec.layout

    @property
    @typing.final
    def dtype(self) -> dtypes.Dtype:
        # Just a sugar getter.
        return self.spec.dtype

    @property
    @typing.final
    def bank(self) -> str:
        # Just a sugar getter for the underlying Spec.
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

    def __post_init__(self):
        self._common_init_checks()

    def __str__(self):
        layout_epi = ""
        if not isinstance(self.layout, layouts.RowMajor):
            layout_epi = f", {self.layout}"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        return f"{type(self).__name__}({dims_part}{layout_epi}, {self.bank})"

    def __getstate__(self):
        return {
            "spec": self.spec,
            "name": self.name,
        }

    def __setstate__(self, state_dict):
        if "__spec" in state_dict:
            object.__setattr__(self, "spec", state_dict["__spec"])
        else:
            object.__setattr__(self, "spec", state_dict["spec"])
        object.__setattr__(self, "name", state_dict["name"])


@dataclasses.dataclass(frozen=True, eq=False)
class Tile(TensorLike):
    source: OperandIdx
    spec: "specs.TensorSpec"
    name: Optional[str]

    def __post_init__(self):
        self._common_init_checks()

    @property
    def root(self) -> Tensor:
        raise NotImplementedError("root will be removed")

    @property
    def dim_sizes(self) -> tuple[int, ...]:
        return self.spec.dim_sizes

    @property
    def bank(self) -> str:
        return self.spec.bank

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

    def __str__(self):
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        layout_epi = ""
        bank_epi = ""
        if not isinstance(self.spec.layout, layouts.RowMajor):
            layout_epi = f", {self.spec.layout}"
        if self.bank != system_config.current_system().default_bank:
            bank_epi = f", {self.bank}"
        return f"{type(self).__name__}({dims_part}{layout_epi}{bank_epi})"

    @property
    def frontiers(self) -> tuple[int, ...]:
        """The sizes of non-overlapping regions between consecutive tiles in each dimension."""
        raise NotImplementedError()

    def __getstate__(self):
        return {
            "source": self.source,
            "spec": self.spec,
            "name": self.name,
        }

    def __setstate__(self, state_dict):
        object.__setattr__(self, "source", state_dict["source"])
        object.__setattr__(self, "spec", state_dict["spec"])
        object.__setattr__(self, "name", state_dict["name"])


@dataclasses.dataclass(frozen=True, eq=False)
class SimpleTile(Tile):
    def steps_dim(self, dim: int, origin_size: int) -> int:
        return math.ceil(origin_size / self.dim_sizes[dim])

    def boundary_size(self, dim: int, origin_size: int) -> int:
        return origin_size % self.dim_sizes[dim]

    @property
    def frontiers(self) -> tuple[int, ...]:
        return tuple(0 for _ in self.dim_sizes)


@dataclasses.dataclass(frozen=True, eq=False)
class ConvolutionImageTile(Tile):

    filter_shape: tuple[int, ...]

    def __post_init__(self):
        super().__post_init__()
        assert len(self.dim_sizes) >= 3
        assert len(self.filter_shape) + 1 == len(
            self.dim_sizes
        ), f"Incompatible ranks; filters was {self.filter_shape} and image was {self.dim_sizes}"

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

    def __getstate__(self):
        state = super().__getstate__()
        state["filter_shape"] = self.filter_shape
        return state

    def __setstate__(self, state_dict):
        super().__setstate__(state_dict)
        object.__setattr__(self, "filter_shape", state_dict["filter_shape"])


def _s(img_size: int, filter_size: int) -> int:
    """Calculates the number of output pixels in a single dimension."""
    return 1 + img_size - filter_size