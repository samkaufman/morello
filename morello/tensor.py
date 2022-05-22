import abc
import dataclasses
import functools
import math
import operator
import typing
from typing import Mapping, Optional, Sequence

from . import dtypes, layouts, specs, system_config

OperandIdx = typing.NewType("OperandIdx", int)


class DisallowedTileShapeError(ValueError):
    pass


class TensorLike(abc.ABC):
    spec: specs.TensorSpec
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

    @property
    @abc.abstractmethod
    def contiguous(self) -> bool:
        """Whether or not elements are contiguous in the underlying memory."""
        # TODO: Expand the above doc to talk about phys. vs. logical contiguousness.
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def address_root(self) -> "TensorBase":
        """Returns the containing tensor with a contiguous address space.

        Essentially, this returns the receiver if in non-cache memory, or its own
        address_space otherwise.
        """
        raise NotImplementedError()

    def replace_tensors(
        self, replacements: Mapping["TensorLike", "TensorLike"]
    ) -> "Tile":
        raise NotImplementedError("replace_tensors will be removed")

    def simple_tile(
        self, operand_idx: OperandIdx, tile_shape: tuple[int, ...]
    ) -> "TensorLike":
        return self._tile(SimpleTile, operand_idx, tile_shape)

    def conv_image_tile(
        self,
        operand_idx: OperandIdx,
        tile_shape: tuple[int, ...],
        filter_shape: tuple[int, ...],
    ) -> "TensorLike":
        return self._tile(
            ConvolutionImageTile, operand_idx, tile_shape, filter_shape=filter_shape
        )

    def _tile(
        self, tile_cls, operand_idx: OperandIdx, new_dims: Sequence[int], **kw
    ) -> "TensorLike":
        if new_dims == self.dim_sizes:
            return self

        if len(new_dims) != len(self.dim_sizes):
            raise ValueError(
                f"Cannot produce rank-{len(new_dims)} tile of shape "
                f"{new_dims} for rank-{len(self.dim_sizes)} tensor of "
                f"shape {self.dim_sizes}"
            )
        if any(td > rd for td, rd in zip(new_dims, self.dim_sizes)):
            raise ValueError(
                f"Tile {new_dims} would be larger than tensor {self.dim_sizes}"
            )

        tile_spec = self.spec.shrink(new_dims)
        return tile_cls(source=operand_idx, spec=tile_spec, name=None, **kw)

    @typing.final
    def __eq__(self, other):
        return self is other

    @typing.final
    def __hash__(self):
        return hash(id(self))


class TensorBase(TensorLike):
    spec: specs.TensorSpec

    @property
    def dim_sizes(self) -> tuple[int, ...]:
        return self.spec.dim_sizes

    @property
    def root(self) -> "TensorBase":
        return self

    @property
    def contiguous(self) -> bool:
        return True

    @property
    def address_root(self) -> "TensorBase":
        # TODO: Don't hardcode cache names here.
        if self.bank in ("L1", "L2") and self.origin:
            return self.origin.address_root
        return self

    @property
    def bank(self) -> str:
        return self.spec.bank


@dataclasses.dataclass(frozen=True, eq=False)
class Tensor(TensorBase):
    """An n-dimensional array."""

    spec: specs.TensorSpec
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
    spec: specs.TensorSpec
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
    @property
    def steps(self) -> int:
        return functools.reduce(
            operator.mul, map(self.steps_dim, range(len(self.dim_sizes))), 1
        )

    def steps_dim(self, dim: int, origin_size: Optional[int] = None) -> int:
        raise NotImplementedError()

    def boundary_size(self, dim: int, origin_size: Optional[int] = None) -> int:
        raise NotImplementedError()

    def __str__(self):
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        layout_epi = ""
        bank_epi = ""
        if not isinstance(self.spec.layout, layouts.RowMajor):
            layout_epi = f", {self.root.layout}"
        if self.bank != system_config.current_system().default_bank:
            bank_epi = f", {self.bank}"
        return f"{type(self).__name__}({dims_part}{layout_epi}{bank_epi})"

    @property
    def contiguous(self) -> bool:
        """Whether or not elements are contiguous in the underlying memory."""
        from . import utils

        return utils.contiguous(self, self.address_root)

    @property
    def address_root(self) -> "TensorBase":
        return self.origin.address_root

    @property
    def frontiers(self) -> tuple[int, ...]:
        """The sizes of non-overlapping regions between consecutive tiles in each dimension."""
        raise NotImplementedError()

    def __getstate__(self):
        return {
            "dim_sizes": self.dim_sizes,
            "name": self.name,
            "origin": self.origin,
        }

    def __setstate__(self, state_dict):
        if "__dim_sizes" in state_dict:
            object.__setattr__(self, "dim_sizes", state_dict["__dim_sizes"])
        else:
            object.__setattr__(self, "dim_sizes", state_dict["dim_sizes"])

        object.__setattr__(self, "name", state_dict["name"])
        object.__setattr__(self, "origin", state_dict["origin"])


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

    def steps_dim(self, dim: int, origin_size: Optional[int] = None) -> int:
        if origin_size is None:
            origin_size = self.origin.dim_sizes[dim]

        # Batch should be a normal tiling.
        if dim == 0:
            return math.ceil(origin_size / self.dim_sizes[dim])

        inner = self.dim_sizes[dim]
        f = self.filter_shape[dim - 1]
        return int(math.ceil(_s(origin_size, f) / _s(inner, f)))

    def boundary_size(self, dim: int, origin_size: Optional[int] = None) -> int:
        if origin_size is None:
            origin_size = self.origin.dim_sizes[dim]

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
