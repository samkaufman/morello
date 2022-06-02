import abc
import dataclasses
import functools
import math
import operator
import typing
from operator import mul
from typing import Mapping, Optional, Union

from . import dtypes, layouts, specs, system_config


class DisallowedTileShapeError(ValueError):
    pass


class TensorLike(abc.ABC):
    spec: specs.TensorSpec
    dim_sizes: tuple[int, ...]
    origin: "Optional[TensorLike]"

    def _common_init_checks(self):
        for dim_size in self.dim_sizes:
            if dim_size <= 0:
                raise ValueError("Invalid dimensions: " + str(self.dim_sizes))

    @property
    @typing.final
    def volume(self) -> int:
        return functools.reduce(mul, self.dim_sizes, 1)

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
    def bytes_used(self) -> int:
        return self.volume * self.dtype.size

    @property
    @typing.final
    def contiguous(self) -> bool:
        """Whether or not elements are contiguous in the underlying memory."""
        # TODO: Expand the above doc to talk about phys. vs. logical contiguousness.
        return self.spec.contiguous

    @property
    @abc.abstractmethod
    def address_root(self) -> "TensorBase":
        """Returns the containing tensor with a contiguous address space.

        Essentially, this returns the receiver if in non-cache memory, or its own
        address_space otherwise.
        """
        raise NotImplementedError()

    def can_move_to(
        self, bank: Optional[str], layout: Optional[layouts.Layout]
    ) -> bool:
        if bank is None:
            bank = self.bank
        if isinstance(layout, layouts.HexagonTranspacked):
            if self.dtype != dtypes.Uint8:
                return False
            if len(self.dim_sizes) != 2:
                return False
            if self.dim_sizes[0] % 4 != 0 or self.dim_sizes[1] % 32 != 0:
                return False
        # TODO: Factor the following check out into a Hexagon-specific tensorlike
        if system_config.current_system().has_hvx and bank == "L2":
            if len([d for d in self.dim_sizes if d != 1]) != 2:
                return False
            if any(d >= 256 for d in self.dim_sizes):
                return False
        if bank == "VMEM":
            if (self.volume * self.dtype.size) % 128 != 0:
                return False
        return True

    def simple_tile(self, tile_shape: tuple[int, ...]) -> "TensorLike":
        return self._tile(SimpleTile, tile_shape)

    def conv_image_tile(
        self, tile_shape: tuple[int, ...], filter_shape: tuple[int, ...]
    ) -> "TensorLike":
        return self._tile(ConvolutionImageTile, tile_shape, filter_shape=filter_shape)

    def replace_tensors(
        self,
        replacements: "Mapping[TensorLike, TensorLike]",
    ) -> "Tensor":
        # Default implementation just replaces the origin.
        new_origin = None
        if self.origin is not None:
            try:
                new_origin = replacements[self.origin]
            except KeyError:
                new_origin = self.origin.replace_tensors(replacements)
        return dataclasses.replace(self, origin=new_origin)

    def _tile(self, tile_cls, tile_shape: tuple[int, ...], **kw) -> "TensorLike":
        if len(tile_shape) != len(self.dim_sizes):
            raise ValueError(
                f"Cannot produce rank-{len(tile_shape)} tile of shape "
                f"{tile_shape} for rank-{len(self.dim_sizes)} tensor of "
                f"shape {self.dim_sizes}"
            )
        if any(td > rd for td, rd in zip(tile_shape, self.dim_sizes)):
            raise ValueError(
                f"Tile {tile_shape} would be larger than tensor {self.dim_sizes}"
            )
        if tile_shape == self.dim_sizes:
            return self
        return tile_cls(dim_sizes=tile_shape, name=None, origin=self, **kw)

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
    origin: Optional[TensorLike] = None

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
            "origin": self.origin,
        }

    def __setstate__(self, state_dict):
        if "__spec" in state_dict:
            object.__setattr__(self, "spec", state_dict["__spec"])
        else:
            object.__setattr__(self, "spec", state_dict["spec"])
        object.__setattr__(self, "name", state_dict["name"])
        object.__setattr__(self, "origin", state_dict["origin"])


@dataclasses.dataclass(frozen=True, eq=False)
class Tile(TensorLike):
    dim_sizes: tuple[int, ...]
    name: Optional[str]
    # TODO: Rename origin, in Tensor and Tile, `source`, which is more descriptive
    origin: Union[Tensor, "Tile"]

    def __post_init__(self):
        self._common_init_checks()
        assert isinstance(
            self.root, TensorBase
        ), f"root was not a tensor; was: {type(self.root)}"
        assert self.origin is not None

    @property
    def root(self) -> Tensor:
        return self.origin.root

    @property
    def bank(self) -> str:
        return self.origin.bank

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

    @functools.cached_property
    def spec(self) -> specs.TensorSpec:
        from . import utils

        target = system_config.current_target()
        layout = layouts.ROW_MAJOR
        if any(d != 1 for d in self.dim_sizes):
            layout = self.root.layout
        contiguous = utils.contiguous(
            (self.dim_sizes, layout),
            (self.address_root.spec.dim_sizes, self.address_root.spec.layout))
        return target.tensor_spec(
            dim_sizes=self.dim_sizes,
            dtype=self.origin.dtype,
            contiguous=contiguous,
            bank=self.origin.bank,
            layout=layout,
        )

    def __str__(self):
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        layout_epi = ""
        bank_epi = ""
        if not isinstance(self.root.layout, layouts.RowMajor):
            layout_epi = f", {self.root.layout}"
        if self.bank != system_config.current_system().default_bank:
            bank_epi = f", {self.bank}"
        return f"{type(self).__name__}({dims_part}{layout_epi}{bank_epi})"

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
    def steps_dim(self, dim: int, origin_size: Optional[int] = None) -> int:
        if origin_size is None:
            origin_size = self.origin.dim_sizes[dim]
        return math.ceil(origin_size / self.dim_sizes[dim])

    def boundary_size(self, dim: int, origin_size: Optional[int] = None) -> int:
        if origin_size is None:
            origin_size = self.origin.dim_sizes[dim]
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