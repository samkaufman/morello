import dataclasses
import functools
import math
from operator import mul
from typing import Optional, Tuple, Union, cast

from . import dtypes, specs, system_config


def layout_ordered_dims(operand: Union["Tensor", "Tile"]) -> Tuple[int, ...]:
    """Returns tuple of operand's height and width; or vice versa if column-major."""
    if len(operand.dim_sizes) == 1:
        return (operand.dim_sizes[0],)
    if operand.root.layout == specs.Layout.ROW_MAJOR:
        lead = [operand.dim_sizes[0], operand.dim_sizes[1]]
    elif operand.root.layout == specs.Layout.COL_MAJOR:
        lead = [operand.dim_sizes[1], operand.dim_sizes[0]]
    else:
        raise NotImplementedError(f"Unknown layout {operand.root.layout}")
    lead.extend(operand.dim_sizes[2:])
    return tuple(lead)


class _TensorLike:
    @property
    def volume(self) -> int:
        return functools.reduce(mul, self.dim_sizes, 1)

    @property
    def layout(self) -> specs.Layout:
        return self.spec.layout

    @property
    def dtype(self) -> dtypes.Dtype:
        # Just a sugar getter.
        return self.spec.dtype

    @property
    def height(self):
        # TODO: Should really remove the `height` property, but useful for backwards
        #  compatibility for the moment.
        assert len(self.dim_sizes) == 2, f"{self} is not a matrix"
        return self.dim_sizes[0]

    @property
    def width(self):
        # TODO: Should really remove the `width` property, but useful for backwards
        #  compatibility for the moment.
        assert len(self.dim_sizes) == 2, f"{self} is not a matrix"
        return self.dim_sizes[1]

    def simple_tile(self, tile_shape: tuple[int, ...]) -> Union["Tensor", "Tile"]:
        return self._tile(SimpleTile, tile_shape)

    def conv_image_tile(
        self, tile_shape: tuple[int, ...], filter_shape: tuple[int, int]
    ) -> Union["Tensor", "Tile"]:
        return self._tile(ConvolutionImageTile, tile_shape, filter_shape=filter_shape)

    def _tile(
        self, tile_cls, tile_shape: tuple[int, ...], **kw
    ) -> Union["Tensor", "Tile"]:
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
            # This cast is safe because there are only two _TensorLike subclasses
            return cast(Union[Tensor, Tile], self)
        return tile_cls(dim_sizes=tile_shape, name=None, origin=self, **kw)


@dataclasses.dataclass(frozen=True)
class Tensor(_TensorLike):
    """An n-dimensional array."""

    spec: specs.TensorSpec
    name: Optional[str]
    origin: Optional[Union["Tensor", "Tile"]] = None

    def __post_init__(self):
        assert isinstance(self.spec, specs.TensorSpec)
        for dim_size in self.dim_sizes:
            assert dim_size > 0

    @property
    def dim_sizes(self) -> Tuple[int, ...]:
        return self.spec.dim_sizes

    @property
    def bank(self) -> str:
        return self.spec.bank

    @property
    def dtype(self) -> dtypes.Dtype:
        return self.spec.dtype

    @property
    def bytes_used(self) -> int:
        return self.volume * self.dtype.size

    def __str__(self):
        layout_epi = ""
        bank_epi = ""
        if self.layout != specs.Layout.ROW_MAJOR:
            layout_epi = f", {self.layout}"
        if self.bank != system_config.current_system().default_bank:
            bank_epi = f", {self.bank}"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        return f"Tensor({dims_part}{layout_epi}{bank_epi})"

    @property
    def root(self):
        return self

    @property
    def contiguous(self):
        """Whether or not elements are contiguous in the underlying memory.

        This is always True for matrices.
        """
        return True

    def replace_tensors(
        self,
        replacements: "Mapping[Union[Tensor, Tile], Union[Tensor, Tile]]",
    ) -> "Tensor":
        new_origin = None
        if self.origin is not None:
            try:
                new_origin = replacements[self.origin]
            except KeyError:
                new_origin = self.origin.replace_tensors(replacements)
        return Tensor(spec=self.spec, name=self.name, origin=new_origin)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

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


@dataclasses.dataclass(frozen=True)
class Tile(_TensorLike):
    dim_sizes: Tuple[int, ...]
    name: Optional[str]
    # TODO: Rename origin, in Tensor and Tile, `source`, which is more descriptive
    origin: Union[Tensor, "Tile"]

    def __post_init__(self):
        assert isinstance(
            self.root, Tensor
        ), f"root was not Tensor; got: {type(self.root)}"
        assert self.origin is not None
        for dim_size in self.dim_sizes:
            if dim_size <= 0:
                raise ValueError("Invalid dimensions: " + str(self.dim_sizes))

    @property
    def root(self) -> Tensor:
        return self.origin.root

    @property
    def bank(self) -> str:
        return self.origin.bank

    @property
    def steps(self) -> int:
        result = 1
        for dim in range(len(self.dim_sizes)):
            result *= self.steps_dim(dim)
        return result

    def steps_dim(self, dim: int, origin_size: Optional[int] = None) -> int:
        raise NotImplementedError()

    @functools.cached_property
    def spec(self) -> specs.TensorSpec:
        layout = specs.Layout.ROW_MAJOR
        if any(d != 1 for d in self.dim_sizes):
            layout = self.root.layout
        return specs.TensorSpec(
            dim_sizes=self.dim_sizes,
            dtype=self.origin.dtype,
            bank=self.origin.bank,
            layout=layout,
        )

    def __str__(self):
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        layout_epi = ""
        bank_epi = ""
        if self.root.layout != specs.Layout.ROW_MAJOR:
            layout_epi = f", {self.root.layout}"
        if self.bank != system_config.current_system().default_bank:
            bank_epi = f", {self.bank}"
        return f"{type(self).__name__}({dims_part}{layout_epi}{bank_epi})"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

    @property
    def contiguous(self):
        """Whether or not elements are contiguous in the underlying memory."""
        # TODO: Throw together some unit tests for this method
        for tile_dim, root_dim in zip(
            layout_ordered_dims(self), layout_ordered_dims(self.root)
        ):
            # The following includes the case where an underlying dimension is 1.
            if tile_dim != root_dim:
                return False
        return True

    @property
    def frontiers(self) -> tuple[int, ...]:
        """The sizes of non-overlapping regions between consecutive tiles in each dimension."""
        raise NotImplementedError()

    def replace_tensors(
        self,
        replacements: "Mapping[Union[Tensor, Tile], Union[Tensor, Tile]]",
    ) -> "Tile":
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


@dataclasses.dataclass(frozen=True)
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

    def replace_tensors(
        self,
        replacements: "Mapping[Union[Tensor, Tile], Union[Tensor, Tile]]",
    ) -> "SimpleTile":
        new_origin = None
        if self.origin is not None:
            try:
                new_origin = replacements[self.origin]
            except KeyError:
                new_origin = self.origin.replace_tensors(replacements)

        return SimpleTile(
            dim_sizes=self.dim_sizes,
            name=self.name,
            origin=new_origin,
        )


@dataclasses.dataclass(frozen=True)
class ConvolutionImageTile(Tile):
    filter_shape: tuple[int, int]

    def steps_dim(self, dim: int, origin_size: Optional[int] = None) -> int:
        if origin_size is None:
            origin_size = self.origin.dim_sizes[dim]
        inner = self.dim_sizes[dim]
        f = self.filter_shape[dim]
        return int(math.ceil(self._s(origin_size, f) / self._s(inner, f)))

    def boundary_size(self, dim: int, origin_size: Optional[int] = None) -> int:
        if origin_size is None:
            origin_size = self.origin.dim_sizes[dim]
        filt = self.filter_shape[dim]
        out_total = 1 + origin_size - filt
        out_for_tile = 1 + self.dim_sizes[dim] - filt
        out_boundary = out_total % out_for_tile
        return out_boundary + filt - 1

    @property
    def steps(self) -> int:
        result = super().steps
        assert result == self._dumb_steps
        return result

    # TODO: Remove this, and filter_shape, if the other calculation works
    @property
    def _dumb_steps(self) -> int:
        origin_out_shape = (
            self._s(self.origin.dim_sizes[0], self.filter_shape[0]),
            self._s(self.origin.dim_sizes[1], self.filter_shape[1]),
        )
        inner_out_shape = (
            self._s(self.dim_sizes[0], self.filter_shape[0]),
            self._s(self.dim_sizes[1], self.filter_shape[1]),
        )
        return math.ceil(origin_out_shape[0] / inner_out_shape[0]) * math.ceil(
            origin_out_shape[1] / inner_out_shape[1]
        )

    @staticmethod
    def _s(img_size: int, filter_size: int) -> int:
        return 1 + img_size - filter_size

    @property
    def frontiers(self) -> tuple[int, ...]:
        # Step size is one, so the frontier in each dimension equals the output
        # size in that dimension.
        return tuple(self._s(w, f) for w, f in zip(self.dim_sizes, self.filter_shape))

    def replace_tensors(
        self,
        replacements: "Mapping[Union[Tensor, Tile], Union[Tensor, Tile]]",
    ) -> "ConvolutionImageTile":
        new_origin = None
        if self.origin is not None:
            try:
                new_origin = replacements[self.origin]
            except KeyError:
                new_origin = self.origin.replace_tensors(replacements)

        return ConvolutionImageTile(
            filter_shape=self.filter_shape,
            dim_sizes=self.dim_sizes,
            name=self.name,
            origin=new_origin,
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["filter_shape"] = self.filter_shape
        return state

    def __setstate__(self, state_dict):
        super().__setstate__(state_dict)
        object.__setattr__(self, "filter_shape", state_dict["filter_shape"])
