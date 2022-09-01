import functools
import operator
import typing
from typing import Any, Optional, Sequence

import cython

from .. import layouts, system_config, utils
from ..dtypes import Dtype, Uint8
from ..system_config import current_system

if typing.TYPE_CHECKING:
    from .. import tensor


@cython.dataclasses.dataclass(unsafe_hash=True)
@cython.cclass
class TensorSpec:
    """A TensorSpec describes an operand to a Spec.

    `contiguous' means that there is a way to iterate over all elements of
    the tensor without skipping addresses.

    `aligned` means that the zero coordinate in the tensor is backed by an address
    multiple of `current_system().line_size`.

    This class is distinct from impl.Tensor and impl.Tile, which describe
    operands in Impl.
    """

    dim_sizes: tuple[int, ...]
    dtype: Dtype
    contiguous_abs: Any
    aligned: bool
    bank: str
    layout: layouts.Layout

    def __init__(
        self,
        dim_sizes: tuple[int, ...],
        dtype: Dtype,
        contiguous_abs: Any,
        aligned: bool = True,
        bank: Optional[str] = None,
        layout: Optional[layouts.Layout] = None,
    ):
        self.dim_sizes = dim_sizes
        self.dtype = dtype
        if bank is None:
            self.bank = current_system().default_bank
        else:
            self.bank = bank
        if layout is None:
            self.layout = layouts.row_major(len(self.dim_sizes))
        else:
            self.layout = layout
        self.contiguous_abs = contiguous_abs
        self.aligned = aligned

        if not len(self.dim_sizes):
            raise ValueError("dim_sizes cannot be empty")
        if all(d == 1 for d in self.dim_sizes):
            if not self.layout.is_row_major:
                raise ValueError("If all dimensions are 1, layout must be row-major")

        if not self.layout.applies_to_shape(dim_sizes, dtype):
            raise ValueError(
                f"Layout {self.layout} does not apply to shape {dim_sizes} with"
                f" dtype {dtype}"
            )

    def shrink(self, new_dim_sizes: Sequence[int], *, aligned: bool) -> "TensorSpec":
        """Returns a clone with new dimensions and updated contiguous- and aligned-ness.

        If new_dim_sizes is all ones, the layout may be changed to row-major.
        """
        contiguous_abs = self.layout.check_tile_contiguity(
            new_dim_sizes, self.dim_sizes, self.contiguous_abs
        )

        new_layout = self.layout
        if all(d == 1 for d in new_dim_sizes):
            new_layout = layouts.row_major(len(new_dim_sizes))
        return TensorSpec(
            tuple(new_dim_sizes),
            dtype=self.dtype,
            bank=self.bank,
            layout=new_layout,
            contiguous_abs=contiguous_abs,
            aligned=aligned,
        )

    @property
    def contiguous(self) -> bool:
        return self.contiguous_abs == len(self.dim_sizes)

    @property
    @typing.final
    def volume(self) -> int:
        return functools.reduce(operator.mul, self.dim_sizes, 1)

    @property
    @typing.final
    def bytes_used(self) -> int:
        return self.volume * self.dtype.size

    def can_move_to(
        self, bank: Optional[str], layout: Optional[layouts.Layout]
    ) -> bool:
        if bank is None:
            bank = self.bank
        if isinstance(layout, layouts.HexagonTranspacked):
            if self.dtype != Uint8:
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

    def is_valid_tile_shape(self, shape: tuple[int, ...]) -> bool:
        """Returns True if can be tiled to this shape."""
        if len(shape) != len(self.dim_sizes):
            return False
        if not all(i <= o for (i, o) in zip(shape, self.dim_sizes)):
            return False
        if not self.layout.applies_to_shape(shape, self.dtype):
            return False
        return True

    def simple_tile(
        self, operand_idx: "tensor.OperandIdx", tile_shape: tuple[int, ...]
    ) -> "tensor.TensorLike":
        from .. import tensor

        return self._tile(tensor.SimpleTile, operand_idx, tile_shape)

    def conv_image_tile(
        self,
        operand_idx: "tensor.OperandIdx",
        tile_shape: tuple[int, ...],
        filter_shape: tuple[int, ...],
    ) -> "tensor.TensorLike":
        from .. import tensor

        return self._tile(
            tensor.ConvolutionImageTile,
            operand_idx,
            tile_shape,
            filter_shape=filter_shape,
        )

    def _tile(
        self, tile_cls, operand_idx: "tensor.OperandIdx", new_dims: Sequence[int], **kw
    ) -> "tensor.TensorLike":
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

        aligned = utils.aligned_approx(tile_cls, new_dims, self)
        tile_spec = self.shrink(new_dims, aligned=aligned)
        return tile_cls(source=operand_idx, spec=tile_spec, name=None, **kw)

    def __str__(self):
        layout_epi = ""
        bank_epi = ""
        c_epi = ""
        a_epi = ""
        if not self.layout.is_row_major:
            layout_epi = f", {self.layout}"
        if self.bank != current_system().default_bank:
            bank_epi = f", {self.bank}"
        if self.contiguous_abs != self.layout.contiguous_top():
            c_epi = f", c{self.contiguous_abs}"
        if not self.aligned:
            a_epi = ", ua"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        return f"({dims_part}, {self.dtype}{bank_epi}{layout_epi}{c_epi}{a_epi})"


@cython.dataclasses.dataclass(unsafe_hash=True)
@cython.cclass
class HvxVmemTensorSpec(TensorSpec):
    vector_shape: tuple[int, ...]

    def __init__(self, *args, vector_shape: tuple[int, ...], **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_shape = vector_shape
        if any(s < vs for s, vs in zip(self.dim_sizes, self.vector_shape)):
            raise ValueError(
                f"Shape {self.dim_sizes} is smaller in some dimensions than vector shape {vector_shape}"
            )

    def is_valid_tile_shape(self, shape: tuple[int, ...]) -> bool:
        if not super().is_valid_tile_shape(shape):
            return False
        if any(i > v for (i, v) in zip(shape, self.vector_shape)):
            return False
        if functools.reduce(operator.mul, shape, 1) % 128 != 0:
            return False
        return True

    def shrink(
        self, new_dim_sizes: Sequence[int], contiguous: bool
    ) -> "HvxVmemTensorSpec":
        new_layout = self.layout
        if all(d == 1 for d in new_dim_sizes):
            new_layout = layouts.row_major(len(new_dim_sizes))
        return HvxVmemTensorSpec(
            tuple(new_dim_sizes),
            dtype=self.dtype,
            contiguous=contiguous,
            bank=self.bank,
            layout=new_layout,
        )

    def __str__(self):
        base_str = super().__str__()[:-1]
        vs_dims_part = "×".join(str(s) for s in self.vector_shape)
        base_str = f"{base_str}, {vs_dims_part})"
        return base_str
