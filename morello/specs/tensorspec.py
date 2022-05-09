import functools
import operator
from typing import Optional

import cython

from .. import layouts
from ..dtypes import Dtype, Uint8
from ..system_config import current_system


@cython.dataclasses.dataclass(unsafe_hash=True)
@cython.cclass
class TensorSpec:
    """A TensorSpec describes an operand to a Spec.

    This class is distinct from impl.Tensor and impl.Tile, which describe operands in
    the scheduling language.
    """

    dim_sizes: tuple[int, ...]
    dtype: Dtype
    bank: str
    layout: layouts.Layout

    def __init__(
        self,
        dim_sizes: tuple[int, ...],
        dtype: Dtype,
        bank: Optional[str] = None,
        layout: layouts.Layout = layouts.ROW_MAJOR,
    ):
        self.dim_sizes = dim_sizes
        self.dtype = dtype
        if bank is None:
            self.bank = current_system().default_bank
        else:
            self.bank = bank
        self.layout = layout

        if not len(self.dim_sizes):
            raise ValueError("dim_sizes cannot be empty")
        if all(d == 1 for d in self.dim_sizes) and self.layout != layouts.ROW_MAJOR:
            raise ValueError("If all dimensions are 1, layout must be row-major")

        if isinstance(self.layout, layouts.HexagonTranspacked):
            if self.dtype != Uint8:
                raise ValueError(
                    f"Cannot create transpacked tensor with type {self.dtype}"
                )
            if len(self.dim_sizes) != 2:
                raise ValueError(
                    f"Cannot create transpacked tensor with rank {len(self.dim_sizes)}"
                )
            if self.dim_sizes[0] % 4 != 0 or self.dim_sizes[1] % 32 != 0:
                raise ValueError(
                    f"Cannot create transpacked tensor with shape "
                    f"{self.dim_sizes}. Must be multiple of 4×32."
                )

    def shrink(self, new_dim_sizes: tuple[int, ...]) -> "TensorSpec":
        """Returns a clone with new dimensions.

        If new_dim_sizes is all ones, the layout may be changed to row-major.
        """
        new_layout = self.layout
        if all(d == 1 for d in new_dim_sizes):
            new_layout = layouts.ROW_MAJOR
        return TensorSpec(
            new_dim_sizes, dtype=self.dtype, bank=self.bank, layout=new_layout
        )

    def is_valid_tile_shape(self, shape: tuple[int, ...]) -> bool:
        """Returns True if self can be tiled in this shape."""
        if len(shape) != len(self.dim_sizes):
            return False
        if not all(i <= o for (i, o) in zip(shape, self.dim_sizes)):
            return False
        if isinstance(self.layout, layouts.HexagonTranspacked):
            if self.dtype != Uint8:
                return False
            if shape[0] % 4 != 0 or shape[1] % 32 != 0:
                return False
        return True

    def __str__(self):
        layout_epi = ""
        bank_epi = ""
        if not isinstance(self.layout, layouts.RowMajor):
            layout_epi = f", {self.layout}"
        if self.bank != current_system().default_bank:
            bank_epi = f", {self.bank}"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        return f"({dims_part}, {self.dtype}{bank_epi}{layout_epi})"


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

    def shrink(self, new_dim_sizes: tuple[int, ...]) -> "HvxVmemTensorSpec":
        new_layout = self.layout
        if all(d == 1 for d in new_dim_sizes):
            new_layout = layouts.ROW_MAJOR
        return HvxVmemTensorSpec(
            new_dim_sizes, dtype=self.dtype, bank=self.bank, layout=new_layout
        )

    def __str__(self):
        base_str = super().__str__()[:-1]
        vs_dims_part = "×".join(str(s) for s in self.vector_shape)
        base_str = f"{base_str}, {vs_dims_part})"
        return base_str
