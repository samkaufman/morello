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


class LayoutDoesntApplyError(ValueError):
    pass


class OversizedVectorError(ValueError):
    pass


@cython.dataclasses.dataclass(unsafe_hash=True, slots=True)
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
    vector_shape: Optional[tuple[int, ...]]

    def __init__(
        self,
        dim_sizes: tuple[int, ...],
        dtype: Dtype,
        contiguous_abs: Any,
        aligned: bool = True,
        bank: Optional[str] = None,
        layout: Optional[layouts.Layout] = None,
        vector_shape: Optional[tuple[int, ...]] = None,
    ):
        system = current_system()

        self.dim_sizes = dim_sizes
        if any(d < 1 for d in dim_sizes):
            raise ValueError(f"Invalid shape: {dim_sizes}")

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
        self.vector_shape = vector_shape

        if not len(self.dim_sizes):
            raise ValueError("dim_sizes cannot be empty")
        if all(d == 1 for d in self.dim_sizes):
            if not self.layout.is_row_major:
                raise LayoutDoesntApplyError(
                    "If all dimensions are 1, layout must be row-major"
                )

        if (self.vector_shape is not None) != system.banks[self.bank].vector_rf:
            raise ValueError(
                f"vector_shape must be specified if and only if the bank ({self.bank})"
                " is a vector register file"
            )

        if vector_shape is not None:
            if len(vector_shape) != len(dim_sizes):
                raise ValueError("vector_shape must have same rank as dim_sizes")
            # if any(i > o for i, o in zip(vector_shape, dim_sizes)):
            #     raise OversizedVectorError(
            #         f"vector_shape must be smaller than dim_sizes, but "
            #         f"were {vector_shape} and {dim_sizes}"
            #     )
        if not self.layout.applies_to_shape(dim_sizes):
            raise LayoutDoesntApplyError(
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
            vector_shape=self.vector_shape,
        )

    @property
    def contiguous(self) -> bool:
        return self.contiguous_abs == self.layout.contiguous_full()

    @property
    @typing.final
    def volume(self) -> int:
        return functools.reduce(operator.mul, self.dim_sizes, 1)

    @property
    @typing.final
    def bytes_used(self) -> int:
        return self.volume * self.dtype.size

    def can_move_to(self, bank: str, layout: layouts.Layout) -> bool:
        system = current_system()
        if layout != self.layout and bank not in system.addressed_banks:
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
        all_ones = all(d == 1 for d in shape)
        if not all_ones and not self.layout.applies_to_shape(shape):
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

    def flatten_inner_contig(self) -> Optional["TensorSpec"]:
        flattened = self.layout.flatten_inner_contiguous_dimensions(
            self.dim_sizes, self.contiguous_abs
        )
        if flattened is None:
            return None
        prefix, _, inner_contig_vol = flattened
        new_layout = layouts.row_major(len(prefix) + 1)
        return TensorSpec(
            dim_sizes=prefix + (inner_contig_vol,),
            dtype=self.dtype,
            contiguous_abs=new_layout.contiguous_one(),
            aligned=self.aligned,
            bank=self.bank,
            layout=new_layout,
        )

    def __str__(self):
        layout_epi = ""
        bank_epi = ""
        c_epi = ""
        a_epi = ""
        v_epi = ""
        if not self.layout.is_row_major:
            layout_epi = f", {self.layout}"
        if self.bank != current_system().default_bank:
            bank_epi = f", {self.bank}"
        if self.contiguous_abs != self.layout.contiguous_full():
            c_epi = f", c{self.contiguous_abs}"
        if not self.aligned:
            a_epi = ", ua"
        if self.vector_shape is not None:
            v_epi = f", {'×'.join(str(s) for s in self.vector_shape)}"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        return f"({dims_part}, {self.dtype}{bank_epi}{layout_epi}{c_epi}{a_epi}{v_epi})"
