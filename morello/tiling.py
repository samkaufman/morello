import dataclasses
from typing import Sequence

from . import specs
from .tensor import ConvolutionImageTile, OperandIdx, SimpleTile, TensorLike, Tile

# TODO: Just make PartialTiles a superclass of Tile or something. That lets us
#   drop tile_to_partial.


@dataclasses.dataclass(frozen=True)
class PartialTile:
    # Shouldn't initialize this base class. Instantiate a subclass instead.
    dim_sizes: tuple[int, ...]

    def tile(self, source_idx: OperandIdx, source: specs.TensorSpec) -> TensorLike:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class PartialSimpleTile(PartialTile):
    def tile(self, source_idx: OperandIdx, source: specs.TensorSpec) -> TensorLike:
        return source.simple_tile(source_idx, self.dim_sizes)


@dataclasses.dataclass(frozen=True)
class PartialConvolutionImageTile(PartialTile):
    # TODO: Rename to something more general. This is just a sliding window.
    filter_shape: tuple[int, ...]

    def __post_init__(self):
        assert len(self.filter_shape) + 1 == len(
            self.dim_sizes
        ), f"Incompatible ranks; filters was {self.filter_shape} and image was {self.dim_sizes}"

    def tile(self, source_idx: OperandIdx, source: specs.TensorSpec) -> TensorLike:
        return source.conv_image_tile(source_idx, self.dim_sizes, self.filter_shape)


def tile_to_partial(tile: Tile) -> PartialTile:
    if not isinstance(tile, Tile):
        raise TypeError("Not a tile: " + str(tile))

    shape = tile.dim_sizes
    if isinstance(tile, SimpleTile):
        return PartialSimpleTile(shape)
    elif isinstance(tile, ConvolutionImageTile):
        filter_shape = tile.filter_shape
        return PartialConvolutionImageTile(shape, filter_shape=filter_shape)
    else:
        raise NotImplementedError(f"Unimplemented for {type(tile).__name__}")


def tile_out(
    spec_type, input_shapes: Sequence[tuple[int, ...]], spec_output: PartialTile
) -> tuple[PartialTile, ...]:
    """Maps a Spec's type, input shapes, and output tile to PartialTiles for its inputs.

    Note that input_shapes refers to original, untiled input shapes, while the
    spec_output describes the final, already-tiled output.

    Compose is not directly represented because tiling a Compose depends on its
    sub-Specs, which are members of the Compose object. As a result, the tile_out
    logic can't be fully defined by the *type* Compose; only by a specific Compose
    instance.
    """
    # We don't handle the case where spec_output is a Tensor on the
    # assumption that the caller will construct a Tile. (Otherwise why
    # bother calling this function?)
    if spec_type == specs.ReduceSum and isinstance(spec_output, PartialSimpleTile):
        assert len(input_shapes) == 1
        return (PartialSimpleTile(spec_output.dim_sizes + (input_shapes[0][-1],)),)
    elif spec_type == specs.ReduceSum and isinstance(
        spec_output, PartialConvolutionImageTile
    ):
        assert len(input_shapes) == 1
        return (
            PartialConvolutionImageTile(
                spec_output.dim_sizes + (input_shapes[0][-1],), spec_output.filter_shape
            ),
        )
    elif spec_type == specs.Convolution and isinstance(
        spec_output, (PartialSimpleTile, PartialConvolutionImageTile)
    ):
        new_batch_size, new_filter_cnt = spec_output.dim_sizes[:2]
        channels = input_shapes[0][1]
        orig_filter_spatials = input_shapes[1][2:]
        new_out_spatials = spec_output.dim_sizes[2:]

        assert channels == input_shapes[1][1], (
            f"Image had {input_shapes[0][1]} channels, but filters had "
            f"{input_shapes[1][1]} channels"
        )

        tile_shape = tuple(
            o + f - 1 for o, f in zip(new_out_spatials, orig_filter_spatials)
        )

        # If the output is a convolution, ensure the input filter/window size
        # is large enough to gather the inputs for the entire output window.
        new_window_spatial_dims = orig_filter_spatials
        if isinstance(spec_output, PartialConvolutionImageTile):
            new_window_spatial_dims = tuple(
                o + i - 1
                for o, i in zip(spec_output.filter_shape[1:], orig_filter_spatials)
            )

        return (
            PartialConvolutionImageTile(
                (new_batch_size, channels) + tile_shape,
                (channels,) + new_window_spatial_dims,
            ),
            PartialSimpleTile((new_filter_cnt, channels) + orig_filter_spatials),
        )
    elif spec_type == specs.Matmul and isinstance(spec_output, PartialSimpleTile):
        (_, k), _ = input_shapes
        m, n = spec_output.dim_sizes
        return PartialSimpleTile((m, k)), PartialSimpleTile((k, n))
    elif (spec_type == specs.Load or spec_type == specs.Store) and isinstance(
        spec_output, PartialSimpleTile
    ):
        return (PartialSimpleTile(spec_output.dim_sizes),)
    elif spec_type == specs.Zero:
        # No inputs for Zero.
        return tuple()
    else:
        raise NotImplementedError(
            f"Unimplemented for {spec_type.__name__} and {type(spec_output).__name__}"
        )
