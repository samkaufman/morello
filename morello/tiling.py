import dataclasses
from typing import Sequence, Union

from . import specs
from .tensor import ConvolutionImageTile, SimpleTile, TensorLike, Tensor, Tile


class UnimplementedCompositionError(NotImplementedError):
    pass


@dataclasses.dataclass(frozen=True)
class PartialTile:
    # Shouldn't initialize this base class. Instantiate a subclass instead.
    dim_sizes: tuple[int, ...]

    def tile(self, source: TensorLike) -> TensorLike:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class PartialSimpleTile(PartialTile):
    def tile(self, source: TensorLike) -> TensorLike:
        return source.simple_tile(self.dim_sizes)


@dataclasses.dataclass(frozen=True)
class PartialConvolutionImageTile(PartialTile):
    filter_shape: tuple[int, int]

    def tile(self, source: TensorLike) -> TensorLike:
        return source.conv_image_tile(self.dim_sizes, self.filter_shape)


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

    Notice that, if tiling, the input_shapes are the shapes of the spec's inputs in
    the outer tiling, while the spec_output refers to the Tile produced by tiling.

    Compose is not directly represented because tiling a Compose depends on its
    subspecs, which are members of the Compose object. As a result, the tile_out
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
        _, (fh, fw, _) = input_shapes
        oh, ow, oc = spec_output.dim_sizes
        tile_shape = (oh + fh - 1, ow + fw - 1)

        soh, sow = 1, 1
        if isinstance(spec_output, PartialConvolutionImageTile):
            soh, sow = spec_output.filter_shape

        window_shape = soh + fh - 1, sow + fw - 1
        return (
            PartialConvolutionImageTile(tile_shape, window_shape),
            PartialSimpleTile((fh, fw, oc)),
        )
    elif spec_type == specs.Matmul and isinstance(spec_output, PartialSimpleTile):
        (_, k), _ = input_shapes
        m, n = spec_output.dim_sizes
        return PartialSimpleTile((m, k)), PartialSimpleTile((k, n))
    else:
        raise NotImplementedError(
            f"Unimplemented for {spec_type.__name__} and {type(spec_output).__name__}"
        )
