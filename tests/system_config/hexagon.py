import functools
import operator

import hypothesis
import pytest
from hypothesis import strategies as st

from morello import specs, dtypes
from morello.system_config import hexagon
from .. import strategies

strategies.register_default_strategies()


@st.composite
def _st_test_tiling_always_produces_vector_indices_in_parent_set(draw):
    tensor_shape = tuple(
        draw(st.lists(st.integers(min_value=1, max_value=256), min_size=1, max_size=5))
    )
    shapes = [tensor_shape]
    for _ in draw(st.integers(min_value=1, max_value=3)):
        shapes.append(
            tuple(
                draw(st.integers(min_value=1, max_value=outer_dim))
                for outer_dim in shapes[-1]
            )
        )
    return shapes


def test_tile_vector_indices_row_major():
    raise NotImplementedError()


def test_tile_vector_indices_col_major():
    raise NotImplementedError()


@pytest.mark.parametrize(
    "tensor_shape, vector_shape, tile_shape, dtype",
    [
        ((1, 128), (1, 128), (1, 128), dtypes.Uint8),
        ((1, 32), (1, 32), (1, 32), dtypes.Uint32),
        ((2, 128), (1, 128), (1, 128), dtypes.Uint8),
        ((3, 128), (1, 128), (1, 128), dtypes.Uint8),
        ((4, 128), (1, 128), (2, 128), dtypes.Uint8),
        ((1, 1, 128), (1, 1, 128), (1, 1, 128), dtypes.Uint8),
        ((2, 1, 128), (2, 1, 64), (2, 1, 128), dtypes.Uint8),
    ],
    ids=str,
)
def test_tiling_across_full_vectors_succeeds(
    tensor_shape, vector_shape, tile_shape, dtype
):
    def inner(layout):
        spec = specs.HvxVmemTensorSpec(
            tensor_shape,
            dtype=dtype,
            bank="VMEM",
            layout=layout,
            vector_shape=vector_shape,
        )
        tensor = hexagon.HvxVmemTensor(spec, None, origin=None)
        tensor.simple_tile(tile_shape)

    for layout in specs.Layout:
        inner(layout)


@pytest.mark.parametrize(
    "tensor_shape, vector_shape, tile_shape, dtype",
    [
        ((2, 128), (1, 128), (2, 64), dtypes.Uint8),
        ((2, 1, 128), (1, 1, 128), (2, 1, 64), dtypes.Uint8),
        ((4, 2, 64), (1, 1, 128), (2, 1, 64), dtypes.Uint8),
    ],
    ids=str,
)
def test_tiling_across_partial_vectors_raises(
    tensor_shape, vector_shape, tile_shape, dtype
):
    def inner(layout):
        spec = specs.HvxVmemTensorSpec(
            tensor_shape,
            dtype=dtype,
            bank="VMEM",
            layout=layout,
            vector_shape=vector_shape,
        )
        tensor = hexagon.HvxVmemTensor(spec, None, origin=None)
        with pytest.raises(ValueError):
            tensor.simple_tile(tile_shape)

    for layout in specs.Layout:
        inner(layout)


@hypothesis.given(_st_test_tiling_always_produces_vector_indices_in_parent_set())
def test_tiling_always_produces_vector_indices_in_parent_set(shapes):
    # TODO: vector_shape shouldn't be the initial tensor size, and it shouldn't use
    #   assume.
    vector_shape = shapes[0]
    hypothesis.assume(functools.reduce(operator.mul, vector_shape, 1) == 128)

    dtype = dtypes.Uint8

    def inner(layout):
        spec = specs.HvxVmemTensorSpec(
            shapes[0],
            dtype=dtype,
            bank="VMEM",
            layout=layout,
            vector_shape=vector_shape,
        )
        prev_tensorlike = hexagon.HvxVmemTensor(spec, None, origin=None)
        for shape in shapes[1:]:
            tile: hexagon.HvxVmemTensorlike = prev_tensorlike.simple_tile(shape)
            prev_tensorlike = tile

    for layout in specs.Layout:
        inner(layout)
