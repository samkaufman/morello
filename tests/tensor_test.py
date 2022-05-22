import io
import pickle

import hypothesis
import pytest
from hypothesis import strategies as st

from morello import dtypes, layouts, specs, system_config, tensor

from . import strategies

strategies.register_default_strategies()


@pytest.mark.parametrize(
    "tensor_shape, tile_shape, expected",
    [
        ((8, 8), (8, 8), True),
        ((8, 8, 8), (8, 8, 8), True),
        ((8, 8, 8), (4, 8, 8), True),
    ],
)
def test_tile_contiguous(tensor_shape, tile_shape, expected):
    # TODO: Vary the following three parameters with hypothesis
    target = system_config.current_target()
    dtype, bank, layout = dtypes.Uint8, "RF", layouts.ROW_MAJOR
    tensor_spec = target.tensor_spec(tensor_shape, dtype, bank, layout)
    t = target.tensor(tensor_spec, name=None)
    tile = t.simple_tile(tensor.OperandIdx(0), tile_shape)
    assert tile.contiguous == expected


@hypothesis.given(st.from_type(specs.TensorSpec), st.text(), st.booleans())
def test_tensors_and_tiles_can_be_pickled_and_unpickled_losslessly(
    spec, name, should_tile
):
    t = system_config.current_target().tensor(spec=spec, name=name)
    if should_tile:
        t = t.simple_tile(tensor.OperandIdx(0), tuple(1 for _ in t.dim_sizes))

    buf = io.BytesIO()
    pickle.dump(t, buf)
    buf.seek(0)
    read_tensor = pickle.load(buf)
    # TODO: Add a deep equality method
    assert str(t) == str(read_tensor)


@pytest.mark.parametrize(
    "outer_shp, tile_shp, filt_shp, expected_batch, expected_height, expected_width",
    [
        ((10, 3, 3), (4, 3, 3), (1, 1), 2, 0, 0),
        ((1, 3, 3), (1, 3, 3), (1, 1), 0, 0, 0),
        ((1, 3, 3), (1, 3, 3), (3, 3), 0, 0, 0),
        ((1, 3, 3), (1, 2, 2), (1, 1), 0, 1, 1),
        ((1, 3, 3), (1, 2, 2), (2, 2), 0, 0, 0),
    ],
)
def test_convolution_image_tile_boundary_size(
    outer_shp, tile_shp, filt_shp, expected_batch, expected_height, expected_width
):
    # TODO: any_level should be something arbitrary from the system_config
    any_level = "GL"
    outer = tensor.Tensor(
        specs.TensorSpec(outer_shp, dtypes.Uint8, any_level, layouts.ROW_MAJOR),
        name=None,
    )
    tile = tensor.ConvolutionImageTile(
        tensor.OperandIdx(0), tile_shp, filter_shape=filt_shp, name=None
    )
    assert tile.boundary_size(0) == expected_batch
    assert tile.boundary_size(1) == expected_height
    assert tile.boundary_size(2) == expected_width
