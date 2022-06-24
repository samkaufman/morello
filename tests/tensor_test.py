import io
import pickle

import hypothesis
import pytest
from hypothesis import strategies as st

from morello import dtypes, layouts, specs, system_config, tensor

from . import strategies

strategies.register_default_strategies()


@hypothesis.given(st.from_type(specs.TensorSpec), st.text(), st.booleans())
def test_tensors_and_tiles_can_be_pickled_and_unpickled_losslessly(
    spec, name, should_tile
):
    t = system_config.current_target().tensor(spec=spec, name=name)
    if should_tile:
        t = t.spec.simple_tile(tensor.OperandIdx(0), tuple(1 for _ in t.dim_sizes))

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
        specs.TensorSpec(outer_shp, dtypes.Uint8, True, any_level, layouts.row_major(len(outer_shp))),
        name=None,
    )
    tile = tensor.ConvolutionImageTile(
        tensor.OperandIdx(0),
        specs.TensorSpec(tile_shp, dtypes.Uint8, any_level, layouts.row_major(len(tile_shp))),
        filter_shape=filt_shp,
        name=None,
    )
    assert tile.boundary_size(0, outer_shp[0]) == expected_batch
    assert tile.boundary_size(1, outer_shp[1]) == expected_height
    assert tile.boundary_size(2, outer_shp[2]) == expected_width
