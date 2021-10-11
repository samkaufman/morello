import io
import pickle

import hypothesis
import pytest
from hypothesis import strategies as st

from morello import dtypes, specs, system_config, tensor

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
    dtype, bank, layout = dtypes.Uint8, "RF", specs.Layout.ROW_MAJOR
    tensor_spec = target.tensor_spec(tensor_shape, dtype, bank, layout)
    t = target.tensor(tensor_spec, name=None, origin=None)
    tile = t.simple_tile(tile_shape)
    assert tile.contiguous == expected


@hypothesis.given(st.from_type(specs.TensorSpec), st.text(), st.booleans())
def test_tensors_and_tiles_can_be_pickled_and_unpickled_losslessly(
    spec, name, should_tile
):
    t = system_config.current_target().tensor(spec=spec, name=name)
    if should_tile:
        t = t.simple_tile(tuple(1 for _ in t.dim_sizes))

    buf = io.BytesIO()
    pickle.dump(t, buf)
    buf.seek(0)
    read_tensor = pickle.load(buf)
    # TODO: Add a deep equality method
    assert str(t) == str(read_tensor)
