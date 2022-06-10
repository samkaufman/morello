import hypothesis
import pytest
from hypothesis import strategies as st

import morello.utils
from morello import dtypes, layouts, system_config, tensor
from morello.impl import utils

from .. import strategies

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
    assert morello.utils.contiguous(tile, t) == expected


@hypothesis.example([1, 1], True)
@hypothesis.given(st.lists(st.integers(0, 20), max_size=4), st.booleans())
def test_gen_tile_sizes_produces_only_smaller_tiles(tile_shape, drop_given):
    for out_shape in utils.gen_tile_sizes(tile_shape, drop_given=drop_given):
        assert all(d <= s for d, s in zip(out_shape, tile_shape))
