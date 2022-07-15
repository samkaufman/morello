import hypothesis
from hypothesis import strategies as st

from morello import dtypes, layouts, system_config, tensor
from morello.impl import utils

from .. import strategies

strategies.register_default_strategies()

def test_nchwc_is_noncontiguous_when_breaking_by_nonmultiple():
    layout = layouts.PackedLayout(4, 1, 32)  # NCHWc with 4-split
    outer_shape = (8, 64, 8, 8)
    bad_shape = (8, 32, 8, 8)

    oi = tensor.OperandIdx(0)

    target = system_config.current_target()
    outer_spec = target.tensor_spec(outer_shape, dtypes.Uint8, bank="GL", layout=layout)
    assert not outer_spec.simple_tile(oi, bad_shape).spec.contiguous


@hypothesis.example([1, 1], True)
@hypothesis.given(st.lists(st.integers(0, 20), max_size=4), st.booleans())
def test_gen_tile_sizes_produces_only_smaller_tiles(tile_shape, drop_given):
    for out_shape in utils.gen_tile_sizes(tile_shape, drop_given=drop_given):
        assert all(d <= s for d, s in zip(out_shape, tile_shape))