import hypothesis
from hypothesis import strategies as st

from morello.impl import utils

from .. import strategies

strategies.register_default_strategies()


@hypothesis.example([1, 1], True)
@hypothesis.given(st.lists(st.integers(0, 20), max_size=4), st.booleans())
def test_gen_tile_sizes_produces_only_smaller_tiles(tile_shape, drop_given):
    for out_shape in utils.gen_tile_sizes(tile_shape, drop_given=drop_given):
        assert all(d <= s for d, s in zip(out_shape, tile_shape))
