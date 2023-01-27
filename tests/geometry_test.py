import pytest
import hypothesis
from hypothesis import strategies as st

from . import strategies

from morello import geometry

strategies.register_default_strategies()


@st.composite
def _arb_simpleblockedrange(draw):
    # TODO: Add reversed ranges.
    start = draw(st.integers(min_value=0, max_value=4))
    length = draw(st.integers(min_value=0, max_value=4))
    block_size = draw(st.integers(min_value=1, max_value=max(1, length)))
    return geometry.SimpleBlockedRange(start, start + length, block_size=block_size)


@hypothesis.given(_arb_simpleblockedrange())
def test_griddimrange_block_points_contains_call(grid_dim):
    for pt in range(grid_dim.start, grid_dim.stop):
        hypothesis.note(f"pt={pt}")
        block = grid_dim.block_index(pt)
        assert pt in list(grid_dim.points_in_block(block))


@pytest.mark.parametrize(
    "length,block_size,expected_block_count",
    [
        (0, 1, 0),
        (1, 1, 1),
        (2, 1, 2),
        (3, 2, 2),
        (4, 2, 2),
        (5, 2, 3),
    ],
)
@pytest.mark.parametrize("step", [-1, 1])
@hypothesis.given(start=st.integers(min_value=0, max_value=4))
def test_griddimrange_block_counts(
    start, step, length, block_size, expected_block_count
):
    b, e = start, start + length
    if step == -1:
        b, e = e - 1, b - 1
    hypothesis.note(
        f"SimpleBlockedRange({b}, {e}, step={step}, block_size={block_size})"
    )
    grid_dim = geometry.SimpleBlockedRange(b, e, step, block_size)
    hypothesis.note("_range is " + str(grid_dim._range))
    assert expected_block_count == grid_dim.block_count


@hypothesis.given(_arb_simpleblockedrange())
def test_griddimrange_points_in_block_raises_indexerror(grid_dim):
    with pytest.raises(IndexError):
        grid_dim.points_in_block(grid_dim.block_count)


@hypothesis.given(_arb_simpleblockedrange())
def test_griddimrange_reversed_produces_reversed_points(blocked_range):
    assert list(reversed(blocked_range)) == list(reversed(list(blocked_range)))


@hypothesis.given(_arb_simpleblockedrange())
def test_griddimrange_reversing_twice_produces_same_grimdimrange(grid_dim):
    assert grid_dim == reversed(reversed(grid_dim))
