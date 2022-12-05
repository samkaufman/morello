import hypothesis
import hypothesis.strategies as st
import pytest
import numpy as np

from morello import bcarray
from . import strategies

strategies.register_default_strategies()


@st.composite
def _arb_test_bcarray_fills_result_array_matching_ndarray_args(draw):
    rank = draw(st.integers(min_value=1, max_value=4))
    arr_shape = tuple(draw(st.integers(min_value=1, max_value=4)) for _ in range(rank))
    block_shape = tuple(draw(st.integers(min_value=1, max_value=d)) for d in arr_shape)
    fill_lower = tuple(draw(st.integers(min_value=0, max_value=d)) for d in arr_shape)
    fill_upper = tuple(
        draw(st.integers(min_value=l, max_value=d))
        for l, d in zip(fill_lower, arr_shape)
    )
    default_value = draw(st.one_of(st.none(), st.integers()))
    fill_value = draw(st.one_of(st.none(), st.integers()))
    return arr_shape, block_shape, default_value, fill_lower, fill_upper, fill_value


@hypothesis.given(_arb_test_bcarray_fills_result_array_matching_ndarray_args())
def test_bcarray_fills_result_array_matching_ndarray(example):
    arr_shape, block_shape, default_value, fill_lower, fill_upper, fill_value = example

    bs_arr = bcarray.BlockCompressedArray(arr_shape, block_shape, default_value)
    bs_arr.fill_range(fill_lower, fill_upper, fill_value)

    np_arr = np.full(arr_shape, fill_value=default_value, dtype=object)
    np_arr[tuple(slice(a, b) for a, b in zip(fill_lower, fill_upper))] = fill_value

    np.testing.assert_array_equal(bs_arr.to_dense(), np_arr)


def test_bcarray_raises_indexerror_on_fill_range_greater_than_shape():
    arr_shape = (3, 3)
    block_shape = (2, 2)
    default_value = 0
    fill_lower = (0, 0)
    fill_upper = (3, 4)
    fill_value = 1

    bs_arr = bcarray.BlockCompressedArray(arr_shape, block_shape, default_value)
    with pytest.raises(IndexError):
        bs_arr.fill_range(fill_lower, fill_upper, fill_value)
