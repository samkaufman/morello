import hypothesis
import hypothesis.strategies as st
import pytest
import numpy as np
import asyncio

from morello import bcarray
from . import strategies

strategies.register_default_strategies()


@st.composite
def _arb_test_bcarray_fills_result_array_matching_ndarray_args(draw):
    loop = asyncio.new_event_loop()

    rank = draw(st.integers(min_value=1, max_value=4))
    arr_shape = tuple(draw(st.integers(min_value=1, max_value=8)) for _ in range(rank))
    block_shape = tuple(draw(st.integers(min_value=1, max_value=d)) for d in arr_shape)
    default_value = draw(st.one_of(st.none(), st.integers()))
    fills = []
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        fill_lower = tuple(
            draw(st.integers(min_value=0, max_value=d)) for d in arr_shape
        )
        fill_upper = tuple(
            draw(st.integers(min_value=l, max_value=d))
            for l, d in zip(fill_lower, arr_shape)
        )
        fill_value = draw(st.one_of(st.none(), st.integers()))
        fills.append((fill_lower, fill_upper, fill_value))
    bs_arr = bcarray.BlockCompressedArray(arr_shape, block_shape, default_value)
    for fill_lower, fill_upper, fill_value in fills:
        loop.run_until_complete(bs_arr.fill_range(fill_lower, fill_upper, fill_value))
    return arr_shape, block_shape, default_value, bs_arr, fills


@hypothesis.given(_arb_test_bcarray_fills_result_array_matching_ndarray_args())
@pytest.mark.asyncio
async def test_bcarray_fills_result_array_matching_ndarray(example):
    (arr_shape, _, default_value, bs_arr, fills) = example

    np_arr = np.full(arr_shape, fill_value=default_value, dtype=object)
    for fill_lower, fill_upper, fill_value in fills:
        np_arr[tuple(slice(a, b) for a, b in zip(fill_lower, fill_upper))] = fill_value

    np.testing.assert_array_equal(await bs_arr.to_dense(), np_arr)


@hypothesis.given(_arb_test_bcarray_fills_result_array_matching_ndarray_args())
@pytest.mark.asyncio
async def test_bcarray_to_dense_returns_correct_shape(example):
    (_, _, _, bs_arr, _) = example
    assert (await bs_arr.to_dense()).shape == bs_arr.shape


@hypothesis.given(_arb_test_bcarray_fills_result_array_matching_ndarray_args())
@pytest.mark.asyncio
async def test_bcarray_iter_values_returns_set_present_in_dense(example):
    (_, _, _, bs_arr, _) = example
    assert set([x async for x in bs_arr.iter_values()]) == set(
        (await bs_arr.to_dense()).flatten().tolist()
    )


@pytest.mark.asyncio
async def test_bcarray_raises_indexerror_on_fill_range_greater_than_shape():
    arr_shape = (3, 3)
    block_shape = (2, 2)
    default_value = 0
    fill_lower = (0, 0)
    fill_upper = (3, 4)
    fill_value = 1

    bs_arr = bcarray.BlockCompressedArray(arr_shape, block_shape, default_value)
    with pytest.raises(IndexError):
        await bs_arr.fill_range(fill_lower, fill_upper, fill_value)
