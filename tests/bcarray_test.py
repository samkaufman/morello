import hypothesis
import hypothesis.strategies as st
import pytest
import numpy as np
import asyncio
import fakeredis.aioredis as fakeredis

from morello import bcarray
from . import strategies

strategies.register_default_strategies()


@st.composite
def _arb_test_bcarray_fills_result_array_matching_ndarray_args(draw):
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

    # Decide whether or not to back with Redis.
    use_redis = None
    if draw(st.booleans()):
        use_redis = (fakeredis.FakeRedis(), "TEST")

    bs_arr = bcarray.BlockCompressedArray(
        arr_shape,
        block_shape,
        default_value,
        dense_block_threshold=draw(st.integers(min_value=0, max_value=5)),
        compress_on_fill=draw(st.booleans()),
        use_redis=use_redis,
    )

    async def ainit():
        for fill_lower, fill_upper, fill_value in fills:
            await bs_arr.fill_range(fill_lower, fill_upper, fill_value)

    return ainit, arr_shape, block_shape, default_value, bs_arr, fills


@hypothesis.settings(deadline=2000)
@hypothesis.given(_arb_test_bcarray_fills_result_array_matching_ndarray_args())
@pytest.mark.asyncio
async def test_bcarray_fills_result_array_matching_ndarray(example):
    (ainit, arr_shape, _, default_value, bs_arr, fills) = example
    await ainit()

    np_arr = np.full(arr_shape, fill_value=default_value, dtype=object)
    for fill_lower, fill_upper, fill_value in fills:
        np_arr[tuple(slice(a, b) for a, b in zip(fill_lower, fill_upper))] = fill_value

    np.testing.assert_array_equal(await bs_arr.to_dense(), np_arr)


@hypothesis.settings(deadline=2000)
@hypothesis.given(_arb_test_bcarray_fills_result_array_matching_ndarray_args())
@pytest.mark.asyncio
async def test_bcarray_to_dense_returns_correct_shape(example):
    (ainit, _, _, _, bs_arr, _) = example
    await ainit()
    assert (await bs_arr.to_dense()).shape == bs_arr.shape


@hypothesis.settings(deadline=2000)
@hypothesis.given(_arb_test_bcarray_fills_result_array_matching_ndarray_args())
@pytest.mark.asyncio
async def test_bcarray_iter_values_returns_set_present_in_dense(example):
    (ainit, _, _, _, bs_arr, _) = example
    await ainit()
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


@st.composite
def _arb_surface_pts_inputs(draw):
    rank = draw(st.integers(min_value=0, max_value=4))
    lower = tuple(draw(st.integers(min_value=1, max_value=16)) for _ in range(rank))
    upper_adds = tuple(
        draw(st.integers(min_value=1, max_value=16)) for _ in range(rank)
    )
    upper = tuple(l + u for l, u in zip(lower, upper_adds))

    lower_misaligned_dim_bits = draw(
        st.lists(st.booleans(), min_size=rank, max_size=rank)
    )
    upper_misaligned_dim_bits = draw(
        st.lists(st.booleans(), min_size=rank, max_size=rank)
    )
    lower_misaligned_dims = [i for i, b in enumerate(lower_misaligned_dim_bits) if b]
    upper_misaligned_dims = [i for i, b in enumerate(upper_misaligned_dim_bits) if b]

    return lower, upper, lower_misaligned_dims, upper_misaligned_dims


@hypothesis.given(_arb_surface_pts_inputs())
def test_surface_pts_never_returns_duplicates(inp):
    lower, upper, lower_misaligned_dims, upper_misaligned_dims = inp
    result = list(
        bcarray.surface_pts(lower, upper, lower_misaligned_dims, upper_misaligned_dims)
    )
    assert len(result) == len(set(result))
