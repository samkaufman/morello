import itertools

import fakeredis.aioredis as fakeredis
import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest

from morello import bcarray
from . import strategies

strategies.register_default_strategies()


@st.composite
def _arb_bcarray(draw):
    rank = draw(st.integers(min_value=1, max_value=3))
    arr_shape = tuple(draw(st.integers(min_value=1, max_value=8)) for _ in range(rank))
    block_shape = tuple(draw(st.integers(min_value=1, max_value=d)) for d in arr_shape)

    # Decide whether to back with Redis.
    if draw(st.booleans()):
        grid = bcarray.BCARedisStore(
            arr_shape,
            block_shape,
            fakeredis.FakeRedis(),
            "TEST",
            bcarray.BCA_DEFAULT_VALUE,
            local_cache={},
        )
    else:
        grid = bcarray.NumpyStore(arr_shape, block_shape, bcarray.BCA_DEFAULT_VALUE)

    bc_arr = bcarray.BlockCompressedArray(arr_shape, block_shape, grid)
    return arr_shape, block_shape, bc_arr


@st.composite
def _arb_bcarray_with_fills(draw, fill_min=1, fill_max=None):
    arr_shape, block_shape, bs_arr = draw(_arb_bcarray())

    if fill_max is None:
        fill_max = bcarray._BCA_DENSE_BLOCK_THRESHOLD + 1

    fills = []
    for _ in range(draw(st.integers(min_value=fill_min, max_value=fill_max))):
        fill_lower = tuple(
            draw(st.integers(min_value=0, max_value=d)) for d in arr_shape
        )
        fill_upper = tuple(
            draw(st.integers(min_value=l, max_value=d))
            for l, d in zip(fill_lower, arr_shape)
        )
        fill_value = draw(st.one_of(st.none(), st.integers()))
        fills.append((fill_lower, fill_upper, fill_value))

    return arr_shape, block_shape, bs_arr, fills


# TODO: Add get_many variant
@hypothesis.settings(deadline=2000)
@pytest.mark.asyncio
@hypothesis.given(_arb_bcarray_with_fills())
async def test_bcarray_fills_match_ndarray(example):
    arr_shape, _, bc_arr, fills = example

    for fill_lower, fill_upper, fill_value in fills:
        await bc_arr.fill_range(fill_lower, fill_upper, fill_value)

    np_arr = np.full(arr_shape, fill_value=bcarray.BCA_DEFAULT_VALUE, dtype=object)
    for fill_lower, fill_upper, fill_value in fills:
        np_arr[tuple(slice(a, b) for a, b in zip(fill_lower, fill_upper))] = fill_value

    results_arr: np.ndarray
    results_arr = np.empty(arr_shape, dtype=object)
    results_list = await bc_arr.get_many(list(np.ndindex(arr_shape)))
    assert len(results_list) == np.prod(arr_shape), (
        f"Expected {np.prod(arr_shape)} results, got {len(results_list)}; "
        f'those results were: "{results_list}"'
    )
    for idx, entry in zip(np.ndindex(arr_shape), results_list, strict=True):
        results_arr[idx] = entry

    np.testing.assert_array_equal(results_arr, np_arr)


@hypothesis.settings(deadline=2000)
@pytest.mark.asyncio
@pytest.mark.parametrize("should_flush", [True, False])
@pytest.mark.parametrize("batch_get", [True, False])
@hypothesis.given(_arb_bcarray_with_fills(fill_min=1, fill_max=1))
async def test_bcarray_fill_followed_by_get_returns_same(
    should_flush, batch_get, example
):
    _, _, bc_arr, fills = example
    lower, upper, fill_value = fills[0]
    await bc_arr.fill_range(lower, upper, fill_value)

    if should_flush:
        await bc_arr.flush()

    query_pts = list(itertools.product(*[range(*t) for t in zip(lower, upper)]))
    if batch_get:
        results = await bc_arr.get_many(query_pts)
    else:
        results = [await bc_arr.get(pt) for pt in query_pts]

    assert all(r == fill_value for r in results)


@hypothesis.settings(deadline=2000)
@hypothesis.given(_arb_bcarray())
@pytest.mark.asyncio
async def test_bcarray_to_dense_returns_correct_shape(example):
    _, _, bc_arr = example
    assert (await bc_arr.to_dense()).shape == bc_arr.shape


@hypothesis.settings(deadline=2000)
@hypothesis.given(_arb_bcarray_with_fills())
@pytest.mark.asyncio
async def test_bcarray_iter_values_returns_set_present_in_dense(example):
    _, _, bc_arr, fills = example
    for fill_lower, fill_upper, fill_value in fills:
        await bc_arr.fill_range(fill_lower, fill_upper, fill_value)
    assert set([x async for x in bc_arr.iter_values()]) == set(
        (await bc_arr.to_dense()).flatten().tolist()
    )


@pytest.mark.asyncio
async def test_bcarray_raises_indexerror_on_fill_range_greater_than_shape():
    arr_shape = (3, 3)
    block_shape = (2, 2)
    fill_lower = (0, 0)
    fill_upper = (3, 4)
    fill_value = 1

    grid = bcarray.NumpyStore(arr_shape, block_shape, bcarray.BCA_DEFAULT_VALUE)
    bc_arr = bcarray.BlockCompressedArray(arr_shape, block_shape, grid)
    with pytest.raises(IndexError):
        await bc_arr.fill_range(fill_lower, fill_upper, fill_value)


@st.composite
def _arb_surface_pts_inputs(draw):
    rank = draw(st.integers(min_value=0, max_value=4))
    lower = tuple(draw(st.integers(min_value=1, max_value=8)) for _ in range(rank))
    upper_adds = tuple(draw(st.integers(min_value=1, max_value=8)) for _ in range(rank))
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


@hypothesis.settings(deadline=1000)
@hypothesis.given(_arb_surface_pts_inputs())
def test_surface_pts_never_returns_duplicates(inp):
    lower, upper, lower_misaligned_dims, upper_misaligned_dims = inp
    result = list(
        bcarray.surface_pts(lower, upper, lower_misaligned_dims, upper_misaligned_dims)
    )
    assert len(result) == len(set(result))
