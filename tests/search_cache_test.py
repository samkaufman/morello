import asyncio
import hypothesis
import pytest
import fakeredis.aioredis
from hypothesis import strategies as st

from morello import dtypes, impl, pformat, pruning, search_cache, specs
from morello.search import dp
from morello.system_config import current_system, current_target

from . import strategies

strategies.register_default_strategies()

# TODO: Add assertion that tests below only put scheduled Impls into the cache.

CACHE_CLASSES = [search_cache.InMemoryScheduleCache, search_cache.RedisCache]


def _make_cache(search_cls):
    if search_cls == search_cache.InMemoryScheduleCache:
        return search_cache.InMemoryScheduleCache()
    elif search_cls == search_cache.RedisCache:
        redis_conn = fakeredis.aioredis.FakeRedis()
        return search_cache.RedisCache(
            redis_conn, "test", lambda s, _: (s.operands[0].dim_sizes[0],)
        )
    else:
        raise NotImplementedError(f"Unsupported type {search_cls}")


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_cls", CACHE_CLASSES)
@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
async def test_cache_common_scenario(cache_cls, dtype):
    cache = _make_cache(cache_cls)
    target = current_target()

    lhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    output = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    fast_schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    fast_wrapped_schedule = search_cache.CachedScheduleSet(
        fast_schedule.spec, ((fast_schedule, 10),), 1
    )

    lhs = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    output = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    slow_schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    slow_wrapped_schedule = search_cache.CachedScheduleSet(
        slow_schedule.spec, ((slow_schedule, 50),), 1
    )

    lhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    output = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    impossible_schedule = impl.MatmulHole(
        specs.Matmul(lhs, rhs, output, serial_only=False)
    )
    impossible_wrapped_schedule = search_cache.CachedScheduleSet(
        impossible_schedule.spec, tuple(), 1
    )

    await cache.put(
        fast_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
    )
    assert {x async for x in cache} == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
        )
    }
    await cache.put(
        slow_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 50, "VRF": 50, "L1": 50, "GL": 0}),
    )
    assert {x async for x in cache} == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
        ),
        (
            slow_schedule.spec,
            slow_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 50, "VRF": 50, "L1": 50, "GL": 0}),
        ),
    }
    await cache.put(
        impossible_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 1, "VRF": 1, "L1": 1, "GL": 0}),
    )
    assert {x async for x in cache} == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
        ),
        (
            slow_schedule.spec,
            slow_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 50, "VRF": 50, "L1": 50, "GL": 0}),
        ),
        (
            impossible_schedule.spec,
            impossible_wrapped_schedule,
            pruning.StandardMemoryLimits(({"RF": 1, "VRF": 1, "L1": 1, "GL": 0})),
        ),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_cls", CACHE_CLASSES)
@hypothesis.given(
    strategies.small_atomic_specs_st, strategies.arb_small_standard_memorylimits()
)
async def test_cache_raises_keyerror_when_empty(cache_cls, spec, limits):
    cache = _make_cache(cache_cls)
    with pytest.raises(KeyError):
        await cache.get(spec, limits)


# TODO: Add Compose Specs (incl. PipelineChildMemoryLimits)
# TODO: Test top_k greater than 1
@pytest.mark.asyncio
@pytest.mark.parametrize("cache_cls", CACHE_CLASSES)
@hypothesis.settings(deadline=240_000)
@hypothesis.given(
    strategies.small_atomic_specs_st,
    strategies.arb_small_standard_memorylimits(),
)
async def test_cache_get_returns_just_put_impls(cache_cls, spec, limits):
    cache = _make_cache(cache_cls)

    optimal = await dp.Search(top_k=1)(spec, limits, parent_summary=None, cache=cache)
    hypothesis.assume(optimal)
    optimal = optimal[0]
    assert isinstance(optimal, impl.Impl)

    results = [imp for imp, _ in (await cache.get(spec, limits)).contents]
    assert len(results) == 1

    hypothesis.note(pformat(results[0], show_cost=True))
    hypothesis.note(pformat(optimal, show_cost=True))
    # TODO: Don't use pformat/string comparison.
    assert pformat(results[0], show_cost=False) == pformat(optimal, show_cost=False)


# TODO: Add Compose Specs (incl. PipelineChildMemoryLimits)
# TODO: Test top_k greater than 1
@pytest.mark.parametrize("cache_cls", CACHE_CLASSES)
@hypothesis.settings(deadline=90_000)
@hypothesis.given(
    strategies.small_atomic_specs_st, strategies.arb_small_standard_memorylimits()
)
def test_cache_reputting_doesnt_increase_size(cache_cls, spec, limits):
    cache = _make_cache(cache_cls)
    optimal = dp.schedule_search(spec, limits, cache=cache)
    hypothesis.assume(optimal)
    optimal = optimal[0]
    assert isinstance(optimal, impl.Impl)

    loop = asyncio.new_event_loop()

    initial_count = cache.count_impls()
    p = loop.run_until_complete(cache.get(spec, limits))
    loop.run_until_complete(cache.put(spec, p, limits))
    assert cache.count_impls() == initial_count


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
@pytest.mark.parametrize("contiguous", [True, False], ids=["contig", "noncontig"])
async def test_cache_updates_when_none_result_put_with_higher_memory_cap(
    dtype, contiguous
):
    t = specs.TensorSpec(
        (8, 8), dtype=dtype, contiguous_abs=(4 if contiguous else 0), bank="RF"
    )
    spec = specs.Matmul(t, t, t, serial_only=False)
    wrapped_schedule = search_cache.CachedScheduleSet(spec, tuple(), 1)

    cache = search_cache.InMemoryScheduleCache()
    await cache.put(
        wrapped_schedule, pruning.StandardMemoryLimits({"RF": 100, "L1": 100, "GL": 0})
    )
    assert {x async for x in aiter(cache)} == {
        (
            spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "L1": 100, "GL": 0}),
        )
    }
    await cache.put(
        wrapped_schedule, pruning.StandardMemoryLimits({"RF": 101, "L1": 101, "GL": 0})
    )
    assert {x async for x in aiter(cache)} == {
        (
            spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 101, "L1": 101, "GL": 0}),
        )
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
async def test_cache_updates_when_schedules_put_with_higher_memory_cap(dtype):
    target = current_target()
    db = current_system().default_bank
    lhs = target.tensor_spec((8, 8), dtype=dtype, bank=db)
    rhs = target.tensor_spec((8, 8), dtype=dtype, bank=db)
    output = target.tensor_spec((8, 8), dtype=dtype, bank=db)
    schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    wrapped_schedule = search_cache.CachedScheduleSet(
        schedule.spec, ((schedule, 10),), 1
    )

    cache = search_cache.InMemoryScheduleCache()
    await cache.put(
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
    )
    assert {x async for x in cache} == {
        (
            schedule.spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
        )
    }
    await cache.put(
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 101, "VRF": 101, "L1": 101, "GL": 0}),
    )
    assert {x async for x in cache} == {
        (
            schedule.spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 101, "VRF": 101, "L1": 101, "GL": 0}),
        )
    }


# def test_references_resolved_correctly_in_saved_caches():
#     spec = specs.Matmul(
#         specs.TensorSpec((2, 1), dtypes.Uint8, bank="L1", contiguous_abs=0),
#         specs.TensorSpec((1, 1), dtypes.Uint8, bank="RF", contiguous_abs=0),
#         specs.TensorSpec((2, 1), dtypes.Uint8, bank="RF", contiguous_abs=0),
#         serial_only=False,
#     )
#
#     dir_path = pathlib.Path(tempfile.mkdtemp())
#     try:
#         with search_cache.persistent_cache(dir_path / "cache.pkl") as cache:
#             dp.schedule_search(spec, cache=cache)
#             assert len(cache)
#             cache.check_children_are_entries(None)
#         with search_cache.persistent_cache(dir_path / "cache.pkl", save=False) as cache:
#             assert len(cache)
#             cache.check_children_are_entries(None)
#     finally:
#         shutil.rmtree(dir_path)
