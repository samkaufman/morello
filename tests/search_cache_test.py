import enum

import fakeredis.aioredis as fakeredis
import hypothesis
import pytest

from morello import dtypes, impl, pformat, pruning, search_cache, specs
from morello.cost import MainCost
from morello.search import dp
from morello.search.common import ScheduleKey
from morello.system_config import current_system, current_target
from morello.utils import snap_availables_up
from . import strategies

strategies.register_default_strategies()


# TODO: Add assertion that tests below only put scheduled Impls into the cache.


class CacheConfig(enum.Enum):
    inmem = enum.auto()
    redis = enum.auto()


def _make_cache(config: CacheConfig):
    match config:
        case CacheConfig.inmem:
            return search_cache.ScheduleCache()
        case CacheConfig.redis:
            return search_cache.ScheduleCache(use_redis=(fakeredis.FakeRedis(), "TEST"))
        case _:
            raise NotImplementedError(f"Unsupported config {config}")


CACHE_CONFIGS = ["inmem", "redis"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_cls", CacheConfig)
@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
async def test_cache_common_scenario(cache_cls, dtype):
    cache = _make_cache(cache_cls)
    target = current_target()

    lhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    output = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    fast_schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    fast_schedule_key = ScheduleKey(
        MainCost(10),
        snap_availables_up(fast_schedule.peak_memory.raw_values),
        fast_schedule.depth,
    )
    fast_wrapped_schedule = search_cache.CachedScheduleSet(
        fast_schedule.spec, ((fast_schedule, fast_schedule_key),), 1
    )

    lhs = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    output = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    slow_schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    slow_schedule_key = ScheduleKey(
        MainCost(50),
        snap_availables_up(slow_schedule.peak_memory.raw_values),
        slow_schedule.depth,
    )
    slow_wrapped_schedule = search_cache.CachedScheduleSet(
        slow_schedule.spec, ((slow_schedule, slow_schedule_key),), 1
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
@pytest.mark.parametrize("cache_cls", CacheConfig)
@hypothesis.given(
    strategies.small_atomic_specs_st, strategies.arb_small_standard_memorylimits()
)
@hypothesis.settings(deadline=10_000)  # TODO: Profile and speed this test up.
async def test_cache_raises_keyerror_when_empty(cache_cls, spec, limits):
    cache = _make_cache(cache_cls)
    with pytest.raises(KeyError):
        await cache.get(spec, limits)


# TODO: Add Compose Specs (incl. PipelineChildMemoryLimits)
# TODO: Test top_k greater than 1
@pytest.mark.asyncio
@pytest.mark.parametrize("cache_cls", CacheConfig)
@pytest.mark.parametrize("use_get_many", [True, False], ids=["get_many", "get"])
@pytest.mark.parametrize("flush_after_put", [True, False], ids=["flush", "no_flush"])
@hypothesis.settings(deadline=240_000)
@hypothesis.given(
    strategies.tiny_atomic_specs_st,
    strategies.arb_small_standard_memorylimits(),
)
async def test_cache_get_returns_just_put_impls(
    cache_cls, use_get_many, flush_after_put, spec, limits
):
    cache = _make_cache(cache_cls)

    # Use DP search below to ensure that we don't produce an Impl with differing
    # sub-Impls for the same Spec.
    optimal = await dp.Search(top_k=1)(spec, limits, parent_summary=None, cache=cache)
    hypothesis.assume(optimal)
    optimal = optimal[0]
    assert isinstance(optimal, impl.Impl)

    # Collect every node of the optimal schedule.
    all_nodes = []
    remaining_nodes = [(optimal, limits)]
    while remaining_nodes:
        node, node_memory_bound = remaining_nodes.pop()
        child_limits = node_memory_bound.transition(node)
        all_nodes.append((node, node_memory_bound))
        remaining_nodes.extend(zip(node.children, child_limits, strict=True))

    if flush_after_put:
        await cache.flush()

    results: list[impl.Impl] = []
    if use_get_many:
        batch_results = await cache.get_many([(n.spec, l) for n, l in all_nodes])
        results.extend(imp for r in batch_results for imp, _ in r.contents)
    else:
        for node, node_limits in all_nodes:
            spec = node.spec
            results.extend(
                imp for imp, _ in (await cache.get(spec, node_limits)).contents
            )

    for result, (expected, _) in zip(results, all_nodes, strict=True):
        assert not isinstance(result, impl.base.AppliedImpl)
        assert not isinstance(expected, impl.base.AppliedImpl)
        assert result.equals_node(expected), (
            f"Result didn't equal expected.\n{pformat(result, show_cost=False)}\n"
            f"!=\n{pformat(expected, show_cost=False)}"
        )

    await cache.flush()


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

    cache = search_cache.ScheduleCache()
    await cache.put(
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
    )
    assert {x async for x in aiter(cache)} == {
        (
            spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
        )
    }

    await cache.put(
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 101, "VRF": 101, "L1": 101, "GL": 0}),
    )
    assert {x async for x in aiter(cache)} == {
        (
            spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 101, "VRF": 101, "L1": 101, "GL": 0}),
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
    k = ScheduleKey(
        MainCost(10),
        snap_availables_up(schedule.peak_memory.raw_values),
        schedule.depth,
    )
    wrapped_schedule = search_cache.CachedScheduleSet(
        schedule.spec, ((schedule, k),), 1
    )

    cache = search_cache.ScheduleCache()
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


# @pytest.mark.asyncio
# async def test_topdown_search_succeeds_with_redis_cache():
#     # Load((1×1, u32, L1, ua), (1×1, u32, RF, ua), serial) != MatmulAccum((1×16, u32, RF), (16×1, u32, L1, c1, ua), (1×1, u
#     #                                                                    │32, L1, ua), serial)
#     spec = specs.Matmul(
#         specs.TensorSpec((1, 32), dtype=dtypes.Uint32, bank="GL"),
#         specs.TensorSpec((32, 5), dtype=dtypes.Uint32, bank="GL"),
#         specs.TensorSpec((1, 5), dtype=dtypes.Uint32, bank="GL"),
#     )

#     cache = _make_cache(cache_cls)

#     # Use DP search below to ensure that we don't produce an Impl with differing
#     # sub-Impls for the same Spec.
#     optimal = await dp.Search(top_k=1)(spec, limits, parent_summary=None, cache=cache)
#     hypothesis.assume(optimal)
#     optimal = optimal[0]
#     assert isinstance(optimal, impl.Impl)

#     results = [imp for imp, _ in (await cache.get(spec, limits)).contents]
#     assert len(results) == 1

#     hypothesis.note(pformat(results[0], show_cost=True))
#     hypothesis.note(pformat(optimal, show_cost=True))
#     # TODO: Don't use pformat/string comparison.
#     assert pformat(results[0], show_cost=False) == pformat(optimal, show_cost=False)


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
