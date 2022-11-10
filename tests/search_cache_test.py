import hypothesis
import pytest
from hypothesis import strategies as st

from morello import dtypes, impl, pformat, pruning, search_cache, specs, utils
from morello.search import dp
from morello.system_config import current_system, current_target

from . import strategies

strategies.register_default_strategies()

# TODO: Add assertion that tests below only put scheduled Impls into the cache.


@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
def test_cache_common_scenario(dtype):
    target = current_target()

    lhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    output = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    fast_schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    fast_wrapped_schedule = search_cache.CachedScheduleSet(((fast_schedule, 10),), 1)

    lhs = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    output = target.tensor_spec((100, 100), dtype=dtype, bank="RF")
    slow_schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    slow_wrapped_schedule = search_cache.CachedScheduleSet(((slow_schedule, 50),), 1)

    lhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    rhs = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    output = target.tensor_spec((8, 8), dtype=dtype, bank="RF")
    impossible_schedule = impl.MatmulHole(
        specs.Matmul(lhs, rhs, output, serial_only=False)
    )
    impossible_wrapped_schedule = search_cache.CachedScheduleSet(tuple(), 1)

    cache = search_cache.InMemoryScheduleCache()
    cache.put(
        fast_schedule.spec,
        fast_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
    )
    assert set(cache) == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
        )
    }
    cache.put(
        slow_schedule.spec,
        slow_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 50, "VRF": 50, "L1": 50, "GL": 0}),
    )
    assert set(cache) == {
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
    cache.put(
        impossible_schedule.spec,
        impossible_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 1, "VRF": 1, "L1": 1, "GL": 0}),
    )
    assert set(cache) == {
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


@st.composite
def _arb_small_limits(draw):
    return pruning.StandardMemoryLimits(
        {
            bank: draw(st.integers(min_value=0, max_value=64))
            for bank in current_system().ordered_banks
        }
    )


# TODO: Add Compose Specs (incl. PipelineChildMemoryLimits)
# TODO: Test top_k greater than 1
@hypothesis.settings(deadline=240_000)
@hypothesis.given(strategies.small_atomic_specs_st, _arb_small_limits(), st.data())
def test_cache_get_returns_just_put_impls(spec, limits, data):
    cache = search_cache.InMemoryScheduleCache()

    optimal = dp.schedule_search(spec, limits, cache=cache)
    hypothesis.assume(optimal)
    optimal = optimal[0]
    assert isinstance(optimal, impl.Impl)

    # Check that we get the right choice for all intermediate limits.
    adjusted_limits_list = [
        data.draw(
            st.integers(min_value=optimal.peak_memory[b], max_value=limits.available[b])
        )
        for b in limits.available.raw_keys
    ]
    adjusted_limits = pruning.StandardMemoryLimits(
        utils.TinyMap(limits.available.raw_keys, tuple(adjusted_limits_list)),
        snap_down=False,
    )
    results = [imp for imp, _ in cache.get(spec, adjusted_limits).contents]
    assert len(results) == 1
    # TODO: Don't use string comparison.

    hypothesis.note(pformat(results[0], show_cost=False))
    hypothesis.note(pformat(optimal, show_cost=False))
    assert str(results[0]) == str(optimal)


# TODO: Add Compose Specs (incl. PipelineChildMemoryLimits)
# TODO: Test top_k greater than 1
@hypothesis.settings(deadline=60_000)
@hypothesis.given(strategies.small_atomic_specs_st, _arb_small_limits())
def test_cache_reputting_doesnt_increase_ize(spec, limits):
    cache = search_cache.InMemoryScheduleCache()

    optimal = dp.schedule_search(spec, limits, cache=cache)
    hypothesis.assume(optimal)
    optimal = optimal[0]
    assert isinstance(optimal, impl.Impl)

    initial_count = cache.count_impls()
    cache.put(spec, cache.get(spec, limits), limits)
    assert cache.count_impls() == initial_count


@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
@pytest.mark.parametrize("contiguous", [True, False], ids=["contig", "noncontig"])
def test_cache_updates_when_none_result_put_with_higher_memory_cap(dtype, contiguous):
    t = specs.TensorSpec(
        (8, 8), dtype=dtype, contiguous_abs=(4 if contiguous else 0), bank="RF"
    )
    spec = specs.Matmul(t, t, t, serial_only=False)
    wrapped_schedule = search_cache.CachedScheduleSet(tuple(), 1)

    cache = search_cache.InMemoryScheduleCache()
    cache.put(
        spec,
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "L1": 100, "GL": 0}),
    )
    assert set(cache) == {
        (
            spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "L1": 100, "GL": 0}),
        )
    }
    cache.put(
        spec,
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 101, "L1": 101, "GL": 0}),
    )
    assert set(cache) == {
        (
            spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 101, "L1": 101, "GL": 0}),
        )
    }


@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
def test_cache_updates_when_schedules_put_with_higher_memory_cap(dtype):
    target = current_target()
    db = current_system().default_bank
    lhs = target.tensor_spec((8, 8), dtype=dtype, bank=db)
    rhs = target.tensor_spec((8, 8), dtype=dtype, bank=db)
    output = target.tensor_spec((8, 8), dtype=dtype, bank=db)
    schedule = impl.MatmulHole(specs.Matmul(lhs, rhs, output, serial_only=False))
    wrapped_schedule = search_cache.CachedScheduleSet(((schedule, 10),), 1)

    cache = search_cache.InMemoryScheduleCache()
    cache.put(
        schedule.spec,
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
    )
    assert set(cache) == {
        (
            schedule.spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "VRF": 100, "L1": 100, "GL": 0}),
        )
    }
    cache.put(
        schedule.spec,
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 101, "VRF": 101, "L1": 101, "GL": 0}),
    )
    assert set(cache) == {
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
