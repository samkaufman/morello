import pytest

from morello import dtypes, impl, pruning, search_cache, specs
from morello.system_config import current_system, current_target

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

    cache = search_cache.ScheduleCache()
    cache.put(
        fast_schedule.spec,
        fast_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "GL": 0}),
    )
    assert set(cache) == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "GL": 0}),
        )
    }
    cache.put(
        slow_schedule.spec,
        slow_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 50, "GL": 0}),
    )
    assert set(cache) == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "GL": 0}),
        ),
        (
            slow_schedule.spec,
            slow_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 50, "GL": 0}),
        ),
    }
    cache.put(
        impossible_schedule.spec,
        impossible_wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 1, "GL": 0}),
    )
    assert set(cache) == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "GL": 0}),
        ),
        (
            slow_schedule.spec,
            slow_wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 50, "GL": 0}),
        ),
        (
            impossible_schedule.spec,
            impossible_wrapped_schedule,
            pruning.StandardMemoryLimits(({"RF": 1, "GL": 0})),
        ),
    }


@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
@pytest.mark.parametrize("contiguous", [True, False], ids=["contig", "noncontig"])
def test_cache_updates_when_none_result_put_with_higher_memory_cap(dtype, contiguous):
    t = specs.TensorSpec(
        (8, 8), dtype=dtype, contiguous=(4 if contiguous else 0), bank="RF"
    )
    spec = specs.Matmul(t, t, t, serial_only=False)
    wrapped_schedule = search_cache.CachedScheduleSet(tuple(), 1)

    cache = search_cache.ScheduleCache()
    cache.put(
        spec, wrapped_schedule, pruning.StandardMemoryLimits({"RF": 100, "GL": 0})
    )
    assert set(cache) == {
        (spec, wrapped_schedule, pruning.StandardMemoryLimits({"RF": 100, "GL": 0}))
    }
    cache.put(
        spec, wrapped_schedule, pruning.StandardMemoryLimits({"RF": 101, "GL": 0})
    )
    assert set(cache) == {
        (spec, wrapped_schedule, pruning.StandardMemoryLimits({"RF": 101, "GL": 0}))
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

    cache = search_cache.ScheduleCache()
    cache.put(
        schedule.spec,
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 100, "GL": 0}),
    )
    assert set(cache) == {
        (
            schedule.spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 100, "GL": 0}),
        )
    }
    cache.put(
        schedule.spec,
        wrapped_schedule,
        pruning.StandardMemoryLimits({"RF": 101, "GL": 0}),
    )
    assert set(cache) == {
        (
            schedule.spec,
            wrapped_schedule,
            pruning.StandardMemoryLimits({"RF": 101, "GL": 0}),
        )
    }
