import pytest

import morello.impl.matmuls
from morello import dtypes, pruning, search_cache, specs
from morello.system_config import current_system, current_target


# TODO: Add assertion that tests below only put scheduled Impls into the cache.


@pytest.mark.parametrize(
    "dtype", [(dtypes.Uint8,), (dtypes.Uint32,)], ids=["u8", "u32"]
)
def test_cache_common_scenario(dtype):
    target = current_target()

    lhs = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank="RF"), name=None
    )
    rhs = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank="RF"), name=None
    )
    output = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank="RF"), name=None
    )
    fast_schedule = morello.impl.matmuls.MatmulHole(lhs, rhs, output, serial_only=False)
    fast_wrapped_schedule = search_cache.CachedSchedule(fast_schedule, cost=10)

    lhs = target.tensor(
        spec=target.tensor_spec((100, 100), dtype=dtype, bank="RF"), name=None
    )
    rhs = target.tensor(
        spec=target.tensor_spec((100, 100), dtype=dtype, bank="RF"), name=None
    )
    output = target.tensor(
        spec=target.tensor_spec((100, 100), dtype=dtype, bank="RF"), name=None
    )
    slow_schedule = morello.impl.matmuls.MatmulHole(lhs, rhs, output, serial_only=False)
    slow_wrapped_schedule = search_cache.CachedSchedule(slow_schedule, cost=50)

    lhs = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank="RF"), name=None
    )
    rhs = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank="RF"), name=None
    )
    output = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank="RF"), name=None
    )
    impossible_schedule = morello.impl.matmuls.MatmulHole(
        lhs, rhs, output, serial_only=False
    )

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
        impossible_schedule.spec, None, pruning.StandardMemoryLimits({"RF": 1, "GL": 0})
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
            None,
            pruning.StandardMemoryLimits(({"RF": 1, "GL": 0})),
        ),
    }


@pytest.mark.parametrize(
    "dtype", [(dtypes.Uint8,), (dtypes.Uint32,)], ids=["u8", "u32"]
)
def test_cache_updates_when_none_result_put_with_higher_memory_cap(dtype):
    t = specs.TensorSpec((8, 8), dtype=dtype, bank="RF")
    spec = specs.Matmul(t, t, t, serial_only=False)
    wrapped_schedule = None

    cache = search_cache.ScheduleCache()
    cache.put(
        spec, wrapped_schedule, pruning.StandardMemoryLimits({"RF": 100, "GL": 0})
    )
    assert set(cache) == {
        (spec, None, pruning.StandardMemoryLimits({"RF": 100, "GL": 0}))
    }
    cache.put(
        spec, wrapped_schedule, pruning.StandardMemoryLimits({"RF": 101, "GL": 0})
    )
    assert set(cache) == {
        (spec, None, pruning.StandardMemoryLimits({"RF": 101, "GL": 0}))
    }


@pytest.mark.parametrize(
    "dtype", [(dtypes.Uint8,), (dtypes.Uint32,)], ids=["u8", "u32"]
)
def test_cache_updates_when_schedules_put_with_higher_memory_cap(dtype):
    target = current_target()
    db = current_system().default_bank
    lhs = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank=db), name=None
    )
    rhs = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank=db), name=None
    )
    output = target.tensor(
        spec=target.tensor_spec((8, 8), dtype=dtype, bank=db), name=None
    )
    schedule = morello.impl.matmuls.MatmulHole(lhs, rhs, output, serial_only=False)
    wrapped_schedule = search_cache.CachedSchedule(schedule, cost=10)

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
