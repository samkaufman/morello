from morello.ops import Matmul, Schedule
from morello import specs, tensor, ops, pruning, search_cache


def test_cache_common_scenario():
    lhs = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    rhs = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    output = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    fast_schedule = ops.Matmul(lhs, rhs, output, serial_only=False)
    fast_wrapped_schedule = search_cache.CachedSchedule(fast_schedule, cost=10)

    lhs = tensor.Tensor(spec=specs.TensorSpec((100, 100), level=0), name=None)
    rhs = tensor.Tensor(spec=specs.TensorSpec((100, 100), level=0), name=None)
    output = tensor.Tensor(spec=specs.TensorSpec((100, 100), level=0), name=None)
    slow_schedule = ops.Matmul(lhs, rhs, output, serial_only=False)
    slow_wrapped_schedule = search_cache.CachedSchedule(slow_schedule, cost=50)

    lhs = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    rhs = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    output = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    impossible_schedule = ops.Matmul(lhs, rhs, output, serial_only=False)

    cache = search_cache.ScheduleCache()
    cache.put(
        fast_schedule.spec,
        fast_wrapped_schedule,
        pruning.StandardMemoryLimits((100, 0)),
    )
    assert set(cache) == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits((100, 0)),
        )
    }
    cache.put(
        slow_schedule.spec, slow_wrapped_schedule, pruning.StandardMemoryLimits((50, 0))
    )
    assert set(cache) == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits((100, 0)),
        ),
        (
            slow_schedule.spec,
            slow_wrapped_schedule,
            pruning.StandardMemoryLimits((50, 0)),
        ),
    }
    cache.put(impossible_schedule.spec, None, pruning.StandardMemoryLimits((1, 0)))
    assert set(cache) == {
        (
            fast_schedule.spec,
            fast_wrapped_schedule,
            pruning.StandardMemoryLimits((100, 0)),
        ),
        (
            slow_schedule.spec,
            slow_wrapped_schedule,
            pruning.StandardMemoryLimits((50, 0)),
        ),
        (impossible_schedule.spec, None, pruning.StandardMemoryLimits((1, 0))),
    }


def test_cache_updates_when_none_result_put_with_higher_memory_cap():
    t = specs.TensorSpec((8, 8), level=0)
    spec = specs.Matmul(t, t, t, serial_only=False)
    wrapped_schedule = None

    cache = search_cache.ScheduleCache()
    cache.put(spec, wrapped_schedule, pruning.StandardMemoryLimits((100, 0)))
    assert set(cache) == {(spec, None, pruning.StandardMemoryLimits((100, 0)))}
    cache.put(spec, wrapped_schedule, pruning.StandardMemoryLimits((101, 0)))
    assert set(cache) == {(spec, None, pruning.StandardMemoryLimits((101, 0)))}


def test_cache_updates_when_schedules_put_with_higher_memory_cap():
    lhs = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    rhs = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    output = tensor.Tensor(spec=specs.TensorSpec((8, 8), level=0), name=None)
    schedule = ops.Matmul(lhs, rhs, output, serial_only=False)
    wrapped_schedule = search_cache.CachedSchedule(schedule, cost=10)

    cache = search_cache.ScheduleCache()
    cache.put(schedule.spec, wrapped_schedule, pruning.StandardMemoryLimits((100, 0)))
    assert set(cache) == {
        (schedule.spec, wrapped_schedule, pruning.StandardMemoryLimits((100, 0)))
    }
    cache.put(schedule.spec, wrapped_schedule, pruning.StandardMemoryLimits((101, 0)))
    assert set(cache) == {
        (schedule.spec, wrapped_schedule, pruning.StandardMemoryLimits((101, 0)))
    }
