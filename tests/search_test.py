import collections
from typing import Optional

import hypothesis
import pytest
from hypothesis import given
from hypothesis import strategies as st

from morello import (
    dtypes,
    op_pprint,
    pruning,
    search,
    search_cache,
    specs,
    system_config,
)
from morello.system_config import current_system, current_target

from . import strategies

strategies.register_default_strategies()


@pytest.mark.skip(reason="No good way to constrain shapes (e.g. Reduce·Reduce·Reduce)")
@given(st.from_type(specs.Spec))
def test_search_passes_on_any_spec(spec: specs.Spec):
    target = current_target()
    inputs = tuple(target.tensor(tensor_spec, name=None) for tensor_spec in spec.inputs)
    output = target.tensor(spec.output, name="output")
    search.schedule_search(spec, inputs, output)


class CountingCache(search_cache.InMemoryScheduleCache):
    def __init__(self):
        super().__init__()
        self.get_counts = collections.defaultdict(lambda: 0)
        self.put_counts = collections.defaultdict(lambda: 0)

    def get(self, spec: specs.Spec, *args) -> Optional[search_cache.CachedScheduleSet]:
        self.get_counts[spec] += 1
        return super().get(spec, *args)

    def put(
        self,
        spec: specs.Spec,
        schedule: Optional[search_cache.CachedScheduleSet],
        memory_limits: pruning.MemoryLimits,
    ) -> None:
        self.put_counts[spec] += 1
        return super().put(spec, schedule, memory_limits)


@pytest.mark.skip("Need structural Impl equality for assert")
@pytest.mark.slow
@given(st.from_type(specs.Spec))
def test_nested_loop_pruning_doesnt_change_solutions(spec):
    token = impl.settings.prune_nested_parallel_loops.set(False)
    try:
        no_pruning_solution = search.schedule_search(spec)
        if no_pruning_solution:
            hypothesis.note(
                f"no_pruning_solution:\n{op_pprint.pformat(no_pruning_solution)}"
            )
    finally:
        impl.settings.prune_nested_parallel_loops.reset(token)

    token = impl.settings.prune_nested_parallel_loops.set(True)
    try:
        with_pruning_solution = search.schedule_search(spec)
        if with_pruning_solution:
            hypothesis.note(
                f"with_pruning_solution:\n{op_pprint.pformat(with_pruning_solution)}"
            )
    finally:
        impl.settings.prune_nested_parallel_loops.reset(token)

    assert no_pruning_solution == with_pruning_solution


# TODO: Generalize beyond just one Compose spec
# The minimum capacity is 5 because that's the number of cache lines required to
# move a single filter and its corresponding image window into registers
# (3*3 + 3*3 + 1 words).
@pytest.mark.skip("Skipping due to performance issues")
@pytest.mark.slow
@hypothesis.settings(deadline=30 * 60 * 1000)
@given(st.integers(min_value=5), st.from_type(dtypes.Dtype))
@hypothesis.example(16)
def test_compose_schedules_improve_as_memory_increases(cap_start, dtype):
    target = current_target()
    system = target.system

    results = []
    for cap in range(cap_start, cap_start + 2):
        img = target.tensor(target.tensor_spec((6, 6), dtype=dtype), name="image")
        filters_a = target.tensor(
            target.tensor_spec((3, 3, 2), dtype=dtype), name="filtersA"
        )
        output = target.tensor(target.tensor_spec((4, 4), dtype=dtype), name="output")

        original_capacity = system.level_configs[0].capacity

        hypothesis.note("---------")
        try:
            system.level_configs[0].capacity = cap
            results.append(
                search.schedule_search(
                    specs.Compose(
                        (specs.ReduceSum, specs.Convolution),
                        (img.spec, filters_a.spec),
                        output.spec,
                        intermediate_dtypes=(dtype,),
                        serial_only=False,
                    ),
                    inputs=(img, filters_a),
                    output=output,
                )
            )
        finally:
            system.level_configs[0].capacity = original_capacity

        if len(results) >= 2:
            print(f"results: {results}")
            hypothesis.note("Left Impl:\n" + op_pprint.pformat(results[-2]))
            hypothesis.note("Right Impl:\n" + op_pprint.pformat(results[-1]))
            # TODO: Need to re-run costing.
            # assert results[-2][1] >= results[-1][1]
