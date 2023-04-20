import asyncio
import collections
from typing import Optional

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from morello import cost, dtypes, op_pprint, pruning, search, search_cache, specs, utils
from morello.system_config import current_target
from . import strategies

strategies.register_default_strategies()


@pytest.mark.skip(reason="No good way to constrain shapes (e.g. Reduce·Reduce·Reduce)")
@pytest.mark.asyncio
@given(st.from_type(specs.Spec))
async def test_search_passes_on_any_spec(s):
    await search.schedule_search(s)


class CountingCache(search_cache.ScheduleCache):
    def __init__(self):
        super().__init__()
        self.get_counts = collections.defaultdict(lambda: 0)
        self.put_counts = collections.defaultdict(lambda: 0)

    async def get(
        self, spec: specs.Spec, *args
    ) -> Optional[search_cache.CachedScheduleSet]:
        self.get_counts[spec] += 1
        return await super().get(spec, *args)

    async def put(
        self,
        spec: specs.Spec,
        schedule: Optional[search_cache.CachedScheduleSet],
        memory_limits: pruning.MemoryLimits,
    ) -> None:
        self.put_counts[spec] += 1
        return await super().put(spec, schedule, memory_limits)


@pytest.mark.skip("Need structural Impl equality for assert")
@pytest.mark.slow
@pytest.mark.asyncio
@given(st.from_type(specs.Spec))
async def test_nested_loop_pruning_doesnt_change_solutions(spec):
    token = impl.settings.prune_nested_parallel_loops.set(False)
    try:
        no_pruning_solution = await search.schedule_search(spec)
        if no_pruning_solution:
            hypothesis.note(
                f"no_pruning_solution:\n{op_pprint.pformat(no_pruning_solution)}"
            )
    finally:
        impl.settings.prune_nested_parallel_loops.reset(token)

    token = impl.settings.prune_nested_parallel_loops.set(True)
    try:
        with_pruning_solution = await search.schedule_search(spec)
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
@pytest.mark.asyncio
@hypothesis.settings(deadline=30 * 60 * 1000)
@given(st.integers(min_value=5), st.from_type(dtypes.Dtype))
@hypothesis.example(16)
async def test_compose_schedules_improve_as_memory_increases(cap_start, dtype):
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
                await search.schedule_search(
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


@pytest.mark.slow
@hypothesis.settings(deadline=400_000)
@hypothesis.example(
    specs.MatmulAccum(
        specs.TensorSpec((2, 2), dtype=dtypes.Uint8, bank="RF", contiguous_abs=2),
        specs.TensorSpec((2, 1), dtype=dtypes.Uint8, bank="L1", contiguous_abs=2),
        specs.TensorSpec((2, 1), dtype=dtypes.Uint8, bank="RF", contiguous_abs=2),
        serial_only=True,
    )
)
@hypothesis.given(strategies.tiny_atomic_specs_st)  # type: ignore
def test_dp_cost_matches_naive_search_cost(spec):
    mlims = pruning.StandardMemoryLimits()

    naive_result = search.naive_search(spec, mlims)
    dp_result = asyncio.run(search.schedule_search(spec, mlims))
    assert (naive_result is None) == (not dp_result)
    hypothesis.assume(naive_result is not None)
    assert cost.compute_cost(naive_result) == cost.compute_cost(
        dp_result[0]
    )  # type: ignore


@hypothesis.given(
    st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=4).map(tuple)
)
def test_next_limits_returns_nothing_at_peak_zero(limits):
    banks = tuple(map(str, range(len(limits))))
    limits = pruning.StandardMemoryLimits(utils.TinyMap(banks, limits))
    peak_map = utils.TinyMap(banks, (0,) * len(banks))
    assert [] == list(search.bottomup.next_limits(limits, peak_map))


def test_next_limits_dim1_a():
    limits = pruning.StandardMemoryLimits(utils.TinyMap(("A",), (8,)))
    peak = pruning.StandardMemoryLimits(utils.TinyMap(("A",), (4,)))
    expected = [pruning.StandardMemoryLimits(utils.TinyMap(("A",), (2,)))]
    assert expected == list(search.bottomup.next_limits(limits, peak.available))


def test_next_limits_dim1_b():
    limits = pruning.StandardMemoryLimits(utils.TinyMap(("A",), (4,)))
    peak = pruning.StandardMemoryLimits(utils.TinyMap(("A",), (4,)))
    expected = [pruning.StandardMemoryLimits(utils.TinyMap(("A",), (2,)))]
    assert expected == list(search.bottomup.next_limits(limits, peak.available))


@hypothesis.given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=4), st.integers(min_value=0, max_value=4)
        ),
        min_size=1,
        max_size=3,
    )
)
def test_next_limits_covers_space_disjointly(inp: list[tuple[int, int]]):
    initial_cap_bits, step = zip(*inp)

    initial_cap = tuple(2 ** (b - 1) if b else 0 for b in initial_cap_bits)
    hypothesis.note("Initial capacity (bits): " + str(initial_cap_bits))

    banks = tuple(map(str, range(len(initial_cap_bits))))
    covered = np.zeros([d + 1 for d in initial_cap_bits], dtype=bool)
    assert not covered.any()
    working_set = collections.deque(
        [pruning.StandardMemoryLimits(utils.TinyMap(banks, initial_cap))]
    )
    hypothesis.note("Initial limits: " + str(working_set[0]))
    while working_set:
        limits = working_set.popleft()
        limits_vals = limits.available.raw_values
        lower_peak_vals = _consume(limits_vals, step)
        lower_peak = utils.TinyMap(banks, lower_peak_vals)
        hypothesis.note(
            "Filling: "
            f"{tuple(slice(p, c + 1) for p, c in zip(lower_peak_vals, limits_vals))}"
        )
        covered[
            tuple(
                slice(p.bit_length(), c.bit_length() + 1)
                for p, c in zip(lower_peak_vals, limits_vals)
            )
        ] = True
        for new_caps in search.bottomup.next_limits(limits, lower_peak):
            hypothesis.note("Yielding new limits: " + str(new_caps))
            working_set.append(new_caps)
    assert covered.all()


def _consume(input: tuple[int, ...], step: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        utils.snap_availables_down(v - s) if v > s else 0 for v, s in zip(input, step)
    )
