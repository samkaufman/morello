import collections
import itertools
from typing import Iterable, Optional, NamedTuple, Tuple, cast

import hypothesis
from hypothesis import given, strategies as st
import pytest

from morello import specs, search, tensor, op_pprint, ops, system_config
from . import strategies

strategies.register_default_strategies()


@pytest.mark.skip(reason="No good way to constrain shapes (e.g. Reduce·Reduce·Reduce)")
@given(st.from_type(specs.Spec))
def test_search_passes_on_any_spec(spec: specs.Spec):
    inputs = tuple(tensor.Tensor(tensor_spec, name=None) for tensor_spec in spec.inputs)
    output = tensor.Tensor(spec.output, name="output")
    search.schedule_search(spec, inputs, output)


class CountingCache(search.ScheduleCache):
    def __init__(self):
        super().__init__()
        self.get_counts = collections.defaultdict(lambda: 0)
        self.put_counts = collections.defaultdict(lambda: 0)

    def get(self, spec: specs.Spec, *args) -> Optional[search.CachedSchedule]:
        self.get_counts[spec] += 1
        return super().get(spec, *args)

    def put(self, spec: specs.Spec, schedule: search.CachedSchedule) -> None:
        self.put_counts[spec] += 1
        return super().put(spec, schedule)


class MatmulConfig(NamedTuple):
    m: int
    k: int
    n: int
    levels: Tuple[int, int, int]
    layouts: Tuple[specs.Layout, specs.Layout, specs.Layout]


def _smaller_matmuls(spec: specs.Matmul) -> Iterable[MatmulConfig]:
    """Yields the sub-specs we expect search to schedule while scheduling a Matmul.

    Scheduling a Matmul should evaluate all Matmuls with smaller dimensions (of original
    size, size 1, or divisible by cache line size), lower memory levels for operands
    such that, left-to-right, levels are non-strictly decreasing in level.
    """
    orig_dims = spec.lhs.dim_sizes + (spec.rhs.dim_sizes[1],)
    operands = spec.inputs + (spec.output,)

    orig_config = MatmulConfig(
        spec.lhs.dim_sizes[0],
        spec.lhs.dim_sizes[1],
        spec.rhs.dim_sizes[1],
        levels=(spec.lhs.level, spec.rhs.level, spec.output.level),
        layouts=(spec.lhs.layout, spec.rhs.layout, spec.output.layout),
    )

    for m, k, n in itertools.product(*[ops.dim_range(d) for d in orig_dims]):
        numels = (m * k, k * n, m * n)
        for levels in itertools.product(*[range(op.level + 1) for op in operands]):
            levels = cast(Tuple[int, int, int], levels)
            for layouts in itertools.product(
                *[
                    list(specs.Layout) if n > 1 else [specs.Layout.ROW_MAJOR]
                    for n in numels
                ]
            ):
                layouts = cast(Tuple[specs.Layout, specs.Layout, specs.Layout], layouts)

                new_config = MatmulConfig(m=m, k=k, n=n, levels=levels, layouts=layouts)
                if new_config == orig_config:
                    continue
                # Exclude any Matmul with a different layout at a level slower than RF
                if any(
                    lvl > 0 and lay != orig_lay
                    for lvl, lay, orig_lay in zip(levels, layouts, orig_config.layouts)
                ):
                    continue
                # Exclude any Matmul with a scalar operand not in row-major
                if any(
                    n == 1 and l != specs.Layout.ROW_MAJOR
                    for n, l in zip(numels, layouts)
                ):
                    continue
                # Exclude any Matmul where the levels of operands aren't non-strictly
                # decreasing
                if any(a > b for a, b in zip(levels[:-1], levels[1:])):
                    continue
                # If the operand started at level 1 or higher, it will have the option
                # of moving directly into level 0 with the desired layout, so we don't
                # expect to explore both
                yield new_config


# TODO: Optionally drop cache line restriction
@pytest.mark.skip(reason="_smaller_matmuls fallen out of sync with action space")
@hypothesis.settings(deadline=10 * 1000)
@given(
    strategies.matmul_spec_st(
        max_dim_size=system_config.DEFAULT_SYSTEM_CONFIG.line_size + 1
    )
)
def test_matmul_search_schedules_every_smaller_op_exactly_once(op_spec):
    """Checks that search of a Matmul spec checks each "smaller" Matmul exactly once.

    This only works with pruning/symmetry-breaking on.
    """
    cache = CountingCache()
    inputs = tuple(tensor.Tensor(t, name=None) for t in op_spec.inputs)
    output = tensor.Tensor(op_spec.output, name="output")
    search.schedule_search(op_spec, inputs, output, cache=cache)

    # Check that given and all smaller Matmuls are queried for and set once
    smaller_specs = {op_spec}
    for matmul_config in _smaller_matmuls(op_spec):
        m, k, n, levels, layouts = matmul_config
        lhs = specs.TensorSpec((m, k), level=levels[0], layout=layouts[0])
        rhs = specs.TensorSpec((k, n), level=levels[1], layout=layouts[1])
        out = specs.TensorSpec((m, n), level=levels[2], layout=layouts[2])
        smaller_specs.add(specs.Matmul(lhs, rhs, out))

    assert set(cache.specs()) == smaller_specs

    for smaller in smaller_specs:
        puts = cache.put_counts[smaller]
        assert puts == 1, f"put count for {smaller} was {puts}"


# TODO: Generalize beyond just one Compose spec
# The minimum capacity is 5 because that's the number of cache lines required to
# move a single filter and its corresponding image window into registers
# (3*3 + 3*3 + 1 words).
@pytest.mark.slow
@hypothesis.settings(deadline=30 * 60 * 1000)
@given(st.integers(min_value=5))
@hypothesis.example(16)
def test_compose_schedules_improve_as_memory_increases(cap_start):
    results = []
    for cap in range(cap_start, cap_start + 2):
        img = tensor.Tensor(specs.TensorSpec((6, 6)), name="image")
        filters_a = tensor.Tensor(specs.TensorSpec((3, 3, 2)), name="filtersA")
        output = tensor.Tensor(specs.TensorSpec((4, 4)), name="output")

        original_capacity = system_config.DEFAULT_SYSTEM_CONFIG.level_configs[
            0
        ].capacity

        hypothesis.note("---------")
        try:
            system_config.DEFAULT_SYSTEM_CONFIG.level_configs[0].capacity = cap
            results.append(
                search.schedule_search(
                    specs.Compose(
                        (specs.ReduceSum, specs.Convolution),
                        (img.spec, filters_a.spec),
                        output.spec,
                        serial_only=False,
                    ),
                    inputs=(img, filters_a),
                    output=output,
                )
            )
        finally:
            system_config.DEFAULT_SYSTEM_CONFIG.level_configs[
                0
            ].capacity = original_capacity

        if len(results) >= 2:
            print(f"results: {results}")
            hypothesis.note("Left Schedule:\n" + op_pprint.pformat(results[-2]))
            hypothesis.note("Right Schedule:\n" + op_pprint.pformat(results[-1]))
            # TODO: Need to re-run costing.
            # assert results[-2][1] >= results[-1][1]
