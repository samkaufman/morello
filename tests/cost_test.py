import hypothesis
from hypothesis import given, strategies as st
import pytest

from morello import cost, ops, specs, system_config, tensor
from . import strategies

strategies.register_default_strategies()


@st.composite
def tile_st(draw):
    root_tile_name = draw(st.text(min_size=1))
    root_tensor = draw(st.from_type(tensor.Tensor))
    dim_sizes = tuple(
        draw(st.integers(min_value=1, max_value=d)) for d in root_tensor.dim_sizes
    )
    return tensor.Tile(root_tensor, dim_sizes, name=root_tile_name)


@st.composite
def _move_operand(
    draw, b: ops.Schedule, underlying_system: system_config.SimpleSystemConfig
):
    operand_idx = draw(st.integers(low=0, high=len(b.inputs) + 1))
    level_configs = underlying_system.level_configs
    level = draw(st.integers(min_value=0, max_value=len(level_configs) - 1))
    layout = draw(st.from_type(specs.Layout))
    b = draw(b)
    if operand_idx == len(b.inputs):
        return b.move_output(operand_idx, level=level, layout=layout)
    return b.move_input(operand_idx, level=level, layout=layout)


@st.composite
def _tile_schedule_op(draw, b: ops.Schedule):
    b = draw(b)
    height = draw(st.integers(min_value=1, max_value=b.innermost.output_height))
    width = draw(st.integers(min_value=1, max_value=b.innermost.output_width))
    return b.tile_out((height, width))


def full_schedule_st(
    underlying_system: system_config.SimpleSystemConfig = system_config.DEFAULT_SYSTEM_CONFIG,
):
    return st.recursive(
        base=st.from_type(ops.Matmul),
        extend=lambda b: st.one_of(
            [_move_operand(b, underlying_system), _tile_schedule_op(b)]
        ),
    ).filter(lambda ex: ex.is_scheduled)


def _make_tiled_matmul(maximize_register_use: bool, c: int, m: int) -> ops.Schedule:
    m_dim = 2 ** c
    tile_width = system_config.DEFAULT_SYSTEM_CONFIG.line_size * m
    hypothesis.assume(tile_width <= m_dim)
    hypothesis.note(f"Non-contig. {m_dim}x{m_dim} Matmul w/ {tile_width}-wide tile")

    a = tensor.Tensor(spec=specs.TensorSpec(dim_sizes=(m_dim, m_dim)), name=None)
    b = tensor.Tensor(spec=specs.TensorSpec(dim_sizes=(m_dim, m_dim)), name=None)
    o = tensor.Tensor(spec=specs.TensorSpec(dim_sizes=(m_dim, m_dim)), name=None)

    schedule = ops.Matmul(a, b, o).tile_out((m_dim, tile_width))
    # if make_contiguous:
    #    schedule = schedule.move_input(1, 0, matrix.Layout.COL_MAJOR)
    #    n = tile_width
    if maximize_register_use:
        # We've already broken columns into tile_width tiles, so lets just move panels
        # into registers
        schedule = (
            schedule.tile_out((1, system_config.DEFAULT_SYSTEM_CONFIG.line_size))
            .move_input(0, level=0)
            .move_input(1, level=0)
        )
    return schedule.complete()


@given(c=st.integers(min_value=2, max_value=4), m=st.integers(min_value=1, max_value=4))
def test_contiguous_copy_lowers_matmul_cost(c: int, m: int):
    contiguous_matmul = _make_tiled_matmul(True, c, m)
    noncontiguous_matmul = _make_tiled_matmul(False, c, m)
    fast_cost = cost.analytical_cost(contiguous_matmul)
    slow_cost = cost.analytical_cost(noncontiguous_matmul)
    hypothesis.note(f"Speeds fast/slow are {fast_cost} and {slow_cost}")
    hypothesis.note(f"Slow, non-contiguous model is {str(noncontiguous_matmul)}")
    hypothesis.note(f"Fast, contig. model is {str(contiguous_matmul)}")
    assert fast_cost < slow_cost


@hypothesis.settings(deadline=10 * 1000)
@given(matmul_spec=strategies.matmul_spec_st())
def test_trivial_tilings_are_same_cost_as_untiled_matmul(matmul_spec: specs.Matmul):
    lhs = ops.Tensor(matmul_spec.lhs, name=None)
    rhs = ops.Tensor(matmul_spec.rhs, name=None)
    out = ops.Tensor(matmul_spec.output, name=None)

    m, k, n = out.dim_sizes[0], lhs.dim_sizes[1], out.dim_sizes[1]
    trivial_tiled_schedule = (
        ops.Matmul(lhs, rhs, out, matmul_spec.serial_only)
        .tile_out((m, n))
        .split(k)
        .complete()
    )

    trivial_untiled_schedule = ops.Matmul(lhs, rhs, out, serial_only=False).complete()
    assert cost.analytical_cost(trivial_tiled_schedule) == cost.analytical_cost(
        trivial_untiled_schedule
    )


@given(
    st.integers(min_value=0, max_value=1),
    st.integers(min_value=2, max_value=64),
    st.integers(min_value=0, max_value=1),
    st.from_type(specs.Layout),
)
def test_cost_is_invariant_to_panel_layouts(dim_idx, elements, level, dest_layout):
    dim_sizes = tuple(elements if dim_idx == i else 1 for i in range(2))
    left = tensor.Tensor(
        spec=specs.TensorSpec(dim_sizes, layout=specs.Layout.ROW_MAJOR, level=level),
        name=None,
    )
    right = tensor.Tensor(
        spec=specs.TensorSpec(dim_sizes, layout=specs.Layout.COL_MAJOR, level=level),
        name=None,
    )
    assert cost.move_cost(left, dest_layout) == cost.move_cost(right, dest_layout)


if __name__ == "__main__":
    pytest.main()
