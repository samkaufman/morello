from typing import Optional

import hypothesis
import pytest
from hypothesis import given
from hypothesis import strategies as st

import morello.impl
from morello import cost, dtypes, specs, system_config, tensor
from morello.impl import Impl
from morello.system_config import cpu

from . import strategies

strategies.register_default_strategies()


@st.composite
def tile_st(draw):
    root_tile_name = draw(st.text(min_size=1))
    root_tensor = draw(st.from_type(tensor.Tensor))
    dim_sizes = tuple(
        draw(st.integers(min_value=1, max_value=d)) for d in root_tensor.dim_sizes
    )
    return tensor.Tile(dim_sizes, name=root_tile_name, origin=root_tensor)


@st.composite
def _move_operand(draw, b: Impl, underlying_system: system_config.SystemDescription):
    operand_idx = draw(st.integers(low=0, high=len(b.inputs) + 1))
    bank = draw(st.sampled_from(sorted(underlying_system.banks)))
    layout = draw(st.from_type(specs.Layout))
    b = draw(b)
    if operand_idx == len(b.inputs):
        return b.move_output(operand_idx, bank=bank, layout=layout)
    return b.move_input(operand_idx, bank=bank, layout=layout)


@st.composite
def _tile_schedule_op(draw, b: Impl):
    b = draw(b)
    height = draw(st.integers(min_value=1, max_value=_get_innermost(b).output_height))
    width = draw(st.integers(min_value=1, max_value=_get_innermost(b).output_width))
    # TODO: This can just draw from the Fpf, once tile_out is an Fpf
    return b.tile_out((height, width))


def _get_innermost(impl: Impl) -> Impl:
    cur = impl
    while len(cur.children):
        assert hasattr(cur, "inner")
        cur = cur.inner
    return cur


def full_schedule_st(
    underlying_system: Optional[system_config.SystemDescription] = None,
):
    if not underlying_system:
        underlying_system = system_config.current_system()
    return st.recursive(
        base=st.from_type(morello.impl.Matmul),
        extend=lambda b: st.one_of(
            [_move_operand(b, underlying_system), _tile_schedule_op(b)]
        ),
    ).filter(lambda ex: ex.is_scheduled)


def _make_tiled_matmul(
    maximize_register_use: bool, c: int, m: int, dtype: dtypes.Dtype
) -> Impl:
    target = system_config.current_target()

    m_dim = 2 ** c
    tile_width = target.system.line_size * m
    hypothesis.assume(tile_width <= m_dim)
    hypothesis.note(f"Non-contig. {m_dim}x{m_dim} Matmul w/ {tile_width}-wide tile")

    a = target.tensor(spec=target.tensor_spec((m_dim, m_dim), dtype), name=None)
    b = target.tensor(spec=target.tensor_spec((m_dim, m_dim), dtype), name=None)
    o = target.tensor(spec=target.tensor_spec((m_dim, m_dim), dtype), name=None)

    schedule = morello.impl.MatmulHole(a, b, o).tile_out((m_dim, tile_width))
    # if make_contiguous:
    #    schedule = schedule.move_input(1, 0, matrix.Layout.COL_MAJOR)
    #    n = tile_width
    if maximize_register_use:
        # We've already broken columns into tile_width tiles, so lets just move panels
        # into registers
        schedule = (
            schedule.tile_out((1, target.system.line_size))
            .move_input(0, level=0)
            .move_input(1, level=0)
        )
    return schedule.complete()


@pytest.mark.skip("Skipping because assumptions cannot be satisfied")
@given(
    st.integers(min_value=2, max_value=4),
    st.integers(min_value=1, max_value=4),
    st.from_type(dtypes.Dtype),
)
def test_contiguous_copy_lowers_matmul_cost(c: int, m: int, dtype: dtypes.Dtype):
    contiguous_matmul = _make_tiled_matmul(True, c, m, dtype)
    noncontiguous_matmul = _make_tiled_matmul(False, c, m, dtype)
    fast_cost = cost.compute_cost(contiguous_matmul)
    slow_cost = cost.compute_cost(noncontiguous_matmul)
    hypothesis.note(f"Speeds fast/slow are {fast_cost} and {slow_cost}")
    hypothesis.note(f"Slow, non-contiguous model is {str(noncontiguous_matmul)}")
    hypothesis.note(f"Fast, contig. model is {str(contiguous_matmul)}")
    assert fast_cost < slow_cost


@hypothesis.settings(deadline=10 * 1000)
@given(matmul_spec=strategies.matmul_spec_st())
def test_trivial_tilings_are_same_cost_as_untiled_matmul(matmul_spec):
    target = system_config.current_target()

    lhs = target.tensor(matmul_spec.lhs, name=None)
    rhs = target.tensor(matmul_spec.rhs, name=None)
    out = target.tensor(matmul_spec.output, name=None)

    m, k, n = out.dim_sizes[0], lhs.dim_sizes[1], out.dim_sizes[1]
    trivial_tiled_schedule = (
        morello.impl.MatmulHole(lhs, rhs, out, matmul_spec.serial_only)
        .tile_out((m, n))
        .split(k)
        .complete()
    )

    trivial_untiled_schedule = morello.impl.MatmulHole(
        lhs, rhs, out, serial_only=False
    ).complete()
    assert cost.compute_cost(trivial_tiled_schedule) == cost.compute_cost(
        trivial_untiled_schedule
    )


@given(
    st.sampled_from(sorted(cpu.CpuTarget().system.banks)),
    st.from_type(dtypes.Dtype),
    st.integers(min_value=0, max_value=1),
    st.integers(min_value=2, max_value=64),
    st.from_type(specs.Layout),
    st.booleans(),
)
def test_cost_is_invariant_to_panel_layouts_cpu(
    bank, dtype, dim_idx, elements, dest_layout, prefetching
):
    # TODO: Add a version of this test for Hexagon targets
    target = cpu.CpuTarget()
    with system_config.with_target(target):
        dim_sizes = tuple(elements if dim_idx == i else 1 for i in range(2))
        left = target.tensor(
            spec=target.tensor_spec(
                dim_sizes, dtype=dtype, layout=specs.ROW_MAJOR, bank=bank
            ),
            name=None,
        )
        right = target.tensor(
            spec=target.tensor_spec(
                dim_sizes, dtype=dtype, layout=specs.COL_MAJOR, bank=bank
            ),
            name=None,
        )

        left_result = cost.move_cost(left, dest_layout, prefetching)
        right_result = cost.move_cost(right, dest_layout, prefetching)
        assert left_result == right_result


if __name__ == "__main__":
    pytest.main()
