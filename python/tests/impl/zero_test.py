import hypothesis
import pytest
from hypothesis import strategies as st

from morello import impl, specs

from .. import strategies

strategies.register_default_strategies()


@pytest.mark.skip(reason="flattening is disabled")
@hypothesis.given(st.from_type(specs.Zero))
def test_zero_flattening_returns_changed_subspec_if_can_flatten(zero_spec: specs.Zero):
    hole = impl.spec_to_hole(zero_spec)
    assert isinstance(hole, impl.ZeroHole)
    hypothesis.assume(hole.can_flatten)
    flattened = hole.flatten()
    assert isinstance(flattened, impl.SpecCast)
    assert flattened.inner.spec != zero_spec, f"{flattened.inner.spec} == {zero_spec}"


@pytest.mark.skip(reason="flattening is disabled")
@hypothesis.given(st.from_type(specs.Zero))
def test_zero_cannot_flatten_with_contiguous_under_two(zero_spec: specs.Zero):
    hypothesis.assume(zero_spec.destination.contiguous_abs < 2)
    hole = impl.spec_to_hole(zero_spec)
    assert isinstance(hole, impl.ZeroHole)
    assert not hole.can_flatten


@pytest.mark.skip(reason="flattening is disabled")
@hypothesis.given(st.from_type(specs.Zero))
def test_zero_flattening_always_decreases_operand_rank(zero_spec: specs.Zero):
    hole = impl.spec_to_hole(zero_spec)
    assert isinstance(hole, impl.ZeroHole)
    hypothesis.assume(hole.can_flatten)
    flattened = hole.flatten()
    assert isinstance(flattened, impl.SpecCast)
    assert len(flattened.inner.spec.operands[0].dim_sizes) < len(
        zero_spec.operands[0].dim_sizes
    ), f"Rank of {flattened.inner.spec} was not lower than original {zero_spec}"
