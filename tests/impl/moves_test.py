from typing import Optional, Union

import hypothesis
import pytest
from hypothesis import strategies as st

from morello import impl, op_pprint, specs

from .. import strategies

strategies.register_default_strategies()


@st.composite
def _atomic_steps_and_tile_out_shape_st(draw):
    root_spec = draw(strategies.atomic_specs_st)
    tile_out_shape = draw(
        st.one_of(
            st.none(),
            st.tuples(
                *[
                    strategies.dim_st(max_size=min(128, t))
                    for t in root_spec.output.dim_sizes
                ]
            ),
        )
    )
    return root_spec, tile_out_shape


@hypothesis.settings(
    deadline=600, suppress_health_check=[hypothesis.HealthCheck.filter_too_much]
)
@hypothesis.given(_atomic_steps_and_tile_out_shape_st())
def test_moves_from_l1_to_rf_have_contiguous_output(tup_input):
    root_spec, tile_shape = tup_input

    hypothesis.assume(root_spec.operands[0].bank == "L1")
    imp = impl.spec_to_hole(root_spec)
    hypothesis.assume(hasattr(imp, "move"))
    if tile_shape:
        try:
            imp = imp.tile_out(tile_shape)
        except (specs.LayoutDoesntApplyError, specs.OversizedVectorError):
            hypothesis.assume(False)
    imp = imp.move(0, "RF")
    hypothesis.note("Impl:\n" + op_pprint.pformat(imp, show_cost=False))

    innermost_impl: Union[impl.Loop, impl.MoveLet] = (
        imp.inner.body if isinstance(imp, impl.Loop) else imp.body
    )
    innermost = innermost_impl.spec
    hypothesis.note(f"Innermost TensorSpec: {innermost.operands[0]}")
    assert innermost.operands[0].contiguous
    assert (
        innermost.operands[0].contiguous_abs
        == innermost.operands[0].layout.contiguous_full()
    )


@hypothesis.given(strategies.atomic_specs_st)
def test_moves_from_gl_to_rf_raise_exception(root_spec: specs.Spec):
    hypothesis.assume(root_spec.operands[0].bank == "GL")
    imp = impl.spec_to_hole(root_spec)
    hypothesis.assume(hasattr(imp, "move"))
    hypothesis.note("First TensorSpec: " + str(root_spec.operands[0]))
    with pytest.raises(ValueError):
        imp.move(0, "RF")
