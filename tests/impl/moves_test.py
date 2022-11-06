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


@hypothesis.settings(deadline=600)
@hypothesis.given(_atomic_steps_and_tile_out_shape_st())
def test_moves_from_l1_to_rf_have_contiguous_output(tup_input):
    root_spec, tile_shape = tup_input

    hypothesis.assume(root_spec.output.bank == "L1")
    imp = impl.spec_to_hole(root_spec)
    if tile_shape:
        try:
            imp = imp.tile_out(tile_shape)
        except (specs.LayoutDoesntApplyError, specs.OversizedVectorError):
            hypothesis.assume(False)
    imp = imp.move_output("RF")
    hypothesis.note("Impl:\n" + op_pprint.pformat(imp, show_cost=False))

    innermost_impl: Union[impl.Loop, impl.MoveLet] = (
        imp.inner.body if isinstance(imp, impl.Loop) else imp.body
    )
    innermost = innermost_impl.spec
    hypothesis.note(f"Innermost TensorSpec: {innermost.output}")
    assert innermost.output.contiguous
    assert innermost.output.contiguous_abs == innermost.output.layout.contiguous_top()


@hypothesis.given(strategies.atomic_specs_st)
def test_moves_from_gl_to_rf_raise_exception(root_spec: specs.Spec):
    hypothesis.assume(root_spec.output.bank == "GL")
    imp = impl.spec_to_hole(root_spec)
    hypothesis.note("Output TensorSpec: " + str(root_spec.output))
    with pytest.raises(ValueError):
        imp.move_output("RF")
