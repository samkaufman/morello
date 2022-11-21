from typing import cast

import hypothesis
import pytest
from hypothesis import strategies as st

from morello import impl, pruning, utils, current_target
from morello.impl.base import spec_to_hole

from . import strategies


# TODO: Test the PipelineChildMemoryLimits case
# TODO: Include Composes in here so we can actually `peel`.
@hypothesis.given(
    strategies.arb_small_standard_memorylimits(),
    st.one_of(strategies.small_atomic_moveable_specs_st).map(spec_to_hole),
    st.data(),
)
def test_transition_always_snaps(base: pruning.StandardMemoryLimits, hole, data):
    target = current_target()

    assert utils.snap_availables_down(base.available) == base.available
    faster = sorted(target.system.faster_destination_banks(hole.spec.output.bank))
    hypothesis.assume(faster)
    new_bank = data.draw(st.sampled_from(faster))
    hypothesis.assume(new_bank)

    move_kws = {"bank": new_bank}
    if target.system.banks[new_bank].vector_rf:
        outer_shape = hole.spec.output.dim_sizes
        dtype = hole.spec.output.dtype
        vector_value_cnt: int = target.system.banks[new_bank].vector_bytes // dtype.size
        all_shapes = list(
            impl.utils.gen_vector_shapes(
                outer_shape=outer_shape,
                dtype=dtype,
                vector_bytes=vector_value_cnt,
            )
        )
        hypothesis.assume(len(all_shapes))
        move_kws["vector_shape"] = data.draw(st.sampled_from(all_shapes))

    if isinstance(hole, impl.ComposeHole) and data.draw(st.booleans()):
        imp = hole.peel(**move_kws)
    else:
        imp = hole.move_output(**move_kws)

    transitioned = base.transition(imp)
    hypothesis.assume(transitioned)  # not None or empty
    transitioned = cast(list[pruning.StandardMemoryLimits], transitioned)
    assert all(
        utils.snap_availables_down(c.available) == c.available for c in transitioned
    )
