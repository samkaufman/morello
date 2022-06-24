from typing import Iterable, Optional, cast

import hypothesis
import pytest
from hypothesis import strategies as st

import morello.impl.utils
from morello import dtypes, impl, op_pprint, specs, tensor
from morello.impl import Impl


def test_dim_range():
    # Common cases
    for mode in impl.TileSizeMode:
        token = impl.tile_size_mode.set(mode)
        try:
            assert list(morello.impl.utils.dim_range(0)) == []
            assert list(morello.impl.utils.dim_range(0, include_end=False)) == []
            assert list(morello.impl.utils.dim_range(1, include_end=False)) == []
            assert list(morello.impl.utils.dim_range(2, include_end=False)) == [1]
        finally:
            impl.tile_size_mode.reset(token)

    token = impl.tile_size_mode.set(impl.TileSizeMode.POWERS_OF_TWO)
    try:
        assert list(morello.impl.utils.dim_range(1)) == [1]
        assert list(morello.impl.utils.dim_range(2)) == [1, 2]
        assert list(morello.impl.utils.dim_range(3)) == [1, 2, 3]
        assert list(morello.impl.utils.dim_range(4)) == [1, 2, 4]
        assert list(morello.impl.utils.dim_range(5)) == [1, 2, 4, 5]
        assert list(morello.impl.utils.dim_range(3, include_end=False)) == [1, 2]
        assert list(morello.impl.utils.dim_range(4, include_end=False)) == [1, 2]
        assert list(morello.impl.utils.dim_range(5, include_end=False)) == [1, 2, 4]
    finally:
        impl.tile_size_mode.reset(token)

    token = impl.tile_size_mode.set(impl.TileSizeMode.CACHE_LINE_MULTIPLES)
    try:
        assert list(morello.impl.utils.dim_range(1)) == [1]
        assert list(morello.impl.utils.dim_range(2)) == [1, 2]
        assert list(morello.impl.utils.dim_range(3)) == [1, 3]
        assert list(morello.impl.utils.dim_range(3, include_end=False)) == [1]
    finally:
        impl.tile_size_mode.reset(token)

    token = impl.tile_size_mode.set(impl.TileSizeMode.ALL)
    try:
        assert list(morello.impl.utils.dim_range(1)) == [1]
        assert list(morello.impl.utils.dim_range(2)) == [1, 2]
        assert list(morello.impl.utils.dim_range(3)) == [1, 2, 3]
        assert list(morello.impl.utils.dim_range(3, include_end=False)) == [1, 2]
    finally:
        impl.tile_size_mode.reset(token)


def test_gen_vector_shapes_1():
    assert list(
        morello.impl.utils.gen_vector_shapes([4, 4], dtypes.Uint8, elements=4 * 4)
    ) == [(4, 4)]


def test_gen_vector_shapes_2():
    assert (
        list(morello.impl.utils.gen_vector_shapes([8], dtypes.Uint8, elements=16)) == []
    )


def test_gen_vector_shapes_3():
    assert list(
        morello.impl.utils.gen_vector_shapes([16, 2], dtypes.Uint8, elements=16)
    ) == [
        (8, 2),
        (16, 1),
    ]


def test_gen_vector_shapes_4():
    assert list(
        morello.impl.utils.gen_vector_shapes([16], dtypes.Uint8, elements=16)
    ) == [(16,)]


@pytest.mark.skip
@pytest.mark.parametrize(
    "intermed_shapes,dtype,op_mems,expected_peaks,expected_additionals",
    [
        (
            [(10,)],
            dtypes.Uint8,
            [{"RF": 20, "GL": 0}, {"RF": 5, "GL": 0}],
            {"RF": 30, "GL": 0},
            [{"RF": 10, "GL": 0}, {"RF": 10, "GL": 0}],
        ),
        (
            [(10,), (90,)],
            dtypes.Uint32,
            [{"RF": 20, "GL": 0}, {"RF": 10, "GL": 10}, {"RF": 10, "GL": 0}],
            {"RF": 410, "GL": 10},
            [{"RF": 40, "GL": 0}, {"RF": 400, "GL": 0}, {"RF": 360, "GL": 0}],
        ),
    ],
)
def test_pipeline_peak_and_additional_memory(
    intermed_shapes, dtype, op_mems, expected_peaks, expected_additionals
):
    class SubImplStub:
        """A stub for an impl.Impl with some arbitrary output and peak mem.

        This doesn't fully implement the impl.Impl abstract class, so we'll
        have to `cast` at construction to satisfy the type checker.
        """

        def __init__(self, mem: dict[str, int], output) -> None:
            super().__init__()
            self._mem = mem
            self._output = output

        @property
        def output(self):
            return self._output

        @property
        def spec(self):
            class SubImplStubSpec:
                @property
                def serial_only(self):
                    return False

            return SubImplStubSpec()

        @property
        def peak_memory(self):
            return self._mem

    assert len(intermed_shapes) + 1 == len(op_mems)

    intermediates: list[Optional[tensor.Tensor]] = [
        tensor.Tensor(spec=specs.TensorSpec(shp, dtype, True, bank="RF"), name=None)
        for shp in intermed_shapes
    ]
    intermediates.append(None)

    pipeline = morello.impl.compose.Pipeline(
        tuple(
            cast(Impl, SubImplStub(m, intermed))
            for m, intermed in zip(op_mems, intermediates)
        )
    )

    assert pipeline.peak_memory == expected_peaks
    assert pipeline.additional_memories == expected_additionals


@pytest.mark.parametrize(
    "outer_batch,inner_batch,img_size,filter_size,channels,patch,out_size,tile_size,steps,dtype",
    [
        (1, 1, 3, 3, 1, 3, 1, 1, 1, dtypes.Uint32),
        (3, 1, 3, 1, 1, 1, 3, 1, 9 * 3, dtypes.Uint32),
        (3, 1, 10, 3, 1, 5, 8, 3, 9 * 3, dtypes.Uint8),
        (1, 1, 5, 3, 1, 4, 3, 2, 4, dtypes.Uint8),
    ],
)
def test_convolution_steps(
    outer_batch,
    inner_batch,
    img_size,
    filter_size,
    channels,
    patch,
    out_size,
    tile_size,
    steps,
    dtype,
):
    print(f"Testing square {tile_size}")
    filter_cnt = 4
    img = specs.TensorSpec((outer_batch, channels, img_size, img_size), dtype, True)
    filters = specs.TensorSpec(
        (filter_cnt, channels, filter_size, filter_size), dtype, True
    )
    out=specs.TensorSpec(
        (outer_batch, filter_cnt, out_size, out_size), dtype, True
    )
    conv = impl.DirectConv(
      specs.Convolution(img, filters, out, serial_only=False)
    )
    print("On a conv: " + str(conv))
    print(f"Calling conv.tile_out({(inner_batch, filter_cnt, tile_size, tile_size)})")
    loop = conv.tile_out((inner_batch, filter_cnt, tile_size, tile_size))
    print(f"Loop:\n{op_pprint.pformat(loop, show_utilization=False, show_cost=False)}")
    if steps == 1:
        assert isinstance(loop, impl.DirectConv)
    else:
        assert loop.steps == steps


def test_evenly_divisible_matmul_tiling():
    lhs = specs.TensorSpec((4, 4), dtypes.Uint32, True)
    rhs = specs.TensorSpec((4, 4), dtypes.Uint32, True)
    out = specs.TensorSpec((4, 4), dtypes.Uint32, True)
    schedule = impl.MatmulHole(specs.Matmul(
        lhs, rhs, out, serial_only=False
    )).tile_out((2, 2))
    assert schedule.spec.output.dim_sizes == (4, 4)
    assert isinstance(schedule.inner, morello.impl.matmuls.MatmulBase)
    assert schedule.inner.spec.output.dim_sizes == (2, 2)
    assert schedule.inner.spec.inputs[0].dim_sizes[1] == 4


@hypothesis.settings(deadline=10 * 1000)
@hypothesis.given(
    st.integers(min_value=1, max_value=11),
    st.integers(min_value=1, max_value=11),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=3),
    st.from_type(dtypes.Dtype),
)
def test_nested_convs_outputs_constant(
    h, w, a, b, fa, fb, th1, tw1, th2, tw2, fi, dtype
):
    # TODO: Update to try different batches counts
    image = tensor.Tensor(
        specs.TensorSpec((1, 1, h + a + fa, w + b + fb), dtype, True), name=None
    )
    # TODO: Update to try different channels counts
    filters = tensor.Tensor(specs.TensorSpec((fi, 1, h, w), dtype, True), name=None)
    expected_output_height = 1 + a + fa
    expected_output_width = 1 + b + fb
    output = tensor.Tensor(
        specs.TensorSpec(
            (1, fi, expected_output_height, expected_output_width), dtype, True
        ),
        name=None,
    )
    schedule = morello.impl.directconv.DirectConv(
        specs.Convolution(
            image.spec, filters.spec, output.spec, serial_only=False
        )
    )
    assert schedule.spec.output.dim_sizes[:2] == (1, fi)
    assert schedule.spec.output.dim_sizes[2] == expected_output_height
    assert schedule.spec.output.dim_sizes[3] == expected_output_width

    hypothesis.assume(th1 <= 1 + image.dim_sizes[2] - filters.dim_sizes[2])
    hypothesis.assume(tw1 <= 1 + image.dim_sizes[3] - filters.dim_sizes[3])
    tiled_schedule_a = schedule.tile_out((1, fi, th1, tw1))
    assert tiled_schedule_a.spec.output.dim_sizes[:2] == (1, fi)
    assert tiled_schedule_a.spec.output.dim_sizes[2] == expected_output_height
    assert tiled_schedule_a.spec.output.dim_sizes[3] == expected_output_width

    hypothesis.assume(th2 <= th1)
    hypothesis.assume(tw2 <= tw1)
    tiled_schedule_b = schedule.tile_out((1, fi, th2, tw2))
    assert tiled_schedule_b.spec.output.dim_sizes[:2] == (1, fi)
    assert tiled_schedule_b.spec.output.dim_sizes[2] == expected_output_height
    assert tiled_schedule_b.spec.output.dim_sizes[3] == expected_output_width


@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
def test_tile_compose_hole_out(dtype):
    img = tensor.Tensor(specs.TensorSpec((1, 1, 8, 8), dtype, True), name="image")
    filters_a = tensor.Tensor(
        specs.TensorSpec((4, 1, 3, 3), dtype, True), name="filtersA"
    )
    filters_b = tensor.Tensor(
        specs.TensorSpec((4, 4, 3, 3), dtype, True), name="filtersB"
    )
    output = tensor.Tensor(specs.TensorSpec((1, 4, 4, 4), dtype, True), name="output")

    compose_spec = specs.Compose(
        (specs.Convolution, specs.Convolution),
        (filters_b.spec, img.spec, filters_a.spec),
        output.spec,
        intermediate_dtypes=(dtype,),
        serial_only=False,
    )

    compose_hole = impl.ComposeHole(compose_spec)
    tiled_compose = compose_hole.tile_out((1, 4, 2, 2))
    assert isinstance(tiled_compose, impl.Loop)
    assert tiled_compose.spec == compose_spec
    assert isinstance(tiled_compose.inner, impl.ComposeHole)
    assert tiled_compose.inner.spec.output.dim_sizes == (1, 4, 2, 2)
    # TODO: Add checks for intermediate shape correctness


def _walk_actions(
    op: Impl, depth: int = 1, parents=None, parent_summary=None
) -> Iterable[Impl]:
    if depth == 0:
        return
    if parents is None:
        parents = []
    for act in op.actions(parent_summary=parent_summary):
        new_tree = act()
        for child in new_tree.children:
            yield child
            yield from _walk_actions(
                child,
                depth=depth - 1,
                parents=list(parents) + [op],
                parent_summary=impl.ParentSummary.update(parent_summary, new_tree),
            )


# TODO: Extend to all Specs, not just a single ComposeHole
@pytest.mark.parametrize("dtype", [dtypes.Uint8, dtypes.Uint32], ids=["u8", "u32"])
def test_composehole_actions_change_spec(dtype):
    # This doesn't test for cycles introduced by sequences of more than one
    # action, but it makes sure that at least every individual step changes the
    # spec.
    img = tensor.Tensor(specs.TensorSpec((1, 1, 8, 8), dtype, True), name="image")
    filters_a = tensor.Tensor(
        specs.TensorSpec((10, 1, 3, 3), dtype, True), name="filtersA"
    )
    filters_b = tensor.Tensor(
        specs.TensorSpec((7, 10, 3, 3), dtype, True), name="filtersB"
    )
    output = tensor.Tensor(specs.TensorSpec((1, 7, 4, 4), dtype, True), name="output")
    initial_spec = specs.Compose(
        (specs.Convolution, specs.Convolution),
        (filters_b.spec, img.spec, filters_a.spec),
        output.spec,
        intermediate_dtypes=(dtype,),
        serial_only=False,
    )
    initial_op = morello.impl.compose.ComposeHole(
        initial_spec
    )

    for child in _walk_actions(initial_op, depth=3):
        assert child.spec != initial_spec, f"{child.spec} == {str(initial_spec)}"


@pytest.mark.skip("Not implemented")
def test_composehole_construction_fails_if_shapes_incompatible():
    raise NotImplementedError()
