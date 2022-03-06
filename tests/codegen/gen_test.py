import functools
import itertools
import operator
from typing import Optional

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st

import morello.impl.actions
import morello.impl.base
import morello.impl.compose
import morello.impl.moves
from morello import dtypes, op_pprint, specs, system_config, tensor
from morello.codegen import indexexpr
from morello.system_config import cpu, hexagon

from .. import strategies

CC_DEADLINE = 60 * 1000
CC_SANITIZE = True

strategies.register_default_strategies()


def test_can_schedule_generate_and_run_parallel_matmul_without_raise() -> None:
    """Manually schedule a Matmul, generate code, compile, and run.

    This test's goals are subsumed by the property-based tests in codegen/gen_test.py.
    It exists to educate new Morello users/developers.
    """
    target = cpu.CpuTarget()
    with system_config.with_target(target):
        spec = specs.Matmul(
            target.tensor_spec((256, 256), dtype=dtypes.Uint32),
            target.tensor_spec((256, 256), dtype=dtypes.Uint32),
            target.tensor_spec((256, 256), dtype=dtypes.Uint32),
            serial_only=False,
        )
        hole = morello.impl.base.spec_to_hole(
            spec,
            (target.tensor(spec.lhs), target.tensor(spec.rhs)),
            target.tensor(spec.output),
        )
        imp = (
            hole.tile_out((8, 8), parallel=True)
            .move_input(0, bank="RF")
            .move_input(1, bank="RF")
            .move_output(bank="RF")
            .complete()
        )

        # Make some arbitrary input values
        input_values = [
            np.arange(16 * 4, dtype=np.uint32).reshape((16, 4)),
            np.arange(4 * 16, dtype=np.uint32).reshape((4, 16)),
        ]

        run_result = target.run_impl(
            imp,
            print_output=True,
            source_cb=lambda s: print("Source Code:\n" + s),
            values=input_values,
        )
        print("stderr of program:\n" + run_result.stderr)
        print("")
        print("stdout of program:\n" + run_result.stdout)


def _read_from_output(output: str) -> np.ndarray:
    reported_rank: Optional[int] = None

    is_3d = False
    accum = []
    for line in output.splitlines():
        # Use leading whitespace to determine intended rank
        if line.strip() and reported_rank is None:
            reported_rank = 1 + len(line) - len(line.lstrip(" "))

        if is_3d:
            if not line.strip():
                accum.append([])
            else:
                row = [float(s) for s in line.split()]
                accum[-1].append(row)
        else:
            if not line.strip():
                is_3d = True
                accum = [accum, []]
            else:
                row = [float(s) for s in line.split()]
                accum.append(row)
    if is_3d and len(accum[-1]) == 0:
        accum.pop()

    result_arr = np.array(accum)
    if reported_rank:
        while len(result_arr.shape) < reported_rank:
            result_arr = result_arr[np.newaxis, :]
    return result_arr


def _conv2d(img: np.ndarray, filters: np.ndarray, out_type) -> np.ndarray:
    fh, fw, fc = filters.shape
    h_out = img.shape[0] - filters.shape[0] + 1
    w_out = img.shape[1] - filters.shape[1] + 1
    out = np.zeros((h_out, w_out, fc), dtype=out_type)
    for i in range(1 + img.shape[0] - filters.shape[0]):
        for j in range(1 + img.shape[1] - filters.shape[1]):
            acts = img[i : i + fh, j : j + fw, np.newaxis] * filters
            acts = acts.sum(axis=(0, 1), keepdims=True)
            out[i, j, :] = acts
    return out


def _count_basic_leaves(imp: morello.impl.base.Impl) -> int:
    """Return the of sub-Impls which aren't Pipelines or ComposeHoles."""
    if isinstance(imp, morello.impl.compose.Pipeline):
        return sum(_count_basic_leaves(s) for s in imp.stages)
    elif isinstance(imp, morello.impl.compose.ComposeHole):
        return 0
    elif isinstance(imp, morello.impl.moves.MoveLet):
        return _count_basic_leaves(imp.inner)
    else:
        assert len(imp.children) == 0
        return 1


@st.composite
def _arb_impls_from_actions(draw, partial_impl: morello.impl.base.Impl):
    """A strategy that expands a given Impl into a fully scheduled Impl."""
    # TODO: Remove the following filter once codegen is implemented for sliding
    #  windows.
    actions = [
        a
        for a in partial_impl.actions()
        if not isinstance(a, morello.impl.actions.SlidingTileOutAction)
    ]

    if partial_impl.is_scheduled:
        if not actions or draw(st.booleans()):
            return partial_impl

    # If we hit a dead end (no actions), then this isn't a viable example
    hypothesis.assume(len(actions))

    action_idx = draw(st.integers(min_value=0, max_value=len(actions) - 1))
    expanded = actions[action_idx]()
    return expanded.replace_children(
        draw(_arb_impls_from_actions(c)) for c in expanded.children
    )


def _spec_to_applied_hole(spec: specs.Spec) -> morello.impl.base.Impl:
    target = system_config.current_target()
    return morello.impl.base.spec_to_hole(
        spec,
        tuple(target.tensor(inp_spec, name=None) for inp_spec in spec.inputs),
        target.tensor(spec.output, name=None),
    )


@st.composite
def _arb_conv_spec(draw):
    """A strategy that yields Convolution specs.

    All tensors will have the same type.
    """
    target = system_config.current_target()

    dtype = draw(st.from_type(dtypes.Dtype))
    inp_h = draw(st.integers(min_value=1, max_value=9))
    inp_w = draw(st.integers(min_value=1, max_value=9))
    fh = draw(st.integers(min_value=1, max_value=inp_h))
    fw = draw(st.integers(min_value=1, max_value=inp_w))
    fc = draw(st.integers(min_value=1, max_value=9))
    out_h, out_w = 1 + inp_h - fh, 1 + inp_w - fw

    return specs.Convolution(
        target.tensor_spec((inp_h, inp_w), dtype=dtype),
        target.tensor_spec((fh, fw, fc), dtype=dtype),
        output=target.tensor_spec((out_h, out_w, fc), dtype=dtype),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_reduce_conv_spec(draw):
    """A strategy that yields Conv-then-Reduce specs.

    All tensors will have the same type.
    """
    target = system_config.current_target()

    dtype = draw(st.from_type(dtypes.Dtype))
    batch_count = draw(st.integers(min_value=1, max_value=9))
    channels = draw(st.integers(min_value=1, max_value=9))
    inp_h = draw(st.integers(min_value=1, max_value=9))
    inp_w = draw(st.integers(min_value=1, max_value=9))
    fh = draw(st.integers(min_value=1, max_value=inp_h))
    fw = draw(st.integers(min_value=1, max_value=inp_w))
    fc = draw(st.integers(min_value=1, max_value=9))
    out_h, out_w = 1 + inp_h - fh, 1 + inp_w - fw

    return specs.Compose(
        (specs.ReduceSum, specs.Convolution),
        inputs=(
            target.tensor_spec((batch_count, channels, inp_h, inp_w), dtype=dtype),
            target.tensor_spec((fc, channels, fh, fw), dtype=dtype),
        ),
        output=target.tensor_spec((batch_count, fc, out_h, out_w), dtype=dtype),
        intermediate_dtypes=(draw(st.from_type(dtypes.Dtype)),),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_conv_reduce_conv_spec(draw):
    """A strategy that yields Conv-Reduce-Conv specs.

    All tensors will have the same dtype.
    """
    target = system_config.current_target()

    dtype = draw(st.from_type(dtypes.Dtype))
    batch_count = draw(st.integers(min_value=1, max_value=9))
    channels = draw(st.integers(min_value=1, max_value=9))
    inp_h = draw(st.integers(min_value=1, max_value=9))
    inp_w = draw(st.integers(min_value=1, max_value=9))
    fa_h = draw(st.integers(min_value=1, max_value=inp_h))
    fa_w = draw(st.integers(min_value=1, max_value=inp_w))
    fa_c = draw(st.integers(min_value=1, max_value=9))
    first_h, first_w = 1 + inp_h - fa_h, 1 + inp_w - fa_w

    fb_h = draw(st.integers(min_value=1, max_value=first_h))
    fb_w = draw(st.integers(min_value=1, max_value=first_w))
    fb_c = draw(st.integers(min_value=1, max_value=9))
    out_h, out_w = 1 + first_h - fb_h, 1 + first_w - fb_w

    return specs.Compose(
        (specs.Convolution, specs.ReduceSum, specs.Convolution),
        inputs=(
            target.tensor_spec((fb_c, channels, fb_h, fb_w), dtype=dtype),
            target.tensor_spec((batch_count, channels, inp_h, inp_w), dtype=dtype),
            target.tensor_spec((fa_c, channels, fa_h, fa_w), dtype=dtype),
        ),
        output=target.tensor_spec((batch_count, fb_c, out_h, out_w), dtype=dtype),
        intermediate_dtypes=(
            draw(st.from_type(dtypes.Dtype)),
            draw(st.from_type(dtypes.Dtype)),
        ),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_matmul_spec(draw):
    """A strategy that yields Matmul specs.

    All tensors will have the same dtype.
    """
    # The max sizes should be at least 16 to explore vectorized ops
    target = system_config.current_target()

    dtype = draw(st.from_type(dtypes.Dtype))
    m = draw(st.integers(min_value=1, max_value=32))
    k = draw(st.integers(min_value=1, max_value=32))
    n = draw(st.integers(min_value=1, max_value=32))
    return specs.Matmul(
        target.tensor_spec((m, k), dtype=dtype),
        target.tensor_spec((k, n), dtype=dtype),
        output=target.tensor_spec((m, n), dtype=dtype),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_matmul_matmul_spec(draw):
    """A strategy that yields Matmul-Matmul specs."""
    # The max sizes should be at least 16 to explore vectorized ops
    target = system_config.current_target()

    dtype = draw(st.from_type(dtypes.Dtype))
    m = draw(st.integers(min_value=1, max_value=32))
    k = draw(st.integers(min_value=1, max_value=32))
    n = draw(st.integers(min_value=1, max_value=32))
    second_n = draw(st.integers(min_value=1, max_value=9))
    return specs.Compose(
        (specs.Matmul, specs.Matmul),
        inputs=(
            target.tensor_spec((n, second_n), dtype=draw(st.from_type(dtypes.Dtype))),
            target.tensor_spec((m, k), dtype=draw(st.from_type(dtypes.Dtype))),
            target.tensor_spec((k, n), dtype=draw(st.from_type(dtypes.Dtype))),
        ),
        output=target.tensor_spec((m, second_n), dtype=dtype),
        intermediate_dtypes=(draw(st.from_type(dtypes.Dtype)),),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_zip_values_for_impl(draw, imp: morello.impl.base.Impl):
    def _value(shape, dtype: dtypes.Dtype):
        num_elements: int = functools.reduce(operator.mul, shape, 1)
        return np.asarray(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=7),
                    max_size=num_elements,
                    min_size=num_elements,
                )
            ),
            dtype=dtype.np_type,
        ).reshape(shape)

    return imp, [_value(op.dim_sizes, op.dtype) for op in imp.inputs]


def _calculator_to_test(spec_st):
    def decorator_wrapper(calc_fn):
        @pytest.mark.slow
        @pytest.mark.parametrize(
            "target",
            [
                cpu.CpuTarget(),
                pytest.param(hexagon.HvxSimulatorTarget(), marks=pytest.mark.hexagon),
            ],
            ids=["cpu", "hexagon"],
        )
        @hypothesis.given(st.data())
        @hypothesis.settings(deadline=CC_DEADLINE)
        @functools.wraps(calc_fn)
        def wrapper(target, data):
            with system_config.with_target(target):
                impl, inp_values = data.draw(
                    spec_st.map(_spec_to_applied_hole)
                    .flatmap(_arb_impls_from_actions)
                    .flatmap(_arb_zip_values_for_impl)
                )
                _test_impl(impl, inp_values, calc_fn)

        return wrapper

    return decorator_wrapper


def _test_impl(imp: morello.impl.base.Impl, inp_values, calc_fn):
    target = system_config.current_target()

    pformatted = op_pprint.pformat(imp, show_utilization=False, show_cost=False)
    hypothesis.note("Impl:\n" + pformatted)

    expected_result = calc_fn(imp.spec, inp_values)

    additional_clang_args = []
    if CC_SANITIZE:
        additional_clang_args += [
            "-fno-omit-frame-pointer",
            "-fsanitize=undefined",
            "-fsanitize=address",
        ]

    run_result = target.run_impl(
        imp,
        print_output=True,
        source_cb=lambda s: hypothesis.note("Source Code:\n" + s),
        values=inp_values,
    )
    hypothesis.note("stderr of program:\n" + run_result.stderr)
    hypothesis.note("stdout of program:\n" + run_result.stdout)
    hypothesis.note("Expected output:\n" + str(expected_result))
    result = _read_from_output(run_result.stdout)
    np.testing.assert_allclose(result, expected_result, rtol=1e-6)


@st.composite
def _st_test_index_exprs_consistent_with_contiguous_props(draw):
    target = system_config.current_target()

    t = draw(st.from_type(tensor.Tensor))

    # TODO: Test column-major as well.
    tensor_spec = draw(
        strategies.tensorspec_st(
            max_dim_size=9, min_dims=1, max_dims=4, layout=specs.Layout.ROW_MAJOR
        )
    )
    t = target.tensor(spec=tensor_spec, name=None, origin=None)

    concrete_tile_idxs: list[list[int]] = []

    depth = draw(st.integers(1, 2))
    for _ in range(depth):
        # TODO: Use a more generic strategy for producing any tile callable
        # TODO: Test convolution tiles
        t = t.simple_tile(tuple(draw(st.integers(1, d)) for d in t.dim_sizes))
        # The following `assume` avoids the case where simple_tile returns `t`
        # itself.
        hypothesis.assume(not isinstance(t, tensor.Tensor))
        assert not isinstance(t, tensor.Tensor)
        concrete_tile_idxs.append(
            [
                draw(st.integers(min_value=0, max_value=t.steps_dim(dim_idx) - 1))
                for dim_idx in range(len(t.dim_sizes))
            ]
        )
    return t, concrete_tile_idxs


@hypothesis.settings(max_examples=1000, deadline=4000)
@hypothesis.given(_st_test_index_exprs_consistent_with_contiguous_props())
def test_index_exprs_consistent_with_contiguous_props(inp):
    """Test that Tiles' `contiguous` property matches walking elements.

    More specifically: test that walking all elements with the highest-numbered
    logical index (`p1` in a 2-dim. tensor) in an innermost loop returns
    adjacent buffer indices.
    """
    tile, concrete_tile_idxs = inp

    # TODO: Modify the strategy so that it always returns a Tile.
    hypothesis.assume(isinstance(tile, tensor.Tile))

    # Compose indexing expressions so that we have a mapping from the final
    # tile's coordinates all the way back to its root tensor.
    stack: list[tensor.Tile] = [tile]
    while isinstance(stack[0].origin, tensor.Tile):
        stack.insert(0, stack[0].origin)
    expr = indexexpr.buffer_indexing_expr(stack[0].origin)
    while stack:
        operand = stack[0]
        del stack[0]
        all_substitutions = {}
        tile_it_vars = concrete_tile_idxs[-1]
        del concrete_tile_idxs[-1]
        for dim_idx, it_var in zip(range(len(tile.dim_sizes)), tile_it_vars):
            e = indexexpr.logical_indexing_expr(operand, dim_idx)
            e = e.subs(f"i{dim_idx}", it_var)
            all_substitutions[f"p{dim_idx}"] = e
        expr = expr.subs(all_substitutions)

    # Walk the elements.
    is_contiguous = True
    last_offset: Optional[int] = None
    for pts in itertools.product(*map(range, tile.dim_sizes)):
        offset = int(expr.evalf(subs={f"p{idx}": pt for idx, pt in enumerate(pts)}))
        assert isinstance(offset, int), f"offset was type {type(offset)}"
        if last_offset is None:
            last_offset = offset
            continue
        if offset != last_offset + 1:
            is_contiguous = False
            break
        last_offset = offset

    assert tile.contiguous == is_contiguous


@_calculator_to_test(_arb_matmul_spec())
def test_codegen_for_matmul(_, inp_values):
    return inp_values[0] @ inp_values[1]


@pytest.mark.slow
@pytest.mark.hexagon
@hypothesis.settings(deadline=CC_DEADLINE)
@hypothesis.given(
    m_mult=st.integers(min_value=1, max_value=6),
    k_mult=st.integers(min_value=1, max_value=6),
    n=st.integers(min_value=32, max_value=144),
    serial_only=st.booleans(),
)
def test_codegen_for_matmul_with_hvx_gemvmpebbw_with_aligned_k_without_split(
    m_mult, k_mult, n, serial_only
):
    m = m_mult * 4
    k = k_mult * 16

    target = hexagon.HvxSimulatorTarget()
    with system_config.with_target(target):
        spec = specs.Matmul(
            target.tensor_spec((m, k), dtype=dtypes.Uint8),
            target.tensor_spec((k, n), dtype=dtypes.Uint8),
            target.tensor_spec((m, n), dtype=dtypes.Uint32),
            serial_only=serial_only,
        )
        hole = morello.impl.base.spec_to_hole(
            spec,
            (target.tensor(spec.lhs), target.tensor(spec.rhs)),
            target.tensor(spec.output),
        )
        imp = hole.tile_out((1, 32))
        # TODO: Re-introduce split
        # if spec.lhs.dim_sizes[1] > 16:
        #     imp = imp.split(16)
        imp = (
            imp.move_input(0, bank="L2")
            .pad_transpack(1)
            .move_input(1, bank="L2")
            .move_output(bank="L2")
            .place_hvx_gemvmpebbw()
        )

        inp_values = [
            np.arange(m * k, dtype=np.uint32).reshape((m, k)),
            np.arange(k * n, dtype=np.uint32).reshape((k, n)),
        ]
        _test_impl(imp, inp_values, lambda _, v: v[0] @ v[1])


@_calculator_to_test(_arb_matmul_matmul_spec())
def test_codegen_for_matmul_matmul(spec, inp_values):
    first_result = np.matmul(
        inp_values[1], inp_values[2], dtype=spec.intermediate_dtypes[0].np_type
    )
    return np.matmul(first_result, inp_values[0], dtype=spec.output.dtype.np_type)


@_calculator_to_test(_arb_conv_spec())
def test_codegen_for_conv(spec, inp_values):
    out_type = spec.output.dtype.np_type
    return _conv2d(*inp_values, out_type)


@_calculator_to_test(_arb_reduce_conv_spec())
def test_codegen_for_reduce_conv(spec, inp_values):
    conv_out_type = spec.intermediate_dtypes[0].np_type
    expected_result = _conv2d(*inp_values, conv_out_type)
    final_out_type = spec.output.dtype.np_type
    expected_result = np.sum(expected_result, axis=-1, dtype=final_out_type)
    return expected_result


@_calculator_to_test(_arb_conv_reduce_conv_spec())
def test_codegen_for_conv_reduce_conv(spec, inp_values):
    expected_result = _conv2d(*inp_values[1:3], spec.intermediate_dtypes[1].np_type)
    expected_result = np.sum(
        expected_result, axis=-1, dtype=spec.intermediate_dtypes[0].np_type
    )
    return _conv2d(expected_result, inp_values[0], spec.output.dtype.np_type)
