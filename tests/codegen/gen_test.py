import functools
import itertools
import operator
from typing import Optional

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st

from morello import dtypes, op_pprint, ops, specs, system_config, tensor
from morello.codegen import indexexpr
from morello.system_config import cpu, hexagon
from .. import strategies

CC_DEADLINE = 30000
CC_SANITIZE = True

strategies.register_default_strategies()


def _read_from_output(output: str) -> np.ndarray:
    is_3d = False
    accum = []
    for line in output.splitlines():
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
    return np.array(accum)


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


def _count_basic_leaves(impl: ops.Schedule) -> int:
    """Return the of sub-Impls which aren't Pipelines or ComposeHoles."""
    if isinstance(impl, ops.Pipeline):
        return sum(_count_basic_leaves(s) for s in impl.stages)
    elif isinstance(impl, ops.ComposeHole):
        return 0
    elif isinstance(impl, ops.MoveLet):
        return _count_basic_leaves(impl.inner)
    else:
        assert len(impl.children) == 0
        return 1


@st.composite
def _arb_impls_from_actions(draw, partial_impl: ops.Schedule):
    # Remember that this strategy only returns Impls that could appear during
    # schedule search. This might mean, for instance, that especially stange
    # output tile shapes are not explored.
    if partial_impl.is_scheduled:
        return partial_impl
    # TODO: Remove the following filter once codegen is implemented for column-major.
    #   and sliding windows.
    actions = [
        a
        for a in partial_impl.actions()
        if (
            not isinstance(a, (ops.MoveAction, ops.PeelAction))
            or a.layout == specs.Layout.ROW_MAJOR
        )
        and not isinstance(a, ops.SlidingTileOutAction)
    ]
    assert actions, f"actions was empty for Impl: {partial_impl}"

    action_idx = draw(st.integers(min_value=0, max_value=len(actions) - 1))

    # TODO: Need to repeatedly draw.
    expanded = actions[action_idx]()
    return expanded.replace_children(
        draw(_arb_impls_from_actions(c)) for c in expanded.children
    )


def _spec_to_applied_hole(spec: specs.Spec) -> ops.Schedule:
    target = system_config.current_target()
    return ops.spec_to_hole(
        spec,
        tuple(target.tensor(inp_spec, name=None) for inp_spec in spec.inputs),
        target.tensor(spec.output, name=None),
    )


@st.composite
def _arb_conv_spec(draw):
    """A strategy that yields Convolution specs.

    All tensors will have the same type.
    """
    dtype = draw(st.from_type(dtypes.Dtype))
    inp_h = draw(st.integers(min_value=1, max_value=9))
    inp_w = draw(st.integers(min_value=1, max_value=9))
    fh = draw(st.integers(min_value=1, max_value=inp_h))
    fw = draw(st.integers(min_value=1, max_value=inp_w))
    fc = draw(st.integers(min_value=1, max_value=9))
    out_h, out_w = 1 + inp_h - fh, 1 + inp_w - fw

    return specs.Convolution(
        specs.TensorSpec((inp_h, inp_w), dtype=dtype),
        specs.TensorSpec((fh, fw, fc), dtype=dtype),
        output=specs.TensorSpec((out_h, out_w, fc), dtype=dtype),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_reduce_conv_spec(draw):
    """A strategy that yields Conv-then-Reduce specs.

    All tensors will have the same type.
    """
    dtype = draw(st.from_type(dtypes.Dtype))
    inp_h = draw(st.integers(min_value=1, max_value=9))
    inp_w = draw(st.integers(min_value=1, max_value=9))
    fh = draw(st.integers(min_value=1, max_value=inp_h))
    fw = draw(st.integers(min_value=1, max_value=inp_w))
    fc = draw(st.integers(min_value=1, max_value=9))
    out_h, out_w = 1 + inp_h - fh, 1 + inp_w - fw

    return specs.Compose(
        (specs.ReduceSum, specs.Convolution),
        inputs=(
            specs.TensorSpec((inp_h, inp_w), dtype=dtype),
            specs.TensorSpec((fh, fw, fc), dtype=dtype),
        ),
        output=specs.TensorSpec((out_h, out_w), dtype=dtype),
        intermediate_dtypes=(draw(st.from_type(dtypes.Dtype)),),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_conv_reduce_conv_spec(draw):
    """A strategy that yields Conv-Reduce-Conv specs.

    All tensors will have the same dtype.
    """
    dtype = draw(st.from_type(dtypes.Dtype))
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
            specs.TensorSpec((fb_h, fb_w, fb_c), dtype=dtype),
            specs.TensorSpec((inp_h, inp_w), dtype=dtype),
            specs.TensorSpec((fa_h, fa_w, fa_c), dtype=dtype),
        ),
        output=specs.TensorSpec((out_h, out_w, fb_c), dtype=dtype),
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
    dtype = draw(st.from_type(dtypes.Dtype))
    m = draw(st.integers(min_value=1, max_value=32))
    k = draw(st.integers(min_value=1, max_value=32))
    n = draw(st.integers(min_value=1, max_value=32))
    return specs.Matmul(
        specs.TensorSpec((m, k), dtype=dtype),
        specs.TensorSpec((k, n), dtype=dtype),
        output=specs.TensorSpec((m, n), dtype=dtype),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_matmul_matmul_spec(draw):
    """A strategy that yields Matmul-Matmul specs."""
    # The max sizes should be at least 16 to explore vectorized ops
    dtype = draw(st.from_type(dtypes.Dtype))
    m = draw(st.integers(min_value=1, max_value=32))
    k = draw(st.integers(min_value=1, max_value=32))
    n = draw(st.integers(min_value=1, max_value=32))
    second_n = draw(st.integers(min_value=1, max_value=9))
    return specs.Compose(
        (specs.Matmul, specs.Matmul),
        inputs=(
            specs.TensorSpec((n, second_n), dtype=draw(st.from_type(dtypes.Dtype))),
            specs.TensorSpec((m, k), dtype=draw(st.from_type(dtypes.Dtype))),
            specs.TensorSpec((k, n), dtype=draw(st.from_type(dtypes.Dtype))),
        ),
        output=specs.TensorSpec((m, second_n), dtype=dtype),
        intermediate_dtypes=(draw(st.from_type(dtypes.Dtype)),),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_zip_values_for_impl(draw, impl: ops.Schedule):
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

    return impl, [_value(op.dim_sizes, op.dtype) for op in impl.inputs]


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
        @hypothesis.given(
            spec_st.map(_spec_to_applied_hole)
            .flatmap(_arb_impls_from_actions)
            .flatmap(_arb_zip_values_for_impl),
        )
        @hypothesis.settings(deadline=CC_DEADLINE)
        @functools.wraps(calc_fn)
        def wrapper(target, pair):
            with system_config.with_target(target):
                impl, inp_values = pair

                hypothesis.note(
                    "Impl:\n"
                    + op_pprint.pformat(impl, show_utilization=False, show_cost=False)
                )

                expected_result = calc_fn(impl.spec, inp_values)

                additional_clang_args = []
                if CC_SANITIZE:
                    additional_clang_args += [
                        "-fno-omit-frame-pointer",
                        "-fsanitize=undefined",
                        "-fsanitize=address",
                    ]

                binary_output, binary_stderr = target.run_impl(
                    impl,
                    print_output=True,
                    source_cb=lambda s: hypothesis.note("Source Code:\n" + s),
                    values=inp_values,
                )
                hypothesis.note("stderr of program:\n" + binary_stderr)
                hypothesis.note("stdout of program:\n" + binary_output)
                hypothesis.note("Expected output:\n" + str(expected_result))
                result = _read_from_output(binary_output)
                np.testing.assert_allclose(result, expected_result, rtol=1e-6)

        return wrapper

    return decorator_wrapper


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


@hypothesis.settings(max_examples=1000, deadline=2000)
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
