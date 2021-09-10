import functools
import io
import operator
import os.path
import subprocess
import tempfile

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st

from morello import op_pprint, ops, specs, system_config, tensor
from morello.codegen import gen

CC_DEADLINE = 9000
CC_SANITIZE = True


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


def _test_c_program(source: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as dirname:
        source_path = os.path.join(dirname, "main.c")
        binary_path = os.path.join(dirname, "a.out")
        with open(source_path, mode="w") as fo:
            print(source, file=fo)
        # TODO: Is there a more secure way to depend on cc? And should we use clang?
        additional_args = []
        if CC_SANITIZE:
            additional_args += [
                "-fno-omit-frame-pointer",
                "-fsanitize=undefined",
                "-fsanitize=address",
            ]
        subprocess.run(
            ["/usr/bin/env", "clang"]
            + additional_args
            + ["-o", binary_path, source_path],
            check=True,
        )
        binary_result = subprocess.run(
            [binary_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        binary_output = binary_result.stdout.decode("utf-8")
        binary_stderr = binary_result.stderr.decode("utf-8")
        hypothesis.note("Output of program:\n" + binary_output)
        hypothesis.note("stderr of program:\n" + binary_stderr)
        binary_result.check_returncode()
        return _read_from_output(binary_output)


def _conv2d(img: np.ndarray, filters: np.ndarray) -> np.ndarray:
    assert img.dtype == filters.dtype
    fh, fw, fc = filters.shape
    h_out = img.shape[0] - filters.shape[0] + 1
    w_out = img.shape[1] - filters.shape[1] + 1
    out = np.zeros((h_out, w_out, fc), dtype=img.dtype)
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
    assert actions, f"actions was empty"

    action_idx = draw(st.integers(min_value=0, max_value=len(actions) - 1))

    # TODO: Need to repeatedly draw.
    expanded = actions[action_idx]()
    return expanded.replace_children(
        draw(_arb_impls_from_actions(c)) for c in expanded.children
    )


def _spec_to_applied_hole(spec: specs.Spec) -> ops.Schedule:
    return ops.spec_to_hole(
        spec,
        tuple(tensor.Tensor(inp_spec, name=None) for inp_spec in spec.inputs),
        tensor.Tensor(spec.output, name=None),
    )


@st.composite
def _arb_conv_spec(draw):
    """A strategy that yields Conv-then-Reduce specs."""
    inp_h = draw(st.integers(min_value=1, max_value=9))
    inp_w = draw(st.integers(min_value=1, max_value=9))
    fh = draw(st.integers(min_value=1, max_value=inp_h))
    fw = draw(st.integers(min_value=1, max_value=inp_w))
    fc = draw(st.integers(min_value=1, max_value=9))
    out_h, out_w = 1 + inp_h - fh, 1 + inp_w - fw

    return specs.Convolution(
        specs.TensorSpec((inp_h, inp_w)),
        specs.TensorSpec((fh, fw, fc)),
        output=specs.TensorSpec((out_h, out_w, fc)),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_reduce_conv_spec(draw):
    """A strategy that yields Conv-then-Reduce specs."""
    inp_h = draw(st.integers(min_value=1, max_value=9))
    inp_w = draw(st.integers(min_value=1, max_value=9))
    fh = draw(st.integers(min_value=1, max_value=inp_h))
    fw = draw(st.integers(min_value=1, max_value=inp_w))
    fc = draw(st.integers(min_value=1, max_value=9))
    out_h, out_w = 1 + inp_h - fh, 1 + inp_w - fw

    return specs.Compose(
        (specs.ReduceSum, specs.Convolution),
        inputs=(
            specs.TensorSpec((inp_h, inp_w)),
            specs.TensorSpec((fh, fw, fc)),
        ),
        output=specs.TensorSpec((out_h, out_w)),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_conv_reduce_conv_spec(draw):
    """A strategy that yields Conv-Reduce-Conv specs."""
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
            specs.TensorSpec((fb_h, fb_w, fb_c)),
            specs.TensorSpec((inp_h, inp_w)),
            specs.TensorSpec((fa_h, fa_w, fa_c)),
        ),
        output=specs.TensorSpec((out_h, out_w, fb_c)),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_matmul_spec(draw):
    """A strategy that yields Matmul specs."""
    m = draw(st.integers(min_value=1, max_value=13))
    k = draw(st.integers(min_value=1, max_value=13))
    n = draw(st.integers(min_value=1, max_value=13))
    return specs.Matmul(
        specs.TensorSpec((m, k)),
        specs.TensorSpec((k, n)),
        output=specs.TensorSpec((m, n)),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_matmul_matmul_spec(draw):
    """A strategy that yields Matmul-Matmul specs."""
    m = draw(st.integers(min_value=1, max_value=9))
    k = draw(st.integers(min_value=1, max_value=9))
    n = draw(st.integers(min_value=1, max_value=9))
    second_n = draw(st.integers(min_value=1, max_value=9))
    return specs.Compose(
        (specs.Matmul, specs.Matmul),
        inputs=(
            specs.TensorSpec((n, second_n)),
            specs.TensorSpec((m, k)),
            specs.TensorSpec((k, n)),
        ),
        output=specs.TensorSpec((m, second_n)),
        serial_only=draw(st.booleans()),
    )


@st.composite
def _arb_zip_values_for_impl(draw, impl: ops.Schedule):
    dtype = system_config.DEFAULT_SYSTEM_CONFIG.dtype

    def _value(shape):
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

    return impl, [_value(op.dim_sizes) for op in impl.inputs]


def _calculator_to_test(spec_st):
    def decorator_wrapper(calc_fn):
        @pytest.mark.slow
        @hypothesis.given(
            spec_st.map(_spec_to_applied_hole)
            .flatmap(_arb_impls_from_actions)
            .flatmap(_arb_zip_values_for_impl)
        )
        @hypothesis.settings(deadline=CC_DEADLINE)
        @functools.wraps(calc_fn)
        def wrapper(pair):
            impl, inp_values = pair

            hypothesis.note(
                "Impl:\n"
                + op_pprint.pformat(impl, show_utilization=False, show_cost=False)
            )

            expected_result = calc_fn(inp_values)

            dtype = system_config.DEFAULT_SYSTEM_CONFIG.dtype
            assert expected_result.dtype == dtype.np_type, (
                f"expected_result should be of system dtype {dtype.np_type} "
                f"but was {expected_result.dtype}"
            )

            with io.StringIO() as buffer:
                gen.generate_c("print_output", impl, buffer, inp_values)
                source_code = buffer.getvalue()
            hypothesis.note("Source:\n" + source_code)
            result = _test_c_program(source_code)

            np.testing.assert_allclose(result, expected_result, rtol=1e-6)

        return wrapper

    return decorator_wrapper


@_calculator_to_test(_arb_matmul_spec())
def test_codegen_for_matmul(inp_values):
    return inp_values[0] @ inp_values[1]


@_calculator_to_test(_arb_matmul_matmul_spec())
def test_codegen_for_matmul_matmul(inp_values):
    out = (inp_values[1] @ inp_values[2]) @ inp_values[0]
    assert out.dtype == inp_values[0].dtype
    return out


@_calculator_to_test(_arb_conv_spec())
def test_codegen_for_conv(inp_values):
    return _conv2d(*inp_values)


@_calculator_to_test(_arb_reduce_conv_spec())
def test_codegen_for_reduce_conv(inp_values):
    dtype = system_config.DEFAULT_SYSTEM_CONFIG.dtype
    expected_result = _conv2d(*inp_values)
    expected_result = np.sum(expected_result, axis=-1, dtype=dtype.np_type)
    return expected_result


@_calculator_to_test(_arb_conv_reduce_conv_spec())
def test_codegen_for_conv_reduce_conv(inp_values):
    dtype = system_config.DEFAULT_SYSTEM_CONFIG.dtype
    expected_result = _conv2d(*inp_values[1:3])
    expected_result = np.sum(expected_result, axis=-1, dtype=dtype.np_type)
    expected_result = _conv2d(expected_result, inp_values[0])
    return expected_result
