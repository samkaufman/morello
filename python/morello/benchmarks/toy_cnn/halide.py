import ctypes

import halide as hl

ctypes.cdll.LoadLibrary("libautoschedule_adams2019.so")


def halide_matmul(lhs, rhs, matrix_size: int, name="matmul") -> hl.Func:
    matmul = hl.Func(name)

    x, y = hl.Var("x"), hl.Var("y")
    k = hl.RDom([(0, matrix_size)], name="k")

    matmul[x, y] = 0.0
    matmul[x, y] += lhs[x, k] * rhs[k, y]

    if not isinstance(lhs, hl.Func):
        assert lhs.dim(0).extent() == matrix_size
        assert lhs.dim(1).extent() == matrix_size
    if not isinstance(rhs, hl.Func):
        assert rhs.dim(0).extent() == matrix_size
        assert rhs.dim(1).extent() == matrix_size
    matmul.set_estimates([(0, matrix_size), (0, matrix_size)])
    return matmul


def halide_gemm3(a, b, c, matrix_size: int, name="gemm3") -> hl.Func:
    stage1 = halide_matmul(a, b, matrix_size, name="gemm3_stage1")
    stage2 = halide_matmul(stage1, c, matrix_size, name="gemm3_stage2")
    return stage2


def halide_conv_layer(input, W: hl.Buffer, name="convolution") -> hl.Func:
    """Builds a 2D convolution over some image with a filters Buffer.

    Image is expected to be 4D: (batch, channel, height, width).
    Filters is expected to be 4D: (filters, channel, height, width).
    The output is in NHWC format: (batch, channel, height, width).
    """
    convolution = hl.Func(name)
    b, c = hl.Var("b"), hl.Var("c")
    x, y = hl.Var("x"), hl.Var("y")
    # TODO: Update this to range in both directions, consistent with other
    #   benchmarks.
    spatial_r = hl.RDom(
        [(0, W.dim(1).extent()), (0, W.dim(2).extent()), (0, W.dim(3).extent())],
        name="spatial_r",
    )
    convolution[b, c, x, y] = 0.0
    convolution[b, c, x, y] += (
        input[b, spatial_r[0], x + spatial_r[1], y + spatial_r[2]]
        * W[c, spatial_r[0], spatial_r[1], spatial_r[2]]
    )
    return convolution


def halide_small_cnn(img: hl.Buffer, filters_a: hl.Buffer, filters_b: hl.Buffer):
    fn = halide_conv_layer(img, filters_a, name="conv_a")
    fn = halide_conv_layer(fn, filters_b, name="conv_b")
    fn.set_estimates([(0, 1), (0, 256), (0, 256), (0, 256)])
    return fn
