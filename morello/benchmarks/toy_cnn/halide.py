import ctypes

import halide as hl

ctypes.cdll.LoadLibrary("libautoschedule_adams2019.so")


def halide_conv_layer(input, W: hl.Buffer, name="convolution") -> hl.Func:
    """Builds a 2D convolution over some image with a filters Buffer.

    Image is expected to be 4D: (batch, channel, height, width).
    Filters is expected to be 4D: (filters, channel, height, width).
    The output is in NHWC format: (batch, channel, height, width).
    """
    convolution = hl.Func(name)
    b, f = hl.Var("b"), hl.Var("f")
    x, y = hl.Var("x"), hl.Var("y")
    spatial_r = hl.RDom(
        [(0, W.dim(1).extent()), (0, W.dim(2).extent()), (0, W.dim(3).extent())],
        name="spatial_r",
    )
    convolution[b, f, x, y] = 0.0
    convolution[b, f, x, y] += (
        input[b, spatial_r[0], x + spatial_r[1], y + spatial_r[2]]
        * W[f, spatial_r[0], spatial_r[1], spatial_r[2]]
    )
    return convolution


# TODO: Does the caller really need to specify inner_size?
def halide_sum(input, inner_size: int) -> hl.Func:
    """Sums over the second dimension of a 4D Func.

    Requires the second dimension size to be provided as `inner_size`.

    Returns a 4D Func with a degenerate second dimension.
    """
    assert input.dimensions() == 4, "Only 4-dim. input supported for now"
    a, b, c = hl.Var("a"), hl.Var("b"), hl.Var("c")
    z = hl.Var("z")
    r = hl.RDom([(0, inner_size)], name="r")
    s = hl.Func("reduce_sum")
    s[a, z, b, c] = 0.0
    s[a, 0, b, c] = hl.sum(input[a, r, b, c])
    return s


def halide_small_cnn(img: hl.Buffer, filters_a: hl.Buffer, filters_b: hl.Buffer):
    fn = halide_conv_layer(img, filters_a, name="conv_a")
    fn = halide_sum(fn, filters_a.dim(0).extent())
    fn = halide_conv_layer(fn, filters_b, name="conv_b")

    # TODO: Remove once the expected output of Morello is NCHW (PyTorch norm.)
    reshaped = hl.Func("reshaped")
    b, c, h, w = hl.Var("b"), hl.Var("c"), hl.Var("h"), hl.Var("w")
    reshaped[b, h, w, c] = fn[b, c, h, w]
    reshaped.set_estimates([(0, 1), (0, 256), (0, 256), (0, 256)])
    return reshaped
