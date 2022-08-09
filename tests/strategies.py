from typing import Any, Callable, Optional, Sequence

from hypothesis import strategies as st
import hypothesis

from morello import dtypes, impl, layouts, specs, system_config, tensor
from morello.system_config import cpu, hexagon

dtype_st = st.sampled_from([dtypes.Uint8, dtypes.Uint32])


@st.composite
def dim_st(draw, max_size: Optional[int] = None) -> int:
    if max_size is None:
        return draw(st.integers(min_value=1))
    return draw(st.integers(min_value=1, max_value=max_size))


@st.composite
def layout_st(
    draw,
    dtype=None,
    numels_ones: Optional[bool] = None,
    dim_sizes: Optional[tuple[int, ...]] = None,
) -> layouts.Layout:
    assert numels_ones is not None or dim_sizes
    if numels_ones is None:
        assert dim_sizes
        numels_ones = all(d == 1 for d in dim_sizes)

    target = system_config.current_target()
    if numels_ones:
        return layouts.row_major(len(dim_sizes))
    available_layouts = [
        l
        for l in target.all_layouts_for_shape(dim_sizes)
        if dim_sizes is None or dtype is None or l.applies_to_shape(dim_sizes, dtype)
    ]
    return draw(st.sampled_from(available_layouts))


@st.composite
def tensor_st(draw, *args, **kwargs):
    target = system_config.current_target()
    return target.tensor(spec=draw(tensorspec_st(*args, **kwargs)), name=draw(st.text()))


@st.composite
def tensorspec_st(
    draw,
    max_dim_size: Optional[int] = 128,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
    layout_fn: Optional[Callable[[int], layouts.Layout]] = None,
    contiguous: Optional[Any] = None,
    fully_contiguous: bool = False,
) -> specs.TensorSpec:
    target = system_config.current_target()

    dim_sizes = draw(
        st.lists(
            dim_st(max_size=max_dim_size), min_size=min_dims, max_size=max_dims
        ).map(tuple)
    )
    dtype = draw(st.from_type(dtypes.Dtype))

    if layout_fn is None:
        layout = draw(layout_st(dim_sizes=dim_sizes, dtype=dtype))
    else:
        layout = layout_fn(len(dim_sizes))

    if contiguous is None:
        if isinstance(layout, layouts.StandardLayout):
            if fully_contiguous:
                contiguous = len(dim_sizes)
            else:
                contiguous = draw(st.integers(min_value=0, max_value=len(dim_sizes)))
        elif isinstance(layout, layouts.PackedLayout):
            if fully_contiguous:
                contiguous = len(dim_sizes) + 1
            else:
                contiguous = draw(
                    st.integers(min_value=0, max_value=len(dim_sizes) + 1)
                )
        else:
            raise NotImplementedError()

    return target.tensor_spec(
        dim_sizes,
        dtype=dtype,
        contiguous=contiguous,
        layout=layout,
        bank=draw(st.sampled_from(sorted(target.system.banks))),
    )


@st.composite
def matmul_spec_st(draw, max_dim_size: Optional[int] = 256):
    target = system_config.current_target()
    lhs_dtype = draw(st.from_type(dtypes.Dtype))
    rhs_dtype = draw(st.from_type(dtypes.Dtype))
    lhs = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=2, max_dims=2))
    rhs_dim_sizes = (lhs.dim_sizes[1], draw(dim_st(max_size=max_dim_size)))
    rhs = target.tensor_spec(
        dim_sizes=rhs_dim_sizes,
        dtype=lhs_dtype,
        contiguous=draw(st.integers(min_value=0, max_value=len(rhs_dim_sizes))),
        layout=draw(layout_st(dim_sizes=rhs_dim_sizes, dtype=lhs_dtype)),
        bank=draw(st.sampled_from(sorted(target.system.banks))),
    )
    out_dim_sizes = (lhs.dim_sizes[0], rhs.dim_sizes[1])
    out = target.tensor_spec(
        dim_sizes=out_dim_sizes,
        dtype=rhs_dtype,
        contiguous=draw(st.integers(min_value=0, max_value=len(out_dim_sizes))),
        layout=draw(layout_st(dim_sizes=out_dim_sizes, dtype=rhs_dtype)),
        bank=draw(st.sampled_from(sorted(target.system.banks))),
    )
    return specs.Matmul(lhs, rhs, out, serial_only=draw(st.booleans()))


@st.composite
def convolution_spec_st(draw, max_dim_size: Optional[int] = 32):
    target = system_config.current_target()
    filters_dtype = draw(st.from_type(dtypes.Dtype))
    out_dtype = draw(st.from_type(dtypes.Dtype))
    image = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=4, max_dims=4))
    filters_dim_sizes = (
        draw(dim_st(max_size=max_dim_size)),
        image.dim_sizes[1],
        draw(dim_st(max_size=image.dim_sizes[2])),
        draw(dim_st(max_size=image.dim_sizes[3])),
    )
    filters = target.tensor_spec(
        dim_sizes=filters_dim_sizes,
        dtype=filters_dtype,
        contiguous=draw(st.integers(min_value=0, max_value=len(filters_dim_sizes))),
        layout=draw(layout_st(dim_sizes=filters_dim_sizes, dtype=filters_dtype)),
        bank=draw(st.sampled_from(sorted(target.system.banks))),
    )
    out_dim_sizes = specs.Convolution.output_shape(image.dim_sizes, filters_dim_sizes)
    out = target.tensor_spec(
        dim_sizes=out_dim_sizes,
        dtype=out_dtype,
        contiguous=draw(st.integers(min_value=0, max_value=len(out_dim_sizes))),
        layout=draw(layout_st(dim_sizes=out_dim_sizes, dtype=out_dtype)),
        bank=draw(st.sampled_from(sorted(target.system.banks))),
    )
    return specs.Convolution(image, filters, out, serial_only=draw(st.booleans()))


@st.composite
def reduce_spec_st(draw, max_dim_size: Optional[int] = 32):
    target = system_config.current_target()
    source = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=2, max_dims=4))
    output_dim_sizes = source.dim_sizes[:-1]
    out_dtype = draw(st.from_type(dtypes.Dtype))
    output = target.tensor_spec(
        dim_sizes=output_dim_sizes,
        dtype=out_dtype,
        contiguous=draw(st.integers(min_value=0, max_value=len(output_dim_sizes))),
        layout=draw(layout_st(dim_sizes=output_dim_sizes, dtype=out_dtype)),
        bank=draw(st.sampled_from(sorted(target.system.banks))),
    )
    return specs.ReduceSum(
        source=source, output=output, serial_only=draw(st.booleans())
    )


@st.composite
def compose_spec_st(draw) -> specs.Compose:
    target = system_config.current_target()

    subspec_classes = draw(
        st.lists(
            st.one_of(
                st.just(specs.ReduceSum),
                st.just(specs.Matmul),
                st.just(specs.Convolution),
            ),
            min_size=2,
        ).map(tuple)
    )
    inputs_count = specs.Compose.calculate_inputs_count(subspec_classes)
    # TODO: Instead of setting min_dims=2, get the minimum dims. from the subspec.
    #   This really needs to be respected in our choices of subspec as well. It
    #   is easy to get into situations where the minimum number of dimensions
    #   for an input tensor is unsatisfiable because the subspecs are, for
    #   instance, a list of Reduces.
    inputs_specs = draw(
        st.lists(
            tensorspec_st(min_dims=2), min_size=inputs_count, max_size=inputs_count
        ).map(tuple)
    )
    output_dim_sizes = specs.Compose.calculate_output(
        subspec_classes, [inp.dim_sizes for inp in inputs_specs]
    )
    out_dtype = draw(st.from_type(dtypes.Dtype))
    output_spec = target.tensor_spec(
        dim_sizes=output_dim_sizes,
        dtype=out_dtype,
        contiguous=draw(st.integers(min_value=0, max_value=len(output_dim_sizes))),
        bank=draw(st.sampled_from(sorted(system_config.current_system().banks))),
        layout=draw(layout_st(dim_sizes=output_dim_sizes, dtype=out_dtype)),
    )
    intermediate_dtypes = draw(
        st.lists(
            st.from_type(dtypes.Dtype),
            min_size=len(subspec_classes) - 1,
            max_size=len(subspec_classes) - 1,
        )
    )
    return specs.Compose(
        subspec_classes=subspec_classes,
        inputs=inputs_specs,
        output=output_spec,
        intermediate_dtypes=intermediate_dtypes,
        serial_only=draw(st.booleans()),
    )


@st.composite
def composehole_op_st(draw) -> impl.ComposeHole:
    target = system_config.current_target()
    spec = draw(compose_spec_st())
    return impl.ComposeHole(
        spec=spec,
        inputs=tuple(target.tensor(s, name=None) for s in spec.inputs),
        output=target.tensor(spec.output, name="output"),
    )


@st.composite
def pipeline_op_st(draw) -> impl.Pipeline:
    raise NotImplementedError()


@st.composite
def tiling_chain_st(draw, *args, chain_len: Optional[int] = None, allow_conv = True, **kwargs):
    base_tensor = draw(tensor_st(*args, **kwargs))

    if len(base_tensor.spec.dim_sizes) < tensor.ConvolutionImageTile.minimum_image_rank():
        allow_conv = False

    if chain_len is None:
        chain_len = draw(st.integers(min_value=1, max_value=3))
    assert chain_len is not None

    chain = [base_tensor]
    for _ in range(chain_len):
        use_conv = draw(st.booleans()) if allow_conv else False
        tile_shape = _smaller_shape(draw, chain[-1].dim_sizes)
        hypothesis.assume(chain[-1].spec.layout.applies_to_shape(tile_shape, chain[-1].spec.dtype))
        if use_conv:
            filter_shape = _smaller_shape(draw, tile_shape[1:])
            chain.append(chain[-1].spec.conv_image_tile(0, tile_shape, filter_shape))
        else:
            chain.append(chain[-1].spec.simple_tile(0, tile_shape))
    return chain


def _smaller_shape(draw, outer: Sequence[int]) -> tuple[int, ...]:
    tile_shape = []
    for dim in outer:
        tile_shape.append(draw(st.integers(min_value=1, max_value=dim)))
    return tuple(tile_shape)


def register_default_strategies():
    st.register_type_strategy(layouts.Layout, layout_st(numels_ones=False))
    st.register_type_strategy(dtypes.Dtype, dtype_st)
    st.register_type_strategy(specs.TensorSpec, tensorspec_st())

    st.register_type_strategy(tensor.Tensor, tensor_st())

    st.register_type_strategy(specs.Compose, compose_spec_st())
    st.register_type_strategy(specs.Matmul, matmul_spec_st())
    st.register_type_strategy(specs.Convolution, convolution_spec_st())
    st.register_type_strategy(specs.ReduceSum, reduce_spec_st())
    st.register_type_strategy(
        specs.Spec,
        st.one_of(
            st.from_type(specs.Compose),
            st.from_type(specs.Matmul),
            st.from_type(specs.Convolution),
            st.from_type(specs.ReduceSum),
        ),
    )

    st.register_type_strategy(impl.ComposeHole, composehole_op_st())
    st.register_type_strategy(impl.Pipeline, pipeline_op_st())
    st.register_type_strategy(
        impl.Impl,
        st.one_of(
            st.from_type(impl.ComposeHole),
            st.from_type(impl.Pipeline),
            st.from_type(impl.DirectConv),
            st.from_type(impl.ReduceSum),
            st.from_type(impl.Loop),
            st.from_type(impl.MoveLet),
        ),
    )