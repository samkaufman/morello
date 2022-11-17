import itertools
from typing import Any, Callable, Optional, Sequence

import hypothesis
from hypothesis import strategies as st

from morello import dtypes, impl, layouts, specs, system_config, tensor
from morello.system_config import cpu, hexagon

dtype_st = st.sampled_from([dtypes.Uint8, dtypes.Uint32])


@st.composite
def dim_st(draw, min_size: int = 1, max_size: Optional[int] = None) -> int:
    return draw(st.integers(min_value=min_size, max_value=max_size))


@st.composite
def layout_st(
    draw,
    dtype=None,
    numels_ones: Optional[bool] = None,
    dim_sizes: Optional[tuple[int, ...]] = None,
) -> layouts.Layout:
    assert (
        numels_ones is not None or dim_sizes
    ), f"numels_ones = {numels_ones} and dim_sizes = {dim_sizes}"
    if numels_ones is None:
        assert dim_sizes
        numels_ones = all(d == 1 for d in dim_sizes)

    target = system_config.current_target()
    if numels_ones:
        return layouts.row_major(len(dim_sizes))
    return draw(st.sampled_from(list(target.all_layouts_for_shape(dim_sizes))))


@st.composite
def tensor_st(draw, *args, **kwargs):
    target = system_config.current_target()
    return target.tensor(
        spec=draw(tensorspec_st(*args, **kwargs)), name=draw(st.text())
    )


@st.composite
def tensorspec_st(
    draw,
    dim_sizes: Optional[tuple[int, ...]] = None,
    max_dim_size: Optional[int] = 128,
    min_dims: int = 1,
    max_dims: int = 20,
    dtype: Optional[dtypes.Dtype] = None,
    layout_fn: Optional[Callable[[int], layouts.Layout]] = None,
    contiguous_abs: Optional[Any] = None,
    bank: Optional[str] = None,
    fully_contiguous: bool = False,
) -> specs.TensorSpec:
    target = system_config.current_target()

    if not dtype:
        dtype = draw(st.from_type(dtypes.Dtype))
    if not bank:
        bank = draw(st.sampled_from(sorted(target.system.banks)))
    assert dtype is not None and bank is not None

    if dim_sizes:
        rank = len(dim_sizes)
    else:
        rank = draw(st.integers(min_value=min_dims, max_value=max_dims))

    # TODO: Instead of rejecting when vector_shapes is empty, enum. vector shapes
    vector_shape = None
    if target.system.banks[bank].vector_rf:
        vector_value_cnt: int = target.system.banks[bank].vector_bytes // dtype.size
        outer_shape = dim_sizes if dim_sizes else (max_dim_size,) * rank
        all_shapes = list(
            impl.utils.gen_vector_shapes(
                outer_shape=outer_shape,
                dtype=dtype,
                vector_bytes=vector_value_cnt,
            )
        )
        hypothesis.assume(len(all_shapes))
        vector_shape = draw(st.sampled_from(all_shapes))

    if not dim_sizes:
        if vector_shape:
            dim_sizes = tuple(
                draw(dim_st(min_size=v, max_size=max_dim_size)) for v in vector_shape
            )
        else:
            dim_sizes = draw(
                st.lists(
                    dim_st(max_size=max_dim_size), min_size=rank, max_size=rank
                ).map(tuple)
            )
    assert dim_sizes
    assert all(d > 0 for d in dim_sizes)

    if layout_fn is None:
        layout = draw(layout_st(dim_sizes=dim_sizes, dtype=dtype))
    else:
        layout = layout_fn(len(dim_sizes))

    if contiguous_abs is None:
        if isinstance(layout, layouts.StandardLayout):
            if fully_contiguous:
                contiguous_abs = len(dim_sizes)
            else:
                trailing_ones = sum(
                    itertools.takewhile(
                        lambda s: s == 1,
                        (dim_sizes[i] for i in reversed(layout.dim_order)),
                    )
                )
                contiguous_abs = draw(
                    st.integers(min_value=trailing_ones, max_value=len(dim_sizes))
                )
        elif isinstance(layout, layouts.PackedLayout):
            if fully_contiguous or all(s == 1 for s in dim_sizes):
                contiguous_abs = len(dim_sizes) + 1
            else:
                contiguous_abs = draw(
                    st.integers(min_value=0, max_value=len(dim_sizes) + 1)
                )
        else:
            raise NotImplementedError()

    return target.tensor_spec(
        dim_sizes,
        dtype=dtype,
        contiguous_abs=contiguous_abs,
        layout=layout,
        bank=bank,
        vector_shape=vector_shape,
    )


@st.composite
def zero_spec_st(draw, max_dim_size: int = 128):
    t = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=1, max_dims=4))
    return specs.Zero(t, serial_only=draw(st.booleans()))


@st.composite
def matmul_spec_st(draw, max_dim_size: Optional[int] = 256, accum=False):
    lhs_dtype = draw(st.from_type(dtypes.Dtype))
    rhs_dtype = draw(st.from_type(dtypes.Dtype))
    lhs = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=2, max_dims=2))
    rhs_dim_sizes = (lhs.dim_sizes[1], draw(dim_st(max_size=max_dim_size)))
    rhs = draw(
        tensorspec_st(
            dim_sizes=rhs_dim_sizes,
            dtype=lhs_dtype,
            layout_fn=lambda _: draw(
                layout_st(dim_sizes=rhs_dim_sizes, dtype=lhs_dtype)
            ),
        )
    )
    out_dim_sizes = (lhs.dim_sizes[0], rhs.dim_sizes[1])
    out = draw(
        tensorspec_st(
            dim_sizes=out_dim_sizes,
            dtype=rhs_dtype,
            layout_fn=lambda _: draw(
                layout_st(dim_sizes=out_dim_sizes, dtype=rhs_dtype)
            ),
        )
    )
    t = specs.MatmulAccum if accum else specs.Matmul
    return t(lhs, rhs, out, serial_only=draw(st.booleans()))


@st.composite
def convolution_spec_st(draw, max_dim_size: Optional[int] = 32, accum=False):
    filters_dtype = draw(st.from_type(dtypes.Dtype))
    out_dtype = draw(st.from_type(dtypes.Dtype))
    image = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=4, max_dims=4))
    filters_dim_sizes = (
        draw(dim_st(max_size=max_dim_size)),
        image.dim_sizes[1],
        draw(dim_st(max_size=image.dim_sizes[2])),
        draw(dim_st(max_size=image.dim_sizes[3])),
    )
    filters = draw(tensorspec_st(dim_sizes=filters_dim_sizes, dtype=filters_dtype))
    out_dim_sizes = specs.Convolution.output_shape(image.dim_sizes, filters_dim_sizes)
    out = draw(tensorspec_st(dim_sizes=out_dim_sizes, dtype=out_dtype))
    t = specs.ConvolutionAccum if accum else specs.Convolution
    return t(image, filters, out, serial_only=draw(st.booleans()))


@st.composite
def reduce_spec_st(draw, max_dim_size: Optional[int] = 32, accum=False):
    source = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=2, max_dims=4))
    output_dim_sizes = source.dim_sizes[:-1]
    output = draw(tensorspec_st(dim_sizes=output_dim_sizes, dtype=source.dtype))
    t = specs.ReduceSumAccum if accum else specs.ReduceSum
    return t(source=source, output=output, serial_only=draw(st.booleans()))


@st.composite
def compose_spec_st(draw) -> specs.Compose:
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

    # assert len(output_dim_sizes), (
    #    f"output_dim_sizes was {output_dim_sizes} and inputs were "
    #    f"{[inp.dim_sizes for inp in inputs_specs]} for "
    #    f"({', '.join([c.__name__ for c in subspec_classes])})"
    # )
    hypothesis.assume(len(output_dim_sizes))

    output_spec = draw(tensorspec_st(dim_sizes=output_dim_sizes))
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
def tiling_chain_st(
    draw, *args, chain_len: Optional[int] = None, allow_conv=True, **kwargs
):
    base_tensor = draw(tensor_st(*args, **kwargs))

    if (
        len(base_tensor.spec.dim_sizes)
        < tensor.ConvolutionImageTile.minimum_image_rank()
    ):
        allow_conv = False

    if chain_len is None:
        chain_len = draw(st.integers(min_value=1, max_value=3))
    assert chain_len is not None

    chain = [base_tensor]
    for _ in range(chain_len):
        use_conv = draw(st.booleans()) if allow_conv else False
        tile_shape = _smaller_shape(draw, chain[-1].dim_sizes, chain[-1].vector_shape)
        hypothesis.assume(chain[-1].spec.layout.applies_to_shape(tile_shape))
        if use_conv:
            filter_shape = _smaller_shape(draw, tile_shape[1:])
            chain.append(chain[-1].spec.conv_image_tile(0, tile_shape, filter_shape))
        else:
            chain.append(chain[-1].spec.simple_tile(0, tile_shape))
    return chain


def _smaller_shape(
    draw, outer: Sequence[int], vector_shape: Optional[Sequence[int]] = None
) -> tuple[int, ...]:
    tile_shape = []
    if vector_shape:
        for dim, v in zip(outer, vector_shape):
            tile_shape.append(draw(st.integers(min_value=v, max_value=dim)))
    else:
        for dim in outer:
            tile_shape.append(draw(st.integers(min_value=1, max_value=dim)))
    return tuple(tile_shape)


atomic_specs_st = st.one_of(
    st.from_type(specs.Zero),
    st.from_type(specs.Matmul),
    st.from_type(specs.Convolution),
    st.from_type(specs.ReduceSum),
)

small_atomic_specs_st = st.one_of(
    zero_spec_st(max_dim_size=4),
    matmul_spec_st(max_dim_size=4),
    convolution_spec_st(max_dim_size=4),
    reduce_spec_st(max_dim_size=4),
)

def register_default_strategies():
    st.register_type_strategy(layouts.Layout, layout_st(numels_ones=False))
    st.register_type_strategy(dtypes.Dtype, dtype_st)
    st.register_type_strategy(specs.TensorSpec, tensorspec_st())

    st.register_type_strategy(tensor.Tensor, tensor_st())

    # Register default strategies for generating Specs.
    st.register_type_strategy(specs.Compose, compose_spec_st())
    st.register_type_strategy(specs.Zero, zero_spec_st())
    st.register_type_strategy(specs.Matmul, matmul_spec_st())
    st.register_type_strategy(specs.Convolution, convolution_spec_st())
    st.register_type_strategy(specs.ReduceSum, reduce_spec_st())
    st.register_type_strategy(
        specs.Spec,
        st.one_of(atomic_specs_st, st.from_type(specs.Compose)),
    )

    # Register default strategies for holes.
    st.register_type_strategy(
        impl.ComposeHole, compose_spec_st().map(impl.spec_to_hole)
    )
    # TODO: Add LoadHole and StoreHole
    # st.register_type_strategy(impl.LoadHole, load_spec_st().map(impl.spec_to_hole))
    # st.register_type_strategy(impl.StoreHole, store_spec_st().map(impl.spec_to_hole))
    st.register_type_strategy(impl.ZeroHole, zero_spec_st().map(impl.spec_to_hole))
    st.register_type_strategy(
        impl.MatmulHole, matmul_spec_st(accum=False).map(impl.spec_to_hole)
    )
    st.register_type_strategy(
        impl.MatmulAccumHole, matmul_spec_st(accum=True).map(impl.spec_to_hole)
    )
    st.register_type_strategy(
        impl.ConvHole, convolution_spec_st(accum=False).map(impl.spec_to_hole)
    )
    st.register_type_strategy(
        impl.ConvAccumHole, convolution_spec_st(accum=True).map(impl.spec_to_hole)
    )
    st.register_type_strategy(
        impl.ReduceSumHole, reduce_spec_st(accum=False).map(impl.spec_to_hole)
    )
    st.register_type_strategy(
        impl.ReduceSumAccumHole, reduce_spec_st(accum=True).map(impl.spec_to_hole)
    )

    # Register default strategies for generating Impls, including Holes, Movelets,
    # and Loops.
    st.register_type_strategy(impl.ComposeHole, composehole_op_st())
    st.register_type_strategy(impl.Pipeline, pipeline_op_st())
    st.register_type_strategy(
        impl.Impl,
        st.one_of(
            st.from_type(impl.ComposeHole),
            st.from_type(impl.Pipeline),
            st.from_type(impl.ConvHole),
            st.from_type(impl.ConvAccumHole),
            st.from_type(impl.MatmulHole),
            st.from_type(impl.MatmulAccumHole),
            st.from_type(impl.ReduceSumHole),
            st.from_type(impl.ReduceSumAccumHole),
            st.from_type(impl.Loop),
            st.from_type(impl.MoveLet),
        ),
    )
