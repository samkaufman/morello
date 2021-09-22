from typing import Optional, Tuple

from hypothesis import strategies as st

from morello import dtypes, ops, specs, system_config, tensor
from morello.system_config import cpu, hexagon

target_st = st.sampled_from([cpu.CpuTarget(), hexagon.HvxSimulatorTarget()])

dtype_st = st.sampled_from([dtypes.Uint8, dtypes.Uint32])


@st.composite
def dim_st(draw, max_size: Optional[int] = None) -> int:
    if max_size is None:
        return draw(st.integers(min_value=1))
    return draw(st.integers(min_value=1, max_value=max_size))


@st.composite
def layout_st(draw, dim_sizes: Optional[Tuple[int, ...]] = None) -> specs.Layout:
    if all(d == 1 for d in dim_sizes):
        return specs.Layout.ROW_MAJOR
    return draw(st.from_type(specs.Layout))


@st.composite
def tensorspec_st(
    draw,
    max_dim_size: Optional[int] = 32,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
) -> specs.TensorSpec:
    system = system_config.current_system()
    dim_sizes = draw(
        st.lists(
            dim_st(max_size=max_dim_size), min_size=min_dims, max_size=max_dims
        ).map(tuple)
    )
    return specs.TensorSpec(
        dim_sizes,
        dtype=draw(st.from_type(dtypes.Dtype)),
        layout=draw(layout_st(dim_sizes=dim_sizes)),
        bank=draw(st.sampled_from(sorted(system.banks))),
    )


@st.composite
def matmul_spec_st(draw, max_dim_size: Optional[int] = 32):
    system = system_config.current_system()
    lhs = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=2, max_dims=2))
    rhs_dim_sizes = (lhs.dim_sizes[1], draw(dim_st(max_size=max_dim_size)))
    rhs = specs.TensorSpec(
        dim_sizes=rhs_dim_sizes,
        dtype=draw(st.from_type(dtypes.Dtype)),
        layout=draw(layout_st(dim_sizes=rhs_dim_sizes)),
        bank=draw(st.sampled_from(sorted(system.banks))),
    )
    out_dim_sizes = (lhs.dim_sizes[0], rhs.dim_sizes[1])
    out = specs.TensorSpec(
        dim_sizes=out_dim_sizes,
        dtype=draw(st.from_type(dtypes.Dtype)),
        layout=draw(layout_st(dim_sizes=out_dim_sizes)),
        bank=draw(st.sampled_from(sorted(system.banks))),
    )
    return specs.Matmul(lhs, rhs, out, serial_only=draw(st.booleans()))


@st.composite
def convolution_spec_st(draw, max_dim_size: Optional[int] = 32):
    system = system_config.current_system()
    image = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=2, max_dims=2))
    filters_dim_sizes = (
        draw(dim_st(max_size=image.dim_sizes[0])),
        draw(dim_st(max_size=image.dim_sizes[1])),
        draw(dim_st(max_size=max_dim_size)),
    )
    filters = specs.TensorSpec(
        dim_sizes=filters_dim_sizes,
        dtype=draw(st.from_type(dtypes.Dtype)),
        layout=draw(layout_st(dim_sizes=filters_dim_sizes)),
        bank=draw(st.sampled_from(sorted(system.banks))),
    )
    out_dim_sizes = specs.Convolution.output_shape(image.dim_sizes, filters_dim_sizes)
    out = specs.TensorSpec(
        dim_sizes=out_dim_sizes,
        dtype=draw(st.from_type(dtypes.Dtype)),
        layout=draw(layout_st(dim_sizes=out_dim_sizes)),
        bank=draw(st.sampled_from(sorted(system.banks))),
    )
    return specs.Convolution(image, filters, out)


@st.composite
def reduce_spec_st(draw, max_dim_size: Optional[int] = 32):
    system = system_config.current_system()
    source = draw(tensorspec_st(max_dim_size=max_dim_size, min_dims=2, max_dims=4))
    output_dim_sizes = source.dim_sizes[:-1]
    output = specs.TensorSpec(
        dim_sizes=output_dim_sizes,
        dtype=draw(st.from_type(dtypes.Dtype)),
        layout=draw(layout_st(dim_sizes=output_dim_sizes)),
        bank=draw(st.sampled_from(sorted(system.banks))),
    )
    return specs.ReduceSum(source=source, output=output)


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
    output_spec = specs.TensorSpec(
        dim_sizes=output_dim_sizes,
        dtype=draw(st.from_type(dtypes.Dtype)),
        bank=draw(st.sampled_from(sorted(system_config.current_system().banks))),
        layout=draw(layout_st(output_dim_sizes)),
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
    )


@st.composite
def composehole_op_st(draw) -> ops.ComposeHole:
    spec = draw(compose_spec_st())
    return ops.ComposeHole(
        spec=spec,
        inputs=tuple(tensor.Tensor(s, name=None) for s in spec.inputs),
        output=tensor.Tensor(spec.output, name="output"),
    )


@st.composite
def pipeline_op_st(draw) -> ops.Pipeline:
    raise NotImplementedError()


def register_default_strategies():
    st.register_type_strategy(dtypes.Dtype, dtype_st)
    st.register_type_strategy(system_config.Target, target_st)
    st.register_type_strategy(specs.TensorSpec, tensorspec_st())

    st.register_type_strategy(
        tensor.Tensor,
        st.builds(
            tensor.Tensor,
            spec=st.from_type(specs.TensorSpec),
            name=st.one_of(st.none(), st.text(min_size=1)),
        ),
    )

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

    st.register_type_strategy(ops.ComposeHole, composehole_op_st())
    st.register_type_strategy(ops.Pipeline, pipeline_op_st())

    st.register_type_strategy(
        ops.Schedule,
        st.one_of(
            st.from_type(ops.ComposeHole),
            st.from_type(ops.Pipeline),
            st.from_type(ops.DirectConv),
            st.from_type(ops.ReduceSum),
            st.from_type(ops.Loop),
            st.from_type(ops.MatmulSplitLoop),
            st.from_type(ops.MoveLet),
        ),
    )
