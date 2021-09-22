from morello import dtypes, tensor, ops, specs, replace


def test_replace_origin_of_tiles():
    origin = tensor.Tensor(
        specs.TensorSpec((10, 10), dtype=dtypes.Uint32, bank="GL"),
        name="origin",
        origin=None,
    )
    l_and_r = origin.simple_tile((4, 4))
    output = origin.simple_tile((4, 4))
    impl = ops.MatmulHole(l_and_r, l_and_r, output, serial_only=False)

    new_origin = tensor.Tensor(
        specs.TensorSpec((20, 20), dtype=dtypes.Uint32, bank="GL"),
        name="new_origin",
        origin=None,
    ).simple_tile((10, 10))
    new_impl = replace.replace(impl, {origin: new_origin})
    assert new_impl.lhs is new_impl.rhs
    assert new_impl.lhs.origin == new_origin
    assert new_impl.output.origin == new_origin


def test_replace_is_no_op_with_no_real_changes():
    origin = tensor.Tensor(
        specs.TensorSpec((10, 10), dtype=dtypes.Uint32, bank="GL"),
        name="origin",
        origin=None,
    )
    lhs = origin.simple_tile((4, 4))
    rhs = origin.simple_tile((4, 4))
    output = origin.simple_tile((4, 4))
    disconnected_tile = origin.simple_tile((1, 1))
    impl = ops.MatmulHole(lhs, rhs, output, serial_only=False)
    new_impl = replace.replace(impl, {disconnected_tile: origin})
    assert new_impl == impl
