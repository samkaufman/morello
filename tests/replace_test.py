import morello.impl.matmuls
from morello import dtypes, replace, system_config


def test_replace_origin_of_tiles():
    target = system_config.current_target()
    origin = target.tensor(
        target.tensor_spec((10, 10), dtype=dtypes.Uint32, bank="GL"),
        name="origin",
    )
    l_and_r = origin.simple_tile((4, 4))
    output = origin.simple_tile((4, 4))
    imp = morello.impl.matmuls.MatmulHole(l_and_r, l_and_r, output, serial_only=False)

    new_origin = target.tensor(
        target.tensor_spec((20, 20), dtype=dtypes.Uint32, bank="GL"),
        name="new_origin",
    ).simple_tile((10, 10))
    new_impl = replace.replace(imp, {origin: new_origin})
    assert new_impl.lhs is new_impl.rhs
    assert new_impl.lhs.origin == new_origin
    assert new_impl.output.origin == new_origin


def test_replace_is_no_op_with_no_real_changes():
    target = system_config.current_target()
    origin = target.tensor(
        target.tensor_spec((10, 10), dtype=dtypes.Uint32, bank="GL"),
        name="origin",
    )
    lhs = origin.simple_tile((4, 4))
    rhs = origin.simple_tile((4, 4))
    output = origin.simple_tile((4, 4))
    disconnected_tile = origin.simple_tile((1, 1))
    imp = morello.impl.matmuls.MatmulHole(lhs, rhs, output, serial_only=False)
    new_impl = replace.replace(imp, {disconnected_tile: origin})
    assert new_impl == imp
