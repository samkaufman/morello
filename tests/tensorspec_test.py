import pytest

from morello import dtypes, tensor, layouts, system_config

@pytest.mark.parametrize(
    "tensor_shape, tile_shape, expected",
    [
        ((8, 8), (8, 8), True),
        ((8, 8, 8), (8, 8, 8), True),
        ((8, 8, 8), (4, 8, 8), True),
    ],
)
def test_tile_contiguous(tensor_shape, tile_shape, expected):
    # TODO: Vary the following three parameters with hypothesis
    target = system_config.current_target()
    dtype, bank, layout = dtypes.Uint8, "RF", layouts.ROW_MAJOR
    tensor_spec = target.tensor_spec(tensor_shape, dtype, True, bank, layout)
    tile = tensor_spec.simple_tile(tensor.OperandIdx(0), tile_shape)
    assert tile.spec.contiguous == expected

