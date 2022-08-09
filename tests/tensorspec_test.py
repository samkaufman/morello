import pytest

from morello import dtypes, layouts, system_config, tensor


@pytest.mark.parametrize(
    "tensor_shape, tile_shape, expected",
    [((8, 8), (8, 8), 2), ((8, 8, 8), (8, 8, 8), 3), ((8, 8, 8), (4, 8, 8), 3),],
)
def test_tile_contiguous(tensor_shape, tile_shape, expected):
    # TODO: Vary the following three parameters with hypothesis
    target = system_config.current_target()
    dtype, bank, layout = dtypes.Uint8, "RF", layouts.row_major(len(tensor_shape))
    tensor_spec = target.tensor_spec(
        tensor_shape, dtype, len(tensor_shape), bank, layout
    )
    tile = tensor_spec.simple_tile(tensor.OperandIdx(0), tile_shape)
    assert tile.spec.contiguous == expected
