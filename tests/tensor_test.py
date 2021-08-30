import io
import pickle

import hypothesis
from hypothesis import strategies as st

from morello import specs, tensor

from . import strategies

strategies.register_default_strategies()


@hypothesis.given(st.from_type(specs.TensorSpec), st.text(), st.booleans())
def test_tensors_and_tiles_can_be_pickled_and_unpickled_losslessly(
    spec, name, should_tile
):
    t = tensor.Tensor(spec=spec, name=name)
    if should_tile:
        t = t.simple_tile(tuple(1 for _ in t.dim_sizes))

    buf = io.BytesIO()
    pickle.dump(t, buf)
    buf.seek(0)
    read_tensor = pickle.load(buf)
    # TODO: Add a deep equality method
    assert str(t) == str(read_tensor)
