from typing import Iterable

import itertools
import re
import hypothesis
import pytest
from hypothesis import strategies as st

from morello import dtypes, layouts, tiling, specs, system_config, tensor, utils

from . import strategies
from . import utils as test_utils

strategies.register_default_strategies()

_POINT_SYMBOL_RE = re.compile(r"^p(\d+)$")


def _shape_ids(arg):
    if isinstance(arg, tuple) and arg and all(isinstance(a, int) for a in arg):
        return "x".join(str(a) for a in arg)
    return str(arg)


@pytest.mark.parametrize(
    "tile_shape, parent_shape, expected",
    [
        ((1,), (1,), True),
        ((1,), (2,), False),
        ((1, 2), (1, 2), True),
        ((16,), (64,), False),
        ((32,), (64,), True),
        ((64,), (64,), True),
        ((64,), (96,), True),
        ((64,), (128,), True),
        ((128,), (256,), True),
        ((2, 16), (2, 64,), False,),
        ((2, 32), (2, 64,), True,),
        ((64, 1), (64, 64), False),
        ((1, 64), (2, 96,), True,),
    ],
    ids=_shape_ids,
)
def test_simpletile_regularlayout_alignment(tile_shape, parent_shape, expected):
    parent = specs.TensorSpec(
        parent_shape, dtypes.Uint8, contiguous_abs=len(parent_shape)
    )
    assert system_config.current_system().line_size == 32
    assert (
        utils.aligned_approx(tiling.SimpleTile, tile_shape, parent=parent) == expected
    )


def test_convtile_nhwc_alignment():
    tile_shape = (1, 4, 512, 512)
    parent_shape = (4, 4, 512, 512)
    expected = True

    parent = specs.TensorSpec(
        parent_shape,
        dtypes.Uint8,
        contiguous_abs=len(parent_shape),
        layout=layouts.NHWC,
    )
    assert system_config.current_system().line_size == 32
    assert (
        utils.aligned_approx(tiling.ConvolutionImageTile, tile_shape, parent=parent)
        == expected
    )


# TODO: Check both under-approximation and exact.
# TODO: Check intermediate tilings too.


@hypothesis.settings(deadline=1000, max_examples=20000)
@hypothesis.given(data=st.data())
@pytest.mark.parametrize(
    "test_conv",
    [False, pytest.param(True, marks=pytest.mark.skip)],
    ids=["simple", "conv"],
)
def test_alignment_approximation_is_correct(test_conv: bool, data):
    tile_chain = data.draw(
        strategies.tiling_chain_st(
            chain_len=1, max_dim_size=96, max_dims=4, allow_conv=test_conv
        )
    )
    if test_conv:
        hypothesis.assume(
            any(isinstance(t, tensor.ConvolutionImageTile) for t in tile_chain)
        )

    parent, final_tile = tile_chain
    val_size = final_tile.spec.dtype.size
    hypothesis.note(f"Parent is {parent}")
    hypothesis.note(f"          {repr(parent)}")
    hypothesis.note(f"Tile is {final_tile}")
    hypothesis.note(f"        {repr(final_tile)}")

    if parent.spec.aligned:
        hypothesis.event("parent is aligned")
    else:
        hypothesis.event("parent is unaligned")
    if final_tile.spec.aligned:
        hypothesis.event("innermost tile is aligned")
    else:
        hypothesis.event("innermost tile is unaligned")

    tile_coords: Iterable[tuple[int, ...]] = itertools.product(
        *(
            range(final_tile.steps_dim(dim_idx, dim_size))
            for dim_idx, dim_size in enumerate(parent.dim_sizes)
        )
    )

    if not parent.spec.aligned or not parent.spec.contiguous:
        assert not final_tile.spec.aligned
    else:
        assert final_tile.spec.aligned == _compute_alignment(
            tile_chain, val_size, tile_coords
        )


def _compute_alignment(tile_chain, val_size, tile_coords):
    is_aligned = True
    for tile_coord in tile_coords:
        idx_expr = test_utils.compose_indexing_exprs(tile_chain, [tile_coord])
        assert all(_POINT_SYMBOL_RE.match(s.name) for s in idx_expr.free_symbols), (
            f"Expected idx_expr to contain only point symbols after index symbol "
            f"substitution, but was: {idx_expr}"
        )

        # Zero all remaining symbols
        offset = int(idx_expr.subs((s, 0) for s in idx_expr.free_symbols)) * val_size
        if offset % system_config.current_system().line_size != 0:
            is_aligned = False
            break
    return is_aligned
