import functools
import itertools
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable, Sequence, TypeVar

if TYPE_CHECKING:
    from . import layouts, specs

T = TypeVar("T")
U = TypeVar("U")

_ALPHABET = list(map(chr, range(97, 123)))
ALPHABET_PRODUCT = _ALPHABET + [
    a + b for a, b in itertools.product(_ALPHABET, _ALPHABET)
]


def zip_dict(
    first: Mapping[T, U], *others: Mapping[T, U], same_keys: bool = False
) -> dict[T, tuple[U, ...]]:
    keys = set(first)
    if same_keys:
        for d in others:
            if set(d) != keys:
                raise ValueError(f"Keys did not match: {set(d)} and {keys}")
    else:
        keys.intersection_update(*others)
    result = {}
    for key in keys:
        result[key] = (first[key],) + tuple(d[key] for d in others)
    return result


def flatten(src):
    """Flattens nested iterables and ndarrays."""
    if hasattr(src, "tolist"):
        src = src.tolist()
    for el in src:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def contiguous_approx(
    tile_shape: Sequence[int], tile_layout: "layouts.Layout", parent: "specs.TensorSpec"
) -> bool:
    """Test whether a tiling breaks contiguousness.

    This is usually used in place of `contiguous`. It has the benefit of being
    compositional in Impl, but cannot "recover" contiguousness in cases where
    a tiling degenerates a dimension (e.g., [4, 4] -> [4, 2] -> [1, 2]), unless
    the new tile has just one value.

    :param tile_shape: The shape of the new tile to test.
    :param tile_layout: The layout of the new tile to test.
    :param parent: The spec of the tensorlike being tiled.
    :returns: `True` if both the parent and its new tile can be determined to be
      contiguous. `False` otherwise.
    """
    if parent.layout != tile_layout:
        raise ValueError(
            f"Only supports same-layout TensorSpecs, but given"
            f" {parent.layout} and {tile_layout}"
        )
    if all(d == 1 for d in tile_shape):
        return True
    if not parent.contiguous:
        return False
    return tile_layout.check_tile_contiguity(parent.dim_sizes, tile_shape)


def factors(n: int) -> Iterable[int]:
    """Returns the factors of an integer, in ascending order.

    Implementation taken from https://stackoverflow.com/a/6800214.
    """
    return sorted(
        set(
            functools.reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
            )
        )
    )


def next_power_of_two(x: int) -> int:
    """Return next highest power of 2, or self if a power of two or zero."""
    if x == 0:
        return 0
    assert x >= 1, f"x must be 1 or greater; was: {x}"
    result = int(2 ** math.ceil(math.log2(x)))
    assert result >= x
    return result
