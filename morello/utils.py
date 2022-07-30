import functools
import itertools
import math
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable, Sequence, TypeVar

from . import layouts, system_config

if TYPE_CHECKING:
    from . import specs

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


def aligned_approx(
    tile_shape: Sequence[int], tile_layout: "layouts.Layout", parent: "specs.TensorSpec"
) -> bool:
    """Test whether a tiling breaks alignment.

    It may report some aligned tilings as unaligned, but may never report an unaligned
    tiling as aligned.
    """
    if parent.layout != tile_layout:
        raise ValueError(
            f"Only supports same-layout TensorSpecs, but given"
            f" {parent.layout} and {tile_layout}"
        )
    if not isinstance(tile_layout, layouts.StandardLayout):
        warnings.warn(
            f"No alignment heuristic support for {tile_layout.__class__.__name__}; "
            "assuming unaligned"
        )
        return False

    # Use a simple, under-approximating strategy. If any dimension would break alignment
    # were that the physically innermost dimension, then this function returns `False`.
    if not parent.aligned:
        return False
    line_size = system_config.current_system().line_size
    return not any(d % line_size for d in tile_shape)


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
