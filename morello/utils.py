import functools
import itertools
import math
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable, Sequence, TypeVar

from . import tensor, layouts, system_config

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
    tile_cls: type, tile_shape: Sequence[int], parent: "specs.TensorSpec",
) -> bool:
    """Test whether a tiling breaks alignment.

    Returns `True` if every concrete tile in a tiling of a given-layout tensor
    has its first address on an aligned address. An aligned address is a
    multiple of `current_system().line_size`.

    Tiles must have the same layout as `parent` (the usual case). 

    It may report some aligned tilings as unaligned, but will never report an
    unaligned tiling as aligned.
    """
    # If the parent isn't contiguous or aligned, then we have no idea if
    # anything is aligned or not.
    if not parent.contiguous or not parent.aligned:
        return False

    line_size = system_config.current_system().line_size

    if isinstance(parent.layout, layouts.StandardLayout) and issubclass(
        tile_cls, tensor.SimpleTile
    ):
        # We want to know if a step in any tiling direction results in a delta
        # that is not a multiple of the line size. In the case of a regular
        # layout and tiling, this is the same as checking that the cumulative
        # tile dimensions (times bytes) are multiples of the line, ignoring
        # dimensions which will never be advanced (tile dim = parent dim).
        cum_inner_volume = 1
        for physical_dim_idx in reversed(parent.layout.dim_order):
            step_values = cum_inner_volume * tile_shape[physical_dim_idx]
            cum_inner_volume *= parent.dim_sizes[physical_dim_idx]
            # Skip dimensions over which we don't iterate.
            if parent.dim_sizes[physical_dim_idx] == tile_shape[physical_dim_idx]:
                continue 
            if step_values * parent.dtype.size % line_size != 0:
                return False
        return True
    else:
        warnings.warn(
            f"No alignment heuristic support for {tile_cls.__name__} and "
            f"{parent.layout.__class__.__name__}; assuming unaligned"
        )
        return False


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
