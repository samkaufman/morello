import functools
import itertools
import math
from collections.abc import Mapping
from typing import Iterable, TypeVar

from . import specs, tensor
from .tensor import TensorLike

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


def contiguous(t, address_root):
    if isinstance(t, tensor.TensorLike):
        self_ordered_dims = layout_ordered_dims(t)
    else:
        self_ordered_dims = layout_ordered_dims(*t)

    if isinstance(address_root, tensor.TensorLike):
        address_root_ordered_dims = layout_ordered_dims(address_root)
    else:
        address_root_ordered_dims = layout_ordered_dims(*address_root)

    pairs = zip(self_ordered_dims, address_root_ordered_dims)

    # Drop leading dimensions where the tile size is one
    pairs = itertools.dropwhile(lambda x: x[0] == 1, pairs)

    # Drop the first
    pairs = itertools.islice(pairs, 1, None)

    for tile_dim_size, root_dim_size in pairs:
        # The following includes the case where an underlying dimension is 1.
        if tile_dim_size != root_dim_size:
            return False
    return True


def layout_ordered_dims(*args) -> tuple[int, ...]:
    """Returns tuple of operand's height and width; or vice versa if column-major."""
    dim_sizes: tuple[int, ...]
    root_layout: specs.Layout
    if len(args) == 1 and isinstance(args[0], TensorLike):
        dim_sizes, root_layout = args[0].dim_sizes, args[0].root.layout
    elif len(args) == 2:
        dim_sizes, root_layout = args
    else:
        raise TypeError("Unknown arguments")

    if len(dim_sizes) == 1:
        return (dim_sizes[0],)
    if root_layout == specs.Layout.ROW_MAJOR:
        lead = [dim_sizes[0], dim_sizes[1]]
    elif root_layout == specs.Layout.COL_MAJOR:
        lead = [dim_sizes[1], dim_sizes[0]]
    else:
        raise NotImplementedError(f"Unknown layout {root_layout}")
    lead.extend(dim_sizes[2:])
    return tuple(lead)


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
