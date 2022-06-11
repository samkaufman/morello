import functools
import itertools
import math
from collections.abc import Mapping
from typing import Iterable, Sequence, TypeVar

from morello.specs.tensorspec import TensorSpec

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


def contiguous(t, address_root) -> bool:
    """Test whether a tensor is contiguous in another tensor's address space."""
    if isinstance(t, specs.TensorSpec):
        self_ordered_dims = layout_ordered_dims(t)
    elif isinstance(t, Sequence):
        self_ordered_dims = layout_ordered_dims(*t)
    else:
        raise TypeError(f"Unexpected first argument type: {type(t).__name__}")

    if isinstance(address_root, specs.TensorSpec):
        address_root_ordered_dims = layout_ordered_dims(address_root)
    elif isinstance(t, Sequence):
        address_root_ordered_dims = layout_ordered_dims(*address_root)
    else:
        raise TypeError(f"Unexpected second argument type: {type(t).__name__}")

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


def contiguous_approx(
    tile_shape: Sequence[int], tile_layout: layouts.Layout, parent: TensorSpec
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
    if all(d == 1 for d in tile_shape):
        return True
    if not parent.contiguous:
        return False
    return contiguous((tile_shape, tile_layout), parent)


# TODO: Can we merge this into Layout?
def layout_ordered_dims(*args) -> tuple[int, ...]:
    """Returns tuple of operand's height and width; or vice versa if column-major."""
    dim_sizes: tuple[int, ...]
    root_layout: layouts.Layout
    if len(args) == 1 and isinstance(args[0], specs.TensorSpec):
        dim_sizes, root_layout = args[0].dim_sizes, args[0].layout
    elif len(args) == 2:
        dim_sizes, root_layout = args
    else:
        raise TypeError(f"Unexpected arguments: {args}")

    if len(dim_sizes) == 1:
        return (dim_sizes[0],)
    if root_layout == layouts.ROW_MAJOR:
        lead = [dim_sizes[0], dim_sizes[1]]
    elif root_layout == layouts.COL_MAJOR:
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
