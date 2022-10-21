import functools
import itertools
import math
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Sequence, TypeVar

from . import layouts, system_config, tensor

if TYPE_CHECKING:
    from . import specs

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

_ALPHABET = list(map(chr, range(97, 123)))
ALPHABET_PRODUCT = _ALPHABET + [
    a + b for a, b in itertools.product(_ALPHABET, _ALPHABET)
]


class TinyMap(Mapping[T, U]):
    __slots__ = ("raw_keys", "raw_values")

    raw_keys: tuple[T, ...]
    raw_values: tuple[U, ...]

    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], TinyMap):
            self.raw_keys = args[0].raw_keys
            self.raw_values = args[0].raw_values
        elif len(args) == 1:
            self.raw_keys = tuple(args[0].keys())
            self.raw_values = tuple(args[0][k] for k in self.raw_keys)
        else:
            keys, values = args
            self.raw_keys = keys
            self.raw_values = values
        assert len(self.raw_keys) == len(self.raw_values)

    def __hash__(self) -> int:
        # Just use the values. In most cases, the keys will be the same.
        return hash(self.raw_values)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TinyMap):
            return NotImplemented
        return self.raw_keys == other.raw_keys and self.raw_values == other.raw_values

    def __len__(self) -> int:
        return len(self.raw_keys)

    def __iter__(self) -> Iterator[T]:
        return iter(self.raw_keys)

    def __getitem__(self, key: T) -> U:
        try:
            idx = self.raw_keys.index(key)
        except ValueError:
            raise KeyError(key)
        return self.raw_values[idx]

    def map_values(self, f: Callable[[U], V]) -> "TinyMap[T, V]":
        return TinyMap(self.raw_keys, tuple(f(v) for v in self.raw_values))

    def replace_value(self, key: T, new_value: U) -> "TinyMap[T, U]":
        index = self.raw_keys.index(key)
        return TinyMap(
            self.raw_keys,
            tuple(
                new_value if i == index else v for i, v in enumerate(self.raw_values)
            ),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.raw_keys)}, {repr(self.raw_values)})"

    def __str__(self) -> str:
        return repr(self)


def zip_dict(
    first: Mapping[T, U], *others: Mapping[T, U], same_keys: bool = False
) -> Iterable[tuple[T, tuple[U, ...]]]:
    # TODO: Just remove the kwargs
    assert same_keys

    # The case where everything is a TinyMap with the same keys can be quick.
    # This should be common: TinyMaps often use a reference to the target's
    # ordered_banks as their keys.
    if (
        isinstance(first, TinyMap)
        and all(isinstance(o, TinyMap) for o in others)
        and all(first.raw_keys == o.raw_keys for o in others)  # type: ignore
    ):
        yield from zip(
            first.raw_keys, zip(first.raw_values, *(o.raw_values for o in others))  # type: ignore
        )
        return

    keys = set(first)
    for d in others:
        if set(d) != keys:
            raise ValueError(f"Keys did not match: {set(d)} and {keys}")
    for key in keys:
        yield key, (first[key],) + tuple(d[key] for d in others)


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
    tile_cls: type,
    tile_shape: Sequence[int],
    parent: "specs.TensorSpec",
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

    if isinstance(parent.layout, layouts.StandardLayout) and issubclass(
        tile_cls, tensor.SimpleTile
    ):
        return _aligned_approx_standard_simple(
            tile_shape, parent.layout, parent.dim_sizes, parent.dtype
        )
    elif isinstance(parent.layout, layouts.PackedLayout) and issubclass(
        tile_cls, tensor.SimpleTile
    ):
        tile_expanded = parent.layout.expand_shape(tile_shape)
        parent_expanded = parent.layout.expand_shape(parent.dim_sizes)
        return _aligned_approx_standard_simple(
            tile_expanded,
            layouts.row_major(len(tile_expanded)),
            parent_expanded,
            parent.dtype,
        )
    elif issubclass(tile_cls, tensor.ConvolutionImageTile):
        # A ConvolutionImageTile defines the first dimension to be a batch dimension;
        # equivalent to a SimpleTile over that dimension. If that's the only dimension
        # that's changed, we can treat it as a SimpleTile.
        if all(a == b for a, b in zip(tile_shape[1:], parent.dim_sizes[1:])):
            return aligned_approx(tensor.SimpleTile, tile_shape, parent)
        else:
            warnings.warn("No alignment analysis for non-batch convolution")
            return False
    else:
        warnings.warn(
            f"No alignment analysis for {tile_cls.__name__} and "
            f"{parent.layout.__class__.__name__}; assuming unaligned"
        )
        return False


def _aligned_approx_standard_simple(
    tile_shape: Sequence[int],
    parent_layout: layouts.StandardLayout,
    parent_shape: Sequence[int],
    parent_dtype,
) -> bool:
    line_size = system_config.current_system().line_size

    # We want to know if a step in any tiling direction results in a delta
    # that is not a multiple of the line size. In the case of a regular
    # layout and tiling, this is the same as checking that the cumulative
    # tile dimensions (times bytes) are multiples of the line, ignoring
    # dimensions which will never be advanced (tile dim = parent dim).
    cum_inner_volume = 1
    for physical_dim_idx in reversed(parent_layout.dim_order):
        step_values = cum_inner_volume * tile_shape[physical_dim_idx]
        cum_inner_volume *= parent_shape[physical_dim_idx]
        # Skip dimensions over which we don't iterate.
        if parent_shape[physical_dim_idx] == tile_shape[physical_dim_idx]:
            continue
        if step_values * parent_dtype.size % line_size != 0:
            return False
    return True


def factors(n: int) -> list[int]:
    """Returns the factors of an integer, in ascending order.

    Implementation taken from https://stackoverflow.com/a/6800214.
    """
    if n == 0:
        return []
    return sorted(
        set(
            functools.reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )
    )


def powers_of_two(maximum: int) -> Iterable[int]:
    """Yields positive powers of two up to maximum, exclusive."""
    return itertools.takewhile(
        lambda v: v < maximum, (2**i for i in itertools.count())
    )


def next_power_of_two(x: int) -> int:
    """Return next highest power of 2, or `x` if a power of two or zero."""
    if x == 0:
        return 0
    assert x >= 1, f"x must be 1 or greater; was: {x}"
    result = int(2 ** math.ceil(math.log2(x)))
    assert result >= x
    return result


def sum_seqs(maxes: Sequence[int], total: int) -> Iterable[tuple[int, ...]]:
    """Return strictly positive integer tuples which sum to `total`, bounded by `maxes`.

    All returned tuples are the same length as `maxes`.

    This is an enumeration of the compositions of the integer `total` with bounded
    parts.
    """
    if len(maxes) == 0:
        return
    elif len(maxes) == 1:
        if maxes[0] >= total:
            yield (total,)
    else:
        # The most it'll be able to do is `start`, so we'll begin with that min.
        obligation = max(total - sum(maxes[1:]), 0)
        for v in range(obligation, min(maxes[0], total) + 1):
            for suffix in sum_seqs(maxes[1:], total - v):
                yield (v,) + suffix