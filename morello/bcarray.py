import itertools
from typing import Iterable, Sequence

import numpy as np


class BlockCompressedArray:
    """A block-compressed, multi-dimensional array.

    Implements a subset of the numpy.ndarray interface.

    Blocks which contain only the same value will be represented as a single
    value in memory.
    """

    grid: np.ndarray

    def __init__(
        self,
        shape: tuple[int, ...],
        block_shape: tuple[int, ...],
        default_value=None,
        dense_block_threshold: int = 4,
        compress_on_fill: bool = False,
    ):
        """Create a new BlockCompressedArray.

        Args:
            shape: The shape of the array.
            block_shape: The maximum shape of each block.
        """
        if isinstance(default_value, (np.ndarray, list)):
            raise ValueError("default_value can not be an ndarray or list")
        self.shape = shape
        self.block_shape = block_shape
        self.default_value = default_value
        self.dense_block_threshold = dense_block_threshold
        self.compress_on_fill = compress_on_fill
        self.grid = np.full(
            tuple((o + b - 1) // b for b, o in zip(block_shape, shape)),
            fill_value=default_value,
            dtype=object,
        )

    def __getitem__(self, pt):
        """Get a value from the array."""
        block_pt = tuple(p // b for p, b in zip(pt, self.block_shape))
        block = self.grid[block_pt]
        if isinstance(block, np.ndarray):
            adjusted_pt = tuple(p % b for p, b in zip(pt, self.block_shape))
            return block[adjusted_pt]
        elif isinstance(block, list):
            # Walk over the list in reverse order so the latest-added entry
            # has priority when there is overlap.
            adjusted_pt = tuple(p % b for p, b in zip(pt, self.block_shape))
            for rng, value in reversed(block):
                if all(l <= p < u for l, p, u in zip(rng[0], adjusted_pt, rng[1])):
                    return value
            raise IndexError()
        else:
            return block

    def count_values(self) -> int:
        """Count the number of stored values.

        Compressed blocks are counted as 1.
        """
        result = 0
        for block_pt in np.ndindex(self.grid.shape):
            block = self.grid[block_pt]
            if isinstance(block, np.ndarray):
                result += block.size
            elif isinstance(block, list):
                result += len(block)
            else:
                result += 1
        return result

    def fill(self, value) -> None:
        """Fill the array with a value."""
        self.grid.fill(value)

    def fill_range(self, lower: Sequence[int], upper: Sequence[int], value) -> None:
        """Fill a (hyper-)rectangular sub-region of the array with a value.

        :param lower: The lower coordinate of the region to fill.
        :param upper: The lower coordinate of the region to fill.
        :param value: The value with which to fill the region.
        """
        if any(l < 0 for l in lower) or any(u > b for u, b in zip(upper, self.shape)):
            raise IndexError(
                f"Range given by {lower} and {upper} is out of bounds for a "
                f"{type(self).__name__} of shape {self.shape}"
            )
        if not all(l < u for l, u in zip(lower, upper)):
            return

        # Fill all blocks which are fully contained within the range.
        block_enclosed_lower = []
        lower_misaligned_dim_idxs: list[int] = []
        for i, (l, b) in enumerate(zip(lower, self.block_shape)):
            block_enclosed_lower.append((l + b - 1) // b)
            if l % b != 0:
                lower_misaligned_dim_idxs.append(i)

        block_enclosed_upper = []
        upper_misaligned_dim_idxs: list[int] = []
        for i, (u, b) in enumerate(zip(upper, self.block_shape)):
            block_enclosed_upper.append(u // b)
            if u % b != 0:
                upper_misaligned_dim_idxs.append(i)

        for block_pt in itertools.product(
            *[range(l, u) for l, u in zip(block_enclosed_lower, block_enclosed_upper)]
        ):
            self.grid[block_pt] = value

        # Fill regions in the boundary blocks.
        for block_pt in _surface_pts(
            block_enclosed_lower,
            block_enclosed_upper,
            lower_misaligned_dim_idxs,
            upper_misaligned_dim_idxs,
        ):
            block_lower = tuple(l * b for l, b in zip(block_pt, self.block_shape))
            block_upper = tuple((l + 1) * b for l, b in zip(block_pt, self.block_shape))
            global_intersection = _intersect(block_lower, block_upper, lower, upper)
            block_intersection: tuple[tuple[int, ...], tuple[int, ...]] = tuple(
                tuple(v - l for v, l in zip(coord, block_lower))
                for coord in global_intersection
            )
            if isinstance(self.grid[block_pt], np.ndarray):
                spt = tuple(slice(a, b) for a, b in zip(*block_intersection))
                self.grid[block_pt][spt].fill(value)
                if self.compress_on_fill:
                    self.grid[block_pt] = _compress_block(self.grid[block_pt])
            elif isinstance(self.grid[block_pt], list):
                _drop_covered_entries(block_intersection, self.grid[block_pt])
                self.grid[block_pt].append((block_intersection, value))
                if len(self.grid[block_pt]) >= self.dense_block_threshold:
                    self._convert_list_block_to_ndarray(block_pt)
            else:
                # Bail here if `value` matches the original value.
                if self.grid[block_pt] == value:
                    continue
                # Turn a single-value block into a list.
                block_shape = self._block_shape_at_point(block_pt)
                self.grid.itemset(
                    block_pt,
                    [
                        (((0,) * len(block_shape), block_shape), self.grid[block_pt]),
                        (block_intersection, value),
                    ],
                )

    def _convert_list_block_to_ndarray(self, block_pt):
        assert isinstance(self.grid[block_pt], list)
        new_arr = np.empty(self._block_shape_at_point(block_pt), dtype=object)
        new_arr.fill(self.default_value)
        # Iterate forward so the later, last-filled entries have priority.
        for rng, value in self.grid[block_pt]:
            spt = tuple(slice(a, b) for a, b in zip(rng[0], rng[1]))
            new_arr[spt].fill(value)
        self.grid.itemset(block_pt, new_arr)

    def _block_shape_at_point(self, pt: Sequence[int]) -> tuple[int, ...]:
        """Get the shape of the block at a given grid coordinate."""
        return tuple(
            min(b, o - (p * b)) for o, b, p in zip(self.shape, self.block_shape, pt)
        )

    def iter_values(self):
        for block_pt in np.ndindex(self.grid.shape):
            block = self.grid[block_pt]
            if isinstance(block, np.ndarray):
                for pt in np.ndindex(block.shape):
                    yield block[pt]
            elif isinstance(block, list):
                for _, value in block:
                    yield value
            else:
                yield block

    def to_dense(self) -> np.ndarray:
        """Convert to a dense numpy array.

        Note that this is slow and memory-hungry. It'll peak at about twice the memory
        required by the result.
        """
        new_grid = self.grid.copy()
        for block_pt in np.ndindex(new_grid.shape):
            if isinstance(new_grid[block_pt], np.ndarray):
                continue

            new_grid[block_pt] = np.empty(
                self._block_shape_at_point(block_pt), dtype=object
            )
            if isinstance(self.grid[block_pt], list):
                # Iterate forward so the later, last-filled entries have priority.
                for rng, value in self.grid[block_pt]:
                    spt = tuple(slice(a, b) for a, b in zip(rng[0], rng[1]))
                    new_grid[block_pt][spt].fill(value)
            else:
                new_grid[block_pt].fill(self.grid[block_pt])
        return np.block(new_grid.tolist())


def _drop_covered_entries(
    covering_rng: tuple[tuple[int, ...], tuple[int, ...]], block_list: list
) -> None:
    to_remove = []
    for idx, ((lower, upper), _) in enumerate(block_list):
        if all(l >= c for l, c in zip(lower, covering_rng[0])) and all(
            u <= c for u, c in zip(upper, covering_rng[1])
        ):
            to_remove.append(idx)
    for idx in reversed(to_remove):
        del block_list[idx]


def _intersect(
    rect1_lower: Sequence[int],
    rect1_upper: Sequence[int],
    rect2_lower: Sequence[int],
    rect2_upper: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute the intersection of two hyperrectangles.

    For example:
    >>> _intersect((0, 0), (2, 2), (1, 1), (3, 3))
    ((1, 1), (2, 2))

    Raises a `ValueError` if the rectangles do not intersect.
    """
    intersect_lower = tuple(max(a, b) for a, b in zip(rect1_lower, rect2_lower))
    intersect_upper = tuple(min(a, b) for a, b in zip(rect1_upper, rect2_upper))
    return intersect_lower, intersect_upper


def _surface_pts(
    lower: Sequence[int],
    upper: Sequence[int],
    lower_misaligned_dim_idxs: Sequence[int],
    upper_misaligned_dim_idxs: Sequence[int],
) -> Iterable[tuple[int, ...]]:
    """Enumerate points on the extruded perimeter of a hyperrectangle.

    With one-dimensional shapes:
    >>> sorted(_surface_pts((0,), (2,), {0}, {0}))
    [(-1,), (2,)]

    With higher-dimensional shapes:
    >>> sorted(_surface_pts((0, 0), (2, 2), {}, {}))
    []
    >>> sorted(_surface_pts((0, 0), (1, 1), {0}, {}))
    [(-1, 0)]
    >>> sorted(_surface_pts((0, 0), (2, 2), {0, 1}, {}))
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)]
    >>> sorted(_surface_pts((0, 0, 0), (2, 2, 2), {1}, {1})) # doctest: +NORMALIZE_WHITESPACE
    [(0, -1, 0), (0, -1, 1), (0, 2, 0), (0, 2, 1),
     (1, -1, 0), (1, -1, 1), (1, 2, 0), (1, 2, 1)]
    >>> sorted(_surface_pts((0, 0, 0), (2, 2, 2), {0, 1}, {})) # doctest: +NORMALIZE_WHITESPACE
    [(-1, -1, 0), (-1, -1, 1),
     (-1, 0, 0), (-1, 0, 1), (-1, 1, 0), (-1, 1, 1),
     (0, -1, 0), (0, -1, 1), (1, -1, 0), (1, -1, 1)]

    Things get a little weirder with zero-volume hyperrectangles:
    >>> sorted(_surface_pts((1,), (1,), {0}, {}))
    [(0,)]
    >>> sorted(_surface_pts((0, 1), (0, 2), {0}, {}))
    [(-1, 1)]
    >>> sorted(_surface_pts((0, 1), (0, 2), {1}, {}))
    []
    """
    for outer_idx in range(len(lower)):
        side_pts = []
        if outer_idx in lower_misaligned_dim_idxs:
            side_pts.append(lower[outer_idx] - 1)
        if outer_idx in upper_misaligned_dim_idxs:
            side_pts.append(upper[outer_idx])

        for outer_dim_pt in side_pts:
            parts = []
            for i in range(len(lower)):
                if i == outer_idx:
                    parts.append([outer_dim_pt])
                else:
                    l, u = lower[i], upper[i]
                    a = 1 if (i in lower_misaligned_dim_idxs and i < outer_idx) else 0
                    b = 1 if (i in upper_misaligned_dim_idxs and i < outer_idx) else 0
                    parts.append(range(l - a, u + b))
            yield from itertools.product(*parts)


def _compress_block(block: np.ndarray):
    # This implementation is slow. To check if all values in `block` are equal, we
    # materialize a whole new block filled with the first element and check (without
    # short-circuiting) for equality between the corresponding elements. To save *some*
    # time in cases where nothing matches, we first check an arbitrary element and bail
    # early, but this is still slow in the general case (and downstream of wanting to
    # lower everything into numpy for speed).
    if block.item(0) != block.item(1):
        return block
    v = block.flatten()[0]
    rhs = np.empty_like(block)
    rhs.fill(v)
    if np.array_equal(block, rhs):
        return v
    return block
