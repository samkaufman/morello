import asyncio
import itertools
import os
import pickle
import random
from typing import Any, AsyncIterator, Iterable, Optional, Sequence, Union

import lz4.frame
import numpy as np

PICKLE_PROTOCOL = 5


class BlockCompressedArray:
    """A block-compressed, multi-dimensional array.

    Implements a subset of the numpy.ndarray interface.

    Blocks which contain only the same value will be represented as a single
    value in memory.
    """

    grid: Union["_NumpyStore", "_BCARedisStore"]

    def __init__(
        self,
        shape: tuple[int, ...],
        block_shape: tuple[int, ...],
        default_value=None,
        dense_block_threshold: int = 4,
        compress_on_fill: bool = False,
        use_redis: Optional[tuple[Any, str]] = None,
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
        if not use_redis:
            self.grid = _NumpyStore(
                np.full(
                    tuple(
                        (o + b - 1) // b for b, o in zip(self.block_shape, self.shape)
                    ),
                    fill_value=self.default_value,
                    dtype=object,
                )
            )
        else:
            redis_client, namespace = use_redis
            prefix = f"{namespace}"
            self.grid = _BCARedisStore(redis_client, prefix)

    async def get(self, pt):
        """Get a value from the array."""
        block_pt = tuple(p // b for p, b in zip(pt, self.block_shape))
        block = await self.grid.get(block_pt)
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

    async def full(self) -> bool:
        # if self.default_value in self.grid:
        if await self.grid.contains(self.default_value):
            return False
        for block_pt in np.ndindex(self.grid.shape):
            block = await self.grid.get(block_pt)
            if isinstance(block, np.ndarray):
                if self.default_value in block:
                    return False
            elif isinstance(block, list):
                # TODO: Below is used here and in iter_values. Extract a method.
                temp_arr = np.empty(self._block_shape_at_point(block_pt), dtype=object)
                for rng, value in await self.grid.get(block_pt):
                    spt = tuple(slice(a, b) for a, b in zip(rng[0], rng[1]))
                    temp_arr[spt].fill(value)
                if self.default_value in temp_arr:
                    return False
            else:
                assert block != self.default_value  # Already checked above
        return True

    async def count_values(self) -> int:
        """Count the number of stored values.

        Compressed blocks are counted as 1.
        """
        result = 0
        for block_pt in np.ndindex(self.grid.shape):
            block = await self.grid.get(block_pt)
            if isinstance(block, np.ndarray):
                result += block.size
            elif isinstance(block, list):
                result += len(block)
            else:
                result += 1
        return result

    async def fill(self, value) -> None:
        """Fill the array with a value."""
        await self.grid.fill(value)

    async def fill_range(
        self, lower: Sequence[int], upper: Sequence[int], value
    ) -> None:
        """Fill a (hyper-)rectangular sub-region of the array with a value.

        :param lower: The lower coordinate of the region to fill.
        :param upper: The upper coordinate of the region to fill. Every dimension must
            be greater than the corresponding dimension in lower or `fill_range` will
            do nothing.
        :param value: The value with which to fill the region.
        """
        if any(l < 0 for l in lower) or any(u > b for u, b in zip(upper, self.shape)):
            raise IndexError(
                f"Range given by {lower} and {upper} is out of bounds for a "
                f"{type(self).__name__} of shape {self.shape}"
            )
        if any(l >= u for l, u in zip(lower, upper)):
            return

        # Fill all blocks which are fully contained within the range.
        # Blocks which are full-enclosed can be filled quickly by replacing those
        # entries in `self.grid` with the value itself; no list or ndarray needed.
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
            await self.grid.itemset(block_pt, value)

        # Fill regions in the boundary blocks.
        surf_pts = list(
            _surface_pts(
                block_enclosed_lower,
                block_enclosed_upper,
                lower_misaligned_dim_idxs,
                upper_misaligned_dim_idxs,
            )
        )
        for block_pt, b in zip(surf_pts, await self.grid.get_many(surf_pts)):
            block_intersection = _block_intersect(
                block_pt, self.block_shape, lower, upper
            )
            if isinstance(b, np.ndarray):
                spt = tuple(slice(a, b) for a, b in zip(*block_intersection))
                b[spt].fill(value)
                if self.compress_on_fill:
                    await self.grid.itemset(block_pt, _compress_block(b))
            elif isinstance(b, list):
                _drop_covered_entries(block_intersection, b)
                b.append((block_intersection, value))
                if len(b) >= self.dense_block_threshold:
                    await self._convert_list_block_to_ndarray(block_pt)
            else:
                # Bail here if `value` matches the original value.
                if b == value:
                    continue
                # Turn a single-value block into a list.
                block_shape = self._block_shape_at_point(block_pt)
                await self.grid.itemset(
                    block_pt,
                    [
                        (((0,) * len(block_shape), block_shape), b),
                        (block_intersection, value),
                    ],
                )

    async def _convert_list_block_to_ndarray(self, block_pt):
        assert isinstance((await self.grid.get(block_pt)), list)
        new_arr = np.empty(self._block_shape_at_point(block_pt), dtype=object)
        new_arr.fill(self.default_value)
        # Iterate forward so the later, last-filled entries have priority.
        for rng, value in await self.grid.get(block_pt):
            spt = tuple(slice(a, b) for a, b in zip(rng[0], rng[1]))
            new_arr[spt].fill(value)
        await self.grid.itemset(block_pt, new_arr)

    def _block_shape_at_point(self, pt: Sequence[int]) -> tuple[int, ...]:
        """Get the shape of the block at a given grid coordinate."""
        return tuple(
            min(b, o - (p * b)) for o, b, p in zip(self.shape, self.block_shape, pt)
        )

    async def iter_values(self) -> AsyncIterator[Any]:
        for block_pt in np.ndindex(self.grid.shape):
            block = await self.grid.get(block_pt)
            if isinstance(block, np.ndarray):
                for pt in np.ndindex(block.shape):
                    yield block[pt]
            elif isinstance(block, list):
                # Lists may contain fully covered ("dead") regions. We could prune
                # these with a line sweep, but instead we'll do the slower, simpler
                # thing: temporarily convert to a dense array.
                temp_arr = np.empty(self._block_shape_at_point(block_pt), dtype=object)
                for rng, value in await self.grid.get(block_pt):
                    spt = tuple(slice(a, b) for a, b in zip(rng[0], rng[1]))
                    temp_arr[spt].fill(value)
                for _, value in np.ndenumerate(temp_arr):
                    yield value
            else:
                yield block

    async def to_dense(self) -> np.ndarray:
        """Convert to a dense numpy array.

        Note that this is slow and memory-hungry. It'll peak at about twice the memory
        required by the result.
        """
        new_grid = await self.grid.copy()
        for block_pt in np.ndindex(new_grid.shape):
            if isinstance(new_grid[block_pt], np.ndarray):
                continue

            new_grid[block_pt] = np.empty(
                self._block_shape_at_point(block_pt), dtype=object
            )
            block_entry = await self.grid.get(block_pt)
            if isinstance(block_entry, list):
                # Iterate forward so the later, last-filled entries have priority.
                for rng, value in block_entry:
                    spt = tuple(slice(a, b) for a, b in zip(rng[0], rng[1]))
                    new_grid[block_pt][spt].fill(value)
            else:
                new_grid[block_pt].fill(block_entry)
        return np.block(new_grid.tolist())


class _NumpyStore:
    def __init__(self, np_arr: np.ndarray):
        self._inner = np_arr

    @property
    def shape(self):
        return self._inner.shape

    async def get(self, key: str):
        try:
            return self._inner[key]
        except IndexError:
            return None

    async def get_many(self, keys) -> list[Any]:
        return [self.get(k) for k in keys]

    async def itemset(self, key, value) -> None:
        self._inner.itemset(key, value)

    async def contains(self, key):
        return key in self._inner

    async def fill(self, value):
        self._inner.fill(value)

    async def copy(self):
        return self._inner.copy()


class _BCARedisStore:
    def __init__(self, redis_client, prefix: str):
        self.redis_client = redis_client
        self.prefix = prefix

    @property
    def shape(self):
        raise NotImplementedError(
            f"shape not implemented for {self.__class__.__name__}"
        )

    # TODO: Don't need this *and* `get_many`.
    async def get(self, key):
        redis_key = self._redis_key(key)
        result = await self.redis_client.get(redis_key)
        if result is None:
            return None
        else:
            return pickle.loads(lz4.frame.decompress(result))

    # async def get_many(self, keys) -> AsyncIterable[Optional[str]]:
    async def get_many(self, keys) -> list[Any]:
        redis_keys = [self._redis_key(key) for key in keys]
        results = await self.redis_client.mget(redis_keys)
        for i in range(len(results)):
            if results[i] is not None:
                results[i] = pickle.loads(lz4.frame.decompress(results[i]))
        return results

    async def itemset(self, key, value) -> None:
        redis_key = self._redis_key(key)
        data = lz4.frame.compress(pickle.dumps(value, protocol=PICKLE_PROTOCOL))
        await self.redis_client.set(redis_key, data)

    async def contains(self, key):
        raise NotImplementedError()

    async def fill(self, value):
        raise NotImplementedError()

    async def copy(self):
        # TODO: Should actually produce an ndarray, despite the name.
        raise NotImplementedError(f"copy not implemented for {self.__class__.__name__}")

    def _redis_key(self, input_key) -> str:
        return f"{self.prefix}:BCA-{','.join(str(k) for k in input_key)}"


def _drop_covered_entries(
    covering_rng: tuple[Sequence[int], Sequence[int]], block_list: list
) -> None:
    to_remove = []
    for idx, ((lower, upper), _) in enumerate(block_list):
        if all(l >= c for l, c in zip(lower, covering_rng[0])) and all(
            u <= c for u, c in zip(upper, covering_rng[1])
        ):
            to_remove.append(idx)
    for idx in reversed(to_remove):
        del block_list[idx]


def _block_intersect(
    block_pt: Sequence[int],
    block_shape: Sequence[int],
    lower: Sequence[int],
    upper: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    global_origin = [l * b for l, b in zip(block_pt, block_shape)]
    block_origin = tuple(max(0, b - a) for a, b in zip(global_origin, lower))
    block_upper = tuple(
        min((p + 1) * s, b) - c
        for p, s, b, c in zip(block_pt, block_shape, upper, global_origin)
    )
    return block_origin, block_upper


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
