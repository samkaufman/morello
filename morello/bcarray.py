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
            self.grid = _NumpyStore(self.shape, self.block_shape, self.default_value)
        else:
            redis_client, prefix = use_redis
            self.grid = _BCARedisStore(
                self.shape, self.block_shape, redis_client, prefix, self.default_value
            )

    async def get(self, pt):
        """Get a value from the array.

        This will raise an IndexError if the point is out of bounds, but
        un-set points which are in-bounds will return the default value.
        """
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

    async def flush(self) -> None:
        await self.grid.flush()

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
            if block_enclosed_lower[i] > block_enclosed_upper[i]:
                block_enclosed_lower[i] = block_enclosed_upper[i]
            if u % b != 0:
                upper_misaligned_dim_idxs.append(i)

        # If any of the dimensions in the block enclosure are equal or inverted, then
        # the following will fill nothing.
        for block_pt in itertools.product(
            *[range(l, u) for l, u in zip(block_enclosed_lower, block_enclosed_upper)]
        ):
            await self.grid.itemset(block_pt, value)

        # Fill regions in the boundary blocks.
        # `surface_pts` requires that block_enclosed_lower be less than
        # block_enclosed_upper in all dimensions, which might not be the case. However,
        # in that case, we don't want to
        surf_pts = list(
            surface_pts(
                block_enclosed_lower,
                block_enclosed_upper,
                lower_misaligned_dim_idxs,
                upper_misaligned_dim_idxs,
            )
        )

        # Strip negative (non-existent) block points.
        # TODO: Can we improve our "snapping" logic to avoid this entirely?
        surf_pts = [pt for pt in surf_pts if all(p >= 0 for p in pt)]

        surf_pts_results = await self.grid.get_many(surf_pts)
        for block_pt, b in zip(surf_pts, surf_pts_results):
            block_intersection = _block_intersect(
                block_pt, self.block_shape, lower, upper
            )
            if block_intersection is None:
                continue

            # TODO: Check for empty intersection, which would be a surface_pts error.
            assert all(l < r for l, r in zip(*block_intersection)), (
                f"Got empty block intersection: {block_intersection}; inputs "
                f"were {block_pt}, {self.block_shape}, {lower}, {upper}."
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
                # TODO: Handle the special case where block_intersection == block_shape.
                block_shape = self._block_shape_at_point(block_pt)
                await self.grid.itemset(
                    block_pt,
                    [
                        (((0,) * len(block_shape), block_shape), b),
                        (block_intersection, value),
                    ],
                )

    async def _convert_list_block_to_ndarray(self, block_pt):
        convertee = await self.grid.get(block_pt)
        assert isinstance(convertee, list), f"Expected list, got {type(convertee)}"
        new_arr = np.empty(self._block_shape_at_point(block_pt), dtype=object)
        new_arr.fill(self.default_value)
        # Iterate forward so the later, last-filled entries have priority.
        for rng, value in convertee:
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
                for rng, value in block:
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
    def __init__(self, expanded_shape, block_shape, default_value):
        self._inner = np.full(
            tuple((o + b - 1) // b for b, o in zip(block_shape, expanded_shape)),
            fill_value=default_value,
            dtype=object,
        )

    @property
    def shape(self):
        return self._inner.shape

    async def get(self, key):
        if any(p < 0 for p in key):
            raise IndexError("Negative indices not supported")
        return self._inner[key]

    async def get_many(self, keys) -> list[Any]:
        return [await self.get(k) for k in keys]

    async def itemset(self, key, value) -> None:
        self._inner.itemset(key, value)

    async def flush(self) -> None:
        pass

    async def contains(self, key):
        return key in self._inner

    async def fill(self, value):
        self._inner.fill(value)

    async def copy(self):
        # Produces an ndarray, despite the name.
        return self._inner.copy()


class _BCARedisStore:
    def __init__(
        self, expanded_shape, block_shape, redis_client, prefix: str, default_value
    ):
        self.shape = tuple(
            (o + b - 1) // b for b, o in zip(block_shape, expanded_shape)
        )
        self.redis_client = redis_client
        self.prefix = prefix
        self.default_value = default_value
        self._local_cache = {}  # TODO: Make an array
        self._updated = set()
        self._prefix_lock = None

    # TODO: Don't need this *and* `get_many`.
    async def get(self, key):
        if any(p < 0 for p in key):
            raise IndexError("Negative indices not supported")
        if key in self._local_cache:
            return self._local_cache[key]
        redis_key = self._redis_key(key)
        result = await self.redis_client.get(redis_key)
        if result is None:
            return self.default_value
        return pickle.loads(lz4.frame.decompress(result))

    async def get_many(self, keys) -> list[Any]:
        for key in keys:
            if any(p < 0 for p in key):
                raise IndexError("Negative indices not supported")

        results = [self.default_value] * len(keys)

        # Get the local results.
        missing = []
        for i, key in enumerate(keys):
            try:
                results[i] = self._local_cache[key]
            except KeyError:
                missing.append((i, key))

        # Fill in the locally-missing results with Redis results.
        redis_results = await self.redis_client.mget(
            [self._redis_key(k) for _, k in missing]
        )
        for (m, _), red_result in zip(missing, redis_results):
            if red_result is not None:
                results[m] = pickle.loads(lz4.frame.decompress(red_result))
        return results

    async def itemset(self, key, value) -> None:
        if not self._prefix_lock:
            self._prefix_lock = self.redis_client.lock(
                f"Lock:{self.prefix}", blocking=False
            )
            for i in range(10):
                lock_result = await self._prefix_lock.acquire()
                if lock_result:
                    break
                if i < 9:
                    await asyncio.sleep(0.1 + random.random())
                else:
                    raise Exception(
                        f"Couldn't acquire lock: Lock:{self.prefix} (PID: {os.getpid()})"
                    )

        self._local_cache[key] = value
        self._updated.add(key)

    async def flush(self) -> None:
        batch_to_set = {}
        for key in self._updated:
            redis_key = self._redis_key(key)
            value = self._local_cache[key]
            # Check for ndarray since that has no __eq__ but also can't be the default.
            if not isinstance(value, np.ndarray) and value == self.default_value:
                # TODO: Instead, just delete the key in this case?
                pass
            data = lz4.frame.compress(pickle.dumps(value, protocol=PICKLE_PROTOCOL))
            batch_to_set[redis_key] = data
        await self.redis_client.mset(batch_to_set)
        self._local_cache.clear()
        self._updated.clear()
        if self._prefix_lock is not None:
            await self._prefix_lock.release()
            self._prefix_lock = None

    async def contains(self, key):
        raise NotImplementedError()

    async def fill(self, value):
        raise NotImplementedError()

    async def copy(self):
        # Produces an ndarray, despite the name.
        result = np.full(self.shape, fill_value=self.default_value, dtype=object)
        for redis_key in await self.redis_client.keys(f"{self.prefix}:BCA-*"):
            pt = tuple(int(k) for k in redis_key.split(":BCA-")[-1].split(","))
            if pt in self._local_cache:
                continue
            result.itemset(
                pt,
                pickle.loads(
                    lz4.frame.decompress(await self.redis_client.get(redis_key))
                ),
            )
        for pt, val in self._local_cache.items():
            result.itemset(pt, val)
        return result

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
) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Intersect the range [lower, upper) with the coordinate range of a block.

    >>> _block_intersect((0,), (3,), (0,), (2,))
    ((0,), (2,))
    >>> _block_intersect((0,), (3,), (3,), (4,)) is None
    True
    >>> _block_intersect((0,), (2,), (0,), (3,))
    ((0,), (2,))
    >>> _block_intersect((1,), (2,), (0,), (3,))
    ((0,), (1,))

    >>> _block_intersect((0, 0), (3, 1), (3, 0), (4, 10)) is None
    True
    """
    if any(l > u for l, u in zip(lower, upper)):
        raise ValueError(f"Invalid range; lower {lower} not less than upper {upper}")
    global_block_origin = [l * b for l, b in zip(block_pt, block_shape)]
    global_block_peak = [o + s for o, s in zip(global_block_origin, block_shape)]

    if any(u < b for u, b in zip(upper, global_block_origin)):
        return None
    if any(l >= t for l, t in zip(lower, global_block_peak)):
        return None

    # The following are the intersection coordinates *inside* the block, snapped
    # to the block boundaries.
    block_origin = tuple(max(0, b - a) for a, b in zip(global_block_origin, lower))
    block_upper = tuple(
        min(t, b) - c for t, b, c in zip(global_block_peak, upper, global_block_origin)
    )
    return block_origin, block_upper


def surface_pts(
    lower: Sequence[int],
    upper: Sequence[int],
    lower_misaligned_dim_idxs: Sequence[int],
    upper_misaligned_dim_idxs: Sequence[int],
) -> Iterable[tuple[int, ...]]:
    """Enumerate points on the extruded perimeter of a hyperrectangle.

    With one-dimensional shapes:
    >>> sorted(surface_pts((0,), (2,), {0}, {0}))
    [(-1,), (2,)]

    With 2-dimensional shapes:
    >>> sorted(surface_pts((0, 0), (2, 2), {}, {}))
    []
    >>> sorted(surface_pts((0, 0), (1, 1), {0}, {}))
    [(-1, 0)]
    >>> sorted(surface_pts((0, 0), (2, 2), {0, 1}, {}))
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)]

    With 3-dimensional shapes:
    >>> sorted(surface_pts((0, 0, 0), (2, 2, 2), {1}, {1})) # doctest: +NORMALIZE_WHITESPACE
    [(0, -1, 0), (0, -1, 1), (0, 2, 0), (0, 2, 1),
     (1, -1, 0), (1, -1, 1), (1, 2, 0), (1, 2, 1)]
    >>> sorted(surface_pts((0, 0, 0), (2, 2, 2), {0, 1}, {})) # doctest: +NORMALIZE_WHITESPACE
    [(-1, -1, 0), (-1, -1, 1),
     (-1, 0, 0), (-1, 0, 1), (-1, 1, 0), (-1, 1, 1),
     (0, -1, 0), (0, -1, 1), (1, -1, 0), (1, -1, 1)]

    Things get a little weirder with zero-volume hyperrectangles:
    >>> sorted(surface_pts((0,), (0,), {0}, {0}))
    [(-1,), (0,)]
    >>> sorted(surface_pts((1,), (1,), {0}, {}))
    [(0,)]
    >>> sorted(surface_pts((1,), (1,), {0}, {0}))
    [(0,), (1,)]
    >>> sorted(surface_pts((0, 1), (0, 2), {0}, {}))
    [(-1, 1)]
    >>> sorted(surface_pts((0, 1), (0, 2), {1}, {}))
    []
    """
    if len(lower) != len(upper):
        raise ValueError("lower and upper must have the same rank")
    if any(l > u for l, u in zip(lower, upper)):
        raise ValueError(f"lower must be <= upper, but given {lower} and {upper}")

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
