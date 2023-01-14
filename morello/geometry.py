import abc
import dataclasses
import typing
from collections.abc import Iterator
from typing import Callable, Generic, Iterable, Sequence

from . import utils

S = typing.TypeVar("S")


@dataclasses.dataclass(frozen=False)
class Grid(Generic[S]):
    inner: list["BlockedRange"]
    mapper: Callable[[tuple[int, ...]], Iterable[S]]
    rev_mapper: Callable[[S], tuple[int, ...]]

    @property
    def block_lens(self) -> tuple[int, ...]:
        return tuple(rng.block_count for rng in self.inner)

    def iter_northeast(self) -> Iterable[Iterable["GridBlock[S]"]]:
        for diagonal in self.block_diagonals_northeast():
            yield diagonal

    def block_diagonals_northeast(self) -> Iterable[Sequence["GridBlock[S]"]]:
        for diagonal in _walk_diagonal_coords(self.block_lens):
            yield [self.get_block(pt) for pt in diagonal]

    def get_block(self, block_pt: Sequence[int]) -> "GridBlock[S]":
        return GridBlock(self, tuple(block_pt))

    def block_at_point(self, element_pt: Sequence[int]) -> "GridBlock[S]":
        if len(element_pt) != len(self.inner):
            raise ValueError(
                f"Point {element_pt} has {len(element_pt)} dimensions, "
                f"but grid has {len(self.inner)} dimensions"
            )
        return self.get_block(
            tuple(rng.block_index(pt) for rng, pt in zip(self.inner, element_pt))
        )


@dataclasses.dataclass(frozen=True)
class GridBlock(Generic[S]):
    parent: Grid[S]
    point: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(rng.block_len(p) for rng, p in zip(self.parent.inner, self.point))

    def point_diagonals_northeast(self) -> Iterable[Sequence[tuple[int, ...]]]:
        for diagonal in _walk_diagonal_coords(self.shape):
            # TODO: We need a real way to do indexing of and then into a block.
            #   The following `list` is bad.
            yield [
                tuple(
                    list(rng.points_in_block(block_pt))[x]
                    for block_pt, x, rng in zip(
                        self.point, unshifted_pt, self.parent.inner
                    )
                )
                for unshifted_pt in diagonal
            ]

    def iter_northeast(self) -> Iterable[Iterable[S]]:
        for inner_diagonal in self.point_diagonals_northeast():
            # Flatten objects within a diagonal.
            yield (s for pt in inner_diagonal for s in self.parent.mapper(pt))

    def __iter__(self):
        for diagonal in self.iter_northeast():
            yield from diagonal

    def __str__(self) -> str:
        return f"GridBlock({self.point})"


class BlockedRange(abc.ABC):
    """A range of integers, divided into blocks."""

    @property
    @abc.abstractmethod
    def block_count(self) -> int:
        pass

    @abc.abstractmethod
    def block_index(self, input: int) -> int:
        pass

    @abc.abstractmethod
    def points_in_block(self, block: int) -> Iterable[int]:
        pass

    @abc.abstractmethod
    def block_len(self, block_idx: int) -> int:
        pass

    @typing.final
    def iter_blocks(self) -> Iterator[Iterable[int]]:
        return iter(self.points_in_block(b) for b in range(self.block_count))

    @abc.abstractmethod
    def __getitem__(self, index: int) -> int:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def reverse(self) -> "BlockedRange":
        pass

    @typing.final
    def __reversed__(self) -> "BlockedRange":
        return self.reverse()


class SimpleBlockedRange(BlockedRange):
    def __init__(
        self, start: int, stop: int, step: int = 1, block_size: int = 1
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size {block_size} must be 1 or greater")
        self.block_size = block_size
        self._range = range(start, stop, step)

    @property
    def start(self) -> int:
        return self._range.start

    @property
    def stop(self) -> int:
        return self._range.stop

    @property
    def step(self) -> int:
        return self._range.step

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SimpleBlockedRange):
            return NotImplemented
        return __o._range == self._range and __o.block_size == self.block_size

    def __hash__(self) -> int:
        return hash((self._range, self.block_size))

    def index(self, value: int) -> int:
        return self._range.index(value)

    @property
    def block_count(self) -> int:
        return (len(self._range) + (self.block_size - 1)) // self.block_size

    def block_index(self, input: int) -> int:
        return self.index(input) // self.block_size

    def points_in_block(self, block: int) -> Iterable[int]:
        if block < 0 or block >= self.block_count:
            raise IndexError(f"Block {block} is out of range")
        return self._range[block * self.block_size : (block + 1) * self.block_size]

    def block_len(self, block: int) -> int:
        if block < 0 or block >= self.block_count:
            raise IndexError(f"Block {block} is out of range")
        return len(self._range[block * self.block_size : (block + 1) * self.block_size])

    def __getitem__(self, index: int) -> int:
        return self._range[index]

    def __len__(self) -> int:
        return len(self._range)

    def reverse(self) -> BlockedRange:
        if self.start == self.stop:
            return self
        d = -1 if self.stop > self.start else 1
        return SimpleBlockedRange(
            self.stop + d,
            self.start + d,
            step=-self.step,
            block_size=self.block_size,
        )


@dataclasses.dataclass(frozen=True)
class BlockedRangeMapper(BlockedRange):
    inner: BlockedRange
    mapper: Callable[[int], int]
    rev_mapper: Callable[[int], int]

    @property
    def block_count(self) -> int:
        return self.inner.block_count

    def block_index(self, input: int) -> int:
        return self.inner.block_index(self.rev_mapper(input))

    def points_in_block(self, block: int) -> Iterable[int]:
        return map(self.mapper, self.inner.points_in_block(block))

    def block_len(self, block_idx: int) -> int:
        return self.inner.block_len(block_idx)

    def __getitem__(self, index: int) -> int:
        return self.mapper(self.inner[index])

    def __len__(self) -> int:
        return len(self.inner)

    def reverse(self) -> "BlockedRange":
        reversed_inner = reversed(self.inner)
        assert isinstance(reversed_inner, BlockedRange)
        return BlockedRangeMapper(self.inner.reverse(), self.mapper, self.rev_mapper)


def log_range(min: int, stop: int, block_size: int):
    if min < 0:
        raise ValueError(f"min {min} must be non-negative")
    if stop < min:
        raise ValueError(f"stop {stop} must be greater than or equal to min {min}")
    if block_size < 1:
        raise ValueError(f"block_size {block_size} must be 1 or greater")

    r = SimpleBlockedRange(
        min.bit_length(),
        (stop - 1).bit_length() + 1 if stop else 0,
        block_size=block_size,
    )
    r = BlockedRangeMapper(r, _log_range_mapper, _log_range_rev_mapper)
    return r


def _log_range_mapper(x):
    return 2 ** (x - 1) if x else 0


def _log_range_rev_mapper(x):
    return x.bit_length() if x else 0


def _walk_diagonal_coords(shape: Sequence[int]) -> Iterable[list[tuple[int, ...]]]:
    diagonal_idx = 0
    while True:
        diagonal_coordinates = list(
            utils.sum_seqs([s - 1 for s in shape], total=diagonal_idx)
        )
        if not diagonal_coordinates:
            break
        yield diagonal_coordinates
        diagonal_idx += 1
