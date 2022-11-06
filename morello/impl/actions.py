import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Tuple, Union

from .. import system_config
from ..layouts import Layout
from ..tensor import Tensor, Tile
from .base import Impl
from . import settings

if TYPE_CHECKING:
    from .compose import ComposeHole


@dataclasses.dataclass(frozen=True)
class MoveAction:
    """Wraps a function which wraps a Impl in a MoveLet.

    This is used instead of a functools.partial to expose fields to
    break_moves_symmetries.
    """

    func: Callable[[Optional[str], Optional[Layout], bool], "Impl"]
    source: Union[Tensor, Tile]
    input_idx: Optional[int]
    prefetching: bool
    bank: Optional[str] = None
    layout: Optional[Layout] = None
    kwargs: Optional[Mapping[Any, Any]] = None

    def __post_init__(self):
        assert (
            any(d > 1 for d in self.source.dim_sizes) or self.layout.is_row_major
        ), f"Layout was {self.layout} for dims. {self.source.dim_sizes}"
        assert (
            self.bank is None
            or self.bank
            in system_config.current_system().faster_destination_banks(self.source.bank)
        )
        assert not self.prefetching or settings.enable_prefetching_moves.get()

    def __call__(self):
        kws = {}
        if self.kwargs is not None:
            kws = self.kwargs
        return self.func(self.bank, self.layout, self.prefetching, **kws)

    def __str__(self):
        return (
            f"MoveAction(input_idx={self.input_idx}, source={str(self.source)},"
            f" prefetching={self.prefetching},"
            f" {self.bank}, layout={str(self.layout)})"
        )


@dataclasses.dataclass(frozen=True)
class PeelAction:
    impl: "ComposeHole"
    bank: Optional[str] = None
    layout: Optional[Layout] = None
    kwargs: Optional[Mapping[Any, Any]] = None

    def __call__(self):
        kws = {}
        if self.kwargs:
            kws = self.kwargs
        return self.impl.peel(self.bank, self.layout, **kws)


@dataclasses.dataclass(frozen=True)
class TileOutAction:
    impl: "Impl"
    shape: Tuple[int, ...]
    parallel: bool

    def __call__(self):
        return self.impl.tile_out(self.shape, parallel=self.parallel)


@dataclasses.dataclass(frozen=True)
class SlidingTileOutAction:
    func: Callable[[int, int, str], "Impl"]
    sliding_dim: int
    output_size: int
    bank: str

    def __call__(self):
        return self.func(self.sliding_dim, self.output_size, self.bank)


@dataclasses.dataclass(frozen=True)
class MatmulSplitAction:
    func: Callable[[int], "Impl"]
    size: int

    def __call__(self):
        return self.func(self.size)
