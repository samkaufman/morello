import contextvars
import enum


class TileSizeMode(enum.Enum):
    ALL = enum.auto()
    CACHE_LINE_MULTIPLES = enum.auto()
    POWERS_OF_TWO = enum.auto()


tile_size_mode: contextvars.ContextVar[TileSizeMode] = contextvars.ContextVar(
    "tile_size_mode", default=TileSizeMode.POWERS_OF_TWO
)
allow_sliding_windows: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "allow_sliding_windows", default=True
)
allow_reduce_splits: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "allow_reduce_splits", default=True
)
prune_nested_parallel_loops: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "prune_nested_parallel_loops", default=True
)
enable_prefetching_moves: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "enable_prefetching_moves", default=False
)
PRUNE_RELAYOUT_CYCLES = True
BREAK_MOVE_SYMMETRIES = False  # TODO: Remove this code entirely
BREAK_SEQUENTIAL_TILES = False
