import dataclasses
import functools
import sys
from typing import FrozenSet, Union, Optional

from .settings import (
    PRUNE_RELAYOUT_CYCLES,
    BREAK_MOVE_SYMMETRIES,
    BREAK_SEQUENTIAL_TILES,
)
from .. import system_config
from ..tensor import Tensor, Tile


@dataclasses.dataclass
class ParentSummary:
    parent: "Impl"
    movements: FrozenSet[tuple[Union[Tensor, Tile], str]]

    @staticmethod
    def update(original: Optional["ParentSummary"], parent: "Impl") -> "ParentSummary":
        from . import moves

        if original is None:
            movements = set()
        else:
            movements = set(original.movements)

        if isinstance(parent, moves.MoveLet):
            movements.add((parent.destination, parent.destination.bank))

        return ParentSummary(parent=parent, movements=frozenset(movements))


def prune_relayout_cycles(func):
    from . import moves

    if not PRUNE_RELAYOUT_CYCLES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        for action in func(*args, **kwargs):
            if not isinstance(action, moves.MoveAction):
                yield action
                continue
            if (action.source, action.bank) in parent_summary.movements:
                continue
            yield action

    return wrapper_decorator


def break_moves_symmetries(func):
    """Wraps a function which yields scheduling actions to filter symmetric moves.

    This places a total ordering over moves' source tensors, requiring that all moves
    be from a tensor greater than or equal to its parent move.

    Pruning using this filtering decorator, unfortunately, means that `func` still
    enumerates unneeded actions, but doesn't expand them, which is the important thing.
    Pruning was implemented as a decorator to separate concerns and make it simple to
    disable for testing.
    """
    from . import actions, moves

    if not BREAK_MOVE_SYMMETRIES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        if not isinstance(parent_summary.parent, moves.MoveLet):
            yield from func(*args, **kwargs)
            return
        for action in func(*args, **kwargs):
            if not isinstance(action, actions.MoveAction):
                yield action
                continue

            # Assign each operand to an integer. Input index if it applies, otherwise:
            # arbitrarily large number
            action_operand_idx = action.input_idx
            parent_schedule_idx = parent_summary.parent.input_idx
            if action_operand_idx is None:
                action_operand_idx = sys.maxsize
            if parent_schedule_idx is None:
                parent_schedule_idx = sys.maxsize

            # Assert lexicographic order
            if action_operand_idx < parent_schedule_idx:
                continue

            # Assert that there is no interleaving of destination levels between moves.
            # Note: destination_banks_closure includes the given bank itself.
            system = system_config.current_system()
            if action.bank not in system.destination_banks_closure(
                parent_summary.parent.destination.bank
            ):
                continue

            yield action

    return wrapper_decorator


def break_tile_out_symmetries(func):
    """Wraps an actions method to never return sequential .tile_outs.

    This is a no-op if BREAK_SEQUENTIAL_TILES is false.
    """
    from . import actions, loops

    if not BREAK_SEQUENTIAL_TILES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        # If no ParentSummary is given, we're at the root, and done.
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        parent_schedule = parent_summary.parent
        # Check if the parent is a loop that tiles output. If not, we're done.
        if (not isinstance(parent_schedule, loops.Loop)) or (
            args[0].output not in [dest for dest, _ in parent_schedule.introduced]
        ):
            yield from func(*args, **kwargs)
            return
        # Filter all .tile_outs with the same parallel flag at the parent.
        for action in func(*args, **kwargs):
            if (
                not isinstance(action, actions.TileOutAction)
                or action.parallel != parent_schedule.parallel
            ):
                yield action

    return wrapper_decorator


def break_matmul_split_symmetries(func):
    from . import actions

    if not BREAK_SEQUENTIAL_TILES:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parent_summary: Optional[ParentSummary] = kwargs.get("parent_summary")
        # If no ParentSummary is given, we're at the root, and done.
        if parent_summary is None:
            yield from func(*args, **kwargs)
            return
        parent_schedule = parent_summary.parent
        # If the parent isn't a matmul split, we're done.
        if not isinstance(parent_schedule, actions.MatmulSplitAction):
            yield from func(*args, **kwargs)
            return
        # Filter out all splits after this point. These splits immediately
        # follow another split.
        for action in func(*args, **kwargs):
            if not isinstance(action, actions.MatmulSplitAction):
                yield action

    return wrapper_decorator
