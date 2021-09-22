from typing import Union, Iterable, Optional

import termcolor

from . import utils
from .tensor import Tensor, Tile


_COLORS = ["blue", "green", "magenta", "red", "yellow", "cyan"]


def _concat_word(left: str, word: str) -> str:
    if left[-1].islower():
        word = word[0].upper() + word[1:]
    elif left[-1].isupper():
        word = word[0].lower() + word[1:]
    return left + word


def _assign_tensor_color(
    tensor: Union[Tensor, Tile],
    assignments: dict[Union[Tensor, Tile], Optional[str]] = {},
    tensors_to_color: Optional[frozenset[Union[Tensor, Tile]]] = None,
) -> None:
    # If a color has been assigned, do nothing
    if tensor in assignments:
        return

    # If possible, choose the same color as the origin
    if tensor.origin:
        _assign_tensor_color(tensor.origin, assignments)
        assignments[tensor] = assignments[tensor.origin]
        return
    if tensor.root and tensor.root != tensor:
        _assign_tensor_color(tensor.root, assignments)
        assignments[tensor] = assignments[tensor.root]
        return

    # Otherwise, we are handling a root tensor, so choose a new color if this
    # tensor is in the list of tensors to color, or assign None if not
    color = None
    if tensors_to_color is None or tensor in tensors_to_color:
        for c in _COLORS:
            if c not in assignments.values():
                color = c
                break
    assignments[tensor] = color


def _assign_tensor_name(
    tensor: Union[Tensor, Tile], assignments: dict[Union[Tensor, Tile], str] = {}
) -> None:
    if tensor in assignments:
        return

    def extend_name(n: str) -> str:
        while n in assignments.values():
            n += "'"
        return n

    # If the tensor has a name, use it or, if the name has already been used
    # (happens if two tensors are assigned the same `name` property value),
    # add "'" until it is unique.
    if tensor.name:
        assignments[tensor] = extend_name(tensor.name)
        return

    if tensor.origin:
        if isinstance(tensor, Tile):
            preferred_suffix = "t"
        elif "RF" in tensor.root.bank:
            preferred_suffix = "r"
        else:
            # 's' is an arbitrary choice, but this case hardly ever occurs in
            # practice.
            preferred_suffix = "s"
        _assign_tensor_name(tensor.origin, assignments)
        assignments[tensor] = extend_name(
            _concat_word(assignments[tensor.origin], preferred_suffix)
        )
        return

    # If we're dealing with a tensor that has no name or origin, just pick an
    # unused character from the alphabet.
    for char in utils.ALPHABET_PRODUCT:
        if char not in assignments.values():
            assignments[tensor] = char
            return
    raise Exception("Exhausted the alphabet")


class TensorNamer:
    def __init__(
        self, tensors_to_color: Optional[Iterable[Union[Tensor, Tile]]] = None
    ) -> None:
        self.name_assignments = dict()
        self.color_assignments = dict()
        self._tensors_to_color = None
        if tensors_to_color:
            self._tensors_to_color = frozenset(tensors_to_color)

    def name(self, tensor: Union[Tensor, Tile], color: bool = False) -> str:
        _assign_tensor_name(tensor, self.name_assignments)
        result = self.name_assignments[tensor]
        if color:
            color_to_use = self._color(tensor)
            if color_to_use:
                result = termcolor.colored(result, color=color_to_use)
        return result

    def _color(self, tensor: Union[Tensor, Tile]) -> Optional[str]:
        _assign_tensor_color(tensor, self.color_assignments, self._tensors_to_color)
        return self.color_assignments[tensor]
