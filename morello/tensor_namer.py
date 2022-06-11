from typing import Mapping, Iterable, Optional

import termcolor

from . import impl, utils
from .tensor import TensorLike, Tile


_COLORS = ["blue", "green", "magenta", "red", "yellow", "cyan"]


def _concat_word(left: str, word: str) -> str:
    if left[-1].islower():
        word = word[0].upper() + word[1:]
    elif left[-1].isupper():
        word = word[0].lower() + word[1:]
    return left + word


def _assign_tensor_color(
    tensor: TensorLike,
    tensors_to_sources: Mapping[TensorLike, TensorLike],
    assignments: dict[TensorLike, Optional[str]] = {},
    tensors_to_color: Optional[frozenset[TensorLike]] = None,
) -> None:
    # If a color has been assigned, do nothing
    if tensor in assignments:
        return

    # If possible, choose the same color as the source
    source = tensors_to_sources.get(tensor)
    if source:
        _assign_tensor_color(source, tensors_to_sources, assignments)
        assignments[tensor] = assignments[source]
        return

    # TODO: Do we actually need the following block.
    # if tensor.root and tensor.root != tensor:
    #     _assign_tensor_color(tensor.root, tensors_to_sources, assignments)
    #     assignments[tensor] = assignments[tensor.root]
    #     return

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
    tensor: TensorLike,
    tensors_to_sources: Mapping[TensorLike, TensorLike],
    assignments: dict[TensorLike, str] = {},
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

    tensor_source = tensors_to_sources.get(tensor)
    if tensor_source:
        if isinstance(tensor, Tile):
            preferred_suffix = "t"
        elif "RF" in tensor.spec.bank:
            preferred_suffix = "r"
        else:
            # 's' is an arbitrary choice, but this case hardly ever occurs in
            # practice.
            preferred_suffix = "s"
        _assign_tensor_name(tensor_source, tensors_to_sources, assignments)
        assignments[tensor] = extend_name(
            _concat_word(assignments[tensor_source], preferred_suffix)
        )
        return

    # If we're dealing with a tensor that has no name or origin, just pick an
    # unused character from the alphabet.
    for char in utils.ALPHABET_PRODUCT:
        if char not in assignments.values():
            assignments[tensor] = char
            return
    raise Exception("Exhausted the alphabet")


def _map_tensors_to_sources(imp, _accum=None) -> Mapping[TensorLike, TensorLike]:
    """Returns a mapping from tensor/tile to its move source or outer tensor."""
    if _accum is None:
        _accum = {}
    for op in imp.operands:
        source = getattr(op, "origin", None)
        if source:
            _accum[op] = source
    if isinstance(imp, impl.MoveLet):
        _accum[imp.destination] = imp.operands[imp.source_idx]
    for child in imp.children:
        _map_tensors_to_sources(child, _accum)
    return _accum


class TensorNamer:
    def __init__(
        self,
        imp: impl.AppliedImpl,
        tensors_to_color: Optional[Iterable[TensorLike]] = None,
    ) -> None:
        self.name_assignments = dict()
        self.color_assignments = dict()
        self._tensors_to_source = _map_tensors_to_sources(imp)
        self._tensors_to_color = None
        if tensors_to_color:
            self._tensors_to_color = frozenset(tensors_to_color)

    def name(self, tensor: TensorLike, color: bool = False) -> str:
        _assign_tensor_name(tensor, self._tensors_to_source, self.name_assignments)
        result = self.name_assignments[tensor]
        if color:
            color_to_use = self._color(tensor)
            if color_to_use:
                result = termcolor.colored(result, color=color_to_use)
        return result

    def _color(self, tensor: TensorLike) -> Optional[str]:
        _assign_tensor_color(
            tensor,
            self._tensors_to_source,
            self.color_assignments,
            self._tensors_to_color,
        )
        return self.color_assignments[tensor]
