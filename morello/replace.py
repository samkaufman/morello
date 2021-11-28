"""Functions for replacing Tensors and Tiles in a Impl.

This module exists because maintaining reference equality for Tensors and Tiles
in the case that their `root` or `origin` properties are changed requires extra
bookkeeping to ensure that the correct, updated instances of those objects are
re-used during subsequent visits.
"""

import functools
import threading
from typing import Optional, TypeVar, Union, cast

from . import impl, specs
from .tensor import ConvolutionImageTile, SimpleTile, Tensor, TensorLike, Tile

T = TypeVar("T")

_tlocal = threading.local()
_tlocal.mutating_replace_stack = []


def _assert_no_cycles(func):
    """Wraps _mutating_replace to ensure there are no cycles."""

    if not __debug__:
        return func

    @functools.wraps(func)
    def wrapper_fail_on_cycles(*args, **kwargs):
        subject_id = id(args[0])
        if subject_id in _tlocal.mutating_replace_stack:
            raise Exception(f"Cycle found: {str(args[0])} (id: {subject_id})")
        _tlocal.mutating_replace_stack.append(subject_id)
        try:
            return func(*args, **kwargs)
        finally:
            _tlocal.mutating_replace_stack.pop()

    return wrapper_fail_on_cycles


@_assert_no_cycles
def _mutating_replace(
    subject: Optional[T], replacements: dict[TensorLike, TensorLike]
) -> Optional[T]:
    if subject is None:
        return None

    if isinstance(subject, TensorLike):
        try:
            return cast(T, replacements[subject])
        except KeyError:
            pass

    if type(subject) is Tensor:
        new_origin = _mutating_replace(subject.origin, replacements)
        if new_origin is subject.origin:
            return cast(T, subject)
        new_tensor = Tensor(
            spec=subject.spec,
            name=subject.name,
            origin=new_origin,
        )
        replacements[subject] = new_tensor
        return cast(T, new_tensor)
    elif type(subject) in (SimpleTile, ConvolutionImageTile):
        # After replacement, it's possible for what was once the subject Tile's
        # root, which should be a Tensor, to become a Tile. In this case, the
        # correct thing to do is to propagate that Tile's root. Since a Tensor's
        # root is itself, we just call `root` below.
        assert subject.origin.root == subject.root
        assert subject.root.root == subject.origin.root
        new_origin = _mutating_replace(subject.origin, replacements)

        if new_origin is subject.origin:
            return cast(T, subject)

        if isinstance(subject, SimpleTile):
            new_tile = SimpleTile(
                dim_sizes=subject.dim_sizes,
                name=subject.name,
                origin=new_origin,
            )
        else:
            new_tile = ConvolutionImageTile(
                filter_shape=subject.filter_shape,
                dim_sizes=subject.dim_sizes,
                name=subject.name,
                origin=new_origin,
            )
        replacements[subject] = new_tile
        return new_tile

    if type(subject) in (impl.MatmulHole, impl.Mult, impl.HvxVrmpyaccVuwVubRub):
        return type(subject)(
            lhs=_mutating_replace(subject.lhs, replacements),
            rhs=_mutating_replace(subject.rhs, replacements),
            output=_mutating_replace(subject.output, replacements),
            serial_only=subject.serial_only,
        )
    elif type(subject) is impl.DirectConv:
        return impl.DirectConv(
            lhs=_mutating_replace(subject.lhs, replacements),
            rhs=_mutating_replace(subject.rhs, replacements),
            output=_mutating_replace(subject.output, replacements),
            serial_only=subject.serial_only,
        )
    elif type(subject) is impl.MoveLet:
        return impl.MoveLet(
            source=_mutating_replace(subject.source, replacements),
            destination=_mutating_replace(subject.destination, replacements),
            prefetching=subject.prefetching,
            input_idx=subject.input_idx,
            inner=_mutating_replace(subject.inner, replacements),
        )
    elif type(subject) is impl.MatmulSplitLoop:
        return impl.MatmulSplitLoop(
            lhs=_mutating_replace(subject.lhs, replacements),
            rhs=_mutating_replace(subject.rhs, replacements),
            output=_mutating_replace(subject.output, replacements),
            inner=_mutating_replace(subject.inner, replacements),
        )
    elif type(subject) is impl.loops.Loop:
        return impl.Loop(
            driving_tile=_mutating_replace(subject.driving_tile, replacements),
            dependent_tiles=frozenset(
                _mutating_replace(t, replacements) for t in subject.dependent_tiles
            ),
            inner=_mutating_replace(subject.inner, replacements),
            parallel=subject.parallel,
        )
    elif type(subject) is impl.SlidingWindowLoop:
        new_inputs = tuple(_mutating_replace(t, replacements) for t in subject.inputs)
        new_output = _mutating_replace(subject.output, replacements)
        new_spec = subject.spec.replace_io(
            tuple(inp.spec for inp in new_inputs), new_output.spec
        )
        return impl.SlidingWindowLoop(
            inputs=new_inputs,
            output=new_output,
            live_tensor=_mutating_replace(subject.live_tensor, replacements),
            frontier_size=subject.frontier_size,
            other_tiles=tuple(
                _mutating_replace(t, replacements) for t in subject.other_tiles
            ),
            spec=new_spec,
            inner=_mutating_replace(subject.inner, replacements),
        )
    elif type(subject) is impl.ReduceSum:
        return impl.ReduceSum(
            source=_mutating_replace(subject.source, replacements),
            output=_mutating_replace(subject.output, replacements),
            serial_only=subject.serial_only,
        )
    elif type(subject) is impl.Pipeline:
        return impl.Pipeline(
            stages=tuple(_mutating_replace(s, replacements) for s in subject.stages)
        )
    elif type(subject) is impl.ComposeHole:
        new_inputs = tuple(_mutating_replace(t, replacements) for t in subject.inputs)
        new_output = _mutating_replace(subject.output, replacements)
        new_spec = specs.Compose(
            subspec_classes=subject.spec.subspec_classes,
            inputs=tuple(inp.spec for inp in new_inputs),
            output=new_output.spec,
            intermediate_dtypes=subject.spec.intermediate_dtypes,
            serial_only=subject.spec.serial_only,
        )
        return impl.ComposeHole(
            spec=new_spec,
            inputs=new_inputs,
            output=new_output,
        )
    else:
        raise NotImplementedError(f"Not implemented for {type(subject)}")


def replace(
    subject: T,
    replacements: dict[Union[Tensor, Tile], Union[Tensor, Tile]],
    update_replacements: bool = False,
) -> T:
    if all(k == v for k, v in replacements.items()):
        return subject
    if not update_replacements:
        replacements = dict(replacements)
    return _mutating_replace(subject, replacements)
