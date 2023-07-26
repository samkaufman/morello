import contextlib
from contextvars import ContextVar
from typing import Union

from .base import SystemDescription, Target

_CURRENT_TARGET: ContextVar["Target"] = ContextVar("_CURRENT_TARGET")


class NoTargetSetException(Exception):
    pass


def current_target() -> Target:
    try:
        return _CURRENT_TARGET.get()
    except LookupError:
        raise NoTargetSetException("No target set. Call set_current_target.")


def set_current_target(target: Union[str, Target]):
    if isinstance(target, str):
        return set_current_target(target_by_name(target))
    _CURRENT_TARGET.set(target)


@contextlib.contextmanager
def with_target(target: Target):
    token = _CURRENT_TARGET.set(target)
    try:
        yield
    finally:
        _CURRENT_TARGET.reset(token)


def target_by_name(target_name: str) -> Target:
    # Import inside the method to solve a circular dependency
    from . import cpu

    if target_name == "x86":
        return cpu.X86Target()
    elif target_name == "arm":
        return cpu.ArmTarget()
    else:
        raise ValueError("Unknown target: " + target_name)


def current_system() -> SystemDescription:
    """Sugar accessor for `current_target().system`."""
    return current_target().system
