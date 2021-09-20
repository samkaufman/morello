import contextlib
from contextvars import ContextVar

from .base import Target, SystemDescription

_CURRENT_TARGET: ContextVar["Target"] = ContextVar("_CURRENT_TARGET")


class NoTargetSetException(Exception):
    pass


def current_target() -> Target:
    try:
        return _CURRENT_TARGET.get()
    except LookupError:
        raise NoTargetSetException()


def set_current_target(target: Target):
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
    from . import cpu, hexagon

    if target_name == "hvxsim":
        return hexagon.HvxSimulatorTarget()
    elif target_name == "cpu":
        return cpu.CpuTarget()
    else:
        raise ValueError("Unknown target: " + target_name)


def current_system() -> SystemDescription:
    """Sugar accessor for `current_target().system`."""
    return current_target().system
