import dataclasses
from typing import TYPE_CHECKING, Iterable, Sequence

from .. import specs, system_config, utils
from .base import AppliedImpl, Impl, make_applied_impl

if TYPE_CHECKING:
    from ..tensor import TensorLike, Tile


@dataclasses.dataclass(frozen=True, slots=True)
class SpecCast(Impl):
    """A simple wrapper for Impls with casting operands."""

    spec: specs.Spec
    inner: Impl
    inner_args: frozenset["Tile"]

    @property
    def is_scheduled(self) -> bool:
        return self.inner.is_scheduled

    @property
    def children(self) -> tuple[Impl, ...]:
        return (self.inner,)

    def replace_children(self, replacements: Iterable[Impl]) -> "SpecCast":
        replacements = tuple(replacements)
        if len(replacements) != 1:
            raise ValueError(
                f"Expected 1 replacement child, but given {len(replacements)}"
            )
        if self.inner.spec != replacements[0].spec:
            raise ValueError(
                f"Cannot replace {self.inner.spec} with {replacements[0].spec}; "
                "specs differ"
            )
        return dataclasses.replace(self, inner=replacements[0])

    def move_input(self, *args, **kwargs) -> "SpecCast":
        return self.replace_children([self.inner.move_input(*args, **kwargs)])

    def move_output(self, *args, **kwargs) -> "SpecCast":
        return self.replace_children([self.inner.move_output(*args, **kwargs)])

    def complete(self) -> Impl:
        return self.replace_children([self.inner.complete()])

    @property
    def additional_memories(self) -> list[utils.TinyMap[str, int]]:
        banks = system_config.current_system().ordered_banks
        return [utils.TinyMap(banks, (0,) * len(banks))]

    @property
    def peak_memory(self) -> utils.TinyMap[str, int]:
        return self.inner.peak_memory

    def apply(self, operands: Sequence["TensorLike"]) -> AppliedImpl:
        assert [o.spec for o in operands] == list(self.spec.operands), (
            f"Operands do not match Spec. Spec operands were "
            f"{', '.join(str(o) for o in self.spec.operands)}, but given "
            f"{', '.join(str(o.spec) for o in operands)}."
        )

        inner_operands = list(operands)
        for inner_arg in self.inner_args:
            inner_operands[inner_arg.source] = inner_arg
        applied_body = self.inner.apply(inner_operands)
        return make_applied_impl(self.replace_children([applied_body]), operands)
