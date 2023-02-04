import dataclasses
from typing import TYPE_CHECKING, Iterable, Sequence

from .. import specs, system_config, utils
from .base import AppliedImpl, Impl, make_applied_impl

if TYPE_CHECKING:
    from ..tensor import TensorLike


@dataclasses.dataclass(frozen=True)
class Block(Impl):
    """A sequence of Impls."""

    spec: specs.Spec
    steps: tuple[Impl, ...]
    op_idxs: tuple[tuple[int, ...], ...]

    @property
    def is_scheduled(self) -> bool:
        return all(op.is_scheduled for op in self.steps)

    @property
    def children(self) -> tuple[Impl, ...]:
        return self.steps

    def replace_children(self, replacements: Iterable[Impl]) -> "Block":
        replacements = tuple(replacements)
        if len(replacements) != len(self.steps):
            raise ValueError(
                f"Expected {len(self.steps)} replacement children, but "
                f"got {len(replacements)}"
            )
        for original, replacement in zip(self.steps, replacements):
            if original.spec != replacement.spec:
                raise ValueError(
                    f"Cannot replace {original.spec} with {replacement.spec}; "
                    "specs differ"
                )
        return dataclasses.replace(self, steps=replacements)

    def complete(self) -> Impl:
        return self.replace_children((c.complete() for c in self.steps))

    def apply(self, operands: Sequence["TensorLike"]) -> AppliedImpl:
        applied_steps = []
        for child, child_idxs in zip(self.steps, self.op_idxs):
            applied_steps.append(child.apply([operands[i] for i in child_idxs]))
        return make_applied_impl(self.replace_children(applied_steps), operands)
