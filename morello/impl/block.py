import dataclasses
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

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

    def subschedule(self, idx: int, fn: Callable[[Impl], Impl]) -> "Block":
        new_steps = list(self.steps)
        new_steps[idx] = fn(new_steps[idx])
        return self.replace_children(new_steps)

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

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        z = {k: 0 for k in system_config.current_system().banks}
        return [z for _ in self.steps]

    @property
    def peak_memory(self) -> dict[str, int]:
        zipped = utils.zip_dict(*[c.peak_memory for c in self.steps], same_keys=True)
        return {k: max(vs) for k, vs in zipped.items()}

    def apply(self, operands: Sequence["TensorLike"]) -> AppliedImpl:
        applied_steps = []
        for child, child_idxs in zip(self.steps, self.op_idxs):
            applied_steps.append(child.apply([operands[i] for i in child_idxs]))
        return make_applied_impl(self.replace_children(applied_steps), operands)

