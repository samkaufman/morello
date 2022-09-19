import dataclasses
import functools
import warnings
from typing import Callable, Iterable, Optional, Sequence

from .. import dtypes, layouts, specs, system_config
from ..tensor import OperandIdx, TensorLike
from .actions import MatmulSplitAction, TileOutAction
from .base import AppliedImpl, Impl, NonAllocatingLeaf, make_applied_impl
from .block import Block
from .loops import Loop
from .moves import Moveable, common_operand_move_actions
from .pruning import (
    ParentSummary,
    break_matmul_split_symmetries,
    break_moves_symmetries,
    break_tile_out_symmetries,
    prune_nested_parallel_loops,
    prune_relayout_cycles,
)
from .utils import assert_stable_spec, dim_range, gen_tile_sizes
from .zero import ZeroHole

_BROADCAST_VEC_MULT_WIDTH = 256 // 8  # bytes


@dataclasses.dataclass(frozen=True)
class MatmulHoleBase(NonAllocatingLeaf, Moveable):
    spec: specs.Spec

    @property
    def is_scheduled(self) -> bool:
        return False

    def replace_spec(self, new_spec: specs.Spec) -> "Impl":
        return type(self)(new_spec)

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        out = self.spec.output

        # Search only over full line sizes
        for h, w in gen_tile_sizes(out.dim_sizes, filter=self._can_tile_out):
            yield TileOutAction(self, (h, w), parallel=False)
            if not self.spec.serial_only:
                yield TileOutAction(self, (h, w), parallel=True)

        yield from common_operand_move_actions(self)

    def apply(self, operands: Sequence[TensorLike]) -> "AppliedImpl":
        return make_applied_impl(self, operands)


class MatmulHole(MatmulHoleBase):
    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from super().actions(parent_summary)
        # NOTE: Don't yield `split` here.
        yield self.to_accum

    @assert_stable_spec
    def split(self, *args, **kwargs) -> "Impl":
        block = self.to_accum()
        matmul_accum = block.children[1]
        assert isinstance(
            matmul_accum, MatmulAccumHole
        ), f"Expected MatmulAccum, but is {type(matmul_accum).__name__}"
        new_children = list(block.children)
        new_children[1] = matmul_accum.split(*args, **kwargs)
        return block.replace_children(new_children)

    @assert_stable_spec
    def complete(self) -> Impl:
        return self.to_accum().complete()

    def to_accum(self) -> Impl:
        s = self.spec
        assert isinstance(s, specs.Matmul)
        zero_hole = ZeroHole(specs.Zero(s.output, serial_only=s.serial_only))
        accum_spec = specs.MatmulAccum(s.lhs, s.rhs, s.output, s.serial_only)
        accum_hole = MatmulAccumHole(accum_spec)

        out_idx = len(s.inputs)
        return Block(
            s, (zero_hole, accum_hole), ((out_idx,), tuple(range(len(s.operands))))
        )


class MatmulAccumHole(MatmulHoleBase):
    def __post_init__(self):
        assert isinstance(self.spec, specs.MatmulAccum)

    @prune_nested_parallel_loops
    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from super().actions(parent_summary)

        lhs, _ = self.spec.inputs
        if lhs.dim_sizes[1] > 1:
            for k in dim_range(lhs.dim_sizes[1], include_end=False):
                if self._split_valid(k):
                    yield MatmulSplitAction(self.split, size=k)

        if Mult.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, Mult)

        if BroadcastVecMult.applies_to_operands(self.spec.operands):
            yield functools.partial(self.place, BroadcastVecMult)

    @assert_stable_spec
    def split(self, size: int) -> "Impl":
        assert size > 0
        lhs, rhs = self.spec.inputs
        if size > lhs.dim_sizes[1]:
            raise ValueError(f"Cannot split {size} with inner dim. {lhs.dim_sizes[1]}")
        if size == lhs.dim_sizes[1]:
            return self
        left_view = lhs.simple_tile(OperandIdx(0), (lhs.dim_sizes[0], size))
        right_view = rhs.simple_tile(OperandIdx(1), (size, rhs.dim_sizes[1]))

        split_subscript = self.spec.operands_dim_subscripts()[0][-1]

        warnings.warn("Not yet specializing spec for split Matmuls")
        return Loop(
            spec=self.spec,
            subscripts=(split_subscript,),
            tiles=frozenset([left_view, right_view]),
            inner=MatmulAccumHole(
                specs.MatmulAccum(
                    left_view.spec, right_view.spec, self.spec.output, serial_only=True
                )
            ),
            parallel=False,  # TODO: Is this parallel correct?
        )

    def _split_valid(self, k: int) -> bool:
        lhs_h, lhs_w = self.spec.inputs[0].dim_sizes
        rhs_h, rhs_w = self.spec.inputs[1].dim_sizes
        assert lhs_w == rhs_h
        if k > lhs_w:
            return False
        if not self.spec.inputs[0].is_valid_tile_shape((lhs_h, k)):
            return False
        if not self.spec.inputs[1].is_valid_tile_shape((k, rhs_w)):
            return False
        return True

    @assert_stable_spec
    def complete(self) -> Impl:
        system = system_config.current_system()

        lhs, rhs = self.spec.inputs
        if lhs.dim_sizes[0] > 1 or rhs.dim_sizes[1] > 1:
            return self.tile_out((1, 1)).complete()
        if lhs.dim_sizes[1] > 1:
            return self.split(1).complete()

        next_general_lhs = system.next_general_bank(lhs.bank)
        if next_general_lhs:
            return self.move_input(0, bank=next_general_lhs).complete()
        next_general_rhs = system.next_general_bank(rhs.bank)
        if next_general_rhs:
            return self.move_input(1, bank=next_general_rhs).complete()
        next_general_out = system.next_general_bank(self.spec.output.bank)
        if next_general_out:
            return self.move_output(bank=next_general_out).complete()
        return self.place(Mult)


@dataclasses.dataclass(frozen=True)
class MatmulLeaf(NonAllocatingLeaf):
    spec: specs.Matmul

    @property
    def is_scheduled(self) -> bool:
        return True

    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from []

    def complete(self, *args, **kwargs):
        return self


@dataclasses.dataclass(frozen=True)
class Mult(MatmulLeaf):
    def __post_init__(self):
        assert all(o.bank in ("RF", "HexagonRF") for o in self.spec.operands)
        assert all(d == 1 for o in self.spec.operands for d in o.dim_sizes)

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return all(
            o.bank in ("RF", "HexagonRF") and all(d == 1 for d in o.dim_sizes)
            for o in operands
        )


@dataclasses.dataclass(frozen=True)
class BroadcastVecMult(MatmulLeaf):
    """A leaf for a scalar-vector multiplication (Clang vector extensions)."""

    def __post_init__(self):
        # TODO: Remove following.
        assert isinstance(self.spec, specs.MatmulAccum)
        check_result = BroadcastVecMult._check_operands(self.spec.operands)
        if check_result:
            raise ValueError(check_result)

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        return BroadcastVecMult._check_operands(operands) is None

    @staticmethod
    def _check_operands(operands: Sequence[specs.TensorSpec]) -> Optional[str]:
        lhs, rhs, out = operands

        # TODO: Maybe model the AVX registers explicitly instead of using RF.
        if lhs.bank != "RF" or rhs.bank != "RF" or out.bank != "RF":
            return "BroadcastVecMult only supports RF operands"

        # The Clang vector extensions require the rhs and output to be aligned.
        if not rhs.aligned:
            return "rhs must be aligned, but was: " + str(rhs)
        if not out.aligned:
            return "out must be aligned, but was: " + str(out)

        # lhs is contiguous because it's 1 value.
        if not rhs.contiguous:
            return "rhs must be contiguous, but was: " + str(rhs)
        if not out.contiguous:
            return "out must be contiguous, but was: " + str(out)

        if lhs.dtype != rhs.dtype:
            return f"Operand value types must match; lhs and rhs were {lhs.dtype} and {rhs.dtype}"
        if lhs.dtype != out.dtype:
            return f"Operand value types must match; lhs and out were {lhs.dtype} and {out.dtype}"

        if any(d != 1 for d in lhs.dim_sizes):
            return f"lhs must have one value, but had shape: {lhs.dim_sizes}"
        if len(rhs.dim_sizes) != 2 or rhs.dim_sizes[0] != 1:
            return f"rhs should have shape 1xn, but had shape: {rhs.dim_sizes}"
        if out.dim_sizes != (1, rhs.dim_sizes[1]):
            return f"out should have shape 1x{rhs.dim_sizes[1]}, but had shape: {out.dim_sizes}"

        assert _BROADCAST_VEC_MULT_WIDTH % out.dtype.size == 0
        if out.dim_sizes[1] != _BROADCAST_VEC_MULT_WIDTH // (out.dtype.size):
            return f"Expects {_BROADCAST_VEC_MULT_WIDTH}-byte operands"

        return None


@dataclasses.dataclass(frozen=True)
class HvxGemvmpybbwAsm(MatmulLeaf):
    """Impl that invokes hexagon_nn's gemvmpybbw_asm function."""

    def __post_init__(self):
        super().__post_init__()
        check_result = HvxGemvmpybbwAsm._check_operands(self.operands)
        if check_result:
            raise ValueError(check_result)

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        if HvxGemvmpybbwAsm._check_operands(operands):
            return False
        return True

    @staticmethod
    def _check_operands(operands: Sequence[specs.TensorSpec]) -> Optional[str]:
        lhs, rhs, out = operands

        if lhs.bank != "L2":
            # The left-hand side will be prefetched into L1 by the operation
            # itself.
            return "lhs must be in L2"
        if rhs.bank != "L2":
            return "rhs must be in L2"
        if out.bank != "L2":
            return "out must be in L2"

        if not lhs.layout.is_row_major:
            return "lhs must be in row-major"
        if rhs.layout != layouts.HEXAGON_TRANSPACKED:
            return "rhs must be transpacked"
        if not out.layout.is_row_major:
            return "out must be in row-major"

        if lhs.dtype != dtypes.Uint8:
            return "lhs should be uint8"
        if rhs.dtype != dtypes.Uint8:
            return "rhs should be uint8"
        if out.dtype != dtypes.Uint32:
            return "out should be uint32"

        # The n dimension below is called m by the implementation.
        m, _ = lhs.dim_sizes
        k, n = rhs.dim_sizes
        if m != 1:
            return f"m must be 1; was: {m}"
        if k < 16 or k % 16 != 0:
            return f"k dimension must be a non-zero multiple of 16; was {k}"
        if n > 32:
            return f"n must be at most 32; was {n}"

        return None


@dataclasses.dataclass(frozen=True)
class HvxVrmpyaccVuwVubRub(MatmulLeaf):
    def __post_init__(self):
        super().__post_init__()
        check_result = HvxVrmpyaccVuwVubRub._check_operands(self.spec.operands)
        if check_result:
            raise ValueError(check_result)

    @staticmethod
    def applies_to_operands(operands: Sequence[specs.TensorSpec]) -> bool:
        if HvxVrmpyaccVuwVubRub._check_operands(operands):
            return False
        return True

    @staticmethod
    def _check_operands(operands: Sequence[specs.TensorSpec]) -> Optional[str]:
        lhs, rhs, out = operands

        if lhs.bank != "VMEM":
            return "lhs must be in vector memory"
        if rhs.bank != "HexagonRF":
            return "rhs must be in scalar registers"
        if out.bank != "VMEM":
            return "out must be in vector memory"

        if lhs.dtype != dtypes.Uint8:
            return "lhs should be uint8"
        if rhs.dtype != dtypes.Uint8:
            return "rhs should be uint8"
        if out.dtype != dtypes.Uint32:
            return "out should be uint8"

        if lhs.dim_sizes != (32, 4):
            return f"lhs must have shape 32x4, but had shape: {lhs.dim_sizes}"
        if rhs.dim_sizes != (4, 1):
            return f"rhs must have shape 4x1, but had shape: {rhs.dim_sizes}"
        if out.dim_sizes != (32, 1):
            return f"out must have shape 1x1, but had shape: {out.dim_sizes}"

        if not lhs.contiguous:
            return "lhs must be contiguous, but was: " + str(lhs)
        if not rhs.contiguous:
            return "rhs must be contiguous, but was: " + str(rhs)
        if not out.contiguous:
            return "out must be contiguous, but was: " + str(out)

        if not lhs.vector_count == 1:
            return f"lhs must be a single HVX vector, but was: {lhs}"
        if not out.vector_count == 1:
            return f"out must be a single HVX vector, but was: {out}"

        return None
