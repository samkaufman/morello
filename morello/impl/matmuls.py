import functools
import warnings
from typing import Union, Callable, Tuple, Iterable, Optional, Sequence

import dataclass_abc

from .actions import TileOutAction, MatmulSplitAction
from .base import Impl
from .loops import MatmulSplitLoop
from .moves import MoveLet, PadTranspack, common_operand_move_actions, common_move
from .pruning import (
    prune_relayout_cycles,
    break_moves_symmetries,
    break_tile_out_symmetries,
    break_matmul_split_symmetries,
    ParentSummary,
)
from .utils import assert_stable_spec, gen_tile_sizes, dim_range
from .. import specs, system_config, dtypes
from ..specs import Layout
from ..system_config import current_target
from ..tensor import Tensor, Tile


@dataclass_abc.dataclass_abc(frozen=True)
class MatmulBase(Impl):
    lhs: Union[Tensor, Tile]  # n-by-m
    rhs: Union[Tensor, Tile]  # m-by-p
    output: Union[Tensor, Tile]  # m-by-n
    serial_only: bool

    def __post_init__(self):
        lw, rh = self.lhs.dim_sizes[1], self.rhs.dim_sizes[0]
        assert lw == rh, f"Inner dims. of Matmul operands don't match: {lw} and {rh}"

    @property
    def inputs(self) -> tuple[Union[Tensor, Tile], ...]:
        return self.lhs, self.rhs

    @functools.cached_property
    def spec(self) -> specs.Matmul:
        return specs.Matmul(
            self.lhs.spec, self.rhs.spec, self.output.spec, serial_only=self.serial_only
        )

    def env_str(
        self,
        name_tensor_fn: Callable[[Union[Tensor, Tile]], str],
        underscore_inner: bool = False,
        fancy: bool = False,
    ):
        return (
            f"{type(self).__name__}({name_tensor_fn(self.lhs)}, "
            f"{name_tensor_fn(self.rhs)}, "
            f"{name_tensor_fn(self.output)})"
        )

    @property
    def children(self) -> Tuple["Impl", ...]:
        return tuple()

    @assert_stable_spec
    def replace_children(self, replacements: Iterable[Impl]) -> Impl:
        replacements = list(replacements)
        if replacements:
            raise Exception("Matmul has no children to replace")
        return self

    @property
    def peak_memory(self) -> dict[str, int]:
        return {k: 0 for k in system_config.current_system().banks}

    def replace_io(
        self, inputs: Iterable[Union[Tensor, Tile]], output: Union[Tensor, Tile]
    ) -> "Impl":
        lhs, rhs = inputs
        return type(self)(lhs, rhs, output, serial_only=self.serial_only)


@dataclass_abc.dataclass_abc(frozen=True)
class MatmulHole(MatmulBase):
    @property
    def is_scheduled(self) -> bool:
        return False

    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        # Search only over full line sizes
        for h, w in gen_tile_sizes(self.output.dim_sizes, filter=self._can_tile_out):
            for parallel in [False] if self.serial_only else [True, False]:
                yield TileOutAction(self, (h, w), parallel)

        if self.lhs.dim_sizes[1] > 1:
            for k in dim_range(self.lhs.dim_sizes[1], include_end=False):
                if self._split_valid(k):
                    yield MatmulSplitAction(self.split, size=k)

        yield from common_operand_move_actions(self)

        if Mult.applies_to_operands(self.operands):
            yield self.place_mult

        if system_config.current_system().has_hvx:
            if HvxVrmpyaccVuwVubRub.applies_to_operands(self.operands):
                yield self.place_hvx_vrmpyacc

    def move_input(
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        if input_idx == 0:
            return common_move(self, "lhs", bank, layout, prefetching, **kwargs)
        elif input_idx == 1:
            return common_move(self, "rhs", bank, layout, prefetching, **kwargs)
        else:
            raise ValueError("input_idx must be 0 or 1")

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        return common_move(self, "output", bank, layout, prefetching, **kwargs)

    @assert_stable_spec
    def pad_transpack(self, input_idx: int) -> "Impl":
        source = self.inputs[input_idx]
        new_mat = current_target().tensor(
            spec=current_target().tensor_spec(
                dim_sizes=source.dim_sizes,
                dtype=source.dtype,
                bank="GL",
                layout=Layout.HEXAGON_TRANSPACKED,
            ),
            origin=source,
        )

        new_inputs = self.inputs[:input_idx] + (new_mat,) + self.inputs[input_idx + 1 :]
        return PadTranspack(
            source=source,
            destination=new_mat,
            input_idx=input_idx,
            inner=self.replace_io(new_inputs, self.output),
        )

    @assert_stable_spec
    def split(self, size: int) -> "Impl":
        assert size > 0
        if size > self.lhs.dim_sizes[1]:
            raise ValueError(
                f"Cannot split {size} with inner dim. {self.lhs.dim_sizes[1]}"
            )
        if size == self.lhs.dim_sizes[1]:
            return self
        left_view = self.lhs.simple_tile((self.lhs.dim_sizes[0], size))
        right_view = self.rhs.simple_tile((size, self.rhs.dim_sizes[1]))
        warnings.warn("Not yet specializing spec for split Matmuls")
        return MatmulSplitLoop(
            lhs=self.lhs,
            rhs=self.rhs,
            output=self.output,
            inner=MatmulHole(left_view, right_view, self.output, self.serial_only),
        )

    def _split_valid(self, k: int) -> bool:
        lhs_h, lhs_w = self.lhs.dim_sizes
        rhs_h, rhs_w = self.rhs.dim_sizes
        assert lhs_w == rhs_h
        if k > lhs_w:
            return False
        if not self.lhs.spec.is_valid_tile_shape((lhs_h, k)):
            return False
        if not self.rhs.spec.is_valid_tile_shape((k, rhs_w)):
            return False
        return True

    @assert_stable_spec
    def complete(self) -> Impl:
        system = system_config.current_system()
        if self.lhs.dim_sizes[0] > 1 or self.rhs.dim_sizes[1] > 1:
            return self.tile_out((1, 1)).complete()
        if self.lhs.dim_sizes[1] > 1:
            return self.split(1).complete()

        next_general_lhs = system.next_general_bank(self.lhs.bank)
        if next_general_lhs:
            return self.move_input(0, bank=next_general_lhs).complete()
        next_general_rhs = system.next_general_bank(self.rhs.bank)
        if next_general_rhs:
            return self.move_input(1, bank=next_general_rhs).complete()
        next_general_out = system.next_general_bank(self.output.bank)
        if next_general_out:
            return self.move_output(bank=next_general_out).complete()
        return self.place_mult()

    @assert_stable_spec
    def place_mult(self) -> "Mult":
        return Mult(self.lhs, self.rhs, self.output, self.serial_only)

    @assert_stable_spec
    def place_hvx_gemvmpebbw(self) -> "HvxGemvmpybbwAsm":
        return HvxGemvmpybbwAsm(self.lhs, self.rhs, self.output, self.serial_only)

    @assert_stable_spec
    def place_hvx_vrmpyacc(self) -> "HvxVrmpyaccVuwVubRub":
        return HvxVrmpyaccVuwVubRub(self.lhs, self.rhs, self.output, self.serial_only)

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return [{k: 0 for k in system_config.current_system().banks}]


@dataclass_abc.dataclass_abc(frozen=True)
class MatmulLeaf(MatmulBase):
    @property
    def is_scheduled(self) -> bool:
        return True

    @prune_relayout_cycles
    @break_moves_symmetries
    @break_tile_out_symmetries
    @break_matmul_split_symmetries
    def actions(
        self, parent_summary: Optional[ParentSummary] = None
    ) -> Iterable[Callable[[], Impl]]:
        yield from []

    def move_input(
        self,
        input_idx: int,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        raise NotImplementedError()

    def move_output(
        self,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        prefetching: bool = False,
        **kwargs,
    ) -> "MoveLet":
        raise NotImplementedError()

    @property
    def additional_memories(self) -> list[dict[str, int]]:
        return []


@dataclass_abc.dataclass_abc(frozen=True)
class Mult(MatmulLeaf):
    # TODO: Replace whole class w/ target-specific implementations

    def __post_init__(self):
        super().__post_init__()
        assert all(o.bank in ("RF", "HexagonRF") for o in self.operands)
        assert all(d == 1 for o in self.operands for d in o.dim_sizes)

    @staticmethod
    def applies_to_operands(operands: Sequence[Union[Tensor, Tile]]) -> bool:
        return all(
            o.bank in ("RF", "HexagonRF") and all(d == 1 for d in o.dim_sizes)
            for o in operands
        )


@dataclass_abc.dataclass_abc(frozen=True)
class HvxGemvmpybbwAsm(MatmulLeaf):
    """Impl that invokes hexagon_nn's gemvmpybbw_asm function."""

    def __post_init__(self):
        super().__post_init__()
        check_result = HvxGemvmpybbwAsm._check_operands(self.operands)
        if check_result:
            raise ValueError(check_result)

    @staticmethod
    def applies_to_operands(operands: Sequence[Union[Tensor, Tile]]) -> bool:
        if HvxGemvmpybbwAsm._check_operands(operands):
            return False
        return True

    @staticmethod
    def _check_operands(operands: Sequence[Union[Tensor, Tile]]) -> Optional[str]:
        lhs, rhs, out = operands

        if lhs.bank != "L2":
            # The left-hand side will be prefetched into L1 by the operation
            # itself.
            return "lhs must be in L2"
        if rhs.bank != "L2":
            return "rhs must be in L2"
        if out.bank != "L2":
            return "out must be in L2"

        if lhs.layout != Layout.ROW_MAJOR:
            return "lhs must be in row-major"
        if rhs.layout != Layout.HEXAGON_TRANSPACKED:
            return "rhs must be transpacked"
        if out.layout != Layout.ROW_MAJOR:
            return "out must be in row-major"

        if lhs.dtype != dtypes.Uint8:
            return "lhs should be uint8"
        if rhs.dtype != dtypes.Uint8:
            return "rhs should be uint8"
        if out.dtype != dtypes.Uint32:
            return "out should be uint8"

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


@dataclass_abc.dataclass_abc(frozen=True)
class HvxVrmpyaccVuwVubRub(MatmulLeaf):
    def __post_init__(self):
        super().__post_init__()
        check_result = HvxVrmpyaccVuwVubRub._check_operands(self.operands)
        if check_result:
            raise ValueError(check_result)

    @staticmethod
    def applies_to_operands(operands: Sequence[Union[Tensor, Tile]]) -> bool:
        if HvxVrmpyaccVuwVubRub._check_operands(operands):
            return False
        return True

    @staticmethod
    def _check_operands(operands: Sequence[Union[Tensor, Tile]]) -> Optional[str]:
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
