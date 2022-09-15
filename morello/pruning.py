import warnings
from collections.abc import Mapping
from typing import Optional

from frozendict import frozendict

from . import impl, specs, system_config, utils
from .utils import TinyMap


# If True, schedules will be saved as if they had memory limits, for all banks,
# that are the next highest power of 2. This discretizes the cache a bit, even
# though it
SNAP_CAP_TO_POWER_OF_TWO = True


class AvailableIsNegativeError(ValueError):
    pass


class IntermediatesTooBigError(ValueError):
    pass


def _pipeline_transition(
    base_available: dict[str, int],
    pipeline: impl.Pipeline,
    carried_input_consumption: dict[str, int],
    carried_output_consumption: dict[str, int],
) -> Optional[list["MemoryLimits"]]:
    def _tensor_mem(tensor: specs.TensorSpec) -> dict[str, int]:
        mem = _zero_banks()
        mem[tensor.bank] = tensor.bytes_used
        return mem

    child_limits: list["MemoryLimits"] = []
    try:
        # The first child carries the base available, minus an output tensor
        child_limits.append(
            PipelineChildMemoryLimits(
                base_available,
                carried_input_consumption,
                _tensor_mem(pipeline.stages[0].spec.output),
            )
        )

        # The intermediate children carry the base available, as well as their
        # input and output intermediate tensors' limits
        for child_idx in range(1, len(pipeline.stages) - 1):
            before = _tensor_mem(pipeline.stages[child_idx - 1].spec.output)
            after = _tensor_mem(pipeline.stages[child_idx].spec.output)
            child_limits.append(
                PipelineChildMemoryLimits(base_available, before, after)
            )

        # The last child carries base available minus its input intermediate
        child_limits.append(
            PipelineChildMemoryLimits(
                base_available,
                _tensor_mem(pipeline.stages[-2].spec.output),
                carried_output_consumption,
            )
        )
        return child_limits
    except IntermediatesTooBigError:
        return None


class MemoryLimits:
    @property
    def available(self) -> Mapping[str, int]:
        raise NotImplementedError()

    def transition(self, schedule: impl.Impl) -> Optional[list["MemoryLimits"]]:
        """Returns new limits for the children of the given schedule.

        Returns `None` if the given schedule violates limits and therefore cannot be
        scheduled.

        Some MemoryLimits, such as PipelineMemoryLimits, may return different memory
        limits for individual children. In the case of pipelines, this is because
        intermediate tensors are only live during the execution of their adjacent
        stages.
        """
        raise NotImplementedError()


class StandardMemoryLimits(MemoryLimits):
    _available: TinyMap[str, int]

    def __init__(self, available_memory: Optional[Mapping[str, int]] = None) -> None:
        super().__init__()
        if available_memory is None:
            system = system_config.current_system()
            self._available = TinyMap(
                system.ordered_banks,
                tuple(system.banks[b].capacity for b in system.ordered_banks),
            )
        else:
            if any(m < 0 for m in available_memory.values()):
                raise AvailableIsNegativeError(
                    f"Given negative available memory: {available_memory}"
                )
            if isinstance(available_memory, TinyMap):
                self._available = available_memory
            else:
                self._available = TinyMap(available_memory)
        self._available = _snap_availables(self._available)

    def __eq__(self, other) -> bool:
        if not isinstance(other, StandardMemoryLimits):
            return NotImplemented
        return self.available == other.available

    def __hash__(self) -> int:
        return hash(self.available)

    def __str__(self) -> str:
        s = f"{type(self).__name__}("
        s += ", ".join(f"{k}={self.available[k]}" for k in sorted(self.available))
        return s + ")"

    @property
    def available(self) -> TinyMap[str, int]:
        return self._available

    def transition(self, schedule: impl.Impl) -> Optional[list["MemoryLimits"]]:
        # base->pipeline
        if isinstance(schedule, impl.Pipeline):
            return _pipeline_transition(
                dict(self.available), schedule, _zero_banks(), _zero_banks()
            )

        # base->base
        child_limits = []
        for adds in schedule.additional_memories:
            assert self._available.raw_keys == adds.raw_keys
            child_limits.append(
                TinyMap(
                    self._available.raw_keys,
                    tuple(
                        m - d
                        for m, d in zip(self._available.raw_values, adds.raw_values)
                    ),
                )
            )
        assert len(child_limits) == len(schedule.children), (
            f"{len(child_limits)} child limits != {len(schedule.children)} "
            f"children for {type(schedule).__name__}"
        )
        if any(
            m < 0
            for memory_after_action in child_limits
            for m in memory_after_action.values()
        ):
            # This violates the limits, so we return None
            return None

        return [StandardMemoryLimits(a) for a in child_limits]


class PipelineChildMemoryLimits(MemoryLimits):
    """A MemoryLimits carrying extra information for a hole in a Pipeline.

    This should not be mutated.
    """

    base_available: dict[str, int]
    input_consumption: dict[str, int]
    output_consumption: dict[str, int]

    def __init__(
        self,
        base: dict[str, int],
        input_consumption: dict[str, int],
        output_consumption: dict[str, int],
    ) -> None:
        super().__init__()
        if any(
            v > b
            for _, (v, b) in utils.zip_dict(input_consumption, base, same_keys=True)
        ):
            raise IntermediatesTooBigError("input tensor doesn't fit in available")
        if any(
            v > b
            for _, (v, b) in utils.zip_dict(output_consumption, base, same_keys=True)
        ):
            raise IntermediatesTooBigError("output tensor doesn't fit in available")

        self.base_available = base

        # Update input and output consumptions using the adjustments
        snapped_base_available = _snap_availables(TinyMap(self.base_available))
        adjustments: Mapping[str, int] = {
            k: orig - v
            for k, (orig, v) in utils.zip_dict(
                self.base_available, snapped_base_available, same_keys=True
            )
        }

        self.input_consumption = {
            k: max(0, orig - adjustment)
            for k, (orig, adjustment) in utils.zip_dict(
                input_consumption, adjustments, same_keys=True
            )
        }
        self.output_consumption = {
            k: max(0, orig - adjustment)
            for k, (orig, adjustment) in utils.zip_dict(
                output_consumption, adjustments, same_keys=True
            )
        }

        # Assert that the important, observable numbers are snapped. (The actual
        # snapping should happen in the transition function.)
        # assert not SNAP_CAP_TO_POWER_OF_TWO or all(
        #     _is_snapped(v) for v in self.base_available.values()
        # )
        # assert not SNAP_CAP_TO_POWER_OF_TWO or all(
        #     _is_snapped(v) for v in self.available.values()
        # )
        # assert not SNAP_CAP_TO_POWER_OF_TWO or all(
        #     _is_snapped(a - b) and _is_snapped(a - c)
        #     for _, (a, b, c) in utils.zip_dict(
        #         self.base_available,
        #         self.input_consumption,
        #         self.output_consumption,
        #         same_keys=True,
        #     )
        # )
        warnings.warn(
            "PipelineChildMemoryLimits snapping behavior isn't very "
            "well-defined yet. Base available memory is snapped, but input- and "
            "output-specific memory adjustments just have the snap differences "
            "removed."
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, PipelineChildMemoryLimits):
            return False
        return (
            self.base_available == other.base_available
            and self.input_consumption == other.input_consumption
            and self.output_consumption == other.output_consumption
        )

    def __hash__(self) -> int:
        return hash(
            (self.base_available, self.input_consumption, self.output_consumption)
        )

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.base_available}, "
            f"{self.input_consumption}, {self.output_consumption})"
        )

    @property
    def available(self) -> Mapping[str, int]:
        # This property is the interface for any caller expecting a base
        # StandardMemoryLimits (i.e. one that can't use extra information about
        # its context in a Pipeline).
        return frozendict(
            {
                bank: a - (b + c)
                for bank, (a, b, c) in utils.zip_dict(
                    self.base_available,
                    self.input_consumption,
                    self.output_consumption,
                    same_keys=True,
                )
            }
        )

    def transition(self, schedule: impl.Impl) -> Optional[list["MemoryLimits"]]:
        # pipeline->base; treat self as a StandardMemoryLimits and transition
        # normally.
        if not isinstance(schedule, impl.Pipeline):
            # If we're transitioning to a non-Pipeline, we lose the precision of a
            # PipelineChildMemoryLimits. It's possible, at this point, that the uniform
            # lower bound on available memory becomes negative, in which case we'll
            # return None.
            try:
                standard_limits = StandardMemoryLimits(self.available)
            except AvailableIsNegativeError:
                return None
            return standard_limits.transition(schedule)

        # pipeline->pipeline; push input/output information into inner
        # PipelineChildMemoryLimits
        return _pipeline_transition(
            self.base_available,
            schedule,
            self.input_consumption,
            self.output_consumption,
        )


def _snap_availables(available: TinyMap[str, int]) -> TinyMap[str, int]:
    """Returns limits that are snapped down according to the snapping strategy."""
    # If SNAP_CAP_TO_POWER_OF_TWO isn't set, don't rebuild the data structure.
    if not SNAP_CAP_TO_POWER_OF_TWO:
        return available
    return available.map_values(_snap_down)


def _snap_down(n: int) -> int:
    """Snaps an integer down according to the snapping strategy.

    No-op for already-snapped integers.
    """
    assert SNAP_CAP_TO_POWER_OF_TWO
    assert n >= 0
    if n == 0:
        return 0
    # Return the greatest power of two equal to or less than n
    return 2 ** (n.bit_length() - 1)


def _zero_banks() -> dict[str, int]:
    return {bank: 0 for bank in system_config.current_system().banks}
