import warnings
from collections.abc import Mapping
from typing import Optional

from . import impl, specs, system_config, utils
from .utils import TinyMap, snap_availables_down


class AvailableIsNegativeError(ValueError):
    pass


class IntermediatesTooBigError(ValueError):
    pass


def _pipeline_transition(
    base_available: TinyMap[str, int],
    pipeline: impl.Pipeline,
    carried_input_consumption: TinyMap[str, int],
    carried_output_consumption: TinyMap[str, int],
) -> Optional[list["MemoryLimits"]]:
    def _tensor_mem(tensor: specs.TensorSpec) -> TinyMap[str, int]:
        return _zero_banks().replace_value(tensor.bank, tensor.bytes_used)

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

    def __init__(
        self, available_memory: Optional[Mapping[str, int]] = None, snap_down=True
    ) -> None:
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
        if snap_down:
            self._available = snap_availables_down(self._available)

    def __eq__(self, other) -> bool:
        if not isinstance(other, StandardMemoryLimits):
            return NotImplemented
        return self.available == other.available

    def __hash__(self) -> int:
        return hash(self.available)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self._available)})"

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
                self.available, schedule, _zero_banks(), _zero_banks()
            )

        # base->base
        child_limits = []
        for adds in schedule.additional_memories:
            assert self._available.raw_keys == adds.raw_keys, (
                f"Memory levels do not match; {self._available.raw_keys} != "
                f"{adds.raw_keys}"
            )
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

    base_available: TinyMap[str, int]
    input_consumption: TinyMap[str, int]
    output_consumption: TinyMap[str, int]

    def __init__(
        self,
        base: TinyMap[str, int],
        input_consumption: TinyMap[str, int],
        output_consumption: TinyMap[str, int],
    ) -> None:
        super().__init__()
        if base.raw_keys != input_consumption.raw_keys:
            raise ValueError("base and input_consumption TinyMaps' keys must match")
        if base.raw_keys != output_consumption.raw_keys:
            raise ValueError("base and output_consumption TinyMaps' keys must match")

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

        # Update input and output consumptions using the adjustments
        adjustments = tuple(
            orig - v
            for orig, v in zip(base.raw_values, snap_availables_down(base).raw_values)
        )

        self.base_available = base
        self.input_consumption = TinyMap(
            base.raw_keys,
            tuple(
                max(0, orig - adjustment)
                for orig, adjustment in zip(input_consumption.raw_values, adjustments)
            ),
        )
        self.output_consumption = TinyMap(
            base.raw_keys,
            tuple(
                max(0, orig - adjustment)
                for orig, adjustment in zip(output_consumption.raw_values, adjustments)
            ),
        )

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
        banks = system_config.current_system().ordered_banks
        assert self.base_available.raw_keys == banks
        assert self.input_consumption.raw_keys == banks
        assert self.output_consumption.raw_keys == banks
        return TinyMap(
            banks,
            tuple(
                a - (b + c)
                for a, b, c in zip(
                    self.base_available.raw_values,
                    self.input_consumption.raw_values,
                    self.output_consumption.raw_values,
                )
            ),
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


def _zero_banks() -> TinyMap[str, int]:
    banks = system_config.current_system().ordered_banks
    return TinyMap(banks, (0,) * len(banks))
