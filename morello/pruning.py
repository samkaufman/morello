import abc
from collections.abc import Mapping
from typing import Optional

from frozendict import frozendict

from . import ops, system_config, utils
from .tensor import Tensor


def _zero_banks() -> dict[str, int]:
    return {bank: 0 for bank in system_config.current_system().banks}


class AvailableIsNegativeError(ValueError):
    pass


class IntermediatesTooBigError(ValueError):
    pass


def _pipeline_transition(
    base_available: dict[str, int],
    pipeline: ops.Pipeline,
    carried_input_consumption: dict[str, int],
    carried_output_consumption: dict[str, int],
) -> Optional[list["MemoryLimits"]]:
    def _tensor_mem(tensor: Tensor) -> dict[str, int]:
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
                _tensor_mem(pipeline.stages[0].output),
            )
        )

        # The intermediate children carry the base available, as well as their
        # input and output intermediate tensors' limits
        for child_idx in range(1, len(pipeline.stages) - 1):
            before = _tensor_mem(pipeline.stages[child_idx - 1].output)
            after = _tensor_mem(pipeline.stages[child_idx].output)
            child_limits.append(
                PipelineChildMemoryLimits(base_available, before, after)
            )

        # The last child carries base available minus its input intermediate
        child_limits.append(
            PipelineChildMemoryLimits(
                base_available,
                _tensor_mem(pipeline.stages[-2].output),
                carried_output_consumption,
            )
        )
        return child_limits
    except IntermediatesTooBigError:
        return None


class MemoryLimits(abc.ABC):
    @property
    @abc.abstractmethod
    def available(self) -> frozendict[str, int]:
        pass

    @abc.abstractmethod
    def transition(self, schedule: ops.Schedule) -> Optional[list["MemoryLimits"]]:
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
    _available: frozendict[str, int]

    def __init__(self, available_memory: Optional[Mapping[str, int]] = None) -> None:
        super().__init__()
        if available_memory is None:
            system = system_config.current_system()
            available = {}
            for bank, bank_config in system.banks.items():
                available[bank] = bank_config.capacity
            self._available = frozendict(available)
        else:
            if any(m < 0 for m in available_memory.values()):
                raise AvailableIsNegativeError(
                    f"Given negative available memory: {available_memory}"
                )
            self._available = frozendict(available_memory)

    def __eq__(self, other) -> bool:
        if not isinstance(other, StandardMemoryLimits):
            return False
        return self.available == other.available

    def __hash__(self) -> int:
        return hash(self.available)

    def __str__(self) -> str:
        s = f"{type(self).__name__}("
        s += ", ".join(f"{k}={self.available[k]}" for k in sorted(self.available))
        return s + ")"

    @property
    def available(self) -> frozendict[str, int]:
        return self._available

    def transition(self, schedule: ops.Schedule) -> Optional[list["MemoryLimits"]]:
        # base->pipeline
        if isinstance(schedule, ops.Pipeline):
            return _pipeline_transition(
                dict(self.available), schedule, _zero_banks(), _zero_banks()
            )

        # base->base
        child_limits = []
        for adds in schedule.additional_memories:
            zd = utils.zip_dict(self._available, adds, same_keys=True).items()
            child_limits.append({k: m - d for k, (m, d) in zd})
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
            for v, b in utils.zip_dict(input_consumption, base, same_keys=True).values()
        ):
            raise IntermediatesTooBigError("input tensor doesn't fit in available")
        if any(
            v > b
            for v, b in utils.zip_dict(
                output_consumption, base, same_keys=True
            ).values()
        ):
            raise IntermediatesTooBigError("output tensor doesn't fit in available")

        self.base_available = base
        self.input_consumption = input_consumption
        self.output_consumption = output_consumption

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
    def available(self) -> frozendict[str, int]:
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
                ).items()
            }
        )

    def transition(self, schedule: ops.Schedule) -> Optional[list["MemoryLimits"]]:
        # pipeline->base; treat self as a StandardMemoryLimits and transition
        # normally.
        if not isinstance(schedule, ops.Pipeline):
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
