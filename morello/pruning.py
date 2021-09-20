import abc
from typing import Optional, Literal, cast

from . import ops, system_config
from .tensor import Tensor


def _zero_tup() -> tuple[Literal[0], ...]:
    return cast(
        tuple[Literal[0], ...],
        tuple(0 for _ in system_config.current_system().level_configs),
    )


class AvailableIsNegativeError(ValueError):
    pass


class IntermediatesTooBigError(ValueError):
    pass


def _pipeline_transition(
    base_available: tuple[int, ...],
    pipeline: ops.Pipeline,
    carried_input_consumption: tuple[int, ...],
    carried_output_consumption: tuple[int, ...],
) -> Optional[list["MemoryLimits"]]:

    # TODO: Make this a method of Tensors
    def _tensor_mem(tensor: Tensor) -> tuple[int, ...]:
        mem = list(_zero_tup())
        mem[tensor.level] = tensor.volume
        return tuple(mem)

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
    def available(self) -> tuple[int, ...]:
        pass

    @abc.abstractmethod
    def transition(self, schedule: ops.Schedule) -> Optional[list["MemoryLimits"]]:
        raise NotImplementedError()


class StandardMemoryLimits(MemoryLimits):
    _available: tuple[int, ...]

    def __init__(self, available_memory: Optional[tuple[int, ...]] = None) -> None:
        super().__init__()
        if available_memory is None:
            system = system_config.current_system()
            self._available = tuple(
                c.capacity * system.line_size for c in system.level_configs
            )
        else:
            if any(m < 0 for m in available_memory):
                raise AvailableIsNegativeError(
                    f"Given negative available memory: {available_memory}"
                )
            self._available = available_memory

    def __eq__(self, other) -> bool:
        if not isinstance(other, StandardMemoryLimits):
            return False
        return self.available == other.available

    def __hash__(self) -> int:
        return hash(self.available)

    @property
    def available(self) -> tuple[int, ...]:
        return self._available

    def transition(self, schedule: ops.Schedule) -> Optional[list["MemoryLimits"]]:
        """Returns new limits for the children (holes) of the given schedule.

        Returns `None` if the given schedule violates limits and is therefore
        unschedulable.
        """
        # base->pipeline
        if isinstance(schedule, ops.Pipeline):
            return _pipeline_transition(
                self.available, schedule, _zero_tup(), _zero_tup()
            )

        # base->base
        child_limits: list[tuple[int, ...]] = []
        for additionals in schedule.additional_memories:
            child_limits.append(
                tuple(m - d for m, d in zip(self._available, additionals))
            )
        assert len(child_limits) == len(schedule.children), (
            f"{len(child_limits)} child limits != {len(schedule.children)} "
            f"children for {type(schedule).__name__}"
        )
        if any(
            m < 0 for memory_after_action in child_limits for m in memory_after_action
        ):
            # This violates the limits, so we return None
            return None
        return [StandardMemoryLimits(a) for a in child_limits]


class PipelineChildMemoryLimits(MemoryLimits):
    """A MemoryLimits carrying extra information for a hole in a Pipeline."""

    base_available: tuple[int, ...]
    input_consumption: tuple[int, ...]
    output_consumption: tuple[int, ...]

    def __init__(
        self,
        base: tuple[int, ...],
        input_consumption: tuple[int, ...],
        output_consumption: tuple[int, ...],
    ) -> None:
        super().__init__()
        if any(v > b for v, b in zip(input_consumption, base)):
            raise IntermediatesTooBigError("input tensor doesn't fit in available")
        if any(v > b for v, b in zip(output_consumption, base)):
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

    @property
    def available(self):
        # This property is the interface for any caller expecting a base
        # StandardMemoryLimits (i.e. one that can't use extra information about
        # its context in a Pipeline).
        return tuple(
            a - (b + c)
            for a, b, c in zip(
                self.base_available, self.input_consumption, self.output_consumption
            )
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
