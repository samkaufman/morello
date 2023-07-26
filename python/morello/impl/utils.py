import functools
import itertools
import contextvars
import operator
from typing import Callable, Iterable, Optional, Sequence

from .. import dtypes, system_config, utils
from . import settings

limit_vectors_to_one_dim = contextvars.ContextVar(
    "limit_vectors_to_one_dim", default=True
)


def assert_stable_spec(func):
    """Assert that a method returns a Impl with the same spec as its first input."""

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        orig_spec = args[0].spec
        value = func(*args, **kwargs)
        assert (
            value.spec == orig_spec
        ), f"Spec {orig_spec} became {value.spec} while executing {func.__name__}"
        return value

    return wrapper_decorator


# TODO: Remove. This shouldn't be needed; just don't enumerate unsupported
#   actions. It was added as a hack on a deadline.
class ActionOutOfDomain(Exception):
    pass


def dim_range(dim: int, include_end: bool = True) -> Iterable[int]:
    """Returns possible dimension sizes up to `dim`.

    If `tile_size_mode` is set to `CACHE_LINE_MULTIPLES`, returned sizes
    will be evenly divisible by the system's cache line size.

    If `tile_size_mode` is set to `POWERS_OF_TWO`, returned sizes
    will be powers of two.

    :param include_end: If False, results will exclude `dim` itself.
    """
    assert dim >= 0
    if dim == 0:
        return
    line_size = system_config.current_system().line_size
    if settings.tile_size_mode.get() == settings.TileSizeMode.CACHE_LINE_MULTIPLES:
        it = range(line_size, dim, line_size)
        if dim > 1:
            it = itertools.chain([1], it)
        if include_end:
            it = itertools.chain(it, [dim])
        yield from it
    elif settings.tile_size_mode.get() == settings.TileSizeMode.POWERS_OF_TWO:
        assert dim >= 0
        if dim == 0:
            return
        power = 0
        while True:
            if 2**power >= dim:
                break
            yield 2**power
            power += 1
        if include_end:
            yield dim
    elif settings.tile_size_mode.get() == settings.TileSizeMode.ALL:
        end = dim
        if include_end:
            end += 1
        yield from range(1, end)
    else:
        raise NotImplementedError(
            f"Unsupported tile size mode: {settings.tile_size_mode.get()}"
        )


def gen_tile_sizes(
    tensor_shape: Sequence[int],
    filter: Optional[Callable[[tuple[int, ...]], bool]] = None,
    drop_given: bool = True,
) -> Iterable[tuple[int, ...]]:
    """Returns tile shapes to explore for a given tensor shape.

    Doesn't return tensor_shape itself.
    """
    if len(tensor_shape) == 0:
        return
    elif len(tensor_shape) == 1:
        for d in dim_range(tensor_shape[0]):
            new_shape = (d,)
            if drop_given and new_shape == tensor_shape:
                continue
            if filter and not filter(new_shape):
                continue
            yield new_shape
    else:
        for rest in gen_tile_sizes(tensor_shape[1:], drop_given=False):
            for d in dim_range(tensor_shape[0]):
                new_shape = (d,) + rest
                if drop_given and new_shape == tensor_shape:
                    continue
                if filter and not filter(new_shape):
                    continue
                yield new_shape


def gen_vector_shapes(
    outer_shape: Optional[Sequence[int]],
    dtype: dtypes.Dtype,
    vector_bytes: int,
    rank: Optional[int] = None,
) -> Iterable[tuple[int, ...]]:
    """Return possible vector shapes.

    :param outer_shape: The shape of the tensor to be vectorized. If `None`, all
      vector shapes for any tensor will be returned.
    """
    if bool(outer_shape) == bool(rank):
        raise ValueError("Must specify either outer_shape or rank, but not both")
    if outer_shape and any(d <= 0 for d in outer_shape):
        raise ValueError(f"Got outer_shape: {outer_shape}")
    if vector_bytes <= 0:
        raise ValueError("vector_bytes must be greater than 0")
    if vector_bytes % dtype.size != 0:
        raise ValueError(f"vector_bytes must be a multiple of dtype size: {dtype.size}")

    if not rank:
        assert outer_shape
        rank = len(outer_shape)

    adjusted_vector_bytes = vector_bytes
    if dtype.size != 1:
        adjusted_vector_bytes //= dtype.size

    if limit_vectors_to_one_dim.get():
        if adjusted_vector_bytes == 1:
            yield (1,) * rank
            return
        for i in reversed(range(rank)):
            if not outer_shape or outer_shape[i] >= adjusted_vector_bytes:
                v: tuple[int, ...] = (
                    ((1,) * i) + (adjusted_vector_bytes,) + ((1,) * (rank - i - 1))
                )
                assert len(v) == rank
                yield v
        return

    if not outer_shape:
        raise NotImplementedError(
            "outer_shape=None is not implemented yet for limit_vectors_to_one_dim=False"
        )

    for result in _gen_vector_shapes_inner(outer_shape, adjusted_vector_bytes):
        assert (
            functools.reduce(operator.mul, result, 1) * dtype.size == vector_bytes
        ), f"{result} was not {vector_bytes} vector bytes"
        yield result


def _gen_vector_shapes_inner(
    outer_shape: Sequence[int], total_elements: int
) -> Iterable[tuple[int, ...]]:
    # The following checks overlap a bit with the checks done by gen_vector_shapes,
    # but are asserts so that -O can elide them.
    assert outer_shape
    assert all(d > 0 for d in outer_shape)

    if len(outer_shape) == 0:
        raise ValueError("Given shape cannot be empty")
    if total_elements == 0:
        yield from []
        return

    if len(outer_shape) == 1:
        if outer_shape[0] >= total_elements:
            yield (total_elements,)
    else:
        for factor in utils.factors(total_elements):
            if factor > outer_shape[0]:
                break
            assert total_elements % factor == 0
            for tail in _gen_vector_shapes_inner(
                outer_shape[1:], total_elements // factor
            ):
                yield (factor,) + tail
