import functools
import itertools
import operator
from typing import Callable, Iterable, Optional, Sequence

from .. import dtypes, system_config, utils
from . import settings


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
            if 2 ** power >= dim:
                break
            yield 2 ** power
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
    outer_shape: Sequence[int], dtype: dtypes.Dtype, elements: int = 128
) -> Iterable[tuple[int, ...]]:
    for result in _gen_vector_shapes_inner(outer_shape, dtype, elements):
        velems = functools.reduce(operator.mul, result, 1)
        assert velems % elements == 0
        yield result


def _gen_vector_shapes_inner(
    outer_shape: Sequence[int], dtype: dtypes.Dtype, elements: int
) -> Iterable[tuple[int, ...]]:
    if dtype.size != 1:
        return _gen_vector_shapes_inner(
            outer_shape, dtypes.Uint8, elements=elements // dtype.size
        )

    if len(outer_shape) == 0:
        raise ValueError("Given shape cannot be empty")
    elif len(outer_shape) == 1:
        if outer_shape[0] < elements:
            return
        yield (elements,)
    else:
        for factor in utils.factors(min(outer_shape[0], elements)):
            for tail in _gen_vector_shapes_inner(
                outer_shape[1:], dtype, elements // factor
            ):
                yield (factor,) + tail
