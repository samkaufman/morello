import contextlib
import functools
import itertools
import logging
import multiprocessing
from collections.abc import Sequence
from typing import Any, Generator, Iterable, Optional

from .. import impl, layouts, pruning, specs
from ..impl import Impl
from . import common

logger = logging.getLogger(__name__)


def _best_schedule(it: Iterable[Impl]):
    """Returns the best schedule if `it` is non-empty; `None` if it is.

    This uses schedule_key, so it will return the lowest cost schedule,
    breaking ties as described in schedule_key's docstring.
    """
    new_it = ((s, common.schedule_key(s)) for s in it)
    return min(new_it, key=lambda x: x[1], default=None)


# TODO: Don't just use a global variable. Use a thread local or contextvars.
_specs_on_stack = []


def _assert_no_cycles_in_stack(func):
    """Decorates schedule_search to ensure no spec ever occurs twice on the stack."""
    global _specs_on_stack

    if not __debug__:
        return func

    # noinspection PyUnreachableCode
    @functools.wraps(func)
    def assert_no_cycles_in_stack_inner(*args, **kwargs):
        if isinstance(args[0], impl.Impl):
            spec = args[0].spec
        else:
            spec = args[0]
        assert isinstance(spec, specs.Spec)
        if spec in _specs_on_stack:
            raise Exception(
                f"Spec {spec} was in the stack:"
                + "".join(f"\n - {s}" for s in _specs_on_stack)
            )
        _specs_on_stack.append(spec)
        try:
            return func(*args, **kwargs)
        finally:
            popped = _specs_on_stack.pop()
            assert popped == spec, "spec on _specs_on_stack doesn't match real stack"

    return assert_no_cycles_in_stack_inner


def naive_search(
    spec: specs.Spec,
    memory_limits=None,
    parent_summary=None,
    parallel_jobs=1,
):
    """Returns the best Impl for a given Spec and memory limits.

    Search explicitly enumerates every possible Impl of the given Spec,
    excluding those that violate memory limits.

    Unlike dynamic programming-based search it does not use or update a cache.

    :param parallel_jobs: The maximum number of parallel processes to launch to
      enumerate subschedules available at the top level.
    """
    # The default available memory is the capacities of current_system().
    # These capacities are measured in cache lines, not words, so: multiply by
    # line size when initializing the limit.
    if memory_limits is None:
        memory_limits = pruning.StandardMemoryLimits()

    # Create a an Impl hole corresponding to the query spec
    leaf = impl.spec_to_hole(spec)
    assert leaf.depth == 1, f"Expected hole to have depth 1; had {leaf.depth}"

    best_result = _best_schedule(
        enumerate_impls(
            leaf, memory_limits, parent_summary, parallel_jobs=parallel_jobs
        )
    )

    if best_result is not None:
        schedule_to_return, _ = best_result
        assert (
            schedule_to_return.spec == spec
        ), f"{str(schedule_to_return.spec)} != {str(spec)}"
        return best_result[0]
    else:
        return None


@_assert_no_cycles_in_stack
def enumerate_impls(
    leaf: impl.Impl, memory_limits, parent_summary, parallel_jobs=1
) -> Iterable[Impl]:
    """Yields all completions of a given Impl.

    If given, only yield Impls which satisfy memory limits.
    """
    # If the leaf is itself scheduled, yield it (i.e. no action) as an option.
    leaf_meets_limits = True
    if memory_limits:
        leaf_meets_limits = all(m >= 0 for m in memory_limits.available.values())
    if leaf_meets_limits and leaf.is_scheduled:
        yield leaf

    parallel_ctx = contextlib.nullcontext()
    if parallel_jobs is None or parallel_jobs > 1:
        live_jobs = 0
        job_async_results = []
        mp_ctx = multiprocessing.get_context("fork")
        m = mp_ctx.Manager()
        results_queue = m.Queue()
        parallel_ctx = mp_ctx.Pool(processes=parallel_jobs)

    with parallel_ctx as pool:
        for act in leaf.actions(parent_summary=parent_summary):
            new_tree = act()
            new_parent_summary = impl.ParentSummary.update(parent_summary, new_tree)

            new_child_memory_limits = [None for _ in new_tree.children]
            if memory_limits:
                new_child_memory_limits = memory_limits.transition(new_tree)
                if new_child_memory_limits is None:
                    continue

            if pool:
                live_jobs += 1
                job_async_results.append(
                    pool.apply_async(
                        _impls_from_new_tree_into_queue,
                        args=(
                            results_queue,
                            new_tree,
                            new_parent_summary,
                            new_child_memory_limits,
                            leaf.spec,
                        ),
                    )
                )
            else:
                yield from _impls_from_new_tree(
                    new_tree, new_parent_summary, new_child_memory_limits, leaf.spec
                )
        if pool:
            logger.info("Launched %s jobs", live_jobs)
            for j in job_async_results:
                j.get()
            pool.close()
            pool.join()
            while live_jobs:
                o = results_queue.get()
                if o is None:
                    live_jobs -= 1
                else:
                    yield o


def _impls_from_new_tree(
    new_tree, new_parent_summary, new_child_memory_limits, expected_spec
) -> Generator[Impl, None, None]:
    for child_options in itertools.product(
        *[
            enumerate_impls(child, mem, new_parent_summary)
            for child, mem in zip(new_tree.children, new_child_memory_limits)
        ]
    ):
        completed = new_tree.replace_children(child_options)
        assert completed.spec == expected_spec, f"{completed.spec} != {expected_spec}"
        yield completed


def _impls_from_new_tree_into_queue(results_queue, *args, **kwargs) -> None:
    for o in _impls_from_new_tree(*args, **kwargs):
        assert o is not None
        results_queue.put(o)
    results_queue.put(None)
