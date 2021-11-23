#!/usr/bin/env python3
"""A script that comparing DP search to beam and random searches."""

import argparse
import contextlib
import dataclasses
import functools
import itertools
import logging
import multiprocessing
import os
import pathlib
import sys
import time
from typing import Iterable, Literal, Optional, Union, Sequence

import pandas as pd

import morello.impl.base
from morello import cost, dtypes, op_pprint, search, search_cache, specs
from morello.search import beam, random
from morello.system_config import set_current_target, target_by_name

logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dp_only", action="store_true")
arg_parser.add_argument("--cachedir", metavar="CACHEDIR", type=pathlib.Path)
arg_parser.add_argument("output", metavar="OUTPUT", type=pathlib.Path)

SKIP_DP_IF_BUDGET_KNOWN = True
PROCS = 26  # None = use all CPUs.
DTYPE = dtypes.Uint32
BEAM_WIDTHS = [1, 10, 100]

target = target_by_name("cpu")
set_current_target(target)


results_queue = multiprocessing.Queue(maxsize=9000)


@dataclasses.dataclass(frozen=True)
class ExperimentResult:
    search_algo: Literal["dp", "beam", "random"]
    spec_name: str
    best_cost: Optional[int]  # May be None if search failed.
    expansions: int
    runtime: float
    best_description: Optional[str]  # pformatted. May be None if search failed.
    all_run_costs: Optional[Sequence[Union[int, float]]]


@dataclasses.dataclass(frozen=True)
class BeamExperimentResult(ExperimentResult):
    beam_width: int
    seq_num: int


def main():
    logging.basicConfig()

    args = arg_parser.parse_args()

    results_path: pathlib.Path = args.output
    if results_path.exists():
        print(f"Path {results_path} already exits; aborting", file=sys.stderr)
        return 100

    budgets: dict[str, int] = {}
    if SKIP_DP_IF_BUDGET_KNOWN:
        for spec_name, _, budget in experiment_specs():
            if budget:
                budgets[spec_name] = budget

    dp_task_partial = functools.partial(dp_task, cache_dir=args.cachedir)
    all_results: list[ExperimentResult] = []
    with multiprocessing.Pool() as pool:
        # Do the dynamic programming runs first in parallel
        experiments_to_run = ((n, s) for n, s, _ in experiment_specs())
        if SKIP_DP_IF_BUDGET_KNOWN:
            experiments_to_run = ((n, s) for n, s, b in experiment_specs() if not b)
        dp_results = list(pool.starmap(dp_task_partial, experiments_to_run))
        all_results += dp_results
        for r in dp_results:
            budgets[r.spec_name] = r.expansions

    # Run the other searches in serial. They are parallel internally.
    try:
        for spec_name, spec, _ in experiment_specs():
            budget = budgets[spec_name]
            if not args.dp_only:
                for result in beam_task(spec_name, spec, budget=budget, parallel=PROCS):
                    all_results.append(result)
                for result in random_task(
                    spec_name, spec, budget=budget, parallel=PROCS
                ):
                    all_results.append(result)
    except Exception:
        raise
    finally:
        print(f"Saving to: {results_path}")
        pd.DataFrame.from_records(map(dataclasses.asdict, all_results)).to_csv(
            results_path
        )
    return 0


def _make_cnn(depth: int = 6) -> specs.Spec:
    subspecs = (
        specs.ReduceSum,
        specs.Convolution,
    ) * depth

    # ResNet-18 uses 224-by-224 image inputs. Let's borrow that.
    inputs = [
        target.tensor_spec((224, 224), dtype=dtypes.Uint8),
        target.tensor_spec((5, 5, 64), dtype=dtypes.Uint8),
    ]
    for _ in range(depth - 1):
        inputs.insert(0, target.tensor_spec((5, 5, 128), dtype=dtypes.Uint8))
    inputs = tuple(inputs)

    output_dim = 512 - 4 * depth
    output = target.tensor_spec((output_dim, output_dim), dtype=dtypes.Uint8)

    intermediate_dtypes = (dtypes.Uint8,) * (len(subspecs) - 1)

    return specs.Compose(
        subspecs,
        inputs,
        output,
        intermediate_dtypes=intermediate_dtypes,
        serial_only=True,
    )


# Let's range over Specs.
def experiment_specs() -> Iterable[tuple[str, specs.Spec, Optional[int]]]:
    """Returns experiments to run.

    :returns: Tuples with: a short name for the experiment, the Spec to schedul, and,
        if available, a known number of expansions required by the DP search.
    """
    yield "matmul-512x512x512", specs.Matmul(
        target.tensor_spec((512, 512), dtype=DTYPE),
        target.tensor_spec((512, 512), dtype=DTYPE),
        target.tensor_spec((512, 512), dtype=DTYPE),
        serial_only=True,
    ), None
    yield "conv-256x256-5x5-32", specs.Convolution(
        target.tensor_spec((256, 256), dtype=DTYPE),
        target.tensor_spec((5, 5, 32), dtype=DTYPE),
        target.tensor_spec((256 - 4, 256 - 4, 32), dtype=DTYPE),
        serial_only=True,
    ), None
    yield "gemm3-256", specs.Compose(
        (specs.Matmul, specs.Matmul),
        (
            target.tensor_spec((256, 256), dtype=DTYPE),
            target.tensor_spec((256, 256), dtype=DTYPE),
            target.tensor_spec((256, 256), dtype=DTYPE),
        ),
        target.tensor_spec((256, 256), dtype=DTYPE),
        intermediate_dtypes=(DTYPE,),
        serial_only=True,
    ), None
    # yield "cnn-3layer", _make_cnn(depth=3), None
    # yield "cnn-6layer", _make_cnn(depth=6), None


def dp_task(
    spec_name: str, spec: specs.Spec, cache_dir: Optional[pathlib.Path]
) -> ExperimentResult:
    """Runs a DP search, then launches beam search with a corresponding budget."""
    print("")
    print(f"Running DP search for {spec_name}")
    sys.stdout.flush()

    start = time.time()
    stats = search.common.SearchStats()
    assert stats.expansions == 0

    cache_context = contextlib.nullcontext(None)
    if cache_dir:
        cache_path = cache_dir / f"cache-{spec_name}.pkl"
        cache_context = search_cache.persistent_cache(cache_path)
    with cache_context as cache:
        search_result = search.dp.schedule_search(
            spec,
            tuple(target.tensor(s) for s in spec.inputs),
            target.tensor(spec.output),
            stats=stats,
            cache=cache,
        )
    print(f"Explored {stats.expansions} schedules")
    sys.stdout.flush()

    best_cost = None
    best_pretty_formatted = None
    if search_result:
        best_cost = cost.analytical_cost(search_result)
        best_pretty_formatted = op_pprint.pformat(search_result)
    else:
        print("No schedule found by DP search")

    runtime = time.time() - start
    print(f"DP search for {spec_name} took {runtime:.2} seconds")
    sys.stdout.flush()

    return ExperimentResult(
        "dp",
        spec_name,
        best_cost,
        stats.expansions,
        runtime,
        best_pretty_formatted,
        all_run_costs=None,
    )


def beam_task(
    spec_name: str, spec: specs.Spec, budget: int, parallel: Optional[int] = 1
) -> Sequence[ExperimentResult]:
    print("")
    print(f"Running beam searches {spec_name} with budget {budget}")
    sys.stdout.flush()

    job_fn = functools.partial(
        _beam_task_job, spec_name, budget, spec, progress_bar=(parallel == 1)
    )
    job_seq = itertools.product(range(10), BEAM_WIDTHS)

    mp_ctx = multiprocessing.get_context("fork")
    if parallel == 1:
        beam_results = [job_fn(*a) for a in job_seq]
    else:
        with mp_ctx.Pool(processes=parallel) as pool:
            beam_results = list(pool.starmap(job_fn, job_seq))
    return beam_results


def _beam_task_job(spec_name, budget, spec, seq_num, beam_width, progress_bar):
    stats = search.common.SearchStats()
    start = time.time()

    beam_result = beam.beam_schedule_search(
        spec,
        tuple(target.tensor(s) for s in spec.inputs),
        target.tensor(spec.output),
        k=beam_width,
        budget=budget,
        stats=stats,
        cost_fn=beam.sampling_heuristic,
        progress_bar=progress_bar,
        return_run_costs=True,
    )

    best_cost = None
    best_pretty_formatted = None
    beam_run_costs = None
    if beam_result is not None:
        assert isinstance(beam_result, tuple)
        search_result, beam_run_costs = beam_result
        best_cost = cost.analytical_cost(search_result)
        best_pretty_formatted = op_pprint.pformat(search_result)
    else:
        print("No schedule found by beam search")

    runtime = time.time() - start
    print(f"Beam search for {spec_name} w/ {beam_width} took {runtime:.2} seconds")
    sys.stdout.flush()

    return BeamExperimentResult(
        "beam",
        spec_name,
        best_cost,
        stats.expansions,
        runtime,
        best_pretty_formatted,
        all_run_costs=beam_run_costs,
        beam_width=beam_width,
        seq_num=seq_num,
    )


def random_task(
    spec_name: str, spec: specs.Spec, budget: int, parallel: Optional[int] = 1
) -> list[ExperimentResult]:
    return [random_search(spec_name, spec, budget, parallel) for _ in range(10)]


def random_search(
    spec_name: str, spec: specs.Spec, budget: int, parallel: Optional[int] = 1
) -> ExperimentResult:
    if parallel is None:
        parallel = os.cpu_count() or 1

    start = time.time()
    original_budget = budget

    inputs = tuple(target.tensor(inp_spec, name=None) for inp_spec in spec.inputs)
    output = target.tensor(spec.output, name=None)
    hole = morello.impl.base.spec_to_hole(spec, inputs, output)

    run_costs: list[int] = []
    best_cost, best_pformatted = None, None

    # In the first phase, run in parallel, accumulating costs until we run out
    # of budget. Ignore that result and any subsequent results.
    if parallel != 1:
        m_ctx = multiprocessing.get_context("fork")
        should_stop = m_ctx.Event()
        results_queue = m_ctx.Queue(maxsize=10000)

        processes = [
            m_ctx.Process(
                target=_randomly_schedule_impls_job,
                args=(hole, results_queue, should_stop),
                daemon=True,
            )
            for _ in range(parallel)
        ]
        for p in processes:
            p.start()
        try:
            while True:
                pformatted, c, steps_taken = results_queue.get()
                if budget - steps_taken < 0:
                    # We're done. Finish up serially.
                    break
                budget -= steps_taken
                run_costs.append(c)
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_pformatted = pformatted
        finally:
            should_stop.set()
            results_queue.close()
            for p in processes:
                p.terminate()

    # Second phase: run serially, passing the budget in.
    while budget:
        scheduled_impl, steps_taken = random.randomly_schedule_impl(hole, budget)
        assert steps_taken <= budget
        if scheduled_impl is not None:
            c = cost.analytical_cost(scheduled_impl)
            run_costs.append(c)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_pformatted = op_pprint.pformat(scheduled_impl)
        else:
            assert steps_taken == budget
        budget -= steps_taken

    runtime = time.time() - start
    print(f"Random search took {runtime:.2}s")
    sys.stdout.flush()
    return ExperimentResult(
        search_algo="random",
        spec_name=spec_name,
        best_cost=best_cost,
        expansions=original_budget - budget,
        # Not recording runtime or descriptions. Implement if needed.
        runtime=runtime,
        best_description=best_pformatted,
        all_run_costs=run_costs,
    )


def _randomly_schedule_impls_job(
    root_impl: morello.impl.base.Impl,
    results_queue: multiprocessing.Queue,
    should_stop,
):
    while not should_stop.is_set():
        scheduled_impl, steps_taken = random.randomly_schedule_impl(root_impl, None)
        assert scheduled_impl and steps_taken
        c = cost.analytical_cost(scheduled_impl)
        try:
            results_queue.put((op_pprint.pformat(scheduled_impl), c, steps_taken))
        except ValueError:
            break


if __name__ == "__main__":
    sys.exit(main())
