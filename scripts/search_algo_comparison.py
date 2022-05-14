#!/usr/bin/env python3
"""A script that comparing DP search to beam and random searches."""

import argparse
import copy
import dataclasses
import functools
import itertools
import logging
import multiprocessing
import os
import pathlib
import sys
import tempfile
import time
from typing import Callable, Iterable, Literal, Optional, Sequence, Union

import pandas as pd

import morello.impl.base
from morello import cost, dtypes, op_pprint, pruning, search, search_cache, specs
from morello.search import beam, random
from morello.system_config import current_target, set_current_target, target_by_name

logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--spec", type=str, action="append", default=None)
arg_parser.add_argument(
    "--processes", type=int, default=None, help="Number of processes to use."
)
arg_parser.add_argument("--no_random", action="store_true")
arg_parser.add_argument("--no_beam", action="store_true")
arg_parser.add_argument("--dp_only", action="store_true")
arg_parser.add_argument(
    "--heuristic", required=True, choices=["random", "noisy_optimal"]
)
arg_parser.add_argument("output", metavar="OUTPUT", type=pathlib.Path)

SKIP_DP_IF_BUDGET_KNOWN = False
DTYPE = dtypes.Uint32
BEAM_TRIALS = 100
BEAM_WIDTHS = [1, 10, 100]

target = target_by_name("cpu")


@dataclasses.dataclass(frozen=True)
class ExperimentResult:
    search_algo: Literal["dp", "beam", "random"]
    spec_name: str
    best_cost: Optional[int]  # May be None if search failed.
    expansions: int
    runtime: float
    execution_time: Optional[Union[int, float]]
    best_impl: Optional[morello.impl.base.Impl]
    best_description: Optional[str]  # pformatted. May be None if search failed.
    all_run_costs: Optional[Sequence[Union[int, float]]]


@dataclasses.dataclass(frozen=True)
class BeamExperimentResult(ExperimentResult):
    beam_width: int
    seq_num: int


def main():
    logging.basicConfig()

    args = arg_parser.parse_args()

    set_current_target(target)

    # Make a temporary directory for the cache.
    tmp_dir = tempfile.TemporaryDirectory()
    cache_root_dir = pathlib.Path(tmp_dir.name)
    assert cache_root_dir.is_dir()
    cache_path = functools.partial(make_cache_path, cache_root_dir)

    results_path: pathlib.Path = args.output
    if results_path.exists():
        print(f"Path {results_path} already exits; aborting", file=sys.stderr)
        return 100

    # Filter experiment specs according to args.
    specs_to_use = []
    for entry in experiment_specs():
        spec_name = entry[0]
        if args.spec is None or spec_name in args.spec:
            specs_to_use.append(entry)

    budgets: dict[str, int] = {}
    if SKIP_DP_IF_BUDGET_KNOWN:
        for spec_name, _, budget in specs_to_use:
            if budget:
                budgets[spec_name] = budget

    procs = args.processes
    if not procs:
        procs = multiprocessing.cpu_count()

    dp_task_partial = functools.partial(dp_task, cache_path=cache_path)
    all_results: list[ExperimentResult] = []
    with multiprocessing.Pool(processes=procs) as pool:
        # Do the dynamic programming runs first in parallel
        experiments_to_run = ((n, s) for n, s, _ in specs_to_use)
        if SKIP_DP_IF_BUDGET_KNOWN:
            experiments_to_run = ((n, s) for n, s, b in specs_to_use if not b)
        dp_results = list(pool.starmap(dp_task_partial, experiments_to_run))
        all_results += dp_results
        for r in dp_results:
            budgets[r.spec_name] = r.expansions

    # Run the other searches in serial. They are parallel internally.
    try:
        for spec_name, spec, _ in specs_to_use:
            budget = budgets[spec_name]
            if not args.dp_only:
                if not args.no_beam:
                    with search_cache.persistent_cache(
                        cache_path(spec_name), save=False
                    ) as cache:
                        for result in beam_task(
                            spec_name,
                            spec,
                            heuristic_name=args.heuristic,
                            cache=cache,
                            budget=budget,
                            parallel=procs,
                        ):
                            all_results.append(result)
                if not args.no_random:
                    for result in random_task(
                        spec_name, spec, budget=budget, parallel=procs
                    ):
                        all_results.append(result)

        # Update the cache with real execution times.
        for idx in range(len(all_results)):
            assert all_results[idx].execution_time is None
            if all_results[idx].best_impl:
                all_results[idx] = dataclasses.replace(
                    all_results[idx],
                    execution_time=current_target().time_impl(
                        all_results[idx].best_impl
                    ),
                )

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
        target.tensor_spec((1, 3, 224, 224), dtype=dtypes.Uint8),
        target.tensor_spec((128, 3, 5, 5), dtype=dtypes.Uint8),
    ]
    for _ in range(depth - 1):
        inputs.insert(0, target.tensor_spec((128, 128, 5, 5), dtype=dtypes.Uint8))
    inputs = tuple(inputs)

    output_dim = 512 - 4 * depth
    output = target.tensor_spec((128, output_dim, output_dim), dtype=dtypes.Uint8)

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
        target.tensor_spec((1, 3, 256, 256), dtype=DTYPE),
        target.tensor_spec((32, 3, 5, 5), dtype=DTYPE),
        target.tensor_spec((1, 32, 256 - 4, 256 - 4), dtype=DTYPE),
        serial_only=True,
    ), None
    yield "gemm3-4", specs.Compose(
        (specs.Matmul, specs.Matmul),
        (
            target.tensor_spec((4, 4), dtype=DTYPE),
            target.tensor_spec((4, 4), dtype=DTYPE),
            target.tensor_spec((4, 4), dtype=DTYPE),
        ),
        target.tensor_spec((4, 4), dtype=DTYPE),
        intermediate_dtypes=(DTYPE,),
        serial_only=True,
    ), None
    yield "gemm3-2048", specs.Compose(
        (specs.Matmul, specs.Matmul),
        (
            target.tensor_spec((2048, 2048), dtype=DTYPE),
            target.tensor_spec((2048, 2048), dtype=DTYPE),
            target.tensor_spec((2048, 2048), dtype=DTYPE),
        ),
        target.tensor_spec((2048, 2048), dtype=DTYPE),
        intermediate_dtypes=(DTYPE,),
        serial_only=True,
    ), None
    # yield "cnn-3layer", _make_cnn(depth=3), None
    # yield "cnn-6layer", _make_cnn(depth=6), None


def dp_task(
    spec_name: str, spec: specs.Spec, cache_path: Callable[[str], pathlib.Path]
) -> ExperimentResult:
    """Runs a DP search, then launches beam search with a corresponding budget."""
    print("")
    print(f"Running DP search for {spec_name}")
    sys.stdout.flush()

    start = time.time()

    cbs = ComposeCountingSearchCallbacks()

    cache_context = search_cache.persistent_cache(cache_path(spec_name))

    with cache_context as cache:
        search_result = search.dp.schedule_search(
            spec,
            tuple(target.tensor(s) for s in spec.inputs),
            target.tensor(spec.output),
            callbacks=cbs,
            cache=cache,
        )
    print(f"Explored {cbs.unseen_compose_visits} Compose sub-problems")
    sys.stdout.flush()

    best_cost = None
    best_pretty_formatted = None
    if search_result:
        best_cost = cost.compute_cost(search_result)
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
        cbs.unseen_compose_visits,
        runtime,
        None,
        search_result,
        best_pretty_formatted,
        all_run_costs=None,
    )


def beam_task(
    spec_name: str,
    spec: specs.Spec,
    budget: int,
    heuristic_name: str,
    cache: search_cache.ScheduleCache,
    parallel: Optional[int] = 1,
) -> Sequence[ExperimentResult]:
    print("")
    print(f"Running beam searches {spec_name} with budget {budget}")
    sys.stdout.flush()

    job_fn = functools.partial(
        _beam_task_job,
        spec_name,
        budget,
        spec,
        progress_bar=(parallel == 1),
        cache=cache,
        heuristic_name=heuristic_name,
    )
    job_seq = itertools.product(range(BEAM_TRIALS), BEAM_WIDTHS)

    if parallel == 1:
        beam_results = [job_fn(*a) for a in job_seq]
    else:
        mp_ctx = multiprocessing.get_context("fork")
        with mp_ctx.Pool(processes=parallel) as pool:
            beam_results = list(pool.starmap(job_fn, job_seq))
    return beam_results


def _beam_task_job(
    spec_name,
    budget,
    spec,
    seq_num,
    beam_width,
    progress_bar,
    heuristic_name: str,
    cache,
):
    if heuristic_name == "noisy_optimal":
        cache = copy.deepcopy(cache)
        cost_fn = functools.partial(beam.noisy_optimal_heuristic, cache=cache)
    elif heuristic_name == "random":
        cost_fn = beam.random_sampling_heuristic
    else:
        raise ValueError(f"Unexpected heuristic name: {heuristic_name}")

    start = time.time()

    remaining_budget = budget
    overall_best_impl = None
    overall_best_cost = None
    overall_best_impl_formatted = None
    all_beam_run_costs = []
    while remaining_budget > 0:
        stats = search.common.SearchStats()
        single_result = beam.beam_schedule_search(
            spec,
            tuple(target.tensor(s) for s in spec.inputs),
            target.tensor(spec.output),
            k=beam_width,
            budget=remaining_budget,
            stats=stats,
            cost_fn=cost_fn,
            progress_bar=progress_bar,
        )
        assert single_result is None or isinstance(single_result, tuple)
        remaining_budget -= stats.expansions
        assert remaining_budget >= 0

        if single_result:
            all_beam_run_costs += list(single_result.all_beam_costs)
            if (
                not overall_best_cost
                or single_result.best_impl_cost < overall_best_cost
            ):
                overall_best_impl = single_result.best_impl
                overall_best_cost = single_result.best_impl_cost
                overall_best_impl_formatted = op_pprint.pformat(single_result.best_impl)

    if not overall_best_impl:
        print("No schedule found by beam search trial")

    runtime = time.time() - start
    print(f"Beam search for {spec_name} w/ {beam_width} took {runtime:.2} seconds")
    sys.stdout.flush()

    return BeamExperimentResult(
        "beam",
        spec_name,
        overall_best_cost,
        budget - remaining_budget,
        runtime,
        None,
        overall_best_impl,
        overall_best_impl_formatted,
        all_run_costs=all_beam_run_costs,
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
    best_impl = None
    while budget:
        scheduled_impl, steps_taken = random.randomly_schedule_impl(hole, budget)
        assert steps_taken <= budget
        if scheduled_impl is not None:
            c = cost.compute_cost(scheduled_impl)
            run_costs.append(c)
            if best_cost is None or c < best_cost:
                best_impl = scheduled_impl
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
        execution_time=None,
        best_impl=best_impl,
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
        c = cost.compute_cost(scheduled_impl)
        try:
            results_queue.put((op_pprint.pformat(scheduled_impl), c, steps_taken))
        except ValueError:
            break


class ComposeCountingSearchCallbacks(search.SearchCallbacks):
    def __init__(self):
        self.unseen_compose_visits = 1

    def enter_unseen(self, spec: specs.Spec, _: pruning.MemoryLimits) -> None:
        if isinstance(spec, specs.Compose):
            self.unseen_compose_visits += 1


def make_cache_path(cache_root_dir: pathlib.Path, spec_name: str) -> pathlib.Path:
    return cache_root_dir / f"cache-{spec_name}.pkl"


if __name__ == "__main__":
    sys.exit(main())
