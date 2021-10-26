import dataclasses
import itertools
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Iterable, Literal, Union

import pandas as pd

import morello.search.beam
from morello import cost, specs, dtypes, op_pprint
from morello.system_config import target_by_name, set_current_target

DTYPE = dtypes.Uint32

target = target_by_name("cpu")
set_current_target(target)


results_queue = multiprocessing.Queue(maxsize=9000)


@dataclasses.dataclass(frozen=True)
class ExperimentResult:
    search_algo: Union[Literal["dp"], Literal["beam"]]
    spec_name: str
    best_cost: int
    expansions: int
    runtime: float
    best_description: str  # pformatted


@dataclasses.dataclass(frozen=True)
class BeamExperimentResult(ExperimentResult):
    beam_width: int
    seq_num: int


def _make_cnn(depth=16):
    pass


# Let's range over Specs.
def experiment_specs() -> Iterable[tuple[str, specs.Spec]]:
    yield "conv", specs.Convolution(
        target.tensor_spec((128, 128), dtype=DTYPE),
        target.tensor_spec((5, 5, 64), dtype=DTYPE),
        target.tensor_spec((124, 124, 64), dtype=DTYPE),
        serial_only=True,
    )
    yield "matmul", specs.Matmul(
        target.tensor_spec((128, 128), dtype=DTYPE),
        target.tensor_spec((128, 128), dtype=DTYPE),
        target.tensor_spec((128, 128), dtype=DTYPE),
        serial_only=True,
    )
    yield "gemm3", specs.Compose(
        (specs.Matmul, specs.Matmul),
        (
            target.tensor_spec((128, 128), dtype=DTYPE),
            target.tensor_spec((128, 128), dtype=DTYPE),
            target.tensor_spec((128, 128), dtype=DTYPE),
        ),
        target.tensor_spec((128, 128), dtype=DTYPE),
        intermediate_dtypes=(DTYPE,),
        serial_only=True,
    )


def job(spec_name: str, spec: specs.Spec):
    dp_result = dp_task(spec_name, spec)
    beam_results = beam_task(spec_name, spec, budget=dp_result.expansions)
    return [dp_result] + beam_results


def dp_task(spec_name: str, spec: specs.Spec) -> ExperimentResult:
    """Runs a DP search, then launches beam search with a corresponding budget."""
    print("")
    print("Running DP search")

    start = time.time()
    try:
        stats = morello.search.common.SearchStats()
        assert stats.expansions == 0
        search_result = morello.search.dp.schedule_search(
            spec,
            tuple(target.tensor(s) for s in spec.inputs),
            target.tensor(spec.output),
            stats=stats,
        )
        print(f"Explored {stats.expansions} schedules")

        best_cost = None
        best_pretty_formatted = None
        if search_result:
            best_cost = cost.analytical_cost(search_result)
            best_pretty_formatted = op_pprint.pformat(search_result)
        else:
            print("No schedule found by DP search")
    finally:
        runtime = time.time() - start
        print(f"Took {runtime} seconds")

    return ExperimentResult(
        "dp", spec_name, best_cost, stats.expansions, runtime, best_pretty_formatted
    )


def beam_task(spec_name: str, spec: specs.Spec, budget: int) -> list[ExperimentResult]:
    print("")
    print(f"Running beam search with budget {budget}")

    beam_results = []

    for seq_num, beam_width in itertools.product(range(10), [20, 200, 2000, 20000]):
        stats = morello.search.common.SearchStats()
        start = time.time()
        try:
            search_result = morello.search.beam.beam_schedule_search(
                spec,
                tuple(target.tensor(s) for s in spec.inputs),
                target.tensor(spec.output),
                k=beam_width,
                budget=budget,
                stats=stats,
            )
            best_cost = None
            best_pretty_formatted = None
            if search_result:
                best_cost = cost.analytical_cost(search_result)
                best_pretty_formatted = op_pprint.pformat(search_result)
                # op_pprint.pprint(search_result)
            else:
                print("No schedule found by beam search")
        finally:
            runtime = time.time() - start
            print(f"Took {runtime} seconds")

        beam_results.append(
            BeamExperimentResult(
                "beam",
                spec_name,
                best_cost,
                stats.expansions,
                runtime,
                best_pretty_formatted,
                beam_width=beam_width,
                seq_num=seq_num,
            )
        )

    return beam_results


def main():
    results_path = Path.cwd() / "dp_beam_results.csv"
    if results_path.exists():
        print(f"Path {results_path} already exits; aborting", file=sys.stderr)
        sys.exit(1)

    all_results = []
    pool = multiprocessing.Pool()
    for results in pool.starmap(job, experiment_specs()):
        all_results += results

    print(f"Saving to: {results_path}")
    pd.DataFrame.from_records(map(dataclasses.asdict, all_results)).to_csv(results_path)


if __name__ == "__main__":
    main()
