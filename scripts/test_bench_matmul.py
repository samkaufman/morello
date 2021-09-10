#!/usr/bin/env python3

import argparse
import contextlib
import functools
import logging
import multiprocessing
import random
import sqlite3
import sys
import time

import numpy as np

from morello import (
    cost,
    op_pprint,
    ops,
    search,
    search_cache,
    specs,
    system_config,
    tensor,
)
from morello.codegen import benchmark

RUNS = 5

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["best", "sample", "perturb", "numpy"])
parser.add_argument("--n_cpus", type=int, default=1)
parser.add_argument("--db_path", type=str, default="samples.db")

spec = specs.Matmul(
    specs.TensorSpec((128, 128)),
    specs.TensorSpec((128, 128)),
    specs.TensorSpec((128, 128)),
    serial_only=True,
)


def sample_completion(partial_impl: ops.Schedule) -> tuple[ops.Schedule, str]:
    if partial_impl.is_scheduled:
        return partial_impl, "random"

    # TODO: Remove the following filter once codegen is implemented for column-major.
    #   and sliding windows.
    actions = [
        a
        for a in partial_impl.actions()
        if (
            not isinstance(a, (ops.MoveAction, ops.PeelAction))
            or a.layout == specs.Layout.ROW_MAJOR
        )
        and not isinstance(a, ops.SlidingTileOutAction)
    ]
    assert actions, f"actions was empty"

    # TODO: Need to repeatedly draw.
    expanded = random.choice(actions)()
    return (
        expanded.replace_children(sample_completion(c)[0] for c in expanded.children),
        "random",
    )


def _sample_randint_on_boundary(upper, overweight_one=False) -> int:
    """Returns a random number in [1, upper) divisible by four, or 1."""
    if overweight_one:
        if random.random() < 0.5:
            return 1
    raw = random.randint(0, (upper - 1) // 4)
    if raw == 0:
        return 1
    return raw * 4


def sample_perturbed(hole: ops.Schedule) -> tuple[ops.Schedule, str]:
    global spec
    m, k, n = [
        _sample_randint_on_boundary(bound)
        for bound in spec.lhs.dim_sizes + (spec.rhs.dim_sizes[0],)
    ]
    impl = hole.tile_out((m, n))
    impl = impl.split(k)
    impl = impl.move_input(0, 0, specs.Layout.ROW_MAJOR)
    m_ = _sample_randint_on_boundary(impl.innermost.output.dim_sizes[0])
    n_ = _sample_randint_on_boundary(impl.innermost.output.dim_sizes[1])
    impl = impl.tile_out((m_, n_))
    impl = impl.move_input(1, 0, specs.Layout.ROW_MAJOR)
    impl = impl.move_output(0, specs.Layout.ROW_MAJOR)
    return impl.complete(), "perturbed3"


def sample_and_benchmark(
    sample_fn,
    hole: ops.Schedule,
    _: int,
):
    impl, procedure = sample_fn(hole)
    r = _benchmark(impl)
    return r, procedure


def benchmark_numpy_impl() -> float:
    global spec
    assert isinstance(spec, specs.Matmul)

    dtype = system_config.DEFAULT_SYSTEM_CONFIG.dtype

    # Make arbitrary args
    (m, n), k = spec.output.dim_sizes, spec.lhs.dim_sizes[1]
    lhs = np.arange(m * k, dtype=dtype.np_type).reshape((m, k))
    rhs = np.arange(k * n, dtype=dtype.np_type).reshape((k, n))
    lhs @ rhs

    start = time.time()
    for _ in range(10):
        lhs @ rhs
    end = time.time()
    return end - start


def _benchmark(impl):
    runtime_secs = min(
        benchmark.time_impl(impl, target_fn=benchmark.build_and_run_on_hexagon_sim)
        for _ in range(RUNS)
    )
    impl_str = op_pprint.pformat(impl)
    c = cost.analytical_cost(impl)
    peak = impl.peak_memory
    return runtime_secs, impl_str, c, peak


# TODO: Remove check_same_thread?
@contextlib.contextmanager
def _open_db(args, *, check_same_thread: bool):
    with sqlite3.connect(args.db_path, check_same_thread=check_same_thread) as db_conn:
        db_conn.execute(
            "CREATE TABLE IF NOT EXISTS samples (procedure text, date datetime, impl text, cost integer, peak_memory text, runtime_secs real)"
        )
        yield db_conn


if __name__ == "__main__":
    logging.basicConfig()

    args = parser.parse_args()

    if args.mode == "numpy":
        runtime_secs = benchmark_numpy_impl()
        with _open_db(args, check_same_thread=False) as db_conn:
            db_conn.execute(
                "INSERT INTO samples VALUES ('numpy', time('now'), NULL, NULL, NULL, ?)",
                (runtime_secs,),
            )
            db_conn.commit()
        sys.exit(0)

    operands = (
        tuple(tensor.Tensor(inp_spec, name=None) for inp_spec in spec.inputs),
        tensor.Tensor(spec.output, name=None),
    )
    hole = ops.spec_to_hole(spec, *operands)

    # Find the best schedule
    with _open_db(args, check_same_thread=False) as db_conn:
        if args.mode == "best":
            with search_cache.persistent_cache(
                "test_bench_matmul_cache.pkl", save=True
            ) as cache:
                impl = search.schedule_search(
                    spec,
                    *operands,
                    cache=cache,
                )
            assert impl is not None
            runtime_secs, impl_str, c, peak = _benchmark(impl)
            db_conn.execute(
                "INSERT INTO samples VALUES (?, time('now'), ?, ?, ?, ?)",
                ("best", impl_str, c, str(peak), runtime_secs),
            )
            db_conn.commit()

        elif args.mode in ("sample", "perturb"):
            sample_fn = sample_completion
            if args.mode == "perturb":
                sample_fn = sample_perturbed

            # TODO: Need to add repeated sampling of the same program
            with multiprocessing.Pool(processes=args.n_cpus) as pool:
                for result in pool.imap_unordered(
                    functools.partial(
                        sample_and_benchmark,
                        sample_fn,
                        hole,
                    ),
                    range(9999999999),
                ):
                    (runtime_secs, impl_str, c, peak), procedure = result
                    db_conn.execute(
                        "INSERT INTO samples VALUES (?, time('now'), ?, ?, ?, ?)",
                        (procedure, impl_str, c, str(peak), runtime_secs),
                    )
                    db_conn.commit()
        else:
            raise Exception("Unexpected mode: " + args.mode)
