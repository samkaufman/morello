#!/usr/bin/env python3

import argparse
import contextlib
import functools
import logging
import multiprocessing
import pathlib
import random
import sqlite3
import time

import numpy as np

import morello.impl.actions
import morello.impl.base
from morello import cost, dtypes, op_pprint, search, search_cache, specs, system_config
from morello.codegen import gen

RUNS = 5
SAMPLE_CNT = 100
DTYPE = dtypes.Uint32

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default="cpu")
parser.add_argument("--cache", type=pathlib.Path, default=None)
parser.add_argument("--n_cpus", type=int, default=1)
parser.add_argument("--db_path", type=pathlib.Path, default="samples.db")
# parser.add_argument("mode", choices=["best", "sample", "perturb", "numpy"])
parser.add_argument("--best", action="store_true")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--perturb", action="store_true")
parser.add_argument("--numpy", action="store_true")

subparsers = parser.add_subparsers(dest="spec")

parser_matmul = subparsers.add_parser("matmul", help="Benchmark matrix multiplication")
parser_matmul.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="An mkn size to benchmark"
)

parser_gemm3 = subparsers.add_parser("gemm3", help="Benchmark GEMM3")
parser_gemm3.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="mkn sizes to benchmark"
)


def make_matmul_spec(d: int) -> specs.Matmul:
    target = system_config.current_target()
    return specs.Matmul(
        target.tensor_spec((d, d), dtype=DTYPE),
        target.tensor_spec((d, d), dtype=DTYPE),
        target.tensor_spec((d, d), dtype=DTYPE),
        serial_only=True,
    )


def make_gemm3_spec(d: int) -> specs.Spec:
    target = system_config.current_target()
    return specs.Compose(
        (specs.Matmul, specs.Matmul),
        inputs=(
            target.tensor_spec((d, d), dtype=DTYPE),
            target.tensor_spec((d, d), dtype=DTYPE),
            target.tensor_spec((d, d), dtype=DTYPE),
        ),
        output=target.tensor_spec((d, d), dtype=DTYPE),
        intermediate_dtypes=(DTYPE,),
        serial_only=True,
    )


def sample_completion(
    partial_impl: morello.impl.base.Impl,
) -> tuple[morello.impl.base.Impl, str]:
    if partial_impl.is_scheduled:
        return partial_impl, "random"

    # TODO: Remove the following filter once codegen is implemented for column-major.
    #   and sliding windows.
    actions = [
        a
        for a in partial_impl.actions()
        if (
            not isinstance(
                a, (morello.impl.actions.MoveAction, morello.impl.actions.PeelAction)
            )
            or a.layout == specs.Layout.ROW_MAJOR
        )
        and not isinstance(a, morello.impl.actions.SlidingTileOutAction)
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


def sample_perturbed(
    hole: morello.impl.base.Impl,
) -> tuple[morello.impl.base.Impl, str]:
    matmul_spec = hole.spec
    assert isinstance(matmul_spec, specs.Matmul)
    m, k, n = [
        _sample_randint_on_boundary(bound)
        for bound in matmul_spec.lhs.dim_sizes + (matmul_spec.rhs.dim_sizes[0],)
    ]
    impl = hole.tile_out((m, n))
    impl = impl.split(k)
    impl = impl.move_input(0, "RF", specs.Layout.ROW_MAJOR)
    m_ = _sample_randint_on_boundary(_get_innermost(impl).output.dim_sizes[0])
    n_ = _sample_randint_on_boundary(_get_innermost(impl).output.dim_sizes[1])
    impl = impl.tile_out((m_, n_))
    impl = impl.move_input(1, "RF", specs.Layout.ROW_MAJOR)
    impl = impl.move_output("RF", specs.Layout.ROW_MAJOR)
    return impl.complete(), "perturbed3"


def _get_innermost(impl: morello.impl.base.Impl) -> morello.impl.base.Impl:
    cur = impl
    while len(cur.children):
        assert hasattr(cur, "inner")
        cur = cur.inner
    return cur


def sample_and_benchmark(
    sample_fn,
    hole: morello.impl.base.Impl,
    _: int,
):
    impl, procedure = sample_fn(hole)
    r = _benchmark(impl)
    return r, procedure


def benchmark_numpy(spec: specs.Spec) -> float:
    if isinstance(spec, specs.Matmul):
        # Make arbitrary args
        (m, n), k = spec.output.dim_sizes, spec.lhs.dim_sizes[1]
        lhs = np.arange(m * k, dtype=DTYPE.np_type).reshape((m, k))
        rhs = np.arange(k * n, dtype=DTYPE.np_type).reshape((k, n))
        lhs @ rhs

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            lhs @ rhs
        end = time.time()
        return (end - start) / gen.BENCH_ITERS
    elif isinstance(spec, specs.Compose) and all(
        s == specs.Matmul for s in spec.subspec_classes
    ):
        m = spec.output.dim_sizes[0]
        assert all(
            d == m for operand in spec.operands for d in operand.dim_sizes
        ), f"All matmul operands' dimensions expected to have the size: {m}."
        a = np.arange(m * m, dtype=DTYPE.np_type).reshape((m, m))
        b = np.arange(m * m, dtype=DTYPE.np_type).reshape((m, m))
        c = np.arange(m * m, dtype=DTYPE.np_type).reshape((m, m))
        (a @ b) @ c

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            (a @ b) @ c
        end = time.time()
        return (end - start) / gen.BENCH_ITERS
    else:
        raise ValueError(f"Unrecognized spec: {spec}")


def _benchmark(impl):
    assert impl.is_scheduled
    runtime_secs = None
    for _ in range(RUNS):
        secs = system_config.current_target().time_impl(impl)
        logger.info(f"Sample runtime result {secs}s:")
        if runtime_secs is None or secs < runtime_secs:
            runtime_secs = secs
    impl_str = op_pprint.pformat(impl)
    c = cost.compute_cost(impl)
    peak = impl.peak_memory
    return runtime_secs, impl_str, c, peak


# TODO: Remove check_same_thread?
@contextlib.contextmanager
def _open_db(args, *, check_same_thread: bool):
    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(
        str(args.db_path), check_same_thread=check_same_thread
    ) as db_conn:
        db_conn.execute(
            "CREATE TABLE IF NOT EXISTS samples (procedure text, date datetime, "
            "impl text, cost integer, peak_memory text, runtime_secs real)"
        )
        yield db_conn


def _run_numpy(args, matmul_spec):
    runtime_secs = benchmark_numpy(matmul_spec)
    print(f"numpy took {runtime_secs:.7f}s")
    with _open_db(args, check_same_thread=False) as db_conn:
        db_conn.execute(
            "INSERT INTO samples VALUES ('numpy', time('now'), NULL, NULL, NULL, ?)",
            (runtime_secs,),
        )
        db_conn.commit()


def _run_best(args, spec):
    with search_cache.persistent_cache(args.cache, save=True) as cache:
        impl = search.schedule_search(spec, cache=cache)
    assert impl is not None
    runtime_secs, impl_str, c, peak = _benchmark(impl)
    print(f"best took {runtime_secs:.7f}s")
    with _open_db(args, check_same_thread=False) as db_conn:
        db_conn.execute(
            "INSERT INTO samples VALUES (?, time('now'), ?, ?, ?, ?)",
            ("best", impl_str, c, str(peak), runtime_secs),
        )
        db_conn.commit()


def _run_sample(sample_fn, args, matmul_spec):
    hole = morello.impl.base.spec_to_hole(matmul_spec)
    with _open_db(args, check_same_thread=False) as db_conn:
        with multiprocessing.Pool(processes=args.n_cpus) as pool:
            for result in pool.imap_unordered(
                functools.partial(
                    sample_and_benchmark,
                    sample_fn,
                    hole,
                ),
                range(SAMPLE_CNT),
            ):
                (runtime_secs, impl_str, c, peak), procedure = result
                print(f"(other) took {runtime_secs:.7f}s")
                db_conn.execute(
                    "INSERT INTO samples VALUES (?, time('now'), ?, ?, ?, ?)",
                    (procedure, impl_str, c, str(peak), runtime_secs),
                )
                db_conn.commit()


def main():
    logging.basicConfig()

    args = parser.parse_args()

    system_config.set_current_target(system_config.target_by_name(args.target))

    print(f"Spec: {args.spec}")
    for n in args.sizes:
        print(f"Size: {n}")
        if args.spec == "matmul":
            spec = make_matmul_spec(n)
        elif args.spec == "gemm3":
            spec = make_gemm3_spec(n)
        else:
            raise NotImplementedError(f"{args.spec} not implemented")

        if args.numpy:
            _run_numpy(args, spec)
        if args.best:
            _run_best(args, spec)
        if args.sample:
            _run_sample(sample_completion, args, spec)
        if args.perturb:
            _run_sample(sample_perturbed, args, spec)


if __name__ == "__main__":
    main()
