#!/usr/bin/env python3

import argparse
import contextlib
import datetime
import functools
import itertools
import logging
import multiprocessing
import os
import pathlib
import random
import re
import sqlite3
import tempfile
import time

import gspread
import halide as hl
import jax
import jax.lib
import jax.numpy as jnp
import numpy as np
import termcolor
import torch
import torch.nn.functional as F
import tvm
import tvm.contrib.graph_executor
from jax import lax
from jax.tools import jax_to_ir
from torch import profiler
from tvm import relay

import morello.impl.actions
import morello.impl.base
from morello import cost, dtypes, op_pprint, search, search_cache, specs, system_config
from morello.benchmarks.toy_cnn import halide as toyhl
from morello.codegen import gen

RUNS = 5
SAMPLE_CNT = 100
DTYPE = dtypes.Uint32
TORCH_DTYPE_NP = np.int32  # Signed version of DTYPE
TORCH_DTYPE = torch.int32

RELAY_VERSION_RE = re.compile(r'^\s*#\[version = "[\d\.]+"\]\s*$')

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default="cpu")
parser.add_argument("--cache", type=pathlib.Path, default=None)
parser.add_argument("--no-save-cache", action="store_false", dest="save_cache")
parser.add_argument("--n_cpus", type=int, default=1)
parser.add_argument("--db_path", type=pathlib.Path, default="samples.db")
parser.add_argument("--best", action="store_true")
parser.add_argument("--baselines", action="store_true")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--perturb", action="store_true")
parser.add_argument("--print_graphs", action="store_true")
parser.add_argument("--log_to_sheet", type=str, default=None)
parser.add_argument("--gsheet_key", type=pathlib.Path, default=None)
parser.add_argument("--hostname", type=str, default=None)
parser.add_argument("-b", "--batch", type=int, action="append")

subparsers = parser.add_subparsers(dest="spec")

parser_matmul = subparsers.add_parser("matmul", help="Benchmark matrix multiplication")
parser_matmul.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="An mkn size to benchmark"
)

parser_gemm3 = subparsers.add_parser("gemm3", help="Benchmark GEMM3")
parser_gemm3.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="mkn sizes to benchmark"
)

parser_conv = subparsers.add_parser("conv", help="Benchmark convolution")
parser_conv.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="image sizes to benchmark"
)

parser_cnn = subparsers.add_parser("cnn", help="Benchmark small CNN")
parser_cnn.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="image sizes to benchmark"
)


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    # Some dtypes aren't supported by PyTorch, but it's fine to convert them to the
    # closest supported type. This shouldn't affect benchmark results meaningfully.
    if arr.dtype == np.uint32:
        arr = arr.astype("int32")
    return torch.from_numpy(arr)


def make_matmul_spec(batch_size: int, d: int) -> specs.Matmul:
    if batch_size != 1:
        raise NotImplementedError("Batched matrix multiplication not yet supported")
    target = system_config.current_target()
    return specs.Matmul(
        target.tensor_spec((d, d), dtype=DTYPE),
        target.tensor_spec((d, d), dtype=DTYPE),
        target.tensor_spec((d, d), dtype=DTYPE),
        serial_only=False,
    )


def make_gemm3_spec(batch_size: int, d: int) -> specs.Spec:
    if batch_size != 1:
        raise NotImplementedError("Batched matrix multiplication not yet supported")
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
        serial_only=False,
    )


def make_conv_spec(batch_size: int, d: int) -> specs.Convolution:
    target = system_config.current_target()

    fh, fw, fc = 5, 5, 32
    out_h, out_w = 1 + d - fh, 1 + d - fw

    return specs.Convolution(
        target.tensor_spec((batch_size, 1, d, d), dtype=DTYPE),
        target.tensor_spec((fc, 1, fh, fw), dtype=DTYPE),
        output=target.tensor_spec((batch_size, fc, out_h, out_w), dtype=DTYPE),
        serial_only=False,
    )


def make_cnn_spec(batch_size: int, d: int) -> specs.Spec:
    target = system_config.current_target()

    img = target.tensor_spec((batch_size, 3, d, d), dtype=DTYPE)
    filters_a = target.tensor_spec((32, 3, 3, 3), dtype=DTYPE)
    filters_b = target.tensor_spec((32, 32, 3, 3), dtype=DTYPE)
    output = target.tensor_spec((batch_size, 32, d - 4, d - 4), dtype=DTYPE)
    return specs.Compose(
        (specs.Convolution, specs.Convolution),
        (filters_b, img, filters_a),
        output,
        intermediate_dtypes=(DTYPE,),
        serial_only=False,
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


def benchmark_baseline(
    spec: specs.Spec, print_graphs: bool = False
) -> dict[str, float]:
    result = {}

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
        result["numpy"] = (end - start) / gen.BENCH_ITERS

        lhs_t, rhs_t = _to_torch(lhs), _to_torch(rhs)
        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            torch.matmul(lhs_t, rhs_t)
        end = time.time()
        result["pytorch"] = (end - start) / gen.BENCH_ITERS
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
        result["numpy"] = (end - start) / gen.BENCH_ITERS

        a_t, b_t, c_t = [_to_torch(x) for x in (a, b, c)]
        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            torch.matmul(torch.matmul(a_t, b_t), c_t)
        end = time.time()
        result["pytorch"] = (end - start) / gen.BENCH_ITERS

        @torch.jit.script
        def jit_gemm3(a, b, c):
            return torch.matmul(torch.matmul(a, b), c)

        torch.jit.wait(torch.jit.fork(jit_gemm3, a_t, b_t, c_t))

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            torch.jit.wait(torch.jit.fork(jit_gemm3, a_t, b_t, c_t))
        end = time.time()
        result["torchscript"] = (end - start) / gen.BENCH_ITERS
    elif isinstance(spec, specs.Convolution):
        batch, channels, h, w = spec.lhs.dim_sizes
        filters_count, _, fh, fw = spec.rhs.dim_sizes

        # Use signed int32 with PyTorch.
        assert DTYPE == dtypes.Uint32

        img = np.arange(batch * channels * h * w, dtype=TORCH_DTYPE_NP).reshape(
            (batch, channels, h, w)
        )
        filters = np.arange(
            filters_count * channels * fh * fw, dtype=TORCH_DTYPE_NP
        ).reshape((filters_count, channels, fh, fw))
        img_t, filters_t = torch.tensor(img), torch.tensor(filters)

        # Which backend is PyTorch using?
        # backend = torch._C._select_conv_backend(
        #     img, filters, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1
        # )
        # print("Conv PyTorch backend:", backend)

        # PyTorch execution on the CPU is synchronous. Useful for this benchmark
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU]) as prof:
            with profiler.record_function("conv2d"):
                F.conv2d(img_t, filters_t)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            F.conv2d(img_t, filters_t)
        end = time.time()
        result["pytorch"] = (end - start) / gen.BENCH_ITERS

        @torch.jit.script
        def jit_conv(i, f):
            return F.conv2d(i, f)

        torch.jit.wait(torch.jit.fork(jit_conv, img_t, filters_t))

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            torch.jit.wait(torch.jit.fork(jit_conv, img_t, filters_t))
        end = time.time()
        result["torchscript"] = (end - start) / gen.BENCH_ITERS

        if print_graphs:
            print("")
            termcolor.cprint("Torch:", attrs=["bold"])
            print(torch.jit.last_executed_optimized_graph())

        def jax_conv(i, f):
            return lax.conv(i, f, (1, 1), "VALID")

        jax_conv_fast = jax.jit(jax_conv)
        jax_conv_fast(img, filters)

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            jax_conv_fast(img, filters)
        end = time.time()
        result["jax"] = (end - start) / gen.BENCH_ITERS

        if print_graphs:
            print(
                jax_to_ir.jax_to_ir(
                    jax_conv_fast,
                    input_shapes=[
                        ("i", jax.ShapedArray(img.shape, img.dtype)),
                        ("f", jax.ShapedArray(filters.shape, filters.dtype)),
                    ],
                    format="HLO",
                )[1]
            )
    elif isinstance(spec, specs.Compose) and spec.subspec_classes == (
        specs.Convolution,
        specs.Convolution,
    ):
        batch_size, channels, h, w = spec.inputs[-2].dim_sizes
        fc_a, _, fh_a, fw_a = spec.inputs[-1].dim_sizes
        fc_b, _, fh_b, fw_b = spec.inputs[-3].dim_sizes

        img = np.arange(batch_size * channels * h * w, dtype=TORCH_DTYPE_NP).reshape(
            (batch_size, channels, h, w)
        )
        filters_a = np.arange(
            fh_a * fw_a * fc_a * channels, dtype=TORCH_DTYPE_NP
        ).reshape((fc_a, channels, fh_a, fw_a))
        filters_b = np.arange(fh_b * fw_b * fc_b * fc_a, dtype=TORCH_DTYPE_NP).reshape(
            (fc_b, fc_a, fh_b, fw_b)
        )
        img_t, filters_a_t, filters_b_t = [
            torch.tensor(x) for x in (img, filters_a, filters_b)
        ]

        F.conv2d(F.conv2d(img_t, filters_a_t), filters_b_t)

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            F.conv2d(F.conv2d(img_t, filters_a_t), filters_b_t)
        end = time.time()
        result["pytorch"] = (end - start) / gen.BENCH_ITERS

        assert (
            torch.int32 == TORCH_DTYPE
        ), "Expected torch.int32, which is inlined into jit_conv"

        def jit_conv(i, fa, fb):
            return F.conv2d(F.conv2d(i, fa), fb)

        jit_conv = torch.jit.trace(jit_conv, (img_t, filters_a_t, filters_b_t))

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            torch.jit.wait(torch.jit.fork(jit_conv, img_t, filters_a_t, filters_b_t))
        end = time.time()
        result["torchscript"] = (end - start) / gen.BENCH_ITERS

        if print_graphs:
            print("")
            termcolor.cprint("Torch:", attrs=["bold"])
            print(torch.jit.last_executed_optimized_graph())

        # PyTorch graph on Relay/TVM
        shape_list = [
            ("input", img_t.shape),
            ("filters_a", filters_a_t.shape),
            ("filters_b", filters_b_t.shape),
        ]
        tvm_target = tvm.target.Target("llvm")
        relay_mod, relay_params = relay.frontend.from_pytorch(jit_conv, shape_list)

        tvm_mod_text = None

        class PIR(tvm.instrument.PassInstrument):
            def run_after_pass(self, mod, info):
                nonlocal tvm_mod_text
                # Save the Relay module if the result of this pass has a useful
                # text representation---in particular, it's not *just* something
                # like '#[version = "0.0.5"]'---and seems to have not yet
                # stripped the definition of the '@main' function.
                new_text = mod.astext(show_meta_data=True)
                if not RELAY_VERSION_RE.match(new_text) and "@main" in new_text:
                    tvm_mod_text = new_text

        instruments = []
        if print_graphs:
            instruments.append(PIR())

        with tvm.transform.PassContext(opt_level=3, instruments=instruments):
            tvm_lib = relay.build(relay_mod, target=tvm_target, params=relay_params)

        tvm_m = tvm.contrib.graph_executor.GraphModule(tvm_lib["default"](tvm.cpu(0)))
        tvm_m.set_input("input", img_t)
        tvm_m.set_input("filters_a", filters_a_t)
        tvm_m.set_input("filters_b", filters_b_t)

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            tvm_m.run()
        end = time.time()
        result["relay"] = (end - start) / gen.BENCH_ITERS

        if print_graphs:
            print("")
            termcolor.cprint("Relay:", attrs=["bold"])
            print(tvm_mod_text)

        def jax_cnn(i, fa, fb):
            a = lax.conv(i, fa, (1, 1), "VALID")
            b = lax.conv(a, fb, (1, 1), "VALID")
            return b

        jax_cnn_fast = jax.jit(jax_cnn)
        jax_cnn_fast(img, filters_a, filters_b).block_until_ready()

        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            jax_cnn_fast(img, filters_a, filters_b).block_until_ready()
        end = time.time()
        result["jax"] = (end - start) / gen.BENCH_ITERS

        # Print the *optimized* HLO. (This API is not stable.)
        if print_graphs:
            print("")
            termcolor.cprint("JAX:", attrs=["bold"])
            c = jax.xla_computation(jax_cnn_fast)(img, filters_a, filters_b)
            print(
                jax.lib.xla_bridge.get_backend().compile(c).hlo_modules()[0].to_string()
            )

        # Benchmark a Halide equivalent.
        img_hl, filters_a_hl, filters_b_hl = (
            hl.Buffer(img, name="img"),
            hl.Buffer(filters_a, name="filters_a"),
            hl.Buffer(filters_b, name="filters_b"),
        )
        fn = toyhl.halide_small_cnn(img_hl, filters_a_hl, filters_b_hl)
        fn = hl.Pipeline(fn)
        fn.auto_schedule("Adams2019", hl.get_jit_target_from_environment())
        fn.compile_jit()
        if print_graphs:
            print("")
            print("Halide: Basic Loop Nest")
            fn.print_loop_nest()
            print("")
            print("Halide: Detailed")
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as t:
                fn.compile_to_lowered_stmt(t.name, [], format=hl.StmtOutputFormat.Text)
            with open(t.name, "r") as fo:
                print(t.name)
                print(fo.read())
        print("")
        print(f"output dims: {spec.output.dim_sizes}")
        start = time.time()
        for _ in range(gen.BENCH_ITERS):
            fn.realize(spec.output.dim_sizes)
        end = time.time()
        result["halide (Adams2019)"] = (end - start) / gen.BENCH_ITERS
    else:
        raise ValueError(f"Unrecognized spec: {spec}")

    return result


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


def _run_baselines(args, spec: specs.Spec, print_graphs: bool = False) -> dict:
    results = {}
    for baseline_name, runtime_secs in benchmark_baseline(
        spec, print_graphs=print_graphs
    ).items():
        print(f"{baseline_name} baseline took {runtime_secs:.7f}s")
        assert baseline_name not in results
        results[baseline_name] = runtime_secs
        with _open_db(args, check_same_thread=False) as db_conn:
            db_conn.execute(
                "INSERT INTO samples VALUES (?, time('now'), NULL, NULL, NULL, ?)",
                (baseline_name, runtime_secs),
            )
            db_conn.commit()
    return results


def _run_best(args, spec) -> dict:
    with search_cache.persistent_cache(args.cache, save=args.save_cache) as cache:
        impl = search.schedule_search(spec, cache=cache)
    assert impl is not None
    runtime_secs, impl_str, c, peak = _benchmark(impl)
    print(f"Best took {runtime_secs:.7f}s")
    print("Impl:\n" + impl_str)
    with _open_db(args, check_same_thread=False) as db_conn:
        db_conn.execute(
            "INSERT INTO samples VALUES (?, time('now'), ?, ?, ?, ?)",
            ("best", impl_str, c, str(peak), runtime_secs),
        )
        db_conn.commit()

    return {"morello": runtime_secs}


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
                print(f"{sample_fn.__name__} took {runtime_secs:.7f}s")
                db_conn.execute(
                    "INSERT INTO samples VALUES (?, time('now'), ?, ?, ?, ?)",
                    (procedure, impl_str, c, str(peak), runtime_secs),
                )
                db_conn.commit()


def main():
    logging.basicConfig()

    args = parser.parse_args()

    batch_sizes = args.batch
    if not batch_sizes:
        batch_sizes = [1]

    if args.log_to_sheet:
        gc_kwargs = {}
        if args.gsheet_key:
            assert isinstance(args.gsheet_key, pathlib.Path)
            gc_kwargs["filename"] = args.gsheet_key
        gc = gspread.service_account(**gc_kwargs)
        sheet = gc.open(args.log_to_sheet).worksheet("Log")

    system_config.set_current_target(args.target)

    # Disable sliding windows, since we can't use them with codegen yet.
    morello.impl.allow_sliding_windows.set(False)

    termcolor.cprint("PyTorch Configuration:", attrs=["bold"])
    print(torch.__config__.show())

    start_time = str(datetime.datetime.now())
    if args.hostname:
        hostname = args.hostname
    else:
        hostname = os.uname().nodename

    print("")
    print(f"Spec: {args.spec}")
    for b, n in itertools.product(batch_sizes, args.sizes):
        print("")
        print(f"Size: {n} (Batch Size: {b})")
        if args.spec == "matmul":
            spec = make_matmul_spec(b, n)
        elif args.spec == "gemm3":
            spec = make_gemm3_spec(b, n)
        elif args.spec == "conv":
            spec = make_conv_spec(b, n)
        elif args.spec == "cnn":
            spec = make_cnn_spec(b, n)
        else:
            raise NotImplementedError(f"{args.spec} not implemented")

        results = {}
        if args.baselines:
            results.update(_run_baselines(args, spec, print_graphs=args.print_graphs))
        if args.best:
            results.update(_run_best(args, spec))
        if args.sample:
            _run_sample(sample_completion, args, spec)
        if args.perturb:
            _run_sample(sample_perturbed, args, spec)

        if args.log_to_sheet:
            sheet.append_rows(
                [
                    [start_time, hostname, args.spec, n, b, name, secs]
                    for name, secs in results.items()
                ],
                value_input_option="USER_ENTERED",
            )


if __name__ == "__main__":
    main()
