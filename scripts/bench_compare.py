#!/usr/bin/env python3

import argparse
import asyncio
import contextlib
import contextvars
import dataclasses
import datetime
import itertools
import logging
import math
import mimetypes
import multiprocessing
import os
import pathlib
import queue
import random
import re
import runpy
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import warnings
from glob import glob
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import gspread
import halide as hl
import jax
import jax.lib
import numpy as np
import oauth2client.service_account
import psutil
import pydrive2
import pydrive2.auth
import pydrive2.drive
import termcolor
import torch
import torch.nn.functional as F
import tvm
import tvm.auto_scheduler
import tvm.contrib.graph_executor
import tvm.te
from google.oauth2 import service_account
from jax import lax
from jax.tools import jax_to_ir
from tvm import auto_scheduler, relay

import morello.impl.actions
import morello.impl.base
from morello import (
    codegen,
    cost,
    dtypes,
    layouts,
    op_pprint,
    search,
    search_cache,
    specs,
    system_config,
)
from morello.benchmarks.toy_cnn import halide as toyhl

DTYPE = dtypes.Uint32
TORCH_DTYPE_NP = np.int32  # Signed version of DTYPE
TORCH_DTYPE = torch.int32
PERF_TERMINATE_TIMEOUT = 60.0  # 1 min.
MIN_TRIAL_TIME_SECS = 2.5
MIN_SAMPLES = 3

RELAY_VERSION_RE = re.compile(r'^\s*#\[version = "[\d\.]+"\]\s*$')

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--ignore-environment", action="store_true")
parser.add_argument("--target", type=str, default="x86")
parser.add_argument("--cache", type=pathlib.Path, default=None)
parser.add_argument("--redis", type=str, default=None)
parser.add_argument("--redis-namespace", type=str, default=None)
parser.add_argument("--load-sqlite-db", type=pathlib.Path, default=None)
parser.add_argument("--sqlite-inmem-cache-kb", type=int, default=None)
parser.add_argument(
    "--trials",
    type=int,
    default=10,
    help=(
        "The number of times to measure each benchmark. (Each measurement is "
        "a loop of individual executions which will be averaged.)"
    ),
)
parser.add_argument("--no-save-cache", action="store_false", dest="save_cache")
parser.add_argument("--backend", type=str, action="append")
parser.add_argument("--log-to-sheet", type=str, default=None)
parser.add_argument("--save-to-gdrive", type=str, default=None)
parser.add_argument(
    "--gsheet-key",
    type=pathlib.Path,
    default=None,
    help="A path to a file containing a Google Sheets key",
)
parser.add_argument("--hostname", type=str, default=None)
parser.add_argument(
    "--configure-governor", nargs="?", type=pathlib.Path, const=True, default=None
)
parser.add_argument("-b", "--batch", type=int, action="append")

subparsers = parser.add_subparsers(dest="spec")

parser_matmul = subparsers.add_parser("matmul", help="Benchmark matrix multiplication")
parser_matmul.add_argument("--serial", action="store_true")
parser_matmul.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="An mkn size to benchmark"
)

parser_gemm3 = subparsers.add_parser("gemm3", help="Benchmark GEMM3")
parser_gemm3.add_argument("--serial", action="store_true")
parser_gemm3.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="mkn sizes to benchmark"
)

parser_conv = subparsers.add_parser("conv", help="Benchmark convolution")
parser_conv.add_argument("--serial", action="store_true")
parser_conv.add_argument(
    "sizes", metavar="N", type=int, nargs="+", help="image sizes to benchmark"
)

for cnn_short in ["cnn", "cnn-nchwc"]:
    parser_cnn = subparsers.add_parser(cnn_short, help="Benchmark small CNN")
    parser_cnn.add_argument("--serial", action="store_true")
    parser_cnn.add_argument(
        "sizes", metavar="N", type=int, nargs="+", help="image sizes to benchmark"
    )


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    # Some dtypes aren't supported by PyTorch, but it's fine to convert them to the
    # closest supported type. This shouldn't affect benchmark results meaningfully.
    if arr.dtype == np.uint32:
        arr = arr.astype("int32")
    return torch.from_numpy(arr)


def _is_realtime():
    return os.sched_getparam(os.getpid()).sched_priority > 0


@contextlib.contextmanager
def _perf_events(output_dir: pathlib.Path, pid=None, tid=None):
    assert not (pid and tid)
    assert pid or tid

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_path = os.getenv("MORELLO_PERF")
    if not bin_path:
        bin_path = shutil.which("perf")
    if not bin_path:
        warnings.warn("perf not found in PATH or MORELLO_PERF; will not profile")
        yield
        return

    data_path = output_dir / "perf.data"
    text_path = output_dir / "perf.txt"

    perf_record_cmd = [str(bin_path)]
    if os.getenv("MORELLO_PERF_SYMFS"):
        perf_record_cmd += ["--symfs", os.getenv("MORELLO_PERF_SYMFS")]
    perf_record_cmd += [
        "stat",
        "--quiet",
        "-ddd",
        "record",
        "-o",
        str(data_path),
    ]
    if pid:
        perf_record_cmd += ["-p", str(pid)]
    else:
        perf_record_cmd += ["-t", str(tid)]

    with subprocess.Popen(
        perf_record_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        yield
        p.send_signal(signal.SIGINT)
        try:
            _, stderr = p.communicate(timeout=PERF_TERMINATE_TIMEOUT)
            print("perf record stderr:")
            print(stderr.decode("utf-8"))
        except TimeoutError:
            pass
        else:
            if stderr:
                logger.warning("perf stderr: %s", stderr.decode("utf8"))
    if p.returncode not in (0, -signal.SIGINT):
        logger.error("perf exited with error code %d", p.returncode)
        return

    # Produce textual output from the saved report
    with text_path.open("w") as f:
        try:
            subprocess.run(
                [bin_path, "stat", "report", "-i", data_path],
                check=True,
                stderr=f,
                stdout=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            logger.error("perf stat report failed; continuing anyway: %s", e)
            if e.stderr:
                logger.error("stderr from perf script: %s", e.stderr.decode("utf8"))


class Benchmark:
    @property
    def spec(self) -> specs.Spec:
        raise NotImplementedError()

    def backends(
        self,
        cache: Union[str, pathlib.Path, None],
        save_cache: bool,
        red: Optional[tuple[str, str]],
        sqlite_db: Optional[pathlib.Path],
        sqlite_cache_kb: Optional[int],
        extras_dir: pathlib.Path,
    ) -> Iterable[tuple[str, Callable[[], "BenchmarkBackend"]]]:
        np_backend = lambda: self._numpy_backend(extras_dir / "numpy")
        torch_backend = lambda: self._torch_backend(extras_dir / "torch")
        torchscript_backend = lambda: TorchScriptBackend(
            torch_backend(), extras_dir / "torchscript"
        )

        yield "numpy", np_backend
        yield "torch", torch_backend
        yield "torchscript", torchscript_backend
        yield "relay", lambda: RelayBackend(torchscript_backend(), extras_dir / "relay")
        yield "tvmautoscheduler", lambda: TVMAutoschedulerBackend(
            torchscript_backend(), extras_dir / "tvmautoscheduler"
        )
        yield "jax", lambda: self._jax_backend(extras_dir / "jax")
        yield "halide", lambda: self._halide_backend(extras_dir / "halide")
        yield "morello", lambda: MorelloBackend(
            self,
            cache,
            save_cache,
            red,
            sqlite_db,
            sqlite_cache_kb,
            extras_dir / "morello",
        )

    @property
    def cpus_used(self) -> int:
        if self.spec.serial_only:
            return 1
        return multiprocessing.cpu_count()

    @property
    def short_name(self) -> str:
        raise NotImplementedError(f"Not implemented for {type(self).__name__}")

    def _numpy_backend(self, extras_dir) -> "BenchmarkBackend":
        raise NotImplementedError()

    def _torch_backend(self, extras_dir) -> "BenchmarkBackend":
        raise NotImplementedError()

    def _jax_backend(self, extras_dir) -> "BenchmarkBackend":
        raise NotImplementedError()

    def _halide_backend(self, extras_dir) -> "BenchmarkBackend":
        raise NotImplementedError()


class BenchmarkBackend:
    extras_dir: pathlib.Path

    def __init__(self, extras_dir: pathlib.Path) -> None:
        self.extras_dir = extras_dir
        extras_dir.mkdir(parents=True, exist_ok=True)

    def run(self, trials: int) -> tuple[list[float], list[int], int, bool]:
        raise NotImplementedError()


class MorelloBackend(BenchmarkBackend):
    def __init__(
        self,
        benchmark: Benchmark,
        cache: Union[str, pathlib.Path, None],
        save_cache: bool,
        red,
        sqlite_db,
        sqlite_cache_kb,
        extras_dir: pathlib.Path,
    ):
        super().__init__(extras_dir)
        self.benchmark = benchmark
        self.cache_path = cache
        self.save_cache = save_cache
        self.red = red
        self.sqlite_db = sqlite_db
        self.sqlite_cache_kb = sqlite_cache_kb

    def run(self, trials: int) -> tuple[list[float], list[int], int, bool]:
        spec = self.benchmark.spec
        with search_cache.persistent_cache(
            self.cache_path,
            self.red,
            self.sqlite_db,
            self.sqlite_cache_kb,
            save=self.save_cache,
        ) as cache:
            start = time.time()
            search_result = asyncio.run(search.schedule_search(spec, cache=cache))
            if not search_result:
                raise Exception(f"Failed to find a schedule for {spec}")
            impl = search_result[0]
            logger.info("Completed synthesis. Took %.2fs", time.time() - start)
        assert impl

        runtime_samples, impl_str, source_code = _benchmark(impl, trials)

        assert runtime_samples
        with (self.extras_dir / "impl.txt").open("w") as fo:
            fo.write(impl_str)
        with (self.extras_dir / "source.c").open("w") as fo:
            fo.write(source_code)

        process = psutil.Process(os.getpid())
        return runtime_samples, process.cpu_affinity(), process.nice(), _is_realtime()

    @property
    def short_name(self) -> str:
        if cost.INST_COST != 1000:
            return f"morello-arith{cost.INST_COST}"
        return "morello"


class LoopingBackend(BenchmarkBackend):
    """A base class for benchmarks which execute a Python codelet in a loop."""

    @property
    def _should_run_rt(self) -> bool:
        return True

    @property
    def _use_subprocess(self) -> bool:
        return False

    @property
    def codelet(self) -> str:
        raise NotImplementedError()

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        raise NotImplementedError()

    def run(self, trials: int) -> tuple[list[float], list[int], int, bool]:
        # TODO: Pull building out of the profiled loop

        rough_secs = self._run_with_profiler(
            1, 1, lambda **_: contextlib.nullcontext()
        )[0][0]
        goal_samples = max(
            MIN_SAMPLES, int(math.ceil(MIN_TRIAL_TIME_SECS / rough_secs))
        )
        logger.info("Goal samples: %d", goal_samples)

        self._run_with_profiler(
            1,
            goal_samples,
            lambda **k: _perf_events(self.extras_dir / "perf", **k),
        )

        return self._run_with_profiler(
            trials, goal_samples, lambda **_: contextlib.nullcontext()
        )

    def _run_with_profiler(
        self, *args, **kwargs
    ) -> tuple[list[float], list[int], int, bool]:
        if self._use_subprocess:
            return self._run_with_profiler_process(*args, **kwargs)
        return self._run_with_profiler_thread(*args, **kwargs)

    def _run_with_profiler_process(
        self, trials: int, samples: int, perf_ctx
    ) -> tuple[list[float], list[int], int, bool]:
        mp_ctx = multiprocessing.get_context("fork")
        go_q, r_q = mp_ctx.Queue(), mp_ctx.Queue()
        child_process = mp_ctx.Process(
            target=self._subproc_wrapper,
            args=(go_q, r_q, None, trials, samples),
            daemon=True,
        )
        child_process.start()
        _, cpu_affinity, nice, is_rt = r_q.get()  # Drop the returned thread ID only.
        with perf_ctx(pid=child_process.pid):
            go_q.put("go")
            success, result = r_q.get()
            child_process.join()
            if not success:
                raise result
        assert result is not None
        return result, cpu_affinity, nice, is_rt

    def _run_with_profiler_thread(
        self, trials: int, samples: int, perf_ctx
    ) -> tuple[list[float], list[int], int, bool]:
        go_q, r_q = queue.Queue(), queue.Queue()

        c = contextvars.copy_context()
        child_thread = threading.Thread(
            target=c.run,
            args=(self._subproc_wrapper, go_q, r_q, trials, samples),
            daemon=True,
            name=f"ProfTh{type(self).__name__}",
        )
        child_thread.start()
        tid, cpu_affinity, nice, is_rt = r_q.get()
        with perf_ctx(tid=tid):
            go_q.put("go")
            success, result = r_q.get()
            child_thread.join()
            if not success:
                raise result
        assert result is not None
        return result, cpu_affinity, nice, is_rt

    def _subproc_wrapper(self, go_q, result_queue, *args, **kwargs):
        """Calls `_subproc_run`, controlled by and reporting to given queues.

        If `self._should_run_rt`, this will set current process to real-time
        priority immediately.

        Meant to be called either in a subprocess or thread.
        """
        try:
            process = psutil.Process(os.getpid())
            result_queue.put(
                (
                    threading.get_native_id(),
                    process.cpu_affinity(),
                    process.nice(),
                    _is_realtime(),
                )
            )
            if self._should_run_rt:
                # Set own priority to realtime (99)
                max_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
                os.sched_setscheduler(0, os.SCHED_FIFO, max_param)  # type: ignore
            msg = go_q.get()
            assert msg == "go"
            result = self._subproc_run(*args, **kwargs)
        except Exception as e:
            result_queue.put((False, e))
        else:
            result_queue.put((True, result))

    def _subproc_run(self, trials, sample_cnt: int) -> list[float]:
        benchmark_code = f"""
global inner_benchmark

def inner_benchmark(self, {', '.join([n for n, _ in self.data_deps])}):
    {self.codelet}

    start = time.perf_counter()
    for _ in range({sample_cnt}):
        {self.codelet}
    end = time.perf_counter()
    return (end - start) / {sample_cnt}
        """
        exec(benchmark_code)

        runs = []
        for _ in range(trials):
            runs.append(inner_benchmark(self, *[v for _, v in self.data_deps]))  # type: ignore
        return runs


class BaseJAXBackend(LoopingBackend):
    def __init__(self, benchmark: Benchmark, extras_dir, print_graphs=True) -> None:
        super().__init__(extras_dir)

        self.benchmark = benchmark
        self.jitted_fn = jax.jit(self.make_jax_func())
        self.set_inputs()

        if print_graphs:
            print(
                jax_to_ir.jax_to_ir(
                    self.jitted_fn,
                    input_shapes=[
                        (n, jax.ShapedArray(arr.shape, arr.dtype))
                        for n, arr in self.data_deps
                    ],
                    format="HLO",
                )[1]
            )

    @property
    def _should_run_rt(self) -> bool:
        return False

    def make_jax_func(self):
        raise NotImplementedError()

    def set_inputs(self):
        raise NotImplementedError()

    def run(self, trials: int) -> tuple[list[float], list[int], int, bool]:
        if not self.benchmark.serial_only:
            return super().run(trials)
        with _affinity_ctx(1):
            return super().run(trials)

    @property
    def codelet(self) -> str:
        dep_part = ", ".join(n for n, _ in self.data_deps)
        return f"self.jitted_fn({dep_part}).block_until_ready()"


@dataclasses.dataclass
class MatmulBenchmark(Benchmark):
    batch_size: int
    size: int
    serial_only: bool

    @classmethod
    def from_args(cls, args) -> Iterable["MatmulBenchmark"]:
        if args.spec != "matmul":
            return
        assert len(args.batch)
        for b, n in itertools.product(args.batch, args.sizes):
            yield MatmulBenchmark(b, n, args.serial)

    @property
    def spec(self) -> specs.Matmul:
        if self.batch_size != 1:
            raise NotImplementedError("Batched matrix multiplication not yet supported")
        target = system_config.current_target()
        return specs.Matmul(
            target.tensor_spec((self.size, self.size), dtype=DTYPE),
            target.tensor_spec((self.size, self.size), dtype=DTYPE),
            target.tensor_spec((self.size, self.size), dtype=DTYPE),
            serial_only=self.serial_only,
        )

    @property
    def short_name(self) -> str:
        return "matmul"

    def _numpy_backend(self, extras_dir) -> "BenchmarkBackend":
        return MatmulNumpy(self, extras_dir)

    def _torch_backend(self, extras_dir) -> "BenchmarkBackend":
        return MatmulTorch(self, extras_dir)

    def _jax_backend(self, extras_dir) -> "BenchmarkBackend":
        return MatmulJAX(self, extras_dir)

    def _halide_backend(self, extras_dir) -> "BenchmarkBackend":
        return MatmulHalide(self, extras_dir)


class MatmulNumpy(LoopingBackend):
    def __init__(self, benchmark: MatmulBenchmark, extras_dir) -> None:
        super().__init__(extras_dir)
        self.benchmark = benchmark
        spec = self.benchmark.spec
        (m, n), k = spec.output.dim_sizes, spec.lhs.dim_sizes[1]
        self.lhs = np.arange(m * k, dtype=DTYPE.np_type).reshape((m, k))
        self.rhs = np.arange(k * n, dtype=DTYPE.np_type).reshape((k, n))

    @property
    def short_name(self) -> str:
        return "numpy"

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("lhs", self.lhs), ("rhs", self.rhs)]

    @property
    def codelet(self) -> str:
        return """lhs @ rhs"""


class MatmulJAX(BaseJAXBackend):
    def make_jax_func(self):
        def jax_mm(lhs, rhs):
            return jax.numpy.matmul(lhs, rhs)

        return jax_mm

    def set_inputs(self):
        numpy_backend = MatmulNumpy(self.benchmark, self.extras_dir)
        self.lhs = jax.numpy.array(numpy_backend.lhs)
        self.rhs = jax.numpy.array(numpy_backend.rhs)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("lhs", self.lhs), ("rhs", self.rhs)]


class BaseTorchBackend(LoopingBackend):
    def run(self, trials: int) -> tuple[list[float], list[int], int, bool]:
        # TODO: Save to file
        termcolor.cprint("PyTorch Configuration:", attrs=["bold"])
        print(torch.__config__.show())

        orig_intraop_threads = torch.get_num_threads()
        torch.set_num_threads(self.expected_threads())
        try:
            return super().run(trials)
        finally:
            torch.set_num_threads(orig_intraop_threads)

    def expected_threads(self) -> int:
        if self.benchmark.spec.serial_only:
            return 1
        return multiprocessing.cpu_count()


class MatmulTorch(BaseTorchBackend):
    def __init__(self, benchmark: MatmulBenchmark, extras_dir) -> None:
        super().__init__(extras_dir)
        self.benchmark = benchmark
        numpy_backend = MatmulNumpy(benchmark, extras_dir)
        self.lhs = _to_torch(numpy_backend.lhs)
        self.rhs = _to_torch(numpy_backend.rhs)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("lhs", self.lhs), ("rhs", self.rhs)]

    @property
    def codelet(self) -> str:
        return """torch.matmul(lhs, rhs)"""


class TorchScriptBackend(BaseTorchBackend):
    def __init__(self, torch_backend: BenchmarkBackend, extras_dir, print_graphs=True):
        super().__init__(extras_dir)
        self.torch_backend = torch_backend
        self.print_graphs = print_graphs
        self.jitted_fn = torch.compile(self._make_jittable())

    def _make_jittable(self):
        codelet = f"""
import torch
import torch.nn.functional as F

global jittable

def jittable({', '.join([n for n, _ in self.torch_backend.data_deps])}):
    return {self.torch_backend.codelet}
        """
        exec(codelet)
        return jittable  # type: ignore

    @property
    def benchmark(self):
        return self.torch_backend.benchmark

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return self.torch_backend.data_deps

    @property
    def codelet(self) -> str:
        return f"self.jitted_fn({', '.join([n for n, _ in self.data_deps])})"

    def run(self, trials: int) -> tuple[list[float], list[int], int, bool]:
        result = super().run(trials)
        if self.print_graphs:

            def custom_backend(graph_module, example_inputs):
                print("")
                termcolor.cprint("Torch:", attrs=["bold"])
                graph_module.graph.print_tabular()
                return graph_module.forward

            torch._dynamo.reset()
            opt_model = torch.compile(self._make_jittable(), backend=custom_backend)
            opt_model(*[t for _, t in self.data_deps])
            torch._dynamo.reset()

        return result


class _RelayBase(LoopingBackend):
    tvm_m: "tvm.contrib.graph_executor.GraphModule"
    relay_mod: "tvm.ir.module.IRModule"
    relay_params: dict[str, "tvm.nd.NDArray"]

    def __init__(
        self, torchscript_backend: TorchScriptBackend, extras_dir: pathlib.Path
    ):
        self.torchscript_backend = torchscript_backend
        self.extras_dir = extras_dir
        extras_dir.mkdir(parents=True, exist_ok=True)

        # PyTorch graph on Relay/TVM
        self._tvm_target = self._make_tvm_target()
        tvm_lib, relay_mod, relay_params = self._make_relay(extras_dir)

        self.tvm_m = tvm.contrib.graph_executor.GraphModule(
            tvm_lib["default"](tvm.cpu(0))
        )
        for n, v in torchscript_backend.data_deps:
            self.tvm_m.set_input(n, v)

    @property
    def benchmark(self):
        return self.torchscript_backend.benchmark

    def run(self, *args, **kwargs):
        orig_val = os.getenv("TVM_NUM_THREADS")
        os.environ["TVM_NUM_THREADS"] = str(self.expected_threads())
        try:
            return super().run(*args, **kwargs)
        finally:
            if orig_val is None:
                del os.environ["TVM_NUM_THREADS"]
            else:
                os.environ["TVM_NUM_THREADS"] = orig_val

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return []

    @property
    def codelet(self) -> str:
        return "self.tvm_m.run()"

    def expected_threads(self) -> int:
        if self.benchmark.spec.serial_only:
            return 1
        return multiprocessing.cpu_count()

    def _make_tvm_target(self):
        # TODO: Produce target from target backend
        return tvm.target.Target("llvm -mcpu=core-avx2")


class RelayBackend(_RelayBase):
    def _make_relay(self, extras_dir: pathlib.Path):
        backend = self.torchscript_backend
        shape_list = [(n, v.shape) for n, v in backend.data_deps]
        relay_mod, relay_params = relay.frontend.from_pytorch(
            backend.jitted_fn, shape_list
        )

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

        with tvm.transform.PassContext(opt_level=3, instruments=[PIR()]):
            tvm_lib = relay.build(
                relay_mod, target=self._tvm_target, params=relay_params
            )

        with (extras_dir / "tvm_mod.txt").open("w") as fo:
            fo.write(tvm_mod_text)
        with (extras_dir / "tvm_source.txt").open("w") as fo:
            fo.write(tvm_lib.get_lib().get_source())
        cc = os.getenv("CXX")
        tvm_lib.export_library(str(extras_dir / "network.so"), cc=cc)

        return tvm_lib, relay_mod, relay_params


class TVMAutoschedulerBackend(_RelayBase):
    def _make_relay(self, extras_dir: pathlib.Path):
        extras_dir.mkdir(parents=True, exist_ok=True)

        backend = self.torchscript_backend
        shape_list = [(n, v.shape) for n, v in backend.data_deps]
        relay_mod, relay_params = relay.frontend.from_pytorch(
            backend.jitted_fn, shape_list
        )

        target = self._make_tvm_target()
        tasks, task_weights = auto_scheduler.extract_tasks(
            relay_mod["main"], relay_params, target
        )

        with (extras_dir / "tasks.txt").open("w") as f:
            for idx, task in enumerate(tasks):
                print(
                    "========== Task %d  (workload key: %s) =========="
                    % (idx, task.workload_key),
                    file=f,
                )
                print(task.compute_dag, file=f)

        # Begin tuning
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        log_file_path = str(extras_dir / "tuning.log")
        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=2000,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file_path)],
        )
        tune_start = time.perf_counter()
        tuner.tune(tuning_options)
        tune_end = time.perf_counter()
        with (extras_dir / "runtime.txt").open("w") as f:
            print(f"Tuning took {tune_end - tune_start:.3f}s", file=f)

        # Compile with the history best
        with auto_scheduler.ApplyHistoryBest(log_file_path):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(relay_mod, target=target, params=relay_params)

        return lib, relay_mod, relay_params


@dataclasses.dataclass
class GEMM3Benchmark(Benchmark):
    batch_size: int
    size: int
    serial_only: bool

    @classmethod
    def from_args(cls, args) -> Iterable["GEMM3Benchmark"]:
        if args.spec != "gemm3":
            return
        assert len(args.batch)
        for b, n in itertools.product(args.batch, args.sizes):
            yield GEMM3Benchmark(b, n, args.serial)

    @property
    def spec(self) -> specs.Spec:
        if self.batch_size != 1:
            raise NotImplementedError("Batched matrix multiplication not yet supported")
        target = system_config.current_target()
        return specs.Compose(
            (specs.Matmul, specs.Matmul),
            inputs=(
                target.tensor_spec((self.size, self.size), dtype=DTYPE),
                target.tensor_spec((self.size, self.size), dtype=DTYPE),
                target.tensor_spec((self.size, self.size), dtype=DTYPE),
            ),
            output=target.tensor_spec((self.size, self.size), dtype=DTYPE),
            intermediate_dtypes=(DTYPE,),
            serial_only=self.serial_only,
        )

    @property
    def short_name(self) -> str:
        return "gemm3"

    def _numpy_backend(self, extras_dir) -> "BenchmarkBackend":
        return GEMM3Numpy(self, extras_dir)

    def _jax_backend(self, extras_dir) -> "BenchmarkBackend":
        return GEMM3JAX(self, extras_dir)

    def _torch_backend(self, extras_dir) -> "BenchmarkBackend":
        return GEMM3Torch(self, extras_dir)

    def _halide_backend(self, extras_dir) -> "BenchmarkBackend":
        return GEMM3Halide(self, extras_dir)


class GEMM3Numpy(LoopingBackend):
    def __init__(self, benchmark: GEMM3Benchmark, extras_dir) -> None:
        super().__init__(extras_dir)
        self.benchmark = benchmark
        m = self.benchmark.size
        self.a = np.arange(m * m, dtype=DTYPE.np_type).reshape((m, m))
        self.b = np.arange(m * m, dtype=DTYPE.np_type).reshape((m, m))
        self.c = np.arange(m * m, dtype=DTYPE.np_type).reshape((m, m))

    @property
    def short_name(self) -> str:
        return "numpy"

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("a", self.a), ("b", self.b), ("c", self.c)]

    @property
    def codelet(self) -> str:
        return """(a @ b) @ c"""


class GEMM3JAX(BaseJAXBackend):
    def make_jax_func(self):
        def jax_gemm3(a, b, c):
            return jax.numpy.matmul(jax.numpy.matmul(a, b), c)

        return jax_gemm3

    def set_inputs(self):
        numpy_backend = GEMM3Numpy(self.benchmark)
        self.a = jax.numpy.array(numpy_backend.a)
        self.b = jax.numpy.array(numpy_backend.b)
        self.c = jax.numpy.array(numpy_backend.c)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("a", self.a), ("b", self.b), ("c", self.c)]


class GEMM3Torch(BaseTorchBackend):
    def __init__(self, benchmark: GEMM3Benchmark, extras_dir) -> None:
        super().__init__(extras_dir)
        self.benchmark = benchmark
        numpy_backend = GEMM3Numpy(benchmark)
        self.a = _to_torch(numpy_backend.a)
        self.b = _to_torch(numpy_backend.b)
        self.c = _to_torch(numpy_backend.c)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("a", self.a), ("b", self.b), ("c", self.c)]

    @property
    def codelet(self) -> str:
        return """torch.matmul(torch.matmul(a, b), c)"""


@dataclasses.dataclass
class ConvBenchmark(Benchmark):
    batch_size: int
    size: int
    serial_only: bool

    @classmethod
    def from_args(cls, args) -> Iterable["ConvBenchmark"]:
        if args.spec != "conv":
            return
        assert len(args.batch)
        for b, n in itertools.product(args.batch, args.sizes):
            yield ConvBenchmark(b, n, args.serial)

    @property
    def spec(self) -> specs.Convolution:
        target = system_config.current_target()

        img_channels = 4
        fh, fw, fc = 5, 5, 32
        out_h, out_w = 1 + self.size - fh, 1 + self.size - fw

        return specs.Convolution(
            target.tensor_spec(
                (self.batch_size, img_channels, self.size, self.size), dtype=DTYPE
            ),
            target.tensor_spec((fc, img_channels, fh, fw), dtype=DTYPE),
            output=target.tensor_spec((self.batch_size, fc, out_h, out_w), dtype=DTYPE),
            serial_only=self.serial_only,
        )

    @property
    def short_name(self) -> str:
        return "conv"

    def _numpy_backend(self, extras_dir) -> "BenchmarkBackend":
        return ConvNumpy(self, extras_dir)

    def _torch_backend(self, extras_dir) -> "BenchmarkBackend":
        return ConvTorch(self, extras_dir)

    def _jax_backend(self, extras_dir) -> "BenchmarkBackend":
        return ConvJAX(self, extras_dir)


class ConvNumpy(LoopingBackend):
    def __init__(self, benchmark: ConvBenchmark, extras_dir) -> None:
        super().__init__(extras_dir)

        self.benchmark = benchmark

        spec = benchmark.spec
        batch, channels, h, w = spec.lhs.dim_sizes
        filters_count, _, fh, fw = spec.rhs.dim_sizes

        # Use signed int32 with PyTorch.
        assert DTYPE == dtypes.Uint32

        self.img = np.arange(batch * channels * h * w, dtype=TORCH_DTYPE_NP).reshape(
            (batch, channels, h, w)
        )
        self.filters = np.arange(
            filters_count * channels * fh * fw, dtype=TORCH_DTYPE_NP
        ).reshape((filters_count, channels, fh, fw))

    @property
    def short_name(self) -> str:
        return "numpy"

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("img", self.img), ("filters", self.filters)]

    @property
    def codelet(self) -> str:
        raise NotImplementedError()


class ConvTorch(BaseTorchBackend):
    def __init__(self, benchmark: ConvBenchmark, extras_dir) -> None:
        self.benchmark = benchmark
        numpy_backend = ConvNumpy(benchmark, extras_dir / "torch")
        self.img = _to_torch(numpy_backend.img)
        self.filters = _to_torch(numpy_backend.filters)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("img", self.img), ("filters", self.filters)]

    @property
    def codelet(self) -> str:
        return """torch.conv2d(img, filters)"""


class ConvJAX(BaseJAXBackend):
    def make_jax_func(self):
        def jax_conv(img, filters):
            return lax.conv(img, filters, (1, 1), "VALID")

        return jax_conv

    def set_inputs(self):
        numpy_backend = ConvNumpy(self.benchmark, self.extras_dir / "jax")
        self.img = jax.numpy.array(numpy_backend.img)
        self.filters = jax.numpy.array(numpy_backend.filters)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("img", self.img), ("filters", self.filters)]


@dataclasses.dataclass
class CNNBenchmark(Benchmark):
    batch_size: int
    image_size: int
    serial_only: bool

    @property
    def size(self):
        return self.image_size

    @classmethod
    def from_args(cls, args) -> Iterable["CNNBenchmark"]:
        if args.spec != "cnn":
            return
        assert len(args.batch)
        for b, n in itertools.product(args.batch, args.sizes):
            yield CNNBenchmark(b, n, args.serial)

    @property
    def spec(self) -> specs.Compose:
        target = system_config.current_target()

        img = target.tensor_spec(
            (self.batch_size, 4, self.image_size, self.image_size),
            dtype=DTYPE,
            layout=layouts.row_major(4),
        )
        filters_a = target.tensor_spec(
            (32, 4, 3, 3), dtype=DTYPE, layout=layouts.row_major(4)
        )
        filters_b = target.tensor_spec(
            (32, 32, 3, 3), dtype=DTYPE, layout=layouts.row_major(4)
        )
        output = target.tensor_spec(
            (self.batch_size, 32, self.image_size - 4, self.image_size - 4),
            dtype=DTYPE,
            layout=layouts.row_major(4),
        )
        return specs.Compose(
            (specs.Convolution, specs.Convolution),
            (filters_b, img, filters_a),
            output,
            intermediate_dtypes=(DTYPE,),
            serial_only=self.serial_only,
        )

    @property
    def short_name(self) -> str:
        return "cnn"

    def _torch_backend(self, extras_dir) -> "BenchmarkBackend":
        return CNNTorch(self, extras_dir)

    def _jax_backend(self, extras_dir) -> "BenchmarkBackend":
        return CNNJAX(self, extras_dir)

    def _halide_backend(self, extras_dir) -> "BenchmarkBackend":
        return CNNHalide(self, extras_dir)


@dataclasses.dataclass
class CNNHCHWcBenchmark(CNNBenchmark):
    batch_size: int
    image_size: int
    serial_only: bool

    @property
    def size(self):
        return self.image_size

    @classmethod
    def from_args(cls, args) -> Iterable["CNNBenchmark"]:
        if args.spec != "cnn-nchwc":
            return
        assert len(args.batch)
        for b, n in itertools.product(args.batch, args.sizes):
            yield CNNHCHWcBenchmark(b, n, args.serial)

    @property
    def spec(self) -> specs.Compose:
        target = system_config.current_target()

        img = target.tensor_spec(
            (self.batch_size, 4, self.image_size, self.image_size),
            dtype=DTYPE,
            layout=layouts.NCHWc4,
        )
        filters_a = target.tensor_spec(
            (32, 4, 3, 3), dtype=DTYPE, layout=layouts.NCHWc4
        )
        filters_b = target.tensor_spec(
            (32, 32, 3, 3), dtype=DTYPE, layout=layouts.NCHWc4
        )
        output = target.tensor_spec(
            (self.batch_size, 32, self.image_size - 4, self.image_size - 4),
            dtype=DTYPE,
            layout=layouts.NCHWc4,
        )
        return specs.Compose(
            (specs.Convolution, specs.Convolution),
            (filters_b, img, filters_a),
            output,
            intermediate_dtypes=(DTYPE,),
            serial_only=self.serial_only,
        )

    @property
    def short_name(self) -> str:
        return "cnn-nchwc"


class RelayCNNNCHWcBackend(LoopingBackend):
    def __init__(self, extras_dir: pathlib.Path):
        self.extras_dir = extras_dir
        self.extras_dir.mkdir(parents=True, exist_ok=True)

        relay_mod = None
        relay_params = None

        layers.conv2d(
            data=data,
            channels=filter_list[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            name="conv0",
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )

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

        with tvm.transform.PassContext(opt_level=3, instruments=[PIR()]):
            tvm_lib = relay.build(relay_mod, target=tvm_target, params=relay_params)

        self.tvm_m = tvm.contrib.graph_executor.GraphModule(
            tvm_lib["default"](tvm.cpu(0))
        )

        # TODO: Set inputs as follows:
        # for n, v in torchscript_backend.data_deps:
        #     self.tvm_m.set_input(n, v)

        with (extras_dir / "tvm_mod.txt").open("w") as fo:
            fo.write(tvm_mod_text)

    @property
    def benchmark(self):
        return self.torchscript_backend.benchmark

    def run(self, *args, **kwargs):
        orig_val = os.getenv("TVM_NUM_THREADS")
        os.environ["TVM_NUM_THREADS"] = str(self.expected_threads())
        try:
            return super().run(*args, **kwargs)
        finally:
            if orig_val is None:
                del os.environ["TVM_NUM_THREADS"]
            else:
                os.environ["TVM_NUM_THREADS"] = orig_val

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return []

    @property
    def codelet(self) -> str:
        return "self.tvm_m.run()"

    def expected_threads(self) -> int:
        if self.benchmark.spec.serial_only:
            return 1
        return multiprocessing.cpu_count()


class CNNTorch(BaseTorchBackend):
    def __init__(self, benchmark: CNNBenchmark, extras_dir) -> None:
        super().__init__(extras_dir)
        self.benchmark = benchmark
        self.img, self.filters_a, self.filters_b = map(
            _to_torch, _make_np_cnn_inputs(self.benchmark.spec)
        )

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("img", self.img), ("fa", self.filters_a), ("fb", self.filters_b)]

    @property
    def codelet(self) -> str:
        return "F.conv2d(F.conv2d(img, fa), fb)"


class CNNJAX(BaseJAXBackend):
    def make_jax_func(self):
        def jax_cnn(img, fa, fb):
            a = lax.conv(img, fa, (1, 1), "VALID")
            b = lax.conv(a, fb, (1, 1), "VALID")
            return b

        return jax_cnn

    def set_inputs(self):
        self.img, self.filters_a, self.filters_b = _make_np_cnn_inputs(
            self.benchmark.spec
        )
        self.img = jax.numpy.array(self.img)
        self.filters_a = jax.numpy.array(self.filters_a)
        self.filters_b = jax.numpy.array(self.filters_b)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("img", self.img), ("fa", self.filters_a), ("fb", self.filters_b)]


class BaseHalideBackend(LoopingBackend):
    def __init__(self, benchmark: Benchmark, extras_dir, print_graphs=True):
        super().__init__(extras_dir)

        self.benchmark = benchmark

        fn = self._make_pipeline()
        machine = hl.MachineParams(
            (1 if benchmark.serial_only else multiprocessing.cpu_count()), 0, 0
        )
        fn.auto_schedule("Adams2019", hl.get_jit_target_from_environment(), machine)
        fn.compile_jit()

        self.fn = fn

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
        print(f"output dims: {self.benchmark.spec.output.dim_sizes}")

    @property
    def codelet(self) -> str:
        return f"self.fn.realize(out_size)"

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("out_size", self.benchmark.spec.output.dim_sizes)]

    def _make_pipeline(self):
        raise NotImplementedError()


class MatmulHalide(BaseHalideBackend):
    def _make_pipeline(self):
        mmnp = MatmulNumpy(self.benchmark, self.extras_dir)
        lhs_hl, rhs_hl = (
            hl.Buffer(mmnp.lhs, name="lhs"),
            hl.Buffer(mmnp.rhs, name="rhs"),
        )
        fn = toyhl.halide_matmul(lhs_hl, rhs_hl, mmnp.lhs.shape[0])
        fn = hl.Pipeline(fn)
        return fn


class GEMM3Halide(BaseHalideBackend):
    def _make_pipeline(self):
        mmnp = GEMM3Numpy(self.benchmark)
        a_hl, b_hl, c_hl = (
            hl.Buffer(mmnp.a, name="a"),
            hl.Buffer(mmnp.b, name="b"),
            hl.Buffer(mmnp.c, name="c"),
        )
        fn = toyhl.halide_gemm3(a_hl, b_hl, c_hl, mmnp.a.shape[0])
        fn = hl.Pipeline(fn)
        return fn


class CNNHalide(BaseHalideBackend):
    def _make_pipeline(self):
        img, filters_a, filters_b = _make_np_cnn_inputs(self.benchmark.spec)
        img_hl, filters_a_hl, filters_b_hl = (
            hl.Buffer(img, name="img"),
            hl.Buffer(filters_a, name="filters_a"),
            hl.Buffer(filters_b, name="filters_b"),
        )
        fn = toyhl.halide_small_cnn(img_hl, filters_a_hl, filters_b_hl)
        fn = hl.Pipeline(fn)
        return fn


def _make_np_cnn_inputs(spec):
    batch_size, channels, h, w = spec.inputs[-2].dim_sizes
    fc_a, _, fh_a, fw_a = spec.inputs[-1].dim_sizes
    fc_b, _, fh_b, fw_b = spec.inputs[-3].dim_sizes

    img = np.arange(batch_size * channels * h * w, dtype=TORCH_DTYPE_NP).reshape(
        (batch_size, channels, h, w)
    )
    filters_a = np.arange(fh_a * fw_a * fc_a * channels, dtype=TORCH_DTYPE_NP).reshape(
        (fc_a, channels, fh_a, fw_a)
    )
    filters_b = np.arange(fh_b * fw_b * fc_b * fc_a, dtype=TORCH_DTYPE_NP).reshape(
        (fc_b, fc_a, fh_b, fw_b)
    )
    return img, filters_a, filters_b


def _benchmark(impl, trials: int):
    assert impl.is_scheduled
    loop = asyncio.new_event_loop()

    # Collect a single rough sample.
    time_check_artifact = loop.run_until_complete(
        system_config.current_target().build_impl(impl, benchmark_samples=1)
    )
    rough_secs = loop.run_until_complete(time_check_artifact.measure_time())
    goal_samples = max(MIN_SAMPLES, int(math.ceil(MIN_TRIAL_TIME_SECS / rough_secs)))
    logger.info("Goal samples: %d", goal_samples)

    artifact = loop.run_until_complete(
        system_config.current_target().build_impl(impl, benchmark_samples=goal_samples)
    )
    assert hasattr(artifact, "source_code")
    source = artifact.source_code  # type: ignore
    runtime_samples = []
    for _ in range(trials):
        secs = loop.run_until_complete(artifact.measure_time())
        logger.info(f"Sample runtime result {secs}s:")
        runtime_samples.append(secs)
    impl_str = op_pprint.pformat(impl, color=False)
    return runtime_samples, impl_str, source  # type: ignore


def _get_benchmark_classes():
    return [
        MatmulBenchmark,
        GEMM3Benchmark,
        ConvBenchmark,
        CNNBenchmark,
        CNNHCHWcBenchmark,
    ]


@contextlib.contextmanager
def _affinity_ctx(num_cpus: int):
    orig_cpus = os.sched_getaffinity(0)
    random_cpus = random.sample(sorted(orig_cpus), num_cpus)
    os.sched_setaffinity(0, random_cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, orig_cpus)


@contextlib.contextmanager
def _scale_governor(path: Optional[pathlib.Path]):
    yield


def _gdrive_upload_dir(
    drive: pydrive2.drive.GoogleDrive,
    local_dir: pathlib.Path,
    remote_root_name: str,
    parent_id: Optional[str] = None,
):
    assert local_dir.is_dir()

    # Get root folder in Drive based on provided name
    if not parent_id:
        remote_root_candidates = drive.ListFile(
            {"q": f"title = '{remote_root_name}' and trashed = False"}
        ).GetList()
        if not remote_root_candidates:
            raise ValueError(f"Found no folders with title '{remote_root_name}'")
        if len(remote_root_candidates) > 1:
            raise ValueError(f"Found multiple folders with title '{remote_root_name}'")
        parent_id = remote_root_candidates[0]["id"]

    # Create new remote subdirectory corresponding to the top of local_dir
    root_meta: dict[str, Any] = {
        "title": local_dir.name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    root_meta["parents"] = [{"id": parent_id}]
    root_item = drive.CreateFile(root_meta)
    root_item.Upload()

    for entry in local_dir.iterdir():
        if entry.is_file():
            # Upload file
            file_meta = {"title": entry.name, "parents": [{"id": root_item["id"]}]}
            guess = mimetypes.guess_type(entry)
            if guess[0]:
                file_meta["mimeType"] = guess[0]
            f = drive.CreateFile(file_meta)
            f.SetContentFile(str(entry.absolute()))
            f.Upload()
        else:
            assert entry.is_dir()
            _gdrive_upload_dir(
                drive, entry, remote_root_name, parent_id=root_item["id"]
            )

    return root_item["alternateLink"]


def _check_environment():
    # TODO: Check irqbalance
    # Check choice of governor
    if any(g != "performance" for g in _get_governor_settings()):
        raise Exception("Clock rate governor not set to 'performance'")
    # # TODO: Check CPU frequency range
    # if any(mi != ma for mi, ma in _get_clock_rates()):
    #     raise Exception("CPU clock rates should be fixed (min == max)")
    # TODO: Check that we're realtime


def _get_governor_settings() -> list[str]:
    govs = []
    for path in glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        with open(path, "r") as fo:
            govs.append(fo.read().strip())
    assert govs
    return govs


def _get_clock_rates() -> list[tuple[float, float]]:
    min_paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_min_freq")
    max_paths = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq")
    assert len(min_paths) == len(max_paths)
    results = []
    for min_path, max_path in zip(min_paths, max_paths):
        with open(min_path, "r") as fo:
            a = int(fo.read())
        with open(max_path, "r") as fo:
            b = int(fo.read())
        results.append((a, b))
    return results


def main():
    logging.basicConfig()

    args = parser.parse_args()

    if not args.ignore_environment:
        _check_environment()

    batch_sizes = args.batch
    if not batch_sizes:
        batch_sizes = [1]

    sheet = None
    if args.log_to_sheet:
        creds = service_account.Credentials.from_service_account_file(
            args.gsheet_key,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.Client(creds)
        sheet = gc.open(args.log_to_sheet).worksheet("Log")

    drive = None
    if args.save_to_gdrive:
        gauth = pydrive2.auth.GoogleAuth()
        gauth.auth_method = "service"
        gauth.credentials = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(
            args.gsheet_key, "https://www.googleapis.com/auth/drive"
        )
        drive = pydrive2.drive.GoogleDrive(gauth)

    system_config.set_current_target(args.target)

    # Disable sliding windows, since we can't use them with codegen yet.
    morello.impl.allow_sliding_windows.set(False)

    start_time = datetime.datetime.now()
    if args.hostname:
        hostname = args.hostname
    else:
        hostname = os.uname().nodename

    governor_ctx = contextlib.nullcontext()
    if args.configure_governor == True:
        governor_ctx = _scale_governor(args.configure_governor)

    with tempfile.TemporaryDirectory() as tdir, governor_ctx:
        for benchmark_cls in _get_benchmark_classes():
            for benchmark in benchmark_cls.from_args(args):
                print("Beginning", str(benchmark))
                work_dir_backend = (
                    pathlib.Path(tdir)
                    / start_time.strftime("%Y-%m-%d_%H:%M:%S")
                    / f"{hostname}-{benchmark.short_name}-{random.randint(0, 9999)}"
                )
                work_dir_backend.mkdir(parents=True, exist_ok=False)
                for short_name, backend_constructor in benchmark.backends(
                    args.cache,
                    args.save_cache,
                    args.redis,
                    args.load_sqlite_db,
                    args.sqlite_inmem_cache_kb,
                    work_dir_backend,
                ):
                    if args.backend and short_name not in args.backend:
                        print(f"Skipping backend named", short_name)
                        continue
                    print(f"Running {short_name}")
                    try:
                        backend = backend_constructor()
                        runtime_samples, cpu_affinity, nice, is_rt = backend.run(
                            args.trials
                        )
                        assert isinstance(runtime_samples, list)
                        runtime_secs = min(runtime_samples)
                    except NotImplementedError:
                        print(f"No implementation for {short_name}. Skipping.")
                        raise
                        continue
                    finally:
                        uploaded_url = ""
                        if args.save_to_gdrive:
                            assert drive is not None
                            uploaded_url = _gdrive_upload_dir(
                                drive, work_dir_backend, args.save_to_gdrive
                            )
                    print(f"{short_name} took {runtime_secs:.7f}s")
                    if args.log_to_sheet:
                        sheet.append_row(
                            [
                                str(start_time),
                                hostname,
                                benchmark.short_name,
                                benchmark.size,
                                benchmark.batch_size,
                                short_name,
                                runtime_secs,
                                benchmark.cpus_used,
                                ", ".join(f"{s:.8f}" for s in runtime_samples),
                                uploaded_url,
                                "amd-boost-off",
                                str(cpu_affinity),
                                nice,
                                str(is_rt),
                            ],
                            value_input_option="USER_ENTERED",
                        )


if __name__ == "__main__":
    main()
