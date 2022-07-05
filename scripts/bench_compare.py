#!/usr/bin/env python3

import argparse
import asyncio
import contextlib
import dataclasses
import datetime
import functools
from glob import glob
import itertools
import logging
import mimetypes
import multiprocessing
import os
import pathlib
import random
import re
import runpy
import tempfile
import time
import typing
from typing import Any, Iterable, Optional, Sequence, Union

import gspread
import halide as hl
import jax
import jax.lib
import numpy as np
import oauth2client.service_account
import pydrive2
import pydrive2.auth
import pydrive2.drive
import termcolor
import torch
import torch.nn.functional as F
import tvm
import tvm.te
import tvm.auto_scheduler
import tvm.contrib.graph_executor
from google.oauth2 import service_account
from jax import lax
from jax.tools import jax_to_ir
from tvm import relay

import morello.impl.actions
import morello.impl.base
from morello import (
    cost,
    codegen,
    dtypes,
    layouts,
    op_pprint,
    search,
    search_cache,
    specs,
    system_config,
)
from morello.benchmarks.toy_cnn import halide as toyhl

RUNS = 100
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
parser.add_argument("--backend", type=str, nargs="+", default=None)
parser.add_argument("--log-to-sheet", type=str, default=None)
parser.add_argument("--save-to-gdrive", type=str, default=None)
parser.add_argument(
    "--gsheet-key",
    type=pathlib.Path,
    default=None,
    help="A path to a file containing a Google Sheets key",
)
parser.add_argument("--hostname", type=str, default=None)
parser.add_argument("--configure-governor", nargs="?", type=pathlib.Path, const=True, default=None)
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


class Benchmark:
    @property
    def spec(self) -> specs.Spec:
        raise NotImplementedError()

    def make_backends(self, cache: Union[str, pathlib.Path, None], save_cache: bool, extras_dir: pathlib.Path) -> Iterable["BenchmarkBackend"]:
        try:
            yield self._numpy_backend()
        except NotImplementedError as e:
            print(f"Not yielding Numpy backend: {e}")

        try:
            torch_backend = self._torch_backend()
        except NotImplementedError as e:
            print(f"Not yielding PyTorch backend: {e}")
        else:
            torchscript_backend = TorchScriptBackend(torch_backend)
            relay_backend = RelayBackend(torchscript_backend, extras_dir)
            # Drop the torch_backend itself. Nearly identicaly runtimes to
            # TorchScript for x86.
            yield from [torchscript_backend, relay_backend]

        try:
            yield self._jax_backend()
        except NotImplementedError as e:
            print(f"Not yielding JAX backend: {e}")

        try:
            yield self._halide_backend()
        except NotImplementedError as e:
            print(f"Not yielding Halide backend: {e}")
    
        yield MorelloBackend(self, cache, save_cache, extras_dir)
    
    @property
    def cpus_used(self) -> int:
        if self.spec.serial_only:
            return 1
        return multiprocessing.cpu_count()

    @property
    def short_name(self) -> str:
        raise NotImplementedError(f"Not implemented for {type(self).__name__}")

    def _numpy_backend(self) -> "BenchmarkBackend":
        raise NotImplementedError()

    def _torch_backend(self) -> "BenchmarkBackend":
        raise NotImplementedError()

    def _jax_backend(self) -> "BenchmarkBackend":
        raise NotImplementedError()

    def _halide_backend(self) -> "BenchmarkBackend":
        raise NotImplementedError()


class BenchmarkBackend:
    @property
    def short_name(self) -> str:
        raise NotImplementedError(f"Not implemented for {type(self).__name__}")

    def run(self) -> list[float]:
        raise NotImplementedError()


class MorelloBackend(BenchmarkBackend):
    def __init__(self, benchmark: Benchmark, cache: Union[str, pathlib.Path, None], save_cache: bool, extras_dir: pathlib.Path):
        if not extras_dir.is_dir():
            raise ValueError("extras_dir should be an existing directory")
        self.benchmark = benchmark
        self.cache_path = cache
        self.save_cache = save_cache
        self.extras_dir = extras_dir

    def run(self) -> list[float]:
        spec = self.benchmark.spec
        with search_cache.persistent_cache(self.cache_path, save=self.save_cache) as cache:
            impl = search.schedule_search(spec, cache=cache)
        assert impl is not None
        runtime_samples, impl_str, source_code = _benchmark(impl)
        assert runtime_samples
        with (self.extras_dir / "impl.txt").open("w") as fo:
            fo.write(impl_str)
        with (self.extras_dir / "source.c").open("w") as fo:
            fo.write(source_code)
        return runtime_samples
    
    @property
    def short_name(self) -> str:
        if cost.INST_COST != 1000:
            return f"morello-arith{cost.INST_COST}"
        return "morello"


class BaselineBackend(BenchmarkBackend):
    @property
    def codelet(self) -> str:
        raise NotImplementedError()

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        raise NotImplementedError()

    def run(self) -> list[float]:
        benchmark_code = f"""
global inner_benchmark

def inner_benchmark(self, {', '.join([n for n, _ in self.data_deps])}):
    {self.codelet}

    start = time.time()
    for _ in range(codegen.gen.BENCH_ITERS):
        {self.codelet}
    end = time.time()
    return (end - start) / codegen.gen.BENCH_ITERS
        """

        exec(benchmark_code)
        runs = []
        for _ in range(RUNS):
            runs.append(inner_benchmark(self, *[v for _, v in self.data_deps]))  # type: ignore
        return runs



class BaseJAXBackend(BaselineBackend):
    def __init__(self, benchmark: Benchmark, print_graphs=True) -> None:
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

    def make_jax_func(self):
        raise NotImplementedError()

    def set_inputs(self):
        raise NotImplementedError()
    
    def run(self) -> list[float]:
        if not self.benchmark.serial_only:
            return super().run()
        with _affinity_ctx(1):
            return super().run()

    @property
    def short_name(self) -> str:
        return "jax"

    @property
    def codelet(self) -> str:
        return f"self.jitted_fn({', '.join(n for n, _ in self.data_deps)}).block_until_ready()"


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

    def _numpy_backend(self) -> "BenchmarkBackend":
        return MatmulNumpy(self)

    def _torch_backend(self) -> "BenchmarkBackend":
        return MatmulTorch(self)
    
    def _jax_backend(self) -> "BenchmarkBackend":
        return MatmulJAX(self)
    
    def _halide_backend(self) -> "BenchmarkBackend":
        return MatmulHalide(self)


class MatmulNumpy(BaselineBackend):
    def __init__(self, benchmark: MatmulBenchmark) -> None:
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
        numpy_backend = MatmulNumpy(self.benchmark)
        self.lhs = jax.numpy.array(numpy_backend.lhs)
        self.rhs = jax.numpy.array(numpy_backend.rhs)

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("lhs", self.lhs), ("rhs", self.rhs)]


class BaseTorchBackend(BaselineBackend):
    def run(self) -> list[float]:
        # TODO: Save to file
        termcolor.cprint("PyTorch Configuration:", attrs=["bold"])
        print(torch.__config__.show())

        orig_intraop_threads = torch.get_num_threads()
        torch.set_num_threads(self.expected_threads())
        try:
            return super().run()
        finally:
            torch.set_num_threads(orig_intraop_threads)
    
    def expected_threads(self) -> int:
        if self.benchmark.spec.serial_only:
            return 1
        return multiprocessing.cpu_count()


class MatmulTorch(BaseTorchBackend):
    def __init__(self, benchmark: MatmulBenchmark) -> None:
        self.benchmark = benchmark
        numpy_backend = MatmulNumpy(benchmark)
        self.lhs = _to_torch(numpy_backend.lhs)
        self.rhs = _to_torch(numpy_backend.rhs)

    @property
    def short_name(self) -> str:
        return "pytorch"

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("lhs", self.lhs), ("rhs", self.rhs)]

    @property
    def codelet(self) -> str:
        return """torch.matmul(lhs, rhs)"""


class TorchScriptBackend(BaseTorchBackend):
    def __init__(self, torch_backend: BenchmarkBackend, print_graphs=True, trace=True):
        self.torch_backend = torch_backend
        self.print_graphs = print_graphs
        self._jitted_fn = None

        if trace:
            self.jitted_fn = self._jit_with_trace()
        else:
            self.jitted_fn = self._jit_with_script()
    
    def _jit_with_trace(self):
        codelet = f"""
import torch
import torch.nn.functional as F

global jittable

def jittable({', '.join([n for n, _ in self.torch_backend.data_deps])}):
    return {self.torch_backend.codelet}
        """
        exec(codelet)
        return torch.jit.trace(jittable, tuple(v for _, v in self.data_deps))  # type: ignore

    def _jit_with_script(self):
        # We'll save the following to a file. TorchScript needs the actual .py
        # to compile, th torch.jit.script, so we can't just exec it.
        jitted_codelet = f"""
import torch
import torch.nn.functional as F

@torch.jit.script
def jitted({', '.join([n for n, _ in self.torch_backend.data_deps])}):
    return {self.torch_backend.codelet}
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fo:
            fo.write(jitted_codelet)
        ran = runpy.run_path(fo.name)
        return ran["jitted"]
    
    @property
    def benchmark(self):
        return self.torch_backend.benchmark

    @property
    def short_name(self) -> str:
        return "torchscript"

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return self.torch_backend.data_deps

    @property
    def codelet(self) -> str:
        return f"torch.jit.wait(torch.jit.fork(self.jitted_fn, {', '.join([n for n, _ in self.data_deps])}))"

    def run(self) -> list[float]:
        result = super().run()
        if self.print_graphs:
            print("")
            termcolor.cprint("Torch:", attrs=["bold"])
            print(torch.jit.last_executed_optimized_graph())
        return result


class RelayBackend(BaselineBackend):
    def __init__(
        self, torchscript_backend: TorchScriptBackend, extras_dir: pathlib.Path
    ):
        self.torchscript_backend = torchscript_backend
        self.extras_dir = extras_dir

        # PyTorch graph on Relay/TVM
        shape_list = [(n, v.shape) for n, v in torchscript_backend.data_deps]
        tvm_target = tvm.target.Target("llvm")
        relay_mod, relay_params = relay.frontend.from_pytorch(
            torchscript_backend.jitted_fn, shape_list
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
        for n, v in torchscript_backend.data_deps:
            self.tvm_m.set_input(n, v)
        
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
    def short_name(self) -> str:
        return "relay"

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

    def _numpy_backend(self) -> "BenchmarkBackend":
        return GEMM3Numpy(self)
    
    def _jax_backend(self) -> "BenchmarkBackend":
        return GEMM3JAX(self)

    def _torch_backend(self) -> "BenchmarkBackend":
        return GEMM3Torch(self)
    
    def _halide_backend(self) -> "BenchmarkBackend":
        return GEMM3Halide(self)


class GEMM3Numpy(BaselineBackend):
    def __init__(self, benchmark: GEMM3Benchmark) -> None:
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
    def __init__(self, benchmark: GEMM3Benchmark) -> None:
        self.benchmark = benchmark
        numpy_backend = GEMM3Numpy(benchmark)
        self.a = _to_torch(numpy_backend.a)
        self.b = _to_torch(numpy_backend.b)
        self.c = _to_torch(numpy_backend.c)

    @property
    def short_name(self) -> str:
        return "pytorch"

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
            target.tensor_spec((self.batch_size, img_channels, self.size, self.size), dtype=DTYPE),
            target.tensor_spec((fc, img_channels, fh, fw), dtype=DTYPE),
            output=target.tensor_spec((self.batch_size, fc, out_h, out_w), dtype=DTYPE),
            serial_only=self.serial_only,
        )
    
    @property
    def short_name(self) -> str:
        return "conv"

    def _numpy_backend(self) -> "BenchmarkBackend":
        return ConvNumpy(self)

    def _torch_backend(self) -> "BenchmarkBackend":
        return ConvTorch(self)
    
    def _jax_backend(self) -> "BenchmarkBackend":
        return ConvJAX(self)


class ConvNumpy(BaselineBackend):
    def __init__(self, benchmark: ConvBenchmark) -> None:
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
    def __init__(self, benchmark: ConvBenchmark) -> None:
        self.benchmark = benchmark
        numpy_backend = ConvNumpy(benchmark)
        self.img = _to_torch(numpy_backend.img)
        self.filters = _to_torch(numpy_backend.filters)

    @property
    def short_name(self) -> str:
        return "pytorch"

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
        numpy_backend = ConvNumpy(self.benchmark)
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

    def _torch_backend(self) -> "BenchmarkBackend":
        return CNNTorch(self)

    def _jax_backend(self) -> "BenchmarkBackend":
        return CNNJAX(self)

    def _halide_backend(self) -> "BencharkBackend":
        return CNNHalide(self)


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
            (32, 4, 3, 3), dtype=DTYPE, layout=layouts.NCHWc4,
        )
        filters_b = target.tensor_spec(
            (32, 32, 3, 3), dtype=DTYPE, layout=layouts.NCHWc4,
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
    
    def make_backends(self, cache: Union[str, pathlib.Path, None], save_cache: bool, extras_dir: pathlib.Path) -> Iterable["BenchmarkBackend"]:
        # TODO: Yield the Relay backend
        # yield RelayCNNNCHWcBackend(extras_dir)
        yield MorelloBackend(self, cache, save_cache, extras_dir)


class RelayCNNNCHWcBackend(BaselineBackend):
    def __init__(self, extras_dir: pathlib.Path):
        self.extras_dir = extras_dir

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
    def short_name(self) -> str:
        return "relay"

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
    def __init__(self, benchmark: CNNBenchmark) -> None:
        self.benchmark = benchmark
        self.img, self.filters_a, self.filters_b = map(
            _to_torch, _make_np_cnn_inputs(self.benchmark.spec)
        )

    @property
    def data_deps(self) -> Sequence[tuple[str, Any]]:
        return [("img", self.img), ("fa", self.filters_a), ("fb", self.filters_b)]
    
    @property
    def short_name(self) -> str:
        return "pytorch"

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


class BaseHalideBackend(BaselineBackend):
    def __init__(self, benchmark: Benchmark, print_graphs=True):
        self.benchmark = benchmark

        fn = self._make_pipeline()
        machine = hl.MachineParams((1 if benchmark.serial_only else multiprocessing.cpu_count()), 0, 0);
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
    def short_name(self) -> str:
        return "halide"

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
        mmnp = MatmulNumpy(self.benchmark)
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


def _benchmark(impl):
    loop = asyncio.get_event_loop()
    assert impl.is_scheduled
    runtime_samples = []
    for _ in range(RUNS):
        secs, source = loop.run_until_complete(system_config.current_target().time_impl(impl, return_source=True))
        logger.info(f"Sample runtime result {secs}s:")
        runtime_samples.append(secs)
    impl_str = op_pprint.pformat(impl, color=False)
    return runtime_samples, impl_str, source


def _get_benchmark_classes():
    return [MatmulBenchmark, GEMM3Benchmark, ConvBenchmark, CNNBenchmark,
            CNNHCHWcBenchmark]


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


def _gdrive_upload_dir(drive: pydrive2.drive.GoogleDrive, local_dir: pathlib.Path, remote_root_name: str, parent_id: Optional[str] = None):
    assert local_dir.is_dir()

    # Get root folder in Drive based on provided name
    if not parent_id:
        remote_root_candidates = drive.ListFile({'q': f"title = '{remote_root_name}' and trashed = False"}).GetList()
        if not remote_root_candidates:
            raise ValueError(f"Found no folders with title '{remote_root_name}'")
        if len(remote_root_candidates) > 1:
            raise ValueError(f"Found multiple folders with title '{remote_root_name}'")
        parent_id = remote_root_candidates[0]["id"]

    # Create new remote subdirectory corresponding to the top of local_dir
    root_meta: dict[str, Any] = {
        "title": local_dir.name, "mimeType": 'application/vnd.google-apps.folder'
    }
    root_meta["parents"] = [{"id": parent_id}]
    root_item = drive.CreateFile(root_meta)
    root_item.Upload()

    for entry in local_dir.iterdir():
        if entry.is_file():
            # TODO: Upload file
            file_meta ={"title": entry.name,
                "parents": [{"id": root_item["id"]}]}
            guess = mimetypes.guess_type(entry)
            if guess[0]:
                file_meta["mimeType"] = guess[0]
            f = drive.CreateFile(file_meta)
            f.SetContentFile(str(entry.absolute()))
            f.Upload()
        else:
            assert entry.is_dir()
            _gdrive_upload_dir(drive, entry, remote_root_name, parent_id=root_item["id"])
    
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

    _check_environment()

    batch_sizes = args.batch
    if not batch_sizes:
        batch_sizes = [1]
    
    sheet = None
    if args.log_to_sheet:
        creds = service_account.Credentials.from_service_account_file(args.gsheet_key, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ])
        gc = gspread.Client(creds)
        sheet = gc.open(args.log_to_sheet).worksheet("Log")
    
    drive = None
    if args.save_to_gdrive:
        gauth = pydrive2.auth.GoogleAuth()
        gauth.auth_method = 'service'
        gauth.credentials = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(args.gsheet_key, "https://www.googleapis.com/auth/drive")
        drive = pydrive2.drive.GoogleDrive(gauth)

    system_config.set_current_target(args.target)

    # Disable sliding windows, since we can't use them with codegen yet.
    morello.impl.allow_sliding_windows.set(False)

    start_time = str(datetime.datetime.now())
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
                work_dir_backend = pathlib.Path(tdir) / f"{start_time}-{hostname}-{benchmark.short_name}-{random.randint(0, 9999)}"
                work_dir_backend.mkdir(parents=True, exist_ok=False)
                for backend in benchmark.make_backends(args.cache, args.save_cache, work_dir_backend):
                    if args.backend and backend.short_name not in args.backend:
                        print(f"Skipping backend named", backend.short_name)
                        continue
                    print(f"Running {backend.short_name}")
                    try:
                        runtime_samples = backend.run()
                        assert isinstance(runtime_samples, list)
                        runtime_secs = min(runtime_samples)
                    except NotImplementedError:
                        print(f"No implementation for {backend}. Skipping.")
                        continue
                    finally:
                        uploaded_url = ""
                        if args.save_to_gdrive:
                            assert drive is not None
                            uploaded_url = _gdrive_upload_dir(drive, work_dir_backend, args.save_to_gdrive)
                    print(f"{backend.short_name} took {runtime_secs:.7f}s")
                    if args.log_to_sheet:
                        sheet.append_row(
                            [
                                start_time,
                                hostname,
                                benchmark.short_name,
                                benchmark.size,
                                benchmark.batch_size,
                                backend.short_name,
                                runtime_secs,
                                benchmark.cpus_used,
                                ", ".join(f"{s:.8f}" for s in runtime_samples),
                                uploaded_url,
                                "improved",
                            ],
                            value_input_option="USER_ENTERED",
                        )


if __name__ == "__main__":
    main()
