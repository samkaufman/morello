import asyncio
import functools
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

from .. import dtypes, layouts, specs
from ..codegen import gen
from ..layouts import Layout
from ..tensor import Tensor, Tile
from .base import MemoryBankConfig, RunResult, SystemDescription, Target

_OUTPUT_RE = re.compile(r"cpu:\s+(\d+)s\s*(\d+)ns")


class CpuTarget(Target):
    def __new__(cls, *args, **kwargs):
        # Singleton pattern. Constructor will return the first instance made.
        it = cls.__dict__.get("__one__")
        if it is not None:
            return it
        cls.__one__ = it = object.__new__(cls)
        return it

    def tensor(
        self,
        spec: specs.TensorSpec,
        name: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")
        return Tensor(spec=spec, name=name)

    def tensor_spec(
        self,
        dim_sizes: tuple[int, ...],
        dtype: dtypes.Dtype,
        bank: Optional[str] = None,
        layout: Optional[Layout] = None,
        **kwargs,
    ) -> specs.TensorSpec:
        if layout is None:
            layout = layouts.ROW_MAJOR
        return specs.TensorSpec(dim_sizes, dtype, bank, layout)

    @functools.cached_property
    def system(self) -> "SystemDescription":
        return SystemDescription(
            line_size=64,
            banks={
                # Selecting 4096 because that's a power of two. We normally
                # overapproximate the peak memory usage of an Impl to the next
                # power of two.
                "RF": MemoryBankConfig(cache_hit_cost=0, capacity=4096),
                "GL": MemoryBankConfig(cache_hit_cost=10, capacity=sys.maxsize),
            },
            default_bank="GL",
            processors=32,
            has_hvx=False,
            faster_destination_banks=self._faster_destination_banks,
            next_general_bank=self._next_general_bank,
            ordered_banks=["RF", "GL"],
        )

    @property
    def all_layouts(self) -> Iterable[Layout]:
        return [layouts.ROW_MAJOR, layouts.COL_MAJOR]

    def _faster_destination_banks(self, source: str) -> set[str]:
        assert isinstance(source, str)
        if source == "RF":
            return set()
        elif source == "GL":
            return {"RF"}
        raise ValueError("Unknown source: " + source)

    def _next_general_bank(self, source: str) -> Optional[str]:
        if source == "RF":
            return None
        elif source == "GL":
            return "RF"
        raise ValueError("Unknown source: " + source)

    async def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        values=None,
    ) -> RunResult:
        with tempfile.TemporaryDirectory() as dirname:
            source_path = os.path.join(dirname, "main.c")
            binary_path = os.path.join(dirname, "a.out")

            with io.StringIO() as source_io:
                if print_output:
                    gen.generate_c("print_output", impl, source_io, values=values)
                else:
                    gen.generate_c("benchmark", impl, source_io, values=values)
                source_code = source_io.getvalue()
            if source_cb:
                source_cb(source_code)
            with open(source_path, mode="w") as fo:
                fo.write(source_code)

            # TODO: Don't need to link OpenMP if the Impl has no parallel loops.
            clang_cmd = [
                _clang_path(),
                "-std=gnu99",
                "-fopenmp",
                "-O3",
                "-o",
                binary_path,
                source_path,
            ]
            clang_proc = await asyncio.create_subprocess_exec(
                *clang_cmd,
            )
            await clang_proc.wait()
            if clang_proc.returncode != 0:
                raise Exception(f"Clang exited with code {clang_proc.returncode}")

            # Run the compiled binary
            binary_proc = await asyncio.create_subprocess_exec(
                binary_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await binary_proc.communicate()
            if binary_proc.returncode != 0:
                raise Exception(f"Binary exited with code {binary_proc.returncode}")
            stdout = stdout.decode("utf8")
            stderr = stderr.decode("utf8")
            return RunResult(stdout, stderr)

    async def time_impl(self, impl) -> float:
        """Executes and benchmarks an Impl on the local machine using Clang.

        Returns the time in seconds. Measured by executing 10 times and
        returning the mean.
        """
        r = await self.run_impl(impl)
        return _parse_benchmark_output(r.stdout) / gen.BENCH_ITERS


def _parse_benchmark_output(output: str) -> float:
    re_match = _OUTPUT_RE.match(output)
    assert re_match is not None
    return int(re_match.group(1)) + (int(re_match.group(2)) / 1e9)


@functools.cache
def _clang_path() -> Path:
    if "CLANG" in os.environ:
        return Path(os.environ["CLANG"])
    raise Exception("Environment variable CLANG is not set")
