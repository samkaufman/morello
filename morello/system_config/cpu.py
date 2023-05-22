import asyncio
import functools
import io
import os
import pathlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from .. import dtypes, layouts, specs
from ..codegen import gen
from ..tensor import Tensor
from .base import BuiltArtifact, MemoryBankConfig, RunResult, SystemDescription, Target

_OUTPUT_RE = re.compile(r"cpu:\s+(\d+)s\s*(\d+)ns")


class _CpuTarget(Target):
    def __new__(cls, *args, **kwargs):
        # Singleton pattern. Constructor will return the first instance made.
        it = cls.__dict__.get("__one__")
        if it is not None:
            return it
        cls.__one__ = it = object.__new__(cls)
        return it

    def tensor(
        self, spec: specs.TensorSpec, name: Optional[str] = None, **kwargs
    ) -> Tensor:
        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")
        return Tensor(spec=spec, name=name)

    def tensor_spec(
        self,
        dim_sizes: tuple[int, ...],
        dtype: dtypes.Dtype,
        contiguous_abs=None,
        aligned: bool = True,
        bank: Optional[str] = None,
        layout: Optional[layouts.Layout] = None,
        vector_shape: Optional[tuple[int, ...]] = None,
        **kwargs,
    ) -> specs.TensorSpec:
        if layout is None:
            layout = layouts.row_major(len(dim_sizes))
        if contiguous_abs is None:
            contiguous_abs = layout.contiguous_full()
        return specs.TensorSpec(
            dim_sizes,
            dtype,
            contiguous_abs,
            aligned,
            bank,
            layout,
            vector_shape,
            **kwargs,
        )

    @functools.cached_property
    def system(self) -> "SystemDescription":
        return SystemDescription(
            line_size=32,
            banks={
                "RF": MemoryBankConfig(cache_hit_cost=0, capacity=64),
                "VRF": MemoryBankConfig(
                    cache_hit_cost=0, capacity=1024, vector_bytes=16  # 128-bit XMMs
                ),
                # 48*1024 would be more accurate for L1, but we want a power of two.
                "L1": MemoryBankConfig(cache_hit_cost=10, capacity=32_768),
                "GL": MemoryBankConfig(cache_hit_cost=100, capacity=1024**3),
            },
            default_bank="GL",
            processors=32,
            faster_destination_banks=self._faster_destination_banks,
            next_general_bank=self._next_general_bank,
            ordered_banks=("RF", "VRF", "L1", "GL"),
            addressed_banks=frozenset(["RF", "VRF", "GL"]),
        )

    def _faster_destination_banks(self, source: str) -> set[str]:
        assert isinstance(source, str)
        if source in ("RF", "VRF"):
            return set()
        elif source == "GL":
            return {"L1"}
        elif source == "L1":
            return {"RF", "VRF"}
        raise ValueError("Unknown source: " + str(source))

    def _next_general_bank(self, source: str) -> Optional[str]:
        if source in ("RF", "VRF"):
            return None
        elif source == "GL":
            return "L1"
        elif source == "L1":
            return "RF"
        raise ValueError("Unknown source: " + str(source))

    async def build_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        values=None,
        extra_clang_args: Optional[Iterable[str]] = None,
    ) -> "CPUBuiltArtifact":
        dirname = pathlib.Path(tempfile.mkdtemp())
        source_path = dirname / "main.c"
        binary_path = dirname / "a.out"

        with io.StringIO() as source_io:
            if print_output:
                gen.generate_c("print_output", impl, source_io, values=values)
            else:
                gen.generate_c("benchmark", impl, source_io, values=values)
            source_code = source_io.getvalue()
        if source_cb:
            source_cb(source_code)
        with source_path.open(mode="w") as fo:
            fo.write(source_code)

        # TODO: Don't need to link OpenMP if the Impl has no parallel loops.
        extra_clang_args = extra_clang_args or []
        clang_cmd = (
            [str(_clang_path())]
            + list(extra_clang_args)
            + list(self._clang_vec_flags())
            + [
                "-std=gnu99",
                "-O3",
                "-o",
                str(binary_path),
                str(source_path),
            ]
        )
        if os.getenv("MORELLO_CLANG_LINK_RT"):
            clang_cmd.append("-lrt")
        clang_proc = await asyncio.create_subprocess_exec(*clang_cmd)
        await clang_proc.wait()
        if clang_proc.returncode != 0:
            raise Exception(f"Clang exited with code {clang_proc.returncode}")

        return CPUBuiltArtifact(binary_path, source_path, dirname)

    async def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        values=None,
        check_flakiness: int = 1,
        extra_clang_args: Optional[Iterable[str]] = None,
    ) -> RunResult:
        artifact = await self.build_impl(
            impl,
            print_output=print_output,
            source_cb=source_cb,
            values=values,
            extra_clang_args=extra_clang_args,
        )
        return await artifact.run(check_flakiness=check_flakiness)

    async def time_impl(
        self, impl, return_source=False
    ) -> Union[float, tuple[float, str]]:
        """Executes and benchmarks an Impl on the local machine using Clang.

        Returns the time in seconds. Measured by executing BENCH_ITERS times and
        returning the mean.
        """
        artifact = await self.build_impl(impl)
        t = await artifact.measure_time()
        if return_source:
            return (t, artifact.source_code)
        return t

    def _clang_vec_flags(self) -> Sequence[str]:
        raise NotImplementedError()


class CPUBuiltArtifact(BuiltArtifact):
    def __init__(
        self,
        binary_path: pathlib.Path,
        source_path: pathlib.Path,
        whole_dir: pathlib.Path,
    ):
        self.binary_path = binary_path
        self.whole_dir = whole_dir
        with source_path.open(mode="r") as fo:
            self.source_code = fo.read()

    async def run(self, check_flakiness: int = 1) -> RunResult:
        # Run the compiled binary
        last_stdout = None
        for it in range(check_flakiness):
            binary_proc = await asyncio.create_subprocess_exec(
                self.binary_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await binary_proc.communicate()

            if binary_proc.returncode != 0:
                raise Exception(
                    f"Binary exited with code {binary_proc.returncode}. "
                    f"Standard error: {stderr}"
                )
            stdout = stdout.decode("utf8")
            stderr = stderr.decode("utf8")

            if last_stdout is not None:
                assert stdout == last_stdout, (
                    f"On iteration {it}, received inconsistent stdouts:"
                    f"\n\n{stdout}\n\n{last_stdout}"
                )
            last_stdout = stdout

        return RunResult(stdout, stderr)  # type: ignore

    async def measure_time(self) -> float:
        """Executes and benchmarks an Impl on the local machine using Clang.

        Returns the time in seconds. Measured by executing BENCH_ITERS times and
        returning the mean.
        """
        r = await self.run()
        return _parse_benchmark_output(r.stdout) / gen.BENCH_ITERS

    def delete(self):
        shutil.rmtree(self.whole_dir)


class X86Target(_CpuTarget):
    def _clang_vec_flags(self) -> Sequence[str]:
        return ["-fopenmp", "-mavx2"]


class ArmTarget(_CpuTarget):
    def _clang_vec_flags(self) -> Sequence[str]:
        return ["-fopenmp"]


def _parse_benchmark_output(output: str) -> float:
    re_match = _OUTPUT_RE.match(output)
    assert re_match is not None
    return int(re_match.group(1)) + (int(re_match.group(2)) / 1e9)


@functools.cache
def _clang_path() -> Path:
    if "CLANG" in os.environ:
        return Path(os.environ["CLANG"])
    raise Exception("Environment variable CLANG is not set")
