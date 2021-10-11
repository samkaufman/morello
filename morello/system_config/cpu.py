import io
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional, Union

from .base import MemoryBankConfig, RunResult, SystemDescription, Target
from .. import dtypes, specs
from ..codegen import gen
from ..tensor import Tensor, Tile

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
        name: Optional[str],
        origin: Optional[Union[Tensor, Tile]] = None,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            raise TypeError("Unexpected keyword argument(s)")
        return Tensor(spec=spec, name=name, origin=origin)

    def tensor_spec(
        self,
        dim_sizes: tuple[int, ...],
        dtype: dtypes.Dtype,
        bank: Optional[str] = None,
        layout: specs.Layout = specs.Layout.ROW_MAJOR,
        **kwargs,
    ) -> specs.TensorSpec:
        return specs.TensorSpec(dim_sizes, dtype, bank, layout)

    @property
    def system(self) -> "SystemDescription":
        return SystemDescription(
            line_size=64,
            banks={
                "RF": MemoryBankConfig(cache_hit_cost=0, capacity=6400),
                "GL": MemoryBankConfig(cache_hit_cost=10, capacity=sys.maxsize),
            },
            default_bank="GL",
            processors=4,
            has_hvx=False,
            faster_destination_banks=self._faster_destination_banks,
            next_general_bank=self._next_general_bank,
            ordered_banks=["RF", "GL"],
        )

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

    def run_impl(
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

            # TODO: Is there a more secure way to depend on clang?
            clang_cmd = [
                "/usr/bin/env",
                "clang",
                "-O3",
            ]
            clang_cmd += [
                "-o",
                binary_path,
                source_path,
            ]
            subprocess.run(clang_cmd, check=True)

            # Run the compiled binary
            binary_result = subprocess.run(
                [binary_path],
                capture_output=True,
                check=True,
            )
            stdout = binary_result.stdout.decode("utf8")
            stderr = binary_result.stderr.decode("utf8")
            return RunResult(stdout, stderr)

    def time_impl(self, impl) -> float:
        stdout, _ = self.run_impl(impl)
        return _parse_benchmark_output(stdout) / gen.BENCH_ITERS


def _parse_benchmark_output(output: str) -> float:
    re_match = _OUTPUT_RE.match(output)
    assert re_match is not None
    return int(re_match.group(1)) + (int(re_match.group(2)) / 1e9)
