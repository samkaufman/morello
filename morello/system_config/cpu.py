import io
import os
import re
import subprocess
import sys
import tempfile

from .base import (
    RunResult,
    Target,
    SystemDescription,
    MemoryLevelConfig,
)
from ..codegen import gen

_OUTPUT_RE = re.compile(r"cpu:\s+(\d+)s\s*(\d+)ns")


class CpuTarget(Target):
    def __new__(cls, *args, **kwds):
        # Singleton pattern. Constructor will return the first instance made.
        it = cls.__dict__.get("__one__")
        if it is not None:
            return it
        cls.__one__ = it = object.__new__(cls)
        return it

    @property
    def system(self) -> "SystemDescription":
        return SystemDescription(
            line_size=64,
            level_configs=[
                MemoryLevelConfig(cache_hit_cost=0, capacity=100),
                MemoryLevelConfig(cache_hit_cost=10, capacity=sys.maxsize),
            ],
            processors=4,
            has_hvx=False,
        )

    def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        extra_clang_args=None,
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
            if extra_clang_args:
                clang_cmd += extra_clang_args
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
