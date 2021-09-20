import contextlib
import dataclasses
import functools
import io
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from . import cpu
from .base import Target, RunResult, SystemDescription
from .. import ops
from ..codegen import gen

HEXAGON_CLANG_ARGS = ["-mhvx", "-mv66"]
HEXAGON_SIM_TARGET_ARG = "--mv66g_1024_rev2"
_REAL_TIME_RE = re.compile(
    r"\s*Ratio to Real Time \(\d+ MHz\) = ~\d+/\d+ \(elapsed time = ([\d.]+)s\)\s*"
)

logger = logging.getLogger(__name__)


class HvxSimulatorTarget(Target):
    def __new__(cls, *args, **kwds):
        # Singleton pattern. Constructor will return the first instance made.
        it = cls.__dict__.get("__one__")
        if it is not None:
            return it
        cls.__one__ = it = object.__new__(cls)
        return it

    @property
    def system(self) -> "SystemDescription":
        # TODO: Implement a SystemDescription more specific to HVX
        return dataclasses.replace(cpu.CpuTarget().system, has_hvx=True)

    def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        extra_clang_args=None,
        values=None,
    ) -> RunResult:
        sim_path = _hexagon_sdk_tools_root() / "Tools" / "bin" / "hexagon-sim"
        with _build_for_hexagon(
            impl, source_cb=source_cb, print_output=print_output, values=values
        ) as binary_path:
            hexagon_sim_cmd = [
                str(sim_path),
                "--verbose",
                HEXAGON_SIM_TARGET_ARG,
                "--profile",
                binary_path,
            ]
            if extra_clang_args:
                hexagon_sim_cmd += extra_clang_args
            result = subprocess.run(
                hexagon_sim_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            stdout = result.stdout.decode("utf8")
            stderr = result.stderr.decode("utf8")
            return RunResult(stdout, stderr)

    def time_impl(self, impl) -> float:
        _, stderr = self.run_impl(impl)
        stderr_lines = stderr.splitlines()
        real_time_line_matches = [_REAL_TIME_RE.match(l) for l in stderr_lines]
        time_matches = [m for m in real_time_line_matches if m is not None]
        assert len(time_matches) == 1, f"Got {len(time_matches)} matches"
        return float(time_matches[0].group(1))


@contextlib.contextmanager
def _build_for_hexagon(
    impl: ops.Schedule, *, source_cb: Callable[[str], None], print_output: bool, values
):
    clang_path = _hexagon_sdk_tools_root() / "Tools" / "bin" / "hexagon-clang"
    if not clang_path.is_file():
        raise Exception(f"{clang_path} is not a file")

    with tempfile.TemporaryDirectory() as dirname:
        # Build with the Hexagon SDK's Clang
        source_path = os.path.join(dirname, "main.c")
        binary_path = os.path.join(dirname, "a.out")

        with io.StringIO() as source_io:
            if print_output:
                gen.generate_c("print_output", impl, source_io, values=values)
            else:
                gen.generate_c("kernel_only", impl, source_io, values=values)
            source_code = source_io.getvalue()
        if source_cb:
            source_cb(source_code)
        with open(source_path, mode="w") as fo:
            fo.write(source_code)

        cmd = (
            ["/usr/bin/env", str(clang_path)]
            + HEXAGON_CLANG_ARGS
            + [
                "-O3",
                "-std=gnu99",
                "-I",
                str(_hexagon_nn_root() / "hexagon" / "include"),
                "-o",
                binary_path,
                source_path,
            ]
        )
        logger.debug("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            result.check_returncode()
        except subprocess.CalledProcessError:
            logger.error(
                "Hexagon Clang failed. stderr was:\n" + result.stderr.decode("utf8")
            )
            raise
        yield binary_path


@functools.cache
def _hexagon_sdk_root() -> Path:
    if "HEXAGON_SDK_ROOT" in os.environ:
        return Path(os.environ["HEXAGON_SDK_ROOT"])
    else:
        raise Exception("Cannot find Hexagon SDK path")


@functools.cache
def _hexagon_sdk_tools_root():
    if "HEXAGON_TOOLS_ROOT" in os.environ:
        tools_root = Path(os.environ["HEXAGON_TOOLS_ROOT"])
    elif "DEFAULT_HEXAGON_TOOLS_ROOT" in os.environ:
        tools_root = Path(os.environ["DEFAULT_HEXAGON_TOOLS_ROOT"])
    else:
        raise Exception("Cannot find Hexagon tools root")
    return tools_root


def _hexagon_nn_root() -> Path:
    return _hexagon_sdk_root() / "libs" / "hexagon_nn" / "2.10.1"
