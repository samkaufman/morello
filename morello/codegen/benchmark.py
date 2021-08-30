import contextlib
import functools
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import ops
from . import gen

_OUTPUT_RE = re.compile(r"cpu:\s+(\d+)s\s*(\d+)ns")
_REAL_TIME_RE = re.compile(
    r"\s*Ratio to Real Time \(\d+ MHz\) = ~\d+/\d+ \(elapsed time = ([\d.]+)s\)\s*"
)

logger = logging.getLogger(__name__)


def _parse_benchmark_output(output: str) -> float:
    re_match = _OUTPUT_RE.match(output)
    assert re_match is not None
    return int(re_match.group(1)) + (int(re_match.group(2)) / 1e9)


def build_and_run_locally(impl: ops.Schedule) -> float:
    with tempfile.TemporaryDirectory() as dirname:
        source_path = os.path.join(dirname, "main.c")
        binary_path = os.path.join(dirname, "a.out")
        with open(source_path, mode="w") as fo:
            gen.generate_c("benchmark", impl, fo)
        # TODO: Is there a more secure way to depend on clang?
        subprocess.run(
            [
                "/usr/bin/env",
                "clang",
                "-O3",
                "-o",
                binary_path,
                source_path,
            ],
            check=True,
        )
        binary_result = subprocess.run(
            [binary_path],
            stdout=subprocess.PIPE,
            check=True,
        )
        binary_result = binary_result.stdout.decode("utf8")
        return _parse_benchmark_output(binary_result) / gen.BENCH_ITERS


@contextlib.contextmanager
def _build_for_hexagon(impl: ops.Schedule):
    clang_path = _hexagon_sdk_tools_root() / "Tools" / "bin" / "hexagon-clang"
    if not clang_path.is_file():
        raise Exception(f"{clang_path} is not a file")

    with tempfile.TemporaryDirectory() as dirname:
        # Build with the Hexagon SDK's Clang
        source_path = os.path.join(dirname, "main.c")
        binary_path = os.path.join(dirname, "a.out")
        with open(source_path, mode="w") as fo:
            gen.generate_c("kernel_only", impl, fo)
        cmd = [
            "/usr/bin/env",
            str(clang_path),
            "-O3",
            "-D__USE_ISOC11",
            "-std=c11",
            "-o",
            binary_path,
            source_path,
        ]
        logger.debug("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(4)
        result.check_returncode()

        yield binary_path


@functools.cache
def _hexagon_sdk_tools_root():
    if "HEXAGON_TOOLS_ROOT" in os.environ:
        tools_root = Path(os.environ["HEXAGON_TOOLS_ROOT"])
    elif "DEFAULT_HEXAGON_TOOLS_ROOT" in os.environ:
        tools_root = Path(os.environ["DEFAULT_HEXAGON_TOOLS_ROOT"])
    else:
        raise Exception("Cannot find Hexagon tools root")
    return tools_root


def build_and_run_on_hexagon_sim(impl: ops.Schedule) -> float:
    sim_path = _hexagon_sdk_tools_root() / "Tools" / "bin" / "hexagon-sim"
    with _build_for_hexagon(impl) as binary_path:
        hexagon_sim_cmd = [
            str(sim_path),
            "--verbose",
            "--mv67g_1024",
            "--profile",
            binary_path,
        ]
        start = time.time()
        result = subprocess.run(
            hexagon_sim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        logger.debug("hexagon-sim took: " + str(time.time() - start))
        resulting_lines = result.stderr.decode("utf8").splitlines()
        real_time_line_matches = [_REAL_TIME_RE.match(l) for l in resulting_lines]
        time_matches = [m for m in real_time_line_matches if m is not None]
        assert len(time_matches) == 1, f"Got {len(time_matches)} matches"
        return float(time_matches[0].group(1))


def time_impl(impl: ops.Schedule, target_fn=build_and_run_locally) -> float:
    """Builds and runs an Impl, returning the seconds taken."""
    try:
        runtime_secs = target_fn(impl)
    except subprocess.CalledProcessError as e:
        print("CalledProcessError occurred.", file=sys.stderr)
        print("stdout was:\n" + str(e.stdout), file=sys.stderr)
        print("stderr was:\n" + str(e.stderr), file=sys.stderr)
        raise
    logger.info(f"Sample runtime result {runtime_secs}s:")
    return runtime_secs
