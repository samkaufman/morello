import os
import re
import subprocess
import tempfile

from .. import ops
from . import gen

_OUTPUT_RE = re.compile(r"cpu:\s+(\d+)s\s*(\d+)ns")


def time_impl(impl: ops.Schedule, nice: int = -10) -> float:
    """Builds and runs an Impl, returning the seconds taken."""

    with tempfile.TemporaryDirectory() as dirname:
        source_path = os.path.join(dirname, "main.c")
        binary_path = os.path.join(dirname, "a.out")
        with open(source_path, mode="w") as fo:
            gen.generate_c("benchmark", impl, fo)
        # TODO: Is there a more secure way to depend on clang?
        subprocess.run(
            [
                "/usr/bin/env",
                "nice",
                "-n",
                str(nice),
                "/usr/bin/env",
                "clang",
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
        binary_output = binary_result.stdout.decode("utf-8")

        re_match = _OUTPUT_RE.match(binary_output)
        assert re_match is not None
        return int(re_match.group(1)) + (int(re_match.group(2)) / 1e9)
