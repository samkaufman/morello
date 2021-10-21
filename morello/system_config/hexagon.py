import abc
import contextlib
import dataclasses
import functools
import io
import logging
import operator
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Union, cast

from .. import dtypes, ops, specs, tensor
from ..codegen import gen
from .base import MemoryBankConfig, RunResult, SystemDescription, Target

HEXAGON_CLANG_ARGS = ["-mhvx", "-mv66"]
HEXAGON_SIM_TARGET_ARG = "--mv66g_1024_rev2"
_REAL_TIME_RE = re.compile(
    r"\s*Ratio to Real Time \(\d+ MHz\) = ~\d+/\d+ \(elapsed time = ([\d.]+)s\)\s*"
)

# L1 cache size is drawn from Linux source:
#   https://docs.huihoo.com/doxygen/linux/kernel/3.7/include_2asm-generic_2cache_8h.html#a9400cc2ba37e33279bdbc510a6311fb4
L1_CACHE_LINE_BYTES = 32
# L2 cache size is a guess based on existing implementations and the description of
# L2FETCH in Qualcomm's HVX Programmer’s Reference Manual.
# TODO: Find a more definitive source.
L2_CACHE_LINE_BYTES = 8 * 1024

logger = logging.getLogger(__name__)


class HvxSimulatorTarget(Target):
    def __new__(cls, *args, **kwds):
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
        origin: Optional[Union[tensor.Tensor, tensor.Tile]] = None,
        **kwargs,
    ) -> tensor.TensorBase:
        if spec.bank == "VMEM":
            assert isinstance(spec, specs.HvxVmemTensorSpec)
            return HvxVmemTensor(spec=spec, name=name, origin=origin)
        return tensor.Tensor(spec=spec, name=name, origin=origin)

    def tensor_spec(
        self,
        dim_sizes: tuple[int, ...],
        dtype: dtypes.Dtype,
        bank: Optional[str] = None,
        layout: specs.Layout = specs.Layout.ROW_MAJOR,
        **kwargs,
    ) -> "TensorSpec":
        if bank == "VMEM":
            return specs.HvxVmemTensorSpec(dim_sizes, dtype, bank, layout, **kwargs)
        return specs.TensorSpec(dim_sizes, dtype, bank, layout)

    @property
    def system(self) -> "SystemDescription":
        # TODO: The `banks` description should model multiple "vector contexts"
        return SystemDescription(
            line_size=64,
            banks={
                "HexagonRF": MemoryBankConfig(cache_hit_cost=0, capacity=32 * 4),
                "VMEM": MemoryBankConfig(cache_hit_cost=0, capacity=32 * 1024),  # HVX
                "L1": MemoryBankConfig(cache_hit_cost=10, capacity=32 * 1024),
                "L2": MemoryBankConfig(cache_hit_cost=10, capacity=2048 * 1024),
                "GL": MemoryBankConfig(cache_hit_cost=10, capacity=sys.maxsize),
            },
            default_bank="GL",
            processors=2,
            has_hvx=True,
            faster_destination_banks=self._faster_destination_banks,
            next_general_bank=self._next_general_bank,
            ordered_banks=["HexagonRF", "VMEM", "L1", "L2", "GL"],
        )

    def _faster_destination_banks(self, source: str) -> set[str]:
        if source in ("HexagonRF", "VMEM"):
            return set()
        elif source == "L1":
            return {"HexagonRF"}
        elif source == "L2":
            return {"L1", "VMEM"}
        elif source == "GL":
            return {"L2"}
        raise ValueError("Unknown source: " + source)

    def _next_general_bank(self, source: str) -> Optional[str]:
        if source in ("HexagonRF", "VMEM"):
            return None
        elif source == "L1":
            return "HexagonRF"
        elif source == "L2":
            return "L1"
        elif source == "GL":
            return "L2"
        raise ValueError("No general next bank for " + source)

    def run_impl(
        self,
        impl,
        print_output=False,
        source_cb=None,
        values=None,
        profile_output: Optional[pathlib.Path] = None,
    ) -> RunResult:
        if profile_output:
            profile_output.mkdir(parents=True, exist_ok=True)

        sim_path = _hexagon_sdk_tools_root() / "Tools" / "bin" / "hexagon-sim"
        with _build_for_hexagon(
            impl, source_cb=source_cb, print_output=print_output, values=values
        ) as binary_path:
            hexagon_sim_cmd = [
                str(sim_path),
                "--timing",
                "--verbose",
                HEXAGON_SIM_TARGET_ARG,
                binary_path,
            ]
            if profile_output:
                hexagon_sim_cmd += [
                    "--packet_analyze=" + str(profile_output / "stats.json"),
                ]
            result = subprocess.run(
                hexagon_sim_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            stdout = result.stdout.decode("utf8")
            stderr = result.stderr.decode("utf8")

            if profile_output:
                profiler_path = (
                    _hexagon_sdk_tools_root() / "Tools" / "bin" / "hexagon-profiler"
                )
                profile_result = subprocess.run(
                    [
                        str(profiler_path),
                        "--packet_analyze",
                        "--json=" + str(profile_output / "stats.json"),
                        "--elf=" + binary_path,
                        "-o",
                        str(profile_output / "output0.html"),
                    ],
                    check=True,
                )

            return RunResult(stdout, stderr)

    def time_impl(
        self,
        impl,
        profile_output: Optional[pathlib.Path] = None,
    ) -> float:
        _, stderr = self.run_impl(impl, profile_output=profile_output)
        stderr_lines = stderr.splitlines()
        real_time_line_matches = [_REAL_TIME_RE.match(l) for l in stderr_lines]
        time_matches = [m for m in real_time_line_matches if m is not None]
        assert len(time_matches) == 1, f"Got {len(time_matches)} matches"
        return float(time_matches[0].group(1))


class HvxVmemTensorlike(tensor.TensorLike):
    def _common_post_init(self):
        assert isinstance(self.spec, specs.HvxVmemTensorSpec)
        assert self.spec.bank == "VMEM"
        if self.bytes_used % 128 != 0:
            raise tensor.DisallowedTileShapeError(
                f"Bytes {self.bytes_used} not divisible by 128"
            )

    @property
    def contiguous(self) -> bool:
        return self.vector_count == 1

    def is_valid_tile_shape(self, shape: tuple[int, ...]) -> bool:
        if len(shape) != len(self.dim_sizes):
            return False
        if any(i > o for (i, o) in zip(shape, self.dim_sizes)):
            return False
        if functools.reduce(operator.mul, shape, 1) % 128 != 0:
            return False
        return True

    def simple_tile(self, tile_shape: tuple[int, ...]) -> "HvxVmemTensorlike":
        return self._tile(HvxVmemSimpleTile, tile_shape)

    def conv_image_tile(
        self, tile_shape: tuple[int, ...], filter_shape: tuple[int, int]
    ) -> tensor.TensorLike:
        raise Exception("Convolution tiling of HVX vectors is not supported")

    @property
    def vector_count(self) -> int:
        return self.volume // functools.reduce(operator.mul, self.root.vector_shape, 1)

    @abc.abstractmethod
    def vector_indices(self, tile_pt: Sequence[int]) -> Sequence[int]:
        """Returns identifiers of the included root HVX vectors."""
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True, eq=False)
class HvxVmemTensor(HvxVmemTensorlike, tensor.TensorBase):
    spec: specs.HvxVmemTensorSpec
    name: Optional[str]
    origin: Optional[tensor.TensorLike] = None

    def __post_init__(self):
        self._common_post_init()
        if functools.reduce(operator.mul, self.vector_shape, self.dtype.size) != 128:
            raise ValueError("vector shape must use 128 bytes")

    @property
    def vector_shape(self) -> tuple[int, ...]:
        return self.spec.vector_shape

    @typing.final
    def vector_indices(self, tile_pt: Sequence[int]) -> Sequence[int]:
        """Returns identifiers of the included root HVX vectors."""
        return list(range(self.vector_count))

    def __str__(self):
        layout_epi = ""
        if self.layout != specs.Layout.ROW_MAJOR:
            layout_epi = f", {self.layout}"
        dims_part = "×".join(str(s) for s in self.dim_sizes)
        vec_part = "×".join(str(s) for s in self.vector_shape)
        return (
            f"{type(self).__name__}({dims_part}{layout_epi}, {self.bank}, {vec_part})"
        )

    def __getstate__(self):
        return {
            "spec": self.spec,
            "name": self.name,
            "origin": self.origin,
        }

    def __setstate__(self, state_dict):
        object.__setattr__(self, "spec", state_dict["spec"])
        object.__setattr__(self, "name", state_dict["name"])
        object.__setattr__(self, "origin", state_dict["origin"])


class HvxVmemSimpleTile(HvxVmemTensorlike, tensor.SimpleTile):
    def __post_init__(self):
        super().__post_init__()
        self._common_post_init()
        for size, vs in zip(self.dim_sizes, self.root.vector_shape):
            tile_str = "×".join(str(s) for s in self.dim_sizes)
            vec_str = "×".join(str(s) for s in self.dim_sizes)
            if size % vs != 0:
                raise ValueError(
                    f"Tile shape {tile_str} not a multiple of vector shape {vec_str}"
                )

    def vector_indices(self, tile_pt: Sequence[int]) -> Sequence[int]:
        raise NotImplementedError()

    @functools.cached_property
    def spec(self) -> specs.HvxVmemTensorSpec:
        orig = super().spec
        return specs.HvxVmemTensorSpec(
            dim_sizes=orig.dim_sizes,
            dtype=orig.dtype,
            bank=orig.bank,
            layout=orig.layout,
            vector_shape=cast(HvxVmemTensor, self.root).vector_shape,
        )

    def __getstate__(self):
        raise NotImplementedError()

    def __setstate__(self, state_dict):
        raise NotImplementedError()


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
                "--save-temps=obj",
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

        # with open(os.path.join(dirname, "main.s"), "r") as fo:
        #     print("")
        #     print(".S:")
        #     print("")
        #     print(fo.read())

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
