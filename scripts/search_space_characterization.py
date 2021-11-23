#!/usr/bin/env python3
"""A script to dump a JSON file summarizing complete enumeration of Impl-spaces."""

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Iterable

import morello.impl.base
from morello import cost, dtypes, specs
from morello.search.naive import enumerate_impls
from morello.system_config.state import (
    current_target,
    set_current_target,
    target_by_name,
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("output", metavar="OUTPUT")


def main():
    args = arg_parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        print(f"{output_path} already exists. Exiting.", file=sys.stderr)
        return 1

    _setup_target()

    start = time.time()
    results = {}
    with multiprocessing.Pool() as pool:
        results.update(*pool.starmap(job, experiment_specs(), chunksize=1))
    print(f"Took {time.time() - start:.2f} seconds")

    with output_path.open("w") as fo:
        json.dump(results, fo)

    return 0


def job(spec_name: str, root_spec: specs.Spec) -> dict[str, list[int]]:
    target = _setup_target()

    inputs = tuple(target.tensor(i) for i in root_spec.inputs)
    output = target.tensor(root_spec.output)
    root_impl = morello.impl.base.spec_to_hole(root_spec, inputs, output)
    return {
        spec_name: [
            cost.analytical_cost(impl)
            for impl in enumerate_impls(root_impl, None, None)
        ]
    }


def experiment_specs() -> Iterable[tuple[str, specs.Spec]]:
    target = current_target()
    yield "gemm3", specs.Compose(
        (specs.Matmul, specs.Matmul),
        (
            target.tensor_spec((4, 4), dtype=dtypes.Uint8),
            target.tensor_spec((4, 2), dtype=dtypes.Uint8),
            target.tensor_spec((2, 4), dtype=dtypes.Uint8),
        ),
        target.tensor_spec((4, 4), dtype=dtypes.Uint8),
        intermediate_dtypes=(dtypes.Uint8,),
        serial_only=True,
    )
    yield "conv-8x8-5x5-2", specs.Convolution(
        target.tensor_spec((8, 8), dtype=dtypes.Uint8, bank="GL"),
        target.tensor_spec((5, 5, 2), dtype=dtypes.Uint8, bank="GL"),
        target.tensor_spec((4, 4, 2), dtype=dtypes.Uint8, bank="GL"),
        serial_only=True,
    )
    yield "matmul-4x4x4", specs.Matmul(
        target.tensor_spec((4, 4), dtype=dtypes.Uint8, bank="GL"),
        target.tensor_spec((4, 4), dtype=dtypes.Uint8, bank="GL"),
        target.tensor_spec((4, 4), dtype=dtypes.Uint8, bank="GL"),
        serial_only=True,
    )


def _setup_target():
    target = target_by_name("cpu")
    set_current_target(target)
    return target


if __name__ == "__main__":
    sys.exit(main())
