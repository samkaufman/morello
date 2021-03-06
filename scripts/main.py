#!/usr/bin/env python3
import argparse
import asyncio
import csv
import doctest
import logging
import os
import pathlib
import sys
import time
from typing import Optional, TypeVar, Union

import morello.impl.base
from morello import (
    cost,
    dtypes,
    impl,
    op_pprint,
    search,
    search_cache,
    specs,
    system_config,
)
from morello.codegen import gen
from morello.impl import TileSizeMode
from morello.search import schedule_search

T = TypeVar("T")

DTYPE = dtypes.Uint32

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=pathlib.Path, default=None)
parser.add_argument("--target", type=str, default="cpu")
parser.add_argument("--cache", type=str)
parser.add_argument("--top", type=int, default=1)
parser.add_argument("--no-save-cache", action="store_false", dest="save_cache")
parser.add_argument("--row-major-only", action="store_true", dest="row_major_only")
parser.add_argument(
    "--tile-sizes",
    choices=["all", "powers_of_two", "cache_line_multiples"],
    type=str.lower,
    default="powers_of_two",
    dest="tile_sizes",
)
parser.add_argument("--print-code", action="store_true")
parser.add_argument("--save-code", type=pathlib.Path)
subparsers = parser.add_subparsers(dest="spec")

parser_matmul = subparsers.add_parser("matmul", help="Impl a matrix multiplication")
parser_matmul.add_argument("m", type=int)
parser_matmul.add_argument("k", type=int)
parser_matmul.add_argument("n", type=int)

parser_conv = subparsers.add_parser("conv", help="Impl a convolution")
parser_conv.add_argument("image_width", type=int)
parser_conv.add_argument("image_height", type=int)
parser_conv.add_argument("filter_width", type=int)
parser_conv.add_argument("filter_height", type=int)
parser_conv.add_argument("filter_count", type=int)

parser_convnet = subparsers.add_parser(
    "convnet", help="Impl a Conv-Reduce-Conv pipeline"
)

parser_gemm3 = subparsers.add_parser("gemm3", help="Impl a GEMM3")
parser_gemm3.add_argument("matrix_size", type=int)


def _matmul_main(m, k, n, top_k: int, cache: search_cache.ScheduleCache):
    target = system_config.current_target()
    left = target.tensor(target.tensor_spec((m, k), dtype=DTYPE), name="left")
    right = target.tensor(target.tensor_spec((k, n), dtype=DTYPE), name="right")
    output = target.tensor(target.tensor_spec((m, n), dtype=DTYPE), name="output")
    start = time.time()
    s = schedule_search(
        specs.Matmul(left.spec, right.spec, output.spec, serial_only=False),
        top_k=top_k,
        cache=cache,
    )
    return s, (time.time() - start)


def _conv_main(
    image_width,
    image_height,
    filter_width,
    filter_height,
    filter_count,
    top_k: int,
    cache: search_cache.ScheduleCache,
) -> tuple[Optional[impl.Impl], float]:
    target = system_config.current_target()
    assert filter_width <= image_width and filter_height <= image_height
    left = target.tensor(
        target.tensor_spec((1, 3, image_width, image_height), dtype=DTYPE), name="image"
    )
    right = target.tensor(
        target.tensor_spec((filter_count, 3, filter_width, filter_height), dtype=DTYPE),
        name="filters",
    )
    output = target.tensor(
        target.tensor_spec(
            (
                1,
                filter_count,
                image_width - filter_width + 1,
                image_height - filter_height + 1,
            ),
            dtype=DTYPE,
        ),
        name="output",
    )
    start = time.time()
    s = schedule_search(
        specs.Convolution(left.spec, right.spec, output.spec, serial_only=False),
        top_k=top_k,
        cache=cache,
    )
    return s, (time.time() - start)


def _convnet_main(top_k: int, cache: search_cache.ScheduleCache):
    target = system_config.current_target()

    d = 32

    img = target.tensor(target.tensor_spec((2, 3, d, d), dtype=DTYPE), name="img")
    filters_a = target.tensor(
        target.tensor_spec((16, 3, 5, 5), dtype=DTYPE), name="filters_a"
    )
    filters_b = target.tensor(
        target.tensor_spec((16, 16, 5, 5), dtype=DTYPE), name="filters_b"
    )
    output = target.tensor(
        target.tensor_spec((2, 16, d - 8, d - 8), dtype=DTYPE), name="output"
    )

    start = time.time()
    s = schedule_search(
        specs.Compose(
            (specs.Convolution, specs.Convolution),
            (filters_b.spec, img.spec, filters_a.spec),
            output.spec,
            intermediate_dtypes=(DTYPE,),
            serial_only=False,
        ),
        top_k=top_k,
        cache=cache,
    )
    return s, (time.time() - start)


def _gemm3_main(matrix_size: int, top_k: int, cache: search_cache.ScheduleCache):
    target = system_config.current_target()
    spec = specs.Compose(
        (specs.Matmul, specs.Matmul),
        (
            target.tensor_spec((matrix_size, matrix_size), dtype=DTYPE),
            target.tensor_spec((matrix_size, matrix_size), dtype=DTYPE),
            target.tensor_spec((matrix_size, matrix_size), dtype=DTYPE),
        ),
        target.tensor_spec((matrix_size, matrix_size), dtype=DTYPE),
        intermediate_dtypes=(DTYPE,),
        serial_only=True,
    )
    start = time.time()
    search_result = schedule_search(spec, top_k=top_k, cache=cache)
    return search_result, (time.time() - start)


def main() -> int:
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Run any doctests
    if doctest.testmod().failed:
        return 1

    # Parse command line arguments
    parsed_args = parser.parse_args()
    if parsed_args.save_code:
        parsed_args.save_code.mkdir(exist_ok=False, parents=True)

    # Set a target
    system_config.set_current_target(parsed_args.target)

    # Configure from args
    if parsed_args.row_major_only:
        col_major_token = search.prune_column_major.set(True)
    else:
        col_major_token = search.prune_column_major.set(False)

    # Apply the --tile-sizes setting
    if parsed_args.tile_sizes == "all":
        tile_size_mode = TileSizeMode.ALL
    elif parsed_args.tile_sizes == "powers_of_two":
        tile_size_mode = TileSizeMode.POWERS_OF_TWO
    elif parsed_args.tile_sizes == "cache_line_multiples":
        tile_size_mode = TileSizeMode.CACHE_LINE_MULTIPLES
    else:
        raise Exception("Unexpectedly got: " + parsed_args.tile_sizes)
    tile_size_mode_token = impl.tile_size_mode.set(tile_size_mode)

    try:
        with search_cache.persistent_cache(
            parsed_args.cache, save=parsed_args.save_cache
        ) as cache:
            if parsed_args.spec == "matmul":
                scheds, runtime = _matmul_main(
                    parsed_args.m, parsed_args.k, parsed_args.n, parsed_args.top, cache
                )
            elif parsed_args.spec == "conv":
                scheds, runtime = _conv_main(
                    parsed_args.image_width,
                    parsed_args.image_height,
                    parsed_args.filter_width,
                    parsed_args.filter_height,
                    parsed_args.filter_count,
                    parsed_args.top,
                    cache,
                )
            elif parsed_args.spec == "convnet":
                scheds, runtime = _convnet_main(parsed_args.top, cache)
            elif parsed_args.spec == "gemm3":
                scheds, runtime = _gemm3_main(
                    parsed_args.matrix_size, parsed_args.top, cache
                )
            else:
                raise Exception("Unknown spec argument: " + parsed_args.spec)
            assert isinstance(scheds, list)
    finally:
        search.prune_column_major.reset(col_major_token)
        impl.tile_size_mode.reset(tile_size_mode_token)

    # Let's apply some "concrete" tensors so we can pprint and generate code.
    scheds = [s.to_applied() for s in scheds]

    # Update the cache with real execution times.
    #
    # Run up to MAX_CONCURRENT_BENCHMARK target programs concurrently.
    # This should probably be kept well below the number of cores on the
    # system to avoid interference.
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_BENCHMARKS", "1")))

    async def get_time(imp):
        async with semaphore:
            return await system_config.current_target().time_impl_robustly(imp)

    benchmarked_runtimes = loop.run_until_complete(
        asyncio.gather(*(get_time(im) for im in scheds))
    )

    results_rows = [["Cost", "Runtime", "Impl", "Source Path"]]
    if scheds:
        for sched_idx, (sched, benchmarked_runtime) in enumerate(
            zip(scheds, benchmarked_runtimes)
        ):
            op_pprint.pprint(sched)
            if parsed_args.print_code:
                print("")
                gen.generate_c("kernel_only", sched, sys.stdout)
            print("")
            print(f"Impl Runtime: {benchmarked_runtime:.4f}s")

            source_path = pathlib.Path("")
            if parsed_args.save_code:
                source_path = parsed_args.save_code / f"{sched_idx:3d}.c"
                with (source_path).open("w") as f:
                    gen.generate_c("kernel_only", sched, f)

            results_rows.append(
                [
                    cost.compute_cost(sched),
                    benchmarked_runtime,
                    op_pprint.pformat(sched, color=False),
                    str(source_path),
                ]
            )
    else:
        results_rows.append(["", "", "", ""])
        print("No schedule found")

    print("")
    print(f"Scheduling took {runtime:.2f}s")

    if parsed_args.results:
        with parsed_args.results.open("w") as f:
            writer = csv.writer(f)
            writer.writerows(results_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())