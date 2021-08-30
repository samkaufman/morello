#!python3

"""Experiments in search over a Kriek Impl-like IR."""
import argparse
import doctest
import logging
import sys
import time
from typing import TypeVar

from morello import ops, op_pprint, search, search_cache, specs
from morello.search import schedule_search
from morello.tensor import Tensor

T = TypeVar("T")

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=str)
parser.add_argument("--no-save-cache", action="store_false", dest="save_cache")
parser.add_argument("--row-major-only", action="store_true", dest="row_major_only")
parser.add_argument(
    "--tile-sizes",
    choices=["all", "powers_of_two", "cache_line_multiples"],
    type=str.lower,
    default="powers_of_two",
    dest="tile_sizes",
)
subparsers = parser.add_subparsers(dest="spec")

parser_matmul = subparsers.add_parser("matmul", help="Schedule a matrix multiplication")
parser_matmul.add_argument("m", type=int)
parser_matmul.add_argument("k", type=int)
parser_matmul.add_argument("n", type=int)

parser_conv = subparsers.add_parser("conv", help="Schedule a convolution")
parser_conv.add_argument("image_width", type=int)
parser_conv.add_argument("image_height", type=int)
parser_conv.add_argument("filter_width", type=int)
parser_conv.add_argument("filter_height", type=int)
parser_conv.add_argument("filter_count", type=int)

parser_convnet = subparsers.add_parser(
    "convnet", help="Schedule a Conv-Reduce-Conv pipeline"
)


def _matmul_main(m, k, n, cache: search_cache.ScheduleCache):
    left = Tensor(specs.TensorSpec((m, k)), name="left")
    right = Tensor(specs.TensorSpec((k, n)), name="right")
    output = Tensor(specs.TensorSpec((m, n)), name="output")
    start = time.time()
    op_pprint.pprint(
        schedule_search(
            specs.Matmul(left.spec, right.spec, output.spec, serial_only=False),
            inputs=(left, right),
            output=output,
            cache=cache,
        )
    )
    print(f"Took {time.time() - start:.2f}s")


def _conv_main(
    image_width,
    image_height,
    filter_width,
    filter_height,
    filter_count,
    cache: search_cache.ScheduleCache,
):
    assert filter_width <= image_width and filter_height <= image_height
    left = Tensor(specs.TensorSpec((image_width, image_height)), name="image")
    right = Tensor(
        specs.TensorSpec((filter_width, filter_height, filter_count)), name="filters"
    )
    output = Tensor(
        specs.TensorSpec(
            (
                image_width - filter_width + 1,
                image_height - filter_height + 1,
                filter_count,
            )
        ),
        name="output",
    )
    start = time.time()
    op_pprint.pprint(
        schedule_search(
            specs.Convolution(left.spec, right.spec, output.spec, serial_only=False),
            inputs=(left, right),
            output=output,
            cache=cache,
        )
    )
    print(f"Took {time.time() - start:.2f}s")


def _convnet_main(cache: search_cache.ScheduleCache):
    img = Tensor(specs.TensorSpec((8, 8)), name="image")
    filters_a = Tensor(specs.TensorSpec((3, 3, 4)), name="filtersA")
    filters_b = Tensor(specs.TensorSpec((3, 3, 4)), name="filtersB")
    output = Tensor(specs.TensorSpec((4, 4, 4)), name="output")

    start = time.time()
    op_pprint.pprint(
        schedule_search(
            specs.Compose(
                (specs.Convolution, specs.ReduceSum, specs.Convolution),
                (filters_b.spec, img.spec, filters_a.spec),
                output.spec,
                serial_only=False,
            ),
            inputs=(filters_b, img, filters_a),
            output=output,
            cache=cache,
        ),
    )
    print(f"Took {time.time() - start:.2f}s")


def main() -> int:
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Run any doctests
    if doctest.testmod().failed:
        return 1

    # Parse command line arguments
    parsed_args = parser.parse_args()

    # Configure from args
    if parsed_args.row_major_only:
        col_major_token = search.prune_column_major.set(True)
    else:
        col_major_token = search.prune_column_major.set(False)

    # Apply the --tile-sizes setting
    if parsed_args.tile_sizes == "all":
        tile_size_mode = ops.TileSizeMode.ALL
    elif parsed_args.tile_sizes == "powers_of_two":
        tile_size_mode = ops.TileSizeMode.POWERS_OF_TWO
    elif parsed_args.tile_sizes == "cache_line_multiples":
        tile_size_mode = ops.TileSizeMode.CACHE_LINE_MULTIPLES
    else:
        raise Exception("Unexpectedly got: " + parsed_args.tile_sizes)
    tile_size_mode_token = ops.tile_size_mode.set(tile_size_mode)

    try:
        with search_cache.persistent_cache(
            parsed_args.cache, save=parsed_args.save_cache
        ) as cache:
            if parsed_args.spec == "matmul":
                _matmul_main(parsed_args.m, parsed_args.k, parsed_args.n, cache)
            elif parsed_args.spec == "conv":
                _conv_main(
                    parsed_args.image_width,
                    parsed_args.image_height,
                    parsed_args.filter_width,
                    parsed_args.filter_height,
                    parsed_args.filter_count,
                    cache,
                )
            elif parsed_args.spec == "convnet":
                _convnet_main(cache)
    finally:
        search.prune_column_major.reset(col_major_token)
        ops.tile_size_mode.reset(tile_size_mode_token)

    return 0


if __name__ == "__main__":
    sys.exit(main())
