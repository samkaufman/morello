import argparse
import asyncio
import concurrent.futures
import contextlib
import functools
import itertools
import logging
import os
import pathlib
import random
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generator, Iterable, Optional, TypeVar, Union

import dask.distributed
import redis.asyncio as redis
import tqdm
import tqdm.contrib.logging

from .. import dtypes, geometry, pruning, search_cache, specs, system_config, utils
from ..impl import spec_to_hole
from ..impl.utils import gen_vector_shapes
from . import dp

if TYPE_CHECKING:
    from .. import impl

T = TypeVar("T")

MERGE_DIAGONALS = True
BANK_GROUPS = (("RF", "VRF"), ("L1",), ("GL",))
NAMESPACE = "BOOP"  # TODO: Generate real namespaces.
PING_TRIES = 720
PING_WAIT_SECS = 5
CONV_CHANNELS = 4
CHECK_CROSS_SPEC_MISSES = False

CacheOrFut = Union[
    search_cache.ScheduleCache,
    concurrent.futures.Future[search_cache.ScheduleCache],
]

logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--scheduler", type=str)
arg_parser.add_argument("--downscale", type=int, default=1)
arg_parser.add_argument("--deploy-k8s", action="store_true")
arg_parser.add_argument("--deploy-k8s-with-custom-image", type=str)
arg_parser.add_argument("--moves-only", action="store_true")
arg_parser.add_argument("--moves-rank", type=int, default=2)
arg_parser.add_argument("--size", type=int, default=64)
arg_parser.add_argument("--image-name", "-t", type=str, default="samkaufman/morello")
arg_parser.add_argument("--moves-cache", metavar="CACHE", type=pathlib.Path)
arg_parser.add_argument("--serial-only", action="store_true")
arg_parser.add_argument("--only-u32", action="store_true")

_tlocal = None


class _DistributedSearch(dp.Search):
    def _iter_expansions(
        self, leaf: "impl.Impl", parent_summary: Optional["impl.ParentSummary"]
    ) -> Iterable["impl.Impl"]:
        # TODO: Document.
        expansions = list(super()._iter_expansions(leaf, parent_summary))
        grouped = {}
        for imp in expansions:
            key = tuple(c.spec for c in imp.children)
            grouped.setdefault(key, []).append(imp)
        for impl_group in grouped.values():
            yield from impl_group


def _spec_coordinate_to_specs(
    top_query: specs.Spec, coord: tuple[int, ...]
) -> Iterable[specs.Spec]:
    target = system_config.current_target()
    operand_count = len(top_query.operands)

    spec_sizes = list(coord[: -operand_count - 1])
    operand_shapes: list[list[int]] = []
    subs_to_sizes: dict[Any, int] = {}
    for operand_subs in top_query.operands_dim_subscripts():
        operand_shapes.append([])
        for sub in operand_subs:
            if sub in subs_to_sizes:
                operand_shapes[-1].append(subs_to_sizes[sub])
            else:
                size = spec_sizes.pop(0)
                operand_shapes[-1].append(size)
                subs_to_sizes[sub] = size

    for layouts, aligneds, tensor_banks in itertools.product(
        itertools.product(
            *[target.all_layouts_for_shape(shp) for shp in operand_shapes]
        ),
        itertools.product([True, False], repeat=operand_count),
        itertools.product(*[BANK_GROUPS[i] for i in coord[-operand_count - 1 : -1]]),
    ):
        for contiguous_abstractions, vec_kwargs in itertools.product(
            itertools.product(*[l.all_contiguous_abs() for l in layouts]),
            itertools.product(
                *[
                    _iter_vector_shape_args(b, shp, o.dtype)
                    for o, b, shp in zip(
                        top_query.operands,
                        tensor_banks,
                        operand_shapes,
                    )
                ]
            ),
        ):
            new_operands = tuple(
                target.tensor_spec(
                    tuple(operand_shapes[i]),
                    dtype=top_query.operands[i].dtype,
                    contiguous_abs=contiguous_abstractions[i],
                    aligned=aligneds[i],
                    bank=tensor_banks[i],
                    layout=layouts[i],
                    **vec_kwargs[i],
                )
                for i in range(operand_count)
            )
            inps, out = new_operands[:-1], new_operands[-1]

            r = top_query.replace_io(inps, out, serial_only=not bool(coord[-1]))
            assert coord == _spec_to_spec_coordinate(r), (
                f"During generation of Specs with grid coordinate {coord}, a Spec "
                f"with {_spec_to_spec_coordinate(r)} was generated. That Spec is {r}."
            )
            yield r


def _spec_to_spec_coordinate(spec: specs.Spec) -> tuple[int, ...]:
    sb = 0 if spec.serial_only else 1

    bank_idxs = []
    for o in spec.operands:
        for idx, bank_group in enumerate(BANK_GROUPS):
            if o.bank in bank_group:
                bank_idxs.append(idx)
                break
    bank_idxs = tuple(bank_idxs)
    assert len(bank_idxs) == len(spec.operands)

    if isinstance(spec, (specs.Matmul, specs.MatmulAccum)):
        m, k = spec.lhs.dim_sizes
        _, n = spec.rhs.dim_sizes
        return (m, k, n) + bank_idxs + (sb,)  # TODO: Need `-1`
    elif isinstance(spec, (specs.Convolution, specs.ConvolutionAccum)):
        b, c, h, w = spec.lhs.dim_sizes
        f, _, fh, fw = spec.rhs.dim_sizes
        return (b, c, h, w, f, fh, fw) + bank_idxs + (sb,)
    elif isinstance(spec, (specs.ReduceSum, specs.ReduceSumAccum)):
        return spec.source.dim_sizes + bank_idxs + (sb,)
    elif isinstance(spec, (specs.Load, specs.Store, specs.Zero)):
        return spec.output.dim_sizes + bank_idxs + (sb,)
    elif isinstance(spec, specs.Compose):
        raise ValueError("Compose Specs not supported")
    else:
        raise NotImplementedError("Unsupported Spec: " + str(spec))


async def _compute_dp_table_graph(
    executor: concurrent.futures.Executor,
    grid: geometry.Grid,
    *,
    redis_url: str,
    desc: Optional[str] = None,
    pb_position: int = 0,
):
    target = system_config.current_target()

    graph_group = random.randint(0, 2**32 - 1)

    curried = functools.partial(
        _compute_block,
        target=target,
        redis_url=redis_url,
        graph_group=graph_group,
    )

    with tqdm.contrib.logging.logging_redirect_tqdm():
        for diagonal in tqdm.tqdm(
            list(grid.block_diagonals_northeast()), desc=desc, position=pb_position
        ):
            await asyncio.gather(
                *[asyncio.wrap_future(executor.submit(curried, x)) for x in diagonal]
            )


# TODO: Can we design this so that we don't need to reproduce the logic of the
#   top-down actions expansion? Hate having the logic in two places.
def _compute_block(
    block: geometry.GridBlock[specs.Spec],
    target: system_config.Target,
    graph_group: int,
    redis_url: str,
) -> None:
    """Compute a block and save into the Redis cache."""
    # TODO: Scope this more nicely, somehow.
    global _tlocal
    if _tlocal is None:
        _tlocal = threading.local()
    if not hasattr(_tlocal, "loop"):
        _tlocal.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_tlocal.loop)
    if getattr(_tlocal, "red", None) is None:
        _tlocal.red = redis.Redis.from_url(redis_url)

        # Wait for Redis to be up.
        for _ in range(PING_TRIES):
            try:
                _tlocal.loop.run_until_complete(_tlocal.red.ping())
            except redis.BusyLoadingError:
                time.sleep(PING_WAIT_SECS)
                pass
            else:
                break

    if getattr(_tlocal, "results_cache_graph_group", None) != graph_group:
        _tlocal.results_cache = None
        _tlocal.results_cache_graph_group = graph_group
    if _tlocal.results_cache is None:
        # Construct a ChainCache where the Redis cache is *first*.
        # The Redis cache will ignore very small Specs, and in these cases,
        # we want to hit the backup local cache.
        #
        # TODO: Assert block membership in lambda
        redis_cache = search_cache.ScheduleCache(use_redis=(_tlocal.red, NAMESPACE))
        local_cache = search_cache.ScheduleCache(max_dim=search_cache.REDIS_MIN_DIM)
        _tlocal.results_cache = search_cache.ChainCache(
            [redis_cache, local_cache], put_all=True
        )
    assert _tlocal.red is not None and _tlocal.results_cache is not None

    with system_config.with_target(target):
        searcher = _DistributedSearch()
        # Run every Spec concurrently, stepping (and reloading) only for new memory limits.
        subproblems_to_run = [(s, pruning.StandardMemoryLimits()) for s in block]
        while subproblems_to_run:
            subproblems_to_run = _step_compute_block(
                _tlocal.results_cache, searcher, subproblems_to_run
            )
        _tlocal.loop.run_until_complete(_tlocal.results_cache.flush())
    _tlocal.loop.run_until_complete(_tlocal.red.close(close_connection_pool=True))


def _step_compute_block(
    results_cache: search_cache.ScheduleCache,
    searcher: dp.Search,
    subproblems_to_run: Iterable[tuple[specs.Spec, pruning.StandardMemoryLimits]],
) -> list[tuple[specs.Spec, pruning.StandardMemoryLimits]]:
    global _tlocal
    assert _tlocal is not None

    search_tasks: dict[int, Generator] = {}
    for spec, mlims in subproblems_to_run:
        # TODO: Pass parent_summary too, if applicable.
        new_task = searcher.interactive_search(spec_to_hole(spec), mlims)
        search_tasks[len(search_tasks)] = new_task

    # TODO: Assert that each of our search generators requests each dependent block
    #   exactly once.
    needs: dict[int, Sequence[tuple[specs.Spec, pruning.MemoryLimits]]] = {
        k: [] for k in search_tasks
    }
    first_step = True  # TODO: Need this, really?
    next_subproblems_to_run = []
    while search_tasks:
        # `subproblems_requested` will have duplicates. Rely on the cache to merge.
        subproblems_requested = list(itertools.chain.from_iterable(needs.values()))
        cache_results_flat = list(
            _tlocal.loop.run_until_complete(
                results_cache.get_many(subproblems_requested)
            )
        )
        assert len(subproblems_requested) == len(cache_results_flat)

        # Check that we don't miss any Specs with coordinates below the current Spec
        # in the data dependency grid.
        # NOTE: This is checking Spec, not block, coordinates, so if we downscale, we
        #       might raise for intra-block misses, which may or may not be expected!
        if CHECK_CROSS_SPEC_MISSES:
            for (target_spec, _), request, cache_result in zip(
                subproblems_to_run, subproblems_requested, cache_results_flat
            ):
                if cache_result is None and _spec_to_spec_coordinate(
                    target_spec
                ) != _spec_to_spec_coordinate(request[0]):
                    raise Exception(
                        f"Unexpected, cross-task miss! While computing {target_spec} "
                        f"at coordinate {_spec_to_spec_coordinate(target_spec)}, "
                        f"missed {', '.join(map(str, request))} at coordinate "
                        f"{_spec_to_spec_coordinate(request[0])}."
                    )

        cache_results_grouped = {}
        taken = 0
        for k in needs:
            cache_results_grouped[k] = cache_results_flat[taken : taken + len(needs[k])]
            taken += len(needs[k])
        assert taken == len(
            cache_results_flat
        ), f"Did not consume all {len(cache_results_flat)} results"

        completed_task_keys = []
        assert len(search_tasks) == len(set(search_tasks.keys()))
        for task_key, search_gen in search_tasks.items():
            try:
                if first_step:
                    msg: dp.SearchMessage = next(search_gen)
                else:
                    # Pop from front of the cache response.
                    msg = search_gen.send(cache_results_grouped[task_key])
            except StopIteration:
                completed_task_keys.append(task_key)
                continue

            needs[task_key] = msg.needed
            for mlims, entry in msg.computed:
                _tlocal.loop.run_until_complete(results_cache.put(entry, mlims))
                # TODO: Clean up dominated limits?
                next_subproblems_to_run.extend(
                    (entry.spec, smaller_mlims)
                    for smaller_mlims in next_limits(mlims, entry.peak_or_zero)
                )
        for task_ident in completed_task_keys:
            del search_tasks[task_ident]
            del needs[task_ident]
        first_step = False

    return next_subproblems_to_run


def next_limits(
    result_limits: pruning.StandardMemoryLimits, result_peak: utils.TinyMap
) -> Iterable[pruning.StandardMemoryLimits]:
    """Returns next memory limits to try, given limits and peak of the last result."""
    if any(
        l < p
        for l, p in zip(result_limits.available.raw_values, result_peak.raw_values)
    ):
        raise ValueError(f"Peak {result_peak} exceeds limits {result_limits}")

    keys_ordered = result_limits.available.raw_keys
    assert keys_ordered == result_peak.raw_keys
    for idx in range(len(keys_ordered)):
        new_values = list(result_limits.available.raw_values)
        if result_peak.raw_values[idx] == 0:
            continue
        if result_peak.raw_values[idx] == 1:
            new_values[idx] = 0
        else:
            new_values[idx] = 2 ** (result_peak.raw_values[idx].bit_length() - 2)
            assert isinstance(new_values[idx], int) and new_values[0] >= 0
        new_cap = pruning.StandardMemoryLimits(
            utils.TinyMap(keys_ordered, tuple(new_values))
        )
        yield new_cap


def _iter_vector_shape_args(
    bank: str, outer_shape: Sequence[int], dtype
) -> Iterable[dict]:
    system = system_config.current_system()
    vector_bytes: Optional[int] = system.banks[bank].vector_bytes
    if vector_bytes is None:
        yield {}
        return
    for shape in gen_vector_shapes(None, dtype, vector_bytes, rank=len(outer_shape)):
        yield {"vector_shape": shape}


def _make_docker_image(image_name: str) -> str:
    current_dir = pathlib.Path(__file__).parent.resolve().parent.parent
    assert (
        current_dir / "Dockerfile"
    ).is_file(), f"{current_dir} does not contain a Dockerfile"

    tag = f"{image_name}:dep{secrets.token_hex(14)}"
    logger.info("Building Docker image with tag %s", tag)
    build_process = subprocess.run(
        f"docker build --target cpu-only -t {tag} -q .",
        stdout=subprocess.PIPE,
        shell=True,
        cwd=current_dir,
    )
    build_process.check_returncode()
    image_id = build_process.stdout.strip()
    logger.info("Built image with ID %s", image_id.decode("utf8"))

    logger.info("Pushing Docker image with tag %s", tag)
    subprocess.run(f"docker push -q {tag}", shell=True, check=True)
    logger.info("Pushed Docker image with tag %s", tag)

    return tag


def _deploy_k8s_cluster(image_tag: str):
    from dask_kubernetes import KubeCluster, make_pod_spec

    logger.info("Starting the Kubernetes cluster")
    cluster = KubeCluster(
        make_pod_spec(
            image=image_tag,
            memory_limit="8.2G",
            memory_request="7G",
            cpu_limit=1,
            cpu_request=1,
            extra_container_config={"imagePullPolicy": "Always"},
        )
    )
    cluster.adapt(minimum=1, maximum=256)
    logger.info("Started the Kubernetes cluster")
    logger.info("Kubernetes scheduler address: %s", cluster.scheduler_address)
    return cluster


def _grid_for_query_spec(spec: specs.Spec, downscale: int) -> geometry.Grid[specs.Spec]:
    """Build a Grid including the given `spec` and its dependencies.

    Overapproximates the dependencies.

    NOTE: `downscale` is only applied to the operand dimensions.
    """
    grid_dims: list[geometry.BlockedRange] = []

    # Add ranges for the Spec dimensions.
    if isinstance(spec, specs.Compose):
        raise ValueError("Compose Specs not supported")
    seen_dims = set()
    for operand, operand_subs in zip(spec.operands, spec.operands_dim_subscripts()):
        for size, sub in zip(operand.dim_sizes, operand_subs):
            if sub in seen_dims:
                continue
            seen_dims.add(sub)
            grid_dims.append(geometry.log_range(1, size + 1, downscale))

    # Add ranges for the operands' banks.
    for _ in range(len(spec.operands)):
        grid_dims.append(geometry.SimpleBlockedRange(0, len(BANK_GROUPS)))

    # Add a dimension for serial-parallel.
    grid_dims.append(geometry.SimpleBlockedRange(0, 1 if spec.serial_only else 2))

    return geometry.Grid(
        grid_dims,
        mapper=functools.partial(_spec_coordinate_to_specs, spec),
        rev_mapper=_spec_to_spec_coordinate,
    )


async def _compute_dtype_graph(args, redis_url, target, executor, dt_idx, dt) -> None:
    if args.moves_rank < 2:
        raise ValueError("args.moves_rank must be at least 2")

    target = system_config.current_target()

    for rank in range(1, args.moves_rank + 1):
        load_spec = specs.Load(
            source=target.tensor_spec((args.size,) * rank, dtype=dt),
            destination=target.tensor_spec((args.size,) * rank, dtype=dt),
            serial_only=args.serial_only,
        )
        store_spec = specs.Store(
            source=target.tensor_spec((args.size,) * rank, dtype=dt),
            destination=target.tensor_spec((args.size,) * rank, dtype=dt),
            serial_only=args.serial_only,
        )
        zero_spec = specs.Zero(
            destination=target.tensor_spec((args.size,) * rank, dtype=dt),
            serial_only=args.serial_only,
        )

        load_grid = _grid_for_query_spec(load_spec, args.downscale)
        store_grid = _grid_for_query_spec(store_spec, args.downscale)
        zero_grid = _grid_for_query_spec(zero_spec, args.downscale)

        # Load and Store have no deps. Zero depends on Store, so it goes last.
        load_fut = _compute_dp_table_graph(
            executor,
            load_grid,
            redis_url=redis_url,
            desc=f"Load, rank {rank}, {dt}",
            pb_position=(dt_idx * 2),
        )
        store_fut = _compute_dp_table_graph(
            executor,
            store_grid,
            redis_url=redis_url,
            desc=f"Store, rank {rank}, {dt}",
            pb_position=(dt_idx * 2) + 1,
        )
        await asyncio.gather(load_fut, store_fut)
        await _compute_dp_table_graph(
            executor, zero_grid, redis_url=redis_url, desc=f"Zero, rank {rank}, {dt}"
        )

    if not args.moves_only:
        for matmul_cls in (specs.MatmulAccum, specs.Matmul):
            matmul_spec = matmul_cls(
                lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                output=target.tensor_spec((args.size, args.size), dtype=dt),
                serial_only=args.serial_only,
            )
            matmul_grid = _grid_for_query_spec(matmul_spec, args.downscale)
            await _compute_dp_table_graph(
                executor,
                matmul_grid,
                redis_url=redis_url,
                pb_position=(dt_idx * 2),
                desc=f"{matmul_cls.__name__}, {dt}",
            )

        if args.moves_rank < 4:
            print("Skipping convolutions because moves rank < 4", file=sys.stderr)
        else:
            for conv_cls in (specs.ConvolutionAccum, specs.Convolution):
                img_shape = (args.size, CONV_CHANNELS, args.size, args.size)
                filters_shape = (args.size, CONV_CHANNELS, args.size, args.size)
                conv_spec = conv_cls(
                    lhs=target.tensor_spec(img_shape, dtype=dt),
                    rhs=target.tensor_spec(filters_shape, dtype=dt),
                    output=target.tensor_spec(
                        conv_cls.output_shape(img_shape, filters_shape), dtype=dt
                    ),
                    serial_only=args.serial_only,
                )
                conv_grid = _grid_for_query_spec(conv_spec, args.downscale)
                await _compute_dp_table_graph(
                    executor,
                    conv_grid,
                    redis_url=redis_url,
                    pb_position=(dt_idx * 2),
                    desc=f"{conv_cls.__name__}, {dt}",
                )


async def main(args=None):
    logging.basicConfig(level=logging.INFO)

    redis_url = os.environ["REDIS_URL"]

    if not args:
        args = arg_parser.parse_args()

    dtypes_to_enum = dtypes.ALL_DTYPES
    if args.only_u32:
        dtypes_to_enum = [dtypes.Uint32]

    start = time.time()

    system_config.set_current_target("cpu")
    target = system_config.current_target()

    if args.scheduler:
        cluster = contextlib.nullcontext()
        cluster_address = args.scheduler
    elif args.deploy_k8s or args.deploy_k8s_with_custom_image:
        if args.deploy_k8s_with_custom_image:
            if args.image_name:
                print("--image_name ignored when using a custom image", file=sys.stderr)
            tag = args.deploy_k8s_with_custom_image
        else:
            assert args.image_name
            tag = _make_docker_image(args.image_name)
        cluster = _deploy_k8s_cluster(tag)
        cluster_address = cluster
    else:
        cluster = dask.distributed.LocalCluster(threads_per_worker=1)
        cluster_address = cluster

    with cluster, dask.distributed.Client(
        cluster_address
    ) as dask_client, dask.config.set(scheduler=dask_client):
        executor = dask_client.get_executor()
        await asyncio.gather(
            *[
                _compute_dtype_graph(args, redis_url, target, executor, dt_idx, dt)
                for dt_idx, dt in enumerate(dtypes_to_enum)
            ]
        )

    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
