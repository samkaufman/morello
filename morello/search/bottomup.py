import abc
import argparse
from collections.abc import Sequence
import concurrent.futures
import contextlib
import copy
import dataclasses
import itertools
import logging
import os
import pathlib
import pickle
import random
import secrets
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING, Union

import dask
import dask.distributed
import redis

from . import dp
from .. import dtypes, pruning, search_cache, specs, system_config, utils
from ..impl.utils import gen_vector_shapes

if TYPE_CHECKING:
    from .. import impl

MERGE_DIAGONALS = True
BANK_GROUPS = (("RF", "VRF"), ("L1",), ("GL",))

CacheOrFut = Union[
    search_cache.InMemoryScheduleCache,
    concurrent.futures.Future[search_cache.InMemoryScheduleCache],
]

logger = logging.getLogger(__name__)

redis_connection: Optional[redis.Redis] = None

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--scheduler", type=str)
arg_parser.add_argument("--downscale", type=int, default=1)
arg_parser.add_argument("--deploy-k8s", action="store_true")
arg_parser.add_argument("--deploy-k8s-with-custom-image", type=str)
arg_parser.add_argument("--moves-only", action="store_true")
arg_parser.add_argument("--moves-rank", type=int, default=2)
arg_parser.add_argument("--size", type=int, default=512)
arg_parser.add_argument("--image-name", "-t", type=str, default="samkaufman/morello")
arg_parser.add_argument("--moves-cache", metavar="CACHE", type=pathlib.Path)
arg_parser.add_argument("--serial-only", action="store_true")


class _DistributedSearch(dp.Search):
    def __init__(
        self, downscale: int, block_coordinate: "Coord", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.downscale = downscale
        self.block_coordinate = block_coordinate

    def _iter_expansions(
        self, leaf: "impl.Impl", parent_summary: Optional["impl.ParentSummary"]
    ) -> Iterable["impl.Impl"]:
        # TODO: Enumerate all expansions and group by the block to which they belong.
        #   This allows downstream search to
        for impl in super()._iter_expansions(leaf, parent_summary):
            yield impl


class RedisCache(search_cache.ScheduleCache):
    """A cache which streams blocks to and from a Redis backend."""

    def __init__(
        self,
        redis_connection: redis.Redis,
        namespace: str,
        subproblem_to_block_coordinate_fn: Callable[
            [specs.Spec, pruning.MemoryLimits], tuple[int, ...]
        ],
    ) -> None:
        super().__init__()
        # TODO: Add an in-process overlay cache
        self.redis_connection = redis_connection
        self.namespace = namespace
        self.overlay = search_cache.InMemoryScheduleCache()
        self._dirty_block: Optional[str] = None
        self._subproblem_to_block_coordinate_fn = subproblem_to_block_coordinate_fn

    def get(
        self, spec: "specs.Spec", memory_limits: pruning.MemoryLimits, recurse_fn=None
    ) -> search_cache.CachedScheduleSet:
        if recurse_fn is not None:
            raise NotImplementedError("recurse_fn must be None")
        block = self._get_block(self._redis_block_key(spec, memory_limits))
        if block is None:
            raise KeyError()
        return block.get(spec, memory_limits, recurse_fn=self.get)

    def put(
        self,
        spec: "specs.Spec",
        schedules_to_put: search_cache.CachedScheduleSet,
        memory_limits: pruning.MemoryLimits,
    ) -> None:
        block_key = self._redis_block_key(spec, memory_limits)
        if self._dirty_block and self._dirty_block != block_key:
            raise Exception(
                "Putting into a block other than the current dirty "
                "block. Call `flush` before `put`ing to a new block. "
                f"Current dirty block is {self._dirty_block} and "
                f"targeted block was {block_key}."
            )
        self._dirty_block = block_key
        self.overlay.put(spec, schedules_to_put, memory_limits)

    def flush(self):
        if not self._dirty_block:
            return

        # TODO: Remove
        from dask import distributed

        distributed.print(f"Flushing block {self._dirty_block}")

        # TODO: There should never be a block, so just check that this is the case.
        block = self._get_block(self._dirty_block)
        if block is not None:
            block.update(self.overlay)
            blob = pickle.dumps(block)
        else:
            blob = pickle.dumps(self.overlay)
        self.redis_connection.set(self._dirty_block, blob)

        self.overlay = search_cache.InMemoryScheduleCache()
        self._dirty_block = None

    def specs(self) -> Iterable["specs.Spec"]:
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def _get_block(self, key: str) -> Optional[search_cache.InMemoryScheduleCache]:
        blob = self.redis_connection.get(key)
        if blob is None:
            return None
        assert isinstance(blob, bytes), f"blob was not bytes, was {type(blob).__name__}"
        decoded = pickle.loads(blob)
        assert isinstance(
            decoded, search_cache.InMemoryScheduleCache
        ), f"decoded was not InMemoryScheduleCache; was {type(decoded).__name__}"
        return decoded

    def _redis_block_key(
        self, spec: "specs.Spec", memory_limits: pruning.MemoryLimits
    ) -> str:
        # TODO: Move the Spec type name prefix outside this class.
        block_pt = self._subproblem_to_block_coordinate_fn(spec, memory_limits)
        block_description = ",".join(map(str, block_pt))
        return f"{self.namespace}-{type(spec).__name__}-{block_description}"


@dataclasses.dataclass(frozen=True)
class Coord:
    dim_coords: tuple[int, ...]
    other_coords: tuple[int, ...]

    def flatten(self) -> tuple[int, ...]:
        return self.dim_coords + self.other_coords

    @staticmethod
    def structure(flattened: tuple[int, ...], reference_spec: specs.Spec) -> "Coord":
        other_coords_cnt = Coord.other_coords_count(type(reference_spec))
        return Coord(
            dim_coords=flattened[:-other_coords_cnt],
            other_coords=flattened[-other_coords_cnt:],
        )

    @staticmethod
    def other_coords_count(spec_type: type):
        if spec_type in (specs.Matmul, specs.MatmulAccum):
            return 4
        elif spec_type in (specs.Convolution, specs.ConvolutionAccum):
            return 4
        elif spec_type in (specs.ReduceSum, specs.ReduceSumAccum):
            return 3
        elif spec_type in (specs.Load, specs.Store):
            return 3
        elif spec_type == specs.Zero:
            return 2
        elif spec_type == specs.Compose:
            raise ValueError("Compose Specs not supported")
        else:
            raise ValueError("Unexpected type: " + str(spec_type))


def _spec_coordinates_in_block(downscale: int, block_coord: Coord) -> Iterable[Coord]:
    for log_pts in itertools.product(
        *[range(b * downscale, (b + 1) * downscale) for b in block_coord.dim_coords]
    ):
        yield Coord(
            tuple(2 ** (p - 1) if p > 0 else 0 for p in log_pts),
            block_coord.other_coords,
        )


def _spec_coordinate_to_block_coord(downscale: int, spec_coord: Coord) -> Coord:
    # Scale the tensor dimensions down.
    # TODO: Handle zero such that exactly `downscale` dimensions in end up in a block.
    dim_coords = tuple(
        (0 if x == 0 else (x - 1).bit_length() + 1) // downscale
        for x in spec_coord.dim_coords
    )
    return Coord(dim_coords, spec_coord.other_coords)


def _spec_to_spec_coordinate(
    spec: specs.Spec, memory_limits: pruning.MemoryLimits
) -> Coord:
    system = system_config.current_system()

    sb = 0 if spec.serial_only else 1
    bank_idxs = []
    for o in spec.operands:
        for idx, bank_group in enumerate(BANK_GROUPS):
            if o.bank in bank_group:
                bank_idxs.append(idx)
                break
    bank_idxs = tuple(bank_idxs)
    assert len(bank_idxs) == len(spec.operands)

    ls = tuple(
        system.banks[b].capacity - memory_limits.available[b]
        for b in system.ordered_banks
    )

    if isinstance(spec, (specs.Matmul, specs.MatmulAccum)):
        m, k = spec.lhs.dim_sizes
        _, n = spec.rhs.dim_sizes
        return Coord((m - 1, k - 1, n - 1) + ls, bank_idxs + (sb,))
    elif isinstance(spec, (specs.Convolution, specs.ConvolutionAccum)):
        b, c, h, w = (d - 1 for d in spec.lhs.dim_sizes)
        f, _, fh, fw = (d - 1 for d in spec.rhs.dim_sizes)
        return Coord((b, c, f, h, w, fh, fw) + ls, bank_idxs + (sb,))
    elif isinstance(spec, (specs.ReduceSum, specs.ReduceSumAccum)):
        return Coord(
            tuple(d - 1 for d in spec.source.dim_sizes) + ls, bank_idxs + (sb,)
        )
    elif isinstance(spec, (specs.Load, specs.Store, specs.Zero)):
        return Coord(
            tuple(d - 1 for d in spec.output.dim_sizes) + ls, bank_idxs + (sb,)
        )
    elif isinstance(spec, specs.Compose):
        raise ValueError("Compose Specs not supported")
    else:
        raise NotImplementedError("Unsupported Spec: " + str(spec))


# TODO: Can we just automatically invert _spec_to_coordinate? Or derive from Spec?
def _destructure_spec_coordinate(
    base_spec: specs.Spec, coord: Coord
) -> tuple[
    Sequence[tuple[int, ...]],
    pruning.StandardMemoryLimits,
    Sequence[Sequence[str]],
    bool,
]:
    """Return a modified copy of the given Spec with the given coordinate."""
    system = system_config.current_system()

    spec_dims = tuple(d + 1 for d in coord.dim_coords[: -len(system.ordered_banks)])
    limit_dims = coord.dim_coords[-len(system.ordered_banks) :]
    limits = pruning.StandardMemoryLimits(
        utils.TinyMap(
            system.ordered_banks,
            tuple(
                system_config.current_system().banks[bank].capacity - bottom
                for bank, bottom in zip(
                    system_config.current_system().ordered_banks, limit_dims
                )
            ),
        )
    )
    bank_grps: Sequence[Sequence[str]] = [
        BANK_GROUPS[b] for b in coord.other_coords[:-1]
    ]
    serial = coord.other_coords[-1] == 0

    if isinstance(base_spec, (specs.Matmul, specs.MatmulAccum)):
        return (
            [spec_dims[:2], spec_dims[1:3], (spec_dims[0], spec_dims[2])],
            limits,
            bank_grps,
            serial,
        )
    elif isinstance(base_spec, (specs.Convolution, specs.ConvolutionAccum)):
        b, c, f, h, w, fh, fw = spec_dims
        return [(b, c, h, w), (f, c, fh, fw), (b, f, h, w)], limits, bank_grps, serial
    elif isinstance(base_spec, (specs.ReduceSum, specs.ReduceSumAccum)):
        return (spec_dims, spec_dims[:-1]), limits, bank_grps, serial
    elif isinstance(base_spec, (specs.Load, specs.Store)):
        return (spec_dims, spec_dims), limits, bank_grps, serial
    elif isinstance(base_spec, specs.Zero):
        return (spec_dims,), limits, bank_grps, serial
    elif isinstance(base_spec, specs.Compose):
        raise ValueError("Compose Specs not supported")
    else:
        raise NotImplementedError("Unsupported Spec: " + str(base_spec))


class DPTableGraph:
    diagonals: list["DPTableGraphDiagonal"]
    downscale: int

    def __init__(
        self,
        executor: concurrent.futures.Executor,
        downscale: int,
        top_query_spec: specs.Spec,
        top_limits: pruning.MemoryLimits,
        base_cache: Optional[CacheOrFut] = None,
    ):
        self.downscale = downscale
        self.diagonals = []

        rid = f"{random.randint(0, 9999999)}"

        top_spec_pt = _spec_to_spec_coordinate(top_query_spec, top_limits)
        top_block = _spec_coordinate_to_block_coord(downscale, top_spec_pt)
        self.top_block = top_block

        diagonal_idx = 0
        while True:
            diagonal_coordinates = [
                Coord.structure(d, top_query_spec)
                for d in utils.sum_seqs(maxes=top_block.flatten(), total=diagonal_idx)
            ]
            if not diagonal_coordinates:
                break

            # Add a number to the task key if all operands have the same rank.
            suffix = type(top_query_spec).__name__
            first_rank = len(top_query_spec.operands[0].dim_sizes)
            if all(len(o.dim_sizes) == first_rank for o in top_query_spec.operands):
                M = ["Zero", "One", "Two", "Three", "Four", "Five"]
                try:
                    suffix += M[len(top_query_spec.operands[0].dim_sizes)]
                except IndexError:
                    pass
            suffix += f"-{diagonal_idx}"

            new_diagonal = DPTableGraphDiagonal(
                executor=executor,
                diagonal_coordinates=diagonal_coordinates,
                rid=rid,
                downscale=downscale,
                suffix=suffix,
                top_query_spec=top_query_spec,
                top_limits=top_limits,
                previous_diagonals=list(self.diagonals),
                initial_cache=(base_cache if diagonal_idx == 0 else None),
            )
            self.diagonals.append(new_diagonal)

            diagonal_idx += 1

        assert (
            tuple(
                map(
                    max,
                    zip(*[c.flatten() for d in self.diagonals for c in d.coordinates]),
                )
            )
            == top_block.flatten()
        )


class DPTableGraphDiagonal:
    coordinates: Sequence[Coord]
    _blocks: list[concurrent.futures.Future[search_cache.InMemoryScheduleCache]]
    _merged_result: Optional[concurrent.futures.Future[search_cache.ScheduleCache]]

    def __init__(
        self,
        executor: concurrent.futures.Executor,
        diagonal_coordinates: Iterable[Coord],
        *,
        rid: str,
        downscale: int,
        suffix: str,
        top_query_spec: specs.Spec,
        top_limits: pruning.MemoryLimits,
        previous_diagonals: Sequence["DPTableGraphDiagonal"],
        initial_cache: Optional[CacheOrFut] = None,
    ):
        self.coordinates = list(diagonal_coordinates)
        self._blocks = []
        self._merged_result = None

        target = system_config.current_target()

        # TODO: The following gets duplicates, right? Clean those up.
        for block_coordinate in self.coordinates:
            new_task = executor.submit(
                _compute_block,
                block_coordinate,
                target,
                top_query_spec,
                top_limits,
                downscale,
                # TODO: Reintroduce key names for Dask, if possible.
                # dask_key_name=f"block-{suffix}-{bidx}-{rid}",
            )
            self._blocks.append(new_task)

        futures_done, _ = concurrent.futures.wait(
            self._blocks, return_when=concurrent.futures.FIRST_EXCEPTION
        )
        for fut in futures_done:
            # `result` will re-raise exceptions, if any.
            fut.result()

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(
        self, idx: int
    ) -> concurrent.futures.Future[search_cache.InMemoryScheduleCache]:
        return self._blocks[idx][0]


def _merge_caches(
    caches: Sequence[CacheOrFut],
    target: system_config.Target,
) -> search_cache.InMemoryScheduleCache:
    if not caches:
        raise ValueError("No caches to merge")

    if len(caches) == 1 and isinstance(caches[0], search_cache.InMemoryScheduleCache):
        return caches[0]

    resolved_caches = []
    future_caches = []
    for c in caches:
        if isinstance(c, search_cache.ScheduleCache):
            resolved_caches.append(c)
        else:
            future_caches.append(c)

    with system_config.with_target(target):
        if isinstance(resolved_caches[0], search_cache.InMemoryScheduleCache):
            cache = copy.deepcopy(resolved_caches[0])
            resolved_caches = resolved_caches[1:]
        else:
            cache = search_cache.InMemoryScheduleCache()

        for c in resolved_caches:
            cache.update(c)
        for c in concurrent.futures.as_completed(future_caches):
            cache.update(c.result())
        return cache


def _enumerate_contig_abstractions(
    spec_type: type, banks: Sequence[str], layouts
) -> Iterable[Sequence[Any]]:
    return itertools.product(*[l.all_contiguous_abs_for_shape() for l in layouts])


# TODO: Can we design this so that we don't need to reproduce the logic of they
#   top-down actions expansion? Hate having the logic in two places.
def _compute_block(
    block_coord: Coord,
    target: system_config.Target,
    top_query_spec: specs.Spec,
    top_limits: pruning.MemoryLimits,
    downscale: int,
) -> search_cache.InMemoryScheduleCache:
    """Compute a block given an input cache and return an overlay cache.

    The returned cache does not contain Specs in the provided `cache`; just the
    additional Specs computed for this block.
    """
    global redis_connection

    if not redis_connection:
        redis_url = os.getenv("REDIS_URL")
        assert isinstance(redis_url, str)
        redis_connection = redis.Redis.from_url(redis_url)

    namespace = "TODO_ns"  # TODO: Generate real namespaces.
    results_cache = RedisCache(
        redis_connection,
        namespace,
        lambda spec, limits: _spec_coordinate_to_block_coord(
            downscale, _spec_to_spec_coordinate(spec, limits)
        ).flatten(),
    )

    with system_config.with_target(target):
        op_count = len(top_query_spec.operands)

        searcher = _DistributedSearch(downscale, block_coord, cache=results_cache)

        top_block_coord = _spec_to_spec_coordinate(top_query_spec, top_limits)
        for spec_coord in _spec_coordinates_in_block(downscale, block_coord):
            # The largest block might contain Specs which are larger than the bounding
            # query shape (`top_query_shape`). Skip those.
            if any(
                s > t for s, t in zip(spec_coord.flatten(), top_block_coord.flatten())
            ):
                continue

            (
                operand_shapes,
                memory_limits,
                tensor_bank_groups,
                serial_only,
            ) = _destructure_spec_coordinate(top_query_spec, spec_coord)
            for layouts, aligneds in itertools.product(
                itertools.product(
                    *[target.all_layouts_for_shape(shp) for shp in operand_shapes]
                ),
                itertools.product([True, False], repeat=op_count),
            ):
                for tensor_banks in itertools.product(*tensor_bank_groups):
                    for contiguous_abstractions in _enumerate_contig_abstractions(
                        type(top_query_spec), tensor_banks, layouts
                    ):
                        for vec_kwargs in itertools.product(
                            *[
                                _iter_vector_shape_args(b, shp, o.dtype)
                                for o, b, shp in zip(
                                    top_query_spec.operands,
                                    tensor_banks,
                                    operand_shapes,
                                )
                            ]
                        ):
                            new_operands = tuple(
                                target.tensor_spec(
                                    operand_shapes[i],
                                    dtype=top_query_spec.operands[i].dtype,
                                    contiguous_abs=contiguous_abstractions[i],
                                    aligned=aligneds[i],
                                    bank=tensor_banks[i],
                                    layout=layouts[i],
                                    **vec_kwargs[i],
                                )
                                for i in range(op_count)
                            )
                            new_inputs, new_output = new_operands[:-1], new_operands[-1]
                            spec = top_query_spec.replace_io(
                                inputs=new_inputs,
                                output=new_output,
                                serial_only=serial_only,
                            )
                            # TODO: Pass parent_summary too, if applicable.
                            searcher(spec, memory_limits=memory_limits)

        results_cache.flush()
        return results_cache.overlay


def _iter_vector_shape_args(
    bank: str, outer_shape: Sequence[int], dtype
) -> Iterable[dict]:
    system = system_config.current_system()
    vector_bytes: Optional[int] = system.banks[bank].vector_bytes
    if vector_bytes is None:
        yield {}
        return
    for shape in gen_vector_shapes(outer_shape, dtype, vector_bytes):
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


def _compute_moves_subgraph(
    executor: concurrent.futures.Executor,
    size: int,
    dt: dtypes.Dtype,
    serial_only: bool,
    downscale: int,
    max_rank: int,
) -> None:
    if max_rank < 2:
        raise ValueError("max_rank must be at least 2")

    target = system_config.current_target()
    limits = pruning.StandardMemoryLimits()

    for rank in range(1, max_rank + 1):
        load_spec = specs.Load(
            source=target.tensor_spec((size,) * rank, dtype=dt),
            destination=target.tensor_spec((size,) * rank, dtype=dt),
            serial_only=serial_only,
        )
        store_spec = specs.Store(
            source=target.tensor_spec((size,) * rank, dtype=dt),
            destination=target.tensor_spec((size,) * rank, dtype=dt),
            serial_only=serial_only,
        )
        zero_spec = specs.Zero(
            destination=target.tensor_spec((size,) * rank, dtype=dt),
            serial_only=serial_only,
        )

        DPTableGraph(executor, downscale, load_spec, top_limits=limits)

        # TODO: Re-enable the following
        # # Zero depends on Store, not Load, so chain those.
        # save_dp_table = DPTableGraph(executor, downscale, store_spec, top_limits=limits)
        # store_zero_cache = DPTableGraph(
        #     executor, downscale, zero_spec, top_limits=limits
        # )


def main(args=None):
    logging.basicConfig(level=logging.INFO)

    if not args:
        args = arg_parser.parse_args()

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
        limits = pruning.StandardMemoryLimits()

        for dt in dtypes.ALL_DTYPES:
            _compute_moves_subgraph(
                executor,
                args.size,
                dt,
                args.serial_only,
                args.downscale,
                args.moves_rank,
            )

            if not args.moves_only:
                matmul_accum_spec = specs.MatmulAccum(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=args.serial_only,
                )
                DPTableGraph(
                    executor, args.downscale, matmul_accum_spec, top_limits=limits
                )

                matmul_spec = specs.Matmul(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=args.serial_only,
                )
                DPTableGraph(
                    executor,
                    args.downscale,
                    matmul_spec,
                    top_limits=limits,
                )

    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()