import argparse
import contextlib
import copy
import dataclasses
import itertools
import logging
import math
import pathlib
import pickle
import random
import secrets
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterable, Optional, Union

import atomicwrites
import dask.distributed

from .. import dtypes, pruning, search_cache, specs, system_config, utils
from . import dp

if TYPE_CHECKING:
    from .. import impl

logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--scheduler", type=str)
arg_parser.add_argument("--downscale", type=int, default=1)
arg_parser.add_argument("--deploy-k8s", action="store_true")
arg_parser.add_argument("--deploy-k8s-with-custom-image", type=str)
arg_parser.add_argument("--moves-only", action="store_true")
arg_parser.add_argument("--size", type=int, default=512)
arg_parser.add_argument("--image-name", "-t", type=str, default="samkaufman/morello")
arg_parser.add_argument("--moves-cache", metavar="CACHE", type=pathlib.Path)
arg_parser.add_argument("out_path", metavar="OUTCACHE", type=pathlib.Path)


# TODO: Make sure we're staying in the expected or dependent Tables.
class _AssertingSearch(dp.Search):
    """Wraps dp.Search to assert that no cache misses occur across block boundaries."""

    def __init__(
        self, downscale: int, block_coordinate: Optional["Coord"], *args, **kwargs
    ) -> None:
        self.block_coordinate = block_coordinate
        super().__init__(*args, **kwargs)
        self.downscale = downscale

    def _choose(
        self,
        spec: specs.Spec,
        leaf: "impl.Impl",
        memory_limits: pruning.MemoryLimits,
        **kwargs,
    ) -> tuple[list[tuple["impl.Impl", tuple]], int]:
        assert self.block_coordinate is None or len(
            _spec_coordinate_to_block_coord(
                self.downscale, _spec_to_spec_coordinate(spec)
            ).dim_coords
        ) == len(self.block_coordinate.dim_coords), (
            f"Expected the same number of dim_coords in the Spec: {spec} and "
            f"{_spec_coordinate_to_block_coord(self.downscale, _spec_to_spec_coordinate(spec))}"
        )
        assert self.block_coordinate is None or len(
            _spec_coordinate_to_block_coord(
                self.downscale, _spec_to_spec_coordinate(spec)
            ).other_coords
        ) == len(self.block_coordinate.other_coords), (
            f"Expected the same number of other_coords in the Spec: {spec} and "
            f"{_spec_coordinate_to_block_coord(self.downscale, _spec_to_spec_coordinate(spec))}"
        )
        assert (
            self.block_coordinate is None
            or _spec_coordinate_to_block_coord(
                self.downscale, _spec_to_spec_coordinate(spec)
            )
            == self.block_coordinate
        ), (
            "Unexpected miss at block coordinate "
            f"{_spec_coordinate_to_block_coord(self.downscale, _spec_to_spec_coordinate(spec))} from "
            f"Spec coordinate {_spec_to_spec_coordinate(spec)}; "
            f"search expected block coord. {self.block_coordinate}; "
            f"Spec = {spec}"
        )
        return super()._choose(spec, leaf, memory_limits, **kwargs)


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
        yield Coord(tuple(2**p for p in log_pts), block_coord.other_coords)


def _spec_coordinate_to_block_coord(downscale: int, spec_coord: Coord) -> Coord:
    # Scale the tensor dimensions down.
    # TODO: Avoid conversion to floats. (Maybe: switch to bit-flipping.)
    dim_coords = tuple(int(math.log2(v)) // downscale for v in spec_coord.dim_coords)
    return Coord(dim_coords, spec_coord.other_coords)


def _spec_to_spec_coordinate(spec: specs.Spec) -> Coord:
    system = system_config.current_system()

    sb = 0 if spec.serial_only else 1
    banks = tuple(system.ordered_banks.index(o.bank) for o in spec.operands)
    if isinstance(spec, (specs.Matmul, specs.MatmulAccum)):
        m, k = spec.lhs.dim_sizes
        _, n = spec.rhs.dim_sizes
        return Coord((m, k, n), banks + (sb,))
    elif isinstance(spec, (specs.Convolution, specs.ConvolutionAccum)):
        b, c, h, w = spec.lhs.dim_sizes
        f, _, fh, fw = spec.rhs.dim_sizes
        return Coord((b, c, f, h, w, fh, fw), banks + (sb,))
    elif isinstance(spec, (specs.ReduceSum, specs.ReduceSumAccum)):
        return Coord(spec.source.dim_sizes, banks + (sb,))
    elif isinstance(spec, (specs.Load, specs.Store, specs.Zero)):
        return Coord(spec.output.dim_sizes, banks + (sb,))
    elif isinstance(spec, specs.Compose):
        raise ValueError("Compose Specs not supported")
    else:
        raise NotImplementedError("Unsupported Spec: " + str(spec))


# TODO: Can we just automatically invert _spec_to_coordinate? Or derive from Spec?
def _destructure_spec_coordinate(
    base_spec: specs.Spec, coord: Coord
) -> tuple[Sequence[tuple[int, ...]], Sequence[str], bool]:
    """Return a modified copy of the given Spec with the given coordinate."""
    system = system_config.current_system()

    spec_dims = coord.dim_coords
    banks = [system.ordered_banks[b] for b in coord.other_coords[:-1]]
    serial = coord.other_coords[-1] == 0

    if isinstance(base_spec, (specs.Matmul, specs.MatmulAccum)):
        return (
            [spec_dims[:2], spec_dims[1:3], (spec_dims[0], spec_dims[2])],
            banks,
            serial,
        )
    elif isinstance(base_spec, (specs.Convolution, specs.ConvolutionAccum)):
        b, c, f, h, w, fh, fw = spec_dims
        return [(b, c, h, w), (f, c, fh, fw), (b, f, h, w)], banks, serial
    elif isinstance(base_spec, (specs.ReduceSum, specs.ReduceSumAccum)):
        return (spec_dims, spec_dims[:-1]), banks, serial
    elif isinstance(base_spec, (specs.Load, specs.Store)):
        return (spec_dims, spec_dims), banks, serial
    elif isinstance(base_spec, specs.Zero):
        return (spec_dims,), banks, serial
    elif isinstance(base_spec, specs.Compose):
        raise ValueError("Compose Specs not supported")
    else:
        raise NotImplementedError("Unsupported Spec: " + str(base_spec))


class DPTableGraph:
    final_cache: dask.distributed.Future
    _block_caches: dict[Coord, dask.distributed.Future]

    def __init__(
        self,
        dask_client: dask.distributed.Client,
        downscale: int,
        top_query_spec: specs.Spec,
        top_limits: pruning.MemoryLimits,
        base_cache: Optional[
            Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]
        ],
    ):
        self._block_caches = {}

        target = system_config.current_target()
        rid = f"{random.randint(0, 99999)}"

        top_spec_pt = _spec_to_spec_coordinate(top_query_spec)
        top_block = _spec_coordinate_to_block_coord(downscale, top_spec_pt)
        diagonal_idx = 0
        last_diagonal: list[tuple[Coord, dask.distributed.Future]] = []
        while True:
            new_diagonal: list[tuple[Coord, dask.distributed.Future]] = []
            block_diagonal = [
                Coord.structure(d, top_query_spec)
                for d in utils.sum_seqs(maxes=top_block.flatten(), total=diagonal_idx)
            ]
            if not block_diagonal:
                break
            for bidx, block in enumerate(block_diagonal):
                # Each block depends on the blocks in the prior diagonal with no greater
                # dimension, as well as *their* dependencies (transitive deps.).
                # (The following loop is slow, but fast enough in practice.)
                last_diag_deps: list[
                    Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]
                ] = [
                    f
                    for c, f in last_diagonal
                    if all(a <= b for a, b in zip(c.flatten(), block.flatten()))
                ]
                if diagonal_idx == 0:
                    assert not last_diag_deps
                    if base_cache:
                        last_diag_deps.append(base_cache)

                # Create the task for the block.
                suffix = f"{type(top_query_spec).__name__}-{diagonal_idx}-{bidx}-{rid}"
                new_task = dask_client.submit(
                    _compute_block,
                    block,
                    target,
                    top_query_spec,
                    top_limits,
                    last_diag_deps,
                    downscale,
                    key=f"block-{suffix}",
                )
                merged_resulting_cache = dask_client.submit(
                    _merge_caches,
                    last_diag_deps + [new_task],
                    target,
                    key=f"resultmerge-{suffix}",
                )
                new_diagonal.append((block, merged_resulting_cache))
                self._block_caches[block] = merged_resulting_cache
            diagonal_idx += 1
            last_diagonal = new_diagonal

        # The final result is the transitive deps. of the final, single-element diagnonal.
        assert len(last_diagonal) == 1
        self.final_cache = last_diagonal[0][1]


def _merge_caches(
    caches: Sequence[search_cache.ScheduleCache],
    target: system_config.Target,
) -> search_cache.InMemoryScheduleCache:
    if not caches:
        raise ValueError("No caches to merge")

    if len(caches) == 1 and isinstance(caches[0], search_cache.InMemoryScheduleCache):
        return caches[0]
    with system_config.with_target(target):
        if isinstance(caches[0], search_cache.InMemoryScheduleCache):
            cache = copy.deepcopy(caches[0])
            caches = caches[1:]
        else:
            cache = search_cache.InMemoryScheduleCache()
        for block_result in caches:
            cache.update(block_result)
        return cache


# TODO: Can we design this so that we don't need to reproduce the logic of the top-down
#   actions expansion? Hate having the logic in two places.
def _compute_block(
    block_coord: Coord,
    target: system_config.Target,
    top_query_spec: specs.Spec,
    top_limits: pruning.MemoryLimits,
    caches: Sequence[search_cache.InMemoryScheduleCache],
    downscale: int,
) -> search_cache.ScheduleCache:
    """Compute a block given an input cache and return an overlay cache.

    The returned cache does not contain Specs in the provided `cache`; just the
    additional Specs computed for this block.
    """
    assert isinstance(top_limits, pruning.StandardMemoryLimits)

    assert caches
    cache = caches[0]
    if len(caches) > 1:
        cache = _merge_caches(caches, target)

    with system_config.with_target(target):
        # Wrap the cache so that updates end up in a separate, "overlay" cache.
        cache = search_cache.CacheChain([search_cache.InMemoryScheduleCache(), cache])

        all_banks = target.system.ordered_banks
        op_count = len(top_query_spec.operands)

        search_obj = _AssertingSearch(downscale, block_coord, cache=cache)

        assert set(top_limits.available.keys()) == set(all_banks)

        top_query_shape = _spec_to_spec_coordinate(top_query_spec)
        for spec_coord in _spec_coordinates_in_block(downscale, block_coord):
            # The larget block might contain Specs which are larger than the bounding
            # query shape (`top_query_shape`). Skip those.
            if any(
                s > t for s, t in zip(spec_coord.flatten(), top_query_shape.flatten())
            ):
                continue

            operand_shapes, tensor_banks, serial_only = _destructure_spec_coordinate(
                top_query_spec, spec_coord
            )
            for layouts in itertools.product(
                *[target.all_layouts_for_shape(shp) for shp in operand_shapes]
            ):
                for contigious_abstractions in itertools.product(
                    *[l.all_contiguous_abs_for_shape() for l in layouts]
                ):
                    for aligneds in itertools.product([True, False], repeat=op_count):
                        # TODO: We should actually step down according to the memory
                        #   used by the best Impl.
                        new_operands = tuple(
                            target.tensor_spec(
                                operand_shapes[i],
                                dtype=top_query_spec.operands[i].dtype,
                                contiguous_abs=contigious_abstractions[i],
                                aligned=aligneds[i],
                                bank=tensor_banks[i],
                                layout=layouts[i],
                            )
                            for i in range(op_count)
                        )
                        new_inputs, new_output = new_operands[:-1], new_operands[-1]
                        for limits in itertools.product(
                            *[
                                utils.powers_of_two(top_limits.available[bank])
                                for bank in all_banks
                            ]
                        ):
                            limits_map = pruning.StandardMemoryLimits(
                                utils.TinyMap(all_banks, limits)
                            )

                            spec = top_query_spec.replace_io(
                                inputs=new_inputs,
                                output=new_output,
                                serial_only=serial_only,
                            )
                            # TODO: Pass parent_summary too, if applicable.
                            # print("Calling search on Spec: " + str(spec))
                            search_obj(spec, memory_limits=limits_map)

        return cache.caches[0]


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
            memory_limit="6G",
            memory_request="3G",
            cpu_limit=1,
            cpu_request=1,
            extra_container_config={"imagePullPolicy": "Always"},
        )
    )
    cluster.adapt(minimum=1, maximum=256)
    logger.info("Started the Kubernetes cluster")
    logger.info("Kubernetes scheduler address: %s", cluster.scheduler_address)
    return cluster


def main():
    logging.basicConfig(level=logging.INFO)

    args = arg_parser.parse_args()

    start = time.time()

    system_config.set_current_target("cpu")
    target = system_config.current_target()

    loaded_moves_cache = None
    if args.moves_cache:
        with args.moves_cache.open(mode="rb") as fo:
            loaded_moves_cache = pickle.load(fo)

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

    with cluster, dask.distributed.Client(cluster_address) as dask_client:
        limits = pruning.StandardMemoryLimits()

        per_dt_caches = []

        for dt in dtypes.ALL_DTYPES:
            if loaded_moves_cache:
                dt_cache = loaded_moves_cache
            else:
                dt_cache = search_cache.InMemoryScheduleCache()
                load_spec = specs.Load(
                    source=target.tensor_spec((args.size, args.size), dtype=dt),
                    destination=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=False,
                )
                store_spec = specs.Store(
                    source=target.tensor_spec((args.size, args.size), dtype=dt),
                    destination=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=False,
                )
                zero_spec = specs.Zero(
                    destination=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=True,
                )

                # Zero depends on Store, not Load, so chain those.
                store_zero_cache = DPTableGraph(
                    dask_client,
                    args.downscale,
                    store_spec,
                    top_limits=limits,
                    base_cache=dt_cache,
                ).final_cache
                store_zero_cache = DPTableGraph(
                    dask_client,
                    args.downscale,
                    zero_spec,
                    top_limits=limits,
                    base_cache=store_zero_cache,
                ).final_cache

                # Merge the Load table with the Store/Zero table.
                dt_cache = dask_client.submit(
                    _merge_caches,
                    [
                        dt_cache,
                        store_zero_cache,
                        DPTableGraph(
                            dask_client,
                            args.downscale,
                            load_spec,
                            top_limits=limits,
                            base_cache=dt_cache,
                        ).final_cache,
                    ],
                    target,
                )

            if not args.moves_only:
                matmul_accum_spec = specs.MatmulAccum(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=False,
                )
                dt_cache = DPTableGraph(
                    dask_client,
                    args.downscale,
                    matmul_accum_spec,
                    top_limits=limits,
                    base_cache=dt_cache,
                ).final_cache

                matmul_spec = specs.Matmul(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=False,
                )
                dt_cache = DPTableGraph(
                    dask_client,
                    args.downscale,
                    matmul_spec,
                    top_limits=limits,
                    base_cache=dt_cache,
                ).final_cache

            per_dt_caches.append(dt_cache)

        assert len(per_dt_caches) == 2

        combined_cache = dask_client.submit(
            _merge_caches, per_dt_caches, target
        ).result()

        with atomicwrites.atomic_write(args.out_path, mode="wb", overwrite=True) as fo:
            pickle.dump(combined_cache, fo)
        logger.info("Saving cache to: %s", str(args.out_path))

    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
