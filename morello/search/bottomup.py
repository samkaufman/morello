import argparse
import copy
import dataclasses
import itertools
import logging
import math
import pathlib
import pickle
import secrets
import subprocess
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterable, Optional, Union

import atomicwrites
import dask.distributed

from .. import dtypes, pruning, search_cache, specs, system_config, utils
from . import dp

if TYPE_CHECKING:
    from .. import impl

FACTOR = 4
USE_TINYMAPS = True

logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--deploy-k8s", action="store_true")
arg_parser.add_argument("--moves-only", action="store_true")
arg_parser.add_argument("--size", type=int, default=512)
arg_parser.add_argument("--image-tag", "-t", type=str, default="samkaufman/morello")


# TODO: Make sure we're staying in the expected or dependent Tables.
class _AssertingSearch(dp.Search):
    """This extends dp.Search to assert that cache misses occur in a single block."""

    def __init__(self, block_coordinate: Optional["Coord"], *args, **kwargs) -> None:
        self.block_coordinate = block_coordinate
        super().__init__(*args, **kwargs)

    def _choose(
        self,
        spec: specs.Spec,
        leaf: "impl.Impl",
        memory_limits: pruning.MemoryLimits,
        **kwargs,
    ) -> tuple[list[tuple["impl.Impl", tuple]], int]:
        assert self.block_coordinate is None or len(
            _spec_coordinate_to_block_coord(_spec_to_spec_coordinate(spec)).dim_coords
        ) == len(self.block_coordinate.dim_coords), (
            f"Expected the same number of dim_coords in the Spec: {spec} and "
            f"{_spec_coordinate_to_block_coord(_spec_to_spec_coordinate(spec))}"
        )
        assert self.block_coordinate is None or len(
            _spec_coordinate_to_block_coord(_spec_to_spec_coordinate(spec)).other_coords
        ) == len(self.block_coordinate.other_coords), (
            f"Expected the same number of other_coords in the Spec: {spec} and "
            f"{_spec_coordinate_to_block_coord(_spec_to_spec_coordinate(spec))}"
        )
        assert (
            self.block_coordinate is None
            or _spec_coordinate_to_block_coord(_spec_to_spec_coordinate(spec))
            == self.block_coordinate
        ), (
            "Unexpected miss at block coordinate "
            f"{_spec_coordinate_to_block_coord(_spec_to_spec_coordinate(spec))} from "
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


def _spec_coordinates_in_block(block: Coord) -> Iterable[Coord]:
    for log_pts in itertools.product(
        *[range(b * FACTOR, (b + 1) * FACTOR) for b in block.dim_coords]
    ):
        yield Coord(tuple(2**p for p in log_pts), block.other_coords)


def _spec_coordinate_to_block_coord(spec_coord: Coord) -> Coord:
    # Scale the tensor dimensions down.
    # TODO: Avoid conversion to floats. (Maybe: switch to bit-flipping.)
    dim_coords = tuple(int(math.log2(v)) // FACTOR for v in spec_coord.dim_coords)
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


def _compute_all(
    dask_client: dask.distributed.Client,
    top_query_spec: specs.Spec,
    top_limits: pruning.MemoryLimits,
    cache: Union[search_cache.InMemoryScheduleCache, dask.distributed.Future],
) -> dask.distributed.Future:
    # Walk frontiers, with synchronization after each.

    target = system_config.current_target()

    top_spec_pt = _spec_to_spec_coordinate(top_query_spec)
    top_block = _spec_coordinate_to_block_coord(top_spec_pt)
    slice_idx = 0
    while True:
        block_diagonal = [
            Coord.structure(d, top_query_spec)
            for d in utils.sum_seqs(maxes=top_block.flatten(), total=slice_idx)
        ]
        if not block_diagonal:
            break
        block_results = dask_client.map(
            _compute_block,
            block_diagonal,
            [target] * len(block_diagonal),
            [top_query_spec] * len(block_diagonal),
            [top_limits] * len(block_diagonal),
            [cache] * len(block_diagonal),
            key=f"block-{type(top_query_spec).__name__}",
        )
        cache = dask_client.submit(_merge_block_results, cache, block_results, target)
        slice_idx += 1

    assert not isinstance(cache, search_cache.InMemoryScheduleCache)
    return cache


def _merge_block_results(
    base_cache: search_cache.InMemoryScheduleCache,
    block_results: Sequence[search_cache.InMemoryScheduleCache],
    target: system_config.Target,
) -> search_cache.InMemoryScheduleCache:
    with system_config.with_target(target):
        cache = copy.deepcopy(base_cache)
        for block_result in block_results:
            cache.update(block_result)
        return cache


# TODO: Can we design this so that we don't need to reproduce the logic of the top-down
#   actions expansion? Hate having the logic in two places.
def _compute_block(
    block_coord: Coord,
    target: system_config.Target,
    top_query_spec: specs.Spec,
    top_limits: pruning.MemoryLimits,
    cache: search_cache.ScheduleCache,
) -> search_cache.ScheduleCache:
    """Compute a block given an input cache and return an overlay cache."""
    assert isinstance(top_limits, pruning.StandardMemoryLimits)

    with system_config.with_target(target):
        all_banks = target.system.ordered_banks
        op_count = len(top_query_spec.operands)

        # Wrap the cache so that updates end up in a separate, "overlay" cache.
        cache = search_cache.CacheChain([search_cache.InMemoryScheduleCache(), cache])

        search_obj = _AssertingSearch(block_coord, cache=cache)

        assert set(top_limits.available.keys()) == set(all_banks)

        top_query_shape = _spec_to_spec_coordinate(top_query_spec)
        for spec_coord in _spec_coordinates_in_block(block_coord):
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
                            if USE_TINYMAPS:
                                limits_map = utils.TinyMap(all_banks, limits)
                            else:
                                limits_map = dict(zip(all_banks, limits))

                            limits_map = pruning.StandardMemoryLimits(limits_map)

                            spec = top_query_spec.replace_io(
                                inputs=new_inputs,
                                output=new_output,
                                serial_only=serial_only,
                            )
                            # TODO: Pass parent_summary too, if applicable.
                            # print("Calling search on Spec: " + str(spec))
                            search_obj(spec, memory_limits=limits_map)

        return cache.caches[0]


def _make_docker_image(image_tag: str) -> str:
    current_dir = pathlib.Path(__file__).parent.resolve().parent.parent
    assert (
        current_dir / "Dockerfile"
    ).is_file(), f"{current_dir} does not contain a Dockerfile"

    tag = f"{image_tag}:dep{secrets.token_hex(14)}"
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

    tag = _make_docker_image(image_tag)

    logger.info("Starting the Kubernetes cluster")
    cluster = KubeCluster(
        make_pod_spec(
            image=tag,
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

    if args.deploy_k8s:
        assert args.image_tag
        cluster = _deploy_k8s_cluster(args.image_tag)
    else:
        cluster = dask.distributed.LocalCluster(threads_per_worker=1)

    with cluster, dask.distributed.Client(cluster) as dask_client:
        limits = pruning.StandardMemoryLimits()

        per_dt_caches = []

        for dt in dtypes.ALL_DTYPES:
            dt_cache = search_cache.InMemoryScheduleCache()

            # TODO: Checkpoint each of the following stages
            futures = []
            spec0 = specs.Load(
                source=target.tensor_spec((args.size, args.size), dtype=dt),
                destination=target.tensor_spec((args.size, args.size), dtype=dt),
                serial_only=False,
            )
            futures.append(
                _compute_all(dask_client, spec0, top_limits=limits, cache=dt_cache)
            )

            spec1 = specs.Store(
                source=target.tensor_spec((args.size, args.size), dtype=dt),
                destination=target.tensor_spec((args.size, args.size), dtype=dt),
                serial_only=False,
            )
            futures.append(
                _compute_all(dask_client, spec1, top_limits=limits, cache=dt_cache)
            )

            # Merge the Load and Store tables.
            dt_cache = dask_client.submit(
                _merge_block_results, dt_cache, futures, target
            )

            spec2 = specs.Zero(
                destination=target.tensor_spec((args.size, args.size), dtype=dt),
                serial_only=False,
            )
            dt_cache = _compute_all(
                dask_client, spec2, top_limits=limits, cache=dt_cache
            )

            if not args.moves_only:
                spec3 = specs.MatmulAccum(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=False,
                )
                dt_cache = _compute_all(
                    dask_client, spec3, top_limits=limits, cache=dt_cache
                )

                spec4 = specs.Matmul(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=False,
                )
                dt_cache = _compute_all(
                    dask_client, spec4, top_limits=limits, cache=dt_cache
                )

            per_dt_caches.append(dt_cache)

        assert len(per_dt_caches) == 2

        combined_cache = dask_client.submit(
            _merge_block_results,
            search_cache.InMemoryScheduleCache(),
            per_dt_caches,
            target,
        ).result()

        out_path = pathlib.Path("./cache.pkl")
        with atomicwrites.atomic_write(out_path, mode="wb", overwrite=True) as fo:
            pickle.dump(combined_cache, fo)
        logger.info("Saving cache to: %s", str(out_path))

    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
