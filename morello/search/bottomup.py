import abc
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
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import atomicwrites
import dask.distributed

from .. import dtypes, pruning, search_cache, specs, system_config, utils
from ..impl.utils import gen_vector_shapes
from . import dp

if TYPE_CHECKING:
    from .. import impl

MERGE_DIAGONALS = True
BANKS_LEVELS = (("RF", "VRF"), ("L1",), ("GL",))

logger = logging.getLogger(__name__)

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
arg_parser.add_argument("out_path", metavar="OUTCACHE", type=pathlib.Path)


# TODO: Make sure we're staying in the expected or dependent Tables.
class _AssertingSearch(dp.Search):
    """Wraps dp.Search to assert that no cache misses occur across block boundaries."""

    def __init__(
        self, downscale: int, block_coordinate: "Coord", *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.downscale = downscale
        self.block_coordinate = block_coordinate

    def _choose(
        self,
        spec: specs.Spec,
        leaf: "impl.Impl",
        memory_limits: pruning.MemoryLimits,
        **kwargs,
    ) -> tuple[list[tuple["impl.Impl", tuple]], int]:
        spec_expected_block = _spec_coordinate_to_block_coord(
            self.downscale, _spec_to_spec_coordinate(spec)
        )
        assert len(spec_expected_block.dim_coords) == len(
            self.block_coordinate.dim_coords
        ), (
            f"Expected the same number of dim_coords in the Spec: {spec} and "
            f"{_spec_coordinate_to_block_coord(self.downscale, _spec_to_spec_coordinate(spec))}"
        )
        assert len(spec_expected_block.other_coords) == len(
            self.block_coordinate.other_coords
        ), (
            f"Expected the same number of other_coords in the Spec: {spec} and "
            f"{_spec_coordinate_to_block_coord(self.downscale, _spec_to_spec_coordinate(spec))}"
        )
        assert spec_expected_block == self.block_coordinate, (
            f"Cache did not contain {str(spec)}, which should have been computed "
            f"as part of block {spec_expected_block}. Miss occurred while computing "
            f"{self.block_coordinate}."
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
    bank_idxs = []
    for o in spec.operands:
        for idx, bank_group in enumerate(BANKS_LEVELS):
            if o.bank in bank_group:
                bank_idxs.append(idx)
                break
    bank_idxs = tuple(bank_idxs)
    assert len(bank_idxs) == len(spec.operands)

    if isinstance(spec, (specs.Matmul, specs.MatmulAccum)):
        m, k = spec.lhs.dim_sizes
        _, n = spec.rhs.dim_sizes
        return Coord((m, k, n), bank_idxs + (sb,))
    elif isinstance(spec, (specs.Convolution, specs.ConvolutionAccum)):
        b, c, h, w = spec.lhs.dim_sizes
        f, _, fh, fw = spec.rhs.dim_sizes
        return Coord((b, c, f, h, w, fh, fw), bank_idxs + (sb,))
    elif isinstance(spec, (specs.ReduceSum, specs.ReduceSumAccum)):
        return Coord(spec.source.dim_sizes, bank_idxs + (sb,))
    elif isinstance(spec, (specs.Load, specs.Store, specs.Zero)):
        return Coord(spec.output.dim_sizes, bank_idxs + (sb,))
    elif isinstance(spec, specs.Compose):
        raise ValueError("Compose Specs not supported")
    else:
        raise NotImplementedError("Unsupported Spec: " + str(spec))


# TODO: Can we just automatically invert _spec_to_coordinate? Or derive from Spec?
def _destructure_spec_coordinate(
    base_spec: specs.Spec, coord: Coord
) -> tuple[Sequence[tuple[int, ...]], Sequence[Sequence[str]], bool]:
    """Return a modified copy of the given Spec with the given coordinate."""
    system = system_config.current_system()

    spec_dims = coord.dim_coords
    banks: Sequence[Sequence[str]] = [BANKS_LEVELS[b] for b in coord.other_coords[:-1]]
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
    diagonals: list["DPTableGraphDiagonal"]
    downscale: int
    _coords_to_transitive_results: dict[
        Coord, list[Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]]
    ]

    def __init__(
        self,
        dask_client: dask.distributed.Client,
        downscale: int,
        top_query_spec: specs.Spec,
        top_limits: pruning.MemoryLimits,
        base_cache: Optional[
            Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]
        ] = None,
        source_link: Optional["IntertableLink"] = None,
    ):
        self.client = dask_client
        self.downscale = downscale

        rid = f"{random.randint(0, 9999999)}"

        self.diagonals = []
        self._coords_to_transitive_results = {}

        top_spec_pt = _spec_to_spec_coordinate(top_query_spec)
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
                dask_client,
                diagonal_coordinates,
                rid=rid,
                downscale=downscale,
                suffix=suffix,
                top_query_spec=top_query_spec,
                top_limits=top_limits,
                last_diagonal=(self.diagonals[-1] if self.diagonals else None),
                initial_cache=(base_cache if diagonal_idx == 0 else None),
                intertable_link=source_link,
            )
            self.diagonals.append(new_diagonal)
            for idx, coord in enumerate(new_diagonal.coordinates):
                self._coords_to_transitive_results[
                    coord
                ] = new_diagonal.get_transitive_block_cache(idx)
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

    def get_transitive_block_cache(
        self, block_coord: Coord
    ) -> Iterable[Union[search_cache.ScheduleCache, dask.distributed.Future]]:
        return self._coords_to_transitive_results[block_coord]

    @property
    def final_cache(
        self,
    ) -> Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]:
        target = system_config.current_target()
        final_diagonal = self.diagonals[-1]
        assert len(final_diagonal) == 1
        final_caches = final_diagonal.get_transitive_block_cache(0)
        assert final_caches
        if len(final_caches) == 1:
            return final_caches[0]
        return self.client.submit(_merge_caches, final_caches, target)


class DPTableGraphDiagonal:
    coordinates: Sequence[Coord]

    def __init__(
        self,
        dask_client,
        diagonal_coordinates: Iterable[Coord],
        *,
        rid: str,
        downscale: int,
        suffix: str,
        top_query_spec: specs.Spec,
        top_limits: pruning.MemoryLimits,
        last_diagonal: Optional["DPTableGraphDiagonal"] = None,
        initial_cache: Optional[
            Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]
        ] = None,
        intertable_link: Optional["IntertableLink"] = None,
    ):
        self.coordinates = list(diagonal_coordinates)
        self._client = dask_client
        self._blocks = []
        self._merged_result = None

        target = system_config.current_target()

        for bidx, block_coordinate in enumerate(self.coordinates):
            # Each block depends on the blocks in the prior diagonal with no greater
            # dimension, as well as *their* dependencies (transitive deps.).
            last_diag_deps = []
            if last_diagonal:
                last_diag_deps.extend(
                    (
                        f
                        for idx, coord in enumerate(last_diagonal.coordinates)
                        for f in last_diagonal.get_transitive_block_cache(idx)
                        if all(
                            a <= b
                            for a, b in zip(coord.flatten(), block_coordinate.flatten())
                        )
                    )
                )
            if intertable_link:
                last_diag_deps.extend(intertable_link(downscale, block_coordinate))
            if initial_cache:
                last_diag_deps.append(initial_cache)

            # Create the task for the block. The result will be just the new Specs.
            new_task = dask_client.submit(
                _compute_block,
                block_coordinate,
                target,
                top_query_spec,
                top_limits,
                last_diag_deps,
                downscale,
                key=f"block-{suffix}-{bidx}-{rid}",
            )
            self._blocks.append((new_task, last_diag_deps))

    def __len__(self) -> int:
        return len(self.coordinates)

    def get_transitive_block_cache(
        self, idx: int
    ) -> list[Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]]:
        """Returns the result of a block and all of its dependencies.

        This may return more than *just* that block's results or its dependencies.
        """
        target = system_config.current_target()

        if MERGE_DIAGONALS:
            if self._merged_result:
                return [self._merged_result]
            src = []
            added_keys = set()
            for result, deps in self._blocks:
                for d in itertools.chain([result], deps):
                    if not hasattr(d, "key"):
                        src.append(d)
                    elif d.key not in added_keys:
                        src.append(d)
                        added_keys.add(d.key)
            self._merged_result = self._client.submit(_merge_caches, src, target)
            return [self._merged_result]

        result, deps = self._blocks[idx]
        return [result] + deps


class IntertableLink(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, destination_downscale: int, destination_block_coord: Coord
    ) -> Iterable[Union[search_cache.ScheduleCache, dask.distributed.Future]]:
        pass


class DirectIntertableLink(IntertableLink):
    """A callable for linking two same-sized DPTableGraphs one-to-one."""

    def __init__(self, source: DPTableGraph):
        self._source = source

    def __call__(
        self, destination_downscale: int, destination_block_coord: Coord
    ) -> Iterable[Union[search_cache.ScheduleCache, dask.distributed.Future]]:
        # TODO: Ideally this method would accept query Specs and not care if the two
        #   graphs are downscaled or blocked in the same way.
        assert destination_downscale == self._source.downscale
        return self._source.get_transitive_block_cache(destination_block_coord)


class ZeroToStoreLink(IntertableLink):
    def __init__(self, source: DPTableGraph):
        self._source = source
        self._source_bank_dim_max = source.top_block.other_coords[0]

    def __call__(
        self, destination_downscale: int, destination_block_coord: Coord
    ) -> Iterable[Union[search_cache.ScheduleCache, dask.distributed.Future]]:
        # TODO: Ideally this method would accept query Specs and not care if the two
        #   graphs are downscaled or blocked in the same way.
        assert destination_downscale == self._source.downscale
        new_other_coords = (
            self._source_bank_dim_max,
        ) + destination_block_coord.other_coords
        new_coord = Coord(destination_block_coord.dim_coords, new_other_coords)
        return self._source.get_transitive_block_cache(new_coord)


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


def _enumerate_contig_abstractions(
    spec_type: type, banks: Sequence[str], layouts
) -> Iterable[Sequence[Any]]:
    return itertools.product(*[l.all_contiguous_abs_for_shape() for l in layouts])


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

    cache = search_cache.CacheChain(
        [search_cache.InMemoryScheduleCache()] + list(caches)
    )

    with system_config.with_target(target):
        all_banks = target.system.ordered_banks
        op_count = len(top_query_spec.operands)

        # Use dp.Search instead of _AssertingSearch because we don't compute all
        # contiguousnesses for all shapes at the moment.
        # TODO: Turn _AssertingSearch back on with more precise checks.
        # search_obj = _AssertingSearch(downscale, block_coord, cache=cache)
        search_obj = dp.Search(cache=cache)

        assert set(top_limits.available.keys()) == set(all_banks)

        top_query_shape = _spec_to_spec_coordinate(top_query_spec)
        for spec_coord in _spec_coordinates_in_block(downscale, block_coord):
            # The larget block might contain Specs which are larger than the bounding
            # query shape (`top_query_shape`). Skip those.
            if any(
                s > t for s, t in zip(spec_coord.flatten(), top_query_shape.flatten())
            ):
                continue

            (
                operand_shapes,
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
                            # TODO: We should actually step down according to the memory
                            #   used by the best Impl.
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
                            # TODO: Avoid constructing the below list. Reverse the generator.
                            for limits in itertools.product(
                                *[
                                    reversed(
                                        list(
                                            utils.powers_of_two(
                                                top_limits.available[bank]
                                            )
                                        )
                                    )
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


def _moves_subgraph(
    dask_client,
    size: int,
    dt: dtypes.Dtype,
    serial_only: bool,
    downscale: int,
    max_rank: int,
) -> dict[
    int, list[Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]]
]:
    if max_rank < 2:
        raise ValueError("max_rank must be at least 2")

    target = system_config.current_target()
    limits = pruning.StandardMemoryLimits()

    final_caches: dict[
        int, list[Union[search_cache.InMemoryScheduleCache, dask.distributed.Future]]
    ] = {}

    for rank in range(1, max_rank + 1):
        final_caches[rank] = []
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

        load_dp_table = DPTableGraph(
            dask_client, downscale, load_spec, top_limits=limits
        )
        final_caches[rank].append(load_dp_table.final_cache)

        # Zero depends on Store, not Load, so chain those.
        save_dp_table = DPTableGraph(
            dask_client, downscale, store_spec, top_limits=limits
        )
        store_zero_cache = DPTableGraph(
            dask_client,
            downscale,
            zero_spec,
            top_limits=limits,
            source_link=ZeroToStoreLink(save_dp_table),
        )
        final_caches[rank].append(store_zero_cache.final_cache)

    return final_caches


def main():
    logging.basicConfig(level=logging.INFO)

    args = arg_parser.parse_args()

    start = time.time()

    system_config.set_current_target("cpu")
    target = system_config.current_target()

    loaded_moves_cache: Optional[
        Union[dask.distributed.Future, search_cache.ScheduleCache]
    ] = None
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
        if loaded_moves_cache:
            scatter_start = time.time()
            logger.info("Scattering loaded moves cache")
            loaded_moves_cache = dask_client.scatter(loaded_moves_cache)  # type: ignore
            logger.info(
                "Scattering loaded moves cache took %.2f seconds",
                time.time() - scatter_start,
            )

        limits = pruning.StandardMemoryLimits()
        final_caches_to_merge = []

        for dt in dtypes.ALL_DTYPES:
            rank2_cache: Union[dask.distributed.Future, search_cache.ScheduleCache]
            if loaded_moves_cache:
                rank2_cache = loaded_moves_cache
            else:
                per_rank_moves_graphs = _moves_subgraph(
                    dask_client,
                    args.size,
                    dt,
                    args.serial_only,
                    args.downscale,
                    args.moves_rank,
                )
                rank2_cache = dask_client.submit(
                    _merge_caches, per_rank_moves_graphs[2], target
                )
                for rank, a in per_rank_moves_graphs.items():
                    if rank != 2:
                        final_caches_to_merge.extend(a)

            if args.moves_only:
                final_caches_to_merge.append(rank2_cache)
            else:
                matmul_accum_spec = specs.MatmulAccum(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=args.serial_only,
                )
                matmul_accum_dp_table = DPTableGraph(
                    dask_client,
                    args.downscale,
                    matmul_accum_spec,
                    top_limits=limits,
                    base_cache=rank2_cache,
                )

                matmul_spec = specs.Matmul(
                    lhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    rhs=target.tensor_spec((args.size, args.size), dtype=dt),
                    output=target.tensor_spec((args.size, args.size), dtype=dt),
                    serial_only=args.serial_only,
                )
                transitive_matmul_graph_table = DPTableGraph(
                    dask_client,
                    args.downscale,
                    matmul_spec,
                    top_limits=limits,
                    source_link=DirectIntertableLink(matmul_accum_dp_table),
                ).final_cache
                final_caches_to_merge.append(transitive_matmul_graph_table)

        combined_cache = dask_client.submit(
            _merge_caches, final_caches_to_merge, target
        ).result()

        with atomicwrites.atomic_write(args.out_path, mode="wb", overwrite=True) as fo:
            pickle.dump(combined_cache, fo)
        logger.info("Saving cache to: %s", str(args.out_path))

    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
