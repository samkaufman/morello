use crate::common::{DimSize, Shape};
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::tiling::spec_app_with_argument_overrides;
use crate::scheduling::{tile_to_apply_err, ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{Target, MEMORY_COUNT};
use crate::tensorspec::{self, TensorSpec};
use crate::tiling::Tiling;
use crate::views::{BoundaryTile, Param, Tensor, View, ViewE};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
#[serde(bound(
    deserialize = "Tgt::Memory: Deserialize<'de>",
    serialize = "Tgt::Memory: Serialize"
))]
pub struct ParallelSplitSoftmaxDenominatorAndMax<Tgt: Target> {
    pub k: DimSize,
    pub partials_level: Tgt::Memory,
    pub partials_layout: Layout,
    pub partials_vector_size: Option<DimSize>,
}

impl<Tgt: Target> ActionT<Tgt> for ParallelSplitSoftmaxDenominatorAndMax<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let LogicalSpec::Primitive(
            PrimitiveBasics {
                typ:
                    PrimitiveSpecType::SoftmaxDenominatorAndMax {
                        scan_dim,
                        accum: false,
                    },
                spec_shape,
                dtypes,
            },
            auxes,
            serial_only,
        ) = &spec.0
        else {
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ParallelSplitSoftmaxDenominatorAndMax only applies to non-accumulating SoftmaxDenominatorAndMax",
            ))));
        };

        if *serial_only {
            return Err(ApplyError::NotApplicable(NotApplicableReason::SerialOnly));
        }
        let scan_dim_us = usize::from(*scan_dim);
        if scan_dim_us >= spec_shape.len() {
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "scan_dim is out of bounds",
            ))));
        }

        let scan_len = spec_shape[scan_dim_us];
        if self.k >= scan_len {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::TileShapeIsLarger,
            ));
        }

        let parameters = spec.0.parameters();
        let dtype = dtypes[0];
        if dtypes.iter().any(|&d| d != dtype) {
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "parallel online softmax split requires matching dtypes",
            ))));
        }

        let full_chunks = scan_len.get() / self.k.get();
        let tail_size = scan_len.get() % self.k.get();
        let scan_tile_count = scan_len.get().div_ceil(self.k.get());
        let mut partial_shape = Shape::from_slice(spec_shape);
        partial_shape[scan_dim_us] = DimSize::new(scan_tile_count).unwrap();
        let partial_max = Rc::new(Tensor::new(
            TensorSpec::new_canon_checked(
                partial_shape.clone(),
                dtype,
                self.partials_level,
                self.partials_layout.clone(),
                self.partials_vector_size,
            )
            .map_err(tensorspec_error_to_apply_err)?,
        ));
        let partial_denom = Rc::new(Tensor::new(
            TensorSpec::new_canon_checked(
                partial_shape.clone(),
                dtype,
                self.partials_level,
                self.partials_layout.clone(),
                self.partials_vector_size,
            )
            .map_err(tensorspec_error_to_apply_err)?,
        ));
        let lowered_limits =
            lowered_memory_limits::<Tgt, _>(&spec.1, [partial_max.spec(), partial_denom.spec()])?;

        let mut input_tile_shape = Shape::from(vec![nz!(1u32); spec_shape.len()]);
        input_tile_shape[scan_dim_us] = self.k;
        let partial_tile_shape = Shape::from(vec![nz!(1u32); partial_shape.len()]);

        let input_tile = LoopTile {
            parameter_index: 0,
            axes: (0..u8::try_from(spec_shape.len()).unwrap()).collect(),
            tile: Tiling::new_simple(input_tile_shape.clone())
                .apply_main(Param::new(0, parameters[0].clone()))
                .map(|v| v.boxed_viewe())
                .map_err(tile_to_apply_err)?,
        };
        let partial_max_tile = LoopTile {
            parameter_index: 1,
            axes: (0..u8::try_from(spec_shape.len()).unwrap()).collect(),
            tile: Tiling::new_simple(partial_tile_shape.clone())
                .apply_main(partial_max.as_ref().clone())
                .map(|v| v.boxed_viewe())
                .map_err(tile_to_apply_err)?,
        };
        let partial_denom_tile = LoopTile {
            parameter_index: 2,
            axes: (0..u8::try_from(spec_shape.len()).unwrap()).collect(),
            tile: Tiling::new_simple(partial_tile_shape)
                .apply_main(partial_denom.as_ref().clone())
                .map(|v| v.boxed_viewe())
                .map_err(tile_to_apply_err)?,
        };

        let mut partial_body_spec = Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::SoftmaxDenominatorAndMax {
                        scan_dim: *scan_dim,
                        accum: false,
                    },
                    spec_shape: input_tile_shape,
                    dtypes: dtypes.clone(),
                },
                vec![
                    auxes[0].clone(),
                    partial_max.spec().aux.clone(),
                    partial_denom.spec().aux.clone(),
                ],
                true,
            ),
            lowered_limits.clone(),
        );
        partial_body_spec
            .canonicalize()
            .map_err(|_| ApplyError::NotApplicable(NotApplicableReason::Other(None)))?;
        let partial_body = SpecApp::new(
            partial_body_spec,
            [
                ViewE::Tile(input_tile.tile.clone()),
                ViewE::Tile(partial_max_tile.tile.clone()),
                ViewE::Tile(partial_denom_tile.tile.clone()),
            ],
        );
        let mut partial_loop_bodies = vec![partial_body.into()];
        if tail_size != 0 {
            partial_loop_bodies.push(tail_body(
                spec,
                partial_max.as_ref(),
                partial_denom.as_ref(),
                full_chunks,
                tail_size,
                lowered_limits.clone(),
            )?);
        }

        let partial_loop = ImplNode::Loop(Loop {
            tiles: vec![input_tile, partial_max_tile, partial_denom_tile],
            bodies: partial_loop_bodies,
            parallel: true,
            spec: None,
        });

        let mut combine_spec = Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::SoftmaxDenominatorAndMaxFromParts {
                        scan_dim: *scan_dim,
                        accum: false,
                    },
                    spec_shape: partial_shape,
                    dtypes: vec![dtype; 4],
                },
                vec![
                    partial_max.spec().aux.clone(),
                    partial_denom.spec().aux.clone(),
                    auxes[1].clone(),
                    auxes[2].clone(),
                ],
                false,
            ),
            lowered_limits,
        );
        combine_spec
            .canonicalize()
            .map_err(|_| ApplyError::NotApplicable(NotApplicableReason::Other(None)))?;
        let combine = SpecApp::new(
            combine_spec,
            [
                ViewE::from(partial_max.as_ref().clone()),
                ViewE::from(partial_denom.as_ref().clone()),
                ViewE::from(Param::new(1, parameters[1].clone())),
                ViewE::from(Param::new(2, parameters[2].clone())),
            ],
        );

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![partial_loop, combine.into()],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![partial_max, partial_denom],
            }],
            spec: Some(spec.clone()),
        }))
    }
}

fn tail_body<Tgt: Target>(
    spec: &Spec<Tgt>,
    partial_max: &Tensor<Tgt>,
    partial_denom: &Tensor<Tgt>,
    full_chunks: u32,
    tail_len: u32,
    memory_limits: MemoryLimits,
) -> Result<ImplNode<Tgt>, ApplyError> {
    let LogicalSpec::Primitive(
        PrimitiveBasics {
            typ:
                PrimitiveSpecType::SoftmaxDenominatorAndMax {
                    scan_dim,
                    accum: false,
                },
            spec_shape,
            dtypes,
        },
        ..,
    ) = &spec.0
    else {
        unreachable!("tail_body is only built for non-accumulating SoftmaxDenominatorAndMax");
    };
    let scan_dim_us = usize::from(*scan_dim);
    let parameters = spec.0.parameters();
    let partial_shape = partial_max.spec().shape();
    debug_assert_eq!(partial_shape, partial_denom.spec().shape());

    let tail_dim = DimSize::new(tail_len).unwrap();
    let mut input_shape = Shape::from_slice(spec_shape);
    input_shape[scan_dim_us] = tail_dim;
    let mut partial_output_shape = Shape::from_slice(partial_shape);
    partial_output_shape[scan_dim_us] = DimSize::new(1).unwrap();

    let mut input_offsets = vec![0; spec_shape.len()];
    input_offsets[scan_dim_us] = spec_shape[scan_dim_us].get() - tail_len;
    let mut partial_offsets = vec![0; partial_shape.len()];
    partial_offsets[scan_dim_us] = full_chunks;

    let input_boundary = BoundaryTile::new(
        input_shape.clone(),
        input_offsets,
        Param::new(0, parameters[0].clone()),
    )
    .map_err(tile_to_apply_err)?;
    let partial_max_boundary = BoundaryTile::new(
        partial_output_shape.clone(),
        partial_offsets.clone(),
        partial_max.clone(),
    )
    .map_err(tile_to_apply_err)?;
    let partial_denom_boundary =
        BoundaryTile::new(partial_output_shape, partial_offsets, partial_denom.clone())
            .map_err(tile_to_apply_err)?;

    let body_spec = Spec(
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::SoftmaxDenominatorAndMax {
                    scan_dim: *scan_dim,
                    accum: false,
                },
                spec_shape: input_shape,
                dtypes: dtypes.to_vec(),
            },
            vec![
                input_boundary.spec().aux.clone(),
                partial_max_boundary.spec().aux.clone(),
                partial_denom_boundary.spec().aux.clone(),
            ],
            true,
        ),
        memory_limits,
    );

    spec_app_with_argument_overrides(
        body_spec,
        [
            (0, ViewE::from(input_boundary)),
            (1, ViewE::from(partial_max_boundary)),
            (2, ViewE::from(partial_denom_boundary)),
        ],
    )
}

fn tensorspec_error_to_apply_err(error: tensorspec::CanonicalizeError) -> ApplyError {
    match error {
        tensorspec::CanonicalizeError::VectorSizeInvalid => {
            ApplyError::NotApplicable(NotApplicableReason::VectorSizeInvalid)
        }
        tensorspec::CanonicalizeError::VectorSizeVolumeIncompatible => {
            ApplyError::NotApplicable(NotApplicableReason::VectorSizeVolumeIncompatible)
        }
        tensorspec::CanonicalizeError::LayoutError(_) => {
            ApplyError::NotApplicable(NotApplicableReason::LayoutIncompatible)
        }
    }
}

fn lowered_memory_limits<'a, Tgt: Target, I>(
    base_limits: &MemoryLimits,
    live_tensors: I,
) -> Result<MemoryLimits, ApplyError>
where
    I: IntoIterator<Item = &'a TensorSpec<Tgt>>,
{
    let mut consumption = [0u64; MEMORY_COUNT];
    for tensor_spec in live_tensors {
        let idx = Tgt::memories()
            .iter()
            .position(|memory| memory == &tensor_spec.memory())
            .unwrap();
        consumption[idx] += tensor_spec.memory_units();
    }
    let mut lowered = MemoryLimits::Standard(match base_limits.to_owned() {
        MemoryLimits::Standard(v) => {
            v.checked_sub_snap_down::<Tgt>(&consumption)
                .map_err(|oom_idx| {
                    ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                        Tgt::memories()[oom_idx].to_string(),
                    ))
                })?
        }
    });
    lowered.discretize::<Tgt>();
    Ok(lowered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::imp::ImplNode;
    use crate::layout::row_major;
    use crate::shape;
    use crate::target::{Avx2Target, CpuMemory, CpuTarget};
    use crate::tensorspec::TensorSpecAux;

    #[test]
    fn test_non_exact_parallel_split_applies() {
        let result = split_action(10, 4).apply(&softmax_denominator_and_max_spec(10));
        assert!(result.is_ok(), "unexpected application result: {result:?}");
    }

    #[test]
    fn test_non_exact_parallel_split_adds_tail_body() {
        let pipeline = split_pipeline(10, 4);
        let ImplNode::Loop(loop_impl) = &pipeline.stages[0] else {
            panic!("expected first stage to be a loop");
        };
        assert_eq!(loop_impl.bodies.len(), 2);
    }

    #[test]
    fn test_non_exact_parallel_split_uses_ceil_partial_shape() {
        let pipeline = split_pipeline(10, 4);
        let ImplNode::SpecApp(combine) = &pipeline.stages[1] else {
            panic!("expected second stage to combine partials");
        };
        assert_eq!((combine.0).0.parameter_shape(0), shape![1, 3]);
    }

    #[test]
    fn test_exact_parallel_split_keeps_single_body() {
        let pipeline = split_pipeline(12, 4);
        let ImplNode::Loop(loop_impl) = &pipeline.stages[0] else {
            panic!("expected first stage to be a loop");
        };
        assert_eq!(loop_impl.bodies.len(), 1);
    }

    #[cfg(not(feature = "softmax-disable-online-rewrites"))]
    #[test]
    fn test_target_generates_non_exact_parallel_split_actions() {
        use crate::scheduling::Action;

        let actions =
            Avx2Target::actions(&softmax_denominator_and_max_spec(4_097).0).collect::<Vec<_>>();
        assert!(actions.iter().any(|action| {
            matches!(
                action,
                Action::ParallelSplitSoftmaxDenominatorAndMax(ParallelSplitSoftmaxDenominatorAndMax {
                    k,
                    ..
                }) if k.get() == 64
            )
        }));
    }

    fn split_pipeline(scan_len: u32, k: u32) -> crate::imp::pipeline::Pipeline<Avx2Target> {
        let imp = split_action(scan_len, k)
            .apply(&softmax_denominator_and_max_spec(scan_len))
            .expect("parallel split should apply");
        let ImplNode::Pipeline(pipeline) = imp else {
            panic!("expected pipeline");
        };
        pipeline
    }

    fn split_action(scan_len: u32, k: u32) -> ParallelSplitSoftmaxDenominatorAndMax<Avx2Target> {
        let partial_shape = shape![1, scan_len.div_ceil(k)];
        ParallelSplitSoftmaxDenominatorAndMax {
            k: DimSize::new(k).unwrap(),
            partials_level: CpuMemory::L1,
            partials_layout: row_major(&partial_shape),
            partials_vector_size: None,
        }
    }

    fn softmax_denominator_and_max_spec(scan_len: u32) -> Spec<Avx2Target> {
        let shape = shape![1, scan_len];
        let basics = PrimitiveBasics {
            typ: PrimitiveSpecType::SoftmaxDenominatorAndMax {
                scan_dim: 1,
                accum: false,
            },
            spec_shape: shape.clone(),
            dtypes: vec![Dtype::Float32; 3],
        };
        let auxes = basics
            .parameter_shapes()
            .into_iter()
            .map(|shape| TensorSpecAux {
                memory: CpuMemory::GL,
                layout: row_major(&shape),
                vector_size: None,
            })
            .collect();
        let mut spec = Spec(
            LogicalSpec::Primitive(basics, auxes, false),
            <Avx2Target as CpuTarget>::max_mem(),
        );
        spec.canonicalize()
            .expect("test softmax spec should canonicalize");
        spec
    }
}
