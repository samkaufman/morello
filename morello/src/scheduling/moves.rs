use crate::common::{DimSize, Dtype};
use crate::imp::moves::{move_cost, movelet_memory_allocation, MoveLet};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{ActionSolver, ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, OperandDirection, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::{self, TensorSpec};
use crate::utils::prev_power_of_two;
use crate::views::{CacheView, Param, Tensor, ViewE};
use serde::{Deserialize, Serialize};
use std::mem;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Move<Tgt: Target> {
    pub source_idx: u8,
    pub destination_dtype: Dtype,
    pub destination_level: Tgt::Level,
    pub destination_layout: Layout,
    pub destination_vector_size: Option<DimSize>,
}

/// Data useful to both a Move's [ActionSolver] or [ImplNode].
struct MoveLetPlan<'a, Tgt: Target> {
    outer_moved_operand_spec: &'a TensorSpec<Tgt>,
    new_spec: TensorSpec<Tgt>,
    prologue_spec: Option<Spec<Tgt>>,
    epilogue_spec: Option<Spec<Tgt>>,
    new_body_spec: Spec<Tgt>,
    new_operands: Vec<TensorSpec<Tgt>>,
    is_cache_miss: bool,
}

impl<Tgt: Target> ActionT<Tgt> for Move<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        let MoveLetPlan {
            outer_moved_operand_spec,
            new_spec,
            prologue_spec,
            epilogue_spec,
            new_body_spec,
            new_operands,
            is_cache_miss,
        } = plan_movelet(
            spec,
            &operands,
            self.source_idx,
            self.destination_dtype,
            self.destination_level,
            &self.destination_layout,
            self.destination_vector_size,
        )?;

        let inner_moved_operand = if is_cache_miss {
            let source = Param::new(self.source_idx, outer_moved_operand_spec.clone());
            ViewE::from(CacheView::new(source, new_spec))
        } else {
            ViewE::from(Tensor::new(new_spec))
        };

        let source_param = ViewE::from(Param::new(
            self.source_idx,
            outer_moved_operand_spec.clone(),
        ));
        let prologue = prologue_spec.map(|s| {
            ImplNode::from(SpecApp::new(
                s.clone(),
                [source_param.clone(), inner_moved_operand.clone()],
            ))
        });
        let epilogue = epilogue_spec.map(|s| {
            ImplNode::from(SpecApp::new(
                s.clone(),
                [inner_moved_operand.clone(), source_param],
            ))
        });

        let main_stage = {
            let inner_operands = new_operands.iter().enumerate().map(|(i, o)| {
                if i == self.source_idx as usize {
                    inner_moved_operand.clone()
                } else {
                    ViewE::from(Param::new(u8::try_from(i).unwrap(), o.clone()))
                }
            });
            ImplNode::from(SpecApp::new(new_body_spec, inner_operands))
        };

        Ok(ImplNode::MoveLet(MoveLet::new(
            self.source_idx,
            outer_moved_operand_spec.clone(),
            inner_moved_operand,
            prologue,
            main_stage,
            epilogue,
            Some(spec.clone()),
        )))
    }

    fn solver(&self, spec: &Spec<Tgt>) -> Result<ActionSolver<Tgt>, ApplyError> {
        let operands = spec.0.parameters();
        let plan = plan_movelet(
            spec,
            &operands,
            self.source_idx,
            self.destination_dtype,
            self.destination_level,
            &self.destination_layout,
            self.destination_vector_size,
        )?;
        let base_main_cost = move_cost(plan.outer_moved_operand_spec, &plan.new_spec);
        let allocation = movelet_memory_allocation(&plan.new_spec);
        Ok(ActionSolver::Move {
            prologue: plan.prologue_spec,
            body: plan.new_body_spec,
            epilogue: plan.epilogue_spec,
            base_main_cost,
            allocation,
        })
    }
}

fn plan_movelet<'a, Tgt: Target>(
    spec: &Spec<Tgt>,
    operands: &'a [TensorSpec<Tgt>],
    source_idx: u8,
    destination_dtype: Dtype,
    destination_level: Tgt::Level,
    destination_layout: &Layout,
    destination_vector_size: Option<DimSize>,
) -> Result<MoveLetPlan<'a, Tgt>, ApplyError> {
    let outer_moved_operand_spec = &operands[usize::from(source_idx)];

    let destination_layout_canonicalized = destination_layout
        .canonicalize(outer_moved_operand_spec.shape())
        .unwrap();
    let mut new_spec = TensorSpec::<Tgt>::new_noncanon(
        outer_moved_operand_spec.shape().into(),
        destination_dtype,
        outer_moved_operand_spec.contiguous_abs(),
        outer_moved_operand_spec.aligned(),
        destination_level,
        destination_layout_canonicalized,
        destination_vector_size,
    );

    let is_cache_miss = move_is_cache_miss(
        outer_moved_operand_spec,
        destination_dtype,
        &destination_level,
        new_spec.layout(),
    );
    // If this is anything other than a simple cache miss, a new buffer will be allocated, so that
    // buffer will be aligned and fully contiguous.
    if !is_cache_miss {
        new_spec.set_aligned(true);
        new_spec.set_contiguous_abs(new_spec.layout().contiguous_full());
        new_spec
            .canonicalize()
            .map_err(|canon_error| match canon_error {
                tensorspec::CanonicalizeError::LayoutError(_) => {
                    ApplyError::NotApplicable(NotApplicableReason::LayoutIncompatible)
                }
                tensorspec::CanonicalizeError::VectorSizeInvalid => {
                    let vs = destination_vector_size
                        .expect("destination vector size is set if VectorSizeInvalid returned");
                    ApplyError::NotApplicable(NotApplicableReason::VectorSizeInvalid(
                        destination_dtype,
                        vs,
                    ))
                }
            })?;
    }
    debug_assert_eq!(
        is_cache_miss,
        move_is_cache_miss(
            outer_moved_operand_spec,
            new_spec.dtype(),
            &new_spec.level(),
            new_spec.layout(),
        ),
        "simple_cache_miss changes after updating alignment and contiguousness"
    );

    assert!(
        new_spec.layout().applies_to_shape(new_spec.shape()),
        "Destination layout {:?} does not apply to shape {:?}",
        new_spec.layout(),
        new_spec.shape()
    );

    // Filter cases where, after canonicalization, the source and destination
    // TensorSpecs match (i.e., within-level copies).
    if outer_moved_operand_spec == &new_spec {
        return Err(ApplyError::NotApplicable(NotApplicableReason::SelfMove));
    }

    let lower_limits: MemoryLimits = {
        let additional = u64::from(destination_dtype.size())
            * u64::from(operands[usize::from(source_idx)].volume().get());
        match &spec.1 {
            MemoryLimits::Standard(base) => {
                let levels = Tgt::levels();
                let updated_level_idx =
                    levels.iter().position(|l| l == &destination_level).unwrap();
                let mut new_limits = base.clone();
                let Some(level_updated) = new_limits
                    .get_unscaled(updated_level_idx)
                    .checked_sub(additional)
                else {
                    return Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                        levels[updated_level_idx].to_string(),
                    )));
                };
                new_limits.set_unscaled(updated_level_idx, prev_power_of_two(level_updated));
                MemoryLimits::Standard(new_limits)
            }
        }
    };

    // Closure which makes a prologue or epilogue sub-Spec.
    let make_logue = |flip, f: &dyn Fn(_, _, _) -> bool| {
        if f(source_idx, &spec.0, is_cache_miss) {
            let mut left_spec = outer_moved_operand_spec;
            let mut right_spec = &new_spec;
            if flip {
                mem::swap(&mut left_spec, &mut right_spec);
            }
            let mut logue_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Move,
                        spec_shape: left_spec.shape().into(),
                        dtypes: vec![left_spec.dtype(), right_spec.dtype()],
                    },
                    vec![left_spec.aux.clone(), right_spec.aux.clone()],
                    spec.0.serial_only(),
                ),
                lower_limits.clone(),
            );
            logue_spec.canonicalize().unwrap();
            Some(logue_spec)
        } else {
            None
        }
    };
    let prologue_spec = make_logue(false, &move_gens_prologue);
    let epilogue_spec = make_logue(true, &move_gens_epilogue);

    let mut new_operands = operands.to_vec();
    new_operands[usize::from(source_idx)] = new_spec.clone();
    let mut new_body_spec = Spec(spec.0.clone(), lower_limits);
    new_body_spec.0.replace_io(&new_operands);
    new_body_spec.canonicalize().unwrap();

    Ok(MoveLetPlan {
        outer_moved_operand_spec,
        new_spec,
        prologue_spec,
        epilogue_spec,
        new_body_spec,
        new_operands,
        is_cache_miss,
    })
}

fn move_gens_prologue<Tgt: Target>(
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
    is_cache_miss: bool,
) -> bool {
    // TODO: Don't allocate whole operand_directions
    let is_read = match logical_spec.operand_directions()[usize::from(source_idx)] {
        OperandDirection::In | OperandDirection::InOut => true,
        OperandDirection::Out => false,
    };
    is_read && !is_cache_miss
}

fn move_gens_epilogue<Tgt: Target>(
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
    is_cache_miss: bool,
) -> bool {
    let source_idx_usize = usize::from(source_idx);
    let is_output = logical_spec.parameter_is_output(source_idx_usize);
    is_output && !is_cache_miss
}

/// Returns `true` if the move is a simple cache miss.
///
/// This is true if the destination is a hardware cache and the layout and data type are
/// unchanged.
fn move_is_cache_miss<Tgt: Target>(
    operand: &TensorSpec<Tgt>,
    destination_dtype: Dtype,
    destination_level: &Tgt::Level,
    destination_layout: &Layout,
) -> bool {
    !destination_level.is_addressed()
        && operand.layout() == destination_layout
        && operand.dtype() == destination_dtype
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::Impl;
    use crate::layout::{batched_col_major, row_major, PhysDim};
    use crate::lspec;
    use crate::scheduling::Action;
    use crate::target::{CpuMemoryLevel, X86Target};
    use nonzero::nonzero as nz;

    #[test]
    fn test_subspecs_when_moving_into_degenerate_packed_layout_solver() {
        shared_test_subspecs_when_moving_into_degenerate_packed_layout(|spec, action| {
            action.solver(spec).unwrap().subspecs().collect()
        })
    }

    #[test]
    fn test_subspecs_when_moving_into_degenerate_packed_layout_apply() {
        shared_test_subspecs_when_moving_into_degenerate_packed_layout(|spec, action| {
            child_impls_into_specs(&action.apply(spec).unwrap())
        })
    }

    #[test]
    fn test_subspecs_when_moving_into_degenerate_packed_layout_apply_unchecked() {
        shared_test_subspecs_when_moving_into_degenerate_packed_layout(|spec, action| {
            child_impls_into_specs(&action.apply_unchecked_canon(spec).unwrap())
        })
    }

    // TODO: Add a variant where only physically innermost dimension is contiguous.
    #[test]
    fn test_move_planning_into_cache_with_extra_degenerate_dims_preserves_layout_and_contig() {
        let batched_cm = batched_col_major(3);
        let logical_spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [1, 8, 128, 8],
            (f32, CpuMemoryLevel::GL, batched_cm.clone()),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let spec = Spec(logical_spec, X86Target::max_mem());
        let parameters = spec.0.parameters();
        let plan = plan_movelet(
            &spec,
            &parameters,
            0,
            Dtype::Float32,
            CpuMemoryLevel::L1,
            &Layout::new(vec![
                (0, PhysDim::Dynamic),
                (1, PhysDim::Dynamic), // param. is 8x128, col-maj., so this is useless.
                (2, PhysDim::Dynamic),
                (1, PhysDim::Packed(nz!(8u32))),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(plan.new_spec.layout(), &batched_cm);
        assert_eq!(plan.new_spec.contiguous_abs(), batched_cm.contiguous_full());
    }

    fn child_impls_into_specs(imp: &ImplNode<X86Target>) -> Vec<Spec<X86Target>> {
        imp.children()
            .iter()
            .map(|child| {
                if let ImplNode::SpecApp(SpecApp(spec, _)) = child {
                    spec.clone()
                } else {
                    panic!("expected a SpecApp child, got {:?}", child)
                }
            })
            .collect()
    }

    fn shared_test_subspecs_when_moving_into_degenerate_packed_layout(
        child_get: impl FnOnce(&Spec<X86Target>, Action<X86Target>) -> Vec<Spec<X86Target>>,
    ) {
        let batched_cm = batched_col_major(3);
        let logical: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [1, 8, 128, 8],
            (f32, CpuMemoryLevel::GL, batched_cm),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let spec = Spec(logical, X86Target::max_mem());
        let action = Action::Move(Move {
            source_idx: 0,
            destination_dtype: Dtype::Float32,
            destination_level: CpuMemoryLevel::L1,
            destination_layout: Layout::new(vec![
                (0, PhysDim::Dynamic),
                (1, PhysDim::Dynamic),
                (2, PhysDim::Dynamic),
                (1, PhysDim::Packed(nz!(8u32))),
            ]),
            destination_vector_size: None,
        });
        match child_get(&spec, action).as_slice() {
            [Spec(logical_spec, _)] => {
                assert!(matches!(
                    logical_spec,
                    LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Matmul { .. },
                            ..
                        },
                        ..
                    )
                ));
            }
            children => panic!("expected one Matmul Spec child, got {:?}", children),
        };
    }
}
