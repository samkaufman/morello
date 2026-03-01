use crate::common::{DimSize, Dtype};
use crate::imp::allocs::{alloc_memory_allocation, move_cost, Alloc};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::scheduling::{ActionSolver, ActionT, ApplyError, MoveActionSolver, NotApplicableReason};
use crate::spec::{LogicalSpec, OperandDirection, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{MemoryLevel, Target, LEVEL_COUNT};
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
struct AllocPlan<Tgt: Target> {
    new_spec: TensorSpec<Tgt>,
    prologue_spec: Option<Spec<Tgt>>,
    epilogue_spec: Option<Spec<Tgt>>,
    new_body_spec: Spec<Tgt>,
    new_operands: Vec<TensorSpec<Tgt>>,
    is_cache_miss: bool,
}

impl<Tgt: Target> ActionT<Tgt> for Move<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let AllocPlan {
            new_spec,
            prologue_spec,
            epilogue_spec,
            new_body_spec,
            new_operands,
            is_cache_miss,
        } = plan_alloc(
            spec,
            self.source_idx,
            self.destination_dtype,
            self.destination_level,
            &self.destination_layout,
            self.destination_vector_size,
        )?;
        let outer_moved_operand_spec = spec.0.parameter(self.source_idx.into());

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

        Ok(ImplNode::Alloc(Alloc::new(
            self.source_idx,
            outer_moved_operand_spec,
            inner_moved_operand,
            prologue,
            main_stage,
            epilogue,
            Some(spec.clone()),
        )))
    }

    fn top_down_solver(&self, spec: &Spec<Tgt>) -> Result<ActionSolver<Tgt>, ApplyError> {
        let plan = plan_alloc(
            spec,
            self.source_idx,
            self.destination_dtype,
            self.destination_level,
            &self.destination_layout,
            self.destination_vector_size,
        )?;
        let source_spec = spec.0.parameter(self.source_idx.into());
        let base_main_cost = move_cost(&source_spec) + move_cost(&plan.new_spec);
        let allocation = alloc_memory_allocation(&plan.new_spec);
        Ok(MoveActionSolver {
            prologue: plan.prologue_spec,
            body: plan.new_body_spec,
            epilogue: plan.epilogue_spec,
            base_main_cost,
            allocation,
        }
        .into())
    }
}

fn plan_alloc<Tgt: Target>(
    spec: &Spec<Tgt>,
    source_idx: u8,
    destination_dtype: Dtype,
    destination_level: Tgt::Level,
    destination_layout: &Layout,
    destination_vector_size: Option<DimSize>,
) -> Result<AllocPlan<Tgt>, ApplyError> {
    let operands = spec.0.parameters();

    let outer_moved_operand_spec = &operands[usize::from(source_idx)];

    let (destination_layout_canonicalized, is_cache_miss) = if !destination_level.has_layout() {
        let empty_layout = Layout::empty();
        let is_cache_miss = move_is_cache_miss(
            outer_moved_operand_spec,
            destination_dtype,
            &destination_level,
            &empty_layout,
        );
        (empty_layout, is_cache_miss)
    } else {
        let mut layout_canon = destination_layout
            .canonicalize(outer_moved_operand_spec.shape())
            .unwrap();
        let is_cache_miss = move_is_cache_miss(
            outer_moved_operand_spec,
            destination_dtype,
            &destination_level,
            &layout_canon,
        );
        if !is_cache_miss {
            layout_canon.set_contiguous_full();
            // Canonicalize again in case setting contiguous affected anything.
            // TODO: Don't call canonicalize twice.
            layout_canon = layout_canon
                .canonicalize(outer_moved_operand_spec.shape())
                .unwrap();
        }
        (layout_canon, is_cache_miss)
    };
    let mut new_spec = TensorSpec::<Tgt>::new_noncanon(
        outer_moved_operand_spec.shape().into(),
        destination_dtype,
        destination_level,
        destination_layout_canonicalized,
        destination_vector_size,
    );

    // If this is anything other than a simple cache miss, a new buffer will be allocated, so that
    // buffer will be aligned and fully contiguous.
    if !is_cache_miss {
        new_spec
            .canonicalize()
            .map_err(|canon_error| match canon_error {
                tensorspec::CanonicalizeError::LayoutError(_) => {
                    ApplyError::NotApplicable(NotApplicableReason::LayoutIncompatible)
                }
                tensorspec::CanonicalizeError::VectorSizeInvalid => {
                    ApplyError::NotApplicable(NotApplicableReason::VectorSizeInvalid)
                }
                tensorspec::CanonicalizeError::VectorSizeVolumeIncompatible => {
                    ApplyError::NotApplicable(NotApplicableReason::VectorSizeVolumeIncompatible)
                }
            })?;
    }

    assert!(
        !new_spec.level().has_layout() || new_spec.layout().applies_to_shape(new_spec.shape()),
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
        let levels = Tgt::levels();
        let updated_level_idx = levels.iter().position(|l| l == &destination_level).unwrap();
        let additional = new_spec.memory_units();
        match &spec.1 {
            MemoryLimits::Standard(base) => {
                let mut new_values: [u64; LEVEL_COUNT] =
                    base.iter().collect::<Vec<_>>().try_into().unwrap();
                let Some(level_updated) = new_values[updated_level_idx].checked_sub(additional)
                else {
                    return Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                        levels[updated_level_idx].to_string(),
                    )));
                };
                if levels[updated_level_idx].counts_registers() {
                    new_values[updated_level_idx] = level_updated;
                } else {
                    new_values[updated_level_idx] = prev_power_of_two(level_updated);
                }
                MemoryLimits::Standard(MemVec::new_for_target::<Tgt>(new_values))
            }
            MemoryLimits::Pipeline {
                limits,
                before,
                after,
            } => {
                let mut limits = limits.clone();
                let current = limits.get_unscaled(updated_level_idx);
                let Some(level_updated) = current.checked_sub(additional) else {
                    return Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                        levels[updated_level_idx].to_string(),
                    )));
                };
                // limits was lowered, but before and after are unchanged. If either
                // difference would go negative, then the move is not possible due to
                // OOM, since the move's buffer would exceed the available memory at
                // that level.
                if level_updated < before[updated_level_idx]
                    || level_updated < after[updated_level_idx]
                {
                    return Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                        levels[updated_level_idx].to_string(),
                    )));
                }
                if levels[updated_level_idx].counts_registers() {
                    limits.set(updated_level_idx, level_updated);
                } else {
                    limits.set(updated_level_idx, prev_power_of_two(level_updated));
                }
                MemoryLimits::Pipeline {
                    limits,
                    before: *before,
                    after: *after,
                }
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

    let mut new_operands = operands;
    new_operands[usize::from(source_idx)] = new_spec.clone();
    let mut new_body_spec = Spec(spec.0.clone(), lower_limits);
    new_body_spec.0.replace_io(&new_operands);
    new_body_spec.canonicalize().unwrap();

    Ok(AllocPlan {
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
    use crate::layout::{batched_col_major, row_major};
    use crate::scheduling::Action;
    use crate::target::{Avx2Target, CpuMemoryLevel};
    use crate::{layout, shape};
    use crate::{lspec, spec};

    #[test]
    fn test_subspecs_when_moving_into_degenerate_packed_layout_solver() {
        shared_test_subspecs_when_moving_into_degenerate_packed_layout(|spec, action| {
            action.top_down_solver(spec).unwrap().subspecs().collect()
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
        let spec: Spec<Avx2Target> = spec!(MatmulAccum(
            [1, 8, 128, 8],
            (f32, CpuMemoryLevel::GL, batched_col_major),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let plan = plan_alloc(
            &spec,
            0,
            Dtype::Float32,
            CpuMemoryLevel::L1,
            &layout![0, 1, 2, 1 p(8)],
            None,
        )
        .unwrap();
        assert_eq!(
            plan.new_spec.layout(),
            &batched_col_major(&spec.0.parameter_shape(0))
                .canonicalize(&spec.0.parameter_shape(0))
                .unwrap()
        );
        assert!(plan.new_spec.is_contiguous());
    }

    /// Test that [plan_alloc] preserves and correctly updates [MemoryLimits::Pipeline].
    #[test]
    fn test_plan_alloc_preserves_pipeline_structure() {
        const ORIGINAL_L1: u64 = 4096;
        let limits = MemVec::new_for_target::<Avx2Target>([16, 16, ORIGINAL_L1, 4096]);
        const BEFORE: [u64; LEVEL_COUNT] = [0, 0, 512, 512];
        const AFTER: [u64; LEVEL_COUNT] = [0, 0, 256, 256];
        let spec: Spec<Avx2Target> = Spec(
            lspec!(Move(
                [8, 8],
                (f32, CpuMemoryLevel::GL, row_major),
                (f32, CpuMemoryLevel::L1, row_major)
            )),
            MemoryLimits::Pipeline {
                limits: limits.clone(),
                before: BEFORE,
                after: AFTER,
            },
        );
        let plan = plan_alloc(
            &spec,
            0,
            Dtype::Float32,
            CpuMemoryLevel::L1,
            &row_major(&shape![8, 8]),
            None,
        )
        .unwrap();
        let MemoryLimits::Pipeline {
            limits: result_limits,
            before: result_before,
            after: result_after,
        } = &plan.new_body_spec.1
        else {
            panic!("expected Pipeline, got Standard");
        };

        // Compute expected L1 limit: original - cache_lines, snapped to prev power of two
        let cache_lines = row_major(&shape![8, 8])
            .estimate_cache_lines::<Avx2Target>(&shape![8, 8], Dtype::Float32);
        let expected_l1 = prev_power_of_two(ORIGINAL_L1 - u64::from(cache_lines));

        // limits should be unchanged at levels 0, 1; decreased at L1 (level 2); zeroed
        // at GL (level 3)
        assert_eq!(result_limits.get_unscaled(0), limits.get_unscaled(0));
        assert_eq!(result_limits.get_unscaled(1), limits.get_unscaled(1));
        assert_eq!(result_limits.get_unscaled(2), expected_l1);
        assert_eq!(result_limits.get_unscaled(3), 0);

        // before and after should be unchanged for levels 0..=2, zeroed at GL (index 3)
        assert_eq!(result_before[..3], BEFORE[..3]);
        assert_eq!(
            result_before[3], 0,
            "before's GL should be zeroed by canonicalization"
        );
        assert_eq!(result_after[..3], AFTER[..3]);
        assert_eq!(result_after[3], 0);
    }

    fn child_impls_into_specs(imp: &ImplNode<Avx2Target>) -> Vec<Spec<Avx2Target>> {
        imp.children()
            .iter()
            .map(|child| {
                if let ImplNode::SpecApp(SpecApp(spec, _)) = child {
                    spec.clone()
                } else {
                    panic!("expected a SpecApp child, got {child:?}")
                }
            })
            .collect()
    }

    fn shared_test_subspecs_when_moving_into_degenerate_packed_layout(
        child_get: impl FnOnce(&Spec<Avx2Target>, Action<Avx2Target>) -> Vec<Spec<Avx2Target>>,
    ) {
        let spec: Spec<Avx2Target> = spec!(MatmulAccum(
            [1, 8, 128, 8],
            (f32, CpuMemoryLevel::GL, batched_col_major),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let action = Action::Move(Move {
            source_idx: 0,
            destination_dtype: Dtype::Float32,
            destination_level: CpuMemoryLevel::L1,
            destination_layout: layout![0, 1, 2, 1 p(8)],
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
            children => panic!("expected one Matmul Spec child, got {children:?}"),
        };
    }
}
