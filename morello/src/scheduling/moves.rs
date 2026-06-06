use crate::common::{DimSize, Dtype};
use crate::imp::allocs::{alloc_memory_allocation, move_cost, Alloc};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::{row_major, Layout};
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{ActionSolver, ActionT, ApplyError, MoveActionSolver, NotApplicableReason};
use crate::spec::{LogicalSpec, OperandDirection, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{Memory, Target, MEMORY_COUNT};
use crate::tensorspec::{self, TensorSpec, TensorSpecAux};
use crate::views::{CacheView, Param, Tensor, ViewE};
use serde::{Deserialize, Serialize};
use std::mem;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Move<Tgt: Target> {
    pub source_idx: u8,
    pub destination_dtype: Dtype,
    pub destination_level: Tgt::Memory,
    pub destination_layout: Layout,
    pub destination_vector_size: Option<DimSize>,
}

/// Data useful to both a Move's [ActionSolver] or [ImplNode].
struct AllocPlan<'a, Tgt: Target> {
    outer_moved_operand_spec: &'a TensorSpec<Tgt>,
    new_spec: TensorSpec<Tgt>,
    prologue_spec: Option<Spec<Tgt>>,
    epilogue_spec: Option<Spec<Tgt>>,
    new_body_spec: Spec<Tgt>,
    new_operands: Vec<TensorSpec<Tgt>>,
    is_cache_miss: bool,
}

struct SolverAllocPlan<Tgt: Target> {
    outer_moved_operand_spec: TensorSpec<Tgt>,
    new_spec: TensorSpec<Tgt>,
    prologue_spec: Option<Spec<Tgt>>,
    epilogue_spec: Option<Spec<Tgt>>,
    new_body_spec: Spec<Tgt>,
}

struct AllocPlanShared<Tgt: Target> {
    new_spec: TensorSpec<Tgt>,
    lower_limits: MemoryLimits,
    is_cache_miss: bool,
}

impl<Tgt: Target> ActionT<Tgt> for Move<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        let AllocPlan {
            outer_moved_operand_spec,
            new_spec,
            prologue_spec,
            epilogue_spec,
            new_body_spec,
            new_operands,
            is_cache_miss,
        } = plan_alloc(
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

        Ok(ImplNode::Alloc(Alloc::new(
            self.source_idx,
            outer_moved_operand_spec.clone(),
            inner_moved_operand,
            prologue,
            main_stage,
            epilogue,
            Some(spec.clone()),
        )))
    }

    fn top_down_solver(&self, spec: &Spec<Tgt>) -> Result<ActionSolver<Tgt>, ApplyError> {
        let plan = plan_alloc_for_solver(
            spec,
            self.source_idx,
            self.destination_dtype,
            self.destination_level,
            &self.destination_layout,
            self.destination_vector_size,
        )?;
        let base_main_cost = move_cost(&plan.outer_moved_operand_spec) + move_cost(&plan.new_spec);
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

fn plan_alloc<'a, Tgt: Target>(
    spec: &Spec<Tgt>,
    operands: &'a [TensorSpec<Tgt>],
    source_idx: u8,
    destination_dtype: Dtype,
    destination_level: Tgt::Memory,
    destination_layout: &Layout,
    destination_vector_size: Option<DimSize>,
) -> Result<AllocPlan<'a, Tgt>, ApplyError> {
    let outer_moved_operand_spec = &operands[usize::from(source_idx)];

    let AllocPlanShared {
        new_spec,
        lower_limits,
        is_cache_miss,
    } = plan_alloc_shared(
        spec,
        outer_moved_operand_spec,
        destination_dtype,
        destination_level,
        destination_layout,
        destination_vector_size,
    )?;

    let prologue_spec = make_logue(
        false,
        move_gens_prologue(source_idx, &spec.0, is_cache_miss),
        outer_moved_operand_spec,
        &new_spec,
        &lower_limits,
        spec.0.serial_only(),
    );
    let epilogue_spec = make_logue(
        true,
        move_gens_epilogue(source_idx, &spec.0, is_cache_miss),
        outer_moved_operand_spec,
        &new_spec,
        &lower_limits,
        spec.0.serial_only(),
    );

    let mut new_operands = operands.to_vec();
    new_operands[usize::from(source_idx)] = new_spec.clone();
    let new_body_spec =
        make_body_spec_for_move(spec, source_idx, &new_spec, &new_operands, lower_limits);

    Ok(AllocPlan {
        outer_moved_operand_spec,
        new_spec,
        prologue_spec,
        epilogue_spec,
        new_body_spec,
        new_operands,
        is_cache_miss,
    })
}

/// Lightweight variant of [plan_alloc].
///
/// The full [plan_alloc] path must keep a replacement operand vector to build child views for the
/// final [Alloc]. The solver only needs prologue/body/epilogue sub-Specs and costs.
fn plan_alloc_for_solver<Tgt: Target>(
    spec: &Spec<Tgt>,
    source_idx: u8,
    destination_dtype: Dtype,
    destination_level: Tgt::Memory,
    destination_layout: &Layout,
    destination_vector_size: Option<DimSize>,
) -> Result<SolverAllocPlan<Tgt>, ApplyError> {
    if !matches!(spec.0, LogicalSpec::Primitive(..)) {
        // Compose replacement currently relies on LogicalSpec::replace_io to propagate shapes,
        // dtypes, and auxes through each component. Keep that less common path on the full planner
        // until it has a specialized updater like the primitive fast path below.
        let operands = spec.0.parameters();
        let plan = plan_alloc(
            spec,
            &operands,
            source_idx,
            destination_dtype,
            destination_level,
            destination_layout,
            destination_vector_size,
        )?;
        return Ok(SolverAllocPlan {
            outer_moved_operand_spec: plan.outer_moved_operand_spec.clone(),
            new_spec: plan.new_spec,
            prologue_spec: plan.prologue_spec,
            epilogue_spec: plan.epilogue_spec,
            new_body_spec: plan.new_body_spec,
        });
    }

    let outer_moved_operand_spec = spec.0.parameter(usize::from(source_idx));
    let AllocPlanShared {
        new_spec,
        lower_limits,
        is_cache_miss,
    } = plan_alloc_shared(
        spec,
        &outer_moved_operand_spec,
        destination_dtype,
        destination_level,
        destination_layout,
        destination_vector_size,
    )?;

    let prologue_spec = make_logue(
        false,
        move_gens_prologue(source_idx, &spec.0, is_cache_miss),
        &outer_moved_operand_spec,
        &new_spec,
        &lower_limits,
        spec.0.serial_only(),
    );
    let epilogue_spec = make_logue(
        true,
        move_gens_epilogue(source_idx, &spec.0, is_cache_miss),
        &outer_moved_operand_spec,
        &new_spec,
        &lower_limits,
        spec.0.serial_only(),
    );

    let new_body_spec =
        make_primitive_body_spec_for_move(spec, usize::from(source_idx), &new_spec, lower_limits);

    Ok(SolverAllocPlan {
        outer_moved_operand_spec,
        new_spec,
        prologue_spec,
        epilogue_spec,
        new_body_spec,
    })
}

fn plan_alloc_shared<Tgt: Target>(
    spec: &Spec<Tgt>,
    outer_moved_operand_spec: &TensorSpec<Tgt>,
    destination_dtype: Dtype,
    destination_level: Tgt::Memory,
    destination_layout: &Layout,
    destination_vector_size: Option<DimSize>,
) -> Result<AllocPlanShared<Tgt>, ApplyError> {
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
        let cache_miss_possible = !destination_level.is_addressed()
            && destination_dtype == outer_moved_operand_spec.dtype();
        if cache_miss_possible {
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
            }
            (layout_canon, is_cache_miss)
        } else {
            let mut layout = destination_layout.clone();
            layout.set_contiguous_full();
            (layout, false)
        }
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
        !new_spec.memory().has_layout() || new_spec.layout().applies_to_shape(new_spec.shape()),
        "Destination layout {:?} does not apply to shape {:?}",
        new_spec.layout(),
        new_spec.shape()
    );

    // Filter cases where, after canonicalization, the source and destination
    // TensorSpecs match (i.e., within-memory copies).
    if outer_moved_operand_spec == &new_spec {
        return Err(ApplyError::NotApplicable(NotApplicableReason::SelfMove));
    }

    let lower_limits: MemoryLimits = {
        let memories = Tgt::memories();
        let updated_level_idx = memories
            .iter()
            .position(|l| l == &destination_level)
            .unwrap();
        let additional = new_spec.memory_units();
        let mut consumed = [0u64; MEMORY_COUNT];
        consumed[updated_level_idx] = additional;
        match &spec.1 {
            MemoryLimits::Standard(base) => MemoryLimits::Standard(
                base.clone()
                    .checked_sub_snap_down::<Tgt>(&consumed)
                    .map_err(|oom_idx| {
                        ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                            memories[oom_idx].to_string(),
                        ))
                    })?,
            ),
        }
    };

    Ok(AllocPlanShared {
        new_spec,
        lower_limits,
        is_cache_miss,
    })
}

fn make_logue<Tgt: Target>(
    flip: bool,
    should_generate: bool,
    outer_moved_operand_spec: &TensorSpec<Tgt>,
    new_spec: &TensorSpec<Tgt>,
    lower_limits: &MemoryLimits,
    serial_only: bool,
) -> Option<Spec<Tgt>> {
    if !should_generate {
        return None;
    }

    let mut left_spec = outer_moved_operand_spec;
    let mut right_spec = new_spec;
    if flip {
        mem::swap(&mut left_spec, &mut right_spec);
    }
    let basics = PrimitiveBasics {
        typ: PrimitiveSpecType::Move,
        spec_shape: left_spec.shape().into(),
        dtypes: vec![left_spec.dtype(), right_spec.dtype()],
    };
    let mut auxes = vec![left_spec.aux.clone(), right_spec.aux.clone()];
    canonicalize_matching_contiguous_move_auxes(&basics.spec_shape, &basics.dtypes, &mut auxes);
    let logue_spec = Spec(
        LogicalSpec::Primitive(basics, auxes, serial_only),
        lower_limits.clone(),
    );
    Some(logue_spec)
}

fn make_body_spec_for_move<Tgt: Target>(
    spec: &Spec<Tgt>,
    source_idx: u8,
    new_spec: &TensorSpec<Tgt>,
    new_operands: &[TensorSpec<Tgt>],
    lower_limits: MemoryLimits,
) -> Spec<Tgt> {
    match &spec.0 {
        LogicalSpec::Primitive(..) => {
            make_primitive_body_spec_for_move(spec, usize::from(source_idx), new_spec, lower_limits)
        }
        LogicalSpec::Compose { .. } => {
            let mut new_body_spec = Spec(spec.0.clone(), lower_limits);
            new_body_spec.0.replace_io(new_operands);
            new_body_spec.canonicalize().unwrap();
            new_body_spec
        }
    }
}

fn make_primitive_body_spec_for_move<Tgt: Target>(
    spec: &Spec<Tgt>,
    source_idx: usize,
    new_spec: &TensorSpec<Tgt>,
    lower_limits: MemoryLimits,
) -> Spec<Tgt> {
    let mut new_body_spec = Spec(spec.0.clone(), lower_limits);
    let LogicalSpec::Primitive(basics, auxes, _) = &mut new_body_spec.0 else {
        unreachable!();
    };
    basics.dtypes[source_idx] = new_spec.dtype();
    auxes[source_idx] = new_spec.aux.clone();

    if matches!(basics.typ, PrimitiveSpecType::Move) {
        canonicalize_matching_contiguous_move_auxes(&basics.spec_shape, &basics.dtypes, auxes);
    }

    new_body_spec
}

/// Applies the only [LogicalSpec::canonicalize] rewrite that is not already guaranteed by
/// [plan_alloc_shared] and the canonical input spec.
///
/// `Move` specs with equal dtypes, equal layouts, and fully-contiguous operands are canonicalized
/// to row-major layouts as a symmetry break. The direct move-planning path constructs logue
/// [Spec]s and primitive moved bodies from already-canonical operand auxes, so it can preserve
/// canonical form by applying just this targeted rule instead of running full
/// [Spec::canonicalize].
fn canonicalize_matching_contiguous_move_auxes<Tgt: Target>(
    shape: &[DimSize],
    dtypes: &[Dtype],
    auxes: &mut [TensorSpecAux<Tgt>],
) {
    debug_assert_eq!(dtypes.len(), auxes.len());
    if dtypes.is_empty() {
        return;
    }

    let first_dtype = dtypes[0];
    let first_layout = &auxes[0].layout;
    if dtypes.iter().all(|&dtype| dtype == first_dtype)
        && auxes
            .iter()
            .all(|aux| &aux.layout == first_layout && aux.layout.is_fully_contiguous())
    {
        for aux in auxes {
            if aux.memory.has_layout() {
                aux.layout = row_major(shape);
            }
        }
    }
}

fn move_gens_prologue<Tgt: Target>(
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
    is_cache_miss: bool,
) -> bool {
    if is_cache_miss {
        return false;
    }
    match logical_spec.parameter_direction(usize::from(source_idx)) {
        OperandDirection::In | OperandDirection::InOut => true,
        OperandDirection::Out => false,
    }
}

fn move_gens_epilogue<Tgt: Target>(
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
    is_cache_miss: bool,
) -> bool {
    if is_cache_miss {
        return false;
    }
    let source_idx_usize = usize::from(source_idx);
    logical_spec.parameter_is_output(source_idx_usize)
}

/// Returns `true` if the move is a simple cache miss.
///
/// This is true if the destination is a hardware cache and the layout and data type are
/// unchanged.
fn move_is_cache_miss<Tgt: Target>(
    operand: &TensorSpec<Tgt>,
    destination_dtype: Dtype,
    destination_level: &Tgt::Memory,
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
    use crate::layout;
    use crate::layout::{batched_col_major, row_major};
    use crate::scheduling::Action;
    use crate::spec;
    use crate::target::{Avx2Target, CpuMemory};
    use proptest::prelude::*;
    use proptest::proptest;
    use proptest::sample::select;

    fn arb_primitive_movable_spec_and_action(
    ) -> impl Strategy<Value = (Spec<Avx2Target>, Move<Avx2Target>)> {
        crate::spec::arb_canonical_primitive_spec::<Avx2Target>(
            Some(DimSize::new(8).unwrap()),
            None,
        )
        .prop_filter_map("Primitive Spec should have Move actions", |spec| {
            let moves = Avx2Target::actions(&spec.0)
                .filter_map(|action| match action {
                    Action::Move(action) => Some(action),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (!moves.is_empty()).then_some((spec, moves))
        })
        .prop_flat_map(|(spec, moves)| (Just(spec), select(moves)))
    }

    proptest! {
        #[test]
        fn test_primitive_body_spec_for_move_is_canonical(
            (spec, action) in arb_primitive_movable_spec_and_action()
        ) {
            let outer_moved_operand_spec = spec.0.parameter(usize::from(action.source_idx));
            let Ok(AllocPlanShared { new_spec, lower_limits, .. }) = plan_alloc_shared(
                &spec,
                &outer_moved_operand_spec,
                action.destination_dtype,
                action.destination_level,
                &action.destination_layout,
                action.destination_vector_size,
            ) else {
                return Ok(());
            };

            let new_body_spec = make_primitive_body_spec_for_move(
                &spec,
                usize::from(action.source_idx),
                &new_spec,
                lower_limits,
            );
            prop_assert!(new_body_spec.is_canonical());
        }

        #[test]
        fn test_logue_spec_for_move_is_canonical(
            (spec, action) in arb_primitive_movable_spec_and_action()
        ) {
            let outer_moved_operand_spec = spec.0.parameter(usize::from(action.source_idx));
            let Ok(AllocPlanShared { new_spec, lower_limits, .. }) = plan_alloc_shared(
                &spec,
                &outer_moved_operand_spec,
                action.destination_dtype,
                action.destination_level,
                &action.destination_layout,
                action.destination_vector_size,
            ) else {
                return Ok(());
            };

            for flip in [false, true] {
                let logue_spec = make_logue(
                    flip,
                    true,
                    &outer_moved_operand_spec,
                    &new_spec,
                    &lower_limits,
                    spec.0.serial_only(),
                )
                .unwrap();
                prop_assert!(logue_spec.is_canonical());
            }
        }
    }

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
            (f32, CpuMemory::GL, batched_col_major),
            (f32, CpuMemory::GL, row_major),
            (f32, CpuMemory::GL, row_major)
        ));
        let parameters = spec.0.parameters();
        let plan = plan_alloc(
            &spec,
            &parameters,
            0,
            Dtype::Float32,
            CpuMemory::L1,
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
            (f32, CpuMemory::GL, batched_col_major),
            (f32, CpuMemory::GL, row_major),
            (f32, CpuMemory::GL, row_major)
        ));
        let action = Action::Move(Move {
            source_idx: 0,
            destination_dtype: Dtype::Float32,
            destination_level: CpuMemory::L1,
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
