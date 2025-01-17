use crate::cost::{Cost, NormalizedCost};
use crate::db::DbKey;
use crate::grid::general::BiMap;
use crate::imp::blocks::Block;
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::{
    make_accum_inits_for_spec, Action, ActionEncodeDecode, ActionT, ApplyError, BottomUpSolver,
    NotApplicableReason, VisitUpdater,
};
use crate::spec::{FillValue, LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::iter;
use std::num::NonZeroU64;

#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToAccum;

type AccumMapKey<Tgt> = (TensorSpec<Tgt>, bool, MemoryLimits);

#[derive(Default)]
pub struct ToAccumSolver<Tgt: Target> {
    zeroes: HashMap<Spec<Tgt>, Vec<NormalizedCost>>,
    accums: HashMap<AccumMapKey<Tgt>, HashMap<Spec<Tgt>, Vec<NormalizedCost>>>,
}

impl<Tgt: Target> ActionT<Tgt> for ToAccum {
    type BSolver = ToAccumSolver<Tgt>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;

        let head = match logical_spec {
            LogicalSpec::Primitive(basics, ..) => basics,
            LogicalSpec::Compose { components, .. } => &components[0],
        };

        let PrimitiveBasics {
            typ:
                PrimitiveSpecType::Matmul { accum }
                | PrimitiveSpecType::Conv { accum }
                | PrimitiveSpecType::Max { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum, .. },
            ..
        } = head
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToAccum is not defined for this Spec kind",
            ))));
        };
        if *accum {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Already accumulating",
            ))));
        }

        let accum_logical_spec = logical_spec.clone_as_accum();
        let mut accum_spec = Spec(accum_logical_spec, spec.1.clone());
        accum_spec
            .canonicalize()
            .expect("ToAccum's introduced accumulating Spec should be canonicalizable");
        let zero_apps = make_accum_inits_for_spec(&accum_spec);
        let accum_app = SpecApp::new_with_default_params(accum_spec).into();

        let mut stages = zero_apps;
        stages.push(accum_app);
        let default_child = Some(stages.len() - 1);
        Ok(ImplNode::Block(Block {
            stages,
            spec: Some(spec.clone()),
            default_child,
        }))
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

impl<Tgt: Target> ToAccumSolver<Tgt> {
    /// Helper for [visit_dependency]. Called when both the Zero and MatmulAccum dependencies are
    /// available.
    fn visit_candidate_dependency_pair<U>(
        &self,
        zero_costs: &[NormalizedCost],
        accum_costs: &[NormalizedCost],
        zero_volume: NonZeroU64,
        matmul_volume: NonZeroU64,
        matmulaccum_spec: &Spec<Tgt>,
        updater: &mut U,
    ) where
        U: VisitUpdater<Tgt>,
    {
        let goal = matmul_from_toaccum_deps(matmulaccum_spec);
        match (accum_costs, zero_costs) {
            ([], _) | (_, []) => {
                // TODO: Prevent calling this twice per goal.
                updater.complete_spec(&goal);
            }
            ([matmulaccum_normalized_cost], [zero_normalized_cost]) => {
                let matmulaccum_cost = matmulaccum_normalized_cost
                    .intensity
                    .into_main_cost_for_volume(matmul_volume);
                let zero_cost = zero_normalized_cost
                    .intensity
                    .into_main_cost_for_volume(zero_volume);
                let peaks = MemoryAllocation::none()
                    .peak_memory_from_child_peaks::<Tgt>(&[
                        matmulaccum_normalized_cost.peaks.clone(),
                        zero_normalized_cost.peaks.clone(),
                    ])
                    .snap_up_for_target::<Tgt>(false);
                let cost_sum = Cost {
                    main: matmulaccum_cost.saturating_add(zero_cost),
                    peaks,
                    depth: 1 + matmulaccum_normalized_cost
                        .depth
                        .max(zero_normalized_cost.depth),
                };
                updater.complete_action(
                    &goal,
                    Action::ToAccum(ToAccum).encode(&goal),
                    NormalizedCost::new(cost_sum, matmul_volume),
                );
                updater.complete_spec(&goal);
            }
            _ => todo!("handle k>2"),
        }
    }
}

impl<Tgt: Target> BottomUpSolver for ToAccumSolver<Tgt> {
    type Tgt = Tgt;

    fn dependencies_for_range<B>(
        &mut self,
        _bimap: &B,
        low: &Spec<Self::Tgt>,
        high: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)>
    where
        B: BiMap<Domain = Spec<Self::Tgt>, Codomain = DbKey>,
    {
        let mut dependencies = vec![];
        if spec_is_plain_matmul(low) {
            if !spec_is_plain_matmul(high) {
                unimplemented!("non-Matmul high with Matmul low");
            }

            dependencies.reserve_exact(2);

            // Request the accumulating Matmul sub-Specs
            let accum_low = Spec(low.0.clone_as_accum(), low.1.clone());
            let accum_high = Spec(high.0.clone_as_accum(), high.1.clone());
            debug_assert!(accum_low.is_canonical());
            debug_assert!(accum_high.is_canonical());
            dependencies.push((accum_low, accum_high));

            // Request the Zero implementations as well.
            let zero_low = zero_for_spec(low);
            let zero_high = zero_for_spec(high);
            debug_assert!(zero_low.is_canonical());
            debug_assert!(zero_high.is_canonical());
            dependencies.push((zero_low, zero_high));
        }
        dependencies
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        if !spec_is_plain_matmul(spec) {
            updater.complete_spec(spec);
        }
    }

    fn visit_dependency<U>(&mut self, spec: &Spec<Tgt>, cost: &[NormalizedCost], updater: &mut U)
    where
        U: VisitUpdater<Tgt>,
    {
        debug_assert!(spec.is_canonical());

        let LogicalSpec::Primitive(basics, _, _) = &spec.0 else {
            unreachable!();
        };

        match basics.typ {
            PrimitiveSpecType::Fill {
                value: FillValue::Zero,
            } => {
                let insert_result = self.zeroes.insert(spec.clone(), cost.into());
                debug_assert!(insert_result.is_none(), "visited twice: {}", spec);
                let zero_volume = spec.0.volume();
                let accum_key = abstract_goal(spec);
                if let Some(accums_map) = self.accums.remove(&accum_key) {
                    for (matmulaccum, matmulaccum_costs) in accums_map {
                        self.visit_candidate_dependency_pair(
                            cost,
                            &matmulaccum_costs,
                            zero_volume,
                            matmulaccum.0.volume(),
                            &matmulaccum,
                            updater,
                        );
                    }
                }
            }
            PrimitiveSpecType::Matmul { accum: true } => {
                // zero_for_spec works both for constructing the Zero dependency of the Matmul *and*
                // the MatmulAccum's paired Zero.
                let corresponding_zero = zero_for_spec(spec);
                if let Some(zero_costs) = self.zeroes.get(&corresponding_zero) {
                    // Already visited the Zero dependency. Compute the final ToAccum Impl's cost.
                    self.visit_candidate_dependency_pair(
                        zero_costs,
                        cost,
                        corresponding_zero.0.volume(),
                        spec.0.volume(),
                        spec,
                        updater,
                    );
                } else {
                    let accum_key = abstract_goal(spec);
                    let final_tensorspec_map = self.accums.entry(accum_key).or_default();
                    let insert_result = final_tensorspec_map.insert(spec.clone(), cost.into());
                    debug_assert!(insert_result.is_none());
                }
            }
            _ => unreachable!(),
        };
    }
}

/// Returns true if the given Spec is a Matmul (not a MatmulAccum or another Spec).
fn spec_is_plain_matmul<Tgt: Target>(spec: &Spec<Tgt>) -> bool {
    match &spec.0 {
        LogicalSpec::Primitive(basics, _, _) => match basics.typ {
            PrimitiveSpecType::Matmul { accum } => !accum,
            _ => false,
        },
        LogicalSpec::Compose { .. } => false,
    }
}

fn matmul_from_toaccum_deps<Tgt: Target>(matmulaccum_spec: &Spec<Tgt>) -> Spec<Tgt> {
    let LogicalSpec::Primitive(accum_basics, accum_auxes, serial) = &matmulaccum_spec.0 else {
        panic!();
    };
    Spec(
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: accum_basics.spec_shape.clone(),
                dtypes: accum_basics.dtypes.clone(),
            },
            accum_auxes.clone(),
            *serial,
        ),
        matmulaccum_spec.1.clone(),
    )
}

/// Creates a canonical Zero with the same output shape, dtype, etc. as the given Spec.
fn zero_for_spec<Tgt: Target>(spec: &Spec<Tgt>) -> Spec<Tgt> {
    let TensorSpec {
        shape: output_shape,
        dtype: output_dtype,
        aux: output_aux,
    } = spec.0.unique_output().unwrap();
    let subspec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Fill {
                value: FillValue::Zero,
            },
            spec_shape: output_shape,
            dtypes: vec![output_dtype],
        },
        vec![output_aux],
        spec.0.serial_only(),
    );
    let mut spec = Spec(subspec, spec.1.clone());
    spec.canonicalize()
        .expect("ToAccum's introduced Zero should be canonicalizable");
    spec
}

/// Build an [AccumMapKey] from a goal Matmul Spec, its MatmulAccum analogue, or its Zero ToAccum
/// dependency.
fn abstract_goal<Tgt: Target>(spec: &Spec<Tgt>) -> AccumMapKey<Tgt> {
    // output_tensorspec is either the accumulating output of the MatmulAccum or the Zero's only
    // parameter (the same).
    let output_tensorspec = spec.0.unique_output().unwrap().clone();
    let serial_only = spec.0.serial_only();
    let mut limits = spec.1.clone();
    limits.zero_levels_slower_than_all::<Tgt>(&[output_tensorspec.level()]);
    (output_tensorspec, serial_only, limits)
}
