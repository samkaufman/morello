use crate::cost::{Cost, NormalizedCost};
use crate::imp::blocks::Block;
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::{
    make_accum_inits_for_spec, Action, ActionEncodeDecode, ActionT, ApplyError, BottomUpSolver,
    DependencyRequest, NotApplicableReason, SpecGeometry, VisitUpdater,
};
use crate::spec::{FillValue, LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::iter;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::rc::Rc;

#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToAccum;

type AccumMapKey<Tgt> = (TensorSpec<Tgt>, bool, MemoryLimits);

pub struct ToAccumSolver<Tgt>(PhantomData<Tgt>);

pub struct ToAccumSolverRequest<Tgt: Target> {
    dependencies: SpecGeometry<Tgt>,
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

        let Some(accum) = head.accum() else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToAccum is not defined for this Spec kind",
            ))));
        };
        if accum {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Already accumulating",
            ))));
        }

        let mut accum_logical_spec = logical_spec.clone();
        *accum_logical_spec.mut_accum().unwrap() = true;
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
        iter::once(ToAccumSolver(PhantomData))
    }
}

impl<Tgt: Target> BottomUpSolver for ToAccumSolver<Tgt> {
    type Tgt = Tgt;
    type Request = ToAccumSolverRequest<Tgt>;

    fn request(&mut self, dependents: &SpecGeometry<Tgt>) -> Self::Request {
        let mut dependencies = SpecGeometry::<Tgt>::new(Rc::clone(dependents.bimap()));
        dependencies.extend(dependents.accums());
        let zeros = dependencies.outputs_zeros().collect::<Vec<_>>();
        dependencies.extend(zeros.into_iter());
        ToAccumSolverRequest {
            dependencies,
            zeroes: HashMap::new(),
            accums: HashMap::new(),
        }
    }
}

impl<Tgt: Target> ToAccumSolverRequest<Tgt> {
    /// Helper for [visit_dependency]. Called when both the Zero and accumulating dependencies are
    /// available.
    fn visit_candidate_dependency_pair<U>(
        &self,
        zero_costs: &[NormalizedCost],
        accum_costs: &[NormalizedCost],
        zero_volume: NonZeroU64,
        accum_volume: NonZeroU64,
        accum_spec: &Spec<Tgt>,
        updater: &mut U,
    ) where
        U: VisitUpdater<Tgt>,
    {
        let mut goal = accum_spec.clone();
        *goal.0.mut_accum().unwrap() = false;

        assert_eq!(goal.0.accum(), Some(false), "goal was not accum: {goal:?}");

        match (accum_costs, zero_costs) {
            ([], _) | (_, []) => {
                // TODO: Prevent calling this twice per goal.
                updater.complete_spec(&goal);
            }
            ([matmulaccum_normalized_cost], [zero_normalized_cost]) => {
                let matmulaccum_cost = matmulaccum_normalized_cost
                    .intensity
                    .into_main_cost_for_volume(accum_volume);
                let zero_cost = zero_normalized_cost
                    .intensity
                    .into_main_cost_for_volume(zero_volume);
                let cost_sum = Cost {
                    main: matmulaccum_cost.saturating_add(zero_cost),
                    peaks: MemoryAllocation::none()
                        .peak_memory_from_child_peaks::<Tgt>(&[
                            matmulaccum_normalized_cost.peaks.clone(),
                            zero_normalized_cost.peaks.clone(),
                        ])
                        .snap_up_for_target::<Tgt>(false),
                    depth: 1 + matmulaccum_normalized_cost
                        .depth
                        .max(zero_normalized_cost.depth),
                };
                updater.complete_action(
                    &goal,
                    Action::ToAccum(ToAccum).encode(&goal),
                    NormalizedCost::new(cost_sum, accum_volume),
                );
                updater.complete_spec(&goal);
            }
            _ => todo!("handle k>2"),
        }
    }
}

impl<Tgt: Target> DependencyRequest for ToAccumSolverRequest<Tgt> {
    type Tgt = Tgt;

    fn queries(&self) -> Option<&SpecGeometry<Tgt>> {
        Some(&self.dependencies)
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        match spec.0.accum() {
            Some(true) | None => updater.complete_spec(spec),
            Some(false) => {}
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
                    for (accum_spec, accum_costs) in accums_map {
                        self.visit_candidate_dependency_pair(
                            cost,
                            &accum_costs,
                            zero_volume,
                            accum_spec.0.volume(),
                            &accum_spec,
                            updater,
                        );
                    }
                }
            }
            _ => {
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
        };
    }
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
