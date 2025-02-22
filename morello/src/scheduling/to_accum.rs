use crate::common::DimSize;
use crate::cost::{Cost, NormalizedCost};
use crate::datadeps::SpecKey;
use crate::imp::blocks::Block;
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::{
    make_accum_inits_for_spec, Action, ActionEncodeDecode, ActionT, ApplyError, BottomUpSolver,
    DependencyRequest, NotApplicableReason, SpecGeometry, SpecGeometryRect, VisitUpdater,
};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::utils::snap_memvec_up;
use crate::views::{Param, ViewE};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::iter;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToAccum;

/// A [HashMap] key for grouping non-Fill Specs by output tensor, serial flag, and memory limits.
/// These are used to look up applicable when visiting Fills.
type AccumMapKey<Tgt> = (TensorSpec<Tgt>, bool, MemoryLimits); // TODO: This doesn't distinguish values

#[derive(Default)]
pub struct ToAccumSolver<Tgt>(PhantomData<Tgt>);

pub struct ToAccumSolverRequest<Tgt: Target> {
    dependencies: SpecGeometry<Tgt>,
    costs: HashMap<Spec<Tgt>, Vec<NormalizedCost>>,
    nonfills_by_key: HashMap<AccumMapKey<Tgt>, Vec<Spec<Tgt>>>,
}

impl<Tgt: Target> ActionT<Tgt> for ToAccum {
    type BSolver = ToAccumSolver<Tgt>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

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

        let accum_logical_spec = logical_spec.clone_as_accum();
        let mut accum_spec = Spec(accum_logical_spec, spec.1.clone());
        accum_spec
            .canonicalize()
            .expect("ToAccum's introduced accumulating Spec should be canonicalizable");
        let app_arguments = operands
            .iter()
            .enumerate()
            .map(|(i, t)| ViewE::from(Param::new(i.try_into().unwrap(), t.clone())));
        let fill_apps = make_accum_inits_for_spec(&accum_spec);
        let accum_app = SpecApp::new(accum_spec, app_arguments).into();

        let mut stages = fill_apps;
        stages.push(accum_app);
        let default_child = Some(stages.len() - 1);
        Ok(ImplNode::Block(Block {
            stages,
            parameters: operands,
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
        let fills = dependencies.outputs_inits().collect::<Vec<_>>();
        dependencies.extend(fills.into_iter());
        ToAccumSolverRequest {
            dependencies,
            costs: HashMap::new(),
            nonfills_by_key: HashMap::new(),
        }
    }
}

impl<Tgt: Target> ToAccumSolverRequest<Tgt> {
    /// Helper for [visit_dependency]. Called when both the Fill and accumulating dependencies are
    /// available.
    fn complete_candidate_dependency_pair<U, C, Cv, D>(
        &self,
        fills_costs: C,
        accum_costs: &[NormalizedCost],
        fills_volumes: D,
        accum_volume: DimSize,
        accum_spec: &Spec<Tgt>,
        updater: &mut U,
    ) where
        U: VisitUpdater<Tgt>,
        C: IntoIterator<Item = Cv>,
        Cv: AsRef<[NormalizedCost]>,
        D: IntoIterator<Item = DimSize>,
    {
        let goal = Spec(accum_spec.0.clone_as_non_accum(), accum_spec.1.clone());

        assert_eq!(goal.0.accum(), Some(false), "goal was not accum: {goal:?}");

        let (mut main_cost_sum, mut child_peaks, mut child_depth) = if accum_costs.is_empty() {
            updater.complete_spec(&goal);
            return;
        } else if accum_costs.len() > 1 {
            todo!("handle k>1");
        } else {
            (
                accum_costs[0]
                    .intensity
                    .into_main_cost_for_volume(accum_volume),
                vec![accum_costs[0].peaks.clone()],
                accum_costs[0].depth,
            )
        };

        for (fill_costs, fill_volume) in fills_costs.into_iter().zip(fills_volumes) {
            if fill_costs.as_ref().is_empty() {
                // TODO: Prevent calling this twice per goal.
                updater.complete_spec(&goal);
            } else if fill_costs.as_ref().len() > 1 {
                todo!("handle k>1");
            } else {
                main_cost_sum = main_cost_sum.saturating_add(
                    fill_costs.as_ref()[0]
                        .intensity
                        .into_main_cost_for_volume(fill_volume),
                );
                child_peaks.push(fill_costs.as_ref()[0].peaks.clone());
                child_depth = child_depth.max(fill_costs.as_ref()[0].depth);
            }
        }

        let complex_cost = Cost {
            main: main_cost_sum,
            peaks: snap_memvec_up(
                MemoryAllocation::none().peak_memory_from_child_peaks::<Tgt>(&child_peaks),
                false,
            ),
            depth: 1 + child_depth,
        };
        updater.complete_action(
            &goal,
            Action::ToAccum(ToAccum).encode(&goal),
            NormalizedCost::new(complex_cost, accum_volume),
        );
        updater.complete_spec(&goal);
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

    fn visit_dependency<U>(
        &mut self,
        rectangle: &SpecGeometryRect<Tgt>,
        cost: &[NormalizedCost],
        updater: &mut U,
    ) where
        U: VisitUpdater<Tgt>,
    {
        rectangle.iter_specs().for_each(|spec| {
            debug_assert!(spec.is_canonical());
            let LogicalSpec::Primitive(basics, _, _) = &spec.0 else {
                unreachable!();
            };

            match basics.typ {
                PrimitiveSpecType::Fill { value: _ } => {
                    // Store the Fill cost.
                    let insert_result = self.costs.insert(spec.clone(), cost.into());
                    debug_assert!(insert_result.is_none(), "visited twice: {}", spec);

                    let abstracted_key = key_for_fill(&spec);
                    if let Some(relevant_non_fills_specs) =
                        self.nonfills_by_key.get(&abstracted_key)
                    {
                        // Look up the Fills for the other parameters of those non-Fills.
                        for non_fill_spec in relevant_non_fills_specs {
                            let Some(non_fill_costs) = self.costs.get(non_fill_spec) else {
                                continue;
                            };
                            let mut fill_parameters = vec![];
                            let mut fill_costs = vec![];
                            for i in 0..non_fill_spec.0.operand_count() {
                                if !non_fill_spec.0.parameter_is_output(i) {
                                    continue;
                                }
                                let parameter = non_fill_spec.0.parameter(i);
                                fill_parameters.push(parameter);
                                fill_costs.push(self.costs.get());
                            }

                            // self.complete_candidate_dependency_pair(fills_costs, accum_costs, fills_volumes, accum_volume, accum_spec, updater);
                        }
                    }

                    let fill_volume = spec.0.volume();
                    let accum_key = key_for_fill(&spec);
                    if let Some(accums_map) = self.nonfills_by_key.remove(&accum_key) {
                        for accum_spec in accums_map {
                            self.complete_candidate_dependency_pair(
                                cost,
                                &self.costs.get(&accum_spec).unwrap(),
                                fill_volume,
                                accum_spec.0.volume(),
                                &accum_spec,
                                updater,
                            );
                        }
                    }
                }
                _ => {
                    // Got the accumulating sub-Spec. If all of the Fill dependencies have been visited,
                    // we can complete the Spec. Otherwise, store the accumulating Spec's cost for later.
                    let mut fills_costs_vec = vec![];
                    let mut fills_volumes_vec = vec![];
                    for (output_idx, initializing_fill) in fills_for_spec(&spec) {
                        let Some(fill_costs) = self.costs.get(&initializing_fill) else {
                            let accum_key = key_for_output(&spec, output_idx);
                            let final_tensorspec_map =
                                self.nonfills_by_key.entry(accum_key).or_default();
                            let insert_result = self.costs.insert(spec.clone(), cost.into());
                            debug_assert!(insert_result.is_none());
                            return;
                        };
                        fills_costs_vec.push(fill_costs);
                        fills_volumes_vec.push(initializing_fill.0.volume());
                    }

                    // Already visited the Fill dependencies. Compute the final ToAccum Impl's cost.
                    self.complete_candidate_dependency_pair(
                        fills_costs_vec,
                        cost,
                        fills_volumes_vec,
                        spec.0.volume(),
                        &spec,
                        updater,
                    );
                    todo!("Call visit_candidate_dependency_pair");
                }
            };
        });
    }
}

/// Creates canonical Fills with the same output shape, dtype, etc. as the given Spec's output
/// parameters.
fn fills_for_spec<Tgt: Target>(spec: &Spec<Tgt>) -> impl Iterator<Item = (usize, Spec<Tgt>)> + '_ {
    (0..spec.0.operand_count())
        .filter(move |&i| spec.0.parameter_is_output(i))
        .map(move |i| {
            let TensorSpec {
                shape: output_shape,
                dtype: output_dtype,
                aux: output_aux,
            } = spec.0.parameter(i);
            let subspec = LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Fill {
                        value: spec.0.initial_accumulating_value_for_output(i).unwrap(),
                    },
                    spec_shape: output_shape,
                    dtypes: vec![output_dtype],
                },
                vec![output_aux],
                spec.0.serial_only(),
            );
            let mut fill_spec = Spec(subspec, spec.1.clone());
            fill_spec
                .canonicalize()
                .expect("ToAccum's introduced Fill should be canonicalizable");
            (i, fill_spec)
        })
}

/// Build an [AccumMapKey] from a Fill dependency.
fn key_for_fill<Tgt: Target>(spec: &Spec<Tgt>) -> AccumMapKey<Tgt> {
    debug_assert_eq!(spec.0.operand_count(), 1);
    key_for_output(spec, 0)
}

fn key_for_output<Tgt: Target>(spec: &Spec<Tgt>, parameter_idx: usize) -> AccumMapKey<Tgt> {
    // output_tensorspec is either the accumulating output of the MatmulAccum or the Fill's only
    // parameter (the same).
    let output_tensorspec = spec.0.parameter(parameter_idx);
    debug_assert!(spec.0.parameter_is_output(parameter_idx));
    let serial_only = spec.0.serial_only();
    let mut limits = spec.1.clone();
    limits.zero_levels_slower_than_all::<Tgt>(&[output_tensorspec.level()]);
    (output_tensorspec, serial_only, limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        db::db_spec_bimap,
        grid::{canon::CanonicalBimap, general::BiMap},
        imp::visit_subspecs,
        spec::arb_canonical_spec,
        target::{ArmTarget, X86Target},
    };
    use itertools::Itertools;
    use proptest::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn test_to_accum_solver_computes_same_goal_costs_x86() {
        shared_test_to_accum_solver_computes_same_goal_costs::<X86Target>();
    }

    #[test]
    fn test_to_accum_solver_computes_same_goal_costs_arm() {
        shared_test_to_accum_solver_computes_same_goal_costs::<ArmTarget>();
    }

    proptest! {
        #[test]
        fn test_to_accum_solver_returns_same_dependencies_x86_unscaled(
            spec in arb_canonical_spec::<X86Target>(None, None)
                .prop_filter("not compose", |s| !matches!(s.0, LogicalSpec::Compose { .. })),
        ) {
            shared_test_to_accum_solver_returns_same_dependencies(spec, false);
        }

        #[test]
        fn test_to_accum_solver_returns_same_dependencies_arm_unscaled(
            spec in arb_canonical_spec::<ArmTarget>(None, None),
        ) {
            shared_test_to_accum_solver_returns_same_dependencies(spec, false);
        }

        #[test]
        fn test_to_accum_solver_returns_same_dependencies_x86_scaled(
            spec in arb_canonical_spec::<X86Target>(None, None),
        ) {
            shared_test_to_accum_solver_returns_same_dependencies(spec, true);
        }

        #[test]
        fn test_to_accum_solver_returns_same_dependencies_arm_scaled(
            spec in arb_canonical_spec::<ArmTarget>(None, None),
        ) {
            shared_test_to_accum_solver_returns_same_dependencies(spec, true);
        }
    }

    // TODO: Modify to take larger Spec dependents, not just a single Spec dependent.
    fn shared_test_to_accum_solver_returns_same_dependencies<Tgt>(
        spec: Spec<Tgt>,
        binary_scale_shapes: bool,
    ) where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        // TODO: Lift the following for proptest
        let bimap = Rc::new(db_spec_bimap(binary_scale_shapes));
        let dependents = SpecGeometry::<Tgt>::single(&spec, bimap);

        let mut expected_dependencies = HashSet::new();
        for rect in dependents.iter() {
            for goal in rect.iter_specs() {
                if let Ok(imp) = ToAccum.apply(&goal) {
                    visit_subspecs(&imp, &mut |subspec| {
                        expected_dependencies.insert(subspec.clone());
                        true
                    });
                }
            }
        }

        let mut solver = ToAccumSolver::default();
        let request = solver.request(&dependents);
        if let Some(queries) = request.queries() {
            let mut queried_specs = HashSet::new();
            for rect in queries.iter() {
                for s in rect.iter_specs() {
                    queried_specs.insert(s.clone());
                }
            }
            assert_eq!(
                queried_specs,
                expected_dependencies,
                "mismatched dependencies for {spec}: {}\nand\n{}",
                queried_specs.iter().join(", "),
                expected_dependencies.iter().join(", "),
            );
        } else {
            assert!(expected_dependencies.is_empty());
        }
    }

    fn shared_test_to_accum_solver_computes_same_goal_costs<Tgt: Target>() {
        todo!()
    }
}
