//! `reconstruct` contains functions for building optimal Impls from sets (e.g., databases) of
//! optimal actions. Callers are responsible for supplying canonical [Spec]s.
use crate::db::ActionCostVec;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::functions::FunctionApp;
use crate::imp::{Impl, ImplNode};
use crate::scheduling::ActionT as _;
use crate::spec::Spec;
use crate::target::Target;

/// Map the `spec` to the optimal Impl according to the action, then replace introduced sub-Specs by
/// looking them up in `lookup`.
pub fn reconstruct_impls_from_actions<Tgt, F>(
    lookup: &F,
    spec: &Spec<Tgt>,
    costs: ActionCostVec,
) -> Vec<ImplNode<Tgt>>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    F: Fn(&Spec<Tgt>) -> Option<ActionCostVec>,
{
    debug_assert!(spec.is_canonical(), "Spec must be canonical: {}", spec);
    let actions = Tgt::actions(&spec.0).collect::<Vec<_>>();
    costs
        .as_ref()
        .iter()
        .map(|(action_num, _)| {
            let root = actions[usize::from(*action_num)]
                .apply(spec)
                .expect("should be able to apply synthesized action");
            let new_children = root
                .children()
                .iter()
                .map(|child| reconstruct_children_from_optima(lookup, child))
                .collect::<Vec<_>>();
            root.replace_children(new_children.into_iter())
        })
        .collect()
}

/// Map every [crate::imp::subspecs::SpecApp] in `tree` by looking up the corresponding [Spec] with
/// `lookup`.
fn reconstruct_children_from_optima<Tgt, F>(lookup: &F, tree: &ImplNode<Tgt>) -> ImplNode<Tgt>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    F: Fn(&Spec<Tgt>) -> Option<ActionCostVec>,
{
    match tree {
        ImplNode::SpecApp(spec_app) => {
            debug_assert!(
                spec_app.0.is_canonical(),
                "Child Spec must be canonical: {}",
                spec_app.0
            );
            let spec = spec_app.0.clone();
            let action_costs = lookup(&spec).unwrap_or_else(|| {
                panic!("Database should contain synthesized Spec: {}", spec_app.0)
            });
            let impls = reconstruct_impls_from_actions(lookup, &spec, action_costs);
            let body = impls.into_iter().next().unwrap_or_else(|| {
                panic!("Database sub-Spec should be satisfiable: {}", spec_app.0)
            });
            ImplNode::FunctionApp(FunctionApp {
                body: Box::new(body),
                parameters: spec_app.1.clone(),
                spec: Some(spec_app.0.clone()),
            })
        }
        _ => tree.replace_children(
            tree.children()
                .iter()
                .map(|child| reconstruct_children_from_optima(lookup, child)),
        ),
    }
}
