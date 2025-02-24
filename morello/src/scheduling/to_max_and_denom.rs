use crate::imp::blocks::Block;
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::scheduling::{
    Action, ActionT, ApplyError, NaiveBottomUpActionProvider, NaiveBottomUpSolver,
    NotApplicableReason,
};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::views::{Param, ViewE};
use serde::{Deserialize, Serialize};
use std::iter;

/// Rewrites a SoftmaxDenominatorAndMax into a Max followed by SoftmaxDenominator.
#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToMaxAndDenominator;

#[derive(Default)]
pub struct ToMaxAndDenominatorActionProvider<Tgt>(std::marker::PhantomData<Tgt>);

impl<Tgt: Target> ActionT<Tgt> for ToMaxAndDenominator {
    type BSolver = NaiveBottomUpSolver<Tgt, ToMaxAndDenominatorActionProvider<Tgt>>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        let LogicalSpec::Primitive(head, _, serial_only) = logical_spec else {
            // TODO: Add a more specific NotApplicableReason
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToMaxAndDenominator only defined for Primitive",
            ))));
        };
        let PrimitiveBasics {
            typ: PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim },
            spec_shape: _,
            dtypes: _,
        } = head
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToMaxAndDenominator is only defined for SoftmaxDenominatorAndMax",
            ))));
        };

        let max_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::Max {
                dim: *scan_dim,
                accum: false,
            },
            [
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(Param::new(1, operands[1].clone())),
            ],
            *serial_only,
            spec.1.clone(),
        ));

        let denom_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::SoftmaxDenominator {
                scan_dim: *scan_dim,
                accum: false,
            },
            [
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(Param::new(1, operands[1].clone())),
                ViewE::from(Param::new(2, operands[2].clone())),
            ],
            *serial_only,
            spec.1.clone(),
        ));

        Ok(ImplNode::Block(Block {
            stages: vec![max_app, denom_app],
            parameters: operands,
            spec: Some(spec.clone()),
            default_child: None,
        }))
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for ToMaxAndDenominatorActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        // TODO: Return the actions!
        vec![]
    }
}
