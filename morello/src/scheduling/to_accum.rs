use crate::cost::Cost;
use crate::imp::blocks::Block;
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::scheduling::{
    make_accum_inits_for_spec, ActionT, ApplyError, BottomUpSolver, NotApplicableReason,
};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToAccum;

#[derive(Default)]
pub struct ToAccumSolver<Tgt>(std::marker::PhantomData<Tgt>);

impl<Tgt: Target> ActionT<Tgt> for ToAccum {
    type BSolver = ToAccumSolver<Tgt>;

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
}

impl<Tgt: Target> BottomUpSolver for ToAccumSolver<Tgt> {
    type Tgt = Tgt;

    fn dependencies_for_spec(&self, spec: &Spec<Self::Tgt>) -> Vec<(Spec<Tgt>, Spec<Tgt>)> {
        todo!()
    }

    fn dependencies_for_range(
        &self,
        low: &Spec<Tgt>,
        high: &Spec<Tgt>,
    ) -> Vec<(Spec<Tgt>, Spec<Tgt>)> {
        todo!()
    }

    fn visit_dependency(&self, spec: &Spec<Tgt>, cost: &Cost) {
        todo!()
    }
}
