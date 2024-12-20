use crate::imp::blocks::Block;
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::scheduling::{ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::views::{Param, ViewE};
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToMaxAndDenominator {}

impl<Tgt: Target> ActionT<Tgt> for ToMaxAndDenominator {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        let LogicalSpec::Primitive(head, auxes, serial_only) = logical_spec else {
            // TODO: Add a more specific NotApplicableReason
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToMaxAndDenominator only defined for Primitive",
            ))));
        };
        let PrimitiveBasics {
            typ: PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim },
            spec_shape,
            dtypes,
        } = head
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToMaxAndDenominator is only defined for SoftmaxDenominatorAndMax",
            ))));
        };

        let max_app = {
            let mut max_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Max {
                            dim: *scan_dim,
                            accum: false,
                        },
                        spec_shape: spec_shape.clone(),
                        dtypes: vec![dtypes[0], dtypes[1]],
                    },
                    vec![operands[0].aux.clone(), operands[1].aux.clone()],
                    *serial_only,
                ),
                spec.1.clone(),
            );
            max_spec.canonicalize().unwrap();
            let app_args = vec![
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(Param::new(1, operands[1].clone())),
            ];
            SpecApp::new(max_spec, app_args).into()
        };

        let denom_app = {
            let mut denom_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::SoftmaxDenominator {
                            scan_dim: *scan_dim,
                            accum: false,
                        },
                        spec_shape: spec_shape.clone(),
                        dtypes: dtypes.clone(),
                    },
                    auxes.clone(),
                    *serial_only,
                ),
                spec.1.clone(),
            );
            denom_spec.canonicalize().unwrap();
            let app_args = vec![
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(Param::new(1, operands[1].clone())),
                ViewE::from(Param::new(2, operands[2].clone())),
            ];
            SpecApp::new(denom_spec, app_args).into()
        };

        Ok(ImplNode::Block(Block {
            stages: vec![max_app, denom_app],
            parameters: operands,
            spec: Some(spec.clone()),
            default_child: None,
        }))
    }
}
