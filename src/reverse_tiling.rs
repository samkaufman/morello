use crate::{
    imp::{loops::Loop, ImplNode},
    scheduling_sugar::SchedulingSugar,
    spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec},
    target::Target,
};

pub fn reverse_tile<Tgt: Target>(outer: &Spec<Tgt>, body: Spec<Tgt>) -> Loop<Tgt, ()> {
    let parallel = !outer.0.serial_only() && body.0.serial_only();

    let reproduced_impl_node = match &outer.0 {
        LogicalSpec::Primitive(outer_basics, ..) => match outer_basics.typ {
            PrimitiveSpecType::Zero | PrimitiveSpecType::Move => {
                let LogicalSpec::Primitive(inner_basics, ..) = &body.0 else {
                    panic!();
                };
                debug_assert_eq!(outer_basics.typ, inner_basics.typ);
                outer.tile_out(&inner_basics.spec_shape, parallel)
            }
            PrimitiveSpecType::Matmul { accum: _ } => {
                let LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Matmul { .. },
                        spec_shape: inner_shape,
                        dtype: _,
                    },
                    ..,
                ) = &body.0
                else {
                    panic!("Outer Spec is Matmul, but inner Spec is not");
                };

                if outer_basics.spec_shape[1] != inner_shape[1] {
                    debug_assert_eq!(outer_basics.spec_shape[0], inner_shape[0]);
                    debug_assert_eq!(outer_basics.spec_shape[2], inner_shape[2]);
                    outer.split(inner_shape[1])
                } else {
                    debug_assert_eq!(outer_basics.spec_shape[1], inner_shape[1]);
                    outer.tile_out(&[inner_shape[0], inner_shape[2]], parallel)
                }
            }
            PrimitiveSpecType::Conv { .. } => {
                let LogicalSpec::Primitive(inner_basics, ..) = &body.0 else {
                    panic!();
                };
                debug_assert_eq!(outer_basics.typ, inner_basics.typ);
                let inner_parameter_shapes = inner_basics.parameter_shapes();
                outer.tile_out(&inner_parameter_shapes[2], parallel)
            }
        },
        LogicalSpec::Compose { .. } => todo!(),
    };

    match reproduced_impl_node {
        ImplNode::Loop(l) => l,
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::reverse_tile;
    use crate::{
        imp::{loops::Loop, ImplNode},
        scheduling::{Action, ApplyError},
        spec::Spec,
        target::{ArmTarget, Target, X86Target},
    };
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_tile_out_in_actions_are_reversible_x86(
            spec in any::<Spec<X86Target>>()
                .prop_filter("Spec should be canonical", |s| s.is_canonical())
        ) {
            assert_action_is_reversible(&spec, |action| matches!(&action, Action::TileOut { .. }));
        }

        #[test]
        fn test_tile_out_in_actions_are_reversible_arm(
            spec in any::<Spec<ArmTarget>>()
                .prop_filter("Spec should be canonical", |s| s.is_canonical())
        ) {
            assert_action_is_reversible(&spec, |action| matches!(&action, Action::TileOut { .. }));
        }

        #[test]
        fn test_split_in_actions_are_reversible_x86(
            spec in any::<Spec<X86Target>>()
                .prop_filter("Spec should be canonical", |s| s.is_canonical())
        ) {
            assert_action_is_reversible(&spec, |action| matches!(&action, Action::Split { .. }));
        }

        #[test]
        fn test_split_in_actions_are_reversible_arm(
            spec in any::<Spec<ArmTarget>>()
                .prop_filter("Spec should be canonical", |s| s.is_canonical())
        ) {
            assert_action_is_reversible(&spec, |action| matches!(&action, Action::Split { .. }));
        }
    }

    fn assert_action_is_reversible<Tgt, F>(spec: &Spec<Tgt>, filter_fn: F)
    where
        Tgt: Target,
        F: Fn(&Action<Tgt>) -> bool,
    {
        for action in spec.0.actions() {
            if !filter_fn(&action) {
                continue;
            }
            let Some((loop_impl, body_spec)) = apply_and_extract_loop_body(spec, &action) else {
                continue;
            };
            let reversed = reverse_tile(spec, body_spec);
            assert_eq!(
                (
                    loop_impl
                        .tiles
                        .iter()
                        .map(|t| t.axes.clone())
                        .collect::<Vec<_>>(),
                    loop_impl.parallel,
                    loop_impl.aux
                ),
                (
                    reversed
                        .tiles
                        .iter()
                        .map(|t| t.axes.clone())
                        .collect::<Vec<_>>(),
                    reversed.parallel,
                    reversed.aux
                )
            );
        }
    }

    fn apply_and_extract_loop_body<Tgt: Target>(
        spec: &Spec<Tgt>,
        action: &Action<Tgt>,
    ) -> Option<(Loop<Tgt, ()>, Spec<Tgt>)> {
        match action.apply(spec) {
            Ok(ImplNode::Loop(loop_node)) => match loop_node.body.as_ref() {
                ImplNode::SpecApp(spec_app) => Some((loop_node.clone(), spec_app.0.clone())),
                _ => panic!("Unexpected loop body node: {:?}", loop_node.body),
            },
            Ok(n) => panic!("Unexpected Impl node from tile_out apply: {n:?}"),
            Err(ApplyError::OutOfMemory | ApplyError::ActionNotApplicable) => None,
            Err(ApplyError::SpecNotCanonical) => unreachable!(),
        }
    }
}
