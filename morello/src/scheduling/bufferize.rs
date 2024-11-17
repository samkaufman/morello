use crate::common::DimSize;
use crate::cost::NormalizedCost;
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{
    make_inner_compose, make_outer_compose, ActionT, ApplyError, BottomUpSolver,
    NotApplicableReason, VisitUpdater,
};
use crate::spec::{LogicalSpec, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::Tensor;
use serde::{Deserialize, Serialize};
use std::iter;
use std::rc::Rc;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Bufferize<Tgt: Target> {
    pub index: usize,
    pub level: Tgt::Level,
    pub layout: Layout,
    pub vector_size: Option<DimSize>,
}

#[derive(Default)]
pub struct BufferizeSolver<Tgt>(std::marker::PhantomData<Tgt>);

impl<Tgt: Target> ActionT<Tgt> for Bufferize<Tgt> {
    type BSolver = BufferizeSolver<Tgt>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;

        let LogicalSpec::Compose {
            components,
            operand_auxes,
            serial_only,
        } = logical_spec
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Not a Compose",
            ))));
        };

        debug_assert!(self.index < components.len() - 1);
        let consumer = &components[self.index];
        let intermediate_tensor = Tensor::new(
            TensorSpec::new_canon_checked(
                consumer.input_shape(0),
                consumer.input_dtype(0),
                self.layout.contiguous_full(),
                true,
                self.level,
                self.layout.clone(),
                self.vector_size,
            )
            .map_err(|_| ApplyError::NotApplicable(NotApplicableReason::LayoutIncompatible))?,
        );

        // Compute the memory limits for the new children.
        let new_limits = {
            // Compute the amount of memory consumed by the new, intermediate
            // tensor.
            // TODO: This shouldn't need to be both here and in `memory_allocated`.
            let first_input_volume: u64 = consumer
                .input_shape(0)
                .into_iter()
                .map(|v| u64::from(v.get()))
                .product();
            let intermediate_mem_consumed_nondiscrete = Tgt::levels().map(|l| {
                if self.level == l {
                    u64::from(consumer.dtypes[0].size()) * first_input_volume
                } else {
                    0u64
                }
            });

            // TODO: Use MemoryLimits::Pipeline where appropriate instead.
            // (Already done by Pipeline::memory_allocated.)
            let mut m = MemoryLimits::Standard(match &spec.1 {
                MemoryLimits::Standard(v) => v
                    .clone()
                    .checked_sub_snap_down(&intermediate_mem_consumed_nondiscrete)
                    .map_err(|oom_idx| {
                        ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                            Tgt::levels()[oom_idx].to_string(),
                        ))
                    })?,
            });
            m.discretize();
            m
        };

        let inner_compose = ImplNode::from(make_inner_compose(
            self.index,
            components,
            operand_auxes,
            *serial_only,
            intermediate_tensor.clone(),
            new_limits.clone(),
        ));
        let outer_compose = ImplNode::from(make_outer_compose(
            self.index,
            components,
            operand_auxes,
            *serial_only,
            intermediate_tensor.clone(),
            new_limits,
        ));

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![inner_compose, outer_compose],
            parameters: spec.0.parameters(),
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(intermediate_tensor)],
            }],
            spec: Some(spec.clone()),
        }))
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

impl<Tgt: Target> BottomUpSolver for BufferizeSolver<Tgt> {
    type Tgt = Tgt;

    fn dependencies_for_spec(&mut self, spec: &Spec<Tgt>) -> Vec<(Spec<Tgt>, Spec<Tgt>)> {
        self.dependencies_for_range(spec, spec)
    }

    fn dependencies_for_range(
        &mut self,
        low: &Spec<Tgt>,
        high: &Spec<Tgt>,
    ) -> Vec<(Spec<Tgt>, Spec<Tgt>)> {
        if matches!(&low.0, LogicalSpec::Compose { .. })
            || matches!(&high.0, LogicalSpec::Compose { .. })
        {
            todo!()
        } else {
            vec![]
        }
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        if !matches!(&spec.0, LogicalSpec::Compose { .. }) {
            updater.complete_spec(spec);
        }
    }

    fn visit_dependency<U>(&mut self, spec: &Spec<Tgt>, cost: &[NormalizedCost], updater: &mut U)
    where
        U: VisitUpdater<Tgt>,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::imp::subspecs::SpecApp;
    use crate::imp::Impl;
    use crate::layout::row_major;
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::shape;
    use crate::spec::{PrimitiveBasics, PrimitiveSpecType};
    use crate::target::{CpuMemoryLevel, X86Target};
    use crate::tensorspec::TensorSpecAux;
    use crate::views::{Param, ViewE};

    /// Test that bufferizing a chain of 3 Matmuls produces the correct sub-Spec applications.
    #[test]
    fn test_bufferize_matmul_chain() {
        let basics0 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![3, 12, 2, 4],
            dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
        };
        let basics1 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![32, 1, 4, 8],
            dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
        };
        let basics2 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![32, 1, 8, 16],
            dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
        };
        let aux = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: true,
            level: CpuMemoryLevel::GL,
            layout: row_major(3),
            vector_size: None,
        };
        let mut spec = Spec::<X86Target>(
            LogicalSpec::Compose {
                components: vec![basics2.clone(), basics1.clone(), basics0.clone()],
                operand_auxes: vec![aux.clone(), aux.clone(), aux.clone(), aux.clone(), aux],
                serial_only: true,
            },
            X86Target::max_mem(),
        );
        spec.canonicalize().unwrap();
        let imp = spec.bufferize(1, CpuMemoryLevel::GL, row_major, None);

        assert_eq!(imp.children().len(), 2);
        assert!(matches!(imp, ImplNode::Pipeline(_)));

        let ImplNode::SpecApp(SpecApp(Spec(first_child_lspec, _), first_child_params)) =
            &imp.children()[0]
        else {
            panic!("expected a SpecApp child, got {:?}", imp.children()[0])
        };
        let LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: first_child_shape,
                dtypes: first_child_dtypes,
            },
            _,
            true,
        ) = first_child_lspec
        else {
            panic!("expected a serial, non-accum Matmul Primitive, got {first_child_lspec:?}");
        };
        assert_eq!(first_child_shape, &basics0.spec_shape);
        assert_eq!(first_child_dtypes, &basics0.dtypes);
        assert_eq!(first_child_params.len(), 3);
        assert!(matches!(
            first_child_params[0],
            ViewE::Param(Param(2, _, _))
        ));
        assert!(matches!(
            first_child_params[1],
            ViewE::Param(Param(3, _, _))
        ));
        assert!(matches!(first_child_params[2], ViewE::Tensor(_)));

        let ImplNode::SpecApp(SpecApp(Spec(second_child_lspec, _), second_child_params)) =
            &imp.children()[1]
        else {
            panic!("expected a SpecApp child, got {:?}", imp.children()[1])
        };
        let LogicalSpec::Compose {
            components,
            operand_auxes: _,
            serial_only: true,
        } = second_child_lspec
        else {
            panic!("expected a serial Compose LogicalSpec, got {second_child_lspec:?}");
        };
        assert_eq!(components, &[basics2.clone(), basics1.clone()]);
        assert_eq!(second_child_params.len(), 4);
        assert!(matches!(
            second_child_params[0],
            ViewE::Param(Param(0, _, _))
        ));
        assert!(matches!(second_child_params[1], ViewE::Tensor(_)));
        assert!(matches!(
            second_child_params[2],
            ViewE::Param(Param(1, _, _))
        ));
        assert!(matches!(
            second_child_params[3],
            ViewE::Param(Param(4, _, _))
        ));
    }
}
