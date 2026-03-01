use crate::common::DimSize;
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{
    make_inner_compose, make_outer_compose, ActionT, ApplyError, NotApplicableReason,
};
use crate::spec::{LogicalSpec, Spec};
use crate::target::{Target, LEVEL_COUNT, ZERO_LIMITS};
use crate::tensorspec::TensorSpec;
use crate::views::{Tensor, View};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Bufferize<Tgt: Target> {
    pub index: usize,
    pub level: Tgt::Level,
    pub layout: Layout,
    pub vector_size: Option<DimSize>,
}

impl<Tgt: Target> ActionT<Tgt> for Bufferize<Tgt> {
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

        if !self.layout.is_fully_contiguous() {
            // The following could be a warning since contiguousness is meant to
            // be an underapproximation. We'll generate code which allocates a contiguous buffer,
            // but it's not necessarily a problem to underapproximate contiguousness.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Buffer layout is not fully contiguous",
            ))));
        }

        debug_assert!(self.index < components.len() - 1);
        let consumer = &components[self.index];
        let intermediate_tensor = Tensor::new(
            TensorSpec::new_canon_checked(
                consumer.input_shape(0),
                consumer.input_dtype(0),
                self.level,
                self.layout.clone(),
                self.vector_size,
            )
            .map_err(|_| ApplyError::NotApplicable(NotApplicableReason::LayoutIncompatible))?,
        );

        // Compute the per-level memory consumed by the new intermediate tensor.
        let buffer_cost: [u64; LEVEL_COUNT] = Tgt::levels().map(|l| {
            if self.level == l {
                intermediate_tensor.spec().memory_units()
            } else {
                0u64
            }
        });

        // Extract the parent's limits, before, and after.
        let (parent_limits, parent_before, parent_after) = match &spec.1 {
            MemoryLimits::Standard(v) => (v.clone(), ZERO_LIMITS, ZERO_LIMITS),
            MemoryLimits::Pipeline {
                limits,
                before,
                after,
            } => (limits.clone(), *before, *after),
        };

        let check_oom =
            |before: &[u64; LEVEL_COUNT], after: &[u64; LEVEL_COUNT]| -> Result<(), ApplyError> {
                for i in 0..LEVEL_COUNT {
                    let needed = before[i] + after[i];
                    if needed > parent_limits.get_unscaled(i) {
                        return Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                            Tgt::levels()[i].to_string(),
                        )));
                    }
                }
                Ok(())
            };

        // Inner child (executed first): inherits parent's before, after = buffer.
        check_oom(&parent_before, &buffer_cost)?;
        let inner_limits = MemoryLimits::Pipeline {
            limits: parent_limits.clone(),
            before: parent_before,
            after: buffer_cost,
        };

        // Outer child (executed second): before = buffer, inherits parent's after.
        check_oom(&buffer_cost, &parent_after)?;
        let outer_limits = MemoryLimits::Pipeline {
            limits: parent_limits,
            before: buffer_cost,
            after: parent_after,
        };

        let inner_compose = ImplNode::from(make_inner_compose(
            self.index,
            components,
            operand_auxes,
            *serial_only,
            intermediate_tensor.clone(),
            inner_limits,
        ));
        let outer_compose = ImplNode::from(make_outer_compose(
            self.index,
            components,
            operand_auxes,
            *serial_only,
            intermediate_tensor.clone(),
            outer_limits,
        ));

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![inner_compose, outer_compose],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(intermediate_tensor)],
            }],
            spec: Some(spec.clone()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::imp::subspecs::SpecApp;
    use crate::imp::Impl;
    use crate::layout::row_major;
    use crate::memorylimits::MemVec;
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::shape;
    use crate::spec::{PrimitiveBasics, PrimitiveSpecType};
    use crate::target::{Avx2Target, CpuMemoryLevel, LEVEL_COUNT};
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

        let aux0_1 = TensorSpecAux {
            level: CpuMemoryLevel::GL,
            layout: row_major(&basics0.parameter_shape(1)),
            vector_size: None,
        };
        let aux1_1 = TensorSpecAux {
            level: CpuMemoryLevel::GL,
            layout: row_major(&basics1.parameter_shape(1)),
            vector_size: None,
        };
        let aux2_0 = TensorSpecAux {
            level: CpuMemoryLevel::GL,
            layout: row_major(&basics2.parameter_shape(0)),
            vector_size: None,
        };
        let aux2_1 = TensorSpecAux {
            level: CpuMemoryLevel::GL,
            layout: row_major(&basics2.parameter_shape(1)),
            vector_size: None,
        };
        let aux0_out = TensorSpecAux {
            level: CpuMemoryLevel::GL,
            layout: row_major(&basics0.parameter_shape(2)),
            vector_size: None,
        };
        let mut spec = Spec::<Avx2Target>(
            LogicalSpec::Compose {
                components: vec![basics0.clone(), basics1.clone(), basics2.clone()],
                operand_auxes: vec![aux0_1, aux1_1, aux2_0, aux2_1, aux0_out],
                serial_only: true,
            },
            Avx2Target::max_mem(),
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
        assert_eq!(first_child_shape, &basics2.spec_shape);
        assert_eq!(first_child_dtypes, &basics2.dtypes);
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
        assert_eq!(components, &[basics0.clone(), basics1.clone()]);
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

    /// Bufferizing a Spec with MemoryLimits::Standard should produce children
    /// with MemoryLimits::Pipeline, where `limits` equals the original Standard
    /// MemVec and `before`/`after` reflect the newly-introduced buffer.
    #[test]
    fn test_bufferize_standard_produces_pipeline_limits() {
        let MemoryLimits::Standard(original_vec) = Avx2Target::max_mem() else {
            panic!("max_mem should be Standard");
        };
        let spec = make_three_matmul_spec(Avx2Target::max_mem());
        let bc = buffer_cost_at_gl(&spec);
        let (m0, m1) = bufferize_child_limits(&spec);

        assert_eq!(
            m0,
            MemoryLimits::Pipeline {
                limits: original_vec.clone(),
                before: ZERO_LIMITS,
                after: bc
            }
        );
        assert_eq!(
            m1,
            MemoryLimits::Pipeline {
                limits: original_vec,
                before: bc,
                after: ZERO_LIMITS
            }
        );
    }

    /// Bufferizing a Spec with MemoryLimits::Pipeline should produce children
    /// whose `limits` is inherited, with parent before/after on the leading/
    /// trailing children and inner fields updated for the new buffer.
    #[test]
    fn test_bufferize_pipeline_produces_pipeline_limits() {
        let base_limits =
            MemVec::new_mixed([16, 16, 1_024, 33_554_432], [true, true, false, false]);
        let parent_before = [0u64, 0, 0, 128];
        let parent_after = [0u64, 0, 0, 256];
        let spec = make_three_matmul_spec(MemoryLimits::Pipeline {
            limits: base_limits.clone(),
            before: parent_before,
            after: parent_after,
        });
        let bc = buffer_cost_at_gl(&spec);
        let (m0, m1) = bufferize_child_limits(&spec);

        assert_eq!(
            m0,
            MemoryLimits::Pipeline {
                limits: base_limits.clone(),
                before: parent_before,
                after: bc
            }
        );
        assert_eq!(
            m1,
            MemoryLimits::Pipeline {
                limits: base_limits,
                before: bc,
                after: parent_after
            }
        );
    }

    /// Build a 3-Matmul Compose Spec with the given memory limits.
    fn make_three_matmul_spec(mem: MemoryLimits) -> Spec<Avx2Target> {
        let b = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![2, 1, 4, 4],
            dtypes: vec![Dtype::Float32; 3],
        };
        let gl_aux = |param: usize| TensorSpecAux {
            level: CpuMemoryLevel::GL,
            layout: row_major(&b.parameter_shape(param)),
            vector_size: None,
        };
        // Compose operand_auxes order: b0.p1, b1.p1, b2.p0, b2.p1, b0.p2
        let mut spec = Spec::<Avx2Target>(
            LogicalSpec::Compose {
                components: vec![b.clone(); 3],
                operand_auxes: vec![gl_aux(1), gl_aux(1), gl_aux(0), gl_aux(1), gl_aux(2)],
                serial_only: true,
            },
            mem,
        );
        spec.canonicalize().unwrap();
        spec
    }

    /// Compute the per-level memory cost of the intermediate buffer introduced
    /// by `bufferize(1, GL, row_major, None)` on a 3-Matmul Compose.
    fn buffer_cost_at_gl(spec: &Spec<Avx2Target>) -> [u64; LEVEL_COUNT] {
        let LogicalSpec::Compose { ref components, .. } = spec.0 else {
            panic!("expected Compose");
        };
        let consumer = &components[1];
        let shape = consumer.input_shape(0);
        let ts = TensorSpec::<Avx2Target>::new_canon_checked(
            shape.clone(),
            consumer.input_dtype(0),
            CpuMemoryLevel::GL,
            row_major(&shape),
            None,
        )
        .unwrap();
        Avx2Target::levels().map(|l| {
            if l == ts.level() {
                ts.memory_units()
            } else {
                0
            }
        })
    }

    /// Apply bufferize and return the two children's MemoryLimits.
    fn bufferize_child_limits(spec: &Spec<Avx2Target>) -> (MemoryLimits, MemoryLimits) {
        let imp = spec.bufferize(1, CpuMemoryLevel::GL, row_major, None);
        assert!(matches!(imp, ImplNode::Pipeline(_)));
        assert_eq!(imp.children().len(), 2);
        let ImplNode::SpecApp(SpecApp(Spec(_, ref m0), _)) = imp.children()[0] else {
            panic!("child 0: expected SpecApp");
        };
        let ImplNode::SpecApp(SpecApp(Spec(_, ref m1), _)) = imp.children()[1] else {
            panic!("child 1: expected SpecApp");
        };
        (m0.clone(), m1.clone())
    }
}
