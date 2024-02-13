//! Extends [crate::codegen::BuiltArtifact] with methods to check correctness of lowered code.
#![cfg(feature = "verification")]

use ndarray::prelude::*;

use crate::{
    codegen::BuiltArtifact,
    common::Dtype,
    spec::{LogicalSpec, PrimitiveBasics, Spec},
    target::Target,
};

impl BuiltArtifact {
    /// Check whether the artifact correctly implements a [Spec].
    ///
    /// This method can be used for a little extra defense against bugs in Morello or the underlying
    /// C compiler.
    pub fn check_correctness<Tgt: Target>(&self, spec: &Spec<Tgt>) -> bool {
        let test_result = match &spec.0 {
            LogicalSpec::Primitive(PrimitiveBasics { dtypes, .. }, _, _) => {
                if dtypes.iter().skip(1).any(|&dt| dtypes[0] != dt) {
                    todo!("Implement correctness check for non-homogeneous types");
                }
                match dtypes[0] {
                    Dtype::Uint8 => test_artifact_correct_inner::<Tgt, u8>(spec, self),
                    Dtype::Sint8 => test_artifact_correct_inner::<Tgt, i8>(spec, self),
                    Dtype::Uint16 => test_artifact_correct_inner::<Tgt, u16>(spec, self),
                    Dtype::Sint16 => test_artifact_correct_inner::<Tgt, i16>(spec, self),
                    Dtype::Uint32 => test_artifact_correct_inner::<Tgt, u32>(spec, self),
                    Dtype::Sint32 => test_artifact_correct_inner::<Tgt, i32>(spec, self),
                }
            }
            LogicalSpec::Compose { .. } => todo!(),
        };
        if test_result {
            log::debug!("Artifact passed correctness check");
        } else {
            log::debug!("Artifact failed correctness check");
        }
        test_result
    }
}

fn test_artifact_correct_inner<Tgt, T>(spec: &Spec<Tgt>, built_artifact: &BuiltArtifact) -> bool
where
    Tgt: Target,
    T: num_traits::Bounded
        + num_traits::WrappingAdd
        + num_traits::ToBytes
        + num_traits::NumAssignRef
        + std::str::FromStr
        + std::fmt::Debug
        + Copy
        + 'static,
    T::Err: std::fmt::Debug,
{
    // Generate arguments.
    let mut concrete_tensors = vec![];
    for input in spec.0.parameters() {
        let value_cnt = input
            .shape()
            .iter()
            .map(|v| usize::try_from(*v).unwrap())
            .product::<_>();
        let unshaped = Array::from_iter(cycle_int_values::<T>().take(value_cnt));
        let shp_usize = input
            .shape()
            .iter()
            .map(|v| usize::try_from(*v).unwrap())
            .collect::<Vec<_>>();
        let arr = unshaped.into_shape(IxDyn(&shp_usize)).unwrap();
        concrete_tensors.push(arr);
    }

    // Compute expected output
    let out_idx = spec.0.output_idx();
    let mut concrete_tensor_views = concrete_tensors
        .iter_mut()
        .map(|v| v.into())
        .collect::<Vec<_>>();
    spec.0.execute(&mut concrete_tensor_views);

    // Split the inputs and output.
    let mut expected_output = None;
    let mut input_tensors = Vec::with_capacity(concrete_tensor_views.len() - 1);
    for (i, v) in concrete_tensor_views.iter().enumerate() {
        if i == out_idx {
            expected_output = Some(v);
        } else {
            input_tensors.push(v.view());
        }
    }
    let expected_output = expected_output.unwrap();

    // Evaluate output
    let lowered_output = built_artifact.run_with_input_data(&input_tensors).unwrap();

    lowered_output == expected_output
}

/// Returns an iterator that yields infinitely all values of a numeric type in ascending order.
fn cycle_int_values<T>() -> impl Iterator<Item = T>
where
    T: num_traits::Bounded + num_traits::WrappingAdd + num_traits::Num + Copy,
{
    // The following can just be `T::min_value()..=T::max_value()` once std::iter::Step is stable.
    let mut v = T::min_value();
    std::iter::from_fn(move || {
        let r = v;
        v = v.wrapping_add(&T::one());
        Some(r)
    })
}
