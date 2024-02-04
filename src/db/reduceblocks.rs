use crate::{grid::linear::BimapInt, ndarray::RleNdArray};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{
    hash::{Hash, Hasher},
    ops::Range,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct ReduceBlock(RleNdArray<Option<(BitF32, SmallVec<[BimapInt; 10]>)>>);

/// Wraps a [f32] to implement [Hash] and [Eq] by comparing bits.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct BitF32(f32);

impl ReduceBlock {
    pub fn partially_filled(
        k: u8,
        shape: &[usize],
        dim_ranges: &[Range<BimapInt>],
        value: &Option<(f32, SmallVec<[BimapInt; 10]>)>,
    ) -> Self {
        assert_ne!(k, 0);
        if k != 1 {
            todo!("k != 1 not implemented");
        }

        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());
        let mut result = ReduceBlock(RleNdArray::new(shape));
        result.fill_region(dim_ranges, value);
        result
    }

    pub fn get(&self, pt: &[usize]) -> Option<(f32, &[BimapInt])> {
        self.0
            .get_with_neighbor(pt)
            .0
            .as_ref()
            .map(|(a, b)| (a.0, &b[..]))
    }

    pub fn fill_region(
        &mut self,
        dim_ranges: &[Range<BimapInt>],
        value: &Option<(f32, SmallVec<[BimapInt; 10]>)>,
    ) {
        let new_value = value.as_ref().map(|(a, b)| (BitF32(*a), b.clone()));
        self.0.fill_region(dim_ranges, &new_value);
    }
}

impl PartialEq for BitF32 {
    fn eq(&self, rhs: &Self) -> bool {
        self.0.to_bits() == rhs.0.to_bits()
    }
}

impl Eq for BitF32 {}

impl Hash for BitF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.0.to_bits());
    }
}
