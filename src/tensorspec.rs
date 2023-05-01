use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::layout::Layout;
use crate::target::{MemoryLevel, Target};

#[derive(Clone, PartialEq, Eq, Debug, Hash, Deserialize, Serialize)]
pub struct TensorSpec<Tgt: Target> {
    dim_sizes: Shape, // TODO: Rename to shape
    dtype: Dtype,
    contiguous_abs: Contig,
    aligned: bool,
    level: Tgt::Level,
    layout: Layout,
    vector_shape: Option<Shape>,
}

impl<Tgt: Target> TensorSpec<Tgt> {
    pub fn new(
        dim_sizes: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_shape: Option<Shape>,
    ) -> Self {
        if dim_sizes.is_empty() || dim_sizes.iter().any(|&d| d < 1) {
            panic!("Invalid shape: {:?}", dim_sizes);
        }

        if dim_sizes.iter().all(|&d| d == 1) && !layout.is_row_major() {
            panic!("If all dimensions are 1, layout must be row-major")
        }

        if !layout.applies_to_shape(&dim_sizes) {
            panic!(
                "Layout {:?} does not apply to shape {:?}",
                layout, dim_sizes
            );
        }

        if vector_shape.is_some() != level.vector_rf() {
            panic!(
                "vector_shape must be specified if and only if the bank ({:?}) is a vector register file", level
            )
        }

        if let Some(vs) = &vector_shape {
            if vs.len() != dim_sizes.len() {
                panic!("vector_shape must have same rank as dim_sizes")
            }
        }

        if !layout.applies_to_shape(&dim_sizes) {
            panic!(
                "Layout {:?} does not apply to shape {:?} with dtype {:?}",
                layout, dim_sizes, dtype
            )
        }

        TensorSpec {
            dim_sizes,
            dtype,
            contiguous_abs,
            aligned,
            level,
            layout,
            vector_shape,
        }
    }

    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }

    pub fn set_layout(&mut self, new_layout: Layout) {
        self.layout = new_layout;
    }

    pub fn is_contiguous(&self) -> bool {
        self.contiguous_abs == self.layout.contiguous_full()
    }

    /// Returns true if this TensorSpec can be tiled to the given shape.
    pub fn is_valid_tile_shape(&self, shape: &[DimSize]) -> bool {
        if shape.len() != self.dim_sizes.len() {
            return false;
        }

        if !shape.iter().zip(self.dim_sizes.iter()).all(|(i, o)| i <= o) {
            return false;
        }

        let all_ones = shape.iter().all(|d| *d == 1);
        if !all_ones && !self.layout.applies_to_shape(shape) {
            return false;
        }

        true
    }

    pub fn bytes_used(&self) -> u32 {
        u32::from(self.dtype.size()) * self.dim_sizes.iter().product::<u32>()
    }

    pub fn dim_sizes(&self) -> &Shape {
        &self.dim_sizes
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn contiguous_abs(&self) -> Contig {
        self.contiguous_abs
    }

    pub fn aligned(&self) -> bool {
        self.aligned
    }

    pub fn level(&self) -> <Tgt as Target>::Level {
        self.level
    }

    pub fn vector_shape(&self) -> Option<&Shape> {
        self.vector_shape.as_ref()
    }

    pub fn set_level(&mut self, level: Tgt::Level, vector_shape: Option<Shape>) {
        assert_eq!(
            level.vector_rf(),
            vector_shape.is_some(),
            "Cannot set level to {:?} with vector shape {:?}",
            level,
            vector_shape
        );
        self.level = level;
        self.vector_shape = vector_shape;
    }

    pub fn set_dim_sizes(&mut self, dim_sizes: Shape) {
        self.dim_sizes = dim_sizes;
    }

    pub fn set_aligned(&mut self, aligned: bool) {
        self.aligned = aligned;
    }
}

impl<Tgt: Target> Display for TensorSpec<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut layout_epi = String::new();
        let mut bank_epi = String::new();
        let mut c_epi = String::new();
        let mut a_epi = String::new();
        let mut v_epi = String::new();

        if !self.layout.is_row_major() {
            layout_epi = format!(", {}", self.layout);
        }

        if Tgt::default_level() != self.level {
            bank_epi = format!(", {}", self.level);
        }

        if self.contiguous_abs != self.layout.contiguous_full() {
            c_epi = format!(", c{}", self.contiguous_abs);
        }

        if !self.aligned {
            a_epi = String::from(", ua");
        }

        if let Some(shape) = &self.vector_shape {
            v_epi = format!(
                ", {}",
                shape
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("×")
            );
        }

        let dims_part = self
            .dim_sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("×");

        write!(
            f,
            "({}, {}{}{}{}{}{})",
            dims_part, self.dtype, bank_epi, layout_epi, c_epi, a_epi, v_epi
        )
    }
}
