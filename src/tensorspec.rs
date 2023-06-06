use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::layout::Layout;
use crate::target::{MemoryLevel, Target};

#[derive(Clone, PartialEq, Eq, Debug, Hash, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct TensorSpec<Tgt: Target> {
    dim_sizes: Shape, // TODO: Rename to shape
    dtype: Dtype,
    aux: TensorSpecAux<Tgt>,
}

// TODO: This probably shouldn't be public.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TensorSpecAux<Tgt: Target> {
    pub contig: Contig,
    pub aligned: bool,
    pub level: Tgt::Level,
    pub layout: Layout,
    pub vector_shape: Option<Shape>,
}

impl<Tgt: Target> TensorSpec<Tgt> {
    pub fn new_canon(
        dim_sizes: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_shape: Option<Shape>,
    ) -> Self {
        let mut r = Self::new_noncanon(
            dim_sizes,
            dtype,
            contiguous_abs,
            aligned,
            level,
            layout,
            vector_shape,
        );
        r.canonicalize();
        r
    }

    pub fn new_noncanon(
        dim_sizes: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_shape: Option<Shape>,
    ) -> Self {
        Self::new_noncanon_with_aux(
            dim_sizes,
            dtype,
            TensorSpecAux {
                contig: contiguous_abs,
                aligned,
                level,
                layout,
                vector_shape,
            },
        )
    }

    pub fn new_noncanon_with_aux(dim_sizes: Shape, dtype: Dtype, aux: TensorSpecAux<Tgt>) -> Self {
        if dim_sizes.is_empty() || dim_sizes.iter().any(|&d| d < 1) {
            panic!("Invalid shape: {:?}", dim_sizes);
        }

        if !aux.layout.applies_to_shape(&dim_sizes) {
            panic!(
                "Layout {:?} does not apply to shape {:?}",
                aux.layout, dim_sizes
            );
        }

        if aux.vector_shape.is_some() != aux.level.vector_rf() {
            panic!(
                "vector_shape must be specified if and only if the bank ({:?}) is a vector register file", aux.level
            )
        }

        if let Some(vs) = &aux.vector_shape {
            if vs.len() != dim_sizes.len() {
                panic!(
                    "vector_shape must have same rank as dim_sizes, but vector_shape was {:?} and dim_sizes was {:?}",
                    vs, dim_sizes
                );
            }
        }

        TensorSpec {
            dim_sizes,
            dtype,
            aux,
        }
    }

    pub fn layout(&self) -> Layout {
        self.aux.layout.clone()
    }

    pub fn set_layout(&mut self, new_layout: Layout) {
        self.aux.layout = new_layout;
    }

    pub fn is_contiguous(&self) -> bool {
        self.aux.contig == self.aux.layout.contiguous_full()
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
        if !all_ones && !self.aux.layout.applies_to_shape(shape) {
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
        self.aux.contig
    }

    pub fn aligned(&self) -> bool {
        self.aux.aligned
    }

    pub fn level(&self) -> <Tgt as Target>::Level {
        self.aux.level
    }

    pub fn vector_shape(&self) -> Option<&Shape> {
        self.aux.vector_shape.as_ref()
    }

    pub fn set_level(&mut self, level: Tgt::Level, vector_shape: Option<Shape>) {
        assert_eq!(
            level.vector_rf(),
            vector_shape.is_some(),
            "Cannot set level to {:?} with vector shape {:?}",
            level,
            vector_shape
        );
        self.aux.level = level;
        self.aux.vector_shape = vector_shape;
    }

    /// Returns a new TensorSpec with the given shape and alignment.
    ///
    /// The result's layout and contiguousness abstraction will have been
    /// canoncialized for the given shape.
    pub fn shrink(&mut self, dim_sizes: &Shape, aligned: bool) {
        self.aux.contig =
            self.layout()
                .tile_contiguity(dim_sizes, &self.dim_sizes, self.aux.contig);
        self.dim_sizes = dim_sizes.clone();
        self.aux.layout = self.aux.layout.canonicalize_for_shape(&self.dim_sizes);
        self.aux.aligned = aligned;
    }

    pub fn canonicalize(&mut self) {
        // Odd implementation, but concise! `shrink` will canonicalize, so we
        // pass the same shape and alignment.
        self.shrink(&self.dim_sizes.clone(), self.aux.aligned);
    }

    // TODO: Shouldn't need this method. Should be implicit in Spec validity.
    pub fn can_move_to(&self, dest_layout: &Layout, dest_level: &Tgt::Level) -> bool {
        if &self.layout() != dest_layout && !dest_level.is_addressed() {
            return false;
        }
        if dest_level.vector_bytes() > 0 {
            let vol: DimSize = self.dim_sizes().iter().product();
            if (vol * DimSize::from(self.dtype.size())) % dest_level.vector_bytes() != 0 {
                return false;
            }
        }
        true
    }
}

impl<Tgt: Target> Display for TensorSpec<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut layout_epi = String::new();
        let mut bank_epi = String::new();
        let mut c_epi = String::new();
        let mut a_epi = String::new();
        let mut v_epi = String::new();

        if !self.aux.layout.is_row_major() {
            layout_epi = format!(", {}", self.aux.layout);
        }

        if Tgt::default_level() != self.aux.level {
            bank_epi = format!(", {}", self.aux.level);
        }

        if self.aux.contig != self.aux.layout.contiguous_full() {
            c_epi = format!(", c{}", self.aux.contig);
        }

        if !self.aux.aligned {
            a_epi = String::from(", ua");
        }

        if let Some(shape) = &self.aux.vector_shape {
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
