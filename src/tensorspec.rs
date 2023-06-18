use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Display;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::layout::{row_major, Layout};
use crate::target::{MemoryLevel, Target};
use crate::utils::join_into_string;

#[derive(Clone, PartialEq, Eq, Debug, Hash, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct TensorSpec<Tgt: Target> {
    pub dim_sizes: Shape, // TODO: Rename to shape
    pub dtype: Dtype,
    pub aux: TensorSpecAux<Tgt>,
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

    pub fn bytes_used(&self) -> u64 {
        u64::from(self.dtype.size())
            * self
                .dim_sizes
                .iter()
                .copied()
                .map(u64::from)
                .product::<u64>()
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
    pub fn shrink(&mut self, dim_sizes: &[DimSize], aligned: bool) {
        // This implementation is similar to `canonicalize`, but the tile
        // contiguousness is computed from both old and new shapes.
        self.aux.contig =
            self.layout()
                .tile_contiguity(dim_sizes, &self.dim_sizes, self.aux.contig);
        self.dim_sizes = Shape::from(dim_sizes);
        self.aux.layout = self.aux.layout.canonicalize_for_shape(&self.dim_sizes);
        self.aux.aligned = aligned;
    }

    pub fn canonicalize(&mut self) {
        self.aux.canonicalize(&self.dim_sizes, self.aux.aligned);
    }

    /// Returns a TensorSpec with given size-one dimensions dropped.
    ///
    /// The given dimension indices must be sorted in ascending order.
    ///
    /// The result will be canoncialized. If any given dimension index is not
    /// size one, this method panics.
    pub fn squeeze_dims(&self, dropped_dims: &[u8]) -> TensorSpec<Tgt> {
        // Make sure dropped_dims is sorted.
        debug_assert!(dropped_dims.windows(2).all(|w| w[0] < w[1]));

        let mut new_dim_sizes = self.dim_sizes().clone();
        for &dim in dropped_dims.iter().rev() {
            assert_eq!(
                new_dim_sizes[usize::from(dim)],
                1,
                "Cannot drop non-degenerate dimension {} of shape {:?}",
                dim,
                new_dim_sizes
            );
            new_dim_sizes.remove(dim.into());
        }

        let mut new_vector_shape = None;
        if let Some(vector_shape) = self.vector_shape() {
            new_vector_shape = Some(vector_shape.to_vec());
            for &dim in dropped_dims.iter().rev() {
                assert_eq!(
                    vector_shape[usize::from(dim)],
                    1,
                    "Cannot drop dimension {} of vector shape {:?}",
                    dim,
                    vector_shape
                );
                new_vector_shape.as_mut().unwrap().remove(dim.into());
            }
        }

        let (new_layout, new_contig) = if new_dim_sizes.iter().all(|&d| d == 1) {
            let new_layout = row_major(new_dim_sizes.len().try_into().unwrap());
            (new_layout.clone(), new_layout.contiguous_full())
        } else {
            let dropped_dims_set = dropped_dims.iter().copied().collect::<HashSet<_>>();
            self.layout()
                .dim_drop(&dropped_dims_set, self.contiguous_abs())
        };

        TensorSpec::new_canon(
            new_dim_sizes,
            self.dtype(),
            new_contig,
            self.aligned(),
            self.level(),
            new_layout,
            new_vector_shape.map(|v| v.into()),
        )
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
        let dims_part = self
            .dim_sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let aux_part = tensorspec_aux_str(&self.aux);

        write!(f, "({}, {}", dims_part, self.dtype)?;
        if aux_part.is_empty() {
            write!(f, ")")
        } else {
            write!(f, ", {})", aux_part)
        }
    }
}

impl<Tgt: Target> TensorSpecAux<Tgt> {
    pub fn canonicalize(&mut self, dim_sizes: &Shape, aligned: bool) {
        self.contig = self
            .layout
            .tile_contiguity(dim_sizes, dim_sizes, self.contig);
        self.layout = self.layout.canonicalize_for_shape(dim_sizes);
        self.aligned = aligned;
    }
}

impl<Tgt: Target> Display for TensorSpecAux<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aux_part = tensorspec_aux_str(self);
        if aux_part.is_empty() {
            write!(f, "(_)")
        } else {
            write!(f, "({})", aux_part)
        }
    }
}

fn tensorspec_aux_str<Tgt: Target>(aux: &TensorSpecAux<Tgt>) -> String {
    let mut parts = Vec::with_capacity(5);

    if aux.level != Tgt::default_level() {
        parts.push(aux.level.to_string());
    }

    if !aux.layout.is_row_major() {
        parts.push(aux.layout.to_string());
    }

    if aux.contig != aux.layout.contiguous_full() {
        parts.push(format!("c{}", aux.contig));
    }

    if !aux.aligned {
        parts.push(String::from("ua"));
    }

    if let Some(shape) = &aux.vector_shape {
        parts.push(
            shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("×"),
        );
    }

    join_into_string(&parts, ", ")
}
