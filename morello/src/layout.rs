use crate::{
    common::{Contig, DimSize, Dtype, Shape},
    expr::{AffineForm, Atom, Bounds, NonAffineExpr},
    layout,
    opaque_symbol::OpaqueSymbol,
    target::Target,
};
use itertools::Itertools;
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use std::fmt;
use std::{collections::HashSet, fmt::Display, hash::Hash};

pub trait LayoutBuilder {
    fn build(self, shape: &[DimSize]) -> Layout;
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deserialize, Serialize)]
pub struct Layout {
    pub(crate) dims: Vec<(u8, PhysDim)>,
    pub(crate) contig: Contig,
}

// TODO: Remove PartialOrd and Ord
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deserialize, Serialize)]
pub enum PhysDim {
    Dynamic,
    Packed(DimSize),
    /// A physical dimension split into two halves, laid out such that values with odd indices in
    /// that dimension come first, followed by even-indexed values. The parameter `0` is the total
    /// size: both halves.
    OddEven(DimSize),
}

#[cfg(test)]
pub struct LayoutArbRankBounds {
    min_rank: std::num::NonZeroU8,
    max_rank: Option<std::num::NonZeroU8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BufferVar {
    TileIdx(u8, OpaqueSymbol),
    // TODO: *Safely* remove OpaqueSymbol from Pt, if possible
    Pt(u8, OpaqueSymbol),
}

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum LayoutError {
    #[error("Layout does not apply to shape {0:?}")]
    InvalidShape(Shape),
}

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum StridesError {
    #[error("Logical dimension {0} maps to multiple, non-sequential physical dimensions")]
    NonseqPhysicalDims(u8),
    #[error("Layout does not apply to shape {0:?}")]
    InvalidShape(Shape),
}

impl LayoutBuilder for Layout {
    fn build(self, _shape: &[DimSize]) -> Layout {
        // TODO: Check that the layout applies to the shape.
        self
    }
}

/// Implements [LayoutBuilder] for functions which accept the rank of the tensor.
impl<F: Fn(u8) -> Layout> LayoutBuilder for F {
    fn build(self, shape: &[DimSize]) -> Layout {
        self(u8::try_from(shape.len()).unwrap())
    }
}

impl Layout {
    pub fn new(dims: Vec<(u8, PhysDim)>) -> Layout {
        #[cfg(debug_assertions)]
        {
            assert!(!dims.is_empty());

            // Check that every logical dimension in the tensor is mentioned at least once.  Also
            // check that, for each logical dimension, only the first mention, if any, is dynamic.
            let logical_rank = dims.iter().map(|&(d, _)| d).max().unwrap();
            let mut seen = vec![false; usize::from(logical_rank) + 1];
            for (d, fixed_size) in &dims {
                if matches!(fixed_size, PhysDim::Dynamic) && seen[usize::from(*d)] {
                    panic!("Non-first occurrence of logical dimension {d} is PhysDim::Dynamic");
                }
                seen[usize::from(*d)] = true;
            }
            assert!(
                seen.iter().all(|&m| m),
                "Logical dimensions missing from layout: {}",
                seen.iter()
                    .enumerate()
                    .filter(|&(_, m)| !m)
                    .map(|(i, _)| i)
                    .join(", ")
            );
        }

        // TODO: Adapt merge_consecutive_dimensions to make the following less convoluted.
        let contig = dims.len().try_into().unwrap();
        let mut l = Layout { dims, contig };
        debug_assert!(l.is_fully_contiguous());
        l.assert_no_consecutive_dimensions();
        l.assert_no_size_1_packings();
        l.assert_no_odd_deinterleaves();
        l.merge_consecutive_dimensions();
        debug_assert!(l.is_fully_contiguous());
        l
    }

    pub fn buffer_indexing_expr(
        &self,
        expr_id: OpaqueSymbol,
        concrete_shape: &[DimSize],
    ) -> NonAffineExpr<BufferVar> {
        let Layout { dims, contig: _ } = self;

        let tensor_rank = concrete_shape.len();
        debug_assert_eq!(
            tensor_rank,
            usize::from(dims.iter().map(|(d, _)| *d).max().unwrap()) + 1
        );

        let physical_shape = self.expand_physical_shape(concrete_shape).unwrap();
        let mut working_expr = AffineForm::zero();
        let mut dim_remaining_volume = Shape::from(concrete_shape);
        for (&(logical_dim, phys_dim), physical_size) in dims.iter().zip(physical_shape) {
            let logical_dim_us = usize::from(logical_dim);
            let prev_remaining_volume = dim_remaining_volume[logical_dim_us];
            debug_assert!(prev_remaining_volume <= concrete_shape[logical_dim_us]);
            let new_remaining_volume =
                DimSize::new(prev_remaining_volume.get() / physical_size.get()).unwrap();
            let new_remaining_volume_i32 = i32::try_from(new_remaining_volume.get()).unwrap();
            dim_remaining_volume[logical_dim_us] = new_remaining_volume;

            // Construct a "term" for this physical dimension: really, an expression parameterized
            // by a logical dimension.
            let mut term: NonAffineExpr<_> = BufferVar::Pt(logical_dim, expr_id).into();
            if prev_remaining_volume != concrete_shape[logical_dim_us] {
                term %= prev_remaining_volume.get().try_into().unwrap();
            }
            match phys_dim {
                PhysDim::OddEven(deinterleave_strip_size) => {
                    let deinterleave_strip_size =
                        i32::try_from(deinterleave_strip_size.get()).unwrap();
                    debug_assert_eq!(deinterleave_strip_size % 2, 0);
                    let half_size = deinterleave_strip_size / 2;

                    let alternating_term =
                        ((term.clone() % deinterleave_strip_size) % 2) * half_size;
                    let linear_term = (term % deinterleave_strip_size) / 2;
                    term = linear_term + alternating_term;
                    term /= new_remaining_volume_i32;
                }
                _ => {
                    if concrete_shape[logical_dim_us] != physical_size {
                        term /= new_remaining_volume_i32;
                    }
                }
            };

            working_expr *= i32::try_from(physical_size.get()).unwrap();
            working_expr += term;
        }

        debug_assert!(dim_remaining_volume.iter().all(|&d| d.get() == 1));
        working_expr
    }

    pub(crate) fn contiguous_full(&self) -> Contig {
        self.dims.len().try_into().unwrap()
    }

    pub fn contiguous_none(&self) -> Contig {
        0
    }

    pub fn all_contiguous_abs(&self) -> impl Iterator<Item = Contig> + Clone {
        0..=self.contiguous_full()
    }

    pub fn is_valid_contiguous_abs(&self, contig: Contig) -> bool {
        contig <= self.contiguous_full()
    }

    pub(crate) fn contig(&self) -> Contig {
        self.contig
    }

    pub fn is_fully_contiguous(&self) -> bool {
        self.contig == self.contiguous_full()
    }

    // TODO: Remove this fn.
    pub(crate) fn set_contig(&mut self, contig: Contig) {
        debug_assert!(self.is_valid_contiguous_abs(contig));
        self.contig = contig;
    }

    pub fn set_contiguous_full(&mut self) {
        self.contig = self.contiguous_full();
    }

    pub fn set_contiguous_none(&mut self) {
        self.contig = self.contiguous_none();
    }

    pub fn estimate_cache_lines<Tgt: Target>(
        &self,
        shape: &[DimSize],
        dtype: Dtype,
        contig: Contig,
    ) -> u32 {
        let Layout { dims, contig: _ } = self;

        assert!(
            usize::from(contig) <= dims.len(),
            "Invalid contig: {contig} for dims: {dims:?}"
        );

        let first_contig_idx = dims.len() - usize::from(contig);
        let line_size = Tgt::line_size();
        let physical_shape = self.expand_physical_shape(shape).unwrap();

        let mut lines = physical_shape[..first_contig_idx]
            .iter()
            .map(|d| d.get())
            .product::<u32>()
            * u32::from(dtype.size());
        lines *= divrem::DivCeil::div_ceil(
            physical_shape[first_contig_idx..]
                .iter()
                .map(|d| d.get())
                .product::<u32>()
                * u32::from(dtype.size()),
            line_size,
        );
        lines
    }

    pub fn applies_to_shape(&self, shape: &[DimSize]) -> bool {
        self.expand_physical_shape(shape).is_ok()
    }

    pub fn is_row_major(&self) -> bool {
        self.dims
            .iter()
            .enumerate()
            .all(|(i, (d, s))| i == usize::from(*d) && *s == PhysDim::Dynamic)
    }

    /// Returns the step size over each logical dimension.
    ///
    /// For example,
    /// ```
    /// # use morello::layout::col_major;
    /// # use morello::common::Shape;
    /// # use nonzero::nonzero as nz;
    /// let layout = col_major(2);
    /// assert_eq!(
    ///     layout.strides(&[nz!(4u32), nz!(6u32)]),
    ///     Ok(Shape::from(vec![nz!(1u32), nz!(4u32)]))
    /// );
    /// ```
    ///
    /// This function returns a [StridesError::NonseqPhysicalDims] for [Layout]s which map each
    /// logical dimension to contiguous sequences of physical dimensions. Put another way, it is not
    /// defined for layouts which "re-visit" a logical dimension while iterating over physical
    /// dimensions.
    pub fn strides(&self, logical_shape: &[DimSize]) -> Result<Shape, StridesError> {
        let Layout { dims, contig: _ } = self;

        if usize::from(*dims.iter().map(|(d, _)| d).max().unwrap()) + 1 != logical_shape.len() {
            return Err(StridesError::InvalidShape(logical_shape.into()));
        }

        let mut seen = vec![false; logical_shape.len()];
        let mut strides = smallvec![nz!(1u32); logical_shape.len()];
        let mut last_stride = nz!(1u32);
        for (logical_dim, _) in &dims.iter().rev().chunk_by(|(dim, _)| *dim) {
            // We won't visit the chunk's contents. We're just interested in the dimension's
            // physical order.
            let logical_dim_usize = usize::from(logical_dim);
            if seen[logical_dim_usize] {
                return Err(StridesError::NonseqPhysicalDims(logical_dim));
            }
            strides[logical_dim_usize] = last_stride;
            last_stride = last_stride
                .checked_mul(logical_shape[logical_dim_usize])
                .unwrap();
            seen[logical_dim_usize] = true;
        }
        Ok(strides)
    }

    /// Build a [Layout] with physical dimensions corresponding to the given
    /// logical dimensions removed.
    ///
    /// The remaining dimensions will have their logical indices shifted.
    pub fn dim_drop(&self, dropped_dims: &HashSet<u8>) -> Layout {
        if dropped_dims.is_empty() {
            return self.clone();
        }

        let Layout {
            ref dims,
            contig: contiguous_abs,
        } = *self;
        let first_contig_idx = self.dims.len() - usize::from(self.contig);

        let mut new_contig = contiguous_abs
            - u8::try_from(
                dims[first_contig_idx..]
                    .iter()
                    .filter(|(d, _)| dropped_dims.contains(d))
                    .count(),
            )
            .unwrap();

        let mut new_dims = Vec::with_capacity(dims.len().saturating_sub(1));
        for (idx, (logical_dim, phys_dim)) in dims.iter().enumerate() {
            if dropped_dims.contains(logical_dim) {
                continue;
            }
            let new_logical_dim = logical_dim
                - u8::try_from(dropped_dims.iter().filter(|&&d| d < *logical_dim).count()).unwrap();

            // Skip if the last dim is Dynamic, this one is Packed, and logical dims match.
            match (new_dims.last_mut(), phys_dim) {
                (None, _) => {
                    // fall through
                }
                (Some((last_logical_dim, _)), _) if *last_logical_dim != new_logical_dim => {
                    // fall through
                }
                (Some((_, PhysDim::Dynamic)), PhysDim::Packed(_)) => {
                    // This Packed is redundant. Skip it.
                    if idx >= first_contig_idx {
                        new_contig -= 1;
                    }
                    continue;
                }
                (Some((_, PhysDim::Packed(last_pack))), PhysDim::Packed(pack)) => {
                    *last_pack = (last_pack.get() * pack.get()).try_into().unwrap();
                    if idx >= first_contig_idx {
                        new_contig -= 1;
                    }
                    continue;
                }
                (Some((_, PhysDim::OddEven(_))), PhysDim::Packed(_))
                | (Some(_), PhysDim::OddEven(_)) => {
                    // fall through
                }
                (Some((_, PhysDim::Packed(_))), PhysDim::Dynamic)
                | (Some((_, PhysDim::OddEven(_))), PhysDim::Dynamic)
                | (Some((_, PhysDim::Dynamic)), PhysDim::Dynamic) => {
                    unreachable!("invalid Layout")
                }
            }

            new_dims.push((new_logical_dim, *phys_dim));
        }

        let mut new_layout = Layout::new(new_dims);
        new_layout.contig = new_contig;
        new_layout
    }

    pub fn swap_dims(&self, dims: (u8, u8)) -> Layout {
        let mut result = Layout::new(
            self.dims
                .iter()
                .copied()
                .map(|(orig_dim, orig_size)| {
                    if orig_dim == dims.0 {
                        (dims.1, orig_size)
                    } else if orig_dim == dims.1 {
                        (dims.0, orig_size)
                    } else {
                        (orig_dim, orig_size)
                    }
                })
                .collect(),
        );
        result.contig = self.contig;
        result
    }

    pub fn canonicalize(&self, shape: &[DimSize]) -> Result<Layout, LayoutError> {
        // TODO: Can we remove this case without affecting behavior?
        if shape.iter().all(|d| d.get() == 1) {
            Ok(row_major(shape.len().try_into().unwrap()))
        } else {
            self.assert_no_consecutive_dimensions();
            let mut new_layout = self.clone();
            // TODO: Expanding is a waste, but it's necessary to return LayoutError::InvalidShape.
            new_layout.expand_physical_shape(shape)?;
            new_layout.drop_unneeded_packings(shape);
            new_layout.reorder_size_one_dynamic_dimensions(shape)?;
            new_layout.increase_contig_through_ones(shape);
            Ok(new_layout)
        }
    }

    pub fn update_for_tiling(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
    ) -> Result<Layout, LayoutError> {
        // TODO: Can we remove this case without affecting behavior?
        if tile_shape.iter().all(|d| d.get() == 1) {
            Ok(row_major(tile_shape.len().try_into().unwrap()))
        } else {
            let new_contig =
                self.lower_contig_to_first_broken_dimension(parent_shape, tile_shape)?;
            debug_assert!(
                parent_shape != tile_shape || new_contig == self.contig,
                "Contig. shouldn't change when the shape ({:?}) doesn't, but {:?} is now {:?}",
                parent_shape,
                self.contig,
                new_contig
            );
            self.assert_no_consecutive_dimensions();
            let mut new_layout = self.clone();
            new_layout.contig = new_contig;
            new_layout.drop_unneeded_packings(tile_shape);
            new_layout.reorder_size_one_dynamic_dimensions(tile_shape)?;
            new_layout.increase_contig_through_ones(tile_shape);
            Ok(new_layout)
        }
    }

    // TODO: Instead of returning the `Contig`, update self.
    /// Drop Contig to the first broken physical dimension.
    fn lower_contig_to_first_broken_dimension(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
    ) -> Result<Contig, LayoutError> {
        let Layout {
            ref dims,
            contig: source_contig,
        } = *self;
        let parent_physical_shape = self.expand_physical_shape(parent_shape)?;
        let tile_physical_shape = self.expand_physical_shape(tile_shape)?;
        debug_assert_eq!(parent_physical_shape.len(), tile_physical_shape.len());
        let matching_suffix_len = parent_physical_shape
            .iter()
            .rev()
            .zip(tile_physical_shape.iter().rev())
            .find_position(|(p, t)| p != t)
            .map(|(idx, _)| idx);
        Ok(matching_suffix_len
            .map(|d| d + 1)
            .unwrap_or(dims.len())
            .min(source_contig.into())
            .try_into()
            .unwrap())
    }

    /// Asserts that there are no consecutive packed or OddEven dimensions with the same logical
    /// dimension.
    ///
    /// This does nothing on release builds.
    fn assert_no_consecutive_dimensions(&self) {
        #[cfg(debug_assertions)]
        {
            for idx in 1..self.dims.len() {
                if self.dims[idx - 1].0 == self.dims[idx].0
                    && matches!(self.dims[idx - 1].1, PhysDim::Dynamic | PhysDim::Packed(_))
                    && matches!(self.dims[idx].1, PhysDim::Dynamic | PhysDim::Packed(_))
                {
                    panic!(
                        "Consecutive matching dimensions for logical dimension {} in layout: {:?}",
                        self.dims[idx].0, self.dims
                    );
                }
            }
        }
    }

    /// Merge matching, consecutive dimensions.
    fn merge_consecutive_dimensions(&mut self) {
        let first_contig_idx = self.dims.len() - usize::from(self.contig);

        let mut new_contig = self.contig;
        let mut new_dims = Vec::with_capacity(self.dims.len());
        new_dims.push(self.dims[0]);

        for (idx, (dim, phys_dim)) in self.dims.iter().skip(1).enumerate() {
            let (last_dim, last_phys_dim): &mut (u8, PhysDim) = new_dims.last_mut().unwrap();
            if dim != last_dim {
                new_dims.push((*dim, *phys_dim));
                continue;
            }

            match (last_phys_dim, phys_dim) {
                (PhysDim::Packed(l), PhysDim::Packed(n)) => {
                    *l = l.checked_mul(*n).unwrap();
                    if idx >= first_contig_idx {
                        new_contig -= 1;
                    }
                }
                (PhysDim::Dynamic, PhysDim::Packed(_))
                | (PhysDim::Dynamic, PhysDim::OddEven(_))
                | (PhysDim::Packed(_), PhysDim::Dynamic)
                | (PhysDim::Packed(_), PhysDim::OddEven(_))
                | (PhysDim::OddEven(_), PhysDim::Dynamic)
                | (PhysDim::OddEven(_), PhysDim::Packed(_)) => {
                    new_dims.push((*dim, *phys_dim));
                }
                (PhysDim::OddEven(_), PhysDim::OddEven(_)) => todo!(),
                (PhysDim::Dynamic, PhysDim::Dynamic) => {
                    panic!("Repeating non-packed dimensions is undefined: {self:?}")
                }
            }
        }

        self.dims = new_dims;
        self.contig = new_contig;
    }

    /// Increase contig through any all-ones prefix.
    fn increase_contig_through_ones(&mut self, tile_shape: &[DimSize]) {
        let physical_tile_shape = self.expand_physical_shape(tile_shape).unwrap();
        let mut first_contig_idx = physical_tile_shape.len() - usize::from(self.contig);
        while first_contig_idx > 0 {
            first_contig_idx -= 1;
            if physical_tile_shape[first_contig_idx].get() != 1 {
                break;
            }
            self.contig += 1;
        }
    }

    fn drop_unneeded_packings(&mut self, tile_shape: &[DimSize]) {
        let Layout {
            ref mut dims,
            contig,
        } = *self;
        let first_contig_idx = dims.len() - usize::from(contig);

        // Count the number of packings applied to each logical dimension.
        let mut packings = vec![0; dims.len()];
        for (logical_dim, s) in dims.as_slice() {
            if matches!(s, PhysDim::Packed(_)) {
                packings[usize::from(*logical_dim)] += 1;
            }
        }

        // Walk from physically innermost to outermost, clearing packings unique to a logical
        // dimension. `new_contig` becomes the new contig. with previously-counted-as-contiguous
        // dimensions removed whenever they are redundant with a cleared packing.
        //
        // Cleared packings are recorded in `logical_dims_noneified` to be used in the next step.
        // Logical dimensions are mapped to the index of the cleared packing for that dimension.
        let mut logical_dims_noneified = vec![None; dims.len()];
        let mut new_contig = contig;
        for idx in (0..dims.len()).rev() {
            let (logical_dim, s) = dims[idx];
            let logical_dim_usize = usize::from(logical_dim);
            if packings[logical_dim_usize] != 1 {
                continue;
            }
            match s {
                PhysDim::Packed(fixed_size) if tile_shape[logical_dim_usize] == fixed_size => {
                    dims[idx] = (logical_dim, PhysDim::Dynamic);
                    logical_dims_noneified[logical_dim_usize] = Some(idx);
                }
                PhysDim::Dynamic
                    if idx >= first_contig_idx
                        && logical_dims_noneified[logical_dim_usize].is_some() =>
                {
                    // We know this will be 1 since we'll have already visited the packed dimension
                    // with the same size as the logical dimension.
                    new_contig -= 1;
                }
                _ => {}
            }
        }

        // Layouts only include a single dynamic size reference to a logical dimension. If any
        // packings were cleared (changed to dynamic size) in the previous step, then any outer
        // reference to that same dimension is removed. (contig. was already updated to be
        // consistent with this removal by the previous step.)
        let mut i = 0;
        dims.retain(|(logical_dim, _)| {
            let should_retain =
                if let Some(noneified_idx) = logical_dims_noneified[usize::from(*logical_dim)] {
                    i >= noneified_idx
                } else {
                    true
                };
            i += 1;
            should_retain
        });

        self.contig = new_contig;
    }

    /// Canonicalize runs of physical dimension which have size 1 for the given `shape`.
    fn reorder_size_one_dynamic_dimensions(
        &mut self,
        shape: &[DimSize],
    ) -> Result<(), LayoutError> {
        let physical_shape = self.expand_physical_shape(shape)?;
        let Layout { dims, contig: _ } = self;
        let mut start = 0;
        while start < physical_shape.len() {
            let mut end = start;
            while end < physical_shape.len() && physical_shape[end].get() == 1 {
                end += 1;
            }
            let ones_slice = &mut dims[start..end];
            ones_slice.sort_by_key(|e| e.0);
            start = end + 1;
        }
        Ok(())
    }

    pub(crate) fn has_noncanon_size_one_dynamic_dimensions(&self, shape: &[DimSize]) -> bool {
        // This implementation could be much more efficient.
        let physical_shape = self.expand_physical_shape(shape).unwrap();
        let mut start = 0;
        while start < physical_shape.len() {
            let mut end = start;
            while end < physical_shape.len() && physical_shape[end].get() == 1 {
                end += 1;
            }
            let ones_slice = &self.dims[start..end];
            if ones_slice.windows(2).any(|w| w[0].0 > w[1].0) {
                return true;
            }
            start = end + 1;
        }
        false
    }

    // TODO: Return iterator instead?
    fn expand_physical_shape(&self, logical_shape: &[DimSize]) -> Result<Shape, LayoutError> {
        let Layout { dims, contig: _ } = self;
        let mut physical_shape = Shape::with_capacity(dims.len());
        let mut logical_shape_remaining: SmallVec<[_; 5]> =
            logical_shape.iter().map(|x| x.get()).collect();
        for (dim, phys_dim) in dims.iter().rev() {
            let remaining_size = &mut logical_shape_remaining[usize::from(*dim)];
            debug_assert_ne!(
                remaining_size, &0,
                "Logical dimension {dim} with unpacked sized already seen in {dims:?}"
            );
            match phys_dim {
                PhysDim::OddEven(s) | PhysDim::Packed(s) => {
                    if *remaining_size % s.get() != 0 {
                        return Err(LayoutError::InvalidShape(logical_shape.into()));
                    }
                    physical_shape.push(*s);
                    *remaining_size /= s.get();
                }
                PhysDim::Dynamic => {
                    physical_shape.push(DimSize::new(*remaining_size).unwrap());
                    *remaining_size = 0; // zero is a special value for error detection
                }
            }
        }
        if logical_shape_remaining.iter().any(|&d| d > 1) {
            return Err(LayoutError::InvalidShape(logical_shape.into()));
        }
        physical_shape.reverse();
        Ok(physical_shape)
    }

    pub(crate) fn physical_size(
        &self,
        physical_dim: u8,
        logical_shape: &[DimSize],
    ) -> Result<DimSize, LayoutError> {
        let Layout { dims, contig: _ } = self;
        let physical_dim_usize = usize::from(physical_dim);
        let corresponding_logical_dim = dims
            .get(physical_dim_usize)
            .ok_or_else(|| LayoutError::InvalidShape(logical_shape.into()))?
            .0;
        let mut current_remaining_size = logical_shape
            .get(usize::from(corresponding_logical_dim))
            .ok_or_else(|| LayoutError::InvalidShape(logical_shape.into()))?
            .get();
        let mut size_of_target_dim = None;

        // TODO: This could short-circuit! It would just have to avoid checking correctness of
        // remaining dimensions.
        for i in (0..dims.len()).rev() {
            let (ldim, pdim) = dims[i];
            if ldim != corresponding_logical_dim {
                continue;
            }
            assert_ne!(current_remaining_size, 0, "Dynamic already seen");
            match pdim {
                PhysDim::Packed(s) | PhysDim::OddEven(s) => {
                    if current_remaining_size % s.get() != 0 {
                        return Err(LayoutError::InvalidShape(logical_shape.into()));
                    }
                    if i == physical_dim_usize {
                        size_of_target_dim = Some(s);
                    }
                    current_remaining_size /= s.get();
                }
                PhysDim::Dynamic => {
                    let dynamic_dim_size = DimSize::new(current_remaining_size)
                        .ok_or_else(|| LayoutError::InvalidShape(logical_shape.into()))?;
                    if i == physical_dim_usize {
                        size_of_target_dim = Some(dynamic_dim_size);
                    }
                    current_remaining_size = 0; // mark consumption by Dynamic
                }
            }
        }

        // If current_remaining_size is 0, a Dynamic dim consumed remaining volume.  If 1, all
        // Packed/OddEven dims divided perfectly. If > 1, the shape is invalid.
        if current_remaining_size > 1 {
            return Err(LayoutError::InvalidShape(logical_shape.into()));
        }
        Ok(size_of_target_dim.expect("No size for physical dimension"))
    }

    fn assert_no_size_1_packings(&self) {
        #[cfg(debug_assertions)]
        {
            for (_, size) in &self.dims {
                debug_assert_ne!(
                    size,
                    &PhysDim::Packed(nz!(1u32)),
                    "Size-1 packings are disallowed"
                );
            }
        }
    }

    fn assert_no_odd_deinterleaves(&self) {
        #[cfg(debug_assertions)]
        {
            for (_, size) in &self.dims {
                if let PhysDim::OddEven(s) = size {
                    debug_assert_eq!(s.get() % 2, 0);
                }
            }
        }
    }
}

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for Layout {
    type Parameters = LayoutArbRankBounds;
    type Strategy = proptest::strategy::BoxedStrategy<Layout>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::{
            prop_oneof,
            strategy::{Just, Strategy},
        };

        let min_rank = usize::from(args.min_rank.get());
        let max_physical_rank = usize::from(args.max_rank.map(|r| r.into()).unwrap_or(5));
        assert!(min_rank <= max_physical_rank);

        let packed_st = (2..=8u32).prop_map(|s| PhysDim::Packed(s.try_into().unwrap()));
        let interleaved_st = (1..=4u32).prop_map(|s| PhysDim::OddEven((s * 2).try_into().unwrap()));
        let non_dynamic_st = prop_oneof![packed_st, interleaved_st];
        let any_phys_dim_st = prop_oneof![
            3 => Just(PhysDim::Dynamic),
            1 => non_dynamic_st.clone()
        ];

        let required_dims = proptest::collection::vec(any_phys_dim_st, min_rank..=min_rank)
            .prop_map(|v| {
                v.into_iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let i = u8::try_from(i).unwrap();
                        (i, p)
                    })
                    .collect::<Vec<_>>()
            });

        let max_adds = max_physical_rank - min_rank;
        let additional_dims = proptest::collection::vec(
            (0..max_adds).prop_flat_map(move |i| {
                let i = u8::try_from(i).unwrap();
                (Just(i), non_dynamic_st.clone())
            }),
            0..=max_adds,
        );

        (required_dims, additional_dims)
            .prop_flat_map(|(prefix, additional)| {
                let new_layout = Layout::new(prefix.into_iter().chain(additional).collect());
                assert_eq!(
                    new_layout.all_contiguous_abs().collect::<Vec<_>>(),
                    (0..=new_layout.contiguous_full()).collect::<Vec<_>>()
                );
                let contigs_range = 0..=new_layout.contiguous_full();
                (Just(new_layout), contigs_range)
            })
            .prop_map(|(mut layout, contig)| {
                layout.contig = contig;
                layout
            })
            .boxed()
    }
}

impl Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Layout { dims, contig } = self;

        if self.is_row_major() {
            write!(f, "RM")?;
        } else if dims[..]
            == [
                (0, PhysDim::Dynamic),
                (2, PhysDim::Dynamic),
                (3, PhysDim::Dynamic),
                (1, PhysDim::Dynamic),
            ]
        {
            write!(f, "NHWC")?;
        } else if dims.iter().all(|(_, s)| s == &PhysDim::Dynamic) {
            write!(
                f,
                "[{}]",
                dims.iter()
                    .map(|(d, _)| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )?;
        } else {
            write!(
                f,
                "<[{}], [{}]>",
                dims.iter()
                    .map(|(d, _)| d.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                dims.iter()
                    .map(|(_, s)| format!("{s:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }

        if !self.is_fully_contiguous() {
            write!(f, ":c{contig}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
impl Default for LayoutArbRankBounds {
    fn default() -> Self {
        Self {
            min_rank: nz!(1u8),
            max_rank: None,
        }
    }
}

impl Display for BufferVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferVar::TileIdx(dim, _) => write!(f, "t{dim}"),
            BufferVar::Pt(dim, _) => write!(f, "p{dim}"),
        }
    }
}

impl Atom for BufferVar {}
impl Bounds for BufferVar {}

pub fn row_major(rank: u8) -> Layout {
    Layout::new((0..rank).map(|d| (d, PhysDim::Dynamic)).collect())
}

pub fn col_major(rank: u8) -> Layout {
    Layout::new((0..rank).rev().map(|d| (d, PhysDim::Dynamic)).collect())
}

pub(crate) fn batched_col_major(rank: u8) -> Layout {
    let rank_us = usize::from(rank);
    let mut layout = row_major(rank);
    layout.dims.swap(rank_us - 1, rank_us - 2);
    layout
}

pub fn nhwc() -> Layout {
    layout![
        (0, PhysDim::Dynamic),
        (2, PhysDim::Dynamic),
        (3, PhysDim::Dynamic),
        (1, PhysDim::Dynamic)
    ]
}

pub mod macros {
    #[macro_export]
    macro_rules! layout {
        ( $($dim:tt),*$(,)* ) => {
            $crate::layout::Layout::new(
                vec![ $( layout!(@inner $dim) ),* ]
            )
        };
        ( @inner ($dim:expr, PhysDim::OddEven($ds:expr)) ) => {{
            use $crate::layout::PhysDim;
            use $crate::spec::macros::internal::IntoDimSize;
            ($dim, PhysDim::OddEven(($ds).into_dim_size()))
        }};
        ( @inner ($dim:expr, PhysDim::Packed($ds:expr)) ) => {{
            use $crate::layout::PhysDim;
            use $crate::spec::macros::internal::IntoDimSize;
            ($dim, PhysDim::Packed(($ds).into_dim_size()))
        }};
        ( @inner ($dim:expr, PhysDim::Dynamic) ) => {{
            use $crate::layout::PhysDim;
            ($dim, PhysDim::Dynamic)
        }};
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::{DimSize, Shape},
        expr::{AffineForm, Bounds, NonAffine, NonAffineExpr, Substitute, Term},
        opaque_symbol::OpaqueSymbol,
        shape,
    };
    use itertools::Itertools;
    use proptest::{
        arbitrary::{any, any_with},
        prelude::prop,
        prop_assert, prop_assert_eq, prop_assume, proptest,
        strategy::{Just, Strategy},
    };
    use std::{collections::HashSet, iter, num::NonZeroU8};

    #[test]
    fn test_expand_physical_shape_1() {
        let layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4))
        ];
        assert_eq!(
            layout.expand_physical_shape(&shape![64, 64]).unwrap()[..],
            *shape![16, 64, 4],
        );
    }

    #[test]
    fn test_expand_physical_shape_2() {
        let layout = layout![(0, PhysDim::Dynamic), (1, PhysDim::Packed(2))];
        assert_eq!(
            layout.expand_physical_shape(&shape![2, 2]).unwrap()[..],
            *shape![2, 2],
        );
    }

    #[test]
    fn test_expand_physical_shape_3() {
        let layout = layout![(0, PhysDim::Packed(4)), (1, PhysDim::Dynamic)];
        assert!(matches!(
            layout.expand_physical_shape(&shape![1, 64]),
            Err(LayoutError::InvalidShape(_))
        ));
    }

    #[test]
    fn test_expand_physical_shape_4() {
        let layout = layout![(0, PhysDim::Packed(2))];
        let expanded = layout.expand_physical_shape(&shape![6]);
        assert!(
            matches!(expanded, Err(LayoutError::InvalidShape(_))),
            "Expected LayoutError::InvalidShape, but was: {expanded:?}",
        );
    }

    #[test]
    fn test_expand_physical_shape_5() {
        let layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4))
        ];
        assert!(matches!(
            layout.expand_physical_shape(&shape![2, 64]),
            Err(LayoutError::InvalidShape(_)),
        ));
    }

    #[test]
    fn test_expand_physical_shape_6() {
        let layout = layout![(0, PhysDim::Packed(2))];
        assert!(matches!(
            layout.expand_physical_shape(&shape![1]),
            Err(LayoutError::InvalidShape(_)),
        ));
    }

    #[test]
    fn test_expand_physical_shape_7() {
        let layout = layout![(0, PhysDim::Packed(4)), (1, PhysDim::Dynamic)];
        let expanded = layout.expand_physical_shape(&shape![8, 64]);
        assert!(
            matches!(expanded, Err(LayoutError::InvalidShape(_))),
            "Expected LayoutError::InvalidShape, but was: {expanded:?}",
        );
    }

    #[test]
    fn test_physical_size_1() {
        let layout = layout![(0, PhysDim::Packed(2))];
        assert!(matches!(
            layout.physical_size(0, &shape![6]),
            Err(LayoutError::InvalidShape(_)),
        ));
    }

    #[test]
    fn test_col_major_slices_are_non_contiguous() {
        let cm = col_major(2);
        let result_layout = cm
            .update_for_tiling(&shape![128, 128], &shape![8, 1])
            .unwrap();
        assert!(result_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_1() {
        let layout = layout![(0, PhysDim::Dynamic), (1, PhysDim::Dynamic)];
        let new_layout = layout.dim_drop(&iter::once(0u8).collect());
        assert_eq!(new_layout, layout![(0, PhysDim::Dynamic)]);
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_2() {
        let layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4))
        ];
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        assert_eq!(
            new_layout,
            // Would be [(0, PhysDim::Dynamic), (0, PhysDim::Packed(4))], but for merging adjacent
            // dimensions.
            layout![(0, PhysDim::Dynamic)]
        );
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_3() {
        let layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4)),
            (2, PhysDim::Dynamic)
        ];
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        assert_eq!(
            new_layout,
            layout![(0, PhysDim::Dynamic), (1, PhysDim::Dynamic)]
        );
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_4() {
        let mut layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4)),
            (2, PhysDim::Dynamic)
        ];
        layout.contig = 3;
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        let mut expected = layout![(0, PhysDim::Dynamic), (1, PhysDim::Dynamic)];
        expected.contig = 1;
        assert_eq!(new_layout, expected);
        // Merging the Dynamic and Packed physical dimensions for logical dimension 0, which are
        // adjacent after dropping dimension 1, results in a loss of contiguousness. Initially,
        // we knew `(0, PhysDim::Packed(4))` was contiguous but `(0, PhysDim::Dynamic)` was not.
        // After merging, we lose the information about those 4-value strips. The final contig.
        // value becomes 1, corresponding to what was `(2, PhysDim::Dynamic)` and is now
        // `(1, PhysDim::Dynamic)`.
        assert_eq!(new_layout.contig(), 1);
    }

    #[test]
    fn test_dim_drop_5() {
        let mut layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4)),
            (2, PhysDim::Dynamic)
        ];
        layout.contig = 3;
        let new_layout = layout.dim_drop(&iter::once(0u8).collect());
        assert_eq!(
            new_layout,
            layout![(0, PhysDim::Dynamic), (1, PhysDim::Dynamic)]
        );
        assert_eq!(new_layout.contig(), 2);
    }

    #[test]
    fn test_dim_drop_6() {
        let mut layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4)),
            (2, PhysDim::Dynamic)
        ];
        layout.contig = 2;
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        let mut expected = layout![(0, PhysDim::Dynamic), (1, PhysDim::Dynamic)];
        expected.contig = 1;
        assert_eq!(new_layout, expected);
        // As in test_dim_drop_4, we lose some contiguousness information here as a result
        // of merging during dim_drop.
        assert_eq!(new_layout.contig(), 1);
    }

    #[test]
    fn test_dim_drop_7() {
        let layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4)),
            (2, PhysDim::Dynamic)
        ];
        let new_layout = layout.dim_drop(&iter::once(0u8).collect());
        assert_eq!(
            new_layout,
            layout![(0, PhysDim::Dynamic), (1, PhysDim::Dynamic)]
        );
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_8() {
        let layout = layout![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(4)),
            (2, PhysDim::Dynamic)
        ];
        let new_layout = layout.dim_drop(&iter::once(2u8).collect());
        assert_eq!(
            new_layout,
            layout![
                (0, PhysDim::Dynamic),
                (1, PhysDim::Dynamic),
                (0, PhysDim::Packed(4))
            ]
        );
        assert!(new_layout.is_fully_contiguous());
    }

    proptest! {
        #[test]
        fn test_expand_physical_shape_preserves_volume(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let pshp = layout.expand_physical_shape(&shape);
            prop_assume!(pshp.is_ok());
            let pshp = pshp.unwrap();

            let physical_volume  = pshp.iter().map(|d| d.get()).product::<u32>();
            let shape_volume = shape.iter().map(|d| d.get()).product::<u32>();
            prop_assert_eq!(physical_volume, shape_volume);
        }

        #[test]
        fn test_expand_physical_shape_matches_physical_size_for_tensor_layouts(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let physical_rank = layout.dims.len();

            let lhs = layout.expand_physical_shape(&shape);

            let mut rhs = Ok(Shape::with_capacity(physical_rank));
            for i in 0..physical_rank {
                if rhs.is_err() {
                    break;
                }
                let i = u8::try_from(i).unwrap();
                match layout.physical_size(i, &shape) {
                    Ok(rhs_size) => rhs.as_mut().unwrap().push(rhs_size),
                    Err(e) => rhs = Err(e),
                }
            }

            prop_assert_eq!(lhs, rhs);
        }

        #[test]
        fn test_physical_shape_raises_error_on_some_dim_when_expansion_does(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let physical_rank = layout.dims.len();

            let lhs = layout.expand_physical_shape(&shape).err();

            let mut raised_error = None;
            for i in 0..physical_rank {
                if raised_error.is_some() {
                    break;
                }
                let i = u8::try_from(i).unwrap();
                match layout.physical_size(i, &shape) {
                    Ok(_) => {},
                    Err(e) => raised_error = Some(e),
                }
            }

            prop_assert_eq!(lhs, raised_error);
        }

        #[test]
        #[should_panic]
        fn test_physical_size_panics_on_oob_dim(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let physical_shape = layout.expand_physical_shape(&shape).unwrap();
            let rank = u8::try_from(physical_shape.len()).unwrap();
            layout.physical_size(rank, &shape).unwrap();
        }

        #[test]
        fn test_layout_underapproximates_fully_contiguous(
            test_input in test_layout_fully_contiguous_or_not_strategy(),
            first_int in 0..3u32,
        ) {
            let (tensor_shape, layout, tile_shape, tile_offset) = test_input;

            // Lay out sequential integers in a Vec according to the layout's indexing expression.
            let e = OpaqueSymbol::new();
            let iexpr = layout.buffer_indexing_expr(e, &tensor_shape);
            let tensor_volume = tensor_shape.iter().map(|d| d.get()).product::<u32>();
            let tensor_buffer = (first_int..(first_int + tensor_volume)).collect::<Vec<_>>();

            let layout = layout.update_for_tiling(&tensor_shape, &tile_shape).unwrap();

            // Walk the memory locations to check correctness.
            let mut visited = HashSet::new();
            for pt in tile_offset
                .iter()
                .zip(&tile_shape)
                .map(|(&off, &within_pt)| off..off + within_pt.get())
                .multi_cartesian_product()
            {
                let buffer_offset_af: NonAffineExpr<BufferVar> =
                    iexpr.clone().map_vars(&mut |v| match v {
                        BufferVar::TileIdx(_, _) => unimplemented!(),
                        BufferVar::Pt(dim, _) => {
                            AffineForm::constant(pt[usize::from(dim)].try_into().unwrap())
                        }
                    });
                let buffer_offset = buffer_offset_af.as_constant().unwrap();
                let Some(v) = tensor_buffer.get(usize::try_from(buffer_offset).unwrap()) else {
                    panic!(
                        "Buffer offset {} is out of bounds for tensor buffer of length {}. \
                        Offset computed at pt {:?} with index expr. {}",
                        buffer_offset,
                        tensor_buffer.len(),
                        pt,
                        iexpr
                    )
                };
                visited.insert(v);
            }

            let is_fully_contig = visited.len()
                == (1 + visited.iter().copied().max().unwrap() - visited.iter().copied().min().unwrap())
                    .try_into()
                    .unwrap();
            let analysis_result_fully_contig = layout.is_fully_contiguous();
            prop_assert_eq!(
                analysis_result_fully_contig, is_fully_contig,
                "Tile is {}fully contiguous but contig.={:?} (full={:?}) (tensor={:?}, tile={:?}, visited={:?})",
                if is_fully_contig { "" } else { "not " },
                layout.contig(),
                layout.contiguous_full(),
                tensor_shape,
                tile_shape,
                visited
            );
        }

        #[test]
        fn test_canonicalize_matches_update_for_tiling(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            match (layout.canonicalize(&shape), layout.update_for_tiling(&shape, &shape)) {
                (Ok(l0), Ok(l1)) if l0 == l1 => {}
                (Err(e0), Err(e1)) if e0 == e1 => {}
                (a, b) => {
                    prop_assert!(false, "{:?} != {:?}", a, b);
                }
            }
        }

        #[test]
        fn test_dim_drop_returns_valid_contig(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let rank = u8::try_from(shape.len()).unwrap();
            for dims_to_drop in (0..rank).powerset() {
                // TODO: Remove this check once we enable support for empty (rank=0) layouts
                if dims_to_drop.len() == shape.len() {
                    continue;
                }
                let new_layout = layout.dim_drop(&dims_to_drop.into_iter().collect());
                prop_assert!(new_layout.is_valid_contiguous_abs(new_layout.contig()));
            }
        }
    }

    fn test_layout_fully_contiguous_or_not_strategy(
    ) -> impl Strategy<Value = (Vec<DimSize>, Layout, Vec<DimSize>, Vec<u32>)> {
        let physical_dim_size_st =
            proptest::collection::vec(1..=3u32, 1..=3).prop_map(|mut mults| {
                if mults[0] == 1 {
                    mults[0] = 2;
                }
                for idx in 1..mults.len() {
                    mults[idx] *= mults[idx - 1];
                }
                mults
            });
        prop::collection::vec((any::<bool>(), physical_dim_size_st), 1..=3)
            .prop_flat_map(|within_dim| {
                let (dim_order, physical_tile_shape): (Vec<_>, Vec<_>) = within_dim
                    .into_iter()
                    .enumerate()
                    .flat_map(|(logical_dim, (prefix_none, per_dim_shape))| {
                        let mut prefix = None;
                        if prefix_none {
                            prefix = Some((u8::try_from(logical_dim).unwrap(), PhysDim::Dynamic));
                        };
                        prefix
                            .into_iter()
                            .chain(per_dim_shape.into_iter().map(move |d| {
                                // TODO: Should also generate OddEven
                                (
                                    u8::try_from(logical_dim).unwrap(),
                                    PhysDim::Packed(d.try_into().unwrap()),
                                )
                            }))
                    })
                    .unzip();
                let rank = usize::from(*dim_order.iter().max().unwrap()) + 1;

                let mut tensor_shape = vec![1; rank];
                for (&dim, &d) in dim_order.iter().zip(&physical_tile_shape) {
                    match d {
                        PhysDim::OddEven(_) => todo!(),
                        PhysDim::Packed(fixed) => {
                            tensor_shape[usize::from(dim)] *= fixed.get();
                        }
                        PhysDim::Dynamic => {
                            // TODO: Multiply by something.
                        }
                    }
                }

                let tile_shape = tensor_shape.iter().map(|&d| 1..=d).collect::<Vec<_>>();
                (
                    Just(tensor_shape),
                    Just(dim_order),
                    Just(physical_tile_shape),
                    tile_shape,
                )
            })
            .prop_filter(
                // TODO: This filter should not be necessary. Instead, avoid this by construction.
                "Layout had consecutive packed dimensions",
                |(_, dim_order, physical_tile_shape, _)| {
                    for idx in 1..dim_order.len() {
                        if dim_order[idx - 1] == dim_order[idx]
                            && matches!(
                                physical_tile_shape[idx - 1],
                                PhysDim::Dynamic | PhysDim::Packed(_)
                            )
                            && matches!(physical_tile_shape[idx], PhysDim::Packed(_))
                        {
                            return false;
                        }
                    }
                    true
                },
            )
            .prop_flat_map(
                |(tensor_shape, dim_order, physical_tile_shape, tile_shape)| {
                    let new_layout =
                        Layout::new(dim_order.into_iter().zip(physical_tile_shape).collect());
                    let tile_offset = tensor_shape
                        .iter()
                        .zip(&tile_shape)
                        .map(|(&o, &i)| 0..=(o - i))
                        .collect::<Vec<_>>();
                    (
                        Just(tensor_shape),
                        Just(new_layout),
                        Just(tile_shape),
                        tile_offset,
                    )
                },
            )
            .prop_map(|(tensor_shape, layout, tile_shape, tile_offset)| {
                (
                    tensor_shape
                        .into_iter()
                        .map(|x| DimSize::new(x).unwrap())
                        .collect::<Vec<_>>(),
                    layout,
                    tile_shape
                        .into_iter()
                        .map(|x| DimSize::new(x).unwrap())
                        .collect::<Vec<_>>(),
                    tile_offset,
                )
            })
            .prop_filter(
                "Layout did not apply to shape",
                |(tensor_shape, layout, tile_shape, _)| {
                    layout.update_for_tiling(tensor_shape, tile_shape).is_ok()
                },
            )
    }

    #[test]
    fn test_row_major_indexing_expression() {
        let rm = row_major(2);
        let expr_id = OpaqueSymbol::new();
        let iexpr = rm.buffer_indexing_expr(expr_id, &shape![16, 4]);
        let expected = NonAffineExpr::constant(0)
            + Term(4, NonAffine::Leaf(BufferVar::Pt(0, expr_id)))
            + Term(1, NonAffine::Leaf(BufferVar::Pt(1, expr_id)));
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_interleaved_indexing_expression_1() {
        let layout = Layout::new(vec![(0, PhysDim::OddEven(DimSize::new(8).unwrap()))]);
        let expr_id = OpaqueSymbol::new();
        let iexpr = layout.buffer_indexing_expr(expr_id, &shape![8]);

        let pt = NonAffineExpr::from(BufferVar::Pt(0, expr_id));
        let expected = (pt.clone() % 8) / 2 + (pt % 2) * 4i32;
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_interleaved_indexing_expression_2() {
        let layout = Layout::new(vec![
            (0, PhysDim::Dynamic),
            (0, PhysDim::OddEven(DimSize::new(8).unwrap())),
        ]);
        let expr_id = OpaqueSymbol::new();
        let iexpr = layout.buffer_indexing_expr(expr_id, &shape![16]);

        let pt = NonAffineExpr::from(BufferVar::Pt(0, expr_id));
        let expected = (pt.clone() / 8) * 8 + (pt.clone() % 8) / 2 + (pt % 2) * 4;
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_interleaved_indexing_expression_3() {
        let layout = Layout::new(vec![
            (0, PhysDim::Dynamic),
            (1, PhysDim::OddEven(DimSize::new(8).unwrap())),
        ]);
        let expr_id = OpaqueSymbol::new();
        let iexpr = layout.buffer_indexing_expr(expr_id, &shape![2, 8]);

        let pt0 = NonAffineExpr::from(BufferVar::Pt(0, expr_id));
        let pt1 = NonAffineExpr::from(BufferVar::Pt(1, expr_id));
        let expected = pt0.clone() * 8 + (pt1.clone() % 8) / 2 + (pt1 % 2) * 4;
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_strides_ok_simple() {
        let layout = row_major(2);
        assert_eq!(
            layout.strides(&[nz!(4u32), nz!(6u32)]),
            Ok(smallvec![nz!(6u32), nz!(1u32)])
        );
    }

    #[test]
    fn test_strides_ok_3d() {
        let layout = Layout::new(vec![
            (2, PhysDim::Dynamic),
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
        ]);
        assert_eq!(
            layout.strides(&[nz!(2u32), nz!(4u32), nz!(6u32)]),
            Ok(smallvec![nz!(4u32), nz!(1u32), nz!(8u32)])
        );
    }

    #[test]
    fn test_strides_err_revisiting() {
        let layout = Layout::new(vec![
            (1, PhysDim::Dynamic),
            (0, PhysDim::Dynamic),
            (1, PhysDim::Packed(nz!(2u32))),
        ]);
        assert_eq!(
            layout.strides(&[nz!(4u32), nz!(6u32)]),
            Err(StridesError::NonseqPhysicalDims(1))
        );
    }

    fn arb_shape_and_same_rank_layout() -> impl Strategy<Value = (Shape, Layout)> {
        proptest::collection::vec(1..=16u32, 1..=3).prop_flat_map(|shape| {
            let shape = shape
                .into_iter()
                .map(|x| DimSize::new(x).unwrap())
                .collect::<Vec<_>>();
            let shape_nz = NonZeroU8::try_from(u8::try_from(shape.len()).unwrap()).unwrap();
            let bounds = LayoutArbRankBounds {
                min_rank: shape_nz,
                max_rank: Some(shape_nz),
            };
            let all_layouts = any_with::<Layout>(bounds);
            (Just(Shape::from(shape)), all_layouts)
        })
    }
}
