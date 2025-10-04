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

/// Maps a tensor's value coordinates to offsets in a buffer, and computes the contiguousness of a
/// tensor in memory.
///
/// Layouts can be seen as functions from a concrete tensor shape and a point in its logical space
/// to a memory offset. The offsets are only defined with respect to a particular concrete shape.
/// As a result, a single [Layout] abstracts over both a tensor and its tilings: only the shape
/// (and potentially the contiguousness) change.
///
/// Layouts are not defined for all shapes. They may only be defined for shapes which are size-one
/// in some dimensions or require some dimensions to be multiples of some size (e.g., evenly
/// divisible by some packing factor).
///
/// Layouts have canonical forms with respect to concrete shapes. The canonical form is logically
/// equivalent for that shape, but not necessarily other shapes.
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
#[derive(Default)]
pub struct LayoutArbRankBounds {
    dims: Option<Vec<u8>>,
}

#[cfg(test)]
impl LayoutArbRankBounds {
    pub fn for_shape(shape: &[DimSize]) -> LayoutArbRankBounds {
        LayoutArbRankBounds {
            dims: Some(
                shape
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.get() != 1)
                    .map(|(logical_dim, _)| u8::try_from(logical_dim).unwrap())
                    .collect(),
            ),
        }
    }
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
impl<F: Fn(&[DimSize]) -> Layout> LayoutBuilder for F {
    fn build(self, shape: &[DimSize]) -> Layout {
        self(shape)
    }
}

impl Layout {
    pub fn new(dims: Vec<(u8, PhysDim)>) -> Layout {
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

    pub fn empty() -> Layout {
        Layout {
            dims: vec![],
            contig: 0,
        }
    }

    pub fn buffer_indexing_expr(
        &self,
        expr_id: OpaqueSymbol,
        concrete_shape: &[DimSize],
    ) -> NonAffineExpr<BufferVar> {
        let Layout { dims, contig: _ } = self;

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

    #[inline]
    pub fn contiguous_none(&self) -> Contig {
        0
    }

    pub fn all_contiguous_abs(&self) -> impl Iterator<Item = Contig> + Clone {
        0..=self.contiguous_full()
    }

    #[inline]
    pub fn is_valid_contiguous_abs(&self, contig: Contig) -> bool {
        contig <= self.contiguous_full()
    }

    #[inline]
    pub(crate) fn contig(&self) -> Contig {
        self.contig
    }

    #[inline]
    pub fn is_fully_contiguous(&self) -> bool {
        self.contig == self.contiguous_full()
    }

    // TODO: Remove this fn.
    pub(crate) fn set_contig(&mut self, contig: Contig) {
        debug_assert!(self.is_valid_contiguous_abs(contig));
        self.contig = contig;
    }

    #[inline]
    pub fn set_contiguous_full(&mut self) {
        self.contig = self.contiguous_full();
    }

    #[inline]
    pub fn set_contiguous_none(&mut self) {
        self.contig = self.contiguous_none();
    }

    pub fn estimate_cache_lines<Tgt: Target>(&self, shape: &[DimSize], dtype: Dtype) -> u32 {
        let Layout { ref dims, contig } = *self;

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
        self.dims.iter().all(|d| matches!(d, (_, PhysDim::Dynamic)))
            && self.dims.iter().map(|(d, _)| *d).is_sorted()
    }

    pub fn is_col_major(&self) -> bool {
        self.dims.iter().all(|d| matches!(d, (_, PhysDim::Dynamic)))
            && self.dims.iter().tuple_windows().all(|(a, b)| b.0 < a.0)
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

    /// Remove all physical dimensions whose logical dimension is size-one in the given shape.
    /// Returns the new contiguousness value, subtracting all dropped physical dims from the
    /// contiguous suffix.
    fn prune_size_one_logical_dims_with_contig(
        &mut self,
        shape: &[DimSize],
        contig: Contig,
    ) -> Contig {
        let first_contig_idx = self.dims.len() - usize::from(contig);
        let mut new_contig = contig;
        let mut idx = 0;
        self.dims.retain(|(ldim, _)| {
            let keep = shape[*ldim as usize].get() != 1;
            if !keep && idx >= first_contig_idx {
                new_contig -= 1;
            }
            idx += 1;
            keep
        });
        new_contig
    }

    pub fn canonicalize(&self, shape: &[DimSize]) -> Result<Layout, LayoutError> {
        // `canonicalize` logic is equivalent to updating for tiling when the shape doesn't change.
        // We just make up a contig and ignore the returned contig.
        self.update_for_tiling(shape, shape)
    }

    pub fn update_for_tiling(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
    ) -> Result<Layout, LayoutError> {
        // If all logical dims are size 1, force row-major layout and contiguous_full. This bypasses
        // potential failures to, for instance, tile Packed physical dimensions.
        // TODO: This should be a natural result of being able to tile within Packed dims.
        if tile_shape.iter().all(|d| d.get() == 1) {
            Ok(row_major(tile_shape))
        } else {
            let mut new_layout = self.clone();
            new_layout.contig =
                new_layout.lower_contig_to_first_broken_dimension(parent_shape, tile_shape)?;
            new_layout.contig =
                new_layout.prune_size_one_logical_dims_with_contig(tile_shape, new_layout.contig);
            new_layout.merge_consecutive_dimensions();
            new_layout.drop_unneeded_packings(tile_shape);
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
        if self.dims.is_empty() {
            return;
        }

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
                (PhysDim::Dynamic, PhysDim::Packed(_)) => {
                    if idx >= first_contig_idx {
                        new_contig -= 1;
                    }
                }
                (PhysDim::Dynamic, PhysDim::OddEven(_))
                | (PhysDim::Packed(_), PhysDim::OddEven(_))
                | (PhysDim::OddEven(_), PhysDim::Dynamic)
                | (PhysDim::OddEven(_), PhysDim::Packed(_)) => {
                    new_dims.push((*dim, *phys_dim));
                }
                (PhysDim::OddEven(_), PhysDim::OddEven(_)) => todo!(),
                (PhysDim::Packed(_), PhysDim::Dynamic) => {
                    unreachable!("Dynamic followed Packed same logical dimension");
                }
                (PhysDim::Dynamic, PhysDim::Dynamic) => {
                    unreachable!("Repeating non-packed dimensions");
                }
            }
        }

        self.dims = new_dims;
        self.contig = new_contig;
    }

    /// Converts Packed dims that exactly match the tile shape back to dynamic
    /// dimensions and removes the now-redundant outer references to those logical
    /// dimensions.
    fn drop_unneeded_packings(&mut self, tile_shape: &[DimSize]) {
        let Layout {
            ref mut dims,
            contig,
        } = *self;
        let first_contig_idx = dims.len() - usize::from(contig);

        // Precompute whether there is any OUTER OddEven for the same logical dim at each index.
        // This is used later to avoid converting Packed dimensions to Dynamic if there is an
        // outer OddEven, even if the volume is covered.
        let outer_oe_exists = {
            let mut outer_oddeven_for_ld: SmallVec<[bool; 5]> = smallvec![false; tile_shape.len()];
            let mut buf: Vec<bool> = Vec::with_capacity(dims.len());
            for (ld, pd) in dims.iter().copied() {
                let ld_us = usize::from(ld);
                buf.push(outer_oddeven_for_ld[ld_us]);
                if matches!(pd, PhysDim::OddEven(_)) {
                    outer_oddeven_for_ld[ld_us] = true;
                }
            }
            buf
        };

        let mut remaining: SmallVec<[_; 5]> = tile_shape.iter().map(|d| d.get()).collect();
        let mut logical_dims_noneified: SmallVec<[_; 5]> = smallvec![None; tile_shape.len()];
        for idx in (0..dims.len()).rev() {
            let (logical_dim, s) = dims[idx];
            let logical_dim_us = usize::from(logical_dim);
            match s {
                PhysDim::Packed(fixed_size) => {
                    let rem = &mut remaining[logical_dim_us];
                    if rem.is_multiple_of(fixed_size.get()) {
                        if !outer_oe_exists[idx] && *rem == fixed_size.get() {
                            dims[idx] = (logical_dim, PhysDim::Dynamic);
                            if logical_dims_noneified[logical_dim_us].is_none() {
                                logical_dims_noneified[logical_dim_us] = Some(idx);
                            }
                        }
                        *rem /= fixed_size.get();
                    } else if *rem != 0 && fixed_size.get() % *rem == 0 {
                        if !outer_oe_exists[idx] {
                            dims[idx] = (logical_dim, PhysDim::Dynamic);
                            if logical_dims_noneified[logical_dim_us].is_none() {
                                logical_dims_noneified[logical_dim_us] = Some(idx);
                            }
                        }
                        *rem = 1;
                    }
                }
                PhysDim::OddEven(oe_size) => {
                    let rem = &mut remaining[logical_dim_us];
                    if rem.is_multiple_of(oe_size.get()) {
                        *rem /= oe_size.get();
                    } else if *rem != 0 && oe_size.get() % *rem == 0 {
                        *rem = 1;
                    }
                }
                PhysDim::Dynamic => {}
            }
        }

        // Rebuild dims while counting drops in the contiguous suffix and collapsing
        // adjacent Dynamics.
        let mut drops_in_contig: usize = 0;
        let mut merged: Vec<(u8, PhysDim)> = Vec::with_capacity(dims.len());
        for (idx, (ld, pd)) in dims.iter().copied().enumerate() {
            let logical_dim_us = usize::from(ld);
            let should_skip = match (logical_dims_noneified[logical_dim_us], merged.last(), pd) {
                (Some(nidx), _, _) if idx < nidx => true,
                (_, Some(&(l, PhysDim::Dynamic)), PhysDim::Dynamic) if l == ld => true,
                _ => false,
            };
            if !should_skip {
                merged.push((ld, pd));
            } else if idx >= first_contig_idx {
                drops_in_contig += 1;
            }
        }
        *dims = merged;

        // Update contiguousness: subtract each dropped dimension that was previously counted as
        // contiguous (i.e., within the contiguous suffix). Clamp to the new number of dims.
        self.contig = usize::from(contig)
            .saturating_sub(drops_in_contig)
            .min(dims.len())
            .try_into()
            .unwrap();
    }

    // TODO: Return iterator instead?
    fn expand_physical_shape(&self, logical_shape: &[DimSize]) -> Result<Shape, LayoutError> {
        let Layout { dims, contig: _ } = self;
        let mut physical_shape = Shape::with_capacity(dims.len());
        let mut logical_shape_remaining: SmallVec<[_; 5]> =
            logical_shape.iter().map(|x| x.get()).collect();
        let mut tiled_packing: SmallVec<[bool; 5]> = smallvec![false; logical_shape.len()];
        for (dim, phys_dim) in dims.iter().rev() {
            let remaining_size = &mut logical_shape_remaining[usize::from(*dim)];
            debug_assert_ne!(
                *remaining_size, 0,
                "Dynamic dimension {dim} already seen in {dims:?}"
            );
            match phys_dim {
                PhysDim::OddEven(pack_size) | PhysDim::Packed(pack_size) => {
                    if *remaining_size < pack_size.get() {
                        if tiled_packing[usize::from(*dim)] {
                            return Err(LayoutError::InvalidShape(logical_shape.into()));
                        }
                        physical_shape.push((*remaining_size).try_into().unwrap());
                        *remaining_size = 1;
                        tiled_packing[usize::from(*dim)] = true;
                    } else if *remaining_size % pack_size.get() != 0 {
                        return Err(LayoutError::InvalidShape(logical_shape.into()));
                    } else {
                        physical_shape.push(*pack_size);
                        *remaining_size /= pack_size.get();
                    }
                }
                PhysDim::Dynamic => {
                    physical_shape.push(DimSize::new(*remaining_size).unwrap());
                    *remaining_size = 0; // zero indicates we've seen Dynamic
                }
            }
        }
        if logical_shape_remaining.iter().any(|&d| d > 1) {
            return Err(LayoutError::InvalidShape(logical_shape.into()));
        }
        physical_shape.reverse();
        Ok(physical_shape)
    }

    #[cfg(test)]
    fn physical_size(
        &self,
        physical_dim: u8,
        logical_shape: &[DimSize],
    ) -> Result<DimSize, LayoutError> {
        let Layout { dims, contig: _ } = self;
        let logical_dim = dims[usize::from(physical_dim)].0;
        let expanded = self.expand_logical_dim_physical_shape(logical_dim, logical_shape)?;
        expanded
            .into_iter()
            .find(|(idx, _)| *idx == usize::from(physical_dim))
            .map(|(_, s)| s)
            .ok_or_else(|| LayoutError::InvalidShape(logical_shape.into()))
    }

    #[cfg(test)]
    fn expand_logical_dim_physical_shape(
        &self,
        logical_dim: u8,
        logical_shape: &[DimSize],
    ) -> Result<Vec<(usize, DimSize)>, LayoutError> {
        let Layout { dims, contig: _ } = self;
        let mut physical_shape = vec![];
        let mut remaining_size = logical_shape[usize::from(logical_dim)].get();
        for (idx, (dim, fixed_size)) in dims.iter().enumerate().rev() {
            if *dim != logical_dim {
                continue;
            }
            debug_assert_ne!(
                remaining_size, 0,
                "Logical dimension {} with unpacked size already seen in {:?}",
                dim, dims
            );
            match fixed_size {
                PhysDim::OddEven(s) | PhysDim::Packed(s) => {
                    if remaining_size < s.get() {
                        physical_shape.push((idx, remaining_size.try_into().unwrap()));
                        remaining_size = 1;
                        continue;
                    }
                    if remaining_size % *s != 0 {
                        return Err(LayoutError::InvalidShape(logical_shape.into()));
                    }
                    physical_shape.push((idx, *s));
                    remaining_size /= s.get();
                }
                PhysDim::Dynamic => {
                    physical_shape.push((idx, DimSize::new(remaining_size).unwrap()));
                    remaining_size = 0; // zero is a special value for error detection
                }
            }
        }
        if remaining_size > 1 {
            return Err(LayoutError::InvalidShape(logical_shape.into()));
        }
        physical_shape.reverse();
        Ok(physical_shape)
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

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty() && self.contig == self.contiguous_none()
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

        let possible_dims = args.dims.unwrap_or_else(|| (1..5).collect());

        let packed_st = (2..=8u32).prop_map(|s| PhysDim::Packed(s.try_into().unwrap()));
        let interleaved_st = (1..=4u32).prop_map(|s| PhysDim::OddEven((s * 2).try_into().unwrap()));
        let non_dynamic_st = prop_oneof![packed_st, interleaved_st];

        let base = Just(possible_dims.clone())
            .prop_shuffle()
            .prop_map(|dims| Layout::new(dims.into_iter().map(|d| (d, PhysDim::Dynamic)).collect()))
            .prop_flat_map(move |dynamic_only_layout| {
                use proptest::prop_oneof;

                let last_logical_dim = dynamic_only_layout.dims.last().map(|(dim, _)| *dim);
                let available_dims: Vec<u8> = possible_dims
                    .iter()
                    .copied()
                    .filter(|&d| Some(d) != last_logical_dim)
                    .collect();

                (
                    Just(dynamic_only_layout),
                    if available_dims.is_empty() {
                        Just(None).boxed()
                    } else {
                        prop_oneof![
                            2 => Just(None),
                            1 => (proptest::sample::select(available_dims), non_dynamic_st.clone())
                                .prop_map(|(logical_dim, phys_dim)| Some((logical_dim, phys_dim)))
                        ]
                        .boxed()
                    },
                )
            })
            .prop_flat_map(|(mut layout, extra)| {
                if let Some(extra) = extra {
                    layout.dims.push(extra);
                }
                let contigs = layout.all_contiguous_abs().collect::<Vec<_>>();
                (Just(layout), proptest::sample::select(contigs))
            })
            .prop_map(|(mut layout, contig)| {
                layout.set_contig(contig);
                layout
            });
        // TODO: Add non-Dynamic dimensions too.
        base.boxed()
    }
}

impl Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Layout { dims, contig } = self;

        if dims[..]
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

pub fn row_major(shape: &[DimSize]) -> Layout {
    let rank = u8::try_from(shape.len()).unwrap();
    Layout::new(
        (0..rank)
            .filter_map(|d| {
                if shape[usize::from(d)].get() == 1 {
                    None
                } else {
                    Some((d, PhysDim::Dynamic))
                }
            })
            .collect(),
    )
}

pub fn col_major(shape: &[DimSize]) -> Layout {
    let mut layout = row_major(shape);
    layout.dims.reverse();
    layout
}

pub(crate) fn batched_col_major(shape: &[DimSize]) -> Layout {
    if shape.is_empty() {
        return Layout::new(vec![]);
    }
    let mut layout = col_major(&shape[1..]);
    for (logical_dim, _) in layout.dims.iter_mut() {
        *logical_dim += 1;
    }
    let Some(batch_size) = shape.first() else {
        unreachable!();
    };
    if batch_size.get() != 1 {
        layout.dims.insert(0, (0, PhysDim::Dynamic));
        layout.contig += 1;
    }
    layout
}

pub fn nhwc(shape: &[DimSize]) -> Layout {
    assert_eq!(shape.len(), 4, "NHWC layout is for 4D tensors");
    let mut l = layout![0, 2, 3, 1];
    l.dims.retain(|(d, _)| shape[usize::from(*d)].get() != 1);
    l.contig = l.dims.len().try_into().unwrap();
    l
}

pub mod macros {
    #[macro_export]
    macro_rules! layout {
        ( @inner [ $( $out:expr, )* ] , ) => {
            $crate::layout::Layout::new(vec![ $( $out, )* ])
        };
        ( @inner [ $( $out:expr, )* ] , , $( $rest:tt )* ) => {
            $crate::layout!{ @inner [ $( $out, )* ] , $( $rest )* }
        };
        ( @inner [ $( $out:expr, )* ] , $dim:tt p ( $ds:expr ) , $( $rest:tt )* ) => {
            $crate::layout!{ @inner [ $( $out, )* (($dim), $crate::layout::PhysDim::Packed($crate::spec::macros::internal::IntoDimSize::into_dim_size($ds))), ] , $( $rest )* }
        };
        ( @inner [ $( $out:expr, )* ] , $dim:tt oe ( $ds:expr ) , $( $rest:tt )* ) => {
            $crate::layout!{ @inner [ $( $out, )* (($dim), $crate::layout::PhysDim::OddEven($crate::spec::macros::internal::IntoDimSize::into_dim_size($ds))), ] , $( $rest )* }
        };
        ( @inner [ $( $out:expr, )* ] , $dim:expr , $( $rest:tt )* ) => {
            $crate::layout!{ @inner [ $( $out, )* ($dim, $crate::layout::PhysDim::Dynamic), ] , $( $rest )* }
        };
        ( $( $t:tt )* ) => {
            $crate::layout!{ @inner [ ] , $( $t )* , }
        };
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
    use std::{collections::HashSet, iter};

    #[test]
    fn test_parenthesized_shorthand_packed() {
        let actual = layout![0, 1 p(8)];
        let expected = Layout::new(vec![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Packed(DimSize::new(8).unwrap())),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_parenthesized_shorthand_oddeven() {
        // layout![0 oe(16), 1] should equal Layout::new([(0, OddEven(16)), (1, Dynamic)])
        let actual = layout![0 oe(16), 1];
        let expected = Layout::new(vec![
            (0, PhysDim::OddEven(DimSize::new(16).unwrap())),
            (1, PhysDim::Dynamic),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_expand_physical_shape_1() {
        let layout = layout![0, 1, 0 p(4)];
        assert_eq!(
            layout.expand_physical_shape(&shape![64, 64]).unwrap()[..],
            *shape![16, 64, 4],
        );
    }

    #[test]
    fn test_expand_physical_shape_2() {
        let layout = layout![0, 1 p(2)];
        assert_eq!(
            layout.expand_physical_shape(&shape![2, 2]).unwrap()[..],
            *shape![2, 2],
        );
    }

    #[test]
    fn test_expand_physical_shape_3() {
        // Layout doesn't apply to 1x64 because there is no Dynamic for dim 0.
        let layout = layout![0 p(4), 1];
        assert_eq!(
            layout.expand_physical_shape(&shape![1, 64]),
            Ok(shape![1, 64])
        );
    }

    #[test]
    fn test_expand_physical_shape_4() {
        let layout = layout![0 p(2)];
        let expanded = layout.expand_physical_shape(&shape![6]);
        assert!(
            matches!(expanded, Err(LayoutError::InvalidShape(_))),
            "Expected LayoutError::InvalidShape, but was: {expanded:?}",
        );
    }

    #[test]
    fn test_expand_physical_shape_5() {
        let layout = layout![0, 1, 0 p(4)];
        assert_eq!(
            layout.expand_physical_shape(&shape![2, 64]),
            Ok(shape![1, 64, 2]),
        );
    }

    #[test]
    fn test_expand_physical_shape_6() {
        // Layout only applies to one shape: [2].
        let layout = layout![0 p(2)];
        let expanded = layout.expand_physical_shape(&shape![1]);
        assert_eq!(expanded, Ok(shape![1]));
    }

    #[test]
    fn test_expand_physical_shape_7() {
        let layout = layout![0 p(4), 1];
        let expanded = layout.expand_physical_shape(&shape![8, 64]);
        assert!(
            matches!(expanded, Err(LayoutError::InvalidShape(_))),
            "Expected LayoutError::InvalidShape, but was: {expanded:?}",
        );
    }

    #[test]
    fn test_expand_physical_shape_8() {
        let layout = layout![1, 0, 1 p(8)];
        let expanded = layout.expand_physical_shape(&shape![10, 4]);
        assert_eq!(expanded, Ok(shape![1, 10, 4]));
    }

    #[test]
    fn test_expand_physical_shape_9() {
        let layout = layout![1, 0, 1 p(4), 0 p(2), 1 p(4)];
        let expanded = layout.expand_physical_shape(&shape![10, 8]);
        assert_eq!(expanded, Ok(shape![1, 5, 2, 2, 4]))
    }

    #[test]
    fn test_expand_physical_shape_10() {
        let layout = layout![1, 0, 1 p(4), 0 p(2), 1 p(4)];
        let expanded = layout.expand_physical_shape(&shape![10, 6]);
        assert!(
            matches!(expanded, Err(LayoutError::InvalidShape(_))),
            "Expected LayoutError::InvalidShape, but was: {expanded:?}",
        )
    }

    #[test]
    fn test_physical_size_1() {
        let layout = layout![0 p(2)];
        assert!(matches!(
            layout.physical_size(0, &shape![6]),
            Err(LayoutError::InvalidShape(_)),
        ));
    }

    #[test]
    fn test_col_major_erases_size1_logical_dim() {
        let cm = col_major(&shape![128, 128]);
        let new_layout = cm
            .update_for_tiling(&shape![128, 128], &shape![8, 1])
            .unwrap();
        assert_eq!(new_layout.dims, [(0, PhysDim::Dynamic)]);
        assert_eq!(new_layout.contig, 1);
    }

    #[test]
    fn test_col_major_slices_are_contiguous() {
        let cm = col_major(&shape![128, 128]);
        let new_layout = cm
            .update_for_tiling(&shape![128, 128], &shape![8, 1])
            .unwrap();
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_1() {
        let layout = layout![0, 1];
        let new_layout = layout.dim_drop(&iter::once(0u8).collect());
        assert_eq!(new_layout, layout![0]);
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_2() {
        let layout = layout![0, 1, 0 p(4)];
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        assert_eq!(
            new_layout,
            // Would be [(0, PhysDim::Dynamic), (0, PhysDim::Packed(4))], but for merging adjacent
            // dimensions.
            layout![0]
        );
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_3() {
        let layout = layout![0, 1, 0 p(4), 2];
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        assert_eq!(new_layout, layout![0, 1]);
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_4() {
        let mut layout = layout![0, 1, 0 p(4), 2];
        layout.contig = 3;
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        let mut expected = layout![0, 1];
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
        let mut layout = layout![0, 1, 0 p(4), 2];
        layout.contig = 3;
        let new_layout = layout.dim_drop(&iter::once(0u8).collect());
        assert_eq!(new_layout, layout![0, 1]);
        assert_eq!(new_layout.contig(), 2);
    }

    #[test]
    fn test_dim_drop_6() {
        let mut layout = layout![0, 1, 0 p(4), 2];
        layout.contig = 2;
        let new_layout = layout.dim_drop(&iter::once(1u8).collect());
        let mut expected = layout![0, 1];
        expected.contig = 1;
        assert_eq!(new_layout, expected);
        // As in test_dim_drop_4, we lose some contiguousness information here as a result
        // of merging during dim_drop.
        assert_eq!(new_layout.contig(), 1);
    }

    #[test]
    fn test_dim_drop_7() {
        let layout = layout![0, 1, 0 p(4), 2];
        let new_layout = layout.dim_drop(&iter::once(0u8).collect());
        assert_eq!(new_layout, layout![0, 1]);
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_dim_drop_8() {
        let layout = layout![0, 1, 0 p(4), 2];
        let new_layout = layout.dim_drop(&iter::once(2u8).collect());
        assert_eq!(new_layout, layout![0, 1, 0 p(4)]);
        assert!(new_layout.is_fully_contiguous());
    }

    #[test]
    fn test_drop_unneeded_packings_1() {
        let mut layout = layout![0, 1, 0 p(4)];
        let expected = layout![1, 0];
        layout.drop_unneeded_packings(&[nz!(4u32), nz!(1024u32)]);
        assert_eq!(layout, expected);
    }

    #[test]
    fn test_drop_unneeded_packings_2() {
        let mut layout = layout![
            0,
            1,
            2, // will be dropped when below is converted
            1 p(256), // after drop, will be contig. with above Dynamic
            2 p(512), // will be converted to Dynamic
            1 p(4),
        ];
        let expected = layout![0, 1, 2, 1 p(nz!(4u32))];
        layout.drop_unneeded_packings(&[nz!(1u32), nz!(2048u32), nz!(512u32)]);
        layout.merge_consecutive_dimensions(); // TODO: Remove
        assert_eq!(layout, expected);
    }

    #[test]
    fn test_drop_unneeded_packings_3() {
        let mut layout = layout![0, 1, 0 p(4), 1, 0 p(4)];
        let expected = layout![1, 0];
        layout.drop_unneeded_packings(&[nz!(4u32), nz!(1024u32)]);
        layout.merge_consecutive_dimensions(); // TODO: Remove
        assert_eq!(layout, expected);
    }

    #[test]
    fn test_drop_unneeded_packings_4() {
        let mut layout = layout![0, 1, 0 p(4), 1, 0 p(4)];
        let expected = layout![1, 0, 1, 0 p(4)];
        layout.drop_unneeded_packings(&[nz!(16u32), nz!(1024u32)]);
        layout.merge_consecutive_dimensions(); // TODO: Remove
        assert_eq!(layout, expected);
    }

    #[test]
    fn test_update_for_tiling_can_tile_packed_dimension() {
        let layout = layout![0, 2, 1, 2 p(32)];
        let tiled_layout = layout
            .update_for_tiling(&shape![2, 4, 64], &shape![1, 4, 16])
            .expect("tiling within a packed physical dimension should succeed");
        let mut expected_layout = layout![1, 2];
        expected_layout.contig = 1;
        assert_eq!(tiled_layout, expected_layout);
    }

    #[test]
    fn test_drop_unneeded_packings_oddeven_contributes_volume() {
        let mut layout = layout![1, 0 oe(4), 0 p(2)];
        let expected = layout![1, 0 oe(4), 0 p(2)];
        layout.drop_unneeded_packings(&[nz!(8u32), nz!(16u32)]);
        assert_eq!(layout, expected);
    }

    #[test]
    fn test_drop_unneeded_packings_oddeven_innermost_drops_outer_packed() {
        let mut layout = layout![0, 1, 0 p(2), 0 oe(4)];
        let expected = layout![1, 0, 0 oe(4)];
        layout.drop_unneeded_packings(&[nz!(8u32), nz!(16u32)]);
        assert_eq!(layout, expected);
    }

    #[test]
    fn test_drop_unneeded_packings_no_change_when_packed_not_noneified_and_oddeven_consumes() {
        let mut layout = layout![0 oe(6), 1, 0 p(4)];
        let expected = layout![0 oe(6), 1, 0 p(4)];
        layout.drop_unneeded_packings(&[nz!(12u32), nz!(16u32)]);
        assert_eq!(layout, expected);
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
            for i in 0..u8::try_from(physical_rank).unwrap() {
                if rhs.is_err() {
                    break;
                }
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
                let new_layout = layout.dim_drop(&dims_to_drop.into_iter().collect());
                prop_assert!(new_layout.is_valid_contiguous_abs(new_layout.contig()));
            }
        }

        #[test]
        fn test_update_for_tiling_is_idempotent(
            (shape, layout) in arb_shape_and_same_rank_layout(),
            same_size in any::<bool>()
        ) {
            // First, get a smaller tile shape for testing tiling
            let tile_shape = if same_size {
                shape.clone()
            } else {
                shape
                    .iter()
                    .map(|&d| DimSize::new(1.max(d.get() / 2)).unwrap())
                    .collect()
            };

            let result1 = layout.update_for_tiling(&shape, &tile_shape);
            prop_assume!(result1.is_ok());
            let layout1 = result1.unwrap();

            let result2 = layout1.update_for_tiling(&tile_shape, &tile_shape);
            prop_assert!(result2.is_ok());
            let layout2 = result2.unwrap();

            prop_assert_eq!(layout1, layout2, "Layout changed after second application");
        }

        #[test]
        fn test_update_for_tiling_returns_no_consecutive_dimensions(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let result = layout.update_for_tiling(&shape, &shape);
            prop_assume!(result.is_ok());

            let new_layout = result.unwrap();
            let Layout { dims, contig: _ } = &new_layout;

            for idx in 1..dims.len() {
                let has_consecutive_dims = dims[idx - 1].0 == dims[idx].0
                    && matches!(dims[idx - 1].1, PhysDim::Dynamic | PhysDim::Packed(_))
                    && matches!(dims[idx].1, PhysDim::Dynamic | PhysDim::Packed(_));
                prop_assert!(!has_consecutive_dims,
                    "update_for_tiling produced consecutive packed dimensions for logical dimension {} in layout: {:?}",
                    dims[idx].0,
                    dims
                );
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
        let rm = row_major(&shape![16, 4]);
        let expr_id = OpaqueSymbol::new();
        let iexpr = rm.buffer_indexing_expr(expr_id, &shape![16, 4]);
        let expected = NonAffineExpr::constant(0)
            + Term(4, NonAffine::Leaf(BufferVar::Pt(0, expr_id)))
            + Term(1, NonAffine::Leaf(BufferVar::Pt(1, expr_id)));
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_interleaved_indexing_expression_1() {
        let layout = layout![0 oe(8)];
        let expr_id = OpaqueSymbol::new();
        let iexpr = layout.buffer_indexing_expr(expr_id, &shape![8]);

        let pt = NonAffineExpr::from(BufferVar::Pt(0, expr_id));
        let expected = (pt.clone() % 8) / 2 + (pt % 2) * 4i32;
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_interleaved_indexing_expression_2() {
        let layout = layout![0, 0 oe(8)];
        let expr_id = OpaqueSymbol::new();
        let iexpr = layout.buffer_indexing_expr(expr_id, &shape![16]);

        let pt = NonAffineExpr::from(BufferVar::Pt(0, expr_id));
        let expected = (pt.clone() / 8) * 8 + (pt.clone() % 8) / 2 + (pt % 2) * 4;
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_interleaved_indexing_expression_3() {
        let layout = layout![0, 1 oe(8)];
        let expr_id = OpaqueSymbol::new();
        let iexpr = layout.buffer_indexing_expr(expr_id, &shape![2, 8]);

        let pt0 = NonAffineExpr::from(BufferVar::Pt(0, expr_id));
        let pt1 = NonAffineExpr::from(BufferVar::Pt(1, expr_id));
        let expected = pt0.clone() * 8 + (pt1.clone() % 8) / 2 + (pt1 % 2) * 4;
        assert_eq!(iexpr, expected, "{iexpr} != {expected}");
    }

    #[test]
    fn test_batched_col_major_rank_0() {
        assert!(batched_col_major(&[]).dims.is_empty());
    }

    #[test]
    fn test_batched_col_major_standard() {
        let layout4 = batched_col_major(&shape![2, 2, 2, 2]);
        assert_eq!(
            layout4.dims,
            vec![
                (0, PhysDim::Dynamic),
                (3, PhysDim::Dynamic),
                (2, PhysDim::Dynamic),
                (1, PhysDim::Dynamic)
            ]
        );
    }

    #[test]
    fn test_layout_rf_1() {
        let original_layout = Layout {
            dims: vec![(3, PhysDim::Dynamic)],
            contig: 0,
        };
        let layout = original_layout
            .canonicalize(&shape![1, 1, 1, 2, 1])
            .unwrap();
        assert_eq!(original_layout, layout);
    }

    fn arb_shape_and_same_rank_layout() -> impl Strategy<Value = (Shape, Layout)> {
        proptest::collection::vec(1..=16u32, 1..=3).prop_flat_map(|shape| {
            let shape = shape
                .into_iter()
                .map(|x| DimSize::new(x).unwrap())
                .collect::<Vec<_>>();
            let bounds = LayoutArbRankBounds::for_shape(&shape);
            let all_layouts = any_with::<Layout>(bounds);
            (Just(Shape::from(shape)), all_layouts)
        })
    }
}
