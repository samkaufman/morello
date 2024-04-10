use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use std::{collections::HashSet, fmt::Display, hash::Hash, iter};

use crate::{
    common::{Contig, DimSize, Dtype, Shape},
    expr::{AffineForm, Atom, Bounds, NonAffine, NonAffineExpr},
    layout,
    opaque_symbol::OpaqueSymbol,
    target::Target,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deserialize, Serialize)]
pub enum Layout {
    New(SmallVec<[(u8, Option<DimSize>); 4]>),
}

#[cfg(test)]
pub struct LayoutArbRankBounds(std::num::NonZeroU8, Option<std::num::NonZeroU8>);

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

impl Layout {
    pub fn new(dims: SmallVec<[(u8, Option<DimSize>); 4]>) -> Layout {
        #[cfg(debug_assertions)]
        {
            assert!(!dims.is_empty());

            // Check that every logical dimension in the tensor is mentioned at least once.
            // Also check that, for each logical dimension, only index 0 can be None.
            let logical_rank = dims.iter().map(|&(d, _)| d).max().unwrap();
            let mut seen = vec![false; usize::from(logical_rank) + 1];
            for (d, fixed_size) in &dims {
                if fixed_size.is_none() && seen[usize::from(*d)] {
                    panic!("Non-first occurrence of logical dimension {} is None", d);
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
        let l = Layout::New(dims);
        l.assert_no_consecutive_dimensions();
        l.assert_no_size_1_packings();
        l.merge_consecutive_dimensions(l.contiguous_full()).0
    }

    pub fn new_standard(dim_order: SmallVec<[u8; 5]>) -> Layout {
        Layout::new(dim_order.iter().map(|&dim| (dim, None)).collect())
    }

    pub fn new_packed(dim_count: u8, strip_dim: u8, strip_size: DimSize) -> Layout {
        Layout::new(
            (0..dim_count)
                .map(|dim| (dim, None))
                .chain(iter::once((strip_dim, Some(strip_size))))
                .collect(),
        )
    }

    pub fn buffer_indexing_expr(
        &self,
        expr_id: &OpaqueSymbol,
        concrete_shape: &[DimSize],
    ) -> NonAffineExpr<BufferVar> {
        let Layout::New(dims) = self;

        let tensor_rank = concrete_shape.len();
        debug_assert_eq!(
            tensor_rank,
            usize::from(dims.iter().map(|(d, _)| *d).max().unwrap()) + 1
        );

        let physical_shape = self.expand_physical_shape(concrete_shape).unwrap();
        let mut working_expr = AffineForm::zero();
        let mut dim_remaining_volume = Shape::from(concrete_shape);
        for (&(logical_dim, _), &physical_size) in dims.iter().zip(&physical_shape) {
            let logical_dim_us = usize::from(logical_dim);
            let prev_remaining_volume = dim_remaining_volume[logical_dim_us];
            debug_assert!(prev_remaining_volume <= concrete_shape[logical_dim_us]);
            dim_remaining_volume[logical_dim_us] =
                DimSize::new(dim_remaining_volume[logical_dim_us].get() / physical_size.get())
                    .unwrap();

            // Construct a "term" for this physical dimension: really, an expression parameterized
            // by a logical dimension.
            let mut term = BufferVar::Pt(logical_dim, expr_id.clone()).into();
            if concrete_shape[logical_dim_us] != physical_size {
                if prev_remaining_volume != concrete_shape[logical_dim_us] {
                    term = NonAffine::Mod(Box::new(term), prev_remaining_volume.get()).into();
                }
                term =
                    NonAffine::FloorDiv(Box::new(term), dim_remaining_volume[logical_dim_us].get())
                        .into();
            }

            working_expr *= i32::try_from(physical_size.get()).unwrap();
            working_expr += term;
        }

        debug_assert!(dim_remaining_volume.iter().all(|&d| d.get() == 1));
        working_expr
    }

    pub fn contiguous_full(&self) -> Contig {
        match self {
            Layout::New(dims) => dims.len().try_into().unwrap(),
        }
    }

    pub fn contiguous_none(&self) -> Contig {
        match self {
            Layout::New(_) => 0,
        }
    }

    pub fn all_contiguous_abs(&self) -> impl Iterator<Item = Contig> + Clone {
        match self {
            Layout::New(dims) => 0..=dims.len().try_into().unwrap(),
        }
    }

    pub fn estimate_cache_lines<Tgt: Target>(
        &self,
        shape: &[DimSize],
        dtype: Dtype,
        contig: Contig,
    ) -> u32 {
        let Layout::New(dims) = self;

        assert!(
            usize::from(contig) <= dims.len(),
            "Invalid contig: {} for dims: {:?}",
            contig,
            dims
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
        let Layout::New(dims) = self;
        dims.iter()
            .enumerate()
            .all(|(i, (d, s))| i == usize::from(*d) && s.is_none())
    }

    // TODO: Do we really need callers to build a HashSet?
    pub fn dim_drop(&self, dropped_dims: &HashSet<u8>, contiguous_abs: Contig) -> (Layout, Contig) {
        if dropped_dims.is_empty() {
            return (self.clone(), contiguous_abs);
        }

        let Layout::New(dims) = self;
        let first_contig_idx = dims.len() - usize::from(contiguous_abs);

        let new_contig = contiguous_abs
            - u8::try_from(
                dims[first_contig_idx..]
                    .iter()
                    .filter(|(d, _)| dropped_dims.contains(d))
                    .count(),
            )
            .unwrap();
        let new_layout = Layout::new(
            dims.iter()
                .filter(|(d, _)| !dropped_dims.contains(d))
                .copied()
                .collect(),
        );
        (new_layout, new_contig)
    }

    pub fn swap_dims(&self, dims: (u8, u8), contiguous_abs: Contig) -> (Layout, Contig) {
        let Layout::New(orig_dims) = self;
        (
            Layout::new(
                orig_dims
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
            ),
            contiguous_abs,
        )
    }

    pub fn update_for_tiling(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
        contig: Contig,
    ) -> Result<(Layout, Contig), LayoutError> {
        // TODO: Can we remove this case without affecting behavior?
        if tile_shape.iter().all(|d| d.get() == 1) {
            let rm_layout = row_major(tile_shape.len().try_into().unwrap());
            let new_contig = rm_layout.contiguous_full();
            Ok((rm_layout, new_contig))
        } else {
            let new_contig =
                self.lower_contig_to_first_broken_dimension(parent_shape, tile_shape, contig)?;
            debug_assert!(parent_shape != tile_shape || new_contig == contig);
            self.assert_no_consecutive_dimensions();
            let mut new_layout = self.clone();
            let new_contig = new_layout.drop_unneeded_packings(tile_shape, new_contig);
            let new_contig = new_layout.increase_contig_through_ones(tile_shape, new_contig);
            Ok((new_layout, new_contig))
        }
    }

    /// Drop Contig to the first broken physical dimension.
    fn lower_contig_to_first_broken_dimension(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
        source_contig: Contig,
    ) -> Result<Contig, LayoutError> {
        let Layout::New(dims) = self;
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

    /// Asserts that there are no consecutive dimensions with the same logical dimension.
    ///
    /// This does nothing on release builds.
    fn assert_no_consecutive_dimensions(&self) {
        #[cfg(debug_assertions)]
        {
            let Layout::New(dims) = self;
            for idx in 1..dims.len() {
                if dims[idx - 1].0 == dims[idx].0
                    && dims[idx - 1].1.is_some()
                    && dims[idx].1.is_some()
                {
                    panic!(
                        "Consecutive packed dimensions for logical dimension {} in layout: {:?}",
                        dims[idx].0, dims
                    );
                }
            }
        }
    }

    /// Merge matching, consecutive dimensions.
    fn merge_consecutive_dimensions(&self, source_contig: Contig) -> (Layout, Contig) {
        let Layout::New(dims) = self;

        let first_contig_idx = dims.len() - usize::from(source_contig);

        let mut new_contig = source_contig;
        let mut new_dims = SmallVec::with_capacity(dims.len());
        new_dims.push(dims[0]);

        for (idx, (dim, packing_size)) in dims.iter().skip(1).enumerate() {
            let (last_dim, last_packing_size): &mut (u8, Option<DimSize>) =
                new_dims.last_mut().unwrap();
            if dim != last_dim {
                new_dims.push((*dim, *packing_size));
                continue;
            }

            match (last_packing_size, packing_size) {
                (Some(l), Some(n)) => {
                    *l = DimSize::new(l.get() * n.get()).unwrap();
                    if idx >= first_contig_idx {
                        new_contig -= 1;
                    }
                }
                (None, Some(_)) | (Some(_), None) => {
                    new_dims.push((*dim, *packing_size));
                }
                (None, None) => {
                    panic!("Repeating non-packed dimensions is undefined: {:?}", self)
                }
            }
        }

        (Layout::New(new_dims), new_contig)
    }

    /// Increase contig through any all-ones prefix.
    fn increase_contig_through_ones(&self, tile_shape: &[DimSize], contig: Contig) -> Contig {
        let physical_tile_shape = self.expand_physical_shape(tile_shape).unwrap();
        let mut first_contig_idx = physical_tile_shape.len() - usize::from(contig);
        let mut new_contig = contig;
        while first_contig_idx > 0 {
            first_contig_idx -= 1;
            if physical_tile_shape[first_contig_idx].get() != 1 {
                break;
            }
            new_contig += 1;
        }
        Contig::from(new_contig)
    }

    fn drop_unneeded_packings(&mut self, tile_shape: &[DimSize], contig: Contig) -> Contig {
        let Layout::New(dims) = self;

        let first_contig_idx = dims.len() - usize::from(contig);

        // Count the number of packings applied to each logical dimension.
        let mut packings = vec![0; dims.len()];
        for (logical_dim, s) in dims.as_slice() {
            if s.is_some() {
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
                Some(fixed_size) if tile_shape[logical_dim_usize] == fixed_size => {
                    dims[idx] = (logical_dim, None);
                    logical_dims_noneified[logical_dim_usize] = Some(idx);
                }
                None if idx >= first_contig_idx
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

        new_contig
    }

    // TODO: Make function private. (aligned_approx needs this.)
    // TODO: Return iterator instead?
    pub(crate) fn expand_physical_shape(
        &self,
        logical_shape: &[DimSize],
    ) -> Result<Shape, LayoutError> {
        let Layout::New(dims) = self;
        let mut physical_shape = SmallVec::with_capacity(dims.len());
        let mut logical_shape_remaining: SmallVec<[u32; 5]> = Shape::from(logical_shape)
            .into_iter()
            .map(|x| x.get())
            .collect();
        for (dim, fixed_size) in dims.iter().rev() {
            let remaining_size = &mut logical_shape_remaining[usize::from(*dim)];
            debug_assert_ne!(
                remaining_size, &0,
                "Logical dimension {} with unpacked sized already seen in {:?}",
                dim, dims
            );
            match fixed_size {
                Some(s) => {
                    if *remaining_size % s.get() != 0 {
                        return Err(LayoutError::InvalidShape(logical_shape.into()));
                    }
                    physical_shape.push(*s);
                    *remaining_size /= s.get();
                }
                None => {
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
        let Layout::New(dims) = self;
        let logical_dim = dims[usize::from(physical_dim)].0;
        let expanded = self.expand_logical_dim_physical_shape(logical_dim, logical_shape)?;
        expanded
            .into_iter()
            .find(|(idx, _)| *idx == usize::from(physical_dim))
            .map(|(_, s)| s)
            .ok_or_else(|| LayoutError::InvalidShape(logical_shape.into()))
    }

    fn expand_logical_dim_physical_shape(
        &self,
        logical_dim: u8,
        logical_shape: &[DimSize],
    ) -> Result<SmallVec<[(usize, DimSize); 3]>, LayoutError> {
        let Layout::New(dims) = self;
        let mut physical_shape = smallvec![];
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
                Some(s) => {
                    if remaining_size % *s != 0 {
                        return Err(LayoutError::InvalidShape(logical_shape.into()));
                    }
                    physical_shape.push((idx, *s));
                    remaining_size /= s.get();
                }
                None => {
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
            use nonzero::nonzero as nz;

            let Layout::New(dims) = self;
            for (_, size) in dims {
                debug_assert_ne!(
                    size,
                    &Some(nz!(1u32)),
                    "Size-1 packing in layout: {:?}",
                    dims
                );
            }
        }
    }
}

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for Layout {
    type Parameters = LayoutArbRankBounds;
    type Strategy = proptest::strategy::BoxedStrategy<Layout>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::strategy::{Just, Strategy};

        let min_rank = usize::from(args.0.get());
        let max_physical_rank = usize::from(args.1.map(|r| r.into()).unwrap_or(5));
        assert!(min_rank <= max_physical_rank);

        let packed_st = (2..=8u32).prop_map(Option::Some);
        let optional_packed_st = Just(None).boxed().prop_union(packed_st.clone().boxed());

        let logical_dims_prefix =
            proptest::collection::vec(optional_packed_st, min_rank..=min_rank).prop_map(|v| {
                v.into_iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let i = u8::try_from(i).unwrap();
                        (i, p)
                    })
                    .collect::<Vec<_>>()
            });

        let max_adds = max_physical_rank - min_rank;
        let additional_logical_dims = proptest::collection::vec(
            (0..max_adds).prop_flat_map(move |i| {
                let i = u8::try_from(i).unwrap();
                (Just(i), packed_st.clone())
            }),
            0..=max_adds,
        );

        (logical_dims_prefix, additional_logical_dims)
            .prop_map(|(prefix, additional)| {
                Layout::New(
                    prefix
                        .into_iter()
                        .chain(additional)
                        .map(|(d, s)| (d, s.map(|x| DimSize::new(x).unwrap())))
                        .collect(),
                )
            })
            .boxed()
    }
}

impl Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Layout::New(dims) = self;
        if self.is_row_major() {
            write!(f, "RM")
        } else if dims.to_vec() == vec![(0, None), (2, None), (3, None), (1, None)] {
            write!(f, "NHWC")
        } else {
            write!(
                f,
                "<[{}], [{}]>",
                dims.iter()
                    .map(|(d, _)| d.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                dims.iter()
                    .map(|(_, s)| format!("{:?}", s))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

#[cfg(test)]
impl Default for LayoutArbRankBounds {
    fn default() -> Self {
        Self(std::num::NonZeroU8::new(1).unwrap(), None)
    }
}

impl Display for BufferVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferVar::TileIdx(dim, _) => write!(f, "t{}", dim),
            BufferVar::Pt(dim, _) => write!(f, "p{}", dim),
        }
    }
}

impl Atom for BufferVar {}
impl Bounds for BufferVar {}

pub fn row_major(rank: u8) -> Layout {
    Layout::new((0..rank).map(|d| (d, None)).collect())
}

pub fn col_major(rank: u8) -> Layout {
    Layout::new((0..rank).rev().map(|d| (d, None)).collect())
}

pub fn nhwc() -> Layout {
    layout![(0, None), (2, None), (3, None), (1, None)]
}

pub mod macros {
    #[macro_export]
    macro_rules! layout {
        ($($dim:tt),*$(,)*) => {
            $crate::layout::Layout::new(
                smallvec::smallvec![ $( $crate::layout::macros::internal::layout_inner!($dim) ),* ]
            )
        };
    }

    pub mod internal {
        // TODO: Make private.
        #[macro_export]
        macro_rules! __layout_inner {
            (($dim:expr, Some($ds:expr))) => {{
                use $crate::spec::macros::internal::IntoDimSize;
                ($dim, Some(($ds).into_dim_size()))
            }};
            (($dim:expr, None)) => {
                ($dim, None)
            };
        }
        pub use __layout_inner as layout_inner;
    }
}

#[cfg(test)]
mod tests {
    use super::{col_major, Layout, LayoutArbRankBounds};
    use crate::{
        common::{DimSize, Shape},
        expr::{AffineForm, Bounds, NonAffine, NonAffineExpr, Substitute, Term},
        layout,
        layout::{row_major, BufferVar, LayoutError},
        opaque_symbol::OpaqueSymbol,
        shape,
    };
    use itertools::Itertools;
    use nonzero::nonzero as nz;
    use proptest::{
        arbitrary::{any, any_with},
        prelude::prop,
        prop_assert_eq, prop_assume, proptest,
        strategy::{Just, Strategy},
    };
    use smallvec::smallvec;
    use std::{collections::HashSet, num::NonZeroU8};

    #[test]
    fn test_expand_physical_shape_1() {
        let layout = layout![(0, None), (1, None), (0, Some(4))];
        assert_eq!(
            layout.expand_physical_shape(&shape![64, 64]).unwrap()[..],
            *shape![16, 64, 4],
        );
    }

    #[test]
    fn test_expand_physical_shape_2() {
        let layout = layout![(0, None), (1, Some(2))];
        assert_eq!(
            layout.expand_physical_shape(&shape![2, 2]).unwrap()[..],
            *shape![2, 2],
        );
    }

    #[test]
    fn test_expand_physical_shape_3() {
        let layout = layout![(0, Some(4)), (1, None)];
        assert!(matches!(
            layout.expand_physical_shape(&shape![1, 64]),
            Err(LayoutError::InvalidShape(_))
        ));
    }

    #[test]
    fn test_expand_physical_shape_4() {
        let layout = layout![(0, Some(2))];
        let expanded = layout.expand_physical_shape(&shape![6]);
        assert!(
            matches!(expanded, Err(LayoutError::InvalidShape(_))),
            "Expected LayoutError::InvalidShape, but was: {expanded:?}",
        );
    }

    #[test]
    fn test_expand_physical_shape_5() {
        let layout = layout![(0, None), (1, None), (0, Some(4))];
        assert!(matches!(
            layout.expand_physical_shape(&shape![2, 64]),
            Err(LayoutError::InvalidShape(_)),
        ));
    }

    #[test]
    fn test_expand_physical_shape_6() {
        let layout = layout![(0, Some(2))];
        assert!(matches!(
            layout.expand_physical_shape(&shape![1]),
            Err(LayoutError::InvalidShape(_)),
        ));
    }

    #[test]
    fn test_expand_physical_shape_7() {
        let layout = layout![(0, Some(4)), (1, None)];
        let expanded = layout.expand_physical_shape(&shape![8, 64]);
        assert!(
            matches!(expanded, Err(LayoutError::InvalidShape(_))),
            "Expected LayoutError::InvalidShape, but was: {expanded:?}",
        );
    }

    #[test]
    fn test_physical_size_1() {
        let layout = layout![(0, Some(2))];
        assert!(matches!(
            layout.physical_size(0, &shape![6]),
            Err(LayoutError::InvalidShape(_)),
        ));
    }

    #[test]
    fn test_col_major_slices_are_non_contiguous() {
        let cm = col_major(2);
        let (_, inner_contig) = cm
            .update_for_tiling(&shape![128, 128], &shape![8, 1], cm.contiguous_full())
            .unwrap();
        assert_eq!(inner_contig, cm.contiguous_full());
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
            let Layout::New(dims) = &layout;
            let physical_rank = dims.len();

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
            let Layout::New(dims) = &layout;
            let physical_rank = dims.len();

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
            let iexpr = layout.buffer_indexing_expr(&e, &tensor_shape);
            let tensor_volume = tensor_shape.iter().map(|d| d.get()).product::<u32>();
            let tensor_buffer = (first_int..(first_int + tensor_volume)).collect::<Vec<_>>();

            let (layout, updated_contig) = layout.update_for_tiling(&tensor_shape, &tile_shape, layout.contiguous_full()).unwrap();

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
            let analysis_result_fully_contig = updated_contig == layout.contiguous_full();
            prop_assert_eq!(
                analysis_result_fully_contig, is_fully_contig,
                "Tile is {}fully contiguous but contig.={:?} (full={:?}) (tensor={:?}, tile={:?}, visited={:?})",
                if is_fully_contig { "" } else { "not " },
                updated_contig,
                layout.contiguous_full(),
                tensor_shape,
                tile_shape,
                visited
            );
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
                            prefix = Some((u8::try_from(logical_dim).unwrap(), None));
                        };
                        prefix.into_iter().chain(
                            per_dim_shape
                                .into_iter()
                                .map(move |d| (u8::try_from(logical_dim).unwrap(), Some(d))),
                        )
                    })
                    .unzip();
                let rank = usize::from(*dim_order.iter().max().unwrap()) + 1;

                let mut tensor_shape = vec![1; rank];
                for (&dim, &d) in dim_order.iter().zip(&physical_tile_shape) {
                    match d {
                        Some(fixed) => {
                            tensor_shape[usize::from(dim)] *= fixed;
                        }
                        None => {
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
                            && physical_tile_shape[idx - 1].is_some()
                            && physical_tile_shape[idx].is_some()
                        {
                            return false;
                        }
                    }
                    true
                },
            )
            .prop_flat_map(
                |(tensor_shape, dim_order, physical_tile_shape, tile_shape)| {
                    let new_layout = Layout::new(
                        dim_order
                            .into_iter()
                            .zip(physical_tile_shape)
                            .map(|(d, s)| (d, s.map(|x| DimSize::new(x).unwrap())))
                            .collect(),
                    );
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
                    layout
                        .update_for_tiling(tensor_shape, tile_shape, layout.contiguous_full())
                        .is_ok()
                },
            )
    }

    #[test]
    fn test_row_major_indexing_expression() {
        let rm = row_major(2);
        let expr_id = OpaqueSymbol::new();
        let iexpr = rm.buffer_indexing_expr(&expr_id, &shape![16, 4]);
        let expected = NonAffineExpr::constant(0)
            + Term(4, NonAffine::Leaf(BufferVar::Pt(0, expr_id.clone())))
            + Term(1, NonAffine::Leaf(BufferVar::Pt(1, expr_id.clone())));
        assert_eq!(iexpr, expected, "{} != {}", iexpr, expected);
    }

    fn arb_shape_and_same_rank_layout() -> impl Strategy<Value = (Shape, Layout)> {
        proptest::collection::vec(1..=16u32, 1..=3).prop_flat_map(|shape| {
            let shape = shape
                .into_iter()
                .map(|x| DimSize::new(x).unwrap())
                .collect::<Vec<_>>();
            let shape_nz = NonZeroU8::try_from(u8::try_from(shape.len()).unwrap()).unwrap();
            let bounds = LayoutArbRankBounds(shape_nz, Some(shape_nz));
            let all_layouts = any_with::<Layout>(bounds);
            (Just(Shape::from(shape)), all_layouts)
        })
    }
}
