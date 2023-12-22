use anyhow::anyhow;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{
    collections::HashSet,
    fmt::Display,
    hash::Hash,
    iter,
};

use crate::{
    common::{Contig, DimSize, Dtype, Shape},
    expr::{AffineForm, Atom, Bounds, NonAffine, NonAffineExpr},
    opaque_symbol::OpaqueSymbol,
    target::Target,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deserialize, Serialize)]
pub enum Layout {
    New(Vec<(u8, Option<DimSize>)>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BufferVar {
    TileIdx(u8, OpaqueSymbol),
    // TODO: *Safely* remove OpaqueSymbol from Pt, if possible
    Pt(u8, OpaqueSymbol),
}

impl Layout {
    pub fn new(dims: Vec<(u8, Option<DimSize>)>) -> Layout {
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

    pub fn new_standard(dim_order: SmallVec<[u8; 5]>, shape: &[DimSize]) -> Layout {
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
            dim_remaining_volume[logical_dim_us] /= physical_size;

            // Construct a "term" for this physical dimension: really, an expression parameterized
            // by a logical dimension.
            let mut term = BufferVar::Pt(logical_dim, expr_id.clone()).into();
            if concrete_shape[logical_dim_us] != physical_size {
                if prev_remaining_volume != concrete_shape[logical_dim_us] {
                    term = NonAffine::Mod(Box::new(term), prev_remaining_volume).into();
                }
                term = NonAffine::FloorDiv(Box::new(term), dim_remaining_volume[logical_dim_us])
                    .into();
            }

            working_expr *= i32::try_from(physical_size).unwrap();
            working_expr += term;
        }

        debug_assert!(dim_remaining_volume.iter().all(|&d| d == 1));
        working_expr
    }

    pub fn contiguous_full(&self) -> Contig {
        match self {
            Layout::New(dims) => dims.len().try_into().unwrap(),
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

        let mut lines = physical_shape[..first_contig_idx].iter().product::<u32>()
            * DimSize::from(dtype.size());
        lines *= divrem::DivCeil::div_ceil(
            physical_shape[first_contig_idx..].iter().product::<u32>() * u32::from(dtype.size()),
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

    // TODO: Merge in contig_tile_transition?
    pub fn update_for_tiling(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
        contig: Contig,
    ) -> anyhow::Result<(Layout, Contig)> {
        // TODO: Can we remove this case without affecting behavior?
        if tile_shape.iter().all(|d| *d == 1) {
            let rm_layout = row_major(tile_shape.len().try_into().unwrap());
            let new_contig = rm_layout.contiguous_full();
            Ok((rm_layout, new_contig))
        } else {
            // println!("shapes first: {:?} and {:?}", parent_shape, tile_shape);
            // println!("layout first: {}", self);
            // println!("contig first: {}", contig);
            let new_layout = self;
            let new_contig = new_layout.drop_contig_for_tiling(parent_shape, tile_shape, contig)?;
            // println!("layout after phase 1: {:?}", new_layout);
            // println!("contig after phase 1: {}", new_contig);
            // let (mut new_layout, new_contig) = new_layout.merge_consecutive_dimensions(new_contig);
            new_layout.assert_no_consecutive_dimensions();
            // println!("layout after phase 2: {:?}", new_layout);
            // println!("contig after phase 2: {}", new_contig);
            let mut new_layout = new_layout.clone();
            let new_contig = new_layout.drop_unneeded_packings(tile_shape, new_contig);
            // println!("layout after phase 3: {:?}", new_layout);
            // println!("contig after phase 3: {}", new_contig);
            // let (new_layout, new_contig) =
            //     new_layout.move_ones_to_inside(parent_shape, tile_shape, new_contig);
            // println!("layout after phase 4: {:?}", new_layout);
            // println!("contig after phase 4: {}", new_contig);
            let new_contig = new_layout.increase_contig_through_ones(tile_shape, new_contig);
            // println!("layout after phase 5: {:?}", new_layout);
            // println!("contig after phase 5: {}", new_contig);
            Ok((new_layout, new_contig))
        }
    }

    /// Drop Contig to the first broken physical dimension.
    fn drop_contig_for_tiling(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
        source_contig: Contig,
    ) -> anyhow::Result<Contig> {
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
        Ok((matching_suffix_len)
            .map(|d| d + 1)
            .unwrap_or(dims.len())
            .min(source_contig.into())
            .try_into()
            .unwrap())
    }

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

    /// Move ones to the inside, sorting the size-one dims., updating Contig as appropriate.
    fn move_ones_to_inside(
        &self,
        parent_shape: &[DimSize],
        tile_shape: &[DimSize],
        source_contig: Contig,
    ) -> (Layout, Contig) {
        let Layout::New(dims) = self;

        let first_contig_idx = dims.len() - usize::from(source_contig);

        let mut new_dims = Vec::with_capacity(dims.len());
        let mut new_dims_back = Vec::new();
        let mut new_contig = source_contig;
        for (idx, (dim, packing_size)) in dims.iter().enumerate() {
            if tile_shape[usize::from(*dim)] != 1 {
                new_dims.push((*dim, *packing_size));
                continue;
            }
            if new_dims_back.iter().any(|(d, _)| d == dim) {
                if idx >= first_contig_idx {
                    new_contig -= 1;
                }
            } else {
                new_dims_back.push((*dim, None));
                if idx < first_contig_idx {
                    new_contig += 1;
                }
            }
        }
        new_dims_back.sort_by_key(|&(d, _)| d);
        new_dims.extend_from_slice(&new_dims_back);
        (Layout::new(new_dims), new_contig)
    }

    /// Merge matching, consecutive dimensions.
    fn merge_consecutive_dimensions(&self, source_contig: Contig) -> (Layout, Contig) {
        let Layout::New(dims) = self;

        let first_contig_idx = dims.len() - usize::from(source_contig);

        let mut new_contig = source_contig;
        let mut new_dims = Vec::with_capacity(dims.len());
        new_dims.push(dims[0]);

        for (idx, (dim, packing_size)) in dims.iter().skip(1).enumerate() {
            let (last_dim, last_packing_size) = new_dims.last_mut().unwrap();
            if dim != last_dim {
                new_dims.push((*dim, *packing_size));
                continue;
            }

            match (last_packing_size, packing_size) {
                (Some(l), Some(n)) => {
                    *l *= *n;
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
            if physical_tile_shape[first_contig_idx] != 1 {
                break;
            }
            new_contig += 1;
        }
        Contig::from(new_contig)
    }

    fn drop_unneeded_packings(&mut self, tile_shape: &[DimSize], contig: Contig) -> Contig {
        let Layout::New(dims) = self;

        let first_contig_idx = dims.len() - usize::from(contig);

        let mut packings = vec![0; dims.len()];
        for (logical_dim, s) in dims.as_slice() {
            if s.is_some() {
                packings[usize::from(*logical_dim)] += 1;
            }
        }

        let mut logical_dims_noneified = std::collections::HashMap::<u8, _>::new();
        let mut new_contig = contig;
        for idx in (0..dims.len()).rev() {
            let (logical_dim, s) = dims[idx];
            if packings[usize::from(logical_dim)] != 1 {
                continue;
            }
            match s {
                Some(fixed_size) if tile_shape[usize::from(logical_dim)] == fixed_size => {
                    dims[idx] = (logical_dim, None);
                    logical_dims_noneified.insert(logical_dim, idx);
                }
                None if idx >= first_contig_idx
                    && logical_dims_noneified.contains_key(&logical_dim) =>
                {
                    // We know this will be 1 since we'll have already visited the packed dimension
                    // with the same size as the logical dimension.
                    new_contig -= 1;
                }
                _ => {}
            }
        }

        let mut i = 0;
        dims.retain(|(logical_dim, _)| {
            let should_retain = if let Some(noneified_idx) = logical_dims_noneified.get(logical_dim)
            {
                i >= *noneified_idx
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
    ) -> anyhow::Result<Vec<DimSize>> {
        let Layout::New(dims) = self;
        let mut physical_shape = Vec::with_capacity(dims.len());
        let mut logical_shape_remaining = logical_shape.to_vec();
        for (dim, fixed_size) in dims.iter().rev() {
            let remaining_size = &mut logical_shape_remaining[usize::from(*dim)];
            debug_assert_ne!(
                remaining_size, &0,
                "Logical dimension {} with unpacked sized already seen in {:?}",
                dim, dims
            );
            match fixed_size {
                Some(s) => {
                    if *remaining_size % *s != 0 {
                        return Err(anyhow!(
                            "Cannot apply shape {:?} to {:?}",
                            self,
                            logical_shape
                        ));
                    }
                    physical_shape.push(*s);
                    *remaining_size /= *s;
                }
                None => {
                    physical_shape.push(*remaining_size);
                    *remaining_size = 0; // zero is a special value for error detection
                }
            }
        }
        physical_shape.reverse();
        Ok(physical_shape)
    }

    fn assert_no_size_1_packings(&self) {
        #[cfg(debug_assertions)]
        {
            let Layout::New(dims) = self;
            for (_, size) in dims {
                debug_assert_ne!(size, &Some(1), "Size-1 packing in layout: {:?}", dims);
            }
        }
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

pub fn nhwc(tensor_shape: &[DimSize]) -> Layout {
    assert_eq!(tensor_shape.len(), 4);
    Layout::new(vec![(0, None), (2, None), (3, None), (1, None)])
}

#[cfg(test)]
mod tests {
    use super::{col_major, Layout};
    use crate::{
        common::DimSize,
        expr::{AffineForm, Bounds, NonAffine, NonAffineExpr, Substitute, Term},
        layout::{row_major, BufferVar},
        opaque_symbol::OpaqueSymbol,
    };
    use itertools::Itertools;
    use proptest::{
        arbitrary::any,
        prelude::prop,
        proptest,
        strategy::{Just, Strategy},
    };
    use std::collections::HashSet;

    #[test]
    fn test_expand_physical_shape() {
        let layout = Layout::new(vec![(0, None), (1, None), (0, Some(4))]);
        assert_eq!(
            layout.expand_physical_shape(&[64, 64]).unwrap(),
            vec![16, 64, 4]
        );
    }

    #[test]
    fn test_col_major_slices_are_non_contiguous() {
        let cm = col_major(2);
        let (_, inner_contig) = cm
            .update_for_tiling(&[128, 128], &[8, 1], cm.contiguous_full())
            .unwrap();
        assert_eq!(inner_contig, cm.contiguous_full());
    }

    proptest! {
        #[test]
        fn test_layout_underapproximates_fully_contiguous(
            test_input in test_layout_fully_contiguous_or_not_strategy(),
            first_int in 0..3u32,
        ) {
            let (tensor_shape, layout, tile_shape, tile_offset) = test_input;

            // Lay out sequential integers in a Vec according to the layout's indexing expression.
            let e = OpaqueSymbol::new();
            let iexpr = layout.buffer_indexing_expr(&e, &tensor_shape);
            let tensor_volume = tensor_shape.iter().copied().product::<u32>();
            let tensor_buffer = (first_int..(first_int + tensor_volume)).collect::<Vec<_>>();

            let (layout, updated_contig) = layout.update_for_tiling(&tensor_shape, &tile_shape, layout.contiguous_full()).unwrap();

            // Walk the memory locations to check correctness.
            let mut visited = HashSet::new();
            for pt in tile_offset
                .iter()
                .zip(&tile_shape)
                .map(|(&off, &within_pt)| off..off + within_pt)
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
            assert_eq!(analysis_result_fully_contig, is_fully_contig,
                "Layout {:?} is {}fully contiguous but contig.={:?} (full={:?})", layout,
                if is_fully_contig { "" } else { "not " },
                updated_contig, layout.contiguous_full());
        }
    }

    fn test_layout_fully_contiguous_or_not_strategy(
    ) -> impl Strategy<Value = (Vec<DimSize>, Layout, Vec<DimSize>, Vec<DimSize>)> {
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
        let iexpr = rm.buffer_indexing_expr(&expr_id, &[16, 4]);
        let expected = NonAffineExpr::constant(0)
            + Term(4, NonAffine::Leaf(BufferVar::Pt(0, expr_id.clone())))
            + Term(1, NonAffine::Leaf(BufferVar::Pt(1, expr_id.clone())));
        assert_eq!(iexpr, expected, "{} != {}", iexpr, expected);
    }
}
