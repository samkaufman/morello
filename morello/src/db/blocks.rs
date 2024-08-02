use super::{ActionCostVec, ActionIdx, ActionNormalizedCostVec, GetPreference};
use crate::{
    common::DimSize,
    cost::{Cost, CostIntensity},
    grid::{
        canon::CanonicalBimap,
        general::BiMap,
        linear::{BimapInt, BimapSInt},
    },
    memorylimits::MemVec,
    ndarray::NDArray,
    spec::Spec,
    target::Target,
};
use enum_dispatch::enum_dispatch;
use rstar::{Envelope, Point, PointDistance, RTree, RTreeObject, RTreeParams, AABB};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::fmt::Debug;
use std::hash::Hash;
use std::{collections::HashSet, ops::Range};

#[cfg(any(feature = "db-stats", test))]
use parking_lot::Mutex;

/// A trait abstracting over concrete-sized RTreeBlockInner variants.
#[enum_dispatch]
pub(crate) trait RTreeBlockGeneric {
    fn get<Tgt: Target>(&self, inner_pt: &[u8], spec_volume: DimSize) -> Option<ActionCostVec>;

    fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    );
}

/// Stores a [Database] block. This may be a single value if all block entries have been filled with
/// the same [ActionCostVec], or an n-dimensional array along with a count of identical entries
/// accumulated until the first differing entry.
///
/// This isn't designed to be compressed in the case that a non-identical value is added and later
/// overwritten with an identical value. In that case, `matches` will be `None` and all values
/// would need to be scanned to determine whether they are all identical and, to set `matches`, how
/// many there are.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DbBlock {
    Whole(Box<WholeBlock>),
    RTree(Box<RTreeBlock>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WholeBlock {
    pub filled: NDArray<u8>, // 0 is empty; otherwise n - 1 = # of actions.
    pub main_costs: NDArray<CostIntensity>,
    pub peaks: NDArray<MemVec>,
    pub depths_actions: NDArray<(u8, ActionIdx)>,
    #[cfg(feature = "db-stats")]
    #[serde(skip)]
    access_counts: Mutex<Option<NDArray<bool>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[enum_dispatch(RTreeBlockGeneric)]
pub enum RTreeBlock {
    // TODO: The need for this enum instead of being runtime-generic in D stinks.
    D1(RTreeBlockInner<1>),
    D2(RTreeBlockInner<2>),
    D3(RTreeBlockInner<3>),
    D4(RTreeBlockInner<4>),
    D5(RTreeBlockInner<5>),
    D6(RTreeBlockInner<6>),
    D7(RTreeBlockInner<7>),
    D8(RTreeBlockInner<8>),
    D9(RTreeBlockInner<9>),
    D10(RTreeBlockInner<10>),
    D11(RTreeBlockInner<11>),
    D12(RTreeBlockInner<12>),
    D13(RTreeBlockInner<13>),
    D14(RTreeBlockInner<14>),
    D15(RTreeBlockInner<15>),
    D16(RTreeBlockInner<16>),
    D17(RTreeBlockInner<17>),
    D18(RTreeBlockInner<18>),
    D19(RTreeBlockInner<19>),
    D20(RTreeBlockInner<20>),
    D21(RTreeBlockInner<21>),
    D22(RTreeBlockInner<22>),
    D23(RTreeBlockInner<23>),
    D24(RTreeBlockInner<24>),
    D25(RTreeBlockInner<25>),
    D26(RTreeBlockInner<26>),
    D27(RTreeBlockInner<27>),
    D28(RTreeBlockInner<28>),
    D29(RTreeBlockInner<29>),
    D30(RTreeBlockInner<30>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RTreeBlockInner<const D: usize> {
    tree: RTree<RTreeBlockRect<D>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct RTreeBlockRect<const D: usize> {
    top: RTreePt<D>,
    bottom: RTreePt<D>, // peak memory can be derived from bottom
    // TODO: Remove peaks (MemVec) from below.
    result: Option<(CostIntensity, MemVec, u8, ActionIdx)>,
}

#[serde_as]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct RTreePt<const D: usize> {
    #[serde_as(as = "[_; D]")] // TODO: Use manual serde impls instead of serde_as.
    arr: [BimapSInt; D],
}

/// Accumulates references to [rstar::RTreeObject]s to remove, then can be used as a
/// [rstar::SelectionFunction].
#[derive(Debug)]
struct BatchRemoveSelFn<O: rstar::RTreeObject> {
    // TODO: Own references, not clones.
    to_remove: HashSet<O>,
    envelope: Option<O::Envelope>,
}

impl DbBlock {
    pub(super) fn get_with_preference<Tgt>(
        &self,
        query: &Spec<Tgt>,
        inner_pt: &[u8],
    ) -> GetPreference<ActionCostVec, Vec<ActionIdx>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let block_result = match self {
            DbBlock::Whole(b) => {
                // TODO: Propogate an action index preference.
                let inner_pt_usize = inner_pt.iter().map(|v| *v as usize).collect::<Vec<_>>();
                b.get(&inner_pt_usize, query.0.volume())
            }
            DbBlock::RTree(b) => b.get::<Tgt>(inner_pt, query.0.volume()),
        };
        match block_result {
            Some(r) => GetPreference::Hit(r),
            None => GetPreference::Miss(None),
        }
    }

    pub(super) fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    ) {
        match self {
            DbBlock::Whole(b) => b.fill_region(k, dim_ranges, value),
            DbBlock::RTree(b) => b.fill_region(k, dim_ranges, value),
        }
    }
}

impl WholeBlock {
    fn empty<Tgt: Target>(k: u8, shape: &[usize]) -> Self {
        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());

        WholeBlock {
            filled: NDArray::new_with_value(shape, 0),
            main_costs: NDArray::new(&shape_with_k),
            peaks: NDArray::new_with_value(&shape_with_k, MemVec::zero::<Tgt>()),
            depths_actions: NDArray::new(&shape_with_k),
            #[cfg(feature = "db-stats")]
            access_counts: Mutex::new(Some(NDArray::new_with_value(shape, false))),
        }
    }

    pub(crate) fn partially_filled<Tgt: Target>(
        k: u8,
        shape: &[usize],
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    ) -> Self {
        let mut e = Self::empty::<Tgt>(k, shape);
        e.fill_region_without_updating_match(k, dim_ranges, value);
        e
    }

    pub(crate) fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    ) {
        self.fill_region_without_updating_match(k, dim_ranges, value);
    }

    fn fill_region_without_updating_match(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    ) {
        let shape = self.filled.shape();
        debug_assert_eq!(dim_ranges.len(), shape.len());

        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());

        self.filled
            .fill_region(dim_ranges, u8::try_from(value.0.len()).unwrap() + 1);
        self.main_costs.fill_broadcast_1d(
            dim_ranges,
            value.0.iter().map(|(_, c)| c.intensity),
            Some(&self.filled),
        );
        self.peaks.fill_broadcast_1d(
            dim_ranges,
            value.0.iter().map(|(_, c)| c.peaks.clone()),
            Some(&self.filled),
        );
        self.depths_actions.fill_broadcast_1d(
            dim_ranges,
            value.0.iter().map(|(a, c)| (c.depth, *a)),
            Some(&self.filled),
        );

        #[cfg(feature = "db-stats")]
        {
            let mut guard = self.access_counts.lock();
            let l =
                guard.get_or_insert_with(|| NDArray::new_with_value(self.filled.shape(), false));
            l.fill_region(dim_ranges, true);
        }
    }

    pub(crate) fn get(&self, pt: &[usize], spec_volume: DimSize) -> Option<ActionCostVec> {
        #[cfg(feature = "db-stats")]
        self.log_access(pt);

        let f = self.filled[pt];
        if f == 0 {
            return None;
        }
        let local_k = f - 1;
        let mut pt_with_k = Vec::with_capacity(pt.len() + 1);
        pt_with_k.extend_from_slice(pt);
        pt_with_k.push(0);
        Some(ActionCostVec(
            (0..local_k)
                .map(move |i| {
                    *pt_with_k.last_mut().unwrap() = i.into();
                    let (depth, action_idx) = self.depths_actions[&pt_with_k];
                    (
                        action_idx,
                        Cost {
                            main: self.main_costs[&pt_with_k]
                                .into_main_cost_for_volume(spec_volume),
                            peaks: self.peaks[&pt_with_k].clone(),
                            depth,
                        },
                    )
                })
                .collect(),
        ))
    }

    #[cfg(feature = "db-stats")]
    fn log_access(&self, pt: &[usize]) {
        let mut guard = self.access_counts.lock();
        let l = guard.get_or_insert_with(|| NDArray::new_with_value(self.filled.shape(), false));
        l.set_pt(pt, true);
    }
}

impl Clone for WholeBlock {
    fn clone(&self) -> Self {
        #[cfg(feature = "db-stats")]
        let access_counts_guard = self.access_counts.lock();

        Self {
            filled: self.filled.clone(),
            main_costs: self.main_costs.clone(),
            peaks: self.peaks.clone(),
            depths_actions: self.depths_actions.clone(),
            #[cfg(feature = "db-stats")]
            access_counts: Mutex::new(access_counts_guard.clone()),
        }
    }
}

impl RTreeBlock {
    pub fn empty(rank: usize) -> Self {
        match rank {
            1 => RTreeBlock::D1(RTreeBlockInner::default()),
            2 => RTreeBlock::D2(RTreeBlockInner::default()),
            3 => RTreeBlock::D3(RTreeBlockInner::default()),
            4 => RTreeBlock::D4(RTreeBlockInner::default()),
            5 => RTreeBlock::D5(RTreeBlockInner::default()),
            6 => RTreeBlock::D6(RTreeBlockInner::default()),
            7 => RTreeBlock::D7(RTreeBlockInner::default()),
            8 => RTreeBlock::D8(RTreeBlockInner::default()),
            9 => RTreeBlock::D9(RTreeBlockInner::default()),
            10 => RTreeBlock::D10(RTreeBlockInner::default()),
            11 => RTreeBlock::D11(RTreeBlockInner::default()),
            12 => RTreeBlock::D12(RTreeBlockInner::default()),
            13 => RTreeBlock::D13(RTreeBlockInner::default()),
            14 => RTreeBlock::D14(RTreeBlockInner::default()),
            15 => RTreeBlock::D15(RTreeBlockInner::default()),
            16 => RTreeBlock::D16(RTreeBlockInner::default()),
            17 => RTreeBlock::D17(RTreeBlockInner::default()),
            18 => RTreeBlock::D18(RTreeBlockInner::default()),
            19 => RTreeBlock::D19(RTreeBlockInner::default()),
            20 => RTreeBlock::D20(RTreeBlockInner::default()),
            21 => RTreeBlock::D21(RTreeBlockInner::default()),
            22 => RTreeBlock::D22(RTreeBlockInner::default()),
            23 => RTreeBlock::D23(RTreeBlockInner::default()),
            24 => RTreeBlock::D24(RTreeBlockInner::default()),
            25 => RTreeBlock::D25(RTreeBlockInner::default()),
            26 => RTreeBlock::D26(RTreeBlockInner::default()),
            27 => RTreeBlock::D27(RTreeBlockInner::default()),
            28 => RTreeBlock::D28(RTreeBlockInner::default()),
            29 => RTreeBlock::D29(RTreeBlockInner::default()),
            30 => RTreeBlock::D30(RTreeBlockInner::default()),
            _ => panic!("Unsupported rank: {}", rank),
        }
    }
}

impl<const D: usize> RTreeBlockGeneric for RTreeBlockInner<D> {
    fn get<Tgt: Target>(&self, inner_pt: &[u8], spec_volume: DimSize) -> Option<ActionCostVec> {
        // TODO: Avoid conversion. Instead forward a slice right into locate_at_point.
        let mut arr = [Default::default(); D];
        for (i, v) in inner_pt.iter().enumerate() {
            arr[i] = (*v).into();
        }
        let rtree_pt = RTreePt { arr };

        // TODO: Return (and test!) k > 1. (The above point may be in a space without k dim.)
        let v = self.tree.locate_at_point(&rtree_pt)?;
        Some(ActionCostVec(match &v.result {
            Some((cost_intensity, peaks, depth, action_idx)) => {
                let cost = Cost {
                    main: cost_intensity.into_main_cost_for_volume(spec_volume),
                    // peaks: MemVec::new_from_binary_scaled(peaks),
                    peaks: peaks.clone(),
                    depth: *depth,
                };
                vec![(*action_idx, cost)]
            }
            None => vec![],
        }))
    }

    fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    ) {
        if k != 1 {
            todo!("Support k > 1");
        }
        let mut bottom = RTreePt::<D>::default();
        let mut top = RTreePt::<D>::default();
        for (i, rng) in dim_ranges.iter().enumerate() {
            assert!(rng.start < rng.end);
            bottom.arr[i] = rng.start.into();
            top.arr[i] = (rng.end - 1).into();
        }

        let new_rect = RTreeBlockRect {
            top,
            bottom,
            result: value.0.first().map(|(action_idx, normalized_cost)| {
                (
                    normalized_cost.intensity,
                    normalized_cost.peaks.clone(),
                    normalized_cost.depth,
                    *action_idx,
                )
            }),
        };

        rtree_merge_insert(&mut self.tree, new_rect);
    }
}

impl<const D: usize> RTreeObject for RTreeBlockRect<D> {
    type Envelope = AABB<RTreePt<D>>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(self.bottom.clone(), self.top.clone())
    }
}

impl<const D: usize> PointDistance for RTreeBlockRect<D> {
    fn distance_2(&self, point: &RTreePt<D>) -> BimapSInt {
        // TODO: Don't allocate an AABB
        let aabb = AABB::from_corners(self.bottom.clone(), self.top.clone());
        aabb.distance_2(point)

        // let closest_within_pt = aabb.min_point(point);
        // let mut length = 0;
        // for (a, b) in closest_within_pt.arr.iter().zip(&point.arr) {
        //     let sq = *a - *b;
        //     length += sq * sq;
        // }
        // length
    }

    fn contains_point(&self, point: &RTreePt<D>) -> bool {
        for ((p, l), h) in point.arr.iter().zip(&self.bottom.arr).zip(&self.top.arr) {
            if *p < *l || *p > *h {
                return false;
            }
        }
        true
    }

    fn distance_2_if_less_or_equal(
        &self,
        point: &RTreePt<D>,
        max_distance_2: BimapSInt,
    ) -> Option<BimapSInt> {
        let distance_2 = self.distance_2(point);
        if distance_2 <= max_distance_2 {
            Some(distance_2)
        } else {
            None
        }
    }
}

// TODO: Remove Default
impl<const D: usize> Default for RTreePt<D> {
    fn default() -> Self {
        RTreePt { arr: [0; D] }
    }
}

impl<const D: usize> From<[BimapSInt; D]> for RTreePt<D> {
    fn from(value: [BimapSInt; D]) -> Self {
        RTreePt { arr: value }
    }
}

impl<const D: usize> Point for RTreePt<D> {
    type Scalar = BimapSInt;
    const DIMENSIONS: usize = D;

    fn generate(generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        RTreePt {
            arr: std::array::from_fn(generator),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        self.arr[index]
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        &mut self.arr[index]
    }
}

impl<O> BatchRemoveSelFn<O>
where
    O: rstar::RTreeObject,
{
    fn clear(&mut self) {
        self.to_remove.clear();
        self.envelope = None;
    }
}

impl<O> BatchRemoveSelFn<O>
where
    O: rstar::RTreeObject + Eq + Hash,
{
    fn queue_removal(&mut self, candidate: O) {
        if let Some(e) = &mut self.envelope {
            e.merge(&candidate.envelope());
        } else {
            self.envelope = Some(candidate.envelope());
        }
        self.to_remove.insert(candidate);
    }
}

impl<O> Default for BatchRemoveSelFn<O>
where
    O: rstar::RTreeObject,
{
    fn default() -> Self {
        BatchRemoveSelFn {
            to_remove: HashSet::new(),
            envelope: None,
        }
    }
}

impl<O> rstar::SelectionFunction<O> for &BatchRemoveSelFn<O>
where
    O: rstar::RTreeObject + Eq + Hash,
{
    fn should_unpack_parent(&self, envelope: &<O as RTreeObject>::Envelope) -> bool {
        self.envelope
            .as_ref()
            .map(|e| e.intersects(envelope))
            .unwrap_or(false)
    }

    fn should_unpack_leaf(&self, leaf: &O) -> bool {
        self.to_remove.contains(leaf)
    }
}

fn rtree_merge_insert<const D: usize, Params: RTreeParams>(
    tree: &mut RTree<RTreeBlockRect<D>, Params>,
    new_rect: RTreeBlockRect<D>,
) {
    // Find the first rectangle which matches except for one dimension which is larger or
    // matches (and has the same cost).
    let mut to_remove = BatchRemoveSelFn::default();
    let mut to_insert = new_rect;
    let mut should_repeat = true;
    let mut skip_insert = false;

    while should_repeat {
        should_repeat = false;
        let candidate_area = AABB::from_corners(
            to_insert.bottom.arr.map(|b| b.saturating_sub(1)).into(),
            to_insert.top.arr.map(|t| t.saturating_add(1)).into(),
        );
        for candidate in tree.locate_in_envelope_intersecting(&candidate_area) {
            // When the inserted rect has matching value and would be fully contained (or is identical),
            // there's nothing to merge. The outer rect. would have already triggered applicable merge
            // rules.
            // TODO: Avoid constructing envelope if possible.
            let candidate_envelope = candidate.envelope();
            let insert_envelope = to_insert.envelope();
            if candidate_envelope.contains_envelope(&insert_envelope) {
                // Assert the MainCost, but not the later tuple elements, are unchanged.
                assert_eq!(
                    candidate.result.as_ref().map(|t| t.0),
                    to_insert.result.as_ref().map(|t| t.0)
                );
                should_repeat = false;
                skip_insert = true;
                break;
            }

            // If the candidate is contained by the to-be-inserted rectangle, remove it.
            let candidate_envelope = candidate.envelope();
            if insert_envelope.contains_envelope(&candidate_envelope) {
                // TODO: Avoid the following clone.
                to_remove.queue_removal(candidate.clone());
                // Assert the MainCost, but not the later tuple elements, are unchanged.
                // TODO: Remove the following assert and lift short-circuit.
                assert_eq!(
                    candidate.result.as_ref().map(|t| t.0),
                    to_insert.result.as_ref().map(|t| t.0)
                );
                continue;
            }

            // If a candidate extrudes the to-be-inserted rect. in exactly one dimension, just grow
            // the to-be-inserted to include it and then remove that rect.
            //
            // TODO: This condition+loop is wasteful.
            if candidate.result == to_insert.result
                && all_dimensions_adjacent_or_overlap(&to_insert, candidate)
                && count_matching_dimensions(&to_insert, candidate) == D - 1
            {
                should_repeat = true;

                let mut merged_envelope = to_insert.envelope();
                merged_envelope.merge(&candidate_envelope);
                to_insert.top = merged_envelope.upper();
                to_insert.bottom = merged_envelope.lower();

                to_remove.queue_removal(candidate.clone());
            }
        }

        let mut remove_count = 0usize;
        let expected_removals = to_remove.to_remove.len();
        for _ in tree.drain_with_selection_function(&to_remove) {
            remove_count += 1;
        }
        assert_eq!(expected_removals, remove_count);
        to_remove.clear();
    }

    if !skip_insert {
        tree.insert(to_insert);
    }
}

fn count_matching_dimensions<const D: usize>(
    new_rect: &RTreeBlockRect<D>,
    candidate: &RTreeBlockRect<D>,
) -> usize {
    let mut matching_dimensions = 0usize;
    for dim in 0..D {
        if new_rect.bottom.arr[dim] == candidate.bottom.arr[dim]
            && new_rect.top.arr[dim] == candidate.top.arr[dim]
        {
            matching_dimensions += 1;
        }
    }
    matching_dimensions
}

fn all_dimensions_adjacent_or_overlap<const D: usize>(
    lhs: &RTreeBlockRect<D>,
    rhs: &RTreeBlockRect<D>,
) -> bool {
    for dim in 0..D {
        let lhs_bottom = lhs.bottom.arr[dim];
        let lhs_top = lhs.top.arr[dim];
        let rhs_bottom = rhs.bottom.arr[dim];
        let rhs_top = rhs.top.arr[dim];
        if (lhs_bottom.saturating_sub(1)..=lhs_top).contains(&rhs_top)
            || (lhs_bottom..=lhs_top.saturating_add(1)).contains(&rhs_bottom)
        {
            continue;
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::X86Target;
    use itertools::Itertools;
    use proptest::strategy::{Just, Strategy};
    use proptest::{prop_assert_eq, proptest};

    #[test]
    #[should_panic]
    fn test_merge_insert_corners_touch_panics() {
        assert_merged(&[([0, 0], [1, 1]), ([1, 1], [2, 2])]);
    }

    #[test]
    #[should_panic]
    fn test_merge_insert_corner_overlap_panics() {
        assert_merged(&[([0, 0], [1, 1]), ([1, 0], [2, 2])]);
    }

    /// Test that two rectangles which match in all but one dimension and, in that dimension,
    /// intersects but does not otherwise overlap.
    #[test]
    fn test_merge_insert_one_dim_extrudes_same_int() {
        assert_merged(&[([0, 0, 0], [1, 1, 1]), ([0, 0, 1], [1, 1, 2])]);
    }

    /// Test that two rectangles which do not intersect but cover adjacent integer coordinates are
    /// merged. For example, that `[0, 1]` and `[2, 3]` are merged into `[0, 3]`.
    #[test]
    fn test_merge_insert_one_dim_extrudes_adjacent_int() {
        assert_merged(&[([0, 0, 0], [1, 1, 1]), ([0, 0, 2], [1, 1, 3])]);
    }

    #[test]
    fn test_merge_insert_one_dim_overlap() {
        assert_merged(&[([0, 0, 0], [1, 4, 1]), ([0, 2, 0], [1, 9, 1])]);
    }

    #[test]
    fn test_merge_insert_filling_gap_merges() {
        assert_merged(&[
            ([0, 0, 0], [1, 1, 1]),
            ([0, 0, 2], [1, 1, 3]),
            ([0, 0, 1], [1, 1, 2]),
        ]);
    }

    #[test]
    fn test_merge_insert_encloses_1() {
        assert_merged(&[([1, 1, 1], [2, 2, 2]), ([1, 1, 1], [2, 3, 2])]);
    }

    #[test]
    fn test_merge_insert_encloses_2() {
        assert_merged(&[([1, 1, 1], [2, 3, 2]), ([1, 1, 1], [2, 2, 2])]);
    }

    #[test]
    fn test_merge_insert_encloses_3() {
        assert_merged(&[([0, 0, 0], [1, 1, 1]), ([0, 0, 0], [0, 0, 0])]);
    }

    #[test]
    fn test_merge_insert_multistep() {
        assert_merged(&[([0, 0], [1, 1]), ([2, 0], [2, 2]), ([0, 2], [1, 2])]);
    }

    proptest! {
        #[test]
        fn test_disjoint_inserts_dont_change_point_results(rects in arb_disjoint_rects::<3>(3)) {
            let mut tree = RTree::new();
            for r in &rects {
                rtree_merge_insert(&mut tree, r.clone());
            }

            // Check the entire applicable space to be sure that the points evaluate the same.
            let mut space_min = rects[0].bottom.clone();
            let mut space_max = rects[0].top.clone();
            for r in &rects[1..] {
                for (a, b) in space_min.arr.iter_mut().zip(&r.bottom.arr) {
                    *a = (*a).min(*b);
                }
            }
            for r in &rects[1..] {
                for (a, b) in space_max.arr.iter_mut().zip(&r.top.arr) {
                    *a = (*a).max(*b);
                }
            }
            for v in space_min.arr.iter_mut() {
                *v = (*v).saturating_sub(1);
            }
            for v in space_max.arr.iter_mut() {
                *v = (*v).saturating_add(1);
            }

            // TODO: Check the points evaluate identically to just checking bounds against the rects.
            let pts_iter = space_min
                .arr
                .iter()
                .zip(&space_max.arr)
                .map(|(l, h)| (*l..=*h))
                .multi_cartesian_product();
            for pt in pts_iter {
                let pt = RTreePt {
                    arr: pt.try_into().unwrap(),
                };
                let tree_value = tree.locate_at_point(&pt).map(|r| r.result.clone());
                let mut expected_value = None;
                for r in rects.iter().rev() {
                    if pt
                        .arr
                        .iter()
                        .zip(&r.bottom.arr)
                        .zip(&r.top.arr)
                        .all(|((p, l), h)| *l <= *p && *p <= *h)
                    {
                        expected_value = Some(r.result.clone());
                        break;
                    }
                }
                prop_assert_eq!(tree_value, expected_value, "values differed at {:?}", pt);
            }
        }
    }

    fn arb_disjoint_rects<const D: usize>(
        count: usize,
    ) -> impl Strategy<Value = Vec<RTreeBlockRect<D>>> {
        proptest::collection::vec(arb_rect::<D>(), count).prop_filter(
            "rectangles overlapped",
            |rects| {
                for i in 0..rects.len() {
                    for j in i + 1..rects.len() {
                        if rects[i].envelope().intersects(&rects[j].envelope()) {
                            return false;
                        }
                    }
                }
                true
            },
        )
    }

    fn arb_rect<const D: usize>() -> impl Strategy<Value = RTreeBlockRect<D>> {
        proptest::collection::vec(arb_rect_range::<D>(), D)
            .prop_flat_map(|rngs| {
                let bottom = rngs.iter().map(|(b, _)| *b).collect::<Vec<_>>();
                let top = rngs.iter().map(|(_, t)| *t).collect::<Vec<_>>();
                (
                    Just(RTreePt::<D> {
                        arr: bottom.try_into().unwrap(),
                    }),
                    Just(RTreePt::<D> {
                        arr: top.try_into().unwrap(),
                    }),
                    arb_rect_value(),
                )
            })
            .prop_map(|(bottom, top, result)| RTreeBlockRect {
                bottom,
                top,
                result,
            })
    }

    fn arb_rect_range<const D: usize>() -> impl Strategy<Value = (BimapSInt, BimapSInt)> {
        (0..BimapSInt::from(4)).prop_flat_map(|b| (Just(b), (b..((b + 4).min(6)))))
    }

    fn arb_rect_value() -> impl Strategy<Value = Option<(CostIntensity, MemVec, u8, ActionIdx)>> {
        (0..4u32, 1..4u32)
            .prop_map(|(a, b)| CostIntensity::new(a, DimSize::try_from(b).unwrap()))
            .prop_flat_map(|intensity| {
                proptest::option::of((
                    Just(intensity),
                    Just(MemVec::zero::<X86Target>()),
                    0..2u8,
                    0..2u16,
                ))
            })
    }

    /// Construct an RTree, insert rects with the given points, and assert that they are merged.
    fn assert_merged<const D: usize>(rects: &[([BimapSInt; D], [BimapSInt; D])]) {
        let mut tree = RTree::new();
        for (bottom, top) in rects {
            rtree_merge_insert(
                &mut tree,
                RTreeBlockRect {
                    top: (*top).into(),
                    bottom: (*bottom).into(),
                    result: None,
                },
            );
        }
        let merged_envelope = rects.iter().fold(
            AABB::<RTreePt<D>>::from_corners(rects[0].0.into(), rects[0].1.into()),
            |mut acc, (b, t)| {
                acc.merge(&AABB::from_corners((*b).into(), (*t).into()));
                acc
            },
        );
        assert_eq!(
            tree.iter().map(|r| r.envelope()).collect::<Vec<_>>(),
            &[merged_envelope]
        );
    }
}
