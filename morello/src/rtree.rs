//! Implementations of R*-Tree data structures storing rectangular geometry.

use crate::grid::linear::BimapSInt;
use enum_dispatch::enum_dispatch;
use rstar::Envelope as _;
use rstar::{Point, PointDistance, RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// A trait abstracting over differently ranked RTree<RTreeRect<_, T>> variants.
#[enum_dispatch]
trait RTreeGeneric<T> {
    fn locate_at_point(&self, pt: &[BimapSInt]) -> Option<&T>;

    // TODO: It would be nice to take low and high by value to avoid a clone.
    fn insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T);

    fn merge_insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T)
    where
        T: PartialEq + Eq + Hash + Clone;
}

macro_rules! rtreedyn_cases {
    ($(#[$meta:meta])* $($n:expr, $name:ident),*) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[allow(private_interfaces)]
        pub enum RTreeDyn<T> {
            $( $name(RTree<RTreeRect<$n, T>>), )*
        }

        impl<T> RTreeDyn<T> {
            pub fn empty(rank: usize) -> Self {
                match rank {
                    $( $n => RTreeDyn::$name(Default::default()), )*
                    _ => panic!("Unsupported rank: {}", rank),
                }
            }

            pub fn locate_at_point(&self, pt: &[BimapSInt]) -> Option<&T> {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::locate_at_point(t, pt), )*
                }
            }

            pub fn insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T) {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::insert(t, low, high, value), )*
                }
            }

            pub fn merge_insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T)
            where
                T: PartialEq + Eq + Hash + Clone,
            {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::merge_insert(t, low, high, value), )*
                }
            }
        }
    };
}

rtreedyn_cases!(
    1, D1, 2, D2, 3, D3, 4, D4, 5, D5, 6, D6, 7, D7, 8, D8, 9, D9, 10, D10, 11, D11, 12, D12, 13,
    D13, 14, D14, 15, D15, 16, D16, 17, D17, 18, D18, 19, D19, 20, D20, 21, D21, 22, D22, 23, D23,
    24, D24, 25, D25, 26, D26, 27, D27, 28, D28, 29, D29, 30, D30, 31, D31, 32, D32, 33, D33, 34,
    D34, 35, D35, 36, D36
);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct RTreeRect<const D: usize, T> {
    top: RTreePt<D>,
    bottom: RTreePt<D>,
    value: T,
}

#[serde_as]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct RTreePt<const D: usize> {
    #[serde_as(as = "[_; D]")] // TODO: Use manual serde impls instead of serde_as.
    arr: [BimapSInt; D],
}

/// An [rstar::SelectionFunction] which removes a set of [rstar::RTreeObject]s.
#[derive(Debug)]
struct BatchRemoveSelFn<O: rstar::RTreeObject> {
    // TODO: Own references, not clones.
    to_remove: HashSet<O>,
    envelope: Option<O::Envelope>,
}

impl<T> RTreeDyn<T> {}

impl<const D: usize, T> RTreeGeneric<T> for RTree<RTreeRect<D, T>> {
    fn locate_at_point(&self, pt: &[BimapSInt]) -> Option<&T> {
        self.locate_at_point(&RTreePt {
            arr: pt.try_into().unwrap(),
        })
        .map(|rect| &rect.value)
    }

    fn insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T) {
        let bottom = RTreePt {
            arr: low.try_into().unwrap(),
        };
        let top = RTreePt {
            arr: high.try_into().unwrap(),
        };
        self.insert(RTreeRect { top, bottom, value });
    }

    fn merge_insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T)
    where
        T: PartialEq + Eq + Hash + Clone,
    {
        let new_rect = RTreeRect {
            top: RTreePt {
                arr: high.try_into().unwrap(),
            },
            bottom: RTreePt {
                arr: low.try_into().unwrap(),
            },
            value,
        };

        // Find the first rectangle which matches except for one dimension which is larger or
        // matches (and has the same cost).
        let mut to_remove = BatchRemoveSelFn::<RTreeRect<D, T>>::default();
        let mut to_insert = new_rect;
        let mut should_repeat = true;
        let mut skip_insert = false;

        while should_repeat {
            should_repeat = false;
            let candidate_area = AABB::from_corners(
                to_insert.bottom.arr.map(|b| b.saturating_sub(1)).into(),
                to_insert.top.arr.map(|t| t.saturating_add(1)).into(),
            );
            for candidate in self.locate_in_envelope_intersecting(&candidate_area) {
                // When the inserted rect has matching value and would be fully contained (or is identical),
                // there's nothing to merge. The outer rect. would have already triggered applicable merge
                // rules.
                // TODO: Avoid constructing envelope if possible.
                let candidate_envelope = candidate.envelope();
                let insert_envelope = to_insert.envelope();
                if candidate_envelope.contains_envelope(&insert_envelope) {
                    // Assert the MainCost, but not the later tuple elements, are unchanged.
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
                    continue;
                }

                // If a candidate extrudes the to-be-inserted rect. in exactly one dimension, just grow
                // the to-be-inserted to include it and then remove that rect.
                //
                // TODO: This condition+loop is wasteful.
                if candidate.value == to_insert.value
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
            for _ in self.drain_with_selection_function(&to_remove) {
                remove_count += 1;
            }
            assert_eq!(expected_removals, remove_count);
            to_remove.clear();
        }

        if !skip_insert {
            self.insert(to_insert);
        }
    }
}

impl<const D: usize, T> RTreeObject for RTreeRect<D, T> {
    type Envelope = AABB<RTreePt<D>>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(self.bottom.clone(), self.top.clone())
    }
}

impl<const D: usize, T> PointDistance for RTreeRect<D, T> {
    fn distance_2(&self, point: &RTreePt<D>) -> BimapSInt {
        // TODO: Don't allocate an AABB
        let aabb = AABB::from_corners(self.bottom.clone(), self.top.clone());
        aabb.distance_2(point)
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

fn count_matching_dimensions<const D: usize, T>(
    new_rect: &RTreeRect<D, T>,
    candidate: &RTreeRect<D, T>,
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

fn all_dimensions_adjacent_or_overlap<const D: usize, T>(
    lhs: &RTreeRect<D, T>,
    rhs: &RTreeRect<D, T>,
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
                tree.merge_insert(&r.bottom.arr, &r.top.arr, r.value);
            }

            // Compute the space-to-check. We'll check all points in this region evaluate correctly.
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
                let pt = RTreePt::<3> {
                    arr: pt.try_into().unwrap(),
                };
                let tree_value = tree.locate_at_point(&pt).map(|r| r.value);
                let mut expected_value = None;
                for r in rects.iter().rev() {
                    if pt
                        .arr
                        .iter()
                        .zip(&r.bottom.arr)
                        .zip(&r.top.arr)
                        .all(|((p, l), h)| *l <= *p && *p <= *h)
                    {
                        expected_value = Some(r.value);
                        break;
                    }
                }
                prop_assert_eq!(tree_value, expected_value, "values differed at {:?}", pt);
            }
        }
    }

    fn arb_disjoint_rects<const D: usize>(
        count: usize,
    ) -> impl Strategy<Value = Vec<RTreeRect<D, u8>>> {
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

    fn arb_rect<const D: usize>() -> impl Strategy<Value = RTreeRect<D, u8>> {
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
                    0..8u8,
                )
            })
            .prop_map(|(bottom, top, value)| RTreeRect { bottom, top, value })
    }

    fn arb_rect_range<const D: usize>() -> impl Strategy<Value = (BimapSInt, BimapSInt)> {
        (0..BimapSInt::from(4)).prop_flat_map(|b| (Just(b), (b..((b + 4).min(6)))))
    }

    // fn arb_rect_value() -> impl Strategy<Value = Option<(CostIntensity, MemVec, u8, ActionIdx)>> {
    //     (0..4u32, 1..4u32)
    //         .prop_map(|(a, b)| CostIntensity::new(a, DimSize::try_from(b).unwrap()))
    //         .prop_flat_map(|intensity| {
    //             proptest::option::of((
    //                 Just(intensity),
    //                 Just(MemVec::zero::<X86Target>()),
    //                 0..2u8,
    //                 0..2u16,
    //             ))
    //         })
    // }

    /// Construct an RTree, insert rects with the given points, and assert that they are merged.
    fn assert_merged<const D: usize>(rects: &[([BimapSInt; D], [BimapSInt; D])]) {
        let mut tree = RTree::new();
        for (bottom, top) in rects {
            tree.merge_insert(bottom, top, ());
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
