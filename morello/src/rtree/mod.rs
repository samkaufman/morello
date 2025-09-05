//! Implementations of R*-Tree data structures storing rectangular geometry.

use crate::grid::linear::BimapSInt;
use crate::rtree::aabb::AABB;
use enum_dispatch::enum_dispatch;
use rstar::Envelope as _;
use rstar::{Point, PointDistance, RTree, RTreeObject};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

mod aabb;

pub type RTreeEntryRef<'a, T> = (&'a [BimapSInt], &'a [BimapSInt], &'a T);

/// A private trait abstracting over differently ranked RTree<RTreeRect<_, T>> variants.
/// It's used internally to implement RTreeDyn (each variant dispatches).
#[enum_dispatch]
trait RTreeGeneric<T> {
    type Intersectable<A>;

    fn size(&self) -> usize;

    fn locate_at_point(&self, pt: &[BimapSInt]) -> Option<&T>;

    fn locate_all_at_point(&self, pt: &[BimapSInt]) -> Box<dyn Iterator<Item = &T> + '_>;

    // TODO: It would be nice to take low and high by value to avoid a clone.
    fn insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T);

    fn merge_insert(
        &mut self,
        low: &[BimapSInt],
        high: &[BimapSInt],
        value: T,
        disallow_overlap: bool,
    ) where
        T: PartialEq + Eq + Hash + Clone;

    /// Insert a rectangle into this [RTree], partitioning it and and any contained. Intersecting
    /// sub-rectangles are combined into one with a new value computed by `fold`.
    ///
    /// For example:
    ///
    /// ```ignore
    /// tree.insert(&[0, 0], &[1, 1], 1);
    /// tree.fold_insert(&[1, 1], &[2, 2], 3, false, |a, b| a + b);
    /// assert_eq!(tree.locate_at_point(&[1, 1]), Some(4));
    /// ```
    fn fold_insert<F>(
        &mut self,
        low: &[BimapSInt],
        high: &[BimapSInt],
        value: T,
        merge: bool,
        fold: F,
    ) where
        // TODO: Shouldn't need to be Clone. Instead, give references.
        T: Clone + Eq + Hash, // Eq needed if `merge` is set
        F: FnMut(T, T) -> T;

    fn subtract(
        &mut self,
        low: &[BimapSInt],
        high: &[BimapSInt],
    ) -> Vec<(Vec<BimapSInt>, Vec<BimapSInt>, T)>
    where
        T: Clone;

    fn subtract_tree<V>(
        &mut self,
        subtrahend_tree: &RTreeDyn<V>,
    ) -> Vec<(Vec<BimapSInt>, Vec<BimapSInt>, T, V)>
    where
        T: Clone,
        V: Clone;

    fn intersection_candidates_with_other_tree<'a, A>(
        &'a self,
        other: &'a Self::Intersectable<A>,
    ) -> Box<dyn Iterator<Item = (RTreeEntryRef<'a, T>, RTreeEntryRef<'a, A>)> + 'a>;
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

            pub fn size(&self) -> usize {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::size(t), )*
                }
            }

            pub fn dim_count(&self) -> usize {
                match self {
                    $( RTreeDyn::$name(_) => $n, )*
                }
            }

            pub fn locate_at_point(&self, pt: &[BimapSInt]) -> Option<&T> {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::locate_at_point(t, pt), )*
                }
            }


            pub fn locate_all_at_point(&self, pt: &[BimapSInt]) -> Box<dyn Iterator<Item = &T> + '_> {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::locate_all_at_point(t, pt), )*
                }
            }

            pub fn insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T) {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::insert(t, low, high, value), )*
                }
            }

            pub fn merge_insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T, disallow_overlap: bool)
            where
                T: PartialEq + Eq + Hash + Clone,
            {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::merge_insert(t, low, high, value, disallow_overlap), )*
                }
            }

            pub fn fold_insert<F>(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T, merge: bool, fold: F)
            where
                // TODO: Shouldn't need to be Clone. Instead, give references.
                T: Clone + Eq + Hash,  // Eq needed if `merge` is set
                F: FnMut(T, T) -> T,
            {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::fold_insert(t, low, high, value, merge, fold), )*
                }
            }

            pub fn subtract(
                &mut self, low: &[BimapSInt], high: &[BimapSInt]
            ) -> Vec<(Vec<BimapSInt>, Vec<BimapSInt>, T)>
            where
                T: Clone, // TODO: Remove this bound. Shouldn't be needed.
            {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::subtract(t, low, high), )*
                }
            }

            /// Update by subtracting the space filled by another [RTreeGeneric].
            /// That tree's values are ignored.
            ///
            /// Note: The current implementation is very slow.
            pub fn subtract_tree<V>(
                &mut self, subtrahend_tree: &RTreeDyn<V>
            ) -> Vec<(Vec<BimapSInt>, Vec<BimapSInt>, T, V)>
            where
                T: Clone,
                V: Clone,
            {
                match self {
                    $( RTreeDyn::$name(t) => RTreeGeneric::subtract_tree(t, subtrahend_tree), )*
                }
            }

            pub fn iter(&self) -> Box<dyn Iterator<Item = (&[BimapSInt], &[BimapSInt], &T)> + '_> {
                match self {
                    $( RTreeDyn::$name(t) => Box::new(
                        t.iter().map(|r| (&r.bottom.arr[..], &r.top.arr[..], &r.value))
                    ), )*
                }
            }

            pub fn locate_in_envelope_intersecting(
                &self,
                bottom: &[BimapSInt],
                top: &[BimapSInt],
            ) -> Box<dyn Iterator<Item = (&[BimapSInt], &[BimapSInt], &T)> + '_> {
                match self {
                    $( RTreeDyn::$name(t) => {
                        Box::new(t.locate_in_envelope_intersecting(&AABB::from_corners(
                            bottom.try_into().unwrap(),
                            top.try_into().unwrap(),
                        )).map(|r| (&r.bottom.arr[..], &r.top.arr[..], &r.value)))
                    } )*
                }
            }

            pub fn intersection_candidates_with_other_tree<'a, A>(
                &'a self,
                other: &'a RTreeDyn<A>,
            ) -> Box<dyn Iterator<Item = (RTreeEntryRef<'a, T>, RTreeEntryRef<'a, A>)> + 'a> {
                match (self, other) {
                    $( (RTreeDyn::$name(t), RTreeDyn::$name(o)) => {
                        RTreeGeneric::intersection_candidates_with_other_tree(t, o)
                    }, )*
                    _ => panic!("Mismatched ranks: {} and {}", self.dim_count(), other.dim_count()),
                }
            }
        }
    };
}

rtreedyn_cases!(
    2, D2, 3, D3, 4, D4, 5, D5, 6, D6, 7, D7, 8, D8, 9, D9, 10, D10, 11, D11, 12, D12, 13, D13, 14,
    D14, 15, D15, 16, D16, 17, D17, 18, D18, 19, D19, 20, D20, 21, D21, 22, D22, 23, D23, 24, D24,
    25, D25, 26, D26, 27, D27, 28, D28, 29, D29, 30, D30, 31, D31, 32, D32, 33, D33, 34, D34, 35,
    D35, 36, D36, 37, D37, 38, D38, 39, D39, 40, D40, 41, D41, 42, D42, 43, D43, 44, D44, 45, D45,
    46, D46, 47, D47, 48, D48, 49, D49, 50, D50, 51, D51, 52, D52, 53, D53, 54, D54
);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct RTreeRect<const D: usize, T> {
    top: RTreePt<D>,
    bottom: RTreePt<D>,
    value: T,
}

#[serde_as]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RTreePt<const D: usize> {
    #[serde_as(as = "[_; D]")] // TODO: Use manual serde impls instead of serde_as.
    arr: [BimapSInt; D],
}

struct RectPartitionIntersection {
    lhs_subrectangles: Vec<(Vec<BimapSInt>, Vec<BimapSInt>)>,
    rhs_subrectangles: Vec<(Vec<BimapSInt>, Vec<BimapSInt>)>,
    intersection: (Vec<BimapSInt>, Vec<BimapSInt>),
}

/// An [rstar::SelectionFunction] which removes a set of [rstar::RTreeObject]s.
#[derive(Debug)]
struct BatchRemoveSelFn<O: rstar::RTreeObject> {
    // TODO: Own references, not clones.
    to_remove: HashSet<O>,
    envelope: Option<O::Envelope>,
}

struct SameValueIntersectionSelFn<'a, const D: usize, T> {
    envelope: <RTreeRect<D, T> as RTreeObject>::Envelope,
    value: &'a T,
}

struct ContainedAndAdjacentSelFn<const D: usize, T> {
    to_insert: RTreeRect<D, T>,
    relevant_area: <RTreeRect<D, T> as RTreeObject>::Envelope,
}

impl<'a, T> IntoIterator for &'a RTreeDyn<T> {
    type Item = (&'a [BimapSInt], &'a [BimapSInt], &'a T);
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<const D: usize, T> RTreeGeneric<T> for RTree<RTreeRect<D, T>> {
    type Intersectable<A> = RTree<RTreeRect<D, A>>;

    fn size(&self) -> usize {
        self.size()
    }

    fn locate_at_point(&self, pt: &[BimapSInt]) -> Option<&T> {
        self.locate_at_point(&pt.try_into().unwrap())
            .map(|rect| &rect.value)
    }

    fn locate_all_at_point(&self, pt: &[BimapSInt]) -> Box<dyn Iterator<Item = &T> + '_> {
        Box::new(
            self.locate_all_at_point(&pt.try_into().unwrap())
                .map(|rect| &rect.value),
        )
    }

    fn insert(&mut self, low: &[BimapSInt], high: &[BimapSInt], value: T) {
        let bottom = low.try_into().unwrap();
        let top = high.try_into().unwrap();
        self.insert(RTreeRect { top, bottom, value });
    }

    fn merge_insert(
        &mut self,
        low: &[BimapSInt],
        high: &[BimapSInt],
        value: T,
        disallow_overlap: bool,
    ) where
        T: PartialEq + Eq + Hash + Clone,
    {
        let new_rect = RTreeRect {
            top: high.try_into().unwrap(),
            bottom: low.try_into().unwrap(),
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
                let value_matches = candidate.value == to_insert.value;

                // When the inserted rect has matching value and would be fully contained (or is identical),
                // there's nothing to merge. The outer rect. would have already triggered applicable merge
                // rules.
                // TODO: Avoid constructing envelope if possible.
                let candidate_envelope = candidate.envelope();
                let insert_envelope = to_insert.envelope();
                if candidate_envelope.contains_envelope(&insert_envelope) && value_matches {
                    // Assert the MainCost, but not the later tuple elements, are unchanged.
                    should_repeat = false;
                    skip_insert = true;
                    break;
                }

                // If the candidate is contained by the to-be-inserted rectangle and has a matching
                // value, remove it.
                if insert_envelope.contains_envelope(&candidate_envelope) && value_matches {
                    // TODO: Avoid the following clone.
                    to_remove.queue_removal(candidate.clone());
                    // Assert the MainCost, but not the later tuple elements, are unchanged.
                    // TODO: Remove the following assert and lift short-circuit.
                    continue;
                }

                // If a candidate extrudes the to-be-inserted rect. in exactly one dimension and the
                // value matches, just grow the to-be-inserted to include it and then remove that
                // rect.
                //
                // TODO: This condition+loop is wasteful.
                if value_matches
                    && all_dimensions_adjacent_or_overlap(
                        &to_insert.bottom.arr,
                        &to_insert.top.arr,
                        &candidate.bottom.arr,
                        &candidate.top.arr,
                    )
                    && count_matching_dimensions(
                        &to_insert.bottom.arr,
                        &to_insert.top.arr,
                        &candidate.bottom.arr,
                        &candidate.top.arr,
                    ) == D - 1
                {
                    should_repeat = true;

                    let mut merged_envelope = to_insert.envelope();
                    merged_envelope.merge(&candidate_envelope);
                    to_insert.top = merged_envelope.upper().clone();
                    to_insert.bottom = merged_envelope.lower().clone();

                    // If we expand insert to merge, we'll need to start over with a new
                    // ContainedAndAdjacentSelFn which has an updated to_insert.
                    // TODO: Starting over for every adjacent/mergeble rect has bad asymptotics!
                    break;
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
            insert_and_subtract_overlap(self, to_insert);
        }
    }

    fn fold_insert<F>(
        &mut self,
        low: &[BimapSInt],
        high: &[BimapSInt],
        value: T,
        merge: bool,
        mut fold: F,
    ) where
        T: Clone + Eq + Hash,
        F: FnMut(T, T) -> T,
    {
        let mut worklist = vec![(low.to_vec(), high.to_vec(), value)];
        while let Some((r_low, r_high, r_value)) = worklist.pop() {
            let mut rhs_subrects = vec![];
            let intersectee_option = self
                .drain_in_envelope_intersecting(AABB::from_corners(
                    r_low.clone().try_into().unwrap(),
                    r_high.clone().try_into().unwrap(),
                ))
                .take(1)
                .next();
            if let Some(intersectee) = intersectee_option {
                let result = rect_partition_intersection(
                    &r_low,
                    &r_high,
                    &intersectee.bottom.arr,
                    &intersectee.top.arr,
                )
                .unwrap();
                rhs_subrects.extend(result.rhs_subrectangles.into_iter().map(|(bottom, top)| {
                    RTreeRect {
                        bottom: bottom.try_into().unwrap(),
                        top: top.try_into().unwrap(),
                        value: intersectee.value.clone(),
                    }
                }));
                worklist.extend(
                    result
                        .lhs_subrectangles
                        .into_iter()
                        .map(|(bottom, top)| (bottom, top, r_value.clone())),
                );
                worklist.push((
                    result.intersection.0,
                    result.intersection.1,
                    fold(r_value.clone(), intersectee.value),
                ));
            } else if merge {
                self.merge_insert(&r_low, &r_high, r_value, true);
            } else {
                self.insert(RTreeRect {
                    bottom: r_low.try_into().unwrap(),
                    top: r_high.try_into().unwrap(),
                    value: r_value,
                });
            }

            if merge {
                rhs_subrects.into_iter().for_each(|r| {
                    let RTreeRect { top, bottom, value } = r;
                    self.merge_insert(&bottom.arr, &top.arr, value, true)
                })
            } else {
                rhs_subrects.into_iter().for_each(|r| self.insert(r));
            }
        }
    }

    fn subtract(
        &mut self,
        low: &[BimapSInt],
        high: &[BimapSInt],
    ) -> Vec<(Vec<BimapSInt>, Vec<BimapSInt>, T)>
    where
        T: Clone,
    {
        assert_eq!(low.len(), high.len());
        let mut subtrahend_tree = RTreeDyn::empty(low.len());
        subtrahend_tree.insert(low, high, ());
        self.subtract_tree(&subtrahend_tree)
            .into_iter()
            .map(|(b, t, v, _)| (b, t, v))
            .collect()
    }

    fn subtract_tree<V>(
        &mut self,
        subtrahend_tree: &RTreeDyn<V>,
    ) -> Vec<(Vec<BimapSInt>, Vec<BimapSInt>, T, V)>
    where
        T: Clone,
        V: Clone,
    {
        let mut subtracted_partitions = vec![];
        let mut new_subrectangles = vec![];
        for (rhs_bottom, rhs_top, rhs_value) in subtrahend_tree.iter() {
            debug_assert!(new_subrectangles.is_empty());
            let rhs_envelope =
                AABB::from_corners(rhs_bottom.try_into().unwrap(), rhs_top.try_into().unwrap());
            for intersecting_rect in self.drain_in_envelope_intersecting(rhs_envelope) {
                let RTreeRect { bottom, top, value } = intersecting_rect;
                let subrectangles = rect_subtract(&bottom.arr, &top.arr, rhs_bottom, rhs_top);
                new_subrectangles.extend(
                    subrectangles
                        .into_iter()
                        .map(|(bottom, top)| (bottom, top, value.clone())),
                );
                let intersection_bottom = bottom
                    .arr
                    .iter()
                    .zip(rhs_bottom)
                    .map(|(b, rb)| *b.max(rb))
                    .collect::<Vec<_>>();
                let intersection_top = top
                    .arr
                    .iter()
                    .zip(rhs_top)
                    .map(|(t, rt)| *t.min(rt))
                    .collect::<Vec<_>>();
                subtracted_partitions.push((
                    intersection_bottom,
                    intersection_top,
                    value,
                    rhs_value.clone(),
                ));
            }
            // TODO: Is there really no bulk insert?
            for (bottom, top, value) in new_subrectangles.drain(..) {
                self.insert(RTreeRect {
                    top: top.try_into().unwrap(),
                    bottom: bottom.try_into().unwrap(),
                    value,
                });
            }
        }
        subtracted_partitions
    }

    fn intersection_candidates_with_other_tree<'a, A>(
        &'a self,
        other: &'a Self::Intersectable<A>,
    ) -> Box<dyn Iterator<Item = (RTreeEntryRef<'a, T>, RTreeEntryRef<'a, A>)> + 'a> {
        Box::new(
            self.intersection_candidates_with_other_tree(other)
                .map(|(a, b)| {
                    (
                        (&a.bottom.arr[..], &a.top.arr[..], &a.value),
                        (&b.bottom.arr[..], &b.top.arr[..], &b.value),
                    )
                }),
        )
    }
}

impl<const D: usize, T> RTreeObject for RTreeRect<D, T> {
    type Envelope = AABB<D>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(self.bottom.clone(), self.top.clone())
    }
}

impl<const D: usize, T> PointDistance for RTreeRect<D, T> {
    fn distance_2(&self, point: &RTreePt<D>) -> BimapSInt {
        // TODO: Don't allocate an AABB/clone.
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

impl<const D: usize> TryFrom<Vec<BimapSInt>> for RTreePt<D> {
    type Error = <[BimapSInt; D] as TryFrom<Vec<BimapSInt>>>::Error;

    fn try_from(value: Vec<BimapSInt>) -> Result<Self, Self::Error> {
        Ok(RTreePt {
            arr: value.try_into()?,
        })
    }
}

impl<'a, const D: usize> TryFrom<&'a [BimapSInt]> for RTreePt<D> {
    type Error = <[BimapSInt; D] as TryFrom<&'a [BimapSInt]>>::Error;

    fn try_from(value: &'a [BimapSInt]) -> Result<Self, Self::Error> {
        Ok(RTreePt {
            arr: value.try_into()?,
        })
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

impl<const D: usize, T> rstar::SelectionFunction<RTreeRect<D, T>>
    for ContainedAndAdjacentSelFn<D, T>
where
    T: PartialEq + Eq,
{
    fn should_unpack_parent(&self, envelope: &<RTreeRect<D, T> as RTreeObject>::Envelope) -> bool {
        self.relevant_area.intersects(envelope)
    }

    fn should_unpack_leaf(&self, candidate: &RTreeRect<D, T>) -> bool {
        if candidate.value == self.to_insert.value {
            let to_insert_envelope = self.to_insert.envelope();
            if to_insert_envelope.contains_envelope(&candidate.envelope()) {
                return true;
            }
            if all_dimensions_adjacent_or_overlap(
                &self.to_insert.bottom.arr,
                &self.to_insert.top.arr,
                &candidate.bottom.arr,
                &candidate.top.arr,
            ) && count_matching_dimensions(
                &self.to_insert.bottom.arr,
                &self.to_insert.top.arr,
                &candidate.bottom.arr,
                &candidate.top.arr,
            ) == D - 1
            {
                return true;
            }
        }
        false
    }
}

impl<'a, const D: usize, T> rstar::SelectionFunction<RTreeRect<D, T>>
    for SameValueIntersectionSelFn<'a, D, T>
where
    T: Eq,
{
    fn should_unpack_parent(&self, envelope: &<RTreeRect<D, T> as RTreeObject>::Envelope) -> bool {
        self.envelope.intersects(envelope)
    }

    fn should_unpack_leaf(&self, leaf: &RTreeRect<D, T>) -> bool {
        leaf.value == *self.value && self.envelope.intersects(&leaf.envelope())
    }
}

/// Insert a new rectangle, partitioning any intersecting rectangles with the same value and
/// removing the overlapping parts.
fn insert_and_subtract_overlap<const D: usize, T>(
    tree: &mut RTree<RTreeRect<D, T>>,
    to_insert: RTreeRect<D, T>,
) where
    T: Clone + Eq,
{
    let selection_fn = SameValueIntersectionSelFn {
        envelope: to_insert.envelope(),
        value: &to_insert.value,
    };
    let mut parts_to_insert = vec![];
    for removed in tree.drain_with_selection_function(selection_fn) {
        // For every intersection removed, call rect_subtract to compute non-intersecting
        // partitions.
        for (bottom, top) in rect_subtract(
            &removed.bottom.arr,
            &removed.top.arr,
            &to_insert.bottom.arr,
            &to_insert.top.arr,
        ) {
            parts_to_insert.push(RTreeRect {
                bottom: bottom.try_into().unwrap(),
                top: top.try_into().unwrap(),
                value: removed.value.clone(),
            });
        }
    }
    for part in parts_to_insert {
        tree.insert(part);
    }
    tree.insert(to_insert);
}

/// Subtract the subtrahend rectangle from the minuend rectangle (both defined by inclusive points),
/// returning replacement rectangles which cover the same space as the minuend but exclude the
/// subtrahend.
fn rect_subtract(
    rect_bottom: &[i64],
    rect_top: &[i64],
    subtrahend_bottom: &[i64],
    subtrahend_top: &[i64],
) -> Vec<(Vec<i64>, Vec<i64>)> {
    assert_eq!(rect_bottom.len(), rect_top.len());
    assert_eq!(rect_bottom.len(), subtrahend_bottom.len());
    assert_eq!(rect_bottom.len(), subtrahend_top.len());

    let mut working_bottom = rect_bottom.to_vec();
    let mut working_top = rect_top.to_vec();
    for dim in 0..rect_bottom.len() {
        if working_bottom[dim] > subtrahend_top[dim] || working_top[dim] < subtrahend_bottom[dim] {
            return vec![(working_bottom, working_top)];
        }
    }

    let mut result = vec![];
    for dim in 0..rect_bottom.len() {
        if working_bottom[dim] < subtrahend_bottom[dim] {
            let orig = working_top[dim];
            working_top[dim] = subtrahend_bottom[dim] - 1;
            result.push((working_bottom.clone(), working_top.clone()));
            working_top[dim] = orig;
        }
        if working_top[dim] > subtrahend_top[dim] {
            let orig = working_bottom[dim];
            working_bottom[dim] = subtrahend_top[dim] + 1;
            result.push((working_bottom.clone(), working_top.clone()));
            working_bottom[dim] = orig;
        }
        working_bottom[dim] = working_bottom[dim].max(subtrahend_bottom[dim]);
        working_top[dim] = working_top[dim].min(subtrahend_top[dim]);
    }
    result
}

/// Compute the intersection of two rectangles and subtract the intersection from each rectangle.
/// The result is a [RectPartitionIntersection] which contains the intersection and the
/// subrectangles/tiles which cover the remainders of each rectangle. If the two rectangles do not
/// intersect, the result will be `None`.
fn rect_partition_intersection(
    lhs_bottom: &[i64],
    lhs_top: &[i64],
    rhs_bottom: &[i64],
    rhs_top: &[i64],
) -> Option<RectPartitionIntersection> {
    assert_eq!(lhs_bottom.len(), lhs_top.len());
    assert_eq!(rhs_bottom.len(), rhs_top.len());
    assert_eq!(lhs_bottom.len(), rhs_bottom.len());

    let mut intersection_bottom = vec![];
    let mut intersection_top = vec![];
    for dim in 0..lhs_bottom.len() {
        let low = lhs_bottom[dim].max(rhs_bottom[dim]);
        let high = lhs_top[dim].min(rhs_top[dim]);
        if low > high {
            return None;
        }
        intersection_bottom.push(low);
        intersection_top.push(high);
    }

    let lhs_subrectangles =
        rect_subtract(lhs_bottom, lhs_top, &intersection_bottom, &intersection_top);
    let rhs_subrectangles =
        rect_subtract(rhs_bottom, rhs_top, &intersection_bottom, &intersection_top);

    Some(RectPartitionIntersection {
        lhs_subrectangles,
        rhs_subrectangles,
        intersection: (intersection_bottom, intersection_top),
    })
}

/// Returns number of matching dimensions between two rectangles, or `None` if they are not all
/// adjacent or overlap.
fn count_merging_dimension(
    new_rect_bottom: &[BimapSInt],
    new_rect_top: &[BimapSInt],
    candidate_bottom: &[BimapSInt],
    candidate_top: &[BimapSInt],
) -> Option<usize> {
    if !all_dimensions_adjacent_or_overlap(
        new_rect_bottom,
        new_rect_top,
        candidate_bottom,
        candidate_top,
    ) {
        None
    } else {
        Some(count_matching_dimensions(
            new_rect_bottom,
            new_rect_top,
            candidate_bottom,
            candidate_top,
        ))
    }
}

fn count_matching_dimensions(
    new_rect_bottom: &[BimapSInt],
    new_rect_top: &[BimapSInt],
    candidate_bottom: &[BimapSInt],
    candidate_top: &[BimapSInt],
) -> usize {
    let mut matching_dimensions = 0usize;
    for dim in 0..new_rect_bottom.len() {
        if new_rect_bottom[dim] == candidate_bottom[dim] && new_rect_top[dim] == candidate_top[dim]
        {
            matching_dimensions += 1;
        }
    }
    matching_dimensions
}

fn all_dimensions_adjacent_or_overlap(
    lhs_bottom: &[BimapSInt],
    lhs_top: &[BimapSInt],
    rhs_bottom: &[BimapSInt],
    rhs_top: &[BimapSInt],
) -> bool {
    for dim in 0..lhs_bottom.len() {
        let lhs_bottom_val = lhs_bottom[dim];
        let lhs_top_val = lhs_top[dim];
        let rhs_bottom_val = rhs_bottom[dim];
        let rhs_top_val = rhs_top[dim];
        if (lhs_bottom_val.saturating_sub(1)..=lhs_top_val).contains(&rhs_top_val)
            || (lhs_bottom_val..=lhs_top_val.saturating_add(1)).contains(&rhs_bottom_val)
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
    use crate::utils::diagonals;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest::strategy::{Just, Strategy};
    use proptest::{prop_assert, prop_assert_eq, proptest};
    use std::rc::Rc;

    #[derive(proptest_derive::Arbitrary, Debug)]
    enum InsertMode {
        Merge,
        Fold,
        FoldMerging,
    }

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
    fn test_merge_insert_one_wide_gap() {
        assert_merged(&[([0, 0], [1, 2]), ([0, 4], [1, 6]), ([0, 3], [1, 3])]);
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

    #[test]
    fn test_merge_insert_merges_values_strict_intersect() {
        let mut tree = RTree::<RTreeRect<2, _>>::new();
        tree.merge_insert(&[0, 1], &[0, 2], "a", false);
        tree.merge_insert(&[0, 1], &[0, 2], "b", false);
        assert_eq!(
            tree.iter().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([
                RTreeRect {
                    top: RTreePt { arr: [0, 2] },
                    bottom: RTreePt { arr: [0, 1] },
                    value: "a",
                },
                RTreeRect {
                    top: RTreePt { arr: [0, 2] },
                    bottom: RTreePt { arr: [0, 1] },
                    value: "b",
                },
            ])
        );
    }

    #[test]
    fn test_merge_insert_merges_values_corners_intersect() {
        let mut tree = RTree::<RTreeRect<2, _>>::new();
        tree.merge_insert(&[0, 1], &[0, 2], "a", false);
        tree.merge_insert(&[0, 2], &[0, 3], "b", false);
        assert_eq!(
            tree.iter().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([
                RTreeRect {
                    top: RTreePt { arr: [0, 2] },
                    bottom: RTreePt { arr: [0, 1] },
                    value: "a",
                },
                RTreeRect {
                    top: RTreePt { arr: [0, 3] },
                    bottom: RTreePt { arr: [0, 2] },
                    value: "b",
                },
            ])
        );
    }

    // TODO: Re-enable the below test when `merge_insert` panics instead of warns again.
    // #[test]
    // #[should_panic(expected = "new rectangle overlaps")]
    // fn test_merge_insert_disallow_overlap_different_values() {
    //     let mut tree = RTree::<RTreeRect<2, _>>::new();
    //     tree.merge_insert(&[0, 0], &[2, 2], "a", false);
    //     tree.merge_insert(&[1, 1], &[3, 3], "b", true); // This should panic
    // }

    #[test]
    fn test_merge_insert_disallow_overlap_same_values() {
        let mut tree = RTree::<RTreeRect<2, _>>::new();
        tree.merge_insert(&[0, 0], &[2, 2], "a", false);
        tree.merge_insert(&[1, 1], &[3, 3], "a", true); // same value
    }

    #[test]
    fn test_merge_insert_disallow_overlap_no_overlap_1() {
        let mut tree = RTree::<RTreeRect<2, _>>::new();
        tree.merge_insert(&[0, 0], &[1, 1], "a", false);
        tree.merge_insert(&[2, 2], &[3, 3], "b", true); // no overlap
    }

    #[test]
    fn test_merge_insert_disallow_overlap_no_overlap_2() {
        let mut tree = RTree::<RTreeRect<2, _>>::new();
        tree.merge_insert(&[0, 0], &[1, 1], "a", false);
        tree.merge_insert(&[2, 0], &[3, 1], "b", true); // no overlap
    }

    #[test]
    fn test_rect_subtract_1() {
        assert_eq!(rect_subtract(&[0, 0], &[1, 1], &[0, 0], &[1, 1]), []);
    }

    #[test]
    fn test_rect_subtract_2() {
        assert_eq!(
            rect_subtract(&[0, 0], &[2, 2], &[0, 0], &[1, 1]),
            [(vec![2, 0], vec![2, 2]), (vec![0, 2], vec![1, 2])]
        );
    }

    #[test]
    fn test_rect_subtract_3() {
        assert_eq!(
            rect_subtract(&[0, 0], &[2, 2], &[1, 1], &[1, 1]),
            [
                (vec![0, 0], vec![0, 2]),
                (vec![2, 0], vec![2, 2]),
                (vec![1, 0], vec![1, 0]),
                (vec![1, 2], vec![1, 2]),
            ]
        );
    }

    #[test]
    fn test_rect_subtract_4() {
        assert_eq!(
            rect_subtract(&[0, 0], &[1, 1], &[0, 1], &[2, 1]),
            [(vec![0, 0], vec![1, 0])]
        );
    }

    #[test]
    fn test_rtree_subtract_1() {
        let mut minuhend: RTree<RTreeRect<2, ()>> = RTree::new();
        minuhend.merge_insert(&[1, 1], &[3, 3], (), true);

        let mut subtrahend: RTree<RTreeRect<2, ()>> = RTree::new();
        subtrahend.merge_insert(&[1, 1], &[2, 2], (), true);
        subtrahend.merge_insert(&[1, 1], &[2, 2], (), true);

        let intersections = minuhend.subtract_tree(&RTreeDyn::D2(subtrahend));
        assert_eq!(
            minuhend.into_iter().collect::<HashSet<_>>(),
            HashSet::from([
                RTreeRect {
                    bottom: [1, 3].into(),
                    top: [2, 3].into(),
                    value: ()
                },
                RTreeRect {
                    bottom: [3, 1].into(),
                    top: [3, 3].into(),
                    value: ()
                },
            ])
        );
        assert_eq!(intersections, &[(vec![1, 1], vec![2, 2], (), ())]);
    }

    #[test]
    fn test_rtree_subtract_2() {
        let mut minuhend: RTree<RTreeRect<2, _>> = RTree::new();
        minuhend.merge_insert(&[1, 1], &[3, 3], "a", true);
        minuhend.merge_insert(&[5, 1], &[5, 7], "b", true);

        let mut subtrahend: RTree<RTreeRect<2, _>> = RTree::new();
        subtrahend.merge_insert(&[1, 1], &[9, 9], "c", true);

        let intersections = minuhend.subtract_tree(&RTreeDyn::D2(subtrahend));
        assert_eq!(minuhend.into_iter().count(), 0);
        assert_eq!(
            intersections,
            &[
                (vec![1, 1], vec![3, 3], "a", "c"),
                (vec![5, 1], vec![5, 7], "b", "c"),
            ]
        );
    }

    #[test]
    fn test_rtree_subtract_3() {
        let mut minuhend: RTree<RTreeRect<2, _>> = RTree::new();
        minuhend.merge_insert(&[1, 1], &[3, 7], "a", true);
        minuhend.merge_insert(&[5, 1], &[7, 7], "b", true);

        let mut subtrahend: RTree<RTreeRect<2, ()>> = RTree::new();
        subtrahend.merge_insert(&[1, 2], &[7, 3], (), true);

        let intersections = minuhend.subtract_tree(&RTreeDyn::D2(subtrahend));
        assert_eq!(
            minuhend.into_iter().collect::<HashSet<_>>(),
            HashSet::from([
                RTreeRect {
                    bottom: [1, 1].into(),
                    top: [3, 1].into(),
                    value: "a"
                },
                RTreeRect {
                    bottom: [5, 1].into(),
                    top: [7, 1].into(),
                    value: "b"
                },
                RTreeRect {
                    bottom: [1, 4].into(),
                    top: [3, 7].into(),
                    value: "a"
                },
                RTreeRect {
                    bottom: [5, 4].into(),
                    top: [7, 7].into(),
                    value: "b"
                },
            ])
        );
        assert_eq!(
            intersections,
            &[
                (vec![1, 2], vec![3, 3], "a", ()),
                (vec![5, 2], vec![7, 3], "b", ()),
            ]
        );
    }

    // Distilled from the trace of an old bug where merge_inset performed "simultaneous"
    // merges with multiple adjacent rectangles, not carrying the updated, merged shape
    // from the first merge.
    #[test]
    fn test_no_overlap_1() {
        let mut tree = RTree::<RTreeRect<2, _>>::new();

        tree.insert(RTreeRect {
            bottom: [2, 3].into(),
            top: [2, 3].into(),
            value: 1,
        });
        tree.insert(RTreeRect {
            bottom: [2, 2].into(),
            top: [2, 2].into(),
            value: 4,
        });
        tree.insert(RTreeRect {
            bottom: [3, 2].into(),
            top: [3, 2].into(),
            value: 1,
        });
        tree.merge_insert(&[3, 3], &[3, 3], 1, true);

        let tree_rects = tree.iter().collect::<Vec<_>>();
        for (i, r) in tree_rects.iter().enumerate() {
            for r2 in &tree_rects[..i] {
                assert!(
                    !r.envelope().intersects(&r2.envelope()),
                    "Found intersection in R*-Tree: [{:?}]",
                    tree.iter()
                        .map(|r| format!("({:?}, {:?}, {:?})", r.bottom.arr, r.top.arr, r.value))
                        .join(", ")
                );
            }
        }
    }

    #[test]
    fn test_fold_insert_no_intersection() {
        let mut tree = RTreeDyn::empty(2);
        tree.insert(&[0, 0], &[1, 1], 1);
        tree.fold_insert(&[2, 2], &[3, 3], 2, false, |a, b| a + b);
        assert_eq!(tree.locate_at_point(&[0, 0]), Some(&1));
        assert_eq!(tree.locate_at_point(&[2, 2]), Some(&2));
    }

    #[test]
    fn test_fold_insert_with_intersection() {
        let mut tree = RTreeDyn::empty(2);
        tree.insert(&[0, 0], &[2, 2], 1);
        tree.fold_insert(&[1, 1], &[3, 3], 2, false, |a, b| a + b);
        assert_eq!(tree.locate_at_point(&[0, 0]), Some(&1));
        assert_eq!(tree.locate_at_point(&[1, 1]), Some(&3));
        assert_eq!(tree.locate_at_point(&[3, 3]), Some(&2));
    }

    #[test]
    fn test_fold_insert_multiple_intersections() {
        let mut tree = RTreeDyn::empty(2);
        tree.insert(&[0, 0], &[2, 2], 1);
        println!("after first insert");
        tree.fold_insert(&[1, 1], &[3, 3], 2, false, |a, b| a + b);
        println!("after first fold_insert");
        for entry in tree.iter() {
            println!("{:?}", entry);
        }
        tree.fold_insert(&[2, 2], &[4, 4], 3, false, |a, b| a + b);
        println!("after second fold_insert");
        assert_eq!(tree.locate_at_point(&[0, 0]), Some(&1));
        assert_eq!(tree.locate_at_point(&[1, 1]), Some(&3));
        assert_eq!(tree.locate_at_point(&[2, 2]), Some(&6));
        assert_eq!(tree.locate_at_point(&[4, 4]), Some(&3));
    }

    proptest! {
        #[test]
        fn test_disjoint_inserts_dont_change_point_results(rects in arb_disjoint_rects::<3>(3)) {
            let mut tree = RTree::new();
            for r in &rects {
                tree.merge_insert(&r.bottom.arr, &r.top.arr, r.value, true);
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

        #[test]
        fn test_rect_subtract_results_are_disjoint((rect, subtrahend) in arb_rect_subtract()) {
            let (rect_low, rect_high) = rect;
            let (sub_low, sub_high) = subtrahend;
            let rects = rect_subtract(&rect_low, &rect_high, &sub_low, &sub_high);
            for a in 0..rects.len() {
                for b in a + 1..rects.len() {
                    let (a_low, a_high) = (&rects[a].0, &rects[a].1);
                    let (b_low, b_high) = (&rects[b].0, &rects[b].1);
                    prop_assert!(
                        (0..a_low.len()).any(|dim| {
                            a_low[dim] > b_high[dim] || a_high[dim] < b_low[dim]
                        }),
                        "rects {:?} and {:?} intersected",
                        rects[a],
                        rects[b]
                    );
                }
            }
        }

        /// Test that every point on the union of `rect_subtract` results intersects correctly.
        #[test]
        fn test_rect_subtract((rect, subtrahend) in arb_rect_subtract()) {
            let (rect_low, rect_high) = rect;
            let (sub_low, sub_high) = subtrahend;

            let bound_low = rect_low.iter().zip(&sub_low).map(|(r, s)| r.min(s)).copied().collect::<Vec<_>>();
            let bound_high = rect_high.iter().zip(&sub_high).map(|(r, s)| r.max(s)).copied().collect::<Vec<_>>();

            let bound_pts = diagonals(&bound_high).flatten().map(|mut unshifted_pt| {
                for dim in 0..unshifted_pt.len() {
                    unshifted_pt[dim] += bound_low[dim];
                }
                unshifted_pt
            });
            for bound_pt in bound_pts {
                let result_match =
                    rect_subtract(&rect_low, &rect_high, &sub_low, &sub_high).iter().any(|rect| {
                        (0..bound_pt.len()).all(|dim| {
                            bound_pt[dim] >= rect.0[dim] && bound_pt[dim] <= rect.1[dim]
                        })
                    });
                let expected = {
                    let mut pt_in_rect = true;
                    let mut pt_in_sub = true;
                    for dim in 0..bound_pt.len() {
                        if bound_pt[dim] < rect_low[dim] || bound_pt[dim] > rect_high[dim] {
                            pt_in_rect = false;
                            break;
                        }
                    }
                    for dim in 0..bound_pt.len() {
                        if bound_pt[dim] < sub_low[dim] || bound_pt[dim] > sub_high[dim] {
                            pt_in_sub = false;
                            break;
                        }
                    }
                    pt_in_rect && !pt_in_sub
                };
                prop_assert_eq!(result_match, expected,
                    "point {:?} was {:?} but expected {:?} (rects were: {:?})",
                    bound_pt, result_match, expected,
                    rect_subtract(&rect_low, &rect_high, &sub_low, &sub_high)
                );
            }
        }

        #[test]
        fn test_rtree_never_contains_overlaps(
            rects in proptest::collection::vec(arb_rect::<3>(), 2..=3),
            insert_mode in any::<InsertMode>(),
        ) {
            let mut tree = RTree::<RTreeRect<3, Rc<u8>>>::new();
            for r in &rects {
                // Box values so that we can later compare memory addresses to determine
                // whether a rects are the same.
                match insert_mode {
                    InsertMode::Merge => tree.merge_insert(&r.bottom.arr, &r.top.arr, Rc::new(r.value), true),
                    InsertMode::Fold => tree.fold_insert(&r.bottom.arr, &r.top.arr, Rc::new(r.value), false, |a, _| a),
                    InsertMode::FoldMerging => tree.fold_insert(&r.bottom.arr, &r.top.arr, Rc::new(r.value), true, |a, _| a),
                };
            }

            for r in tree.iter() {
                let envelope = AABB::from_corners(r.bottom.clone(), r.top.clone());
                let intersecting = tree.locate_in_envelope_intersecting(&envelope);
                for other in intersecting {
                    if Rc::ptr_eq(&other.value, &r.value) {
                        continue;
                    }
                    prop_assert!(
                        r.value != other.value || !r.envelope().intersects(&other.envelope()),
                        "rectangles {:?} and {:?} intersected; all: {:?}",
                        r,
                        other,
                        tree.iter().collect::<Vec<_>>(),
                    );
                }
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

    // fn arb_rect_value() -> impl Strategy<Value = Option<(CostIntensity, MemVec, u8, ActionNum)>> {
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

    #[allow(clippy::type_complexity)]
    fn arb_rect_subtract() -> impl Strategy<Value = ((Vec<i64>, Vec<i64>), (Vec<i64>, Vec<i64>))> {
        proptest::collection::vec((0..6i64, 0..6i64, 0..6i64, 0..6i64), 0..4).prop_map(
            |dim_tuples| {
                let mut result = ((vec![], vec![]), (vec![], vec![]));
                for (num0, num1, num2, num3) in dim_tuples {
                    result.0 .0.push(num0.min(num1));
                    result.0 .1.push(num1.max(num0));
                    result.1 .0.push(num2.min(num3));
                    result.1 .1.push(num3.max(num2));
                }
                result
            },
        )
    }

    /// Construct an RTree, insert rects with the given points and the same value, and assert that
    /// they are all merged into a single rectangle
    fn assert_merged<const D: usize>(rects: &[([BimapSInt; D], [BimapSInt; D])]) {
        let mut tree = RTree::new();
        for (bottom, top) in rects {
            tree.merge_insert(bottom, top, (), true);
        }
        let merged_envelope = rects.iter().fold(
            AABB::<D>::from_corners(rects[0].0.into(), rects[0].1.into()),
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

    #[test]
    fn test_rect_partition_intersection_no_overlap() {
        let lhs_bottom = vec![0, 0];
        let lhs_top = vec![1, 1];
        let rhs_bottom = vec![2, 2];
        let rhs_top = vec![3, 3];
        let result = rect_partition_intersection(&lhs_bottom, &lhs_top, &rhs_bottom, &rhs_top);
        assert!(result.is_none());
    }

    #[test]
    fn test_rect_partition_intersection_full_overlap() {
        let lhs_bottom = vec![0, 0];
        let lhs_top = vec![3, 3];
        let rhs_bottom = vec![1, 1];
        let rhs_top = vec![2, 2];

        let expected_lhs_coverage = HashSet::from([
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 0],
            vec![2, 0],
            vec![1, 3],
            vec![2, 3],
            vec![3, 0],
            vec![3, 1],
            vec![3, 2],
            vec![3, 3],
        ]);

        let result =
            rect_partition_intersection(&lhs_bottom, &lhs_top, &rhs_bottom, &rhs_top).unwrap();
        assert_eq!(result.intersection, (vec![1, 1], vec![2, 2]));
        let merged_lhs_coverage: HashSet<_> = result
            .lhs_subrectangles
            .iter()
            .flat_map(|(bottom, top)| covered_points(bottom, top))
            .collect();
        assert_eq!(merged_lhs_coverage, expected_lhs_coverage);
        assert_eq!(result.rhs_subrectangles, vec![]);
    }

    #[test]
    fn test_rect_partition_intersection_partial_overlap() {
        let lhs_bottom = vec![0, 0];
        let lhs_top = vec![2, 2];
        let rhs_bottom = vec![1, 1];
        let rhs_top = vec![3, 3];

        let expected_lhs_coverage =
            HashSet::from([vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![0, 2]]);
        let expected_rhs_coverage =
            HashSet::from([vec![3, 1], vec![1, 3], vec![2, 3], vec![3, 2], vec![3, 3]]);

        let result =
            rect_partition_intersection(&lhs_bottom, &lhs_top, &rhs_bottom, &rhs_top).unwrap();
        assert_eq!(result.intersection, (vec![1, 1], vec![2, 2]));

        let merged_lhs_coverage: HashSet<_> = result
            .lhs_subrectangles
            .iter()
            .flat_map(|(bottom, top)| covered_points(bottom, top))
            .collect();
        let merged_rhs_coverage: HashSet<_> = result
            .rhs_subrectangles
            .iter()
            .flat_map(|(bottom, top)| covered_points(bottom, top))
            .collect();
        assert_eq!(merged_lhs_coverage, expected_lhs_coverage);
        assert_eq!(merged_rhs_coverage, expected_rhs_coverage);
    }

    fn covered_points(bottom: &[BimapSInt], top: &[BimapSInt]) -> HashSet<Vec<BimapSInt>> {
        assert_eq!(bottom.len(), top.len());
        bottom
            .iter()
            .zip(top)
            .map(|(b, t)| (*b..=*t))
            .multi_cartesian_product()
            .collect()
    }
}
