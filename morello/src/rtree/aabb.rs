use super::{RTreeInt, RTreeMetric, RTreePt};
use rstar::{Point, RTreeObject};
use serde::{Deserialize, Serialize};
use wide::{i16x16, i8x16, i8x32, CmpGt};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct AABB<const D: usize> {
    lower: RTreePt<D>,
    upper: RTreePt<D>,
}

impl<const D: usize> AABB<D> {
    #[allow(dead_code)]
    pub fn from_corners(p1: RTreePt<D>, p2: RTreePt<D>) -> Self {
        let mut low = p1.arr;
        let mut up = p2.arr;
        for i in 0..D {
            if low[i] > up[i] {
                core::mem::swap(&mut low[i], &mut up[i]);
            }
        }
        AABB {
            lower: RTreePt { arr: low },
            upper: RTreePt { arr: up },
        }
    }

    pub(crate) fn from_bounds(lower: RTreePt<D>, upper: RTreePt<D>) -> Self {
        debug_assert!(lower.arr.iter().zip(&upper.arr).all(|(l, u)| l <= u));
        AABB { lower, upper }
    }

    pub(crate) fn ordered_corners_contain_point(
        lower: &[RTreeInt; D],
        upper: &[RTreeInt; D],
        point: &[RTreeInt; D],
    ) -> bool {
        let mut i = 0usize;
        while i + 32 <= D {
            let pv = i8x32::from(&point[i..i + 32]);
            let l = i8x32::from(&lower[i..i + 32]);
            let u = i8x32::from(&upper[i..i + 32]);
            if (l.simd_gt(pv) | pv.simd_gt(u)).any() {
                return false;
            }
            i += 32;
        }
        while i + 16 <= D {
            let pv = i8x16::from(&point[i..i + 16]);
            let l = i8x16::from(&lower[i..i + 16]);
            let u = i8x16::from(&upper[i..i + 16]);
            if (l.simd_gt(pv) | pv.simd_gt(u)).any() {
                return false;
            }
            i += 16;
        }
        while i < D {
            let p = point[i];
            if p < lower[i] || p > upper[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    pub(crate) fn ordered_corners_distance_2(
        lower: &[RTreeInt; D],
        upper: &[RTreeInt; D],
        point: &[RTreeInt; D],
    ) -> RTreeMetric {
        let mut acc = 0;
        for i in 0..D {
            let p = RTreeMetric::from(point[i]);
            let l = RTreeMetric::from(lower[i]);
            let u = RTreeMetric::from(upper[i]);
            let below = (l - p).max(0);
            let above = (p - u).max(0);
            let d = below + above;
            acc += d * d;
        }
        acc
    }

    #[allow(dead_code)]
    #[inline]
    pub fn lower(&self) -> &RTreePt<D> {
        &self.lower
    }

    #[allow(dead_code)]
    #[inline]
    pub fn upper(&self) -> &RTreePt<D> {
        &self.upper
    }
}

impl<const D: usize> rstar::Envelope for AABB<D> {
    type Point = RTreePt<D>;

    fn new_empty() -> Self {
        AABB {
            lower: RTreePt {
                arr: [RTreeInt::MAX; D],
            },
            upper: RTreePt {
                arr: [RTreeInt::MIN; D],
            },
        }
    }

    fn contains_point(&self, point: &Self::Point) -> bool {
        Self::ordered_corners_contain_point(&self.lower.arr, &self.upper.arr, &point.arr)
    }

    fn contains_envelope(&self, other: &Self) -> bool {
        let mut fail32 = i8x32::ZERO;
        let mut i = 0usize;
        while i + 32 <= D {
            let al = i8x32::from(&self.lower.arr[i..i + 32]);
            let bl = i8x32::from(&other.lower.arr[i..i + 32]);
            let au = i8x32::from(&self.upper.arr[i..i + 32]);
            let bu = i8x32::from(&other.upper.arr[i..i + 32]);
            fail32 |= al.simd_gt(bl) | bu.simd_gt(au);
            i += 32;
        }
        if fail32.any() {
            return false;
        }

        let mut fail16 = i8x16::ZERO;
        while i + 16 <= D {
            let al = i8x16::from(&self.lower.arr[i..i + 16]);
            let bl = i8x16::from(&other.lower.arr[i..i + 16]);
            let au = i8x16::from(&self.upper.arr[i..i + 16]);
            let bu = i8x16::from(&other.upper.arr[i..i + 16]);
            fail16 |= al.simd_gt(bl) | bu.simd_gt(au);
            i += 16;
        }
        if fail16.any() {
            return false;
        }
        while i < D {
            if self.lower.arr[i] > other.lower.arr[i] || self.upper.arr[i] < other.upper.arr[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    fn merge(&mut self, other: &Self) {
        let mut i = 0usize;
        while i + 32 <= D {
            // lower = min(self.lower, other.lower)
            let al = i8x32::from(&self.lower.arr[i..i + 32]);
            let bl = i8x32::from(&other.lower.arr[i..i + 32]);
            let m_l = al.simd_gt(bl);
            let lmin = (bl & m_l) | (al & !m_l);
            self.lower.arr[i..i + 32].copy_from_slice(lmin.as_array());

            // upper = max(self.upper, other.upper)
            let au = i8x32::from(&self.upper.arr[i..i + 32]);
            let bu = i8x32::from(&other.upper.arr[i..i + 32]);
            let m_u = au.simd_gt(bu);
            let umax = (au & m_u) | (bu & !m_u);
            self.upper.arr[i..i + 32].copy_from_slice(umax.as_array());

            i += 32;
        }

        while i + 16 <= D {
            // lower = min(self.lower, other.lower)
            let al = i8x16::from(&self.lower.arr[i..i + 16]);
            let bl = i8x16::from(&other.lower.arr[i..i + 16]);
            let m_l = al.simd_gt(bl);
            let lmin = (bl & m_l) | (al & !m_l);
            self.lower.arr[i..i + 16].copy_from_slice(lmin.as_array());

            // upper = max(self.upper, other.upper)
            let au = i8x16::from(&self.upper.arr[i..i + 16]);
            let bu = i8x16::from(&other.upper.arr[i..i + 16]);
            let m_u = au.simd_gt(bu);
            let umax = (au & m_u) | (bu & !m_u);
            self.upper.arr[i..i + 16].copy_from_slice(umax.as_array());

            i += 16;
        }

        while i < D {
            if other.lower.arr[i] < self.lower.arr[i] {
                self.lower.arr[i] = other.lower.arr[i];
            }
            if other.upper.arr[i] > self.upper.arr[i] {
                self.upper.arr[i] = other.upper.arr[i];
            }
            i += 1;
        }
    }

    fn merged(&self, other: &Self) -> Self {
        let mut res = self.clone();
        res.merge(other);
        res
    }

    fn intersects(&self, other: &Self) -> bool {
        let mut i = 0usize;
        while i + 32 <= D {
            let al = i8x32::from(&self.lower.arr[i..i + 32]);
            let au = i8x32::from(&self.upper.arr[i..i + 32]);
            let bl = i8x32::from(&other.lower.arr[i..i + 32]);
            let bu = i8x32::from(&other.upper.arr[i..i + 32]);
            if (al.simd_gt(bu) | bl.simd_gt(au)).any() {
                return false;
            }
            i += 32;
        }
        while i + 16 <= D {
            let al = i8x16::from(&self.lower.arr[i..i + 16]);
            let au = i8x16::from(&self.upper.arr[i..i + 16]);
            let bl = i8x16::from(&other.lower.arr[i..i + 16]);
            let bu = i8x16::from(&other.upper.arr[i..i + 16]);
            if (al.simd_gt(bu) | bl.simd_gt(au)).any() {
                return false;
            }
            i += 16;
        }
        while i < D {
            if self.lower.arr[i] > other.upper.arr[i] || self.upper.arr[i] < other.lower.arr[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    fn area(&self) -> <Self::Point as Point>::ComposedScalar {
        let mut prod = 1;
        for i in 0..D {
            let d = (RTreeMetric::from(self.upper.arr[i]) - RTreeMetric::from(self.lower.arr[i]))
                .max(0);
            prod *= d;
        }
        prod
    }

    fn distance_2(&self, point: &Self::Point) -> <Self::Point as Point>::ComposedScalar {
        Self::ordered_corners_distance_2(&self.lower.arr, &self.upper.arr, &point.arr)
    }

    fn min_max_dist_2(&self, point: &Self::Point) -> <Self::Point as Point>::ComposedScalar {
        let mut acc_sum = 0;
        let mut acc_maxdiff = RTreeMetric::MIN;
        for i in 0..D {
            let p = RTreeMetric::from(point.arr[i]);
            let dl = RTreeMetric::from(self.lower.arr[i]) - p;
            let du = RTreeMetric::from(self.upper.arr[i]) - p;
            let a = dl * dl;
            let b = du * du;
            let (mmax, mmin) = if a >= b { (a, b) } else { (b, a) };
            acc_sum += mmax;
            let diff = mmax - mmin;
            if diff > acc_maxdiff {
                acc_maxdiff = diff;
            }
        }
        acc_sum - acc_maxdiff
    }

    fn center(&self) -> Self::Point {
        let mut out = [0; D];
        let mut i = 0usize;
        while i + 16 <= D {
            let l = i16x16::from(i8x16::from(&self.lower.arr[i..i + 16]));
            let u = i16x16::from(i8x16::from(&self.upper.arr[i..i + 16]));
            let mid = l + ((u - l) >> 1);
            out[i..i + 16].copy_from_slice(i8x16::from_i16x16_saturate(mid).as_array());
            i += 16;
        }
        while i < D {
            let l = i64::from(self.lower.arr[i]);
            let u = i64::from(self.upper.arr[i]);
            out[i] = (l + ((u - l) >> 1)) as RTreeInt;
            i += 1;
        }
        RTreePt { arr: out }
    }

    fn intersection_area(&self, other: &Self) -> <Self::Point as Point>::ComposedScalar {
        let mut prod = 1;
        let zero = i16x16::ZERO;
        let mut i = 0usize;
        while i + 16 <= D {
            let al = i16x16::from(i8x16::from(&self.lower.arr[i..i + 16]));
            let au = i16x16::from(i8x16::from(&self.upper.arr[i..i + 16]));
            let bl = i16x16::from(i8x16::from(&other.lower.arr[i..i + 16]));
            let bu = i16x16::from(i8x16::from(&other.upper.arr[i..i + 16]));
            let lmax = al.simd_gt(bl).blend(al, bl);
            let umin = au.simd_gt(bu).blend(bu, au);
            let d_raw = umin - lmax;
            let d = d_raw.simd_gt(zero).blend(d_raw, zero);
            for lane in d.as_array() {
                prod *= RTreeMetric::from(*lane);
            }
            i += 16;
        }
        while i < D {
            let lmax = self.lower.arr[i].max(other.lower.arr[i]);
            let umin = self.upper.arr[i].min(other.upper.arr[i]);
            let d = (RTreeMetric::from(umin) - RTreeMetric::from(lmax)).max(0);
            prod *= d;
            i += 1;
        }
        prod
    }

    fn perimeter_value(&self) -> <Self::Point as Point>::ComposedScalar {
        let mut sum = 0;
        let mut i = 0usize;
        while i + 16 <= D {
            let l = i16x16::from(i8x16::from(&self.lower.arr[i..i + 16]));
            let u = i16x16::from(i8x16::from(&self.upper.arr[i..i + 16]));
            for lane in (u - l).as_array() {
                sum += RTreeMetric::from(*lane);
            }
            i += 16;
        }
        while i < D {
            sum += RTreeMetric::from(self.upper.arr[i]) - RTreeMetric::from(self.lower.arr[i]);
            i += 1;
        }
        sum.max(0)
    }

    fn sort_envelopes<T: RTreeObject<Envelope = Self>>(axis: usize, envelopes: &mut [T]) {
        envelopes.sort_by(|l, r| {
            let lval = l.envelope().lower.arr[axis];
            let rval = r.envelope().lower.arr[axis];
            lval.partial_cmp(&rval).unwrap()
        });
    }

    fn partition_envelopes<T: RTreeObject<Envelope = Self>>(
        axis: usize,
        envelopes: &mut [T],
        selection_size: usize,
    ) {
        envelopes.select_nth_unstable_by(selection_size, |l, r| {
            let lval = l.envelope().lower.arr[axis];
            let rval = r.envelope().lower.arr[axis];
            lval.partial_cmp(&rval).unwrap()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstar::{Envelope, RTreeObject};

    // Convenience to use AABB directly with sort_envelopes/partition_envelopes
    impl RTreeObject for AABB<3> {
        type Envelope = AABB<3>;
        fn envelope(&self) -> Self::Envelope {
            self.clone()
        }
    }

    #[derive(Clone)]
    struct RObj {
        low: RTreePt<3>,
        high: RTreePt<3>,
    }
    impl RTreeObject for RObj {
        type Envelope = rstar::AABB<RTreePt<3>>;
        fn envelope(&self) -> Self::Envelope {
            rstar::AABB::from_corners(self.low.clone(), self.high.clone())
        }
    }

    proptest! {
        #[test]
        fn test_eq_contains_point(ours in arb_aabb(), p in prop::collection::vec(-2i8..18i8, 3)) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            let pt = RTreePt::<3> { arr: [p[0], p[1], p[2]] };
            prop_assert_eq!(ours.contains_point(&pt), theirs.contains_point(&pt));
        }

        #[test]
        fn test_eq_intersects(ours0 in arb_aabb(), ours1 in arb_aabb()) {
            let theirs0 = rstar::AABB::from_corners(ours0.lower().clone(), ours0.upper().clone());
            let theirs1 = rstar::AABB::from_corners(ours1.lower().clone(), ours1.upper().clone());
            prop_assert_eq!(ours0.intersects(&ours1), theirs0.intersects(&theirs1));
        }

        #[test]
        fn test_eq_contains_envelope(ours0 in arb_aabb(), ours1 in arb_aabb()) {
            let theirs0 = rstar::AABB::from_corners(ours0.lower().clone(), ours0.upper().clone());
            let theirs1 = rstar::AABB::from_corners(ours1.lower().clone(), ours1.upper().clone());
            prop_assert_eq!(ours0.contains_envelope(&ours1), theirs0.contains_envelope(&theirs1));
        }

        #[test]
        fn test_eq_merge_and_merged(mut ours0 in arb_aabb(), ours1 in arb_aabb()) {
            let theirs0 = rstar::AABB::from_corners(ours0.lower().clone(), ours0.upper().clone());
            let theirs1 = rstar::AABB::from_corners(ours1.lower().clone(), ours1.upper().clone());

            ours0.merge(&ours1);
            let mut theirs0m = theirs0.clone();
            theirs0m.merge(&theirs1);
            prop_assert_eq!(ours0.lower().arr, theirs0m.lower().arr);
            prop_assert_eq!(ours0.upper().arr, theirs0m.upper().arr);

            let ours2 = ours0.clone().merged(&ours1);
            let theirs2 = theirs0.merged(&theirs1);
            prop_assert_eq!(ours2.lower().arr, theirs2.lower().arr);
            prop_assert_eq!(ours2.upper().arr, theirs2.upper().arr);
        }

        #[test]
        fn test_eq_area(ours in arb_small_aabb()) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            prop_assert_eq!(ours.area(), theirs.area());
        }

        #[test]
        fn test_eq_min_max_dist_2(ours in arb_small_aabb(), p in prop::collection::vec(-2i8..6i8, 3)) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            let pt = RTreePt::<3> { arr: [p[0], p[1], p[2]] };
            prop_assert_eq!(ours.min_max_dist_2(&pt), theirs.min_max_dist_2(&pt));
        }

        #[test]
        fn test_eq_perimeter_value(ours in arb_small_aabb()) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            prop_assert_eq!(ours.perimeter_value(), theirs.perimeter_value());
        }

        #[test]
        fn test_eq_intersection_area(ours0 in arb_small_aabb(), ours1 in arb_small_aabb()) {
            let theirs0 = rstar::AABB::from_corners(ours0.lower().clone(), ours0.upper().clone());
            let theirs1 = rstar::AABB::from_corners(ours1.lower().clone(), ours1.upper().clone());
            prop_assert_eq!(ours0.intersection_area(&ours1), theirs0.intersection_area(&theirs1));
        }

        #[test]
        fn test_eq_center(ours in arb_aabb()) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            prop_assert_eq!(ours.center().arr, theirs.center().arr);
        }

        // distance_2 should match rstar::AABB for small ranges (no overflow)
        #[test]
        fn test_eq_distance_2_small(ours in arb_small_aabb(), p in prop::collection::vec(-2i8..6i8, 3)) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            let pt = RTreePt::<3> { arr: [p[0], p[1], p[2]] };
            prop_assert_eq!(ours.distance_2(&pt), theirs.distance_2(&pt));
        }

        #[test]
        fn test_eq_sort_axis0(v in prop::collection::vec(arb_aabb(), 0..8)) {
            let mut ours = v.clone();
            let mut theirs: Vec<RObj> = v
                .iter()
                .map(|a| RObj { low: a.lower().clone(), high: a.upper().clone() })
                .collect();

            AABB::<3>::sort_envelopes(0, &mut ours);
            rstar::AABB::<RTreePt<3>>::sort_envelopes(0, &mut theirs);

            let ours_boxes: Vec<([RTreeInt;3],[RTreeInt;3])> =
                ours.iter().map(|a| (a.lower().arr, a.upper().arr)).collect();
            let theirs_boxes: Vec<([RTreeInt;3],[RTreeInt;3])> =
                theirs.iter().map(|o| (o.envelope().lower().arr, o.envelope().upper().arr)).collect();
            prop_assert_eq!(ours_boxes, theirs_boxes);
        }

        #[test]
        fn test_eq_partition_axis0(v in prop::collection::vec(arb_aabb(), 1..8)) {
            let mut ours = v.clone();
            let mut theirs: Vec<RObj> = v
                .iter()
                .map(|a| RObj { low: a.lower().clone(), high: a.upper().clone() })
                .collect();

            // partition equivalence: same pivot key at k
            let k = ours.len() / 2;
            AABB::<3>::partition_envelopes(0, &mut ours, k);
            rstar::AABB::<RTreePt<3>>::partition_envelopes(0, &mut theirs, k);

            // Compare pivot key and ensure strict partitions (< and > pivot) match as multisets.
            let to_box = |a: &AABB<3>| (a.lower().arr, a.upper().arr);
            let to_box_r = |o: &RObj| { let e = o.envelope(); (e.lower().arr, e.upper().arr) };

            let ours_pivot_key = ours[k].lower().arr[0];
            let theirs_pivot_key = theirs[k].envelope().lower().arr[0];
            prop_assert_eq!(ours_pivot_key, theirs_pivot_key);
            let pivot_key = theirs_pivot_key;

            let mut left_lt_ours: Vec<_> = ours[..k]
                .iter().filter(|a| a.lower().arr[0] < pivot_key).map(to_box).collect();
            let mut left_lt_theirs: Vec<_> = theirs[..k]
                .iter().filter(|o| o.envelope().lower().arr[0] < pivot_key).map(to_box_r).collect();
            left_lt_ours.sort();
            left_lt_theirs.sort();
            prop_assert_eq!(left_lt_ours, left_lt_theirs);

            let mut right_gt_ours: Vec<_> = ours[k+1..]
                .iter().filter(|a| a.lower().arr[0] > pivot_key).map(to_box).collect();
            let mut right_gt_theirs: Vec<_> = theirs[k+1..]
                .iter().filter(|o| o.envelope().lower().arr[0] > pivot_key).map(to_box_r).collect();
            right_gt_ours.sort();
            right_gt_theirs.sort();
            prop_assert_eq!(right_gt_ours, right_gt_theirs);
        }
    }

    /// Test merge computes correct min/max with extreme values.
    ///
    /// This targets a prior bug where using subtraction-derived masks could overflow
    /// and select the wrong lane, producing an invalid envelope.
    #[test]
    fn test_merge_extreme_values() {
        let a = AABB::<6>::from_corners(
            RTreePt {
                arr: [RTreeInt::MIN; 6],
            },
            RTreePt { arr: [0; 6] },
        );
        let b = AABB::<6>::from_corners(
            RTreePt {
                arr: [RTreeInt::MAX; 6],
            },
            RTreePt {
                arr: [RTreeInt::MAX; 6],
            },
        );

        let mut m = a.clone();
        m.merge(&b);
        assert_eq!(m.lower().arr, [RTreeInt::MIN; 6]);
        assert_eq!(m.upper().arr, [RTreeInt::MAX; 6]);
    }

    // Generate an axis-aligned box by sampling a base point and a non-negative offset.
    // Returns an AABB directly. Name reflects that: arb_aabb.
    fn arb_aabb() -> impl Strategy<Value = AABB<3>> {
        prop::collection::vec(0i8..16i8, 3)
            .prop_flat_map(|base| {
                prop::collection::vec(0i8..16i8, 3).prop_map(move |delta| (base.clone(), delta))
            })
            .prop_map(|(base, delta)| {
                let low = RTreePt::<3> {
                    arr: [base[0], base[1], base[2]],
                };
                let high = RTreePt::<3> {
                    arr: [base[0] + delta[0], base[1] + delta[1], base[2] + delta[2]],
                };
                AABB::from_corners(low, high)
            })
    }

    fn arb_small_aabb() -> impl Strategy<Value = AABB<3>> {
        prop::collection::vec(0i8..4i8, 3)
            .prop_flat_map(|base| {
                prop::collection::vec(0i8..4i8, 3).prop_map(move |delta| (base.clone(), delta))
            })
            .prop_map(|(base, delta)| {
                let low = RTreePt::<3> {
                    arr: [base[0], base[1], base[2]],
                };
                let high = RTreePt::<3> {
                    arr: [base[0] + delta[0], base[1] + delta[1], base[2] + delta[2]],
                };
                AABB::from_corners(low, high)
            })
    }
}
