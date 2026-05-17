use super::RTreePt;
use crate::grid::linear::BimapSInt;
use rstar::{Point, RTreeObject};
use serde::{Deserialize, Serialize};
use wide::{i32x8, CmpGt};

#[cfg(target_feature = "avx512f")]
use wide::i32x16;

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
        lower: &[BimapSInt; D],
        upper: &[BimapSInt; D],
        point: &[BimapSInt; D],
    ) -> bool {
        let mut i = 0usize;
        // TODO: Uncomment the below case if we switch to u32, because u32x16 has an any fn, even
        //       though i32x16 does not.
        // #[cfg(target_feature = "avx512f")]
        // while i + 16 <= D {
        //     let pv = i32x16::from(&point[i..i + 16]);
        //     let l = i32x16::from(&lower[i..i + 16]);
        //     let u = i32x16::from(&upper[i..i + 16]);
        //     if (l.simd_gt(pv) | pv.simd_gt(u)).any() {
        //         return false;
        //     }
        //     i += 16;
        // }
        while i + 8 <= D {
            let pv = i32x8::from(&point[i..i + 8]);
            let l = i32x8::from(&lower[i..i + 8]);
            let u = i32x8::from(&upper[i..i + 8]);
            if (l.simd_gt(pv) | pv.simd_gt(u)).any() {
                return false;
            }
            i += 8;
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
        lower: &[BimapSInt; D],
        upper: &[BimapSInt; D],
        point: &[BimapSInt; D],
    ) -> BimapSInt {
        let mut vacc = i32x8::ZERO;
        let mut i = 0usize;
        while i + 8 <= D {
            let p = i32x8::from(&point[i..i + 8]);
            let l = i32x8::from(&lower[i..i + 8]);
            let u = i32x8::from(&upper[i..i + 8]);
            let dl_raw = l - p;
            let du_raw = p - u;
            let dl_nonneg = dl_raw & !(dl_raw >> 31i32);
            let du_nonneg = du_raw & !(du_raw >> 31i32);
            let d = dl_nonneg + du_nonneg;
            vacc += d * d;
            i += 8;
        }

        let lanes = vacc.as_array();
        let mut acc: BimapSInt = lanes.iter().sum();

        while i < D {
            let p = point[i];
            let l = lower[i];
            let u = upper[i];
            let below = (l - p).max(0);
            let above = (p - u).max(0);
            let d = below + above;
            acc += d * d;
            i += 1;
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
                arr: [BimapSInt::MAX; D],
            },
            upper: RTreePt {
                arr: [BimapSInt::MIN; D],
            },
        }
    }

    fn contains_point(&self, point: &Self::Point) -> bool {
        Self::ordered_corners_contain_point(&self.lower.arr, &self.upper.arr, &point.arr)
    }

    fn contains_envelope(&self, other: &Self) -> bool {
        let mut fail = i32x8::ZERO;
        let mut i = 0usize;
        while i + 8 <= D {
            let al = i32x8::from(&self.lower.arr[i..i + 8]);
            let bl = i32x8::from(&other.lower.arr[i..i + 8]);
            let au = i32x8::from(&self.upper.arr[i..i + 8]);
            let bu = i32x8::from(&other.upper.arr[i..i + 8]);
            fail |= al.simd_gt(bl) | bu.simd_gt(au);
            i += 8;
        }
        if fail.any() {
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
        // Vectorized min for lower and max for upper per 4-lane chunk.
        let mut i = 0usize;
        while i + 8 <= D {
            // lower = min(self.lower, other.lower)
            let al = i32x8::from(&self.lower.arr[i..i + 8]);
            let bl = i32x8::from(&other.lower.arr[i..i + 8]);
            let m_l = al.simd_gt(bl);
            let lmin = (bl & m_l) | (al & !m_l);
            self.lower.arr[i..i + 8].copy_from_slice(lmin.as_array());

            // upper = max(self.upper, other.upper)
            let au = i32x8::from(&self.upper.arr[i..i + 8]);
            let bu = i32x8::from(&other.upper.arr[i..i + 8]);
            let m_u = au.simd_gt(bu);
            let umax = (au & m_u) | (bu & !m_u);
            self.upper.arr[i..i + 8].copy_from_slice(umax.as_array());

            i += 8;
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
        while i + 8 <= D {
            let al = i32x8::from(&self.lower.arr[i..i + 8]);
            let au = i32x8::from(&self.upper.arr[i..i + 8]);
            let bl = i32x8::from(&other.lower.arr[i..i + 8]);
            let bu = i32x8::from(&other.upper.arr[i..i + 8]);
            if (al.simd_gt(bu) | bl.simd_gt(au)).any() {
                return false;
            }
            i += 8;
        }
        while i < D {
            if self.lower.arr[i] > other.upper.arr[i] || self.upper.arr[i] < other.lower.arr[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    fn area(&self) -> <Self::Point as Point>::Scalar {
        let ones = [1; 8];
        let mut vacc = i32x8::from(&ones[..]);
        let mut i = 0usize;
        while i + 8 <= D {
            let l = i32x8::from(&self.lower.arr[i..i + 8]);
            let u = i32x8::from(&self.upper.arr[i..i + 8]);
            let d = u - l;
            vacc *= d;
            i += 8;
        }
        let lanes = vacc.as_array();
        let mut prod: BimapSInt = lanes.iter().product();
        while i < D {
            prod *= self.upper.arr[i] - self.lower.arr[i];
            i += 1;
        }
        prod
    }

    fn distance_2(&self, point: &Self::Point) -> <Self::Point as Point>::Scalar {
        Self::ordered_corners_distance_2(&self.lower.arr, &self.upper.arr, &point.arr)
    }

    fn min_max_dist_2(&self, point: &Self::Point) -> <Self::Point as Point>::Scalar {
        let mut sum_max = i32x8::ZERO;
        let mut max_diff = i32x8::MIN;
        let mut i = 0usize;
        while i + 8 <= D {
            let p = i32x8::from(&point.arr[i..i + 8]);
            let l = i32x8::from(&self.lower.arr[i..i + 8]);
            let u = i32x8::from(&self.upper.arr[i..i + 8]);

            let dl = l - p;
            let du = u - p;
            let a = dl * dl;
            let b = du * du;

            let c = a - b;
            let k = c >> 31i32; // all ones if a < b, else 0
            let mmax = a - (c & k);
            let mmin = b + (c & k);
            let diff = mmax - mmin;
            sum_max += mmax;

            let cd = max_diff - diff;
            let kd = cd >> 31i32;
            max_diff -= cd & kd;
            i += 8;
        }

        let lanes_sum = sum_max.as_array();
        let mut acc_sum: BimapSInt = lanes_sum.iter().sum();
        let lanes_maxd = max_diff.as_array();
        let mut acc_maxdiff: BimapSInt = *lanes_maxd.iter().max().unwrap();

        while i < D {
            let p = point.arr[i];
            let dl = self.lower.arr[i] - p;
            let du = self.upper.arr[i] - p;
            let a = dl * dl;
            let b = du * du;
            let (mmax, mmin) = if a >= b { (a, b) } else { (b, a) };
            acc_sum += mmax;
            let diff = mmax - mmin;
            if diff > acc_maxdiff {
                acc_maxdiff = diff;
            }
            i += 1;
        }
        acc_sum - acc_maxdiff
    }

    fn center(&self) -> Self::Point {
        let mut out = [0; D];
        let mut i = 0usize;
        while i + 8 <= D {
            let l = i32x8::from(&self.lower.arr[i..i + 8]);
            let u = i32x8::from(&self.upper.arr[i..i + 8]);
            let mid = l + ((u - l) >> 1i32);
            // Store lanes into the output slice without horizontal ops.
            out[i..i + 8].copy_from_slice(mid.as_array());
            i += 8;
        }
        while i < D {
            let l = self.lower.arr[i];
            let u = self.upper.arr[i];
            out[i] = l + ((u - l) >> 1);
            i += 1;
        }
        RTreePt { arr: out }
    }

    fn intersection_area(&self, other: &Self) -> <Self::Point as Point>::Scalar {
        let mut vacc = i32x8::ONE;
        let mut i = 0usize;
        while i + 8 <= D {
            // lmax = max(al, bl)
            let al = i32x8::from(&self.lower.arr[i..i + 8]);
            let bl = i32x8::from(&other.lower.arr[i..i + 8]);
            let c1 = al - bl;
            let k1 = c1 >> 31i32;
            let lmax = al - (c1 & k1);

            // umin = min(au, bu)
            let au = i32x8::from(&self.upper.arr[i..i + 8]);
            let bu = i32x8::from(&other.upper.arr[i..i + 8]);
            let c2 = au - bu;
            let k2 = c2 >> 31i32;
            let umin = bu + (c2 & k2);

            // d = max(0, umin - lmax)
            let d_raw = umin - lmax;
            let d = d_raw & !(d_raw >> 31i32);
            vacc *= d;
            i += 8;
        }
        let lanes = vacc.as_array();
        let mut prod: BimapSInt = lanes.iter().product();
        while i < D {
            let lmax = self.lower.arr[i].max(other.lower.arr[i]);
            let umin = self.upper.arr[i].min(other.upper.arr[i]);
            let d = (umin - lmax).max(0);
            prod *= d;
            i += 1;
        }
        prod
    }

    fn perimeter_value(&self) -> <Self::Point as Point>::Scalar {
        let mut vacc = i32x8::ZERO;
        let mut i = 0usize;
        while i + 8 <= D {
            let l = i32x8::from(&self.lower.arr[i..i + 8]);
            let u = i32x8::from(&self.upper.arr[i..i + 8]);
            vacc += u - l;
            i += 8;
        }
        let lanes = vacc.as_array();
        let mut sum: BimapSInt = lanes.iter().sum();
        while i < D {
            sum += self.upper.arr[i] - self.lower.arr[i];
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
        fn test_eq_contains_point(ours in arb_aabb(), p in prop::collection::vec(-2..18, 3)) {
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
        fn test_eq_area(ours in arb_aabb()) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            prop_assert_eq!(ours.area(), theirs.area());
        }

        #[test]
        fn test_eq_min_max_dist_2(ours in arb_aabb(), p in prop::collection::vec(-2..18, 3)) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            let pt = RTreePt::<3> { arr: [p[0], p[1], p[2]] };
            prop_assert_eq!(ours.min_max_dist_2(&pt), theirs.min_max_dist_2(&pt));
        }

        #[test]
        fn test_eq_perimeter_value(ours in arb_aabb()) {
            let theirs = rstar::AABB::from_corners(ours.lower().clone(), ours.upper().clone());
            prop_assert_eq!(ours.perimeter_value(), theirs.perimeter_value());
        }

        #[test]
        fn test_eq_intersection_area(ours0 in arb_aabb(), ours1 in arb_aabb()) {
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
        fn test_eq_distance_2_small(ours in arb_aabb(), p in prop::collection::vec(-2..18, 3)) {
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

            let ours_boxes: Vec<([BimapSInt;3],[BimapSInt;3])> =
                ours.iter().map(|a| (a.lower().arr, a.upper().arr)).collect();
            let theirs_boxes: Vec<([BimapSInt;3],[BimapSInt;3])> =
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
                arr: [BimapSInt::MIN; 6],
            },
            RTreePt { arr: [0; 6] },
        );
        let b = AABB::<6>::from_corners(
            RTreePt {
                arr: [BimapSInt::MAX; 6],
            },
            RTreePt {
                arr: [BimapSInt::MAX; 6],
            },
        );

        let mut m = a.clone();
        m.merge(&b);
        assert_eq!(m.lower().arr, [BimapSInt::MIN; 6]);
        assert_eq!(m.upper().arr, [BimapSInt::MAX; 6]);
    }

    // Generate an axis-aligned box by sampling a base point and a non-negative offset.
    // Returns an AABB directly. Name reflects that: arb_aabb.
    fn arb_aabb() -> impl Strategy<Value = AABB<3>> {
        prop::collection::vec(0..16, 3)
            .prop_flat_map(|base| {
                prop::collection::vec(0..16, 3).prop_map(move |delta| (base.clone(), delta))
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
