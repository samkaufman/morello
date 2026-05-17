use super::{ActionCostVec, ActionNormalizedCostVec, ActionNum, DbValue, GetPreference};
use crate::{
    cost::NormalizedCost,
    grid::{
        canon::CanonicalBimap,
        general::BiMap,
        linear::{BimapInt, BimapSInt},
    },
    rtree::{RTreeDyn, RegionScanResult},
    spec::Spec,
    target::Target,
};
use serde::{Deserialize, Serialize};
use std::cmp;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::ops::Range;

/// Stores a [Database] block. This may be a single value if all block entries have been filled with
/// the same [ActionCostVec], or an n-dimensional array along with a count of identical entries
/// accumulated until the first differing entry.
///
/// This isn't designed to be compressed in the case that a non-identical value is added and later
/// overwritten with an identical value. In that case, `matches` will be `None` and all values
/// would need to be scanned to determine whether they are all identical and, to set `matches`, how
/// many there are.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PageContents {
    RTree(Box<RTreePageContents>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTreePageContents(RTreeDyn<DbValue>);

impl PageContents {
    pub fn get_with_preference<Tgt>(
        &self,
        query: &Spec<Tgt>,
        inner_pt: &[BimapInt],
    ) -> GetPreference<ActionCostVec, Vec<ActionNum>>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let block_result = match self {
            PageContents::RTree(b) => b.get(inner_pt, query.0.volume()),
        };
        match block_result {
            Some(r) => GetPreference::Hit(r),
            None => GetPreference::Miss(None),
        }
    }

    pub fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    ) {
        match self {
            PageContents::RTree(b) => b.fill_region(k, dim_ranges, value),
        }
    }
}

impl RTreePageContents {
    pub fn empty(rank: usize) -> Self {
        RTreePageContents(RTreeDyn::empty(rank))
    }

    pub fn tree(&self) -> &RTreeDyn<DbValue> {
        &self.0
    }

    #[cfg(feature = "db-stats")]
    pub fn rect_count(&self) -> usize {
        self.0.size()
    }

    #[cfg(feature = "db-stats")]
    pub fn spec_count(&self) -> u128 {
        self.0
            .iter()
            .map(|(bottom, top, _)| {
                bottom
                    .iter()
                    .zip(top)
                    .map(|(bottom, top)| u128::try_from(top - bottom + 1).unwrap())
                    .product::<u128>()
            })
            .sum()
    }

    pub fn get(&self, inner_pt: &[BimapInt], spec_volume: NonZeroU64) -> Option<ActionCostVec> {
        let arr = inner_pt
            .iter()
            .map(|&v| local_coord_to_rtree(v))
            .collect::<Vec<_>>();

        // TODO: Return (and test!) k > 1. (The above point may be in a space without k dim.)
        let value = self.0.locate_at_point(&arr)?;
        Some(ActionCostVec(match &value {
            Some((cost_intensity, peaks, depth, action_num)) => {
                let cost = NormalizedCost {
                    intensity: *cost_intensity,
                    peaks: peaks.clone(),
                    depth: *depth,
                }
                .into_cost(spec_volume);
                vec![(*action_num, cost)]
            }
            None => vec![],
        }))
    }

    pub fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        normalized_action_costs: &ActionNormalizedCostVec,
    ) {
        if k != 1 {
            todo!("Support k > 1");
        }
        let mut bottom = Vec::with_capacity(dim_ranges.len());
        let mut top = Vec::with_capacity(dim_ranges.len());
        for rng in dim_ranges {
            assert!(rng.start < rng.end);
            bottom.push(local_coord_to_rtree(rng.start));
            top.push(local_coord_to_rtree(rng.end - 1));
        }

        let value = normalized_action_costs
            .0
            .first()
            .map(|(action_num, normalized_cost)| {
                (
                    normalized_cost.intensity,
                    normalized_cost.peaks.clone(),
                    normalized_cost.depth,
                    *action_num,
                )
            });

        match self.0.covered_or_all_intersections_match(
            &bottom,
            &top,
            |existing| action_dominates(existing, &value),
            |existing| action_dominates(&value, existing),
            |existing| existing == &value,
        ) {
            RegionScanResult::Covered => {}
            RegionScanResult::AllIntersectionsMatched {
                all_intersections_equal: true,
            } => {
                self.0
                    .merge_insert(&bottom, &top, value, cfg!(debug_assertions));
            }
            RegionScanResult::AllIntersectionsMatched {
                all_intersections_equal: false,
            } => {
                self.0.replace(&bottom, &top, value, true);
            }
            RegionScanResult::SomeIntersectionsUnmatched => {
                self.0
                    .fold_insert(&bottom, &top, value, true, |inserted, existing| {
                        if action_dominates(&inserted, &existing) {
                            inserted
                        } else {
                            existing
                        }
                    });
            }
        }
    }
}

fn local_coord_to_rtree(coord: BimapInt) -> BimapSInt {
    BimapSInt::try_from(coord).expect("database page-local coordinate exceeded R-tree precision")
}

fn action_dominates(lhs: &DbValue, rhs: &DbValue) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => {
            assert_eq!(
                lhs.0, rhs.0,
                "Overlapping rectangles must have matching cost intensities: lhs={:?}, rhs={:?}",
                lhs.0, rhs.0
            );
            lhs.1
                .lex_cmp(&rhs.1)
                .then(lhs.2.cmp(&rhs.2))
                .then(lhs.3.cmp(&rhs.3))
                != cmp::Ordering::Greater
        }
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => {
            panic!("Overlapping rectangles must both have values or both be empty")
        }
    }
}
