use super::{ActionCostVec, ActionNormalizedCostVec, ActionNum, GetPreference};
use crate::{
    cost::{CostIntensity, NormalizedCost},
    grid::{canon::CanonicalBimap, general::BiMap, linear::BimapInt},
    memorylimits::MemVec,
    rtree::RTreeDyn,
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
pub struct RTreePageContents(RTreeDyn<Option<(CostIntensity, MemVec, u8, ActionNum)>>);

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

    pub fn tree(&self) -> &RTreeDyn<Option<(CostIntensity, MemVec, u8, ActionNum)>> {
        &self.0
    }

    #[cfg(feature = "db-stats")]
    pub fn rect_count(&self) -> usize {
        self.0.size()
    }

    pub fn get(&self, inner_pt: &[BimapInt], spec_volume: NonZeroU64) -> Option<ActionCostVec> {
        // TODO: Avoid conversion. Instead forward a slice right into locate_at_point.
        let arr = inner_pt.iter().map(|v| (*v).into()).collect::<Vec<_>>();

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
            bottom.push(rng.start.into());
            top.push((rng.end - 1).into());
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
        self.0
            .fold_insert(&bottom, &top, value, true, min_overlapping_action);
    }
}

fn min_overlapping_action(
    inserted: Option<(CostIntensity, MemVec, u8, ActionNum)>,
    existing: Option<(CostIntensity, MemVec, u8, ActionNum)>,
) -> Option<(CostIntensity, MemVec, u8, ActionNum)> {
    match (inserted, existing) {
        (Some(inserted), Some(existing)) => {
            assert_eq!(
                    inserted.0, existing.0,
                    "Overlapping rectangles must have matching cost intensities: inserted={:?}, existing={:?}",
                    inserted.0, existing.0
                );
            Some(cmp::min_by(inserted, existing, |inserted, existing| {
                inserted
                    .1
                    .lex_cmp(&existing.1)
                    .then(inserted.2.cmp(&existing.2))
            }))
        }
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => {
            panic!("Overlapping rectangles must both have values or both be empty")
        }
    }
}
