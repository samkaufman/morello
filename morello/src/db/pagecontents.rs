use super::{ActionCostVec, ActionNormalizedCostVec, ActionNum, GetPreference};
use crate::{
    common::DimSize,
    cost::{Cost, CostIntensity},
    grid::{canon::CanonicalBimap, general::BiMap, linear::BimapInt},
    memorylimits::MemVec,
    rtree::RTreeDyn,
    spec::Spec,
    target::Target,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
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
        inner_pt: &[u8],
    ) -> GetPreference<ActionCostVec, Vec<ActionNum>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
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

    #[cfg(feature = "db-stats")]
    pub fn rect_count(&self) -> usize {
        self.0.size()
    }

    pub fn get(&self, inner_pt: &[u8], spec_volume: DimSize) -> Option<ActionCostVec> {
        // TODO: Avoid conversion. Instead forward a slice right into locate_at_point.
        let arr = inner_pt.iter().map(|v| (*v).into()).collect::<Vec<_>>();

        // TODO: Return (and test!) k > 1. (The above point may be in a space without k dim.)
        let value = self.0.locate_at_point(&arr)?;
        Some(ActionCostVec(match &value {
            Some((cost_intensity, peaks, depth, action_num)) => {
                let cost = Cost {
                    main: cost_intensity.into_main_cost_for_volume(spec_volume),
                    // peaks: MemVec::new_from_binary_scaled(peaks),
                    peaks: peaks.clone(),
                    depth: *depth,
                };
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
        self.0.merge_insert(&bottom, &top, value);
    }
}
