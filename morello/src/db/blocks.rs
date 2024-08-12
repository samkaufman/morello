use super::{ActionCostVec, ActionIdx, ActionNormalizedCostVec, GetPreference};
use crate::{
    common::DimSize,
    cost::{Cost, CostIntensity},
    grid::{canon::CanonicalBimap, general::BiMap, linear::BimapInt},
    memorylimits::MemVec,
    ndarray::NDArray,
    rtree::RTreeDyn,
    spec::Spec,
    target::Target,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::Range;

#[cfg(feature = "db-stats")]
use parking_lot::Mutex;

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
pub struct RTreeBlock(RTreeDyn<Option<(CostIntensity, MemVec, u8, ActionIdx)>>);

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
            DbBlock::RTree(b) => b.get(inner_pt, query.0.volume()),
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
    pub fn with_single_rect(
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionNormalizedCostVec,
    ) -> Self {
        let mut tree_block = RTreeBlock(RTreeDyn::empty(dim_ranges.len()));
        tree_block.fill_region(k, dim_ranges, value);
        tree_block
    }

    pub fn get(&self, inner_pt: &[u8], spec_volume: DimSize) -> Option<ActionCostVec> {
        // TODO: Avoid conversion. Instead forward a slice right into locate_at_point.
        let arr = inner_pt.iter().map(|v| (*v).into()).collect::<Vec<_>>();

        // TODO: Return (and test!) k > 1. (The above point may be in a space without k dim.)
        let value = self.0.locate_at_point(&arr)?;
        Some(ActionCostVec(match &value {
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
            .map(|(action_idx, normalized_cost)| {
                (
                    normalized_cost.intensity,
                    normalized_cost.peaks.clone(),
                    normalized_cost.depth,
                    *action_idx,
                )
            });
        self.0.merge_insert(&bottom, &top, value);
    }
}
