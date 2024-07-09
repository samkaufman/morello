use super::{ActionCostVec, ActionIdx, ActionNormalizedCostVec, GetPreference};
use crate::{
    common::DimSize,
    cost::{Cost, CostIntensity},
    grid::{canon::CanonicalBimap, general::BiMap, linear::BimapInt},
    memorylimits::MemVec,
    ndarray::NDArray,
    spec::Spec,
    target::Target,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
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
#[serde(bound = "")]
pub enum DbBlock {
    Whole(Box<WholeBlock>),
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

impl DbBlock {
    pub fn get_with_preference<Tgt>(
        &self,
        query: &Spec<Tgt>,
        inner_pt: &[u8],
    ) -> GetPreference<ActionCostVec, Vec<ActionIdx>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let inner_pt_usize = inner_pt.iter().map(|v| *v as usize).collect::<Vec<_>>();
        match self {
            DbBlock::Whole(b) => {
                // TODO: Propogate an action index preference.
                match b.get(&inner_pt_usize, query.0.volume()) {
                    Some(r) => GetPreference::Hit(r),
                    None => GetPreference::Miss(None),
                }
            }
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

    #[cfg(feature = "db-stats")]
    pub fn accesses(&self) -> (usize, usize) {
        let guard = self.access_counts.lock();
        let Some(l) = guard.as_ref() else {
            let shape = self.filled.shape();
            let volume = shape.iter().product();
            return (0, volume);
        };
        let total = l.data.len();
        let read = l
            .data
            .runs()
            .filter_map(|r| {
                if *r.value {
                    Some(usize::try_from(r.len).unwrap())
                } else {
                    None
                }
            })
            .sum();
        (read, total)
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
