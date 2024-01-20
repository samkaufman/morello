use crate::cost::{Cost, MainCost};
use crate::db::{ActionCostVec, ActionIdx, DashmapDiskDatabase, Database, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::grid::linear::BimapInt;
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemVec;
use crate::ndarray::NDArray;
use crate::spec::Spec;
use crate::target::Target;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::ops::Range;

/// Stores a [Database] block. This may be a single value if all block entries have been filled with
/// the same [ActionCostVec], or an n-dimensional array along with a count of identical entries
/// accumulated until the first differing entry.
///
/// This isn't designed to be compressed in the case that a non-identical value is added and later
/// overwritten with an identical value. In that case, `matches` will be `None` and all values
/// would need to be scanned to determine whether they are all identical and, to set `matches`, how
/// many there are.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum DbBlock {
    Rle(Box<RleBlock>),
    ActionOnly(ActionOnlyBlock),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RleBlock {
    pub filled: NDArray<u8>, // 0 is empty; otherwise n - 1 = # of actions.
    pub main_costs: NDArray<MainCost>,
    pub peaks: NDArray<MemVec>,
    pub depths_actions: NDArray<(u8, ActionIdx)>,
    shape: SmallVec<[usize; 10]>,
}

// TODO: Replace [Option<u16>] with just [u16] offset by one.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActionOnlyBlock(pub NDArray<Option<SmallVec<[u16; 1]>>>);

impl DbBlock {
    pub fn get_with_preference<Tgt>(
        &self,
        containing_db: &DashmapDiskDatabase,
        query: &Spec<Tgt>,
        inner_pt: &[u8],
    ) -> GetPreference<ActionCostVec, SmallVec<[ActionIdx; 1]>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let inner_pt_usize = inner_pt
            .iter()
            .map(|v| *v as usize)
            .collect::<SmallVec<[_; 10]>>();
        match self {
            DbBlock::ActionOnly(v) => {
                let (inner_result, neighbor) = v.0.get_with_neighbor(&inner_pt_usize);
                match inner_result {
                    Some(inner) => {
                        GetPreference::Hit(ActionCostVec(
                            inner
                                .iter()
                                .map(|&action_idx| {
                                    let action = query
                                        .0
                                        .actions()
                                        .into_iter()
                                        .nth(action_idx.into())
                                        .unwrap();
                                    let imp = action.apply(query).unwrap();
                                    let recomputed_cost = compute_cost(&imp, &|s| {
                                        let Some(ActionCostVec(inner_decisions)) = containing_db.get(&s.0)
                                        else {
                                            panic!(
                                                "Missed sub-Impl {} while computing cost for {}",
                                                s.0, query
                                            );
                                        };
                                        if inner_decisions.is_empty() {
                                            panic!(
                                                "No actions for sub-Impl {} while computing cost for {}",
                                                s.0, query
                                            );
                                        } else if inner_decisions.len() == 1 {
                                            inner_decisions[0].1.clone()
                                        } else {
                                            todo!();
                                        }
                                    });
                                    (action_idx, recomputed_cost)
                                })
                                .collect(),
                        ))
                    },
                    None => {
                        GetPreference::Miss(neighbor.map(|v| v.clone().expect("neighbor of a miss should be a hit")))
                    }
                }
            }
            DbBlock::Rle(b) => {
                // TODO: Propogate an action index preference.
                match b.get(&inner_pt_usize) {
                    Some(r) => GetPreference::Hit(r),
                    None => GetPreference::Miss(None),
                }
            }
        }
    }

    pub fn compact(&mut self) {
        match self {
            DbBlock::ActionOnly(b) => b.0.shrink_to_fit(),
            DbBlock::Rle(e) => {
                e.compact();
            }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            DbBlock::ActionOnly(b) => b.0.shape(),
            DbBlock::Rle(e) => e.shape(),
        }
    }
}

impl RleBlock {
    fn empty<Tgt: Target>(k: u8, shape: &[usize]) -> Self {
        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());

        RleBlock {
            filled: NDArray::new_with_value(shape, 0),
            main_costs: NDArray::new(&shape_with_k),
            peaks: NDArray::new_with_value(&shape_with_k, MemVec::zero::<Tgt>()),
            depths_actions: NDArray::new(&shape_with_k),
            shape: shape.into(),
        }
    }

    pub(crate) fn partially_filled<Tgt: Target>(
        k: u8,
        shape: &[usize],
        dim_ranges: &[Range<BimapInt>],
        value: &ActionCostVec,
    ) -> Self {
        let mut e = Self::empty::<Tgt>(k, shape);
        e.fill_region_without_updating_match(k, dim_ranges, value);
        e
    }

    pub(crate) fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionCostVec,
    ) {
        self.fill_region_without_updating_match(k, dim_ranges, value);
    }

    fn fill_region_without_updating_match(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionCostVec,
    ) {
        let shape = self.filled.shape();
        debug_assert_eq!(dim_ranges.len(), shape.len());

        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());

        self.filled
            .fill_region(dim_ranges, &(u8::try_from(value.len()).unwrap() + 1));
        self.main_costs.fill_broadcast_1d(
            dim_ranges,
            value.0.iter().map(|(_, c)| c.main),
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
    }

    pub(crate) fn get(&self, pt: &[usize]) -> Option<ActionCostVec> {
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
                            main: self.main_costs[&pt_with_k],
                            peaks: self.peaks[&pt_with_k].clone(),
                            depth,
                        },
                    )
                })
                .collect(),
        ))
    }

    pub(crate) fn compact(&mut self) {
        self.filled.shrink_to_fit();
        self.main_costs.shrink_to_fit();
        self.peaks.shrink_to_fit();
        self.depths_actions.shrink_to_fit();
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Compute the cost of an incomplete Impl.
fn compute_cost<Tgt, Aux, F>(imp: &ImplNode<Tgt, Aux>, lookup: &F) -> Cost
where
    Tgt: Target,
    Aux: Clone,
    F: Fn(&SpecApp<Tgt, Spec<Tgt>, Aux>) -> Cost,
{
    match imp {
        ImplNode::SpecApp(s) => lookup(s),
        _ => {
            let children = imp.children();
            let mut child_costs: SmallVec<[_; 3]> = SmallVec::with_capacity(children.len());
            for c in children {
                child_costs.push(compute_cost(c, lookup));
            }
            Cost::from_child_costs(imp, &child_costs)
        }
    }
}
