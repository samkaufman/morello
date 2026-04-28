use crate::cost::NormalizedCost;
use crate::db::ActionCostVec;
use crate::grid::general::BiMap;
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::spatial_query::SpatialQuery;
use crate::spec::Spec;
use crate::target::Target;
use std::hash::Hash;

pub mod fallback;

pub trait SpatialActionSolverT<Tgt: Target> {
    fn spatial_query<B, K>(&self, bimap: &B) -> SpatialQuery<Tgt, B, K>
    where
        B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
        K: Eq + Hash;

    fn resolve<B, K>(
        &mut self,
        bimap: &B,
        table_key: &K,
        bottom: &[BimapSInt],
        top: &[BimapSInt],
        normalized_cost: Option<&NormalizedCost>,
    ) where
        B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
        K: Clone + Eq + Hash;

    fn resolve_unmemoizable_dependency(&mut self, spec: &Spec<Tgt>, result: &ActionCostVec);

    fn finalize(self);
}
