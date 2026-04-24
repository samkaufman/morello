use crate::grid::general::BiMap;
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::rtree::RTreeDyn;
use crate::spec::Spec;
use crate::target::Target;
use itertools::Itertools;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct SpatialQuery<Tgt: Target, B, K> {
    tables: HashMap<K, RTreeDyn<()>>,
    /// Dependencies that cannot be represented in `tables` because they are outside the
    /// bimap's domain.
    unmemoizable_specs: Vec<Spec<Tgt>>,
    marker: PhantomData<B>, // Carry `B` to enforce consistency across calls
}

impl<Tgt, B, K> SpatialQuery<Tgt, B, K>
where
    Tgt: Target,
    K: Eq + Hash,
{
    #[cfg(test)]
    pub fn rect_count(&self) -> usize {
        self.tables.values().map(RTreeDyn::size).sum()
    }

    pub(crate) fn tables(&self) -> impl Iterator<Item = (&K, &RTreeDyn<()>)> {
        self.tables.iter()
    }

    pub(crate) fn unmemoizable_specs(&self) -> &[Spec<Tgt>] {
        &self.unmemoizable_specs
    }

    pub(crate) fn subtract_rect(&mut self, table_key: &K, bottom: &[BimapSInt], top: &[BimapSInt]) {
        if let Some(table) = self.tables.get_mut(table_key) {
            table.subtract(bottom, top);
        }
    }

    pub(crate) fn insert_point(&mut self, table_key: K, rank: usize, point: &[BimapInt]) {
        let point = point
            .iter()
            .map(|&coord| BimapSInt::from(coord))
            .collect::<Vec<_>>();
        if !self.contains_point(&table_key, &point) {
            self.table_mut(table_key, rank)
                .merge_insert(&point, &point, (), true);
        }
    }

    pub fn insert_rect(
        &mut self,
        table_key: K,
        rank: usize,
        bottom: &[BimapInt],
        top: &[BimapInt],
    ) {
        let bottom = bottom
            .iter()
            .map(|&coord| BimapSInt::from(coord))
            .collect::<Vec<_>>();
        let top = top
            .iter()
            .map(|&coord| BimapSInt::from(coord))
            .collect::<Vec<_>>();
        self.table_mut(table_key, rank)
            .merge_insert(&bottom, &top, (), true);
    }

    pub(crate) fn contains_point(&self, table_key: &K, point: &[BimapSInt]) -> bool {
        self.tables
            .get(table_key)
            .is_some_and(|table| table.locate_at_point(point).is_some())
    }

    #[cfg(test)]
    pub(crate) fn rectangles_for_table(
        &self,
        table_key: &K,
    ) -> impl Iterator<Item = (&[BimapSInt], &[BimapSInt])> + '_ {
        self.tables
            .get(table_key)
            .into_iter()
            .flat_map(|tree| tree.iter().map(|(bottom, top, ())| (bottom, top)))
    }

    fn table_mut(&mut self, table_key: K, rank: usize) -> &mut RTreeDyn<()> {
        self.tables
            .entry(table_key)
            .and_modify(|table| assert_eq!(table.dim_count(), rank))
            .or_insert_with(|| RTreeDyn::empty(rank))
    }
}

impl<Tgt, B, K> SpatialQuery<Tgt, B, K>
where
    Tgt: Target,
    B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
    K: Eq + Hash,
{
    pub fn from_subspecs(bimap: &B, subspecs: impl IntoIterator<Item = Spec<Tgt>>) -> Self {
        let mut query = SpatialQuery::default();
        subspecs.into_iter().for_each(|subspec| {
            query.insert_spec(bimap, subspec);
        });
        query
    }

    pub fn insert_spec(&mut self, bimap: &B, spec: Spec<Tgt>) {
        assert!(spec.is_canonical());
        if BiMap::defined_for(bimap, &spec) {
            let (table_key, global_pt) = BiMap::apply(bimap, &spec);
            self.insert_point(table_key, global_pt.len(), &global_pt);
        } else {
            self.unmemoizable_specs.push(spec);
        }
    }

    pub fn insert_rectangle(
        &mut self,
        bimap: &B,
        lower_corner: &Spec<Tgt>,
        upper_corner: &Spec<Tgt>,
    ) {
        let mut lower_corner = lower_corner.clone();
        lower_corner
            .canonicalize()
            .expect("SpatialQuery rectangle lower corner must be canonicalizable");
        let mut upper_corner = upper_corner.clone();
        upper_corner
            .canonicalize()
            .expect("SpatialQuery rectangle upper corner must be canonicalizable");

        assert!(BiMap::defined_for(bimap, &lower_corner));
        assert!(BiMap::defined_for(bimap, &upper_corner));

        let (table_key, lower_pt) = BiMap::apply(bimap, &lower_corner);
        let (upper_table_key, upper_pt) = BiMap::apply(bimap, &upper_corner);
        assert!(table_key == upper_table_key); // no Debug bound, so use ==

        let bottom = lower_pt
            .iter()
            .zip(&upper_pt)
            .map(|(&lhs, &rhs)| lhs.min(rhs))
            .collect::<Vec<_>>();
        let top = lower_pt
            .iter()
            .zip(&upper_pt)
            .map(|(&lhs, &rhs)| lhs.max(rhs))
            .collect::<Vec<_>>();
        self.insert_rect(table_key, bottom.len(), &bottom, &top);
    }

    #[cfg(test)]
    pub fn contains(&self, bimap: &B, spec: &Spec<Tgt>) -> bool {
        if BiMap::defined_for(bimap, spec) {
            let (table_key, global_pt) = BiMap::apply(bimap, spec);
            let global_pt = global_pt
                .iter()
                .map(|&coord| BimapSInt::from(coord))
                .collect::<Vec<_>>();
            self.contains_point(&table_key, &global_pt)
        } else {
            self.unmemoizable_specs.contains(spec)
        }
    }

    pub fn memoizable_specs<'a>(&'a self, bimap: &'a B) -> impl Iterator<Item = Spec<Tgt>> + 'a
    where
        K: Clone,
    {
        self.tables.iter().flat_map(|(table_key, tree)| {
            tree.iter().flat_map(|(bottom, top, ())| {
                bottom
                    .iter()
                    .zip(top)
                    .map(|(&bottom, &top)| {
                        let bottom = BimapInt::try_from(bottom).unwrap();
                        let top = BimapInt::try_from(top).unwrap();
                        bottom..=top
                    })
                    .multi_cartesian_product()
                    .map(|global_pt| BiMap::apply_inverse(bimap, &(table_key.clone(), global_pt)))
            })
        })
    }
}

impl<Tgt, B, K> Clone for SpatialQuery<Tgt, B, K>
where
    Tgt: Target,
    K: Clone,
{
    fn clone(&self) -> Self {
        SpatialQuery {
            tables: self.tables.clone(),
            unmemoizable_specs: self.unmemoizable_specs.clone(),
            marker: PhantomData,
        }
    }
}

impl<Tgt: Target, B, K> Default for SpatialQuery<Tgt, B, K> {
    fn default() -> Self {
        SpatialQuery {
            tables: HashMap::new(),
            unmemoizable_specs: Vec::new(),
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Avx2Target;

    type TestQuery = SpatialQuery<Avx2Target, (), ()>;

    #[test]
    fn test_spatial_query_can_coalesce_adjacent_unit_rectangles() {
        let mut query = TestQuery::default();
        query.insert_point((), 2, &[0, 0]);
        query.insert_point((), 2, &[1, 0]);

        assert_eq!(query.rect_count(), 1);
        assert_eq!(
            query
                .rectangles_for_table(&())
                .map(|(bottom, top)| (bottom.to_vec(), top.to_vec()))
                .collect::<Vec<_>>(),
            vec![(vec![0, 0], vec![1, 0])]
        );
        assert!(query.contains_point(&(), &[0, 0]));
        assert!(query.contains_point(&(), &[1, 0]));
        assert!(!query.contains_point(&(), &[2, 0]));
    }

    #[test]
    fn test_spatial_query_deduplicates_points() {
        let mut query = TestQuery::default();
        query.insert_point((), 2, &[0, 0]);
        query.insert_point((), 2, &[0, 0]);

        assert_eq!(query.rect_count(), 1);
        assert!(query.contains_point(&(), &[0, 0]));
    }

    #[test]
    fn test_spatial_query_contains_point_checks_rect_membership() {
        let mut query = TestQuery::default();
        query.insert_point((), 2, &[0, 0]);
        query.insert_point((), 2, &[1, 0]);

        assert!(query.contains_point(&(), &[0, 0]));
        assert!(query.contains_point(&(), &[1, 0]));
        assert!(!query.contains_point(&(), &[2, 0]));
        assert!(!query.contains_point(&(), &[0, 1]));
    }
}
