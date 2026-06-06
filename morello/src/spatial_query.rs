use crate::grid::general::BiMap;
use crate::grid::linear::BimapInt;
use crate::rtree::{RTreeDyn, RTreeInt};
use crate::spec::Spec;
use crate::target::{Target, MEMORY_COUNT};
use crate::utils::multi_range_product;
use itertools::Itertools;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct SpatialQuery<Tgt: Target, B, K> {
    tables: HashMap<K, HashMap<Vec<BimapInt>, RTreeDyn<()>>>,
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
        self.tables
            .values()
            .flat_map(HashMap::values)
            .map(RTreeDyn::size)
            .sum()
    }

    pub(crate) fn page_tables(&self) -> impl Iterator<Item = (&K, &[BimapInt], &RTreeDyn<()>)> {
        self.tables
            .iter()
            .flat_map(move |(table_key, page_tables)| {
                page_tables
                    .iter()
                    .map(move |(page_point, tree)| (table_key, page_point.as_slice(), tree))
            })
    }

    pub(crate) fn insert_point(&mut self, table_key: K, rank: usize, point: &[BimapInt]) {
        assert_eq!(point.len(), rank);
        let page_point = blockify_point(point);
        let local_point = localize_point(point, &page_point);
        if !self.contains_point(&table_key, point) {
            self.table_mut(table_key, page_point, rank).merge_insert(
                &local_point,
                &local_point,
                (),
                true,
            );
        }
    }

    pub fn insert_rect(
        &mut self,
        table_key: K,
        rank: usize,
        bottom: &[BimapInt],
        top: &[BimapInt],
    ) {
        assert_eq!(bottom.len(), rank);
        assert_eq!(top.len(), rank);
        let page_bottom = blockify_point(bottom);
        let page_top = blockify_point(top);
        let page_tables = self.tables.entry(table_key).or_default();

        let mut local_bottom = Vec::with_capacity(rank);
        let mut local_top = Vec::with_capacity(rank);
        multi_range_product(&page_bottom, &page_top, |page_point: &[BimapInt]| {
            local_bottom.clear();
            local_top.clear();
            for (dim, &page_coord) in page_point.iter().enumerate() {
                let block_dim_size = block_size_dim(dim, rank);
                let block_start = page_coord * block_dim_size;
                let start = block_start.max(bottom[dim]);
                // Avoid `(page_coord + 1) * block_dim_size`: the final page can sit at the top of
                // BimapInt's range, so the mathematical exclusive end may overflow.
                let end_inclusive = block_end_inclusive(block_start, block_dim_size).min(top[dim]);
                debug_assert!(start <= end_inclusive);

                local_bottom.push(local_coord_to_rtree(start - block_start));
                local_top.push(local_coord_to_rtree(end_inclusive - block_start));
            }

            page_table_mut(page_tables, page_point.to_vec(), rank).merge_insert(
                &local_bottom,
                &local_top,
                (),
                true,
            );
        });
    }

    pub(crate) fn contains_point(&self, table_key: &K, point: &[BimapInt]) -> bool {
        let page_point = blockify_point(point);
        let local_point = localize_point(point, &page_point);
        self.tables
            .get(table_key)
            .and_then(|page_tables| page_tables.get(&page_point))
            .is_some_and(|table| table.locate_at_point_int(&local_point).is_some())
    }

    fn table_mut(
        &mut self,
        table_key: K,
        page_point: Vec<BimapInt>,
        rank: usize,
    ) -> &mut RTreeDyn<()> {
        let page_tables = self.tables.entry(table_key).or_default();
        page_table_mut(page_tables, page_point, rank)
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
            self.contains_point(&table_key, &global_pt)
        } else {
            self.unmemoizable_specs.contains(spec)
        }
    }

    pub fn memoizable_specs<'a>(&'a self, bimap: &'a B) -> impl Iterator<Item = Spec<Tgt>> + 'a
    where
        K: Clone,
    {
        self.tables
            .iter()
            .flat_map(move |(table_key, page_tables)| {
                page_tables.iter().flat_map(move |(page_point, tree)| {
                    let rank = tree.dim_count();
                    tree.iter().flat_map(move |(bottom, top, ())| {
                        bottom
                            .iter()
                            .zip(top)
                            .enumerate()
                            .map(move |(dim, (&bottom, &top))| {
                                let block_start = page_point[dim] * block_size_dim(dim, rank);
                                let bottom = block_start + BimapInt::try_from(bottom).unwrap();
                                let top = block_start + BimapInt::try_from(top).unwrap();
                                bottom..=top
                            })
                            .multi_cartesian_product()
                            .map(|global_pt| {
                                BiMap::apply_inverse(bimap, &(table_key.clone(), global_pt))
                            })
                    })
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

fn page_table_mut(
    page_tables: &mut HashMap<Vec<BimapInt>, RTreeDyn<()>>,
    page_point: Vec<BimapInt>,
    rank: usize,
) -> &mut RTreeDyn<()> {
    page_tables
        .entry(page_point)
        .and_modify(|table| assert_eq!(table.dim_count(), rank))
        .or_insert_with(|| RTreeDyn::empty(rank))
}

fn block_size_dim(dim: usize, dim_count: usize) -> BimapInt {
    if dim >= dim_count.saturating_sub(MEMORY_COUNT) {
        31
    } else {
        8
    }
}

#[inline]
fn block_end_inclusive(block_start: BimapInt, block_dim_size: BimapInt) -> BimapInt {
    block_start.saturating_add(block_dim_size - 1)
}

fn blockify_point(pt: &[BimapInt]) -> Vec<BimapInt> {
    let rank = pt.len();
    pt.iter()
        .enumerate()
        .map(|(dim, &coord)| coord / block_size_dim(dim, rank))
        .collect()
}

fn localize_point(pt: &[BimapInt], page_pt: &[BimapInt]) -> Vec<RTreeInt> {
    debug_assert_eq!(pt.len(), page_pt.len());
    let rank = pt.len();
    pt.iter()
        .zip(page_pt)
        .enumerate()
        .map(|(dim, (&coord, &page_coord))| {
            local_coord_to_rtree(coord - page_coord * block_size_dim(dim, rank))
        })
        .collect()
}

fn local_coord_to_rtree(coord: BimapInt) -> RTreeInt {
    RTreeInt::try_from(coord)
        .expect("spatial query page-local coordinate exceeded R-tree precision")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Avx2Target;

    type TestQuery = SpatialQuery<Avx2Target, (), ()>;

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
