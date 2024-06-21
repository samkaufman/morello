use super::{
    general::{BiMapExt, SurMap},
    tablemeta::{DimensionType, TableMeta},
};

/// A [BiMap] which applies the wrapped [BiMap] to each element of a [Vec].
pub struct MapBiMap<T>(T);

impl<T: BiMapExt> SurMap for MapBiMap<T> {
    type Domain = Vec<T::Domain>;
    type Codomain = Vec<T::Codomain>;
    type DomainIter = [Self::Domain; 1];

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        t.iter().map(|t| self.0.apply(t)).collect()
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        [i.iter().map(|t| self.0.invert(t)).collect()]
    }
}

impl<T> TableMeta for MapBiMap<T>
where
    T: TableMeta + BiMapExt,
{
    fn dimension_types(&self, input: &Self::Domain) -> Vec<DimensionType> {
        input
            .iter()
            .flat_map(|t| self.0.dimension_types(t))
            .collect()
    }
}
