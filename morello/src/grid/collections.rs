use super::{
    general::{BiMap, SurMap},
    tablemeta::{DimensionType, TableMeta},
};

/// A [BiMap] which applies the wrapped [BiMap] to each element of a [Vec].
pub struct MapBiMap<T>(T);

impl<T: BiMap> BiMap for MapBiMap<T> {
    type Domain = Vec<<T as BiMap>::Domain>;
    type Codomain = Vec<<T as BiMap>::Codomain>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        t.iter().map(|t| <T as BiMap>::apply(&self.0, t)).collect()
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        i.iter()
            .map(|t| <T as BiMap>::apply_inverse(&self.0, t))
            .collect()
    }
}

impl<T> TableMeta for MapBiMap<T>
where
    T: TableMeta + BiMap<Domain = <T as SurMap>::Domain>,
{
    fn dimension_types(&self, input: &Self::Domain) -> Vec<DimensionType> {
        input
            .iter()
            .flat_map(|t| self.0.dimension_types(t))
            .collect()
    }
}
