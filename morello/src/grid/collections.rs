use super::general::BiMap;

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
