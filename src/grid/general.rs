pub trait Bimap {
    type Domain;
    type Codomain;
    type DomainIter: Iterator<Item = Self::Domain>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain;
    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter;
}
