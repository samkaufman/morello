use super::general::Bimap;

pub type BimapInt = u32;

#[derive(Debug, Clone, Copy)]
pub struct BoolBimap;

#[derive(Debug, Clone, Copy, Default)]
pub struct AddBimap<T>(T)
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>;

impl Bimap for BoolBimap {
    type Domain = bool;
    type Codomain = BimapInt;
    type DomainIter = std::iter::Once<bool>;

    fn apply(&self, b: &bool) -> BimapInt {
        if *b {
            1
        } else {
            0
        }
    }

    fn apply_inverse(&self, i: &BimapInt) -> Self::DomainIter {
        std::iter::once(*i != 0)
    }
}

impl<T> Bimap for AddBimap<T>
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>,
{
    type Domain = T;
    type Codomain = BimapInt;
    type DomainIter = std::iter::Once<T>;

    fn apply(&self, v: &T) -> BimapInt {
        *v + self.0
    }

    fn apply_inverse(&self, i: &BimapInt) -> Self::DomainIter {
        std::iter::once(*i - self.0)
    }
}
