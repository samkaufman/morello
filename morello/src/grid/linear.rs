use super::general::SurMap;

pub type BimapInt = u32;

#[derive(Debug, Clone, Copy)]
pub struct BoolBimap;

// TODO: Rename to Shift and document
#[derive(Debug, Clone, Copy, Default)]
pub struct AddBimap<T>(T)
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>;

impl SurMap for BoolBimap {
    type Domain = bool;
    type Codomain = BimapInt;
    type DomainIter = [bool; 1];

    fn apply(&self, b: &bool) -> BimapInt {
        if *b {
            1
        } else {
            0
        }
    }

    fn apply_inverse(&self, i: &BimapInt) -> [bool; 1] {
        [*i != 0]
    }
}

impl<T> SurMap for AddBimap<T>
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>,
{
    type Domain = T;
    type Codomain = BimapInt;
    type DomainIter = [T; 1];

    fn apply(&self, v: &T) -> BimapInt {
        *v + self.0
    }

    fn apply_inverse(&self, i: &BimapInt) -> [T; 1] {
        [*i - self.0]
    }
}
