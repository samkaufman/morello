use super::general::BiMap;

pub type BimapInt = u32;

#[derive(Debug, Clone, Copy)]
pub struct BoolBimap;

// TODO: Rename to Shift and document
#[derive(Debug, Clone, Copy, Default)]
pub struct AddBimap<T>(T)
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>;

impl BiMap for BoolBimap {
    type Domain = bool;
    type Codomain = BimapInt;

    fn apply(&self, b: &bool) -> BimapInt {
        if *b {
            1
        } else {
            0
        }
    }

    fn apply_inverse(&self, i: &BimapInt) -> bool {
        *i != 0
    }
}

impl<T> BiMap for AddBimap<T>
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>,
{
    type Domain = T;
    type Codomain = BimapInt;

    fn apply(&self, v: &T) -> BimapInt {
        *v + self.0
    }

    fn apply_inverse(&self, i: &BimapInt) -> T {
        *i - self.0
    }
}
