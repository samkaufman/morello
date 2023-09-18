use crate::grid::general::IntBiMap;

pub type BimapInt = u32;

#[derive(Debug, Clone, Copy)]
pub struct BoolBimap;

#[derive(Debug, Clone, Copy, Default)]
pub struct AddBimap<T>(T)
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>;

impl IntBiMap for BoolBimap {
    type Domain = bool;

    fn apply(&self, b: &bool) -> BimapInt {
        if *b {
            1
        } else {
            0
        }
    }

    fn apply_inverse(&self, i: BimapInt) -> bool {
        i != 0
    }
}

impl<T> IntBiMap for AddBimap<T>
where
    T: std::ops::Add<Output = BimapInt> + Copy,
    BimapInt: std::ops::Sub<T, Output = T>,
{
    type Domain = T;

    fn apply(&self, v: &T) -> BimapInt {
        *v + self.0
    }

    fn apply_inverse(&self, i: BimapInt) -> T {
        i - self.0
    }
}
