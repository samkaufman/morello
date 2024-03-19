use std::ops::Index;

use crate::scheduling::Action;
use crate::target::Target;

pub trait ActionSeq<Tgt>:
    IntoIterator<Item = Action<Tgt>> + Index<usize, Output = Action<Tgt>>
where
    Tgt: Target,
    Self::IntoIter: ExactSizeIterator<Item = Action<Tgt>>,
{
}

// Ducktypin' any T which satisfies the bounds.
impl<T, Tgt> ActionSeq<Tgt> for T
where
    T: IntoIterator<Item = Action<Tgt>>,
    T: Index<usize, Output = Action<Tgt>>,
    Tgt: Target,
    T::IntoIter: ExactSizeIterator,
{
}
