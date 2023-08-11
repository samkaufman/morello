use itertools::Either;
use std::{
    num::TryFromIntError,
    ops::{Add, AddAssign, Mul, Sub},
};

#[derive(Debug, PartialEq, Clone)]
pub struct AffineExpr<T>(pub Vec<Term<T>>, pub i32);

#[derive(Debug, PartialEq, Clone)]
pub struct Term<T>(pub i32, pub T);

impl<T: PartialEq> AffineExpr<T> {
    pub fn subs(mut self, symbol: &T, replacement: AffineExpr<T>) -> AffineExpr<T> {
        match self.0.iter().position(|t| &t.1 == symbol) {
            Some(idx) => {
                let coef = self.0[idx].0;
                self.0.swap_remove(idx);
                self + (replacement * coef)
            }
            None => self,
        }
    }

    pub fn map_terms<U, F>(mut self, mut mapper: F) -> AffineExpr<U>
    where
        F: FnMut(T) -> Either<U, i32>,
        U: PartialEq,
    {
        let mut accum = AffineExpr(vec![], self.1);
        for Term(c, s) in self.0.drain(..) {
            match mapper(s) {
                Either::Left(u) => accum += Term(c, u),
                Either::Right(i) => accum += c * i,
            }
        }
        accum
    }
}

impl<T> PartialEq<i32> for &AffineExpr<T> {
    fn eq(&self, rhs: &i32) -> bool {
        self.0.is_empty() && self.1 == *rhs
    }
}

impl<T: PartialEq> Add for AffineExpr<T> {
    type Output = Self;

    fn add(mut self, rhs: AffineExpr<T>) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: PartialEq> Add<Term<T>> for AffineExpr<T> {
    type Output = Self;

    fn add(self, rhs: Term<T>) -> Self::Output {
        self + AffineExpr::from(rhs)
    }
}

impl<T> Add<i32> for AffineExpr<T> {
    type Output = Self;

    fn add(mut self, rhs: i32) -> Self::Output {
        self.1 += rhs;
        self
    }
}

impl<T: PartialEq> AddAssign for AffineExpr<T> {
    fn add_assign(&mut self, rhs: Self) {
        // TODO: Keep the terms sorted, then join by merging (popping smaller head).
        let AffineExpr(terms, intercept) = self;
        *intercept += rhs.1;
        for Term(c, s) in rhs.0 {
            if let Some(Term(c2, _)) = terms.iter_mut().find(|Term(_, s2)| &s == s2) {
                *c2 += c;
            } else {
                terms.push(Term(c, s));
            }
        }
    }
}

impl<T: PartialEq> AddAssign<Term<T>> for AffineExpr<T> {
    fn add_assign(&mut self, rhs: Term<T>) {
        self.add_assign(AffineExpr::from(rhs));
    }
}

impl<T> AddAssign<i32> for AffineExpr<T> {
    fn add_assign(&mut self, rhs: i32) {
        self.1 += rhs;
    }
}

impl<T> Sub<i32> for AffineExpr<T> {
    type Output = Self;

    fn sub(mut self, rhs: i32) -> Self::Output {
        self.1 -= rhs;
        self
    }
}

impl<T> Mul<i32> for AffineExpr<T> {
    type Output = Self;

    fn mul(mut self, rhs: i32) -> Self::Output {
        self.0.iter_mut().for_each(|Term(c, _)| *c *= rhs);
        self.1 *= rhs;
        self
    }
}

impl<T> From<Term<T>> for AffineExpr<T> {
    fn from(t: Term<T>) -> Self {
        AffineExpr(vec![t], 0)
    }
}

impl<T> From<i32> for AffineExpr<T> {
    fn from(i: i32) -> Self {
        AffineExpr(vec![], i)
    }
}

impl<T> TryFrom<usize> for AffineExpr<T> {
    type Error = TryFromIntError;

    fn try_from(i: usize) -> Result<Self, Self::Error> {
        Ok(AffineExpr(vec![], i.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use super::{AffineExpr, Term};

    #[test]
    fn test_intercept_scalar_addition() {
        assert_eq!(AffineExpr::<()>(vec![], 1) + 2, AffineExpr::<()>(vec![], 3));
        assert_eq!(AffineExpr::<()>(vec![], 1) + 0, AffineExpr::<()>(vec![], 1));
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn test_intercept_scalar_multiplication() {
        assert_eq!(AffineExpr::<()>(vec![], 1) * 2, AffineExpr::<()>(vec![], 2));
        assert_eq!(AffineExpr::<()>(vec![], 1) * 0, AffineExpr::<()>(vec![], 0));
    }

    #[test]
    fn test_subs() {
        let e = AffineExpr(
            vec![Term(2, String::from("x")), Term(4, String::from("y"))],
            1,
        );
        let replacement = AffineExpr(
            vec![Term(1, String::from("y")), Term(2, String::from("z"))],
            1,
        );
        let expected = AffineExpr(
            vec![
                Term(2, String::from("x")),
                Term(4, String::from("y")),
                Term(8, String::from("z")),
            ],
            5,
        );
        assert_eq!(e.subs(&"y".to_string(), replacement), expected);
    }
}
