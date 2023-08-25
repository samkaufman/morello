use itertools::Either;
use std::ops::{Add, AddAssign, Mul, Sub};

pub type NonAffineExpr<T> = AffineForm<NonAffine<T>>;

pub trait Substitute<R> {
    type Atom: Atom;
    type Output;

    fn subs(self, atom: &Self::Atom, replacement: &R) -> Self::Output
    where
        Self: Sized,
        R: Clone + From<Self::Atom>,
    {
        self.map_vars(&mut |a| {
            if atom == &a {
                replacement.clone()
            } else {
                a.into()
            }
        })
    }

    fn map_vars(self, mapper: &mut impl FnMut(Self::Atom) -> R) -> Self::Output;
}

pub trait Atom: Clone + Eq {}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AffineForm<T>(pub Vec<Term<T>>, pub i32);

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Term<T>(pub i32, pub T);

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum NonAffine<T> {
    // TODO: `Constant` should be a kind of leaf
    Constant(i32),
    Leaf(T),
    FloorDiv(Box<NonAffineExpr<T>>, u32),
    Mod(Box<NonAffineExpr<T>>, u32),
}

impl<T> AffineForm<T> {
    pub const fn zero() -> Self {
        AffineForm(vec![], 0)
    }

    pub const fn constant(c: i32) -> Self {
        AffineForm(vec![], c)
    }
}

impl<T: PartialEq> AffineForm<T> {
    pub fn map_terms<U, F>(mut self, mut mapper: F) -> AffineForm<U>
    where
        F: FnMut(T) -> Either<U, i32>,
        U: PartialEq,
    {
        let mut accum = AffineForm(vec![], self.1);
        for Term(c, s) in self.0.drain(..) {
            match mapper(s) {
                Either::Left(u) => accum += Term(c, u),
                Either::Right(i) => accum += c * i,
            }
        }
        accum
    }
}

// An AffineForm can sub. an R for its atoms if it contains terms which themselves yield AffineForms
// over R.
impl<T, R, RO> Substitute<R> for AffineForm<T>
where
    T: Substitute<R, Output = AffineForm<RO>>,
    R: Clone + Eq,
    RO: Eq,
{
    type Atom = T::Atom;
    type Output = AffineForm<RO>;

    fn map_vars(mut self, mapper: &mut impl FnMut(Self::Atom) -> R) -> Self::Output {
        let mut accum = AffineForm(vec![], self.1);
        for Term(c, s) in self.0.drain(..) {
            // Flatten AffineForms resulting from the substitution.
            accum += s.map_vars(mapper) * c;
        }
        accum
    }
}

impl<T, R, RO> Substitute<R> for NonAffine<T>
where
    T: Substitute<R, Output = NonAffineExpr<RO>>,
    R: Clone + Eq,
    RO: Eq,
{
    type Atom = T::Atom;
    type Output = NonAffineExpr<RO>;

    fn map_vars(self, mapper: &mut impl FnMut(Self::Atom) -> R) -> Self::Output {
        match self {
            NonAffine::Constant(c) => NonAffineExpr::constant(c),
            NonAffine::Leaf(v) => v.map_vars(mapper),
            NonAffine::FloorDiv(v, d) => {
                AffineForm::from(NonAffine::FloorDiv(Box::new(v.map_vars(mapper)), d))
            }
            NonAffine::Mod(v, m) => {
                AffineForm::from(NonAffine::Mod(Box::new(v.map_vars(mapper)), m))
            }
        }
    }
}

// Any type implementing `Atom` can be replaced with anything into which that type can be converted.
// The implementation just checks equality and returns either the (converted) atom or the
// replacement.
impl<T: Atom, R: Clone> Substitute<R> for T {
    type Atom = T;
    type Output = R;

    fn map_vars(self, mapper: &mut impl FnMut(Self::Atom) -> R) -> R {
        mapper(self)
    }
}

impl<T> From<T> for AffineForm<T> {
    fn from(t: T) -> Self {
        AffineForm(vec![Term(1, t)], 0)
    }
}

impl<T> From<Term<T>> for AffineForm<T> {
    fn from(t: Term<T>) -> Self {
        AffineForm(vec![t], 0)
    }
}

impl<T: Atom> From<T> for NonAffineExpr<T> {
    fn from(t: T) -> Self {
        AffineForm(vec![Term(1, NonAffine::Leaf(t))], 0)
    }
}

impl<T: Atom> From<Option<T>> for NonAffineExpr<T> {
    fn from(t: Option<T>) -> Self {
        match t {
            Some(t) => AffineForm(vec![Term(1, NonAffine::Leaf(t))], 0),
            None => NonAffineExpr::constant(0),
        }
    }
}

impl<T> PartialEq<i32> for &AffineForm<T> {
    fn eq(&self, rhs: &i32) -> bool {
        self.0.is_empty() && self.1 == *rhs
    }
}

impl<T: PartialEq> Add for AffineForm<T> {
    type Output = Self;

    fn add(mut self, rhs: AffineForm<T>) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: PartialEq> Add<Term<T>> for AffineForm<T> {
    type Output = Self;

    fn add(self, rhs: Term<T>) -> Self::Output {
        self + AffineForm::from(rhs)
    }
}

impl<T> Add<i32> for AffineForm<T> {
    type Output = Self;

    fn add(mut self, rhs: i32) -> Self::Output {
        self.1 += rhs;
        self
    }
}

impl<T: PartialEq> AddAssign for AffineForm<T> {
    fn add_assign(&mut self, rhs: Self) {
        // TODO: Keep the terms sorted, then join by merging (popping smaller head).
        let AffineForm(terms, intercept) = self;
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

impl<T: PartialEq> AddAssign<Term<T>> for AffineForm<T> {
    fn add_assign(&mut self, rhs: Term<T>) {
        self.add_assign(AffineForm::from(rhs));
    }
}

impl<T> AddAssign<i32> for AffineForm<T> {
    fn add_assign(&mut self, rhs: i32) {
        self.1 += rhs;
    }
}

impl<T> Sub<i32> for AffineForm<T> {
    type Output = Self;

    fn sub(mut self, rhs: i32) -> Self::Output {
        self.1 -= rhs;
        self
    }
}

impl<T> Mul<i32> for AffineForm<T> {
    type Output = Self;

    fn mul(mut self, rhs: i32) -> Self::Output {
        self.0.iter_mut().for_each(|Term(c, _)| *c *= rhs);
        self.1 *= rhs;
        self
    }
}

impl<T: Atom> From<T> for NonAffine<T> {
    fn from(t: T) -> Self {
        NonAffine::Leaf(t)
    }
}

impl Atom for String {}
impl Atom for &str {}

#[cfg(test)]
mod tests {
    use crate::expr::Substitute;

    use super::{AffineForm, NonAffine, Term};

    #[test]
    fn test_intercept_scalar_addition() {
        assert_eq!(AffineForm::<()>(vec![], 1) + 2, AffineForm::<()>(vec![], 3));
        assert_eq!(AffineForm::<()>(vec![], 1) + 0, AffineForm::<()>(vec![], 1));
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn test_intercept_scalar_multiplication() {
        assert_eq!(AffineForm::<()>(vec![], 1) * 2, AffineForm::<()>(vec![], 2));
        assert_eq!(AffineForm::<()>(vec![], 1) * 0, AffineForm::<()>(vec![], 0));
    }

    #[test]
    fn test_substitute_affine_expr_var_for_affine_expr() {
        let e = AffineForm(
            vec![Term(2, String::from("x")), Term(4, String::from("y"))],
            1,
        );
        let replacement = AffineForm(
            vec![Term(1, String::from("y")), Term(2, String::from("z"))],
            1,
        );
        let expected = AffineForm(
            vec![
                Term(2, String::from("x")),
                Term(4, String::from("y")),
                Term(8, String::from("z")),
            ],
            5,
        );
        let result = e.subs(&String::from("y"), &replacement);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_substitute_non_affine_expr_var_for_non_affine_term() {
        let original_expr = AffineForm(vec![Term(1, NonAffine::Leaf("x"))], 1);
        let replacement = AffineForm::from(NonAffine::FloorDiv(
            Box::new(NonAffine::Leaf("y").into()),
            2,
        ));

        let result = original_expr.subs(&"x", &replacement);

        let expected = AffineForm(
            vec![Term(
                1,
                NonAffine::FloorDiv(Box::new(NonAffine::Leaf("y").into()), 2),
            )],
            1,
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_substitute_non_affine_expr_var_for_affine_expr_1() {
        let x = NonAffine::Leaf("x");
        let y = NonAffine::Leaf("y");
        let z = NonAffine::Leaf("z");

        let original_expr = AffineForm(vec![Term(1, x)], 1);
        let result = original_expr.subs(
            &"x",
            &AffineForm(vec![Term(1, y.clone()), Term(2, z.clone())], 1),
        );
        assert_eq!(result, AffineForm(vec![Term(1, y), Term(2, z)], 2,));
    }

    #[test]
    fn test_substitute_non_affine_expr_var_for_affine_expr_2() {
        let x = NonAffine::Leaf("x");
        let y = NonAffine::Leaf("y");

        let original_expr =
            AffineForm(vec![Term(2, NonAffine::FloorDiv(Box::new(x.into()), 4))], 1);
        let result = original_expr.subs(
            &"x",
            &AffineForm::from(NonAffine::FloorDiv(Box::new(y.clone().into()), 2)),
        );
        assert_eq!(
            result,
            AffineForm(
                vec![Term(
                    2,
                    NonAffine::FloorDiv(
                        Box::new(NonAffine::FloorDiv(Box::new(y.into()), 2).into()),
                        4
                    )
                )],
                1,
            )
        );
    }

    #[test]
    fn test_substitute_non_affine_expr_var_for_affine_expr_3() {
        let x = NonAffine::Leaf("x");
        let y = NonAffine::Leaf("y");

        let original_expr = AffineForm(vec![Term(2, NonAffine::Mod(Box::new(x.into()), 4))], 1);
        let result = original_expr.subs(
            &"x",
            &AffineForm::from(NonAffine::Mod(Box::new(y.clone().into()), 2)),
        );
        assert_eq!(
            result,
            AffineForm(
                vec![Term(
                    2,
                    NonAffine::Mod(Box::new(NonAffine::Mod(Box::new(y.into()), 2).into()), 4)
                )],
                1,
            )
        );
    }
}
