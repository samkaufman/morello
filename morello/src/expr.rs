use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub},
};

pub type NonAffineExpr<T> = AffineForm<NonAffine<T>>;

pub trait Bounds {
    /// The inclusive bounds of the value, if known.
    fn bounds(&self) -> Option<(u32, u32)> {
        None
    }

    fn as_constant(&self) -> Option<i32> {
        // TODO: Weird this returns i32, not u32. Unify types as u32 if possible.
        if let Some((lower_bound, upper_bound)) = self.bounds() {
            if lower_bound == upper_bound {
                return Some(lower_bound.try_into().unwrap());
            }
        }
        None
    }
}

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

pub trait Atom: Clone + Eq + Bounds {}

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

impl<T: Bounds> Bounds for AffineForm<T> {
    fn bounds(&self) -> Option<(u32, u32)> {
        let mut minimum = 0u32;
        let mut maximum: Option<u32> = None;
        for term in &self.0 {
            let (term_min, term_max) = term.1.bounds()?;
            let coeff = u32::try_from(term.0).unwrap();
            minimum = minimum.min(coeff * term_min);
            maximum = Some(maximum.unwrap_or(0).max(coeff * term_max));
        }
        // maximum is `None` if there are no terms. In this case, minimum is 0.
        let c = u32::try_from(self.1).unwrap();
        Some((minimum + c, maximum.unwrap_or(0) + c))
    }
}

impl<T: Bounds> Bounds for NonAffine<T> {
    fn bounds(&self) -> Option<(u32, u32)> {
        match self {
            NonAffine::Constant(v) => {
                let v = (*v).try_into().unwrap();
                Some((v, v))
            }
            NonAffine::Leaf(v) => v.bounds(),
            NonAffine::FloorDiv(v, d) => v.bounds().map(|(v_min, v_max)| (v_min / d, v_max / d)),
            NonAffine::Mod(v, m) => {
                v.bounds().map(|(_, v_max)| {
                    // TODO: Tighten (raise) the minimum bound.
                    let adjusted_max = v_max.min(m - 1);
                    (0, adjusted_max)
                })
            }
        }
    }
}

impl Bounds for String {}
impl Bounds for &str {}

// An AffineForm can sub. an R for its atoms if it contains terms which themselves yield AffineForms
// over R.
impl<T, R, RO> Substitute<R> for AffineForm<T>
where
    T: Substitute<R, Output = AffineForm<RO>> + Bounds,
    R: Clone + Eq,
    RO: Bounds + Eq,
{
    type Atom = T::Atom;
    type Output = AffineForm<RO>;

    fn map_vars(mut self, mapper: &mut impl FnMut(Self::Atom) -> R) -> Self::Output {
        let mut accum = AffineForm(vec![], self.1);
        for Term(c, s) in self.0.drain(..) {
            // Flatten AffineForms resulting from the substitution.
            let subbed = s.map_vars(mapper);
            if subbed.as_constant() != Some(0) {
                accum += subbed * c;
            }
        }
        accum
    }
}

impl<T, R, RO> Substitute<R> for NonAffine<T>
where
    T: Substitute<R, Output = NonAffineExpr<RO>> + Bounds,
    R: Clone + Eq,
    RO: Bounds + Eq,
{
    type Atom = T::Atom;
    type Output = NonAffineExpr<RO>;

    fn map_vars(self, mapper: &mut impl FnMut(Self::Atom) -> R) -> Self::Output {
        match self {
            NonAffine::Constant(c) => NonAffineExpr::constant(c),
            NonAffine::Leaf(v) => v.map_vars(mapper),
            NonAffine::FloorDiv(v, d) => {
                let subbed = v.map_vars(mapper);
                if let Some(c) = subbed.as_constant() {
                    NonAffineExpr::constant(c / i32::try_from(d).unwrap())
                } else {
                    AffineForm::from(NonAffine::FloorDiv(Box::new(subbed), d))
                }
            }
            NonAffine::Mod(v, m) => {
                let subbed = v.map_vars(mapper);
                if let Some(c) = subbed.as_constant() {
                    NonAffineExpr::constant(c % i32::try_from(m).unwrap())
                } else {
                    AffineForm::from(NonAffine::Mod(Box::new(subbed), m))
                }
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

impl<T> PartialEq<i32> for AffineForm<T> {
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
        self *= rhs;
        self
    }
}

impl<T> MulAssign<i32> for AffineForm<T> {
    fn mul_assign(&mut self, rhs: i32) {
        self.0.iter_mut().for_each(|Term(c, _)| *c *= rhs);
        self.1 *= rhs;
    }
}

impl<T> Div<u32> for AffineForm<NonAffine<T>> {
    type Output = Self;

    fn div(self, rhs: u32) -> Self::Output {
        debug_assert_ne!(rhs, 0);
        if rhs == 1 {
            self
        } else {
            NonAffine::FloorDiv(Box::new(self), rhs).into()
        }
    }
}

impl<T> DivAssign<u32> for AffineForm<NonAffine<T>> {
    fn div_assign(&mut self, rhs: u32) {
        *self = std::mem::take(self) / rhs
    }
}

impl<T> Rem<u32> for AffineForm<NonAffine<T>> {
    type Output = Self;

    fn rem(mut self, rhs: u32) -> Self::Output {
        if self.0.is_empty() {
            AffineForm::constant(self.1 % i32::try_from(rhs).unwrap())
        } else if self.0.len() == 1 && self.0[0].0 == 1 && self.1 == 0 {
            (self.0.pop().unwrap().1 % rhs).into()
        } else {
            NonAffine::Mod(Box::new(self), rhs).into()
        }
    }
}

impl<T> Rem<u32> for NonAffine<T> {
    type Output = Self;

    fn rem(self, rhs: u32) -> Self::Output {
        match self {
            // TODO: Below i32 cast shouldn't be necesary. May make more sense to use i32 for Mod.
            NonAffine::Constant(c) => NonAffine::Constant(c % i32::try_from(rhs).unwrap()),
            leaf @ NonAffine::Leaf(_) => NonAffine::Mod(Box::new(leaf.into()), rhs),
            // e.g., (a % 8) % 2 == a % 2
            NonAffine::Mod(a, r) if r % rhs == 0 => NonAffine::Mod(a, rhs),
            // e.g., (a % 2) % 8 == a % 2
            NonAffine::Mod(a, r) if rhs % r == 0 => NonAffine::Mod(a, r),
            other => NonAffine::Mod(Box::new(other.into()), rhs),
        }
    }
}

impl<T> RemAssign<u32> for AffineForm<NonAffine<T>> {
    fn rem_assign(&mut self, rhs: u32) {
        *self = std::mem::take(self) % rhs
    }
}

impl<T> RemAssign<u32> for NonAffine<T> {
    fn rem_assign(&mut self, rhs: u32) {
        *self = std::mem::take(self) % rhs
    }
}

impl<T: Atom> From<T> for NonAffine<T> {
    fn from(t: T) -> Self {
        NonAffine::Leaf(t)
    }
}

impl<T> Default for AffineForm<T> {
    fn default() -> Self {
        AffineForm::zero()
    }
}

impl<T> Default for NonAffine<T> {
    fn default() -> Self {
        NonAffine::Constant(0)
    }
}

impl<T: Display> Display for AffineForm<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Some((first_term, rest_terms)) = self.0.split_first() else {
            return write!(f, "{}", self.1);
        };

        write_affine_term(f, first_term)?;
        for t in rest_terms {
            write!(f, " + ")?;
            write_affine_term(f, t)?;
        }
        if self.1 != 0 {
            write!(f, " + {}", self.1)?;
        }
        Ok(())
    }
}

impl<T: Display> Display for NonAffine<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NonAffine::Constant(v) => write!(f, "{}", v),
            NonAffine::Leaf(v) => write!(f, "{}", v),
            NonAffine::FloorDiv(v, d) => write!(f, "({}) / {}", v, d),
            NonAffine::Mod(v, m) => write!(f, "({}) % {}", v, m),
        }
    }
}

impl Atom for String {}
impl Atom for &str {}

fn write_affine_term<T: Display>(f: &mut std::fmt::Formatter<'_>, t: &Term<T>) -> std::fmt::Result {
    if t.0 == 1 {
        write!(f, "({})", t.1)
    } else {
        write!(f, "{}({})", t.0, t.1)
    }
}

#[cfg(test)]
mod tests {
    use crate::expr::{NonAffineExpr, Substitute};

    use super::{AffineForm, NonAffine, Term};

    #[test]
    fn test_intercept_scalar_addition() {
        assert_eq!(
            AffineForm::<&str>(vec![], 1) + 2,
            AffineForm::<&str>(vec![], 3)
        );
        assert_eq!(
            AffineForm::<&str>(vec![], 1) + 0,
            AffineForm::<&str>(vec![], 1)
        );
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn test_intercept_scalar_multiplication() {
        assert_eq!(
            AffineForm::<&str>(vec![], 1) * 2,
            AffineForm::<&str>(vec![], 2)
        );
        assert_eq!(
            AffineForm::<&str>(vec![], 1) * 0,
            AffineForm::<&str>(vec![], 0)
        );
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

    #[test]
    fn test_subs_div() {
        let x = NonAffine::Leaf("x");
        let original_expr = AffineForm(
            vec![Term(1, NonAffine::FloorDiv(Box::new(x.into()), 16))],
            0,
        );

        let result_1 = original_expr.clone().subs(&"x", &AffineForm::constant(1));
        assert_eq!(result_1, AffineForm::constant(0));

        let result_2 = original_expr.subs(&"x", &AffineForm::constant(16));
        assert_eq!(result_2, AffineForm::constant(1));
    }

    #[test]
    fn test_subs_mod_1() {
        let x = NonAffine::Leaf("x");
        let original_expr = AffineForm(vec![Term(1, NonAffine::Mod(Box::new(x.into()), 16))], 0);

        let result_1 = original_expr.clone().subs(&"x", &AffineForm::constant(1));
        assert_eq!(result_1, AffineForm::constant(1));

        let result_2 = original_expr.subs(&"x", &AffineForm::constant(16));
        assert_eq!(result_2, AffineForm::constant(0));
    }

    #[test]
    fn test_subs_mod_2() {
        let root = NonAffineExpr::<()>::constant(0) % 8;
        assert_eq!(root, 0i32);
    }
}
