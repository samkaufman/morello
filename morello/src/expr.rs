use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub},
};

pub type NonAffineExpr<T> = AffineForm<NonAffine<T>>;

pub trait Bounds {
    /// The inclusive bounds of the value, if known.
    fn bounds(&self) -> Option<(i32, i32)> {
        None
    }

    fn as_constant(&self) -> Option<i32> {
        // TODO: Weird this returns i32, not u32. Unify types as u32 if possible.
        if let Some((lower_bound, upper_bound)) = self.bounds() {
            if lower_bound == upper_bound {
                return Some(lower_bound);
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
    FloorDiv(Box<NonAffineExpr<T>>, i32),
    Mod(Box<NonAffineExpr<T>>, i32),
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
    fn bounds(&self) -> Option<(i32, i32)> {
        let mut minimum = self.1;
        let mut maximum = Some(self.1);
        for Term(coeff, sym) in &self.0 {
            let (sym_min, sym_max) = sym.bounds()?;
            if *coeff < 0 {
                // TODO: Convert the following to checked_add and checked_mul.
                minimum += *coeff * sym_max;
                maximum = maximum.map(|m| m + (*coeff * sym_min));
            } else {
                // TODO: Convert the following to checked_add and checked_mul.
                minimum += *coeff * sym_min;
                maximum = maximum.map(|m| m + (*coeff * sym_max));
            }
        }
        // maximum is `None` if there are no terms. In this case, minimum is 0.
        Some((minimum, maximum.unwrap_or(0)))
    }
}

impl<T: Bounds> Bounds for NonAffine<T> {
    fn bounds(&self) -> Option<(i32, i32)> {
        match self {
            NonAffine::Constant(v) => Some((*v, *v)),
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
                let mut subbed = v.map_vars(mapper);
                subbed.simplify_div_with_bounds(d);
                subbed
            }
            NonAffine::Mod(v, m) => {
                let mut subbed = v.map_vars(mapper);
                subbed.simplify_mod_with_bounds(m);
                subbed
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
        terms.retain(|Term(c, _)| *c != 0);
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
        self.0.retain(|Term(c, _)| *c != 0);
        self.1 *= rhs;
    }
}

impl<T> Div<i32> for AffineForm<NonAffine<T>> {
    type Output = Self;

    fn div(mut self, rhs: i32) -> Self::Output {
        debug_assert_ne!(rhs, 0);
        if rhs == 1 {
            self
        } else if self.0.is_empty() {
            AffineForm::constant(self.1 / rhs)
        } else if self.div_through(rhs) || self.div_distribute_if_divisor_doesnt_duplicate(rhs) {
            self
        } else {
            NonAffine::FloorDiv(Box::new(self), rhs).into()
        }
    }
}

impl<T> AffineForm<NonAffine<T>> {
    /// Divide all coefficients and intercept by `rhs` if all are divisible. Returns `true` if so.
    fn div_through(&mut self, rhs: i32) -> bool {
        if self.1 % rhs == 0 && self.0.iter().all(|Term(c, _)| c % rhs == 0) {
            self.0.iter_mut().for_each(|Term(c, _)| *c /= rhs);
            self.1 /= rhs;
            true
        } else {
            false
        }
    }

    /// If every term's coefficient is 1 or `rhs`, there is at most one coeff. equal to 1, and the
    /// intercept is divisible by `rhs`, then distribute and apply division to all terms/intercept.
    fn div_distribute_if_divisor_doesnt_duplicate(&mut self, rhs: i32) -> bool {
        // Short-circuit if any condition doesn't apply.
        if self.1 % rhs != 0 {
            return false;
        }
        let mut seen_one = false;
        for Term(c, _) in &self.0 {
            if *c == 1 {
                if seen_one {
                    return false;
                }
                seen_one = true;
            } else if *c != rhs {
                return false;
            }
        }

        // Apply transformation in place: divide intercept; turn rhs-coeff terms into coeff-1;
        // wrap the single coeff-1 term in an inner FloorDiv.
        self.1 /= rhs;
        for Term(c, s) in &mut self.0 {
            if *c == rhs {
                *c = 1;
            } else {
                debug_assert_eq!(*c, 1);
                let old = std::mem::take(s);
                *s = NonAffine::FloorDiv(Box::new(AffineForm(vec![Term(1, old)], 0)), rhs);
            }
        }
        true
    }
}

impl<T> DivAssign<i32> for AffineForm<NonAffine<T>> {
    fn div_assign(&mut self, rhs: i32) {
        *self = std::mem::take(self) / rhs
    }
}

impl<T> Rem<i32> for AffineForm<NonAffine<T>> {
    type Output = Self;

    fn rem(mut self, rhs: i32) -> Self::Output {
        assert_ne!(rhs, 0);
        if rhs == 1 {
            AffineForm::constant(0)
        } else if self.0.is_empty() {
            AffineForm::constant(self.1 % rhs)
        } else if self.0.len() == 1 && self.0[0].0 == 1 && self.1 == 0 {
            (self.0.pop().unwrap().1 % rhs).into()
        } else {
            let reduced_intercept = self.1 % rhs;
            let reduced_terms: Vec<Term<NonAffine<T>>> = self
                .0
                .into_iter()
                .map(|Term(c, s)| (c % rhs, s))
                .filter_map(|(rc, s)| (rc != 0).then(|| Term(rc, s)))
                .collect();
            if reduced_terms.is_empty() {
                // (c % m) % m == (c % m)
                AffineForm::constant(reduced_intercept)
            } else {
                NonAffine::Mod(Box::new(AffineForm(reduced_terms, reduced_intercept)), rhs).into()
            }
        }
    }
}

impl<T> Rem<i32> for NonAffine<T> {
    type Output = Self;

    fn rem(self, rhs: i32) -> Self::Output {
        assert_ne!(rhs, 0);
        if rhs == 1 {
            return NonAffine::Constant(0);
        }
        match self {
            // TODO: Below i32 cast shouldn't be necesary. May make more sense to use i32 for Mod.
            NonAffine::Constant(c) => NonAffine::Constant(c % rhs),
            leaf @ NonAffine::Leaf(_) => NonAffine::Mod(Box::new(leaf.into()), rhs),
            // e.g., (a % 8) % 2 == a % 2
            NonAffine::Mod(a, r) if r % rhs == 0 => NonAffine::Mod(a, rhs),
            // e.g., (a % 2) % 8 == a % 2
            NonAffine::Mod(a, r) if rhs % r == 0 => NonAffine::Mod(a, r),
            other => NonAffine::Mod(Box::new(other.into()), rhs),
        }
    }
}

impl<T: Bounds> AffineForm<NonAffine<T>> {
    /// If the implied remainder is in [0, rhs-1], make `self` the exact affine quotient; otherwise
    /// replace `self` with a FloorDiv.
    fn simplify_div_with_bounds(&mut self, rhs: i32) {
        assert_ne!(rhs, 0);

        // Fast path: if rhs is 1 or there are no terms, just divide and bail.
        if self.0.is_empty() || rhs == 1 {
            self.1 /= rhs;
            return;
        }

        // All coefficients and intercept must be >=0. Otherwise, bail.
        if self.1 < 0 || self.0.iter().any(|Term(c, _)| *c < 0) {
            *self = std::mem::take(self) / rhs;
            return;
        }

        // Compute conservative bounds for the remainder when dividing by `rhs`.
        let base = self.1 % rhs;
        let total_remainder = self
            .0
            .iter()
            .try_fold((base, base), |(mn, mx), Term(c, s)| {
                let r = c % rhs;
                if r == 0 {
                    Ok((mn, mx))
                } else if let Some((smin, smax)) = s.bounds() {
                    Ok((mn + r * smin, mx + r * smax))
                } else {
                    Err(())
                }
            });

        match total_remainder {
            Ok((mn, mx)) if mn >= 0 && mx < rhs => {
                self.0.iter_mut().for_each(|Term(c, _)| *c /= rhs);
                self.0.retain(|Term(c, _)| *c != 0);
                self.1 /= rhs;
            }
            _ => {
                // Fall back to just constructing the division node normally.
                *self = NonAffine::FloorDiv(Box::new(std::mem::take(self)), rhs).into();
            }
        }
    }

    /// If already in [0, rhs-1], do nothing; otherwise compute `self % rhs`.
    fn simplify_mod_with_bounds(&mut self, rhs: i32) {
        assert_ne!(rhs, 0);

        // Fast path: mod 1 is always 0.
        if rhs == 1 {
            *self = AffineForm::constant(0);
            return;
        }

        // If bounds already guarantee the value lies in [0, rhs-1], nothing to do.
        if let Some((min_b, max_b)) = self.bounds() {
            if min_b >= 0 && max_b < rhs {
                return;
            }
        }

        // Fallback: compute modulo structurally.
        *self = std::mem::take(self) % rhs;
    }
}

impl<T> RemAssign<i32> for AffineForm<NonAffine<T>> {
    fn rem_assign(&mut self, rhs: i32) {
        *self = std::mem::take(self) % rhs
    }
}

impl<T> RemAssign<i32> for NonAffine<T> {
    fn rem_assign(&mut self, rhs: i32) {
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
            NonAffine::Constant(v) => write!(f, "{v}"),
            NonAffine::Leaf(v) => write!(f, "{v}"),
            NonAffine::FloorDiv(v, d) => write!(f, "({v}) / {d}"),
            NonAffine::Mod(v, m) => write!(f, "({v}) % {m}"),
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
    use super::{AffineForm, Atom, NonAffine, Term};
    use crate::expr::{Bounds, NonAffineExpr, Substitute};

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct TestAtomBounded(&'static str, i32, i32);
    impl Bounds for TestAtomBounded {
        fn bounds(&self) -> Option<(i32, i32)> {
            Some((self.1, self.2))
        }
    }
    impl Atom for TestAtomBounded {}

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
    fn test_affineform_bounds_intercept() {
        assert_eq!(AffineForm::<&str>(vec![], 1).bounds(), Some((1, 1)))
    }

    #[test]
    fn test_affineform_bounds_positive_signs() {
        struct Trt(i32, i32);

        impl Bounds for Trt {
            fn bounds(&self) -> Option<(i32, i32)> {
                Some((self.0, self.1))
            }
        }

        assert_eq!(
            AffineForm(vec![Term(1, Trt(0, 10)), Term(2, Trt(0, 10))], 0).bounds(),
            Some((0, 30))
        )
    }

    #[test]
    fn test_affineform_bounds_mixed_signs() {
        struct Trt(i32, i32);

        impl Bounds for Trt {
            fn bounds(&self) -> Option<(i32, i32)> {
                Some((self.0, self.1))
            }
        }

        assert_eq!(
            AffineForm(vec![Term(1, Trt(0, 10)), Term(-2, Trt(-5, 10))], 1).bounds(),
            Some((-19, 21))
        )
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

    /// Test that (a % 2) % 4 simplifies to (a % 2).
    #[test]
    fn test_substitute_non_affine_expr_var_for_affine_expr_3() {
        let x = NonAffine::Leaf("x");
        let y = NonAffine::Leaf("y");

        let original_expr: NonAffineExpr<_> = Term(2, NonAffine::Mod(Box::new(x.into()), 4)).into();
        let result = original_expr.subs(
            &"x",
            &AffineForm::from(NonAffine::Mod(Box::new(y.clone().into()), 2)),
        );
        assert_eq!(
            result,
            AffineForm::from(Term(2, NonAffine::Mod(Box::new(y.into()), 2)))
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

    #[test]
    fn test_mul_by_zero_clears_terms() {
        let mut e = AffineForm(vec![Term(3, "x")], 5);
        e *= 0;
        let expected: AffineForm<&str> = AffineForm::constant(0);
        assert_eq!(e, expected);
    }

    #[test]
    fn test_addassign_merging_to_zero_removes_term() {
        let mut e = AffineForm(vec![Term(3, "x")], 0);
        e += AffineForm(vec![Term(-3, "x")], 0);
        let expected: AffineForm<&str> = AffineForm::constant(0);
        assert_eq!(e, expected);
    }

    #[test]
    fn test_division_of_constant_affine_folds() {
        let e: AffineForm<NonAffine<()>> = AffineForm::constant(6);
        let got = e / 2;
        assert_eq!(got, AffineForm::constant(3));
    }

    #[test]
    fn test_mod_by_one_is_zero() {
        let e: AffineForm<NonAffine<&str>> = AffineForm(vec![Term(2, NonAffine::Leaf("x"))], 3);
        let m = 1;
        assert_eq!(e % m, AffineForm::constant(0));

        let n = NonAffine::Leaf("x") % m;
        assert_eq!(n, NonAffine::Constant(0));
    }

    #[test]
    fn test_mod_reduces_coefficients_and_intercept_1() {
        // (5*x + 10) % 5 == 0
        let e: AffineForm<NonAffine<&str>> = AffineForm(vec![Term(5, NonAffine::Leaf("x"))], 10);
        assert_eq!(e % 5, AffineForm::constant(0));
    }

    // TODO: Remove
    #[test]
    fn test_mod_reduces_coefficients_and_intercept_2() {
        // (5*x + 11) % 5 == 1
        let e: AffineForm<NonAffine<&str>> = AffineForm(vec![Term(5, NonAffine::Leaf("x"))], 11);
        assert_eq!(e % 5, AffineForm::constant(1));
    }

    #[test]
    fn test_mod_reduces_coefficients_and_intercept_3() {
        // (7*x + 3*y + 9) % 5 == (2*x + 3*y + 4) % 5
        let e: AffineForm<NonAffine<&str>> = AffineForm(
            vec![Term(7, NonAffine::Leaf("x")), Term(3, NonAffine::Leaf("y"))],
            9,
        );
        let expected: AffineForm<NonAffine<&str>> = NonAffine::Mod(
            Box::new(AffineForm(
                vec![Term(2, NonAffine::Leaf("x")), Term(3, NonAffine::Leaf("y"))],
                4,
            )),
            5,
        )
        .into();
        assert_eq!(e % 5, expected);
    }

    #[test]
    fn test_mod_reduces_coefficients_and_intercept_4() {
        // (6*x + 8*y + 7) % 3 == (2*y + 1) % 3
        let e: AffineForm<NonAffine<&str>> = AffineForm(
            vec![Term(6, NonAffine::Leaf("x")), Term(8, NonAffine::Leaf("y"))],
            7,
        );
        let expected: AffineForm<NonAffine<&str>> = NonAffine::Mod(
            Box::new(AffineForm(vec![Term(2, NonAffine::Leaf("y"))], 1)),
            3,
        )
        .into();
        assert_eq!(e % 3, expected);
    }

    #[test]
    fn test_divide_through_all_divisible() {
        // (16*x + 32*y + 48) / 16 == (1*x + 2*y + 3)
        let e: AffineForm<NonAffine<&str>> = AffineForm(
            vec![
                Term(16, NonAffine::Leaf("x")),
                Term(32, NonAffine::Leaf("y")),
            ],
            48,
        );
        let got = e / 16;
        let expected: AffineForm<NonAffine<&str>> = AffineForm(
            vec![Term(1, NonAffine::Leaf("x")), Term(2, NonAffine::Leaf("y"))],
            3,
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn test_divide_through_mixed_divisible() {
        // (16*x + 18*y + 4) / 4 => FloorDiv when not all coefficients divisible
        let e: AffineForm<NonAffine<&str>> = AffineForm(
            vec![
                Term(16, NonAffine::Leaf("x")),
                Term(18, NonAffine::Leaf("y")),
            ],
            4,
        );
        let got = e.clone() / 4;
        let expected_affine: AffineForm<NonAffine<&str>> =
            NonAffine::FloorDiv(Box::new(e), 4).into();
        assert_eq!(got, expected_affine);
    }

    // Regression: big-constant divide-through similar to codegen cases
    #[test]
    fn test_divide_through_codegen_like_example() {
        // (16*n063 + 128*n062) / 16 => n063 + 8*n062
        let e: AffineForm<NonAffine<&str>> = AffineForm(
            vec![
                Term(16, NonAffine::Leaf("n063")),
                Term(128, NonAffine::Leaf("n062")),
            ],
            0,
        );
        let got = e / 16;
        let expected: AffineForm<NonAffine<&str>> = AffineForm(
            vec![
                Term(1, NonAffine::Leaf("n063")),
                Term(8, NonAffine::Leaf("n062")),
            ],
            0,
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn test_divide_through_with_intercept_regression() {
        // (32 + 16*x + 128*y) / 16 => 2 + x + 8*y
        let e: AffineForm<NonAffine<&str>> = AffineForm(
            vec![
                Term(16, NonAffine::Leaf("x")),
                Term(128, NonAffine::Leaf("y")),
            ],
            32,
        );
        let got = e / 16;
        let expected: AffineForm<NonAffine<&str>> = AffineForm(
            vec![Term(1, NonAffine::Leaf("x")), Term(8, NonAffine::Leaf("y"))],
            2,
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn test_substitution_floor_div_divide_through_regression() {
        // (x / 16) with x := 16*y  => y
        let x = NonAffine::Leaf("x");
        let y = NonAffine::Leaf("y");
        let original_expr = AffineForm(
            vec![Term(1, NonAffine::FloorDiv(Box::new(x.into()), 16))],
            0,
        );
        let result = original_expr.subs(&"x", &AffineForm(vec![Term(16, y.clone())], 0));
        assert_eq!(result, AffineForm(vec![Term(1, y)], 0));
    }

    #[test]
    fn test_substitution_mod_reduction_regression() {
        // (x % 16) with x := 16*y  => 0
        let original_expr =
            AffineForm::from(NonAffine::Mod(Box::new(NonAffine::Leaf("x").into()), 16));
        let result = original_expr.subs(&"x", &AffineForm(vec![Term(16, NonAffine::Leaf("y"))], 0));
        assert_eq!(result, AffineForm::constant(0));
    }

    #[test]
    fn test_mod_outside_bounds_simplifies_away_1() {
        let x = TestAtomBounded("x", 0, 1023);
        let original: AffineForm<_> =
            NonAffine::Mod(Box::new(NonAffine::Leaf(x.clone()).into()), 1024).into();
        let result = original.subs(&x, &NonAffineExpr::from(x.clone()));
        assert_eq!(result, AffineForm::from(x));
    }

    #[test]
    fn test_mod_outside_bounds_simplifies_away_2() {
        let x = TestAtomBounded("x", 0, 5);
        let e = AffineForm::from(NonAffine::Leaf(x.clone())) + 2; // bounds become [2, 7]
        let original = AffineForm::from(NonAffine::Mod(Box::new(e.clone()), 8));
        let result = original.subs(&x, &NonAffineExpr::from(x.clone()));
        assert_eq!(result, e);
    }

    #[test]
    fn test_bounds_aware_floor_div_affine() {
        // (x + 16*y) / 16 where 0 <= x <= 15 => y
        let x = TestAtomBounded("x", 0, 15);
        let y = TestAtomBounded("y", 0, 100);
        let e = AffineForm(
            vec![
                Term(1, NonAffine::Leaf(x.clone())),
                Term(16, NonAffine::Leaf(y.clone())),
            ],
            0,
        );
        let original = AffineForm::from(NonAffine::FloorDiv(Box::new(e), 16));

        let got = original.subs(&x, &NonAffineExpr::from(x.clone())); // no-op forcing simplification
        let expected = AffineForm::from(Term(1, NonAffine::Leaf(y)));
        assert_eq!(got, expected);
    }

    #[test]
    fn test_bounds_aware_floor_div_substitution() {
        // x / 16
        //   x := (z + 16*y)
        //   where z in [0,15]
        // produces: floor(z/16) + y
        // simplifies to: y
        let x_atom = TestAtomBounded("x", 0, i32::MAX);
        let x = NonAffine::Leaf(x_atom.clone());
        let original = AffineForm(
            vec![Term(1, NonAffine::FloorDiv(Box::new(x.into()), 16))],
            0,
        );
        let z = NonAffine::Leaf(TestAtomBounded("z", 0, 15));
        let y = NonAffine::Leaf(TestAtomBounded("y", 0, 100));
        let replacement = AffineForm(vec![Term(1, z), Term(16, y.clone())], 0);
        let got = original.subs(&x_atom, &replacement);
        let expected = AffineForm(vec![Term(1, y)], 0);
        assert_eq!(got, expected);
    }
}
