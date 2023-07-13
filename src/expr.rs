use std::ops::{Add, Mul};

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
}

impl<T: PartialEq> Add for AffineExpr<T> {
    type Output = Self;

    fn add(mut self, rhs: AffineExpr<T>) -> Self::Output {
        // TODO: Keep the terms sorted for faster iteration.
        let AffineExpr(terms, intercept) = &mut self;
        *intercept += rhs.1;
        for Term(c, s) in rhs.0 {
            if let Some(Term(c2, _)) = terms.iter_mut().find(|Term(_, s2)| &s == s2) {
                *c2 += c;
            } else {
                terms.push(Term(c, s));
            }
        }
        self
    }
}

impl<T> Add<i32> for AffineExpr<T> {
    type Output = Self;

    fn add(mut self, rhs: i32) -> Self::Output {
        self.1 += rhs;
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

#[cfg(test)]
mod tests {
    use super::{AffineExpr, Term};

    #[test]
    fn test_intercept_scalar_addition() {
        assert_eq!(AffineExpr::<()>(vec![], 1) + 2, AffineExpr::<()>(vec![], 3));
        assert_eq!(AffineExpr::<()>(vec![], 1) + 0, AffineExpr::<()>(vec![], 1));
    }

    #[test]
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
        // let expected = todo!("Encode 2x + 4y + 8z + 5");
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
