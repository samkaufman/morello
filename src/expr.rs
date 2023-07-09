use itertools::Itertools;
use std::iter;
use std::ops::{Add, Mul};

#[derive(Debug, PartialEq)]
pub struct AffineExpr(pub Vec<Term>, pub i32);

#[derive(Debug, PartialEq)]
pub struct Term(pub i32, pub String);

impl AffineExpr {
    pub fn c_expr(&self) -> String {
        let mut buf = self
            .0
            .iter()
            .map(|Term(coef, sym)| match &coef {
                0 => panic!("AffineExpr contained zero term"),
                1 => sym.to_string(),
                _ => format!("{} * {}", coef, sym),
            })
            .join(" + ");
        if self.1 != 0 {
            if buf.is_empty() {
                buf = self.1.to_string();
            } else {
                buf += &format!(" + {}", self.1);
            }
        }
        buf
    }
}

impl Add<i32> for AffineExpr {
    type Output = Self;

    fn add(mut self, rhs: i32) -> Self::Output {
        self.1 += rhs;
        self
    }
}

impl Mul<i32> for AffineExpr {
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
        assert_eq!(AffineExpr(vec![], 1) + 2, AffineExpr(vec![], 3));
        assert_eq!(AffineExpr(vec![], 1) + 0, AffineExpr(vec![], 1));
    }

    #[test]
    fn test_intercept_scalar_multiplication() {
        assert_eq!(AffineExpr(vec![], 1) * 2, AffineExpr(vec![], 2));
        assert_eq!(AffineExpr(vec![], 1) * 0, AffineExpr(vec![], 0));
    }

    #[test]
    fn test_expr_zero_not_emitted() {
        assert_eq!(AffineExpr(vec![], 0).c_expr(), "")
    }

    #[test]
    fn test_intercept_zero_not_emitted() {
        assert_eq!(
            AffineExpr(vec![Term(2, "x".to_string())], 0).c_expr(),
            "2 * x"
        )
    }

    #[test]
    fn test_lower_to_c_expr() {
        assert_eq!(AffineExpr(vec![], 1).c_expr(), "1");
        assert_eq!(
            AffineExpr(vec![Term(1, "x".to_string())], 1).c_expr(),
            "x + 1"
        );
        assert_eq!(
            AffineExpr(vec![Term(2, "y".to_string())], 3).c_expr(),
            "2 * y + 3"
        );
    }
}
