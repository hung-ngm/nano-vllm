//! The Rust port of `nanovllm/layers/`.
//!
//! `divide` is the only export in this unit whose inputs and outputs are plain numbers.
//! The torch, Triton, and module exports stay in Python: see `rust-anything.json`.

#![forbid(unsafe_code)]

use crate::pynum::Num;

/// Port of `divide` (`nanovllm/layers/linear.py:7`).
///
/// `assert numerator % denominator == 0` then `numerator // denominator`.
/// An int divided by an int stays an int. A float argument makes the result a float.
pub fn divide(numerator: Num, denominator: Num) -> Result<Num, String> {
    let remainder = numerator.modulo(denominator)?;
    if !is_zero(remainder) {
        return Err("AssertionError: ".into());
    }
    Ok(floor_div(numerator, denominator))
}

fn is_zero(n: Num) -> bool {
    match n {
        Num::Int(0) => true,
        Num::Float(x) => x == 0.0,
        Num::Int(_) => false,
    }
}

/// Python `//` for a pair that already divided evenly.
fn floor_div(numerator: Num, denominator: Num) -> Num {
    match (numerator, denominator) {
        (Num::Int(x), Num::Int(y)) => x
            .checked_div_euclid(y)
            .map_or_else(|| Num::Float((x as f64 / y as f64).floor()), Num::Int),
        (numerator, denominator) => Num::Float((numerator.f() / denominator.f()).floor()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_stay_ints() {
        assert_eq!(divide(Num::Int(8), Num::Int(2)).unwrap(), Num::Int(4));
        assert_eq!(divide(Num::Int(-8), Num::Int(2)).unwrap(), Num::Int(-4));
        assert_eq!(divide(Num::Int(8), Num::Int(-2)).unwrap(), Num::Int(-4));
        assert_eq!(divide(Num::Int(-8), Num::Int(-2)).unwrap(), Num::Int(4));
    }

    #[test]
    fn floats_stay_floats() {
        assert_eq!(
            divide(Num::Float(8.0), Num::Float(2.0)).unwrap(),
            Num::Float(4.0)
        );
        assert_eq!(
            divide(Num::Float(-8.0), Num::Int(2)).unwrap(),
            Num::Float(-4.0)
        );
        assert_eq!(
            divide(Num::Float(-0.0), Num::Float(1.0)).unwrap(),
            Num::Float(-0.0)
        );
    }

    #[test]
    fn rejects_a_remainder_and_zero() {
        assert_eq!(
            divide(Num::Int(3), Num::Int(2)).unwrap_err(),
            "AssertionError: "
        );
        assert_eq!(
            divide(Num::Int(1), Num::Int(0)).unwrap_err(),
            "ZeroDivisionError: integer modulo by zero"
        );
        assert_eq!(
            divide(Num::Float(1.0), Num::Float(0.0)).unwrap_err(),
            "ZeroDivisionError: float modulo"
        );
    }
}
