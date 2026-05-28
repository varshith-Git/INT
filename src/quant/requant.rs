//! # Requantization
//!
//! After `matmul_wide`, the accumulator is `i32` at scale `s_a × s_b`.
//! This module converts that `i32` back to a narrow type (`i8`, `i16`) at
//! the *output* scale `s_out`, using only integer arithmetic.
//!
//! ## Why integer-only?
//!
//! On MCUs without an FPU (Cortex-M0, M3, many RISC-V parts), floating-point
//! division is expensive or absent.  The standard solution (CMSIS-NN,
//! TFLite Micro) is to precompute a fixed-point multiplier and right-shift
//! from the scale ratio, then apply both in a single `i32` multiply + shift.
//!
//! ## The math
//!
//! ```text
//! scale_ratio   = (s_a × s_b) / s_out
//! multiplier    = round(scale_ratio × 2^shift)   [fits in i32]
//! shift         = number of fractional bits chosen so multiplier ∈ [2^30, 2^31)
//!
//! requantized   = clamp(
//!                   round((acc × multiplier) >> (31 + shift)),
//!                   T::MIN, T::MAX
//!                 )
//! ```
//!
//! [`RequantShift`] holds the precomputed `(multiplier, shift)` pair and
//! is cheap to apply per-element in the inner loop.

use crate::matrix::Matrix;
use crate::quant::clip::Clamp;
use core::fmt;
use libm::roundf;

// ─────────────────────────────────────────────────────────────────────────────
// RequantShift
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed integer multiplier + right-shift for requantization.
///
/// Construct once per layer with [`RequantShift::from_scales`], then call
/// [`RequantShift::apply`] per accumulator element.
///
/// # Fields
/// - `multiplier` — fixed-point scale ratio in Q31 format
/// - `shift`      — total right-shift to apply after multiply
#[derive(Clone, Copy, PartialEq)]
pub struct RequantShift {
    /// Fixed-point multiplier (Q31).
    pub multiplier: i32,
    /// Right-shift applied after multiply.
    pub shift: u8,
}

impl RequantShift {
    /// Construct from three scales: input A, input B, desired output.
    ///
    /// ```text
    /// effective_scale = s_a * s_b / s_out
    /// ```
    /// A floating-point operation here is fine — this happens *once* at
    /// layer initialization, not in the hot path.
    pub fn from_scales(s_a: f32, s_b: f32, s_out: f32) -> Self {
        debug_assert!(s_a > 0.0 && s_b > 0.0 && s_out > 0.0);
        let ratio = (s_a * s_b) / s_out;
        Self::from_ratio(ratio)
    }

    /// Construct from a pre-computed ratio `(s_a × s_b) / s_out`.
    pub fn from_ratio(ratio: f32) -> Self {
        // Decompose ratio into mantissa × 2^exp
        // We want multiplier ∈ [2^30, 2^31) to maximise precision.
        let (mantissa, exponent) = frexp(ratio);

        // mantissa is in [0.5, 1.0); scale to [2^30, 2^31) range
        // multiplier = round(mantissa × 2^31)
        let multiplier = roundf(mantissa * (1u64 << 31) as f32) as i64;
        let multiplier = multiplier.clamp(0, i32::MAX as i64) as i32;

        // shift accounts for the 2^31 we already baked in + the exponent
        // total shift = 31 - exponent  (right-shift to undo the baked-in scale)
        let shift = (31i32 - exponent).clamp(0, 62) as u8;

        Self { multiplier, shift }
    }

    /// Apply requantization to a single `i32` accumulator value.
    ///
    /// Result is a `i32`.
    ///
    /// Uses a rounding right-shift: adds `1 << (shift-1)` before shifting
    /// to implement round-half-away-from-zero.
    #[inline(always)]
    pub fn apply(&self, acc: i32) -> i32 {
        // Add 1 << (shift - 1) to achieve round-to-nearest instead of floor
        let val = (acc as i64 * self.multiplier as i64 + (1i64 << (self.shift - 1))) >> self.shift;
        val as i32
    }

    /// Apply requantization to a single `i32` accumulator value, returning `i32` without clamping to i8 bounds.
    #[inline(always)]
    pub fn apply_i32(&self, acc: i32) -> i32 {
        self.apply(acc) as i32
    }

    /// Requantize and clamp to output type `Out`.
    ///
    /// # Example
    /// ```ignore
    /// let rs = RequantShift::from_scales(s_a, s_b, s_out);
    /// let out_val: i8 = rs.apply_clamped(acc_val);
    /// ```
    #[inline]
    pub fn apply_clamped<Out: Clamp>(&self, acc: i32) -> Out {
        let shifted = self.apply(acc);
        Out::clamp_from_i64(shifted as i64)
    }
}

impl fmt::Debug for RequantShift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RequantShift {{ multiplier: {:#010x}, shift: {} }}",
            self.multiplier, self.shift
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix-level requantization
// ─────────────────────────────────────────────────────────────────────────────

/// Requantize an entire `i32` accumulator matrix to a narrow output type `Out`.
///
/// Applies [`RequantShift::apply_clamped`] to every element.
/// The output `Matrix<Out, R, C>` has the same dimensions as the input.
pub fn requantize_matrix<Out, const R: usize, const C: usize>(
    acc: &Matrix<i32, R, C>,
    rs: &RequantShift,
) -> Matrix<Out, R, C>
where
    Out: Clamp + Copy + Default,
{
    let mut out = Matrix::<Out, R, C>::zeros();
    for r in 0..R {
        for c in 0..C {
            out.data[r][c] = rs.apply_clamped(acc.data[r][c]);
        }
    }
    out
}

/// Requantize an entire `i32` accumulator matrix to an `i32` matrix scaled correctly, without clamping.
pub fn requantize_matrix_i32<const R: usize, const C: usize>(
    acc: &Matrix<i32, R, C>,
    rs: &RequantShift,
) -> Matrix<i32, R, C> {
    let mut out = Matrix::<i32, R, C>::zeros();
    for r in 0..R {
        for c in 0..C {
            out.data[r][c] = rs.apply_i32(acc.data[r][c]);
        }
    }
    out
}

/// Requantize an entire `i32` accumulator matrix to a narrow output type `Out` using per-channel scales.
pub fn requantize_matrix_per_channel<Out, const R: usize, const C: usize>(
    acc: &Matrix<i32, R, C>,
    rs: &[RequantShift; C],
) -> Matrix<Out, R, C>
where
    Out: Clamp + Copy + Default,
{
    let mut out = Matrix::<Out, R, C>::zeros();
    for r in 0..R {
        for c in 0..C {
            out.data[r][c] = rs[c].apply_clamped(acc.data[r][c]);
        }
    }
    out
}

/// Requantize an entire `i32` accumulator matrix using per-channel scales, returning `i32`.
pub fn requantize_matrix_i32_per_channel<const R: usize, const C: usize>(
    acc: &Matrix<i32, R, C>,
    rs: &[RequantShift; C],
) -> Matrix<i32, R, C> {
    let mut out = Matrix::<i32, R, C>::zeros();
    for r in 0..R {
        for c in 0..C {
            out.data[r][c] = rs[c].apply_i32(acc.data[r][c]);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Split `x` into `(mantissa, exponent)` such that `x = mantissa × 2^exponent`
/// and `mantissa ∈ [0.5, 1.0)`.  Like C's `frexp`.
fn frexp(x: f32) -> (f32, i32) {
    if x == 0.0 { return (0.0, 0); }
    let bits = x.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    let exp = exp_bits - 126;          // biased exponent; mantissa = x / 2^exp ∈ [0.5,1)
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F00_0000;  // exponent → 126 (= 0.5..1)
    (f32::from_bits(mantissa_bits), exp)
}

/// Rounding arithmetic right-shift: round-half-away-from-zero.
#[inline(always)]
fn rounding_right_shift(x: i64, shift: u32) -> i64 {
    if shift == 0 { return x; }
    let rounding = 1i64 << (shift - 1);
    (x + rounding) >> shift
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_frexp_basic() {
        let (m, e) = frexp(8.0_f32);
        assert!((m - 0.5).abs() < 1e-6, "mantissa should be 0.5, got {m}");
        assert_eq!(e, 4);
    }

    #[test]
    fn test_frexp_one() {
        let (m, e) = frexp(1.0_f32);
        assert!((m - 0.5).abs() < 1e-6);
        assert_eq!(e, 1);
    }

    #[test]
    fn test_rounding_shift() {
        // 5 >> 1 with rounding = round(2.5) = 3
        assert_eq!(rounding_right_shift(5, 1), 3);
        // 4 >> 1 with rounding = 2
        assert_eq!(rounding_right_shift(4, 1), 2);
        // negative: -5 >> 1 = round(-2.5) = -2
        assert_eq!(rounding_right_shift(-5, 1), -2);
    }

    #[test]
    fn test_requant_identity_scale() {
        // If s_a=1, s_b=1, s_out=1 → ratio=1 → output ≈ input
        let rs = RequantShift::from_scales(1.0, 1.0, 1.0);
        let result: i8 = rs.apply_clamped(64i32);
        assert_eq!(result, 64i8);
    }

    #[test]
    fn test_requant_scale_down() {
        // s_a=0.1, s_b=0.1, s_out=1.0 → ratio=0.01 → large accum shrinks
        let rs = RequantShift::from_scales(0.1, 0.1, 1.0);
        // acc=10000 → real value = 10000 × 0.01 = 100.0 → clamp i8 → 100
        let result: i8 = rs.apply_clamped(10000i32);
        // Allow ±2 for fixed-point rounding
        assert!((result as i32 - 100).abs() <= 2,
            "expected ≈100, got {result}");
    }

    #[test]
    fn test_requant_clamps_to_i8_max() {
        let rs = RequantShift::from_scales(1.0, 1.0, 1.0);
        let result: i8 = rs.apply_clamped(1000i32);
        assert_eq!(result, i8::MAX);
    }

    #[test]
    fn test_requant_clamps_to_i8_min() {
        let rs = RequantShift::from_scales(1.0, 1.0, 1.0);
        let result: i8 = rs.apply_clamped(-1000i32);
        assert_eq!(result, i8::MIN);
    }

    #[test]
    fn test_requant_matrix() {
        let acc = Matrix::from_array([[64i32, -64], [127, -127]]);
        let rs = RequantShift::from_scales(1.0, 1.0, 1.0);
        let out: Matrix<i8, 2, 2> = requantize_matrix(&acc, &rs);
        assert_eq!(out.data[0][0], 64i8);
        assert_eq!(out.data[0][1], -64i8);
        assert_eq!(out.data[1][0], 127i8);
        assert_eq!(out.data[1][1], -127i8);
    }
}