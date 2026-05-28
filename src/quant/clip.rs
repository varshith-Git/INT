//! # Clamp and activation functions
//!
//! Provides:
//! - [`Clamp`]  — trait for saturating narrow from a wide `i64` value
//! - [`clamp_i8`] / [`clamp_i16`] / [`clamp_i32`] — concrete helpers
//! - [`relu_i8`] / [`relu6_i8`] — post-requantization activations
//! - [`apply_relu`] / [`apply_relu6`] — in-place matrix activation
//!
//! ## Why saturating clamp matters on embedded
//!
//! On MCUs, overflow wraps silently (unless `-C overflow-checks=on`).
//! A saturating clamp at the output of requantization ensures that an
//! over-range accumulator produces `i8::MAX` / `i8::MIN` instead of
//! wrapping to a random value — the correct behaviour for quantized inference.
//!
//! ## ReLU in quantized space
//!
//! For a symmetric tensor (zero_point = 0):
//! ```text
//! relu(q) = max(q, 0)   [float 0.0 → quantized 0]
//! ```
//! For affine tensors, float 0.0 maps to `zero_point`, so:
//! ```text
//! relu(q) = max(q, zero_point)
//! ```
//! Pass `zero_point` explicitly to the relu helpers when using asymmetric
//! quantization.

use crate::matrix::Matrix;

// ─────────────────────────────────────────────────────────────────────────────
// Clamp trait
// ─────────────────────────────────────────────────────────────────────────────

/// Saturating narrow: convert an `i64` to `Self`, clamping to `[MIN, MAX]`.
///
/// Implemented for every integer type the engine supports.
pub trait Clamp: Copy {
    fn clamp_from_i64(x: i64) -> Self;
    fn min_val() -> i64;
    fn max_val() -> i64;
}

macro_rules! impl_clamp {
    ($t:ty) => {
        impl Clamp for $t {
            #[inline(always)]
            fn clamp_from_i64(x: i64) -> Self {
                x.clamp(<$t>::MIN as i64, <$t>::MAX as i64) as $t
            }
            #[inline(always)] fn min_val() -> i64 { <$t>::MIN as i64 }
            #[inline(always)] fn max_val() -> i64 { <$t>::MAX as i64 }
        }
    };
}

impl_clamp!(i8);
impl_clamp!(i16);
impl_clamp!(i32);
impl_clamp!(i64);
impl_clamp!(u8);
impl_clamp!(u16);
impl_clamp!(u32);
impl_clamp!(u64);

// ─────────────────────────────────────────────────────────────────────────────
// Scalar clamp helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Saturating clamp to `i8` range.
#[inline(always)]
pub fn clamp_i8(x: i32) -> i8 {
    x.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Saturating clamp to `i16` range.
#[inline(always)]
pub fn clamp_i16(x: i32) -> i16 {
    x.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Saturating clamp to `u8` range (e.g. after unsigned quantization).
#[inline(always)]
pub fn clamp_u8(x: i32) -> u8 {
    x.clamp(u8::MIN as i32, u8::MAX as i32) as u8
}

// ─────────────────────────────────────────────────────────────────────────────
// ReLU activations
// ─────────────────────────────────────────────────────────────────────────────

/// ReLU in quantized `i8` space.
///
/// `zero_point` is the quantized representation of float 0.0:
/// - Symmetric quantization: `zero_point = 0`
/// - Asymmetric (affine):    `zero_point = params.zero_point as i8`
///
/// # Example
/// ```ignore
/// // symmetric: max(q, 0)
/// let activated = relu_i8(-10i8, 0);
/// assert_eq!(activated, 0i8);
///
/// // asymmetric with zero_point=128
/// let activated = relu_i8(120i8, 128i8);
/// assert_eq!(activated, 128i8);  // clamps to zero_point
/// ```
#[inline(always)]
pub fn relu_i8(q: i8, zero_point: i8) -> i8 {
    q.max(zero_point)
}

/// ReLU6 in quantized `i8` space.
///
/// Clips the output to the quantized value representing float 6.0.
/// `six_q` = `round(6.0 / scale)` for symmetric tensors.
///
/// Used after lightweight convolutional layers (MobileNet-style).
#[inline(always)]
pub fn relu6_i8(q: i8, zero_point: i8, six_q: i8) -> i8 {
    q.clamp(zero_point, six_q)
}

/// ReLU in quantized `i32` space (applied to accumulator before requant).
#[inline(always)]
pub fn relu_i32(acc: i32, zero_point: i32) -> i32 {
    acc.max(zero_point)
}

// ─────────────────────────────────────────────────────────────────────────────
// In-place matrix activations
// ─────────────────────────────────────────────────────────────────────────────

/// Apply ReLU in-place to every element of a `Matrix<i8>`.
pub fn apply_relu<const R: usize, const C: usize>(
    mat: &mut Matrix<i8, R, C>,
    zero_point: i8,
) {
    for r in 0..R {
        for c in 0..C {
            mat.data[r][c] = relu_i8(mat.data[r][c], zero_point);
        }
    }
}

/// Apply ReLU6 in-place to every element of a `Matrix<i8>`.
pub fn apply_relu6<const R: usize, const C: usize>(
    mat: &mut Matrix<i8, R, C>,
    zero_point: i8,
    six_q: i8,
) {
    for r in 0..R {
        for c in 0..C {
            mat.data[r][c] = relu6_i8(mat.data[r][c], zero_point, six_q);
        }
    }
}

/// Apply ReLU in-place to an `i32` accumulator matrix.
///
/// Typically called before requantization when the activation is fused
/// into the output step.
pub fn apply_relu_i32<const R: usize, const C: usize>(
    acc: &mut Matrix<i32, R, C>,
    zero_point: i32,
) {
    for r in 0..R {
        for c in 0..C {
            acc.data[r][c] = relu_i32(acc.data[r][c], zero_point);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix-level clamp helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Clamp an entire `i32` matrix down to `i8`.
pub fn clamp_matrix_i8<const R: usize, const C: usize>(
    mat: &Matrix<i32, R, C>,
) -> Matrix<i8, R, C> {
    let mut out = Matrix::<i8, R, C>::zeros();
    for r in 0..R {
        for c in 0..C {
            out.data[r][c] = clamp_i8(mat.data[r][c]);
        }
    }
    out
}

/// Dynamically scales an `i32` matrix down to `i8` using a single global scale factor.
/// Returns `(Matrix<i8>, scale_factor)` where `scale_factor` is the multiplier that was
/// applied (1.0 if no scaling was needed). The caller MUST multiply `out_scale * scale_factor`
/// when constructing the resulting QTensor, otherwise amplitude is permanently lost.
pub fn scale_matrix_i32_to_i8<const R: usize, const C: usize>(
    mat: &Matrix<i32, R, C>,
) -> (Matrix<i8, R, C>, f32) {
    // Find global absmax across entire matrix
    let mut global_max: i32 = 0;
    for r in 0..R {
        for c in 0..C {
            let val = mat.data[r][c].abs();
            if val > global_max {
                global_max = val;
            }
        }
    }

    let mut out = Matrix::<i8, R, C>::zeros();
    if global_max <= 127 {
        for r in 0..R {
            for c in 0..C {
                out.data[r][c] = mat.data[r][c] as i8;
            }
        }
        (out, 1.0)
    } else {
        let scale_factor = (global_max as f32) / 127.0;
        for r in 0..R {
            for c in 0..C {
                out.data[r][c] = libm::roundf((mat.data[r][c] as f32) / scale_factor) as i8;
            }
        }
        (out, scale_factor)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Clamp trait ──────────────────────────────────────────────────────────

    #[test]
    fn test_clamp_i8_in_range() {
        assert_eq!(i8::clamp_from_i64(50), 50i8);
        assert_eq!(i8::clamp_from_i64(-50), -50i8);
    }

    #[test]
    fn test_clamp_i8_overflow() {
        assert_eq!(i8::clamp_from_i64(1000), i8::MAX);
        assert_eq!(i8::clamp_from_i64(-1000), i8::MIN);
    }

    #[test]
    fn test_clamp_u8() {
        assert_eq!(clamp_u8(255), 255u8);
        assert_eq!(clamp_u8(256), 255u8);
        assert_eq!(clamp_u8(-1), 0u8);
    }

    // ── ReLU ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_relu_i8_positive_passthrough() {
        assert_eq!(relu_i8(42i8, 0), 42i8);
    }

    #[test]
    fn test_relu_i8_negative_clamp() {
        assert_eq!(relu_i8(-10i8, 0), 0i8);
    }

    #[test]
    fn test_relu_i8_asymmetric_zero_point() {
        // zero_point = -128 (i.e. 0.0 maps to -128 in affine scheme)
        assert_eq!(relu_i8(-100i8, -128i8), -100i8);  // above zero
        assert_eq!(relu_i8(-10i8,  0),       0i8);
    }

    #[test]
    fn test_relu6_i8_clips_max() {
        // scale = 6.0/127 ≈ 0.047; six_q = 127
        assert_eq!(relu6_i8(127i8, 0, 127), 127i8);
        assert_eq!(relu6_i8(-5i8,  0, 127), 0i8);
    }

    // ── Matrix ReLU ──────────────────────────────────────────────────────────

    #[test]
    fn test_apply_relu_matrix() {
        let mut m = Matrix::from_array([[-5i8, 10], [0, -1]]);
        apply_relu(&mut m, 0);
        assert_eq!(m.data, [[0i8, 10], [0, 0]]);
    }

    #[test]
    fn test_apply_relu_i32_matrix() {
        let mut acc = Matrix::from_array([[-100i32, 500], [0, -1]]);
        apply_relu_i32(&mut acc, 0);
        assert_eq!(acc.data, [[0i32, 500], [0, 0]]);
    }
}