//! # Scale management
//!
//! Defines [`QuantParams`], which attaches semantic meaning to a raw integer
//! tensor: "this `i8` value of 42 represents the float 42 × scale + zero_point".
//!
//! ## Two quantization schemes
//!
//! ### Affine (asymmetric) — TFLite style
//! ```text
//! real = (q - zero_point) × scale
//! q    = clamp(round(real / scale) + zero_point, T::MIN, T::MAX)
//! ```
//! - `zero_point ≠ 0` — lets the quantized range cover asymmetric float ranges
//!   (e.g. ReLU output lives in [0, 6], not [-3, 3]).
//!
//! ### Symmetric — CMSIS-NN style
//! ```text
//! real = q × scale          (zero_point always 0)
//! q    = clamp(round(real / scale), T::MIN, T::MAX)
//! ```
//! - Simpler multiply-accumulate: no zero_point correction term.
//! - Preferred for weights; activations may use affine.
//!
//! ## Scale propagation through matmul
//!
//! If A has scale `s_a` and B has scale `s_b`, the output has scale:
//! ```text
//! s_out = s_a × s_b
//! ```
//! The caller is responsible for requantizing the accumulator back to the
//! desired output scale via [`crate::quant::requant`].

use core::fmt;
use libm::roundf;

// ─────────────────────────────────────────────────────────────────────────────
// QuantParams
// ─────────────────────────────────────────────────────────────────────────────

/// Quantization parameters for a tensor.
///
/// A pair of `(scale, zero_point)` gives every raw integer value a real
/// floating-point meaning:
/// ```text
/// real_value = (raw - zero_point) as f32 * scale
/// ```
#[derive(Clone, Copy, PartialEq)]
pub struct QuantParams {
    /// Floating-point step size between adjacent quantized values.
    /// Must be > 0.
    pub scale: f32,

    /// Quantized value that represents float zero.
    /// 0 for symmetric quantization, non-zero for asymmetric.
    pub zero_point: i32,
}

impl QuantParams {
    /// Symmetric quantization: `zero_point = 0`.
    #[inline]
    pub const fn symmetric(scale: f32) -> Self {
        Self { scale, zero_point: 0 }
    }

    /// Asymmetric (affine) quantization with an explicit zero-point.
    #[inline]
    pub const fn affine(scale: f32, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }

    /// Returns `true` if this is symmetric (zero_point == 0).
    #[inline]
    pub const fn is_symmetric(&self) -> bool {
        self.zero_point == 0
    }

    /// Scale product rule: output scale after matmul of two tensors.
    ///
    /// If A ~ s_a and B ~ s_b, their product accumulates at scale s_a × s_b.
    /// Pass this to [`requant::RequantShift::from_scales`] to find the
    /// shift/multiplier pair needed to requantize to a target scale.
    #[inline]
    pub fn matmul_out_scale(&self, other: &QuantParams) -> f32 {
        self.scale * other.scale
    }

    // ── float ↔ quantized conversions ────────────────────────────────────────

    /// Quantize a `f32` value to `i8` using this scheme.
    ///
    /// Rounds to nearest, then saturating-clamps to `[i8::MIN, i8::MAX]`.
    #[inline]
    pub fn quantize_f32_to_i8(&self, x: f32) -> i8 {
        let q = roundf(x / self.scale) as i32 + self.zero_point;
        q.clamp(i8::MIN as i32, i8::MAX as i32) as i8
    }

    /// Quantize a `f32` value to `i32` (used for bias tensors).
    #[inline]
    pub fn quantize_f32_to_i32(&self, x: f32) -> i32 {
        roundf(x / self.scale) as i32 + self.zero_point
    }

    /// Dequantize an `i8` value back to `f32`.
    #[inline]
    pub fn dequantize_i8(&self, q: i8) -> f32 {
        (q as i32 - self.zero_point) as f32 * self.scale
    }

    /// Dequantize an `i32` accumulator value back to `f32`.
    ///
    /// Used after `matmul_wide` to inspect the raw accumulator in float space.
    #[inline]
    pub fn dequantize_i32(&self, q: i32) -> f32 {
        (q - self.zero_point) as f32 * self.scale
    }
}

impl fmt::Debug for QuantParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_symmetric() {
            write!(f, "QuantParams::symmetric(scale={:.6e})", self.scale)
        } else {
            write!(
                f,
                "QuantParams::affine(scale={:.6e}, zero_point={})",
                self.scale, self.zero_point
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute symmetric `QuantParams` from a float tensor's absolute maximum.
///
/// Calibrates the scale so that `abs_max` maps to `i8::MAX` (127).
///
/// # Example
/// ```ignore
/// // Tensor values in [-2.0, +1.8] → abs_max = 2.0
/// let params = symmetric_params_from_absmax(2.0_f32);
/// assert_eq!(params.zero_point, 0);
/// // scale ≈ 2.0 / 127.0 ≈ 0.01575
/// ```
#[inline]
pub fn symmetric_params_from_absmax(abs_max: f32) -> QuantParams {
    debug_assert!(abs_max > 0.0, "abs_max must be positive");
    QuantParams::symmetric(abs_max / i8::MAX as f32)
}

/// Compute affine `QuantParams` from a float tensor's `[min, max]` range.
///
/// Calibrates scale and zero_point so that `min` maps to `i8::MIN` (-128)
/// and `max` maps to `i8::MAX` (127).  This is the TFLite-style scheme.
#[inline]
pub fn affine_params_from_range(min: f32, max: f32) -> QuantParams {
    debug_assert!(min < max, "min must be strictly less than max");
    let scale = (max - min) / (i8::MAX as f32 - i8::MIN as f32);
    // zero_point: the integer value that represents float 0.0
    let zero_point = roundf(-(min / scale)) as i32 + i8::MIN as i32;
    let zero_point = zero_point.clamp(i8::MIN as i32, i8::MAX as i32);
    QuantParams::affine(scale, zero_point)
}