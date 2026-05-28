//! # QTensor — quantized tensor with semantic meaning
//!
//! A raw `Matrix<i8, R, C>` is just bits.  A `QTensor` knows what those bits
//! *mean* in float space:
//!
//! ```text
//! real[r][c] = (data[r][c] - zero_point) as f32 * scale
//! ```
//!
//! This is the primary type passed between layers in a quantized inference
//! runtime.  Weights, activations, and bias tensors are all `QTensor`s with
//! different `QuantParams`.
//!
//! ## Type parameters
//! - `T`   — storage integer type (`i8`, `u8`, `i16`, `i32`, …)
//! - `R`   — row count (const)
//! - `C`   — column count (const)
//!
//! The storage type `T` is kept generic so that:
//! - `QTensor<i8,  R, C>` — narrow input/output (activations, weights)
//! - `QTensor<i32, R, C>` — wide accumulator (post-matmul, pre-requant)
//! - `QTensor<i16, R, C>` — audio / DSP mid-precision

use crate::matrix::Matrix;
use crate::quant::scale::QuantParams;
use core::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// QTensor
// ─────────────────────────────────────────────────────────────────────────────

/// A quantized tensor: integer matrix + quantization parameters.
///
/// The integer data lives in `data`; `params` gives every element a real
/// floating-point interpretation.
#[derive(Clone, Copy, PartialEq)]
pub struct QTensor<T, const R: usize, const C: usize>
where
    T: Copy + Default,
{
    /// Raw quantized values stored in row-major order.
    pub data: Matrix<T, R, C>,

    /// Scale and zero-point that map raw integers to real floats.
    pub params: QuantParams,
}

impl<T: Copy + Default, const R: usize, const C: usize> QTensor<T, R, C> {
    // ── constructors ─────────────────────────────────────────────────────────

    /// Wrap an existing `Matrix` with quantization params.
    #[inline]
    pub fn new(data: Matrix<T, R, C>, params: QuantParams) -> Self {
        Self { data, params }
    }

    /// Create a zero-valued tensor (useful as accumulator seed).
    #[inline]
    pub fn zeros(params: QuantParams) -> Self {
        Self { data: Matrix::zeros(), params }
    }

    /// Create from a 2-D array literal.
    #[inline]
    pub fn from_array(raw: [[T; C]; R], params: QuantParams) -> Self {
        Self { data: Matrix::from_array(raw), params }
    }

    // ── accessors ─────────────────────────────────────────────────────────────

    /// Raw integer at `(row, col)`.
    #[inline]
    pub fn raw(&self, row: usize, col: usize) -> T {
        self.data.get(row, col)
    }

    #[inline]
    pub const fn rows(&self) -> usize { R }

    #[inline]
    pub const fn cols(&self) -> usize { C }

    #[inline]
    pub fn scale(&self) -> f32 { self.params.scale }

    #[inline]
    pub fn zero_point(&self) -> i32 { self.params.zero_point }
}

// ── float dequantization helpers (on i8 tensors) ─────────────────────────────

impl<const R: usize, const C: usize> QTensor<i8, R, C> {
    /// Dequantize element `(row, col)` to `f32`.
    #[inline]
    pub fn dequant(&self, row: usize, col: usize) -> f32 {
        self.params.dequantize_i8(self.data.get(row, col))
    }

    /// Build a `QTensor<i8>` by quantizing a 2-D array of `f32` values.
    ///
    /// Uses the provided `params` — caller must have calibrated these first
    /// (e.g. via [`scale::symmetric_params_from_absmax`]).
    pub fn quantize_from(floats: [[f32; C]; R], params: QuantParams) -> Self {
        let mut raw = [[0i8; C]; R];
        for r in 0..R {
            for c in 0..C {
                raw[r][c] = params.quantize_f32_to_i8(floats[r][c]);
            }
        }
        Self::from_array(raw, params)
    }
}

// ── i32 accumulator dequantization ───────────────────────────────────────────

impl<const R: usize, const C: usize> QTensor<i32, R, C> {
    /// Dequantize element `(row, col)` from `i32` accumulator to `f32`.
    #[inline]
    pub fn dequant_i32(&self, row: usize, col: usize) -> f32 {
        self.params.dequantize_i32(self.data.get(row, col))
    }
}

impl<T: Copy + Default + fmt::Display, const R: usize, const C: usize> fmt::Debug
    for QTensor<T, R, C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QTensor [{R}×{C}] {:?}", self.params)?;
        for r in 0..R {
            write!(f, "  [")?;
            for c in 0..C {
                if c > 0 { write!(f, ", ")?; }
                write!(f, "{:>6}", self.data.get(r, c))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}