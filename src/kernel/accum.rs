//! # Accumulator Widening Trait
//!
//! Defines the relationship between a narrow input type `T` and its
//! wider accumulation type `Accum`.  This mirrors the ISA-level widening
//! multiply-accumulate found on ARM (SMLAL), RISC-V (P-ext), and DSPs.
//!
//! ## Type pairs shipped
//!
//! | Input `T` | Accumulator `Accum` | Rationale                          |
//! |-----------|---------------------|------------------------------------|
//! | `i8`      | `i32`               | Quantized inference (CMSIS-NN style)|
//! | `u8`      | `u32`               | Unsigned quantized / image kernels  |
//! | `i16`     | `i64`               | Audio / fixed-point DSP             |
//! | `u16`     | `u64`               | Unsigned 16-bit accumulation        |
//! | `i32`     | `i64`               | General-purpose wide accumulation   |
//! | `u32`     | `u64`               | Unsigned 32-bit wide accumulation   |
//! | `i64`     | `i64`               | Already widest signed — identity    |
//! | `u64`     | `u64`               | Already widest unsigned — identity  |

use core::ops::{Add, Mul};
use core::fmt;

/// Describes a scalar type that can widen into a larger accumulator type.
///
/// # Contract
/// - `Self::Accum` must be at least as wide as `Self`
/// - `widen(x)` is lossless (no truncation)
/// - `Accum` supports `Add` and `Mul` natively (used inside the kernel)
pub trait WideAccum: Copy + fmt::Debug {
    /// The wider type used to accumulate products.
    type Accum: Copy
        + Default
        + Add<Output = Self::Accum>
        + Mul<Output = Self::Accum>
        + PartialEq
        + fmt::Debug;

    /// Zero value of the accumulator type.
    fn accum_zero() -> Self::Accum;

    /// Widen `self` losslessly into `Accum`.
    fn widen(self) -> Self::Accum;
}

// ── Signed pairs ────────────────────────────────────────────────────────────

impl WideAccum for i8 {
    type Accum = i32;
    #[inline(always)] fn accum_zero() -> i32 { 0 }
    #[inline(always)] fn widen(self) -> i32  { self as i32 }
}

impl WideAccum for i16 {
    type Accum = i64;
    #[inline(always)] fn accum_zero() -> i64 { 0 }
    #[inline(always)] fn widen(self) -> i64  { self as i64 }
}

impl WideAccum for i32 {
    type Accum = i64;
    #[inline(always)] fn accum_zero() -> i64 { 0 }
    #[inline(always)] fn widen(self) -> i64  { self as i64 }
}

impl WideAccum for i64 {
    type Accum = i64;  // identity — already widest
    #[inline(always)] fn accum_zero() -> i64 { 0 }
    #[inline(always)] fn widen(self) -> i64  { self }
}

// ── Unsigned pairs ───────────────────────────────────────────────────────────

impl WideAccum for u8 {
    type Accum = u32;
    #[inline(always)] fn accum_zero() -> u32 { 0 }
    #[inline(always)] fn widen(self) -> u32  { self as u32 }
}

impl WideAccum for u16 {
    type Accum = u64;
    #[inline(always)] fn accum_zero() -> u64 { 0 }
    #[inline(always)] fn widen(self) -> u64  { self as u64 }
}

impl WideAccum for u32 {
    type Accum = u64;
    #[inline(always)] fn accum_zero() -> u64 { 0 }
    #[inline(always)] fn widen(self) -> u64  { self as u64 }
}

impl WideAccum for u64 {
    type Accum = u64;  // identity — already widest
    #[inline(always)] fn accum_zero() -> u64 { 0 }
    #[inline(always)] fn widen(self) -> u64  { self }
}