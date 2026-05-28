//! # matmul_engine
//!
//! A `no_std` integer matrix multiplication engine with **accumulator
//! type separation** — narrow inputs, wide accumulators.
//!
//! ```text
//! Classic:   T  ×  T  →  T          (overflow-prone)
//! This crate: T  ×  T  →  T::Accum  (safe, CMSIS-NN style)
//! ```
//!
//! ## Crate layout
//!
//! ```text
//! matmul_engine
//! ├── matrix.rs          Matrix<T,R,C> / DynMatrix<T>
//! └── kernel/
//!     ├── accum.rs       WideAccum trait + all type pairs
//!     ├── static_wide.rs matmul_wide()     (stack, no alloc)
//!     └── dyn_wide.rs    dyn_matmul_wide() (heap, feature="alloc")
//! ```
//!
//! ## Feature flags
//! | Flag    | What it adds                        |
//! |---------|-------------------------------------|
//! | `alloc` | `DynMatrix<T>` + `dyn_matmul_wide`  |
//! | `std`   | implies `alloc` (for tests/benches) |

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod kernel;
pub mod matrix;
pub mod quant;
pub mod nn;
pub mod runtime;
pub mod model;
pub mod examples;

#[cfg(test)]
pub mod testing;

// ── Public re-exports ────────────────────────────────────────────────────────

pub use kernel::accum::WideAccum;
pub use kernel::static_wide::matmul_wide;
pub use matrix::{Matrix, MatMulScalar};

#[cfg(feature = "alloc")]
pub use kernel::dyn_wide::dyn_matmul_wide;
#[cfg(feature = "alloc")]
pub use matrix::DynMatrix;

// ── Homogeneous kernel (T×T→T, kept from v1) ────────────────────────────────

const TILE: usize = 4;

/// Homogeneous multiply: both input and output share the same type `T`.
///
/// Use `matmul_wide` instead when overflow is a concern.
pub fn matmul<T, const R: usize, const K: usize, const C: usize>(
    a: &Matrix<T, R, K>,
    b: &Matrix<T, K, C>,
) -> Matrix<T, R, C>
where
    T: MatMulScalar,
{
    let mut out = Matrix::<T, R, C>::zeros();
    let mut i = 0;
    while i < R {
        let i_end = te(i, R);
        let mut k = 0;
        while k < K {
            let k_end = te(k, K);
            let mut j = 0;
            while j < C {
                let j_end = te(j, C);
                let mut ii = i;
                while ii < i_end {
                    let mut kk = k;
                    while kk < k_end {
                        let a_ik = a.data[ii][kk];
                        let mut jj = j;
                        while jj < j_end {
                            out.data[ii][jj] = out.data[ii][jj] + a_ik * b.data[kk][jj];
                            jj += 1;
                        }
                        kk += 1;
                    }
                    ii += 1;
                }
                j += TILE;
            }
            k += TILE;
        }
        i += TILE;
    }
    out
}

#[inline(always)]
const fn te(start: usize, limit: usize) -> usize {
    let e = start + TILE;
    if e < limit { e } else { limit }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WideAccum trait verification ─────────────────────────────────────────

    #[test]
    fn test_widen_i8_to_i32() {
        let x: i8 = 127;
        let w: i32 = x.widen();
        assert_eq!(w, 127i32);

        let y: i8 = -128;
        assert_eq!(y.widen(), -128i32);
    }

    #[test]
    fn test_widen_u8_to_u32() {
        let x: u8 = 255;
        assert_eq!(x.widen(), 255u32);
    }

    #[test]
    fn test_widen_i16_to_i64() {
        let x: i16 = i16::MIN;
        assert_eq!(x.widen(), i16::MIN as i64);
    }

    #[test]
    fn test_widen_i32_to_i64() {
        let x: i32 = i32::MAX;
        assert_eq!(x.widen(), i32::MAX as i64);
    }

    // ── Core: i8 × i8 → i32 ──────────────────────────────────────────────────

    #[test]
    fn test_i8_to_i32_2x2() {
        let a = Matrix::from_array([[1i8, 2], [3, 4]]);
        let b = Matrix::from_array([[5i8, 6], [7, 8]]);
        let c: Matrix<i32, 2, 2> = matmul_wide(&a, &b);
        assert_eq!(c.data, [[19i32, 22], [43, 50]]);
    }

    /// Key overflow test: without widening, i8 max_dot would overflow.
    /// K=4 of max i8 (127): 127*127*4 = 64516 — fits i32, NOT i8.
    #[test]
    fn test_i8_overflow_would_happen_without_widening() {
        // dot product of [127,127,127,127] · [127,127,127,127] = 64516
        let a = Matrix::from_array([[127i8, 127, 127, 127]]);
        let b = Matrix::from_array([[127i8], [127], [127], [127]]);
        let c: Matrix<i32, 1, 1> = matmul_wide(&a, &b);
        assert_eq!(c.data[0][0], 64_516i32);  // correct in i32

        // Demonstrate the i8 version would overflow (wraps to garbage):
        // 127*127 = 16129 which is way beyond i8::MAX (127)
        // We just assert the wide result is correct.
    }

    #[test]
    fn test_i8_negative_values() {
        let a = Matrix::from_array([[-1i8, -2], [3, -4]]);
        let b = Matrix::from_array([[2i8, -3], [-4, 5]]);
        let c: Matrix<i32, 2, 2> = matmul_wide(&a, &b);
        // row0: [-1*2 + -2*-4,  -1*-3 + -2*5] = [6, -7]
        // row1: [ 3*2 + -4*-4,   3*-3 + -4*5] = [22, -29]
        assert_eq!(c.data, [[6i32, -7], [22, -29]]);
    }

    // ── u8 × u8 → u32 ────────────────────────────────────────────────────────

    #[test]
    fn test_u8_to_u32_2x2() {
        let a = Matrix::from_array([[200u8, 200], [100, 100]]);
        let b = Matrix::from_array([[200u8, 100], [200, 100]]);
        let c: Matrix<u32, 2, 2> = matmul_wide(&a, &b);
        // [200*200+200*200, 200*100+200*100] = [80000, 40000]
        // [100*200+100*200, 100*100+100*100] = [40000, 20000]
        assert_eq!(c.data, [[80_000u32, 40_000], [40_000, 20_000]]);
    }

    // ── i16 × i16 → i64 ──────────────────────────────────────────────────────

    #[test]
    fn test_i16_to_i64_audio_dsp() {
        // Simulates a 2-tap FIR filter (common in audio DSP)
        let signal  = Matrix::from_array([[1000i16, -2000]]);
        let weights = Matrix::from_array([[300i16], [400i16]]);
        let out: Matrix<i64, 1, 1> = matmul_wide(&signal, &weights);
        // 1000*300 + (-2000)*400 = 300000 - 800000 = -500000
        assert_eq!(out.data[0][0], -500_000i64);
    }

    // ── i32 × i32 → i64 ──────────────────────────────────────────────────────

    #[test]
    fn test_i32_to_i64_wide_accum() {
        let a = Matrix::from_array([[1_000_000i32, 2_000_000]]);
        let b = Matrix::from_array([[3_000_000i32], [4_000_000]]);
        let c: Matrix<i64, 1, 1> = matmul_wide(&a, &b);
        assert_eq!(c.data[0][0], 11_000_000_000_000i64);
    }

    // ── Non-square i8 → i32 ──────────────────────────────────────────────────

    #[test]
    fn test_i8_nonsquare_3x4_times_4x2() {
        let a = Matrix::from_array([
            [1i8, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]);
        let b = Matrix::from_array([
            [10i8, 20],
            [30, 40],
            [50, 60],
            [70, 80],
        ]);
        let c: Matrix<i32, 3, 2> = matmul_wide(&a, &b);
        // identity-like selection
        assert_eq!(c.data, [[10i32, 20], [30, 40], [50, 60]]);
    }

    // ── identity: i64×i64→i64 (already widest) ───────────────────────────────

    #[test]
    fn test_i64_identity_accum() {
        let a = Matrix::from_array([[2i64, 3], [4, 5]]);
        let b = Matrix::from_array([[1i64, 0], [0, 1]]);
        let c: Matrix<i64, 2, 2> = matmul_wide(&a, &b);
        assert_eq!(c.data, [[2i64, 3], [4, 5]]);
    }

    // ── Homogeneous kernel still works (regression) ───────────────────────────

    #[test]
    fn test_homogeneous_i32_regression() {
        let a = Matrix::from_array([[1i32, 2], [3, 4]]);
        let b = Matrix::from_array([[5i32, 6], [7, 8]]);
        let c = matmul(&a, &b);
        assert_eq!(c.data, [[19i32, 22], [43, 50]]);
    }

    // ── Dynamic wide (alloc) ──────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    mod dyn_tests {
        use super::super::*;

        #[test]
        fn test_dyn_i8_to_i32() {
            let a = DynMatrix::<i8>::from_slice(2, 2, &[1, 2, 3, 4]);
            let b = DynMatrix::<i8>::from_slice(2, 2, &[5, 6, 7, 8]);
            let c: DynMatrix<i32> = dyn_matmul_wide(&a, &b);
            assert_eq!(c.get(0, 0), 19);
            assert_eq!(c.get(0, 1), 22);
            assert_eq!(c.get(1, 0), 43);
            assert_eq!(c.get(1, 1), 50);
        }

        #[test]
        fn test_dyn_u8_overflow_safety() {
            // 255*255*2 = 130050 — fine in u32, overflows u8
            let a = DynMatrix::<u8>::from_slice(1, 2, &[255, 255]);
            let b = DynMatrix::<u8>::from_slice(2, 1, &[255, 255]);
            let c: DynMatrix<u32> = dyn_matmul_wide(&a, &b);
            assert_eq!(c.get(0, 0), 130_050u32);
        }

        #[test]
        #[should_panic(expected = "inner dimensions must match")]
        fn test_dyn_dim_mismatch_message() {
            let a = DynMatrix::<i8>::zeros(2, 3);
            let b = DynMatrix::<i8>::zeros(2, 2);
            let _ = dyn_matmul_wide(&a, &b);
        }
    }
}