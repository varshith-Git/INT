//! # Static widening matrix multiplication kernel
//!
//! `matmul_wide` multiplies two stack matrices `A: R×K`, `B: K×C`
//! and returns an output matrix `Out: R×C` whose element type is
//! `T::Accum` — wider than the input type `T`.
//!
//! ## Example
//! ```ignore
//! // i8 inputs → i32 outputs (no overflow on any K ≤ 256)
//! let a: Matrix<i8, 2, 4> = ...;
//! let b: Matrix<i8, 4, 2> = ...;
//! let c: Matrix<i32, 2, 2> = matmul_wide(&a, &b);
//! ```
//!
//! ## Tiling
//! A tile size of 4 is used throughout.  This fits comfortably in the
//! register file of Cortex-M4/M7 (16 × 32-bit GPRs) without spilling.

use super::accum::WideAccum;
use crate::matrix::Matrix;

const TILE: usize = 4;

/// Multiply `a` (R×K) by `b` (K×C), accumulating into wider type.
///
/// Input elements are of type `T`; output elements are `T::Accum`.
/// Tiled 4×4 for cache efficiency on small embedded caches.
#[inline]
pub fn matmul_wide<T, const R: usize, const K: usize, const C: usize>(
    a: &Matrix<T, R, K>,
    b: &Matrix<T, K, C>,
) -> Matrix<T::Accum, R, C>
where
    T: WideAccum,
    T::Accum: Copy + Default,
{
    let mut out = Matrix::<T::Accum, R, C>::zeros();

    let mut i = 0;
    while i < R {
        let i_end = tile_end(i, R);
        let mut k = 0;
        while k < K {
            let k_end = tile_end(k, K);
            let mut j = 0;
            while j < C {
                let j_end = tile_end(j, C);

                // ── 4×4 tile kernel ──────────────────────────────────────
                let mut ii = i;
                while ii < i_end {
                    let mut kk = k;
                    while kk < k_end {
                        // widen once per (ii,kk) pair — reused across j-tile
                        let a_wide: T::Accum = a.data[ii][kk].widen();
                        let mut jj = j;
                        while jj < j_end {
                            // widen b element, multiply in accum space
                            let b_wide: T::Accum = b.data[kk][jj].widen();
                            out.data[ii][jj] =
                                out.data[ii][jj] + a_wide * b_wide;
                            jj += 1;
                        }
                        kk += 1;
                    }
                    ii += 1;
                }
                // ────────────────────────────────────────────────────────

                j += TILE;
            }
            k += TILE;
        }
        i += TILE;
    }

    out
}

#[inline(always)]
const fn tile_end(start: usize, limit: usize) -> usize {
    let end = start + TILE;
    if end < limit { end } else { limit }
}