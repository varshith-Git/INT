//! # Dynamic widening matrix multiplication kernel
//!
//! Heap-backed equivalent of `static_wide::matmul_wide`.
//! Requires `feature = "alloc"`.
//!
//! Uses the same tiling strategy and `WideAccum` abstraction;
//! input type `T` widens to `T::Accum` during accumulation.

use super::accum::WideAccum;
use crate::matrix::DynMatrix;

const TILE: usize = 4;

/// Multiply `a` by `b` with widening accumulation.
///
/// # Panics
/// Panics if `a.cols() != b.rows()`.
pub fn dyn_matmul_wide<T>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
) -> DynMatrix<T::Accum>
where
    T: WideAccum + Default,
    T::Accum: Copy + Default,
{
    assert_eq!(
        a.cols(), b.rows(),
        "matmul_wide: inner dimensions must match ({}×{} × {}×{})",
        a.rows(), a.cols(), b.rows(), b.cols()
    );

    let (r, k, c) = (a.rows(), a.cols(), b.cols());
    let mut out = DynMatrix::<T::Accum>::zeros(r, c);

    let mut i = 0;
    while i < r {
        let i_end = tile_end(i, r);
        let mut kk = 0;
        while kk < k {
            let k_end = tile_end(kk, k);
            let mut j = 0;
            while j < c {
                let j_end = tile_end(j, c);

                let mut ii = i;
                while ii < i_end {
                    let mut kkk = kk;
                    while kkk < k_end {
                        let a_wide: T::Accum = a.get(ii, kkk).widen();
                        let mut jj = j;
                        while jj < j_end {
                            let b_wide: T::Accum = b.get(kkk, jj).widen();
                            let prev = out.get(ii, jj);
                            out.set(ii, jj, prev + a_wide * b_wide);
                            jj += 1;
                        }
                        kkk += 1;
                    }
                    ii += 1;
                }

                j += TILE;
            }
            kk += TILE;
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