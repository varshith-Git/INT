//! Matrix storage types.
//!
//! - [`Matrix<T, R, C>`]  — stack-allocated, const-generic
//! - [`DynMatrix<T>`]     — heap-backed, requires `feature = "alloc"`

use core::fmt;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// ─────────────────────────────────────────────────────────────────────────────
// MatMulScalar  (homogeneous T×T→T, kept for the original kernel)
// ─────────────────────────────────────────────────────────────────────────────

use core::ops::{Add, Mul};

/// Scalar trait for homogeneous (non-widening) multiplication.
pub trait MatMulScalar:
    Copy + Default + Add<Output = Self> + Mul<Output = Self> + PartialEq + fmt::Debug
{
    fn zero() -> Self;
}

macro_rules! impl_scalar {
    ($($t:ty),*) => {
        $(impl MatMulScalar for $t { #[inline(always)] fn zero() -> Self { 0 } })*
    };
}
impl_scalar!(i8, i16, i32, i64, u8, u16, u32, u64);

// ─────────────────────────────────────────────────────────────────────────────
// Matrix<T, R, C>  — stack storage
// ─────────────────────────────────────────────────────────────────────────────

/// Stack-allocated, row-major matrix.
///
/// `data[r][c]` accesses row `r`, column `c`.
/// Zero heap, zero runtime size overhead.
#[derive(Clone, Copy, PartialEq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub data: [[T; C]; R],
}

impl<T: Copy + Default, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Create a matrix filled with `T::default()` (zero for all numerics).
    #[inline]
    pub fn zeros() -> Self {
        Self { data: [[T::default(); C]; R] }
    }

    /// Create from a 2-D array literal.
    #[inline]
    pub fn from_array(data: [[T; C]; R]) -> Self {
        Self { data }
    }

    #[inline] pub fn get(&self, r: usize, c: usize) -> T  { self.data[r][c] }
    #[inline] pub fn set(&mut self, r: usize, c: usize, v: T) { self.data[r][c] = v; }
    #[inline] pub const fn rows(&self) -> usize { R }
    #[inline] pub const fn cols(&self) -> usize { C }
}

impl<T: Copy + Default + From<u8>, const N: usize> Matrix<T, N, N> {
    /// Square identity matrix.
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..N { m.data[i][i] = T::from(1u8); }
        m
    }
}

impl<T: Copy + Default + fmt::Display, const R: usize, const C: usize>
    fmt::Debug for Matrix<T, R, C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..R {
            write!(f, "[")?;
            for c in 0..C {
                if c > 0 { write!(f, ", ")?; }
                write!(f, "{:>6}", self.data[r][c])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DynMatrix<T>  — heap storage (feature = "alloc")
// ─────────────────────────────────────────────────────────────────────────────

/// Heap-backed row-major matrix for runtime-determined sizes.
#[cfg(feature = "alloc")]
#[derive(Clone, PartialEq)]
pub struct DynMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: Copy + Default> DynMatrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: (0..rows * cols).map(|_| T::default()).collect() }
    }

    pub fn from_slice(rows: usize, cols: usize, src: &[T]) -> Self {
        assert_eq!(src.len(), rows * cols, "DynMatrix::from_slice: length mismatch");
        Self { rows, cols, data: src.to_vec() }
    }

    #[inline] pub fn get(&self, r: usize, c: usize) -> T  { self.data[r * self.cols + c] }
    #[inline] pub fn set(&mut self, r: usize, c: usize, v: T) { self.data[r * self.cols + c] = v; }
    #[inline] pub fn rows(&self) -> usize { self.rows }
    #[inline] pub fn cols(&self) -> usize { self.cols }
}

#[cfg(feature = "alloc")]
impl<T: Copy + Default + fmt::Display> fmt::Debug for DynMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.rows {
            write!(f, "[")?;
            for c in 0..self.cols {
                if c > 0 { write!(f, ", ")?; }
                write!(f, "{:>6}", self.get(r, c))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}