//! Low-level compute kernels.
//!
//! This module is intentionally flat: consumers import from the
//! crate root, not from here directly.

pub mod accum;
pub mod static_wide;

#[cfg(feature = "alloc")]
pub mod dyn_wide;