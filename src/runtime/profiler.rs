// src/runtime/profiler.rs

/// An abstract clock trait to allow `no_std` timing injections
/// while supporting `std::time` for host/test environments.
pub trait RuntimeClock {
    /// Returns the current time in platform-specific ticks or microseconds.
    fn now_ticks() -> u64;
}

/// A dummy clock for `no_std` environments without a provided timer.
pub struct DummyClock;

impl RuntimeClock for DummyClock {
    fn now_ticks() -> u64 {
        0
    }
}

#[cfg(feature = "std")]
pub struct StdClock;

#[cfg(feature = "std")]
impl RuntimeClock for StdClock {
    fn now_ticks() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
}
