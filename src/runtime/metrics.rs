// src/runtime/metrics.rs

/// Metrics tracker for runtime observability.
/// Can be feature-gated internally but the struct is always exposed.
#[derive(Debug, Default, Clone)]
pub struct RuntimeMetrics {
    pub tokens_generated: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub max_activation: i32,
    pub softmax_calls: usize,
    pub attention_ops: usize,
    
    // Saturation events track how often quantization hits i8 bounds (-128 or 127).
    // Extreme saturation indicates quantization collapse or drift.
    pub saturation_events_neg: usize,
    pub saturation_events_pos: usize,
}

impl RuntimeMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "debug-runtime")]
    pub fn record_saturation(&mut self, value: i32) {
        if value <= -128 {
            self.saturation_events_neg += 1;
        } else if value >= 127 {
            self.saturation_events_pos += 1;
        }
    }

    #[cfg(not(feature = "debug-runtime"))]
    #[inline(always)]
    pub fn record_saturation(&mut self, _value: i32) {
        // No-op in release unless explicitly debugging runtime
    }

    pub fn record_token_generated(&mut self) {
        self.tokens_generated += 1;
    }
}
