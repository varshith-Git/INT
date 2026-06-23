// src/runtime/sampler.rs
//! Deterministic sampling for autoregressive decoding.
//!
//! Greedy decoding is reproducible but degenerate. This module adds
//! **temperature**, **top-k**, and **top-p (nucleus)** sampling that stay
//! *bit-reproducible*: the RNG is a seeded SplitMix64 stream, so
//! `(seed, step, logits) -> token` is fixed on every architecture (x86, ARM,
//! RISC-V, WASM). Log the seed and the entire generation replays exactly — the
//! same determinism contract the rest of the engine keeps. No global state, no
//! entropy source, no heap.

use libm::expf;

/// Deterministic, `no_std` PRNG (SplitMix64). Same seed → identical stream on
/// every target. Used so sampled generation is reproducible from a logged seed.
#[derive(Debug, Clone)]
pub struct DetRng {
    state: u64,
}

impl DetRng {
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64 bits of the SplitMix64 stream.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)`. Uses the top 24 bits for an exact dyadic value.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }
}

/// Sampling configuration.
///
/// `temperature <= 0.0` selects **greedy** (argmax) and ignores the other knobs.
/// `top_k == 0` disables top-k. `top_p >= 1.0` disables nucleus filtering.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl SamplingConfig {
    /// Greedy decoding (deterministic argmax). The seed is irrelevant.
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 0, top_p: 1.0, seed: 0 }
    }

    /// Plain temperature sampling with a fixed seed.
    pub fn temperature(temp: f32, seed: u64) -> Self {
        Self { temperature: temp, top_k: 0, top_p: 1.0, seed }
    }

    /// Top-k sampling with a fixed seed.
    pub fn top_k(temp: f32, k: usize, seed: u64) -> Self {
        Self { temperature: temp, top_k: k, top_p: 1.0, seed }
    }

    /// Nucleus (top-p) sampling with a fixed seed.
    pub fn top_p(temp: f32, p: f32, seed: u64) -> Self {
        Self { temperature: temp, top_k: 0, top_p: p, seed }
    }
}

/// Sample one token index from `logits` according to `cfg`, drawing from `rng`.
///
/// Pure and deterministic: identical `(logits, cfg, rng-state)` yields an
/// identical index on every target. Probabilities are computed in `f32`
/// (consistent with the engine's hybrid softmax), filtered by top-k / top-p,
/// then drawn by inverse-CDF.
pub fn sample<const VOCAB: usize>(
    logits: &[i8; VOCAB],
    cfg: &SamplingConfig,
    rng: &mut DetRng,
) -> usize {
    // Greedy fast-path — also the `temperature -> 0` limit.
    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }

    // 1. Scaled logits → numerically-stable softmax in f32.
    let inv_t = 1.0 / cfg.temperature;
    let mut max_l = f32::NEG_INFINITY;
    for &l in logits.iter() {
        let s = (l as f32) * inv_t;
        if s > max_l {
            max_l = s;
        }
    }
    let mut probs = [0.0f32; VOCAB];
    let mut sum = 0.0f32;
    for v in 0..VOCAB {
        let p = expf((logits[v] as f32) * inv_t - max_l);
        probs[v] = p;
        sum += p;
    }
    let inv_sum = 1.0 / (sum + 1e-9);
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // 2. Rank indices by probability, descending. Insertion sort is adequate for
    //    the small vocabularies this engine targets; a partial heap is the
    //    follow-up for large-vocab models.
    let mut order = [0usize; VOCAB];
    for (v, slot) in order.iter_mut().enumerate() {
        *slot = v;
    }
    for i in 1..VOCAB {
        let mut j = i;
        while j > 0 && probs[order[j]] > probs[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }

    // 3. top-k: keep at most k highest. top-p: keep the smallest ranked prefix
    //    whose cumulative mass reaches p. Both operate on the ranked order.
    let k_limit = if cfg.top_k == 0 { VOCAB } else { cfg.top_k.min(VOCAB) };
    let mut kept = 0usize;
    let mut kept_mass = 0.0f32;
    while kept < k_limit {
        let idx = order[kept];
        kept_mass += probs[idx];
        kept += 1;
        if cfg.top_p < 1.0 && kept_mass >= cfg.top_p {
            break;
        }
    }
    if kept == 0 {
        return order[0];
    }

    // 4. Inverse-CDF draw within the kept (renormalized) set.
    let target = rng.next_f32() * kept_mass;
    let mut acc = 0.0f32;
    for r in 0..kept {
        let idx = order[r];
        acc += probs[idx];
        if acc >= target {
            return idx;
        }
    }
    order[kept - 1] // floating-point guard
}

#[inline]
fn argmax<const VOCAB: usize>(logits: &[i8; VOCAB]) -> usize {
    let mut best = 0;
    for v in 1..VOCAB {
        if logits[v] > logits[best] {
            best = v;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_across_instances() {
        let mut a = DetRng::new(42);
        let mut b = DetRng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_next_f32_in_unit_interval() {
        let mut r = DetRng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits: [i8; 5] = [3, -1, 9, 2, 8];
        let cfg = SamplingConfig::greedy();
        let mut rng = DetRng::new(123);
        // index 2 is the max
        assert_eq!(sample(&logits, &cfg, &mut rng), 2);
    }

    #[test]
    fn top_k_1_always_returns_argmax() {
        let logits: [i8; 6] = [0, 5, 1, 9, 2, 4];
        for seed in 0..50 {
            let cfg = SamplingConfig::top_k(1.0, 1, seed);
            let mut rng = DetRng::new(seed);
            assert_eq!(sample(&logits, &cfg, &mut rng), 3, "argmax must win under top_k=1");
        }
    }

    #[test]
    fn same_seed_same_token() {
        let logits: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let cfg = SamplingConfig::temperature(1.5, 99);
        let t1 = sample(&logits, &cfg, &mut DetRng::new(cfg.seed));
        let t2 = sample(&logits, &cfg, &mut DetRng::new(cfg.seed));
        assert_eq!(t1, t2);
    }

    #[test]
    fn sample_index_always_in_range() {
        let logits: [i8; 4] = [10, -20, 30, 5];
        for seed in 0..200 {
            let cfg = SamplingConfig::temperature(0.8, seed);
            let idx = sample(&logits, &cfg, &mut DetRng::new(seed));
            assert!(idx < 4);
        }
    }

    #[test]
    fn tiny_top_p_collapses_to_top_token() {
        // A peaked distribution + a tiny nucleus must always pick the top token.
        let logits: [i8; 5] = [0, 0, 40, 0, 0];
        for seed in 0..50 {
            let cfg = SamplingConfig::top_p(1.0, 0.05, seed);
            let idx = sample(&logits, &cfg, &mut DetRng::new(seed));
            assert_eq!(idx, 2);
        }
    }

    #[test]
    fn higher_temperature_spreads_mass() {
        // With near-uniform logits and high temperature, sampling should not be
        // pinned to one index across many seeds (sanity check on the RNG path).
        let logits: [i8; 4] = [1, 0, 1, 0];
        let mut seen = [false; 4];
        for seed in 0..400 {
            let cfg = SamplingConfig::temperature(5.0, seed);
            let idx = sample(&logits, &cfg, &mut DetRng::new(seed));
            seen[idx] = true;
        }
        let distinct = seen.iter().filter(|&&s| s).count();
        assert!(distinct >= 2, "expected spread, saw {distinct} distinct tokens");
    }
}
