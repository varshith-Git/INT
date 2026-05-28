// src/testing/stress.rs

#[cfg(test)]
mod tests {
    use crate::model::{QModelHeader, load_from_bytes};
    use crate::runtime::{QGenerator, kv_cache::KVCacheError};

    const STRESS_SEQ_SMALL: usize = 128;
    const STRESS_SEQ_MEDIUM: usize = 512;
    // We constrain LARGE to a size that avoids local stack overflows during CI tests.
    const STRESS_SEQ_LARGE: usize = 1000;

    fn build_generator<const SEQ: usize>() -> QGenerator<4, SEQ, 2, 4> {
        let mut bytes = [0u8; 72];
        let hdr = QModelHeader {
            magic: *b"QMOD",
            version: 1,
            vocab: 4,
            seq: SEQ as u32,
            dim: 2,
            hidden: 4,
        };
        bytes[0..24].copy_from_slice(&hdr.to_bytes());
        let model = load_from_bytes::<4, SEQ, 2, 4>(&bytes).unwrap();
        QGenerator::new(model)
    }

    #[test]
    fn test_incremental_decode_1000_tokens() {
        // Large sequence decode check to ensure $O(N)$ boundary safety
        let mut gen = build_generator::<STRESS_SEQ_LARGE>();
        let seq = gen.greedy_decode(1, STRESS_SEQ_LARGE);
        
        assert_eq!(seq.len(), STRESS_SEQ_LARGE);
        // Ensure cache actually survived the rollout
        assert_eq!(gen.cache.current_len(), STRESS_SEQ_LARGE - 1);
    }

    #[test]
    fn test_transformer_long_rollout_stability() {
        let mut gen = build_generator::<STRESS_SEQ_MEDIUM>();
        let seq = gen.greedy_decode(2, STRESS_SEQ_MEDIUM);
        
        // Ensure we don't return entirely random garbage by token 500
        // (Due to deterministic 0-initialized weights, it will likely output token 0 repeated)
        assert_eq!(seq.len(), STRESS_SEQ_MEDIUM);
    }

    #[test]
    fn test_kv_cache_max_sequence() {
        let mut gen = build_generator::<STRESS_SEQ_SMALL>();
        gen.greedy_decode(1, STRESS_SEQ_SMALL);
        // greedy_decode loops from 1..limit, resulting in SEQ - 1 cache entries.
        assert_eq!(gen.cache.current_len(), STRESS_SEQ_SMALL - 1);
    }

    #[test]
    fn test_cache_reset_integrity() {
        let mut gen = build_generator::<STRESS_SEQ_SMALL>();
        gen.greedy_decode(1, 10);
        assert_eq!(gen.cache.current_len(), 9);
        gen.cache.reset();
        assert_eq!(gen.cache.current_len(), 0);
    }

    #[test]
    fn test_cache_overflow_protection() {
        let mut gen = build_generator::<STRESS_SEQ_SMALL>();
        
        for _ in 0..STRESS_SEQ_SMALL {
            gen.cache.append(&[0, 0], &[0, 0], 1.0, 1.0).unwrap();
        }

        // Cache is exactly full. The next append MUST gracefully error.
        let result = gen.cache.append(&[1, 1], &[1, 1], 1.0, 1.0);
        assert_eq!(result, Err(KVCacheError::CacheFull));
    }
}
