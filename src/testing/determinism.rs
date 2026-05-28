// src/testing/determinism.rs
use crate::runtime::kv_cache::QKVCache;

/// FNV-1a style rolling checksum for deterministic state validation.
/// Does not require heap allocation or `std::collections::hash_map`.
pub fn rolling_checksum_i8(data: &[i8]) -> u32 {
    let mut hash = 2166136261u32;
    for &byte in data {
        hash ^= byte as u8 as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

pub fn runtime_state_hash<const SEQ: usize, const DIM: usize>(
    cache: &QKVCache<SEQ, DIM>,
    logits: &[i8],
) -> u32 {
    let mut hash = 2166136261u32;
    
    // Hash valid cache region
    let len = cache.current_len();
    for t in 0..len {
        for d in 0..DIM {
            hash ^= cache.keys.data[t][d] as u8 as u32;
            hash = hash.wrapping_mul(16777619);
            
            hash ^= cache.values.data[t][d] as u8 as u32;
            hash = hash.wrapping_mul(16777619);
        }
    }
    
    // Hash logits
    for &byte in logits {
        hash ^= byte as u8 as u32;
        hash = hash.wrapping_mul(16777619);
    }
    
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{QModelHeader, load_from_bytes};
    use crate::runtime::QGenerator;

    fn build_dummy_generator() -> QGenerator<4, 8, 2, 4> {
        let mut bytes = [0u8; 72];
        let hdr = QModelHeader {
            magic: *b"QMOD",
            version: 1,
            vocab: 4,
            seq: 8,
            dim: 2,
            hidden: 4,
        };
        bytes[0..24].copy_from_slice(&hdr.to_bytes());
        let model = load_from_bytes::<4, 8, 2, 4>(&bytes).unwrap();
        QGenerator::new(model)
    }

    #[test]
    fn test_deterministic_generation_replay() {
        let mut gen1 = build_dummy_generator();
        let mut gen2 = build_dummy_generator();

        let seq1 = gen1.greedy_decode(1, 8);
        let seq2 = gen2.greedy_decode(1, 8);

        // Prove absolute array equality for generated tokens
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn test_cache_state_divergence() {
        let mut gen1 = build_dummy_generator();
        let mut gen2 = build_dummy_generator();

        gen1.greedy_decode(1, 4);
        gen2.greedy_decode(1, 4);

        let hash1 = runtime_state_hash(&gen1.cache, &[]);
        let hash2 = runtime_state_hash(&gen2.cache, &[]);

        assert_eq!(hash1, hash2, "Cache divergence detected between identical runs!");
    }
}
