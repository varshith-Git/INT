// src/runtime/generator.rs
use crate::runtime::kv_cache::{QKVCache, KVCacheError};
use crate::model::QModel;
use crate::quant::QTensor;
use crate::matrix::Matrix;
use crate::runtime::decode::argmax_logits;

/// A quantized autoregressive generator runtime.
pub struct QGenerator<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
> {
    pub model: QModel<VOCAB, SEQ, DIM, HIDDEN>,
    pub cache: QKVCache<SEQ, DIM>,
}

impl<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
> QGenerator<VOCAB, SEQ, DIM, HIDDEN> {
    
    pub fn new(model: QModel<VOCAB, SEQ, DIM, HIDDEN>) -> Self {
        Self {
            model,
            cache: QKVCache::new(),
        }
    }

    /// Forward pass for a single token using incremental KV cache.
    /// Returns `Err(KVCacheError::CacheFull)` if the context window is exhausted.
    pub fn forward_step_incremental(
        &mut self,
        token: usize,
    ) -> Result<QTensor<i8, 1, VOCAB>, KVCacheError> {
        let x_full = self.model.embedding.lookup_batch([token]);

        let mut x_mat = Matrix::<i8, 1, DIM>::zeros();
        for d in 0..DIM {
            x_mat.data[0][d] = x_full.raw(0, d);
        }
        let x = QTensor::new(x_mat, x_full.params);

        let hidden = self.model.transformer.forward_incremental(&x, &mut self.cache)?;
        Ok(self.model.output.forward(&hidden))
    }

    /// Iterative greedy decoding using cached state.
    /// Stops early if the KV cache fills before `max_len` (returns tokens generated so far).
    pub fn greedy_decode(&mut self, start_token: usize, max_len: usize) -> [usize; SEQ] {
        self.cache.reset();

        let mut tokens = [0; SEQ];
        tokens[0] = start_token;
        let limit = max_len.min(SEQ);

        for i in 1..limit {
            match self.forward_step_incremental(tokens[i - 1]) {
                Ok(logits) => {
                    let mut row = [0; VOCAB];
                    for v in 0..VOCAB {
                        row[v] = logits.raw(0, v);
                    }
                    tokens[i] = argmax_logits(&row);
                }
                Err(_) => break, // context window full — return what we have
            }
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::load_from_bytes;
    use crate::model::QModelHeader;

    fn build_generator() -> QGenerator<4, 4, 2, 4> {
        // build deterministic byte array matching a 4,4,2,4 model exactly
        let mut bytes = [0u8; 72];
        let hdr = QModelHeader {
            magic: *b"QMOD",
            version: 1,
            vocab: 4,
            seq: 4,
            dim: 2,
            hidden: 4,
        };
        bytes[0..24].copy_from_slice(&hdr.to_bytes());
        let model = load_from_bytes::<4, 4, 2, 4>(&bytes).unwrap();
        QGenerator::new(model)
    }

    #[test]
    fn test_generator_single_step() {
        let mut gen = build_generator();
        let logits = gen.forward_step_incremental(1).expect("cache not full");
        assert_eq!(logits.rows(), 1);
        assert_eq!(logits.cols(), 4);
        assert_eq!(gen.cache.current_len(), 1);
    }

    #[test]
    fn test_generator_greedy_decode() {
        let mut gen = build_generator();
        let seq = gen.greedy_decode(1, 4);
        assert_eq!(seq[0], 1);
    }

    #[test]
    fn test_generator_deterministic_output() {
        let mut gen = build_generator();
        let seq1 = gen.greedy_decode(1, 4);
        let seq2 = gen.greedy_decode(1, 4);
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn test_generator_sequence_growth() {
        let mut gen = build_generator();
        let seq = gen.greedy_decode(1, 3);
        assert_eq!(seq[0], 1);
        assert_eq!(seq[3], 0);
    }

    #[test]
    fn test_generator_runtime_stability() {
        let mut gen = build_generator();
        let seq = gen.greedy_decode(2, 4);
        assert_eq!(seq.len(), 4);
    }
    
    #[test]
    fn test_incremental_generation() {
        let mut gen = build_generator();
        gen.greedy_decode(1, 4);
        // Ensure cache actually populated correctly through sequence
        assert_eq!(gen.cache.current_len(), 3); 
    }
}
