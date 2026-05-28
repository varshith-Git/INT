// src/nn/embedding.rs
use crate::matrix::Matrix;
use crate::quant::QTensor;

/// A quantized embedding layer mapping token IDs to INT8 dense vectors.
pub struct QEmbedding<const VOCAB: usize, const DIM: usize> {
    pub weights: QTensor<i8, VOCAB, DIM>,
}

impl<const VOCAB: usize, const DIM: usize> QEmbedding<VOCAB, DIM> {
    /// Creates a new quantized embedding layer.
    pub fn new(weights: QTensor<i8, VOCAB, DIM>) -> Self {
        Self { weights }
    }

    /// Looks up a single token ID, returning a 1xDIM tensor.
    ///
    /// Preserves quantization metadata (scale, zero_point).
    /// Panics if token >= VOCAB.
    pub fn lookup(&self, token: usize) -> QTensor<i8, 1, DIM> {
        assert!(token < VOCAB, "Token {} out of bounds for vocab size {}", token, VOCAB);
        
        let mut row = [0i8; DIM];
        for d in 0..DIM {
            row[d] = self.weights.raw(token, d);
        }
        
        let data = Matrix::from_array([row]);
        QTensor::new(data, self.weights.params)
    }

    /// Looks up a sequence of token IDs, returning a SEQxDIM tensor.
    ///
    /// Returns stacked embeddings preserving quantization metadata.
    /// Panics if any token >= VOCAB.
    pub fn lookup_batch<const SEQ: usize>(&self, tokens: [usize; SEQ]) -> QTensor<i8, SEQ, DIM> {
        let mut batch_data = Matrix::<i8, SEQ, DIM>::zeros();
        
        for (i, &token) in tokens.iter().enumerate() {
            assert!(token < VOCAB, "Token {} out of bounds for vocab size {}", token, VOCAB);
            for d in 0..DIM {
                batch_data.data[i][d] = self.weights.raw(token, d);
            }
        }
        
        QTensor::new(batch_data, self.weights.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::scale::QuantParams;

    fn make_test_embedding() -> QEmbedding<4, 3> {
        // Vocab=4, Dim=3
        let params = QuantParams::symmetric(0.5);
        let data = Matrix::from_array([
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ]);
        QEmbedding::new(QTensor::new(data, params))
    }

    #[test]
    fn test_embedding_lookup_single() {
        let emb = make_test_embedding();
        
        let out = emb.lookup(1);
        assert_eq!(out.rows(), 1);
        assert_eq!(out.cols(), 3);
        assert_eq!(out.raw(0, 0), 10);
        assert_eq!(out.raw(0, 1), 11);
        assert_eq!(out.raw(0, 2), 12);
    }

    #[test]
    fn test_embedding_lookup_batch() {
        let emb = make_test_embedding();
        
        let tokens = [2, 0];
        let out = emb.lookup_batch(tokens);
        
        assert_eq!(out.rows(), 2);
        assert_eq!(out.cols(), 3);
        
        // Check first token (2)
        assert_eq!(out.raw(0, 0), 20);
        assert_eq!(out.raw(0, 1), 21);
        assert_eq!(out.raw(0, 2), 22);
        
        // Check second token (0)
        assert_eq!(out.raw(1, 0), 0);
        assert_eq!(out.raw(1, 1), 1);
        assert_eq!(out.raw(1, 2), 2);
    }

    #[test]
    fn test_embedding_preserves_quant_params() {
        let emb = make_test_embedding();
        
        let out_single = emb.lookup(3);
        assert_eq!(out_single.scale(), emb.weights.scale());
        assert_eq!(out_single.zero_point(), emb.weights.zero_point());

        let out_batch = emb.lookup_batch([0, 1]);
        assert_eq!(out_batch.scale(), emb.weights.scale());
        assert_eq!(out_batch.zero_point(), emb.weights.zero_point());
    }

    #[test]
    #[should_panic(expected = "Token 4 out of bounds for vocab size 4")]
    fn test_embedding_out_of_bounds() {
        let emb = make_test_embedding();
        emb.lookup(4); // Panics!
    }
}
