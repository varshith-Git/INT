// src/nn/pipeline.rs
use crate::nn::embedding::QEmbedding;
use crate::nn::linear::QLinear;
use crate::quant::QTensor;

/// A simple end-to-end quantized feedforward pipeline.
pub struct TinyQModel<
    const VOCAB: usize,
    const EMBED: usize,
    const HIDDEN: usize,
    const OUT: usize,
> {
    pub embedding: QEmbedding<VOCAB, EMBED>,
    pub linear1: QLinear<EMBED, HIDDEN>,
    pub linear2: QLinear<HIDDEN, OUT>,
}

impl<
    const VOCAB: usize,
    const EMBED: usize,
    const HIDDEN: usize,
    const OUT: usize,
> TinyQModel<VOCAB, EMBED, HIDDEN, OUT> {
    /// Creates a new inference model.
    pub fn new(
        embedding: QEmbedding<VOCAB, EMBED>,
        linear1: QLinear<EMBED, HIDDEN>,
        linear2: QLinear<HIDDEN, OUT>,
    ) -> Self {
        Self {
            embedding,
            linear1,
            linear2,
        }
    }

    /// Forward pass through the model.
    ///
    /// token -> QEmbedding -> QLinear1 (with ReLU) -> QLinear2 -> logits
    pub fn forward(&self, token: usize) -> QTensor<i8, 1, OUT> {
        let x = self.embedding.lookup(token);
        let hidden = self.linear1.forward(&x);
        let logits = self.linear2.forward(&hidden);
        logits
    }

    /// Predicts the token with the highest logit score.
    ///
    /// Pure integer comparison only.
    pub fn predict(&self, token: usize) -> usize {
        let logits = self.forward(token);
        
        let mut max_val = core::i8::MIN;
        let mut max_idx = 0;
        
        for i in 0..OUT {
            let val = logits.raw(0, i);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        max_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use crate::quant::scale::QuantParams;

    fn build_test_model() -> TinyQModel<4, 3, 2, 4> {
        // We use symmetric(1.0) so scaling ratios are 1.0 everywhere, simplifying exact math
        let params = QuantParams::symmetric(1.0);

        // Embedding: 4 vocab, 3 dim
        let emb_data = Matrix::from_array([
            [1, 2, 3], // Token 0
            [-1, -2, -3], // Token 1
            [0, 10, 0], // Token 2
            [5, 5, 5], // Token 3
        ]);
        let embedding = QEmbedding::new(QTensor::new(emb_data, params));

        // Linear1: 3 in, 2 out
        let l1_w_data = Matrix::from_array([
            [2, 0],
            [0, 2],
            [1, -1],
        ]);
        let linear1 = QLinear::new(
            l1_w_data,
            &[1.0; 2],
            Some(Matrix::from_array([[1, -1]])), // Bias
            1.0, // input scale
            params,
            true, // ReLU enabled!
        );

        // Linear2: 2 in, 4 out
        let l2_w_data = Matrix::from_array([
            [1, 0, 0, -1],
            [0, 1, -1, 0],
        ]);
        let linear2 = QLinear::new(
            l2_w_data,
            &[1.0; 4],
            None, // No bias
            1.0, // input scale
            params,
            false, // NO ReLU on output layer
        );

        TinyQModel::new(embedding, linear1, linear2)
    }

    #[test]
    fn test_tiny_pipeline_forward() {
        let model = build_test_model();
        
        // Pass token 0:
        // Emb: [1, 2, 3]
        // L1 matmul: [1,2,3] * [[2,0], [0,2], [1,-1]] 
        //   = [2+0+3, 0+4-3] = [5, 1]
        // L1 bias: [5+1, 1-1] = [6, 0]
        // L1 relu: [6, 0]
        // L2 matmul: [6, 0] * [[1,0,0,-1], [0,1,-1,0]]
        //   = [6+0, 0+0, 0+0, -6+0] = [6, 0, 0, -6]
        
        let logits = model.forward(0);
        assert_eq!(logits.rows(), 1);
        assert_eq!(logits.cols(), 4);
        assert_eq!(logits.raw(0, 0), 6);
        assert_eq!(logits.raw(0, 1), 0);
        assert_eq!(logits.raw(0, 2), 0);
        assert_eq!(logits.raw(0, 3), -6);
    }

    #[test]
    fn test_tiny_pipeline_predict() {
        let model = build_test_model();
        // Logits for token 0 are [6, 0, 0, -6]. Max is 6 at index 0.
        let pred = model.predict(0);
        assert_eq!(pred, 0);
    }

    #[test]
    fn test_pipeline_quant_params_consistency() {
        let model = build_test_model();
        let logits = model.forward(0);
        // We set everything to symmetric(1.0)
        assert_eq!(logits.scale(), 1.0);
        assert_eq!(logits.zero_point(), 0);
    }

    #[test]
    fn test_pipeline_relu_stability() {
        let model = build_test_model();
        // Pass token 1:
        // Emb: [-1, -2, -3]
        // L1 matmul: [-1,-2,-3] * [[2,0], [0,2], [1,-1]]
        //   = [-2+0-3, 0-4+3] = [-5, -1]
        // L1 bias: [-5+1, -1-1] = [-4, -2]
        // L1 relu: [0, 0]    <-- Negative activations clamped!
        // L2 matmul: [0, 0] * [[1,0,0,-1], [0,1,-1,0]] = [0, 0, 0, 0]
        
        let logits = model.forward(1);
        assert_eq!(logits.raw(0, 0), 0);
        assert_eq!(logits.raw(0, 1), 0);
        assert_eq!(logits.raw(0, 2), 0);
        assert_eq!(logits.raw(0, 3), 0);
    }
}
