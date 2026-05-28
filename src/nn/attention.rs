// src/nn/attention.rs
use crate::matrix::Matrix;
use crate::quant::QTensor;
use crate::kernel::static_wide::matmul_wide;
use crate::runtime::kv_cache::QKVCache;
use crate::nn::softmax::QSoftmax;

/// A stateless quantized attention primitive.
/// Computes raw unscaled scores and routes values without softmax.

pub struct QAttention<const SEQ: usize, const DIM: usize> {}

impl<const SEQ: usize, const DIM: usize> QAttention<SEQ, DIM> {
    /// Computes attention scores: Q x K^T
    ///
    /// INT8 x INT8 -> INT32 accumulation.
    /// No floating point, no normalization, no softmax.
    pub fn compute_scores(
        q: &QTensor<i8, SEQ, DIM>,
        k: &QTensor<i8, SEQ, DIM>,
    ) -> Matrix<i32, SEQ, SEQ> {
        // Transpose K manually into a temporary DIM x SEQ matrix
        // so we can utilize the existing highly optimized matmul_wide kernel.
        let mut k_t = Matrix::<i8, DIM, SEQ>::zeros();
        for r in 0..SEQ {
            for c in 0..DIM {
                k_t.data[c][r] = k.raw(r, c);
            }
        }
        
        // Q (SEQ x DIM) * K^T (DIM x SEQ) = Scores (SEQ x SEQ)
        matmul_wide(&q.data, &k_t)
    }

    /// Routes values based on normalized attention probabilities: probs x V
    ///
    /// Temporarily computes weighted aggregation in f32 logic.
    pub fn apply_softmax_values(
        probs: &Matrix<f32, SEQ, SEQ>,
        v: &QTensor<i8, SEQ, DIM>,
    ) -> Matrix<f32, SEQ, DIM> {
        let mut out = Matrix::<f32, SEQ, DIM>::zeros();
        
        for i in 0..SEQ {
            for j in 0..DIM {
                let mut sum: f32 = 0.0;
                for k in 0..SEQ {
                    let prob = probs.data[i][k];
                    let val = v.raw(k, j) as f32;
                    sum += prob * val;
                }
                out.data[i][j] = sum;
            }
        }
        
        out
    }

    /// End-to-end stateless forward pipeline for attention logic.
    pub fn forward(
        q: &QTensor<i8, SEQ, DIM>,
        k: &QTensor<i8, SEQ, DIM>,
        v: &QTensor<i8, SEQ, DIM>,
    ) -> Matrix<f32, SEQ, DIM> {
        let scores = Self::compute_scores(q, k);
        let probs = QSoftmax::softmax_matrix(&scores, SEQ);
        Self::apply_softmax_values(&probs, v)
    }

    /// Incremental attention over cached K/V using explicit active length bounds.
    pub fn compute_scores_incremental(
        q: &QTensor<i8, 1, DIM>,
        cache: &QKVCache<SEQ, DIM>,
    ) -> Matrix<i32, 1, SEQ> {
        let mut scores = Matrix::<i32, 1, SEQ>::zeros();
        let len = cache.current_len();

        for t in 0..len {
            let mut sum: i32 = 0;
            for d in 0..DIM {
                let q_val = q.raw(0, d) as i32;
                let k_val = cache.keys.data[t][d] as i32;
                sum = sum.wrapping_add(q_val.wrapping_mul(k_val));
            }
            scores.data[0][t] = sum;
        }
        scores
    }

    pub fn apply_softmax_values_incremental(
        probs: &Matrix<f32, 1, SEQ>,
        cache: &QKVCache<SEQ, DIM>,
    ) -> Matrix<f32, 1, DIM> {
        let mut out = Matrix::<f32, 1, DIM>::zeros();
        let len = cache.current_len();

        for d in 0..DIM {
            let mut sum: f32 = 0.0;
            for t in 0..len {
                let prob = probs.data[0][t];
                let v_val = cache.values.data[t][d] as f32;
                sum += prob * v_val;
            }
            out.data[0][d] = sum;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::scale::QuantParams;

    fn build_tensors() -> (QTensor<i8, 3, 2>, QTensor<i8, 3, 2>, QTensor<i8, 3, 2>) {
        let params = QuantParams::symmetric(1.0);
        
        // Q: [3, 2]
        let q_data = Matrix::from_array([
            [1, 0],
            [0, 1],
            [-1, -1]
        ]);
        let q = QTensor::new(q_data, params);

        // K: [3, 2]
        let k_data = Matrix::from_array([
            [1, 0],
            [0, 1],
            [1, 1]
        ]);
        let k = QTensor::new(k_data, params);

        // V: [3, 2]
        let v_data = Matrix::from_array([
            [10, 20],
            [30, 40],
            [50, 60]
        ]);
        let v = QTensor::new(v_data, params);
        
        (q, k, v)
    }

    #[test]
    fn test_attention_score_shape() {
        let (q, k, _) = build_tensors();
        let scores = QAttention::<3, 2>::compute_scores(&q, &k);
        assert_eq!(scores.rows(), 3);
        assert_eq!(scores.cols(), 3);
    }

    #[test]
    fn test_attention_value_shape() {
        let (q, k, v) = build_tensors();
        let scores = QAttention::<3, 2>::compute_scores(&q, &k);
        let probs = QSoftmax::softmax_matrix(&scores, 3);
        let out = QAttention::<3, 2>::apply_softmax_values(&probs, &v);
        assert_eq!(out.rows(), 3);
        assert_eq!(out.cols(), 2);
    }

    #[test]
    fn test_attention_self_similarity() {
        // verify identical Q/K produce stronger diagonal scores
        let params = QuantParams::symmetric(1.0);
        let x_data = Matrix::from_array([
            [2, 0],
            [0, 2]
        ]);
        let x = QTensor::new(x_data, params);
        
        let scores = QAttention::<2, 2>::compute_scores(&x, &x);
        
        assert_eq!(scores.data[0][0], 4);
        assert_eq!(scores.data[1][1], 4);
    }

    #[test]
    fn test_attention_forward_pipeline() {
        let (q, k, v) = build_tensors();
        
        let out = QAttention::<3, 2>::forward(&q, &k, &v);
        // We know V is [10, 20], [30, 40], [50, 60]. Max possible is 50, 60.
        // As long as it doesn't crash and shape is correct, we are good.
        assert_eq!(out.rows(), 3);
        assert_eq!(out.cols(), 2);
    }

    #[test]
    fn test_attention_negative_values() {
        let (q, k, _v) = build_tensors();
        let params = QuantParams::symmetric(1.0);
        let v_neg = QTensor::new(Matrix::<i8, 3, 2>::from_array([
            [-10, -20],
            [-30, -40],
            [-50, -60]
        ]), params);
        let out = QAttention::<3, 2>::forward(&q, &k, &v_neg);
        assert!(out.data[2][0] < 0.0);
        assert!(out.data[2][1] < 0.0);
    }

    #[test]
    fn test_attention_with_cache() {
        use crate::runtime::kv_cache::QKVCache;
        let params = QuantParams::symmetric(1.0);

        let mut cache = QKVCache::<3, 2>::new();
        
        // Step 1
        cache.append(&[1, 0], &[5, 5], 1.0, 1.0).unwrap();
        let q1 = QTensor::new(Matrix::<i8, 1, 2>::from_array([[1, 1]]), params);
        let k1 = QTensor::new(Matrix::<i8, 1, 2>::from_array([[1, 0]]), params);
        let v1 = QTensor::new(Matrix::<i8, 1, 2>::from_array([[5, 5]]), params);
        
        let full1_scores = QAttention::<1, 2>::compute_scores(&q1, &k1);
        let full1_probs = QSoftmax::softmax_matrix(&full1_scores, 1);
        let full1_out = QAttention::<1, 2>::apply_softmax_values(&full1_probs, &v1);

        let score1 = QAttention::<3, 2>::compute_scores_incremental(&q1, &cache);
        let probs1 = QSoftmax::softmax_matrix(&score1, cache.current_len());
        let out1 = QAttention::<3, 2>::apply_softmax_values_incremental(&probs1, &cache);
        assert!((full1_out.data[0][0] - out1.data[0][0]).abs() < 1e-5);

        // Step 2
        cache.append(&[0, 1], &[6, 6], 1.0, 1.0).unwrap();
        let q2 = QTensor::new(Matrix::<i8, 2, 2>::from_array([[1, 1], [2, 2]]), params);
        let k2 = QTensor::new(Matrix::<i8, 2, 2>::from_array([[1, 0], [0, 1]]), params);
        let v2 = QTensor::new(Matrix::<i8, 2, 2>::from_array([[5, 5], [6, 6]]), params);
        
        let full2_scores = QAttention::<2, 2>::compute_scores(&q2, &k2);
        let full2_probs = QSoftmax::softmax_matrix(&full2_scores, 2);
        let full2_out = QAttention::<2, 2>::apply_softmax_values(&full2_probs, &v2);

        let q2_inc = QTensor::new(Matrix::<i8, 1, 2>::from_array([[2, 2]]), params);
        let score2 = QAttention::<3, 2>::compute_scores_incremental(&q2_inc, &cache);
        let probs2 = QSoftmax::softmax_matrix(&score2, cache.current_len());
        let out2 = QAttention::<3, 2>::apply_softmax_values_incremental(&probs2, &cache);
        
        assert!((full2_out.data[1][0] - out2.data[0][0]).abs() < 1e-5);
    }
}
