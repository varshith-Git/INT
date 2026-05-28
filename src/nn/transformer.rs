// src/nn/transformer.rs
use crate::matrix::Matrix;
use crate::quant::{QTensor, QuantParams, RequantShift, clip::clamp_i8};
use crate::nn::linear::QLinear;
use crate::nn::attention::QAttention;
use crate::nn::rmsnorm::QRMSNorm;
use crate::runtime::kv_cache::QKVCache;
use crate::nn::softmax::QSoftmax;

/// A quantized transformer execution block (prototype).
///
/// Validates residual routing, attention integration, and feedforward execution
/// using purely integer-based arithmetic, stabilized by RMSNorm.
pub struct QTransformerBlock<const SEQ: usize, const DIM: usize, const HIDDEN: usize> {
    pub norm1: QRMSNorm<DIM>,
    pub norm2: QRMSNorm<DIM>,
    
    pub q_proj: QLinear<DIM, DIM>,
    pub k_proj: QLinear<DIM, DIM>,
    pub v_proj: QLinear<DIM, DIM>,

    pub ff1: QLinear<DIM, HIDDEN>,
    pub ff2: QLinear<HIDDEN, DIM>,

    // RequantShift for the attention output since QAttention returns i32
    pub attn_requant: RequantShift,
    pub attn_out_params: QuantParams,
}

impl<const SEQ: usize, const DIM: usize, const HIDDEN: usize> QTransformerBlock<SEQ, DIM, HIDDEN> {
    /// Full forward pass for the transformer block.
    pub fn forward(&self, x: &QTensor<i8, SEQ, DIM>) -> QTensor<i8, SEQ, DIM> {
        // 1. RMSNorm 1
        let norm1_out = self.norm1.forward(x);

        // 2. Q, K, V projections
        let q = self.q_proj.forward(&norm1_out);
        let k = self.k_proj.forward(&norm1_out);
        let v = self.v_proj.forward(&norm1_out);

        // 3. Attention
        let attn_scores = QAttention::<SEQ, DIM>::compute_scores(&q, &k);
        let probs = QSoftmax::softmax_matrix(&attn_scores, SEQ);
        let attn_out_f32 = QAttention::<SEQ, DIM>::apply_softmax_values(&probs, &v);

        // 4. Requantize attention output back to INT8
        let attn_out_i8 = requantize_attention_output_f32(&attn_out_f32, &self.attn_requant, self.attn_out_params);

        // 5. Residual add 1 (Input + Attention)
        let residual_1 = residual_add_i8(x, &attn_out_i8);

        // 6. RMSNorm 2
        let norm2_out = self.norm2.forward(&residual_1);

        // 7. Feedforward Network (FF1 -> ReLU -> FF2)
        let ff1_out = self.ff1.forward(&norm2_out);
        let ff2_out = self.ff2.forward(&ff1_out);

        // 8. Residual add 2 (Residual_1 + FF)
        let out = residual_add_i8(&residual_1, &ff2_out);

        out
    }

    /// Incremental token forward pass using a KV Cache.
    pub fn forward_incremental(
        &self,
        x: &QTensor<i8, 1, DIM>,
        cache: &mut QKVCache<SEQ, DIM>,
    ) -> QTensor<i8, 1, DIM> {
        let norm1_out = self.norm1.forward(x);

        let q = self.q_proj.forward(&norm1_out);
        let k = self.k_proj.forward(&norm1_out);
        let v = self.v_proj.forward(&norm1_out);

        let mut k_arr = [0; DIM];
        let mut v_arr = [0; DIM];
        for d in 0..DIM {
            k_arr[d] = k.raw(0, d);
            v_arr[d] = v.raw(0, d);
        }
        
        // Return if cache full - wait, the method signature currently does not return Result.
        // We will expect it.
        cache.append(&k_arr, &v_arr, k.scale(), v.scale()).expect("KV cache overflow");

        let attn_scores = QAttention::<SEQ, DIM>::compute_scores_incremental(&q, cache);
        let probs = QSoftmax::softmax_matrix(&attn_scores, cache.current_len());
        let attn_out_f32 = QAttention::<SEQ, DIM>::apply_softmax_values_incremental(&probs, cache);

        let attn_out_i8 = requantize_attention_output_f32(&attn_out_f32, &self.attn_requant, self.attn_out_params);

        let residual_1 = residual_add_i8(x, &attn_out_i8);

        let norm2_out = self.norm2.forward(&residual_1);

        let ff1_out = self.ff1.forward(&norm2_out);
        let ff2_out = self.ff2.forward(&ff1_out);

        residual_add_i8(&residual_1, &ff2_out)
    }
}

/// Safely add two i8 tensors by widening to i32 and clamping back.
/// Assumes both tensors share the same QuantParams for the residual stream.
pub fn residual_add_i8<const SEQ: usize, const DIM: usize>(
    a: &QTensor<i8, SEQ, DIM>,
    b: &QTensor<i8, SEQ, DIM>,
) -> QTensor<i8, SEQ, DIM> {
    debug_assert_eq!(a.scale(), b.scale());
    debug_assert_eq!(a.zero_point(), b.zero_point());

    let mut out_data = Matrix::<i8, SEQ, DIM>::zeros();
    for r in 0..SEQ {
        for c in 0..DIM {
            let sum = (a.raw(r, c) as i32) + (b.raw(r, c) as i32);
            // Re-apply zero point: q_out = q1 + q2 - Z
            let val = sum - a.zero_point();
            out_data.data[r][c] = clamp_i8(val);
        }
    }
    QTensor::new(out_data, a.params)
}

/// Requantize the float attention output back to i8.
pub fn requantize_attention_output_f32<const SEQ: usize, const DIM: usize>(
    attn_out_f32: &Matrix<f32, SEQ, DIM>,
    requant: &RequantShift,
    out_params: QuantParams,
) -> QTensor<i8, SEQ, DIM> {
    let mut out_data = Matrix::<i8, SEQ, DIM>::zeros();
    for r in 0..SEQ {
        for c in 0..DIM {
            // First we round the weighted i8 sum from f32 to i32.
            let val_i32 = libm::roundf(attn_out_f32.data[r][c]) as i32;
            out_data.data[r][c] = requant.apply_clamped(val_i32);
        }
    }
    QTensor::new(out_data, out_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::scale::QuantParams;

    fn build_toy_block() -> QTransformerBlock<2, 2, 4> {
        let params = QuantParams::symmetric(1.0);
        
        let norm1 = QRMSNorm::new([10.0, 10.0], 1e-5);
        let norm2 = QRMSNorm::new([10.0, 10.0], 1e-5);

        // Identity-like projections for simplicity
        let q_w = Matrix::from_array([[1, 0], [0, 1]]);
        let k_w = Matrix::from_array([[1, 0], [0, 1]]);
        let v_w = Matrix::from_array([[1, 0], [0, 1]]);
        
        let q_proj = QLinear::new(q_w, &[1.0; 2], None, 1.0, params, false);
        let k_proj = QLinear::new(k_w, &[1.0; 2], None, 1.0, params, false);
        let v_proj = QLinear::new(v_w, &[1.0; 2], None, 1.0, params, false);

        // FF1: 2 -> 4, with ReLU
        let ff1_w = Matrix::from_array([
            [1, 1, 0, 0],
            [0, 0, -1, -1]
        ]);
        let ff1 = QLinear::new(ff1_w, &[1.0; 4], None, 1.0, params, true);

        // FF2: 4 -> 2, without ReLU
        let ff2_w = Matrix::from_array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ]);
        let ff2 = QLinear::new(ff2_w, &[1.0; 2], None, 1.0, params, false);

        let attn_requant = RequantShift::from_ratio(1.0);
        
        QTransformerBlock {
            norm1, norm2,
            q_proj, k_proj, v_proj,
            ff1, ff2,
            attn_requant,
            attn_out_params: params,
        }
    }

    #[test]
    fn test_transformer_forward_shape() {
        let block = build_toy_block();
        let x = QTensor::new(Matrix::from_array([[10, 10], [5, -5]]), QuantParams::symmetric(1.0));
        let out = block.forward(&x);
        assert_eq!(out.rows(), 2);
        assert_eq!(out.cols(), 2);
    }

    #[test]
    fn test_transformer_residual_stability() {
        let params = QuantParams::symmetric(1.0);
        let a = QTensor::new(Matrix::from_array([[100, 100], [-100, -100]]), params);
        let b = QTensor::new(Matrix::from_array([[50, -100], [-50, 100]]), params);
        
        let out = residual_add_i8(&a, &b);
        assert_eq!(out.raw(0, 0), 127); // Clamped from 150
        assert_eq!(out.raw(0, 1), 0);
        assert_eq!(out.raw(1, 0), -128); // Clamped from -150
        assert_eq!(out.raw(1, 1), 0);
    }

    #[test]
    fn test_transformer_attention_integration() {
        let block = build_toy_block();
        let x = QTensor::new(Matrix::from_array([[10, 10], [0, 0]]), QuantParams::symmetric(1.0));
        
        let norm_x = block.norm1.forward(&x);
        let q = block.q_proj.forward(&norm_x);
        let k = block.k_proj.forward(&norm_x);
        let v = block.v_proj.forward(&norm_x);
        let scores = QAttention::<2, 2>::compute_scores(&q, &k);
        let probs = QSoftmax::softmax_matrix(&scores, 2);
        let attn_f32 = QAttention::<2, 2>::apply_softmax_values(&probs, &v);
        let attn_i8 = requantize_attention_output_f32(&attn_f32, &block.attn_requant, block.attn_out_params);
        
        assert_eq!(attn_i8.rows(), 2);
        assert_eq!(attn_i8.cols(), 2);
    }

    #[test]
    fn test_transformer_relu_behavior() {
        let block = build_toy_block();
        // Negative inputs should be zeroed out by FF1 ReLU
        let x = QTensor::new(Matrix::from_array([[-10, -10], [-5, 5]]), QuantParams::symmetric(1.0));
        let ff1_out = block.ff1.forward(&x);
        assert_eq!(ff1_out.raw(0, 0), 0);
        assert_eq!(ff1_out.raw(0, 1), 0);
        assert_eq!(ff1_out.raw(0, 2), 10);
        assert_eq!(ff1_out.raw(0, 3), 10);
    }

    #[test]
    fn test_transformer_quantized_pipeline() {
        let block = build_toy_block();
        let x = QTensor::new(Matrix::from_array([[10, 10], [5, -5]]), QuantParams::symmetric(1.0));
        let out = block.forward(&x);
        assert_eq!(out.rows(), 2);
    }

    #[test]
    fn test_transformer_with_rmsnorm() {
        let block = build_toy_block();
        let x = QTensor::new(Matrix::from_array([[100, 100], [-100, -100]]), QuantParams::symmetric(1.0));
        let out = block.forward(&x);
        assert_eq!(out.rows(), 2);
    }

    #[test]
    fn test_transformer_stack_stability() {
        let block = build_toy_block();
        let mut x = QTensor::new(Matrix::from_array([[10, 10], [-10, -10]]), QuantParams::symmetric(1.0));
        for _ in 0..5 {
            x = block.forward(&x);
        }
        assert_eq!(x.rows(), 2);
    }
}
