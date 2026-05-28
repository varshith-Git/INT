// src/nn/gpt.rs
use crate::matrix::Matrix;
use crate::quant::{QTensor, QuantParams, RequantShift};
use crate::nn::linear::QLinear;
use crate::nn::embedding::QEmbedding;
use crate::nn::rmsnorm::QRMSNorm;
use crate::runtime::kv_cache::QKVCache;
use crate::nn::transformer::requantize_attention_output_f32;

pub struct QMultiHeadAttention<const SEQ: usize, const DIM: usize, const HEADS: usize> {}

impl<const SEQ: usize, const DIM: usize, const HEADS: usize> QMultiHeadAttention<SEQ, DIM, HEADS> {
    pub const HEAD_DIM: usize = DIM / HEADS;

    /// Computes multi-head attention scores, applies causal masking and softmax,
    /// and accumulates values in float.
    pub fn forward(
        q: &QTensor<i8, SEQ, DIM>,
        k: &QTensor<i8, SEQ, DIM>,
        v: &QTensor<i8, SEQ, DIM>,
        scale: f32,
        causal: bool,
    ) -> Matrix<f32, SEQ, DIM> {
        let mut out = Matrix::<f32, SEQ, DIM>::zeros();
        
        for h in 0..HEADS {
            let h_offset = h * Self::HEAD_DIM;
            
            // 1. Compute scores: Q_h * K_h^T in float (correct regardless of calibration)
            let q_scale = q.scale();
            let k_scale = k.scale();
            let head_dim_scale = 1.0 / libm::sqrtf(Self::HEAD_DIM as f32);
            let mut scores_f32 = Matrix::<f32, SEQ, SEQ>::zeros();
            for i in 0..SEQ {
                for j in 0..SEQ {
                    let mut sum: f32 = 0.0;
                    for d in 0..Self::HEAD_DIM {
                        let q_val = (q.raw(i, h_offset + d) as f32) * q_scale;
                        let k_val = (k.raw(j, h_offset + d) as f32) * k_scale;
                        sum += q_val * k_val;
                    }
                    scores_f32.data[i][j] = sum * head_dim_scale;
                }
            }
            
            // 2. Softmax with causal masking
            let mut probs = Matrix::<f32, SEQ, SEQ>::zeros();
            for i in 0..SEQ {
                let active_len = if causal { i + 1 } else { SEQ };
                probs.data[i] = softmax_row_f32(&scores_f32.data[i], active_len);
            }
            
            // 3. Apply values: probs * V_h
            for i in 0..SEQ {
                for d in 0..Self::HEAD_DIM {
                    let mut sum: f32 = 0.0;
                    for j in 0..SEQ {
                        sum += probs.data[i][j] * (v.raw(j, h_offset + d) as f32);
                    }
                    out.data[i][h_offset + d] = sum;
                }
            }
        }
        out
    }

    /// Incremental multi-head attention using KV cache.
    pub fn forward_incremental(
        q: &QTensor<i8, 1, DIM>,
        cache: &QKVCache<SEQ, DIM>,
        scale: f32,
    ) -> Matrix<f32, 1, DIM> {
        let mut out = Matrix::<f32, 1, DIM>::zeros();
        let len = cache.current_len();
        
        for h in 0..HEADS {
            let h_offset = h * Self::HEAD_DIM;
            
            // 1. Compute scores: Q_h * K_h^T in float (correct regardless of calibration)
            let q_scale = q.scale();
            let k_scale = cache.key_scale;
            let head_dim_scale = 1.0 / libm::sqrtf(Self::HEAD_DIM as f32);
            let mut scores_f32 = [0.0; SEQ];
            for t in 0..len {
                let mut sum: f32 = 0.0;
                for d in 0..Self::HEAD_DIM {
                    let q_val = (q.raw(0, h_offset + d) as f32) * q_scale;
                    let k_val = (cache.keys.data[t][h_offset + d] as f32) * k_scale;
                    sum += q_val * k_val;
                }
                scores_f32[t] = sum * head_dim_scale;
            }
            
            // 2. Softmax
            let probs = softmax_row_f32(&scores_f32, len);
            
            // 3. Apply values: probs * V_h (raw i8, scale handled by requantize_attention_output_f32)
            for d in 0..Self::HEAD_DIM {
                let mut sum: f32 = 0.0;
                for t in 0..len {
                    sum += probs[t] * (cache.values.data[t][h_offset + d] as f32);
                }
                out.data[0][h_offset + d] = sum;
            }
        }
        out
    }
}

pub fn softmax_row_f32<const N: usize>(scores: &[f32; N], active_len: usize) -> [f32; N] {
    let mut out = [0.0; N];
    if active_len == 0 {
        return out;
    }
    let limit = active_len.min(N);
    let mut max_val = scores[0];
    for i in 1..limit {
        if scores[i] > max_val {
            max_val = scores[i];
        }
    }
    let mut sum_exp = 0.0;
    for i in 0..limit {
        let exp_val = libm::expf(scores[i] - max_val);
        out[i] = exp_val;
        sum_exp += exp_val;
    }
    let inv_sum = 1.0 / (sum_exp + 1e-6);
    for i in 0..limit {
        out[i] *= inv_sum;
    }
    out
}

/// Compute head/token attention entropy: H = -sum p * ln(p + 1e-9)
pub fn compute_entropy<const N: usize>(probs: &[f32; N], active_len: usize) -> f32 {
    let mut h = 0.0;
    let limit = active_len.min(N);
    for i in 0..limit {
        let p = probs[i];
        if p > 0.0 {
            h -= p * libm::logf(p + 1e-9);
        }
    }
    h
}

pub struct QGPTBlock<const SEQ: usize, const DIM: usize, const HIDDEN: usize, const HEADS: usize> {
    pub norm1: QRMSNorm<DIM>,
    pub norm2: QRMSNorm<DIM>,
    
    pub q_proj: QLinear<DIM, DIM>,
    pub k_proj: QLinear<DIM, DIM>,
    pub v_proj: QLinear<DIM, DIM>,
    pub out_proj: QLinear<DIM, DIM>,

    pub ff1: QLinear<DIM, HIDDEN>,
    pub ff2: QLinear<HIDDEN, DIM>,

    pub attn_scale: f32,
    pub attn_requant: RequantShift,
    pub attn_out_params: QuantParams,
    pub s_res_in: QuantParams,
    pub s_res_out: QuantParams,
}

impl<const SEQ: usize, const DIM: usize, const HIDDEN: usize, const HEADS: usize> QGPTBlock<SEQ, DIM, HIDDEN, HEADS> {
    pub fn forward(
        &self,
        x_i32: &mut Matrix<i32, SEQ, DIM>,
    ) -> (Matrix<f32, SEQ, SEQ>, Matrix<f32, SEQ, SEQ>) {
        use crate::quant::clip::scale_matrix_i32_to_i8;
        use crate::quant::requant::requantize_matrix_i32_per_channel;

        let (x_i8_data, dyn_scale) = scale_matrix_i32_to_i8(x_i32);
        let x_i8 = QTensor::new(x_i8_data, QuantParams::symmetric(self.s_res_in.scale * dyn_scale));

        let norm1_out = self.norm1.forward(&x_i8);

        let q = self.q_proj.forward_dynamic(&norm1_out);
        let k = self.k_proj.forward_dynamic(&norm1_out);
        let v = self.v_proj.forward_dynamic(&norm1_out);

        // Debug extraction for head 0
        let head_dim = DIM / HEADS;
        let mut debug_scores = Matrix::<f32, SEQ, SEQ>::zeros();
        let mut debug_probs = Matrix::<f32, SEQ, SEQ>::zeros();
        for i in 0..SEQ {
            let mut row_scores = [0.0; SEQ];
            for j in 0..SEQ {
                let mut sum = 0;
                for d in 0..head_dim {
                    sum += (q.raw(i, d) as i32) * (k.raw(j, d) as i32);
                }
                row_scores[j] = (sum as f32) * self.attn_scale;
                debug_scores.data[i][j] = row_scores[j];
            }
            
            let active_len = i + 1;
            let probs = softmax_row_f32(&row_scores, active_len);
            for j in 0..SEQ {
                debug_probs.data[i][j] = probs[j];
            }
        }

        let attn_out_f32 = QMultiHeadAttention::<SEQ, DIM, HEADS>::forward(&q, &k, &v, self.attn_scale, true);
        let attn_out_i8 = requantize_attention_output_f32(&attn_out_f32, &self.attn_requant, self.attn_out_params);
        
        let attn_proj_out_acc = self.out_proj.forward_i32(&attn_out_i8);
        let attn_proj_out_i32 = requantize_matrix_i32_per_channel(&attn_proj_out_acc, &self.out_proj.requant);

        // Add to residual stream
        for i in 0..SEQ {
            for d in 0..DIM {
                x_i32.data[i][d] = x_i32.data[i][d].saturating_add(attn_proj_out_i32.data[i][d]);
            }
        }

        let (x_i8_data2, dyn_scale2) = scale_matrix_i32_to_i8(x_i32);
        let x_i8_2 = QTensor::new(x_i8_data2, QuantParams::symmetric(self.s_res_in.scale * dyn_scale2));
        let norm2_out = self.norm2.forward(&x_i8_2);
        let mut ff1_out = self.ff1.forward_dynamic(&norm2_out);
        
        // ReLU
        for i in 0..SEQ {
            for d in 0..HIDDEN {
                if ff1_out.data.data[i][d] < 0 {
                    ff1_out.data.data[i][d] = 0;
                }
            }
        }
        
        let ff2_out_acc = self.ff2.forward_i32(&ff1_out);
        let ff2_out_i32 = requantize_matrix_i32_per_channel(&ff2_out_acc, &self.ff2.requant);

        // Add to residual stream
        for i in 0..SEQ {
            for d in 0..DIM {
                x_i32.data[i][d] = x_i32.data[i][d].saturating_add(ff2_out_i32.data[i][d]);
            }
        }

        (debug_scores, debug_probs)
    }

    pub fn forward_incremental(
        &self,
        x_i32: &mut Matrix<i32, 1, DIM>,
        cache: &mut QKVCache<SEQ, DIM>,
    ) {
        use crate::quant::clip::scale_matrix_i32_to_i8;
        use crate::quant::requant::requantize_matrix_i32_per_channel;

        let (x_i8_data, dyn_scale) = scale_matrix_i32_to_i8(x_i32);
        let x_i8 = QTensor::new(x_i8_data, QuantParams::symmetric(self.s_res_in.scale * dyn_scale));

        let norm1_out = self.norm1.forward(&x_i8);

        let q = self.q_proj.forward_dynamic(&norm1_out);
        let k = self.k_proj.forward_dynamic(&norm1_out);
        let v = self.v_proj.forward_dynamic(&norm1_out);

        let mut k_arr = [0; DIM];
        let mut v_arr = [0; DIM];
        for d in 0..DIM {
            k_arr[d] = k.raw(0, d);
            v_arr[d] = v.raw(0, d);
        }
        cache.append(&k_arr, &v_arr, k.scale(), v.scale()).expect("KV cache overflow");

        let attn_out_f32 = QMultiHeadAttention::<SEQ, DIM, HEADS>::forward_incremental(&q, cache, self.attn_scale);
        let attn_out_i8 = requantize_attention_output_f32(&attn_out_f32, &self.attn_requant, self.attn_out_params);
        
        let attn_proj_out_acc = self.out_proj.forward_i32(&attn_out_i8);
        let attn_proj_out_i32 = requantize_matrix_i32_per_channel(&attn_proj_out_acc, &self.out_proj.requant);

        // Add to residual stream
        for d in 0..DIM {
            x_i32.data[0][d] = x_i32.data[0][d].saturating_add(attn_proj_out_i32.data[0][d]);
        }

        let (x_i8_data2, dyn_scale2) = scale_matrix_i32_to_i8(x_i32);
        let x_i8_2 = QTensor::new(x_i8_data2, QuantParams::symmetric(self.s_res_in.scale * dyn_scale2));
        let norm2_out = self.norm2.forward(&x_i8_2);
        let mut ff1_out = self.ff1.forward_dynamic(&norm2_out);
        
        // ReLU
        for d in 0..HIDDEN {
            if ff1_out.data.data[0][d] < 0 {
                ff1_out.data.data[0][d] = 0;
            }
        }
        
        let ff2_out_acc = self.ff2.forward_i32(&ff1_out);
        let ff2_out_i32 = requantize_matrix_i32_per_channel(&ff2_out_acc, &self.ff2.requant);

        for d in 0..DIM {
            x_i32.data[0][d] = x_i32.data[0][d].saturating_add(ff2_out_i32.data[0][d]);
        }
    }
}

pub struct QGPTModel<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
    const HEADS: usize,
    const LAYERS: usize,
> {
    pub token_embeddings: QEmbedding<VOCAB, DIM>,
    pub pos_embeddings: QTensor<i8, SEQ, DIM>,
    pub blocks: [QGPTBlock<SEQ, DIM, HIDDEN, HEADS>; LAYERS],
    pub final_norm: QRMSNorm<DIM>,
    pub lm_head: QLinear<DIM, VOCAB>,
}

impl<
    const VOCAB: usize,
    const SEQ: usize,
    const DIM: usize,
    const HIDDEN: usize,
    const HEADS: usize,
    const LAYERS: usize,
> QGPTModel<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS> {
    pub fn forward(
        &self,
        tokens: &[usize; SEQ],
    ) -> (
        Matrix<f32, SEQ, VOCAB>,
        Matrix<f32, SEQ, SEQ>,
        Matrix<f32, SEQ, SEQ>,
        Matrix<i8, SEQ, DIM>,
        Matrix<i8, SEQ, DIM>,
    ) {
        let mut x_mat = Matrix::<i32, SEQ, DIM>::zeros();
        for i in 0..SEQ {
            let tok = tokens[i];
            let tok_scale = self.token_embeddings.weights.scale();
            let pos_scale = self.pos_embeddings.scale();
            let out_scale = self.blocks[0].s_res_in.scale;
            
            for d in 0..DIM {
                let tok_val = (self.token_embeddings.weights.raw(tok, d) as f32) * tok_scale;
                let pos_val = (self.pos_embeddings.raw(i, d) as f32) * pos_scale;
                let sum_val = tok_val + pos_val;
                x_mat.data[i][d] = libm::roundf(sum_val / out_scale) as i32;
            }
        }
        
        let (l0_scores, _) = self.blocks[0].forward(&mut x_mat);
        let (l0_hidden, _) = crate::quant::clip::scale_matrix_i32_to_i8(&x_mat);
        
        let mut last_scores = crate::matrix::Matrix::<f32, SEQ, SEQ>::zeros();
        for i in 1..LAYERS {
            let (scores, _) = self.blocks[i].forward(&mut x_mat);
            if i == LAYERS - 1 {
                last_scores = scores;
            }
        }
        let (l1_hidden, _) = crate::quant::clip::scale_matrix_i32_to_i8(&x_mat);
        
        let (final_i8, dyn_scale) = crate::quant::clip::scale_matrix_i32_to_i8(&x_mat);
        let final_norm_tensor = QTensor::new(
            final_i8,
            QuantParams::symmetric(self.blocks[LAYERS - 1].s_res_out.scale * dyn_scale)
        );
        let final_norm_out = self.final_norm.forward(&final_norm_tensor);
        let logits = self.lm_head.forward_f32(&final_norm_out);
        
        (logits, l0_scores, last_scores, l0_hidden, l1_hidden)
    }

    pub fn forward_incremental(
        &self,
        token: usize,
        pos: usize,
        caches: &mut [QKVCache<SEQ, DIM>; LAYERS],
    ) -> Matrix<f32, 1, VOCAB> {
        let mut x_mat = Matrix::<i32, 1, DIM>::zeros();
        let tok_scale = self.token_embeddings.weights.scale();
        let pos_scale = self.pos_embeddings.scale();
        let out_scale = self.blocks[0].s_res_in.scale;
        
        for d in 0..DIM {
            let tok_val = (self.token_embeddings.weights.raw(token, d) as f32) * tok_scale;
            let pos_val = (self.pos_embeddings.raw(pos, d) as f32) * pos_scale;
            let sum_val = tok_val + pos_val;
            x_mat.data[0][d] = libm::roundf(sum_val / out_scale) as i32;
        }
        
        for i in 0..LAYERS {
            self.blocks[i].forward_incremental(&mut x_mat, &mut caches[i]);
        }
        
        let (final_i8, dyn_scale) = crate::quant::clip::scale_matrix_i32_to_i8(&x_mat);
        let final_norm_tensor = QTensor::new(
            final_i8,
            QuantParams::symmetric(self.blocks[LAYERS - 1].s_res_out.scale * dyn_scale)
        );
        let final_norm_out = self.final_norm.forward(&final_norm_tensor);
        self.lm_head.forward_f32(&final_norm_out)
    }
}
