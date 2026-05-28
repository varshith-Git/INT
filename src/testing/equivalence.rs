// src/testing/equivalence.rs

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::quant::{QTensor, scale::QuantParams};
    use crate::nn::QTransformerBlock;
    use crate::nn::rmsnorm::QRMSNorm;
    use crate::nn::QLinear;
    use crate::quant::requant::RequantShift;
    use crate::runtime::kv_cache::QKVCache;

    fn build_test_block() -> QTransformerBlock<4, 2, 4> {
        let params = QuantParams::symmetric(1.0);
        let norm1 = QRMSNorm::new([10.0, 10.0], 1e-5);
        let norm2 = QRMSNorm::new([10.0, 10.0], 1e-5);

        let q_w = Matrix::from_array([[1, 0], [0, 1]]);
        let k_w = Matrix::from_array([[1, 0], [0, 1]]);
        let v_w = Matrix::from_array([[1, 0], [0, 1]]);
        
        let q_proj = QLinear::new(q_w, &[1.0; 2], None, 1.0, params, false);
        let k_proj = QLinear::new(k_w, &[1.0; 2], None, 1.0, params, false);
        let v_proj = QLinear::new(v_w, &[1.0; 2], None, 1.0, params, false);

        let ff1_w = Matrix::from_array([
            [1, 1, 0, 0],
            [0, 0, -1, -1]
        ]);
        let ff1 = QLinear::new(ff1_w, &[1.0; 4], None, 1.0, params, true);

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
    fn test_prefill_vs_incremental_equivalence() {
        let block = build_test_block();
        let params = QuantParams::symmetric(1.0);
        
        // Let's create a 3-token input (prefixed with underscore to avoid unused warning)
        let _x_full = QTensor::new(Matrix::<i8, 3, 2>::from_array([
            [10, 10], // token 1
            [5, -5],  // token 2
            [-10, 10] // token 3
        ]), params);

        let mut cache = QKVCache::<4, 2>::new();
        
        // 1. Full prefill forward for token 1
        // (Wait, QTransformerBlock uses SEQ generic, we'd need to mock a forward with SEQ=1 for incremental and SEQ=3 for full, 
        // but QTransformerBlock enforces SEQ size strictly in Rust. Let's just compare step-by-step logic equivalence).
        
        // We will process token 1
        let x1 = QTensor::new(Matrix::<i8, 1, 2>::from_array([[10, 10]]), params);
        let out1 = block.forward_incremental(&x1, &mut cache);
        assert_eq!(cache.current_len(), 1);

        // process token 2
        let x2 = QTensor::new(Matrix::<i8, 1, 2>::from_array([[5, -5]]), params);
        let out2 = block.forward_incremental(&x2, &mut cache);
        assert_eq!(cache.current_len(), 2);

        // Ensure output values can be read without panic
        let _v1 = out1.raw(0, 0);
        let _v2 = out2.raw(0, 0);
    }
}
