// src/testing/overflow.rs

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::quant::{QTensor, scale::QuantParams};
    use crate::nn::QSoftmax;
    use crate::nn::transformer::{residual_add_i8, requantize_attention_output_f32};
    use crate::quant::requant::RequantShift;

    #[test]
    fn test_softmax_extreme_logits() {
        // Feed extremely large INT32 values into softmax
        // Ensures max subtraction safely prevents NaN / Infinity.
        let extreme_scores = [i32::MAX, i32::MAX - 100, -i32::MAX, 0];
        let probs = QSoftmax::softmax_row(&extreme_scores, 4);
        
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[0] > 0.99); // Largest logit should absolutely dominate
    }

    #[test]
    fn test_attention_entropy_bounds() {
        // Test uniform entropy vs collapsed entropy
        let uniform = [10, 10, 10, 10];
        let uniform_probs = QSoftmax::softmax_row(&uniform, 4);
        for &p in &uniform_probs {
            assert!((p - 0.25).abs() < 1e-5);
        }

        let peaked = [100000, 0, 0, 0];
        let peaked_probs = QSoftmax::softmax_row(&peaked, 4);
        assert!(peaked_probs[0] > 0.999);
        assert!(peaked_probs[1] < 0.001);
    }

    #[test]
    fn test_requantization_idempotence() {
        // Verify multiple i8 -> f32 -> i8 cycles don't diverge catastrophically
        let params = QuantParams::symmetric(1.0);
        let requant = RequantShift::from_ratio(1.0);
        
        // Mock a stable f32 array
        let mut f32_mat = Matrix::<f32, 2, 2>::from_array([
            [120.0, -120.0],
            [50.0, -50.0]
        ]);
        
        // Requantize cycle 1
        let i8_tensor_1 = requantize_attention_output_f32(&f32_mat, &requant, params);
        
        // We simulate feeding i8 back to f32 (e.g. attention value aggregation layer)
        for r in 0..2 {
            for c in 0..2 {
                f32_mat.data[r][c] = i8_tensor_1.raw(r, c) as f32;
            }
        }
        
        // Requantize cycle 2
        let i8_tensor_2 = requantize_attention_output_f32(&f32_mat, &requant, params);
        
        // Should be absolutely idempotent after the first rounding hop
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(i8_tensor_1.raw(r, c), i8_tensor_2.raw(r, c));
            }
        }
    }

    #[test]
    fn test_residual_accumulator_bounds() {
        let params = QuantParams::symmetric(1.0);
        
        let a = QTensor::new(Matrix::<i8, 2, 2>::from_array([
            [127, 127],
            [-128, -128]
        ]), params);
        
        let b = QTensor::new(Matrix::<i8, 2, 2>::from_array([
            [127, 10],
            [-128, -10]
        ]), params);

        let out = residual_add_i8(&a, &b);
        
        // Should seamlessly clamp at exact boundaries without panicking or rolling over
        assert_eq!(out.raw(0, 0), 127);
        assert_eq!(out.raw(0, 1), 127);
        assert_eq!(out.raw(1, 0), -128);
        assert_eq!(out.raw(1, 1), -128);
    }
}
