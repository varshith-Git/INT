// src/nn/softmax.rs
use crate::matrix::Matrix;

pub struct QSoftmax;

impl QSoftmax {
    pub fn softmax_row<const N: usize>(scores: &[i32; N], active_len: usize) -> [f32; N] {
        let mut out = [0.0; N];
        if active_len == 0 {
            return out;
        }

        let limit = active_len.min(N);
        
        // 1. Find max for stabilization
        let mut max_val = scores[0];
        for i in 1..limit {
            if scores[i] > max_val {
                max_val = scores[i];
            }
        }

        // 2. Compute exponentials and sum
        let mut sum_exp = 0.0;
        for i in 0..limit {
            let diff = scores[i].saturating_sub(max_val);
            let exp_val = libm::expf(diff as f32);
            out[i] = exp_val;
            sum_exp += exp_val;
        }

        // 3. Epsilon protection and normalization
        let inv_sum = 1.0 / (sum_exp + 1e-6);
        for i in 0..limit {
            out[i] *= inv_sum;
        }

        out
    }

    pub fn softmax_matrix<const R: usize, const C: usize>(
        scores: &Matrix<i32, R, C>,
        active_len: usize,
    ) -> Matrix<f32, R, C> {
        let mut out = Matrix::<f32, R, C>::zeros();
        for r in 0..R {
            out.data[r] = Self::softmax_row(&scores.data[r], active_len);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_softmax_row_sums_to_one() {
        let scores = [10, 20, 30, 0];
        // Only consider active_len = 3
        let probs = QSoftmax::softmax_row(&scores, 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(probs[3], 0.0);
    }
    
    #[test]
    fn test_softmax_stable_large_values() {
        let scores = [100000, 100000, 100000, 100000];
        let probs = QSoftmax::softmax_row(&scores, 4);
        assert!((probs[0] - 0.25).abs() < 1e-5);
    }
    
    #[test]
    fn test_softmax_negative_values() {
        let scores = [-100, -50, -10];
        let probs = QSoftmax::softmax_row(&scores, 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
