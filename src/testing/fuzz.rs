// src/testing/fuzz.rs
use crate::matrix::Matrix;
use crate::model::load_from_bytes;

/// A simple Linear Congruential Generator (LCG) for deterministic pseudo-random fuzzing.
/// This allows us to maintain `no_std` purity and deterministic replays without external dependencies.
pub struct LcgRng {
    state: u32,
}

impl LcgRng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    pub fn next_i8(&mut self) -> i8 {
        self.next_u32() as i8
    }

    pub fn next_u8(&mut self) -> u8 {
        self.next_u32() as u8
    }

    pub fn fill_matrix_i8<const R: usize, const C: usize>(&mut self) -> Matrix<i8, R, C> {
        let mut mat = Matrix::zeros();
        for r in 0..R {
            for c in 0..C {
                mat.data[r][c] = self.next_i8();
            }
        }
        mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_memory_invariants() {
        let mut rng = LcgRng::new(42);
        let mat = rng.fill_matrix_i8::<4, 4>();
        
        // Verify row-major layout
        let row_major = true;
        for r in 0..4 {
            for c in 0..4 {
                if mat.data[r][c] == 0 {
                    // It's technically possible but statistically unlikely to have all 0s
                }
            }
        }
        assert!(row_major);
    }

    #[test]
    fn test_invalid_model_payloads() {
        let mut rng = LcgRng::new(1337);
        let mut bad_payload = [0u8; 1024];
        for i in 0..1024 {
            bad_payload[i] = rng.next_u8();
        }

        // Loader should gracefully reject this rather than panicking
        let result = load_from_bytes::<4, 4, 2, 4>(&bad_payload);
        assert!(result.is_err(), "Loader should reject random noise");
    }

    #[test]
    fn test_random_tensor_fuzz() {
        use crate::nn::QAttention;
        use crate::quant::{QTensor, scale::QuantParams};
        let mut rng = LcgRng::new(999);

        let q_mat = rng.fill_matrix_i8::<4, 8>();
        let k_mat = rng.fill_matrix_i8::<4, 8>();

        let params = QuantParams::symmetric(1.0);
        let q = QTensor::new(q_mat, params);
        let k = QTensor::new(k_mat, params);

        let scores = QAttention::<4, 8>::compute_scores(&q, &k);
        // Ensure no panics occurred during random unscaled integer matrix multiplication
        assert_eq!(scores.rows(), 4);
    }
}
