// src/nn/rmsnorm.rs
use crate::matrix::Matrix;
use crate::quant::{QTensor, QuantParams};
use crate::quant::clip::clamp_i8;
use libm::sqrtf;

/// A quantized RMSNorm primitive for transformer stabilization.
pub struct QRMSNorm<const DIM: usize> {
    pub weight: [f32; DIM],
    pub eps: f32,
}

impl<const DIM: usize> QRMSNorm<DIM> {
    pub fn new(weight: [f32; DIM], eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward<const SEQ: usize>(&self, x: &QTensor<i8, SEQ, DIM>) -> QTensor<i8, SEQ, DIM> {
        let mut f32_out = Matrix::<f32, SEQ, DIM>::zeros();
        let mut max_abs: f32 = 0.0;

        for r in 0..SEQ {
            let mean_sq = compute_rms_i32(&x.data.data[r], x.zero_point());
            let rms = sqrtf(mean_sq + self.eps);

            for c in 0..DIM {
                let v = (x.raw(r, c) as i32) - x.zero_point();
                let normalized = (v as f32) / rms;
                let val = normalized * self.weight[c];
                f32_out.data[r][c] = val;
                
                let val_abs = libm::fabsf(val);
                if val_abs > max_abs {
                    max_abs = val_abs;
                }
            }
        }
        
        let optimal_scale = max_abs / 127.0;
        let final_scale = if optimal_scale > 0.0 { optimal_scale } else { x.scale() };
        
        let mut out_data = Matrix::<i8, SEQ, DIM>::zeros();
        for r in 0..SEQ {
            for c in 0..DIM {
                let scaled = f32_out.data[r][c] / final_scale;
                out_data.data[r][c] = crate::quant::clip::clamp_i8(libm::roundf(scaled) as i32);
            }
        }

        QTensor::new(out_data, QuantParams::symmetric(final_scale))
     }

    /// Forward pass taking an un-clamped i32 residual stream directly.
    /// RMS normalization is scale invariant, so we can compute it directly on the i32 values.
    pub fn forward_i32<const BATCH: usize>(&self, x: &Matrix<i32, BATCH, DIM>) -> QTensor<i8, BATCH, DIM> {
        let mut out_data = Matrix::<i8, BATCH, DIM>::zeros();
        let mut f32_out = [[0.0_f32; DIM]; BATCH];
        let mut max_abs = 0.0_f32;

        for r in 0..BATCH {
            let mean_sq = crate::nn::rmsnorm::compute_rms_i32_from_i32(&x.data[r]);
            let rms = libm::sqrtf(mean_sq + self.eps);

            for c in 0..DIM {
                let v = x.data[r][c] as f32;
                let normalized = v / rms;
                let scaled = normalized * self.weight[c];
                f32_out[r][c] = scaled;
                
                let abs_val = libm::fabsf(scaled);
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
            }
        }
        
        let optimal_scale = max_abs / 127.0;
        let final_scale = if optimal_scale > 0.0 { optimal_scale } else { 1.0 };
        let inv_scale = 1.0 / final_scale;

        for r in 0..BATCH {
            for c in 0..DIM {
                out_data.data[r][c] = crate::quant::clip::clamp_i8(libm::roundf(f32_out[r][c] * inv_scale) as i32);
            }
        }
        
        QTensor::new(out_data, QuantParams::symmetric(final_scale))
    }
}

/// Accumulates square of integers into i32 and returns the mean.
pub fn compute_rms_i32<const DIM: usize>(row: &[i8; DIM], zero_point: i32) -> f32 {
    let mut sum_sq: i32 = 0;
    for &val in row.iter() {
        let v = (val as i32) - zero_point;
        sum_sq += v * v;
    }
    (sum_sq as f32) / (row.len() as f32)
}

pub fn compute_rms_i32_from_i32(row: &[i32]) -> f32 {
    let mut sum_sq: f32 = 0.0;
    for &val in row {
        let vf = val as f32;
        sum_sq += vf * vf;
    }
    sum_sq / (row.len() as f32)
}

/// Requantizes the float result back to i8 safely.
pub fn requantize_norm_output(scaled: f32, zero_point: i32) -> i8 {
    let q = libm::roundf(scaled) as i32 + zero_point;
    clamp_i8(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::scale::QuantParams;

    fn build_rmsnorm() -> QRMSNorm<4> {
        QRMSNorm::new([10.0, 10.0, 10.0, 10.0], 1e-5)
    }

    #[test]
    fn test_rmsnorm_shape_preservation() {
        let norm = build_rmsnorm();
        let x = QTensor::new(Matrix::from_array([[1, 2, 3, 4]]), QuantParams::symmetric(1.0));
        let out = norm.forward(&x);
        assert_eq!(out.rows(), 1);
        assert_eq!(out.cols(), 4);
    }

    #[test]
    fn test_rmsnorm_stabilizes_large_values() {
        let norm = build_rmsnorm();
        let x = QTensor::new(Matrix::from_array([[100, 100, 100, 100]]), QuantParams::symmetric(1.0));
        let out = norm.forward(&x);
        // mean_sq = 10000, rms = 100, normalized = 1.0, scaled = 10.0 -> requant = 10
        assert_eq!(out.raw(0, 0), 10);
    }

    #[test]
    fn test_rmsnorm_negative_values() {
        let norm = build_rmsnorm();
        let x = QTensor::new(Matrix::from_array([[-100, -100, -100, -100]]), QuantParams::symmetric(1.0));
        let out = norm.forward(&x);
        assert_eq!(out.raw(0, 0), -10);
    }

    #[test]
    fn test_rmsnorm_quant_pipeline() {
        let norm = build_rmsnorm();
        let params = QuantParams::symmetric(0.5);
        let x = QTensor::new(Matrix::from_array([[10, 20, 30, 40]]), params);
        let out = norm.forward(&x);
        assert_eq!(out.scale(), 0.5);
    }

    #[test]
    fn test_rmsnorm_deterministic_output() {
        let norm = build_rmsnorm();
        let x = QTensor::new(Matrix::from_array([[50, -25, 10, -5]]), QuantParams::symmetric(1.0));
        let out1 = norm.forward(&x);
        let out2 = norm.forward(&x);
        for c in 0..4 {
            assert_eq!(out1.raw(0, c), out2.raw(0, c));
        }
    }
}
