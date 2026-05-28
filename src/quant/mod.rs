//! # Quantization pipeline
//!
//! This module is the inference runtime core.  It composes the other modules
//! into a single coherent pipeline:
//!
//! ```text
//! f32 tensor
//!    │  scale::quantize_f32_to_i8
//!    ▼
//! QTensor<i8, R, K>  ×  QTensor<i8, K, C>
//!    │  kernel::matmul_wide  (i8 × i8 → i32)
//!    ▼
//! Matrix<i32, R, C>           ← accumulator, scale = s_a × s_b
//!    │  clip::apply_relu_i32  (optional fused activation)
//!    │  requant::requantize_matrix  (i32 → i8, integer multiply-shift)
//!    ▼
//! QTensor<i8, R, C>           ← output, scale = s_out
//! ```
//!
//! ## Submodules
//!
//! | Module        | Responsibility                                    |
//! |---------------|---------------------------------------------------|
//! | `scale`       | [`QuantParams`], float↔fixed conversion, calibration |
//! | `qtensor`     | [`QTensor`]: Matrix + QuantParams                 |
//! | `requant`     | [`RequantShift`]: integer multiply-shift requant  |
//! | `clip`        | [`Clamp`] trait, ReLU, ReLU6, saturating helpers  |

pub mod clip;
pub mod qtensor;
pub mod requant;
pub mod scale;

// ── Flat re-exports for ergonomic use from the crate root ────────────────────
pub use clip::{Clamp, apply_relu, apply_relu6, apply_relu_i32, clamp_i8, clamp_i16};
pub use qtensor::QTensor;
pub use requant::{RequantShift, requantize_matrix};
pub use scale::{QuantParams, affine_params_from_range, symmetric_params_from_absmax};

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests — full pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod integration {
    use super::*;
    use crate::kernel::static_wide::matmul_wide;
    use crate::matrix::Matrix;

    /// Full round-trip:
    /// float → quantize → matmul_wide (i8×i8→i32) → requantize → i8 output
    ///
    /// Uses a simple 2×2 case where we can verify the exact float result.
    #[test]
    fn test_full_pipeline_i8_to_i8() {
        // Float weights and activations
        let a_float = [[1.0f32, 2.0], [3.0, 4.0]];
        let b_float = [[0.5f32, 0.0], [0.0, 0.5]];

        // Calibrate: abs_max of A = 4.0, B = 0.5
        let params_a = symmetric_params_from_absmax(4.0);
        let params_b = symmetric_params_from_absmax(0.5);

        // Quantize inputs to i8 QTensors
        let qa: QTensor<i8, 2, 2> = QTensor::quantize_from(a_float, params_a);
        let qb: QTensor<i8, 2, 2> = QTensor::quantize_from(b_float, params_b);

        // matmul_wide: i8 × i8 → i32 accumulator
        let acc: Matrix<i32, 2, 2> = matmul_wide(&qa.data, &qb.data);

        // Expected float result:
        // C[0][0] = 1.0*0.5 + 2.0*0.0 = 0.5
        // C[0][1] = 1.0*0.0 + 2.0*0.5 = 1.0
        // C[1][0] = 3.0*0.5 + 4.0*0.0 = 1.5
        // C[1][1] = 3.0*0.0 + 4.0*0.5 = 2.0
        // → abs_max_out = 2.0
        let params_out = symmetric_params_from_absmax(2.0);

        // Requantize accumulator: scale = s_a * s_b / s_out
        let rs = RequantShift::from_scales(
            params_a.scale,
            params_b.scale,
            params_out.scale,
        );
        let out: Matrix<i8, 2, 2> = requantize_matrix(&acc, &rs);
        let qout = QTensor::new(out, params_out);

        // Check dequantized output is close to float ground truth
        let expected = [[0.5f32, 1.0], [1.5, 2.0]];
        for r in 0..2 {
            for c in 0..2 {
                let got = qout.dequant(r, c);
                let want = expected[r][c];
                assert!(
                    (got - want).abs() < 0.1,
                    "mismatch at [{r},{c}]: got {got:.4}, want {want:.4}"
                );
            }
        }
    }

    /// Pipeline with fused ReLU: negative accumulators are zeroed before
    /// requantization, which is cheaper than a separate pass.
    #[test]
    fn test_pipeline_with_fused_relu() {
        // A has negative values, B is identity-ish
        let a_float = [[-2.0f32, 1.0], [3.0, -1.0]];
        let b_float = [[1.0f32, 0.0], [0.0, 1.0]];

        let params_a = symmetric_params_from_absmax(3.0);
        let params_b = symmetric_params_from_absmax(1.0);

        let qa: QTensor<i8, 2, 2> = QTensor::quantize_from(a_float, params_a);
        let qb: QTensor<i8, 2, 2> = QTensor::quantize_from(b_float, params_b);

        let mut acc: Matrix<i32, 2, 2> = matmul_wide(&qa.data, &qb.data);

        // Fused ReLU in i32 space (symmetric → zero_point = 0)
        apply_relu_i32(&mut acc, 0);

        let params_out = symmetric_params_from_absmax(3.0);
        let rs = RequantShift::from_scales(params_a.scale, params_b.scale, params_out.scale);
        let out: Matrix<i8, 2, 2> = requantize_matrix(&acc, &rs);

        // After ReLU: negatives → 0; positives unchanged
        // C (before relu) = A × I ≈ A = [[-2,1],[3,-1]]
        // After relu:                    [[ 0,1],[3, 0]]
        let qout = QTensor::new(out, params_out);
        assert!(qout.dequant(0, 0).abs() < 0.15,    "relu should zero [-2,0]");
        assert!((qout.dequant(0, 1) - 1.0).abs() < 0.15, "should be ~1.0");
        assert!((qout.dequant(1, 0) - 3.0).abs() < 0.15, "should be ~3.0");
        assert!(qout.dequant(1, 1).abs() < 0.15,    "relu should zero [-1,1]");
    }

    /// Asymmetric (affine) quantization: typical for activations after ReLU,
    /// where the float range is non-negative [0, max] and asymmetric params
    /// use the full i8 range.
    #[test]
    fn test_affine_quantize_dequantize_roundtrip() {
        let params = affine_params_from_range(0.0, 6.0);  // ReLU6 output range
        let original = 3.14f32;
        let q = params.quantize_f32_to_i8(original);
        let back = params.dequantize_i8(q);
        // Round-trip error bounded by one quantization step
        assert!((back - original).abs() < params.scale + 1e-5,
            "round-trip error too large: orig={original}, back={back}");
    }

    /// Scale product rule: confirm that matmul_out_scale is consistent.
    #[test]
    fn test_scale_product_rule() {
        let pa = QuantParams::symmetric(0.05);
        let pb = QuantParams::symmetric(0.02);
        let expected_out_scale = 0.05 * 0.02;
        assert!((pa.matmul_out_scale(&pb) - expected_out_scale).abs() < 1e-9);
    }
}