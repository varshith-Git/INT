// src/nn/linear.rs
use crate::matrix::Matrix;
use crate::quant::{QTensor, QuantParams, RequantShift, clip::apply_relu_i32};
use crate::kernel::static_wide::matmul_wide;

/// A fully connected (linear) layer for quantized inference.
///
/// Implements the standard quantized pipeline:
/// INT8 input → wide matmul → bias add → optional ReLU (i32) → requantize → INT8 output.
pub struct QLinear<const IN_DIM: usize, const OUT_DIM: usize> {
    pub weights: Matrix<i8, IN_DIM, OUT_DIM>,
    pub bias: Option<Matrix<i32, 1, OUT_DIM>>,
    pub weight_scales: [f32; OUT_DIM],
    pub requant: [RequantShift; OUT_DIM],
    pub out_params: QuantParams,
    pub relu: bool,
    pub input_scale: f32,
}

impl<const IN_DIM: usize, const OUT_DIM: usize> QLinear<IN_DIM, OUT_DIM> {
    /// Creates a new quantized linear layer and precomputes the requantization shift.
    pub fn new(
        weights: Matrix<i8, IN_DIM, OUT_DIM>,
        weight_scales: &[f32; OUT_DIM],
        bias: Option<Matrix<i32, 1, OUT_DIM>>,
        input_scale: f32,
        out_params: QuantParams,
        relu: bool,
    ) -> Self {
        let requant = core::array::from_fn(|i| {
            RequantShift::from_scales(input_scale, weight_scales[i], out_params.scale)
        });
        
        let mut weight_scales_arr = [0.0; OUT_DIM];
        for i in 0..OUT_DIM {
            weight_scales_arr[i] = weight_scales[i];
        }

        Self {
            weights,
            bias,
            weight_scales: weight_scales_arr,
            requant,
            out_params,
            relu,
            input_scale,
        }
    }

    /// Forward pass through the linear layer.
    ///
    /// Batch size `BATCH` is resolved at compile time for the input tensor.
    pub fn forward<const BATCH: usize>(
        &self,
        x: &QTensor<i8, BATCH, IN_DIM>,
    ) -> QTensor<i8, BATCH, OUT_DIM> {
        // 1. Wide matmul (i8 x i8 -> i32 accumulator)
        let mut acc = matmul_wide(&x.data, &self.weights);

        // 2. Add bias
        if let Some(bias) = &self.bias {
            for r in 0..BATCH {
                for c in 0..OUT_DIM {
                    acc.data[r][c] += bias.data[0][c];
                }
            }
        }

        // 3. Optional fused ReLU
        // Applied in i32 space before requantization. 
        // Assuming symmetric quantization (zero_point = 0) for the intermediate accumulation.
        if self.relu {
            apply_relu_i32(&mut acc, 0);
        }

        // 4. Requantize back to i8
        let out_data = crate::quant::requant::requantize_matrix_per_channel(&acc, &self.requant);
        
        QTensor::new(out_data, self.out_params)
    }

    /// Forward pass returning raw i32 accumulations before requantization.
    pub fn forward_i32<const BATCH: usize>(
        &self,
        x: &QTensor<i8, BATCH, IN_DIM>,
    ) -> Matrix<i32, BATCH, OUT_DIM> {
        let mut acc = matmul_wide(&x.data, &self.weights);

        if let Some(bias) = &self.bias {
            for r in 0..BATCH {
                for c in 0..OUT_DIM {
                    acc.data[r][c] += bias.data[0][c];
                }
            }
        }

        if self.relu {
            apply_relu_i32(&mut acc, 0);
        }

        acc
    }

    /// Forward pass that handles inputs with dynamically shifting scales.
    pub fn forward_dynamic<const BATCH: usize>(
        &self,
        x: &QTensor<i8, BATCH, IN_DIM>,
    ) -> QTensor<i8, BATCH, OUT_DIM> {
        let acc = self.forward_i32(x);
        
        let mut out_data = Matrix::<i8, BATCH, OUT_DIM>::zeros();
        for c in 0..OUT_DIM {
            let scale_f32 = (x.scale() * self.weight_scales[c]) / self.out_params.scale;
            for r in 0..BATCH {
                let val = acc.data[r][c] as f32;
                let scaled = val * scale_f32;
                out_data.data[r][c] = crate::quant::clip::clamp_i8(libm::roundf(scaled) as i32);
            }
        }
        
        QTensor::new(out_data, self.out_params)
    }

    /// Forward pass through the linear layer.

    /// Forward pass returning fully dequantized f32 accumulations.
    pub fn forward_f32<const BATCH: usize>(
        &self,
        x: &QTensor<i8, BATCH, IN_DIM>,
    ) -> Matrix<f32, BATCH, OUT_DIM> {
        let acc = self.forward_i32(x);
        let mut out = Matrix::<f32, BATCH, OUT_DIM>::zeros();
        for r in 0..BATCH {
            for c in 0..OUT_DIM {
                out.data[r][c] = (acc.data[r][c] as f32) * x.scale() * self.weight_scales[c];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_qlinear_forward() {
        // Simple 1x2 * 2x3 -> 1x3 linear layer
        let _w_params = QuantParams::symmetric(1.0);
        let w_data = Matrix::from_array([
            [10i8, 0, -10],
            [0, 10, -10]
        ]);
        let weight_scales = [1.0, 1.0, 1.0];

        let bias = Matrix::from_array([[5i32, -5, 0]]);

        let x_params = QuantParams::symmetric(1.0);
        let x_data = Matrix::from_array([[5i8, 5]]);
        let x = QTensor::new(x_data, x_params);

        let out_params = QuantParams::symmetric(1.0);

        let linear = QLinear::new(w_data, &weight_scales, Some(bias), x.scale(), out_params, true);
        let out = linear.forward(&x);

        assert_eq!(out.rows(), 1);
        assert_eq!(out.cols(), 3);
        
        // Verify ReLU zeroed the last negative accumulator (-100)
        assert_eq!(out.raw(0, 2), 0);
        assert!(out.raw(0, 0) > 0);
        assert!(out.raw(0, 1) > 0);
    }
}
