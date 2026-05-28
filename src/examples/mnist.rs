// src/examples/mnist.rs
use crate::quant::{QTensor, QuantParams};
use crate::nn::linear::QLinear;
use crate::model::loader::MLPWeights;

pub struct QMLP {
    fc1: QLinear<784, 128>,
    fc2: QLinear<128, 10>,
}

impl QMLP {
    pub fn new(weights: MLPWeights<784, 128, 10>) -> Self {
        let fc1 = QLinear::new(
            weights.fc1_w,
            &[weights.fc1_w_scale; 128],
            Some(weights.fc1_b),
            weights.img_scale,
            weights.fc1_out_params,
            true, // ReLU fused
        );
        
        let fc2 = QLinear::new(
            weights.fc2_w,
            &[weights.fc2_w_scale; 10],
            Some(weights.fc2_b),
            weights.fc1_out_params.scale,
            QuantParams::symmetric(1.0), // out_params scale not heavily strictly needed for argmax, but we keep it symmetric(1.0)
            false,
        );

        Self { fc1, fc2 }
    }

    pub fn forward(&self, x: &QTensor<i8, 1, 784>) -> QTensor<i8, 1, 10> {
        let h = self.fc1.forward(x);
        self.fc2.forward(&h)
    }

    pub fn predict(&self, x: &QTensor<i8, 1, 784>) -> usize {
        let logits = self.forward(x);
        
        let mut max_val = i8::MIN;
        let mut max_idx = 0;
        
        for i in 0..10 {
            let val = logits.raw(0, i);
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        max_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::fs;
    use crate::model::loader::load_mlp_from_bytes;
    use crate::matrix::Matrix;

    #[test]
    fn test_mlp_real_weights() {
        let bin_data = fs::read("artifacts/mnist_int8.bin").expect("Run Python scripts first!");
        let weights = load_mlp_from_bytes::<784, 128, 10>(&bin_data).expect("Failed to load MLP weights");
        let mlp = QMLP::new(weights);
        
        // Assert parameters loaded
        assert_eq!(mlp.fc1.weights.rows(), 784);
        assert_eq!(mlp.fc1.weights.cols(), 128);
        assert_eq!(mlp.fc2.weights.rows(), 128);
        assert_eq!(mlp.fc2.weights.cols(), 10);
    }

    #[test]
    fn test_mnist_parity() {
        let bin_data = fs::read("artifacts/mnist_int8.bin").expect("Run Python scripts first!");
        let weights = load_mlp_from_bytes::<784, 128, 10>(&bin_data).unwrap();
        let img_scale = weights.img_scale;
        
        let mlp = QMLP::new(weights);

        // Load quantized sample image
        let img_data = fs::read("artifacts/sample_image_int8.npy").expect("Run Python scripts first!");
        
        // Numpy format header for 1x1x28x28 int8 is 128 bytes usually (standard format)
        // We will just read the last 784 bytes.
        let raw_pixels = &img_data[img_data.len() - 784..];
        
        let mut x_mat = Matrix::<i8, 1, 784>::zeros();
        for i in 0..784 {
            x_mat.data[0][i] = raw_pixels[i] as i8;
        }
        
        let x = QTensor::new(x_mat, QuantParams::symmetric(img_scale));
        
        let pred = mlp.predict(&x);
        
        let expected_txt = fs::read_to_string("artifacts/expected_pred.txt").expect("Run Python scripts first!");
        let expected_pred: usize = expected_txt.trim().parse().unwrap();
        
        assert_eq!(pred, expected_pred, "Rust INT8 runtime prediction diverges from PyTorch!");
        
        // Load expected float logits from python to check scale
        let _expected_logits_data = fs::read("artifacts/expected_logits.npy").unwrap();
        // Just print output quantization bounds
        let logits = mlp.forward(&x);
        std::println!("Rust Quantized Logits (i8): {:?}", logits.data.data[0]);
    }
}
