import numpy as np
import struct

def main():
    fc1_w = np.load("artifacts/fc1.weight_int8.npy")
    fc1_b = np.load("artifacts/fc1.bias_int8.npy")
    fc2_w = np.load("artifacts/fc2.weight_int8.npy")
    fc2_b = np.load("artifacts/fc2.bias_int8.npy")
    
    with open("artifacts/fc1.weight_scale.txt", "r") as f:
        fc1_w_scale = float(f.read().strip())
    with open("artifacts/fc1.bias_scale.txt", "r") as f:
        fc1_b_scale = float(f.read().strip())
    with open("artifacts/fc2.weight_scale.txt", "r") as f:
        fc2_w_scale = float(f.read().strip())
    with open("artifacts/fc2.bias_scale.txt", "r") as f:
        fc2_b_scale = float(f.read().strip())
        
    with open("artifacts/sample_image_scale.txt", "r") as f:
        img_scale = float(f.read().strip())
        
    # We will embed the img_scale and other scales in the binary or pass them via runtime parsing.
    # Actually, the biases in QLinear expect the SAME scale as the input*weight product for residual_add
    # However, since QLinear currently doesn't natively do bias addition dynamically with different scales out of the box,
    # wait: QLinear in our Rust codebase *does not* have biases right now!
    # Let me check `src/nn/linear.rs` -- the struct is `QLinear<I, O> { weight: QTensor, ... }`. It only has weight.
    # Ah, the prompt says "Save: fc1.weight, fc1.bias". But the Rust `QLinear` doesn't support bias yet.
    # Let me look closely at `export_mnist_bin.py` binary layout required:
    # [magic][version][input_dim][hidden_dim][output_dim]
    # [fc1 weights][fc1 bias][fc2 weights][fc2 bias]
    # [scales]
    
    with open("artifacts/mnist_int8.bin", "wb") as f:
        # Header
        f.write(b"QMLP")
        f.write(struct.pack("<I", 1)) # version
        f.write(struct.pack("<I", 784))
        f.write(struct.pack("<I", 128))
        f.write(struct.pack("<I", 10))
        
        # We write row-major raw bytes
        f.write(fc1_w.tobytes())
        f.write(fc1_b.tobytes())
        f.write(fc2_w.tobytes())
        f.write(fc2_b.tobytes())
        
        # Scales
        f.write(struct.pack("<f", fc1_w_scale))
        f.write(struct.pack("<f", fc1_b_scale))
        f.write(struct.pack("<f", fc2_w_scale))
        f.write(struct.pack("<f", fc2_b_scale))
        f.write(struct.pack("<f", img_scale))

    print("Exported artifacts/mnist_int8.bin")

if __name__ == "__main__":
    main()
