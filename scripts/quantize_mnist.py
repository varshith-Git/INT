import numpy as np
import os

def quantize_tensor(tensor_name, x):
    absmax = np.max(np.abs(x))
    scale = absmax / 127.0
    if scale == 0:
        scale = 1.0
    
    q_x = np.round(x / scale)
    q_x = np.clip(q_x, -127, 127).astype(np.int8)
    
    err = np.mean(np.abs((q_x.astype(np.float32) * scale) - x))
    sat = np.sum((q_x == -127) | (q_x == 127)) / q_x.size * 100.0
    
    print(f"--- {tensor_name} ---")
    print(f"Shape: {x.shape}, Float range: [{np.min(x):.4f}, {np.max(x):.4f}]")
    print(f"Scale: {scale:.6f}, Error: {err:.6f}, Saturation: {sat:.2f}%")
    
    return q_x, scale

def main():
    tensors = [
        ("fc1.weight", "artifacts/fc1.weight.npy"),
        ("fc1.bias", "artifacts/fc1.bias.npy"),
        ("fc2.weight", "artifacts/fc2.weight.npy"),
        ("fc2.bias", "artifacts/fc2.bias.npy"),
        ("sample_image", "artifacts/sample_image.npy")
    ]
    
    for name, path in tensors:
        x = np.load(path)
        q_x, scale = quantize_tensor(name, x)
        
        np.save(f"artifacts/{name}_int8.npy", q_x)
        with open(f"artifacts/{name}_scale.txt", "w") as f:
            f.write(str(scale))

if __name__ == "__main__":
    main()
