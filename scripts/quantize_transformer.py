import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from train_tiny_transformer import TinyGPT

def compute_quantization_error(float_tensor, scale):
    if scale == 0:
        return 0.0, 0.0
    q = torch.clamp(torch.round(float_tensor / scale), -128, 127)
    dequant = q * scale
    error = torch.mean((float_tensor - dequant) ** 2).item()
    saturation = (torch.sum(torch.abs(q) == 128) + torch.sum(torch.abs(q) == 127)).item() / float_tensor.numel() * 100
    return error, saturation

def main():
    encoded = np.load("artifacts/encoded_corpus.npy")
    vocab_size = int(np.max(encoded)) + 1
    
    dim = 64
    heads = 4
    layers = 3
    hidden = 256
    seq_len = 64
    
    # Re-instantiate float model and load weights
    model = TinyGPT(vocab_size, seq_len, dim, heads, hidden, layers)
    state = model.state_dict()
    
    state["token_emb.weight"] = torch.tensor(np.load("artifacts/token_embeddings.npy"))
    state["pos_emb.weight"] = torch.tensor(np.load("artifacts/pos_embeddings.npy"))
    for l in range(layers):
        state[f"blocks.{l}.norm1.weight"] = torch.tensor(np.load(f"artifacts/l{l}_norm1.npy"))
        state[f"blocks.{l}.attn.q_proj.weight"] = torch.tensor(np.load(f"artifacts/l{l}_q_proj.npy"))
        state[f"blocks.{l}.attn.k_proj.weight"] = torch.tensor(np.load(f"artifacts/l{l}_k_proj.npy"))
        state[f"blocks.{l}.attn.v_proj.weight"] = torch.tensor(np.load(f"artifacts/l{l}_v_proj.npy"))
        state[f"blocks.{l}.attn.out_proj.weight"] = torch.tensor(np.load(f"artifacts/l{l}_out_proj.npy"))
        state[f"blocks.{l}.norm2.weight"] = torch.tensor(np.load(f"artifacts/l{l}_norm2.npy"))
        state[f"blocks.{l}.ff1.weight"] = torch.tensor(np.load(f"artifacts/l{l}_ff1.npy"))
        state[f"blocks.{l}.ff2.weight"] = torch.tensor(np.load(f"artifacts/l{l}_ff2.npy"))
    state["final_norm.weight"] = torch.tensor(np.load("artifacts/final_norm.npy"))
    state["lm_head.weight"] = torch.tensor(np.load("artifacts/lm_head.npy"))
    
    model.load_state_dict(state)
    model.eval()
    
    # 1. Calibration for Activation Scales
    # Let's take some random sequences from the corpus to calibrate activation ranges
    print("Calibrating activation scales...")
    num_calib = 50
    np.random.seed(42)
    start_idxs = np.random.randint(0, len(encoded) - seq_len, size=num_calib)

    # Per-layer residual maximums — replaces single global max_residual
    max_res_in  = [1e-5] * layers   # residual entering layer l
    max_res_out = [1e-5] * layers   # residual exiting layer l
    max_q       = [1e-5] * layers
    max_k       = [1e-5] * layers
    max_v       = [1e-5] * layers
    max_ff1_out = [1e-5] * layers

    model.eval()
    with torch.no_grad():
        for start in start_idxs:
            seq = torch.tensor(
                encoded[start : start + seq_len],
                dtype=torch.long
            ).unsqueeze(0)

            tok_emb = model.token_emb(seq)
            pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
            pos_emb = model.pos_emb(pos)
            x = tok_emb + pos_emb

            for l in range(layers):
                max_res_in[l] = max(max_res_in[l], x.abs().max().item())

                layer  = model.blocks[l]
                normed = layer.norm1(x)

                max_q[l]   = max(max_q[l],   layer.attn.q_proj(normed).abs().max().item())
                max_k[l]   = max(max_k[l],   layer.attn.k_proj(normed).abs().max().item())
                max_v[l]   = max(max_v[l],   layer.attn.v_proj(normed).abs().max().item())

                normed2      = layer.norm2(x)
                ff1_act      = F.relu(layer.ff1(normed2))
                max_ff1_out[l] = max(max_ff1_out[l], ff1_act.abs().max().item())

                # Step through the block manually to get exactly what x is at the end of the block
                attn_out = layer.attn(normed)
                x = x + attn_out
                
                norm2_out = layer.norm2(x)
                ff_out = layer.ff2(F.relu(layer.ff1(norm2_out)))
                x = x + ff_out
                
                max_res_out[l] = max(max_res_out[l], x.abs().max().item())
                
            x = model.final_norm(x)

    # Per-layer scales
    scale_res_in  = [v / 127.0 for v in max_res_in]
    scale_res_out = [v / 127.0 for v in max_res_out]
    scale_q       = [max_q[l]       / 127.0 for l in range(layers)]
    scale_k       = [max_k[l]       / 127.0 for l in range(layers)]
    scale_v       = [max_v[l]       / 127.0 for l in range(layers)]
    scale_ff1_out = [max_ff1_out[l] / 127.0 for l in range(layers)]

    print("Per-layer residual scales:")
    for l in range(layers):
        print(f"  Layer {l}: in={scale_res_in[l]:.5f}  out={scale_res_out[l]:.5f}")
    print(f"Q scales: {[f'{s:.5f}' for s in scale_q]}")
    print(f"K scales: {[f'{s:.5f}' for s in scale_k]}")
    
    # We still need scale_residual for anything that still expects it globally 
    # (like the final norm / lm_head which haven't been per-layerified yet)
    scale_residual = max(scale_res_out)
    for l in range(layers):
        np.save(f"artifacts/l{l}_res_in_scale.npy", np.array([scale_res_in[l]], dtype=np.float32))
        np.save(f"artifacts/l{l}_res_out_scale.npy", np.array([scale_res_out[l]], dtype=np.float32))
    scale_q = [max_q[l] / 127.0 for l in range(layers)]
    scale_k = [max_k[l] / 127.0 for l in range(layers)]
    scale_v = [max_v[l] / 127.0 for l in range(layers)]
    scale_ff1_out = [max_ff1_out[l] / 127.0 for l in range(layers)]
    
    # 2. Weight Scale Extraction & Quantization Errors
    report = {
        "calibration": {
            "residual_scale": scale_residual,
            "layer_scales": []
        },
        "weights": {}
    }
    
    def process_weight_per_channel(name, tensor):
        # tensor is [OUT, IN]
        scales = tensor.abs().max(dim=1)[0] / 127.0
        scales[scales == 0] = 1.0 # avoid div by zero
        
        q = torch.clamp(torch.round(tensor / scales.unsqueeze(1)), -128, 127).to(torch.int8)
        
        dequant = q.float() * scales.unsqueeze(1)
        err = torch.mean((tensor - dequant) ** 2).item()
        
        report["weights"][name] = {
            "scale_mean": scales.mean().item(),
            "scale_min": scales.min().item(),
            "scale_max": scales.max().item(),
            "mse": err,
        }
        np.save(f"artifacts/{name}_int8.npy", q.numpy())
        np.save(f"artifacts/{name}_scale.npy", scales.numpy().astype(np.float32))

    # Embeddings are lookup tables, not Linear layers. But we quantize them per-row too.
    process_weight_per_channel("token_embeddings", state["token_emb.weight"])
    process_weight_per_channel("pos_embeddings", state["pos_emb.weight"])
    def quantize_layer(weight, per_channel=True):
        if per_channel:
            scales = np.abs(weight).max(axis=1) / 127.0
            scales = np.where(scales == 0, 1e-9, scales)
            w_int8 = np.clip(np.round(weight / scales[:, None]), -128, 127).astype(np.int8)
            return w_int8, scales
        else:
            scale = np.abs(weight).max() / 127.0
            scale = scale if scale > 0 else 1e-9
            w_int8 = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)
            # Broadcast scalar to array so Rust loader doesn't panic
            scales = np.full((weight.shape[0],), scale, dtype=np.float32)
            return w_int8, scales

    for l in range(layers):
        q_w = state[f'blocks.{l}.attn.q_proj.weight'].numpy()
        k_w = state[f'blocks.{l}.attn.k_proj.weight'].numpy()
        v_w = state[f'blocks.{l}.attn.v_proj.weight'].numpy()
        out_w = state[f'blocks.{l}.attn.out_proj.weight'].numpy()
        ff1_w = state[f'blocks.{l}.ff1.weight'].numpy()
        ff2_w = state[f'blocks.{l}.ff2.weight'].numpy()
        
        # Q, K, V are per-tensor
        q_int8, q_scale = quantize_layer(q_w, per_channel=False)
        k_int8, k_scale = quantize_layer(k_w, per_channel=False)
        v_int8, v_scale = quantize_layer(v_w, per_channel=False)
        
        # Everything else remains per-channel
        out_int8, out_scale = quantize_layer(out_w, per_channel=True)
        ff1_int8, ff1_scale = quantize_layer(ff1_w, per_channel=True)
        ff2_int8, ff2_scale = quantize_layer(ff2_w, per_channel=True)
        
        np.save(f"artifacts/l{l}_q_proj_int8.npy", q_int8)
        np.save(f"artifacts/l{l}_q_proj_scale.npy", q_scale.astype(np.float32))
        np.save(f"artifacts/l{l}_k_proj_int8.npy", k_int8)
        np.save(f"artifacts/l{l}_k_proj_scale.npy", k_scale.astype(np.float32))
        np.save(f"artifacts/l{l}_v_proj_int8.npy", v_int8)
        np.save(f"artifacts/l{l}_v_proj_scale.npy", v_scale.astype(np.float32))
        np.save(f"artifacts/l{l}_out_proj_int8.npy", out_int8)
        np.save(f"artifacts/l{l}_out_proj_scale.npy", out_scale.astype(np.float32))
        np.save(f"artifacts/l{l}_ff1_int8.npy", ff1_int8)
        np.save(f"artifacts/l{l}_ff1_scale.npy", ff1_scale.astype(np.float32))
        np.save(f"artifacts/l{l}_ff2_int8.npy", ff2_int8)
        np.save(f"artifacts/l{l}_ff2_scale.npy", ff2_scale.astype(np.float32))
        
        # Norm weights are float32, keep them as is
        np.save(f"artifacts/l{l}_norm1_float.npy", state[f"blocks.{l}.norm1.weight"].numpy())
        np.save(f"artifacts/l{l}_norm2_float.npy", state[f"blocks.{l}.norm2.weight"].numpy())
        
        report["calibration"]["layer_scales"].append({
            "layer": l,
            "q_scale": scale_q[l],
            "k_scale": scale_k[l],
            "v_scale": scale_v[l],
            "ff1_out_scale": scale_ff1_out[l]
        })
        
    np.save("artifacts/final_norm_float.npy", state["final_norm.weight"].numpy())
    process_weight_per_channel("lm_head", state["lm_head.weight"])
    
    # Save active calibration scales as numpy arrays for loader script
    np.save("artifacts/scale_residual.npy", np.array([scale_residual], dtype=np.float32))
    for l in range(layers):
        np.save(f"artifacts/l{l}_q_scale.npy", np.array([scale_q[l]], dtype=np.float32))
        np.save(f"artifacts/l{l}_k_scale.npy", np.array([scale_k[l]], dtype=np.float32))
        np.save(f"artifacts/l{l}_v_scale.npy", np.array([scale_v[l]], dtype=np.float32))
        np.save(f"artifacts/l{l}_ff1_out_scale.npy", np.array([scale_ff1_out[l]], dtype=np.float32))
        
    # Save report
    with open("artifacts/quantization_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("Quantization report generated and saved to artifacts/quantization_report.json")

if __name__ == "__main__":
    main()
