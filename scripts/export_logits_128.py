import torch
import numpy as np
from train_tiny_transformer import TinyGPT
import json
import os

def load_weights(model):
    state = model.state_dict()
    state["token_emb.weight"] = torch.tensor(np.load("artifacts/token_embeddings.npy"))
    state["pos_emb.weight"] = torch.tensor(np.load("artifacts/pos_embeddings.npy"))
    for l in range(3):
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

def main():
    # Force sequence length to 64 to match training
    seq_len = 64
    model = TinyGPT(vocab_size=61, seq_len=seq_len, dim=64, heads=4, hidden=256, layers=3)
    load_weights(model)
    model.eval()

    py_gen = np.load("artifacts/expected_generated_tokens_128.npy")
    
    top3_data = {}
    
    for step in range(6, 106):
        start_idx = step - seq_len if step >= seq_len else 0
        context = py_gen[start_idx:step]
        inp = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(inp)[0, -1, :]
            
            top3_vals, top3_idx = torch.topk(logits, 3)
            
            top3_data[step] = {
                "indices": top3_idx.tolist(),
                "logits": top3_vals.tolist()
            }
            
    with open("artifacts/py_top3_128.txt", "w") as f:
        for step, data in top3_data.items():
            idx_str = ",".join(str(i) for i in data["indices"])
            val_str = ",".join(f"{v:.4f}" for v in data["logits"])
            f.write(f"{step}|{idx_str}|{val_str}\n")
    print("Saved PyTorch top-3 logits to artifacts/py_top3_128.txt")

if __name__ == "__main__":
    main()
