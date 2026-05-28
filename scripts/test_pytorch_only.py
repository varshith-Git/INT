import torch
import json
import sys
import os

# Import the model architecture
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.train_tiny_transformer import TinyGPT

# Load vocab
with open('artifacts/vocab.json') as f:
    vocab_raw = json.load(f)
    vocab = vocab_raw['stoi'] if 'stoi' in vocab_raw else vocab_raw
idx_to_char = {v: k for k, v in vocab.items()}

# Load model
model = TinyGPT(vocab_size=len(vocab), seq_len=64, dim=64, heads=4, hidden=256, layers=3)
state = model.state_dict()
import numpy as np
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
model.eval()

# Encode "ACT I\n"
prompt = "ACT I\n"
char_to_idx = vocab
tokens = [char_to_idx[c] for c in prompt]
x = torch.tensor([tokens])

# Run 10 steps of greedy decode
print("PyTorch float32 generation:")
with torch.no_grad():
    for i in range(20):
        logits = model(x)
        next_token = logits[0, -1, :].argmax().item()
        char = idx_to_char.get(next_token, '?')
        print(f"  Step {i+1}: token={next_token} char='{repr(char)}' logit={logits[0,-1,next_token]:.4f}")
        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

# Now test: does the model produce stable, non-degenerate output?
print("\nFirst 5 logit distributions:")
x2 = torch.tensor([tokens])
with torch.no_grad():
    for i in range(5):
        logits = model(x2)
        top5 = logits[0, -1, :].topk(5)
        print(f"  Step {i+1} top5: {list(zip(top5.indices.tolist(), [f'{v:.3f}' for v in top5.values.tolist()]))}")
        x2 = torch.cat([x2, torch.tensor([[top5.indices[0].item()]])], dim=1)
