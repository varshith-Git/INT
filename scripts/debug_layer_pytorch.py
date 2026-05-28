import torch
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.train_tiny_transformer import TinyGPT

with open('artifacts/vocab.json') as f:
    vocab_raw = json.load(f)
    vocab = vocab_raw['stoi'] if 'stoi' in vocab_raw else vocab_raw

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

prompt = "ACT I\n"
tokens = [vocab[c] for c in prompt]
x = torch.tensor([tokens])

print("=== EMBEDDING OUTPUT (first token position) ===")
with torch.no_grad():
    token_emb = model.token_emb(x)
    pos_emb = model.pos_emb(torch.arange(x.size(1), device=x.device))
    emb = token_emb + pos_emb
    print(f"  shape: {emb.shape}")
    print(f"  absmax: {emb.abs().max().item():.4f}")
    print(f"  mean:   {emb.mean().item():.4f}")
    print(f"  top5 values at pos 0: {emb[0,0,:].topk(5).values.tolist()}")

print("\n=== LAYER 0 OUTPUT ===")
with torch.no_grad():
    out0 = model.blocks[0](emb)
    print(f"  absmax: {out0.abs().max().item():.4f}")
    print(f"  mean:   {out0.mean().item():.4f}")
    print(f"  top5 at pos -1: {out0[0,-1,:].topk(5).values.tolist()}")
    print(f"  top5 idx at pos -1: {out0[0,-1,:].topk(5).indices.tolist()}")

print("\n=== LAYER 1 OUTPUT ===")
with torch.no_grad():
    out1 = model.blocks[1](out0)
    print(f"  absmax: {out1.abs().max().item():.4f}")
    print(f"  mean:   {out1.mean().item():.4f}")
    print(f"  top5 at pos -1: {out1[0,-1,:].topk(5).values.tolist()}")
    print(f"  top5 idx at pos -1: {out1[0,-1,:].topk(5).indices.tolist()}")

print("\n=== LAYER 2 OUTPUT ===")
with torch.no_grad():
    out2 = model.blocks[2](out1)
    print(f"  absmax: {out2.abs().max().item():.4f}")
    print(f"  mean:   {out2.mean().item():.4f}")
    print(f"  top5 at pos -1: {out2[0,-1,:].topk(5).values.tolist()}")
    print(f"  top5 idx at pos -1: {out2[0,-1,:].topk(5).indices.tolist()}")

print("\n=== FINAL LOGITS ===")
with torch.no_grad():
    normed = model.final_norm(out2)
    logits = model.lm_head(normed)
    print(f"  absmax: {logits.abs().max().item():.4f}")
    print(f"  top5 at pos -1: {logits[0,-1,:].topk(5).values.tolist()}")
    print(f"  top5 idx at pos -1: {logits[0,-1,:].topk(5).indices.tolist()}")
