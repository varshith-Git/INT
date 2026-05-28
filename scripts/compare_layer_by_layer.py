import torch
import json
import os
import subprocess

# 1. PyTorch Evaluation
import sys
sys.path.append('.')
import scripts.train_tiny_transformer as train
sys.modules['__main__'].TinyGPT = train.TinyGPT
sys.modules['__main__'].TransformerBlock = train.TransformerBlock
sys.modules['__main__'].CausalSelfAttention = train.CausalSelfAttention
sys.modules['__main__'].RMSNorm = train.RMSNorm
model = torch.load('artifacts/tiny_transformer.pt', map_location='cpu', weights_only=False)
model.eval()

with open('artifacts/vocab.json') as f:
    vocab = json.load(f)

prompt = "ACT I\n"
tokens = [vocab['stoi'][c] for c in prompt]
# pad to seq len 64 with 0s
tokens_padded = tokens + [0] * (64 - len(tokens))

x = torch.tensor([tokens_padded])

print("=== PyTorch ===")
with torch.no_grad():
    pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
    emb = model.token_emb(x) + model.pos_emb(pos)
    print(f"After embedding - absmax: {emb.abs().max():.4f}")
    
    out = emb
    for i, layer in enumerate(model.blocks):
        print(f"Before layer {i} - absmax: {out.abs().max():.4f}")
        out = layer(out)
        print(f"After layer {i} - absmax: {out.abs().max():.4f}")
    
    out = model.final_norm(out)
    logits = model.lm_head(out)
    # Get top 3 of the last token in the unpadded prompt
    last_token_idx = len(tokens) - 1
    last_logits = logits[0, last_token_idx]
    vals, idxs = torch.topk(last_logits, 3)
    print(f"Logits Top 3:")
    for v, idx in zip(vals, idxs):
        print(f"  {idx.item()}: {v.item():.4f}")

# 2. Rust Evaluation
print("\n=== Rust ===")
# We need to run a rust binary that processes the same prompt
# First, let's create a temporary rust file that does exactly this.
