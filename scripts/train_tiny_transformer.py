import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

# Deterministic setup
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, return_debug=False):
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Q K^T
        raw_scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = raw_scores / (self.head_dim ** 0.5)
        
        # Causal Mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        scaled_scores = scaled_scores.masked_fill(mask, float('-inf'))
        
        probs = F.softmax(scaled_scores, dim=-1)
        
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        if return_debug:
            return out, raw_scores, probs
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.ff1 = nn.Linear(dim, hidden, bias=False)
        self.ff2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x, return_debug=False):
        norm1_out = self.norm1(x)
        if return_debug:
            attn_out, raw_scores, probs = self.attn(norm1_out, return_debug=True)
        else:
            attn_out = self.attn(norm1_out)
        h1 = x + attn_out
        
        norm2_out = self.norm2(h1)
        ff_out = self.ff2(F.relu(self.ff1(norm2_out)))
        h2 = h1 + ff_out
        
        if return_debug:
            return h2, raw_scores, probs
        return h2

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, dim, heads, hidden, layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, hidden) for _ in range(layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, return_debug=False):
        B, T = idx.size()
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        
        x = tok_emb + pos_emb
        
        debug_info = {}
        
        for i, block in enumerate(self.blocks):
            if return_debug:
                x_new, scores, probs = block(x, return_debug=True)
                debug_info[f'l{i}_hidden'] = x_new
                debug_info[f'l{i}_scores'] = scores
                debug_info[f'l{i}_probs'] = probs
                x = x_new
            else:
                x = block(x)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if return_debug:
            return logits, debug_info
        return logits

def get_batch(data, seq_len, batch_size):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1], dtype=torch.long) for i in ix])
    return x, y

def main():
    encoded = np.load("artifacts/encoded_corpus.npy")
    vocab_size = int(np.max(encoded)) + 1
    
    dim = 64
    heads = 4
    layers = 3
    hidden = 256
    seq_len = 64
    batch_size = 64
    epochs = 20
    batches_per_epoch = 150
    
    model = TinyGPT(vocab_size, seq_len, dim, heads, hidden, layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    
    print("Training tiny transformer...")
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        for _ in range(batches_per_epoch):
            x, y = get_batch(encoded, seq_len, batch_size)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {loss_sum/batches_per_epoch:.4f}")
        
    model.eval()
    
    # Save float weights
    state = model.state_dict()
    os.makedirs("artifacts", exist_ok=True)
    
    np.save("artifacts/token_embeddings.npy", state["token_emb.weight"].numpy())
    np.save("artifacts/pos_embeddings.npy", state["pos_emb.weight"].numpy())
    
    for l in range(layers):
        np.save(f"artifacts/l{l}_norm1.npy", state[f"blocks.{l}.norm1.weight"].numpy())
        np.save(f"artifacts/l{l}_q_proj.npy", state[f"blocks.{l}.attn.q_proj.weight"].numpy())
        np.save(f"artifacts/l{l}_k_proj.npy", state[f"blocks.{l}.attn.k_proj.weight"].numpy())
        np.save(f"artifacts/l{l}_v_proj.npy", state[f"blocks.{l}.attn.v_proj.weight"].numpy())
        np.save(f"artifacts/l{l}_out_proj.npy", state[f"blocks.{l}.attn.out_proj.weight"].numpy())
        np.save(f"artifacts/l{l}_norm2.npy", state[f"blocks.{l}.norm2.weight"].numpy())
        np.save(f"artifacts/l{l}_ff1.npy", state[f"blocks.{l}.ff1.weight"].numpy())
        np.save(f"artifacts/l{l}_ff2.npy", state[f"blocks.{l}.ff2.weight"].numpy())
        
    np.save("artifacts/final_norm.npy", state["final_norm.weight"].numpy())
    np.save("artifacts/lm_head.npy", state["lm_head.weight"].numpy())
    torch.save(model, "artifacts/tiny_transformer.pt")
    print("Saved all float weights to artifacts/")
    
    # Generate reference debug sequence and activations using a sample prompt
    # Let's take the first 6 tokens from corpus as prompt
    sample_prompt = encoded[:6]
    np.save("artifacts/sample_prompt.npy", sample_prompt)
    with open("artifacts/sample_prompt.bin", "wb") as f:
        f.write(sample_prompt.astype(np.int32).tobytes())
    
    prompt_tensor = torch.tensor(sample_prompt, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits, debug = model(prompt_tensor, return_debug=True)
        
        # Save reference activations (npy)
        np.save("artifacts/layer0_hidden.npy", debug["l0_hidden"].numpy())
        np.save("artifacts/layer1_hidden.npy", debug["l1_hidden"].numpy())
        np.save("artifacts/attention_scores.npy", debug["l0_scores"].numpy()) # raw scores
        np.save("artifacts/attention_probs.npy", debug["l0_probs"].numpy())
        np.save("artifacts/final_logits.npy", logits.numpy())
        
        # Save reference activations (bin)
        with open("artifacts/layer0_hidden.bin", "wb") as f:
            f.write(debug["l0_hidden"].numpy().astype(np.float32).tobytes())
        with open("artifacts/layer1_hidden.bin", "wb") as f:
            f.write(debug["l1_hidden"].numpy().astype(np.float32).tobytes())
        with open("artifacts/attention_scores.bin", "wb") as f:
            f.write(debug["l0_scores"].numpy().astype(np.float32).tobytes())
        with open("artifacts/attention_probs.bin", "wb") as f:
            f.write(debug["l0_probs"].numpy().astype(np.float32).tobytes())
        with open("artifacts/final_logits.bin", "wb") as f:
            f.write(logits.numpy().astype(np.float32).tobytes())
        
    print("Saved reference forward pass activations.")
    
    # Autoregressive generation rollout parity references (32, 64, 128 steps)
    # Greedy decoding temperature=0
    for length in [32, 64, 128]:
        generated = list(sample_prompt)
        for _ in range(length - len(sample_prompt)):
            inp = torch.tensor(generated[-seq_len:], dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                logits = model(inp)
                next_token = torch.argmax(logits[0, -1, :]).item()
                generated.append(next_token)
        np.save(f"artifacts/expected_generated_tokens_{length}.npy", np.array(generated, dtype=np.int32))
        with open(f"artifacts/expected_generated_tokens_{length}.bin", "wb") as f:
            f.write(np.array(generated, dtype=np.int32).tobytes())
        print(f"Generated expected sequence of length {length}")

if __name__ == "__main__":
    main()
