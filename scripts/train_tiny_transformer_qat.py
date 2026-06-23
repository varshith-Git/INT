"""
Quantization-Aware Training (QAT) for TinyGPT-INT8.
=================================================================================
STATUS: written to match the engine's quantization scheme, but UNRUN in the
authoring environment (PyTorch failed to import there). Treat this as a reviewed
first draft + runbook, not a tested artifact. Validate per the runbook at the
bottom before trusting the numbers.

WHY THIS EXISTS
The post-training-quantized (PTQ) model collapses with depth: ~79% at 2 layers,
~41% at 3 layers (see README "Quantization Degradation Analysis"). Root cause:
the float weights were trained with no awareness of int8 rounding, so per-channel
rounding error compounds across layers and pushes activations out of the
distribution later layers expect. QAT fixes this at the source by simulating int8
rounding *inside the forward pass during training* (FakeQuantize + straight-
through estimator), so the weights learn to be robust to it.

DESIGN — must mirror the Rust runtime exactly, or QAT won't transfer:
  * Weights      : per-output-channel, symmetric int8, scale = amax(row)/127.
  * Activations  : per-tensor, symmetric int8, scale = amax/127 (dynamic).
  * Rounding     : round-to-nearest, clamp to [-127, 127] (matches requant.rs).
  * RMSNorm      : stays float32 (the engine runs RMSNorm in float — hybrid).
  * Export       : identical .npy layout to train_tiny_transformer.py, so the
                   existing quantize_transformer.py + export_transformer_bin.py
                   consume the QAT weights with NO changes.

USAGE
  python3 scripts/prepare_text.py            # produces artifacts/encoded_corpus.npy
  python3 scripts/train_tiny_transformer_qat.py
  python3 scripts/quantize_transformer.py
  python3 scripts/export_transformer_bin.py
  cargo run --release --features std --example eval_100   # check the new accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

# Deterministic setup (same seed as the float trainer).
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ── FakeQuantize: simulate int8 in the forward pass, pass gradients straight ──
class _FakeQuantSTE(torch.autograd.Function):
    """round(x/scale).clamp(-127,127)*scale forward; identity gradient (STE)."""

    @staticmethod
    def forward(ctx, x, scale):
        q = torch.clamp(torch.round(x / scale), -127.0, 127.0)
        return q * scale

    @staticmethod
    def backward(ctx, grad_out):
        # Straight-through: gradient flows unchanged to x; scale is a constant.
        return grad_out, None


def fake_quant_per_channel(w, eps=1e-8):
    """Per-output-row symmetric int8 (matches per-channel weight quant)."""
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    scale = amax / 127.0
    return _FakeQuantSTE.apply(w, scale)


def fake_quant_per_tensor(x, eps=1e-8):
    """Per-tensor symmetric int8 (matches dynamic activation scaling)."""
    amax = x.abs().amax().clamp_min(eps)
    scale = amax / 127.0
    return _FakeQuantSTE.apply(x, scale)


class QATLinear(nn.Linear):
    """nn.Linear that fake-quantizes input (per-tensor) and weight (per-channel)."""

    def forward(self, x):
        xq = fake_quant_per_tensor(x)
        wq = fake_quant_per_channel(self.weight)
        return F.linear(xq, wq, self.bias)


# ── Model — identical topology to train_tiny_transformer.py, QAT-wrapped ─────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight  # float, by design


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.q_proj = QATLinear(dim, dim, bias=False)
        self.k_proj = QATLinear(dim, dim, bias=False)
        self.v_proj = QATLinear(dim, dim, bias=False)
        self.out_proj = QATLinear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.ff1 = QATLinear(dim, hidden, bias=False)
        self.ff2 = QATLinear(hidden, dim, bias=False)

    def forward(self, x):
        h1 = x + self.attn(self.norm1(x))
        h2 = h1 + self.ff2(F.relu(self.ff1(self.norm2(h1))))
        return h2


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, dim, heads, hidden, layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, hidden) for _ in range(layers)])
        self.final_norm = RMSNorm(dim)
        self.lm_head = QATLinear(dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        # Embeddings are quantized per-tensor in the engine; fake-quant them too.
        x = fake_quant_per_tensor(self.token_emb(idx)) + fake_quant_per_tensor(self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def get_batch(data, seq_len, batch_size):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i + seq_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i + 1:i + seq_len + 1], dtype=torch.long) for i in ix])
    return x, y


def main():
    encoded = np.load("artifacts/encoded_corpus.npy")
    vocab_size = int(np.max(encoded)) + 1

    # ── Model config ─────────────────────────────────────────────────────────────
    dim, heads, layers, hidden, seq_len = 512, 8, 12, 2048, 64
    batch_size, batches_per_epoch = 32, 300
    epochs, lr = 30, 3e-4

    # ── Device: use Apple MPS (GPU) if available ─────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Training on Apple MPS (GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on CUDA")
    else:
        device = torch.device("cpu")
        print("Training on CPU (will be slow)")

    model = TinyGPT(vocab_size, seq_len, dim, heads, hidden, layers).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        for _ in range(batches_per_epoch):
            x, y = get_batch(encoded, seq_len, batch_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss_sum / batches_per_epoch:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    model.eval()

    # ── Export float weights — identical layout; quantize_transformer.py reads these
    state = {k: v.cpu() for k, v in model.state_dict().items()}
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
    print(f"Saved QAT float weights ({layers} layers, dim={dim}) to artifacts/")


if __name__ == "__main__":
    main()
