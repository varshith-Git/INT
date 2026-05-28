import torch
import numpy as np
import json
from train_tiny_transformer import TinyGPT
from export_logits_128 import load_weights

def main():
    with open("artifacts/vocab.json", "r") as f:
        vocab = json.load(f)
        stoi = vocab["stoi"]
        itos = vocab["itos"]

    model = TinyGPT(vocab_size=61, seq_len=64, dim=64, heads=4, hidden=256, layers=3)
    load_weights(model)
    model.eval()

    prompt = "ACT I\n"
    context = [stoi[c] for c in prompt]
    
    print(f"Prompt: {prompt!r}")
    print("Generating 64 tokens...")
    
    with torch.no_grad():
        for _ in range(64):
            # PyTorch max context size is 64
            start_idx = len(context) - 64 if len(context) >= 64 else 0
            inp = torch.tensor(context[start_idx:], dtype=torch.long).unsqueeze(0)
            
            logits = model(inp)[0, -1, :]
            next_token = torch.argmax(logits).item()
            context.append(next_token)
            
    decoded = "".join([itos.get(str(t), "?") for t in context])
    print("\n--- Final Generated Text ---")
    print(decoded)
    print("----------------------------\n")

if __name__ == "__main__":
    main()
