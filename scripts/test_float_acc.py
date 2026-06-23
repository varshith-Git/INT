import torch
import numpy as np

# Load corpus
encoded = np.load("artifacts/encoded_corpus.npy")
length = 1050
sample = encoded[:length].astype(np.int64)

# Load PyTorch model
model = torch.load("artifacts/tiny_transformer.pt", map_location="cpu")
model.eval()

correct = 0
total = 1000

print("Running float model eval...")
for i in range(1000):
    seq = sample[i:i+6]
    target = sample[i+6]
    
    with torch.no_grad():
        logits = model(torch.tensor(seq).unsqueeze(0))
    
    pred = logits[0, -1, :].argmax().item()
    if pred == target:
        correct += 1
        
print(f"Float model top-1 accuracy: {correct}/{total} = {correct/total*100:.1f}%")
