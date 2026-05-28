import sys

content = open("scripts/train_tiny_transformer.py").read()

old = '''    print("Saved all float weights to artifacts/")'''
new = '''    import torch
    torch.save(model, "artifacts/tiny_transformer.pt")
    print("Saved all float weights to artifacts/")'''

assert old in content, "Failed old"
content = content.replace(old, new, 1)

open("scripts/train_tiny_transformer.py", "w").write(content)
print("patched")
