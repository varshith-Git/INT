import sys

content = open("scripts/compare_layer_by_layer.py").read()

old = "model = torch.load('artifacts/tiny_transformer.pt', map_location='cpu')"
new = """import sys
sys.path.append('.')
from scripts.train_tiny_transformer import TinyGPT
model = torch.load('artifacts/tiny_transformer.pt', map_location='cpu', weights_only=False)"""

assert old in content, "Failed old"
content = content.replace(old, new, 1)

open("scripts/compare_layer_by_layer.py", "w").write(content)
print("patched")
