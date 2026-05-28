import sys

content = open("scripts/compare_layer_by_layer.py").read()

old = "tokens = [vocab[c] for c in prompt]"
new = "tokens = [vocab['stoi'][c] for c in prompt]"

assert old in content, "Failed old"
content = content.replace(old, new, 1)

open("scripts/compare_layer_by_layer.py", "w").write(content)
print("patched")
