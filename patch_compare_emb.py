import sys

content = open("scripts/compare_layer_by_layer.py").read()

old = "emb = model.embedding(x)"
new = """pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
    emb = model.token_emb(x) + model.pos_emb(pos)"""

assert old in content, "Failed old"
content = content.replace(old, new, 1)

open("scripts/compare_layer_by_layer.py", "w").write(content)
print("patched")
