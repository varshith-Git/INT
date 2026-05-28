import sys

content = open("scripts/quantize_transformer.py").read()

old = '''    # We still need scale_residual for anything that still expects it globally 
    # (like the final norm / lm_head which haven't been per-layerified yet)
    scale_residual = max(scale_res_out)
    for l in range(layers):
        import numpy as np
        np.save(f"artifacts/l{l}_res_in_scale.npy", np.array([scale_res_in[l]], dtype=np.float32))
        np.save(f"artifacts/l{l}_res_out_scale.npy", np.array([scale_res_out[l]], dtype=np.float32))'''
    
new = '''    # We still need scale_residual for anything that still expects it globally 
    # (like the final norm / lm_head which haven't been per-layerified yet)
    scale_residual = max(scale_res_out)
    for l in range(layers):
        np.save(f"artifacts/l{l}_res_in_scale.npy", np.array([scale_res_in[l]], dtype=np.float32))
        np.save(f"artifacts/l{l}_res_out_scale.npy", np.array([scale_res_out[l]], dtype=np.float32))'''

assert old in content, "Failed to patch quantize_transformer.py"
content = content.replace(old, new, 1)

open("scripts/quantize_transformer.py", "w").write(content)
print("quantize_transformer.py patched")
