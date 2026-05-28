import sys

content = open("scripts/export_transformer_bin.py").read()

# Replace loading
old_load = '''    calib_ff1_out_scale = []
    
    for l in range(layers):'''
new_load = '''    calib_ff1_out_scale = []
    scale_res_in = []
    scale_res_out = []
    
    for l in range(layers):
        scale_res_in.append(float(np.load(f"artifacts/l{l}_res_in_scale.npy")[0]))
        scale_res_out.append(float(np.load(f"artifacts/l{l}_res_out_scale.npy")[0]))'''
assert old_load in content, "Failed to find load block in export"
content = content.replace(old_load, new_load, 1)

# Replace writing
old_write = '''        # 6. calibration activation scales (f32)
        f.write(struct.pack("<f", scale_residual))
        for l in range(layers):
            f.write(struct.pack("<f", calib_q_scale[l]))
            f.write(struct.pack("<f", calib_k_scale[l]))
            f.write(struct.pack("<f", calib_v_scale[l]))
            f.write(struct.pack("<f", calib_ff1_out_scale[l]))'''
new_write = '''        # 6. calibration activation scales (f32)
        for l in range(layers):
            f.write(struct.pack("<f", scale_res_in[l]))    # new: per-layer residual in
            f.write(struct.pack("<f", scale_res_out[l]))   # new: per-layer residual out
            f.write(struct.pack("<f", calib_q_scale[l]))
            f.write(struct.pack("<f", calib_k_scale[l]))
            f.write(struct.pack("<f", calib_v_scale[l]))
            f.write(struct.pack("<f", calib_ff1_out_scale[l]))'''
assert old_write in content, "Failed to find write block in export"
content = content.replace(old_write, new_write, 1)

open("scripts/export_transformer_bin.py", "w").write(content)
print("export_transformer_bin.py patched")
