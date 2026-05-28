import numpy as np
import struct
import os

def main():
    # Load dimensions
    vocab_size = int(np.load("artifacts/encoded_corpus.npy").max()) + 1
    seq_len = 64
    dim = 64
    heads = 4
    layers = 3
    hidden = 256
    
    # 1. Load weights
    token_emb = np.load("artifacts/token_embeddings_int8.npy")
    token_emb_scale = np.load("artifacts/token_embeddings_scale.npy").astype(np.float32)
        
    pos_emb = np.load("artifacts/pos_embeddings_int8.npy")
    pos_emb_scale = np.load("artifacts/pos_embeddings_scale.npy").astype(np.float32)
        
    # Layer weights
    l_q_proj = []
    l_q_scale = []
    l_k_proj = []
    l_k_scale = []
    l_v_proj = []
    l_v_scale = []
    l_out_proj = []
    l_out_scale = []
    l_ff1 = []
    l_ff1_scale = []
    l_ff2 = []
    l_ff2_scale = []
    
    l_norm1 = []
    l_norm2 = []
    
    for l in range(layers):
        l_q_proj.append(np.load(f"artifacts/l{l}_q_proj_int8.npy"))
        l_q_scale.append(np.load(f"artifacts/l{l}_q_proj_scale.npy").astype(np.float32))
            
        l_k_proj.append(np.load(f"artifacts/l{l}_k_proj_int8.npy"))
        l_k_scale.append(np.load(f"artifacts/l{l}_k_proj_scale.npy").astype(np.float32))
            
        l_v_proj.append(np.load(f"artifacts/l{l}_v_proj_int8.npy"))
        l_v_scale.append(np.load(f"artifacts/l{l}_v_proj_scale.npy").astype(np.float32))
            
        l_out_proj.append(np.load(f"artifacts/l{l}_out_proj_int8.npy"))
        l_out_scale.append(np.load(f"artifacts/l{l}_out_proj_scale.npy").astype(np.float32))
            
        l_ff1.append(np.load(f"artifacts/l{l}_ff1_int8.npy"))
        l_ff1_scale.append(np.load(f"artifacts/l{l}_ff1_scale.npy").astype(np.float32))
            
        l_ff2.append(np.load(f"artifacts/l{l}_ff2_int8.npy"))
        l_ff2_scale.append(np.load(f"artifacts/l{l}_ff2_scale.npy").astype(np.float32))
            
        l_norm1.append(np.load(f"artifacts/l{l}_norm1_float.npy"))
        l_norm2.append(np.load(f"artifacts/l{l}_norm2_float.npy"))
        
    final_norm = np.load("artifacts/final_norm_float.npy")
    lm_head = np.load("artifacts/lm_head_int8.npy")
    lm_head_scale = np.load("artifacts/lm_head_scale.npy").astype(np.float32)
        
    # Calibration scales
    scale_residual = float(np.load("artifacts/scale_residual.npy")[0])
    
    calib_q_scale = []
    calib_k_scale = []
    calib_v_scale = []
    calib_ff1_out_scale = []
    scale_res_in = []
    scale_res_out = []
    
    for l in range(layers):
        scale_res_in.append(float(np.load(f"artifacts/l{l}_res_in_scale.npy")[0]))
        scale_res_out.append(float(np.load(f"artifacts/l{l}_res_out_scale.npy")[0]))
        calib_q_scale.append(float(np.load(f"artifacts/l{l}_q_scale.npy")[0]))
        calib_k_scale.append(float(np.load(f"artifacts/l{l}_k_scale.npy")[0]))
        calib_v_scale.append(float(np.load(f"artifacts/l{l}_v_scale.npy")[0]))
        calib_ff1_out_scale.append(float(np.load(f"artifacts/l{l}_ff1_out_scale.npy")[0]))
        
    # Write binary file
    with open("artifacts/tiny_transformer_int8.bin", "wb") as f:
        # Header (32 bytes)
        f.write(b"QTRN") # magic
        f.write(struct.pack("<I", 1)) # version
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", seq_len))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<I", heads))
        f.write(struct.pack("<I", layers))
        f.write(struct.pack("<I", hidden))
        
        # 1. token embeddings (i8) + scale (f32)
        f.write(token_emb.tobytes())
        f.write(token_emb_scale.tobytes())
        
        # 2. pos embeddings (i8) + scale (f32)
        f.write(pos_emb.tobytes())
        f.write(pos_emb_scale.tobytes())
        
        # 3. layers
        for l in range(layers):
            # norm1 weight (f32)
            f.write(l_norm1[l].astype(np.float32).tobytes())
            
            # q_proj (i8) + scale (f32)
            f.write(l_q_proj[l].tobytes())
            f.write(l_q_scale[l].tobytes())
            
            # k_proj (i8) + scale (f32)
            f.write(l_k_proj[l].tobytes())
            f.write(l_k_scale[l].tobytes())
            
            # v_proj (i8) + scale (f32)
            f.write(l_v_proj[l].tobytes())
            f.write(l_v_scale[l].tobytes())
            
            # out_proj (i8) + scale (f32)
            f.write(l_out_proj[l].tobytes())
            f.write(l_out_scale[l].tobytes())
            
            # norm2 weight (f32)
            f.write(l_norm2[l].astype(np.float32).tobytes())
            
            # ff1 weight (i8) + scale (f32)
            f.write(l_ff1[l].tobytes())
            f.write(l_ff1_scale[l].tobytes())
            
            # ff2 weight (i8) + scale (f32)
            f.write(l_ff2[l].tobytes())
            f.write(l_ff2_scale[l].tobytes())
            
        # 4. final norm (f32)
        f.write(final_norm.astype(np.float32).tobytes())
        
        # 5. lm_head (i8) + scale (f32)
        f.write(lm_head.tobytes())
        f.write(lm_head_scale.tobytes())
        
        # 6. calibration activation scales (f32)
        for l in range(layers):
            f.write(struct.pack("<f", scale_res_in[l]))    # new: per-layer residual in
            f.write(struct.pack("<f", scale_res_out[l]))   # new: per-layer residual out
            f.write(struct.pack("<f", calib_q_scale[l]))
            f.write(struct.pack("<f", calib_k_scale[l]))
            f.write(struct.pack("<f", calib_v_scale[l]))
            f.write(struct.pack("<f", calib_ff1_out_scale[l]))
            
    print(f"Successfully exported artifacts/tiny_transformer_int8.bin (vocab_size={vocab_size})")

if __name__ == "__main__":
    main()
