import numpy as np
import struct
import os

def main():
    vocab_size = int(np.load("artifacts/encoded_corpus.npy").max()) + 1
    seq_len = 64

    # Infer dim and layers from what quantize_transformer.py wrote
    token_emb_i8 = np.load("artifacts/token_embeddings_int8.npy")
    dim = token_emb_i8.shape[1]
    layers = 0
    while os.path.exists(f"artifacts/l{layers}_q_proj_int8.npy"):
        layers += 1
    ff1 = np.load("artifacts/l0_ff1_int8.npy")
    hidden = ff1.shape[0]
    heads_dim = np.load("artifacts/l0_q_proj_int8.npy").shape[0]
    # heads is encoded in the Rust loader via DIM/HEAD_DIM — we just store the
    # projection shapes; the loader derives heads from dim and the weight shape.
    # Write heads as dim//64 (standard head_dim=64 convention).
    heads = max(1, dim // 64)

    print(f"Exporting: vocab={vocab_size} seq={seq_len} dim={dim} hidden={hidden} heads={heads} layers={layers}")

    token_emb_scale = np.load("artifacts/token_embeddings_scale.npy").astype(np.float32)
    pos_emb        = np.load("artifacts/pos_embeddings_int8.npy")
    pos_emb_scale  = np.load("artifacts/pos_embeddings_scale.npy").astype(np.float32)

    l_q_proj, l_q_scale = [], []
    l_k_proj, l_k_scale = [], []
    l_v_proj, l_v_scale = [], []
    l_out_proj, l_out_scale = [], []
    l_ff1, l_ff1_scale = [], []
    l_ff2, l_ff2_scale = [], []
    l_norm1, l_norm2 = [], []
    calib_q_scale, calib_k_scale, calib_v_scale = [], [], []
    calib_ff1_out_scale, scale_res_in, scale_res_out = [], [], []

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
        # Norm weights are stored as float32
        l_norm1.append(np.load(f"artifacts/l{l}_norm1.npy").astype(np.float32))
        l_norm2.append(np.load(f"artifacts/l{l}_norm2.npy").astype(np.float32))
        scale_res_in.append(float(np.load(f"artifacts/l{l}_res_in_scale.npy")[0]))
        scale_res_out.append(float(np.load(f"artifacts/l{l}_res_out_scale.npy")[0]))
        calib_q_scale.append(float(np.load(f"artifacts/l{l}_q_scale.npy")[0]))
        calib_k_scale.append(float(np.load(f"artifacts/l{l}_k_scale.npy")[0]))
        calib_v_scale.append(float(np.load(f"artifacts/l{l}_v_scale.npy")[0]))
        calib_ff1_out_scale.append(float(np.load(f"artifacts/l{l}_ff1_out_scale.npy")[0]))

    final_norm   = np.load("artifacts/final_norm.npy").astype(np.float32)
    lm_head      = np.load("artifacts/lm_head_int8.npy") if os.path.exists("artifacts/lm_head_int8.npy") else np.load("artifacts/lm_head.npy").astype(np.int8)
    lm_head_scale = np.load("artifacts/lm_head_scale.npy").astype(np.float32) if os.path.exists("artifacts/lm_head_scale.npy") else np.array([1.0], dtype=np.float32)

    with open("artifacts/tiny_transformer_int8.bin", "wb") as f:
        # Header: magic(4) + version(4) + vocab(4) + seq(4) + dim(4) + heads(4) + layers(4) + hidden(4)
        f.write(b"QTRN")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", seq_len))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<I", heads))
        f.write(struct.pack("<I", layers))
        f.write(struct.pack("<I", hidden))

        f.write(token_emb_i8.tobytes());  f.write(token_emb_scale.tobytes())
        f.write(pos_emb.tobytes());       f.write(pos_emb_scale.tobytes())

        for l in range(layers):
            f.write(l_norm1[l].tobytes())
            f.write(l_q_proj[l].tobytes());   f.write(l_q_scale[l].tobytes())
            f.write(l_k_proj[l].tobytes());   f.write(l_k_scale[l].tobytes())
            f.write(l_v_proj[l].tobytes());   f.write(l_v_scale[l].tobytes())
            f.write(l_out_proj[l].tobytes()); f.write(l_out_scale[l].tobytes())
            f.write(l_norm2[l].tobytes())
            f.write(l_ff1[l].tobytes());      f.write(l_ff1_scale[l].tobytes())
            f.write(l_ff2[l].tobytes());      f.write(l_ff2_scale[l].tobytes())

        f.write(final_norm.tobytes())
        f.write(lm_head.tobytes()); f.write(lm_head_scale.tobytes())

        for l in range(layers):
            f.write(struct.pack("<f", scale_res_in[l]))
            f.write(struct.pack("<f", scale_res_out[l]))
            f.write(struct.pack("<f", calib_q_scale[l]))
            f.write(struct.pack("<f", calib_k_scale[l]))
            f.write(struct.pack("<f", calib_v_scale[l]))
            f.write(struct.pack("<f", calib_ff1_out_scale[l]))

    size_mb = os.path.getsize("artifacts/tiny_transformer_int8.bin") / 1e6
    print(f"Exported artifacts/tiny_transformer_int8.bin  ({size_mb:.1f} MB, {layers} layers, dim={dim})")

if __name__ == "__main__":
    main()
