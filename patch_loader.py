import sys

content = open("src/model/loader.rs").read()

# 1. Update load_transformer_from_bytes
old1 = '''    let residual_scale = reader.extract_f32()?;
    
    let mut calib_q_scale = [0.0; LAYERS];
    let mut calib_k_scale = [0.0; LAYERS];
    let mut calib_v_scale = [0.0; LAYERS];
    let mut calib_ff1_out_scale = [0.0; LAYERS];
    
    for l in 0..LAYERS {
        calib_q_scale[l] = reader.extract_f32()?;
        calib_k_scale[l] = reader.extract_f32()?;
        calib_v_scale[l] = reader.extract_f32()?;
        calib_ff1_out_scale[l] = reader.extract_f32()?;
    }
    
    let mut block_weights_iter = block_weights.into_iter();
    let blocks = core::array::from_fn(|i| {
        construct_gpt_block(
            block_weights_iter.next().unwrap(),
            residual_scale,
            calib_q_scale[i],
            calib_k_scale[i],
            calib_v_scale[i],
            calib_ff1_out_scale[i],
        )
    });
    
    let lm_head = QLinear::new(
        lm_head_w,
        &lm_head_scale,
        None,
        residual_scale,
        QuantParams::symmetric(1.0),
        false,
    );'''

new1 = '''    let mut s_res_in = [0.0; LAYERS];
    let mut s_res_out = [0.0; LAYERS];
    let mut calib_q_scale = [0.0; LAYERS];
    let mut calib_k_scale = [0.0; LAYERS];
    let mut calib_v_scale = [0.0; LAYERS];
    let mut calib_ff1_out_scale = [0.0; LAYERS];
    
    for l in 0..LAYERS {
        s_res_in[l] = reader.extract_f32()?;
        s_res_out[l] = reader.extract_f32()?;
        calib_q_scale[l] = reader.extract_f32()?;
        calib_k_scale[l] = reader.extract_f32()?;
        calib_v_scale[l] = reader.extract_f32()?;
        calib_ff1_out_scale[l] = reader.extract_f32()?;
    }
    
    let mut block_weights_iter = block_weights.into_iter();
    let blocks = core::array::from_fn(|i| {
        construct_gpt_block(
            block_weights_iter.next().unwrap(),
            s_res_in[i],
            s_res_out[i],
            calib_q_scale[i],
            calib_k_scale[i],
            calib_v_scale[i],
            calib_ff1_out_scale[i],
        )
    });
    
    let lm_head = QLinear::new(
        lm_head_w,
        &lm_head_scale,
        None,
        s_res_out[LAYERS - 1], // Final residual out
        QuantParams::symmetric(1.0),
        false,
    );'''

assert old1 in content, "Failed to find old1 in loader.rs"
content = content.replace(old1, new1, 1)

# 2. Update construct_gpt_block signature
old2 = '''fn construct_gpt_block<const SEQ: usize, const DIM: usize, const HIDDEN: usize, const HEADS: usize>(
    weights: LoadedBlockWeights<DIM, HIDDEN>,
    residual_scale: f32,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    ff1_out_scale: f32,
) -> crate::nn::gpt::QGPTBlock<SEQ, DIM, HIDDEN, HEADS> {'''

new2 = '''fn construct_gpt_block<const SEQ: usize, const DIM: usize, const HIDDEN: usize, const HEADS: usize>(
    weights: LoadedBlockWeights<DIM, HIDDEN>,
    s_res_in: f32,
    s_res_out: f32,
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    ff1_out_scale: f32,
) -> crate::nn::gpt::QGPTBlock<SEQ, DIM, HIDDEN, HEADS> {'''

assert old2 in content, "Failed to find old2 in loader.rs"
content = content.replace(old2, new2, 1)

# 3. Replace residual_scale -> s_res_in inside construct_gpt_block linear layers
old3 = '''    let q_proj = QLinear::new(
        q_w_t,
        &weights.q_scale,
        None,
        residual_scale,
        QuantParams::symmetric(q_scale),
        false,
    );
    let k_proj = QLinear::new(
        k_w_t,
        &weights.k_scale,
        None,
        residual_scale,
        QuantParams::symmetric(k_scale),
        false,
    );
    let v_proj = QLinear::new(
        v_w_t,
        &weights.v_scale,
        None,
        residual_scale,
        QuantParams::symmetric(v_scale),
        false,
    );
    let out_proj = QLinear::new(
        out_w_t,
        &weights.out_scale,
        None,
        v_scale,
        QuantParams::symmetric(residual_scale),
        false,
    );
    let ff1 = QLinear::new(
        ff1_w_t,
        &weights.ff1_scale,
        None,
        residual_scale,
        QuantParams::symmetric(ff1_out_scale),
        true,
    );
    let ff2 = QLinear::new(
        ff2_w_t,
        &weights.ff2_scale,
        None,
        ff1_out_scale,
        QuantParams::symmetric(residual_scale),
        false,
    );'''

new3 = '''    let q_proj = QLinear::new(
        q_w_t,
        &weights.q_scale,
        None,
        s_res_in,
        QuantParams::symmetric(q_scale),
        false,
    );
    let k_proj = QLinear::new(
        k_w_t,
        &weights.k_scale,
        None,
        s_res_in,
        QuantParams::symmetric(k_scale),
        false,
    );
    let v_proj = QLinear::new(
        v_w_t,
        &weights.v_scale,
        None,
        s_res_in,
        QuantParams::symmetric(v_scale),
        false,
    );
    let out_proj = QLinear::new(
        out_w_t,
        &weights.out_scale,
        None,
        v_scale,
        QuantParams::symmetric(s_res_in),
        false,
    );
    let ff1 = QLinear::new(
        ff1_w_t,
        &weights.ff1_scale,
        None,
        s_res_in,
        QuantParams::symmetric(ff1_out_scale),
        true,
    );
    let ff2 = QLinear::new(
        ff2_w_t,
        &weights.ff2_scale,
        None,
        ff1_out_scale,
        QuantParams::symmetric(s_res_in),
        false,
    );'''

assert old3 in content, "Failed to find old3 in loader.rs"
content = content.replace(old3, new3, 1)

# 4. update returned QGPTBlock
old4 = '''    crate::nn::gpt::QGPTBlock {
        norm1: weights.norm1,
        norm2: weights.norm2,
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        ff1,
        ff2,
        attn_scale,
        attn_requant,
        attn_out_params: QuantParams::symmetric(v_scale),
    }'''

new4 = '''    crate::nn::gpt::QGPTBlock {
        norm1: weights.norm1,
        norm2: weights.norm2,
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        ff1,
        ff2,
        attn_scale,
        attn_requant,
        attn_out_params: QuantParams::symmetric(v_scale),
        s_res_in: QuantParams::symmetric(s_res_in),
        s_res_out: QuantParams::symmetric(s_res_out),
    }'''

assert old4 in content, "Failed to find old4 in loader.rs"
content = content.replace(old4, new4, 1)

open("src/model/loader.rs", "w").write(content)
print("loader.rs patched")
