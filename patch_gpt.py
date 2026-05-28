import sys

content = open("src/nn/gpt.rs").read()

# 1. Update struct QGPTBlock
old1 = '''    pub attn_scale: f32,
    pub attn_requant: RequantShift,
    pub attn_out_params: QuantParams,
}'''
new1 = '''    pub attn_scale: f32,
    pub attn_requant: RequantShift,
    pub attn_out_params: QuantParams,
    pub s_res_in: QuantParams,
    pub s_res_out: QuantParams,
}'''
assert old1 in content, "Failed old1"
content = content.replace(old1, new1, 1)

# 2. Update forward
old2 = '''        let s_res_scale = self.q_proj.input_scale;
        let (x_i8_data, dyn_scale) = scale_matrix_i32_to_i8(x_i32);
        let x_i8 = QTensor::new(x_i8_data, QuantParams::symmetric(s_res_scale * dyn_scale));

        let norm1_out = self.norm1.forward(&x_i8);'''
new2 = '''        let (x_i8_data, dyn_scale) = scale_matrix_i32_to_i8(x_i32);
        let x_i8 = QTensor::new(x_i8_data, QuantParams::symmetric(self.s_res_in.scale * dyn_scale));

        let norm1_out = self.norm1.forward(&x_i8);'''
assert old2 in content, "Failed old2"
content = content.replace(old2, new2, 1)

old3 = '''        let norm2_out = self.norm2.forward_i32(x_i32);
        let mut ff1_out = self.ff1.forward_dynamic(&norm2_out);'''
new3 = '''        let (x_i8_data2, dyn_scale2) = scale_matrix_i32_to_i8(x_i32);
        let x_i8_2 = QTensor::new(x_i8_data2, QuantParams::symmetric(self.s_res_in.scale * dyn_scale2));
        let norm2_out = self.norm2.forward(&x_i8_2);
        let mut ff1_out = self.ff1.forward_dynamic(&norm2_out);'''
assert old3 in content, "Failed old3"
content = content.replace(old3, new3, 1)

# 3. Update forward_incremental
old4 = '''        let s_res_scale = self.q_proj.input_scale;
        let (x_i8_data, dyn_scale) = scale_matrix_i32_to_i8(x_i32);
        let x_i8 = QTensor::new(x_i8_data, QuantParams::symmetric(s_res_scale * dyn_scale));

        let norm1_out = self.norm1.forward(&x_i8);'''
new4 = '''        let (x_i8_data, dyn_scale) = scale_matrix_i32_to_i8(x_i32);
        let x_i8 = QTensor::new(x_i8_data, QuantParams::symmetric(self.s_res_in.scale * dyn_scale));

        let norm1_out = self.norm1.forward(&x_i8);'''
assert old4 in content, "Failed old4"
content = content.replace(old4, new4, 1)

old5 = '''        let norm2_out = self.norm2.forward_i32(x_i32);
        let mut ff1_out = self.ff1.forward_dynamic(&norm2_out);'''
new5 = '''        let (x_i8_data2, dyn_scale2) = scale_matrix_i32_to_i8(x_i32);
        let x_i8_2 = QTensor::new(x_i8_data2, QuantParams::symmetric(self.s_res_in.scale * dyn_scale2));
        let norm2_out = self.norm2.forward(&x_i8_2);
        let mut ff1_out = self.ff1.forward_dynamic(&norm2_out);'''
assert old5 in content, "Failed old5"
content = content.replace(old5, new5, 1)

# 4. Update QGPTModel forward
old6 = '''        for i in 0..SEQ {
            let tok = tokens[i];
            let tok_scale = self.token_embeddings.weights.scale();
            let pos_scale = self.pos_embeddings.scale();
            let out_scale = self.blocks[0].q_proj.input_scale; // S_residual
            
            for d in 0..DIM {
                let tok_val = (self.token_embeddings.weights.raw(tok, d) as f32) * tok_scale;
                let pos_val = (self.pos_embeddings.raw(i, d) as f32) * pos_scale;
                let sum_val = tok_val + pos_val;
                x_mat.data[i][d] = libm::roundf(sum_val / out_scale) as i32;
            }
        }'''
new6 = '''        for i in 0..SEQ {
            let tok = tokens[i];
            let tok_scale = self.token_embeddings.weights.scale();
            let pos_scale = self.pos_embeddings.scale();
            let out_scale = self.blocks[0].s_res_in.scale;
            
            for d in 0..DIM {
                let tok_val = (self.token_embeddings.weights.raw(tok, d) as f32) * tok_scale;
                let pos_val = (self.pos_embeddings.raw(i, d) as f32) * pos_scale;
                let sum_val = tok_val + pos_val;
                x_mat.data[i][d] = libm::roundf(sum_val / out_scale) as i32;
            }
        }'''
assert old6 in content, "Failed old6"
content = content.replace(old6, new6, 1)

old7 = '''        let final_norm_out = self.final_norm.forward_i32(&x_mat);
        let logits = self.lm_head.forward_f32(&final_norm_out);'''
new7 = '''        let (final_i8, dyn_scale) = crate::quant::clip::scale_matrix_i32_to_i8(&x_mat);
        let final_norm_tensor = QTensor::new(
            final_i8,
            QuantParams::symmetric(self.blocks[LAYERS - 1].s_res_out.scale * dyn_scale)
        );
        let final_norm_out = self.final_norm.forward(&final_norm_tensor);
        let logits = self.lm_head.forward_f32(&final_norm_out);'''
assert old7 in content, "Failed old7"
content = content.replace(old7, new7, 1)

# 5. Update QGPTModel forward_incremental
old8 = '''        let tok_scale = self.token_embeddings.weights.scale();
        let pos_scale = self.pos_embeddings.scale();
        let out_scale = self.blocks[0].q_proj.input_scale;'''
new8 = '''        let tok_scale = self.token_embeddings.weights.scale();
        let pos_scale = self.pos_embeddings.scale();
        let out_scale = self.blocks[0].s_res_in.scale;'''
assert old8 in content, "Failed old8"
content = content.replace(old8, new8, 1)

old9 = '''        let (final_i8, dyn_scale) = crate::quant::clip::scale_matrix_i32_to_i8(&x_mat);
        let final_norm_tensor = QTensor::new(
            final_i8,
            QuantParams::symmetric(out_scale * dyn_scale)
        );'''
new9 = '''        let (final_i8, dyn_scale) = crate::quant::clip::scale_matrix_i32_to_i8(&x_mat);
        let final_norm_tensor = QTensor::new(
            final_i8,
            QuantParams::symmetric(self.blocks[LAYERS - 1].s_res_out.scale * dyn_scale)
        );'''
assert old9 in content, "Failed old9"
content = content.replace(old9, new9, 1)

open("src/nn/gpt.rs", "w").write(content)
print("gpt.rs patched")
