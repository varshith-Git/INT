import sys

content = open("src/nn/gpt.rs").read()

# Helper function to get absmax from x_mat
absmax_helper = '''        #[cfg(feature = "std")]
        let get_absmax = |mat: &Matrix<i32, SEQ, DIM>, scale: f32| -> f32 {
            let mut max_val = 0;
            for i in 0..SEQ {
                for d in 0..DIM {
                    let val = mat.data[i][d].abs();
                    if val > max_val { max_val = val; }
                }
            }
            (max_val as f32) * scale
        };'''

old1 = '''        let (l0_scores, _) = self.blocks[0].forward(&mut x_mat);'''
new1 = absmax_helper + '''
        #[cfg(feature = "std")]
        eprintln!("Rust After embedding - absmax: {:.4}", get_absmax(&x_mat, self.blocks[0].s_res_in.scale));
        
        #[cfg(feature = "std")]
        eprintln!("Rust Before layer 0 - absmax: {:.4}", get_absmax(&x_mat, self.blocks[0].s_res_in.scale));
        
        let (l0_scores, _) = self.blocks[0].forward(&mut x_mat);
        
        #[cfg(feature = "std")]
        eprintln!("Rust After layer 0 - absmax: {:.4} (using s_res_out)", get_absmax(&x_mat, self.blocks[0].s_res_out.scale));
'''
assert old1 in content, "Failed old1"
content = content.replace(old1, new1, 1)

old2 = '''        for i in 1..LAYERS {
            let (scores, _) = self.blocks[i].forward(&mut x_mat);
            if i == LAYERS - 1 {
                last_scores = scores;
            }
        }'''
new2 = '''        for i in 1..LAYERS {
            #[cfg(feature = "std")]
            eprintln!("Rust Before layer {} - absmax: {:.4} (using prev s_res_out)", i, get_absmax(&x_mat, self.blocks[i-1].s_res_out.scale));
            #[cfg(feature = "std")]
            eprintln!("Rust Before layer {} - absmax: {:.4} (using my s_res_in)", i, get_absmax(&x_mat, self.blocks[i].s_res_in.scale));

            let (scores, _) = self.blocks[i].forward(&mut x_mat);
            
            #[cfg(feature = "std")]
            eprintln!("Rust After layer {} - absmax: {:.4} (using s_res_out)", i, get_absmax(&x_mat, self.blocks[i].s_res_out.scale));

            if i == LAYERS - 1 {
                last_scores = scores;
            }
        }'''
assert old2 in content, "Failed old2"
content = content.replace(old2, new2, 1)

open("src/nn/gpt.rs", "w").write(content)
print("gpt.rs patched with eprintln!")
