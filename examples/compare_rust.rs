#[cfg(feature = "std")]
fn main() {
    use std::fs;
    use matmul_engine::model::load_transformer_from_bytes;

    const VOCAB: usize = 61;
    const SEQ: usize = 64;
    const DIM: usize = 64;
    const HIDDEN: usize = 256;
    const HEADS: usize = 4;
    const LAYERS: usize = 3;

    let bin_data = fs::read("artifacts/tiny_transformer_int8.bin").unwrap();
    let model = load_transformer_from_bytes::<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>(&bin_data).unwrap();

    let mut tokens = [0usize; SEQ];
    let prompt_tokens = [11, 13, 30, 1, 19, 0];
    for (i, &t) in prompt_tokens.iter().enumerate() {
        tokens[i] = t;
    }
    let prompt_len = 6;
    
    // We already have the eprintln! inside forward
    let (logits, _, _, _, _) = model.forward(&tokens);
    
    let last_token_idx = prompt_len - 1;
    
    let mut logits_vec = Vec::new();
    for v in 0..VOCAB {
        logits_vec.push((v, logits.data[last_token_idx][v]));
    }
    
    logits_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("Logits Top 3:");
    for i in 0..3 {
        println!("  {}: {:.4}", logits_vec[i].0, logits_vec[i].1);
    }
}

#[cfg(not(feature = "std"))]
fn main() {}
