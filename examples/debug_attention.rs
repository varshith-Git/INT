// examples/debug_attention.rs
#[cfg(feature = "std")]
fn main() {
    use std::fs;
    use matmul_engine::model::load_transformer_from_bytes;
    use matmul_engine::runtime::kv_cache::QKVCache;

    const VOCAB: usize = 61;
    const SEQ: usize = 64;
    const DIM: usize = 64;
    const HIDDEN: usize = 256;
    const HEADS: usize = 4;
    const LAYERS: usize = 3;

    let bin_data = fs::read("artifacts/coherent_transformer_int8.bin").unwrap();
    let model = load_transformer_from_bytes::<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>(&bin_data).unwrap();

    // "ACT I\n" mapped using vocab.json
    let prompt_tokens = vec![11, 13, 30, 1, 19, 0];
    
    let mut caches = [
        QKVCache::<SEQ, DIM>::new(),
        QKVCache::<SEQ, DIM>::new(),
        QKVCache::<SEQ, DIM>::new(),
    ];
    
    println!("--- Processing Prompt ---");
    let mut current_token = 0;
    for (i, &t) in prompt_tokens.iter().enumerate() {
        let logits = model.forward_incremental(t, i, &mut caches);
        
        let mut max_val = f32::MIN;
        let mut max_idx = 0;
        for v in 0..VOCAB {
            if logits.data[0][v] > max_val {
                max_val = logits.data[0][v];
                max_idx = v;
            }
        }
        current_token = max_idx;
    }
    
    println!("--- Generation Steps ---");
    let mut current_pos = prompt_tokens.len();
    for step in 1..=5 {
        println!("*** Generating Step {} (pos {}) ***", step, current_pos);
        let logits = model.forward_incremental(current_token, current_pos, &mut caches);
        
        let mut max_val = f32::MIN;
        let mut max_idx = 0;
        for v in 0..VOCAB {
            if logits.data[0][v] > max_val {
                max_val = logits.data[0][v];
                max_idx = v;
            }
        }
        current_token = max_idx;
        current_pos += 1;
    }
}

#[cfg(not(feature = "std"))]
fn main() {}
