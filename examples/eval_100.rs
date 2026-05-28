// examples/eval_100.rs
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

    let bin_data = fs::read("artifacts/tiny_transformer_int8.bin")
        .expect("Run python scripts/export_transformer_bin.py first!");
    let model = load_transformer_from_bytes::<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>(&bin_data).unwrap();

    let py_gen_bytes = fs::read("artifacts/expected_generated_tokens_128.bin")
        .expect("Run python scripts/train_tiny_transformer.py first!");
    let py_gen: &[i32] = unsafe {
        std::slice::from_raw_parts(py_gen_bytes.as_ptr() as *const i32, py_gen_bytes.len() / 4)
    };

    let py_top3_str = fs::read_to_string("artifacts/py_top3_128.txt").unwrap_or_default();
    let mut py_top3_map = std::collections::HashMap::new();
    for line in py_top3_str.lines() {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() == 3 {
            let s: usize = parts[0].parse().unwrap();
            let indices: Vec<usize> = parts[1].split(',').map(|x| x.parse().unwrap()).collect();
            let logits: Vec<f32> = parts[2].split(',').map(|x| x.parse().unwrap()).collect();
            py_top3_map.insert(s, (indices, logits));
        }
    }

    let prompt_len = 6;
    let total_steps = 100;
    
    let mut matches = 0;
    let mut mismatches = 0;

    println!("Running teacher-forced parity evaluation for 100 tokens...");

    for step in prompt_len..(prompt_len + total_steps) {
        // PyTorch uses generated[-seq_len:]
        let start_idx = if step >= SEQ { step - SEQ } else { 0 };
        let context = &py_gen[start_idx..step];
        
        let mut caches = [
            QKVCache::<SEQ, DIM>::new(),
            QKVCache::<SEQ, DIM>::new(),
            QKVCache::<SEQ, DIM>::new(),
        ];
        let mut rust_argmax = 0;
        let mut rust_all_logits = Vec::new();
        
        for (pos, &t) in context.iter().enumerate() {
            let logits = model.forward_incremental(t as usize, pos, &mut caches);
            
            // Only capture the argmax of the final token in the sequence
            if pos == context.len() - 1 {
                let mut max_val = f32::MIN;
                let mut max_idx = 0;
                for v in 0..VOCAB {
                    let val = logits.data[0][v];
                    rust_all_logits.push((v, val));
                    if val > max_val {
                        max_val = val;
                        max_idx = v;
                    }
                }
                rust_argmax = max_idx;
            }
        }
        
        let py_target = py_gen[step] as usize;
        
        if rust_argmax == py_target {
            matches += 1;
        } else {
            mismatches += 1;
            println!("Mismatch at step {}:", step);
            rust_all_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            println!("  Rust Top 3: [{} (val={}), {} (val={}), {} (val={})]", 
                rust_all_logits[0].0, rust_all_logits[0].1,
                rust_all_logits[1].0, rust_all_logits[1].1,
                rust_all_logits[2].0, rust_all_logits[2].1
            );
            
            if let Some((py_idx, py_vals)) = py_top3_map.get(&step) {
                println!("  PyTorch Top 3: [{} (val={:.4}), {} (val={:.4}), {} (val={:.4})]", 
                    py_idx[0], py_vals[0],
                    py_idx[1], py_vals[1],
                    py_idx[2], py_vals[2]
                );
            }
            println!("");
        }
    }
    
    let accuracy = (matches as f64 / total_steps as f64) * 100.0;
    println!("\n===========================================");
    println!("Teacher-Forced 100-Token Evaluation:");
    println!("Matches: {} / {}", matches, total_steps);
    println!("Mismatches: {}", mismatches);
    println!("Accuracy: {:.2}%", accuracy);
    println!("===========================================\n");
}

#[cfg(not(feature = "std"))]
fn main() {}
