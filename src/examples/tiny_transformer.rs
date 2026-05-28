// src/examples/tiny_transformer.rs
#![cfg(feature = "std")]

use std::fs;
use std::io::Write;
use std::vec::Vec;
use std::{print, println};
use crate::model::load_transformer_from_bytes;
use crate::runtime::kv_cache::QKVCache;

pub fn run_generation_example() {
    let bin_data = fs::read("artifacts/coherent_transformer_int8.bin")
        .expect("Run python scripts/export_transformer_bin.py first!");
        
    let vocab_str = fs::read_to_string("artifacts/vocab.txt")
        .expect("Run python scripts/prepare_text.py first!");
        
    let chars: Vec<char> = vocab_str.chars().collect();
    let vocab_size = chars.len();
    
    println!("Loaded vocabulary of size {} characters.", vocab_size);
    
    // Constant parameters matching training architecture
    const VOCAB: usize = 61;
    const SEQ: usize = 64;
    const DIM: usize = 64;
    const HIDDEN: usize = 256;
    const HEADS: usize = 4;
    const LAYERS: usize = 3;
    
    assert_eq!(vocab_size, VOCAB, "Vocab mismatch!");
    
    println!("Loading quantized GPT-style model...");
    let model = load_transformer_from_bytes::<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>(&bin_data)
        .expect("Failed to load quantized transformer model");
        
    println!("Model successfully loaded! Initializing generation...");
    
    // Load sample prompt tokens
    // We will parse it from artifacts/sample_prompt.npy or just use a custom string.
    // Let's decode prompt "ACT " or "KING "
    let prompt_str = "ACT I\n";
    println!("Prompt: {:?}", prompt_str);
    
    let mut prompt_tokens = Vec::new();
    for c in prompt_str.chars() {
        if let Some(idx) = chars.iter().position(|&x| x == c) {
            prompt_tokens.push(idx);
        } else {
            prompt_tokens.push(0); // fallback
        }
    }
    
    let mut caches = [
        QKVCache::<SEQ, DIM>::new(),
        QKVCache::<SEQ, DIM>::new(),
        QKVCache::<SEQ, DIM>::new(),
    ];
    
    // Step 1: Prefill prompt tokens
    // To do this, we can run forward_incremental step-by-step for the prompt tokens,
    // populating the KV cache.
    let mut current_token = 0;
    print!("Generated: ");
    for (i, &token) in prompt_tokens.iter().enumerate() {
        let print_char = chars[token];
        print!("{}", print_char);
        
        let logits = model.forward_incremental(token, i, &mut caches);
        
        // Next token is greedy argmax
        let mut max_val = f32::MIN;
        let mut max_idx = 0;
        for v in 0..VOCAB {
            let val = logits.data[0][v];
            if val > max_val {
                max_val = val;
                max_idx = v;
            }
        }
        current_token = max_idx;
    }
    
    // Step 2: Autoregressive Decode Loop (up to 128 tokens total)
    let total_to_generate = 128 - prompt_tokens.len();
    let mut start_pos = prompt_tokens.len();
    
    for _step in 0..total_to_generate {
        if start_pos >= SEQ {
            println!("\nReached maximum sequence length SEQ={}", SEQ);
            break;
        }
        
        // Print character
        let gen_char = chars[current_token];
        print!("{}", gen_char);
        Write::flush(&mut std::io::stdout()).unwrap();
        
        // Run single forward step
        let logits = model.forward_incremental(current_token, start_pos, &mut caches);
        
        // Compute entropy of the logits as a diagnostic metric
        // Note: For attention entropy tracking per instructions, we track it on probs.
        // Let's compute prediction argmax
        let mut max_val = f32::MIN;
        let mut max_idx = 0;
        for v in 0..VOCAB {
            let val = logits.data[0][v];
            if val > max_val {
                max_val = val;
                max_idx = v;
            }
        }
        
        current_token = max_idx;
        start_pos += 1;
    }
    println!("\nGeneration complete.");
}
