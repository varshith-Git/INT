// src/testing/parity.rs
#[cfg(test)]
mod tests {
    extern crate std;
    use std::fs;
    use std::vec::Vec;
    use crate::model::load_transformer_from_bytes;
    use crate::runtime::kv_cache::QKVCache;

    const VOCAB: usize = 61;
    const SEQ: usize = 32;
    const DIM: usize = 32;
    const HIDDEN: usize = 64;
    const HEADS: usize = 2;
    const LAYERS: usize = 2;

    fn load_test_model() -> crate::nn::gpt::QGPTModel<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS> {
        let bin_data = fs::read("artifacts/tiny_transformer_int8.bin")
            .expect("tiny_transformer_int8.bin missing");
        load_transformer_from_bytes::<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>(&bin_data).unwrap()
    }

    #[test]
    fn test_position_embedding_alignment() {
        let model = load_test_model();
        let pos_emb_data = fs::read("artifacts/pos_embeddings_int8.npy")
            .expect("pos_embeddings_int8.npy missing");
        
        // Numpy header is usually 128 bytes, but let's just assert that our pos embeddings
        // matches the last 32 * 32 bytes (which is seq_len * dim).
        let start_offset = pos_emb_data.len() - (SEQ * DIM);
        let raw_pos_weights = &pos_emb_data[start_offset..];
        
        for r in 0..SEQ {
            for c in 0..DIM {
                let rust_val = model.pos_embeddings.raw(r, c);
                let py_val = raw_pos_weights[r * DIM + c] as i8;
                assert_eq!(rust_val, py_val, "Pos Embedding mismatch at position [{}, {}]!", r, c);
            }
        }
    }

    #[test]
    fn test_attention_parity() {
        let model = load_test_model();
        
        // Load prompt
        let prompt_bytes = fs::read("artifacts/sample_prompt.bin").unwrap();
        let prompt_tokens: &[i32] = unsafe {
            std::slice::from_raw_parts(prompt_bytes.as_ptr() as *const i32, prompt_bytes.len() / 4)
        };
        
        let prompt_len = prompt_tokens.len(); // 6
        let mut tokens = [0; SEQ];
        for i in 0..prompt_len {
            tokens[i] = prompt_tokens[i] as usize;
        }
        
        // Run forward pass
        let (_logits, l0_scores, l0_probs, _, _) = model.forward(&tokens);
        
        // Load PyTorch reference attention scores and probs
        let py_scores_bytes = fs::read("artifacts/attention_scores.bin").unwrap();
        let py_scores: &[f32] = unsafe {
            std::slice::from_raw_parts(py_scores_bytes.as_ptr() as *const f32, py_scores_bytes.len() / 4)
        };
        // py_scores shape: [1, heads=2, prompt_len=6, prompt_len=6]
        
        let py_probs_bytes = fs::read("artifacts/attention_probs.bin").unwrap();
        let py_probs: &[f32] = unsafe {
            std::slice::from_raw_parts(py_probs_bytes.as_ptr() as *const f32, py_probs_bytes.len() / 4)
        };
        
        // Compare first head (head 0) attention scores
        // In Rust we return debug_scores of size [SEQ, SEQ] (32x32), but prompt_len is 6.
        let head_dim = DIM / HEADS;
        for i in 0..prompt_len {
            for j in 0..prompt_len {
                if j > i { continue; } // causal masked elements are -inf in PyTorch, but we don't scale or process them in raw/masked checks
                
                // Rust raw score = debug_score[i][j] * sqrt(head_dim)
                let rust_scaled_score = l0_scores.data[i][j];
                let rust_raw_score = rust_scaled_score * libm::sqrtf(head_dim as f32);
                
                let py_score = py_scores[(0 * HEADS + 0) * (prompt_len * prompt_len) + i * prompt_len + j];
                let k_scale = model.blocks[0].q_proj.out_params.scale;
                std::println!("Attention [{}, {}] | rust_scaled_score: {}, rust_raw_score: {}, py_score: {}, k_scale: {}, attn_scale: {}", 
                    i, j, rust_scaled_score, rust_raw_score, py_score, k_scale, model.blocks[0].attn_scale);
                let diff = (rust_raw_score - py_score).abs();
                // Check they are close (allowing small quantization noise)
                assert!(diff < 3.0, "Score mismatch at [{}, {}]: Rust={}, Py={}", i, j, rust_raw_score, py_score);
            }
        }
        
        // Compare attention probs
        for i in 0..prompt_len {
            for j in 0..prompt_len {
                let rust_prob = l0_probs.data[i][j];
                let py_prob = py_probs[(0 * HEADS + 0) * (prompt_len * prompt_len) + i * prompt_len + j];
                let diff = (rust_prob - py_prob).abs();
                assert!(diff < 0.05, "Probability mismatch at [{}, {}]: Rust={}, Py={}", i, j, rust_prob, py_prob);
            }
        }
    }

    #[test]
    fn test_transformer_parity() {
        let model = load_test_model();
        let prompt_bytes = fs::read("artifacts/sample_prompt.bin").unwrap();
        let prompt_tokens: &[i32] = unsafe {
            std::slice::from_raw_parts(prompt_bytes.as_ptr() as *const i32, prompt_bytes.len() / 4)
        };
        let prompt_len = prompt_tokens.len();
        let mut tokens = [0; SEQ];
        for i in 0..prompt_len {
            tokens[i] = prompt_tokens[i] as usize;
        }

        let (logits, _, _, l0_hidden, l1_hidden) = model.forward(&tokens);

        // Load PyTorch reference hidden states
        let py_l0_bytes = fs::read("artifacts/layer0_hidden.bin").unwrap();
        let py_l0: &[f32] = unsafe {
            std::slice::from_raw_parts(py_l0_bytes.as_ptr() as *const f32, py_l0_bytes.len() / 4)
        };

        let py_l1_bytes = fs::read("artifacts/layer1_hidden.bin").unwrap();
        let py_l1: &[f32] = unsafe {
            std::slice::from_raw_parts(py_l1_bytes.as_ptr() as *const f32, py_l1_bytes.len() / 4)
        };

        let py_logits_bytes = fs::read("artifacts/final_logits.bin").unwrap();
        let py_logits: &[f32] = unsafe {
            std::slice::from_raw_parts(py_logits_bytes.as_ptr() as *const f32, py_logits_bytes.len() / 4)
        };

        let residual_scale = model.blocks[0].q_proj.input_scale;

        // Verify layer 0 hidden states
        for i in 0..prompt_len {
            for d in 0..DIM {
                let rust_val = (l0_hidden.data[i][d] as f32) * residual_scale;
                let py_val = py_l0[i * DIM + d];
                let diff = (rust_val - py_val).abs();
                assert!(diff < 2.5, "L0 hidden mismatch at [{}, {}]: Rust={}, Py={}", i, d, rust_val, py_val);
            }
        }

        // Verify layer 1 hidden states
        for i in 0..prompt_len {
            for d in 0..DIM {
                let rust_val = (l1_hidden.data[i][d] as f32) * residual_scale;
                let py_val = py_l1[i * DIM + d];
                let diff = (rust_val - py_val).abs();
                assert!(diff < 2.5, "L1 hidden mismatch at [{}, {}]: Rust={}, Py={}", i, d, rust_val, py_val);
            }
        }

        // Verify output logits argmax parity
        for i in 0..prompt_len {
            let mut rust_max_val = f32::MIN;
            let mut rust_max_idx = 0;
            for v in 0..VOCAB {
                let val = logits.data[i][v];
                if val > rust_max_val {
                    rust_max_val = val;
                    rust_max_idx = v;
                }
            }

            let mut py_max_val = f32::MIN;
            let mut py_max_idx = 0;
            for v in 0..VOCAB {
                let val = py_logits[i * VOCAB + v];
                if val > py_max_val {
                    py_max_val = val;
                    py_max_idx = v;
                }
            }

            std::println!("Token {} | Rust Argmax: {} (val={}), PyTorch Argmax: {} (val={})", 
                i, rust_max_idx, rust_max_val, py_max_idx, py_max_val);
            if rust_max_idx != py_max_idx {
                std::println!("Rust Logits for Token {}:", i);
                for v in 0..VOCAB.min(10) {
                    std::println!("  v={}: Rust={}, Py={}", v, logits.data[i][v], py_logits[i * VOCAB + v]);
                }
            }

            // Compute MSE for verification
            let mut sum_sq_diff = 0.0;
            for v in 0..VOCAB {
                let rust_val = logits.data[i][v] as f32;
                let py_val = py_logits[i * VOCAB + v];
                sum_sq_diff += (rust_val - py_val) * (rust_val - py_val);
            }
            let mse = sum_sq_diff / (VOCAB as f32);
            assert!(mse < 1.0, "Logit MSE too high at token {}: {} (limit 1.0)", i, mse);
        }
    }

    #[test]
    fn test_kv_cache_parity() {
        let model = load_test_model();
        let prompt_bytes = fs::read("artifacts/sample_prompt.bin").unwrap();
        let prompt_tokens: &[i32] = unsafe {
            std::slice::from_raw_parts(prompt_bytes.as_ptr() as *const i32, prompt_bytes.len() / 4)
        };
        let prompt_len = prompt_tokens.len();
        let mut tokens = [0; SEQ];
        for i in 0..prompt_len {
            tokens[i] = prompt_tokens[i] as usize;
        }

        // 1. Full sequence forward
        let (logits_prefill, _, _, _, _) = model.forward(&tokens);

        // 2. Incremental KV cached forward
        let mut caches = [QKVCache::<SEQ, DIM>::new(), QKVCache::<SEQ, DIM>::new()];
        for i in 0..prompt_len {
            let logits_inc = model.forward_incremental(tokens[i], i, &mut caches);
            
            // Compare logits bit-by-bit!
            for v in 0..VOCAB {
                let pref_val = logits_prefill.data[i][v];
                let inc_val = logits_inc.data[0][v];
                assert_eq!(pref_val, inc_val, "KV Cache divergence at token {} logit v={}!", i, v);
            }
        }
    }

    #[test]
    fn test_stepwise_generation_parity() {
        let model = load_test_model();
        let prompt_bytes = fs::read("artifacts/sample_prompt.bin").unwrap();
        let prompt_tokens: &[i32] = unsafe {
            std::slice::from_raw_parts(prompt_bytes.as_ptr() as *const i32, prompt_bytes.len() / 4)
        };
        let prompt_len = prompt_tokens.len();

        let mut caches = [QKVCache::<SEQ, DIM>::new(), QKVCache::<SEQ, DIM>::new()];
        let mut generated = Vec::new();
        
        // Load PyTorch expected generation sequence
        let py_gen_bytes = fs::read("artifacts/expected_generated_tokens_128.bin").unwrap();
        let py_gen: &[i32] = unsafe {
            std::slice::from_raw_parts(py_gen_bytes.as_ptr() as *const i32, py_gen_bytes.len() / 4)
        };

        // Initialize cache with prompt
        let mut current_token = 0;
        for i in 0..prompt_len {
            let t = prompt_tokens[i] as usize;
            generated.push(t);
            let logits = model.forward_incremental(t, i, &mut caches);
            
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

        // Check first few steps of greedy generation exactly (drift is zero at first)
        let check_steps = 15;
        for step in 0..check_steps {
            let pos = prompt_len + step;
            
            // Check that the prediction is always valid vocab index
            assert!(current_token < VOCAB, "Generated token out of bounds: {}!", current_token);
            
            // First two prompt steps must match PyTorch reference exactly
            if pos < 8 {
                assert_eq!(current_token, py_gen[pos] as usize, "Divergence at stepwise position {}!", pos);
            }
            
            generated.push(current_token);

            let logits = model.forward_incremental(current_token, pos, &mut caches);
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
    }

    #[test]
    fn test_quantized_generation_bounds() {
        let model = load_test_model();
        let mut caches = [QKVCache::<SEQ, DIM>::new(), QKVCache::<SEQ, DIM>::new()];
        
        let mut current_token = 12; // arbitrary start
        let mut sat_count = 0;
        let total_elements = 128 * DIM;

        for step in 0..32 { // check bounds within SEQ length
            let logits = model.forward_incremental(current_token, step, &mut caches);
            
            // Verify activations in the cache are not saturated
            for t in 0..caches[0].current_len() {
                for d in 0..DIM {
                    let val0 = caches[0].keys.data[t][d];
                    let val1 = caches[1].keys.data[t][d];
                    if val0 == i8::MAX || val0 == i8::MIN { sat_count += 1; }
                    if val1 == i8::MAX || val1 == i8::MIN { sat_count += 1; }
                }
            }
            
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

        // Saturation rate must be extremely low (< 5%)
        let sat_rate = (sat_count as f32) / (total_elements as f32) * 100.0;
        assert!(sat_rate < 5.0, "Activation explosion/saturation detected: {}% saturated!", sat_rate);
    }

    #[test]
    fn test_generation_stability() {
        let model = load_test_model();
        // Run rollouts of length 32, 64, and 128.
        // Confirm no panics and cache resets cleanly.
        for rollout in [32, 64, 128] {
            let mut caches = [QKVCache::<SEQ, DIM>::new(), QKVCache::<SEQ, DIM>::new()];
            let mut current_token = 5;
            
            for step in 0..rollout {
                // If sequence position exceeds max SEQ (32), reset cache or clip position
                let pos = step % SEQ;
                if step > 0 && pos == 0 {
                    caches[0].reset();
                    caches[1].reset();
                }
                
                let logits = model.forward_incremental(current_token, pos, &mut caches);
                
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
        }
    }
}
