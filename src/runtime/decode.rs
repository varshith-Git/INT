// src/runtime/decode.rs

/// Pure integer argmax decoding.
pub fn argmax_logits(logits: &[i8]) -> usize {
    let mut max_val = core::i8::MIN;
    let mut max_idx = 0;
    
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    max_idx
}
