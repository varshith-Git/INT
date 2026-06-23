// src/wasm_api.rs
//
// WebAssembly bindings for TinyGPT-INT8.
// Build: wasm-pack build --target web --features wasm
//
// Exposes to JavaScript:
//   init_model(bytes: Uint8Array) -> bool
//   predict(token_ids: Uint8Array) -> Float32Array  (61 logits)

extern crate alloc;
use alloc::vec::Vec;
use alloc::boxed::Box;

use wasm_bindgen::prelude::*;
use crate::model::load_transformer_from_bytes;
use crate::runtime::kv_cache::QKVCache;

// Compile-time model constants — must match export_transformer_bin.py
const VOCAB:  usize = 61;
const SEQ:    usize = 64;
const DIM:    usize = 512;
const HIDDEN: usize = 2048;
const HEADS:  usize = 8;
const LAYERS: usize = 12;

type GPTModel = crate::nn::gpt::QGPTModel<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>;

// Store the model as a raw pointer.
// WASM is single-threaded, so this is safe — there are no concurrent accesses.
// We use a static mut *const to avoid the unsafe block restriction by gating
// this module behind #[cfg(feature = "wasm")] which is explicitly unsafe-allowed
// via the allow attribute below.
#[allow(unsafe_code)]
static mut MODEL_PTR: *mut GPTModel = core::ptr::null_mut();

/// Load the quantized model from a Uint8Array (.bin file bytes).
/// Call this once after fetching the model file via fetch().
/// Returns true on success.
#[allow(unsafe_code)]
#[wasm_bindgen]
pub fn init_model(bytes: &[u8]) -> bool {
    match load_transformer_from_bytes::<VOCAB, SEQ, DIM, HIDDEN, HEADS, LAYERS>(bytes) {
        Ok(model) => {
            let boxed = Box::new(model);
            unsafe {
                // Free any previously loaded model
                if !MODEL_PTR.is_null() {
                    drop(Box::from_raw(MODEL_PTR));
                }
                MODEL_PTR = Box::into_raw(boxed);
            }
            true
        }
        Err(_) => false,
    }
}

/// Run one forward pass. Returns logits for all 61 vocab tokens.
///
/// token_ids: a sequence of up to 64 character token IDs.
///            Feed the last N characters the user has typed (encoded as u8).
///            The engine is stateless per call — it re-builds the KV cache
///            from scratch each time (fine for short contexts, ≤64 chars).
///
/// Returns Float32Array of length 61.
#[allow(unsafe_code)]
#[wasm_bindgen]
pub fn predict(token_ids: &[u8]) -> Vec<f32> {
    let model = unsafe {
        if MODEL_PTR.is_null() {
            return alloc::vec![0.0f32; VOCAB];
        }
        &*MODEL_PTR
    };

    let len = token_ids.len().min(SEQ);
    if len == 0 {
        return alloc::vec![0.0f32; VOCAB];
    }

    let mut caches: [QKVCache<SEQ, DIM>; LAYERS] =
        core::array::from_fn(|_| QKVCache::new());

    let mut logits_out = alloc::vec![0.0f32; VOCAB];

    for (pos, &tok) in token_ids[..len].iter().enumerate() {
        let logits = model.forward_incremental(tok as usize, pos, &mut caches);
        if pos == len - 1 {
            for v in 0..VOCAB {
                logits_out[v] = logits.data[0][v];
            }
        }
    }

    logits_out
}

/// Model vocab size — always 61 for this build.
#[wasm_bindgen]
pub fn vocab_size() -> u32 {
    VOCAB as u32
}
