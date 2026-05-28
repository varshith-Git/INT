// examples/debug_layer.rs
#[cfg(feature = "std")]
fn main() {
    use matmul_engine::model::load_transformer_from_bytes;
    use matmul_engine::runtime::kv_cache::QKVCache;
    use matmul_engine::tensor::QTensor;
    use std::fs;

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
    
    // "ACT I\n" = 6 tokens
    let tokens = &py_gen[0..6];

    let mut cache = QKVCache::<SEQ, DIM, HEADS, LAYERS>::new();

    println!("=== EMBEDDING OUTPUT (first token position) ===");
    let emb = model.embed(tokens);
    let mut absmax = 0.0;
    let mut sum = 0.0;
    for d in 0..DIM {
        let val = (emb.raw(0, d) as f32) * emb.scale();
        if val.abs() > absmax { absmax = val.abs(); }
        sum += val;
    }
    println!("  shape: [1, 6, 64]");
    println!("  absmax: {:.4}", absmax);
    println!("  mean:   {:.4}", sum / (DIM as f32 * 6.0)); // simple mean proxy
    
    let mut v: Vec<(usize, f32)> = (0..DIM).map(|d| (d, (emb.raw(0, d) as f32) * emb.scale())).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    print!("  top5 values at pos 0: [");
    for i in 0..5 { print!("{:.4}{}", v[i].1, if i==4 {""} else {", "}); }
    println!("]");

    let mut x_mat = emb;
    
    for l in 0..LAYERS {
        println!("\n=== LAYER {} OUTPUT ===", l);
        x_mat = model.blocks[l].forward_incremental(&x_mat, &cache);
        
        let mut absmax = 0.0;
        let mut sum = 0.0;
        for d in 0..DIM {
            let val = (x_mat.raw(0, d) as f32) * x_mat.scale();
            if val.abs() > absmax { absmax = val.abs(); }
            sum += val;
        }
        println!("  absmax: {:.4}", absmax);
        println!("  mean:   {:.4}", sum / (DIM as f32)); // mean of last pos
        
        let mut v: Vec<(usize, f32)> = (0..DIM).map(|d| (d, (x_mat.raw(0, d) as f32) * x_mat.scale())).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        print!("  top5 at pos -1: [");
        for i in 0..5 { print!("{:.4}{}", v[i].1, if i==4 {""} else {", "}); }
        println!("]");
        print!("  top5 idx at pos -1: [");
        for i in 0..5 { print!("{}{}", v[i].0, if i==4 {""} else {", "}); }
        println!("]");
    }
    
    println!("\n=== FINAL LOGITS ===");
    let logits = model.forward(&x_mat);
    let mut absmax = 0.0;
    for i in 0..VOCAB {
        if logits[i].abs() > absmax { absmax = logits[i].abs(); }
    }
    println!("  absmax: {:.4}", absmax);
    
    let mut v: Vec<(usize, f32)> = (0..VOCAB).map(|i| (i, logits[i])).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    print!("  top5 at pos -1: [");
    for i in 0..5 { print!("{:.4}{}", v[i].1, if i==4 {""} else {", "}); }
    println!("]");
    print!("  top5 idx at pos -1: [");
    for i in 0..5 { print!("{}{}", v[i].0, if i==4 {""} else {", "}); }
    println!("]");
}
