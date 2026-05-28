// examples/tiny_transformer.rs
fn main() {
    #[cfg(feature = "std")]
    matmul_engine::examples::tiny_transformer::run_generation_example();
    
    #[cfg(not(feature = "std"))]
    println!("Please run with std feature enabled: cargo run --example tiny_transformer --features std");
}
