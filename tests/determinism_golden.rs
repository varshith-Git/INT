//! Cross-architecture determinism golden test.
//!
//! The engine's headline guarantee is that the *same* code produces *byte-
//! identical* output on every architecture (x86_64, aarch64, riscv64, wasm32).
//! The existing unit tests only prove *same-run* equality on one machine. This
//! test pins a **golden fingerprint** over the deterministic primitives and
//! asserts the live fingerprint equals it. CI (`.github/workflows/determinism.yml`)
//! runs this exact test on multiple architectures — if any arch computes a
//! different fingerprint, this assertion fails and the moat is proven broken.
//!
//! Why each primitive is here:
//!  * `matmul_wide` (i8 -> i32): pure integer — must be identical everywhere.
//!  * `DetRng` u64 stream: pure integer — must be identical everywhere.
//!  * sampler float path (`expf` via **libm**, not hardware/`std`): the only
//!    floating-point surface. It is deterministic *because* the crate uses a
//!    software float impl (`libm`, `default-features = false`) instead of
//!    hardware FMA-contractable math — this test is what verifies that claim
//!    holds across ISAs.
//!
//! Run locally:   cargo test --test determinism_golden
//! Regenerate:    flip PRINT_ONLY to true, run with `-- --nocapture`, paste the
//!                printed value into GOLDEN, flip it back.

use matmul_engine::runtime::sampler::{sample, DetRng, SamplingConfig};
use matmul_engine::{matmul_wide, Matrix};

/// Set true to print the computed fingerprint instead of asserting (for refresh).
const PRINT_ONLY: bool = false;

/// Golden fingerprint over all primitives below. Identical on every conforming
/// architecture. Captured on aarch64 (Apple Silicon) 2026-06-22.
const GOLDEN: u64 = 0x0904_5538_e992_6ea4;

// ── Dependency-free FNV-1a (64-bit) so the fingerprint needs no extra crates ──
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(FNV_OFFSET)
    }
    fn eat(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }
}

fn compute_fingerprint() -> u64 {
    let mut h = Fnv::new();

    // 1. Wide integer matmul over fixed inputs (8x6 · 6x4 -> 8x4, i8 -> i32).
    let a = Matrix::from_array([
        [3i8, -7, 12, 0, -1, 5],
        [-9, 4, 8, -2, 11, -6],
        [1, 1, -1, -1, 2, -2],
        [127, -128, 64, -64, 32, -32],
        [10, 20, -30, 40, -50, 60],
        [-5, -4, -3, -2, -1, 0],
        [6, 7, 8, 9, 10, 11],
        [-12, 13, -14, 15, -16, 17],
    ]);
    let b = Matrix::from_array([
        [2i8, -3, 4, -5],
        [6, -7, 8, -9],
        [-10, 11, -12, 13],
        [14, -15, 16, -17],
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
    ]);
    let c: Matrix<i32, 8, 4> = matmul_wide(&a, &b);
    for row in &c.data {
        for v in row {
            h.eat(&v.to_le_bytes());
        }
    }

    // 2. Pure-integer RNG stream.
    let mut rng = DetRng::new(0x0BAD_F00D_DEAD_BEEF);
    for _ in 0..512 {
        h.eat(&rng.next_u64().to_le_bytes());
    }

    // 3. RNG -> f32 conversion stream (integer -> single IEEE mul; arch-stable).
    let mut rng2 = DetRng::new(0x1234_5678_9ABC_DEF0);
    for _ in 0..256 {
        h.eat(&rng2.next_f32().to_bits().to_le_bytes());
    }

    // 4. Sampler float path: softmax(expf via libm) + top-p + inverse-CDF draw,
    //    over fixed logits across many seeds. This is the real cross-ISA stress.
    let logits: [i8; 16] = [12, -3, 40, 7, -20, 5, 33, 1, -8, 19, 27, -1, 6, 14, -11, 9];
    for seed in 0..128u64 {
        let cfg = SamplingConfig::top_p(0.9, 0.95, seed);
        let idx = sample(&logits, &cfg, &mut DetRng::new(seed));
        h.eat(&(idx as u64).to_le_bytes());
        // also exercise plain temperature + top-k paths
        let idx2 = sample(&logits, &SamplingConfig::top_k(0.8, 4, seed), &mut DetRng::new(seed ^ 0xFF));
        h.eat(&(idx2 as u64).to_le_bytes());
    }

    h.0
}

#[test]
fn determinism_golden() {
    let fp = compute_fingerprint();
    if PRINT_ONLY {
        eprintln!("determinism fingerprint = {fp:#018x}");
        return;
    }
    assert_eq!(
        fp, GOLDEN,
        "\nDETERMINISM BROKEN: this architecture computed {fp:#018x}, \
         expected {GOLDEN:#018x}.\nThe same code must produce the same bytes on \
         every ISA — investigate before shipping."
    );
}

/// Sanity: the fingerprint is stable within a single run (cheap regression guard).
#[test]
fn fingerprint_is_stable_in_process() {
    assert_eq!(compute_fingerprint(), compute_fingerprint());
}
