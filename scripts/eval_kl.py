#!/usr/bin/env python3
"""
eval_kl.py — compute richer accuracy metrics between Rust engine logits
and the PyTorch float32 reference, using full logit distributions.

Metrics reported:
  1. Top-1 accuracy (same as eval_100 baseline)
  2. Mean Reciprocal Rank (MRR) of the reference top-1 token in Rust ranking
  3. Top-3 and Top-5 accuracy
  4. Mean rank of the correct token
  5. Mean cross-entropy of Rust distribution w.r.t. reference distribution
     (proxy for KL: CE(ref, rust) = -sum(ref_prob * log(rust_prob)))
     Uses softmax over available top-3 logits as an approximation.
  6. Per-step delta: difference in log-prob of the correct token (old vs new)

Usage:
    # First build both logit dumps:
    cargo run --release --example eval_kl --features std
    # (optionally restore old artifacts, rebuild, rename output, then swap back)

    python scripts/eval_kl.py [--old artifacts/rust_logits_old.bin] [--new artifacts/rust_logits.bin]

The script works with a single dump too (just prints absolute metrics).
"""

import numpy as np
import json
import argparse
import sys

VOCAB       = 61
TOTAL_STEPS = 100
PROMPT_LEN  = 6

def load_logits(path):
    """Load a flat binary file of float32 logits: [steps, vocab]"""
    data = np.frombuffer(open(path, "rb").read(), dtype=np.float32)
    assert data.size == TOTAL_STEPS * VOCAB, \
        f"Expected {TOTAL_STEPS * VOCAB} values, got {data.size} in {path}"
    return data.reshape(TOTAL_STEPS, VOCAB)

def softmax(logits):
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()

def load_py_reference():
    """
    Load PyTorch reference top-1 tokens (ground truth targets) and top-3 logits.
    Returns:
        targets:  [100] int array — the correct token at each step
        py_top3_idx:   {step: [i0, i1, i2]}
        py_top3_logit: {step: [l0, l1, l2]}
    """
    py_gen_bytes = open("artifacts/expected_generated_tokens_128.bin", "rb").read()
    py_gen = np.frombuffer(py_gen_bytes, dtype=np.int32)

    targets = np.array([py_gen[PROMPT_LEN + s] for s in range(TOTAL_STEPS)], dtype=np.int32)

    with open("artifacts/py_top3_128.json") as f:
        py_top3_raw = json.load(f)

    py_top3_idx   = {}
    py_top3_logit = {}
    for step in range(PROMPT_LEN, PROMPT_LEN + TOTAL_STEPS):
        key = str(step)
        if key in py_top3_raw:
            py_top3_idx[step]   = py_top3_raw[key]["indices"]
            py_top3_logit[step] = py_top3_raw[key]["logits"]

    return targets, py_top3_idx, py_top3_logit

def compute_metrics(logits, targets, py_top3_idx, py_top3_logit, label=""):
    """Compute all metrics for a [100, 61] logit array."""
    n = TOTAL_STEPS

    top1_hits  = 0
    top3_hits  = 0
    top5_hits  = 0
    ranks      = []
    mrr        = 0.0
    cross_ents = []
    log_probs  = []  # log-prob of correct token under rust distribution

    for s in range(n):
        step = PROMPT_LEN + s
        logit = logits[s]                            # [61]
        target = int(targets[s])

        # Rankings
        order = np.argsort(logit)[::-1]              # descending
        rank = int(np.where(order == target)[0][0]) + 1  # 1-indexed

        if rank == 1: top1_hits += 1
        if rank <= 3: top3_hits += 1
        if rank <= 5: top5_hits += 1
        ranks.append(rank)
        mrr += 1.0 / rank

        # Log-prob of correct token
        rust_probs = softmax(logit)
        log_probs.append(float(np.log(rust_probs[target] + 1e-9)))

        # Cross-entropy against PyTorch reference (top-3 approximation)
        key = step
        if key in py_top3_idx and key in py_top3_logit:
            py_idx   = py_top3_idx[key]
            py_lgts  = np.array(py_top3_logit[key], dtype=np.float32)
            py_probs_top3 = softmax(py_lgts)          # over top-3 only
            # CE: -sum(py_prob_i * log(rust_prob_i)) for i in top3
            ce = 0.0
            for i, (idx, pp) in enumerate(zip(py_idx, py_probs_top3)):
                rp = rust_probs[idx]
                ce -= pp * np.log(rp + 1e-9)
            cross_ents.append(ce)

    mrr /= n
    mean_rank = float(np.mean(ranks))
    mean_ce   = float(np.mean(cross_ents)) if cross_ents else float('nan')
    mean_lp   = float(np.mean(log_probs))

    hdr = f"  [{label}]" if label else ""
    print(f"{hdr}")
    print(f"  Top-1 accuracy:            {top1_hits:3d}/100  = {100*top1_hits/n:.1f}%")
    print(f"  Top-3 accuracy:            {top3_hits:3d}/100  = {100*top3_hits/n:.1f}%")
    print(f"  Top-5 accuracy:            {top5_hits:3d}/100  = {100*top5_hits/n:.1f}%")
    print(f"  Mean Reciprocal Rank:       {mrr:.4f}  (higher = better)")
    print(f"  Mean rank of correct token: {mean_rank:.2f}  (lower = better)")
    print(f"  Mean log-prob(correct):     {mean_lp:.4f}  (less negative = better)")
    print(f"  Mean CE (ref top3 → rust):  {mean_ce:.4f}  (lower = better, approx)")

    return {
        "top1": top1_hits, "top3": top3_hits, "top5": top5_hits,
        "mrr": mrr, "mean_rank": mean_rank, "mean_lp": mean_lp,
        "mean_ce": mean_ce, "log_probs": log_probs, "ranks": ranks,
    }

def compare_metrics(old_m, new_m):
    """Print delta between two metric dicts."""
    print("\n─" * 30)
    print("  DELTA (new − old):")
    print(f"  Top-1:     {new_m['top1'] - old_m['top1']:+d}")
    print(f"  Top-3:     {new_m['top3'] - old_m['top3']:+d}")
    print(f"  Top-5:     {new_m['top5'] - old_m['top5']:+d}")
    print(f"  MRR:       {new_m['mrr'] - old_m['mrr']:+.4f}")
    print(f"  Mean rank: {new_m['mean_rank'] - old_m['mean_rank']:+.2f}  (negative = improvement)")
    print(f"  Mean lp:   {new_m['mean_lp'] - old_m['mean_lp']:+.4f}  (positive = improvement)")
    print(f"  Mean CE:   {new_m['mean_ce'] - old_m['mean_ce']:+.4f}  (negative = improvement)")

    # Step-level analysis: per-step log-prob change
    old_lp = np.array(old_m["log_probs"])
    new_lp = np.array(new_m["log_probs"])
    delta   = new_lp - old_lp

    improved = (delta > 0.05).sum()
    degraded = (delta < -0.05).sum()
    neutral  = len(delta) - improved - degraded

    print(f"\n  Step-level log-prob changes (threshold |Δ|>0.05):")
    print(f"    Improved: {improved} steps  (new log-prob higher by >0.05)")
    print(f"    Neutral:  {neutral} steps")
    print(f"    Degraded: {degraded} steps")

    # Top-5 most improved and degraded steps
    order = np.argsort(delta)
    print(f"\n  Top-5 most degraded steps: {[PROMPT_LEN + i for i in order[:5].tolist()]}")
    print(f"  Top-5 most improved steps: {[PROMPT_LEN + i for i in order[-5:][::-1].tolist()]}")

    # Statistical test: one-sample Wilcoxon signed-rank (symmetric null)
    # Simple: report mean and std of delta
    print(f"\n  Log-prob delta: mean={delta.mean():+.4f}  std={delta.std():.4f}  "
          f"(SE≈{delta.std()/np.sqrt(len(delta)):.4f})")
    t_stat = delta.mean() / (delta.std() / np.sqrt(len(delta)))
    print(f"  t-statistic = {t_stat:.2f}  "
          f"(|t|>2 → p<0.05 approx; this is a continuous metric, not top-1 noise)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", default="artifacts/rust_logits.bin",
                        help="Path to new (current) rust logit dump")
    parser.add_argument("--old", default=None,
                        help="Path to old rust logit dump (optional, enables delta)")
    args = parser.parse_args()

    print("=" * 60)
    print("  eval_kl.py — full-distribution accuracy metrics")
    print("=" * 60)

    targets, py_top3_idx, py_top3_logit = load_py_reference()

    print(f"\nLoading: {args.new}")
    new_logits = load_logits(args.new)
    print()
    new_m = compute_metrics(new_logits, targets, py_top3_idx, py_top3_logit, label=args.new)

    if args.old:
        print(f"\nLoading: {args.old}")
        old_logits = load_logits(args.old)
        print()
        old_m = compute_metrics(old_logits, targets, py_top3_idx, py_top3_logit, label=args.old)
        compare_metrics(old_m, new_m)

    print("\nDone.")

if __name__ == "__main__":
    main()
