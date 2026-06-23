# INT — Go-to-Market: beachhead, pilot, expansion

This is the business plan for steps 2–4 of the sequence. Step 1 (run a real model
via QAT) is the technical precondition — see `scripts/train_tiny_transformer_qat.py`
and its runbook. **Nothing below is sellable until Step 1 lands a model that is
"real enough" for a demo** (a small but genuinely coherent model, not the 41% toy).

---

## Step 2 — Beachhead vertical: **Defense / Aerospace edge** (recommended)

The choice was between **medical devices** and **defense/aerospace**. Recommend
**defense/aerospace edge first**, with medical as the expansion market. Concrete
niche: **on-board inference for space (satellites, probes) and defense autonomous
sensing (UAVs, munitions, unattended ground sensors)** — places with no MMU, no
heap to spare, no cloud link, and a hard "must not crash / must be auditable"
requirement.

**Why defense/aerospace is the sharper first wedge:**

| Factor | Defense / Aerospace edge | Medical devices |
|---|---|---|
| Tolerates early-stage tech (low TRL) | **Yes** — funds it via research contracts | No — needs mature, validated tech |
| Values determinism + bounded memory + no-OOM | **Acute, non-negotiable** | High, but software-of-unknown-provenance rules dominate |
| Values "prove what the model did" (receipts) | **Acute** (assurance, accountability) | High (audit trail) |
| Needs consumer-grade accuracy to start | **No** — narrow tasks, classification, anomaly | Often yes (clinical claims) |
| Procurement path for immature product | **SBIR/STTR, AFWERX/SpaceWERX, DARPA, In-Q-Tel, prime innovation arms** | FDA pathway = years |
| Time-to-first-dollar | Months (research/pilot contract) | Years (clearance) |

The one-line reason: **defense/aerospace will *pay you to raise the TRL*; medical
makes you raise it first, then sell.** For a product that is differentiated on
assurance but immature on capability, you want the buyer who funds maturation.

**Why this niche specifically fits INT's *current* moat (true today, pre-accuracy):**
- Deterministic, bit-identical output across radiation-hardened / mixed silicon →
  reproducible behavior is a certification asset.
- Fixed, pre-measured memory; cannot OOM or fragment → the `forbid(unsafe_code)` +
  panic-free runtime maps directly to DO-178C / space-grade "no undefined behavior."
- Proof-carrying inference (receipt per output) → accountability for autonomous
  decisions, the exact thing defense autonomy programs are blocked on.

**Disqualifier to be honest about:** ITAR/export control and clearances add
friction. Start with **dual-use / commercial-space** customers (smallsat operators,
commercial UAV) to avoid the clearance wall on day one, then move to defense primes.

> If you'd rather lead with medical: the pilot structure below ports, but swap the
> procurement path to a device OEM's advanced-R&D group and the success metric to
> "deterministic + auditable inference inside an IEC 62304 software unit."

---

## Step 3 — Land one design-partner / pilot

**What you are selling in the pilot (not the toy):** a *capability*, framed as —
> "Deterministic, crash-proof, on-device inference that fits in your existing MCU
> budget and emits a cryptographic receipt for every output. Verifiable by
> recomputation on the ground."

### Target partner archetypes (pick 1, get 1 champion inside it)
1. **Commercial smallsat operator / payload builder** — wants on-board inference
   (image triage, anomaly detection) without downlinking raw data. Pain: power,
   memory, "can we trust/audit the on-board call?"
2. **Defense autonomy group at a prime's innovation arm** (or a well-funded
   autonomy startup) — blocked on *explainable/auditable* edge decisions.
3. **Dual-use UAV / unattended-sensor vendor** — needs tiny, deterministic
   classification at the edge, intermittent or no connectivity.

### The 90-day pilot — concrete scope
- **You provide:** INT + `qmatmul` running *their* one narrow model (a small
  classifier or detector — not an LLM) on *their* target MCU class, producing a
  per-inference receipt, plus the Valori-side verifier that re-derives and checks
  receipts on the ground.
- **They provide:** one real task + dataset, the target hardware (or its specs),
  and one engineer as champion.
- **Deliverable:** a demo on real hardware + a short report: footprint (KB),
  determinism proof (byte-identical output across two machines), and an audit
  trail of receipts verified against re-computation.

### Success metrics (agree these up front)
- Runs within their memory/power budget on the target class (hard number).
- **Zero** runtime panics / allocations on the hot path (show the `forbid` guarantee).
- 100% of outputs reproduced bit-for-bit by the ground verifier.
- Accuracy within an agreed delta of their float baseline (this is where Step 1 /
  QAT must already be good enough — set the delta honestly).

### Outreach angle (how to open the door)
- Lead with the **assurance story**, not accuracy: "verifiable + crash-proof +
  tiny." That is what they can't buy elsewhere; accuracy they can get from anyone.
- Entry vehicles: an **SBIR/STTR topic** that matches (autonomy assurance, edge
  AI, space compute), **SpaceWERX/AFWERX open calls**, or a warm intro to a
  prime's innovation/IRAD group. In-Q-Tel for the intelligence-adjacent angle.
- Proof asset to show first: the `proof-carrying-inference` demo + the cross-arch
  determinism result (build that CI — it's the cheapest credibility you can buy).

---

## Step 4 — Expand using the Valori bundle

The expansion is **land on compute, expand to the full audited stack.**

1. **Land:** INT verifiable inference in one edge program (Step 3).
2. **Attach Valori:** the receipts have to live somewhere tamper-evident. Valori
   *is* that store (BLAKE3 audit chain, deterministic memory). Sell the pair:
   > "Verifiable compute (INT) + verifiable memory (Valori) = the only AI stack
   > you can audit end-to-end, from the satellite to the ground."
   No competitor offers both halves.
3. **Expand within the account:** from one payload → fleet-wide audited inference;
   from edge → the ground-station verification + long-term audit archive (Valori
   cluster, S3 offload, DR — all already built).
4. **Cross the chasm to adjacent regulated markets** once the assurance story is
   proven in defense/aerospace: **medical devices** (auditable on-device
   inference), **industrial/critical infrastructure**, **finance** (provable model
   decisions). Each reuses the same two-halves pitch with a different compliance
   wrapper (DO-178C → IEC 62304 → SR 11-7 / model-risk).

**Business model:** open-source `qmatmul` (credibility + adoption); commercial
license for INT in regulated/embedded products; subscription for the Valori
verification + audit-archive layer. Land with services (the pilot), expand to
product + subscription.

---

## Reality check — the gap to honor

- **The product is the guarantee, not the model.** Sell determinism, bounded
  memory, and receipts — all true *today*. Do not sell accuracy yet.
- **Step 1 is the gate.** Until QAT (or a loaded real small model) clears an
  honest accuracy bar on a narrow task, every pilot conversation stalls at "show
  me it actually works." Build that first.
- **Cheapest credibility next:** cross-arch determinism CI (proves the moat) and a
  hardware footprint table. Both are small and make the pitch real.
