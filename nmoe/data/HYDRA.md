# HYDRA: Dual‑Head Latent Grader over gpt‑oss‑20B

Purpose
- Provide an elegant, minimal, high‑throughput quality gate for pretraining/midtraining data using a single heavy backbone pass and lightweight heads.
- Reuse the existing gpt‑oss‑20B backbone once per document, keep KV/prefix hot, and attach two small heads to its latents to drive the “three gates” philosophy (this doc focuses on the Quality gate).
- Zero orchestration frameworks; narrow, production‑ready interfaces; deterministic resume; atomic artifacts.

Non‑Goals (here)
- New storage formats (keep npy+idx + manifest with hashes).
- Reward‑model grading for bulk (RM is only for calibration/borderlines).
- Any NCCL all‑to‑all or TP in training; this is inference‑time curation.

Backbone (source‑of‑truth: the model’s config)
- Core LLM: gpt‑oss‑20B (24 layers; hidden 2880; head_dim 64; n_heads 64; n_kv 8; experts 32, top‑4; vocab ≈201k; ctx ≥4096).
- Precision: BF16 activations, FP8 KV preferred for generation; head params in BF16 with FP32 master for stability.

Scoring Schema (CONTRACTS‑aligned)
- Dimensions (all 0–4): helpfulness, correctness, coherence, complexity, density.
- Aggregated overall in [0,1] via normalized weighted sum: overall = w·(d/4), w fit on a calibration slice (no magic numbers). Persist w and thresholds (τ_drop, τ_keep) in manifest + SQLite.

Design Overview
- Early‑exit schedule across the frozen backbone:
  1) Layers 0–18 → Probe (cheap) → DROP early if overall < τ_drop.
  2) Layers 19–24 → Judge (sequence‑aware) → KEEP if overall ≥ τ_keep, else optional escalation (RM or LLM) for band [τ_drop, τ_keep).
  3) If KEEP and rephrase is requested, reuse KV/prefix for generation (no re‑prefill of the document).

Heads
1) Probe (L18, mean‑pooled)
- Input: mean_pool(hidden_states[layer=18]) ∈ R^{2880}.
- Head: Linear(2880 → 5), outputs the 5 scores on 0–4 scale; clamp to [0,4].
- Purpose: ultra‑cheap early rejection of obvious garbage; also provides the L18 pooled embedding used for semantic dedup/routing.
- Params: ~14.4k (weights) + bias (negligible); runs sub‑ms on B200 for batch O(128).

2) Judge (L24, token‑aware)
- Input: full‑sequence hidden at layer=24 (T×2880), with doc length T ≤ 4096.
- Projector: Linear(2880 → 512).
- Encoder: 1× Transformer block (D=512, H=8, RoPE, RMSNorm, SwiGLU MLP 512→1536→512, Flash‑attn kernels).
- Pool: mean over valid tokens (or learned attention pool) → 512.
- Head: Linear(512 → 5) for the 5 dimensions.
- Optional aux heads (same encoder output):
  - is_synth: Linear(512 → 1) with BCE.
  - domain: Linear(512 → K) with CE (K per CONTRACTS; e.g., 16 classes) for routing/analytics.
- Purpose: fine judgement of coherence/structure/nuance with minimal added compute; still negligible next to the 20B forward.
- Params: ≈3–5M total; runs O(<10ms) per batch at T≈2–4k on B200.

Runtime Flow (single heavy pass; KV reuse)
```
tokens = encode(text)                    # HarmonyGptOss
kv, h18 = backbone(tokens, up_to=18, return_h=18, keep_kv=True)
emb18 = mean_pool(h18)
s_probe = Probe(emb18)                   # 5 dims in 0–4; overall = agg(s_probe)
if overall(s_probe) < τ_drop:
    record_drop("probe") ; continue

kv, h24 = backbone(tokens, from_layer=19, to_layer=24, kv=kv, return_h=24)
s_judge = Judge(h24)                     # 5 dims in 0–4; overall = agg(s_judge)
if overall(s_judge) < τ_keep and escalate:
    s_escal = RM_or_LLM(text)            # small fraction, only band cases
    fuse/calibrate and decide

if KEEP and need_rephrase:
    # reuse KV + optional soft/hard prompt, zero prefill of doc
    out = generate_with_kv(kv, prompt=style(domain), max_new_tokens=...
```

Three Gates (how HYDRA supports them)
- Quality (this doc): Probe+Judge produce overall ∈ [0,1]. Thresholds τ_drop and τ_keep define DROP/KEEP/ESCALATE.
- Diversity: Persist emb18 (float16 or PQ’d int8) and run exact+semantic dedup (FAISS + MinHash). Dedup happens before Judge to avoid spending compute on near‑dupes where possible.
- Fidelity: For rephrased items, cosine(emb18_original, emb18_rephrase) ≥ ρ; escalate borderlines to a strong LLM/RM verify. Persist pass/fail and margins.

Training (distilled supervision, frozen backbone)
- Labels:
  - Primary: gpt‑oss‑20B multi‑attribute judgments with the HYDRA rubric (0–4 dims), generated once on a 50k–200k slice (Dolma3 pool/mix + a small amount of code/math/conversation).
  - Auxiliary (optional): HelpSteer2 attributes; provenance for is_synth; domain tags from metadata or weak heuristics.
- Losses:
  - Probe: masked Huber/MSE over 5 dims; optional pairwise ranking on overall for robustness.
  - Judge: same primary loss; add BCE for is_synth and CE for domain if enabled (multi‑task, weighted).
- Freezing: backbone weights frozen; gradients only through heads (and projector/1‑layer encoder for Judge).
- Data handling: cache tokenized inputs (Harmony), respect 4k ctx; stratify batches across domains.
- Optim: AdamW, β=(0.9, 0.95), lr 1e‑3 (Probe), 5e‑4 (Judge), cosine decay, wd 0.01; BF16 compute + FP32 master.
- Validation: held‑out set; target Pearson/Spearman on overall ≥0.80 for Judge, ≥0.75 for Probe; MAE per dimension ≤0.35.

Calibration (no magic numbers)
- Fit weights w for overall via least squares on a 10k calibration set to maximize agreement with escalation labels (or a high‑end auditor like Skywork‑Reward‑Gemma‑2‑27B on the same slice).
- Sweep τ_drop and τ_keep to produce acceptance curves; pick τs that achieve desired keep rate and measured downstream utility.
- Persist: {w, τ_drop, τ_keep, dataset hash, model SHA, rubric hash} in manifest + SQLite; render curves in NVIZ.

Persistence & Artifacts
- Shards: unchanged npy+idx layout.
- Embeddings: store pooled L18 per doc (float16, or product‑quantized int8) in sidecar for dedup/routing; include checksum.
- Manifests: atomic write via *.tmp → rename; stamp tokenizer name, vocab size, eos id, tokenizer hash; persist head hashes and calibration tuple (w, τs).
- Metrics (SQLite via nmoe/metrics.py):
  - doc/sec, ms/doc (probe, judge, escalate), f_drop/f_keep/f_escalate, tokens saved by early‑exit, dedup stats, fidelity pass rate, disagreement counters vs auditor.

Interfaces (minimal, stable)
- Encoder hooks in `nmoe/data/model.py` (surgical; if not already present):
  - `forward(..., return_hidden_states=True, up_to_layer=None, from_layer=None, keep_kv=False)`
  - `pool_hidden(hidden, mask) -> emb`
- Heads (Torch):
  - `ProbeHead(2880→5)`; `JudgeHead(2880→512→[1×Encoder]→5)` (+ optional aux).
- Aggregation:
  - `aggregate(scores: dict[str,float], w: dict[str,float]) -> float in [0,1]` (normalize dims ÷4 before dot).
- Runner (smart packer): streaming API, batch size and ctx configurable; returns per‑doc decision + reasons.

SLOs / Acceptance
- Correctness: Probe/Judge parity vs oracle on held‑out meets targets above; router load sane when used for domain routing.
- Performance: ≥15% net compute saved at corpus scale via early‑exit; ≤100ms/doc @4k ctx, batch≥128 on B200 for Judge path; Probe <1ms/doc.
- Stability: 10k‑doc soak; deterministic resume; atomic manifests.

Risks & Mitigations
- Underrating top‑tier educational content: add few‑shot rubric examples during oracle labeling; calibrate τ_keep with auditor; allow domain‑specific w.
- Parse fragility (LLM oracle phase): use Harmony StreamableParser and 4k ctx; store raw and parsed outputs for audits.
- Distribution shift across domains: stratified calibration + domain‑aware thresholds (optional).

Operational Playbook
1) Label slice with oracle (LLM/RM) using HYDRA rubric (4096 ctx, StreamableParser) → write labels.
2) Train Probe then Judge heads on frozen 20B latents; validate; export head weights + hashes.
3) Calibrate w, τ_drop, τ_keep on 10k slice; persist; cut acceptance curves.
4) Run smart packer at scale: HF stream → dedup (exact/semantic via emb18) → Probe → Judge → optional escalate → shard write (atomic) + metrics.
5) Periodically audit disagreements vs auditor; refresh calibration as needed.

Config (TOML sketch; read by CLI)
```toml
[hydra]
backbone = "gpt-oss-20b"
probe_layer = 18
judge_layer = 24
project_dim = 512
use_aux_is_synth = true
use_aux_domain = false

[hydra.aggregate]
# Learned on calibration; defaults only as placeholders.
weights = { helpfulness=0.35, correctness=0.20, coherence=0.15, complexity=0.20, density=0.10 }
tau_drop = 0.30
tau_keep = 0.55

[hydra.runtime]
max_seq_len = 4096
batch_size = 128
escalate_band = true
```

Backlog (if needed later; not required for MVP)
- Learned attention pooling for Judge; soft prompts for rephrase; PQ’d embedding store tooling; bandit to tune τ online.

References (context only; do not couple to them at runtime)
- FineWeb‑Edu (LLM‑as‑judge → classifier distill), Seed‑Coder (LLM oracle → small scorer), Nemotron Reward RMs for auditing; Early‑exit/MTP inspirations.
