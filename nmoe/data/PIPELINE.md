# nmoe Data Pipeline Architecture

A comprehensive, model-agnostic data curation pipeline for production-grade LLM pretraining.

## Design Principles

1. **Filter first, augment later** - Don't waste compute rephrasing/generating from garbage
2. **Model-agnostic slots** - Pipeline design is independent of specific model choices
3. **Same quality bar for all sources** - Synthetic data meets identical thresholds as organic
4. **Provenance is first-class** - Full metadata for every document, enabling debugging and analysis
5. **Deterministic and resumable** - Identical inputs produce identical outputs; can resume from any stage
6. **Joint quality-diversity optimization** - Not sequential filtering then balancing (QuaDMix insight)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
├─────────────────┬─────────────────┬─────────────────────────────────────────┤
│   Raw Corpus    │    Curated      │           (Generated Later)             │
│  (CommonCrawl,  │   (arXiv,       │   Rephrased + Synthetic                 │
│   Dolma, etc.)  │   GitHub, etc.) │   (fed back through same pipeline)      │
└────────┬────────┴────────┬────────┴─────────────────────────────────────────┘
         │                 │
         └────────┬────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: HEURISTIC PRE-FILTER                                              │
│  ─────────────────────────────────────────────────────────────              │
│  Fast, CPU-based filters to remove obvious garbage                          │
│                                                                              │
│  • Language ID (fastText lid.176, threshold ≥0.65)                          │
│  • Length filters (min 50 words, max 100k words)                            │
│  • Deduplication:                                                            │
│      - MinHash fuzzy (shingle=13, perms=128, Jaccard≥0.82)                  │
│      - URL/exact hash dedup                                                  │
│      - Line-level dedup (remove repeated lines)                             │
│      - Paragraph-level dedup                                                │
│  • N-gram repetition filter (max 20% repeated 10-grams)                     │
│  • Perplexity filter (KenLM or small LM, remove outliers ±3σ)               │
│  • Symbol/special char ratio (max 30%)                                      │
│  • Boilerplate removal (trafilatura patterns)                               │
│  • Toxicity classifier (HateBERT or Detoxify)                               │
│  • PII blocklist (emails, phones, SSNs, etc.)                               │
│                                                                              │
│  Target: ~40-50% retention                                                  │
│  Output: Filtered docs + heuristic_scores{}                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: QUALITY SCORING                                                    │
│  ─────────────────────────────────────────────────────────────              │
│  Multi-dimensional quality assessment via [QUALITY_SCORER_MODEL]            │
│                                                                              │
│  Dimensions (Nemotron-style, 0-4 scale each):                               │
│  ┌─────────────┬────────────────────────────────────────────────┐           │
│  │ helpfulness │ Does this provide useful information?          │           │
│  │ correctness │ Is the information factually accurate?         │           │
│  │ coherence   │ Is it well-structured and logical?             │           │
│  │ complexity  │ Depth of reasoning and nuance?                 │           │
│  │ verbosity   │ Appropriate length (penalize padding)?         │           │
│  └─────────────┴────────────────────────────────────────────────┘           │
│                                                                              │
│  Domain-specific dimensions (0-10 scale):                                   │
│  ┌─────────────┬────────────────────────────────────────────────┐           │
│  │ CODE        │ readability, modularity, clarity, reusability  │           │
│  │ MATH        │ correctness, reasoning, notation, completeness │           │
│  │ PROSE       │ informativeness, style, engagement             │           │
│  └─────────────┴────────────────────────────────────────────────┘           │
│                                                                              │
│  Two-tier scoring (optional, based on benchmark results):                   │
│  • Bulk: [BULK_SCORER_MODEL] for all documents                              │
│  • Escalation: [ESCALATION_SCORER_MODEL] for borderline band (middle 20%)  │
│  • Record escalated_by in provenance                                        │
│                                                                              │
│  Output: quality_scores{} per document                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: DOMAIN CLASSIFICATION                                              │
│  ─────────────────────────────────────────────────────────────              │
│  Classify documents into semantic domains via [DOMAIN_CLASSIFIER_MODEL]     │
│                                                                              │
│  Architecture options:                                                       │
│  • LLM hidden states + linear head (trained on 100k self-labeled samples)  │
│  • Fine-tuned encoder (Gemma-2-27B, etc.)                                   │
│  • Direct LLM classification                                                │
│                                                                              │
│  Domains (16 classes):                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ code, math, science, instruction, prose, conversation, reference, │     │
│  │ legal, medical, finance, news, creative, qa, academic, technical, │     │
│  │ other                                                              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Hidden-state classifier spec (if using LLM backbone):                      │
│  • Layer: ~24 (intermediate often outperforms final by 2-16%)               │
│  • Head: Linear(hidden_dim, 16) + Softmax                                   │
│  • Training: Freeze backbone, 100k samples, 1 epoch, lr=1e-4                │
│  • Export: classifier.pt separate from backbone                             │
│                                                                              │
│  Output: domain_id, domain_confidence per document                          │
└─────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: QUALITY-DIVERSITY SAMPLING (QuaDMix)                              │
│  ─────────────────────────────────────────────────────────────              │
│  Joint optimization of quality AND diversity (not sequential)               │
│                                                                              │
│  Step 1: Quality Aggregation (per-domain weights)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  q̄ = Σ σ(qₙ) × αₙ,ₘ                                                 │    │
│  │                                                                      │    │
│  │  where:                                                              │    │
│  │    qₙ = individual quality scores (helpfulness, correctness, etc.)  │    │
│  │    αₙ,ₘ = learned weights per domain m                               │    │
│  │    σ = normalization (sigmoid)                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Step 2: Percentile Ranking (within each domain)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  r̄ = rank(q̄) / count(domain)                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Step 3: Sigmoid Sampling (per-domain parameters)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  S(r̄) = (2/(1+e^(-λₘ(ωₘ-r̄))))^ηₘ + εₘ   if r̄ ≤ ωₘ                  │    │
│  │        = εₘ                               if r̄ > ωₘ                  │    │
│  │                                                                      │    │
│  │  Parameters (learned via proxy model optimization):                  │    │
│  │    λₘ ∈ [1, 1000]   - decay rate (steepness)                        │    │
│  │    ωₘ ∈ [0, 0.1]    - quality threshold percentile                  │    │
│  │    ηₘ ∈ [0, 1]      - scaling exponent                              │    │
│  │    εₘ ∈ [0, 0.001]  - baseline probability                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Hyperparameter optimization:                                                │
│  • Train proxy models (530M-1.3B) on candidate mixtures                     │
│  • LightGBM regression to predict downstream performance                    │
│  • 100 iterations, select top-10 configurations, average                    │
│                                                                              │
│  Target: ~10-15% retention (of Stage 1 output)                              │
│  Output: Sampled high-quality corpus                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: DATA AUGMENTATION                                                  │
│  ─────────────────────────────────────────────────────────────              │
│  Generate synthetic data from high-quality filtered corpus                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  5a: REPHRASING (K2-style)                                          │    │
│  │  ────────────────────────                                           │    │
│  │  • Model: [REPHRASING_MODEL]                                        │    │
│  │  • Up to 10 rephrasings per document (K2: 10 rephrasings > 10 epochs)│   │
│  │  • Style-diverse prompts (formal, casual, technical, simplified)    │    │
│  │  • Chunk-wise for long docs (2k token chunks with context)          │    │
│  │  • Track: parent_doc_id, prompt_id, seed                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  5b: SYNTHETIC GENERATION                                           │    │
│  │  ────────────────────────────                                       │    │
│  │  • Model: [GENERATION_MODEL] (may be larger for quality)            │    │
│  │  • Types: reasoning traces, QA pairs, instruction-response          │    │
│  │  • Track: synthetic_kind, model_name, prompt_template               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: Augmented corpus (original + rephrased + synthetic)                │
│          All marked with is_synthetic=true, synthetic_kind={rephrase|gen}   │
└─────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: SYNTHETIC QUALITY GATE                                             │
│  ─────────────────────────────────────────────────────────────              │
│  Synthetic data must pass the SAME quality bar as originals                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  6a: FIDELITY CHECK (rephrased only)                                │    │
│  │  ─────────────────────────────────                                  │    │
│  │  Verify semantic equivalence to original                            │    │
│  │                                                                      │    │
│  │  Embedding model: [EMBEDDING_MODEL]                                 │    │
│  │  • Cosine > 0.85: PASS                                              │    │
│  │  • Cosine ∈ [0.80, 0.85]: LLM verify                                │    │
│  │  • Cosine < 0.80: REJECT (no LLM call)                              │    │
│  │                                                                      │    │
│  │  LLM verification: [FIDELITY_VERIFIER_MODEL]                        │    │
│  │  • Compare (original, rephrased) pair                               │    │
│  │  • Score 1-5 semantic alignment                                     │    │
│  │  • Threshold: ≥4 to pass                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  6b: QUALITY SCORING                                                │    │
│  │  ───────────────────────                                            │    │
│  │  Same [QUALITY_SCORER_MODEL] and thresholds as Stage 2              │    │
│  │  Synthetic must meet identical quality bar                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  6c: DIVERSITY CHECK                                                │    │
│  │  ─────────────────────                                              │    │
│  │  Ensure we're not just echoing ourselves                            │    │
│  │                                                                      │    │
│  │  • Embedding clustering (RAPIDS k-means or FAISS)                   │    │
│  │  • Self-BLEU within rephrasings of same document                    │    │
│  │  • Per-source budget caps to prevent mode collapse                  │    │
│  │  • Flag and review if cluster becomes too dense                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: Verified synthetic corpus                                          │
│          gates{fidelity_pass, quality_pass, diversity_pass} in provenance   │
└─────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: FINAL ASSEMBLY                                                     │
│  ─────────────────────────────────────────────────────────────              │
│  Merge all sources and prepare for training                                 │
│                                                                              │
│  • Merge: original + rephrased + synthetic                                  │
│  • Domain balancing: Apply final mixture weights                            │
│  • FIM: Fill-in-Middle for code domain (rate=0.1)                           │
│  • Tokenize: o200k_harmony (vocab_size=201088)                              │
│  • Shard: .npy + .idx files (500M tokens per shard)                         │
│  • Manifest: Full provenance + quality metadata                             │
│                                                                              │
│  Output:                                                                     │
│  ├── shards/shard_XXXX.npy                                                  │
│  ├── shards/shard_XXXX.idx                                                  │
│  └── manifest.jsonl (per-doc metadata)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Slots (To Be Filled by Benchmarking)

| Slot | Role | Candidates | Decision Criteria |
|------|------|------------|-------------------|
| `QUALITY_SCORER_MODEL` | Multi-dimensional quality scoring | gpt-oss-20B, Nemotron-340B-Reward, Skywork-27B | RewardBench correlation, throughput |
| `ESCALATION_SCORER_MODEL` | Borderline cases (optional) | gpt-oss-120B, Nemotron-340B | Quality on hard cases |
| `DOMAIN_CLASSIFIER_MODEL` | 16-class domain classification | gpt-oss-20B + linear head, Gemma-2-27B fine-tuned | F1 on held-out labels |
| `EMBEDDING_MODEL` | Fidelity cosine, diversity clustering | NV-Embed-v2, Qwen3-Embedding-8B, gpt-oss-20B hidden | MTEB score, serving ease |
| `FIDELITY_VERIFIER_MODEL` | LLM comparison for borderline fidelity | gpt-oss-20B, gpt-oss-120B | Accuracy vs human labels |
| `REPHRASING_MODEL` | K2-style knowledge rephrasing | gpt-oss-20B | Already implemented |
| `GENERATION_MODEL` | Synthetic data generation | gpt-oss-20B, gpt-oss-120B | Quality of traces/QA |

---

## Provenance Schema

Every document carries full metadata in `manifest.jsonl`:

```json
{
  "doc_id": "sha256:abc123...",
  "source_id": "dolma_v3_cc_2024_10",
  "url_hash": "sha256:def456...",
  "license": "cc-by-4.0",
  "lang": "en",
  "lang_confidence": 0.98,

  "is_synthetic": false,
  "synthetic_kind": null,
  "parent_doc_id": null,
  "prompt_id": null,
  "seed": null,
  "model_name": null,

  "heuristic_scores": {
    "word_count": 1523,
    "perplexity": 45.2,
    "repetition_ratio": 0.03,
    "symbol_ratio": 0.02,
    "toxicity_score": 0.01
  },

  "quality_scores": {
    "helpfulness": 3.2,
    "correctness": 3.8,
    "coherence": 3.5,
    "complexity": 2.9,
    "verbosity": 2.1,
    "domain_specific": 7.5,
    "aggregated": 0.78
  },
  "scorer_model": "gpt-oss-20B",
  "scorer_version": "v1.0",
  "escalated_by": null,

  "domain": {
    "id": 0,
    "name": "code",
    "confidence": 0.94
  },

  "gates": {
    "heuristic_pass": true,
    "quality_pass": true,
    "sampled": true,
    "fidelity_pass": null,
    "diversity_pass": null
  },

  "tokens": 2048,
  "shard": "shard_0042.npy",
  "offset": 1048576
}
```

For synthetic documents, additional fields are populated:

```json
{
  "is_synthetic": true,
  "synthetic_kind": "rephrase",
  "parent_doc_id": "sha256:original123...",
  "prompt_id": "rephrase_formal_v2",
  "seed": 42,
  "model_name": "gpt-oss-20B",

  "gates": {
    "fidelity_pass": true,
    "fidelity_score": 0.87,
    "quality_pass": true,
    "diversity_pass": true
  }
}
```

---

## Determinism & Resume

Each stage persists state for exact reproducibility:

```json
// state_stage_1.json
{
  "stage": 1,
  "stage_name": "heuristic_filter",
  "started_at": "2025-01-15T10:30:00Z",
  "completed_at": null,
  "status": "running",

  "rng_state": {
    "python": "base64_encoded_state",
    "numpy": "base64_encoded_state",
    "torch": "base64_encoded_state"
  },

  "cursor": {
    "source": "dolma_v3_cc_2024_10",
    "shard": 42,
    "offset": 1048576
  },

  "metrics": {
    "docs_processed": 1523456,
    "docs_retained": 723456,
    "retention_rate": 0.475
  }
}
```

Resume behavior:
- Load state from `state_stage_N.json`
- Restore RNG states
- Seek to cursor position
- Continue processing

---

## Metrics (SQLite)

Aligned with `nmoe/metrics.py` contract. Pipeline emits to the metrics directory (configured in `configs/storage.toml`):

```sql
-- Per-stage retention
INSERT INTO metrics (run, tag, step, value) VALUES
  ('data_v1', 'stage/retention/1_heuristic', 0, 0.47),
  ('data_v1', 'stage/retention/2_quality', 0, 0.85),
  ('data_v1', 'stage/retention/4_sampling', 0, 0.12);

-- Domain histograms
INSERT INTO metrics (run, tag, step, value) VALUES
  ('data_v1', 'domain/count/code', 0, 15234567),
  ('data_v1', 'domain/count/math', 0, 8234567),
  ('data_v1', 'domain/count/prose', 0, 45234567);

-- Quality distributions
INSERT INTO metrics (run, tag, step, value) VALUES
  ('data_v1', 'quality/mean/helpfulness', 0, 3.2),
  ('data_v1', 'quality/std/helpfulness', 0, 0.8);

-- Synthetic metrics
INSERT INTO metrics (run, tag, step, value) VALUES
  ('data_v1', 'fidelity/pass_rate', 0, 0.92),
  ('data_v1', 'diversity/cluster_coverage', 0, 0.85);

-- Throughput
INSERT INTO metrics (run, tag, step, value) VALUES
  ('data_v1', 'throughput/docs_per_s', 0, 523.4),
  ('data_v1', 'latency/p50_ms', 0, 45.2),
  ('data_v1', 'latency/p95_ms', 0, 123.5);

-- Escalation (if two-tier scoring)
INSERT INTO metrics (run, tag, step, value) VALUES
  ('data_v1', 'escalation/rate', 0, 0.18);
```

---

## Module Structure

```
nmoe/data/
├── pipeline.py              # Main orchestrator, CLI entry point
├── config.py                # PipelineConfig dataclass
│
├── filters/                 # Stage 1
│   ├── __init__.py
│   ├── heuristic.py         # Length, symbol ratio, repetition
│   ├── language.py          # fastText lid.176
│   ├── dedup.py             # MinHash, exact, line-level, paragraph
│   ├── perplexity.py        # KenLM or small LM
│   ├── toxicity.py          # HateBERT / Detoxify
│   └── pii.py               # Blocklist patterns
│
├── scoring/                 # Stage 2
│   ├── __init__.py
│   ├── base.py              # Abstract QualityScorer interface
│   ├── prompts.py           # Scoring prompts (multi-dimensional)
│   └── aggregator.py        # Combine scores with domain weights
│
├── classification/          # Stage 3
│   ├── __init__.py
│   ├── base.py              # Abstract DomainClassifier interface
│   └── head.py              # Linear head on LLM hidden states
│
├── sampling/                # Stage 4
│   ├── __init__.py
│   ├── quadmix.py           # QuaDMix implementation
│   └── optimizer.py         # Proxy model hyperparameter search
│
├── augmentation/            # Stage 5
│   ├── __init__.py
│   ├── rephrase.py          # K2-style rephrasing (existing)
│   ├── synthetic.py         # Trace/QA/instruction generation
│   └── prompts.py           # Style-diverse prompt templates
│
├── verification/            # Stage 6
│   ├── __init__.py
│   ├── fidelity.py          # Embedding + LLM verification
│   └── diversity.py         # Clustering, self-BLEU
│
├── assembly/                # Stage 7
│   ├── __init__.py
│   ├── merge.py             # Combine sources
│   ├── fim.py               # Fill-in-Middle for code
│   └── tokenize.py          # Tokenization and sharding
│
├── provenance/              # Metadata
│   ├── __init__.py
│   ├── schema.py            # Document provenance dataclass
│   └── manifest.py          # Manifest writer
│
├── metrics/                 # Observability
│   ├── __init__.py
│   └── pipeline_metrics.py  # SQLite writer for pipeline stages
│
├── model.py                 # BatchedGenerator (existing)
└── rephrase.py              # Legacy import (existing)
```

---

## CLI Interface (Design Sketch)

```bash
# Full pipeline
python -m nmoe.data.pipeline run \
    --config configs/data/pipeline_v1.toml \
    --source hf:allenai/dolma-v3 \
    --output /data/nmoe/curated_v1

# Individual stages
python -m nmoe.data.pipeline stage1 --input /raw --output /stage1
python -m nmoe.data.pipeline stage2 --input /stage1 --output /stage2
python -m nmoe.data.pipeline stage3 --input /stage2 --output /stage3
python -m nmoe.data.pipeline stage4 --input /stage3 --output /stage4
python -m nmoe.data.pipeline stage5 --input /stage4 --output /stage5
python -m nmoe.data.pipeline stage6 --input /stage5 --output /stage6
python -m nmoe.data.pipeline stage7 --input /stage6 --output /final

# Resume from checkpoint
python -m nmoe.data.pipeline run --resume /data/nmoe/curated_v1

# Quality analysis
python -m nmoe.data.pipeline analyze --input /stage2 --report quality.html

# Benchmark scorers (for model selection)
python -m nmoe.data.pipeline benchmark \
    --models gpt-oss-20B,nemotron-340b,skywork-27b \
    --samples /data/benchmark_samples.jsonl \
    --output /results/scorer_benchmark.json
```

---

## Configuration (TOML)

```toml
[pipeline]
name = "nmoe_curated_v1"
output_dir = "/data/nmoe/curated_v1"
resume = true

[stage1.heuristic]
min_words = 50
max_words = 100000
lang_threshold = 0.65
allowed_langs = ["en"]
minhash_shingle = 13
minhash_perms = 128
minhash_threshold = 0.82
max_repetition_ratio = 0.20
max_symbol_ratio = 0.30
perplexity_sigma = 3.0

[stage2.quality]
# Model slot - filled after benchmarking
model = "${QUALITY_SCORER_MODEL}"
model_path = "${QUALITY_SCORER_PATH}"
batch_size = 64
max_input_tokens = 4096
max_output_tokens = 256
temperature = 0.1

# Two-tier (optional)
enable_escalation = true
escalation_model = "${ESCALATION_SCORER_MODEL}"
escalation_band = [0.4, 0.6]  # Middle 20%

[stage3.domain]
model = "${DOMAIN_CLASSIFIER_MODEL}"
classifier_path = "/models/domain_classifier.pt"
hidden_layer = 24
num_domains = 16

[stage4.sampling]
# QuaDMix parameters (learned via proxy optimization)
params_path = "/models/quadmix_params.json"
target_retention = 0.12

[stage5.augmentation]
# Rephrasing
rephrase_model = "${REPHRASING_MODEL}"
max_rephrasings = 10
prompt_templates = ["formal", "casual", "technical", "simplified", "concise"]

# Synthetic generation
generation_model = "${GENERATION_MODEL}"
synthetic_types = ["reasoning_trace", "qa_pair", "instruction"]

[stage6.verification]
# Fidelity
embedding_model = "${EMBEDDING_MODEL}"
cosine_pass_threshold = 0.85
cosine_verify_threshold = 0.80
fidelity_verifier = "${FIDELITY_VERIFIER_MODEL}"
fidelity_score_threshold = 4

# Diversity
enable_clustering = true
max_cluster_density = 0.95
self_bleu_threshold = 0.7

[stage7.assembly]
tokenizer = "o200k_harmony"
vocab_size = 201088
eos_token_id = 199999
shard_size_tokens = 500_000_000
fim_rate = 0.1
fim_domains = ["code"]
```

---

## Measurement Plan

Before filling model slots, benchmark with this harness:

### Quality Scorer Benchmark

```yaml
inputs:
  - samples: 1000 docs per domain (16k total)
  - length_bins: [256, 1024, 2048, 4096]
  - human_labels: 500 docs with human quality scores

models:
  - gpt-oss-20B
  - gpt-oss-120B
  - Nemotron-340B-Reward
  - Skywork-Reward-27B
  - Skywork-Reward-8B

metrics:
  - correlation_with_human: Pearson/Spearman vs human labels
  - rewardbench_score: If applicable
  - throughput_docs_per_s: Per GPU
  - latency_p50_ms: Per document
  - latency_p95_ms: Per document
  - memory_gb: Peak GPU memory
  - cost_per_1k_docs: Compute cost estimate
```

### Embedding Model Benchmark

```yaml
inputs:
  - samples: 5000 (original, rephrased) pairs
  - human_fidelity_labels: 500 pairs with human judgment

models:
  - NV-Embed-v2
  - Qwen3-Embedding-8B
  - gpt-oss-20B (hidden states)

metrics:
  - fidelity_auprc: AUPRC vs human fidelity labels
  - clustering_silhouette: Quality of domain clusters
  - throughput_docs_per_s: Per GPU
  - memory_gb: Peak GPU memory
  - serving_compatibility: vLLM/SGLang/HF support
```

### Domain Classifier Benchmark

```yaml
inputs:
  - samples: 100k self-labeled by gpt-oss-120B
  - held_out: 10k for testing

models:
  - gpt-oss-20B hidden + linear head (layer 24)
  - gpt-oss-20B hidden + linear head (layer sweep)
  - Gemma-2-27B fine-tuned
  - Direct LLM classification

metrics:
  - f1_macro: Across 16 domains
  - f1_per_domain: Per-domain breakdown
  - throughput_docs_per_s: Per GPU
  - training_time: For head/fine-tune approaches
```

---

## Validation Protocol

### Proxy Model Evaluation (FineWeb Protocol)

1. Train 1.3B proxy model on curated subset (100B tokens)
2. Evaluate on held-out benchmarks
3. If improvement at 1.3B, likely scales to 70B+
4. Iterate on scoring weights / thresholds

### Target Metrics

| Benchmark | Baseline (raw) | Target (curated) |
|-----------|----------------|------------------|
| MMLU | 35% | 50%+ |
| ARC-Easy | 55% | 70%+ |
| HellaSwag | 45% | 55%+ |
| HumanEval | 15% | 30%+ |
| SimpleQA | 24% | 35%+ |

### Quick-Reject Rule

If a candidate mixture regresses any benchmark by >2% vs baseline, reject without full evaluation.

---

## References

### Data Curation
- [DCLM](https://arxiv.org/abs/2406.11794) - fastText classifier, instruction data as positive class
- [FineWeb-Edu](https://arxiv.org/abs/2406.17557) - LLM annotation → classifier distillation
- [QuaDMix](https://arxiv.org/abs/2504.16511) - Joint quality-diversity optimization
- [Seed-Coder](https://arxiv.org/abs/2506.03524) - 4-dimension code quality
- [K2 (Kimi)](https://arxiv.org/abs/2507.20534) - 10 rephrasings > 10 epochs
- [CLIMB](https://research.nvidia.com/labs/lpr/climb/) - Clustering-based mixture search
- [Data Quality Illusion](https://arxiv.org/abs/2510.00866) - CQF limitations

### Models & Benchmarks
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding models
- [RewardBench](https://huggingface.co/spaces/allenai/reward-bench) - Reward models
- [Nemotron-340B-Reward](https://huggingface.co/nvidia/Nemotron-4-340B-Reward) - 5-dimension scoring
- [Skywork-Reward](https://github.com/SkyworkAI/Skywork-Reward) - SOTA reward models
- [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) - 72.31 MTEB
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B) - 70.58 MTEB multilingual
