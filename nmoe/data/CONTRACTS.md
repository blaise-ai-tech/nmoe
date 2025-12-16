# Data Pipeline Contracts

Preprocessing contracts for deterministic, atomic, and self-describing data pipelines.

## Purpose

- Make preprocessing deterministic, atomic, and self-describing.
- Keep public surfaces small; no format churn (npy+idx + JSON manifest).
- Clarify boundaries: preprocessing vs grading (three gates live in grading).

---

## Tokenizer Contract

### Derivation

- Derive `vocab_size` and `eos_token_id` from the tokenizer at runtime; never hardcode.
- CLI may display derived values for transparency.

### Stamps (manifest)

| Field | Type | Description |
|-------|------|-------------|
| `tokenizer_name` | string | Tokenizer identifier (e.g., "o200k_harmony") |
| `tokenizer_hash` | string | Short digest of tokenizer config/vocab |
| `tokenizer_version` | string | Version if available, else null |

### Drift

- If derived values differ from existing manifest, abort with `E-TOKENIZER-DRIFT`.
- This ensures the same tokenizer is used across all stages and runs.

---

## Manifest Schema

### Top-Level Fields (additions)

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Manifest schema version (starts at 1) |
| `pipeline_version` | string | Pipeline version (e.g., "data_v1") |
| `git_sha` | string | Git commit hash at pipeline run |
| `cuda_arch` | string | CUDA architecture (e.g., "sm_100a") |
| `tokenizer_name` | string | Tokenizer identifier |
| `tokenizer_hash` | string | Tokenizer config digest |

### Existing Fields (unchanged)

- `dataset`, `version`, `tokenizer`, `vocab_size`, `eos_token_id`
- `dtype`, `created_at`, `total_tokens`, `total_documents`, `num_shards`
- `shards[]`, `source_info`

### Shard Entry (unchanged)

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Path to .npy shard |
| `index_path` | string | Path to .idx file |
| `num_tokens` | int | Token count in shard |
| `num_documents` | int | Document count in shard |
| `checksum` | string | SHA256 of shard file |

### Atomicity

- Write manifests to `*.tmp` then rename to final path.
- Incomplete writes must not leave partial manifests.

---

## Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `E-MANIFEST-COMMIT` | Atomic rename/write failure | Check disk space/permissions |
| `E-INDEX-MISSING` | Missing .idx for shard | Regenerate index or re-run stage |
| `E-SHARD-CHECKSUM` | Checksum mismatch on verification | Shard corrupt; re-run stage |
| `E-SOURCE-NOTFOUND` | No input files or HF split invalid | Check source path/config |
| `E-TOKENIZER-DRIFT` | Tokenizer-derived values differ from manifest | Use consistent tokenizer |
| `E-RESUME-CURSOR-MISMATCH` | Last processed record checksum mismatch on resume | State file stale or data corrupted |

### CLI Behavior

- Exit with non-zero code on any error.
- Print single-line actionable message to stderr.
- Format: `ERROR [E-CODE]: description`

---

## Resume State

### File: `state_stage_{N}.json`

```json
{
  "stage": "heuristic_pack",
  "started_at": "2025-01-15T10:30:00Z",
  "completed_at": null,
  "status": "running",

  "rng_state": {
    "python": "base64_encoded_state",
    "numpy": "base64_encoded_state"
  },

  "cursor": {
    "source": "dolma_v3_cc",
    "shard_dir": "/data/raw/dolma_v3",
    "local_doc_count": 1523456
  },

  "manifest_digest": "sha256:abc123...",

  "schema_version": 1,
  "pipeline_version": "data_v1",
  "git_sha": "abc123def456"
}
```

### Semantics

- **Exactly-once**: On rerun, verify last processed record checksum before resuming.
- **Byte-identical guarantee**: Same inputs + config → identical outputs.
- If verification fails, abort with `E-RESUME-CURSOR-MISMATCH`.

---

## Gating Statement

### Preprocessing Gates (always on)

These are structural filters that run unconditionally:

- Deduplication (exact hash + MinHash fuzzy)
- Language ID (fastText lid.176, threshold from config)
- Length bounds (min/max words from config)

### Diagnostics (log-only by default)

These are computed and logged but do not gate unless promoted via policy:

- Perplexity (domain-normalized z-scores)
- Repetition ratio (n-gram analysis)
- Toxicity score (Detoxify/HateBERT)
- PII matches (blocklist patterns)
- Symbol/special char ratio

### Quality Gates (three, live in grading)

These belong to the grading stage, not preprocessing:

| Gate | Description |
|------|-------------|
| `quality_agg` | Aggregated quality score ≥ threshold |
| `fidelity` | Semantic equivalence for rephrased content |
| `diversity_density` | Embedding cluster density ≤ threshold |

---

## Hidden States & Domain Head

### model.forward() Extension

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    return_hidden_states: bool = False,
    hidden_layer: int = 24,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        input_ids: [batch, seq_len] token IDs
        positions: [batch, seq_len] position indices
        return_hidden_states: If True, also return hidden states
        hidden_layer: Which layer's hidden states to return (default: 24)

    Returns:
        logits: [batch, seq_len, vocab_size]
        hidden_states: [batch, seq_len, hidden_dim] (only if return_hidden_states=True)
    """
```

- Default behavior unchanged (returns logits only).
- Hidden states are from the specified layer, before the final LM head.

### DomainClassifierHead

```python
class DomainClassifierHead(nn.Module):
    """Linear classifier on LLM hidden states."""

    def __init__(self, hidden_dim: int, num_domains: int = 16):
        self.head = nn.Linear(hidden_dim, num_domains)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            logits: [batch, num_domains] (uses last token's hidden state)
        """
        return self.head(hidden_states[:, -1, :])
```

- Head is a separate artifact (`classifier.pt`), backbone frozen.
- Training: 100k self-labeled samples, 1 epoch, lr=1e-4.

---

## Scorer Output Schema

### JSON Contract (strict)

```json
{
  "helpfulness": 3.2,
  "correctness": 3.8,
  "coherence": 3.5,
  "complexity": 2.9,
  "verbosity": 2.1,
  "aggregated": 0.78,
  "domain": {
    "id": 0,
    "confidence": 0.94
  },
  "scorer_model": "gpt-oss-20B",
  "scorer_version": "v1.0"
}
```

### Field Ranges

| Field | Type | Range |
|-------|------|-------|
| `helpfulness` | float | [0, 5] |
| `correctness` | float | [0, 4] |
| `coherence` | float | [0, 4] |
| `complexity` | float | [0, 4] |
| `verbosity` | float | [0, 3] |
| `aggregated` | float | [0, 1] |
| `domain.id` | int | [0, 15] |
| `domain.confidence` | float | [0, 1] |

### Validation

- Parser must hard-fail on missing or out-of-range fields.
- Map validation errors to structured error codes.

---

## Fidelity Contract

### Embedding Pre-Gate

| Setting | Value |
|---------|-------|
| Model | NV-Embed-v2 (default) |
| Method | Cosine similarity on pooled vectors |
| Threshold | From calibration artifact (no magic default) |

### Decision Logic

```
cosine > threshold_high  → PASS (no LLM call)
cosine ∈ [threshold_low, threshold_high] → LLM VERIFY
cosine < threshold_low   → REJECT (no LLM call)
```

### LLM Verify (borderline only)

**Input:**
```json
{
  "original": "...",
  "rephrased": "..."
}
```

**Output:**
```json
{
  "fidelity_score": 4,
  "pass": true,
  "verifier_model": "gpt-oss-20B",
  "version": "v1.0"
}
```

- Pass rule: `fidelity_score >= 4` (unless calibration overrides).
- Score range: 1-5 (1 = major info loss, 5 = perfect preservation).

---

## Metrics

### Reuse Policy

- Reuse `nmoe/metrics.py` if/when needed; no new subsystem.
- Pipeline may emit metrics but is not required to.

### Optional Tags (if emitted)

| Tag | Description |
|-----|-------------|
| `stage/retention/1_prep` | Fraction retained after preprocessing |
| `throughput/docs_per_s` | Processing throughput |

---

## Freeze Points

Once a run starts, the following must not change:

- `pipeline_version`
- Scorer prompts
- Quality thresholds
- Tokenizer (name + config)

Changing any of these requires a new run with a new `pipeline_version`.

---

## Acceptance Criteria

| Phase | Acceptance Test |
|-------|-----------------|
| 0 | CONTRACTS.md exists with clear statements |
| 1 | Kill-and-resume → byte-identical outputs; tokenizer hash stamped |
| 2 | Hidden states returned when requested; default path unchanged |
| 3 | CSV shows scorer comparison; adapter validates on sample |
| 4 | Domain classifier F1 > 0.85 on held-out set |
| 5 | Fidelity function returns pass/fail with calibrated threshold |
