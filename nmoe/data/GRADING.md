# Data Quality Grading & Filtering

State-of-the-art approaches for LLM training data curation (2024-2025).

## Overview

Modern data pipelines use model-based filtering rather than heuristics. Key finding: **classifier-based quality filtering (CQF) is the most impactful component** for assembling high-quality training sets.

---

## 1. DCLM (DataComp-LM)

**Source**: [GitHub](https://github.com/mlfoundations/dclm) | [Paper](https://arxiv.org/abs/2406.11794) | [Website](https://www.datacomp.ai/dclm/)

### Approach
- **fastText bigram classifier** for document scoring
- Simple, CPU-based, scales to trillions of tokens
- Trained on ~400k documents (200k positive, 200k negative)

### Training Data
| Class | Source | Notes |
|-------|--------|-------|
| Positive | OpenHermes 2.5 + ELI5 | Instruction-formatted data outperforms Wikipedia |
| Negative | Random RefinedWeb samples | Low-quality web crawl baseline |

### Key Finding
> Instruction-formatted positive examples beat traditional "high quality" sources (Wikipedia, books). This suggests data quality definitions differ from conventional assumptions.

### Filtering
- Score all documents with classifier
- **Keep top 10%** (stricter thresholds outperform permissive ones)
- Top-10% >> Top-15% >> Top-20%

### Results
- 7B model trained on DCLM-Baseline: **64% MMLU** with 2.6T tokens
- +6.6pp improvement over MAP-Neo with 40% less compute

---

## 2. FineWeb-Edu

**Source**: [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | [Paper](https://arxiv.org/html/2406.17557v1)

### Approach
Two-stage pipeline:
1. **LLM-as-judge**: Large model annotates samples
2. **Train small classifier**: Distill LLM judgments into fast classifier

### Stage 1: LLM Annotation
- **Model**: Llama-3-70B-Instruct
- **Samples**: 460k FineWeb documents
- **Scale**: 0-5 additive scoring

**Prompt Design**:
- Focus on "grade-school and middle-school level knowledge"
- Avoids bias toward highly technical content (arXiv abstracts)
- Additive scale (Yuan et al.) works best

### Stage 2: Classifier Training
- **Embedding**: Snowflake-arctic-embed-m (frozen)
- **Head**: Linear regression
- **Training**: 410k samples, 20 epochs, lr=3e-4
- **Validation**: 50k samples, select best F1 checkpoint
- **F1 Score**: 82% (binary classification at threshold 3)

### Filtering
- **Threshold**: Score ≥ 3 (out of 5)
- Balances knowledge benchmarks (MMLU, ARC) vs other tasks (HellaSwag)

### Results
- 15T tokens → 1.3T tokens after filtering
- **10x more efficient**: matches C4/Dolma MMLU with 10x fewer tokens
- Cost: 6k H100 GPU hours for 15T token classification

---

## 3. Kimi K2 Knowledge Rephrasing

**Source**: [Paper](https://arxiv.org/html/2507.20534v1) | [GitHub](https://github.com/MoonshotAI/Kimi-K2)

### Approach
Synthetic data augmentation via rephrasing with fidelity verification.

### Pipeline Components

#### 1. Style/Perspective-Diverse Prompting
- Varied linguistic styles
- Different perspectives
- Maintains factual integrity

#### 2. Chunk-wise Autoregressive Rewriting
- Divide long documents into segments
- Rephrase each chunk with context from previous
- Stitch back together
- Mitigates LLM output length limitations

#### 3. Fidelity Verification
- **LLM compares original vs rephrased**
- Checks semantic alignment
- Quality control before training

### Results
| Strategy | SimpleQA Accuracy |
|----------|-------------------|
| 10 epochs original data | 23.76% |
| 1 rephrase + 10 epochs | 27.39% |
| **10 rephrasings + 1 epoch** | **28.94%** |

> Quality beats quantity: +5.18pp improvement from rephrasing vs repetition

### Production Notes
- Each corpus rephrased **at most twice** (diminishing returns)
- Key challenges: hallucinations, toxicity, domain generalization

---

## 4. ByteDance Seed-Coder (Code-Specific)

**Source**: [Paper](https://arxiv.org/abs/2506.03524) | [GitHub](https://github.com/ByteDance-Seed/Seed-Coder) | [Website](https://bytedance-seed-coder.github.io/)

### Approach
LLM-based code quality filtering across 4 dimensions, distilled to small classifier.

### The Four Quality Dimensions

| Dimension | Description |
|-----------|-------------|
| **Readability** | Comments, naming conventions, formatting, structural practices |
| **Modularity** | Well-structured, avoids complex/lengthy functions, clear separation of logic |
| **Clarity** | Minimizes redundancy, no excessive debug prints, clear intent |
| **Reusability** | Code can be adapted for other purposes |

### Pipeline

1. **Oracle Annotation**: DeepSeek-V2-Chat scores 222k code files on 4 dimensions
2. **Distillation**: Fine-tune Llama-2-1.3B with regression head (1 epoch)
3. **Filtering**: Apply scorer to all GitHub data, filter bottom ~10%

### Key Finding
> LLM filters capture nuanced quality standards that rule-based filters cannot. Rule-based filters are "prone to subjective biases, limited in scalability, and costly to extend across languages."

### Results
- 6T token corpus (5T regular + 1T continued pretraining)
- Ablation shows LLM-filtered code significantly outperforms rule-filtered

---

## 5. ByteDance QuaDMix

**Source**: [Paper](https://arxiv.org/abs/2504.16511)

### Approach
Joint optimization of **quality AND diversity** (not sequential).

### Problem
> Traditional pipelines apply quality filtering first, then domain balancing. This overlooks the trade-off: high-quality data often has domain bias, diverse data may compromise quality.

### Framework (3 Stages)

#### Stage 1: Feature Extraction
- **Quality Scores**: AskLLM, FineWeb-Edu, DCLM (3 metrics)
- **Domain Classification**: DeBERTa-V3 classifier → 26 domains

#### Stage 2: Quality Aggregation
Per-domain weighted combination of quality scores:
```
q̄ = Σ σ(qₙ) × αₙ,ₘ
```
- Different domains weight quality criteria differently
- Domain-specific merging parameters αₙ,ₘ

#### Stage 3: Quality-Diversity Sampling
Sigmoid-based sampling function per domain:
```
S(r̄) = (2/(1+e^(-λₘ(ωₘ-r̄))))^ηₘ + ϵₘ  if r̄ ≤ ωₘ
      = ϵₘ                              if r̄ > ωₘ
```

### Hyperparameters (per domain)
- **αₙ,ₘ**: Quality criteria weights
- **λₘ**: Quality-based decay rate
- **ωₘ**: Quality percentile threshold (0.0-0.1 range)
- **ηₘ**: Sampling value scaling
- **ϵₘ**: Baseline sampling for low-quality data

### Optimization
- Train proxy models (530M params) on sampled mixtures
- Use LightGBM to search parameter space
- Avoid exhaustive full-model retraining

### Results
- **+7.2%** average improvement across benchmarks
- Outperforms DCLM, FineWeb-Edu, AskLLM, DSIR, RegMix

---

## 6. NVIDIA CLIMB

**Source**: [Paper](https://research.nvidia.com/labs/lpr/climb/) | [ClimbMix Dataset](https://research.nvidia.com/labs/lpr/climb/)

### Approach
Clustering-based iterative data mixture bootstrapping.

### Pipeline

1. **Embed**: Encode text with pretrained encoder
2. **Cluster**: K-means into semantic groups
3. **Prune/Merge**: Remove low-quality clusters, merge redundant
4. **Iterate**:
   - Sample candidate mixtures
   - Train proxy models on subset
   - Update predictor to estimate performance
   - Prune poor mixtures, keep promising ones

### Key Innovation
> Progressive refinement eliminates suboptimal candidates early, converging toward optimized mixture without exhaustive manual curation.

### Results
- 950M model + 400B ClimbMix tokens **exceeds Llama-3.2-1B** by 2.0% on 12 reasoning tasks
- Domain-specific optimization: +5% for Social Sciences

### Released Datasets
- **ClimbLab**: 1.3T filtered tokens, 20 semantic clusters (research playground)
- **ClimbMix**: 400B optimized tokens for efficient pretraining

---

## 7. NVIDIA NeMo Curator

**Source**: [GitHub](https://github.com/NVIDIA-NeMo/Curator) | [Docs](https://docs.nvidia.com/nemo/curator/latest/) | [Blog](https://developer.nvidia.com/blog/curating-trillion-token-datasets-introducing-nemo-data-curator/)

### Approach
GPU-accelerated modular pipeline for text, image, video, audio.

### Text Processing Components

| Component | Description |
|-----------|-------------|
| **30+ Heuristic Filters** | Word count, line endings, boilerplate detection |
| **fastText Classification** | Language ID, domain classification |
| **GPU Classifiers** | Quality, safety, content type |
| **Deduplication** | Fuzzy (MinHash), exact, semantic |
| **Data Blending** | Mix sources with configurable ratios |

### Performance
- **16x faster** fuzzy dedup on 8TB RedPajama v2 (1.78T tokens)
- **40% lower TCO** vs CPU alternatives
- Uses RAPIDS (cuDF, cuML, cuGraph) + Ray for multi-node scaling

### Modality Support
- Text, Image, Video, Audio
- Each has specialized filters (aesthetic, NSFW, WER, motion, etc.)

---

## 8. Qwen Data Pipeline

**Source**: [Qwen2.5 Report](https://arxiv.org/pdf/2412.15115) | [Qwen3 Report](https://arxiv.org/pdf/2505.09388)

### Approach
Multi-stage pretraining with LLM-based quality filters.

### Qwen2.5
- **Filter Model**: Qwen2-Instruct as quality filter
- **Multi-dimensional analysis**: Comprehensive scoring across criteria
- **Multilingual**: Enhanced filtering across languages (not just EN/ZH)
- **Scale**: 7T → 18T tokens

### Qwen3
- **Three-stage pretraining**:
  1. 30T tokens: General knowledge foundation
  2. Knowledge-intensive: STEM, coding (Qwen2.5-Math, Qwen2.5-Coder for synthetic)
  3. Long-context: 4k → 32k context extension
- **PDF extraction**: Qwen2.5-VL fine-tuned for text extraction
- **Synthetic data**: Domain-specific models generate targeted content

### Key Insight
> Use the model itself (or domain-specific variants) for data curation. Qwen2-Instruct filters for Qwen2.5, specialized models for domain data.

---

## 9. DeepSeek Data Pipeline

**Source**: [DeepSeek-V3 Report](https://arxiv.org/html/2412.19437v1) | [GitHub](https://github.com/deepseek-ai/DeepSeek-V3)

### Approach
Aggressive deduplication + quality filtering + data remixing.

### Pipeline Components

1. **Deduplication**: MinHash/SimHash fuzzy dedup across crawl dumps
2. **Quality Filtering**: Linguistic + semantic assessment
   - Individual level: clarity, coherence
   - Global level: remove low-quality domains
3. **Ranking + Manual Review**: Quality ranking, human annotation of metadata
4. **Data Remixing**: Analyze token share per domain, reweight for balance
5. **FIM Strategy**: Fill-in-Middle at document level (rate=0.1)

### Scale
- 14.8T diverse, high-quality tokens
- Enriched with math, code, multilingual beyond EN/ZH

### Key Insight
> Multiple iterations until desired quality reached. Remixing ensures broad domain coverage without heavy bias.

---

## 10. Implementation Recommendations

### For Quality Filtering (New Data)

**Option A: DCLM-style (Fast, Proven)**
```
1. Collect positive examples (instruction data, high-quality QA)
2. Sample negative examples from raw crawl
3. Train fastText classifier
4. Filter to top 10%
```
- Pro: CPU-only, scales to 100T+ tokens
- Con: Less nuanced than LLM-based

**Option B: FineWeb-Edu style (Higher Quality)**
```
1. Sample 500k documents from corpus
2. Score with gpt-oss-20b (0-5 scale)
3. Train embedding + linear classifier
4. Filter at threshold 3
```
- Pro: Better discrimination, customizable criteria
- Con: Requires GPU for annotation phase

### For Rephrasing Verification

**K2-style Fidelity Check**
```
1. Generate rephrasings with gpt-oss-20b
2. For each (original, rephrased) pair:
   - Prompt LLM: "Do these contain the same facts? Score 1-5"
   - Or: Use embedding similarity as fast proxy
3. Filter rephrasing with score < 4
```

### Hybrid Approach (Recommended)

```
┌─────────────────┐
│   Raw Corpus    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DCLM Classifier │  ← fastText, top-30% (coarse filter)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Grading    │  ← gpt-oss-20b, score 0-5
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train Classifier│  ← Embed + linear on LLM scores
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Filter    │  ← threshold ≥ 3, get top ~10%
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  K2 Rephrasing  │  ← 2x rephrasing with fidelity check
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training Data   │
└─────────────────┘
```

---

## 11. Key Metrics

| Metric | Benchmark | Notes |
|--------|-----------|-------|
| MMLU | Knowledge | Primary quality signal |
| ARC | Reasoning | Educational content quality |
| HellaSwag | Commonsense | Avoid over-filtering |
| SimpleQA | Factual | Rephrasing effectiveness |

### Evaluation Protocol (FineWeb)
- Train small model (1.3B) on data subset
- Measure benchmark performance
- If improvement at 1.3B, likely scales to 70B+
- Iterate on filtering criteria

---

## 12. Summary: Key Patterns Across SOTA

| Approach | Quality Signal | Diversity | Scale | Key Innovation |
|----------|---------------|-----------|-------|----------------|
| DCLM | fastText (instruction pos) | - | 100T+ | Instruction data as positive class |
| FineWeb-Edu | LLM→Embed+Linear | - | 15T | LLM distillation to fast classifier |
| K2 Rephrase | LLM fidelity check | Style prompts | - | 10x rephrasing > 10x epochs |
| Seed-Coder | LLM (4 dims)→1.3B | - | 6T | Domain-specific quality dimensions |
| QuaDMix | Multi-score aggregate | 26 domains | - | Joint quality-diversity optimization |
| CLIMB | Cluster quality | Semantic clusters | 1.3T | Iterative mixture search |
| NeMo | GPU classifiers | - | 100T+ | 16x faster with RAPIDS |
| Qwen | Self-model filter | Multi-stage | 30T+ | Use own model for curation |
| DeepSeek | Linguistic+Semantic | Remixing | 14.8T | Iterative until quality target |

### Common Themes
1. **LLM-as-judge → Small classifier**: Large model annotates, small model filters at scale
2. **Instruction data > Wikipedia**: For positive examples in classifier training
3. **Joint quality-diversity**: Don't filter then balance—optimize together
4. **Proxy model evaluation**: Test on small models before full training
5. **Iterative refinement**: Multiple passes until quality threshold met
6. **Domain-specific criteria**: Different quality weights per domain/modality

---

## 13. References

1. **DCLM**: Li et al. "DataComp-LM: In search of the next generation of training sets for language models" (2024) [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)

2. **FineWeb**: Penedo et al. "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" (2024) [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)

3. **Kimi K2**: Moonshot AI "Kimi K2: Open Agentic Intelligence" (2025) [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)

4. **Seed-Coder**: ByteDance "Seed-Coder: Let the Code Model Curate Data for Itself" (2025) [arXiv:2506.03524](https://arxiv.org/abs/2506.03524)

5. **QuaDMix**: ByteDance "Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining" (2025) [arXiv:2504.16511](https://arxiv.org/abs/2504.16511)

6. **CLIMB**: NVIDIA "CLustering-based Iterative Data Mixture Bootstrapping" (2025) [Website](https://research.nvidia.com/labs/lpr/climb/)

7. **NeMo Curator**: NVIDIA [GitHub](https://github.com/NVIDIA-NeMo/Curator) | [Docs](https://docs.nvidia.com/nemo/curator/latest/)

8. **Qwen2.5**: Alibaba "Qwen2.5 Technical Report" (2025) [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

9. **Qwen3**: Alibaba "Qwen3 Technical Report" (2025) [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

10. **DeepSeek-V3**: DeepSeek "DeepSeek-V3 Technical Report" (2024) [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

11. **Data Quality Illusion**: "Rethinking Classifier-based quality filtering for LLM Pretraining" (2025) [arXiv:2510.00866](https://arxiv.org/abs/2510.00866)

---

## 14. Code Oracle (Grounded) — Seed‑Coder Pattern, Hydra‑Ready

Purpose
- Use a grounded LLM oracle to score code with evidence‑anchored reasoning, but emit only strict JSON. Distill these targets into fast heads over a frozen backbone (Hydra), enabling bulk gating with one heavy pass.

Schema (strict JSON; final channel only)
```json
{
  "readability": 0.0,
  "modularity": 0.0,
  "clarity": 0.0,
  "reusability": 0.0,
  "zero_score_reason": null,
  "meta": {
    "language": "py",
    "has_tests": false,
    "loc": 187
  }
}
```
- Scales: each axis ∈ [0,10]. If `zero_score_reason` != null (enum: `auto_generated|data_or_config|binary_blob|duplicate|compile_error|security_risk|other`), all four axes must be 0 by contract.

Grounded Oracle Prompt (pattern)
- System: “You are a code‑quality oracle for pretraining. Think privately. Output only strict JSON matching the schema in your final channel.”
- Hidden thinking steps (not persisted):
  1) Analyze code structure and dependencies.
  2) Identify logic errors and bad patterns.
  3) Assess readability and documentation.
  4) Assign 0–10 scores for [readability, modularity, clarity, reusability]; set `zero_score_reason` when applicable.
- Parsing: Use Harmony StreamableParser and read only the final channel. Context ≥ 4096 to avoid truncation.

Aggregation and Mapping
- Aggregated (code) overall in [0,1]:
```
overall_code = (w_r * r + w_m * m + w_c * c + w_u * u) / 10
```
  - Defaults (tuned via calibration): w = {r=0.30, m=0.25, c=0.25, u=0.20}.
- For Hydra training/decisions:
  - Use `overall_code` as the acceptance scalar (quality gate). Define code thresholds τ_drop_code, τ_keep_code on [0,1].
  - When targets are needed on a 0–4 scale: `code_overall_0_4 = 4 * overall_code` (for compatibility with general Hydra heads). Optionally, train a code‑specific head on the 4 axes directly.

Calibration Protocol (deterministic)
- Sample 10k code files (stratified by language/length). Score with the oracle using the grounded prompt.
- Fit weights w by least squares (or simple grid) to align with a chosen auditor or downstream acceptance set.
- Sweep τ_drop_code and τ_keep_code to produce acceptance curves; select τ to meet target keep‑rate and quality. Persist curves in SQLite.
- Persist in manifest: oracle prompt hash/version, tokenizer name+hash, weights w, thresholds τ_drop_code/τ_keep_code, date/commit.

Hydra Integration (one heavy pass)
- Probe (early exit): L18 mean‑pool → Linear(2880→5) can be trained to predict `code_overall_0_4` (or a small code‑specific head with 4 outputs).
- Judge (sequence‑aware): identical‑layer MTP block appended after the last backbone layer; readout from a <QUALITY> token → Linear(2880→5) or a 4‑axis code head. Train on oracle labels with Huber; optional pairwise ranking on overall.
- KV reuse: If rephrasing/augmentation is requested, reuse the document KV; no re‑prefill.

Metrics (SQLite via existing subsystem)
- doc/sec; ms/doc (probe, judge); f_drop/f_keep/f_escalate; distribution of `zero_score_reason`; acceptance curves; disagreement vs auditor; tokens saved by early‑exit.

Notes
- Reasoning is never persisted—only strict JSON outputs and minimal metadata.
- Keep exact+semantic dedup (emb18) and fidelity checks orthogonal to code scoring (three gates: quality, diversity, fidelity).
