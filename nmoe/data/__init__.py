"""
nmoe.data - Data loading, preprocessing, inference, and mixture utilities.

Primary exports:
- build_loader: Main entry point for creating data loaders
- DeterministicLoader: Production loader with SWRR mixing
- MixturePlan: Data structure for mixture configuration

Preprocessing:
- PrepPipeline: Local preprocessing pipeline
- ParallelPrepPipeline: Multiprocess preprocessing
- PrepConfig: Pipeline configuration
- HuggingFaceSource: Stream from HuggingFace Hub

Index/Shard utilities:
- IndexReader, IndexWriter: Document boundary indices
- ShardedWriter: Write .npy + .idx shards
- ManifestInfo: Dataset manifest
"""
# Mixture (no torch dependency)
from .mixture import MixturePlan, StagePlan, SourcePlan, HFSource, resolve_plan, populate_paths

# Index (no torch dependency)
from .index import IndexReader, IndexWriter, regenerate_index_from_shard

# Sources (no torch dependency)
from .sources import (
    DataSource,
    Document,
    HuggingFaceSource,
    JSONLSource,
    TextFileSource,
    ArrowSource,
    create_source,
)

# Sinks (no torch dependency)
from .sinks import (
    ShardWriter,
    ShardedWriter,
    ShardInfo,
    ManifestInfo,
    load_manifest,
    verify_manifest,
)

# Preprocessing (no torch dependency)
from .prep import PrepConfig, PrepPipeline, ParallelPrepPipeline

# Transforms (no torch dependency)
from .transforms import normalize_text, tokenize, pack_document, get_shard_id, make_file_name


# Lazy imports for torch-dependent modules
def __getattr__(name):
    if name in ("build_loader", "DeterministicLoader"):
        from .loader import build_loader, DeterministicLoader
        return {"build_loader": build_loader, "DeterministicLoader": DeterministicLoader}[name]
    if name in ("NumpyFSLDataset", "Cursor"):
        from .dataset import NumpyFSLDataset, Cursor
        return {"NumpyFSLDataset": NumpyFSLDataset, "Cursor": Cursor}[name]
    if name in ("format_prompt", "create_parser", "stop_tokens", "sample_top_p"):
        from .rephrase import format_prompt, create_parser, stop_tokens, sample_top_p
        return {"format_prompt": format_prompt, "create_parser": create_parser,
                "stop_tokens": stop_tokens, "sample_top_p": sample_top_p}[name]
    if name in ("build_prompt", "right_trim", "grade_prompts", "compute_aggregated", "compute_aggregated_code"):
        from .score import build_prompt, right_trim, grade_prompts, compute_aggregated, compute_aggregated_code
        return {"build_prompt": build_prompt, "right_trim": right_trim, "grade_prompts": grade_prompts,
                "compute_aggregated": compute_aggregated, "compute_aggregated_code": compute_aggregated_code}[name]
    if name in ("minhash_signature", "jaccard_from_signature", "stream_near_dedup", "LSHIndex",
                "exact_hash", "dedup_exact", "dedup_line_level", "dedup_paragraph_level"):
        from .dedup import (
            minhash_signature,
            jaccard_from_signature,
            stream_near_dedup,
            LSHIndex,
            exact_hash,
            dedup_exact,
            dedup_line_level,
            dedup_paragraph_level,
        )
        return {
            "minhash_signature": minhash_signature,
            "jaccard_from_signature": jaccard_from_signature,
            "stream_near_dedup": stream_near_dedup,
            "LSHIndex": LSHIndex,
            "exact_hash": exact_hash,
            "dedup_exact": dedup_exact,
            "dedup_line_level": dedup_line_level,
            "dedup_paragraph_level": dedup_paragraph_level,
        }[name]
    if name in ("cluster_embeddings", "coverage_metrics"):
        from .diversity import cluster_embeddings, coverage_metrics
        return {"cluster_embeddings": cluster_embeddings, "coverage_metrics": coverage_metrics}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Loader
    "build_loader",
    "DeterministicLoader",
    # Mixture
    "MixturePlan",
    "resolve_plan",
    "populate_paths",
    # Dataset
    "NumpyFSLDataset",
    "Cursor",
    # Index
    "IndexReader",
    "IndexWriter",
    "regenerate_index_from_shard",
    # Sources
    "DataSource",
    "Document",
    "HuggingFaceSource",
    "JSONLSource",
    "TextFileSource",
    "ArrowSource",
    "create_source",
    # Sinks
    "ShardWriter",
    "ShardedWriter",
    "ShardInfo",
    "ManifestInfo",
    "load_manifest",
    "verify_manifest",
    # Preprocessing
    "PrepConfig",
    "PrepPipeline",
    "ParallelPrepPipeline",
    # Transforms
    "normalize_text",
    "tokenize",
    "pack_document",
    "get_shard_id",
    "make_file_name",
    # Scoring
    "build_prompt",
    "right_trim",
    "grade_prompts",
    "compute_aggregated",
    "compute_aggregated_code",
    # Rephrasing
    "format_prompt",
    "create_parser",
    "stop_tokens",
    "sample_top_p",
    # Dedup
    "minhash_signature",
    "jaccard_from_signature",
    "stream_near_dedup",
    "LSHIndex",
    "exact_hash",
    "dedup_exact",
    "dedup_line_level",
    "dedup_paragraph_level",
    # Diversity
    "cluster_embeddings",
    "coverage_metrics",
]
