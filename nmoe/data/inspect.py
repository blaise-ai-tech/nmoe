"""
Shard data inspection utilities.

Provides tools for validating preprocessed data quality:
- Double EOS detection (common preprocessing bug)
- Token distribution analysis
- Vocabulary coverage metrics
- Sample token visualization
"""
from __future__ import annotations

import logging
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np

from .sinks import load_manifest

log = logging.getLogger(__name__)


def inspect_shards(
    *,
    manifest_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    eos_token_id: Optional[int] = None,
    num_shards: int = 1,
    check_all: bool = False,
    verbose: bool = False,
    sample_size: Optional[int] = None,
    compute_stats: bool = False,
) -> int:
    """Inspect shard data for quality issues.

    Args:
        manifest_path: Path to manifest.json (if provided, reads eos_token_id from manifest)
        data_dir: Directory containing .npy shards (used if manifest not provided)
        eos_token_id: EOS token ID (required if manifest not provided)
        num_shards: Number of shards to inspect (default: 1)
        check_all: Inspect all shards (ignores num_shards)
        verbose: Show detailed per-issue output
        sample_size: Number of random tokens to sample per shard
        compute_stats: Compute token distribution statistics

    Returns:
        0 if no issues found, 1 if issues detected
    """
    # Load manifest if provided, otherwise scan directory
    if manifest_path:
        if not manifest_path.exists():
            log.error(f"Manifest not found: {manifest_path}")
            return 1
        manifest = load_manifest(manifest_path)
        data_dir = manifest_path.parent
        shard_pattern = str(data_dir / "*.npy")
    elif data_dir:
        if not data_dir.exists():
            log.error(f"Data directory not found: {data_dir}")
            return 1
        shard_pattern = str(data_dir / "*.npy")
        manifest = None
    else:
        log.error("Either --manifest or --data-dir must be provided")
        return 1

    shards = sorted(glob(shard_pattern))
    if not shards:
        log.error(f"No .npy shards found in {data_dir}")
        return 1

    log.info(f"Found {len(shards)} shards in {data_dir}")
    if manifest:
        log.info(f"Manifest: {manifest.total_documents:,} docs, {manifest.total_tokens:,} tokens")
        eos_token_id = manifest.eos_token_id
    else:
        if eos_token_id is None:
            log.error("--eos-token-id required when --manifest not provided")
            return 1
        log.info(f"No manifest, using --eos-token-id={eos_token_id}")

    # Inspect first N shards (or all if --all)
    shards_to_check = shards if check_all else shards[:num_shards]
    log.info(f"Inspecting {len(shards_to_check)} shard(s)")

    total_double_eos = 0
    total_tokens = 0
    total_eos_tokens = 0
    token_stats = []

    for shard_path in shards_to_check:
        log.info(f"  Loading: {Path(shard_path).name}")
        shard = np.load(shard_path, mmap_mode='r')
        n_tokens = len(shard)
        total_tokens += n_tokens

        # Check for double EOS
        if eos_token_id is not None:
            double_eos = 0
            for i in range(n_tokens - 1):
                if shard[i] == eos_token_id and shard[i+1] == eos_token_id:
                    double_eos += 1
                    if verbose and double_eos <= 3:
                        log.warning(f"    Double EOS at positions {i},{i+1}")

            eos_count = int((shard == eos_token_id).sum())
            total_eos_tokens += eos_count
            total_double_eos += double_eos

            log.info(f"    Tokens: {n_tokens:,}, EOS: {eos_count:,}, Double EOS: {double_eos}")

        # Sample tokens if requested
        if sample_size:
            sample_sz = min(sample_size, n_tokens)
            sample_indices = np.random.choice(n_tokens, sample_sz, replace=False)
            sample_tokens = shard[sample_indices]
            log.info(f"    Sample tokens: {sample_tokens[:10].tolist()}")

        # Token distribution stats
        if compute_stats:
            unique, counts = np.unique(shard, return_counts=True)
            token_stats.append({
                'shard': Path(shard_path).name,
                'unique_tokens': len(unique),
                'total_tokens': n_tokens,
                'vocab_coverage': len(unique) / (manifest.vocab_size if manifest else 201088),
            })

    # Summary
    log.info("")
    log.info("=== Inspection Summary ===")
    log.info(f"Shards inspected: {len(shards_to_check)}")
    log.info(f"Total tokens: {total_tokens:,}")
    if eos_token_id is not None:
        log.info(f"Total EOS tokens: {total_eos_tokens:,}")
        log.info(f"Total double EOS sequences: {total_double_eos}")
        if total_double_eos > 0:
            log.error(f"ISSUE: Found {total_double_eos} double EOS sequences (should be 0)")
        else:
            log.info("✓ No double EOS sequences found")

    if compute_stats and token_stats:
        log.info("")
        log.info("=== Token Distribution ===")
        for stat in token_stats:
            log.info(f"  {stat['shard']}: {stat['unique_tokens']:,} unique tokens, {stat['vocab_coverage']:.1%} vocab coverage")

    return 1 if total_double_eos > 0 else 0
