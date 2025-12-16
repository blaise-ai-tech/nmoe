"""
Stub for QuaDMix joint quality‑diversity sampler.

Goal:
- Sample batches/documents optimizing a joint objective over quality scores and
  diversity signals (clusters, distances), as described in PIPELINE.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class QuaDMixConfig:
    weight_quality: float = 1.0
    weight_diversity: float = 0.2
    min_per_cluster: int = 0
    temperature: float = 1.0


def sample_joint(
    items: Iterable[Dict[str, Any]],
    *,
    cfg: QuaDMixConfig,
) -> List[Dict[str, Any]]:
    """Select a subset under a joint quality‑diversity objective (placeholder)."""
    return list(items)

