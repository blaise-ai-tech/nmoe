"""
Stubs for fidelity verification utilities.

Planned features:
- Embedding cosine similarity using L18 pooled hidden (emb18) from HYDRA backbone.
- LLM verify step for borderline cosine scores (e.g., Harmony final‑only prompt).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def cosine_emb18(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between two batches of emb18 vectors (placeholder).

    Args:
        emb_a: [B, H]
        emb_b: [B, H]

    Returns:
        [B] cosine similarity in [-1, 1].
    """
    if emb_a.ndim != 2 or emb_b.ndim != 2 or emb_a.shape != emb_b.shape:
        raise ValueError("embeddings must be [B,H] and same shape")
    a = torch.nn.functional.normalize(emb_a, dim=-1)
    b = torch.nn.functional.normalize(emb_b, dim=-1)
    return (a * b).sum(dim=-1)


def llm_verify(
    original_text: str,
    rephrased_text: str,
    *,
    checkpoint: str,
    max_new: int = 256,
) -> Tuple[bool, Dict[str, Any]]:
    """Borderline fidelity verification with an LLM (placeholder).

    Returns:
        (pass_flag, details)
    """
    raise NotImplementedError("fidelity.llm_verify stub not implemented yet")

