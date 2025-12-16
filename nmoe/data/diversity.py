"""
Diversity analysis utilities: clustering and coverage metrics.

Implements MiniBatch K-Means clustering in pure PyTorch/NumPy (no sklearn
dependency) with deterministic seeding. Suitable for large embedding arrays
via minibatches. For very large corpora consider FAISS/RAFT for acceleration.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

@torch.no_grad()
def cluster_embeddings(
    emb: torch.Tensor,
    k: int,
    *,
    seed: int = 42,
    iters: int = 50,
    batch_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MiniBatch K-Means on embeddings.

    Args:
        emb: [N, H] float tensor (cpu or cuda)
        k: number of clusters
        seed: RNG seed for reproducible init
        iters: number of minibatch iterations
        batch_size: minibatch size

    Returns:
        labels [N] (cpu), centroids [k, H] (same dtype/device as emb)
    """
    assert emb.ndim == 2 and k > 0
    N, H = int(emb.shape[0]), int(emb.shape[1])
    device = emb.device
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # k-means++ style seeding (simplified): pick first random, then pick farthest
    idx0 = torch.randint(low=0, high=N, size=(1,), generator=gen).item()
    centroids = emb[idx0 : idx0 + 1].clone()
    while centroids.shape[0] < k:
        # compute distance to nearest existing centroid on a sample to save time
        sample_idx = torch.randint(0, N, (min(8192, N),), generator=gen)
        x = emb.index_select(0, sample_idx)
        dists = torch.cdist(x, centroids, p=2)  # [S, C]
        min_d, _ = dists.min(dim=1)
        farthest = sample_idx[min_d.argmax()]
        centroids = torch.cat([centroids, emb[farthest : farthest + 1]], dim=0)

    # Running sums and counts per centroid for online updates
    sums = torch.zeros_like(centroids)
    counts = torch.zeros(k, device=device, dtype=torch.long)

    for _ in range(max(1, iters)):
        # Sample a minibatch
        bidx = torch.randint(0, N, (min(batch_size, N),), generator=gen)
        x = emb.index_select(0, bidx)
        # Assign to nearest centroid
        d = torch.cdist(x, centroids, p=2)  # [B, k]
        assign = d.argmin(dim=1)  # [B]
        # Online updates per centroid
        for c in range(k):
            mask = assign == c
            if mask.any():
                xb = x[mask]
                sums[c] += xb.sum(dim=0)
                counts[c] += mask.sum()
                centroids[c] = sums[c] / counts[c].clamp_min(1)

    # Final full assignment (can be heavy but accurate)
    labels = torch.cdist(emb, centroids, p=2).argmin(dim=1).to("cpu")
    return labels, centroids


def coverage_metrics(labels: torch.Tensor, k: int) -> Dict[str, float]:
    """Compute coverage/balance metrics from cluster labels.

    Returns:
      - cluster_coverage: fraction of non-empty clusters
      - entropy: normalized Shannon entropy in [0,1]
      - gini: Gini coefficient of the distribution
    """
    counts = torch.bincount(labels.to(torch.long), minlength=k).float()
    total = counts.sum().item()
    nonempty = (counts > 0).float().sum().item()
    coverage = nonempty / max(1, k)
    if total <= 0:
        return {"cluster_coverage": float(coverage), "entropy": 0.0, "gini": 0.0}
    p = (counts / total).clamp_min(1e-12)
    entropy = float((-p * p.log()).sum() / torch.log(torch.tensor(float(k))))
    # Gini: sum_i sum_j |p_i - p_j| / (2k * mean)
    diffs = torch.abs(p.unsqueeze(0) - p.unsqueeze(1)).mean().item()
    gini = float(diffs / (2.0 / k))
    return {"cluster_coverage": float(coverage), "entropy": float(entropy), "gini": float(gini)}
