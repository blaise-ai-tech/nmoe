"""Grouped blockscaled GEMM for MoE expert MLPs.

This is the blockscaled analogue of `nmoe/ggemm.py` (BF16 grouped_mm path).
The public surface stays tiny: quantize expert weights once, then run the
expert MLP on RDEP-dispatched packed activations.
"""

from __future__ import annotations

import torch

from nmoe.blockscaled.grouped import QuantizedWeightsFused, expert_blockscaled, quantize_weights


def expert(
  Xe_q_pad: torch.Tensor,
  Xe_sf_pad: torch.Tensor,
  W_cache: QuantizedWeightsFused,
  offs_pad: torch.Tensor,
) -> torch.Tensor:
  """Expert MLP (blockscaled): Y = (SiLU(X @ W1) * (X @ W3)) @ W2.

  Args:
    Xe_q_pad: [M_pad, H_packed, ...] packed quantized activations from RDEP
    Xe_sf_pad: [E, M_e_stride, sf_k_pad] uint8 SFA (MMA layout) from RDEP
    W_cache: QuantizedWeightsFused from quantize_weights(...)
    offs_pad: [E] int32 cumulative padded offsets from RDEP (no leading 0)
  """
  return expert_blockscaled(Xe_q_pad, Xe_sf_pad, W_cache, offs_pad)

