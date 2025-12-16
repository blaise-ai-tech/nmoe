"""
Grouped GEMM for MoE expert MLPs: Y = (SiLU(X @ W1) * (X @ W3)) @ W2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def expert(
  Xe_pad: torch.Tensor,
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  offs_pad: torch.Tensor,
) -> torch.Tensor:
  """Expert MLP: Y = (SiLU(X @ W1) * (X @ W3)) @ W2

  BF16 path using torch._grouped_mm.

  Args:
    Xe_pad: [M_pad, H] pre-padded BF16 input from rdep.dispatch_sorted
    W1, W3: [E, H, Dff] gate/up weights
    W2: [E, Dff, H] down weight
    offs_pad: [E] cumulative padded offsets from rdep

  Returns:
    [M_pad, H] BF16 output (caller uses dest to select valid rows)
  """
  if Xe_pad.size(0) == 0:
    return Xe_pad

  H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
  H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
  return torch._grouped_mm(F.silu(H1).mul_(H3), W2, offs=offs_pad)


def expert_mlp_bf16(
  Xe: torch.Tensor,
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  offsets: torch.Tensor,
  align: int = 128,  # Match blockscaled for consistent padding
) -> torch.Tensor:
  """
  Expert MLP: Y = (SiLU(X @ W1) * (X @ W3)) @ W2

  Args:
    Xe: [M, H] input activations sorted by expert
    W1, W3: [E, H, Dff] gate/up weights
    W2: [E, Dff, H] down weight
    offsets: [E+1] expert boundaries (offsets[0]=0, offsets[E]=M)
    align: row alignment for grouped_mm

  Returns:
    [M, H] output
  """
  M = Xe.size(0)
  if M == 0:
    return Xe.new_zeros(0, Xe.size(1))

  E = offsets.numel() - 1
  H = Xe.size(1)
  dev = Xe.device

  cnt = (offsets[1:] - offsets[:-1]).long()
  cnt_pad = ((cnt + align - 1) // align) * align
  offs_pad = torch.cumsum(cnt_pad.int(), dim=0, dtype=torch.int32)

  starts_real = torch.zeros(E, device=dev, dtype=torch.long)
  starts_pad = torch.zeros(E, device=dev, dtype=torch.long)
  if E > 1:
    starts_real[1:] = cnt[:-1].cumsum(0)
    starts_pad[1:] = cnt_pad[:-1].cumsum(0)

  ar = torch.arange(M, device=dev, dtype=torch.long)
  eid = torch.repeat_interleave(torch.arange(E, device=dev), cnt)
  pos = ar - starts_real[eid]
  dest = starts_pad[eid] + pos

  M_pad = cnt_pad.sum().item()
  Xe_pad = Xe.new_zeros(M_pad, H)
  Xe_pad.index_copy_(0, dest, Xe)

  H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
  H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
  Yp = torch._grouped_mm(F.silu(H1).mul_(H3), W2, offs=offs_pad)

  return Yp.index_select(0, dest)
