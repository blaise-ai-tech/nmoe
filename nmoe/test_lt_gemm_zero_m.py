"""Regression: cuBLASLt wgrad must zero m==0 experts.

Blockscaled MoE backward uses cuBLASLt grouped GEMMs to compute BF16 wgrads for
expert weights. When an expert receives zero tokens in a step (m==0), its
gradient slice must be exactly zero; leaving it uninitialized can inject NaNs
into optimizer state and crash training.

Usage:
  python -m nmoe.test_lt_gemm_zero_m
"""

from __future__ import annotations

import torch

from nmoe.csrc import rdep as _C


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise RuntimeError(msg)


def _prefix_offsets(ms: list[int]) -> torch.Tensor:
  offs: list[int] = []
  s = 0
  for m in ms:
    _require(m >= 0, "m must be non-negative")
    s += int(m)
    offs.append(s)
  return torch.tensor(offs, dtype=torch.int32, device='cpu')


def _require_zero_slice(x: torch.Tensor, e: int) -> None:
  sl = x[e]
  _require(torch.isfinite(sl).all().item(), f"expert {e} slice has non-finite values")
  if (sl != 0).any().item():
    max_abs = float(sl.float().abs().max().item())
    raise RuntimeError(f"expert {e} slice not zero (max_abs={max_abs})")


def main() -> None:
  _require(torch.cuda.is_available(), "CUDA required")

  dev = torch.device('cuda')
  torch.manual_seed(0)

  E = 5
  H = 128
  Dff = 256
  ms = [0, 37, 0, 13, 0]  # leading/middle/trailing zero-token experts
  offs = _prefix_offsets(ms)
  M = int(offs[-1].item())
  _require(M > 0, "Need at least one token row")

  stream = torch.cuda.current_stream(dev)

  X = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
  dH = torch.randn(M, Dff, device=dev, dtype=torch.bfloat16)
  dW13 = torch.full((E, H, Dff), float('nan'), device=dev, dtype=torch.bfloat16)
  _C.bf16_wgrad_w13_cublaslt(
      X.data_ptr(),
      dH.data_ptr(),
      dW13.data_ptr(),
      offs.data_ptr(),
      int(E), int(H), int(Dff),
      stream,
  )

  A = torch.randn(M, Dff, device=dev, dtype=torch.bfloat16)
  dY = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
  dW2 = torch.full((E, Dff, H), float('nan'), device=dev, dtype=torch.bfloat16)
  _C.bf16_wgrad_w2_cublaslt(
      A.data_ptr(),
      dY.data_ptr(),
      dW2.data_ptr(),
      offs.data_ptr(),
      int(E), int(H), int(Dff),
      stream,
  )

  torch.cuda.synchronize(dev)

  zero_es = [i for i, m in enumerate(ms) if m == 0]
  _require(len(zero_es) > 0, "Test must include m==0 experts")
  for e in zero_es:
    _require_zero_slice(dW13, e)
    _require_zero_slice(dW2, e)

  _require(torch.isfinite(dW13).all().item(), "dW13 has non-finite values")
  _require(torch.isfinite(dW2).all().item(), "dW2 has non-finite values")

  print("OK: zero-token experts produce zero wgrad slices")


if __name__ == '__main__':
  main()

