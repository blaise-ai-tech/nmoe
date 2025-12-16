"""Smoke + microbench for blockscaled dense ops (single GPU).

Runs accuracy + perf checks versus BF16 baselines:
  - dense linear: vs torch.nn.functional.linear
  - dense MLP: vs BF16 MLP built from F.linear

Usage (on debug pod):
  python -m nmoe.test_blockscaled_dense
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from nmoe.blockscaled.dense import (
  linear as bs_linear,
  mlp as bs_mlp,
  quantize_linear_weight,
  quantize_mlp_weights,
)


def _rms(x: torch.Tensor) -> float:
  return float(x.pow(2).mean().sqrt().item())


def _bench_ms(fn, *, warmup: int = 20, iters: int = 100) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(iters):
    fn()
  end.record()
  end.synchronize()
  return float(start.elapsed_time(end)) / float(iters)


@torch.no_grad()
def main() -> None:
  torch.manual_seed(0)
  torch.cuda.set_device(0)
  dev = torch.device("cuda")

  # Keep sizes small-ish but aligned for the blockscaled kernels.
  M = 2048   # multiple of 128
  K = 1024   # multiple of 32
  N = 4096   # multiple of 64
  H = K
  Dff = 4096  # 2*Dff multiple of 64

  x = torch.randn(M, K, device=dev, dtype=torch.bfloat16).contiguous()
  w = torch.randn(N, K, device=dev, dtype=torch.bfloat16).contiguous()

  # ---------------------------------------------------------------------------
  # Dense linear
  # ---------------------------------------------------------------------------
  y_ref = F.linear(x, w)

  print(f"== dense linear: M={M} K={K} N={N} ==")
  ms_ref = _bench_ms(lambda: F.linear(x, w))
  print(f"bf16 F.linear: {ms_ref:.3f} ms")

  for profile in ("fp8", "nvfp4"):
    Wq = quantize_linear_weight(w, profile=profile)
    y = bs_linear(x, Wq)

    err = (y - y_ref).float()
    rms_err = _rms(err)
    rms_ref = _rms(y_ref.float())
    rel = rms_err / max(rms_ref, 1e-8)

    ms = _bench_ms(lambda: bs_linear(x, Wq))
    print(f"{profile} blockscaled: {ms:.3f} ms  rms={rms_err:.4e}  rel_rms={rel:.4e}  max_abs={float(err.abs().max().item()):.4e}")

  # ---------------------------------------------------------------------------
  # Dense MLP
  # ---------------------------------------------------------------------------
  W1 = torch.randn(Dff, H, device=dev, dtype=torch.bfloat16).contiguous()
  W3 = torch.randn(Dff, H, device=dev, dtype=torch.bfloat16).contiguous()
  W2 = torch.randn(H, Dff, device=dev, dtype=torch.bfloat16).contiguous()

  def mlp_ref() -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, W1)).mul_(F.linear(x, W3)), W2)

  y_ref = mlp_ref()

  print(f"\n== dense MLP: M={M} H={H} Dff={Dff} ==")
  ms_ref = _bench_ms(mlp_ref)
  print(f"bf16 MLP: {ms_ref:.3f} ms")

  for profile in ("fp8", "nvfp4"):
    Wq = quantize_mlp_weights(W1, W3, W2, profile=profile)

    def mlp_bs() -> torch.Tensor:
      return bs_mlp(x, Wq)

    y = mlp_bs()
    err = (y - y_ref).float()
    rms_err = _rms(err)
    rms_ref = _rms(y_ref.float())
    rel = rms_err / max(rms_ref, 1e-8)

    ms = _bench_ms(mlp_bs)
    print(f"{profile} blockscaled: {ms:.3f} ms  rms={rms_err:.4e}  rel_rms={rel:.4e}  max_abs={float(err.abs().max().item()):.4e}")


if __name__ == "__main__":
  t0 = time.time()
  main()
  torch.cuda.synchronize()
  print(f"\nOK in {time.time() - t0:.2f}s")

