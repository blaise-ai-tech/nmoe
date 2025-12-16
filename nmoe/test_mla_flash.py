"""Correctness + microbench for MLA attention (FA4 fwd + FlashMLA bwd) on SM100.

This targets the training-critical shape:
  - dtype: bf16
  - causal, full-seq, fixed-length packed
  - (d_qk, d_v) = (192, 128)

Usage:
  python -m nmoe.test_mla_flash
  python -m nmoe.test_mla_flash --bench

Memory safety (tiny case):
  compute-sanitizer --tool memcheck python -m nmoe.test_mla_flash --tiny
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from nmoe.attention.mla import _MlaFa4FwdFlashMlaBwd


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise RuntimeError(msg)


def _sm100_only() -> None:
  _require(torch.cuda.is_available(), "CUDA required (B200 / SM100).")
  major, minor = torch.cuda.get_device_capability()
  _require(major == 10, f"SM100 required. Got compute capability {major}.{minor}.")


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


def _ref_grads_bf16(
  q_bshd: torch.Tensor,
  k_bshd: torch.Tensor,
  v_bshd: torch.Tensor,
  d_out_bshd: torch.Tensor,
  *,
  softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # Reference: explicit FP32 matmul -> causal mask -> softmax -> matmul.
  bsz, seqlen, n_heads, _ = q_bshd.shape
  q = q_bshd.transpose(1, 2)  # [B,H,S,Dq]
  k = k_bshd.transpose(1, 2)  # [B,H,S,Dq]
  v = v_bshd.transpose(1, 2)  # [B,H,S,Dv]
  d_out = d_out_bshd.transpose(1, 2)  # [B,H,S,Dv]

  scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * softmax_scale
  mask = torch.triu(torch.ones((seqlen, seqlen), device=scores.device, dtype=torch.bool), diagonal=1)
  scores = scores.masked_fill(mask, float("-inf"))
  p = torch.softmax(scores, dim=-1)
  out = torch.matmul(p, v.float())
  loss = (out * d_out.float()).sum()
  dq, dk, dv = torch.autograd.grad(loss, (q_bshd, k_bshd, v_bshd))
  return dq, dk, dv


def _stats(name: str, got: torch.Tensor, ref: torch.Tensor) -> str:
  diff = (got - ref).abs()
  return f"{name} max {diff.max().item():.6f} mean {diff.mean().item():.6f}"


def _run_case(*, bsz: int, seqlen: int, n_heads: int, seed: int) -> None:
  torch.manual_seed(seed)
  d_qk = 192
  d_v = 128
  softmax_scale = 1.0 / math.sqrt(d_qk)

  q = torch.randn((bsz, seqlen, n_heads, d_qk), device="cuda", dtype=torch.bfloat16, requires_grad=True)
  k = torch.randn((bsz, seqlen, n_heads, d_qk), device="cuda", dtype=torch.bfloat16, requires_grad=True)
  v = torch.randn((bsz, seqlen, n_heads, d_v), device="cuda", dtype=torch.bfloat16, requires_grad=True)
  d_out = torch.randn((bsz, seqlen, n_heads, d_v), device="cuda", dtype=torch.bfloat16)

  dq_ref, dk_ref, dv_ref = _ref_grads_bf16(q, k, v, d_out, softmax_scale=softmax_scale)

  out = _MlaFa4FwdFlashMlaBwd.apply(q, k, v, softmax_scale)
  loss = (out * d_out).sum()
  dq, dk, dv = torch.autograd.grad(loss, (q, k, v))

  print(f"B={bsz} S={seqlen} H={n_heads}: {_stats('dq', dq, dq_ref)} | {_stats('dk', dk, dk_ref)} | {_stats('dv', dv, dv_ref)}")

  # Tight, BF16-appropriate tolerances (empirically: max is typically 1-4 BF16 quanta).
  _require(float((dq - dq_ref).abs().max().item()) <= 0.0625, "dq max abs diff too large")
  _require(float((dk - dk_ref).abs().max().item()) <= 0.0625, "dk max abs diff too large")
  _require(float((dv - dv_ref).abs().max().item()) <= 0.0625, "dv max abs diff too large")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--bench", action="store_true", help="Run a small perf microbench.")
  ap.add_argument("--tiny", action="store_true", help="Run a single tiny case (useful for compute-sanitizer).")
  args = ap.parse_args()

  _sm100_only()
  torch.cuda.set_device(0)

  if args.tiny:
    _run_case(bsz=1, seqlen=33, n_heads=2, seed=0)
    return

  # Correctness matrix (includes odd seqlens to exercise tail predicates).
  for (bsz, seqlen, n_heads) in [
    (1, 1, 8),
    (1, 127, 8),
    (2, 129, 4),
    (2, 257, 8),
    (4, 64, 16),
  ]:
    _run_case(bsz=bsz, seqlen=seqlen, n_heads=n_heads, seed=123)

  if args.bench:
    bsz, seqlen, n_heads = 8, 4096, 16
    d_qk, d_v = 192, 128
    softmax_scale = 1.0 / math.sqrt(d_qk)
    q = torch.randn((bsz, seqlen, n_heads, d_qk), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((bsz, seqlen, n_heads, d_qk), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((bsz, seqlen, n_heads, d_v), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    d_out = torch.randn((bsz, seqlen, n_heads, d_v), device="cuda", dtype=torch.bfloat16)

    def fwd():
      return _MlaFa4FwdFlashMlaBwd.apply(q, k, v, softmax_scale)

    def bwd():
      out = fwd()
      loss = (out * d_out).sum()
      torch.autograd.grad(loss, (q, k, v))

    ms_fwd = _bench_ms(lambda: fwd(), warmup=10, iters=50)
    ms_bwd = _bench_ms(lambda: bwd(), warmup=5, iters=20)
    print(f"\nbench B={bsz} S={seqlen} H={n_heads}: fwd {ms_fwd:.3f} ms | fwd+bwd {ms_bwd:.3f} ms")


if __name__ == "__main__":
  t0 = time.time()
  main()
  torch.cuda.synchronize()
  print(f"\nOK in {time.time() - t0:.2f}s")
