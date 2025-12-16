"""Smoke + microbench for blockscaled grouped MoE expert ops (single GPU).

Compares against the BF16 grouped_mm path:
  - BF16 expert MLP: nmoe.ggemm.expert
  - Blockscaled expert MLP: nmoe.blockscaled.ggemm.expert (FP8 / NVFP4)

Usage (on debug pod):
  python -m nmoe.test_blockscaled_grouped
"""

from __future__ import annotations

import time

import torch

from nmoe import ggemm
from nmoe.csrc import rdep
from nmoe.quant import quantize_fp8, quantize_nvfp4
from nmoe.blockscaled.grouped import quantize_weights
from nmoe.blockscaled.ggemm import expert as bs_expert


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


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


def _packed_u16_from_fp8(x: torch.Tensor) -> torch.Tensor:
  # quantize_fp8 returns [M, H, 1] float8_e4m3fn backed by uint16 storage.
  if x.dtype != torch.float8_e4m3fn or x.ndim != 3 or x.shape[-1] != 1:
    raise TypeError(f"expected fp8 [M,H,1], got dtype={x.dtype} shape={tuple(x.shape)}")
  M, H, _ = map(int, x.shape)
  return x.view(torch.uint8).view(M, H).view(torch.uint16)


def _packed_u16_from_nvfp4(x: torch.Tensor) -> torch.Tensor:
  # quantize_nvfp4 returns [M, H//2, 1] uint8 backed by uint16 storage.
  if x.dtype != torch.uint8 or x.ndim != 3 or x.shape[-1] != 1:
    raise TypeError(f"expected nvfp4 [M,H//2,1] uint8, got dtype={x.dtype} shape={tuple(x.shape)}")
  M, Hp, _ = map(int, x.shape)
  return x.view(torch.uint8).view(M, Hp).view(torch.uint16)


@torch.no_grad()
def main() -> None:
  torch.manual_seed(0)
  torch.cuda.set_device(0)
  dev = torch.device("cuda")

  # Keep sizes aligned for all kernels.
  E = 8
  H = 1024    # multiple of 64
  Dff = 4096  # multiple of 32; 2*Dff multiple of 64
  M_per = 1024  # per-expert rows, multiple of 128
  M_pad = E * M_per

  offs = torch.arange(0, M_pad + 1, step=M_per, device=dev, dtype=torch.int32).contiguous()  # [E+1]
  offs_pad = offs[1:].contiguous()  # [E]

  # Inputs are already expert-sorted (one contiguous block per expert).
  x = torch.randn(M_pad, H, device=dev, dtype=torch.bfloat16).contiguous()

  # Expert weights.
  W1 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16).contiguous()
  W3 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16).contiguous()
  W2 = torch.randn(E, Dff, H, device=dev, dtype=torch.bfloat16).contiguous()

  # BF16 reference.
  y_ref = ggemm.expert(x, W1, W3, W2, offs_pad)

  flops = 6.0 * float(M_pad) * float(H) * float(Dff)  # 2*(H1+H3+Y) = 6*M*H*Dff
  ms_ref = _bench_ms(lambda: ggemm.expert(x, W1, W3, W2, offs_pad))
  tflops_ref = flops / (ms_ref * 1e-3) / 1e12

  print(f"== grouped expert MLP: E={E} M_per={M_per} (M_pad={M_pad}) H={H} Dff={Dff} ==")
  print(f"bf16 grouped_mm: {ms_ref:.3f} ms  {tflops_ref:.2f} TF/s")

  for profile in ("fp8", "nvfp4"):
    if profile == "fp8":
      x_q, x_sf = quantize_fp8(x)
      Xe_q_pad_u16 = _packed_u16_from_fp8(x_q)
    else:
      x_q, x_sf = quantize_nvfp4(x)
      Xe_q_pad_u16 = _packed_u16_from_nvfp4(x_q)

    # Swizzle SFA from MKL row-major to per-expert strided MMA layout.
    x_sf_mkl = x_sf.squeeze(-1).contiguous()
    sf_k = int(x_sf_mkl.shape[1])
    sf_k_pad = _ceil_div(sf_k, 4) * 4
    Xe_sf_pad = torch.empty((E, M_per, sf_k_pad), device=dev, dtype=torch.uint8)
    rdep.swizzle_sf_strided(
      x_sf_mkl.data_ptr(), Xe_sf_pad.data_ptr(),
      offs.data_ptr(), E, sf_k, sf_k_pad, M_pad, M_per, torch.cuda.current_stream(dev)
    )

    # Quantize weights once per profile.
    W_cache = quantize_weights(W1, W3, W2, profile=profile)

    def run_bs() -> torch.Tensor:
      return bs_expert(Xe_q_pad_u16, Xe_sf_pad, W_cache, offs_pad)

    y = run_bs()
    err = (y - y_ref).float()
    rms_err = _rms(err)
    rms_ref = _rms(y_ref.float())
    rel = rms_err / max(rms_ref, 1e-8)

    ms = _bench_ms(run_bs)
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{profile} blockscaled: {ms:.3f} ms  {tflops:.2f} TF/s  rms={rms_err:.4e}  rel_rms={rel:.4e}  max_abs={float(err.abs().max().item()):.4e}")


if __name__ == "__main__":
  t0 = time.time()
  main()
  torch.cuda.synchronize()
  print(f"\nOK in {time.time() - t0:.2f}s")

