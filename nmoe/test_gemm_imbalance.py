"""Test blockscaled GEMM numerics with imbalanced expert loads.

This tests whether the grouped GEMM kernels produce correct results when
expert loads are heavily imbalanced (simulating router collapse scenario).

Usage:
  python -m nmoe.test_gemm_imbalance
"""

from __future__ import annotations

import torch

from nmoe import ggemm
from nmoe.quant import quantize_fp8, quantize_nvfp4
from nmoe.blockscaled.grouped import quantize_weights
from nmoe.blockscaled.ggemm import expert as bs_expert
from nmoe.csrc import rdep


def _rms(x: torch.Tensor) -> float:
  return float(x.pow(2).mean().sqrt().item())


def _stats(x: torch.Tensor, name: str) -> None:
  """Print tensor statistics."""
  if x.numel() == 0:
    print(f"  {name}: empty")
    return
  xf = x.float()
  print(f"  {name}: shape={tuple(x.shape)} min={xf.min():.4f} max={xf.max():.4f} "
        f"mean={xf.mean():.4f} std={xf.std():.4f} nan={torch.isnan(xf).sum().item()} inf={torch.isinf(xf).sum().item()}")


def _packed_u16_from_fp8(x: torch.Tensor) -> torch.Tensor:
  M, H, _ = map(int, x.shape)
  return x.view(torch.uint8).view(M, H).view(torch.uint16)


def _packed_u16_from_nvfp4(x: torch.Tensor) -> torch.Tensor:
  M, Hp, _ = map(int, x.shape)
  return x.view(torch.uint8).view(M, Hp).view(torch.uint16)


@torch.no_grad()
def test_imbalanced_load(profile: str = "fp8") -> None:
  """Test GEMM with heavily imbalanced expert loads."""
  torch.manual_seed(42)
  torch.cuda.set_device(0)
  dev = torch.device("cuda")

  E = 8       # 8 experts
  H = 2048    # hidden dim (matches moonlet)
  Dff = 1408  # expert FFN dim (matches moonlet moe_inter_dim)

  # Simulate heavily imbalanced load: expert 0 gets 14%, others share rest
  # Total tokens ~ 3000 (similar to batch_size=8 * seq_len=4096 / 8 experts * top_k=6)
  total_tokens = 3072

  # Imbalanced: expert 0 gets 50%, expert 1 gets 25%, others get ~4% each
  loads = [1536, 768, 128, 128, 128, 128, 128, 128]  # sum = 3072

  print(f"\n=== Imbalanced Load Test ({profile}) ===")
  print(f"E={E} H={H} Dff={Dff}")
  print(f"Loads per expert: {loads}")
  print(f"Load %: {[f'{100*l/sum(loads):.1f}%' for l in loads]}")

  # Build offsets
  offs_list = [0]
  for l in loads:
    offs_list.append(offs_list[-1] + l)
  offs = torch.tensor(offs_list, device=dev, dtype=torch.int32)
  offs_pad = offs[1:].contiguous()
  M_pad = sum(loads)

  # Pad to 128 alignment per expert (as RDEP does)
  loads_padded = [((l + 127) // 128) * 128 for l in loads]
  offs_padded_list = [0]
  for l in loads_padded:
    offs_padded_list.append(offs_padded_list[-1] + l)
  offs_padded = torch.tensor(offs_padded_list[1:], device=dev, dtype=torch.int32)
  M_pad_aligned = sum(loads_padded)

  print(f"Padded loads: {loads_padded}")
  print(f"M_pad={M_pad} M_pad_aligned={M_pad_aligned}")

  # Generate inputs with realistic magnitude
  x = torch.randn(M_pad_aligned, H, device=dev, dtype=torch.bfloat16) * 0.1

  # Zero out padding regions
  idx = 0
  for i, (real, padded) in enumerate(zip(loads, loads_padded)):
    if real < padded:
      x[idx + real:idx + padded] = 0
    idx += padded

  # Expert weights (normal init like in model)
  W1 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16) * 0.02
  W3 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16) * 0.02
  W2 = torch.randn(E, Dff, H, device=dev, dtype=torch.bfloat16) * 0.02

  print("\nInputs:")
  _stats(x, "x")
  _stats(W1, "W1")
  _stats(W2, "W2")
  _stats(W3, "W3")

  # BF16 reference
  print("\n--- BF16 Reference ---")
  y_ref = ggemm.expert(x, W1, W3, W2, offs_padded)
  _stats(y_ref, "y_ref")

  # Check per-expert outputs
  idx = 0
  for i, (real, padded) in enumerate(zip(loads, loads_padded)):
    expert_out = y_ref[idx:idx + real]
    nan_count = torch.isnan(expert_out).sum().item()
    inf_count = torch.isinf(expert_out).sum().item()
    if nan_count > 0 or inf_count > 0:
      print(f"  Expert {i}: NaN={nan_count} Inf={inf_count} (load={real})")
    idx += padded

  # Blockscaled
  print(f"\n--- Blockscaled ({profile}) ---")

  # Quantize weights
  W_cache = quantize_weights(W1, W3, W2, profile=profile)

  # Quantize activations
  if profile == "fp8":
    x_q, x_sf = quantize_fp8(x)
    Xe_q_pad_u16 = _packed_u16_from_fp8(x_q)
  else:
    x_q, x_sf = quantize_nvfp4(x)
    Xe_q_pad_u16 = _packed_u16_from_nvfp4(x_q)

  # Build SFA in MMA layout using proper swizzle kernel
  M_e_stride = ((max(loads_padded) + 127) // 128) * 128
  sf_k = H // 32
  sf_k_pad = ((sf_k + 3) // 4) * 4
  Xe_sf_mma = torch.zeros(E, M_e_stride, sf_k_pad, device=dev, dtype=torch.uint8)

  # Swizzle SFA from MKL row-major to per-expert strided MMA layout
  # x_sf has shape [M, sf_k, 1] - squeeze the last dim
  x_sf_mkl = x_sf.squeeze(-1).contiguous() if x_sf.ndim == 3 else x_sf.contiguous()
  # offs for swizzle needs leading 0 - rebuild it
  offs_with0 = torch.cat([torch.zeros(1, device=dev, dtype=torch.int32), offs_padded])
  rdep.swizzle_sf_strided(
    x_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
    offs_with0.data_ptr(), E, sf_k, sf_k_pad, M_pad_aligned, M_e_stride,
    torch.cuda.current_stream(dev)
  )

  # Run blockscaled expert
  y_bs = bs_expert(Xe_q_pad_u16, Xe_sf_mma, W_cache, offs_padded)
  _stats(y_bs, "y_bs")

  # Check per-expert outputs
  idx = 0
  for i, (real, padded) in enumerate(zip(loads, loads_padded)):
    expert_out = y_bs[idx:idx + real]
    nan_count = torch.isnan(expert_out).sum().item()
    inf_count = torch.isinf(expert_out).sum().item()
    rms = _rms(expert_out) if expert_out.numel() > 0 else 0
    if nan_count > 0 or inf_count > 0 or rms > 10:
      print(f"  Expert {i}: NaN={nan_count} Inf={inf_count} rms={rms:.4f} (load={real})")
    idx += padded

  # Compare
  diff = (y_bs.float() - y_ref.float()).abs()
  rel_err = diff / (y_ref.float().abs() + 1e-6)
  print(f"\nComparison:")
  print(f"  Abs diff: max={diff.max():.6f} mean={diff.mean():.6f}")
  print(f"  Rel err:  max={rel_err.max():.6f} mean={rel_err.mean():.6f}")

  # Amplitude ratio (to see if blockscaled has lower amplitude)
  amp_ref = _rms(y_ref)
  amp_bs = _rms(y_bs)
  print(f"  Amplitude: ref={amp_ref:.6f} bs={amp_bs:.6f} ratio={amp_bs/amp_ref:.4f}")


@torch.no_grad()
def test_extreme_imbalance(profile: str = "fp8") -> None:
  """Test GEMM with extreme imbalance (one expert gets almost everything)."""
  torch.manual_seed(42)
  torch.cuda.set_device(0)
  dev = torch.device("cuda")

  E = 8
  H = 2048
  Dff = 1408

  # Extreme: expert 0 gets 90%, others get 1-2%
  loads = [2560, 128, 128, 64, 64, 64, 32, 32]  # sum = 3072

  print(f"\n=== Extreme Imbalance Test ({profile}) ===")
  print(f"Loads per expert: {loads}")
  print(f"Load %: {[f'{100*l/sum(loads):.1f}%' for l in loads]}")

  loads_padded = [((l + 127) // 128) * 128 for l in loads]
  offs_padded_list = [0]
  for l in loads_padded:
    offs_padded_list.append(offs_padded_list[-1] + l)
  offs_padded = torch.tensor(offs_padded_list[1:], device=dev, dtype=torch.int32)
  M_pad_aligned = sum(loads_padded)

  x = torch.randn(M_pad_aligned, H, device=dev, dtype=torch.bfloat16) * 0.1

  # Zero out padding
  idx = 0
  for i, (real, padded) in enumerate(zip(loads, loads_padded)):
    if real < padded:
      x[idx + real:idx + padded] = 0
    idx += padded

  W1 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16) * 0.02
  W3 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16) * 0.02
  W2 = torch.randn(E, Dff, H, device=dev, dtype=torch.bfloat16) * 0.02

  # BF16 reference
  y_ref = ggemm.expert(x, W1, W3, W2, offs_padded)

  # Blockscaled
  W_cache = quantize_weights(W1, W3, W2, profile=profile)

  if profile == "fp8":
    x_q, x_sf = quantize_fp8(x)
    Xe_q_pad_u16 = _packed_u16_from_fp8(x_q)
  else:
    x_q, x_sf = quantize_nvfp4(x)
    Xe_q_pad_u16 = _packed_u16_from_nvfp4(x_q)

  M_e_stride = ((max(loads_padded) + 127) // 128) * 128
  sf_k = H // 32
  sf_k_pad = ((sf_k + 3) // 4) * 4
  Xe_sf_mma = torch.zeros(E, M_e_stride, sf_k_pad, device=dev, dtype=torch.uint8)

  # Swizzle SFA from MKL row-major to per-expert strided MMA layout
  x_sf_mkl = x_sf.squeeze(-1).contiguous() if x_sf.ndim == 3 else x_sf.contiguous()
  offs_with0 = torch.cat([torch.zeros(1, device=dev, dtype=torch.int32), offs_padded])
  rdep.swizzle_sf_strided(
    x_sf_mkl.data_ptr(), Xe_sf_mma.data_ptr(),
    offs_with0.data_ptr(), E, sf_k, sf_k_pad, M_pad_aligned, M_e_stride,
    torch.cuda.current_stream(dev)
  )

  y_bs = bs_expert(Xe_q_pad_u16, Xe_sf_mma, W_cache, offs_padded)

  # Check for NaN/Inf
  nan_ref = torch.isnan(y_ref).sum().item()
  nan_bs = torch.isnan(y_bs).sum().item()
  inf_ref = torch.isinf(y_ref).sum().item()
  inf_bs = torch.isinf(y_bs).sum().item()

  print(f"\nResults:")
  print(f"  BF16:  NaN={nan_ref} Inf={inf_ref} rms={_rms(y_ref):.6f}")
  print(f"  {profile.upper()}: NaN={nan_bs} Inf={inf_bs} rms={_rms(y_bs):.6f}")

  if nan_bs > 0 or inf_bs > 0:
    print("  WARNING: Blockscaled produced NaN/Inf!")
  else:
    diff = (y_bs.float() - y_ref.float()).abs()
    print(f"  Max abs diff: {diff.max():.6f}")


@torch.no_grad()
def test_stress_values(profile: str = "fp8") -> None:
  """Test GEMM with stress input values (large/small magnitudes)."""
  torch.manual_seed(42)
  torch.cuda.set_device(0)
  dev = torch.device("cuda")

  E = 8
  H = 2048
  Dff = 1408
  M_per = 256
  M_pad = E * M_per

  offs = torch.arange(0, M_pad + 1, step=M_per, device=dev, dtype=torch.int32)
  offs_pad = offs[1:].contiguous()

  print(f"\n=== Stress Value Test ({profile}) ===")

  for scale, name in [(0.01, "small"), (0.1, "normal"), (1.0, "large"), (10.0, "very_large")]:
    x = torch.randn(M_pad, H, device=dev, dtype=torch.bfloat16) * scale
    W1 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16) * 0.02
    W3 = torch.randn(E, H, Dff, device=dev, dtype=torch.bfloat16) * 0.02
    W2 = torch.randn(E, Dff, H, device=dev, dtype=torch.bfloat16) * 0.02

    y_ref = ggemm.expert(x, W1, W3, W2, offs_pad)

    W_cache = quantize_weights(W1, W3, W2, profile=profile)
    if profile == "fp8":
      x_q, x_sf = quantize_fp8(x)
      Xe_q_pad_u16 = _packed_u16_from_fp8(x_q)
    else:
      x_q, x_sf = quantize_nvfp4(x)
      Xe_q_pad_u16 = _packed_u16_from_nvfp4(x_q)

    M_e_stride = M_per
    sf_k = H // 32
    sf_k_pad = ((sf_k + 3) // 4) * 4
    Xe_sf_mma_padded = torch.zeros(E, M_e_stride, sf_k_pad, device=dev, dtype=torch.uint8)

    # Swizzle SFA from MKL row-major to per-expert strided MMA layout
    x_sf_mkl = x_sf.squeeze(-1).contiguous() if x_sf.ndim == 3 else x_sf.contiguous()
    offs_with0 = torch.cat([torch.zeros(1, device=dev, dtype=torch.int32), offs_pad])
    rdep.swizzle_sf_strided(
      x_sf_mkl.data_ptr(), Xe_sf_mma_padded.data_ptr(),
      offs_with0.data_ptr(), E, sf_k, sf_k_pad, M_pad, M_e_stride,
      torch.cuda.current_stream(dev)
    )

    y_bs = bs_expert(Xe_q_pad_u16, Xe_sf_mma_padded, W_cache, offs_pad)

    nan_bs = torch.isnan(y_bs).sum().item()
    inf_bs = torch.isinf(y_bs).sum().item()
    amp_ratio = _rms(y_bs) / (_rms(y_ref) + 1e-9)

    status = "FAIL" if (nan_bs > 0 or inf_bs > 0) else "OK"
    print(f"  {name} (scale={scale}): {status} NaN={nan_bs} Inf={inf_bs} amp_ratio={amp_ratio:.4f}")


if __name__ == "__main__":
  print("Testing blockscaled GEMM numerics with imbalanced loads...")

  for profile in ["fp8", "nvfp4"]:
    test_imbalanced_load(profile)
    test_extreme_imbalance(profile)
    test_stress_values(profile)

  print("\n=== All tests complete ===")
