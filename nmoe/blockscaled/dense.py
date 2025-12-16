"""Blockscaled dense ops for SM100 (B200).

Public surfaces:
  - blockscaled dense linear (F.linear equivalent)
  - blockscaled dense MLP

All ops:
  - Inputs: BF16
  - Outputs: BF16
  - Accumulation: FP32 (BF16 output)

This is the dense counterpart to the grouped MoE expert path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nmoe.csrc import rdep
from nmoe.quant import quantize_fp8, quantize_nvfp4
from nmoe.blockscaled.grouped import _swizzle_sf_to_mma, run_grouped_blockscaled_strided


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


_DENSE_OFFS_CACHE: dict[tuple[int, int], torch.Tensor] = {}

@dataclass
class _ActQuantTmp:
  """Reusable per-call intermediates for activation quantization."""

  M: int
  K: int
  u16: torch.Tensor     # [M, K//2] (fp8) or [M, K//4] (nvfp4)
  sf_mma: torch.Tensor  # [1, M, sf_k_pad] uint8, swizzled SFA in MMA layout


# Per-(device, profile) single-slot cache. Keeps the hot path allocation-free.
_ACT_QUANT_TMP: dict[tuple[int, str], _ActQuantTmp] = {}

@dataclass
class _Bf16Tmp:
  """Reusable BF16 workspace tensor (internal intermediates only)."""

  shape: tuple[int, int]
  buf: torch.Tensor


_BF16_TMP: dict[tuple[int, tuple[int, int]], _Bf16Tmp] = {}


def _get_dense_offs(device: torch.device, M: int) -> torch.Tensor:
  """Return cached offs=[0,M] on GPU (avoid CPU->GPU copy in hot path)."""
  device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
  key = (device_index, int(M))
  offs = _DENSE_OFFS_CACHE.get(key)
  if offs is None:
    offs = torch.tensor([0, M], device=device, dtype=torch.int32)
    _DENSE_OFFS_CACHE[key] = offs
  return offs


def _require_bf16_cuda_contig(x: torch.Tensor, *, name: str) -> None:
  if x.dtype != torch.bfloat16:
    raise TypeError(f"{name} must be BF16, got {x.dtype}.")
  if not x.is_cuda:
    raise TypeError(f"{name} must be CUDA tensor.")
  if not x.is_contiguous():
    raise TypeError(f"{name} must be contiguous.")


def _require_multiple(x: int, mult: int, *, name: str) -> None:
  if x % mult != 0:
    raise ValueError(f"{name} must be multiple of {mult}, got {x}.")


def _get_act_quant_tmp(device: torch.device, *, profile: str, M: int, K: int) -> _ActQuantTmp:
  """Get (or allocate) activation quant intermediates for this (M,K,profile)."""
  device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
  key = (device_index, str(profile))
  tmp = _ACT_QUANT_TMP.get(key)
  if tmp is not None and tmp.M == int(M) and tmp.K == int(K):
    return tmp

  sf_k = K // 32
  sf_k_pad = _ceil_div(sf_k, 4) * 4
  if sf_k_pad != sf_k:
    raise ValueError(f"K must be a multiple of 128 (sf_k%4==0). Got K={K}.")

  if profile == "fp8":
    u16 = torch.empty((M, K // 2), device=device, dtype=torch.uint16)
  elif profile == "nvfp4":
    u16 = torch.empty((M, K // 4), device=device, dtype=torch.uint16)
  else:
    raise ValueError("profile must be 'fp8' or 'nvfp4'")

  sf_mma = torch.empty((1, M, sf_k_pad), device=device, dtype=torch.uint8)
  tmp = _ActQuantTmp(M=int(M), K=int(K), u16=u16, sf_mma=sf_mma)
  _ACT_QUANT_TMP[key] = tmp
  return tmp


def _get_bf16_tmp(device: torch.device, *, shape: tuple[int, int]) -> torch.Tensor:
  """Get (or allocate) a reusable BF16 workspace buffer for internal intermediates."""
  device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
  key = (device_index, tuple(map(int, shape)))
  tmp = _BF16_TMP.get(key)
  if tmp is not None:
    return tmp.buf
  buf = torch.empty(shape, device=device, dtype=torch.bfloat16)
  _BF16_TMP[key] = _Bf16Tmp(shape=shape, buf=buf)
  return buf


@dataclass(frozen=True)
class QuantizedLinearWeights:
  """Quantized dense linear weights (W is [out, in])."""

  W_q: torch.Tensor        # [N, K_packed, 1] FP8 (float8) or NVFP4 (uint8 packed)
  W_sf_mma: torch.Tensor   # [N_pad, sf_k_pad, 1] uint8 E8M0 SFB in MMA layout (colwise)
  N: int
  K: int
  profile: str


def quantize_linear_weight(W: torch.Tensor, *, profile: str) -> QuantizedLinearWeights:
  """Quantize BF16 dense linear weight (nn.Linear-style) for blockscaled GEMM.

  Args:
    W: [N, K] BF16, row-major (out_features, in_features)
    profile: "fp8" or "nvfp4"
  """
  _require_bf16_cuda_contig(W, name="W")
  if W.ndim != 2:
    raise ValueError(f"W must be 2D [N,K], got shape={tuple(W.shape)}.")

  N, K = map(int, W.shape)
  _require_multiple(N, 64, name="N")
  _require_multiple(K, 128, name="K")

  if profile == "fp8":
    W_q, W_sf = quantize_fp8(W)
  elif profile == "nvfp4":
    W_q, W_sf = quantize_nvfp4(W)
  else:
    raise ValueError("profile must be 'fp8' or 'nvfp4'")

  W_sf_mma = _swizzle_sf_to_mma(W_sf)
  return QuantizedLinearWeights(W_q=W_q.contiguous(), W_sf_mma=W_sf_mma, N=N, K=K, profile=profile)


def _linear_impl(
  x: torch.Tensor,
  W: QuantizedLinearWeights,
  *,
  bias: torch.Tensor | None = None,
  out: torch.Tensor | None = None,
) -> torch.Tensor:
  """Blockscaled dense linear (F.linear equivalent): y = x @ W.T (+ bias).

  Notes:
  - Weight quantization is caller-owned via QuantizedLinearWeights (cache it).
  - Activation quantization + SFA swizzle are part of the op.
  """
  _require_bf16_cuda_contig(x, name="x")
  if x.ndim != 2:
    raise ValueError(f"x must be 2D [M,K], got shape={tuple(x.shape)}.")
  M, K = map(int, x.shape)
  if K != int(W.K):
    raise ValueError(f"K mismatch: x.shape[1]={K} vs W.K={W.K}.")

  _require_multiple(M, 128, name="M")
  _require_multiple(K, 128, name="K")
  _require_multiple(int(W.N), 64, name="N")

  # Quantize activation and write SFA directly to the per-group MMA layout.
  # This avoids the standalone swizzle kernel (and its memset).
  E = 1
  offs = _get_dense_offs(x.device, M)  # [0, M]
  tmp = _get_act_quant_tmp(x.device, profile=W.profile, M=M, K=K)
  A_u16 = tmp.u16
  A_sf_mma = tmp.sf_mma
  stream = torch.cuda.current_stream(x.device)

  if W.profile == "fp8":
    rdep.quant_fp8_sf_strided_mma(
      x.data_ptr(), x.stride(0),
      A_u16.data_ptr(), A_u16.stride(0),
      A_sf_mma.data_ptr(),
      offs.data_ptr(), E, M,
      M, K,
      stream
    )
    A_q = A_u16.view(torch.uint8).view(M, K, 1).view(torch.float8_e4m3fn)
  elif W.profile == "nvfp4":
    rdep.quant_nvfp4_sf_strided_mma(
      x.data_ptr(), x.stride(0),
      A_u16.data_ptr(), A_u16.stride(0),
      A_sf_mma.data_ptr(),
      offs.data_ptr(), E, M,
      M, K,
      stream
    )
    A_q = A_u16.view(torch.uint8).view(M, K // 2, 1)
  else:
    raise ValueError(f"Unsupported profile: {W.profile}")

  if out is None:
    y = torch.empty((M, int(W.N)), device=x.device, dtype=torch.bfloat16)
  else:
    _require_bf16_cuda_contig(out, name="out")
    if out.ndim != 2 or tuple(map(int, out.shape)) != (M, int(W.N)):
      raise ValueError(f"out must have shape {(M, int(W.N))}, got {tuple(out.shape)}.")
    y = out

  run_grouped_blockscaled_strided(
    A_q, A_sf_mma,
    W.W_q.unsqueeze(0), W.W_sf_mma.unsqueeze(0),
    y.unsqueeze(-1), offs,
    profile=W.profile, N=int(W.N), K=K,
  )

  if bias is not None:
    if bias.dtype != torch.bfloat16 or (not bias.is_cuda) or (not bias.is_contiguous()) or bias.ndim != 1:
      raise TypeError("bias must be 1D contiguous CUDA BF16 tensor.")
    if bias.numel() != int(W.N):
      raise ValueError(f"bias length {bias.numel()} != N {W.N}.")
    y.add_(bias)
  return y


def linear(
  x: torch.Tensor,
  W: QuantizedLinearWeights,
  *,
  bias: torch.Tensor | None = None,
) -> torch.Tensor:
  return _linear_impl(x, W, bias=bias, out=None)


@dataclass(frozen=True)
class QuantizedMLPWeights:
  """Quantized dense MLP weights for Y = (SiLU(X@W1.T) * (X@W3.T)) @ W2.T."""

  W13: QuantizedLinearWeights  # Interleaved gate/up, output N=2*Dff
  W2: QuantizedLinearWeights   # Down projection, output N=H
  H: int
  Dff: int
  profile: str


def quantize_mlp_weights(
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  *,
  profile: str,
) -> QuantizedMLPWeights:
  """Quantize dense MLP weights from nn.Linear-style matrices.

  Args:
    W1: [Dff, H] BF16 (gate)
    W3: [Dff, H] BF16 (up)
    W2: [H, Dff] BF16 (down)
  """
  _require_bf16_cuda_contig(W1, name="W1")
  _require_bf16_cuda_contig(W3, name="W3")
  _require_bf16_cuda_contig(W2, name="W2")
  if W1.ndim != 2 or W3.ndim != 2 or W2.ndim != 2:
    raise ValueError("W1/W3/W2 must all be 2D matrices.")

  Dff, H = map(int, W1.shape)
  if tuple(W3.shape) != (Dff, H):
    raise ValueError(f"W3 shape {tuple(W3.shape)} != {(Dff, H)}.")
  if tuple(W2.shape) != (H, Dff):
    raise ValueError(f"W2 shape {tuple(W2.shape)} != {(H, Dff)}.")

  _require_multiple(H, 128, name="H")
  _require_multiple(Dff, 128, name="Dff")
  _require_multiple(2 * Dff, 64, name="2*Dff")

  # Interleave W1.T and W3.T along output columns: [H, 2*Dff] then transpose to [2*Dff, H].
  W13 = torch.stack([W1.T, W3.T], dim=-1).view(H, 2 * Dff).T.contiguous()
  W13_q = quantize_linear_weight(W13, profile=profile)
  W2_q = quantize_linear_weight(W2.contiguous(), profile=profile)
  return QuantizedMLPWeights(W13=W13_q, W2=W2_q, H=H, Dff=Dff, profile=profile)


def mlp(x: torch.Tensor, W: QuantizedMLPWeights) -> torch.Tensor:
  """Blockscaled dense MLP (BF16 output, FP32 accumulate)."""
  _require_bf16_cuda_contig(x, name="x")
  if x.ndim != 2:
    raise ValueError(f"x must be 2D [M,H], got shape={tuple(x.shape)}.")
  M, H = map(int, x.shape)
  if H != int(W.H):
    raise ValueError(f"H mismatch: x.shape[1]={H} vs W.H={W.H}.")

  _require_multiple(M, 128, name="M")

  # GEMM 1+2: H13 = X @ W13.T, where W13 is interleaved [gate0, up0, gate1, up1, ...].
  H13 = _get_bf16_tmp(x.device, shape=(M, 2 * int(W.Dff)))
  _linear_impl(x, W.W13, out=H13)

  # SwiGLU + quant/pack for GEMM3 input, writing SFA directly to MMA layout.
  offs = _get_dense_offs(x.device, M)  # [0, M]
  tmp = _get_act_quant_tmp(x.device, profile=W.profile, M=M, K=int(W.Dff))
  A_u16 = tmp.u16
  A_sf_mma = tmp.sf_mma
  stream = torch.cuda.current_stream(x.device)
  if W.profile == "fp8":
    rdep.swiglu_quant_fp8_sf_strided_mma(
      H13.data_ptr(), H13.stride(0),
      A_u16.data_ptr(), A_u16.stride(0),
      A_sf_mma.data_ptr(),
      offs.data_ptr(), 1, M,
      M, int(W.Dff),
      stream,
    )
    A_q = A_u16.view(torch.uint8).view(M, int(W.Dff), 1).view(torch.float8_e4m3fn)
  elif W.profile == "nvfp4":
    rdep.swiglu_quant_nvfp4_sf_strided_mma(
      H13.data_ptr(), H13.stride(0),
      A_u16.data_ptr(), A_u16.stride(0),
      A_sf_mma.data_ptr(),
      offs.data_ptr(), 1, M,
      M, int(W.Dff),
      stream,
    )
    A_q = A_u16.view(torch.uint8).view(M, int(W.Dff) // 2, 1)
  else:
    raise ValueError(f"Unsupported profile: {W.profile}")

  # GEMM 3: Y = A @ W2.T  (A is already quantized)
  y = torch.empty((M, int(W.H)), device=x.device, dtype=torch.bfloat16)
  run_grouped_blockscaled_strided(
    A_q, A_sf_mma,
    W.W2.W_q.unsqueeze(0), W.W2.W_sf_mma.unsqueeze(0),
    y.unsqueeze(-1), offs,
    profile=W.profile, N=int(W.H), K=int(W.Dff),
  )
  return y
