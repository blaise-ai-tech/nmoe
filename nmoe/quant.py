"""Python wrapper for BF16 -> FP8/NVFP4 quantization kernels.

Uses quant.cu kernels via the rdep C-extension (nmoe.rdep._C).

Output formats:
  - FP8: out [M, K/2] uint16 (packed), sfa [M, ceil(K/32)] uint8
  - NVFP4: out [M, K/4] uint16 (packed), sfa [M, ceil(K/32)] uint8
"""
from __future__ import annotations

import torch
from nmoe.rdep import _C


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to FP8 E4M3 with E8M0 scale factors.

    Args:
        x: [M, K] BF16 tensor, K must be multiple of 32

    Returns:
        out: [M, K, 1] float8_e4m3fn for blockscaled runners
        sfa: [M, K//32, 1] uint8 E8M0 scale factors
    """
    assert x.dtype == torch.bfloat16, f"Expected BF16, got {x.dtype}"
    assert x.is_cuda and x.is_contiguous()
    M, K = x.shape
    assert K % 32 == 0, f"K must be multiple of 32, got {K}"

    out_u16 = torch.empty(M, K // 2, dtype=torch.uint16, device=x.device)
    sfa = torch.empty(M, K // 32, dtype=torch.uint8, device=x.device)

    _C.quant_fp8(
        x.data_ptr(), K,
        out_u16.data_ptr(), K // 2,
        sfa.data_ptr(), K // 32,
        M, K, torch.cuda.current_stream(x.device)
    )

    out = out_u16.view(torch.uint8).view(M, K, 1).view(torch.float8_e4m3fn)
    sfa = sfa.unsqueeze(-1)
    return out, sfa


def quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to NVFP4 E2M1 with E8M0 scale factors.

    Args:
        x: [M, K] BF16 tensor, K must be multiple of 32

    Returns:
        out: [M, K//2, 1] uint8 packed bytes (2 FP4 codes per byte)
        sfa: [M, K//32, 1] uint8 E8M0 scale factors
    """
    assert x.dtype == torch.bfloat16, f"Expected BF16, got {x.dtype}"
    assert x.is_cuda and x.is_contiguous()
    M, K = x.shape
    assert K % 32 == 0, f"K must be multiple of 32, got {K}"

    out_u16 = torch.empty(M, K // 4, dtype=torch.uint16, device=x.device)
    sfa = torch.empty(M, K // 32, dtype=torch.uint8, device=x.device)

    _C.quant_nvfp4(
        x.data_ptr(), K,
        out_u16.data_ptr(), K // 4,
        sfa.data_ptr(), K // 32,
        M, K, torch.cuda.current_stream(x.device)
    )

    out = out_u16.view(torch.uint8).view(M, K // 2, 1)
    sfa = sfa.unsqueeze(-1)
    return out, sfa


def quantize_fp8_with_sfa(x: torch.Tensor, sfa: torch.Tensor) -> torch.Tensor:
    """Quantize BF16 tensor to FP8 E4M3 using PRECOMPUTED E8M0 SFA.

    Args:
        x: [M, K] BF16 tensor (K % 32 == 0)
        sfa: [M, K//32] uint8 E8M0 scale factors (MKL row‑major)

    Returns:
        out: [M, K, 1] float8_e4m3fn (packed to uint16 under the hood)
    """
    assert x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
    assert sfa.dtype == torch.uint8 and sfa.is_cuda and sfa.is_contiguous()
    M, K = x.shape
    assert sfa.shape == (M, K // 32)
    out_u16 = torch.empty(M, K // 2, dtype=torch.uint16, device=x.device)
    _C.quant_fp8_with_sfa(x.data_ptr(), K,
                          out_u16.data_ptr(), K // 2,
                          sfa.data_ptr(), K // 32,
                          M, K, torch.cuda.current_stream(x.device))
    return out_u16.view(torch.uint8).view(M, K, 1).view(torch.float8_e4m3fn)


def quantize_nvfp4_with_sfa(x: torch.Tensor, sfa: torch.Tensor) -> torch.Tensor:
    """Quantize BF16 tensor to NVFP4 E2M1 using PRECOMPUTED E8M0 SFA.

    Args:
        x: [M, K] BF16 tensor (K % 32 == 0)
        sfa: [M, K//32] uint8 E8M0 scale factors

    Returns:
        out: [M, K//2, 1] uint8 (each byte packs two FP4 nibbles)
    """
    assert x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
    assert sfa.dtype == torch.uint8 and sfa.is_cuda and sfa.is_contiguous()
    M, K = x.shape
    assert sfa.shape == (M, K // 32)
    out_u16 = torch.empty(M, K // 4, dtype=torch.uint16, device=x.device)
    _C.quant_nvfp4_with_sfa(x.data_ptr(), K,
                            out_u16.data_ptr(), K // 4,
                            sfa.data_ptr(), K // 32,
                            M, K, torch.cuda.current_stream(x.device))
    return out_u16.view(torch.uint8).view(M, K // 2, 1)
