import math
import torch


def _e8m0_from_scale(scale: torch.Tensor) -> torch.Tensor:
    """Encode positive scale values to E8M0 (uint8).

    scale_byte = 127 + round(log2(scale)), with guards for non‑positive.
    """
    s = torch.clamp(scale, min=1e-8)
    exp = torch.round(torch.log2(s))
    byte = 127 + exp.to(torch.int32)
    return torch.clamp(byte, 0, 255).to(torch.uint8)


class BlockscaleActivationTracker:
    """Blockscale EMA stabilizer for activation scales.

    Granularity: per row, per K‑block of size sf_vec (32 for FP8, 32 BF16 for NVFP4).
    Stabilizes per‑block scales using a block‑column EMA across rows (and over steps),
    then clamps per‑row amax to a band around the EMA before computing scales.
    """

    def __init__(self, H: int, profile: str = "fp8", beta: float = 0.99,
                 margin: float = 0.95, band: tuple[float, float] = (0.7, 1.3)):
        assert profile in ("fp8", "nvfp4")
        self.profile = profile
        self.beta = beta
        self.margin = margin
        self.alpha_lo, self.alpha_hi = band
        self.sf_vec = 32  # both paths use 32 BF16 elements per scale vector
        self.sf_k = (H + self.sf_vec - 1) // self.sf_vec
        self.register_state(H)

    def register_state(self, H: int):
        device = torch.device("cuda")
        self.ema_blk = torch.zeros(self.sf_k, device=device, dtype=torch.float32)

    @torch.no_grad()
    def sfa(self, X_bf16: torch.Tensor) -> torch.Tensor:
        """Compute stabilized SFA for X_bf16 [M, H]. Returns [M, sf_k] uint8.
        """
        assert X_bf16.is_cuda and X_bf16.dtype == torch.bfloat16
        M, H = X_bf16.shape
        sf_k = self.sf_k
        # Per-row, per‑block amax
        Xf = X_bf16.float()
        # Pad H to sf_k*sf_vec to simplify view
        pad_cols = sf_k * self.sf_vec - H
        if pad_cols:
            Xf = torch.nn.functional.pad(Xf, (0, pad_cols))
        Xb = Xf.view(M, sf_k, self.sf_vec)
        amax_row_blk = Xb.abs().amax(dim=2)  # [M, sf_k]

        # Block‑column statistic across rows (p99 or mean of top‑k)
        # Use top‑k mean as a cheap p‑approximation
        k = max(1, int(0.01 * M))
        vals, _ = torch.topk(amax_row_blk, k, dim=0)
        amax_batch_blk = vals.mean(dim=0)  # [sf_k]

        # EMA update
        if torch.count_nonzero(self.ema_blk).item() == 0:
            self.ema_blk.copy_(amax_batch_blk)
        else:
            self.ema_blk.mul_(self.beta).maximum_(amax_batch_blk.mul(1 - self.beta))

        # Clamp per‑row block amax to band around EMA
        lo = self.ema_blk * self.alpha_lo
        hi = self.ema_blk * self.alpha_hi
        amax_eff = torch.clamp(amax_row_blk, min=lo, max=hi)

        # Convert to scale with headroom
        fp_max = 448.0 if self.profile == "fp8" else 6.0
        scale = torch.clamp(amax_eff / (fp_max * self.margin), min=1e-8)
        # E8M0 encode (power‑of‑two snapping implicit via rounding in _e8m0_from_scale)
        sfa_u8 = _e8m0_from_scale(scale)
        return sfa_u8.contiguous()


class BlockscaleWeightTracker:
    """Per‑channel, per‑block scales for weights.

    For each row (output channel) and K‑block, compute amax and emit E8M0 scale with margin.
    Uses optional EMA across calls if desired (not required for first iteration).
    """

    def __init__(self, K: int, profile: str = "fp8", margin: float = 0.95):
        assert profile in ("fp8", "nvfp4")
        self.profile = profile
        self.margin = margin
        self.sf_vec = 32
        self.sf_k = (K + self.sf_vec - 1) // self.sf_vec

    @torch.no_grad()
    def sfb(self, W_rowmajor: torch.Tensor) -> torch.Tensor:
        """Compute SFB for W [N, K] BF16 (row‑major). Returns [N, sf_k] uint8.
        """
        assert W_rowmajor.is_cuda and W_rowmajor.dtype == torch.bfloat16
        N, K = W_rowmajor.shape
        Wf = W_rowmajor.float()
        pad_cols = self.sf_k * self.sf_vec - K
        if pad_cols:
            Wf = torch.nn.functional.pad(Wf, (0, pad_cols))
        Wb = Wf.view(N, self.sf_k, self.sf_vec)
        amax = Wb.abs().amax(dim=2)  # [N, sf_k]
        fp_max = 448.0 if self.profile == "fp8" else 6.0
        scale = torch.clamp(amax / (fp_max * self.margin), min=1e-8)
        return _e8m0_from_scale(scale).contiguous()

