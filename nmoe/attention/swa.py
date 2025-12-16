import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from nmoe.attention.rope import rotate_pe
from nmoe.config import Config
import nmoe.triton.swa as swa_k


class SWA(nn.Module):
  """Sliding-Window Attention wrapper around Triton kernel (nmoe.triton.swa).

  Fused QKV + OUT + learned sinks. Applies RoPE to q,k and calls the kernel with
  layout expected by SWA: q[B,T,Hkv,G,D], k[B,T,Hkv,D], v[B,T,Hkv,D]. For now we
  assume GQA groups = 1 (Hkv == H), which is sufficient for initial integration
  and weight conversion from gpt-oss.
  """

  def __init__(self, config: Config):
    super().__init__()
    self.dim = config.dim
    self.n_heads = config.n_heads
    self.head_dim = config.v_head_dim
    qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if qk_dim != self.head_dim:
      raise RuntimeError(
        f"SWA requires qk_head_dim == v_head_dim (got {qk_dim} vs {self.head_dim})."
      )
    swa_opts = getattr(config, "attn_swa", {}) or {}
    self.n_kv_heads = int(swa_opts.get("kv_heads", self.n_heads))
    if self.n_heads % self.n_kv_heads != 0:
      raise RuntimeError(f"SWA requires H % KV == 0 (H={self.n_heads}, KV={self.n_kv_heads}).")
    self.repeat_kv = self.n_heads // self.n_kv_heads
    out_feats = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
    self.qkv = nn.Linear(self.dim, out_feats, bias=True, dtype=torch.bfloat16)
    self.out = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=True, dtype=torch.bfloat16)
    self.sinks = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.bfloat16))
    self.window = 0
    self.softmax_scale = self.head_dim ** -0.5

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
    if self.qkv.bias is not None:
      nn.init.zeros_(self.qkv.bias)
    nn.init.trunc_normal_(self.out.weight, mean=0.0, std=init_std)
    if self.out.bias is not None:
      nn.init.zeros_(self.out.bias)
    with torch.no_grad():
      self.sinks.zero_()

  @record_function("attn")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.size()
    qkv = self.qkv(x)
    H, KV, D = self.n_heads, self.n_kv_heads, self.head_dim
    q, k, v = torch.split(qkv, [H * D, KV * D, KV * D], dim=-1)
    q = q.view(B, T, H, D)
    k = k.view(B, T, KV, D)
    v = v.view(B, T, KV, D)

    q = rotate_pe(q, cos, sin)
    k = rotate_pe(k, cos, sin)

    G = self.repeat_kv
    assert H == KV * G, f"Heads must satisfy H = KV*G (got H={H}, KV={KV}, G={G})"
    q = q.view(B, T, KV, G, D)

    start_q = torch.tensor([0], dtype=torch.int32, device=x.device)
    bandwidth = int(self.window) if self.window and self.window > 0 else 0

    with record_function("attn.kernel[swa]"):
      o = swa_k.attention(q, k, v, self.sinks, self.softmax_scale, bandwidth, start_q)
    return self.out(o)
