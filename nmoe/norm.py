import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

  def forward(self, x: torch.Tensor):
    return F.rms_norm(x, (self.dim,), self.weight, self.eps)
