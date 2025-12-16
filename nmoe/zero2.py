"""ZeRO-2 for replicated (dense) parameters.

Elegant minimal implementation:
- step_dense_adamw: RS(AVG) grads → AdamW shard update → AG params

State lives in caller-provided dict to keep this module stateless.
Works seamlessly for single GPU (no-op) and multi-GPU (ZeRO-2).
"""
import os
from contextlib import nullcontext
import math
import torch
import torch.distributed as dist


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


def _get_or_init_flat_group(
  group: dict,
  *,
  params: list[torch.nn.Parameter],
  rank: int,
  world: int,
  dtype: torch.dtype,
) -> dict:
  """Return (and lazily initialize) a persistent flat buffer plan for a param group/dtype.

  Stored on the *group dict* (not in optimizer state) to avoid checkpointing large buffers.
  """
  cache = group.setdefault('_zero2_flat', {})
  key = (dtype, int(world))
  if key in cache:
    flat = cache[key]
    # Re-init if param list changed (should not happen in training).
    if flat.get('n_params') == len(params):
      return flat

  if not params:
    raise RuntimeError("ZeRO-2: empty param list")

  dev = params[0].device
  for p in params:
    if p.device != dev:
      raise RuntimeError("ZeRO-2: params must be on a single device per group")
    if p.dtype != dtype:
      raise RuntimeError("ZeRO-2: dtype grouping invariant violated")

  offsets: list[tuple[int, int]] = []
  total = 0
  for p in params:
    n = int(p.numel())
    offsets.append((total, total + n))
    total += n

  shard_size = _ceil_div(total, world)
  padded_total = shard_size * world

  flat_param = torch.empty(padded_total, dtype=dtype, device=dev)
  # Initialize from current param values.
  for p, (a, b) in zip(params, offsets):
    flat_param[a:b].copy_(p.detach().view(-1))
  if total < padded_total:
    flat_param[total:padded_total].zero_()

  # Re-point params to views into flat_param (no per-step unpack copies).
  for p, (a, b) in zip(params, offsets):
    p.data = flat_param[a:b].view_as(p)  # type: ignore[assignment]

  # Scratch (persistent): full grad buffer + RS output + local param shard.
  flat_grad = torch.empty(padded_total, dtype=dtype, device=dev)
  grad_shard = torch.empty(shard_size, dtype=dtype, device=dev)
  if world == 1:
    param_shard = flat_param[:shard_size]  # view (no comm)
  else:
    param_shard = torch.empty(shard_size, dtype=dtype, device=dev)
    a = rank * shard_size
    b = a + shard_size
    param_shard.copy_(flat_param[a:b])

  flat = {
    'n_params': len(params),
    'offsets': offsets,
    'total': total,
    'padded_total': padded_total,
    'shard_size': shard_size,
    'flat_param': flat_param,
    'flat_grad': flat_grad,
    'grad_shard': grad_shard,
    'param_shard': param_shard,
  }
  cache[key] = flat
  return flat


@torch.no_grad()
def step_dense_adamw(
  param_groups: list[dict],
  *,
  state: dict,
  pg: dist.ProcessGroup | None = None,
  betas: tuple[float, float] = (0.9, 0.95),
  eps: float = 1e-8,
) -> None:
  """ZeRO-2 shard step for dense param groups using AdamW.

  For each group:
  1. Flatten grads and params per dtype
  2. Reduce-scatter(AVG) grads → local shard
  3. AdamW update on shard with shard-local state (exp_avg/exp_avg_sq)
  4. All-gather updated param shards

  Single GPU: just runs AdamW on full params (no RS/AG).
  Multi-GPU: full ZeRO-2 with sharding.
  """
  timers_on = os.getenv('NMOE_TIMERS', '1') not in ('0', 'false', 'False')
  if timers_on:
    try:
      from nmoe.metrics import cuda_time as _cuda_time  # local import to avoid hard dependency
      time_ctx = _cuda_time
    except Exception:
      time_ctx = lambda _tag: nullcontext()
  else:
    time_ctx = lambda _tag: nullcontext()

  world = dist.get_world_size(pg) if (dist.is_available() and dist.is_initialized()) else 1

  rank = dist.get_rank(pg) if (dist.is_available() and dist.is_initialized()) else 0

  for group_idx, group in enumerate(param_groups):
    params = list(group['params'])
    if not params:
      continue

    # Group by dtype
    by_dtype = {}
    for p in params:
      if p.dtype not in by_dtype:
        by_dtype[p.dtype] = []
      by_dtype[p.dtype].append(p)

    for dtype, group_params in by_dtype.items():
      flat = _get_or_init_flat_group(group, params=group_params, rank=rank, world=world, dtype=dtype)
      flat_param = flat['flat_param']
      flat_grad = flat['flat_grad']
      grad_shard = flat['grad_shard']
      param_shard = flat['param_shard']
      offsets: list[tuple[int, int]] = flat['offsets']
      total = int(flat['total'])
      padded_total = int(flat['padded_total'])
      shard_size = int(flat['shard_size'])

      # Pack grads into persistent flat buffer (no torch.cat in hot path).
      flat_grad.zero_()
      for p, (a, b) in zip(group_params, offsets):
        if p.grad is None:
          continue
        flat_grad[a:b].copy_(p.grad.detach().view(-1))

      # Reduce-scatter(AVG) grads (or local shard when world==1).
      if world == 1:
        grad_shard.copy_(flat_grad[:shard_size])
      else:
        with time_ctx('time_ms/zero2_reduce_scatter'):
          dist.reduce_scatter_tensor(grad_shard, flat_grad, op=dist.ReduceOp.AVG, group=pg)

      # Get or initialize shard state
      state_key = f"shard_{rank}_{group_idx}_{dtype}"
      if state_key not in state:
        state[state_key] = {
          'step': 0,
          'exp_avg': torch.zeros_like(grad_shard),
          'exp_avg_sq': torch.zeros_like(grad_shard),
        }

      s = state[state_key]
      s['step'] += 1

      # AdamW update on shard
      beta1, beta2 = betas
      exp_avg = s['exp_avg']
      exp_avg_sq = s['exp_avg_sq']

      exp_avg.mul_(beta1).add_(grad_shard, alpha=1 - beta1)
      exp_avg_sq.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1 - beta2)

      # Bias correction
      bias_correction1 = 1 - beta1 ** s['step']
      bias_correction2 = 1 - beta2 ** s['step']
      step_size = group['lr'] / bias_correction1

      denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

      # Weight decay
      if group['weight_decay'] > 0:
        param_shard.mul_(1 - group['lr'] * group['weight_decay'])

      # Update shard
      param_shard.addcdiv_(exp_avg, denom, value=-step_size)

      # Sync updated params to all ranks (no-op when world==1).
      if world > 1:
        with time_ctx('time_ms/zero2_all_gather'):
          dist.all_gather_into_tensor(flat_param, param_shard, group=pg)
      elif total < padded_total:
        # Keep padding clean for determinism.
        flat_param[total:padded_total].zero_()
