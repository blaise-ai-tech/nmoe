"""RDEP forward correctness across modes and precisions.

Runs the same forward test under:
  - single GPU: world == 1
  - single node: world == local_world (CUDA IPC)
  - multi node: world > local_world (IPC + NVSHMEM)

And for profiles:
  - bf16
  - fp8
  - nvfp4

This test is intentionally minimal and deterministic:
  - fixed routing pattern that exercises local + remote paths
  - fixed per-expert diagonal scale (no GEMM correctness confounders)
  - single, explicit oracle: out_ref[t] = Σ_k gate[t,k] * scale[eid[t,k]] * x[t]

Usage:
  python  -m nmoe.test_rdep_forward --profile bf16
  torchrun --nproc_per_node=8 -m nmoe.test_rdep_forward --profile nvfp4
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from nmoe import runtime
from nmoe.rdep import Rdep


def _rms(x: torch.Tensor) -> float:
  return float(x.float().pow(2).mean().sqrt().item())


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise RuntimeError(msg)


def _decode_row_id(row_id: torch.Tensor, *, T: int, K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # Row ID encoding: (rank * T + tok) * K + slot.
  # Ensure int64 math on-device.
  rid = row_id.to(torch.int64)
  slot = torch.remainder(rid, K)
  tmp = rid // K
  tok = torch.remainder(tmp, T)
  src_rank = tmp // T
  return src_rank, tok, slot


def _routing_pattern(
  *,
  T: int,
  K: int,
  world: int,
  local_world: int,
  n_local: int,
  rank: int,
  device,
  same_node_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Deterministic routing that covers:
  - slot0: always local rank
  - slot1: in IPC -> same-node peer; in hybrid -> alternating same-node / other-node (unless same_node_only)
  """
  _require(K == 2, "This test currently requires K=2 (fixed minimal oracle).")

  base = (rank // local_world) * local_world
  local_rank = rank - base
  peer_same_node = base + ((local_rank + 1) % local_world) if world > 1 else rank
  peer_other_node = (rank + local_world) % world if world > local_world else peer_same_node

  tok = torch.arange(T, device=device, dtype=torch.int64)

  # Slot 0: always local.
  dest0 = torch.full((T,), int(rank), device=device, dtype=torch.int64)
  eid0 = dest0 * n_local + (tok % n_local)

  # Slot 1: cover remote paths when available.
  if world == 1:
    dest1 = dest0
  elif world == local_world:
    dest1 = torch.full((T,), int(peer_same_node), device=device, dtype=torch.int64)
  else:
    if same_node_only:
      dest1 = torch.full((T,), int(peer_same_node), device=device, dtype=torch.int64)
    else:
      # Hybrid: alternate destinations to exercise both IPC and NVSHMEM paths.
      dest1 = torch.where((tok & 1) == 0,
                          torch.full((T,), int(peer_same_node), device=device, dtype=torch.int64),
                          torch.full((T,), int(peer_other_node), device=device, dtype=torch.int64))
  eid1 = dest1 * n_local + ((tok + 1) % n_local)

  eid = torch.stack([eid0, eid1], dim=1).to(torch.int32).contiguous()  # [T,2]

  # Gates: fixed non-degenerate mixture.
  gates = torch.empty((T, K), device=device, dtype=torch.float32)
  gates[:, 0] = 0.7
  gates[:, 1] = 0.3
  return eid, gates


def _scale_table(*, world: int, n_local: int, device) -> torch.Tensor:
  # Small set of exactly representable scales (BF16 / FP8 / FP4 friendly).
  vals = torch.tensor([0.25, 0.5, 1.0, 2.0], device=device, dtype=torch.float32)
  gid = torch.arange(world * n_local, device=device, dtype=torch.int64)
  return vals[gid % vals.numel()]


def _bf16_expert_scale(
  Xe_pad: torch.Tensor,
  dest: torch.Tensor,
  row_id: torch.Tensor,
  scales: torch.Tensor,
  *,
  T: int,
  K: int,
  world: int,
  local_world: int,
  n_local: int,
  rank: int,
  same_node_only: bool,
) -> torch.Tensor:
  """Compute Ye_sorted [M,H] in sorted order for BF16 path.

  Important: in multi-GPU runs, each rank receives rows routed by *all* src
  ranks. The correct expert identity for a received row is determined by the
  *source rank's* routing decision (decoded from row_id). This avoids assuming
  any particular expert segment ordering in Xe_pad.
  """
  Xe_sorted = Xe_pad.index_select(0, dest.to(torch.int64))  # [M,H]
  src_rank, tok, slot = _decode_row_id(row_id, T=T, K=K)
  src_rank = src_rank.to(torch.int64)
  tok = tok.to(torch.int64)
  slot = slot.to(torch.int64)

  if world == 1:
    dest1 = src_rank
  elif world == local_world:
    base = (src_rank // int(local_world)) * int(local_world)
    local_rank = src_rank - base
    dest1 = base + torch.remainder(local_rank + 1, int(local_world))
  else:
    base = (src_rank // int(local_world)) * int(local_world)
    local_rank = src_rank - base
    peer_same_node = base + torch.remainder(local_rank + 1, int(local_world))
    peer_other_node = torch.remainder(src_rank + int(local_world), int(world))
    if same_node_only:
      dest1 = peer_same_node
    else:
      dest1 = torch.where((tok & 1) == 0, peer_same_node, peer_other_node)

  dest_rank = torch.where(slot == 0, src_rank, dest1)
  local_eid = torch.where(slot == 0, torch.remainder(tok, int(n_local)), torch.remainder(tok + 1, int(n_local)))
  eid_row = dest_rank * int(n_local) + local_eid

  _require(bool((dest_rank == int(rank)).all().item()),
           "Dispatch invariant failed: received row with non-local dest rank.")
  scale_row = scales.index_select(0, eid_row).to(torch.bfloat16)
  return (Xe_sorted * scale_row.unsqueeze(1)).contiguous()


def _blockscaled_diag_gemm(
  *,
  profile: str,
  Xe_q_pad_u16: torch.Tensor,
  Xe_sf_pad: torch.Tensor,
  offs_pad: torch.Tensor,
  scales_local: torch.Tensor,
  H: int,
) -> torch.Tensor:
  """Compute Ye_pad [M_pad,H] using diagonal weights (per expert).

  For fp8/nvfp4 this stays hermetic: it avoids CuTeDSL JIT (cuda-python), and
  instead uses our own kernels:
    1) unswizzle activation SFs (MMA -> row-major)
    2) dequantize Xe_q + SFA -> BF16 Xe
    3) apply per-expert diagonal scale in BF16
  """
  from nmoe.csrc import rdep as _C

  dev = Xe_q_pad_u16.device
  stream = torch.cuda.current_stream(dev)

  M_pad = int(Xe_q_pad_u16.shape[0])
  E = int(offs_pad.numel())

  # Activation SFA dims.
  sf_k = (H + 31) // 32
  sf_k_pad = ((sf_k + 3) // 4) * 4
  M_e_swizzle = int(Xe_sf_pad.shape[1])

  # Offsets for (un)swizzle kernels require a leading 0.
  offs = torch.cat([torch.zeros(1, device=dev, dtype=torch.int32), offs_pad.to(torch.int32)])

  # Unswizzle activation SFs to row-major [M_pad, sf_k] for dequant.
  Xe_sf_mkl = torch.empty(M_pad, sf_k, device=dev, dtype=torch.uint8)
  _C.unswizzle_sf_strided(
    Xe_sf_pad.data_ptr(), Xe_sf_mkl.data_ptr(),
    offs.data_ptr(), E, sf_k, sf_k_pad, M_pad, M_e_swizzle,
    stream,
  )

  Xe_bf16 = torch.empty(M_pad, H, device=dev, dtype=torch.bfloat16)
  if profile == "fp8":
    Hp = H // 2
    _C.dequant_fp8_to_bf16(Xe_q_pad_u16.data_ptr(), Hp,
                           Xe_sf_mkl.data_ptr(), sf_k,
                           M_pad, H,
                           Xe_bf16.data_ptr(), H,
                           stream)
  elif profile == "nvfp4":
    Hp = H // 4
    _C.dequant_nvfp4_to_bf16(Xe_q_pad_u16.data_ptr(), Hp,
                             Xe_sf_mkl.data_ptr(), sf_k,
                             M_pad, H,
                             Xe_bf16.data_ptr(), H,
                             stream)
  else:
    raise ValueError(f"unsupported profile: {profile}")

  # Per-row expert scale from padded end offsets.
  rows = torch.arange(M_pad, device=dev, dtype=torch.int64)
  local_eid = torch.searchsorted(offs_pad.to(torch.int64), rows, right=True).to(torch.int64)
  scale_row = scales_local.index_select(0, local_eid).to(torch.bfloat16)
  return (Xe_bf16 * scale_row.unsqueeze(1)).contiguous()


def _oracle_out(
  *,
  x: torch.Tensor,
  eid: torch.Tensor,
  gates: torch.Tensor,
  scales: torch.Tensor,
  T: int,
  K: int,
  H: int,
) -> torch.Tensor:
  """Oracle in BF16->FP32 accumulate (matches return_scatter contract)."""
  x_bf16 = x.to(torch.bfloat16)
  out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
  for k in range(K):
    eid_k = eid[:, k].to(torch.int64)
    s_k = scales.index_select(0, eid_k).to(torch.bfloat16)
    y_k = (x_bf16 * s_k.unsqueeze(1)).float()
    out += y_k * gates[:, k].unsqueeze(1)
  return out.to(torch.bfloat16)


@torch.no_grad()
def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile", choices=["bf16", "fp8", "nvfp4"], required=True)
  ap.add_argument("--same-node-only", action="store_true",
                  help="In hybrid runs, route all traffic within the local node (IPC only).")
  args = ap.parse_args()

  rank, world = runtime.init(seed=0)
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  dev = torch.device("cuda", local_rank)

  # Fixed, minimal test sizes (satisfy all kernel constraints).
  T = 128
  H = 256
  K = 2
  n_local = 8
  local_world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

  capacity = int(T * K * max(1, world))

  rdep = Rdep(dim=H, n_local=n_local, topk=K, profile=args.profile, capacity=capacity)

  # Per-rank determinism (avoid identical payloads across ranks).
  torch.manual_seed(1234 + rank)
  torch.cuda.manual_seed(1234 + rank)

  if dist.is_initialized():
    dist.barrier()

  # Deterministic skew to catch missing completion sync.
  if world > 1:
    time.sleep(0.01 * (rank % 8))

  x = torch.randn(T, H, device=dev, dtype=torch.bfloat16).contiguous()
  eid, gates = _routing_pattern(
    T=T, K=K, world=world, local_world=local_world, n_local=n_local, rank=rank, device=dev,
    same_node_only=bool(args.same_node_only),
  )
  scales = _scale_table(world=world, n_local=n_local, device=dev)

  h = rdep.dispatch(x, eid, gates)

  _require(int(h.M) == int(h.dest.numel()), "Dispatch invariant failed: M != dest.numel().")
  _require(int(h.M) == int(h.row_id.numel()), "Dispatch invariant failed: M != row_id.numel().")
  _require(int(h.M) == int(h.gate.numel()), "Dispatch invariant failed: M != gate.numel().")
  _require(int(h.M) <= int(capacity), "Dispatch invariant failed: M > capacity.")
  _require(bool((h.dest.to(torch.int64) >= 0).all().item()), "Dispatch invariant failed: negative dest index.")
  _require(bool((h.dest.to(torch.int64) < int(h.M_pad)).all().item()), "Dispatch invariant failed: dest index >= M_pad.")
  _require(bool((h.offs.to(torch.int64) >= 0).all().item()), "Dispatch invariant failed: negative offs.")
  _require(int(h.offs.numel()) == int(n_local), "Dispatch invariant failed: offs length != n_local.")
  _require(int(h.M_pad) >= int(h.M), "Dispatch invariant failed: M_pad < M.")
  _require(int(h.M_pad) == int(h.offs[-1].item()), "Dispatch invariant failed: offs[-1] != M_pad.")

  # Compute per-rank reference (BF16 output).
  out_ref = _oracle_out(x=x, eid=eid, gates=gates, scales=scales, T=T, K=K, H=H)

  if args.profile == "bf16":
    Ye_sorted = _bf16_expert_scale(
      h.Xe, h.dest, h.row_id, scales,
      T=T, K=K, world=world, local_world=local_world, n_local=n_local, rank=rank,
      same_node_only=bool(args.same_node_only),
    )
  else:
    scales_local = scales.index_select(0, (torch.arange(n_local, device=dev, dtype=torch.int64) + rank * n_local))
    Ye_pad = _blockscaled_diag_gemm(
      profile=args.profile,
      Xe_q_pad_u16=h.Xe_q,
      Xe_sf_pad=h.Xe_sf,
      offs_pad=h.offs,
      scales_local=scales_local,
      H=H,
    )
    Ye_sorted = Ye_pad.index_select(0, h.dest.to(torch.int64)).contiguous()

  out = rdep.return_scatter(Ye_sorted, h, T, gates)

  _require(bool(torch.isfinite(out).all().item()), "Output contains NaN/Inf.")

  err = (out.float() - out_ref.float()).contiguous()
  rms_err = _rms(err)
  rms_ref = max(_rms(out_ref.float()), 1e-8)
  rel = rms_err / rms_ref
  max_abs = float(err.abs().max().item())

  if args.profile == "bf16":
    rel_limit = 1e-3
    max_abs_limit = 2e-2
  elif args.profile == "fp8":
    rel_limit = 5e-2
    max_abs_limit = 4e-1
  else:  # nvfp4
    rel_limit = 1.5e-1
    max_abs_limit = 1.5

  _require(rel <= rel_limit and max_abs <= max_abs_limit,
           f"Forward mismatch: rel_rms={rel:.3e} (limit {rel_limit:.3e}) "
           f"max_abs={max_abs:.3e} (limit {max_abs_limit:.3e})")

  if dist.is_initialized():
    dist.barrier()
  if rank == 0:
    print(f"OK rdep_forward profile={args.profile} world={world} local_world={local_world} T={T} H={H} K={K}")


if __name__ == "__main__":
  t0 = time.time()
  main()
  torch.cuda.synchronize()
  print(f"done in {time.time() - t0:.2f}s")
