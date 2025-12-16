"""RDEP backward comm correctness across modes and precisions.

Validates the two distributed backward primitives that unblock training:
  1) gather_dy_dist_bf16: stages dOut via IPC/NVSHMEM (push), computes dYe + dGate
  2) scatter_dx_dist_bf16: sends dXe rows back to token owners and reduces into dX

Runs under:
  - single GPU: world == 1
  - single node: world == local_world (CUDA IPC)
  - multi node: world > local_world (IPC + NVSHMEM)

And for profiles:
  - bf16
  - fp8
  - nvfp4

Usage:
  python  -m nmoe.test_rdep_backward --profile bf16
  torchrun --nproc_per_node=8 -m nmoe.test_rdep_backward --profile nvfp4
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from nmoe import runtime
from nmoe.rdep import Rdep


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise RuntimeError(msg)


def _decode_row_id(row_id: torch.Tensor, *, T: int, K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # Row ID encoding: (rank * T + tok) * K + slot.
  rid = row_id.to(torch.int64)
  slot = torch.remainder(rid, K)
  tmp = rid // K
  tok = torch.remainder(tmp, T)
  src_rank = tmp // T
  return src_rank, tok, slot


def _routing_pattern(*, T: int, K: int, world: int, local_world: int, n_local: int, rank: int, device) -> tuple[torch.Tensor, torch.Tensor]:
  _require(K == 2, "This test currently requires K=2 (fixed minimal oracle).")

  base = (rank // local_world) * local_world
  local_rank = rank - base
  peer_same_node = base + ((local_rank + 1) % local_world) if world > 1 else rank
  peer_other_node = (rank + local_world) % world if world > local_world else peer_same_node

  tok = torch.arange(T, device=device, dtype=torch.int64)

  # Slot 0: always local.
  dest0 = torch.full((T,), int(rank), device=device, dtype=torch.int64)
  eid0 = dest0 * n_local + (tok % n_local)

  # Slot 1: exercise remote paths when available.
  if world == 1:
    dest1 = dest0
  elif world == local_world:
    dest1 = torch.full((T,), int(peer_same_node), device=device, dtype=torch.int64)
  else:
    dest1 = torch.where((tok & 1) == 0,
                        torch.full((T,), int(peer_same_node), device=device, dtype=torch.int64),
                        torch.full((T,), int(peer_other_node), device=device, dtype=torch.int64))
  eid1 = dest1 * n_local + ((tok + 1) % n_local)

  eid = torch.stack([eid0, eid1], dim=1).to(torch.int32).contiguous()  # [T,2]

  gates = torch.empty((T, K), device=device, dtype=torch.float32)
  gates[:, 0] = 0.7
  gates[:, 1] = 0.3
  return eid, gates


def _rms(x: torch.Tensor) -> float:
  return float(x.float().pow(2).mean().sqrt().item())


@torch.no_grad()
def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile", choices=["bf16", "fp8", "nvfp4"], required=True)
  args = ap.parse_args()

  rank, world = runtime.init(seed=0)
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  dev = torch.device("cuda", local_rank)
  local_world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

  # Small sizes (fast, satisfy vectorization constraints).
  T = 128
  H = 256
  K = 2
  n_local = 8

  capacity = int(T * K * max(1, world))
  rdep = Rdep(dim=H, n_local=n_local, topk=K, profile=args.profile, capacity=capacity)

  torch.manual_seed(1234 + rank)
  torch.cuda.manual_seed(1234 + rank)

  if dist.is_initialized():
    dist.barrier()

  if world > 1:
    time.sleep(0.01 * (rank % 8))

  x = torch.randn(T, H, device=dev, dtype=torch.bfloat16).contiguous()
  eid, gates = _routing_pattern(T=T, K=K, world=world, local_world=local_world, n_local=n_local, rank=rank, device=dev)

  h = rdep.dispatch(x, eid, gates)
  _require(int(h.M) == int(h.row_id.numel()), "Dispatch invariant failed: M != row_id.numel().")

  M = int(h.M)
  row_id = h.row_id
  gate_sorted = h.gate

  # Deterministic per-rank payloads.
  torch.manual_seed(4321 + rank)
  torch.cuda.manual_seed(4321 + rank)
  Ye_sorted = torch.randn(M, H, device=dev, dtype=torch.bfloat16).contiguous()
  dOut = torch.randn(T, H, device=dev, dtype=torch.bfloat16).contiguous()

  # =========================
  # gather_dy (distributed)
  # =========================
  from nmoe.csrc import rdep as _C

  # Pre-fill with NaN to catch any silent no-op/partial writes.
  dYe = torch.full((M, H), float("nan"), device=dev, dtype=torch.bfloat16)
  dGate_sorted = torch.full((M,), float("nan"), device=dev, dtype=torch.float32)
  dGates_tk = torch.full((T, K), float("nan"), device=dev, dtype=torch.float32)

  stream = torch.cuda.current_stream(dev)
  if dist.is_available() and dist.is_initialized() and world > 1:
    _C.gather_dy_dist_bf16(
      dOut.data_ptr(),
      eid.data_ptr(),
      Ye_sorted.data_ptr(),
      row_id.data_ptr(),
      gate_sorted.data_ptr(),
      dYe.data_ptr(),
      dGate_sorted.data_ptr(),
      dGates_tk.data_ptr(),
      M, T, H, K,
      stream,
    )
  else:
    _C.gather_dy_bf16(
      dOut.data_ptr(),
      Ye_sorted.data_ptr(),
      row_id.data_ptr(),
      gate_sorted.data_ptr(),
      dYe.data_ptr(),
      dGate_sorted.data_ptr(),
      M, T, H, K,
      stream,
    )
    _C.scatter_gate_bf16(
      dGate_sorted.data_ptr(),
      row_id.data_ptr(),
      dGates_tk.data_ptr(),
      M, T, K,
      stream,
    )

  # Reference: build global dOut[world*T, H] then gather by src_tok.
  if dist.is_available() and dist.is_initialized():
    gathered = [torch.empty_like(dOut) for _ in range(world)]
    dist.all_gather(gathered, dOut)
    dOut_all = torch.stack(gathered, dim=0).reshape(world * T, H)  # [world*T, H]
  else:
    dOut_all = dOut.reshape(T, H)

  src_rank, tok, slot = _decode_row_id(row_id, T=T, K=K)
  src_tok = (src_rank * int(T) + tok).to(torch.int64)
  dy_ref = dOut_all.index_select(0, src_tok).to(torch.bfloat16)  # [M,H]

  dYe_ref = (dy_ref.float() * gate_sorted.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
  dGate_ref = (Ye_sorted.float() * dy_ref.float()).sum(dim=1)

  _require(bool(torch.isfinite(dYe).all().item()), "dYe contains NaN/Inf.")
  _require(bool(torch.isfinite(dGate_sorted).all().item()), "dGate_sorted contains NaN/Inf.")
  _require(bool(torch.isfinite(dGates_tk).all().item()), "dGates_tk contains NaN/Inf.")

  # dYe is BF16, allow exact match (should be deterministic).
  max_abs_dYe = float((dYe.float() - dYe_ref.float()).abs().max().item())
  _require(max_abs_dYe == 0.0, f"gather_dy dYe mismatch: max_abs={max_abs_dYe:.3e}")

  # dGate is FP32 reduction; allow tiny tolerance.
  err_gate = (dGate_sorted - dGate_ref).abs()
  max_abs_gate = float(err_gate.max().item())
  _require(max_abs_gate <= 5e-3, f"gather_dy dGate_sorted mismatch: max_abs={max_abs_gate:.3e}")

  # Validate dGates_tk by gathering dGate_ref + row_id across ranks and scattering to tok-slots.
  if dist.is_available() and dist.is_initialized():
    # Gather variable-length tensors via padding.
    m = torch.tensor([M], device=dev, dtype=torch.int32)
    m_all = [torch.zeros_like(m) for _ in range(world)]
    dist.all_gather(m_all, m)
    m_max = int(torch.stack(m_all).max().item())

    rid_pad = torch.full((m_max,), -1, device=dev, dtype=torch.int64)
    gate_pad = torch.zeros((m_max,), device=dev, dtype=torch.float32)
    rid_pad[:M] = row_id
    gate_pad[:M] = dGate_ref

    rid_all = [torch.empty_like(rid_pad) for _ in range(world)]
    gate_all = [torch.empty_like(gate_pad) for _ in range(world)]
    dist.all_gather(rid_all, rid_pad)
    dist.all_gather(gate_all, gate_pad)

    rid_cat = torch.cat(rid_all, dim=0)
    gate_cat = torch.cat(gate_all, dim=0)
    valid = rid_cat >= 0
    rid_cat = rid_cat[valid]
    gate_cat = gate_cat[valid]

    src_rank2, tok2, slot2 = _decode_row_id(rid_cat, T=T, K=K)
    mine = src_rank2 == int(rank)
    tok2 = tok2[mine].to(torch.int64)
    slot2 = slot2[mine].to(torch.int64)
    gate2 = gate_cat[mine]

    idx = (tok2 * int(K) + slot2).to(torch.int64)
    _require(int(idx.numel()) == int(T * K),
             f"dGates reference coverage failed: got {int(idx.numel())} entries, expected {int(T * K)}")
    _require(int(idx.unique().numel()) == int(T * K),
             "dGates reference indices not unique (duplicate tok-slot writes?)")

    dGates_ref_flat = torch.full((int(T * K),), float("nan"), device=dev, dtype=torch.float32)
    dGates_ref_flat.index_copy_(0, idx, gate2)
    _require(not bool(torch.isnan(dGates_ref_flat).any().item()),
             "dGates reference has holes (missing tok-slot writes?)")
    dGates_ref = dGates_ref_flat.view(T, K)
  else:
    src_rank2, tok2, slot2 = _decode_row_id(row_id, T=T, K=K)
    _require(bool((src_rank2 == 0).all().item()), "world==1 invariant failed: src_rank != 0")
    idx = (tok2.to(torch.int64) * int(K) + slot2.to(torch.int64)).to(torch.int64)
    dGates_ref_flat = torch.full((int(T * K),), float("nan"), device=dev, dtype=torch.float32)
    dGates_ref_flat.index_copy_(0, idx, dGate_ref)
    _require(not bool(torch.isnan(dGates_ref_flat).any().item()),
             "world==1 dGates reference has holes (missing tok-slot writes?)")
    dGates_ref = dGates_ref_flat.view(T, K)

  max_abs_gates = float((dGates_tk - dGates_ref).abs().max().item())
  _require(max_abs_gates <= 5e-3, f"gather_dy dGates_tk mismatch: max_abs={max_abs_gates:.3e}")

  # =========================
  # scatter_dx (distributed)
  # =========================
  torch.manual_seed(9876 + rank)
  torch.cuda.manual_seed(9876 + rank)
  dXe_sorted = torch.randn(M, H, device=dev, dtype=torch.bfloat16).contiguous()

  dX = torch.zeros(T, H, device=dev, dtype=torch.float32)
  if dist.is_available() and dist.is_initialized() and world > 1:
    _C.scatter_dx_dist_bf16(
      dXe_sorted.data_ptr(),
      row_id.data_ptr(),
      dX.data_ptr(),
      M, T, H, K,
      stream,
    )
  else:
    # Single GPU reference path using existing scatter_dx_bf16 (needs padded layout).
    # Build dXe_pad by scattering sorted rows into padded indices via `dest`.
    M_pad = int(h.M_pad)
    dXe_pad = torch.zeros(M_pad, H, device=dev, dtype=torch.bfloat16)
    _C.scatter_sorted_to_pad_with_dest_bf16(
      dXe_sorted.data_ptr(),
      h.dest.data_ptr(),
      dXe_pad.data_ptr(),
      M, H,
      stream,
    )
    _C.scatter_dx_bf16(
      dXe_pad.data_ptr(),
      h.dest.data_ptr(),
      row_id.data_ptr(),
      dX.data_ptr(),
      M, T, H, K,
      stream,
    )

  # Reference via gathering all (row_id, dXe_sorted) and index_add on tok.
  if dist.is_available() and dist.is_initialized():
    m = torch.tensor([M], device=dev, dtype=torch.int32)
    m_all = [torch.zeros_like(m) for _ in range(world)]
    dist.all_gather(m_all, m)
    m_max = int(torch.stack(m_all).max().item())

    rid_pad = torch.full((m_max,), -1, device=dev, dtype=torch.int64)
    dxe_pad = torch.zeros((m_max, H), device=dev, dtype=torch.bfloat16)
    rid_pad[:M] = row_id
    dxe_pad[:M] = dXe_sorted

    rid_all = [torch.empty_like(rid_pad) for _ in range(world)]
    dxe_all = [torch.empty_like(dxe_pad) for _ in range(world)]
    dist.all_gather(rid_all, rid_pad)
    dist.all_gather(dxe_all, dxe_pad)

    rid_cat = torch.cat(rid_all, dim=0)
    dxe_cat = torch.cat(dxe_all, dim=0)
    valid = rid_cat >= 0
    rid_cat = rid_cat[valid]
    dxe_cat = dxe_cat[valid]

    src_rank3, tok3, _slot3 = _decode_row_id(rid_cat, T=T, K=K)
    mine = src_rank3 == int(rank)
    tok3 = tok3[mine].to(torch.int64)
    dxe3 = dxe_cat[mine].float()

    dX_ref = torch.zeros(T, H, device=dev, dtype=torch.float32)
    dX_ref.index_add_(0, tok3, dxe3)
  else:
    src_rank3, tok3, _slot3 = _decode_row_id(row_id, T=T, K=K)
    _require(bool((src_rank3 == 0).all().item()), "world==1 invariant failed: src_rank != 0")
    dX_ref = torch.zeros(T, H, device=dev, dtype=torch.float32)
    dX_ref.index_add_(0, tok3.to(torch.int64), dXe_sorted.float())

  err_dx = (dX - dX_ref)
  rms_dx = _rms(err_dx)
  max_abs_dx = float(err_dx.abs().max().item())
  _require(rms_dx <= 1e-3 and max_abs_dx <= 5e-3,
           f"scatter_dx mismatch: rms={rms_dx:.3e} max_abs={max_abs_dx:.3e}")

  if dist.is_initialized():
    dist.barrier()
  if rank == 0:
    print(f"OK rdep_backward profile={args.profile} world={world} local_world={local_world} T={T} H={H} K={K}")


if __name__ == "__main__":
  t0 = time.time()
  main()
  torch.cuda.synchronize()
  print(f"done in {time.time() - t0:.2f}s")
