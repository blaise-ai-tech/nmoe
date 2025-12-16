from dataclasses import dataclass
from typing import Optional
import os
import sys

import torch
import torch.distributed as dist
import numpy as np
from contextlib import nullcontext

# Import C extension (built in csrc/)
from .csrc import rdep as _C
from .ggemm import expert as expert_bf16


@dataclass
class DispatchHandle:
    """Handle returned by dispatch() for BF16 path."""
    Xe: torch.Tensor         # [M_pad, H] BF16 - padded recv buffer
    offs: torch.Tensor       # [n_local] cumulative offsets (int32)
    dest: torch.Tensor       # [M] destination indices (int32)
    row_id: torch.Tensor     # [M] encoded (src_rank,tok,slot) int64, sorted order
    gate: torch.Tensor       # [M] gate weights float32, sorted order
    M: int                   # actual received count
    M_pad: int               # padded count for grouped GEMM


@dataclass
class DispatchBlockscaledHandle:
    """Handle returned by dispatch() for blockscaled path."""
    Xe_q: torch.Tensor       # [M_pad, Hp] packed quantized activations (uint16)
    Xe_sf: torch.Tensor      # [E, M_e_swizzle, sf_k_pad] SFA per-expert, MMA layout (uint8)
    offs: torch.Tensor       # [n_local] cumulative offsets (int32)
    dest: torch.Tensor       # [M] destination indices (int32)
    row_id: torch.Tensor     # [M] encoded (src_rank,tok,slot) int64, sorted order
    gate: torch.Tensor       # [M] gate weights float32, sorted order
    M: int                   # actual received count
    M_pad: int               # padded count


def _get_local_world_size() -> int:
    """Get local world size (GPUs on this node)."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

_CPU_PG = None
_CPU_PG_WORLD: int | None = None


def _cpu_pg():
    """CPU-only process group for bootstrap collectives (Gloo)."""
    global _CPU_PG, _CPU_PG_WORLD
    if not dist.is_initialized():
        return None
    world = int(dist.get_world_size())
    if world <= 1:
        return None
    if _CPU_PG is None or _CPU_PG_WORLD != world:
        _CPU_PG = dist.new_group(backend="gloo")
        _CPU_PG_WORLD = world
    return _CPU_PG


_CUDA_TIME = None  # lazily initialized to avoid import cycles (rdep <-> metrics <-> model)
_NVTX = None


def _maybe_cuda_time(tag: str):
    global _CUDA_TIME
    if _CUDA_TIME is None:
        if os.getenv('NMOE_TIMERS', '1') in ('0', 'false', 'False'):
            _CUDA_TIME = False
        else:
            try:
                from nmoe.metrics import cuda_time as _ct  # local import (metrics imports model)
                _CUDA_TIME = _ct
            except Exception:
                _CUDA_TIME = False
    if _CUDA_TIME is False:
        return nullcontext()
    return _CUDA_TIME(tag)  # type: ignore[operator]


def _maybe_nvtx(tag: str):
    global _NVTX
    if _NVTX is None:
        _NVTX = os.getenv('NMOE_NVTX', '0') in ('1', 'true', 'True')
    if not _NVTX:
        return nullcontext()
    try:
        return torch.cuda.nvtx.range(tag)
    except Exception:
        return nullcontext()


class Rdep:
    #TODO(xjdr): this is an odd global here
    PROFILES = {'bf16': -1, 'fp8': 0, 'nvfp4': 1}

    def __init__(self, dim: int, n_local: int, topk: int, profile: str = 'nvfp4',
                 capacity: int = 65536):
        assert profile in self.PROFILES, f"profile must be one of {list(self.PROFILES.keys())}"

        self.dim = dim
        self.n_local = n_local
        self.topk = topk
        self.profile = profile
        self.capacity = capacity
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_world = _get_local_world_size()
        _C.init(self.rank, self.world, self.local_world)
        mode_int = _C.get_mode()
        self._mode = {0: 'single', 1: 'ipc', 2: 'hybrid'}[mode_int]
        if self._mode == 'hybrid' and not _C.has_nvshmem():
            raise RuntimeError(
                f"Multi-node configuration (world={self.world} > local_world={self.local_world}) "
                "requires NVSHMEM support. Rebuild rdep with NVSHMEM or use single-node."
            )

        if self._mode == 'hybrid':
            self._setup_hybrid()
        elif self._mode == 'ipc':
            _C.alloc_bf16(capacity, dim, n_local)
            if profile != 'bf16':
                _C.alloc_blockscaled(capacity, dim, n_local, self.PROFILES[profile])
            self._setup_ipc()
        elif self._mode == 'single':
            _C.alloc_bf16(capacity, dim, n_local)
            _C.sync_buffer_ptrs_bf16()
            if profile != 'bf16':
                _C.alloc_blockscaled(capacity, dim, n_local, self.PROFILES[profile])
                _C.sync_buffer_ptrs_blockscaled()

        self._expert_loads = None

    def _setup_hybrid(self):
        if cpu_pg is None:
            raise RuntimeError("[RDEP] internal error: expected dist to be initialized for hybrid bootstrap")

        uid_size = _C.nvshmem_get_uid_size()
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: Getting UID (size={uid_size})...", flush=True)
            uid = _C.nvshmem_get_uid()
            print(f"[RDEP] rank={self.rank}: Got UID", flush=True)
        else:
            uid = None

        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: Broadcasting UID via CPU...", flush=True)
        uid_list = [uid]
        dist.broadcast_object_list(uid_list, src=0, group=cpu_pg)
        uid = uid_list[0]
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: UID broadcast complete", flush=True)
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: Initializing NVSHMEM...", flush=True)
        _C.nvshmem_init(uid, self.rank, self.world, self.local_world)
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: NVSHMEM initialized!", flush=True)

        _C.nvshmem_alloc_bf16(self.capacity, self.dim, self.n_local)
        local_rank = self.rank % self.local_world
        node_id = self.rank // self.local_world
        local_handle_bf16 = _C.nvshmem_get_ipc_handle_bf16()
        all_handles_bf16 = [None] * self.world
        dist.all_gather_object(all_handles_bf16, local_handle_bf16, group=cpu_pg)
        local_handles_bf16 = []
        for r in range(self.world):
            if r // self.local_world == node_id:
                local_handles_bf16.append(all_handles_bf16[r])
        local_handles_bf16_np = np.concatenate(local_handles_bf16)
        _C.nvshmem_open_ipc_handles_bf16(local_handles_bf16_np, self.local_world)
        _C.nvshmem_sync_ipc_buffer_ptrs_bf16()
        dist.barrier(group=cpu_pg)

    def _setup_ipc(self):
        """Exchange IPC handles via NCCL all_gather (one-time at init)."""
        local_handle_bf16 = _C.get_ipc_handle_bf16()
        handle_tensor_bf16 = torch.from_numpy(local_handle_bf16).cuda()

        all_handles_bf16 = [torch.zeros_like(handle_tensor_bf16) for _ in range(self.world)]
        dist.all_gather(all_handles_bf16, handle_tensor_bf16)

        all_handles_bf16_np = np.concatenate([h.cpu().numpy() for h in all_handles_bf16])
        _C.open_ipc_handles_bf16(all_handles_bf16_np, self.world)
        _C.sync_buffer_ptrs_bf16()

        if self.profile != 'bf16':
            local_handle_block = _C.get_ipc_handle_blockscaled()
            handle_tensor_block = torch.from_numpy(local_handle_block).cuda()
            all_handles_block = [torch.zeros_like(handle_tensor_block) for _ in range(self.world)]
            dist.all_gather(all_handles_block, handle_tensor_block)
            all_handles_block_np = np.concatenate([h.cpu().numpy() for h in all_handles_block])
            _C.open_ipc_handles_blockscaled(all_handles_block_np, self.world)
            _C.sync_buffer_ptrs_blockscaled()

    def dispatch(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                 stream: Optional[torch.cuda.Stream] = None):
        T, H = x.shape
        K = eid.shape[1]
        device = x.device
        if stream is None:
            stream = torch.cuda.current_stream(device)

        need = int(T * K * self.world)
        if self.capacity < need:
            raise RuntimeError(
                f"[RDEP] capacity too small: capacity={self.capacity:,} need>={need:,} (T={T:,} K={K} world={self.world}). "
                "Set capacity to worst-case T*K*world (no silent truncation)."
            )

        x = x.contiguous().bfloat16()
        eid = eid.contiguous().int()
        gates_fp32 = gates.contiguous().float()

        if self.profile == 'bf16':
            Xe_pad, offs_pad, dest, row_id, gate_sorted = _DispatchBf16.apply(self, x, eid, gates_fp32)
            M_recv = int(dest.numel())
            M_pad = int(Xe_pad.size(0))
            h = DispatchHandle(
                Xe=Xe_pad,
                offs=offs_pad,
                dest=dest,
                row_id=row_id,
                gate=gate_sorted,
                M=M_recv,
                M_pad=M_pad,
            )
            self._last_offs = h.offs
            return h
        else:
            pack_factor = 2 if self.profile == 'fp8' else 4
            Hp = H // pack_factor
            Hsf = (H + 31) // 32

            align = 128
            max_pad = int(self.capacity + self.n_local * (align - 1))
            sf_k_pad = ((Hsf + 3) // 4) * 4
            M_e_swizzle_cap = ((self.capacity + 127) // 128) * 128

            scratch = getattr(self, "_dispatch_scratch_blockscaled", None)
            if scratch is None or scratch.get("device") != device:
                scratch = {
                    "device": device,
                    "Xe_q_pad": torch.empty(max_pad, Hp, device=device, dtype=torch.uint16),
                    "Xe_sf_pad": torch.empty(self.n_local, M_e_swizzle_cap, sf_k_pad, device=device, dtype=torch.uint8),
                    "offs_pad": torch.empty(self.n_local, device=device, dtype=torch.int32),
                    "dest": torch.empty(self.capacity, device=device, dtype=torch.int32),
                    "row_id": torch.empty(self.capacity, device=device, dtype=torch.int64),
                    "gate_sorted": torch.empty(self.capacity, device=device, dtype=torch.float32),
                    "M_pad_host": torch.empty(1, device="cpu", dtype=torch.int32, pin_memory=True),
                }
                self._dispatch_scratch_blockscaled = scratch

            Xe_q_pad = scratch["Xe_q_pad"]
            Xe_sf_pad = scratch["Xe_sf_pad"]
            offs_pad = scratch["offs_pad"]
            dest = scratch["dest"]
            row_id = scratch["row_id"]
            gate_sorted = scratch["gate_sorted"]
            M_pad_tensor = scratch["M_pad_host"]

            if self._mode == 'hybrid':
                Xe_pad = torch.empty(max_pad, int(H), device=device, dtype=torch.bfloat16)

                M_recv = _C.dispatch(
                    x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
                    int(T), int(K),
                    Xe_pad.data_ptr(),
                    offs_pad.data_ptr(),
                    dest.data_ptr(),
                    row_id.data_ptr(),
                    gate_sorted.data_ptr(),
                    M_pad_tensor.data_ptr(),
                    stream,
                )
                torch.cuda.current_stream(device).synchronize()  # Need M_pad from host
                M_pad = int(M_pad_tensor.item())

                if M_recv <= 0:
                    h = DispatchBlockscaledHandle(
                        Xe_q=Xe_q_pad[:0],
                        Xe_sf=Xe_sf_pad,
                        offs=offs_pad,
                        dest=dest[:0],
                        row_id=row_id[:0],
                        gate=gate_sorted[:0],
                        M=0,
                        M_pad=0,
                    )
                    self._last_offs = h.offs
                    return h

                offs_with0 = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offs_pad])

                if self.profile == 'fp8':
                    _C.quant_fp8_sf_strided_mma(
                        Xe_pad.data_ptr(), int(H),
                        Xe_q_pad.data_ptr(), Hp,
                        Xe_sf_pad.data_ptr(),
                        offs_with0.data_ptr(),
                        self.n_local, M_e_swizzle_cap,
                        int(M_pad), int(H),
                        stream,
                    )
                else:  # nvfp4
                    _C.quant_nvfp4_sf_strided_mma(
                        Xe_pad.data_ptr(), int(H),
                        Xe_q_pad.data_ptr(), Hp,
                        Xe_sf_pad.data_ptr(),
                        offs_with0.data_ptr(),
                        self.n_local, M_e_swizzle_cap,
                        int(M_pad), int(H),
                        stream,
                    )
            else:
                M_recv = _C.dispatch_blockscaled(
                    x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
                    T, K,
                    Xe_q_pad.data_ptr(), Xe_sf_pad.data_ptr(),
                    offs_pad.data_ptr(), dest.data_ptr(),
                    row_id.data_ptr(), gate_sorted.data_ptr(),
                    M_pad_tensor.data_ptr(),
                    stream)
                M_pad = int(M_pad_tensor.item())

            h = DispatchBlockscaledHandle(
                Xe_q=Xe_q_pad[:M_pad],
                Xe_sf=Xe_sf_pad,
                offs=offs_pad,
                dest=dest[:M_recv],
                row_id=row_id[:M_recv],
                gate=gate_sorted[:M_recv],
                M=M_recv,
                M_pad=M_pad,
            )
            self._last_offs = h.offs
            return h

    def return_scatter(self, Ye: torch.Tensor, handle, T: int, gates_tk: torch.Tensor,
                       y_shared: Optional[torch.Tensor] = None,
                       stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        H = Ye.shape[1]
        device = Ye.device
        if stream is None:
            stream = torch.cuda.current_stream(device)

        out_bf16 = _ReturnScatter.apply(self, Ye, handle.row_id, handle.gate, gates_tk, int(T), int(self.topk))
        out_f32 = out_bf16.float()

        if y_shared is not None:
            out_f32 = out_f32 + y_shared.float()

        return out_f32.to(dtype=torch.bfloat16)

    def moe_bf16(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        if self.profile != 'bf16':
            raise RuntimeError("moe_bf16() requires profile='bf16'")
        return _MoEBf16Fused.apply(self, x, eid, gates, W1, W3, W2)

    def moe_blockscaled(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                        W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
                        W_cache) -> torch.Tensor:
        if self.profile == 'bf16':
            raise RuntimeError("moe_blockscaled() requires profile in {'fp8','nvfp4'}")
        return _MoEBlockscaledFused.apply(self, x, eid, gates, W1, W3, W2, W_cache)

    def get_expert_loads(self) -> torch.Tensor:
        """Get expert load counts from last dispatch for aux-free balancing.

        Returns:
            loads: [n_local] tensor of per-expert token counts (normalized)
        """
        # Load counts from last dispatch padded offsets
        if not hasattr(self, "_last_offs") or self._last_offs is None:
            raise RuntimeError("get_expert_loads() requires a prior dispatch; no offsets cached.")
        offs = self._last_offs
        z = torch.zeros(1, dtype=offs.dtype, device=offs.device)
        cnt = torch.diff(torch.cat([z, offs])).float()
        total = cnt.sum().clamp_min(1.0)
        return cnt / total


class _MoEBf16Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        device = x.device
        stream = torch.cuda.current_stream(device)

        x = x.contiguous().bfloat16()
        eid = eid.contiguous().int()
        gates = gates.contiguous().bfloat16()
        gates_fp32 = gates.detach().float()

        T, H = x.shape
        K = int(eid.shape[1])

        offs_pad = torch.empty(rdep.n_local, device=device, dtype=torch.int32)
        # dispatch_meta_bf16 uses this host int32 (pinned) as scratch to read back M_recv.
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        # BF16 fused path uses align=128 for consistent GEMM padding
        align = 128

        with _maybe_nvtx('rdep/dispatch_meta'), _maybe_cuda_time('time_ms/rdep_dispatch_meta'):
            M_recv = _C.dispatch_meta_bf16(
                x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
                int(T), int(K), align,
                offs_pad.data_ptr(), M_host.data_ptr(),
                stream,
            )

        out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
        if M_recv <= 0:
            ctx.save_for_backward(x, eid, gates, W1, W3, W2)
            return out_f32.to(dtype=torch.bfloat16)

        # Avoid a second host sync for exact M_pad:
        # - Exact padded total is sum_e align_up(cnt_e, align) and depends on routing.
        # - For BF16 grouped GEMM we only need per-expert offsets to be aligned.
        # - Over-allocate to a deterministic upper bound and extend the *last* expert's
        #   padded region. Extra rows are zeroed and therefore compute to zero.
        max_pad = (int(M_recv) + int(rdep.n_local) * (align - 1) + (align - 1)) // align * align
        # Ensure the last expert's padded segment reaches max_pad (keeps per-expert alignment).
        offs_pad[-1] = int(max_pad)

        Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_xe'), _maybe_cuda_time('time_ms/rdep_gather_xe'):
            _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

        with _maybe_nvtx('rdep/expert_mlp_bf16'), _maybe_cuda_time('time_ms/expert_mlp_bf16'):
            Ye_pad = expert_bf16(Xe_pad, W1, W3, W2, offs_pad)
        Ye_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_from_pad'), _maybe_cuda_time('time_ms/rdep_gather_from_pad'):
            _C.gather_from_pad_bf16(Ye_pad.data_ptr(), Ye_sorted.data_ptr(), int(M_recv), int(H), stream)

        with _maybe_nvtx('rdep/return_scatter'), _maybe_cuda_time('time_ms/rdep_return_scatter'):
            _C.return_scatter(
                Ye_sorted.data_ptr(),
                out_f32.data_ptr(),
                int(M_recv), int(T), int(K),
                stream,
            )

        ctx.save_for_backward(x, eid, gates, W1, W3, W2)
        return out_f32.to(dtype=torch.bfloat16)

    @staticmethod
    def backward(ctx, dOut: torch.Tensor):
        x, eid, gates, W1, W3, W2 = ctx.saved_tensors
        device = x.device
        stream = torch.cuda.current_stream(device)

        x = x.contiguous().bfloat16()
        eid = eid.contiguous().int()
        gates = gates.contiguous().bfloat16()
        gates_fp32 = gates.detach().float()
        dOut = dOut.contiguous().bfloat16()

        T, H = x.shape
        K = int(eid.shape[1])

        offs_pad = torch.empty(int(W1.size(0)), device=device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        # BF16 fused path uses align=128 for consistent GEMM padding
        align = 128

        with _maybe_nvtx('rdep/dispatch_meta'), _maybe_cuda_time('time_ms/rdep_dispatch_meta'):
            M_recv = _C.dispatch_meta_bf16(
                x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
                int(T), int(K), align,
                offs_pad.data_ptr(), M_host.data_ptr(),
                stream,
            )

        if M_recv <= 0:
            dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
            dGates = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)
            dW1 = torch.zeros_like(W1)
            dW3 = torch.zeros_like(W3)
            dW2 = torch.zeros_like(W2)
            return None, dX, None, dGates, dW1, dW3, dW2
        max_pad = (int(M_recv) + int(offs_pad.numel()) * (align - 1) + (align - 1)) // align * align
        offs_pad[-1] = int(max_pad)

        Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_xe'), _maybe_cuda_time('time_ms/rdep_gather_xe'):
            _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

        row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
        gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
        _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

        with torch.enable_grad():
            Xe_pad = Xe_pad.requires_grad_(True)
            with _maybe_nvtx('rdep/expert_mlp_bf16'), _maybe_cuda_time('time_ms/expert_mlp_bf16'):
                Ye_pad = expert_bf16(Xe_pad, W1, W3, W2, offs_pad)

        Ye_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_from_pad'), _maybe_cuda_time('time_ms/rdep_gather_from_pad'):
            _C.gather_from_pad_bf16(Ye_pad.detach().data_ptr(), Ye_sorted.data_ptr(), int(M_recv), int(H), stream)

        dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
        dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
        dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            _C.gather_dy_dist_bf16(
                dOut.data_ptr(),
                eid.data_ptr(),
                Ye_sorted.data_ptr(),
                row_id.data_ptr(),
                gate_sorted.data_ptr(),
                dYe_sorted.data_ptr(),
                dGate_sorted.data_ptr(),
                dGates_tk_f32.data_ptr(),
                int(M_recv), int(T), int(H), int(K),
                stream,
            )
        else:
            _C.gather_dy_bf16(
                dOut.data_ptr(),
                Ye_sorted.data_ptr(),
                row_id.data_ptr(),
                gate_sorted.data_ptr(),
                dYe_sorted.data_ptr(),
                dGate_sorted.data_ptr(),
                int(M_recv), int(T), int(H), int(K),
                stream,
            )
            _C.scatter_gate_bf16(
                dGate_sorted.data_ptr(),
                row_id.data_ptr(),
                dGates_tk_f32.data_ptr(),
                int(M_recv), int(T), int(K),
                stream,
            )

        dYe_pad = torch.zeros(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/scatter_sorted_to_pad'), _maybe_cuda_time('time_ms/rdep_scatter_sorted_to_pad'):
            _C.scatter_sorted_to_pad_bf16(dYe_sorted.data_ptr(), dYe_pad.data_ptr(), int(M_recv), int(H), stream)

        dXe_pad, dW1, dW3, dW2 = torch.autograd.grad(
            outputs=Ye_pad,
            inputs=(Xe_pad, W1, W3, W2),
            grad_outputs=dYe_pad,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
        dXe_pad_bf16 = dXe_pad.to(dtype=torch.bfloat16)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dXe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
            _C.gather_from_pad_bf16(dXe_pad_bf16.data_ptr(), dXe_sorted.data_ptr(), int(M_recv), int(H), stream)
            _C.scatter_dx_dist_bf16(
                dXe_sorted.data_ptr(),
                row_id.data_ptr(),
                dX.data_ptr(),
                int(M_recv), int(T), int(H), int(K),
                stream,
            )
        else:
            _C.scatter_dx_bf16_internal(
                dXe_pad_bf16.data_ptr(),
                row_id.data_ptr(),
                dX.data_ptr(),
                int(M_recv), int(T), int(H), int(K),
                stream,
            )

        dGates = dGates_tk_f32.to(dtype=torch.bfloat16)
        return None, dX, None, dGates, dW1, dW3, dW2


class _MoEBlockscaledFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor, W_cache) -> torch.Tensor:
        device = x.device
        stream = torch.cuda.current_stream(device)

        x = x.contiguous().bfloat16()
        eid = eid.contiguous().int()
        gates = gates.contiguous().bfloat16()
        gates_fp32 = gates.detach().float()

        T, H = x.shape
        K = int(eid.shape[1])
        E = int(rdep.n_local)

        # Option A: Use BF16 dispatch + local quantization
        # This ensures Xe_pad (BF16) is available for backward STE
        offs_pad = torch.empty(E, device=device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
        align = 128  # Required for blockscaled SF swizzle

        with _maybe_nvtx('rdep/dispatch_meta'), _maybe_cuda_time('time_ms/rdep_dispatch_meta'):
            M_recv = _C.dispatch_meta_bf16(
                x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
                int(T), int(K), align,
                offs_pad.data_ptr(), M_host.data_ptr(),
                stream,
            )

        out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
        if M_recv <= 0:
            ctx.rdep = rdep
            ctx.W_cache = W_cache
            ctx.T = int(T)
            ctx.H = int(H)
            ctx.K = int(K)
            ctx.save_for_backward(x, eid, gates, W1, W3, W2)
            return out_f32.to(dtype=torch.bfloat16)

        # Compute max_pad and extend last expert's padded region
        max_pad = (int(M_recv) + E * (align - 1) + (align - 1)) // align * align
        offs_pad[-1] = int(max_pad)

        # Gather BF16 activations
        Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_xe'), _maybe_cuda_time('time_ms/rdep_gather_xe'):
            _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

        # Quantize locally: BF16 -> FP8/NVFP4
        pack_factor = 2 if rdep.profile == 'fp8' else 4
        Hp = H // pack_factor
        sf_k = H // 32
        sf_k_pad = ((sf_k + 3) // 4) * 4
        M_e_stride = ((rdep.capacity + 127) // 128) * 128  # 128-aligned capacity per expert

        Xe_q = torch.empty(int(max_pad), Hp, device=device, dtype=torch.uint16)
        Xe_sf = torch.empty(E, M_e_stride, sf_k_pad, device=device, dtype=torch.uint8)

        # Quant kernels expect offs_with0 [E+1] with leading 0: [0, offs_pad[0], offs_pad[1], ...]
        offs_with0 = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offs_pad])

        with _maybe_nvtx('rdep/quant_act'), _maybe_cuda_time('time_ms/quant_act'):
            if rdep.profile == 'fp8':
                _C.quant_fp8_sf_strided_mma(
                    Xe_pad.data_ptr(), int(H),
                    Xe_q.data_ptr(), Hp,
                    Xe_sf.data_ptr(),
                    offs_with0.data_ptr(),
                    E, M_e_stride,
                    int(max_pad), int(H),
                    stream,
                )
            else:  # nvfp4
                _C.quant_nvfp4_sf_strided_mma(
                    Xe_pad.data_ptr(), int(H),
                    Xe_q.data_ptr(), Hp,
                    Xe_sf.data_ptr(),
                    offs_with0.data_ptr(),
                    E, M_e_stride,
                    int(max_pad), int(H),
                    stream,
                )

        # Expert compute (blockscaled)
        from nmoe.blockscaled.ggemm import expert as expert_blockscaled
        with _maybe_nvtx('rdep/expert_blockscaled'), _maybe_cuda_time('time_ms/expert_blockscaled'):
            Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad)

        # Gather sorted and return scatter
        Ye_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_from_pad'), _maybe_cuda_time('time_ms/rdep_gather_from_pad'):
            _C.gather_from_pad_bf16(Ye_pad.data_ptr(), Ye_sorted.data_ptr(), int(M_recv), int(H), stream)

        with _maybe_nvtx('rdep/return_scatter'), _maybe_cuda_time('time_ms/rdep_return_scatter'):
            _C.return_scatter(
                Ye_sorted.data_ptr(),
                out_f32.data_ptr(),
                int(M_recv), int(T), int(K),
                stream,
            )

        ctx.rdep = rdep
        ctx.W_cache = W_cache
        ctx.T = int(T)
        ctx.H = int(H)
        ctx.K = int(K)
        ctx.save_for_backward(x, eid, gates, W1, W3, W2)
        return out_f32.to(dtype=torch.bfloat16)

    @staticmethod
    def backward(ctx, dOut: torch.Tensor):
        x, eid, gates, W1, W3, W2 = ctx.saved_tensors
        rdep: Rdep = ctx.rdep
        W_cache = ctx.W_cache

        device = dOut.device
        stream = torch.cuda.current_stream(device)

        dOut = dOut.contiguous().bfloat16()
        x = x.contiguous().bfloat16()
        eid = eid.contiguous().int()
        gates = gates.contiguous().bfloat16()
        gates_fp32 = gates.detach().float()

        T = int(ctx.T)
        H = int(ctx.H)
        K = int(ctx.K)
        E = int(rdep.n_local)

        # Option A: Use BF16 dispatch to get correct Xe_pad from all ranks
        # This fixes the distributed bug where local x was used for remote rows
        offs_pad = torch.empty(E, device=device, dtype=torch.int32)
        M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
        align = 128  # Required for blockscaled SF swizzle

        with _maybe_nvtx('rdep/dispatch_meta'), _maybe_cuda_time('time_ms/rdep_dispatch_meta'):
            M_recv = _C.dispatch_meta_bf16(
                x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
                int(T), int(K), align,
                offs_pad.data_ptr(), M_host.data_ptr(),
                stream,
            )

        if M_recv <= 0:
            dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
            dGates = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)
            dW1 = torch.zeros_like(W1)
            dW3 = torch.zeros_like(W3)
            dW2 = torch.zeros_like(W2)
            return None, dX, None, dGates, dW1, dW3, dW2, None

        # Compute max_pad and extend last expert's padded region
        max_pad = (int(M_recv) + E * (align - 1) + (align - 1)) // align * align
        offs_pad[-1] = int(max_pad)

        # Gather BF16 activations (correct from all source ranks via IPC buffer!)
        Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_xe'), _maybe_cuda_time('time_ms/rdep_gather_xe'):
            _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

        # Get row_id and gate_sorted for dGate computation
        row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
        gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
        _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

        # Quantize and run expert forward for Ye recomputation (needed for dGate)
        pack_factor = 2 if rdep.profile == 'fp8' else 4
        Hp = H // pack_factor
        sf_k = H // 32
        sf_k_pad = ((sf_k + 3) // 4) * 4
        M_e_stride = ((rdep.capacity + 127) // 128) * 128

        Xe_q = torch.empty(int(max_pad), Hp, device=device, dtype=torch.uint16)
        Xe_sf = torch.empty(E, M_e_stride, sf_k_pad, device=device, dtype=torch.uint8)

        # Quant kernels expect offs_with0 [E+1] with leading 0: [0, offs_pad[0], offs_pad[1], ...]
        offs_with0 = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offs_pad])

        with _maybe_nvtx('rdep/quant_act'), _maybe_cuda_time('time_ms/quant_act'):
            if rdep.profile == 'fp8':
                _C.quant_fp8_sf_strided_mma(
                    Xe_pad.data_ptr(), int(H),
                    Xe_q.data_ptr(), Hp,
                    Xe_sf.data_ptr(),
                    offs_with0.data_ptr(),
                    E, M_e_stride,
                    int(max_pad), int(H),
                    stream,
                )
            else:  # nvfp4
                _C.quant_nvfp4_sf_strided_mma(
                    Xe_pad.data_ptr(), int(H),
                    Xe_q.data_ptr(), Hp,
                    Xe_sf.data_ptr(),
                    offs_with0.data_ptr(),
                    E, M_e_stride,
                    int(max_pad), int(H),
                    stream,
                )

        from nmoe.blockscaled.ggemm import expert as expert_blockscaled
        with _maybe_nvtx('rdep/expert_blockscaled'), _maybe_cuda_time('time_ms/expert_blockscaled'):
            Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad)

        # Gather sorted Ye for dGate
        Ye_sorted_unscaled = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
        with _maybe_nvtx('rdep/gather_from_pad'), _maybe_cuda_time('time_ms/rdep_gather_from_pad'):
            _C.gather_from_pad_bf16(Ye_pad.data_ptr(), Ye_sorted_unscaled.data_ptr(), int(M_recv), int(H), stream)

        # TODO(perf): The gather_dy kernels still compute dGate internally (dot product of Ye*dOut).
        # This is wasted compute (~negligible). To fully remove, modify CUDA kernels in rdep.cu.
        dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
        dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)  # unused, but kernel writes to it

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            _C.gather_dy_dist_bf16(
                dOut.data_ptr(),
                eid.data_ptr(),
                Ye_sorted_unscaled.data_ptr(),  # Ye only used for dGate which we discard
                row_id.data_ptr(),
                gate_sorted.data_ptr(),
                dYe_sorted.data_ptr(),
                dGate_sorted.data_ptr(),
                torch.empty(0, device=device).data_ptr(),  # dGates_tk unused
                int(M_recv), int(T), int(H), int(K),
                stream,
            )
        else:
            _C.gather_dy_bf16(
                dOut.data_ptr(),
                Ye_sorted_unscaled.data_ptr(),  # Ye only used for dGate which we discard
                row_id.data_ptr(),
                gate_sorted.data_ptr(),
                dYe_sorted.data_ptr(),
                dGate_sorted.data_ptr(),  # computed but discarded
                int(M_recv), int(T), int(H), int(K),
                stream,
            )

        dYe_pad = torch.zeros(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
        _C.scatter_sorted_to_pad_bf16(
            dYe_sorted.data_ptr(),
            dYe_pad.data_ptr(),
            int(M_recv), int(H),
            stream,
        )

        offs_pinned = torch.empty(E, dtype=torch.int32, device='cpu', pin_memory=True)
        offs_pinned.copy_(offs_pad, non_blocking=True)
        copy_event = torch.cuda.Event()
        copy_event.record(stream)
        Dff = int(W2.size(1))
        H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
        H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
        dA = torch._grouped_mm(dYe_pad, W2.transpose(1, 2), offs=offs_pad)
        A = torch.empty_like(H1)
        dH1 = torch.empty_like(H1)
        dH3 = torch.empty_like(H3)
        _C.swiglu_bwd_bf16(
            H1.data_ptr(), int(Dff),
            H3.data_ptr(), int(Dff),
            dA.data_ptr(), int(Dff),
            A.data_ptr(), int(Dff),
            dH1.data_ptr(), int(Dff),
            dH3.data_ptr(), int(Dff),
            int(max_pad), int(Dff),
            stream,
        )

        copy_event.synchronize()
        offs_host = offs_pinned
        dW2 = torch.empty_like(W2)
        _C.bf16_wgrad_w2_cublaslt(
            A.data_ptr(),
            dYe_pad.data_ptr(),
            dW2.data_ptr(),
            offs_host.data_ptr(),
            int(E), int(H), int(Dff),
            stream,
        )

        dW1 = torch.empty_like(W1)
        _C.bf16_wgrad_w13_cublaslt(
            Xe_pad.data_ptr(),
            dH1.data_ptr(),
            dW1.data_ptr(),
            offs_host.data_ptr(),
            int(E), int(H), int(Dff),
            stream,
        )

        dW3 = torch.empty_like(W3)
        _C.bf16_wgrad_w13_cublaslt(
            Xe_pad.data_ptr(),
            dH3.data_ptr(),
            dW3.data_ptr(),
            offs_host.data_ptr(),
            int(E), int(H), int(Dff),
            stream,
        )

        dX_pad = torch._grouped_mm(dH1, W1.transpose(1, 2), offs=offs_pad)
        dX_pad.add_(torch._grouped_mm(dH3, W3.transpose(1, 2), offs=offs_pad))
        dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dX_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
            _C.gather_from_pad_bf16(dX_pad.data_ptr(), dX_sorted.data_ptr(), int(M_recv), int(H), stream)
            _C.scatter_dx_dist_bf16(
                dX_sorted.data_ptr(),
                row_id.data_ptr(),
                dX.data_ptr(),
                int(M_recv), int(T), int(H), int(K),
                stream,
            )
        else:
            _C.scatter_dx_bf16_internal(
                dX_pad.data_ptr(),
                row_id.data_ptr(),
                dX.data_ptr(),
                int(M_recv), int(T), int(H), int(K),
                stream,
            )

        dGates = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)
        return None, dX, None, dGates, dW1, dW3, dW2, None


class _DispatchBf16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates_fp32: torch.Tensor):
        device = x.device
        stream = torch.cuda.current_stream(device)
        T, H = x.shape
        K = eid.shape[1]

        align = 128  # Match blockscaled for consistent padding
        max_pad = int(rdep.capacity + rdep.n_local * (align - 1))
        Xe_pad = torch.zeros(max_pad, H, device=device, dtype=torch.bfloat16)
        offs_pad = torch.zeros(rdep.n_local, device=device, dtype=torch.int32)
        dest = torch.zeros(rdep.capacity, device=device, dtype=torch.int32)
        row_id = torch.zeros(rdep.capacity, device=device, dtype=torch.int64)
        gate_sorted = torch.zeros(rdep.capacity, device=device, dtype=torch.float32)
        M_pad_tensor = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

        M_recv = _C.dispatch(
            x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
            int(T), int(K),
            Xe_pad.data_ptr(),
            offs_pad.data_ptr(),
            dest.data_ptr(),
            row_id.data_ptr(),
            gate_sorted.data_ptr(),
            M_pad_tensor.data_ptr(),
            stream,
        )
        stream.synchronize()
        M_pad = int(M_pad_tensor.item())

        dest = dest[:M_recv]
        row_id = row_id[:M_recv]
        gate_sorted = gate_sorted[:M_recv]

        ctx.T = int(T)
        ctx.H = int(H)
        ctx.K = int(K)
        ctx.M = int(M_recv)
        ctx.save_for_backward(dest, row_id)

        return Xe_pad[:M_pad], offs_pad, dest, row_id, gate_sorted

    @staticmethod
    def backward(ctx, dXe_pad: torch.Tensor, d_offs, d_dest, d_row_id, d_gate_sorted):
        dest, row_id = ctx.saved_tensors
        device = dXe_pad.device
        stream = torch.cuda.current_stream(device)

        dX = torch.zeros(ctx.T, ctx.H, device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dXe_sorted = dXe_pad.index_select(0, dest)
            _C.scatter_dx_dist_bf16(
                dXe_sorted.contiguous().bfloat16().data_ptr(),
                row_id.contiguous().to(torch.int64).data_ptr(),
                dX.data_ptr(),
                ctx.M, ctx.T, ctx.H, ctx.K,
                stream,
            )
        else:
            _C.scatter_dx_bf16(
                dXe_pad.contiguous().bfloat16().data_ptr(),
                dest.contiguous().int().data_ptr(),
                row_id.contiguous().to(torch.int64).data_ptr(),
                dX.data_ptr(),
                ctx.M, ctx.T, ctx.H, ctx.K,
                stream,
            )
        return None, dX, None, None


class _ReturnScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rdep: Rdep, Ye: torch.Tensor, row_id: torch.Tensor, gate_sorted: torch.Tensor,
                gates_tk: torch.Tensor, T: int, K: int):
        device = Ye.device
        stream = torch.cuda.current_stream(device)
        out_f32 = torch.zeros(int(T), Ye.shape[1], device=device, dtype=torch.float32)

        if rdep.profile == 'bf16' or rdep._mode == 'hybrid':
            _C.return_scatter(Ye.data_ptr(), out_f32.data_ptr(), int(row_id.numel()), int(T), int(K), stream)
        else:
            _C.return_scatter_blockscaled(Ye.data_ptr(), out_f32.data_ptr(), int(row_id.numel()), int(T), int(K), stream)

        ctx.T = int(T)
        ctx.K = int(K)
        ctx.M = int(row_id.numel())
        ctx.H = int(Ye.shape[1])
        ctx.rdep = rdep
        ctx.save_for_backward(Ye, row_id, gate_sorted)
        return out_f32.to(dtype=torch.bfloat16)

    @staticmethod
    def backward(ctx, dOut: torch.Tensor):
        Ye, row_id, gate_sorted = ctx.saved_tensors
        rdep: Rdep = ctx.rdep
        device = dOut.device
        stream = torch.cuda.current_stream(device)

        dYe = torch.empty(ctx.M, ctx.H, device=device, dtype=torch.bfloat16)
        dGate_sorted = torch.empty(ctx.M, device=device, dtype=torch.float32)
        dGates_tk = torch.zeros(ctx.T, ctx.K, device=device, dtype=torch.float32)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            if _C.get_mode() != 1 or rdep.profile != 'bf16':
                raise RuntimeError(
                    "_ReturnScatter.backward in distributed runs currently supports BF16 IPC only. "
                    "Use the fused MoE paths (Rdep.moe_bf16 / Rdep.moe_blockscaled) for training."
                )
            _C.gather_dy_ipc_bf16(
                dOut.contiguous().bfloat16().data_ptr(),
                Ye.contiguous().bfloat16().data_ptr(),
                row_id.contiguous().to(torch.int64).data_ptr(),
                gate_sorted.contiguous().float().data_ptr(),
                dYe.data_ptr(),
                dGate_sorted.data_ptr(),
                dGates_tk.data_ptr(),
                ctx.M, ctx.T, ctx.H, ctx.K,
                stream,
            )
        else:
            _C.gather_dy_bf16(
                dOut.contiguous().bfloat16().data_ptr(),
                Ye.contiguous().bfloat16().data_ptr(),
                row_id.contiguous().to(torch.int64).data_ptr(),
                gate_sorted.contiguous().float().data_ptr(),
                dYe.data_ptr(),
                dGate_sorted.data_ptr(),
                ctx.M, ctx.T, ctx.H, ctx.K,
                stream,
            )
            _C.scatter_gate_bf16(
                dGate_sorted.data_ptr(),
                row_id.contiguous().to(torch.int64).data_ptr(),
                dGates_tk.data_ptr(),
                ctx.M, ctx.T, ctx.K,
                stream,
            )
        return None, dYe, None, None, dGates_tk.to(dtype=torch.bfloat16), None, None
