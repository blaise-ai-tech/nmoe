"""FlashAttention w/support for learned sinks and banded attention.

This is an expanded version of the Flash Attention v2 implementation (see https://tridao.me/publications/flash2/flash2.pdf)
which can be found at https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.

This version has been extended to support banded attention and learned attention sinks.
"""

import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# pytest is only needed for local unit tests; guard at runtime so
# training images don't require it.
try:  # pragma: no cover - optional test dependency
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None  # type: ignore



@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    M,
    Out,  #
    Start_q,
    Z,
    H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    if BANDWIDTH:
        lo_bound = start_q + start_m * BLOCK_M - BANDWIDTH
        lo, hi = tl.maximum(0, lo_bound), start_q + (start_m + 1) * BLOCK_M
    else:
        # Global attention: consider all keys up to the current query position
        lo, hi = 0, start_q + (start_m + 1) * BLOCK_M

    # Early exit if no keys are valid for this M-block (saves work for strict causal/windows)
    if hi <= lo:
        # write zeros to Out and -inf to M for valid rows
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N_Q_CTX
        zero = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        Out.store([off_z, off_h, start_m * BLOCK_M, 0], zero.to(Out.dtype)[None, None, :, :])
        neg_inf = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        m_ptrs = M + off_hz * N_Q_CTX + offs_m
        tl.store(m_ptrs, neg_inf, mask=m_mask)
        return

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T
        qk = tl.dot(q, k, allow_tf32=False)

        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc, allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    sink = tl.math.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]
    m_i += tl.math.log(l_i)
    m_ptrs = M + off_hz * N_Q_CTX + offs_m
    tl.store(m_ptrs, m_i)
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        assert len(start_q) == 1
        bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM_Q = q.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K = k.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_V = v.shape
        n_heads = n_kv_heads * repeat_kv
        q = q.view(bs, n_ctx, n_heads, HEAD_DIM_Q)
        k = k.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        q = q.transpose(1, 2).contiguous()
        k = k.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v = v.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        BLOCK_M = 64
        BLOCK_N = 64
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        # pad q to multiple of its block size in the n_ctx dimension (-2)
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))
        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        # pad k and v to multiple of their block size in the n_kv_ctx dimension
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        o = torch.empty_like(q)
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)
        _attn_fwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, HEAD_DIM_K]),
            sinks,
            sm_scale,
            M,
            TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM_K]),
            start_q,
            q.shape[0],
            q.shape[1],
            N_Q_CTX=n_ctx + m_pad_size,
            N_KV_CTX=n_kv_ctx,
            HEAD_DIM=HEAD_DIM_K,
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=3,
        )

        # Save tensors and metadata needed for backward
        ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
        ctx.sm_scale = sm_scale
        ctx.bandwidth = bandwidth
        # Save head grouping info to reconstruct original shapes/grads
        ctx.n_kv_heads = n_kv_heads
        ctx.repeat_kv = repeat_kv
        ctx.head_dim = HEAD_DIM_K

        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()
        o = o.view(bs, n_ctx, n_heads * HEAD_DIM_V)
        return o

    @staticmethod
    def backward(ctx, grad_out):
        """Triton backward kernel (FlashAttention-style) with sliding window and sinks.

        Computes grads for q, k, v, sinks. Uses a two-pass tile sweep to obtain
        row-wise softmax statistics (m_i, l_i) and then accumulate dV, s, dQ, dK.
        """
        q_saved, k_saved, v_saved, sinks, o_saved, M_saved, start_q = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        bandwidth = ctx.bandwidth if ctx.bandwidth is not None else 0
        n_kv_heads = ctx.n_kv_heads
        repeat_kv = ctx.repeat_kv

        B, H, M_pad, D = q_saved.shape
        N_pad = k_saved.shape[2]
        M = grad_out.shape[1]
        # We don't have the original N_KV_CTX; assume K/V are only padded at tail
        # and use their logical length from saved forward inputs (N_pad may be > N_KV_CTX).
        N_kv = k_saved.shape[2]

        # Reshape grad_out to [B, H, M, D]
        dO = grad_out.view(B, M, H, D).transpose(1, 2).contiguous()

        # Allocate grads: use fp32 for DK/DV to allow atomic adds accurately.
        dQ = torch.zeros_like(q_saved)
        dK = torch.zeros((B, H, N_pad, D), device=q_saved.device, dtype=torch.float32)
        dV = torch.zeros((B, H, N_pad, D), device=q_saved.device, dtype=torch.float32)
        dSinks = torch.zeros((H,), device=q_saved.device, dtype=torch.float32)

        # Strides (in elements)
        def strides(x):
            s = x.stride()
            return tuple(int(si) for si in s)

        qsz = strides(q_saved)
        ksz = strides(k_saved)
        vsz = strides(v_saved)
        dosz = strides(dO)
        dqsz = strides(dQ)
        dksz = strides(dK)
        dvsz = strides(dV)

        BLOCK_M = 64
        BLOCK_N = 64

        # Heuristic: use column-block backward for larger sequence lengths
        # Column-block backward reduces DK/DV atomics and wins on global attention.
        # For sliding-window (BANDWIDTH>0), row-block often performs better due to fewer DQ atomics.
        use_col = (bandwidth == 0 and D <= 64)

        if use_col:
            # Allocate float32 accumulator for dQ
            dQf32 = torch.zeros_like(q_saved, dtype=torch.float32)
            grid = (triton.cdiv(N_kv, BLOCK_N), B * H, 1)
            _attn_bwd_col[grid](
                q_saved, k_saved, v_saved, dO,
                dQf32, dK, dV, sinks, dSinks, M_saved,
                start_q,
                sm_scale,
                B, H,
                M, N_kv,
                *qsz, *ksz, *vsz, *dosz, *dqsz, *dksz, *dvsz,
                *M_saved.stride(),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BANDWIDTH=bandwidth, HEAD_DIM=D,
                num_warps=8,
                num_stages=2,
            )
            dQ = dQf32.to(dQ.dtype)
        else:
            grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
            row_stages = 2 if D >= 128 else 3
            _attn_bwd[grid](
                q_saved, k_saved, v_saved, dO,
                dQ, dK, dV, sinks, dSinks,
                start_q,
                sm_scale,
                B, H,
                M, N_kv,
                *qsz, *ksz, *vsz, *dosz, *dqsz, *dksz, *dvsz,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BANDWIDTH=bandwidth, HEAD_DIM=D,
                num_warps=8,
                num_stages=row_stages,
            )

        # Cast DK/DV to input dtype and slice valid lengths
        dQ = dQ[:, :, :M, :]
        dK = dK[:, :, :N_kv, :].to(k_saved.dtype)
        dV = dV[:, :, :N_kv, :].to(v_saved.dtype)

        # Map head-expanded grads back to original input shapes
        dQ_out = dQ.transpose(1, 2).contiguous().view(B, M, n_kv_heads, repeat_kv, D)
        dK_grouped = dK.view(B, n_kv_heads, repeat_kv, N_kv, D).sum(dim=2)
        dV_grouped = dV.view(B, n_kv_heads, repeat_kv, N_kv, D).sum(dim=2)
        dK_out = dK_grouped.transpose(1, 2).contiguous()
        dV_out = dV_grouped.transpose(1, 2).contiguous()

        # Sink grads to original dtype
        dSink = dSinks.to(sinks.dtype)

        return dQ_out, dK_out, dV_out, dSink, None, None, None


@triton.jit
def _attn_bwd(
    Q_ptr, K_ptr, V_ptr, DO_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    SINKS_ptr, dSINKS_ptr,
    Start_q,
    sm_scale,
    Z, H,
    N_Q_CTX, N_KV_CTX,
    # Strides (elements)
    Q_sZ, Q_sH, Q_sM, Q_sD,
    K_sZ, K_sH, K_sN, K_sD,
    V_sZ, V_sH, V_sN, V_sD,
    DO_sZ, DO_sH, DO_sM, DO_sD,
    DQ_sZ, DQ_sH, DQ_sM, DQ_sD,
    DK_sZ, DK_sH, DK_sN, DK_sD,
    DV_sZ, DV_sH, DV_sN, DV_sD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BANDWIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Valid masks for bounds
    m_mask = offs_m < N_Q_CTX

    # Load Q and dO tiles: [BLOCK_M, D]
    q_ptrs = Q_ptr + off_z * Q_sZ + off_h * Q_sH + offs_m[:, None] * Q_sM + offs_d[None, :] * Q_sD
    do_ptrs = DO_ptr + off_z * DO_sZ + off_h * DO_sH + offs_m[:, None] * DO_sM + offs_d[None, :] * DO_sD
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0).to(tl.float32)
    dO = tl.load(do_ptrs, mask=m_mask[:, None], other=0).to(tl.float32)

    # Sinks
    sink = tl.load(SINKS_ptr + off_h).to(tl.float32)

    # Pass 1: compute row-wise m_i and l_i (like forward)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    # Initialize with sink to include it in the max
    m_i = tl.maximum(m_i, sink)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    if BANDWIDTH:
        lo_bound = start_q + start_m * BLOCK_M - BANDWIDTH
        lo = tl.maximum(0, lo_bound)
        hi = start_q + (start_m + 1) * BLOCK_M
    else:
        # Global attention: include all prior keys up to query position
        lo = 0
        hi = start_q + (start_m + 1) * BLOCK_M

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < N_KV_CTX

        # mask causal and window constraints
        causal = n_idx[None, :] > (start_q + offs_m)[:, None]
        if BANDWIDTH:
            too_old = n_idx[None, :] < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask_mn = causal | too_old
        else:
            mask_mn = causal
        mask_mn = mask_mn | (~n_mask[None, :]) | (~m_mask[:, None])

        k_ptrs = K_ptr + off_z * K_sZ + off_h * K_sH + n_idx[:, None] * K_sN + offs_d[None, :] * K_sD
        k_nt = tl.load(k_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)  # [BLOCK_N, D]
        k = k_nt.T  # [D, BLOCK_N]

        qk = tl.dot(q, k, allow_tf32=False)
        qk = qk * sm_scale + tl.where(mask_mn, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Sink term for normalization
    sink_exp = tl.math.exp(sink - m_i)
    z_row = l_i + sink_exp

    # Pass 2: accumulate dV and s = sum(dP * P_norm) over all keys
    s_row = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < N_KV_CTX
        causal = n_idx[None, :] > (start_q + offs_m)[:, None]
        if BANDWIDTH:
            too_old = n_idx[None, :] < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask_mn = causal | too_old
        else:
            mask_mn = causal
        mask_mn = mask_mn | (~n_mask[None, :]) | (~m_mask[:, None])

        k_ptrs = K_ptr + off_z * K_sZ + off_h * K_sH + n_idx[:, None] * K_sN + offs_d[None, :] * K_sD
        v_ptrs = V_ptr + off_z * V_sZ + off_h * V_sH + n_idx[:, None] * V_sN + offs_d[None, :] * V_sD
        k_nt = tl.load(k_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)  # [BLOCK_N, D]
        k = k_nt.T
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)

        qk = tl.dot(q, k, allow_tf32=False)
        qk = qk * sm_scale + tl.where(mask_mn, -1.0e6, 0.0)
        p = tl.math.exp(qk - m_i[:, None])
        p = p / z_row[:, None]

        # dP = dO @ V^T
        dP = tl.dot(dO, v.T, allow_tf32=False)
        s_row += tl.sum(dP * p, 1)

        # dV += P^T @ dO
        dV_tile = tl.dot(p.T, dO, allow_tf32=False)
        # Atomic add to global DV
        dv_ptrs = DV_ptr + off_z * DV_sZ + off_h * DV_sH + n_idx[:, None] * DV_sN + offs_d[None, :] * DV_sD
        tl.atomic_add(dv_ptrs, dV_tile, mask=n_mask[:, None])

    # dSink: -p_sink * s
    p_sink = sink_exp / z_row
    dSink_block = -p_sink * s_row
    tl.atomic_add(dSINKS_ptr + off_h, tl.sum(dSink_block, 0))

    # Pass 3: accumulate dQ and dK
    dQ_tile = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < N_KV_CTX
        causal = n_idx[None, :] > (start_q + offs_m)[:, None]
        if BANDWIDTH:
            too_old = n_idx[None, :] < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask_mn = causal | too_old
        else:
            mask_mn = causal
        mask_mn = mask_mn | (~n_mask[None, :]) | (~m_mask[:, None])

        k_ptrs = K_ptr + off_z * K_sZ + off_h * K_sH + n_idx[:, None] * K_sN + offs_d[None, :] * K_sD
        v_ptrs = V_ptr + off_z * V_sZ + off_h * V_sH + n_idx[:, None] * V_sN + offs_d[None, :] * V_sD
        k_nt = tl.load(k_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)  # [BLOCK_N, D]
        k = k_nt.T
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)

        qk = tl.dot(q, k, allow_tf32=False)
        qk = qk * sm_scale + tl.where(mask_mn, -1.0e6, 0.0)
        p = tl.math.exp(qk - m_i[:, None])
        p = p / z_row[:, None]

        # dP and dlogits
        dP = tl.dot(dO, v.T, allow_tf32=False)
        dlogits = (dP - s_row[:, None]) * p
        dlogits = dlogits * sm_scale

        # dQ += dlogits @ K (use non-transposed k: [BLOCK_N, D])
        dQ_tile += tl.dot(dlogits, k_nt, allow_tf32=False)

        # dK += dlogits^T @ Q (atomic)
        dK_tile = tl.dot(dlogits.T, q, allow_tf32=False)
        dk_ptrs = DK_ptr + off_z * DK_sZ + off_h * DK_sH + n_idx[:, None] * DK_sN + offs_d[None, :] * DK_sD
        tl.atomic_add(dk_ptrs, dK_tile, mask=n_mask[:, None])

    # Store dQ
    dq_ptrs = DQ_ptr + off_z * DQ_sZ + off_h * DQ_sH + offs_m[:, None] * DQ_sM + offs_d[None, :] * DQ_sD
    tl.store(dq_ptrs, dQ_tile.to(tl.bfloat16), mask=m_mask[:, None])


@triton.jit
def _attn_bwd_col(
    Q_ptr, K_ptr, V_ptr, DO_ptr,
    DQf32_ptr, DK_ptr, DV_ptr,
    SINKS_ptr, dSINKS_ptr, LSE_ptr,
    Start_q,
    sm_scale,
    Z, H,
    N_Q_CTX, N_KV_CTX,
    # Strides
    Q_sZ, Q_sH, Q_sM, Q_sD,
    K_sZ, K_sH, K_sN, K_sD,
    V_sZ, V_sH, V_sN, V_sD,
    DO_sZ, DO_sH, DO_sM, DO_sD,
    DQ_sZ, DQ_sH, DQ_sM, DQ_sD,
    DK_sZ, DK_sH, DK_sN, DK_sD,
    DV_sZ, DV_sH, DV_sN, DV_sD,
    L_sZ, L_sH, L_sM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BANDWIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Column tile indices
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n < N_KV_CTX

    # Load K and V for this column block
    k_ptrs = K_ptr + off_z * K_sZ + off_h * K_sH + offs_n[:, None] * K_sN + offs_d[None, :] * K_sD
    v_ptrs = V_ptr + off_z * V_sZ + off_h * V_sH + offs_n[:, None] * V_sN + offs_d[None, :] * V_sD
    k_nt = tl.load(k_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)  # [BLOCK_N, D]
    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0).to(tl.float32)     # [BLOCK_N, D]
    kT = k_nt.T  # [D, BLOCK_N]

    # Accumulators for this column
    dk_tile = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_tile = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dSink_acc = tl.zeros((), dtype=tl.float32)

    # Loop over row blocks
    num_m_blocks = tl.cdiv(N_Q_CTX, BLOCK_M)
    for start_m in range(0, num_m_blocks):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N_Q_CTX

        # Row pointers
        q_ptrs = Q_ptr + off_z * Q_sZ + off_h * Q_sH + offs_m[:, None] * Q_sM + offs_d[None, :] * Q_sD
        do_ptrs = DO_ptr + off_z * DO_sZ + off_h * DO_sH + offs_m[:, None] * DO_sM + offs_d[None, :] * DO_sD
        l_ptrs = LSE_ptr + off_z * L_sZ + off_h * L_sH + offs_m * L_sM

        # Load Q, dO, LSE
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0).to(tl.float32)   # [M,D]
        dO = tl.load(do_ptrs, mask=m_mask[:, None], other=0).to(tl.float32) # [M,D]
        lse = tl.load(l_ptrs, mask=m_mask).to(tl.float32)                   # [M]

        # Build masks
        causal = offs_n[None, :] > (start_q + offs_m)[:, None]
        if BANDWIDTH:
            too_old = offs_n[None, :] < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask_mn = causal | too_old
        else:
            mask_mn = causal
        mask_mn = mask_mn | (~n_mask[None, :]) | (~m_mask[:, None])

        # Compute softmax probabilities for this tile using saved LSE
        qk = tl.dot(q, kT, allow_tf32=False) * sm_scale
        p = tl.math.exp(qk - lse[:, None])
        p = tl.where(mask_mn, 0.0, p)

        # dP, s_tile
        dP = tl.dot(dO, v.T, allow_tf32=False)
        s_tile = tl.sum(dP * p, 1)

        # Update dV
        dv_tile += tl.dot(p.T, dO, allow_tf32=False)

        # dscores and dK
        dscores = (dP - s_tile[:, None]) * p * sm_scale
        dk_tile += tl.dot(dscores.T, q, allow_tf32=False)

        # dQ atomic add
        dq_add = tl.dot(dscores, k_nt, allow_tf32=False)  # [M,D]
        dq_ptrs = DQf32_ptr + off_z * DQ_sZ + off_h * DQ_sH + offs_m[:, None] * DQ_sM + offs_d[None, :] * DQ_sD
        tl.atomic_add(dq_ptrs, dq_add, mask=m_mask[:, None])

        # dSink: -exp(sink - lse) * s_tile
        sink = tl.load(SINKS_ptr + off_h).to(tl.float32)
        p_sink_row = tl.math.exp(sink - lse)
        dSink_acc += tl.sum(-p_sink_row * s_tile, 0)

    # Write DK/DV tiles
    kv_mask = n_mask[:, None]
    dk_out_ptrs = DK_ptr + off_z * DK_sZ + off_h * DK_sH + offs_n[:, None] * DK_sN + offs_d[None, :] * DK_sD
    dv_out_ptrs = DV_ptr + off_z * DV_sZ + off_h * DV_sH + offs_n[:, None] * DV_sN + offs_d[None, :] * DV_sD
    tl.store(dk_out_ptrs, dk_tile.to(tl.bfloat16), mask=kv_mask)
    tl.store(dv_out_ptrs, dv_tile.to(tl.bfloat16), mask=kv_mask)

    # Accumulate dSink
    tl.atomic_add(dSINKS_ptr + off_h, dSink_acc)

attention = _attention.apply


def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
):
    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups * head_dim).bfloat16()
    return output


if pytest is not None:  # pragma: no cover - tests are optional at runtime
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_queries", [1, 128])
    @pytest.mark.parametrize("num_keys", [128, 32])
    @pytest.mark.parametrize("num_key_value_heads", [8])
    @pytest.mark.parametrize("num_key_value_groups", [8])
    @pytest.mark.parametrize("head_dim", [64])
    @pytest.mark.parametrize("sm_scale", [0.125])
    @pytest.mark.parametrize("sliding_window", [None, 128])
    @pytest.mark.parametrize("start_q", [0, 5])
    def test_eq(batch_size, num_queries, num_keys, num_key_value_heads, num_key_value_groups, head_dim, sm_scale, sliding_window, start_q):
        if num_queries > num_keys:
            pytest.skip("too many queries")

        q = torch.randn(batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim).bfloat16().cuda()
        k = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
        v = torch.randn(batch_size, num_keys, num_key_value_heads, head_dim).bfloat16().cuda()
        sinks = torch.randn(num_key_value_heads * num_key_value_groups).bfloat16().cuda()

        start_q = torch.tensor([start_q], dtype=torch.int32).cuda()

        o1 = attention(q, k, v, sinks, sm_scale, sliding_window, start_q)
        o2 = attention_ref(q, k, v, sinks, sm_scale, sliding_window, start_q)

        torch.testing.assert_close(o1, o2)
