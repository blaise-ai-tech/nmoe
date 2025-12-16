Single-GPU MoE Flow Summary

  Data Structures

  Token Layout:     [T, H]     - T tokens, H hidden dim
  Expert Layout:    [M, H]     - M = T*K routed tokens (sorted by expert)
  Padded Layout:    [M_pad, H] - M_pad = sum of per-expert aligned counts
  Expert Weights:   [E, H, Dff] for W1/W3, [E, Dff, H] for W2

  Key Mappings

  | Name        | Shape     | Purpose                                     |
  |-------------|-----------|---------------------------------------------|
  | row_id      | [M] int64 | Encoded (src_rank * T + tok) * K + slot     |
  | dest        | [M] int32 | Maps sorted index → padded index            |
  | offs_pad    | [E] int32 | Cumulative padded boundaries (no leading 0) |
  | gate_sorted | [M] f32   | Gate weights in sorted order                |

  BF16 Forward Flow

  x[T,H] ──┬──▶ dispatch_meta_bf16 ──▶ sorting, offs_pad, M_recv
           │
           └──▶ gather_xe_bf16 ──▶ Xe_pad[M_pad,H]
                                        │
                                        ▼
                                expert_bf16(Xe_pad, W1, W3, W2, offs_pad)
                                        │
                                        ▼
                                Ye_pad[M_pad,H]
                                        │
                                gather_from_pad ──▶ Ye_sorted[M,H]
                                        │
                                return_scatter(Ye_sorted, row_id, gate)
                                        │
                                        ▼
                                out[T,H] (gated, accumulated)

  BF16 Backward Flow (Recomputation Style)

  dOut[T,H] ─────────────────────────────────────────────────────┐
                                                                 │
  x,eid,gates ──▶ RE-DISPATCH (get sorting info) ──▶ row_id, gate│
           │                                                     │
           └──▶ RE-GATHER Xe_pad[M_pad,H]                        │
                        │                                        │
                        ▼                                        ▼
                expert_bf16 WITH requires_grad=True    gather_dy_bf16
                        │                                    │
                        ▼                                    ▼
                Ye_pad[M_pad,H] ◀─────────────── dYe_sorted[M,H], dGate
                        │                               │
                torch.autograd.grad ◀── scatter_to_pad ─┘
                        │
                        ▼
                dXe_pad, dW1, dW3, dW2
                        │
                scatter_dx_bf16 ──▶ dX[T,H]

  Blockscaled Forward Flow

  x[T,H] ──▶ dispatch (FUSED quant) ──▶ Xe_q[M_pad,Hp], Xe_sf[E,M_e,sf_k]
                                                │
                                                ▼
                                        expert_blockscaled(Xe_q, Xe_sf, W_cache, offs)
                                                │
                                                ▼
                                        Ye_pad[M_pad,H] (BF16)
                                                │
                                        index_select(dest) ──▶ Ye_sorted
                                                │
                                        return_scatter_blockscaled
                                                │
                                                ▼
                                        out[T,H]

  Blockscaled Backward (STE Style)

  Key Insight: Forward uses quantized compute, backward uses BF16 "as if" compute.

  dOut[T,H] ───────────────────────────────────────────────────────┐
                                                                   │
  x,eid,gates ──▶ RE-DISPATCH ──▶ h (Xe_q, Xe_sf, row_id, gate)    │
                                          │                        │
                                          ▼                        ▼
                                  expert_blockscaled        gather_dy_bf16
                                          │                        │
                                          ▼                        ▼
                                  Ye_pad, Ye_sorted         dYe_sorted, dGate
                                                                   │
           ┌───────────────────────────────────────────────────────┘
           │
           ▼
  X_sorted = x.index_select(0, tok)  ◀── THE BUG: uses local x!
           │
           ▼
  X_pad = scatter_sorted_to_pad(X_sorted)
           │
           ▼
  EXPLICIT BF16 BACKWARD (no autograd):
    H1 = torch._grouped_mm(X_pad, W1, offs)
    H3 = torch._grouped_mm(X_pad, W3, offs)
    dA = torch._grouped_mm(dYe_pad, W2.T, offs)
    swiglu_bwd_bf16(H1, H3, dA, ...) ──▶ A, dH1, dH3
    dW2 = wgrad_w2(A, dYe_pad, offs)        ◀── needs offs_host.cpu()!
    dW1 = wgrad_w13(X_pad, dH1, offs)
    dW3 = wgrad_w13(X_pad, dH3, offs)
    dX_pad = dH1 @ W1.T + dH3 @ W3.T
           │
           ▼
  scatter_dx ──▶ dX[T,H]

  ---
