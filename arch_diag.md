   ## Executive Summary

   The nmoe codebase implements a distributed Mixture-of-Experts (MoE) training framework optimized for B200 GPUs (SM100). The architecture is split into three major
   components: Python orchestration (rdep.py, model.py), Python compute libraries (blockscaled/grouped.py, ggemm.py), and CUDA kernels (csrc/). The system supports three
    execution modes: single-GPU, intra-node IPC, and multi-node hybrid (IPC + NVSHMEM).

   **Key insight**: The codebase is HIGHLY SPECIALIZED for B200 with SM100-specific kernels, making it a state-of-the-art but narrowly-focused implementation.

   ---

   ## 1. Entry Points: train.py → model.py → rdep.py

   ### train.py (183 LoC) - Orchestration
   **Role**: High-level training loop
   ```
   main()
     ↓
   train(cfg)
     ├─ runtime.init()  [distributed setup]
     ├─ build_loader()  [data pipeline]
     ├─ Transformer(cfg).cuda()  [model creation]
     │   └─ creates Rdep instance if n_layers > n_dense_layers
     ├─ build_optimizer()  [optimizer setup]
     ├─ Training loop:
     │   ├─ loader.next()  [get batch]
     │   ├─ model(inputs)  [FORWARD: calls MoE layers]
     │   ├─ loss.backward()  [BACKWARD: calls MoE autograd]
     │   └─ step()  [optimizer step]
   ```

   **Observations**:
   - Very clean, linear training flow
   - MoE complexity is entirely encapsulated in model.py and rdep.py
   - No direct rdep calls in train.py

   ### model.py (250 LoC) - Model Definition
   **Key classes**:
   1. **Transformer** (lines 169-250)
      - Creates single `Rdep` instance at init time (if has_moe=True)
      - Distributes same Rdep instance to all MoE layers
      - Passes through to each TransformerBlock

   2. **TransformerBlock** (lines 139-166)
      - Decides at init: dense MLP vs MoE based on layer_id
      - MoE layers call rdep.moe_bf16() or rdep.moe_blockscaled()

   3. **MoE** (lines 81-136) - THE KEY INTEGRATION POINT
      ```python
      def forward(self, x):
          X = x.view(-1, x.size(-1))
          g, eid = self.router(X)  # Route: [T,K] gates, [T,K] expert IDs

          if self._use_blockscaled:
              out = self._rdep.moe_blockscaled(X, eid, g, W1, W3, W2, W_cache)
          else:
              out = self._rdep.moe_bf16(X, eid, g, W1, W3, W2)
          return out.view_as(x)
      ```
      - Stores expert weights W1, W3, W2 as Parameters
      - Calls rdep.moe_* which returns autograd-enabled output

   **Data flow**:
   ```
   inputs [batch, seq, dim]
     ↓ (embedding)
   x [batch*seq, dim]
     ↓ (routing)
   g [batch*seq, K]  (gate weights)
   eid [batch*seq, K]  (expert indices)
     ↓ (rdep.moe_*)
   out [batch*seq, dim]
     ↓ (view, loss, backward)
   ```

   ---

   ## 2. MoE Data Flow: Dispatch → Compute → Return

   ### rdep.py (1047 LoC) - Core MoE Infrastructure

   **High-level structure**:
   ```
   Rdep.__init__()
     ├─ _setup_single()    [MODE_SINGLE: world=1]
     ├─ _setup_ipc()       [MODE_IPC: intra-node]
     └─ _setup_hybrid()    [MODE_HYBRID: multi-node]

   moe_bf16()    → _MoEBf16Fused.apply()
   moe_blockscaled() → _MoEBlockscaledFused.apply()
   ```

   ### Token Flow (BF16 Path): _MoEBf16Fused.forward()

   ```
   x [T, H]              {T tokens, H hidden}
   eid [T, K]            {expert indices}
   gates [T, K]          {gate weights}

     ↓ _MoEBf16Fused.forward()

   [Step 1] dispatch_meta_bf16()  [_C kernel]
     ├─ Input:  x, eid, gates, T, K, align=128
     ├─ Output: offs_pad [E] {expert offsets}
     │         M_recv {actual token count}
     └─ Action: GPU-side routing via atomics
                Sorts tokens by expert, pads to 128

   [Step 2] gather_xe_bf16()  [_C kernel]
     ├─ Input:  M_recv, max_pad, stream
     ├─ Output: Xe_pad [max_pad, H]
     └─ Action: Gather padded BF16 activations from shared buffer

   [Step 3] expert_bf16()  [torch._grouped_mm]
     ├─ Input:  Xe_pad, W1, W3, W2, offs_pad
     ├─ Formula: Y = (SiLU(X@W1) * (X@W3)) @ W2
     └─ Action: Per-expert MLP via grouped GEMM
                Uses torch._grouped_mm (PyTorch internal)

   [Step 4] gather_from_pad_bf16()  [_C kernel]
     ├─ Input:  Ye_pad [max_pad, H]
     ├─ Output: Ye_sorted [M_recv, H]
     └─ Action: Unpad expert outputs

   [Step 5] return_scatter()  [_C kernel]
     ├─ Input:  Ye_sorted, row_id, gates, T, K
     ├─ Output: out [T, H]
     └─ Action: Scatter sorted outputs back to original token positions
                Weighted by gate scores
   ```

   ### Token Flow (Blockscaled Path): _MoEBlockscaledFused.forward()

   ```
   Same as BF16 up to Step 2, then:

   [Step 2.5] Quantization
     ├─ quant_fp8_sf_strided_mma() OR quant_nvfp4_sf_strided_mma()
     ├─ Input:  Xe_pad [max_pad, H] BF16
     ├─ Output: Xe_q [max_pad, H/2 or H/4] uint16 (packed)
     │          Xe_sf [E, M_e_stride, sf_k_pad] uint8 (MMA layout)
     └─ Action: BF16 → FP8 or NVFP4 quantization
                Scale factors in CUTLASS MMA layout

   [Step 3] expert_blockscaled()  [CuTeDSL kernel from grouped.py]
     ├─ Input:  Xe_q, Xe_sf, W_cache (quantized weights)
     ├─ Output: Ye_pad [max_pad, H]
     └─ Action: Blockscaled grouped GEMM
                Vendored CuTeDSL kernel (SM100-only)

   [Steps 4-5] Same as BF16
   ```

   ### Key Data Structures

   **DispatchHandle** (BF16 path):
   ```python
   Xe: [M_pad, H]         # Padded BF16 activations
   offs: [n_local]        # Cumulative expert offsets
   dest: [M]              # Destination indices for scattering back
   row_id: [M]            # Encoded (rank, token, slot)
   gate: [M]              # Gate weights
   M: int                 # Actual received count
   M_pad: int             # Padded count
   ```

   **DispatchBlockscaledHandle** (blockscaled path):
   ```python
   Xe_q: [M_pad, H_packed]       # Packed quantized activations
   Xe_sf: [E, M_e_swizzle, ...]  # Scale factors (MMA layout)
   offs: [n_local]               # Same as BF16
   dest: [M]
   row_id: [M]
   gate: [M]
   M: int
   M_pad: int
   ```

   ---

   ## 3. rdep.py Analysis: ~1000 LoC Structure

   ### Major Sections

   **Lines 1-95**: Imports, dataclasses, module-level utilities
   - `_get_local_world_size()`: Read LOCAL_WORLD_SIZE env
   - `_cpu_pg()`: CPU-only process group for bootstrap collectives
   - `_maybe_cuda_time()`, `_maybe_nvtx()`: Profiling hooks (circular import guards)

   **Lines 95-383**: Rdep class - Main dispatcher
   ```python
   class Rdep:
       __init__()          # lines 99-132: init modes, alloc buffers
       _setup_hybrid()     # lines 136-174: NVSHMEM bootstrap
       _setup_ipc()        # lines 176-195: NCCL IPC handle exchange
       dispatch()          # lines 197-338: Route and pad tokens
       return_scatter()    # lines 340-354: Scatter expert outputs
       moe_bf16()          # lines 356-360: Entry point for BF16
       moe_blockscaled()   # lines 362-367: Entry point for blockscaled
       get_expert_loads()  # lines 369-382: Load balancing
   ```

   **Lines 385-448**: _MoEBf16Fused.forward()
   - dispatch_meta_bf16() to get offsets
   - gather_xe_bf16() to pull activations
   - expert_bf16() for grouped GEMM
   - gather_from_pad_bf16() to unpad
   - return_scatter() to reassemble

   **Lines 450-579**: _MoEBf16Fused.backward()
   - Mirrors forward path
   - Scatters gradients back: scatter_dx_bf16() or scatter_dx_dist_bf16()

   **Lines 581-906**: _MoEBlockscaledFused.forward() & backward()
   - Similar to BF16 but with quantization step
   - Calls expert_blockscaled() from blockscaled/grouped.py

   **Lines 909-977**: _DispatchBf16 (lower-level autograd)
   - Used only in manual dispatch mode (not in fused path)
   - Returns sorted handle

   **Lines 980-1047**: _ReturnScatter (lower-level autograd)
   - Return path for manual dispatch mode

   ### Key Insight: Two Code Paths

   **Path 1: Fused (_MoEBf16Fused)**
   - Single autograd.Function wraps all 5 steps
   - Complete forward + backward in one place
   - Called from model.py via rdep.moe_bf16()

   **Path 2: Manual (dispatch → compute → return)**
   - _DispatchBf16 + compute + _ReturnScatter
   - Lower-level, for experimentation
   - NOT used in normal training

   ### Critical Dependencies in rdep.py

   **External imports**:
   ```python
   from .csrc import rdep as _C     # Line 12: C extension (dispatch/return/scatter)
   from .ggemm import expert as expert_bf16  # Line 13: BF16 grouped GEMM
   ```

   **Internal imports** (NO circular imports observed):
   - `torch.distributed`: For IPC handle exchange
   - `numpy`: For IPC handle arrays
   - `torch.cuda.nvtx`: Optional profiling
   - `nmoe.metrics.cuda_time`: Lazy-loaded in _maybe_cuda_time()

   ### Boundary Between Communication and Compute

   **Communication Layer** (lines 136-195, inside _C):
   - `_setup_hybrid()`: NVSHMEM UID broadcast, IPC handle exchange
   - `dispatch_meta_bf16()`: GPU-side routing + atomics
   - `return_scatter()`: Gather scattered outputs

   **Compute Layer** (lines 666-668, blockscaled path):
   ```python
   from nmoe.blockscaled.ggemm import expert as expert_blockscaled
   Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad)
   ```

   **Boundary observation**:
   - Dispatch and return are in _C (CUDA kernels)
   - Expert compute calls blockscaled/grouped.py (CuTeDSL)
   - Clean separation at the function boundary

   ---

   ## 4. Three Execution Modes

   ### MODE_SINGLE (world=1)
   ```
   _setup_single():
     ├─ _C.alloc_bf16(capacity, dim, n_local)
     ├─ _C.sync_buffer_ptrs_bf16()  [dummy, local only]
     └─ For blockscaled: _C.alloc_blockscaled() + sync

   Dispatch: All tokens routed locally
   Return: Direct GPU memory, no sync needed
   ```

   ### MODE_IPC (world=local_world)
   ```
   _setup_ipc():
     ├─ _C.alloc_bf16(capacity, dim, n_local)
     ├─ _C.get_ipc_handle_bf16()  [get cudaIpcMemHandle_t]
     ├─ dist.all_gather(handles via NCCL)  [one-time bootstrap]
     ├─ _C.open_ipc_handles_bf16(all_handles)  [map remote buffers]
     └─ _C.sync_buffer_ptrs_bf16()  [broadcast pointers]

   Dispatch: GPU-side P2P via IPC handles
   Return: Same, via IPC
   Hot path: Zero NCCL (uses GPU atomics for sync)
   ```

   ### MODE_HYBRID (world > local_world)
   ```
   _setup_hybrid():
     ├─ _C.nvshmem_get_uid()  [rank 0 only]
     ├─ dist.broadcast_object_list(uid, src=0)  [via CPU Gloo]
     ├─ _C.nvshmem_init(uid, rank, world, local_world)
     │   └─ NVSHMEM establishes symmetric heaps
     ├─ _C.nvshmem_alloc_bf16()  [allocate from NVSHMEM]
     ├─ Exchange intra-node IPC handles via NCCL all_gather
     ├─ _C.nvshmem_open_ipc_handles_bf16()  [map intra-node)
     └─ dist.barrier() to sync

   Dispatch:
     ├─ Intra-node: IPC P2P
     ├─ Inter-node: NVSHMEM puts (GPU-side, no host involvement)
     └─ Atomics for sync (GPU-side)

   Return: Same (IPC intra-node, NVSHMEM inter-node)
   Hot path: Zero NCCL
   ```

   ---

   ## 5. blockscaled/grouped.py Analysis (~2400 LoC)

   ### Purpose
   Provides **SM100-specific grouped blockscaled GEMM** for expert computation.

   ### Architecture

   **Top-level API** (lines 2295-2399):
   ```python
   def expert_blockscaled(Xe_q_pad, Xe_sf_pad, W_cache, offs_pad):
       """Calls run_grouped_blockscaled_strided()"""
       return run_grouped_blockscaled_strided(...)
   ```

   **Key classes**:
   1. **NmoeGroupedScaledGemmKernel** (lines 254-1804)
      - CuTeDSL kernel definition
      - Params: GEMM dimensions, tensor layouts, MMA configs
      - Compiles to GPU kernel at first call
      - CUTLASS MMA SM100-specific

   2. **_StridedCompileKey** (lines 1801-1819)
      - Cache key for compiled kernels
      - Keyed by: problem size, data types, scheduling

   3. **_ExpertScratch** (lines 1821-1830)
      - Workspace cache for grouped GEMM metadata

   4. **QuantizedWeightsFused** (lines 2134-2152)
      - Container for quantized expert weights
      - Stores: W1_q, W1_sf, W3_q, W3_sf, W2_q, W2_sf

   ### Quantization Pipeline

   **quantize_weights()** (lines 2154-2293):
   ```
   W1, W3, W2 [E, H, Dff] BF16
     ├─ For each weight:
     │   ├─ quantize_fp8() or quantize_nvfp4()
     │   │   └─ Packs BF16 → FP8/NVFP4
     │   ├─ Calls _C.swizzle_sf_mkl_to_mma()
     │   │   └─ Swizzles scale factors to CUTLASS layout
     │   └─ Returns W_q, W_sf (quantized + MMA-layout SF)
     └─ Returns QuantizedWeightsFused
   ```

   **Observation**: Quantization is **offline** (post-optimizer step), cached in model.py

   ### Invocation in rdep.py

   **Blockscaled forward** (rdep.py lines 666-668):
   ```python
   from nmoe.blockscaled.ggemm import expert as expert_blockscaled
   Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad)
   ```

   **Blockscaled backward** (rdep.py lines 787-840):
   - Gradients propagate through expert_blockscaled() via autograd
   - Quantization is **non-differentiable** (stopped with .detach())
   - Gradients flow back to Xe_pad (BF16 before quantization)

   ---

   ## 6. ggemm.py and grouped.py Interaction

   ### ggemm.py (91 LoC)
   **Role**: BF16 grouped GEMM wrapper

   ```python
   def expert(Xe_pad, W1, W3, W2, offs_pad):
       """BF16 MLP: Y = (SiLU(X@W1) * (X@W3)) @ W2

       Uses torch._grouped_mm (PyTorch internal API)
       """
       H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
       H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
       return torch._grouped_mm(F.silu(H1).mul_(H3), W2, offs=offs_pad)
   ```

   **Critical observation**: Relies on **torch._grouped_mm**, which is:
   - PyTorch internal API (not guaranteed stable)
   - Highly optimized for NVIDIA GPUs
   - Used by DeepSeek for MoE

   ### blockscaled/grouped.py (2400 LoC)
   **Role**: Blockscaled (FP8/NVFP4) grouped GEMM

   **Key difference from ggemm.py**:
   - Handles quantized activations + scale factors
   - Manages per-expert scale factor layouts (MMA swizzle)
   - Uses vendored CuTeDSL kernel (no PyTorch API dependency)
   - SM100-specific (requires B200 GPU)

   ### Dependency: No Circular Imports

   **ggemm.py** → rdep.py:
   - Line 13 in rdep.py: `from .ggemm import expert as expert_bf16`

   **blockscaled/grouped.py** → rdep.py:
   - Line 666 in rdep.py: `from nmoe.blockscaled.ggemm import expert as expert_blockscaled`
   - Called only inside _MoEBlockscaledFused.forward()

   **blockscaled/ggemm.py** (wrapper):
   - Line 12: `from nmoe.blockscaled.grouped import QuantizedWeightsFused, expert_blockscaled`

   **No cycle**: rdep imports ggemm/blockscaled, but ggemm/blockscaled don't import rdep

   ---

   ## 7. csrc/ Structure (~15,200 LoC CUDA)

   ### File Organization

   ```
   csrc/
   ├─ rdep.cu               (154 KB)  Main dispatch/return kernels
   ├─ rdep_nvshmem.cu       (107 KB)  NVSHMEM + bootstrap
   ├─ rdep_nvshmem.cuh      (13 KB)   NVSHMEM headers
   ├─ quant.cu              (101 KB)  Quantization (FP8/NVFP4)
   ├─ ptx.cu                (23 KB)   PTX primitives (memory ordering)
   ├─ adamw.cu              (20 KB)   Optimizer kernels
   ├─ lt_gemm.cu            (15 KB)   Low-precision GEMM
   ├─ grouped_gemm.cu       (22 KB)   Grouped GEMM (old, unused?)
   ├─ bindings.cpp          (79 KB)   Pybind11 bindings
   ├─ gpu.cpp               (7 KB)    GPU telemetry (NVML)
   ├─ Makefile              (8 KB)    Build system
   ├─ rdep/                 Subdirectory (DeepEP-based)
   │  ├─ rdep.cu            (47 KB)   DeepEP dispatch/combine
   │  ├─ rdep.cuh           (1 KB)    Headers
   │  ├─ configs.cuh        (2 KB)    Configuration
   │  ├─ contract.cuh       (10 KB)   Verify invariants
   │  ├─ utils.cuh          (23 KB)   Utilities
   │  ├─ exception.cuh      (1 KB)    Error handling
   │  └─ ibgda_device.cuh   (21 KB)   Inter-GPU atomic ops
   └─ bindings_*.cpp (generated)
   ```

   ### rdep.cu vs rdep/rdep.cu

   **rdep.cu** (top-level, 154 KB):
   - NCCL bootstrap for MODE_IPC and MODE_HYBRID
   - Wraps rdep/rdep.cu functions
   - Python entry points (`rdep_init`, `rdep_dispatch`, etc.)
   - Falls back to C extension API

   **rdep/rdep.cu** (47 KB):
   - **DeepEP-based** dispatch/combine primitives
   - Actual per-expert atomics, P2P writes
   - Low-level GPU-side routing

   ### quant.cu (101 KB)

   **Functions**:
   - `quant_fp8_sf_strided_mma()`: BF16 → FP8 with MMA swizzle
   - `quant_nvfp4_sf_strided_mma()`: BF16 → NVFP4 with MMA swizzle
   - Per-expert quantization with atomic updates

   ### ptx.cu (23 KB)

   **PTX primitives** (GPU-side memory ordering):
   - `f32_to_e4m3_byte()`: Float → FP8 E4M3
   - `f32x4_to_e2m1x4_packed()`: Float x4 → NVFP4 (packed)
   - `e8m0_encode_from_pos_f32()`: Float → E8M0 scale factor
   - IPC memory ordering (atomicAdd_system, atomicSub_system)

   ### Bindings (pybind11, bindings.cpp + 79 KB)

   **Python API exposed**:
   ```python
   _C.init(rank, world, local_world)
   _C.get_mode() → {0:single, 1:ipc, 2:hybrid}
   _C.has_nvshmem() → bool
   _C.alloc_bf16(capacity, dim, n_local)
   _C.alloc_blockscaled(capacity, dim, n_local, profile)
   _C.dispatch_meta_bf16(...) → M_recv
   _C.gather_xe_bf16(...)
   _C.gather_from_pad_bf16(...)
   _C.return_scatter(...)
   _C.dispatch_blockscaled(...)
   _C.quant_fp8_sf_strided_mma(...)
   _C.quant_nvfp4_sf_strided_mma(...)
   _C.swizzle_sf_mkl_to_mma(...)
   ... (and many more)
   ```

   ### Key Observation: Tight CUDA Integration

   **rdep.cu includes ptx.cu directly** (line 52 in rdep.cu):
   ```cpp
   #include "ptx.cu"
   using namespace nmoe::ptx;
   ```

   This is **code organization via inclusion**, not separate compilation. Unusual but valid pattern for header-only PTX utilities.

   ---

   ## 8. Dependency Graph

   ```
   train.py
     ├─ model.py
     │  ├─ Rdep (created once, shared across layers)
     │  ├─ MoE.forward() calls:
     │  │  ├─ rdep.moe_bf16() or rdep.moe_blockscaled()
     │  │  └─ blockscaled.ggemm.quantize_weights()
     │  └─ ggemm.expert() [BF16 path]
     │
     ├─ rdep.py
     │  ├─ _C (csrc/rdep module)
     │  │  ├─ rdep.cu [dispatch/return via IPC/NVSHMEM]
     │  │  ├─ rdep/rdep.cu [DeepEP primitives]
     │  │  ├─ quant.cu [quantization]
     │  │  ├─ ptx.cu [PTX primitives]
     │  │  └─ rdep_nvshmem.cu [NVSHMEM bootstrap]
     │  │
     │  ├─ ggemm.expert() [torch._grouped_mm]
     │  │
     │  ├─ blockscaled.ggemm.expert()
     │  │  └─ blockscaled.grouped.run_grouped_blockscaled_strided()
     │  │     └─ NmoeGroupedScaledGemmKernel [CuTeDSL]
     │  │
     │  └─ torch.distributed [bootstrap, IPC handle exchange]
     │
     └─ blockscaled/
        ├─ ggemm.py [quantize_weights, expert]
        │  └─ grouped.py [QuantizedWeightsFused, expert_blockscaled]
        │     └─ CuTeDSL (nvidia-cutlass-dsl >= 4.3.1)
        │
        └─ (blockscaled doesn't import rdep)

   quant.py
     └─ _C (same csrc/rdep module)

   Circular dependencies: NONE DETECTED
   ```

   ---

   ## 9. Architecture Assessment

   ### Clean Separations ✓

   1. **Training loop** (train.py) cleanly isolated from MoE logic
   2. **Model definition** (model.py) only knows about rdep.moe_*() interface
   3. **Dispatch/compute/return** well-separated in rdep.py
   4. **Communication** (GPU-side P2P) separate from **compute** (expert MLP)
   5. **BF16 path** vs **blockscaled path** are independent code paths
   6. **Bootstrap** (one-time) vs **hot path** (zero NCCL) clearly demarcated

   ### Tangled Dependencies ⚠️ 

   1. **rdep.py** has TOO MANY RESPONSIBILITIES:
      - Dispatch logic (routes tokens)
      - Multiple autograd.Function classes (_MoEBf16Fused, _MoEBlockscaledFused, _DispatchBf16, _ReturnScatter)
      - Distribution mode detection and setup
      - Profiling hooks and lazy imports (circular import guard)
      - ~1047 LoC in single file

   2. **blockscaled/grouped.py** is MASSIVE (2399 LoC):
      - CuTeDSL kernel definition (lines 254-1804)
      - Compilation cache logic
      - Quantization pipeline
      - High coupling to CUTLASS internals

   3. **torch._grouped_mm** dependency (ggemm.py):
      - PyTorch internal API, not documented
      - Could break on future PyTorch versions
      - No fallback or abstraction layer

   4. **Inverse includes** (csrc/rdep.cu includes ptx.cu):
      - Unusual compilation pattern (code-via-inclusion)
      - Makes refactoring harder

   ### Code Placement Issues ⚠️ 

   1. **quant.py** is thin wrapper around _C:
      - Only 4 public functions (quantize_fp8, quantize_nvfp4, + helpers)
      - Could be merged into blockscaled/grouped.py
      - Current separation is artificial

   2. **ggemm.py** is simple thin wrapper:
      - Only 2 functions (expert, expert_mlp_bf16)
      - Could be moved into rdep.py or blockscaled/ggemm.py
      - Current location (nmoe root) is odd compared to blockscaled/ analog

   3. **Two grouped GEMM implementations**:
      - csrc/grouped_gemm.cu (22 KB) - appears unused
      - blockscaled/grouped.py + CuTeDSL - used for production
      - Duplication suggests incomplete refactoring

   4. **rdep/rdep.cu** location:
      - Nested under csrc/rdep/ (good)
      - But only rdep.cu includes it, not called separately
      - DeepEP-inspired but modified for nmoe

   ### Duplication

   1. **Quantization logic**:
      - quant.py (Python wrapper)
      - quant.cu (kernel)
      - blockscaled/grouped.py (also calls quant kernels)
      - Creates 3-layer abstraction

   2. **Dispatch/return paths**:
      - Fused path (_MoEBf16Fused)
      - Manual path (_DispatchBf16 + _ReturnScatter)
      - Both in rdep.py, only one used
      - Dead code?

   3. **GEMM implementations**:
      - grouped_gemm.cu (old?)
      - blockscaled/grouped.py (new)
      - Suggests incomplete migration

   ---

   ## 10. Hidden Complexity and Gotchas

   ### 1. Global State in rdep.py
   ```python
   _CPU_PG = None          # Lazy-init process group
   _CPU_PG_WORLD = None
   _CUDA_TIME = None       # Lazy-init profiling
   _NVTX = None
   ```
   - Used for profiling and distributed setup
   - Circular import guards to avoid model.py → metrics → model.py
   - Module-level state is fragile

   ### 2. Profiling Hooks Everywhere
   - `_maybe_cuda_time('tag')`: Returns nullcontext or CUDA timer
   - `_maybe_nvtx('tag')`: Returns nullcontext or NVTX range
   - Lazy imports to break circular dependency with metrics.py
   - ~20+ profiling points in rdep.py

   ### 3. Buffer Management Complexity
   ```python
   # MODE_SINGLE: Local only
   # MODE_IPC: Alloc + exchange handles
   # MODE_HYBRID: NVSHMEM + IPC handles
   ```
   - 3 different initialization paths
   - _setup_ipc() and _setup_hybrid() use different approaches
   - Easy to miss edge case in one mode

   ### 4. Row ID Encoding
   ```c
   int64_t encode_rid(int rank, int tok, int slot, int T, int K) {
       return (static_cast<int64_t>(rank) * T + tok) * K + slot;
   }
   ```
   - Encodes (rank, token_index, expert_slot) into single int64
   - Used for backward pass to route gradients
   - Requires knowledge of T, K at decode time
   - Magic numbers hardcoded in multiple places

   ### 5. Scale Factor Swizzle
   ```python
   # blockscaled/grouped.py lines 50-75
   def _swizzle_sf_to_mma(sf_mkl):
       """Swizzle scale factors from MKL row-major to MMA layout."""
       # Pad, call rdep.swizzle_sf_mkl_to_mma(), return
   ```
   - Scale factors MUST be in CUTLASS MMA layout
   - "IMPORTANT: Returns the full padded tensor to preserve swizzle pattern"
   - One-way transformation, non-obvious intent
   - If accidentally called twice or sliced incorrectly, SILENT CORRECTNESS BUG

   ### 6. Padding Asymmetry
   ```python
   # rdep.py line 237-238
   max_pad = int(self.capacity + self.n_local * (align - 1))
   sf_k_pad = ((Hsf + 3) // 4) * 4
   ```
   - Different padding rules for different tensors
   - max_pad for activations, sf_k_pad for scale factors
   - Both aligned to 128 or 4, but computed differently
   - Easy to get shapes wrong

   ---

   ## 11. Summary: Strengths and Weaknesses

   ### Strengths
   ✓ **High performance**: SM100-specific kernels, optimized for B200
   ✓ **Clean separation of concerns**: Train → Model → Rdep → CUDA
   ✓ **Multiple modes**: Single, IPC, hybrid (multi-node)
   ✓ **Comprehensive testing**: test_rdep_forward.py, test_rdep_backward.py, etc.
   ✓ **Profiling infrastructure**: cuda_time, NVTX support

   ### Weaknesses
   ⚠️  **Monolithic rdep.py** (1047 LoC): Too many responsibilities
   ⚠️  **blockscaled/grouped.py** (2399 LoC): Massive, hard to understand
   ⚠️  **torch._grouped_mm dependency**: Undocumented PyTorch internal API
   ⚠️  **Duplication**: Multiple GEMM implementations, quantization wrappers
   ⚠️  **Global state + lazy imports**: Fragile, hard to reason about
   ⚠️  **SM100-only**: No fallback for other GPU types (Hopper, etc.)

   ---

   ## 12. Recommendations for Refactoring

   ### If considering moving to attic/:

   **Phase 1: Extract core abstractions**
   1. Split rdep.py into:
      - `dispatch.py`: Dispatch logic (_MoEBf16Fused, _MoEBlockscaledFused)
      - `modes.py`: Single/IPC/hybrid setup
      - `buffers.py`: Memory management
      - `primitives.py`: Lower-level _DispatchBf16, _ReturnScatter

   2. Consolidate quantization:
      - Move quant.py logic into blockscaled/
      - Or vice versa

   3. Create GEMM abstraction:
      - Remove torch._grouped_mm dependency
      - Wrap in custom grouped_gemm.py with fallback

   **Phase 2: Decouple CUDA**
   1. Keep csrc/ as-is (already well-organized)
   2. Python layer should call csrc/ via clear API
   3. Bindings are in good shape (pybind11 → Python dict)

   **Phase 3: Test and validate**
   1. Existing tests (test_rdep_forward.py, etc.) should continue passing
   2. Add tests for each mode (single, IPC, hybrid)
   3. Benchmark before/after refactoring

   ---

   ## File Inventory

   **Python** (3879 LoC):
   ```
   train.py                     183 LoC    Entry point
   model.py                     250 LoC    Transformer + MoE
   rdep.py                    1047 LoC    Dispatch + autograd
   ggemm.py                     91 LoC    BF16 GEMM wrapper
   quant.py                    ~100 LoC   Quantization wrapper
   blockscaled/
     ├─ ggemm.py               31 LoC    Blockscaled GEMM wrapper
     └─ grouped.py           2399 LoC    CuTeDSL kernel + quant pipeline
   ```

   **CUDA** (15,210 LoC):
   ```
   csrc/
     ├─ rdep.cu              154 KB    Dispatch/return (main)
     ├─ rdep/rdep.cu          47 KB    DeepEP primitives
     ├─ rdep_nvshmem.cu      107 KB    NVSHMEM + bootstrap
     ├─ quant.cu             101 KB    Quantization kernels
     ├─ ptx.cu                23 KB    PTX primitives
     ├─ adamw.cu              20 KB    Optimizer
     ├─ lt_gemm.cu            15 KB    Low-precision GEMM
     ├─ grouped_gemm.cu       22 KB    Grouped GEMM (old?)
     ├─ bindings.cpp          79 KB    Pybind11
     └─ Makefile               8 KB    Build

• Minimum Viable Perfect Core (the “train.py → model.py” path)

  Python: required for MoE training

  - nmoe/train.py:37 — calls Transformer(cfg) and runs fwd/bwd/step.
  - nmoe/model.py:169 — constructs a single Rdep and passes it into MoE blocks.
  - nmoe/model.py:81 — MoE.forward() is the only integration point that must remain stable (router → rdep.moe_* → reshape).
  - nmoe/model.py:49 — Router.forward() (produces g, eid).
  - nmoe/model.py:111 — MoE.refresh_weight_cache() (blockscaled only; uses quantize_weights).
  - nmoe/rdep.py:110 — Rdep init + mode bootstrap (single/ipc/hybrid).
  - nmoe/rdep.py:385 — _MoEBf16Fused (production BF16 MoE path).
  - nmoe/rdep.py:581 — _MoEBlockscaledFused (production FP8/NVFP4 MoE path).
  - nmoe/ggemm.py:1 — expert() (BF16 grouped MLP; safe to inline into moe.py if you want).
  - nmoe/blockscaled/grouped.py:2154 — quantize_weights() (creates W_cache).
  - nmoe/blockscaled/grouped.py:2295 — expert_blockscaled() (runs the blockscaled expert MLP).
  - nmoe/blockscaled/ggemm.py:1 — thin wrapper (can stay or be folded).

  CUDA extension surface that the production path actually calls

  - Bootstrap / mode setup (from nmoe/rdep.py:110):
      - _C.init, _C.get_mode, _C.has_nvshmem
      - Single/IPC: _C.alloc_bf16, _C.get_ipc_handle_bf16, _C.open_ipc_handles_bf16, _C.sync_buffer_ptrs_bf16
      - Blockscaled buffers (if profile fp8/nvfp4): _C.alloc_blockscaled, _C.get_ipc_handle_blockscaled, _C.open_ipc_handles_blockscaled, _C.sync_buffer_ptrs_blockscaled
      - Hybrid: _C.nvshmem_get_uid_size, _C.nvshmem_get_uid, _C.nvshmem_init(...), _C.nvshmem_alloc_bf16 / _C.nvshmem_alloc_blockscaled, plus the nvshmem_*ipc_handle/open/
        sync_* trio.
  - Hot path (BF16 fused) (nmoe/rdep.py:385):
      - _C.dispatch_meta_bf16, _C.gather_xe_bf16, _C.gather_from_pad_bf16, _C.return_scatter
      - Backward: _C.gather_meta_sorted_bf16, _C.gather_dy_bf16 or _C.gather_dy_dist_bf16, _C.scatter_gate_bf16, _C.scatter_sorted_to_pad_bf16, _C.scatter_dx_bf16_internal
        or _C.scatter_dx_dist_bf16
  - Hot path (blockscaled fused) (nmoe/rdep.py:581):
      - Same dispatch/gather/return as BF16
      - Activation quant: _C.quant_fp8_sf_strided_mma / _C.quant_nvfp4_sf_strided_mma
      - Weight-grad helpers used in the custom backward: _C.swiglu_bwd_bf16, _C.bf16_wgrad_w2_cublaslt, _C.bf16_wgrad_w13_cublaslt
  - Blockscaled expert compute plumbing (nmoe/blockscaled/grouped.py:2154 and nmoe/blockscaled/grouped.py:2295):
      - _C.quant_fp8_sf_strided_mma / _C.quant_nvfp4_sf_strided_mma (weights + activations)
      - _C.swiglu_quant_fp8_sf_strided_mma / _C.swiglu_quant_nvfp4_sf_strided_mma
      - _C.build_grouped_gemm_metadata, _C.swizzle_sf_mkl_to_mma
  - Optimizer coupling (if you use ExpertAdamW):
      - nmoe/opt.py:134 calls _rdep_ext.expert_adamw_step(...) (so that symbol is part of the “core” if blockscaled training uses ExpertAdamW).

  Clear “not on the production path” candidates (i.e., attic if you want one clear path)

  - Manual MoE API + autograd wrappers in nmoe/rdep.py:197 (dispatch()), nmoe/rdep.py:909 (_DispatchBf16), nmoe/rdep.py:980 (_ReturnScatter) — not used by MoE.forward; used
    by tests (nmoe/test_rdep_forward.py, nmoe/test_rdep_backward.py).
  - nmoe/opt.py:17 imports _muon_ext but never uses it (pure dead code today).
  - nmoe/blockscaled/grouped.py:16 imports quantize_fp8/quantize_nvfp4 from nmoe.quant but never references them (import-only).
  - The “v2” DeepEP prototype is currently a build-cost/mental-cost liability: nmoe/csrc/rdep/rdep.cu exports rdep_v2_*, but nmoe/rdep.py never uses it, and nmoe/csrc/
    bindings.cpp:91 only declares it (no Python m.def bindings).
