  ELEGANT MINIMALISM AUDIT - CONSOLIDATED FINDINGS

  Overview

  Audited ~12,000+ lines across 40+ Python files in 4 major areas. Overall grade: B+ - Production-ready with cleanup opportunities.

  ---
  CRITICAL ISSUES (Must Fix Before Launch)

  1. STUB FILES WITH NO IMPLEMENTATION

  | File                   | Lines | Issue                                   |
  |------------------------|-------|-----------------------------------------|
  | nmoe/data/fidelity.py  | 49    | llm_verify() raises NotImplementedError |
  | nmoe/data/quadmix.py   | 30    | sample_joint() is a no-op placeholder   |
  | nmoe/data/synthetic.py | 63    | Entirely redundant with rephrase.py     |

  Action: Delete all three files or implement properly.

  2. DEAD MODULES (Never Imported)

  | File            | Lines | Issue                                                              |
  |-----------------|-------|--------------------------------------------------------------------|
  | nmoe/scaling.py | 110   | BlockscaleActivationTracker and BlockscaleWeightTracker never used |

  Action: Delete scaling.py.

  3. MASSIVE DEAD CODE IN EVAL

  | Location            | Lines | Issue                                                                                    |
  |---------------------|-------|------------------------------------------------------------------------------------------|
  | nmoe/eval/runner.py | ~141  | Entire simulation framework (SimModel, --simulate args, _run_*_sim funcs) never wired up |

  Action: Remove simulation code or implement it.

  ---
  HIGH PRIORITY (Clean Up)

  Dead Functions

  | File                  | Function                  | Lines      |
  |-----------------------|---------------------------|------------|
  | nmoe/opt.py           | _classify_param()         | 151-162    |
  | nmoe/ggemm.py         | expert_mlp_bf16()         | 38-91      |
  | nmoe/triton/dsa.py    | lightning_indexer_ref()   | 174-196    |
  | nmoe/triton/dsa.py    | lightning_indexer_fused() | 350-398    |
  | nmoe/eval/adapters.py | _safe_get()               | 33-40      |
  | nmoe/log.py           | debug, print0, check      | 70, 73, 82 |
  | nmoe/metrics.py       | Attention import          | 17-21      |

  Unused Imports (Across All Files)

  | File                    | Import                       |
  |-------------------------|------------------------------|
  | nmoe/data/dataset.py    | gzip                         |
  | nmoe/data/prep.py       | multiprocessing as mp        |
  | nmoe/data/transforms.py | re                           |
  | nmoe/eval/hooks.py      | Optional                     |
  | nmoe/eval/runner.py     | math, random, Any, Iterable  |
  | nmoe/eval/adapters.py   | Iterable, Optional           |
  | nmoe/metrics.py         | import os as _os (duplicate) |

  Duplicate Code

  | Location                                   | Issue                                     |
  |--------------------------------------------|-------------------------------------------|
  | nmoe/attention/kda.py + nmoe/triton/kda.py | Triton allocator setup duplicated         |
  | nmoe/data/cli.py                           | import glob appears twice (lines 88, 100) |

  ---
  MEDIUM PRIORITY (Refactoring)

  Test Code in Production Modules

  | File               | Lines   | Issue               |
  |--------------------|---------|---------------------|
  | nmoe/triton/swa.py | 617-641 | Inline pytest tests |
  | nmoe/triton/nsa.py | 523-575 | Inline pytest tests |

  Action: Move to separate test files.

  Overly Large Files

  | File                | Lines | Issue                                         |
  |---------------------|-------|-----------------------------------------------|
  | nmoe/data/cli.py    | 866   | Monolithic CLI with 8+ commands               |
  | nmoe/data/prep.py   | 538   | Contains unused Beam integration (~130 lines) |
  | nmoe/eval/runner.py | 414   | 35% is dead simulation code                   |

  Outdated Comments

  | File         | Line | Issue                                        |
  |--------------|------|----------------------------------------------|
  | nmoe/rdep.py | 6    | TODO says "MODE_HYBRID" but it's implemented |

  ---
  SUMMARY STATISTICS

  | Category              | Count   | Impact      |
  |-----------------------|---------|-------------|
  | Stub files to remove  | 3       | ~140 lines  |
  | Dead modules          | 1       | ~110 lines  |
  | Dead functions        | 7       | ~250 lines  |
  | Unused imports        | 12      | Minor       |
  | Duplicate code blocks | 2       | ~30 lines   |
  | Test code in prod     | 2 files | ~100 lines  |
  | Dead eval simulation  | 1       | ~141 lines  |
  | TOTAL REMOVABLE       |         | ~770+ lines |

  ---
  RECOMMENDED CLEANUP ORDER

  Phase 1: Delete Dead Files

  rm nmoe/data/fidelity.py
  rm nmoe/data/quadmix.py
  rm nmoe/data/synthetic.py
  rm nmoe/scaling.py

  Phase 2: Remove Dead Functions

  - nmoe/opt.py: Delete _classify_param() (lines 151-162)
  - nmoe/ggemm.py: Delete expert_mlp_bf16() (lines 38-91)
  - nmoe/triton/dsa.py: Delete lightning_indexer_ref() and lightning_indexer_fused() and helpers (~200 lines)
  - nmoe/eval/runner.py: Delete all simulation code (~141 lines)
  - nmoe/log.py: Delete unused functions debug, print0, check

  Phase 3: Clean Up Imports

  Remove 12 unused imports listed above.

  Phase 4: Consolidate Duplicates

  - Create nmoe/triton/utils.py with shared triton allocator setup
  - Fix duplicate import glob in data/cli.py

  Phase 5: Move Test Code

  - Move triton/swa.py tests to tests/triton/test_swa.py
  - Move triton/nsa.py tests to tests/triton/test_nsa.py

  ---
  POSITIVE FINDINGS

  The codebase demonstrates excellent architectural discipline:

  - Clean separation of concerns
  - Minimal abstraction layers
  - Single responsibility per module
  - No multiple ways to do the same thing (except dead code)
  - Clear naming conventions
  - Comprehensive docstrings in most files
  - Strong adherence to "fail fast, fail loud" in core modules

  After cleanup, this will be a exemplary minimal codebase.

● The audit is complete. I've identified ~770+ lines of dead code that can be removed to achieve elegant minimalism.

  Key findings:
  - 3 stub files to delete entirely (fidelity.py, quadmix.py, synthetic.py)
  - 1 dead module (scaling.py - never imported anywhere)
  - 7 dead functions across various files
  - 12 unused imports
  - ~141 lines of unused simulation infrastructure in eval
  - 2 duplicate code blocks (triton allocator setup)

  Would you like me to proceed with the cleanup? I can:
  1. Delete all dead files and code (Phase 1-2)
  2. Clean up unused imports (Phase 3)
  3. Consolidate duplicates (Phase 4)

  Or if you prefer, I can tackle specific items first. What's your priority?

> i think there is more duplication, dead code and non elegant minimalism in our various *.cu files

● Search(pattern: "**/*.cu")
  ⎿  Found 77 files (ctrl+o to expand)

● Search(pattern: "**/*.cuh")
  ⎿  Found 27 files (ctrl+o to expand)

● Bash(ls -la /mlstore/src/noumena/nmoe/nmoe/csrc/)
  ⎿  total 593
     drwxrwxr-x 2 xjdr xjdr     15 Dec 15 10:40 .
     drwxrwxr-x 9 xjdr xjdr     29 Dec 15 11:11 ..
     … +13 lines (ctrl+o to expand)

● Let me audit all the CUDA source files in nmoe/csrc/. These are the core files (not third_party):

● 4 Task agents launched (ctrl+o to expand)
   ├─ Audit rdep CUDA files · Running in background
   │  ⎿  Launched
   ├─ Audit quant.cu · Running in background
   │  ⎿  Launched
   ├─ Audit optimizer and GEMM kernels · Running in background
   │  ⎿  Launched
   └─ Audit bindings.cpp · Running in background
      ⎿  Launched

● While the agents audit in detail, let me do quick scans for common issues:

● Search(pattern: "TODO|FIXME|HACK|XXX|DEPRECATED", path: "nmoe/csrc", output_mode: "content")
  ⎿  Found 0 lines

● Search(pattern: "#if 0|#ifdef DEAD|#ifdef UNUSED|#ifdef DEBUG", path: "nmoe/csrc", output_mode: "content")
  ⎿  Found 0 lines

● Search(pattern: "printf|std::cout|fprintf", path: "nmoe/csrc", output_mode: "content")
  ⎿  Found 84 lines (ctrl+o to expand)

● Search(pattern: "^//.*kernel|^/\*.*kernel", path: "nmoe/csrc", output_mode: "content")
  ⎿  Found 25 lines (ctrl+o to expand)

● Bash(grep -n "DEBUG_TEST\|__attribute__((unused))\|#pragma\s*GCC\s*diagnostic\s*ignored" /mlstore/src/noumena/nmoe/nmoe/csrc/*.cu /mlstore/src/noumena/nmoe/nmoe/csrc…)
  ⎿  /mlstore/src/noumena/nmoe/nmoe/csrc/bindings.cpp:362:    fprintf(stderr, "[DEBUG_TEST] C++ module is loaded!\n");

● Bash(wc -l /mlstore/src/noumena/nmoe/nmoe/csrc/*.cu /mlstore/src/noumena/nmoe/nmoe/csrc/*.cpp /mlstore/src/noumena/nmoe/nmoe/csrc/*.cuh | sort -n)
  ⎿      46 /mlstore/src/noumena/nmoe/nmoe/csrc/flashmla_sm100_bindings.cpp
        213 /mlstore/src/noumena/nmoe/nmoe/csrc/gpu.cpp
        300 /mlstore/src/noumena/nmoe/nmoe/csrc/rdep_nvshmem.cuh
     … +10 lines (ctrl+o to expand)

● Search(pattern: "^__global__ void", path: "nmoe/csrc", output_mode: "content")
  ⎿  Found 92 lines (ctrl+o to expand)

● Task Output a825521
  ⎿  Read output (ctrl+o to expand)

● Task Output ab8612e
  ⎿  Read output (ctrl+o to expand)

● Task Output a5a79bb
  ⎿  Read output (ctrl+o to expand)

● Task Output a3fe9fc
  ⎿  Read output (ctrl+o to expand)

● ---
  CUDA ELEGANT MINIMALISM AUDIT - CONSOLIDATED FINDINGS

  Overview

  Audited ~13,500 lines of CUDA/C++ code across 9 files. Found significant cleanup opportunities.

  ---
  CRITICAL ISSUES (Must Fix)

  1. Debug Code in Production

  | File         | Line | Issue                                                    |
  |--------------|------|----------------------------------------------------------|
  | bindings.cpp | 362  | fprintf(stderr, "[DEBUG_TEST] C++ module is loaded!\n"); |

  2. Dead Kernels

  | File     | Kernel                          | Lines     | Impact                 |
  |----------|---------------------------------|-----------|------------------------|
  | rdep.cu  | k_barrier_bf16                  | 2654-2657 | Never launched         |
  | rdep.cu  | k_barrier_blockscaled           | 2659-2661 | Never launched         |
  | quant.cu | k_dispatch_quant_fp8            | 1307-1366 | Never called           |
  | quant.cu | k_dispatch_quant_nvfp4          | 1389-1439 | Never called           |
  | quant.cu | k_unpack_nvfp4_to_bytes_variant | 1102-1155 | Debug-only, never used |

  ---
  HIGH PRIORITY (Clean Up)

  Dead Functions in ptx.cu (~150 lines)

  | Function                                                  | Status                           |
  |-----------------------------------------------------------|----------------------------------|
  | host_e8m0_encode, host_e8m0_decode                        | Never used (host-side reference) |
  | f32x4_to_e2m1x4_packed_sr                                 | Never used (stochastic rounding) |
  | st_na_relaxed_gpu_* (7 functions)                         | Never used                       |
  | ld_na_relaxed_gpu_* (4 functions)                         | Never used                       |
  | atom_cas_acquire_cta_shared, atom_exch_release_cta_shared | Never used                       |
  | lg2_approx, ex2_approx                                    | Never used                       |
  | get_laneid, bar_sync_count, elect_sync                    | Never used                       |
  | Warp reduction templates (5 structs + 4 functions)        | Never used                       |
  | UNROLLED_WARP_COPY macro                                  | Never used                       |

  Dead Functions in muon.cu

  | Item                      | Lines   | Issue                |
  |---------------------------|---------|----------------------|
  | COEFF_QUINTIC array       | 73-79   | Never used           |
  | COEFF_POLAR_EXPRESS array | 82-91   | Never used           |
  | NMOE_CHECK_CUBLASLT macro | 54-62   | Never used           |
  | coeff_mode parameter      | Various | Accepted but ignored |

  Unused Python Bindings (~52 functions)

  | Category               | Count | Examples                             |
  |------------------------|-------|--------------------------------------|
  | cuBLAS grouped-batched | 4     | bf16_dgrad_w2_cublas_grouped, etc.   |
  | Dense GEMM variants    | 6     | fused_dense_nvfp4_gemm_bf16, etc.    |
  | NVSHMEM blockscaled    | 6     | nvshmem_alloc_blockscaled, etc.      |
  | SwiGLU non-strided     | 2     | swiglu_quant_fp8, swiglu_quant_nvfp4 |
  | Debug/utility          | 3     | debug_test, reset_bf16, etc.         |
  | Other                  | 31    | Various unused exports               |

  ---
  MEDIUM PRIORITY (Consolidate)

  Duplicate Code Across Files

  | Pattern                                         | Files                           | Lines |
  |-------------------------------------------------|---------------------------------|-------|
  | Helper functions (encode_rid, decode_rid, etc.) | rdep.cu, rdep_nvshmem.cu        | ~100  |
  | Meta struct                                     | rdep.cu, rdep_nvshmem.cu        | ~10   |
  | Buffer offset calculations                      | rdep.cu, rdep_nvshmem.cu        | ~150  |
  | Triton allocator setup                          | attention/kda.py, triton/kda.py | ~30   |
  | cutlass_sf_swizzle_offset definitions           | quant.cu                        | ~80   |

  Duplicate Kernel Patterns in quant.cu

  | Pattern                      | Count     | Potential Savings        |
  |------------------------------|-----------|--------------------------|
  | FP8/NVFP4 quantization pairs | 8 kernels | ~800 lines via templates |
  | Fused GEMM variants          | 3 kernels | ~200 lines via templates |

  Duplicate Declarations in bindings.cpp

  | Function                        | Issue                           |
  |---------------------------------|---------------------------------|
  | dequant_fp8_to_bf16             | Declared twice (lines 108, 113) |
  | fused_dense_quant_fp8_gemm_bf16 | Declared twice (lines 253, 259) |

  ---
  SUMMARY STATISTICS

  | File            | Lines  | Dead Code | Duplicate | Cleanup % |
  |-----------------|--------|-----------|-----------|-----------|
  | rdep.cu         | 3,553  | ~50       | ~250      | 8%        |
  | rdep_nvshmem.cu | 2,476  | 0         | ~150      | 6%        |
  | quant.cu        | 2,613  | ~275      | ~1,000    | 49%       |
  | bindings.cpp    | 1,458  | ~800      | ~20       | 56%       |
  | ptx.cu          | 627    | ~170      | 0         | 27%       |
  | muon.cu         | 515    | ~50       | 0         | 10%       |
  | adamw.cu        | 576    | 0         | 0         | 0%        |
  | grouped_gemm.cu | 698    | 0         | 0         | 0%        |
  | lt_gemm.cu      | 404    | 0         | 0         | 0%        |
  | TOTAL           | 13,479 | ~1,345    | ~1,420    | ~20%      |

  ---
  RECOMMENDED CLEANUP ORDER

  Phase 1: Delete Dead Code (~1,345 lines)

  1. rdep.cu: Remove k_barrier_bf16, k_barrier_blockscaled
  2. quant.cu: Remove k_dispatch_quant_*, k_unpack_nvfp4_to_bytes_variant
  3. ptx.cu: Remove 32 unused functions/templates/macros
  4. muon.cu: Remove COEFF_* arrays, NMOE_CHECK_CUBLASLT
  5. bindings.cpp: Remove debug_test, duplicate declarations

  Phase 2: Remove Unused Bindings (~800 lines in bindings.cpp)

  1. Remove all cuBLAS grouped-batched functions (4 functions)
  2. Remove unused NVSHMEM blockscaled functions (6 functions)
  3. Remove unused dense GEMM variants (6 functions)
  4. Remove non-strided SwiGLU variants (2 functions)
  5. Remove other unused exports (34+ functions)

  Phase 3: Consolidate Duplicates (~1,420 lines potential savings)

  1. Create rdep_common.cuh with shared helpers
  2. Template-ize FP8/NVFP4 kernel pairs in quant.cu
  3. Consolidate cutlass_sf_swizzle_offset definitions
  4. Unify buffer offset calculations

  ---
  TOTAL IMPACT

  | Metric               | Before | After   | Savings      |
  |----------------------|--------|---------|--------------|
  | Total CUDA/C++ lines | 13,479 | ~10,700 | ~2,765 (20%) |
  | Kernel count         | ~100   | ~85     | ~15 kernels  |
  | Python bindings      | ~80    | ~40     | ~50%         |

  ---
  FILES READY FOR PRODUCTION

  These files require no cleanup:
  - adamw.cu - Clean, minimal
  - grouped_gemm.cu - Clean, sophisticated async design
  - lt_gemm.cu - Clean, minimal cuBLASLt wrapper
  - rdep_nvshmem.cuh - Clean header file

● ---
  The CUDA audit is complete. Here's what I found:

  Major Findings:

  1. ~2,765 lines removable (20% of total CUDA/C++ code)
  2. Critical dead code:
    - 2 dead barrier kernels in rdep.cu
    - 3 dead dispatch/debug kernels in quant.cu
    - Debug print statement in bindings.cpp
  3. Biggest cleanup opportunities:
    - bindings.cpp: 52 unused Python exports (~800 lines removable)
    - quant.cu: FP8/NVFP4 pairs could be templated (~1,000 lines consolidation)
    - ptx.cu: 32 unused functions from DeepEP copy (~170 lines)
  4. Clean files (no changes needed):
    - adamw.cu, grouped_gemm.cu, lt_gemm.cu - exemplary minimalism

  Would you like me to start implementing the cleanup? I'd suggest starting with:
  1. Phase 1: Delete the clearly dead code (safest, immediate impact)
  2. Phase 2: Remove unused Python bindings
  3. Phase 3: Consolidate duplicates (more complex refactoring)

