#pragma once

#include "configs.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace nmoe {
namespace rdep {

// ============================================================================
// Meta: 16-byte aligned metadata that travels with each dispatched token
// ============================================================================
//
// Encoding:
//   row_id = (rank * T + tok) * K + slot
//   local_eid = expert index on destination GPU (or packed with dest_nvl for proxy)
//   gate = gating weight for return accumulation
//
// The 16-byte alignment enables vectorized int4 stores for P2P/RDMA writes.

struct alignas(16) Meta {
    int64_t row_id;      // Encodes (src_rank, tok, slot) for return routing
    int32_t local_eid;   // Expert ID on destination GPU
    float   gate;        // Gating weight
};
static_assert(sizeof(Meta) == 16, "Meta must be 16 bytes for int4 stores");

// Row ID encoding/decoding
__device__ __host__ __forceinline__
int64_t encode_rid(int rank, int tok, int slot, int T, int K) {
    return (static_cast<int64_t>(rank) * T + tok) * K + slot;
}

__device__ __host__ __forceinline__
void decode_rid(int64_t rid, int T, int K, int* rank, int* tok, int* slot) {
    *slot = rid % K;
    int64_t tmp = rid / K;
    *tok = tmp % T;
    *rank = tmp / T;
}

// Proxy routing: pack dest_nvl_rank into high bits of local_eid for inter-node
static constexpr int META_DEST_NVL_SHIFT = 16;

__device__ __host__ __forceinline__
int meta_pack_local_eid_dest_nvl(int local_eid, int dest_nvl_rank) {
    return (dest_nvl_rank << META_DEST_NVL_SHIFT) | (local_eid & 0xFFFF);
}

__device__ __host__ __forceinline__
int meta_unpack_local_eid(int packed) {
    return packed & 0xFFFF;
}

__device__ __host__ __forceinline__
int meta_unpack_dest_nvl(int packed) {
    return (packed >> META_DEST_NVL_SHIFT) & 0xFFFF;
}

// ============================================================================
// Layout Output: what the layout phase produces
// ============================================================================
//
// The layout phase takes (eids[T,K], gates[T,K]) and produces deterministic
// mappings that tell dispatch/return where each token-slot goes.
//
// Key invariant: this is computed entirely on GPU with no host sync.
// The tok-slot protocol means each (tok, slot) pair maps to a fixed offset.

struct LayoutOutput {
    // Per-slot mappings (size = T * K = M_expanded)
    int* dest;           // [M] destination rank for each (tok, slot)
    int* local_eid;      // [M] local expert ID on destination
    int* order;          // [M] permutation for sorted access (by dest, then local_eid)

    // Expert boundaries in padded space
    int* offs_pad;       // [n_local + 1] start offset for each local expert (padded)
    int  M_pad;          // Total padded size (sum of per-expert capacities)

    // Per-rank counts (for tok-slot protocol)
    int* rank_counts;    // [world] number of tokens sent to each rank
    int* rank_prefix;    // [world + 1] prefix sum of rank_counts
};

// ============================================================================
// Dispatch Args: canonical kernel arguments for all transport modes
// ============================================================================
//
// The dispatch kernel moves x[T,H] tokens to their destination buffers.
// All modes (single/IPC/hybrid) receive the same args; only the transport differs.

template <typename T_data>
struct DispatchArgs {
    // Input tokens
    const T_data* x;           // [T, H] input activations
    const int*    eids;        // [T, K] expert IDs (global)
    const float*  gates;       // [T, K] gate values

    // Layout (from LayoutOutput)
    const int*    dest;        // [M] destination rank
    const int*    local_eid;   // [M] local expert ID on destination
    const int*    order;       // [M] sorted permutation
    const int*    offs_pad;    // [n_local + 1] expert boundaries
    const int*    rank_prefix; // [world + 1] prefix sum of per-rank counts

    // Dimensions
    int T;                     // Number of tokens
    int H;                     // Hidden dimension
    int Ha;                    // Aligned hidden dimension
    int K;                     // Top-K
    int n_local;               // Number of local experts
    int capacity;              // Max tokens per destination buffer

    // Rank info
    int my_rank;
    int world;
};

// ============================================================================
// Return Args: canonical kernel arguments for return scatter
// ============================================================================
//
// The return kernel moves expert outputs Ye[M_recv,H] back to source ranks
// and accumulates into out[T,H] with gating.
//
// Key invariant: uses tok-slot protocol. Each return writes to:
//   tok_y[tok * K + slot, :] and tok_gate[tok * K + slot]
// Then a reduction kernel accumulates: out[tok] = sum_slot(tok_gate[slot] * tok_y[slot])

template <typename T_data>
struct ReturnArgs {
    // Expert outputs
    const T_data* Ye;          // [M_recv, H] expert outputs (reordered)
    const int*    order;       // [M_recv] permutation
    const Meta*   meta;        // [M_recv] metadata for routing

    // Output accumulator
    float* out;                // [T, H] output (accumulated)

    // Tok-slot buffers (deterministic, no append counter)
    T_data* tok_y;             // [T * K, Ha] per-(tok,slot) return buffer
    float*  tok_gate;          // [T * K] per-(tok,slot) gate weights

    // Dimensions
    int M_recv;                // Received tokens (can be computed from layout)
    int T;
    int H;
    int Ha;
    int K;

    // Rank info
    int my_rank;
    int world;
};

// ============================================================================
// Buffer Layout Constants
// ============================================================================
//
// Each rank has a fixed buffer layout. For tok-slot protocol:
//   tok_slots = T * K (one slot per (token, expert) pair)
//
// Buffer sections (BF16 mode):
//   [0, capacity * Ha * 2)                - x_buf: dispatch receive buffer
//   [x_buf_end, x_buf_end + capacity * 16) - meta: metadata array
//   [meta_end, meta_end + 4)               - counter: (legacy, unused in tok-slot)
//   [counter_end, counter_end + 4)         - dropped: overflow counter
//   [dropped_end, dropped_end + MAX_RANKS * 4) - barrier_signals
//   [signals_end, signals_end + tok_slots * Ha * 2) - tok_y: return buffer
//   [tok_y_end, tok_y_end + tok_slots * 4) - tok_gate: gate weights

constexpr int BUFFER_ALIGNMENT = 128;  // Bytes

struct BufferLayout {
    size_t x_off;              // Offset to x_buf
    size_t meta_off;           // Offset to meta
    size_t counter_off;        // Offset to counter (legacy)
    size_t dropped_off;        // Offset to dropped
    size_t barrier_off;        // Offset to barrier_signals
    size_t tok_y_off;          // Offset to tok_y
    size_t tok_gate_off;       // Offset to tok_gate
    size_t total_size;         // Total buffer size
};

__host__ inline BufferLayout compute_buffer_layout_bf16(
    int capacity, int Ha, int tok_slots, int max_ranks) {

    BufferLayout layout;
    size_t offset = 0;

    // x_buf: [capacity, Ha] of uint16_t
    layout.x_off = offset;
    offset += static_cast<size_t>(capacity) * Ha * sizeof(uint16_t);
    offset = (offset + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

    // meta: [capacity] of Meta (16 bytes each)
    layout.meta_off = offset;
    offset += static_cast<size_t>(capacity) * sizeof(Meta);
    offset = (offset + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

    // counter: single int (legacy, unused in tok-slot protocol)
    layout.counter_off = offset;
    offset += sizeof(int);
    offset = (offset + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

    // dropped: single int
    layout.dropped_off = offset;
    offset += sizeof(int);
    offset = (offset + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

    // barrier_signals: [max_ranks] ints
    layout.barrier_off = offset;
    offset += static_cast<size_t>(max_ranks) * sizeof(int);
    offset = (offset + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

    // tok_y: [tok_slots, Ha] of uint16_t
    layout.tok_y_off = offset;
    offset += static_cast<size_t>(tok_slots) * Ha * sizeof(uint16_t);
    offset = (offset + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

    // tok_gate: [tok_slots] of float
    layout.tok_gate_off = offset;
    offset += static_cast<size_t>(tok_slots) * sizeof(float);
    offset = (offset + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);

    layout.total_size = offset;
    return layout;
}

// ============================================================================
// Transport Mode (runtime selection, not compile-time)
// ============================================================================

enum class TransportMode {
    LOCAL,   // Single GPU, no communication
    IPC,     // CUDA IPC (intra-node)
    HYBRID   // IPC (intra-node) + RDEP/NVSHMEM (inter-node)
};

// ============================================================================
// Invariants (enforced at runtime in debug builds)
// ============================================================================
//
// 1. tok_slots >= T * K (each token-expert pair has a dedicated slot)
// 2. capacity >= world * tok_slots (enough space for all incoming tokens)
// 3. No append counters on hot paths (tok-slot protocol is deterministic)
// 4. Meta.row_id uniquely identifies (src_rank, tok, slot)
// 5. All modes produce identical numerical results

#ifdef RDEP_DEBUG
#define RDEP_INVARIANT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("RDEP invariant violation: %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
            asm("trap;"); \
        } \
    } while (0)
#else
#define RDEP_INVARIANT(cond, msg) ((void)0)
#endif

} // namespace rdep
} // namespace nmoe
