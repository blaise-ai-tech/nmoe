#pragma once

// RDEP: Remote Dispatch/Expert-Parallel primitives for MoE
//
// This is the canonical header for the RDEP implementation.
// Import this single header to get all RDEP functionality.
//
// Architecture:
//   1. Layout (GPU): eids[T,K] + gates[T,K] → deterministic routing tables
//   2. Transport (GPU): move tokens between ranks (local/IPC/hybrid)
//   3. Return (GPU): scatter expert outputs back with gating
//
// Key invariants:
//   - Tok-slot protocol everywhere (no append counters on hot paths)
//   - Identical numerical results across all transport modes
//   - No host sync in steady state

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"
#include "contract.cuh"

#ifndef DISABLE_NVSHMEM
#include "ibgda_device.cuh"
#endif

namespace nmoe {
namespace rdep {

// Version for ABI compatibility
constexpr int RDEP_VERSION_MAJOR = 1;
constexpr int RDEP_VERSION_MINOR = 0;

} // namespace rdep
} // namespace nmoe
