// Pybind11 bindings for RDEP
// Module name: rdep

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

namespace py = pybind11;

static inline std::string cuda_err(cudaError_t err) {
  return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
}

// RDEP C symbols from rdep.cu
#ifdef BUILD_RDEP
extern "C" {
  // Init
  void rdep_init(int rank, int world, int local_world);
  int  rdep_get_mode();
  bool rdep_has_nvshmem();

#ifdef WITH_NVSHMEM
  // NVSHMEM functions (from rdep_nvshmem.cu) - use rdep_ prefix to avoid conflicts
  void rdep_nvshmem_get_uid(void* uid_out);
  int  rdep_nvshmem_get_uid_size();
  void rdep_nvshmem_init_with_uid(const void* uid, int rank, int world, int local_world);
  void rdep_nvshmem_finalize();
  void rdep_nvshmem_alloc_bf16(size_t capacity, int H, int n_local);
  void rdep_nvshmem_alloc_blockscaled(size_t capacity, int H, int n_local, int profile);
  void rdep_nvshmem_barrier();
  void rdep_nvshmem_quiet();
  // NVSHMEM IPC buffer functions (separate cudaMalloc'd buffer for intra-node IPC)
  void rdep_nvshmem_get_ipc_handle_bf16(void* handle_out);
  void rdep_nvshmem_open_ipc_handles_bf16(const void* handles, int local_world);
  void rdep_nvshmem_sync_ipc_buffer_ptrs_bf16();
  void rdep_nvshmem_get_ipc_handle_blockscaled(void* handle_out);
  void rdep_nvshmem_open_ipc_handles_blockscaled(const void* handles, int local_world);
  void rdep_nvshmem_sync_ipc_buffer_ptrs_blockscaled();
#endif
  void rdep_get_ipc_handle_bf16(void* handle_out);
  void rdep_get_ipc_handle_blockscaled(void* handle_out);
  void rdep_open_ipc_handles_bf16(const void* handles, int world);
  void rdep_open_ipc_handles_blockscaled(const void* handles, int world);
  void rdep_sync_buffer_ptrs_bf16();
  void rdep_sync_buffer_ptrs_blockscaled();

  // BF16 path
  void rdep_alloc_bf16(size_t capacity, int H, int n_local);
  void rdep_reset_bf16();
  int  rdep_dispatch_meta_bf16(const void* x, const int* eids, const float* gates,
                               int T, int K, int align,
                               int* offs_pad_out, int* M_pad_out,
                               cudaStream_t stream);
  void rdep_gather_xe_bf16(void* Xe_out, int M_recv, int M_pad, cudaStream_t stream);
  void rdep_gather_meta_sorted_bf16(int64_t* row_id_out, float* gate_out, int M_recv, cudaStream_t stream);
  void rdep_gather_from_pad_bf16(const void* in_pad, void* out_sorted, int M_recv, int H, cudaStream_t stream);
  void rdep_scatter_sorted_to_pad_bf16(const void* in_sorted, void* out_pad, int M_recv, int H, cudaStream_t stream);
  void rdep_scatter_sorted_to_pad_with_dest_bf16(const void* in_sorted, const int* dest,
                                                 void* out_pad, int M_recv, int H, cudaStream_t stream);
  int  rdep_dispatch(const void* x, const int* eids, const float* gates,
                     int T, int K,
                     void* Xe_out, int* offs_pad_out, int* dest_out,
                     int64_t* row_id_out, float* gate_out,
                     int* M_pad_out, cudaStream_t stream);
  void rdep_return_scatter(const void* Ye, void* out, int M_recv, int T, int K,
                           cudaStream_t stream);
  void rdep_gather_dy_bf16(const void* dY, const void* Ye, const int64_t* row_id, const float* gate,
                           void* dYe_out, float* dGate_out,
                           int M, int T, int H, int K, cudaStream_t stream);
  void rdep_scatter_gate_bf16(const float* dGate_sorted, const int64_t* row_id,
                              float* dGates_tk, int M, int T, int K, cudaStream_t stream);
  void rdep_scatter_dx_bf16(const void* dXe_pad, const int* dest, const int64_t* row_id,
                            void* dX_out, int M, int T, int H, int K, cudaStream_t stream);
  void rdep_scatter_dx_bf16_internal(const void* dXe_pad, const int64_t* row_id,
                                     void* dX_out, int M, int T, int H, int K, cudaStream_t stream);
  void rdep_gather_dy_ipc_bf16(const void* dY_local, const void* Ye_sorted,
                               const int64_t* row_id, const float* gate_sorted,
                               void* dYe_out, float* dGate_sorted_out, float* dGates_tk_out,
                               int M, int T, int H, int K, cudaStream_t stream);
  void rdep_scatter_dx_ipc_bf16(const void* dXe_sorted, const int64_t* row_id,
                                void* dX_out, int M, int T, int H, int K, cudaStream_t stream);
  void rdep_gather_dy_dist_bf16(const void* dY_local, const int* eids, const void* Ye_sorted,
                                const int64_t* row_id, const float* gate_sorted,
                                void* dYe_out, float* dGate_sorted_out, float* dGates_tk_out,
                                int M, int T, int H, int K, cudaStream_t stream);
  void rdep_scatter_dx_dist_bf16(const void* dXe_sorted, const int64_t* row_id,
                                 void* dX_out, int M, int T, int H, int K, cudaStream_t stream);

  // ========== DeepEP-aligned API (rdep/rdep.cu) ==========
  // These will eventually replace the above functions
  int rdep_v2_init(int rank, int num_ranks, int num_channels,
                   int buf_tokens, int H, int K, int n_experts,
                   void** ipc_handles, cudaStream_t stream);
  void rdep_v2_shutdown();
  cudaIpcMemHandle_t rdep_v2_get_ipc_handle();
  int rdep_v2_dispatch(const void* x, const int* eids, const float* gates,
                       void* recv_x, int T, cudaStream_t stream);
  void rdep_v2_combine(void* out, const void* expert_out, const float* gates,
                       int T, int M_recv, cudaStream_t stream);
  int rdep_v2_get_rank();
  int rdep_v2_get_num_ranks();
  int rdep_v2_get_hidden();
  int rdep_v2_get_topk();
  int rdep_v2_get_num_experts();
  int rdep_v2_get_capacity();
  int* rdep_v2_get_recv_src_idx();
  float* rdep_v2_get_recv_gate();

  // Blockscaled path
  void rdep_alloc_blockscaled(size_t capacity, int H, int n_local, int profile);
  void rdep_reset_blockscaled();
  int  rdep_dispatch_blockscaled(const void* x, const int* eids, const float* gates,
                                  int T, int K,
                                  void* Xe_q_out, void* Xe_sf_out,
                                  int* offs_pad_out, int* dest_out,
                                  int64_t* row_id_out, float* gate_out,
                                  int* M_pad_out, cudaStream_t stream);
  void rdep_return_scatter_blockscaled(const void* Ye, void* out, int M_recv, int T, int K,
                                       cudaStream_t stream);

  // Quantization and swizzle (dense/weights usage only)
  cudaError_t quant_fp8(const void* x, int ldx,
                        void* out, int ld_out,
                        void* sfa, int ld_sf,
                        int M, int K, cudaStream_t stream);
  cudaError_t dequant_fp8_to_bf16(const void* q, int ldq,
                                  const void* sfa, int ld_sf,
                                  int M, int K,
                                  void* out, int ldo,
                                  cudaStream_t stream);
  cudaError_t dequant_fp8_to_bf16(const void* q, int ldq,
                                  const void* sfa, int ld_sf,
                                  int M, int K,
                                  void* out, int ldo,
                                  cudaStream_t stream);
  cudaError_t quant_nvfp4(const void* x, int ldx,
                          void* out, int ld_out,
                          void* sfa, int ld_sf,
                          int M, int K, cudaStream_t stream);
  cudaError_t quant_fp8_sf_strided_mma(const void* x, int ldx,
                                       void* out, int ld_out,
                                       void* sf_mma,
                                       const int32_t* offs,
                                       int E, int M_e_stride,
                                       int M_pad, int K, cudaStream_t stream);
  cudaError_t quant_nvfp4_sf_strided_mma(const void* x, int ldx,
                                         void* out, int ld_out,
                                         void* sf_mma,
                                         const int32_t* offs,
                                         int E, int M_e_stride,
                                         int M_pad, int K, cudaStream_t stream);
  cudaError_t swiglu_bwd_bf16(const void* h1, int ld_h1,
                              const void* h3, int ld_h3,
                              const void* dA, int ld_dA,
                              void* A_out, int ld_A,
                              void* dH1_out, int ld_dH1,
                              void* dH3_out, int ld_dH3,
                              int M, int K, cudaStream_t stream);
  cudaError_t swiglu_quant_fp8(const void* h13, int ld_h13,
                               void* out, int ld_out,
                               void* sfa, int ld_sf,
                               int M, int K, cudaStream_t stream);
  cudaError_t swiglu_quant_nvfp4(const void* h13, int ld_h13,
                                 void* out, int ld_out,
                                 void* sfa, int ld_sf,
                                 int M, int K, cudaStream_t stream);
  // BF16 grouped GEMM via cuBLASLt (MoE backward).
  cudaError_t bf16_dgrad_w2_cublaslt(const void* dY,
                                     void* dA_out,
                                     const void* W2,
                                     const int32_t* offs_pad,
                                     int E, int H, int Dff,
                                     cudaStream_t stream);
  cudaError_t bf16_dgrad_w2_cublas_grouped(const void* dY,
                                           void* dA_out,
                                           const void* W2,
                                           const int32_t* offs_pad,
                                           int E, int H, int Dff,
                                           cudaStream_t stream);
  cudaError_t bf16_wgrad_w2_cublaslt(const void* A,
                                     const void* dY,
                                     void* dW2_out,
                                     const int32_t* offs_pad,
                                     int E, int H, int Dff,
                                     cudaStream_t stream);
  cudaError_t bf16_wgrad_w2_cublas_grouped(const void* A,
                                           const void* dY,
                                           void* dW2_out,
                                           const int32_t* offs_pad,
                                           int E, int H, int Dff,
                                           cudaStream_t stream);
  cudaError_t bf16_wgrad_w13_cublaslt(const void* X,
                                      const void* dH,
                                      void* dW_out,
                                      const int32_t* offs_pad,
                                      int E, int H, int Dff,
                                      cudaStream_t stream);
  cudaError_t bf16_wgrad_w13_cublas_grouped(const void* X,
                                            const void* dH,
                                            void* dW_out,
                                            const int32_t* offs_pad,
                                            int E, int H, int Dff,
                                            cudaStream_t stream);
  cudaError_t bf16_dgrad_w13_cublaslt(const void* dH,
                                      const void* W,
                                      void* dX_out,
                                      const int32_t* offs_pad,
                                      int E, int H, int Dff,
                                      float beta,
                                      cudaStream_t stream);
  cudaError_t bf16_dgrad_w13_cublas_grouped(const void* dH,
                                            const void* W,
                                            void* dX_out,
                                            const int32_t* offs_pad,
                                            int E, int H, int Dff,
                                            float beta,
                                            cudaStream_t stream);
  cudaError_t swiglu_quant_fp8_sf_strided_mma(const void* h13, int ld_h13,
                                              void* out, int ld_out,
                                              void* sf_mma,
                                              const int32_t* offs,
                                              int E, int M_e_stride,
                                              int M_pad, int K, cudaStream_t stream);
  cudaError_t swiglu_quant_nvfp4_sf_strided_mma(const void* h13, int ld_h13,
                                                void* out, int ld_out,
                                                void* sf_mma,
                                                const int32_t* offs,
                                                int E, int M_e_stride,
                                                int M_pad, int K, cudaStream_t stream);
  cudaError_t quant_fp8_with_sfa(const void* x, int ldx,
                                 void* out, int ld_out,
                                 const void* sfa, int ld_sf,
                                 int M, int K, cudaStream_t stream);
  cudaError_t quant_nvfp4_with_sfa(const void* x, int ldx,
                                   void* out, int ld_out,
                                   const void* sfa, int ld_sf,
                                   int M, int K, cudaStream_t stream);
  cudaError_t dequant_nvfp4_to_bf16(const void* q, int ldq,
                                    const void* sfa, int ld_sf,
                                    int M, int K,
                                    void* out, int ldo,
                                    cudaStream_t stream);
  cudaError_t unpack_nvfp4_to_bytes(const void* q, int ldq,
                                    void* out_bytes, int ld_bytes,
                                    int M, int K,
                                    cudaStream_t stream);
  cudaError_t unpack_nvfp4_to_bytes_variant(const void* q, int ldq,
                                            void* out_bytes, int ld_bytes,
                                            int M, int K, int variant,
                                            cudaStream_t stream);
  cudaError_t repack_nvfp4_dense_perm(const void* in_u8, int ldi_bytes,
                                      void* out_u8, int ldo_bytes,
                                      const void* perm32,
                                      int M, int K,
                                      cudaStream_t stream);
  cudaError_t swizzle_sf_dense_nvfp4(const void* sf_mkl, void* sf_mma,
                                     int M, int sf_k, cudaStream_t stream);
  cudaError_t fused_dense_nvfp4_gemm_bf16(const void* A_q, int ldq,
                                          const void* SFA, int ld_sf,
                                          const void* W, int ldw,
                                          const void* bias,
                                          void* Y, int ldy,
                                          int M, int N, int K,
                                          cudaStream_t stream);
  cudaError_t fused_dense_quant_nvfp4_gemm_bf16(const void* A, int lda,
                                                const void* W, int ldw,
                                                const void* bias,
                                                void* Y, int ldy,
                                                int M, int N, int K,
                                                cudaStream_t stream);
  cudaError_t fused_dense_quant_fp8_gemm_bf16(const void* A, int lda,
                                              const void* W, int ldw,
                                              const void* bias,
                                              void* Y, int ldy,
                                              int M, int N, int K,
                                              cudaStream_t stream);
  cudaError_t fused_dense_quant_fp8_gemm_bf16(const void* A, int lda,
                                              const void* W, int ldw,
                                              const void* bias,
                                              void* Y, int ldy,
                                              int M, int N, int K,
                                              cudaStream_t stream);
  cudaError_t swizzle_sf_mkl_to_mma(const void* sf_mkl, void* sf_mma,
                                    int M, int sf_k, cudaStream_t stream);
  // Per-expert strided swizzle for activation SFs
  cudaError_t swizzle_sf_strided(
      const void* sf_mkl,   // [M_pad, sf_k] uint8, row-major
      void* sf_mma,         // [E, M_e_swizzle, sf_k_pad] uint8, per-expert swizzled
      const int32_t* offs,  // [E+1] cumulative offsets with leading 0
      int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
      cudaStream_t stream);
  cudaError_t unswizzle_sf_strided(
      const void* sf_mma,   // [E, M_e_swizzle, sf_k_pad] uint8, per-expert swizzled
      void* sf_mkl,         // [M_pad, sf_k] uint8, row-major
      const int32_t* offs,  // [E+1] cumulative offsets with leading 0
      int E, int sf_k, int sf_k_pad, int M_pad, int M_e_swizzle,
      cudaStream_t stream);
  cudaError_t expert_adamw_step(
      int profile,
      void* W1, const void* dW1, void* m1, void* v1,
      void* W3, const void* dW3, void* m3, void* v3,
      void* W2, const void* dW2, void* m2, void* v2,
      void* W13_q, void* W13_sf_mma,
      void* W2_q, void* W2_sf_mma,
      int E, int H, int Dff,
      float lr, float beta1, float beta2,
      float weight_decay, float eps,
      float step_size, float inv_bias_correction2_sqrt,
      cudaStream_t stream);
  cudaError_t build_grouped_gemm_metadata(
      const int32_t* offs, int E,
      int64_t A_base, int64_t A_row_bytes,
      int64_t B_base, int64_t B_expert_bytes,
      int64_t C_base, int64_t C_row_bytes,
      int64_t SFA_base, int64_t SFA_expert_bytes,
      int64_t SFB_base, int64_t SFB_expert_bytes,
      int32_t A_stride0_elem, int32_t A_stride1_elem,
      int32_t B_stride0_elem, int32_t B_stride1_elem,
      int32_t C_stride0_elem, int32_t C_stride1_elem,
      int32_t N, int32_t K,
      int32_t* sizes_mnkl, int32_t* strides_abc, int64_t* ptrs_abc, int64_t* ptrs_sfasfb,
      cudaStream_t stream);
  cudaError_t grouped_dense_nvfp4_gemm_bf16_strided(
      const int32_t* sizes_mnkl,
      const int32_t* strides_abc,
      const int64_t* ptrs_abc,
      const int64_t* ptrs_sfasfb,
      int E, int sf_k,
      cudaStream_t stream);
}
#endif

static inline cudaStream_t to_stream(py::object s) {
  if (s.is_none()) return nullptr;
  auto uptr = s.attr("cuda_stream").cast<uintptr_t>();
  return reinterpret_cast<cudaStream_t>(uptr);
}

// IPC handle size is 64 bytes
constexpr int IPC_HANDLE_SIZE = 64;

#ifdef BUILD_MUON
// Muon module (built separately)
extern "C" void* muon_plan_create(int Bmax, int M, int N);
extern "C" void  muon_plan_destroy(void* plan);
extern "C" void  muon_plan_run(void* plan, void* x_bf16, long long B, int M, int N, int steps, int coeff_mode, cudaStream_t stream);
extern "C" void  muon(void* x_bf16, long long B, int M, int N, int steps, int coeff_mode, cudaStream_t stream);
PYBIND11_MODULE(muon, m2) {
  m2.doc() = "Muon orthogonalization (CUDA/cuBLAS)";
  m2.def("plan_create", [](int Bmax, int M, int N){
    return reinterpret_cast<uintptr_t>(muon_plan_create(Bmax, M, N));
  }, py::arg("Bmax"), py::arg("M"), py::arg("N"));
  m2.def("plan_destroy", [](uintptr_t plan){ muon_plan_destroy(reinterpret_cast<void*>(plan)); }, py::arg("plan"));
  m2.def("plan_run", [](uintptr_t plan, uintptr_t x_ptr, long long B, int M, int N, int steps, int coeff_mode, py::object stream){
    muon_plan_run(reinterpret_cast<void*>(plan), reinterpret_cast<void*>(x_ptr), B, M, N, steps, coeff_mode, to_stream(stream));
  }, py::arg("plan"), py::arg("x"), py::arg("B"), py::arg("M"), py::arg("N"), py::arg("steps") = 5, py::arg("coeff_mode") = 1, py::arg("stream") = py::none());
  m2.def("muon", [](uintptr_t x_ptr, long long B, int M, int N, int steps, int coeff_mode, py::object stream) {
    muon(reinterpret_cast<void*>(x_ptr), B, M, N, steps, coeff_mode, to_stream(stream));
  }, py::arg("x"), py::arg("B"), py::arg("M"), py::arg("N"), py::arg("steps") = 5, py::arg("coeff_mode") = 1, py::arg("stream") = py::none());
}
#endif

#ifdef BUILD_RDEP
PYBIND11_MODULE(rdep, m) {
  m.doc() = "RDEP: Expert-parallel dispatch/return for MoE";

  // ========== Init ==========
  m.def("init", [](int rank, int world, int local_world) {
    rdep_init(rank, world, local_world);
  }, py::arg("rank"), py::arg("world"), py::arg("local_world"),
     "Initialize RDEP with rank, world size, and local world size");

  m.def("get_mode", &rdep_get_mode,
        "Get current mode (0=SINGLE, 1=IPC, 2=HYBRID)");

  m.def("has_nvshmem", &rdep_has_nvshmem,
        "Check if NVSHMEM support is compiled in");

  m.def("debug_test", []() {
    fprintf(stderr, "[DEBUG_TEST] C++ module is loaded!\n");
    fflush(stderr);
    return 42;
  }, "Test that C++ module is loaded");

  m.def("get_ipc_handle_bf16", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_get_ipc_handle_bf16(handle.mutable_data());
    return handle;
  }, "Get local IPC handle for BF16 buffer");

  m.def("get_ipc_handle_blockscaled", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_get_ipc_handle_blockscaled(handle.mutable_data());
    return handle;
  }, "Get local IPC handle for blockscaled buffer");

  m.def("open_ipc_handles_bf16", [](py::array_t<uint8_t> handles, int world) {
    rdep_open_ipc_handles_bf16(handles.data(), world);
  }, py::arg("handles"), py::arg("world"),
     "Open remote IPC handles for BF16 path");

  m.def("open_ipc_handles_blockscaled", [](py::array_t<uint8_t> handles, int world) {
    rdep_open_ipc_handles_blockscaled(handles.data(), world);
  }, py::arg("handles"), py::arg("world"),
     "Open remote IPC handles for blockscaled path");

  m.def("sync_buffer_ptrs_bf16", &rdep_sync_buffer_ptrs_bf16,
        "Sync BF16 buffer pointers to device");
  m.def("sync_buffer_ptrs_blockscaled", &rdep_sync_buffer_ptrs_blockscaled,
        "Sync blockscaled buffer pointers to device");

  // ========== BF16 Path ==========
  m.def("alloc_bf16", [](size_t capacity, int H, int n_local) {
    rdep_alloc_bf16(capacity, H, n_local);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"),
     "Allocate BF16 dispatch buffers");

  m.def("reset_bf16", &rdep_reset_bf16,
        "Reset BF16 counters");

  m.def("dispatch_meta_bf16", [](uintptr_t x_ptr, uintptr_t eids_ptr, uintptr_t gates_ptr,
                                 int T, int K, int align,
                                 uintptr_t offs_pad_ptr, uintptr_t M_pad_ptr,
                                 py::object stream) {
    return rdep_dispatch_meta_bf16(
        reinterpret_cast<const void*>(x_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const float*>(gates_ptr),
        T, K, align,
        reinterpret_cast<int*>(offs_pad_ptr),
        reinterpret_cast<int*>(M_pad_ptr),
        to_stream(stream));
  }, py::arg("x"), py::arg("eids"), py::arg("gates"),
     py::arg("T"), py::arg("K"), py::arg("align"),
     py::arg("offs_pad"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Dispatch BF16 tokens to experts (meta only: sort + pad mapping). "
     "align: per-expert row padding (8 for BF16, 128 for blockscaled). "
     "NOTE: `M_pad` is treated as a pinned host scratch (used to read back M_recv).");

  m.def("gather_xe_bf16", [](uintptr_t Xe_out_ptr, int M_recv, int M_pad, py::object stream) {
    rdep_gather_xe_bf16(
        reinterpret_cast<void*>(Xe_out_ptr),
        M_recv, M_pad,
        to_stream(stream));
  }, py::arg("Xe_out"), py::arg("M_recv"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Gather BF16 activations into padded expert layout (requires prior dispatch_meta_bf16)");

  m.def("gather_meta_sorted_bf16", [](uintptr_t row_id_ptr, uintptr_t gate_sorted_ptr, int M_recv, py::object stream) {
    rdep_gather_meta_sorted_bf16(
        reinterpret_cast<int64_t*>(row_id_ptr),
        reinterpret_cast<float*>(gate_sorted_ptr),
        M_recv,
        to_stream(stream));
  }, py::arg("row_id"), py::arg("gate_sorted"), py::arg("M_recv"),
     py::arg("stream") = py::none(),
     "Gather sorted row_id and gate (requires prior dispatch_meta_bf16)");

  m.def("gather_from_pad_bf16", [](uintptr_t in_pad_ptr, uintptr_t out_sorted_ptr, int M_recv, int H, py::object stream) {
    rdep_gather_from_pad_bf16(
        reinterpret_cast<const void*>(in_pad_ptr),
        reinterpret_cast<void*>(out_sorted_ptr),
        M_recv, H,
        to_stream(stream));
  }, py::arg("in_pad"), py::arg("out_sorted"), py::arg("M_recv"), py::arg("H"),
     py::arg("stream") = py::none(),
     "Gather BF16 rows from padded layout into sorted layout (requires prior dispatch_meta_bf16)");

  m.def("scatter_sorted_to_pad_bf16", [](uintptr_t in_sorted_ptr, uintptr_t out_pad_ptr, int M_recv, int H, py::object stream) {
    rdep_scatter_sorted_to_pad_bf16(
        reinterpret_cast<const void*>(in_sorted_ptr),
        reinterpret_cast<void*>(out_pad_ptr),
        M_recv, H,
        to_stream(stream));
  }, py::arg("in_sorted"), py::arg("out_pad"), py::arg("M_recv"), py::arg("H"),
     py::arg("stream") = py::none(),
     "Scatter BF16 rows from sorted layout into padded layout (requires prior dispatch_meta_bf16)");

  m.def("scatter_sorted_to_pad_with_dest_bf16", [](uintptr_t in_sorted_ptr, uintptr_t dest_ptr,
                                                  uintptr_t out_pad_ptr,
                                                  int M_recv, int H, py::object stream) {
    rdep_scatter_sorted_to_pad_with_dest_bf16(
        reinterpret_cast<const void*>(in_sorted_ptr),
        reinterpret_cast<const int*>(dest_ptr),
        reinterpret_cast<void*>(out_pad_ptr),
        M_recv, H,
        to_stream(stream));
  }, py::arg("in_sorted"), py::arg("dest"), py::arg("out_pad"),
     py::arg("M_recv"), py::arg("H"),
     py::arg("stream") = py::none(),
     "Scatter BF16 rows from sorted layout into padded layout (explicit dest mapping)");

  m.def("dispatch", [](uintptr_t x_ptr, uintptr_t eids_ptr, uintptr_t gates_ptr,
                       int T, int K,
                       uintptr_t Xe_out_ptr, uintptr_t offs_pad_ptr,
                       uintptr_t dest_ptr, uintptr_t row_id_ptr, uintptr_t gate_sorted_ptr, uintptr_t M_pad_ptr,
                       py::object stream) {
    return rdep_dispatch(
        reinterpret_cast<const void*>(x_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const float*>(gates_ptr),
        T, K,
        reinterpret_cast<void*>(Xe_out_ptr),
        reinterpret_cast<int*>(offs_pad_ptr),
        reinterpret_cast<int*>(dest_ptr),
        reinterpret_cast<int64_t*>(row_id_ptr),
        reinterpret_cast<float*>(gate_sorted_ptr),
        reinterpret_cast<int*>(M_pad_ptr),
        to_stream(stream));
  }, py::arg("x"), py::arg("eids"), py::arg("gates"),
     py::arg("T"), py::arg("K"),
     py::arg("Xe_out"), py::arg("offs_pad"), py::arg("dest"), py::arg("row_id"), py::arg("gate_sorted"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Dispatch BF16 tokens to experts");

  m.def("return_scatter", [](uintptr_t Ye_ptr, uintptr_t out_ptr,
                             int M_recv, int T, int K,
                             py::object stream) {
    rdep_return_scatter(
        reinterpret_cast<const void*>(Ye_ptr),
        reinterpret_cast<void*>(out_ptr),
        M_recv, T, K, to_stream(stream));
  }, py::arg("Ye"), py::arg("out"),
     py::arg("M_recv"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Scatter expert outputs back to tokens (BF16)");

  // ========== BF16 Backward Helpers (single-GPU milestone) ==========
  m.def("gather_dy_bf16", [](uintptr_t dY_ptr, uintptr_t Ye_ptr,
                            uintptr_t row_id_ptr, uintptr_t gate_ptr,
                            uintptr_t dYe_ptr, uintptr_t dGate_ptr,
                            int M, int T, int H, int K,
                            py::object stream) {
    rdep_gather_dy_bf16(
        reinterpret_cast<const void*>(dY_ptr),
        reinterpret_cast<const void*>(Ye_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(gate_ptr),
        reinterpret_cast<void*>(dYe_ptr),
        reinterpret_cast<float*>(dGate_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dY"), py::arg("Ye"),
     py::arg("row_id"), py::arg("gate_sorted"),
     py::arg("dYe_out"), py::arg("dGate_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Backward gather: dOut[T,H] -> dYe[M,H] and dGate[M] (BF16)");

  m.def("scatter_gate_bf16", [](uintptr_t dGate_sorted_ptr, uintptr_t row_id_ptr,
                               uintptr_t dGates_tk_ptr,
                               int M, int T, int K,
                               py::object stream) {
    rdep_scatter_gate_bf16(
        reinterpret_cast<const float*>(dGate_sorted_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<float*>(dGates_tk_ptr),
        M, T, K,
        to_stream(stream));
  }, py::arg("dGate_sorted"), py::arg("row_id"), py::arg("dGates_tk"),
     py::arg("M"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Scatter dGate[M] back to [T,K] (BF16 path)");

  m.def("scatter_dx_bf16", [](uintptr_t dXe_pad_ptr, uintptr_t dest_ptr, uintptr_t row_id_ptr,
                             uintptr_t dX_out_ptr,
                             int M, int T, int H, int K,
                             py::object stream) {
    rdep_scatter_dx_bf16(
        reinterpret_cast<const void*>(dXe_pad_ptr),
        reinterpret_cast<const int*>(dest_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dX_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dXe_pad"), py::arg("dest"), py::arg("row_id"),
     py::arg("dX_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Backward scatter: dXe_pad[M_pad,H] -> dX[T,H] (float32 accum)");

  m.def("scatter_dx_bf16_internal", [](uintptr_t dXe_pad_ptr, uintptr_t row_id_ptr,
                                      uintptr_t dX_out_ptr,
                                      int M, int T, int H, int K,
                                      py::object stream) {
    rdep_scatter_dx_bf16_internal(
        reinterpret_cast<const void*>(dXe_pad_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dX_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dXe_pad"), py::arg("row_id"), py::arg("dX_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Backward scatter: dXe_pad[M_pad,H] -> dX[T,H] using internal dest mapping (float32 accum)");

  m.def("gather_dy_ipc_bf16", [](uintptr_t dY_ptr, uintptr_t Ye_ptr,
                                uintptr_t row_id_ptr, uintptr_t gate_ptr,
                                uintptr_t dYe_ptr, uintptr_t dGate_sorted_ptr, uintptr_t dGates_tk_ptr,
                                int M, int T, int H, int K,
                                py::object stream) {
    rdep_gather_dy_ipc_bf16(
        reinterpret_cast<const void*>(dY_ptr),
        reinterpret_cast<const void*>(Ye_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(gate_ptr),
        reinterpret_cast<void*>(dYe_ptr),
        reinterpret_cast<float*>(dGate_sorted_ptr),
        reinterpret_cast<float*>(dGates_tk_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dY"), py::arg("Ye"),
     py::arg("row_id"), py::arg("gate_sorted"),
     py::arg("dYe_out"), py::arg("dGate_sorted_out"), py::arg("dGates_tk_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "IPC backward gather: stages dY, computes dYe and returns/scatters dGate to [T,K]");

  m.def("scatter_dx_ipc_bf16", [](uintptr_t dXe_sorted_ptr, uintptr_t row_id_ptr,
                                 uintptr_t dX_out_ptr,
                                 int M, int T, int H, int K,
                                 py::object stream) {
    rdep_scatter_dx_ipc_bf16(
        reinterpret_cast<const void*>(dXe_sorted_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dX_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dXe_sorted"), py::arg("row_id"), py::arg("dX_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "IPC backward scatter: sends dXe rows to sources and accumulates into dX[T,H]");

  m.def("gather_dy_dist_bf16", [](uintptr_t dY_ptr, uintptr_t eids_ptr, uintptr_t Ye_ptr,
                                 uintptr_t row_id_ptr, uintptr_t gate_ptr,
                                 uintptr_t dYe_ptr, uintptr_t dGate_sorted_ptr, uintptr_t dGates_tk_ptr,
                                 int M, int T, int H, int K,
                                 py::object stream) {
    rdep_gather_dy_dist_bf16(
        reinterpret_cast<const void*>(dY_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const void*>(Ye_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<const float*>(gate_ptr),
        reinterpret_cast<void*>(dYe_ptr),
        reinterpret_cast<float*>(dGate_sorted_ptr),
        reinterpret_cast<float*>(dGates_tk_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dY"), py::arg("eids"), py::arg("Ye"),
     py::arg("row_id"), py::arg("gate_sorted"),
     py::arg("dYe_out"), py::arg("dGate_sorted_out"), py::arg("dGates_tk_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Distributed backward gather (IPC+hybrid): push-stages dY, computes dYe, returns dGate to [T,K]");

  m.def("scatter_dx_dist_bf16", [](uintptr_t dXe_sorted_ptr, uintptr_t row_id_ptr,
                                  uintptr_t dX_out_ptr,
                                  int M, int T, int H, int K,
                                  py::object stream) {
    rdep_scatter_dx_dist_bf16(
        reinterpret_cast<const void*>(dXe_sorted_ptr),
        reinterpret_cast<const int64_t*>(row_id_ptr),
        reinterpret_cast<void*>(dX_out_ptr),
        M, T, H, K,
        to_stream(stream));
  }, py::arg("dXe_sorted"), py::arg("row_id"), py::arg("dX_out"),
     py::arg("M"), py::arg("T"), py::arg("H"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Distributed backward scatter (IPC+hybrid): sends dXe rows to sources and reduces into dX[T,H]");

  // ========== Blockscaled Path ==========
  m.def("alloc_blockscaled", [](size_t capacity, int H, int n_local, int profile) {
    rdep_alloc_blockscaled(capacity, H, n_local, profile);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"), py::arg("profile"),
     "Allocate blockscaled dispatch buffers (profile: 0=fp8, 1=nvfp4)");

  m.def("reset_blockscaled", &rdep_reset_blockscaled,
        "Reset blockscaled counters");

  m.def("dispatch_blockscaled", [](uintptr_t x_ptr, uintptr_t eids_ptr, uintptr_t gates_ptr,
                                   int T, int K,
                                   uintptr_t Xe_q_ptr, uintptr_t Xe_sf_ptr,
                                   uintptr_t offs_pad_ptr, uintptr_t dest_ptr,
                                   uintptr_t row_id_ptr, uintptr_t gate_sorted_ptr,
                                   uintptr_t M_pad_ptr, py::object stream) {
    return rdep_dispatch_blockscaled(
        reinterpret_cast<const void*>(x_ptr),
        reinterpret_cast<const int*>(eids_ptr),
        reinterpret_cast<const float*>(gates_ptr),
        T, K,
        reinterpret_cast<void*>(Xe_q_ptr),
        reinterpret_cast<void*>(Xe_sf_ptr),
        reinterpret_cast<int*>(offs_pad_ptr),
        reinterpret_cast<int*>(dest_ptr),
        reinterpret_cast<int64_t*>(row_id_ptr),
        reinterpret_cast<float*>(gate_sorted_ptr),
        reinterpret_cast<int*>(M_pad_ptr),
        to_stream(stream));
  }, py::arg("x"), py::arg("eids"), py::arg("gates"),
     py::arg("T"), py::arg("K"),
     py::arg("Xe_q"), py::arg("Xe_sf"), py::arg("offs_pad"), py::arg("dest"), py::arg("row_id"), py::arg("gate_sorted"), py::arg("M_pad"),
     py::arg("stream") = py::none(),
     "Dispatch blockscaled tokens to experts (fused quant+sort+pack)");

  m.def("return_scatter_blockscaled", [](uintptr_t Ye_ptr, uintptr_t out_ptr,
                                         int M_recv, int T, int K,
                                         py::object stream) {
    rdep_return_scatter_blockscaled(
        reinterpret_cast<const void*>(Ye_ptr),
        reinterpret_cast<void*>(out_ptr),
        M_recv, T, K, to_stream(stream));
  }, py::arg("Ye"), py::arg("out"),
     py::arg("M_recv"), py::arg("T"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Scatter expert outputs back to tokens (blockscaled)");

  // ========== Expert Optimizer (AdamW + cache emit) ==========
  m.def(
      "expert_adamw_step",
      [](int profile,
         uintptr_t W1_ptr, uintptr_t dW1_ptr, uintptr_t m1_ptr, uintptr_t v1_ptr,
         uintptr_t W3_ptr, uintptr_t dW3_ptr, uintptr_t m3_ptr, uintptr_t v3_ptr,
         uintptr_t W2_ptr, uintptr_t dW2_ptr, uintptr_t m2_ptr, uintptr_t v2_ptr,
         uintptr_t W13_q_ptr, uintptr_t W13_sf_ptr,
         uintptr_t W2_q_ptr, uintptr_t W2_sf_ptr,
         int E, int H, int Dff,
         float lr, float beta1, float beta2,
         float weight_decay, float eps,
         float step_size, float inv_bias_correction2_sqrt,
         py::object stream) {
        auto err = expert_adamw_step(
            profile,
            reinterpret_cast<void*>(W1_ptr), reinterpret_cast<const void*>(dW1_ptr),
            reinterpret_cast<void*>(m1_ptr), reinterpret_cast<void*>(v1_ptr),
            reinterpret_cast<void*>(W3_ptr), reinterpret_cast<const void*>(dW3_ptr),
            reinterpret_cast<void*>(m3_ptr), reinterpret_cast<void*>(v3_ptr),
            reinterpret_cast<void*>(W2_ptr), reinterpret_cast<const void*>(dW2_ptr),
            reinterpret_cast<void*>(m2_ptr), reinterpret_cast<void*>(v2_ptr),
            reinterpret_cast<void*>(W13_q_ptr), reinterpret_cast<void*>(W13_sf_ptr),
            reinterpret_cast<void*>(W2_q_ptr), reinterpret_cast<void*>(W2_sf_ptr),
            E, H, Dff,
            lr, beta1, beta2,
            weight_decay, eps,
            step_size, inv_bias_correction2_sqrt,
            to_stream(stream));
        if (err != cudaSuccess) throw std::runtime_error("expert_adamw_step failed");
      },
      py::arg("profile"),
      py::arg("W1"),
      py::arg("dW1"),
      py::arg("m1"),
      py::arg("v1"),
      py::arg("W3"),
      py::arg("dW3"),
      py::arg("m3"),
      py::arg("v3"),
      py::arg("W2"),
      py::arg("dW2"),
      py::arg("m2"),
      py::arg("v2"),
      py::arg("W13_q"),
      py::arg("W13_sf_mma"),
      py::arg("W2_q"),
      py::arg("W2_sf_mma"),
      py::arg("E"),
      py::arg("H"),
      py::arg("Dff"),
      py::arg("lr"),
      py::arg("beta1"),
      py::arg("beta2"),
      py::arg("weight_decay"),
      py::arg("eps"),
      py::arg("step_size"),
      py::arg("inv_bias_correction2_sqrt"),
      py::arg("stream") = py::none(),
      "Fused expert AdamW update + packed weight cache emission (FP8/NVFP4)");

  // ========== Quantization + Swizzle (dense / weight cache only) ==========
  m.def("quant_fp8", [](uintptr_t x_ptr, int ldx,
                        uintptr_t out_ptr, int ld_out,
                        uintptr_t sfa_ptr, int ld_sf,
                        int M, int K, py::object stream) {
    auto err = quant_fp8(reinterpret_cast<const void*>(x_ptr), ldx,
                         reinterpret_cast<void*>(out_ptr), ld_out,
                         reinterpret_cast<void*>(sfa_ptr), ld_sf,
                         M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_fp8 failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
      "BF16 -> FP8 (packed u16) with SFA (rowwise, sf_vec=32)");

  m.def("dequant_fp8_to_bf16", [](uintptr_t q_ptr, int ldq,
                                   uintptr_t sfa_ptr, int ld_sf,
                                   int M, int K,
                                   uintptr_t out_ptr, int ldo,
                                   py::object stream) {
    auto err = dequant_fp8_to_bf16(reinterpret_cast<const void*>(q_ptr), ldq,
                                   reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                   M, K,
                                   reinterpret_cast<void*>(out_ptr), ldo,
                                   to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("dequant_fp8_to_bf16 failed");
  }, py::arg("q"), py::arg("ldq"), py::arg("sfa"), py::arg("ld_sf"),
     py::arg("M"), py::arg("K"), py::arg("out"), py::arg("ldo"),
     py::arg("stream") = py::none(),
     "FP8 (packed u16) + SFA (E8M0, per-32) -> BF16");

  m.def("quant_nvfp4", [](uintptr_t x_ptr, int ldx,
                           uintptr_t out_ptr, int ld_out,
                           uintptr_t sfa_ptr, int ld_sf,
                           int M, int K, py::object stream) {
    auto err = quant_nvfp4(reinterpret_cast<const void*>(x_ptr), ldx,
                           reinterpret_cast<void*>(out_ptr), ld_out,
                           reinterpret_cast<void*>(sfa_ptr), ld_sf,
                           M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_nvfp4 failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> NVFP4 (packed u16) with SFA (rowwise, sf_vec=32)");

  m.def("quant_fp8_sf_strided_mma", [](uintptr_t x_ptr, int ldx,
                                       uintptr_t out_ptr, int ld_out,
                                       uintptr_t sf_mma_ptr,
                                       uintptr_t offs_ptr,
                                       int E, int M_e_stride,
                                       int M_pad, int K,
                                       py::object stream) {
    auto err = quant_fp8_sf_strided_mma(reinterpret_cast<const void*>(x_ptr), ldx,
                                        reinterpret_cast<void*>(out_ptr), ld_out,
                                        reinterpret_cast<void*>(sf_mma_ptr),
                                        reinterpret_cast<const int32_t*>(offs_ptr),
                                        E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_fp8_sf_strided_mma failed");
  }, py::arg("x"), py::arg("ldx"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> FP8 (packed u16) with SFA written directly to per-expert MMA layout");

  m.def("quant_nvfp4_sf_strided_mma", [](uintptr_t x_ptr, int ldx,
                                         uintptr_t out_ptr, int ld_out,
                                         uintptr_t sf_mma_ptr,
                                         uintptr_t offs_ptr,
                                         int E, int M_e_stride,
                                         int M_pad, int K,
                                         py::object stream) {
    auto err = quant_nvfp4_sf_strided_mma(reinterpret_cast<const void*>(x_ptr), ldx,
                                          reinterpret_cast<void*>(out_ptr), ld_out,
                                          reinterpret_cast<void*>(sf_mma_ptr),
                                          reinterpret_cast<const int32_t*>(offs_ptr),
                                          E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_nvfp4_sf_strided_mma failed");
  }, py::arg("x"), py::arg("ldx"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> NVFP4 (packed u16) with SFA written directly to per-expert MMA layout");

  m.def("swiglu_bwd_bf16", [](uintptr_t h1_ptr, int ld_h1,
                              uintptr_t h3_ptr, int ld_h3,
                              uintptr_t dA_ptr, int ld_dA,
                              uintptr_t A_out_ptr, int ld_A,
                              uintptr_t dH1_out_ptr, int ld_dH1,
                              uintptr_t dH3_out_ptr, int ld_dH3,
                              int M, int K, py::object stream) {
    auto err = swiglu_bwd_bf16(reinterpret_cast<const void*>(h1_ptr), ld_h1,
                               reinterpret_cast<const void*>(h3_ptr), ld_h3,
                               reinterpret_cast<const void*>(dA_ptr), ld_dA,
                               reinterpret_cast<void*>(A_out_ptr), ld_A,
                               reinterpret_cast<void*>(dH1_out_ptr), ld_dH1,
                               reinterpret_cast<void*>(dH3_out_ptr), ld_dH3,
                               M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_bwd_bf16 failed");
  }, py::arg("h1"), py::arg("ld_h1"),
     py::arg("h3"), py::arg("ld_h3"),
     py::arg("dA"), py::arg("ld_dA"),
     py::arg("A_out"), py::arg("ld_A"),
     py::arg("dH1_out"), py::arg("ld_dH1"),
     py::arg("dH3_out"), py::arg("ld_dH3"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "SwiGLU backward (BF16): (h1, h3, dA) -> (A, dH1, dH3)");

  m.def("bf16_dgrad_w2_cublaslt", [](uintptr_t dY_ptr,
                                     uintptr_t dA_out_ptr,
                                     uintptr_t W2_ptr,
                                     uintptr_t offs_pad_ptr,
                                     int E, int H, int Dff,
                                     py::object stream) {
    auto err = bf16_dgrad_w2_cublaslt(reinterpret_cast<const void*>(dY_ptr),
                                      reinterpret_cast<void*>(dA_out_ptr),
                                      reinterpret_cast<const void*>(W2_ptr),
                                      reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                      E, H, Dff,
                                      to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_dgrad_w2_cublaslt failed");
  }, py::arg("dY"),
     py::arg("dA_out"),
     py::arg("W2"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 dgrad for W2 via cuBLASLt: dA = dY @ W2^T (FP32 accum)");

  m.def("bf16_wgrad_w2_cublaslt", [](uintptr_t A_ptr,
                                     uintptr_t dY_ptr,
                                     uintptr_t dW2_out_ptr,
                                     uintptr_t offs_pad_ptr,
                                     int E, int H, int Dff,
                                     py::object stream) {
    auto err = bf16_wgrad_w2_cublaslt(reinterpret_cast<const void*>(A_ptr),
                                      reinterpret_cast<const void*>(dY_ptr),
                                      reinterpret_cast<void*>(dW2_out_ptr),
                                      reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                      E, H, Dff,
                                      to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w2_cublaslt failed");
  }, py::arg("A"),
     py::arg("dY"),
     py::arg("dW2_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 wgrad for W2 via cuBLASLt: dW2 = A^T @ dY (FP32 accum)");

  m.def("bf16_wgrad_w13_cublaslt", [](uintptr_t X_ptr,
                                      uintptr_t dH_ptr,
                                      uintptr_t dW_out_ptr,
                                      uintptr_t offs_pad_ptr,
                                      int E, int H, int Dff,
                                      py::object stream) {
    auto err = bf16_wgrad_w13_cublaslt(reinterpret_cast<const void*>(X_ptr),
                                       reinterpret_cast<const void*>(dH_ptr),
                                       reinterpret_cast<void*>(dW_out_ptr),
                                       reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                       E, H, Dff,
                                       to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w13_cublaslt failed");
  }, py::arg("X"),
     py::arg("dH"),
     py::arg("dW_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 wgrad via cuBLASLt: dW = X^T @ dH (FP32 accum)");

  m.def("bf16_dgrad_w13_cublaslt", [](uintptr_t dH_ptr,
                                      uintptr_t W_ptr,
                                      uintptr_t dX_out_ptr,
                                      uintptr_t offs_pad_ptr,
                                      int E, int H, int Dff,
                                      float beta,
                                      py::object stream) {
    auto err = bf16_dgrad_w13_cublaslt(reinterpret_cast<const void*>(dH_ptr),
                                       reinterpret_cast<const void*>(W_ptr),
                                       reinterpret_cast<void*>(dX_out_ptr),
                                       reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                       E, H, Dff,
                                       beta,
                                       to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_dgrad_w13_cublaslt failed");
  }, py::arg("dH"),
     py::arg("W"),
     py::arg("dX_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("beta"),
     py::arg("stream") = py::none(),
     "Grouped BF16 dgrad via cuBLASLt: dX = dH @ W^T (beta controls accumulation)");

  m.def("bf16_dgrad_w2_cublas_grouped", [](uintptr_t dY_ptr,
                                           uintptr_t dA_out_ptr,
                                           uintptr_t W2_ptr,
                                           uintptr_t offs_pad_ptr,
                                           int E, int H, int Dff,
                                           py::object stream) {
    auto err = bf16_dgrad_w2_cublas_grouped(reinterpret_cast<const void*>(dY_ptr),
                                            reinterpret_cast<void*>(dA_out_ptr),
                                            reinterpret_cast<const void*>(W2_ptr),
                                            reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                            E, H, Dff,
                                            to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_dgrad_w2_cublas_grouped failed");
  }, py::arg("dY"),
     py::arg("dA_out"),
     py::arg("W2"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 dgrad for W2 via cuBLAS grouped-batched: dA = dY @ W2^T (FP32 accum)");

  m.def("bf16_wgrad_w2_cublas_grouped", [](uintptr_t A_ptr,
                                           uintptr_t dY_ptr,
                                           uintptr_t dW2_out_ptr,
                                           uintptr_t offs_pad_ptr,
                                           int E, int H, int Dff,
                                           py::object stream) {
    auto err = bf16_wgrad_w2_cublas_grouped(reinterpret_cast<const void*>(A_ptr),
                                            reinterpret_cast<const void*>(dY_ptr),
                                            reinterpret_cast<void*>(dW2_out_ptr),
                                            reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                            E, H, Dff,
                                            to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w2_cublas_grouped failed");
  }, py::arg("A"),
     py::arg("dY"),
     py::arg("dW2_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 wgrad for W2 via cuBLAS grouped-batched: dW2 = A^T @ dY (FP32 accum)");

  m.def("bf16_wgrad_w13_cublas_grouped", [](uintptr_t X_ptr,
                                            uintptr_t dH_ptr,
                                            uintptr_t dW_out_ptr,
                                            uintptr_t offs_pad_ptr,
                                            int E, int H, int Dff,
                                            py::object stream) {
    auto err = bf16_wgrad_w13_cublas_grouped(reinterpret_cast<const void*>(X_ptr),
                                             reinterpret_cast<const void*>(dH_ptr),
                                             reinterpret_cast<void*>(dW_out_ptr),
                                             reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                             E, H, Dff,
                                             to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_wgrad_w13_cublas_grouped failed");
  }, py::arg("X"),
     py::arg("dH"),
     py::arg("dW_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("stream") = py::none(),
     "Grouped BF16 wgrad via cuBLAS grouped-batched: dW = X^T @ dH (FP32 accum)");

  m.def("bf16_dgrad_w13_cublas_grouped", [](uintptr_t dH_ptr,
                                            uintptr_t W_ptr,
                                            uintptr_t dX_out_ptr,
                                            uintptr_t offs_pad_ptr,
                                            int E, int H, int Dff,
                                            float beta,
                                            py::object stream) {
    auto err = bf16_dgrad_w13_cublas_grouped(reinterpret_cast<const void*>(dH_ptr),
                                             reinterpret_cast<const void*>(W_ptr),
                                             reinterpret_cast<void*>(dX_out_ptr),
                                             reinterpret_cast<const int32_t*>(offs_pad_ptr),
                                             E, H, Dff,
                                             beta,
                                             to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("bf16_dgrad_w13_cublas_grouped failed");
  }, py::arg("dH"),
     py::arg("W"),
     py::arg("dX_out"),
     py::arg("offs_pad"),
     py::arg("E"), py::arg("H"), py::arg("Dff"),
     py::arg("beta"),
     py::arg("stream") = py::none(),
     "Grouped BF16 dgrad via cuBLAS grouped-batched: dX = dH @ W^T (beta controls accumulation)");

  m.def("swiglu_quant_fp8", [](uintptr_t h13_ptr, int ld_h13,
                               uintptr_t out_ptr, int ld_out,
                               uintptr_t sfa_ptr, int ld_sf,
                               int M, int K, py::object stream) {
    auto err = swiglu_quant_fp8(reinterpret_cast<const void*>(h13_ptr), ld_h13,
                                reinterpret_cast<void*>(out_ptr), ld_out,
                                reinterpret_cast<void*>(sfa_ptr), ld_sf,
                                M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_quant_fp8 failed");
  }, py::arg("h13"), py::arg("ld_h13"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused SwiGLU + FP8 quantization (packed u16) with rowwise SFA (sf_vec=32)");

  m.def("swiglu_quant_nvfp4", [](uintptr_t h13_ptr, int ld_h13,
                                 uintptr_t out_ptr, int ld_out,
                                 uintptr_t sfa_ptr, int ld_sf,
                                 int M, int K, py::object stream) {
    auto err = swiglu_quant_nvfp4(reinterpret_cast<const void*>(h13_ptr), ld_h13,
                                  reinterpret_cast<void*>(out_ptr), ld_out,
                                  reinterpret_cast<void*>(sfa_ptr), ld_sf,
                                  M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_quant_nvfp4 failed");
  }, py::arg("h13"), py::arg("ld_h13"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused SwiGLU + NVFP4 quantization (packed u16) with rowwise SFA (sf_vec=32)");

  m.def("swiglu_quant_fp8_sf_strided_mma", [](uintptr_t h13_ptr, int ld_h13,
                                              uintptr_t out_ptr, int ld_out,
                                              uintptr_t sf_mma_ptr,
                                              uintptr_t offs_ptr,
                                              int E, int M_e_stride,
                                              int M_pad, int K,
                                              py::object stream) {
    auto err = swiglu_quant_fp8_sf_strided_mma(reinterpret_cast<const void*>(h13_ptr), ld_h13,
                                               reinterpret_cast<void*>(out_ptr), ld_out,
                                               reinterpret_cast<void*>(sf_mma_ptr),
                                               reinterpret_cast<const int32_t*>(offs_ptr),
                                               E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_quant_fp8_sf_strided_mma failed: " + cuda_err(err));
  }, py::arg("h13"), py::arg("ld_h13"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused SwiGLU + FP8 quantization with SFA written directly to per-expert MMA layout");

  m.def("swiglu_quant_nvfp4_sf_strided_mma", [](uintptr_t h13_ptr, int ld_h13,
                                                uintptr_t out_ptr, int ld_out,
                                                uintptr_t sf_mma_ptr,
                                                uintptr_t offs_ptr,
                                                int E, int M_e_stride,
                                                int M_pad, int K,
                                                py::object stream) {
    auto err = swiglu_quant_nvfp4_sf_strided_mma(reinterpret_cast<const void*>(h13_ptr), ld_h13,
                                                 reinterpret_cast<void*>(out_ptr), ld_out,
                                                 reinterpret_cast<void*>(sf_mma_ptr),
                                                 reinterpret_cast<const int32_t*>(offs_ptr),
                                                 E, M_e_stride, M_pad, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swiglu_quant_nvfp4_sf_strided_mma failed: " + cuda_err(err));
  }, py::arg("h13"), py::arg("ld_h13"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("sf_mma"),
     py::arg("offs"), py::arg("E"), py::arg("M_e_stride"),
     py::arg("M_pad"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused SwiGLU + NVFP4 quantization with SFA written directly to per-expert MMA layout");

  m.def("quant_fp8_with_sfa", [](uintptr_t x_ptr, int ldx,
                                  uintptr_t out_ptr, int ld_out,
                                  uintptr_t sfa_ptr, int ld_sf,
                                  int M, int K, py::object stream) {
    auto err = quant_fp8_with_sfa(reinterpret_cast<const void*>(x_ptr), ldx,
                                  reinterpret_cast<void*>(out_ptr), ld_out,
                                  reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                  M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_fp8_with_sfa failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> FP8 using provided SFA (rowwise, sf_vec=32)");

  m.def("quant_nvfp4_with_sfa", [](uintptr_t x_ptr, int ldx,
                                     uintptr_t out_ptr, int ld_out,
                                     uintptr_t sfa_ptr, int ld_sf,
                                     int M, int K, py::object stream) {
    auto err = quant_nvfp4_with_sfa(reinterpret_cast<const void*>(x_ptr), ldx,
                                    reinterpret_cast<void*>(out_ptr), ld_out,
                                    reinterpret_cast<const void*>(sfa_ptr), ld_sf,
                                    M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("quant_nvfp4_with_sfa failed");
  }, py::arg("x"), py::arg("ldx"), py::arg("out"), py::arg("ld_out"),
     py::arg("sfa"), py::arg("ld_sf"), py::arg("M"), py::arg("K"),
     py::arg("stream") = py::none(),
     "BF16 -> NVFP4 using provided SFA (rowwise, sf_vec=32)");

  // NVFP4 -> BF16 dequantization (dense helper)
  m.def("dequant_nvfp4_to_bf16", [](uintptr_t q_ptr, int ldq,
                                     uintptr_t sfa_ptr, int ld_sf,
                                     int M, int K,
                                     uintptr_t out_ptr, int ldo,
                                     py::object stream) {
    auto err = dequant_nvfp4_to_bf16(
        reinterpret_cast<const void*>(q_ptr), ldq,
        reinterpret_cast<const void*>(sfa_ptr), ld_sf,
        M, K,
        reinterpret_cast<void*>(out_ptr), ldo,
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("dequant_nvfp4_to_bf16 failed");
  }, py::arg("q"), py::arg("ldq"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("M"), py::arg("K"),
     py::arg("out"), py::arg("ld_out"),
     py::arg("stream") = py::none(),
     "NVFP4 (packed u16) + SFA (E8M0, per-32) -> BF16");

  // NVFP4: expand nibble-packed to one-code-per-byte (for CuTe Float4E2M1FN)
  m.def("unpack_nvfp4_to_bytes", [](uintptr_t q_ptr, int ldq,
                                     uintptr_t out_ptr, int ld_bytes,
                                     int M, int K,
                                     py::object stream) {
    auto err = unpack_nvfp4_to_bytes(
        reinterpret_cast<const void*>(q_ptr), ldq,
        reinterpret_cast<void*>(out_ptr), ld_bytes,
        M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("unpack_nvfp4_to_bytes failed");
  }, py::arg("q"), py::arg("ldq"), py::arg("out_bytes"), py::arg("ld_bytes"),
     py::arg("M"), py::arg("K"), py::arg("stream") = py::none(),
     "NVFP4 nibble-packed u16 -> bytes [M,K] (low nibble per byte)");

  // Debug: variant reorder for NVFP4 unpack
  m.def("unpack_nvfp4_to_bytes_variant", [](uintptr_t q_ptr, int ldq,
                                             uintptr_t out_ptr, int ld_bytes,
                                             int M, int K, int variant,
                                             py::object stream) {
    auto err = unpack_nvfp4_to_bytes_variant(
        reinterpret_cast<const void*>(q_ptr), ldq,
        reinterpret_cast<void*>(out_ptr), ld_bytes,
        M, K, variant, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("unpack_nvfp4_to_bytes_variant failed");
  }, py::arg("q"), py::arg("ldq"), py::arg("out_bytes"), py::arg("ld_bytes"),
     py::arg("M"), py::arg("K"), py::arg("variant"), py::arg("stream") = py::none(),
     "NVFP4 nibble-packed u16 -> bytes [M,K] with nibble reordering variant");

  // NVFP4 dense repack using 32-lane LUT
  m.def("repack_nvfp4_dense_perm", [](uintptr_t in_ptr, int ldi_bytes,
                                       uintptr_t out_ptr, int ldo_bytes,
                                       uintptr_t perm32_ptr,
                                       int M, int K, py::object stream) {
    auto err = repack_nvfp4_dense_perm(reinterpret_cast<const void*>(in_ptr), ldi_bytes,
                                       reinterpret_cast<void*>(out_ptr), ldo_bytes,
                                       reinterpret_cast<const void*>(perm32_ptr),
                                       M, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("repack_nvfp4_dense_perm failed");
  }, py::arg("in_u8"), py::arg("ldi_bytes"), py::arg("out_u8"), py::arg("ldo_bytes"),
     py::arg("perm32"), py::arg("M"), py::arg("K"), py::arg("stream") = py::none());

  // Dense NVFP4 SF swizzle (rowwise)
  m.def("swizzle_sf_mkl_to_mma_dense_nvfp4", [](uintptr_t sf_mkl, uintptr_t sf_mma,
                                                 int M, int sf_k, py::object stream) {
    auto err = swizzle_sf_dense_nvfp4(reinterpret_cast<const void*>(sf_mkl),
                                      reinterpret_cast<void*>(sf_mma),
                                      M, sf_k, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swizzle_sf_mkl_to_mma_dense_nvfp4 failed");
  }, py::arg("sf_mkl"), py::arg("sf_mma"), py::arg("M"), py::arg("sf_k"), py::arg("stream") = py::none());

  // Fused dense GEMM: (NVFP4,SFA) x (BF16 W) -> BF16 Y
  m.def("fused_dense_nvfp4_gemm_bf16", [](uintptr_t Aq_ptr, int ldq,
                                           uintptr_t sfa_ptr, int ld_sf,
                                           uintptr_t w_ptr, int ldw,
                                           uintptr_t bias_ptr,
                                           uintptr_t y_ptr, int ldy,
                                           int M, int N, int K,
                                           py::object stream) {
    auto err = fused_dense_nvfp4_gemm_bf16(
        reinterpret_cast<const void*>(Aq_ptr), ldq,
        reinterpret_cast<const void*>(sfa_ptr), ld_sf,
        reinterpret_cast<const void*>(w_ptr), ldw,
        reinterpret_cast<const void*>(bias_ptr),
        reinterpret_cast<void*>(y_ptr), ldy,
        M, N, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("fused_dense_nvfp4_gemm_bf16 failed");
  }, py::arg("A_q"), py::arg("ldq"),
     py::arg("sfa"), py::arg("ld_sf"),
     py::arg("W"), py::arg("ldw"),
     py::arg("bias"),
     py::arg("Y"), py::arg("ld_out"),
     py::arg("M"), py::arg("N"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused dense GEMM: (NVFP4 A + per-32 SFA) x (BF16 W) -> BF16 Y");

  // Fused dense GEMM with on-the-fly quantization: (BF16 A) ->NVFP4-> x (BF16 W) -> BF16 Y
  m.def("fused_dense_quant_nvfp4_gemm_bf16", [](uintptr_t a_ptr, int lda,
                                                 uintptr_t w_ptr, int ldw,
                                                 uintptr_t bias_ptr,
                                                 uintptr_t y_ptr, int ldy,
                                                 int M, int N, int K,
                                                 py::object stream) {
    auto err = fused_dense_quant_nvfp4_gemm_bf16(
        reinterpret_cast<const void*>(a_ptr), lda,
        reinterpret_cast<const void*>(w_ptr), ldw,
        reinterpret_cast<const void*>(bias_ptr),
        reinterpret_cast<void*>(y_ptr), ldy,
        M, N, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("fused_dense_quant_nvfp4_gemm_bf16 failed");
  }, py::arg("A"), py::arg("ldA"),
     py::arg("W"), py::arg("ldW"),
     py::arg("bias"),
     py::arg("Y"), py::arg("ld_out"),
     py::arg("M"), py::arg("N"), py::arg("K"),
     py::arg("stream") = py::none(),
      "Fused dense GEMM: (BF16 A quantized per-32 NVFP4 on-the-fly) x (BF16 W) -> BF16 Y");

  // Fused dense GEMM with on-the-fly FP8 quantization
  m.def("fused_dense_quant_fp8_gemm_bf16", [](uintptr_t a_ptr, int lda,
                                              uintptr_t w_ptr, int ldw,
                                              uintptr_t bias_ptr,
                                              uintptr_t y_ptr, int ldy,
                                              int M, int N, int K,
                                              py::object stream) {
    auto err = fused_dense_quant_fp8_gemm_bf16(
        reinterpret_cast<const void*>(a_ptr), lda,
        reinterpret_cast<const void*>(w_ptr), ldw,
        reinterpret_cast<const void*>(bias_ptr),
        reinterpret_cast<void*>(y_ptr), ldy,
        M, N, K, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("fused_dense_quant_fp8_gemm_bf16 failed");
  }, py::arg("A"), py::arg("ldA"), py::arg("W"), py::arg("ldW"), py::arg("bias"),
     py::arg("Y"), py::arg("ld_out"), py::arg("M"), py::arg("N"), py::arg("K"),
     py::arg("stream") = py::none(),
     "Fused dense GEMM: (BF16 A quantized per-32 FP8 E4M3 on-the-fly) x (BF16 W) -> BF16 Y");

  m.def("swizzle_sf_mkl_to_mma", [](uintptr_t sf_mkl_ptr, uintptr_t sf_mma_ptr,
                                     int M, int sf_k, py::object stream) {
    auto err = swizzle_sf_mkl_to_mma(reinterpret_cast<const void*>(sf_mkl_ptr),
                                     reinterpret_cast<void*>(sf_mma_ptr),
                                     M, sf_k, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swizzle_sf_mkl_to_mma failed");
  }, py::arg("sf_mkl"), py::arg("sf_mma"), py::arg("M"), py::arg("sf_k"),
     py::arg("stream") = py::none(),
     "Rowwise SF swizzle to MMA layout (A-side)");

  // Per-expert strided SF swizzle (activation A-side, grouped strided path)
  m.def("swizzle_sf_strided", [](uintptr_t sf_mkl_ptr, uintptr_t sf_mma_ptr,
                                  uintptr_t offs_ptr, int E, int sf_k, int sf_k_pad,
                                  int M_pad, int M_e_swizzle, py::object stream) {
    auto err = swizzle_sf_strided(
        reinterpret_cast<const void*>(sf_mkl_ptr),
        reinterpret_cast<void*>(sf_mma_ptr),
        reinterpret_cast<const int32_t*>(offs_ptr),
        E, sf_k, sf_k_pad, M_pad, M_e_swizzle, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("swizzle_sf_strided failed");
  }, py::arg("sf_mkl"), py::arg("sf_mma"), py::arg("offs"),
     py::arg("E"), py::arg("sf_k"), py::arg("sf_k_pad"), py::arg("M_pad"), py::arg("M_e_swizzle"),
     py::arg("stream") = py::none(),
     "Per-expert strided swizzle to MMA layout (A-side)");

  m.def("unswizzle_sf_strided", [](uintptr_t sf_mma_ptr, uintptr_t sf_mkl_ptr,
                                   uintptr_t offs_ptr, int E, int sf_k, int sf_k_pad,
                                   int M_pad, int M_e_swizzle, py::object stream) {
    auto err = unswizzle_sf_strided(
        reinterpret_cast<const void*>(sf_mma_ptr),
        reinterpret_cast<void*>(sf_mkl_ptr),
        reinterpret_cast<const int32_t*>(offs_ptr),
        E, sf_k, sf_k_pad, M_pad, M_e_swizzle, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("unswizzle_sf_strided failed");
  }, py::arg("sf_mma"), py::arg("sf_mkl"), py::arg("offs"),
     py::arg("E"), py::arg("sf_k"), py::arg("sf_k_pad"), py::arg("M_pad"), py::arg("M_e_swizzle"),
     py::arg("stream") = py::none(),
     "Per-expert strided unswizzle from MMA layout to row-major (A-side)");

  // Build grouped GEMM metadata (GPU) for strided grouped interface
  m.def("build_grouped_gemm_metadata", [](uintptr_t offs_ptr, int E,
                                          int64_t A_base, int64_t A_row_bytes,
                                          int64_t B_base, int64_t B_expert_bytes,
                                          int64_t C_base, int64_t C_row_bytes,
                                          int64_t SFA_base, int64_t SFA_expert_bytes,
                                          int64_t SFB_base, int64_t SFB_expert_bytes,
                                          int32_t A_s0, int32_t A_s1,
                                          int32_t B_s0, int32_t B_s1,
                                          int32_t C_s0, int32_t C_s1,
                                          int32_t N, int32_t K,
                                          uintptr_t sizes_ptr, uintptr_t strides_ptr,
                                          uintptr_t ptrs_abc_ptr, uintptr_t ptrs_sfasfb_ptr,
                                          py::object stream) {
    auto err = build_grouped_gemm_metadata(
        reinterpret_cast<const int32_t*>(offs_ptr), E,
        A_base, A_row_bytes,
        B_base, B_expert_bytes,
        C_base, C_row_bytes,
        SFA_base, SFA_expert_bytes,
        SFB_base, SFB_expert_bytes,
        A_s0, A_s1,
        B_s0, B_s1,
        C_s0, C_s1,
        N, K,
        reinterpret_cast<int32_t*>(sizes_ptr),
        reinterpret_cast<int32_t*>(strides_ptr),
        reinterpret_cast<int64_t*>(ptrs_abc_ptr),
        reinterpret_cast<int64_t*>(ptrs_sfasfb_ptr),
        to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("build_grouped_gemm_metadata failed");
  }, py::arg("offs"), py::arg("E"),
     py::arg("A_base"), py::arg("A_row_bytes"),
     py::arg("B_base"), py::arg("B_expert_bytes"),
     py::arg("C_base"), py::arg("C_row_bytes"),
     py::arg("SFA_base"), py::arg("SFA_expert_bytes"),
     py::arg("SFB_base"), py::arg("SFB_expert_bytes"),
     py::arg("A_s0"), py::arg("A_s1"),
     py::arg("B_s0"), py::arg("B_s1"),
     py::arg("C_s0"), py::arg("C_s1"),
     py::arg("N"), py::arg("K"),
     py::arg("sizes_mnkl"), py::arg("strides_abc"),
     py::arg("ptrs_abc"), py::arg("ptrs_sfasfb"),
     py::arg("stream") = py::none(),
     "Build grouped GEMM metadata on GPU for strided grouped interface");

  // Grouped dense GEMM (skeleton): iterate experts and call fused dense
  m.def("grouped_dense_nvfp4_gemm_bf16_strided", [](uintptr_t sizes_ptr,
                                                     uintptr_t strides_ptr,
                                                     uintptr_t ptrs_abc_ptr,
                                                     uintptr_t ptrs_sfasfb_ptr,
                                                     int E, int sf_k,
                                                     py::object stream) {
    auto err = grouped_dense_nvfp4_gemm_bf16_strided(
        reinterpret_cast<const int32_t*>(sizes_ptr),
        reinterpret_cast<const int32_t*>(strides_ptr),
        reinterpret_cast<const int64_t*>(ptrs_abc_ptr),
        reinterpret_cast<const int64_t*>(ptrs_sfasfb_ptr),
        E, sf_k, to_stream(stream));
    if (err != cudaSuccess) throw std::runtime_error("grouped_dense_nvfp4_gemm_bf16_strided failed");
  }, py::arg("sizes_mnkl"), py::arg("strides_abc"),
     py::arg("ptrs_abc"), py::arg("ptrs_sfasfb"),
     py::arg("E"), py::arg("sf_k"), py::arg("stream") = py::none(),
     "Grouped dense GEMM (NVFP4 A + per-32 SFA) using per-expert pointers and strides");

#ifdef WITH_NVSHMEM
  // ========== NVSHMEM Functions ==========
  m.def("nvshmem_get_uid", []() {
    int size = rdep_nvshmem_get_uid_size();
    py::array_t<uint8_t> uid(size);
    rdep_nvshmem_get_uid(uid.mutable_data());
    return uid;
  }, "Get NVSHMEM UID for bootstrap (call on rank 0 only)");

  m.def("nvshmem_get_uid_size", &rdep_nvshmem_get_uid_size,
        "Get NVSHMEM UID size in bytes");

  m.def("nvshmem_init", [](py::array_t<uint8_t> uid, int rank, int world, int local_world) {
    rdep_nvshmem_init_with_uid(uid.data(), rank, world, local_world);
  }, py::arg("uid"), py::arg("rank"), py::arg("world"), py::arg("local_world"),
     "Initialize NVSHMEM with UID from rank 0");

  m.def("nvshmem_finalize", &rdep_nvshmem_finalize,
        "Finalize NVSHMEM");

  m.def("nvshmem_alloc_bf16", [](size_t capacity, int H, int n_local) {
    rdep_nvshmem_alloc_bf16(capacity, H, n_local);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"),
     "Allocate NVSHMEM symmetric buffers for BF16 path");

  m.def("nvshmem_alloc_blockscaled", [](size_t capacity, int H, int n_local, int profile) {
    rdep_nvshmem_alloc_blockscaled(capacity, H, n_local, profile);
  }, py::arg("capacity"), py::arg("H"), py::arg("n_local"), py::arg("profile"),
     "Allocate NVSHMEM symmetric buffers for blockscaled path");

  m.def("nvshmem_barrier", &rdep_nvshmem_barrier,
        "NVSHMEM barrier (all PEs)");

  m.def("nvshmem_quiet", &rdep_nvshmem_quiet,
        "NVSHMEM quiet (ensure all puts complete)");

  // NVSHMEM IPC buffer functions (separate cudaMalloc'd buffer)
  // These are different from rdep get_ipc_handle_bf16 - they get handles
  // from the cudaMalloc'd IPC buffer, NOT from NVSHMEM symmetric heap
  m.def("nvshmem_get_ipc_handle_bf16", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_nvshmem_get_ipc_handle_bf16(handle.mutable_data());
    return handle;
  }, "Get IPC handle for NVSHMEM's separate IPC buffer (BF16)");

  m.def("nvshmem_open_ipc_handles_bf16", [](py::array_t<uint8_t> handles, int local_world) {
    rdep_nvshmem_open_ipc_handles_bf16(handles.data(), local_world);
  }, py::arg("handles"), py::arg("local_world"),
     "Open remote IPC handles for NVSHMEM's IPC buffer (BF16)");

  m.def("nvshmem_sync_ipc_buffer_ptrs_bf16", &rdep_nvshmem_sync_ipc_buffer_ptrs_bf16,
        "Sync NVSHMEM IPC buffer pointers to device");

  m.def("nvshmem_get_ipc_handle_blockscaled", []() {
    py::array_t<uint8_t> handle(IPC_HANDLE_SIZE);
    rdep_nvshmem_get_ipc_handle_blockscaled(handle.mutable_data());
    return handle;
  }, "Get IPC handle for NVSHMEM's separate IPC buffer (blockscaled)");

  m.def("nvshmem_open_ipc_handles_blockscaled", [](py::array_t<uint8_t> handles, int local_world) {
    rdep_nvshmem_open_ipc_handles_blockscaled(handles.data(), local_world);
  }, py::arg("handles"), py::arg("local_world"),
     "Open remote IPC handles for NVSHMEM's IPC buffer (blockscaled)");

  m.def("nvshmem_sync_ipc_buffer_ptrs_blockscaled", &rdep_nvshmem_sync_ipc_buffer_ptrs_blockscaled,
        "Sync NVSHMEM IPC buffer pointers to device (blockscaled)");
#endif
}
#endif
