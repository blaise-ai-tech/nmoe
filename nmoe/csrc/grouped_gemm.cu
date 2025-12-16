#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
 
#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>
 
namespace {
 
inline cudaError_t cuda_from_cublas(cublasStatus_t s) {
  return (s == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}
 
struct HandleState {
  cublasHandle_t handle = nullptr;
  bool initialized = false;
};
 
HandleState& handle_state() {
  static thread_local HandleState s;
  return s;
}
 
cudaError_t ensure_cublas(cudaStream_t stream) {
  auto& s = handle_state();
  if (s.handle == nullptr) {
    auto st = cublasCreate(&s.handle);
    if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
    st = cublasSetMathMode(s.handle, CUBLAS_TENSOR_OP_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
    st = cublasSetPointerMode(s.handle, CUBLAS_POINTER_MODE_HOST);
    if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
    s.initialized = true;
  }
  auto st = cublasSetStream(s.handle, stream);
  return cuda_from_cublas(st);
}

struct PendingGroupedCall {
  cudaEvent_t done = nullptr;
  std::vector<cublasOperation_t> transa;
  std::vector<cublasOperation_t> transb;
  std::vector<int> m;
  std::vector<int> n;
  std::vector<int> k;
  std::vector<int> lda;
  std::vector<int> ldb;
  std::vector<int> ldc;
  std::vector<int> group_size;
  std::vector<float> alpha;
  std::vector<float> beta;
  std::vector<const void*> A;
  std::vector<const void*> B;
  std::vector<void*> C;
 
  PendingGroupedCall() = default;
  PendingGroupedCall(const PendingGroupedCall&) = delete;
  PendingGroupedCall& operator=(const PendingGroupedCall&) = delete;
  PendingGroupedCall(PendingGroupedCall&& other) noexcept
      : done(other.done),
        transa(std::move(other.transa)),
        transb(std::move(other.transb)),
        m(std::move(other.m)),
        n(std::move(other.n)),
        k(std::move(other.k)),
        lda(std::move(other.lda)),
        ldb(std::move(other.ldb)),
        ldc(std::move(other.ldc)),
        group_size(std::move(other.group_size)),
        alpha(std::move(other.alpha)),
        beta(std::move(other.beta)),
        A(std::move(other.A)),
        B(std::move(other.B)),
        C(std::move(other.C)) {
    other.done = nullptr;
  }
  PendingGroupedCall& operator=(PendingGroupedCall&& other) noexcept {
    if (this == &other) return *this;
    if (done) cudaEventDestroy(done);
    done = other.done;
    other.done = nullptr;
    transa = std::move(other.transa);
    transb = std::move(other.transb);
    m = std::move(other.m);
    n = std::move(other.n);
    k = std::move(other.k);
    lda = std::move(other.lda);
    ldb = std::move(other.ldb);
    ldc = std::move(other.ldc);
    group_size = std::move(other.group_size);
    alpha = std::move(other.alpha);
    beta = std::move(other.beta);
    A = std::move(other.A);
    B = std::move(other.B);
    C = std::move(other.C);
    return *this;
  }
 
  ~PendingGroupedCall() {
    if (done) cudaEventDestroy(done);
  }
};
 
std::deque<PendingGroupedCall>& pending_calls() {
  static thread_local std::deque<PendingGroupedCall> q;
  return q;
}
 
void reap_pending_calls() {
  auto& q = pending_calls();
  for (auto it = q.begin(); it != q.end();) {
    const auto st = cudaEventQuery(it->done);
    if (st == cudaSuccess) {
      it = q.erase(it);
      continue;
    }
    if (st == cudaErrorNotReady) {
      ++it;
      continue;
    }
    // On unexpected CUDA errors, stop reaping to avoid spinning.
    break;
  }
}
 
template <typename MakePtrsFn>
static inline void group_ptrs_by_tokens(
    int E,
    const int32_t* offs_pad,
    MakePtrsFn make_ptrs,
    std::vector<int>* group_m,
    std::vector<std::vector<const void*>>* group_A,
    std::vector<std::vector<const void*>>* group_B,
    std::vector<std::vector<void*>>* group_C) {
  std::unordered_map<int, int> m_to_group;
  int32_t start = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad[e];
    const int tokens = static_cast<int>(end - start);
    if (tokens > 0) {
      auto it = m_to_group.find(tokens);
      int gid = 0;
      if (it == m_to_group.end()) {
        gid = static_cast<int>(group_m->size());
        m_to_group.emplace(tokens, gid);
        group_m->push_back(tokens);
        group_A->emplace_back();
        group_B->emplace_back();
        group_C->emplace_back();
      } else {
        gid = it->second;
      }
      const void* A = nullptr;
      const void* B = nullptr;
      void* C = nullptr;
      make_ptrs(e, start, tokens, &A, &B, &C);
      (*group_A)[gid].push_back(A);
      (*group_B)[gid].push_back(B);
      (*group_C)[gid].push_back(C);
    }
    start = end;
  }
}
 
}  // namespace

extern "C" cudaError_t bf16_dgrad_w2_cublas_grouped(const void* dY,
                                                    void* dA_out,
                                                    const void* W2,
                                                    const int32_t* offs_pad,
                                                    int E,
                                                    int H,
                                                    int Dff,
                                                    cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;
 
  const auto* dY_bf16 = reinterpret_cast<const __nv_bfloat16*>(dY);
  auto* dA_bf16 = reinterpret_cast<__nv_bfloat16*>(dA_out);
  const auto* W2_bf16 = reinterpret_cast<const __nv_bfloat16*>(W2);
 
  // Column-major view trick:
  // dA_row [m, Dff] == dA_col [Dff, m]
  // dY_row [m, H]   == dY_col [H, m]
  // W2_row [Dff, H] == W2_col [H, Dff] (transpose)
  // dA_col = (W2_col)^T * dY_col
 
  auto err = ensure_cublas(stream);
  if (err != cudaSuccess) return err;
  reap_pending_calls();
 
  const auto make_ptrs = [&](int e, int32_t start, int /*m*/,
                             const void** Ap, const void** Bp, void** Cp) {
    *Ap = W2_bf16 + static_cast<size_t>(e) * static_cast<size_t>(Dff) * static_cast<size_t>(H);
    *Bp = dY_bf16 + static_cast<size_t>(start) * static_cast<size_t>(H);
    *Cp = dA_bf16 + static_cast<size_t>(start) * static_cast<size_t>(Dff);
  };
 
  std::vector<int> group_m;
  std::vector<std::vector<const void*>> group_A;
  std::vector<std::vector<const void*>> group_B;
  std::vector<std::vector<void*>> group_C;
  group_ptrs_by_tokens(E, offs_pad, make_ptrs, &group_m, &group_A, &group_B, &group_C);
 
  const int group_count = static_cast<int>(group_m.size());
  if (group_count == 0) return cudaSuccess;
 
  std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_T);
  std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_N);
  std::vector<int> m_array(group_count, Dff);
  std::vector<int> n_array(group_count);
  std::vector<int> k_array(group_count, H);
  std::vector<int> lda_array(group_count, H);
  std::vector<int> ldb_array(group_count, H);
  std::vector<int> ldc_array(group_count, Dff);
  std::vector<int> group_size(group_count);
 
  auto& q = pending_calls();
  q.emplace_back();
  auto& p = q.back();
  if (!p.done) {
    auto st = cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming);
    if (st != cudaSuccess) {
      q.pop_back();
      return st;
    }
  }
 
  p.transa = std::move(transa);
  p.transb = std::move(transb);
  p.m = std::move(m_array);
  p.n = std::move(n_array);
  p.k = std::move(k_array);
  p.lda = std::move(lda_array);
  p.ldb = std::move(ldb_array);
  p.ldc = std::move(ldc_array);
  p.group_size = std::move(group_size);
  p.alpha.assign(group_count, 1.0f);
  p.beta.assign(group_count, 0.0f);
 
  size_t total = 0;
  for (int g = 0; g < group_count; ++g) total += group_A[g].size();
  p.A.clear();
  p.B.clear();
  p.C.clear();
  p.A.reserve(total);
  p.B.reserve(total);
  p.C.reserve(total);
 
  for (int g = 0; g < group_count; ++g) {
    p.group_size[g] = static_cast<int>(group_A[g].size());
    p.n[g] = group_m[g];
    p.A.insert(p.A.end(), group_A[g].begin(), group_A[g].end());
    p.B.insert(p.B.end(), group_B[g].begin(), group_B[g].end());
    p.C.insert(p.C.end(), group_C[g].begin(), group_C[g].end());
  }
 
  const cudaDataType_t bf16 = CUDA_R_16BF;
  const cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
  auto& s = handle_state();
  const cublasStatus_t st = cublasGemmGroupedBatchedEx(
      s.handle,
      p.transa.data(),
      p.transb.data(),
      p.m.data(),
      p.n.data(),
      p.k.data(),
      p.alpha.data(),
      p.A.data(),
      bf16,
      p.lda.data(),
      p.B.data(),
      bf16,
      p.ldb.data(),
      p.beta.data(),
      p.C.data(),
      bf16,
      p.ldc.data(),
      group_count,
      p.group_size.data(),
      compute);
  if (st != CUBLAS_STATUS_SUCCESS) {
    q.pop_back();
    return cuda_from_cublas(st);
  }
  auto ev = cudaEventRecord(p.done, stream);
  if (ev != cudaSuccess) {
    q.pop_back();
    return ev;
  }
  return cudaSuccess;
}
 
extern "C" cudaError_t bf16_wgrad_w2_cublas_grouped(const void* A,
                                                    const void* dY,
                                                    void* dW2_out,
                                                    const int32_t* offs_pad,
                                                    int E,
                                                    int H,
                                                    int Dff,
                                                    cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;
 
  const auto* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A);
  const auto* dY_bf16 = reinterpret_cast<const __nv_bfloat16*>(dY);
  auto* dW2_bf16 = reinterpret_cast<__nv_bfloat16*>(dW2_out);
 
  // Output dW2_row [Dff, H] == dW2_col [H, Dff]
  // dW2_col = dY_col [H, m] * (A_col [Dff, m])^T
 
  const auto make_ptrs = [&](int e, int32_t start, int /*m*/,
                             const void** Ap, const void** Bp, void** Cp) {
    *Ap = dY_bf16 + static_cast<size_t>(start) * static_cast<size_t>(H);
    *Bp = A_bf16 + static_cast<size_t>(start) * static_cast<size_t>(Dff);
    *Cp = dW2_bf16 + static_cast<size_t>(e) * static_cast<size_t>(Dff) * static_cast<size_t>(H);
  };
 
  auto err = ensure_cublas(stream);
  if (err != cudaSuccess) return err;
  reap_pending_calls();
 
  std::vector<int> group_m;
  std::vector<std::vector<const void*>> group_A;
  std::vector<std::vector<const void*>> group_B;
  std::vector<std::vector<void*>> group_C;
  group_ptrs_by_tokens(E, offs_pad, make_ptrs, &group_m, &group_A, &group_B, &group_C);
 
  const int group_count = static_cast<int>(group_m.size());
  if (group_count == 0) return cudaSuccess;
 
  std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_T);
  std::vector<int> m_array(group_count, H);
  std::vector<int> n_array(group_count, Dff);
  std::vector<int> k_array(group_count);
  std::vector<int> lda_array(group_count, H);
  std::vector<int> ldb_array(group_count, Dff);
  std::vector<int> ldc_array(group_count, H);
  std::vector<int> group_size(group_count);
 
  auto& q = pending_calls();
  q.emplace_back();
  auto& p = q.back();
  if (!p.done) {
    auto st = cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming);
    if (st != cudaSuccess) {
      q.pop_back();
      return st;
    }
  }
 
  p.transa = std::move(transa);
  p.transb = std::move(transb);
  p.m = std::move(m_array);
  p.n = std::move(n_array);
  p.k = std::move(k_array);
  p.lda = std::move(lda_array);
  p.ldb = std::move(ldb_array);
  p.ldc = std::move(ldc_array);
  p.group_size = std::move(group_size);
  p.alpha.assign(group_count, 1.0f);
  p.beta.assign(group_count, 0.0f);
 
  size_t total = 0;
  for (int g = 0; g < group_count; ++g) total += group_A[g].size();
  p.A.clear();
  p.B.clear();
  p.C.clear();
  p.A.reserve(total);
  p.B.reserve(total);
  p.C.reserve(total);
 
  for (int g = 0; g < group_count; ++g) {
    p.group_size[g] = static_cast<int>(group_A[g].size());
    p.k[g] = group_m[g];
    p.A.insert(p.A.end(), group_A[g].begin(), group_A[g].end());
    p.B.insert(p.B.end(), group_B[g].begin(), group_B[g].end());
    p.C.insert(p.C.end(), group_C[g].begin(), group_C[g].end());
  }
 
  const cudaDataType_t bf16 = CUDA_R_16BF;
  const cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
  auto& s = handle_state();
  const cublasStatus_t st = cublasGemmGroupedBatchedEx(
      s.handle,
      p.transa.data(),
      p.transb.data(),
      p.m.data(),
      p.n.data(),
      p.k.data(),
      p.alpha.data(),
      p.A.data(),
      bf16,
      p.lda.data(),
      p.B.data(),
      bf16,
      p.ldb.data(),
      p.beta.data(),
      p.C.data(),
      bf16,
      p.ldc.data(),
      group_count,
      p.group_size.data(),
      compute);
  if (st != CUBLAS_STATUS_SUCCESS) {
    q.pop_back();
    return cuda_from_cublas(st);
  }
  auto ev = cudaEventRecord(p.done, stream);
  if (ev != cudaSuccess) {
    q.pop_back();
    return ev;
  }
  return cudaSuccess;
}
 
extern "C" cudaError_t bf16_wgrad_w13_cublas_grouped(const void* X,
                                                     const void* dH,
                                                     void* dW_out,
                                                     const int32_t* offs_pad,
                                                     int E,
                                                     int H,
                                                     int Dff,
                                                     cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;
 
  const auto* X_bf16 = reinterpret_cast<const __nv_bfloat16*>(X);
  const auto* dH_bf16 = reinterpret_cast<const __nv_bfloat16*>(dH);
  auto* dW_bf16 = reinterpret_cast<__nv_bfloat16*>(dW_out);
 
  auto err = ensure_cublas(stream);
  if (err != cudaSuccess) return err;
  reap_pending_calls();
 
  std::unordered_map<int, int> m_to_group;
  std::vector<int> group_m;
  std::vector<std::vector<const void*>> group_A;
  std::vector<std::vector<const void*>> group_B;
  std::vector<std::vector<void*>> group_C;
 
  int32_t start = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad[e];
    const int tokens = static_cast<int>(end - start);
    if (tokens > 0) {
      auto it = m_to_group.find(tokens);
      int gid = 0;
      if (it == m_to_group.end()) {
        gid = static_cast<int>(group_m.size());
        m_to_group.emplace(tokens, gid);
        group_m.push_back(tokens);
        group_A.emplace_back();
        group_B.emplace_back();
        group_C.emplace_back();
      } else {
        gid = it->second;
      }
      const void* Ap = dH_bf16 + static_cast<size_t>(start) * static_cast<size_t>(Dff);
      const void* Bp = X_bf16 + static_cast<size_t>(start) * static_cast<size_t>(H);
      void* Cp = dW_bf16 + static_cast<size_t>(e) * static_cast<size_t>(H) * static_cast<size_t>(Dff);
      group_A[gid].push_back(Ap);
      group_B[gid].push_back(Bp);
      group_C[gid].push_back(Cp);
    }
    start = end;
  }
 
  const int group_count = static_cast<int>(group_m.size());
  if (group_count == 0) return cudaSuccess;
 
  std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_T);
  std::vector<int> m_array(group_count, Dff);
  std::vector<int> n_array(group_count, H);
  std::vector<int> k_array(group_count);
  std::vector<int> lda_array(group_count, Dff);
  std::vector<int> ldb_array(group_count, H);
  std::vector<int> ldc_array(group_count, Dff);
  std::vector<int> group_size(group_count);
 
  auto& q = pending_calls();
  q.emplace_back();
  auto& p = q.back();
  if (!p.done) {
    auto st = cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming);
    if (st != cudaSuccess) {
      q.pop_back();
      return st;
    }
  }
 
  p.transa = std::move(transa);
  p.transb = std::move(transb);
  p.m = std::move(m_array);
  p.n = std::move(n_array);
  p.k = std::move(k_array);
  p.lda = std::move(lda_array);
  p.ldb = std::move(ldb_array);
  p.ldc = std::move(ldc_array);
  p.group_size = std::move(group_size);
  p.alpha.assign(group_count, 1.0f);
  p.beta.assign(group_count, 0.0f);
 
  size_t total = 0;
  for (int g = 0; g < group_count; ++g) total += group_A[g].size();
  p.A.clear();
  p.B.clear();
  p.C.clear();
  p.A.reserve(total);
  p.B.reserve(total);
  p.C.reserve(total);
 
  for (int g = 0; g < group_count; ++g) {
    p.group_size[g] = static_cast<int>(group_A[g].size());
    p.k[g] = group_m[g];
    p.A.insert(p.A.end(), group_A[g].begin(), group_A[g].end());
    p.B.insert(p.B.end(), group_B[g].begin(), group_B[g].end());
    p.C.insert(p.C.end(), group_C[g].begin(), group_C[g].end());
  }
 
  const cudaDataType_t bf16 = CUDA_R_16BF;
  const cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
  auto& s = handle_state();
  const cublasStatus_t st = cublasGemmGroupedBatchedEx(
      s.handle,
      p.transa.data(),
      p.transb.data(),
      p.m.data(),
      p.n.data(),
      p.k.data(),
      p.alpha.data(),
      p.A.data(),
      bf16,
      p.lda.data(),
      p.B.data(),
      bf16,
      p.ldb.data(),
      p.beta.data(),
      p.C.data(),
      bf16,
      p.ldc.data(),
      group_count,
      p.group_size.data(),
      compute);
  if (st != CUBLAS_STATUS_SUCCESS) {
    q.pop_back();
    return cuda_from_cublas(st);
  }
  auto ev = cudaEventRecord(p.done, stream);
  if (ev != cudaSuccess) {
    q.pop_back();
    return ev;
  }
  return cudaSuccess;
}
 
extern "C" cudaError_t bf16_dgrad_w13_cublas_grouped(const void* dH,
                                                     const void* W,
                                                     void* dX_out,
                                                     const int32_t* offs_pad,
                                                     int E,
                                                     int H,
                                                     int Dff,
                                                     float beta_scalar,
                                                     cudaStream_t stream) {
  if (E <= 0 || H <= 0 || Dff <= 0) return cudaSuccess;
 
  const auto* dH_bf16 = reinterpret_cast<const __nv_bfloat16*>(dH);
  const auto* W_bf16 = reinterpret_cast<const __nv_bfloat16*>(W);
  auto* dX_bf16 = reinterpret_cast<__nv_bfloat16*>(dX_out);
 
  auto err = ensure_cublas(stream);
  if (err != cudaSuccess) return err;
  reap_pending_calls();
 
  std::unordered_map<int, int> m_to_group;
  std::vector<int> group_m;
  std::vector<std::vector<const void*>> group_A;
  std::vector<std::vector<const void*>> group_B;
  std::vector<std::vector<void*>> group_C;
 
  int32_t start = 0;
  for (int e = 0; e < E; ++e) {
    const int32_t end = offs_pad[e];
    const int tokens = static_cast<int>(end - start);
    if (tokens > 0) {
      auto it = m_to_group.find(tokens);
      int gid = 0;
      if (it == m_to_group.end()) {
        gid = static_cast<int>(group_m.size());
        m_to_group.emplace(tokens, gid);
        group_m.push_back(tokens);
        group_A.emplace_back();
        group_B.emplace_back();
        group_C.emplace_back();
      } else {
        gid = it->second;
      }
      const void* Ap = W_bf16 + static_cast<size_t>(e) * static_cast<size_t>(H) * static_cast<size_t>(Dff);
      const void* Bp = dH_bf16 + static_cast<size_t>(start) * static_cast<size_t>(Dff);
      void* Cp = dX_bf16 + static_cast<size_t>(start) * static_cast<size_t>(H);
      group_A[gid].push_back(Ap);
      group_B[gid].push_back(Bp);
      group_C[gid].push_back(Cp);
    }
    start = end;
  }
 
  const int group_count = static_cast<int>(group_m.size());
  if (group_count == 0) return cudaSuccess;
 
  std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_T);
  std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_N);
  std::vector<int> m_array(group_count, H);
  std::vector<int> n_array(group_count);
  std::vector<int> k_array(group_count, Dff);
  std::vector<int> lda_array(group_count, Dff);
  std::vector<int> ldb_array(group_count, Dff);
  std::vector<int> ldc_array(group_count, H);
  std::vector<int> group_size(group_count);
 
  auto& q = pending_calls();
  q.emplace_back();
  auto& p = q.back();
  if (!p.done) {
    auto st = cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming);
    if (st != cudaSuccess) {
      q.pop_back();
      return st;
    }
  }
 
  p.transa = std::move(transa);
  p.transb = std::move(transb);
  p.m = std::move(m_array);
  p.n = std::move(n_array);
  p.k = std::move(k_array);
  p.lda = std::move(lda_array);
  p.ldb = std::move(ldb_array);
  p.ldc = std::move(ldc_array);
  p.group_size = std::move(group_size);
  p.alpha.assign(group_count, 1.0f);
  p.beta.assign(group_count, beta_scalar);
 
  size_t total = 0;
  for (int g = 0; g < group_count; ++g) total += group_A[g].size();
  p.A.clear();
  p.B.clear();
  p.C.clear();
  p.A.reserve(total);
  p.B.reserve(total);
  p.C.reserve(total);
 
  for (int g = 0; g < group_count; ++g) {
    p.group_size[g] = static_cast<int>(group_A[g].size());
    p.n[g] = group_m[g];
    p.A.insert(p.A.end(), group_A[g].begin(), group_A[g].end());
    p.B.insert(p.B.end(), group_B[g].begin(), group_B[g].end());
    p.C.insert(p.C.end(), group_C[g].begin(), group_C[g].end());
  }
 
  const cudaDataType_t bf16 = CUDA_R_16BF;
  const cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
  auto& s = handle_state();
  const cublasStatus_t st = cublasGemmGroupedBatchedEx(
      s.handle,
      p.transa.data(),
      p.transb.data(),
      p.m.data(),
      p.n.data(),
      p.k.data(),
      p.alpha.data(),
      p.A.data(),
      bf16,
      p.lda.data(),
      p.B.data(),
      bf16,
      p.ldb.data(),
      p.beta.data(),
      p.C.data(),
      bf16,
      p.ldc.data(),
      group_count,
      p.group_size.data(),
      compute);
  if (st != CUBLAS_STATUS_SUCCESS) {
    q.pop_back();
    return cuda_from_cublas(st);
  }
  auto ev = cudaEventRecord(p.done, stream);
  if (ev != cudaSuccess) {
    q.pop_back();
    return ev;
  }
  return cudaSuccess;
}
