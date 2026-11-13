// One phase of the one-block step alone (kvm_fgpt1.cuh's own functions), timed in one block of NT threads on gpt-tiny's shapes: does a
// phase cost what its dependent chain predicts when nothing else shares the kernel? The forward's phase clock reads the softmax at 5.1 us,
// the norm at 2.4, the logits and CE at 5.5, the q|k|v GEMM at 1.8.
//   nvcc <kbuild's flags> -Iinclude bench/fused_gpt/fgpt1_iso.cu -o fgpt1_iso && ./fgpt1_iso
#include <cstdio>
#include <cuda_runtime.h>
#include "kvm_fgpt1.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s\n", cudaGetErrorString(e_), #x); return 1; } } while (0)
using namespace kvm_fgpt1;

__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// PHASE: 0 softmax (64 rows of 32 keys), 1 LayerNorm (32 x 32), 2 CE rows (32 x 64), 3 q|k|v GEMM (32 x 96 x 32), 4 a bare __syncthreads
template <int PHASE>
__global__ void __launch_bounds__(NT, 1) iso(int reps, unsigned long long* ns, float* sink) {   // ns[PHASE]; ns has 6 slots
  extern __shared__ float sm[];
  const int tid = threadIdx.x;
  constexpr int N = 32, T = 32, H = 2, D = 32, V = 64, T16 = 32;
  const int ldS = kvm_fgpt_ldk(T16), ldd = kvm_fgpt_ldk(D), ldw = kvm_fgpt_ldmn(3 * D), ldq = kvm_fgpt_ldk(3 * D);
  float* const A = sm;                   // up to 16 K floats of operands and outputs
  float* const B = sm + 16384;
  float* const C = sm + 24576;
  int* const tg = reinterpret_cast<int*>(sm + 32768);
  for (int i = tid; i < 32768; i += NT) sm[i] = 0.001f * (float)((i * 7919) % 1000) - 0.5f;
  for (int i = tid; i < N; i += NT) tg[i] = (i * 13) % V;
  __syncthreads();
  const unsigned long long t0 = gt();
  for (int r = 0; r < reps; ++r) {
    if constexpr (PHASE == 0) softmax_rows(A, ldS, T, T16, H * T, C);
    else if constexpr (PHASE == 1) ln_fwd_p(A, D, B, B + D, C, ldd, B + 2 * D, B + 2 * D + N, N, D, 1e-5f);
    else if constexpr (PHASE == 2) ce_rows(A, N, V, tg, C, C + N);
    else if constexpr (PHASE == 3) gemm<false, false>(A, ldd, B, ldw, N, 3 * D, D, false, [&](int m, int n, float a) { C[m * ldq + n] = a; });
    else if constexpr (PHASE == 5) softmax_rows_t(A, ldS, T, T16, H * T, C);
    __syncthreads();
  }
  const unsigned long long t1 = gt();
  if (tid == 0) { ns[PHASE] = t1 - t0; sink[0] = C[1] + A[3]; }
}

// Both softmaxes on the same scores, once: their probabilities and log-sum-exps compared bit for bit.
__global__ void __launch_bounds__(NT, 1) sm_check(int* bad) {
  extern __shared__ float sm[];
  constexpr int T = 32, H = 2, T16 = 32;
  const int ldS = kvm_fgpt_ldk(T16), n = H * T16 * ldS;
  float* const A = sm; float* const B = sm + n; float* const la = sm + 2 * n; float* const lb = la + H * T;
  for (int i = threadIdx.x; i < n; i += NT) { const float x = 0.37f * (float)((i * 2654435761u) % 1000) / 1000.f - 0.1f; A[i] = x; B[i] = x; }
  __syncthreads();
  softmax_rows(A, ldS, T, T16, H * T, la);
  __syncthreads();
  softmax_rows_t(B, ldS, T, T16, H * T, lb);
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += NT) if (__float_as_uint(A[i]) != __float_as_uint(B[i])) atomicAdd(bad, 1);
  for (int i = threadIdx.x; i < H * T; i += NT) if (__float_as_uint(la[i]) != __float_as_uint(lb[i])) atomicAdd(bad + 1, 1);
}

template <int PHASE>
static int run(const char* name, unsigned long long* dns, float* sink, double inforward) {
  const int reps = 2000, smem = (32768 + 64) * 4;
  CK(cudaFuncSetAttribute(iso<PHASE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  iso<PHASE><<<1, NT, smem>>>(10, dns, sink);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  unsigned long long best = ~0ull;
  for (int k = 0; k < 5; ++k) {
    iso<PHASE><<<1, NT, smem>>>(reps, dns, sink);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long v[6]; CK(cudaMemcpy(v, dns, sizeof v, cudaMemcpyDeviceToHost));
    if (v[PHASE] < best) best = v[PHASE];
  }
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, iso<PHASE>));
  printf("%-28s %7.3f us a pass alone (%3d registers, %4zu B local); in the forward %.2f us\n", name, best / 1000.0 / reps,
         fa.numRegs, (size_t)fa.localSizeBytes, inforward);
  return 0;
}

int main() {
  unsigned long long* dns; float* sink;
  CK(cudaMalloc(&dns, 6 * sizeof(unsigned long long))); CK(cudaMalloc(&sink, 16));
  {
    int* dbad; CK(cudaMalloc(&dbad, 8)); CK(cudaMemset(dbad, 0, 8));
    const int smem = (2 * 2 * 32 * kvm_fgpt_ldk(32) + 4 * 64) * 4;
    CK(cudaFuncSetAttribute(sm_check, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    sm_check<<<1, NT, smem>>>(dbad); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int bad[2]; CK(cudaMemcpy(bad, dbad, 8, cudaMemcpyDeviceToHost));
    printf("softmax, a thread a row against a warp a row: %s (%d probabilities and %d log-sum-exps differ)\n",
           bad[0] == 0 && bad[1] == 0 ? "BITWISE" : "DIFFERS", bad[0], bad[1]);
  }
  if (run<4>("bare __syncthreads", dns, sink, 0.0)) return 1;
  if (run<0>("softmax, 64 rows x 32", dns, sink, 5.12)) return 1;
  if (run<5>("softmax, a thread a row", dns, sink, 5.12)) return 1;
  if (run<1>("LayerNorm, 32 x 32", dns, sink, 2.41)) return 1;
  if (run<2>("CE rows, 32 x 64", dns, sink, 5.53)) return 1;
  if (run<3>("q|k|v GEMM, 32 x 96 x 32", dns, sink, 1.84)) return 1;
  printf("ISO_END\n");
  return 0;
}
