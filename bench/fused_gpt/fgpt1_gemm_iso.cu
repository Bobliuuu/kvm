// One GEMM of the one-block step alone (kvm_fused_gpt.cuh's gemm, as kvm_fgpt1.cuh calls it), in one block of NT threads: where does a
// small GEMM's time go, its k-steps or its fixed part (the epilogue, the barrier, the tile setup)? Time against K gives the k-step's cost
// (the slope) and the fixed cost (the intercept), with and without whole tiles (gemm's Full flag: no bounds branch between a lane's
// epilogue elements), and with an epilogue that does real work per element (gelu_grad's tanhf). Two epilogue edits of a
// few instructions an element each moved about 1.2 us in the step; the q|k|v GEMM alone takes about 370 cycles a k-step.
//   nvcc <kbuild's flags> -Iinclude bench/fused_gpt/fgpt1_gemm_iso.cu -o fgpt1_gemm_iso && ./fgpt1_gemm_iso
#include <cstdio>
#include <cuda_runtime.h>
#include "kvm_fgpt1.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s\n", cudaGetErrorString(e_), #x); return 1; } } while (0)
using namespace kvm_fgpt1;

__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

constexpr int OFF_B = 8448, OFF_C = 18944;   // A: M x ldk(K) floats (M 32, K at most 256); B: K x ldmn(N); C: M x ldk(N)
constexpr int SMEM_FLOATS = 22272;

// The shape is the kernel's arguments, as in the step, where M, N and K come from the run's descriptor: a compile-time shape would let the
// compiler fold the bounds checks that the whole-tile flag removes (and the K loop's trip count). EPI: 0 a plain store; 1 gelu_grad of C's
// own element times the product (the backward's dU epilogue before it read gelu'(z) from the forward).
template <bool FULL, bool PREC, int EPI>
__global__ void __launch_bounds__(NT, 1) iso(int reps, int M, int N, int K, unsigned long long* out, float* sink) {   // out: ns, cycles
  extern __shared__ float sm[];
  const int lda = kvm_fgpt_ldk(K), ldb = kvm_fgpt_ldmn(N), ldc = kvm_fgpt_ldk(N);
  float* const A = sm;
  float* const B = sm + OFF_B;
  float* const C = sm + OFF_C;
  for (int i = threadIdx.x; i < SMEM_FLOATS; i += NT) sm[i] = 0.001f * (float)((i * 7919) % 1000) - 0.5f;
  __syncthreads();
  const unsigned long long t0 = gt();
  const long long c0 = clock64();
  for (int r = 0; r < reps; ++r) {
    if constexpr (EPI == 0) gemm<false, false, NT, FULL>(A, lda, B, ldb, M, N, K, PREC, [=](int m, int n, float a) { C[m * ldc + n] = a; });
    else gemm<false, false, NT, FULL>(A, lda, B, ldb, M, N, K, PREC, [=](int m, int n, float a) { C[m * ldc + n] = a * gelu_grad(C[m * ldc + n]); });
  }
  const long long c1 = clock64();
  const unsigned long long t1 = gt();
  if (threadIdx.x == 0) { out[0] = t1 - t0; out[1] = (unsigned long long)(c1 - c0); sink[0] = C[1] + C[ldc + 3]; }
}

template <bool FULL, bool PREC, int EPI>
static int run(int M, int N, int K, unsigned long long* dout, float* sink) {
  const int reps = 2000, smem = SMEM_FLOATS * 4;
  if (M != 32 || K > 256 || (long long)K * kvm_fgpt_ldmn(N) > OFF_C - OFF_B || (long long)M * kvm_fgpt_ldk(N) > SMEM_FLOATS - OFF_C) {
    printf("GEMM %d x %d x %d does not fit the bed's layout\n", M, N, K); return 1;
  }
  auto* k = iso<FULL, PREC, EPI>;
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  k<<<1, NT, smem>>>(50, M, N, K, dout, sink);   // warm: the code into the caches, the clock up
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  unsigned long long best_ns = ~0ull, best_cy = 0;
  for (int t = 0; t < 5; ++t) {
    k<<<1, NT, smem>>>(reps, M, N, K, dout, sink);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long v[2]; CK(cudaMemcpy(v, dout, sizeof v, cudaMemcpyDeviceToHost));
    if (v[0] < best_ns) { best_ns = v[0]; best_cy = v[1]; }
  }
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k));
  printf("GEMM %2d x %2d x %3d  %-5s %-6s %-9s %8.3f us %8.0f cycles a pass at %4.0f MHz (%2d tiles, %2d k-steps; %3d registers)\n", M, N, K,
         FULL ? "full" : "bound", PREC ? "3xTF32" : "1xTF32", EPI ? "gelu_grad" : "store", best_ns / 1000.0 / reps, (double)best_cy / reps,
         1e3 * (double)best_cy / (double)best_ns, (M / 16) * (N / 8), K / 8, fa.numRegs);
  return 0;
}

int main() {
  unsigned long long* dout; float* sink;
  CK(cudaMalloc(&dout, 2 * sizeof(unsigned long long))); CK(cudaMalloc(&sink, 16));
  // the k-step's cost: 32 x 32 at four depths (8 tiles: as dh2, dhf and dh1), bounds-checked and whole-tile
  for (int K : {32, 64, 128, 256}) if (run<false, false, 0>(32, 32, K, dout, sink)) return 1;
  for (int K : {32, 64, 128, 256}) if (run<true, false, 0>(32, 32, K, dout, sink)) return 1;
  // the epilogue's cost: the q|k|v shape (24 tiles), a store against gelu_grad's tanhf, bounds-checked and whole-tile
  if (run<false, false, 0>(32, 96, 32, dout, sink) || run<true, false, 0>(32, 96, 32, dout, sink) || run<false, false, 1>(32, 96, 32, dout, sink) ||
      run<true, false, 1>(32, 96, 32, dout, sink)) return 1;
  // 3xTF32 (attention's products) at one depth, for the precise mode's price
  if (run<true, true, 0>(32, 32, 128, dout, sink)) return 1;
  printf("GEMM_ISO_END\n");
  return 0;
}
