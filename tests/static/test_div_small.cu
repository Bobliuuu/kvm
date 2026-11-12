// div_small (include/kvm_fused_gpt.cuh) against the integer division, on the device it runs on:
//   1. every x in [0, 2^21) for every d in [1, 512];
//   2. grid_loop's own domain: every x in [0, 1024] (its BT and the thread index) for every d in [1, 2^20];
//   3. the range's top edge for d in [1, 2048], every power of two up to 2^30 and its neighbours, and 2^31 - 1: x in [2^21 - 4096, 2^21),
//      and x = d - 1, d and d + 1 below 2^21;
//   4. 2^24 pseudo-random (x, d) pairs, x in [0, 2^21) and d over the whole positive int range.
// x from 2^21 up is outside div_small's contract, so it is not tested.
// Prints the first mismatch and exits 1, else "div_small: PASS" and the counts. nvcc -O3 -arch=sm_XX -std=c++17 -Iinclude tests/static/test_div_small.cu
#include <cstdio>
#include <cstdlib>
#include "kvm_fused_gpt.cuh"

using kvm_fgpt::div_small;
__device__ unsigned long long g_checked;
__device__ int g_bad, g_bad_x, g_bad_d, g_bad_q;

__device__ __forceinline__ void check(int x, int d) {
  const int q = div_small(x, d, __fdividef(1.f, (float)d));
  if (q != x / d && atomicCAS(&g_bad, 0, 1) == 0) { g_bad_x = x; g_bad_d = d; g_bad_q = q; }
}
__global__ void dense(int xmax) {   // every x below xmax for d = blockIdx.y + 1
  const int d = blockIdx.y + 1;
  unsigned long long c = 0;
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < xmax; x += gridDim.x * blockDim.x, ++c) check(x, d);
  atomicAdd(&g_checked, c);
}
__global__ void domain(int dmax) {   // grid_loop's own domain: every x in [0, 1024] for every d in [1, dmax]
  unsigned long long c = 0;
  for (int d = blockIdx.x + 1; d <= dmax; d += gridDim.x)
    for (int x = threadIdx.x; x <= 1024; x += blockDim.x, ++c) check(x, d);
  atomicAdd(&g_checked, c);
}
__global__ void edges(const int* ds, int nd) {
  unsigned long long c = 0;
  for (int k = blockIdx.x; k < nd; k += gridDim.x) {
    const int d = ds[k];
    for (int x = kvm_fgpt::kDivSmallMax - 4096 + threadIdx.x; x < kvm_fgpt::kDivSmallMax; x += blockDim.x, ++c) check(x, d);
    if (threadIdx.x < 3) { const long long x = (long long)d - 1 + threadIdx.x; if (x >= 0 && x < kvm_fgpt::kDivSmallMax) { check((int)x, d); ++c; } }
  }
  atomicAdd(&g_checked, c);
}
__global__ void random_pairs(unsigned long long n, unsigned long long seed) {
  unsigned long long c = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x, ++c) {
    unsigned long long z = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
    z = (z ^ (z >> 31)) * 0xBF58476D1CE4E5B9ull; z ^= z >> 29;
    const int x = (int)(z & (kvm_fgpt::kDivSmallMax - 1)), d = (int)((z >> 32) & 0x7fffffff) | 1;
    check(x >> (int)(i % 21), d >> (int)((i / 21) % 31) | 1);
  }
  atomicAdd(&g_checked, c);
}
#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s\n", cudaGetErrorString(e_), #x); return 2; } } while (0)
int main() {
  int ds[2048 + 3 * 31 + 1], nd = 0;
  for (int d = 1; d <= 2048; ++d) ds[nd++] = d;
  for (int k = 12; k <= 30; ++k) { ds[nd++] = (1 << k) - 1; ds[nd++] = 1 << k; ds[nd++] = (1 << k) + 1; }
  ds[nd++] = 0x7fffffff;
  int* dd; CK(cudaMalloc(&dd, sizeof(int) * nd)); CK(cudaMemcpy(dd, ds, sizeof(int) * nd, cudaMemcpyHostToDevice));
  dense<<<dim3(64, 512), 256>>>(kvm_fgpt::kDivSmallMax);
  domain<<<4096, 256>>>(1 << 20);
  edges<<<512, 256>>>(dd, nd);
  random_pairs<<<1024, 256>>>(1ull << 24, 0x5DEECE66Dull);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  int bad, bx, bdv, bq; unsigned long long checked;
  CK(cudaMemcpyFromSymbol(&bad, g_bad, sizeof bad)); CK(cudaMemcpyFromSymbol(&checked, g_checked, sizeof checked));
  if (bad) {
    CK(cudaMemcpyFromSymbol(&bx, g_bad_x, sizeof bx)); CK(cudaMemcpyFromSymbol(&bdv, g_bad_d, sizeof bdv)); CK(cudaMemcpyFromSymbol(&bq, g_bad_q, sizeof bq));
    printf("div_small: FAIL at x %d, d %d: %d, not %d (%llu checked)\n", bx, bdv, bq, bx / bdv, checked);
    return 1;
  }
  printf("div_small: PASS, %llu pairs checked (every x below 2^21 for d up to 512, grid_loop's domain, the top edge, 2^24 random pairs)\n", checked);
  return 0;
}
