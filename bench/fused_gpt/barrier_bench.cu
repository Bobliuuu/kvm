// The gang barriers, alone: the counter barrier (kvm_gang::barrier) against the fused GPT step's flag line (kvm_fgpt::cross_line), K rounds
// of two crossings by G co-resident blocks of 256 threads; ns a crossing from block 0's globaltimer, the median of 5 launches after one
// warm-up. Each round also checks that the barrier orders data: block b writes a value before the first crossing, and every block reads
// every other block's value after it (a stale read counts an error).
//   nvcc -O3 -std=c++17 -arch=sm_XX -Iinclude bench/fused_gpt/barrier_bench.cu -o barrier_bench && ./barrier_bench [K=2000]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "kvm_fused_gpt.cuh"   // kvm_fgpt::cross_line, the fused GPT step's barrier, and kvm_gang.cuh's counter barrier
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

template <int MODE>
__global__ void __launch_bounds__(256, 1) bench(unsigned* ctl, unsigned* flags, int* data, int K, int G, unsigned long long* ns, int* errors) {
  unsigned epoch = flags[blockIdx.x];   // every flag starts equal (zeroed once), and every block crosses K times a launch
  const unsigned long long t0 = gtimer();
  int bad = 0;
  for (int k = 0; k < K; ++k) {
    if (threadIdx.x == 0) data[blockIdx.x] = k * 1000 + blockIdx.x;             // written before crossing k ...
    if (MODE == 0) kvm_gang::barrier(ctl, G); else kvm_fgpt::cross_line(flags, blockIdx.x, G, epoch, nullptr);
    if (threadIdx.x < (unsigned)G) bad += *(volatile int*)&data[threadIdx.x] < k * 1000;   // ... read after it (k*1000+b, or k+1's)
    if (MODE == 0) kvm_gang::barrier(ctl, G); else kvm_fgpt::cross_line(flags, blockIdx.x, G, epoch, nullptr);  // nobody writes k+1 before all have read k
  }
  if (bad) atomicAdd(errors, bad);
  if (blockIdx.x == 0 && threadIdx.x == 0) *ns = gtimer() - t0;
}

int main(int argc, char** argv) {
  const int K = argc > 1 ? atoi(argv[1]) : 2000;
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  unsigned *ctl, *flags; int *data, *errors; unsigned long long* ns;
  CK(cudaMalloc(&ctl, 2 * sizeof(unsigned))); CK(cudaMalloc(&flags, 128)); CK(cudaMalloc(&data, 64 * sizeof(int)));
  CK(cudaMalloc(&errors, sizeof(int))); CK(cudaMalloc(&ns, sizeof(unsigned long long)));
  printf("%s (sm_%d%d, %d SMs), %d crossings a run (two a round, data checked between), ns a crossing (median of 5 runs)\n", p.name, p.major, p.minor, sms, 2 * K);
  printf("%4s %12s %12s %8s %s\n", "G", "counter", "flag line", "ratio", "errors");
  for (int G : {2, 4, 8, 16, 32}) {
    if (G > sms) break;
    double med[2]; int errs[2];
    for (int mode = 0; mode < 2; ++mode) {
      CK(cudaMemset(ctl, 0, 2 * sizeof(unsigned))); CK(cudaMemset(flags, 0, 128)); CK(cudaMemset(errors, 0, sizeof(int)));
      std::vector<double> t;
      for (int r = 0; r < 6; ++r) {   // the first launch warms the clock and loads the code; it is not counted
        if (mode == 0) bench<0><<<G, 256>>>(ctl, flags, data, K, G, ns, errors); else bench<1><<<G, 256>>>(ctl, flags, data, K, G, ns, errors);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long h; CK(cudaMemcpy(&h, ns, sizeof h, cudaMemcpyDeviceToHost));
        if (r) t.push_back((double)h / (2.0 * K));
      }
      std::sort(t.begin(), t.end()); med[mode] = t[t.size() / 2];
      CK(cudaMemcpy(&errs[mode], errors, sizeof(int), cudaMemcpyDeviceToHost));
    }
    printf("%4d %12.0f %12.0f %8.2f %d/%d\n", G, med[0], med[1], med[0] / med[1], errs[0], errs[1]);
  }
  return 0;
}
