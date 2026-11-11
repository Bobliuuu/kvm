// The gang barriers alone (include/kvm_gang.cuh): the counter form (`barrier`) against the flag-line form (`barrier_lines`), K rounds of
// two crossings by G co-resident blocks of 256 threads, for every G up to the SM count that the gangs of a fused step use; ns a crossing
// from block 0's globaltimer, the median of 5 launches after one warm-up. Each round also checks that the barrier orders data: every block
// writes a value before the first crossing and reads every other block's after it (a stale read counts an error).
//   nvcc -O3 -std=c++17 -arch=sm_XX -Iinclude bench/gang/barrier_bench.cu -o barrier_bench && ./barrier_bench [K=2000]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kvm_gang.cuh"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

template <int FORM>
__global__ void __launch_bounds__(256, 1) bench(unsigned* ctl, int* data, int K, int G, unsigned long long* ns, int* errors) {
  unsigned epoch = FORM == 0 ? 0u : kvm_gang::lines_epoch(ctl, blockIdx.x);
  const auto cross = [&]() { return FORM == 0 ? kvm_gang::barrier(ctl, G) : kvm_gang::barrier_lines(ctl, blockIdx.x, G, epoch); };
  const unsigned long long t0 = gtimer();
  int bad = 0;
  for (int k = 0; k < K; ++k) {
    if (threadIdx.x == 0) data[blockIdx.x] = k * 1000 + blockIdx.x;          // written before the first crossing of round k ...
    cross();
    for (int b = threadIdx.x; b < G; b += blockDim.x) bad += *(volatile int*)&data[b] < k * 1000;   // ... read after it
    cross();                                                                   // nobody writes round k + 1 before all have read k
  }
  if (bad) atomicAdd(errors, bad);
  if (blockIdx.x == 0 && threadIdx.x == 0) *ns = gtimer() - t0;
}

int main(int argc, char** argv) {
  const int K = argc > 1 ? atoi(argv[1]) : 2000;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  const int sms = p.multiProcessorCount, words = kvm_gang::line_words(sms);
  unsigned* ctl; int *data, *errors; unsigned long long* ns;
  CK(cudaMalloc(&ctl, words * sizeof(unsigned))); CK(cudaMalloc(&data, sms * sizeof(int)));
  CK(cudaMalloc(&errors, sizeof(int))); CK(cudaMalloc(&ns, sizeof(unsigned long long)));
  printf("%s (sm_%d%d, %d SMs): %d crossings a launch (two a round, data checked between), ns a crossing (median of 5 launches)\n", p.name, p.major, p.minor, sms, 2 * K);
  printf("%5s %10s %10s %7s %s\n", "G", "counter", "flag line", "ratio", "errors");
  for (int G : {2, 4, 8, 16, 32, 33, 48, 64, 96, 108, 128}) {
    if (G > sms) continue;
    double med[2]; int errs[2];
    for (int form = 0; form < 2; ++form) {
      CK(cudaMemset(ctl, 0, words * sizeof(unsigned))); CK(cudaMemset(errors, 0, sizeof(int)));
      std::vector<double> t;
      for (int r = 0; r < 6; ++r) {   // the first launch warms the clock and loads the code; it is not counted
        if (form == 0) bench<0><<<G, 256>>>(ctl, data, K, G, ns, errors); else bench<1><<<G, 256>>>(ctl, data, K, G, ns, errors);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long h; CK(cudaMemcpy(&h, ns, sizeof h, cudaMemcpyDeviceToHost));
        if (r) t.push_back((double)h / (2.0 * K));
      }
      std::sort(t.begin(), t.end()); med[form] = t[t.size() / 2];
      CK(cudaMemcpy(&errs[form], errors, sizeof(int), cudaMemcpyDeviceToHost));
    }
    printf("%5d %10.0f %10.0f %7.2f %d/%d\n", G, med[0], med[1], med[0] / med[1], errs[0], errs[1]);
  }
  return 0;
}
