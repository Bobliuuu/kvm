// The gang barriers alone (include/kvm_gang.cuh): the shipped crossing (`cross`), its two forms (the counted barrier and the flag lines)
// and the counter form they replaced, kept here as a reference, so a new GPU re-measures the size rule. K rounds of two crossings by G
// co-resident blocks of 256 threads, for every G up to the SM count that the gangs of a fused step use; ns a crossing from block 0's
// globaltimer, the median of 5 launches after one warm-up. Each round also checks that the barrier orders data: every block writes a
// value before the first crossing and reads every other block's after it (a stale read counts an error).
//   nvcc -O3 -std=c++17 -arch=sm_XX -Iinclude bench/gang/barrier_bench.cu -o barrier_bench && ./barrier_bench [K=2000]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kvm_gang.cuh"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// The reference: the counter form. A member reads the generation, adds itself with an acq_rel atomic (a round trip), and the last arrival
// resets the count and publishes the next generation, which the others poll.
namespace ref {
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) { unsigned o; asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o; }
__device__ __forceinline__ void counter(unsigned* ctl, int members) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned g0 = kvm_gang::ld_acquire(ctl + 1);
    if (atom_add_acq_rel(ctl, 1u) + 1u == (unsigned)members) {
      asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(ctl), "r"(0u) : "memory");
      kvm_gang::st_release(ctl + 1, g0 + 1u);
    } else {
      while (kvm_gang::ld_acquire(ctl + 1) == g0) {}
    }
  }
  __syncthreads();
}
}  // namespace ref

// FORM 0: the shipped crossing; 1: the counted barrier; 2: the flag lines; 3: the counter (the reference). `ctl` starts zeroed.
constexpr int kForms = 4;
constexpr const char* kFormNames[kForms] = {"cross", "counted", "flag lines", "counter"};
template <int FORM>
__global__ void __launch_bounds__(256, 1) bench(unsigned* ctl, int* data, int K, int G, unsigned long long* ns, int* errors) {
  unsigned long long at = FORM == 0 ? kvm_gang::cross_start(ctl, blockIdx.x, G) : FORM == 1 ? kvm_gang::counted_base(ctl, G)
                        : FORM == 2 ? (unsigned long long)kvm_gang::lines_epoch(ctl, blockIdx.x) : 0ull;
  const auto cross = [&]() {
    if (FORM == 0) kvm_gang::cross(ctl, blockIdx.x, G, at);
    else if (FORM == 1) kvm_gang::barrier_counted(ctl, G, at);
    else if (FORM == 2) { unsigned e = (unsigned)at; kvm_gang::barrier_lines(ctl, blockIdx.x, G, e); at = e; }
    else ref::counter(ctl, G);
  };
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
  printf("crossing: the flag lines up to KVM_GANG_LINES_MAX = %d members, the counted barrier past them\n", KVM_GANG_LINES_MAX);
  printf("%5s", "G");
  for (int f = 0; f < kForms; ++f) printf(" %11s", kFormNames[f]);
  printf("  errors (in that order)\n");
  for (int G : {2, 4, 8, 16, 24, 32, 33, 40, 48, 56, 64, 96, 108, 128, 132}) {
    if (G > sms) continue;
    double med[kForms]; int errs[kForms];
    for (int form = 0; form < kForms; ++form) {
      CK(cudaMemset(ctl, 0, words * sizeof(unsigned))); CK(cudaMemset(errors, 0, sizeof(int)));
      std::vector<double> t;
      for (int r = 0; r < 6; ++r) {   // the first launch warms the clock and loads the code; it is not counted
        if (form == 0) bench<0><<<G, 256>>>(ctl, data, K, G, ns, errors);
        else if (form == 1) bench<1><<<G, 256>>>(ctl, data, K, G, ns, errors);
        else if (form == 2) bench<2><<<G, 256>>>(ctl, data, K, G, ns, errors);
        else bench<3><<<G, 256>>>(ctl, data, K, G, ns, errors);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long h; CK(cudaMemcpy(&h, ns, sizeof h, cudaMemcpyDeviceToHost));
        if (r) t.push_back((double)h / (2.0 * K));
      }
      std::sort(t.begin(), t.end()); med[form] = t[t.size() / 2];
      CK(cudaMemcpy(&errs[form], errors, sizeof(int), cudaMemcpyDeviceToHost));
    }
    printf("%5d", G);
    for (int f = 0; f < kForms; ++f) printf(" %11.0f", med[f]);
    printf("  %d/%d/%d/%d\n", errs[0], errs[1], errs[2], errs[3]);
  }
  return 0;
}
