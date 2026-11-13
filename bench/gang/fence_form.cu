// One arrival-fence form of the shipped gang barriers, for fence_bench.cu: built once per KVM_GANG_FENCE value (0, 1, 2), each
// build in its own namespace FENCE_NS, so one process launches the three forms' barrier_counted and barrier_lines (include/kvm_gang.cuh),
// called as the fused steps call them, with the run's abort word.
#include "kvm_gang.cuh"
#ifndef FENCE_NS
#error "build with -DKVM_GANG_FENCE=F -DFENCE_NS=fF, F in 0, 1, 2 (fence_bench.cu's header)"
#endif

namespace FENCE_NS {
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// K rounds of two crossings by G co-resident blocks; ns from block 0's globaltimer. Each round checks that the barrier orders data:
// every block writes a value before the first crossing and reads every other block's after it (a stale read counts an error).
template <bool Lines>
__global__ void __launch_bounds__(256, 1) bench(unsigned* ctl, int* data, int K, int G, const int* abort, unsigned long long* ns, int* errors) {
  unsigned long long target = Lines ? 0ull : kvm_gang::counted_base(ctl, G);
  unsigned epoch = Lines ? kvm_gang::lines_epoch(ctl, blockIdx.x) : 0u;
  const auto cross = [&]() {
    if constexpr (Lines) return kvm_gang::barrier_lines(ctl, blockIdx.x, G, epoch, abort);
    else return kvm_gang::barrier_counted(ctl, G, target, abort);
  };
  const unsigned long long t0 = gtimer();
  int bad = 0;
  for (int k = 0; k < K; ++k) {
    if (threadIdx.x == 0) data[blockIdx.x] = k * 1000 + blockIdx.x;          // written before the first crossing of round k ...
    if (!cross()) break;
    for (int b = threadIdx.x; b < G; b += blockDim.x) bad += *(volatile int*)&data[b] < k * 1000;   // ... read after it
    if (!cross()) break;                                                       // nobody writes round k + 1 before all have read k
  }
  if (bad) atomicAdd(errors, bad);
  if (blockIdx.x == 0 && threadIdx.x == 0) *ns = gtimer() - t0;
}

int fence() { return KVM_GANG_FENCE; }
void launch(bool lines, unsigned* ctl, int* data, int K, int G, const int* abort, unsigned long long* ns, int* errors) {
  if (lines) bench<true><<<G, 256>>>(ctl, data, K, G, abort, ns, errors);
  else bench<false><<<G, 256>>>(ctl, data, K, G, abort, ns, errors);
}
}  // namespace FENCE_NS
