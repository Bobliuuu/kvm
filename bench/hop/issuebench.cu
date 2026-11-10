// How an instruction sequence between two barriers costs on an A100 SM when every warp of a block runs it, against one warp running it while the others wait at the barrier.
//
//   nvcc -O3 -arch=sm_80 -std=c++17 bench/hop/issuebench.cu -o issuebench && ./issuebench
//
// The body is integer work that does not touch memory: its time is instruction issue. For each loop count the kernel runs both modes on every SM (one 256-thread block each) and thread 0
// of block 0 reports the SM cycles between the two barriers. Printed: cycles per loop iteration of each mode, and the ratio. If the block's warps share the SM's issue slots, the
// all-warps mode costs up to 8 times the one-warp mode per iteration.
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void issue_kernel(int n, int one_warp, long long* cycles, unsigned* sink) {
  unsigned x = threadIdx.x * 7u + 1u + blockIdx.x;
  __syncthreads();
  const long long t0 = clock64();
  if (!one_warp || threadIdx.x < 32) {
    for (int i = 0; i < n; ++i) x = (x ^ (x >> 3)) * 2654435761u + (unsigned)i;
  }
  __syncthreads();
  const long long t1 = clock64();
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

int main() {
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int blocks = sms;
  long long* cycles;
  unsigned* sink;
  CK(cudaMalloc(&cycles, blocks * sizeof(long long)));
  CK(cudaMalloc(&sink, (size_t)blocks * 256 * sizeof(unsigned)));
  std::vector<long long> host(blocks);
  printf("%d SMs, one 256-thread block on each\n%8s %12s %12s %8s\n", sms, "iters", "all warps", "one warp", "ratio");
  for (int n : {16, 64, 256, 1024, 4096}) {
    double per[2];
    for (int mode = 0; mode < 2; ++mode) {
      double best = 1e30;
      for (int rep = 0; rep < 5; ++rep) {
        issue_kernel<<<blocks, 256>>>(n, mode, cycles, sink);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host.data(), cycles, blocks * sizeof(long long), cudaMemcpyDeviceToHost));
        std::vector<long long> v(host);
        std::sort(v.begin(), v.end());
        best = std::min(best, (double)v[v.size() / 2]);
      }
      per[mode] = best / n;
    }
    printf("%8d %12.2f %12.2f %8.2f   (cycles per iteration, median block, best of 5 launches)\n", n, per[0], per[1], per[0] / per[1]);
  }
  cudaFree(cycles);
  cudaFree(sink);
  return 0;
}
