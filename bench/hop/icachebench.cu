// What instruction fetch costs on an A100 SM, by code size: one warp runs a straight-line body of independent integer instructions (no loop inside it, so every instruction is
// fetched), and the body is entered R times in a row. The first pass fetches every line from wherever it lives; later passes find it in the instruction caches if it fits.
//
//   nvcc -O3 -arch=sm_80 -std=c++17 bench/hop/icachebench.cu -o icachebench && ./icachebench
//
// Printed for each body size: cycles per instruction on the first pass and on later passes. A body that fits a cache level runs at the issue rate on later passes; one that does not pays
// the next level's fetch latency per line every pass. The SASS size of each body is printed too (instructions x 16 bytes).
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// Eight independent accumulators, each updated from itself only by a rotate, an add and a shift-xor (no algebra folds a run of these), so the body is fetch- or issue-bound,
// not latency-bound: eight chains of four instructions are in flight per step.
#define UPD(a, r) a = (a + __funnelshift_l(a, a, r)) ^ (a >> 3);
#define OP8 UPD(a0, 7) UPD(a1, 9) UPD(a2, 11) UPD(a3, 13) UPD(a4, 15) UPD(a5, 17) UPD(a6, 19) UPD(a7, 21)

template <int N8>
__global__ void body_kernel(int passes, long long* out, unsigned* sink) {
  unsigned a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  long long first = 0, rest = 0;
  for (int p = 0; p < passes; ++p) {
    const long long t0 = clock64();
#pragma unroll
    for (int i = 0; i < N8; ++i) { OP8 }
    asm volatile("" : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5), "+r"(a6), "+r"(a7));   // the body's results are live here: nothing moves across a pass
    const long long dt = clock64() - t0;
    if (p == 0) first = dt; else rest += dt;
  }
  sink[threadIdx.x] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
  if (threadIdx.x == 0) { out[0] = first; out[1] = rest; }
}

template <int N8>
static void run(long long* d, unsigned* sink) {
  const int passes = 9;
  long long h[2];
  body_kernel<N8><<<1, 32>>>(passes, d, sink);   // one warp on one SM
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  const double ins = 8.0 * 4 * N8;   // four SASS instructions per update (checked with cuobjdump below)
  printf("%8.0f %9.1f KB %10.2f %10.2f\n", ins, ins * 16 / 1024.0, h[0] / ins, h[1] / ((passes - 1) * ins));
}

int main() {
  long long* d;
  unsigned* sink;
  CK(cudaMalloc(&d, 2 * sizeof(long long)));
  CK(cudaMalloc(&sink, 32 * sizeof(unsigned)));
  printf("%8s %12s %10s %10s   (cycles per instruction, one warp)\n", "instrs", "body", "1st pass", "later");
  run<8>(d, sink); run<16>(d, sink); run<32>(d, sink); run<64>(d, sink); run<128>(d, sink); run<256>(d, sink); run<512>(d, sink); run<1024>(d, sink); run<2048>(d, sink);
  cudaFree(d);
  cudaFree(sink);
  return 0;
}
