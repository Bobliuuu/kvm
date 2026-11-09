// The dependent-hop floor on this GPU: how fast can one block tell another "your input is ready"?
//   nvcc -O3 -arch=sm_80 -std=c++17 bench/hop/hopbench.cu -o hopbench && ./hopbench [--quick] > out.jsonl
// Each line of the output is one JSON record ("case": ..., hop_ns = half a measured round trip between two blocks through global memory).
// A round trip is the cleanest dependent-hop measurement: the producer's signal store, the L2, the consumer's detecting load, then
// the same hop back; half of it is one hop including the producer's signal and the consumer's detection.
// Cases:
//   ping_rel_acq / ping_fence_rel_acq / ping_atom_acq / ping_rel_cg / ping_rel_cv : one hop through global memory, per signal/poll primitive
//   ping_*_pollers<Q>: the same hop with Q idle "worker" blocks polling (the executor's load on the chip)
//   ping_data_rel_acq / ping_data_fence: the producer writes a data word, fences, signals; the consumer verifies the word (a checked dependency)
//   ping_join: the join path (fence, indegree atomic, release store), the executor's indegree-counter hop
//   barrier: one block's __syncthreads(), 256 threads (the cheapest "hop" that stays inside a block)
//   graph_edge: a chain of N one-store kernels in a CUDA graph (the hardware front end's edge, the conventional reference)
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); exit(1); } } while (0)

__device__ __forceinline__ unsigned long long hb_now() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int hb_smid() { unsigned r; asm("mov.u32 %0, %%smid;" : "=r"(r)); return (int)r; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ int atom_add_ar(int* p, int v) { int r; asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],%2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ void fence_acq() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }

// A signal and a poll on one 128-byte line of its own.
enum Prim { PRIM_REL_ACQ = 0, PRIM_FENCE_REL_ACQ = 1, PRIM_ATOM_ACQ = 2, PRIM_REL_CG = 3, PRIM_REL_CV = 4, PRIM_DATA_REL_ACQ = 5, PRIM_DATA_FENCE = 6, PRIM_JOIN = 7 };

__device__ __forceinline__ int poll_acq(const int* p, int want, int prim) {
  for (;;) {
    int v;
    if (prim == PRIM_REL_CG) { v = __ldcg(p); }
    else if (prim == PRIM_REL_CV) { v = __ldcv(p); }
    else v = ld_acq(p);
    if (v >= want) { if (prim == PRIM_REL_CG || prim == PRIM_REL_CV) fence_acq(); return v; }
  }
}

__device__ __forceinline__ void do_signal(int* p, int v, int prim, int* data, int* join_counter) {
  switch (prim) {
    case PRIM_REL_ACQ: case PRIM_REL_CG: case PRIM_REL_CV: st_rel(p, v); break;
    case PRIM_FENCE_REL_ACQ: __threadfence(); st_rel(p, v); break;
    case PRIM_ATOM_ACQ: atom_add_ar(p, 1); break;
    case PRIM_DATA_REL_ACQ: *(volatile int*)data = v; st_rel(p, v); break;
    case PRIM_DATA_FENCE: *(volatile int*)data = v; __threadfence(); st_rel(p, v); break;
    case PRIM_JOIN:   // the executor's completion: fence the op's stores, decrement the successor's indegree, release the mailbox if it hit zero
      *(volatile int*)data = v; __threadfence();
      if (atom_add_ar(join_counter, -1) == 1) st_rel(p, v);
      break;
  }
}

struct alignas(128) HbLine { int v; int d; int join; int pad[29]; };
static_assert(sizeof(HbLine) == 128, "one line");

struct PairStat { long long rt_ns, rt_cycles; int smid_a, smid_b, mismatches, pad; };

// Pair p = blocks 2p (driver) and 2p+1 (echo). Only thread 0 acts on each side; every other thread leaves at once (the block is a warp, 32 threads).
__global__ void __launch_bounds__(32) pingpong_kernel(HbLine* lines, PairStat* out, int pairs, int rounds, int warmup, int prim, int* hot, int* stop) {
  const int b = blockIdx.x;
  if (b >= 2 * pairs) {       // a background "idle worker": polls one shared word at full rate until the pairs are done
    if (threadIdx.x == 0) { while (!ld_acq(stop)) (void)ld_acq(hot); }
    return;
  }
  if (threadIdx.x != 0) return;
  const int p = b / 2; const bool driver = (b & 1) == 0;
  HbLine* a = lines + 2 * p; HbLine* e = lines + 2 * p + 1;
  if (driver) {
    unsigned long long t0 = 0; long long c0 = 0; int bad = 0;
    for (int i = 1; i <= warmup + rounds; ++i) {
      if (prim == PRIM_JOIN) *(volatile int*)&e->join = 1;       // the echo's "indegree": one edge per hop
      if (i == warmup + 1) { t0 = hb_now(); c0 = clock64(); }
      do_signal(&a->v, i, prim, &a->d, &e->join);
      poll_acq(&e->v, i, prim);
      if (prim == PRIM_DATA_REL_ACQ || prim == PRIM_DATA_FENCE || prim == PRIM_JOIN) { if (*(const volatile int*)&e->d != i) ++bad; }
    }
    PairStat* s = out + p;
    s->rt_ns = (long long)(hb_now() - t0); s->rt_cycles = clock64() - c0; s->smid_a = hb_smid(); s->mismatches = bad; s->pad = rounds;
    if (atomicAdd(stop + 1, 1) == pairs - 1) st_rel(stop, 1);      // all pairs done: let the pollers leave
  } else {
    int bad = 0;
    for (int i = 1; i <= warmup + rounds; ++i) {
      poll_acq(&a->v, i, prim);
      if (prim == PRIM_DATA_REL_ACQ || prim == PRIM_DATA_FENCE || prim == PRIM_JOIN) { if (*(const volatile int*)&a->d != i) ++bad; }
      if (prim == PRIM_JOIN) *(volatile int*)&a->join = 1;        // reset the driver's join for the echo's signal... each direction has its own counter
      do_signal(&e->v, i, prim, &e->d, &a->join);
    }
    out[p].smid_b = hb_smid(); out[p].mismatches += bad;
  }
}

__global__ void __launch_bounds__(256) barrier_kernel(long long* out, int n) {
  long long t0 = clock64();
  for (int i = 0; i < n; ++i) __syncthreads();
  if (threadIdx.x == 0) { out[0] = clock64() - t0; }
}

__global__ void noop_kernel(int* c) { atomicAdd(c, 1); }

// Keep the clocks up before timing: short kernels alone leave the GPU at its idle clock (a ping-pong at 210 MHz is not the floor).
__global__ void __launch_bounds__(256) warm_kernel(float* x, unsigned long long ms) {
  const unsigned long long t0 = hb_now();
  float a = 1.0f + (float)threadIdx.x;
  while (hb_now() - t0 < ms * 1000000ull) for (int k = 0; k < 256; ++k) a = a * 1.0000001f + 0.5f;
  if (a > 0) x[blockIdx.x * 256 + threadIdx.x] = a;
}
static void warm(unsigned long long ms) { static float* x = nullptr; if (!x) CK(cudaMalloc(&x, 1 << 20)); warm_kernel<<<108, 256>>>(x, ms); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }

static void json_str(FILE* f, const char* k, const char* v, bool comma = true) { fprintf(f, "\"%s\":\"%s\"%s", k, v, comma ? "," : ""); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; }

struct Case { const char* name; int pairs, pollers, prim, rounds; };

int main(int argc, char** argv) {
  bool quick = argc > 1 && !strcmp(argv[1], "--quick");
  const int rounds = quick ? 2000 : 20000, warmup = 128;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  fprintf(stderr, "device: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
  HbLine* lines; PairStat* out; int* hot; int* stop;
  CK(cudaMalloc(&lines, sizeof(HbLine) * 512)); CK(cudaMalloc(&out, sizeof(PairStat) * 256)); CK(cudaMalloc(&hot, 256)); CK(cudaMalloc(&stop, 16));
  std::vector<Case> cases = {
    {"ping_rel_acq", 1, 0, PRIM_REL_ACQ, rounds}, {"ping_fence_rel_acq", 1, 0, PRIM_FENCE_REL_ACQ, rounds}, {"ping_atom_acq", 1, 0, PRIM_ATOM_ACQ, rounds},
    {"ping_rel_cg", 1, 0, PRIM_REL_CG, rounds}, {"ping_rel_cv", 1, 0, PRIM_REL_CV, rounds},
    {"ping_data_rel_acq", 1, 0, PRIM_DATA_REL_ACQ, rounds}, {"ping_data_fence", 1, 0, PRIM_DATA_FENCE, rounds}, {"ping_join", 1, 0, PRIM_JOIN, rounds},
    {"ping_rel_acq_8pairs", 8, 0, PRIM_REL_ACQ, rounds}, {"ping_rel_acq_32pairs", 32, 0, PRIM_REL_ACQ, rounds},
    {"ping_rel_acq_pollers104", 2, 104, PRIM_REL_ACQ, rounds}, {"ping_rel_cg_pollers104", 2, 104, PRIM_REL_CG, rounds}, {"ping_join_pollers104", 2, 104, PRIM_JOIN, rounds},
  };
  warm(1500);
  for (const Case& c : cases) {
    std::vector<double> hops, cyc; int same = 0, bad = 0, kept = 0;
    for (int rep = 0; rep < (quick ? 1 : 5); ++rep) {
      warm(100);
      CK(cudaMemset(lines, 0, sizeof(HbLine) * 512)); CK(cudaMemset(out, 0, sizeof(PairStat) * 256)); CK(cudaMemset(hot, 0, 256)); CK(cudaMemset(stop, 0, 16));
      pingpong_kernel<<<2 * c.pairs + c.pollers, 32>>>(lines, out, c.pairs, c.rounds, warmup, c.prim, hot, stop);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      std::vector<PairStat> h(c.pairs); CK(cudaMemcpy(h.data(), out, sizeof(PairStat) * c.pairs, cudaMemcpyDeviceToHost));
      for (auto& s : h) { hops.push_back(s.rt_ns / (2.0 * s.pad)); cyc.push_back(s.rt_cycles / (2.0 * s.pad)); same += s.smid_a == s.smid_b; bad += s.mismatches; ++kept; }
    }
    double mn = *std::min_element(hops.begin(), hops.end());
    printf("{"); json_str(stdout, "case", c.name); printf("\"pairs\":%d,\"pollers\":%d,\"reps\":%d,\"rounds\":%d,\"hop_ns_median\":%.1f,\"hop_ns_min\":%.1f,\"hop_cycles_median\":%.1f,\"implied_mhz\":%.0f,\"same_sm_pairs\":%d,\"pairs_total\":%d,\"mismatches\":%d}\n",
           c.pairs, c.pollers, quick ? 1 : 5, c.rounds, median(hops), mn, median(cyc), median(hops) > 0 ? 1000.0 * median(cyc) / median(hops) : 0.0, same, kept, bad);
    fflush(stdout);
  }
  // Barrier: one block of 256 threads on one SM, N barriers.
  long long* bo; CK(cudaMalloc(&bo, 8));
  warm(200);
  for (int n : {1000}) {
    std::vector<double> v;
    for (int rep = 0; rep < 5; ++rep) { barrier_kernel<<<1, 256>>>(bo, n); CK(cudaDeviceSynchronize()); long long x; CK(cudaMemcpy(&x, bo, 8, cudaMemcpyDeviceToHost)); v.push_back((double)x / n); }
    printf("{\"case\":\"barrier_256_threads\",\"n\":%d,\"cycles_median\":%.1f}\n", n, median(v));
  }
  // Graph edge: a chain of noop kernels in a CUDA graph (Jerry's graph-edge reference: about 1.3 us).
  for (int N : {1024, 4096}) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    int* ctr; CK(cudaMalloc(&ctr, 64)); CK(cudaMemset(ctr, 0, 64));
    cudaGraph_t g; CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < N; ++i) noop_kernel<<<1, 1, 0, s>>>(ctr);
    CK(cudaStreamEndCapture(s, &g)); cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge, g, 0));
    std::vector<double> v;
    for (int rep = 0; rep < 5; ++rep) {
      auto t0 = std::chrono::high_resolution_clock::now();
      CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
      v.push_back(std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    int got; CK(cudaMemcpy(&got, ctr, 4, cudaMemcpyDeviceToHost));
    printf("{\"case\":\"graph_edge\",\"nodes\":%d,\"launch_to_sync_us\":%.2f,\"edge_us\":%.3f,\"counter\":%d,\"expected\":%d}\n", N, median(v), median(v) / N, got, N * 5);
    CK(cudaGraphDestroy(g)); CK(cudaGraphExecDestroy(ge)); CK(cudaStreamDestroy(s)); CK(cudaFree(ctr));
  }
  return 0;
}
