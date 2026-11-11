// fused_mlp_bed: can a whole training step of a small MLP run as one block-resident program, activations in shared memory and one reduction hop per step?
// A standalone bed, no executor around it. Shapes are run-time: embedding V x E, L hidden layers (GELU) of widths H[..], head H[L-1] x V, softmax CE, SGD.
//   nvcc -O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off fused_mlp_bed.cu -o fused_mlp_bed   (seconds, not minutes)
//   ./fused_mlp_bed --steps 200 --hidden 72,72 [--tokens 96] [--vocab 100] [--embed 48] [--slice 32] [--check 8]
//   ./fused_mlp_bed --bench-gemm
//
// Design. The batch's tokens are cut into row slices of R tokens, one CTA per slice for the whole run. A CTA copies the weights from global memory into shared memory at
// the start of each step and runs forward and backward on its R rows with every activation resident in shared memory (GEMMs by TF32 mma over shared-memory fragments):
// the step's global-memory round trips are the weight load, the token/embedding gather and the per-slice partial gradients written back. The partial gradients are then
// reduced across slices (one hop on polled counters) and SGD updates the weights (fixed reduction order: the result is deterministic, independent of slice timing).
// Everything below is one persistent launch. The layout of shared memory (offsets of weights and activations for the shape) is computed on the host.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <type_traits>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "kvm_fused_mlp.h"
#include "kvm_fused_mlp.cuh"

constexpr int MAXL = KVM_FMLP_MAXL;
constexpr float LR = 0.05f;
__host__ __device__ inline int rnd16(int x) { return (x + 15) & ~15; }
__host__ __device__ inline int rnd8(int x) { return (x + 7) & ~7; }
constexpr int MAXSLICE = 8, MAXSM = 40 * 1024;     // (floats: 160 KB of the A100's 164 KB)
extern __shared__ __align__(128) float sm[];

__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }

constexpr int NPH = 16;
struct Globals {
  const KvmFusedDesc* d;                 // the run's descriptor (device memory, as the op reads it)
  int bt, nsl;
  int* ids, *tgt;                        // [steps][BT]
  float* loss;                           // [steps * nsl]: each slice's sum of row losses
  int* progress;                         // [nsl * 32] (one 128-byte line per slice)
  long long* ph;                         // [NPH] accumulated cycles per phase (slice 0, thread 0)
  int steps;
};

// The fused program of all slices: one CTA per slice for the whole run, each step the op bodies plus the executor's hops (partial publish, wait for every slice, the owner's share of the updates, publish, wait).
__global__ void __launch_bounds__(256) fused_kernel(const Globals g) {
#ifdef __CUDA_ARCH__
  const int sc = blockIdx.x, nsl = g.nsl, bt = g.bt;
  __shared__ long long t_ph;
  const bool ph0 = sc == 0;
  kvm_fmlp::PhaseTimer ph{ph0 ? g.ph : nullptr, &t_ph};
  int* pc = g.progress;
  for (int st = 0; st < g.steps; ++st) {
    kvm_fmlp::prefetch_desc(g.d, sm);
    kvm_fmlp::slice_run(sc, g.ids + (size_t)st * bt, g.tgt + (size_t)st * bt, g.loss + (size_t)st * nsl + sc, sm, ph, kvm_fmlp::NoFail{});
    __threadfence();
    __syncthreads(); ph(11);
    if (threadIdx.x == 0) st_rel(pc + (size_t)sc * 32, 2 * st + 1);
    if (threadIdx.x < nsl) { while (ld_acq(pc + (size_t)threadIdx.x * 32) < 2 * st + 1) {} }
    __syncthreads(); ph(12);
    kvm_fmlp::prefetch_desc(g.d, sm);
    kvm_fmlp::sgd_run(sc, nullptr, sm);
    __threadfence();
    __syncthreads(); ph(13);
    if (threadIdx.x == 0) st_rel(pc + (size_t)sc * 32, 2 * st + 2);
    if (threadIdx.x < nsl) { while (ld_acq(pc + (size_t)threadIdx.x * 32) < 2 * st + 2) {} }
    __syncthreads(); ph(14);
  }
#endif  // __CUDA_ARCH__
}

// Micro-bench of the GEMM helper: thread 0 times `iters` calls per shape; A, B are shared memory with random contents.
// kind: 0 NN with MAXI 8 (the original cases), 1 NN MAXI 4 (the slice's forward and head), 2 NT MAXI 4 (B stored [n][k]: the activation gradients), 3 TN MAXI 8 (A stored [k][m]: the weight gradients),
// 4 and 5: NN and NT with one row block fixed at compile time (gemm_r at R = 16)
struct GemmCase { int M, N, K, lda, ldb, kind; };
__global__ void __launch_bounds__(256) gemm_bench_kernel(long long* out, const float* src, int iters, GemmCase c) {
#ifdef __CUDA_ARCH__
  const int lda = c.lda, ldb = c.ldb;
  const int asz = (c.kind == 3 ? c.K : c.M) * lda, bsz = (c.kind == 2 || c.kind == 5 ? c.N : c.K) * ldb;
  float* A = sm, *B = sm + asz + 16;
  float* C = B + bsz + 16;
  for (int i = threadIdx.x; i < asz + bsz; i += 256) (i < asz ? A[i] : B[i - asz]) = src[i & 4095];
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    if (c.kind == 0) kvm_fmlp::gemm<false, false, 8>(A, lda, B, ldb, C, ldb, c.M, c.N, c.K);
    else if (c.kind == 1) kvm_fmlp::gemm<false, false, 4>(A, lda, B, ldb, C, ldb, c.M, c.N, c.K);
    else if (c.kind == 2) kvm_fmlp::gemm<false, true, 4>(A, lda, B, ldb, C, ldb, c.M, c.N, c.K);
    else if (c.kind == 3) kvm_fmlp::gemm<true, false, 8>(A, lda, B, ldb, C, ldb, c.M, c.N, c.K);
    else if (c.kind == 4) kvm_fmlp::gemm<false, false, 1, false, 1>(A, lda, B, ldb, C, ldb, 16, c.N, c.K);
    else kvm_fmlp::gemm<false, true, 1, false, 1>(A, lda, B, ldb, C, ldb, 16, c.N, c.K);
    __syncthreads();
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) *out = t1 - t0;
#endif
}

// ---------------- host ----------------
struct Shape {   // the model the bed trains (host, for the CPU reference); the device reads KvmFusedDesc
  int V, E, L, bt, nsl, H[MAXL];
  int K(int l) const { return l == 0 ? E : H[l - 1]; }
};
struct Model { Shape sh; std::vector<float> we, wh, w[MAXL], b[MAXL]; };

// CPU reference of one training step in float32 (same algorithm, no TF32): checks the device's arithmetic, not its speed.
static float ref_step(Model& m, const int* ids, const int* tgt) {
  const Shape& s = m.sh; const int BT = s.bt, V = s.V, E = s.E, L = s.L;
  std::vector<std::vector<float>> Z(L), H(L);
  std::vector<float> X(BT * E), Y(BT * V), dY(BT * V);
  auto gelu = [](float x) { const float k0 = 0.7978845608028654f; return 0.5f * x * (1.f + std::tanh(k0 * (x + 0.044715f * x * x * x))); };
  auto gelug = [](float x) { const float k0 = 0.7978845608028654f; float t = std::tanh(k0 * (x + 0.044715f * x * x * x)); return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * k0 * (1.f + 3.f * 0.044715f * x * x); };
  for (int r = 0; r < BT; ++r) for (int c = 0; c < E; ++c) X[r * E + c] = m.we[ids[r] * E + c];
  for (int l = 0; l < L; ++l) {
    Z[l].resize(BT * s.H[l]); H[l].resize(BT * s.H[l]);
    const std::vector<float>& A = l == 0 ? X : H[l - 1]; const int K = s.K(l);
    for (int r = 0; r < BT; ++r) for (int c = 0; c < s.H[l]; ++c) { float z = 0; for (int k = 0; k < K; ++k) z += A[r * K + k] * m.w[l][k * s.H[l] + c]; z += m.b[l][c]; Z[l][r * s.H[l] + c] = z; H[l][r * s.H[l] + c] = gelu(z); }
  }
  { const int K = s.H[L - 1]; for (int r = 0; r < BT; ++r) for (int c = 0; c < V; ++c) { float z = 0; for (int k = 0; k < K; ++k) z += H[L - 1][r * K + k] * m.wh[k * V + c]; Y[r * V + c] = z; } }
  double loss = 0;
  for (int r = 0; r < BT; ++r) {
    float mx = -INFINITY, S = 0;
    for (int c = 0; c < V; ++c) mx = std::max(mx, Y[r * V + c]);
    for (int c = 0; c < V; ++c) S += std::exp(Y[r * V + c] - mx);
    loss += std::log(S) + mx - Y[r * V + tgt[r]];
    for (int c = 0; c < V; ++c) dY[r * V + c] = (std::exp(Y[r * V + c] - mx) / S - (c == tgt[r] ? 1.f : 0.f)) / BT;
  }
  { const int K = s.H[L - 1];
    std::vector<float> gh(K * V, 0.f);
    for (int k = 0; k < BT; ++k) for (int i = 0; i < K; ++i) for (int j = 0; j < V; ++j) gh[i * V + j] += H[L - 1][k * K + i] * dY[k * V + j];
    std::vector<float> D(BT * K);
    for (int r = 0; r < BT; ++r) for (int c = 0; c < K; ++c) { float z = 0; for (int k = 0; k < V; ++k) z += dY[r * V + k] * m.wh[c * V + k]; D[r * K + c] = z; }
    for (int l = L - 1; l >= 0; --l) {
      const int Hl = s.H[l], KL = s.K(l);
      std::vector<float> gb(Hl, 0.f), gw(KL * Hl, 0.f), Dn(BT * KL);
      for (int r = 0; r < BT; ++r) for (int c = 0; c < Hl; ++c) { D[r * Hl + c] *= gelug(Z[l][r * Hl + c]); gb[c] += D[r * Hl + c]; }
      const std::vector<float>& Ap = l > 0 ? H[l - 1] : X;
      for (int k = 0; k < BT; ++k) for (int i = 0; i < KL; ++i) for (int j = 0; j < Hl; ++j) gw[i * Hl + j] += Ap[k * KL + i] * D[k * Hl + j];
      for (int r = 0; r < BT; ++r) for (int c = 0; c < KL; ++c) { float z = 0; for (int k = 0; k < Hl; ++k) z += D[r * Hl + k] * m.w[l][c * Hl + k]; Dn[r * KL + c] = z; }
      if (l == 0) { std::vector<float> ge(V * E, 0.f); for (int r = 0; r < BT; ++r) for (int c = 0; c < E; ++c) ge[ids[r] * E + c] += Dn[r * E + c]; for (int i = 0; i < V * E; ++i) m.we[i] -= LR * ge[i]; }
      for (int i = 0; i < KL * Hl; ++i) m.w[l][i] -= LR * gw[i];
      for (int i = 0; i < Hl; ++i) m.b[l][i] -= LR * gb[i];
      D = Dn;
    }
    for (int i = 0; i < K * V; ++i) m.wh[i] -= LR * gh[i];
  }
  return (float)(loss / BT);
}

int main(int argc, char** argv) {
  int steps = 200, check = 0, reps = 3, bt = 96, V = 100, E = 48, slice = 32; std::vector<int> hidden = {72, 72};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--bench-gemm") {
      int iters = 4000; float* src; long long* out; CK(cudaMalloc(&src, 4096 * 4)); CK(cudaMalloc(&out, 8));
      std::vector<float> s(4096); { std::mt19937 r(7); std::uniform_real_distribution<float> d(-0.3f, 0.3f); for (auto& x : s) x = d(r); }
      CK(cudaMemcpy(src, s.data(), 4096 * 4, cudaMemcpyHostToDevice));
      CK(cudaFuncSetAttribute(gemm_bench_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXSM * 4));
      // the original NN cases, then the slice's own shapes at R = 16 (h72, V 100): forward/head (NN, MAXI 4), activation gradients (NT), weight gradients (TN)
      GemmCase cases[] = {{16, 16, 8, 0, 0, 0}, {16, 16, 32, 0, 0, 0}, {16, 16, 72, 0, 0, 0}, {32, 16, 72, 0, 0, 0}, {32, 80, 8, 0, 0, 0}, {32, 80, 72, 0, 0, 0}, {32, 112, 72, 0, 0, 0},
                          {80, 112, 32, 0, 0, 0}, {128, 128, 64, 0, 0, 0},
                          {16, 80, 48, 0, 0, 1}, {16, 80, 72, 0, 0, 1}, {16, 112, 72, 0, 0, 1}, {16, 80, 72, 0, 0, 2}, {16, 80, 104, 0, 0, 2}, {80, 80, 16, 0, 0, 3}, {80, 112, 16, 0, 0, 3},
                          {16, 80, 48, 0, 0, 4}, {16, 80, 72, 0, 0, 4}, {16, 80, 72, 0, 0, 5}, {16, 80, 104, 0, 0, 5}};
      for (GemmCase& c : cases) {
        // pitches: A [M][K] (TN: [K][M]), B [K][N] (NT: [N][K]), each congruent to 4 mod 16 like the slice's
        c.lda = kvm_fmlp_pitch(c.kind == 3 ? c.M : c.K); c.ldb = kvm_fmlp_pitch(c.kind == 2 || c.kind == 5 ? c.K : c.N);
        gemm_bench_kernel<<<1, 256, MAXSM * 4>>>(out, src, iters, c); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        long long v; CK(cudaMemcpy(&v, out, 8, cudaMemcpyDeviceToHost));
        static const char* kinds[] = {"NN8", "NN4", "NT4", "TN8", "NN1f", "NT1f"};
        printf("  %s M=%3d N=%3d K=%3d: %8.1f cycles/call  (%.1f FLOP/cycle)\n", kinds[c.kind], c.M, c.N, c.K, (double)v / iters, (double)c.M * c.N * c.K * 2 / ((double)v / iters));
      }
      return 0;
    }
    else if (a == "--steps") steps = atoi(argv[++i]); else if (a == "--check") check = atoi(argv[++i]); else if (a == "--reps") reps = atoi(argv[++i]);
    else if (a == "--tokens") bt = atoi(argv[++i]); else if (a == "--vocab") V = atoi(argv[++i]); else if (a == "--embed") E = atoi(argv[++i]); else if (a == "--slice") slice = atoi(argv[++i]);
    else if (a == "--hidden") { hidden.clear(); std::stringstream ss(argv[++i]); std::string tok; while (std::getline(ss, tok, ',')) hidden.push_back(atoi(tok.c_str())); }
  }
  Shape s{}; s.V = V; s.E = E; s.L = (int)hidden.size(); s.bt = bt; s.nsl = bt / slice;
  if (s.bt % slice || s.nsl > MAXSLICE || s.L > MAXL || (slice & 15) || slice > KVM_FMLP_MAX_SLICE_ROWS || (E & 7) || (V & 3)) { fprintf(stderr, "shape constraints: tokens a multiple of --slice, at most %d slices and %d layers, slice a multiple of 16 up to %d, E a multiple of 8, V of 4\n", MAXSLICE, MAXL, KVM_FMLP_MAX_SLICE_ROWS); return 2; }
  KvmFusedDesc d{}; d.R = slice; d.V = V; d.E = E; d.L = s.L; d.nsl = s.nsl; d.T = bt; d.lr = LR; d.groups = s.nsl;   // the bed: one owner per slice
  for (int l = 0; l < s.L; ++l) { if (hidden[l] & 7) { fprintf(stderr, "hidden widths must be multiples of 8\n"); return 2; } s.H[l] = d.H[l] = hidden[l]; }
  // the budget as the executor sizes it: the kernel's opt-in maximum less its own static shared memory (kept activations fit when they fit there)
  int budget = 0;
  { int dev; CK(cudaGetDevice(&dev)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev)); cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, fused_kernel));
    budget = std::min(MAXSM, (int)(((long long)p.sharedMemPerBlockOptin - (long long)fa.sharedSizeBytes) / 4) & ~15); }
  const int total_floats = kvm_fmlp_layout(d, budget);
  if (total_floats > MAXSM) { fprintf(stderr, "shape needs %d floats of shared memory (limit %d): use a smaller --slice\n", total_floats, MAXSM); return 2; }
  std::mt19937 rng(42);
  auto mk = [&](size_t n, float sc_) { std::vector<float> v(n); std::uniform_real_distribution<float> d(-sc_, sc_); for (auto& x : v) x = d(rng); return v; };
  Model ref; ref.sh = s; ref.we = mk((size_t)V * E, 0.3f); ref.wh = mk((size_t)s.H[s.L - 1] * V, 0.12f);
  for (int l = 0; l < s.L; ++l) { ref.w[l] = mk((size_t)s.K(l) * s.H[l], 0.13f); ref.b[l] = mk(s.H[l], 0.02f); }
  std::vector<int> ids((size_t)steps * bt), tgt((size_t)steps * bt);
  { std::uniform_int_distribution<int> d(0, V - 1); for (auto& x : ids) x = d(rng); for (auto& x : tgt) x = d(rng); }
  Globals g{}; g.bt = bt; g.nsl = s.nsl; g.steps = steps;
  auto dm = [](size_t n) { float* p; CK(cudaMalloc(&p, n * 4)); CK(cudaMemset(p, 0, n * 4)); return p; };
  d.we = dm((size_t)V * E); d.ge = dm((size_t)s.nsl * V * E); d.wh = dm((size_t)s.H[s.L - 1] * V); d.loss_part = dm(s.nsl);
  d.ghld = kvm_fmlp_pad16(V); d.gh = dm((size_t)s.nsl * rnd16(s.H[s.L - 1]) * d.ghld);
  for (int l = 0; l < s.L; ++l) {
    d.w[l] = dm((size_t)s.K(l) * s.H[l]); d.b[l] = dm(s.H[l]); d.gwld[l] = kvm_fmlp_pad16(s.H[l]);
    d.gw[l] = dm((size_t)s.nsl * rnd16(s.K(l)) * d.gwld[l]); d.gb[l] = dm((size_t)s.nsl * s.H[l]);
  }
  KvmFusedDesc* ddesc; CK(cudaMalloc(&ddesc, sizeof(KvmFusedDesc))); CK(cudaMemcpy(ddesc, &d, sizeof(KvmFusedDesc), cudaMemcpyHostToDevice));
  g.d = ddesc;
  CK(cudaMalloc(&g.ph, NPH * 8)); CK(cudaMemset(g.ph, 0, NPH * 8));
  CK(cudaMalloc(&g.ids, (size_t)steps * bt * 4)); CK(cudaMalloc(&g.tgt, (size_t)steps * bt * 4)); CK(cudaMalloc(&g.loss, (size_t)steps * s.nsl * 4)); CK(cudaMalloc(&g.progress, (size_t)s.nsl * 32 * 4));
  CK(cudaMemcpy(g.ids, ids.data(), (size_t)steps * bt * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(g.tgt, tgt.data(), (size_t)steps * bt * 4, cudaMemcpyHostToDevice));
  const int smem_bytes = total_floats * 4 + 4096;
  CK(cudaFuncSetAttribute(fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fused_kernel, 256, smem_bytes));
  printf("fused_mlp_bed: tokens=%d V=%d E=%d hidden=", bt, V, E); for (int l = 0; l < s.L; ++l) printf("%d%s", s.H[l], l + 1 < s.L ? "," : ""); printf(" slices=%d (R=%d), steps=%d; shared memory %d KB, %d resident block(s)/SM\n", s.nsl, slice, steps, smem_bytes / 1024, occ);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto reset = [&]() {
    CK(cudaMemcpy(d.we, ref.we.data(), (size_t)V * E * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.wh, ref.wh.data(), (size_t)s.H[s.L - 1] * V * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < s.L; ++l) { CK(cudaMemcpy(d.w[l], ref.w[l].data(), (size_t)s.K(l) * s.H[l] * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.b[l], ref.b[l].data(), s.H[l] * 4, cudaMemcpyHostToDevice)); }
    CK(cudaMemset(d.ge, 0, (size_t)s.nsl * V * E * 4)); CK(cudaMemset(g.progress, 0, (size_t)s.nsl * 32 * 4)); CK(cudaDeviceSynchronize());
  };
  double best = 1e30;
  for (int rep = 0; rep <= reps; ++rep) {
    reset();
    CK(cudaEventRecord(e0)); fused_kernel<<<s.nsl, 256, smem_bytes>>>(g); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); CK(cudaGetLastError());
    if (rep) { best = std::min(best, (double)ms); printf("rep %d: %.3f ms = %.2f us/step\n", rep, ms, ms * 1000.f / steps); }
    else printf("warm-up (rep 0): %.3f ms\n", ms);
    CK(cudaMemset(g.progress, 0, (size_t)s.nsl * 32 * 4)); CK(cudaMemset(d.ge, 0, (size_t)s.nsl * V * E * 4));
  }
  printf("best: %.2f us/step\n", best * 1000. / steps);
  {
    long long hp[NPH]; CK(cudaMemcpy(hp, g.ph, sizeof hp, cudaMemcpyDeviceToHost));
    static const char* names[NPH] = {"load weights+ids", "embedding gather", "forward GEMMs (+gelu)", "head GEMM", "softmax CE", "head backward GEMMs", "dZ gelu' (+sync)", "bias colsum", "gelu_f again (l>0)", "weight gradient GEMM (A^T dZ)", "activation gradient GEMM (dZ W^T)", "publish partials (atomics, fence)", "wait partials (hop)", "reduce + SGD + fence", "wait updates (hop 2)"};
    long long tot = 0; for (int i = 0; i < NPH; ++i) tot += hp[i];
    printf("phase breakdown (slice 0, cycles per step, %d steps x%d runs):\n", steps, reps + 1);
    for (int i = 0; i < NPH; ++i) if (hp[i]) printf("  %-34s %8.0f  (%.1f%%)\n", names[i], (double)hp[i] / (steps * (reps + 1)), 100.0 * hp[i] / tot);
    printf("  total %.0f cycles per step\n", (double)tot / (steps * (reps + 1)));
  }
  if (check > 0) {
    reset();
    int s0 = g.steps; g.steps = check;
    fused_kernel<<<s.nsl, 256, smem_bytes>>>(g); CK(cudaDeviceSynchronize()); g.steps = s0;
    std::vector<float> dl((size_t)check * s.nsl);
    CK(cudaMemcpy(dl.data(), g.loss, dl.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int st = 0; st < check; ++st) {
      double dev = 0; for (int c = 0; c < s.nsl; ++c) dev += dl[(size_t)st * s.nsl + c];
      dev /= bt;
      double rf = ref_step(ref, ids.data() + (size_t)st * bt, tgt.data() + (size_t)st * bt);
      double rel = std::fabs(dev - rf) / std::max(1e-9, std::fabs(rf));
      worst = std::max(worst, rel);
      printf("step %d: loss device %.6f, cpu %.6f (rel %.2e)\n", st, dev, rf, rel);
    }
    std::vector<float> w((size_t)s.H[s.L - 1] * V); CK(cudaMemcpy(w.data(), d.wh, w.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0; for (size_t i = 0; i < w.size(); ++i) { num += (w[i] - ref.wh[i]) * (w[i] - ref.wh[i]); den += ref.wh[i] * ref.wh[i]; }
    printf("after %d steps: head weights relative L2 to the fp32 reference: %.2e; worst per-step loss difference %.2e\n", check, std::sqrt(num / den), worst);
  }
  return 0;
}
