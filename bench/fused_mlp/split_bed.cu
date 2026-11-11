// split_bed: a fused MLP training step for widths one block cannot hold, each row slice spread over S blocks by hidden unit, at depth 1-4.
//   nvcc -O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -Iinclude bench/fused_mlp/split_bed.cu -o split_bed
//   ./split_bed [--hidden 576] [--depth 1] [--shares 5] [--slices 6] [--steps 50] [--check 3] [--embed 48] [--vocab 100] [--seed 1]
//
// The model is the panel's MLP: embedding V x E, `depth` hidden layers of H units (linear with bias, then tanh GELU), a head H x V without bias,
// softmax cross-entropy over the batch's T = 16 x slices tokens, SGD (lr 0.05). Block (q, s) of the Q x S grid takes slice q's 16 rows and
// share s's hidden units [h_s, h_s + w_s) of every layer: its columns of each layer's weights and bias, and its rows of the head, loaded from
// global memory each step, so a share's data is 1/S of the model. The blocks of a slice exchange activations through L2, each crossing a
// grid barrier (kvm_gang.cuh's form); a step has 2 x depth + 1 of them:
//   * forward, layer l: Z = A W_l,s + b_l,s and H = gelu(Z) are local. Below the top layer, each block writes its columns of H, and after the
//     barrier every block of the slice reads the whole H, the next layer's input A (the all-gather).
//   * head: the logits are a sum over shares, so each block writes its partial (16 x V); after the barrier, every block of the slice sums its
//     slice's S partials in share order and runs the cross-entropy.
//   * backward, layer l from the top: dZ = dH_s gelu'(Z) is local, and the partial gradients dW_l,s (from the layer's whole input A, read back
//     from L2) and db_l,s are written for the owners. Above layer 0, the slice's dH_(l-1) is a sum over shares: each block writes its
//     full-width partial dZ W_l,s^T, and after the barrier each sums the S partials of its own columns (the reduce-scatter). At layer 0 the
//     partial is the embedding's, dX0.
//   * owners, after a barrier: the Q blocks holding share s split its parameters, sum the Q slices' partials in slice order and write the
//     updated values; every block also owns a few embedding rows and applies the tokens' gradients in token order. A last barrier ends the step.
// Fixed summation orders throughout, so a run is deterministic. The step is include/kvm_fused_split.cuh's split_run, the body of the executor's
// gang op `fmlp_split`, so this bed times and checks exactly the code the op runs; the kernel only loops it over the steps on one gang.
// Prints block 0's per-phase cycles (each phase ends at its last warp's end), the step time by the kernel's own %globaltimer, and, with
// --check K, the first K steps against an fp64 CPU reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "kvm_fused_split.cuh"
using kvm_fmlp::NT;

constexpr float LR = 0.05f;
constexpr int R = KVM_SPLIT_R;
constexpr int MAXL = KVM_SPLIT_MAXL;
constexpr int NPH = kvm_fsplit::NPH;
extern __shared__ __align__(128) float smem[];

struct Timing { long long* phase; unsigned long long* gtime; unsigned* smid; };   // [NPH] cycles summed over steps (block 0); [2] its %globaltimer; [blocks] each block's SM
__device__ __forceinline__ unsigned sm_id() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned long long global_ns() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// Block 0's phase clock: every warp stamps its own end of a phase (a shared atomic max) and thread 0 turns the stamps into durations one step later,
// so a phase ends at its last warp's end (thread 0's clock read after a barrier runs when warp 0 arrives: sm_80's barrier holds a warp only at its
// next memory access). The stamps alternate between two rows by step parity; no barrier is added.
struct Stamps {
  unsigned long long* row;
  __device__ __forceinline__ void operator()(int i) const { if (blockIdx.x == 0 && (threadIdx.x & 31) == 0) atomicMax(row + i, (unsigned long long)clock64()); }
};

// One gang of Q x S blocks running split_run over the steps; the descriptor is prefetched before each step, as the static worker does.
template <int S, int Q>
__global__ void __launch_bounds__(NT, 1) split_kernel(const KvmSplitDesc* gd, const int* ids, const int* tgt, float* loss, int steps, int T, Timing tm) {
  const int tid = threadIdx.x;
  if (tid == 0) tm.smid[blockIdx.x] = sm_id();
  __shared__ unsigned long long stamp[2][NPH];
  for (int i = tid; i < 2 * NPH; i += NT) stamp[i / NPH][i % NPH] = 0ull;
  __syncthreads();
  unsigned long long prev_end = 0;
  long long acc[NPH];
#pragma unroll
  for (int i = 0; i < NPH; ++i) acc[i] = 0;
  if (blockIdx.x == 0 && tid == 0) { tm.gtime[0] = global_ns(); prev_end = clock64(); }
  auto take = [&](int step) {
#pragma unroll
    for (int i = 0; i < NPH; ++i) { const unsigned long long e = stamp[step & 1][i]; acc[i] += (long long)(e - prev_end); prev_end = e; }
  };
  for (int step = 0; step < steps; ++step) {
    if (step > 0 && blockIdx.x == 0 && tid == 0) take(step - 1);   // every warp of block 0 is past step - 1's last stamp: it passed the last barrier
    kvm_fsplit::prefetch_desc(gd, smem);
    kvm_fsplit::split_run<S, Q>(blockIdx.x, ids + (size_t)step * T, tgt + (size_t)step * T, loss + 2 * step, smem, nullptr, Stamps{stamp[step & 1]});
  }
  __syncthreads();
  if (blockIdx.x == 0 && tid == 0) {
    take(steps - 1); tm.gtime[1] = global_ns();
#pragma unroll
    for (int i = 0; i < NPH; ++i) tm.phase[i] = acc[i];
  }
}

// The instantiated (shares, slices) pairs, the op's own (KVM_SPLIT_SHARE_LIST x KVM_SPLIT_Q): a run-time bound on the partial-sum loops would serialize
// each cell's loads (measured on the fused slice's owners: about 7.6 GB/s an SM, latency-bound).
using KernelFn = void (*)(const KvmSplitDesc*, const int*, const int*, float*, int, int, Timing);
static KernelFn pick_kernel(int S, int Q) {
  if (Q != KVM_SPLIT_Q) return nullptr;
  switch (S) {
#define X(S_) case S_: return split_kernel<S_, KVM_SPLIT_Q>;
    KVM_SPLIT_SHARE_LIST
#undef X
    default: return nullptr;
  }
}

// ---------------- host: the fp64 reference of the same steps
static double gelu_d(double x) { const double y = 0.7978845608028654 * (x + 0.044715 * x * x * x); return 0.5 * x * (1.0 + std::tanh(y)); }
static double gelu_gd(double x) { const double y = 0.7978845608028654 * (x + 0.044715 * x * x * x), t = std::tanh(y); return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x); }
struct Model { std::vector<double> W[MAXL], b[MAXL], Wh, emb; };
static std::vector<double> ref_steps(const KvmSplitDesc& d, Model& m, const std::vector<int>& ids, const std::vector<int>& tgt, int k) {
  const int E = d.E, V = d.V, H = d.H, T = d.T, L = d.L; std::vector<double> losses;
  for (int step = 0; step < k; ++step) {
    const int* id = &ids[(size_t)step * T]; const int* tg = &tgt[(size_t)step * T];
    std::vector<std::vector<double>> A(L + 1), Z(L);   // A[l]: layer l's input (A[0] the embedding rows, T x E; A[l > 0] T x H); A[L] the head's input
    A[0].assign((size_t)T * E, 0.0);
    for (int r = 0; r < T; ++r) for (int e = 0; e < E; ++e) A[0][(size_t)r * E + e] = m.emb[(size_t)id[r] * E + e];
    for (int l = 0; l < L; ++l) {
      const int K = l ? H : E;
      Z[l].assign((size_t)T * H, 0.0); A[l + 1].assign((size_t)T * H, 0.0);
      for (int r = 0; r < T; ++r) for (int c = 0; c < H; ++c) { double a = m.b[l][c]; for (int kk = 0; kk < K; ++kk) a += A[l][(size_t)r * K + kk] * m.W[l][(size_t)kk * H + c]; Z[l][(size_t)r * H + c] = a; A[l + 1][(size_t)r * H + c] = gelu_d(a); }
    }
    std::vector<double> Lg((size_t)T * V), dL((size_t)T * V);
    double loss = 0.0;
    for (int r = 0; r < T; ++r) {
      double mx = -1e300; for (int v = 0; v < V; ++v) { double a = 0; for (int c = 0; c < H; ++c) a += A[L][(size_t)r * H + c] * m.Wh[(size_t)c * V + v]; Lg[(size_t)r * V + v] = a; mx = std::max(mx, a); }
      double z = 0; for (int v = 0; v < V; ++v) z += std::exp(Lg[(size_t)r * V + v] - mx);
      loss += mx + std::log(z) - Lg[(size_t)r * V + tg[r]];
      for (int v = 0; v < V; ++v) dL[(size_t)r * V + v] = (std::exp(Lg[(size_t)r * V + v] - mx) / z - (v == tg[r] ? 1.0 : 0.0)) / T;
    }
    losses.push_back(loss / T);
    std::vector<double> gWh((size_t)H * V, 0.0), gEmb((size_t)V * E, 0.0), dH((size_t)T * H, 0.0);
    std::vector<std::vector<double>> gW(L), gb(L);
    for (int c = 0; c < H; ++c) for (int v = 0; v < V; ++v) { double a = 0; for (int r = 0; r < T; ++r) a += A[L][(size_t)r * H + c] * dL[(size_t)r * V + v]; gWh[(size_t)c * V + v] = a; }
    for (int r = 0; r < T; ++r) for (int c = 0; c < H; ++c) { double a = 0; for (int v = 0; v < V; ++v) a += dL[(size_t)r * V + v] * m.Wh[(size_t)c * V + v]; dH[(size_t)r * H + c] = a; }
    for (int l = L - 1; l >= 0; --l) {
      const int K = l ? H : E;
      std::vector<double> dZ((size_t)T * H);
      for (size_t i = 0; i < dZ.size(); ++i) dZ[i] = dH[i] * gelu_gd(Z[l][i]);
      gW[l].assign((size_t)K * H, 0.0); gb[l].assign(H, 0.0);
      for (int kk = 0; kk < K; ++kk) for (int c = 0; c < H; ++c) { double a = 0; for (int r = 0; r < T; ++r) a += A[l][(size_t)r * K + kk] * dZ[(size_t)r * H + c]; gW[l][(size_t)kk * H + c] = a; }
      for (int c = 0; c < H; ++c) { double a = 0; for (int r = 0; r < T; ++r) a += dZ[(size_t)r * H + c]; gb[l][c] = a; }
      std::vector<double> dA((size_t)T * K, 0.0);
      for (int r = 0; r < T; ++r) for (int kk = 0; kk < K; ++kk) { double a = 0; for (int c = 0; c < H; ++c) a += dZ[(size_t)r * H + c] * m.W[l][(size_t)kk * H + c]; dA[(size_t)r * K + kk] = a; }
      if (l == 0) { for (int r = 0; r < T; ++r) for (int e = 0; e < E; ++e) gEmb[(size_t)id[r] * E + e] += dA[(size_t)r * E + e]; }
      else dH = dA;
    }
    for (int l = 0; l < L; ++l) { for (size_t i = 0; i < m.W[l].size(); ++i) m.W[l][i] -= LR * gW[l][i]; for (size_t i = 0; i < m.b[l].size(); ++i) m.b[l][i] -= LR * gb[l][i]; }
    for (size_t i = 0; i < m.Wh.size(); ++i) m.Wh[i] -= LR * gWh[i];
    for (size_t i = 0; i < m.emb.size(); ++i) m.emb[i] -= LR * gEmb[i];
  }
  return losses;
}
static double rel_delta(const std::vector<float>& gpu, const std::vector<double>& ref, const std::vector<float>& init) {   // relative L2 of the update (after - before)
  double n = 0, dd = 0; for (size_t i = 0; i < ref.size(); ++i) { const double r = ref[i] - init[i], g = (double)gpu[i] - init[i]; n += r * r; dd += (g - r) * (g - r); }
  return std::sqrt(dd / std::max(n, 1e-300));
}

int main(int argc, char** argv) {
  KvmSplitDesc d{}; d.V = 100; d.E = 48; d.L = 1; d.H = 576; d.S = 5; d.Q = 6; d.lr = LR;
  int steps = 50, check = 3; unsigned seed = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i]; const int v = atoi(argv[i + 1]);
    if (a == "--hidden") d.H = v; else if (a == "--depth") d.L = v; else if (a == "--shares") d.S = v; else if (a == "--slices") d.Q = v; else if (a == "--steps") steps = v;
    else if (a == "--check") check = v; else if (a == "--embed") d.E = v; else if (a == "--vocab") d.V = v; else if (a == "--seed") seed = (unsigned)v;
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  const int total = kvm_split_layout(d);
  if (total < 0) { fprintf(stderr, "a shape fmlp_split does not take (kvm_split_layout): H a multiple of 8, E of 16 (at most 128), V of 4 (padded at most 128), the widest share at most 128, depth 1-%d, shares 1-%d\n", MAXL, KVM_SPLIT_MAXS); return 2; }
  int wmax = 0; for (int s = 0; s < d.S; ++s) wmax = std::max(wmax, d.share0[s + 1] - d.share0[s]);
  const KernelFn kern = pick_kernel(d.S, d.Q);
  if (!kern) { fprintf(stderr, "no kernel instantiated for %d shares and %d slices (pick_kernel: S in 2-6, 8, 9, 10, 12, 18 with 6 slices)\n", d.S, d.Q); return 2; }
  const size_t smem_bytes = (size_t)total * sizeof(float);
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0; CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  const int nslot = d.Q * d.S, T = d.T;
  printf("%s, %d SMs, %zu KB shared memory a block (opt-in); model E %d V %d H %d depth %d, T %d (%d slices of %d), S %d shares (width <= %d, padded %d): %d blocks, %.1f KB a block, %d barriers a step\n",
         prop.name, prop.multiProcessorCount, (size_t)prop.sharedMemPerBlockOptin / 1024, d.E, d.V, d.H, d.L, T, d.Q, R, d.S, wmax, d.HP, nslot, smem_bytes / 1024.0, 2 * d.L + 1);
  if (smem_bytes > prop.sharedMemPerBlockOptin) { printf("does not fit: %zu > %zu bytes a block; raise --shares\n", smem_bytes, (size_t)prop.sharedMemPerBlockOptin); return 3; }
  CK(cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, NT, smem_bytes));
  if (occ * prop.multiProcessorCount < nslot) { printf("the grid (%d blocks) cannot be co-resident: %d a SM x %d SMs\n", nslot, occ, prop.multiProcessorCount); return 3; }

  std::mt19937 rng(seed); std::uniform_real_distribution<float> u(-1.f, 1.f); std::uniform_int_distribution<int> tok(0, d.V - 1);
  std::vector<float> W[MAXL], bvec[MAXL], Wh((size_t)d.H * d.V), emb((size_t)d.V * d.E);
  for (auto& x : emb) x = 0.5f * u(rng);
  for (int l = 0; l < d.L; ++l) {
    const int K = l ? d.H : d.E;
    W[l].resize((size_t)K * d.H); for (auto& x : W[l]) x = u(rng) / std::sqrt((float)K);
    bvec[l].resize(d.H); for (auto& x : bvec[l]) x = 0.1f * u(rng);
  }
  for (auto& x : Wh) x = u(rng) / std::sqrt((float)d.H);
  const int nsteps = std::max(steps, check);
  std::vector<int> ids((size_t)nsteps * T), tgt((size_t)nsteps * T);
  for (auto& x : ids) x = tok(rng);
  for (auto& x : tgt) x = tok(rng);

  auto dalloc = [](size_t n) { float* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(float))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(float))); return p; };
  for (int l = 0; l < d.L; ++l) {
    d.w[l] = dalloc(W[l].size()); d.b[l] = dalloc(d.H);
    d.dW[l] = dalloc((size_t)nslot * (l ? d.HR : d.E) * d.HP); d.db[l] = dalloc((size_t)nslot * d.HP);
    if (l + 1 < d.L) d.XH[l] = dalloc((size_t)d.Q * R * d.HR);   // zero once: the pad columns past H are never written
  }
  d.wh = dalloc(Wh.size()); d.we = dalloc(emb.size());
  d.P = dalloc((size_t)nslot * R * d.VP); d.dWh = dalloc((size_t)nslot * d.HP * d.VP); d.dX0 = dalloc((size_t)nslot * R * d.E);
  for (int k = 0; k < 2; ++k) d.XD[k] = d.L > 1 ? dalloc((size_t)nslot * R * d.HR) : nullptr;   // layer l's reduce-scatter in XD[l & 1]
  d.loss_part = dalloc(d.Q);
  CK(cudaMalloc(&d.ctl, 2 * sizeof(unsigned))); CK(cudaMemset(d.ctl, 0, 2 * sizeof(unsigned)));   // once: the gang barrier needs no reset between launches
  KvmSplitDesc* gd; CK(cudaMalloc(&gd, sizeof(KvmSplitDesc))); CK(cudaMemcpy(gd, &d, sizeof(KvmSplitDesc), cudaMemcpyHostToDevice));
  int *dids, *dtgt; CK(cudaMalloc(&dids, ids.size() * sizeof(int))); CK(cudaMalloc(&dtgt, tgt.size() * sizeof(int)));
  CK(cudaMemcpy(dids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtgt, tgt.data(), tgt.size() * sizeof(int), cudaMemcpyHostToDevice));
  float* dloss = dalloc((size_t)nsteps * 2);
  Timing tm{};
  CK(cudaMalloc(&tm.phase, NPH * sizeof(long long))); CK(cudaMalloc(&tm.gtime, 2 * sizeof(unsigned long long))); CK(cudaMalloc(&tm.smid, (size_t)nslot * sizeof(unsigned)));
  auto upload = [&]() {
    for (int l = 0; l < d.L; ++l) { CK(cudaMemcpy(d.w[l], W[l].data(), W[l].size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.b[l], bvec[l].data(), bvec[l].size() * 4, cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(d.wh, Wh.data(), Wh.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.we, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(tm.phase, 0, NPH * sizeof(long long)));
  };
  auto launch = [&](int n) {
    int nn = n, TT = T; void* args[] = {&gd, &dids, &dtgt, &dloss, &nn, &TT, &tm};
    CK(cudaLaunchCooperativeKernel((void*)kern, dim3(nslot), dim3(NT), args, smem_bytes, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  };
  int bad = 0;
  if (check > 0) {   // the first `check` steps against fp64
    upload(); launch(check);
    std::vector<float> loss((size_t)check * 2), gW[MAXL], gb[MAXL], gWh(Wh.size()), gemb(emb.size());
    CK(cudaMemcpy(loss.data(), dloss, loss.size() * 4, cudaMemcpyDeviceToHost));
    for (int l = 0; l < d.L; ++l) { gW[l].resize(W[l].size()); gb[l].resize(d.H); CK(cudaMemcpy(gW[l].data(), d.w[l], gW[l].size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gb[l].data(), d.b[l], gb[l].size() * 4, cudaMemcpyDeviceToHost)); }
    CK(cudaMemcpy(gWh.data(), d.wh, gWh.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gemb.data(), d.we, gemb.size() * 4, cudaMemcpyDeviceToHost));
    Model m;
    for (int l = 0; l < d.L; ++l) { m.W[l].assign(W[l].begin(), W[l].end()); m.b[l].assign(bvec[l].begin(), bvec[l].end()); }
    m.Wh.assign(Wh.begin(), Wh.end()); m.emb.assign(emb.begin(), emb.end());
    const std::vector<double> rl = ref_steps(d, m, ids, tgt, check);
    double worst = 0; for (int st = 0; st < check; ++st) { const double g = loss[(size_t)st * 2]; worst = std::max(worst, std::fabs(g - rl[st]) / std::fabs(rl[st])); printf("check step %d: loss %.6f, fp64 %.6f\n", st, g, rl[st]); }
    double eW = 0, eb = 0;
    for (int l = 0; l < d.L; ++l) { eW = std::max(eW, rel_delta(gW[l], m.W[l], W[l])); eb = std::max(eb, rel_delta(gb[l], m.b[l], bvec[l])); }
    const double eWh = rel_delta(gWh, m.Wh, Wh), eemb = rel_delta(gemb, m.emb, emb);
    printf("check after %d steps: loss max rel %.2e; update rel L2 (worst layer) W %.2e, b %.2e; Wh %.2e, emb %.2e (TF32 operands: a few 1e-3 expected)\n", check, worst, eW, eb, eWh, eemb);
    bad = worst > 1e-3 || eW > 2e-2 || eb > 2e-2 || eWh > 2e-2 || eemb > 2e-2;
    printf("%s\n", bad ? "CHECK FAILED" : "check ok");
  }
  // timing: a warm launch, then the timed one
  upload(); launch(std::min(steps, 20));
  upload();
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0)); launch(steps); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  long long hph[NPH]; CK(cudaMemcpy(hph, tm.phase, sizeof(hph), cudaMemcpyDeviceToHost));
  const char* names[NPH] = {"load (weights, embedding rows)", "forward (GEMMs, GELU, all-gathers)", "logits partial GEMM (to L2)", "barrier: logits",
                            "logits summed, cross-entropy", "backward (GEMMs, reduce-scatters)", "barrier: partials", "owners: parameters", "owners: embedding + last barrier"};
  long long tot = 0; for (int i = 0; i < NPH; ++i) tot += hph[i];
  unsigned long long hgt[2]; CK(cudaMemcpy(hgt, tm.gtime, sizeof(hgt), cudaMemcpyDeviceToHost));
  const double kern_us = (hgt[1] - hgt[0]) / 1e3, rated = clock_khz / 1e3, eff = (double)tot / std::max(1e-9, kern_us);
  printf("timed: %d steps: %.2f us a step in the kernel (block 0's %%globaltimer over its counted loop), %.2f us by events (launch and host delays included)\n",
         steps, kern_us / steps, 1e3 * ms / steps);
  printf("block 0 by phase (cycles a step; %d of the step's %d barriers sit inside the forward and backward phases):\n", 2 * (d.L - 1), 2 * d.L + 1);
  for (int i = 0; i < NPH; ++i) printf("  %-38s %8.0f  (%4.1f%%)\n", names[i], (double)hph[i] / steps, 100.0 * hph[i] / std::max(1ll, tot));
  printf("  %-38s %8.0f cycles\n", "total", (double)tot / steps);
  printf("block 0's phases: from the last warp's end of the previous phase to the last warp's end of this one (per-warp stamps)\n");
  {   // one block a SM is the design; say so when blocks shared SMs
    std::vector<unsigned> sids((size_t)nslot); CK(cudaMemcpy(sids.data(), tm.smid, sids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::sort(sids.begin(), sids.end());
    const int distinct = (int)(std::unique(sids.begin(), sids.end()) - sids.begin());
    printf("blocks ran on %d distinct SMs of %d blocks%s\n", distinct, nslot, distinct < nslot ? ": SOME SHARED AN SM" : " (one block a SM)");
  }
  printf("effective SM clock %.0f MHz (block 0's cycles over its kernel time), rated %.0f MHz: %s\n", eff, rated,
         eff > 1.02 * rated ? "the GPU boosted above its rated clock" : eff < 0.98 * rated ? "below the rated clock (a power, thermal or application-clock limit)" : "at the rated clock");
  return bad;
}
