// grid_bed: the grid MLP step for MLPs too wide for the split step: one copy of the model, each weight matrix cut K x N over the gang's P blocks.
//   nvcc -O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -Iinclude bench/fused_mlp/grid_bed.cu -o grid_bed
//   ./grid_bed [--hidden 1024] [--depth 2] [--members 108] [--steps 50] [--check 3] [--embed 48] [--vocab 100] [--seed 1]
// The step is include/kvm_fused_grid.cuh's grid_run (kvm_fused_grid.h has the design), looped over the steps on one gang of P blocks, so this bed
// times and checks the code an executor op would run. The model and data are mlp_ref.h's, drawn as split_bed draws them, so the two beds train the
// same model on one seed and shape. Prints block 0's per-phase cycles (each phase ends at its last warp's end), the step time by the kernel's own
// %globaltimer, and, with --check K, the first K steps against the float64 reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "kvm_fused_grid.cuh"
#include "bed_clock.cuh"
#include "mlp_ref.h"
using kvm_fmlp::NT;

constexpr float LR = 0.05f;
constexpr int NPH = kvm_fgrid::NPH;
extern __shared__ __align__(128) float smem[];

// One gang of P blocks running grid_run over the steps; the descriptor is prefetched before each step, as the static worker does.
__global__ void __launch_bounds__(NT, 1) grid_kernel(const KvmGridDesc* gd, const int* ids, const int* tgt, float* loss, int steps, Timing tm) {
  const int tid = threadIdx.x, T = kvm_fgrid::T96;
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
    kvm_fgrid::prefetch_desc(gd, smem);
    kvm_fgrid::grid_run(blockIdx.x, ids + (size_t)step * T, tgt + (size_t)step * T, loss + 2 * step, smem, nullptr, Stamps{stamp[step & 1]});
  }
  __syncthreads();
  if (blockIdx.x == 0 && tid == 0) {
    take(steps - 1); tm.gtime[1] = global_ns();
#pragma unroll
    for (int i = 0; i < NPH; ++i) tm.phase[i] = acc[i];
  }
}

int main(int argc, char** argv) {
  KvmGridDesc d{}; d.V = 100; d.E = 48; d.L = 2; d.H = 1024; d.T = kvm_fgrid::T96; d.P = 108; d.lr = LR;
  int steps = 50, check = 3; unsigned seed = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i]; const int v = atoi(argv[i + 1]);
    if (a == "--hidden") d.H = v; else if (a == "--depth") d.L = v; else if (a == "--members") d.P = v; else if (a == "--steps") steps = v;
    else if (a == "--check") check = v; else if (a == "--embed") d.E = v; else if (a == "--vocab") d.V = v; else if (a == "--seed") seed = (unsigned)v;
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  const int total = kvm_grid_layout(d);
  if (total < 0) { fprintf(stderr, "a shape the grid step does not take (kvm_grid_layout): H and E multiples of 16 (E at most 128), V a multiple of 4 (padded at most 128), depth 1-%d, T = 96\n", KVM_GRID_MAXL); return 2; }
  const size_t smem_bytes = (size_t)total * sizeof(float);
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0; CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  const int T = d.T, barriers = 4 * d.L + 2;
  printf("%s, %d SMs, %zu KB shared memory a block (opt-in); model E %d V %d H %d depth %d, T %d; %d members: layer 0 1 x %d, hidden %d x %d, head %d x 1; %.1f KB a block, %d barriers a step\n",
         prop.name, prop.multiProcessorCount, (size_t)prop.sharedMemPerBlockOptin / 1024, d.E, d.V, d.H, d.L, T, d.P, d.nb0, d.ka, d.nb, d.kah, smem_bytes / 1024.0, barriers);
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, grid_kernel));   // the kernel's static shared memory (the phase stamps) counts against the same limit
  if (smem_bytes + fa.sharedSizeBytes > prop.sharedMemPerBlockOptin) {
    printf("does not fit: %zu dynamic + %zu static > %zu bytes a block; raise --members\n", smem_bytes, (size_t)fa.sharedSizeBytes, (size_t)prop.sharedMemPerBlockOptin); return 3; }
  CK(cudaFuncSetAttribute((const void*)grid_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, grid_kernel, NT, smem_bytes));
  if (occ * prop.multiProcessorCount < d.P) { printf("the gang (%d blocks) cannot be co-resident: %d a SM x %d SMs\n", d.P, occ, prop.multiProcessorCount); return 3; }

  const int nsteps = std::max(steps, check);
  const mlp_ref::Data data = mlp_ref::draw(d, seed, nsteps);   // the model and the steps' tokens, drawn as every fused MLP bed draws them
  auto dalloc = [](size_t n) { float* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(float))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(float))); return p; };
  const size_t TH = (size_t)T * d.H;
  for (int l = 0; l < d.L; ++l) { d.w[l] = dalloc(data.W[l].size()); d.b[l] = dalloc(d.H); d.Y[l] = dalloc(TH); }
  d.wh = dalloc(data.Wh.size()); d.emb = dalloc(data.emb.size());
  d.part = dalloc(d.L > 1 ? (size_t)d.ka * TH : 1); d.hpart = dalloc((size_t)d.kah * T * d.VP); d.dlog = dalloc((size_t)T * d.VP);
  d.dyp = dalloc((size_t)std::max(d.nb, 1) * TH); d.dz = dalloc(TH); d.dx0 = dalloc((size_t)d.nb0 * T * d.E); d.rloss = dalloc(T);
  CK(cudaMalloc(&d.ctl, 2 * sizeof(unsigned))); CK(cudaMemset(d.ctl, 0, 2 * sizeof(unsigned)));   // once: the gang barrier needs no reset between launches
  KvmGridDesc* gd; CK(cudaMalloc(&gd, sizeof(KvmGridDesc))); CK(cudaMemcpy(gd, &d, sizeof(KvmGridDesc), cudaMemcpyHostToDevice));
  int *dids, *dtgt; CK(cudaMalloc(&dids, data.ids.size() * sizeof(int))); CK(cudaMalloc(&dtgt, data.tgt.size() * sizeof(int)));
  CK(cudaMemcpy(dids, data.ids.data(), data.ids.size() * sizeof(int), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtgt, data.tgt.data(), data.tgt.size() * sizeof(int), cudaMemcpyHostToDevice));
  float* dloss = dalloc((size_t)nsteps * 2);
  Timing tm{};
  CK(cudaMalloc(&tm.phase, NPH * sizeof(long long))); CK(cudaMalloc(&tm.gtime, 2 * sizeof(unsigned long long))); CK(cudaMalloc(&tm.smid, (size_t)d.P * sizeof(unsigned)));
  auto upload = [&]() {
    for (int l = 0; l < d.L; ++l) { CK(cudaMemcpy(d.w[l], data.W[l].data(), data.W[l].size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.b[l], data.b[l].data(), data.b[l].size() * 4, cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(d.wh, data.Wh.data(), data.Wh.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.emb, data.emb.data(), data.emb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(tm.phase, 0, NPH * sizeof(long long)));
  };
  auto launch = [&](int n) {
    int nn = n; void* args[] = {&gd, &dids, &dtgt, &dloss, &nn, &tm};
    CK(cudaLaunchCooperativeKernel((void*)grid_kernel, dim3(d.P), dim3(NT), args, smem_bytes, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  };
  int bad = 0;
  if (check > 0) {   // the first `check` steps against fp64
    upload(); launch(check);
    std::vector<float> loss((size_t)check * 2), gW[mlp_ref::MAXL], gb[mlp_ref::MAXL], gWh(data.Wh.size()), gemb(data.emb.size());
    CK(cudaMemcpy(loss.data(), dloss, loss.size() * 4, cudaMemcpyDeviceToHost));
    for (int l = 0; l < d.L; ++l) { gW[l].resize(data.W[l].size()); gb[l].resize(d.H); CK(cudaMemcpy(gW[l].data(), d.w[l], gW[l].size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gb[l].data(), d.b[l], gb[l].size() * 4, cudaMemcpyDeviceToHost)); }
    CK(cudaMemcpy(gWh.data(), d.wh, gWh.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gemb.data(), d.emb, gemb.size() * 4, cudaMemcpyDeviceToHost));
    mlp_ref::Model m = mlp_ref::model_of(data, d.L);
    const std::vector<double> rl = mlp_ref::ref_steps(d, m, data.ids, data.tgt, check, LR);
    double worst = 0; for (int st = 0; st < check; ++st) { const double g = loss[(size_t)st * 2]; worst = std::max(worst, std::fabs(g - rl[st]) / std::fabs(rl[st])); printf("check step %d: loss %.6f, fp64 %.6f\n", st, g, rl[st]); }
    double eW = 0, eb = 0;
    for (int l = 0; l < d.L; ++l) { eW = std::max(eW, mlp_ref::rel_delta(gW[l], m.W[l], data.W[l])); eb = std::max(eb, mlp_ref::rel_delta(gb[l], m.b[l], data.b[l])); }
    const double eWh = mlp_ref::rel_delta(gWh, m.Wh, data.Wh), eemb = mlp_ref::rel_delta(gemb, m.emb, data.emb);
    printf("check after %d steps: loss max rel %.2e; update rel L2 (worst layer) W %.2e, b %.2e; Wh %.2e, emb %.2e (TF32 operands: a few 1e-3 expected)\n", check, worst, eW, eb, eWh, eemb);
    bad = !(worst <= 1e-3 && eW <= 2e-2 && eb <= 2e-2 && eWh <= 2e-2 && eemb <= 2e-2);   // a NaN fails too
    printf("%s\n", bad ? "CHECK FAILED" : "check ok");
  }
  // timing: a warm launch, then the timed one
  upload(); launch(std::min(steps, 20));
  upload();
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0)); launch(steps); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  long long hph[NPH]; CK(cudaMemcpy(hph, tm.phase, sizeof(hph), cudaMemcpyDeviceToHost));
  const char* names[NPH] = {"load (blocks, embedding rows)", "forward (layer 0, hidden layers)", "head partial logits", "cross-entropy",
                            "backward: head", "backward: hidden layers", "backward: layer 0", "embedding, write-back, last barrier"};
  long long tot = 0; for (int i = 0; i < NPH; ++i) tot += hph[i];
  unsigned long long hgt[2]; CK(cudaMemcpy(hgt, tm.gtime, sizeof(hgt), cudaMemcpyDeviceToHost));
  const double kern_us = (hgt[1] - hgt[0]) / 1e3, rated = clock_khz / 1e3, eff = (double)tot / std::max(1e-9, kern_us);
  printf("timed: %d steps: %.2f us a step in the kernel (block 0's %%globaltimer over its counted loop), %.2f us by events (launch and host delays included)\n",
         steps, kern_us / steps, 1e3 * ms / steps);
  printf("block 0 by phase (cycles a step; each phase ends at a barrier, which its time includes):\n");
  for (int i = 0; i < NPH; ++i) printf("  %-38s %8.0f  (%4.1f%%)\n", names[i], (double)hph[i] / steps, 100.0 * hph[i] / std::max(1ll, tot));
  printf("  %-38s %8.0f cycles\n", "total", (double)tot / steps);
  {   // one block a SM is the design; say so when blocks shared SMs
    std::vector<unsigned> sids((size_t)d.P); CK(cudaMemcpy(sids.data(), tm.smid, sids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::sort(sids.begin(), sids.end());
    const int distinct = (int)(std::unique(sids.begin(), sids.end()) - sids.begin());
    printf("blocks ran on %d distinct SMs of %d blocks%s\n", distinct, d.P, distinct < d.P ? ": SOME SHARED AN SM" : " (one block a SM)");
  }
  printf("effective SM clock %.0f MHz (block 0's cycles over its kernel time), rated %.0f MHz: %s\n", eff, rated,
         eff > 1.02 * rated ? "the GPU boosted above its rated clock" : eff < 0.98 * rated ? "below the rated clock (a power, thermal or application-clock limit)" : "at the rated clock");
  return bad;
}
