// grid_bed: the grid MLP step for MLPs too wide for the split step: one copy of the model, each weight matrix cut K x N over the gang's P blocks.
//   nvcc -O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -Iinclude bench/fused_mlp/grid_bed.cu -o grid_bed
//   ./grid_bed [--hidden 1024] [--depth 2] [--members 108] [--steps 50] [--check 3] [--embed 48] [--vocab 100] [--seed 1] [--dump FILE]
//              [--phys P_PHYS] [--runs R]
// --phys runs the arithmetic of --members on fewer physical members, each carrying several (mvm_fused_grid_virtual.cuh: blocks streamed);
// --runs R launches R gangs side by side, each with a model of its own trained on the same tokens, and checks every copy equals the first
// byte for byte. A virtual run's dump compares to the resident run's of the same --members byte for byte.
// The step is include/mvm_fused_grid.cuh's grid_run (mvm_fused_grid.h has the design), looped over the steps on one gang of P blocks, so this bed
// times and checks the code an executor op would run. The model and data are mlp_ref.h's, drawn as split_bed draws them, so the two beds train the
// same model on one seed and shape. Prints block 0's per-phase cycles (each phase ends at its last warp's end), the step time by the kernel's own
// %globaltimer, and, with --check K, the first K steps against the float64 reference. --dump writes the losses and the model after the check's
// steps as raw floats, in split_bed's order (each step's loss and 1 / T, then every layer's W and b, the head and the embedding), so another
// form's dump of the same shape, seed and steps compares to it byte for byte.
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "mvm_fused_grid.cuh"
#include "mvm_fused_grid_virtual.cuh"
#include "bed_clock.cuh"
#include "mlp_ref.h"
using mvm_fmlp::NT;

constexpr float LR = 0.05f;
constexpr int NPH = mvm_fgrid::NPH;
constexpr int NALL = NPH + mvm_fgrid::NPH_SUB;   // every stamp index: the phases, then the sub-phases (mvm_fused_grid.cuh)
extern __shared__ __align__(128) float smem[];

// R gangs of `phys` blocks, gang r = blocks [r phys, (r + 1) phys), each running its own run (descriptor gd[r], loss slots loss + r * lstride) over
// the steps: grid_run (resident, phys = P) or grid_run_chunked; the descriptor is prefetched before each step, as the static worker does.
template <bool VIRT>
__global__ void __launch_bounds__(NT, 1) grid_kernel(const MvmGridDesc* gd, int phys, const int* ids, const int* tgt, float* loss, int lstride, int steps, Timing tm) {
  const int tid = threadIdx.x, T = mvm_fgrid::T96, run = blockIdx.x / phys, member = blockIdx.x % phys;
  gd += run; loss += (size_t)run * lstride;
  if (tid == 0) tm.smid[blockIdx.x] = sm_id();
  __shared__ unsigned long long stamp[2][NALL];
  for (int i = tid; i < 2 * NALL; i += NT) stamp[i / NALL][i % NALL] = 0ull;
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
  // Each step's every stamp, raw, for the host's sub-phase walk: tm.phase[NPH + step * NALL + i], copied by warp 0's lanes when thread 0 takes the step
  unsigned long long* const raw = reinterpret_cast<unsigned long long*>(tm.phase + NPH);
  for (int step = 0; step < steps; ++step) {
    if (step > 0 && blockIdx.x == 0 && tid == 0) take(step - 1);   // every warp of block 0 is past step - 1's last stamp: it passed the last barrier
    if (step > 0 && blockIdx.x == 0 && tid < 32) for (int i = tid; i < NALL; i += 32) raw[(size_t)(step - 1) * NALL + i] = stamp[(step - 1) & 1][i];
    mvm_fgrid::prefetch_desc(gd, smem);
    if constexpr (VIRT) mvm_fgrid::grid_run_chunked(member, ids + (size_t)step * T, tgt + (size_t)step * T, loss + 2 * step, smem, nullptr, Stamps{stamp[step & 1]});
    else mvm_fgrid::grid_run(member, ids + (size_t)step * T, tgt + (size_t)step * T, loss + 2 * step, smem, nullptr, Stamps{stamp[step & 1]});
  }
  __syncthreads();
  if (blockIdx.x == 0 && tid == 0) {
    take(steps - 1); tm.gtime[1] = global_ns();
    for (int i = 0; i < NALL; ++i) raw[(size_t)(steps - 1) * NALL + i] = stamp[(steps - 1) & 1][i];
#pragma unroll
    for (int i = 0; i < NPH; ++i) tm.phase[i] = acc[i];
  }
}

int main(int argc, char** argv) {
  MvmGridDesc d{}; d.V = 100; d.E = 48; d.L = 2; d.H = 1024; d.T = mvm_fgrid::T96; d.P = 108; d.lr = LR;
  int steps = 50, check = 3, phys = 0, runs = 1, budget = 40928; unsigned seed = 1; std::string dump;   // budget: the A100 executor's fused floats
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i]; const int v = atoi(argv[i + 1]);
    if (a == "--dump") dump = argv[i + 1];
    else if (a == "--hidden") d.H = v; else if (a == "--depth") d.L = v; else if (a == "--members") d.P = v; else if (a == "--steps") steps = v;
    else if (a == "--check") check = v; else if (a == "--embed") d.E = v; else if (a == "--vocab") d.V = v; else if (a == "--seed") seed = (unsigned)v;
    else if (a == "--phys") phys = v; else if (a == "--runs") runs = v; else if (a == "--budget") budget = v;
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (!dump.empty() && check < 1) { fprintf(stderr, "--dump writes the model after the check's steps: give --check 1 or more\n"); return 2; }
  if (runs < 1) { fprintf(stderr, "--runs takes 1 or more\n"); return 2; }
  if (phys == 0) phys = d.P;
  const bool virt = phys < d.P;
  if (phys > d.P) { fprintf(stderr, "--phys %d: at most --members (%d)\n", phys, d.P); return 2; }
  d.phys = virt ? phys : 0;
  const int total = virt ? mvm_grid_layout_chunked(d, budget) : mvm_grid_layout(d);
  if (total < 0) { fprintf(stderr, "a shape the grid step does not take (mvm_grid_layout%s): H a multiple of 8, E of 16 (E at most 128), V a multiple of 4 (padded at most 128), depth 1-%d, T = 96\n", virt ? "_chunked" : "", MVM_GRID_MAXL); return 2; }
  const auto kern = virt ? grid_kernel<true> : grid_kernel<false>;
  const size_t smem_bytes = (size_t)total * sizeof(float);
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0; CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  const int T = d.T, barriers = 4 * d.L + 2;
  printf("%s, %d SMs, %zu KB shared memory a block (opt-in); model E %d V %d H %d depth %d, T %d; %d members: layer 0 1 x %d, hidden %d x %d, head %d x 1; %.1f KB a block, %d barriers a step\n",
         prop.name, prop.multiProcessorCount, (size_t)prop.sharedMemPerBlockOptin / 1024, d.E, d.V, d.H, d.L, T, d.P, d.nb0, d.ka, d.nb, d.kah, smem_bytes / 1024.0, barriers);
  printf("member: %s; %d run(s) at once, %d blocks\n", virt ? "virtual (blocks streamed)" : "resident", runs, runs * phys);
  if (virt) printf("  virtual (chunked): %d physical members carry the %d members' arithmetic; hidden blocks a load G %d, budget %d floats\n", phys, d.P, d.G, budget);
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, (const void*)kern));   // the kernel's static shared memory (the phase stamps) counts against the same limit
  if (smem_bytes + fa.sharedSizeBytes > prop.sharedMemPerBlockOptin) {
    printf("does not fit: %zu dynamic + %zu static > %zu bytes a block; raise --members\n", smem_bytes, (size_t)fa.sharedSizeBytes, (size_t)prop.sharedMemPerBlockOptin); return 3; }
  CK(cudaFuncSetAttribute((const void*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, NT, smem_bytes));
  if (occ * prop.multiProcessorCount < runs * phys) { printf("the grid (%d blocks) cannot be co-resident: %d a SM x %d SMs\n", runs * phys, occ, prop.multiProcessorCount); return 3; }

  const int nsteps = std::max(steps, check);
  const mlp_ref::Data data = mlp_ref::draw(d, seed, nsteps);   // the model and the steps' tokens, drawn as every fused MLP bed draws them
  auto dalloc = [](size_t n) { float* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(float))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(float))); return p; };
  const size_t TH = (size_t)T * d.Hp;   // the step's activations and partials, at the padded width
  std::vector<MvmGridDesc> ds((size_t)runs, d);   // run r: the same layout and model, a model and scratch of its own
  const size_t lines = mvm_gang::line_words(phys) * sizeof(unsigned);
  for (MvmGridDesc& x : ds) {
    for (int l = 0; l < x.L; ++l) { x.w[l] = dalloc(data.W[l].size()); x.b[l] = dalloc(x.H); x.Y[l] = dalloc(TH); }
    x.wh = dalloc(data.Wh.size()); x.emb = dalloc(data.emb.size());
    x.part = dalloc(x.L > 1 ? (size_t)x.ka * TH : 1); x.hpart = dalloc((size_t)x.kah * T * x.VP); x.dlog = dalloc((size_t)T * x.VP);
    x.dyp = dalloc((size_t)std::max(x.nb, 1) * TH); x.dz = dalloc(TH); x.dx0 = dalloc((size_t)x.nb0 * T * x.E); x.rloss = dalloc(T);
    x.gq = virt ? dalloc((size_t)phys * x.gqs) : nullptr;   // each physical member's gelu' (the chunked layout's gqs)
    CK(cudaMalloc(&x.ctl, lines)); CK(cudaMemset(x.ctl, 0, lines));   // once: the gang's control words (mvm_gang::cross) need no reset between launches
  }
  MvmGridDesc* gd; CK(cudaMalloc(&gd, ds.size() * sizeof(MvmGridDesc))); CK(cudaMemcpy(gd, ds.data(), ds.size() * sizeof(MvmGridDesc), cudaMemcpyHostToDevice));
  int *dids, *dtgt; CK(cudaMalloc(&dids, data.ids.size() * sizeof(int))); CK(cudaMalloc(&dtgt, data.tgt.size() * sizeof(int)));
  CK(cudaMemcpy(dids, data.ids.data(), data.ids.size() * sizeof(int), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtgt, data.tgt.data(), data.tgt.size() * sizeof(int), cudaMemcpyHostToDevice));
  const int lstride = nsteps * 2;   // each run's loss slots
  float* dloss = dalloc((size_t)runs * lstride);
  Timing tm{};
  const size_t phase_words = NPH + (size_t)nsteps * NALL;   // the phases' sums, then every step's raw stamps
  CK(cudaMalloc(&tm.phase, phase_words * sizeof(long long))); CK(cudaMalloc(&tm.gtime, 2 * sizeof(unsigned long long))); CK(cudaMalloc(&tm.smid, (size_t)runs * phys * sizeof(unsigned)));
  auto upload = [&]() {
    for (const MvmGridDesc& x : ds) {
      for (int l = 0; l < x.L; ++l) { CK(cudaMemcpy(x.w[l], data.W[l].data(), data.W[l].size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(x.b[l], data.b[l].data(), data.b[l].size() * 4, cudaMemcpyHostToDevice)); }
      CK(cudaMemcpy(x.wh, data.Wh.data(), data.Wh.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(x.emb, data.emb.data(), data.emb.size() * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(tm.phase, 0, phase_words * sizeof(long long)));
  };
  auto launch = [&](int n) {
    int nn = n, ph = phys, ls = lstride; void* args[] = {&gd, &ph, &dids, &dtgt, &dloss, &ls, &nn, &tm};
    CK(cudaLaunchCooperativeKernel((const void*)kern, dim3(runs * phys), dim3(NT), args, smem_bytes, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  };
  // A run's state after the check's steps, as --dump writes it: its losses, then every layer's W and b, the head and the embedding.
  auto state_of = [&](const MvmGridDesc& x, int run) {
    std::vector<float> out((size_t)check * 2);
    CK(cudaMemcpy(out.data(), dloss + (size_t)run * lstride, out.size() * 4, cudaMemcpyDeviceToHost));
    auto take = [&](const float* p, size_t n) { const size_t o = out.size(); out.resize(o + n); CK(cudaMemcpy(out.data() + o, p, n * 4, cudaMemcpyDeviceToHost)); };
    for (int l = 0; l < x.L; ++l) { take(x.w[l], data.W[l].size()); take(x.b[l], (size_t)x.H); }
    take(x.wh, data.Wh.size()); take(x.emb, data.emb.size());
    return out;
  };
  int bad = 0;
  if (check > 0) {   // the first `check` steps against fp64
    upload(); launch(check);
    std::vector<float> loss((size_t)check * 2), gW[mlp_ref::MAXL], gb[mlp_ref::MAXL], gWh(data.Wh.size()), gemb(data.emb.size());
    const MvmGridDesc& x0 = ds[0];   // run 0 against fp64; every other run against run 0, byte for byte
    CK(cudaMemcpy(loss.data(), dloss, loss.size() * 4, cudaMemcpyDeviceToHost));
    for (int l = 0; l < d.L; ++l) { gW[l].resize(data.W[l].size()); gb[l].resize(d.H); CK(cudaMemcpy(gW[l].data(), x0.w[l], gW[l].size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gb[l].data(), x0.b[l], gb[l].size() * 4, cudaMemcpyDeviceToHost)); }
    CK(cudaMemcpy(gWh.data(), x0.wh, gWh.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gemb.data(), x0.emb, gemb.size() * 4, cudaMemcpyDeviceToHost));
    mlp_ref::Model m = mlp_ref::model_of(data, d.L);
    const std::vector<double> rl = mlp_ref::ref_steps(d, m, data.ids, data.tgt, check, LR);
    double worst = 0; for (int st = 0; st < check; ++st) { const double g = loss[(size_t)st * 2]; worst = std::max(worst, std::fabs(g - rl[st]) / std::fabs(rl[st])); printf("check step %d: loss %.6f, fp64 %.6f\n", st, g, rl[st]); }
    double eW = 0, eb = 0;
    for (int l = 0; l < d.L; ++l) { eW = std::max(eW, mlp_ref::rel_delta(gW[l], m.W[l], data.W[l])); eb = std::max(eb, mlp_ref::rel_delta(gb[l], m.b[l], data.b[l])); }
    const double eWh = mlp_ref::rel_delta(gWh, m.Wh, data.Wh), eemb = mlp_ref::rel_delta(gemb, m.emb, data.emb);
    printf("check after %d steps: loss max rel %.2e; update rel L2 (worst layer) W %.2e, b %.2e; Wh %.2e, emb %.2e (TF32 operands: a few 1e-3 expected)\n", check, worst, eW, eb, eWh, eemb);
    bad = !(worst <= 1e-3 && eW <= 2e-2 && eb <= 2e-2 && eWh <= 2e-2 && eemb <= 2e-2);   // a NaN fails too
    printf("%s\n", bad ? "CHECK FAILED" : "check ok");
    if (runs > 1) {   // every copy computes the first's bits: R gangs at once change nothing a run computes
      const std::vector<float> first = state_of(ds[0], 0);
      int same = 1;
      for (int r = 1; r < runs; ++r) { const std::vector<float> x = state_of(ds[(size_t)r], r); same += x.size() == first.size() && !memcmp(x.data(), first.data(), x.size() * 4); }
      printf("copies: %d of %d equal to the first, byte for byte, after %d steps%s\n", same, runs, check, same == runs ? "" : ": COPIES DIFFER");
      bad |= same != runs;
    }
    if (!dump.empty()) {   // the check's losses, then every layer's W and b, the head and the embedding, as raw floats in that order (split_bed's)
      FILE* f = fopen(dump.c_str(), "wb");
      if (!f) { fprintf(stderr, "cannot write %s\n", dump.c_str()); return 2; }
      size_t n = fwrite(loss.data(), 4, loss.size(), f), want = loss.size();
      for (int l = 0; l < d.L; ++l) { n += fwrite(gW[l].data(), 4, gW[l].size(), f) + fwrite(gb[l].data(), 4, gb[l].size(), f); want += gW[l].size() + gb[l].size(); }
      n += fwrite(gWh.data(), 4, gWh.size(), f) + fwrite(gemb.data(), 4, gemb.size(), f); want += gWh.size() + gemb.size();
      if (fclose(f) != 0 || n != want) { fprintf(stderr, "short write to %s (%zu of %zu floats)\n", dump.c_str(), n, want); return 2; }
      printf("dumped %zu floats to %s\n", want, dump.c_str());
    }
  }
  // timing: a warm launch, then the timed one
  upload(); launch(std::min(steps, 20));
  upload();
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0)); launch(steps); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  std::vector<long long> hall(phase_words); CK(cudaMemcpy(hall.data(), tm.phase, phase_words * sizeof(long long), cudaMemcpyDeviceToHost));
  long long hph[NPH]; std::copy(hall.begin(), hall.begin() + NPH, hph);
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
  {   // the sub-phases (mvm_fused_grid.cuh: sub_f0, sub_fh, sub_bh), each step's raw stamps walked in execution order from the step before's last stamp
    using mvm_fgrid::sub_f0; using mvm_fgrid::sub_fh; using mvm_fgrid::sub_bh;
    std::vector<int> order{0, sub_f0(0), sub_f0(1)};
    for (int l = 1; l < d.L; ++l) for (int k = 0; k < mvm_fgrid::NSUB_FH; ++k) order.push_back(sub_fh(l, k));
    for (int i : {1, 2, 3, 4}) order.push_back(i);
    for (int q = 0; q + 1 < d.L; ++q) for (int k = 0; k < mvm_fgrid::NSUB_BH; ++k) order.push_back(sub_bh(q, k));
    for (int i : {5, 6, 7}) order.push_back(i);
    std::vector<double> dur(NALL, 0.0); long long negative = 0; int walked = 0;
    const unsigned long long* rows = reinterpret_cast<const unsigned long long*>(hall.data() + NPH);
    for (int k = 1; k < steps; ++k, ++walked) {
      const unsigned long long* row = rows + (size_t)k * NALL;
      unsigned long long prev = rows[(size_t)(k - 1) * NALL + 7];
      for (int idx : order) { const unsigned long long e = row[idx]; if (!e) continue; const long long t = (long long)(e - prev); negative += t < 0; dur[idx] += (double)t; prev = e; }
    }
    if (walked > 0) {
      const char* fh[4] = {"partials (GEMMs)", "barrier", "reduce (sum, GELU, H)", "barrier"};
      const char* bh[4] = {"dZ piece", "barrier", "input-gradient partial + update", "barrier"};
      const double tot_step = (double)tot / steps;
      auto line = [&](const std::string& n, int idx) { if (dur[idx] != 0.0) printf("  %-44s %8.0f  (%4.1f%%)\n", n.c_str(), dur[idx] / walked, 100.0 * dur[idx] / walked / std::max(1.0, tot_step)); };
      printf("sub-phases (cycles a step over steps 1-%d; each from the last warp's end of the stamp before it):\n", steps - 1);
      line("forward layer 0: work", sub_f0(0)); line("forward layer 0: barrier", sub_f0(1));
      for (int l = 1; l < d.L; ++l) for (int k = 0; k < 4; ++k) line("forward layer " + std::to_string(l) + ": " + fh[k], sub_fh(l, k));
      line("forward: the rest, to its phase's end", 1);
      for (int q = 0; q + 1 < d.L; ++q) for (int k = 0; k < 4; ++k) line("backward layer " + std::to_string(d.L - 1 - q) + ": " + bh[k], sub_bh(q, k));
      line("backward hidden: the rest, to its phase's end", 5);
      printf("sub-phase walk: %d steps, %lld negative intervals%s\n", walked, negative, negative ? " (STAMPS OUT OF ORDER: the table above is not a partition)" : " (a partition of each step)");
    }
  }
  {   // one block a SM is the design; say so when blocks shared SMs
    std::vector<unsigned> sids((size_t)runs * phys); CK(cudaMemcpy(sids.data(), tm.smid, sids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::sort(sids.begin(), sids.end());
    const int distinct = (int)(std::unique(sids.begin(), sids.end()) - sids.begin()), blocks = runs * phys;
    printf("blocks ran on %d distinct SMs of %d blocks%s\n", distinct, blocks, distinct < blocks ? ": SOME SHARED AN SM" : " (one block a SM)");
  }
  printf("effective SM clock %.0f MHz (block 0's cycles over its kernel time), rated %.0f MHz: %s\n", eff, rated,
         eff > 1.02 * rated ? "the GPU boosted above its rated clock" : eff < 0.98 * rated ? "below the rated clock (a power, thermal or application-clock limit)" : "at the rated clock");
  return bad;
}
