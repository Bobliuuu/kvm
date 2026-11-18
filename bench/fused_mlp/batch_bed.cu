// batch_bed: the batched MLP step, R runs of one MLP carried by one gang (include/mvm_fused_batch.h has the design).
//   nvcc -O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -Iinclude bench/fused_mlp/batch_bed.cu -o batch_bed
//   ./batch_bed [--hidden 640] [--depth 4] [--runs 16] [--members 0 (every SM)] [--steps 50] [--check 3] [--seed 1] [--warm-ms 500] [--bwd auto|fused|split] [--fm 1|2|3|6] [--nw 16|32|64|128] [--virtual SV | --grid P] [--dump FILE]
// --virtual SV computes the split step's arithmetic for SV virtual shares (mvm_fused_batch.h: MvmBatchDesc::Sv), so the dump equals split_bed's
// at --virtual SV for any of its share counts; 0 (the default) keeps the op's own arithmetic, each output over its whole K in one chain.
// --grid P computes the grid step's arithmetic at a gang of P members (mvm_fused_batch.h: MvmBatchDesc::gHp, the partition mvm_grid_layout takes),
// so the dump equals grid_bed's at --members P; it runs the fused backward and tiles of at most MVM_BATCH_GRID_MAXRB (3) row blocks a warp.
// --fm and --nw force the forward tiles' rows (fm x 16) and columns, and the split backward's tiles with them; --kw the fused backward's row
// blocks (16, 32 or 64 rows); like --bwd, they change only how the step is spread over the gang, so a dump must not change with them (and a
// card with less shared memory a block than the A100's, the laptop's 99 KB, takes the smaller tiles).
// --dump FILE writes run 0's model and losses after the checked steps (raw floats), so two modes or builds can be compared byte for byte.
// Every run trains the same model on the same tokens (mlp_ref.h's draw, as the other fused MLP beds draw them), so after the checked steps
// every run's model and losses must equal run 0's bit for bit, and run 0 is checked against the float64 reference. Prints the layout, block
// 0's cycles by phase (each phase ends at its crossing, which its time includes), and the step time by the kernel's own %globaltimer.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "mvm_fused_batch.h"
// the item clock (mvm_fused_batch.cuh): block 0's thread 0 adds each item's segments: operand loop, epilogue, end barrier
__device__ long long g_item_clk[CK_KINDS][3];
__device__ long long g_item_last;
__device__ int g_item_n[CK_KINDS];
#define MVM_BATCH_CLOCK(kind, mark) do { if (blockIdx.x == 0 && threadIdx.x == 0) { const long long n_ = clock64(); \
    if ((mark) == 0) ++g_item_n[kind]; else g_item_clk[kind][(mark) - 1] += n_ - g_item_last; g_item_last = n_; } } while (0)
// the chunk clock (mvm_fused_batch.cuh's k_loop), in a build with -DBED_CHUNK_CLOCK only (its stamps cost every warp a few instructions a chunk):
// block 0's warp 0 (an MMA warp) and warp 7 (on small tiles, a copier) add each item's chunk intervals: wait + barrier, copies, k-steps, chunks
__device__ long long g_chunk_clk[CK_KINDS][2][4];
#ifdef BED_CHUNK_CLOCK
#define MVM_BATCH_CHUNK_CLOCK(kind, t) do { if (blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 224)) { const int w_ = threadIdx.x ? 1 : 0; \
    for (int i_ = 0; i_ < 4; ++i_) atomicAdd((unsigned long long*)&g_chunk_clk[kind][w_][i_], (unsigned long long)(t)[i_]); } } while (0)
#endif
#include "mvm_fused_batch.cuh"
#include "mvm_fused_grid.h"
#include "bed_clock.cuh"
#include "mlp_ref.h"
using mvm_fmlp::NT;

constexpr float LR = 0.05f;
constexpr int MAXPH = 2 * MVM_BATCH_MAXL + 3;
extern __shared__ __align__(128) float smem[];

struct Shape { int V, E, H, T, L; };   // what mlp_ref reads

template <bool GRID>   // the grid step's arithmetic or not: a kernel and a call tree each (mvm_fused_batch.cuh: run_item)
__global__ void __launch_bounds__(NT, 1) batch_kernel(const MvmBatchDesc* gd, int steps, Timing tm) {
  const MvmBatchDesc d = *gd;
  if (threadIdx.x == 0) tm.smid[blockIdx.x] = sm_id();
  const int np = mvm_batch_phases(d.L);
  __shared__ long long ph_acc[MAXPH];
  __shared__ long long last;
  if (threadIdx.x < MAXPH) ph_acc[threadIdx.x] = 0;
  unsigned long long at = mvm_gang::cross_start(d.ctl, blockIdx.x, d.members);   // this member's place in the gang's crossings
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) { tm.gtime[0] = global_ns(); last = clock64(); }
  auto ph = [&](int p) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { const long long n = clock64(); ph_acc[p] += n - last; last = n; }
  };
  for (int s = 0; s < steps; ++s)
    if (!mvm_fbatch::batch_step<GRID>(blockIdx.x, d, s, smem, at, nullptr, ph)) break;   // a bed has no error word: a crossing cannot abort
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    tm.gtime[1] = global_ns();
    for (int p = 0; p < np; ++p) tm.phase[p] = ph_acc[p];
  }
}

int main(int argc, char** argv) {
  MvmBatchDesc d{}; d.E = 48; d.V = 100; d.H = 640; d.L = 4; d.T = MVM_BATCH_T; d.R = 16; d.members = 0; d.lr = LR;
  int steps = 50, check = 3, warm_ms = 500, bwd = -1, fm = 0, nw = 0, xfm = 0, xnw = 0, wfm = 0, wnw = 0, kw = 0, grid_p = 0; unsigned seed = 1; std::string dump;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) { fprintf(stderr, "option %s needs a value\n", a.c_str()); return 2; }
    if (a == "--bwd") { const std::string m = argv[++i]; bwd = m == "fused" ? 0 : m == "split" ? 1 : m == "auto" ? -1 : -2;
                        if (bwd == -2) { fprintf(stderr, "--bwd takes auto, fused or split\n"); return 2; } continue; }
    if (a == "--dump") { dump = argv[++i]; continue; }
    const int v = atoi(argv[++i]);
    if (a == "--hidden") d.H = v; else if (a == "--depth") d.L = v; else if (a == "--runs") d.R = v; else if (a == "--members") d.members = v;
    else if (a == "--virtual") d.Sv = v;
    else if (a == "--grid") grid_p = v;
    else if (a == "--fm") fm = v; else if (a == "--nw") nw = v; else if (a == "--xfm") xfm = v; else if (a == "--xnw") xnw = v;
    else if (a == "--wfm") wfm = v; else if (a == "--wnw") wnw = v; else if (a == "--kw") kw = v;
    else if (a == "--steps") steps = v; else if (a == "--check") check = v; else if (a == "--seed") seed = (unsigned)v; else if (a == "--warm-ms") warm_ms = v;
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  if (d.members <= 0) d.members = prop.multiProcessorCount;
  if (grid_p) {   // the grid step's arithmetic at a gang of grid_p members: its partition, as mvm_grid_layout takes it (grid_bed --members grid_p)
    if (d.Sv) { fprintf(stderr, "--grid and --virtual exclude each other: a shape's arithmetic is the split step's or the grid step's\n"); return 2; }
    if (bwd == 1) { fprintf(stderr, "--grid runs the fused backward only (--bwd split is the split step's tiles)\n"); return 2; }
    MvmGridDesc g{}; g.V = d.V; g.E = d.E; g.L = d.L; g.H = d.H; g.T = d.T; g.P = grid_p;
    if (mvm_grid_layout(g) < 0) { fprintf(stderr, "--grid %d: a shape the grid step does not take (mvm_grid_layout)\n", grid_p); return 2; }
    d.gHp = g.Hp; d.gka = g.ka; d.gnb = g.nb; d.gnb0 = g.nb0; d.gkah = g.kah;
    printf("grid arithmetic at %d members: Hp %d, layer 0 in %d column blocks, hidden %d x %d, head %d row blocks\n", grid_p, g.Hp, g.nb0, g.ka, g.nb, g.kah);
  }
  const int floats = mvm_batch_layout(d);
  if (floats < 0) { fprintf(stderr, "a shape the batched step does not take (mvm_batch_layout): H a multiple of 8 up to 4096, E 48, V up to 112, T 96, depth 1-%d, runs 1-%d, --virtual 0-%d\n", MVM_BATCH_MAXL, MVM_BATCH_MAXR, MVM_BATCH_MAXS); return 2; }
  if (fm || nw || xfm || xnw || wfm || wnw || kw) {   // forced tile shapes: recompute what depends on them (the split backward's follow fm and nw unless given)
    auto row_blocks = [](int x) { return x == 1 || x == 2 || x == 3 || x == 6; };
    auto width = [](int x) { return x == 16 || x == 32 || x == 64 || x == 128; };
    if ((fm && !row_blocks(fm)) || (xfm && !row_blocks(xfm)) || (wfm && !row_blocks(wfm))) { fprintf(stderr, "--fm, --xfm and --wfm take 1, 2, 3 or 6\n"); return 2; }
    if ((nw && !width(nw)) || (xnw && !width(xnw)) || (wnw && !width(wnw))) { fprintf(stderr, "--nw, --xnw and --wnw take 16, 32, 64 or 128\n"); return 2; }
    if (kw && kw != 16 && kw != 32 && kw != 64) { fprintf(stderr, "--kw takes 16, 32 or 64\n"); return 2; }
    if (grid_p && mvm_batch_warp_rb(fm ? fm : d.fm, (nw ? nw : d.nw) / 16) > MVM_BATCH_GRID_MAXRB) {   // the grid path's row blocks a warp (mvm_batch_layout)
      fprintf(stderr, "--grid takes forward tiles of at most %d row blocks of 16 a warp (fm 6 with nw <= 64, fm 3 with nw <= 128, fm 1-2 any)\n", MVM_BATCH_GRID_MAXRB); return 2; }
    if (kw) { d.kw = kw; d.kt = (d.H + kw - 1) / kw; }
    if (fm) { d.fm = fm; d.fr = 6 / fm; }
    if (nw) { d.nw = nw; d.nt = (d.H + nw - 1) / nw; }
    d.xfm = xfm ? xfm : d.fm; d.xnw = xnw ? xnw : d.nw; d.wfm = wfm ? wfm : d.fm; d.wnw = wnw ? wnw : d.nw;
    d.smem_floats = mvm_batch_smem(d.nw, d.fm, d.kw, d.xfm, d.xnw, d.wfm, d.wnw);
  }
  const int floats_used = d.smem_floats;
  if (bwd >= 0) d.bsplit = bwd;   // both modes fit the layout's shared memory (mvm_batch_smem takes the larger)
  const size_t smem_bytes = (size_t)floats_used * sizeof(float);
  const int np = mvm_batch_phases(d.L);
  printf("%s, %d SMs, %zu KB shared memory a block (opt-in); model E %d V %d H %d depth %d, T %d; %d runs on one gang of %d members; "
         "forward tiles %d x %d rows x %d x %d columns a run, backward %s; %.1f KB a member; %d phases (crossings) a step\n",
         prop.name, prop.multiProcessorCount, (size_t)prop.sharedMemPerBlockOptin / 1024, d.E, d.V, d.H, d.L, d.T, d.R, d.members, d.fr, 16 * d.fm, d.nt, d.nw,
         d.bsplit ? ("split (dX tiles " + std::to_string(16 * d.xfm) + " x " + std::to_string(d.xnw) + ", then update tiles " + std::to_string(16 * d.wfm) + " x " +
                     std::to_string(d.wnw) + ")").c_str() : ("fused, " + std::to_string(d.kt) + " row blocks of " + std::to_string(d.kw) + " a run").c_str(),
         smem_bytes / 1024.0, np);
  printf("items a phase:");
  for (int p = 0; p < np; ++p) printf(" %d", mvm_batch_items(d, p));
  printf("\n");
  void (*const kernel)(const MvmBatchDesc*, int, Timing) = d.gHp > 0 ? batch_kernel<true> : batch_kernel<false>;
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kernel));
  if (smem_bytes + fa.sharedSizeBytes > prop.sharedMemPerBlockOptin) { printf("does not fit: %zu dynamic + %zu static > %zu bytes a block\n", smem_bytes, (size_t)fa.sharedSizeBytes, (size_t)prop.sharedMemPerBlockOptin); return 3; }
  CK(cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, NT, smem_bytes));
  if (occ * prop.multiProcessorCount < d.members) { printf("the gang (%d blocks) cannot be co-resident: %d a SM x %d SMs\n", d.members, occ, prop.multiProcessorCount); return 3; }

  const int nsteps = std::max(steps, check);
  d.steps = nsteps;
  const Shape sh{d.V, d.E, d.H, d.T, d.L};
  const mlp_ref::Data data = mlp_ref::draw(sh, seed, nsteps);
  auto dalloc = [](size_t n) { float* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(float))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(float))); return p; };
  int32_t *dids, *dtgt;
  CK(cudaMalloc(&dids, data.ids.size() * sizeof(int32_t))); CK(cudaMalloc(&dtgt, data.tgt.size() * sizeof(int32_t)));
  CK(cudaMemcpy(dids, data.ids.data(), data.ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dtgt, data.tgt.data(), data.tgt.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  const size_t TH = (size_t)d.T * d.H;
  // each step's tokens and targets (every run trains on the same), and each run's per-step loss pairs (the mean, then 1 / T), as pointer arrays:
  // the executor points the same fields at the bundle's own buffers of each step
  auto ptrs = [&](auto base, size_t stride) {
    using P = decltype(base);
    std::vector<P> h(nsteps); for (int s = 0; s < nsteps; ++s) h[s] = base + (size_t)s * stride;
    P* dp; CK(cudaMalloc(&dp, nsteps * sizeof(P))); CK(cudaMemcpy(dp, h.data(), nsteps * sizeof(P), cudaMemcpyHostToDevice)); return dp;
  };
  const int32_t* const* ids_s = ptrs((const int32_t*)dids, d.T);
  const int32_t* const* tgt_s = ptrs((const int32_t*)dtgt, d.T);
  std::vector<float*> loss_store(d.R);
  std::vector<MvmBatchRun> runs(d.R);
  for (int g = 0; g < d.R; ++g) {
    MvmBatchRun& r = runs[g];
    r.emb = dalloc(data.emb.size());
    for (int l = 0; l < d.L; ++l) { r.w[l] = dalloc(data.W[l].size()); r.b[l] = dalloc(d.H); r.y[l] = dalloc(TH); r.g[l] = dalloc(TH); }
    r.wh = dalloc(data.Wh.size());
    r.ids = ids_s; r.tgt = tgt_s;
    loss_store[g] = dalloc((size_t)nsteps * 2); r.loss = ptrs(loss_store[g], 2);
    r.x0 = dalloc((size_t)d.T * d.E); r.dlog = dalloc((size_t)d.T * d.VP); r.dz[0] = dalloc(TH); r.dz[1] = dalloc(TH); r.dz[2] = dalloc(TH);
    r.dx0 = dalloc((size_t)d.T * d.E); r.rloss = dalloc(d.T);
  }
  MvmBatchRun* druns; CK(cudaMalloc(&druns, runs.size() * sizeof(MvmBatchRun))); CK(cudaMemcpy(druns, runs.data(), runs.size() * sizeof(MvmBatchRun), cudaMemcpyHostToDevice));
  d.runs = druns;
  const size_t lines = mvm_gang::line_words(d.members) * sizeof(unsigned);
  CK(cudaMalloc(&d.ctl, lines)); CK(cudaMemset(d.ctl, 0, lines));   // once: the gang's control words need no reset between launches (mvm_gang::cross)
  if (d.gHp > 0) {   // the grid sums' walks, one word a piece (mvm_batch_grid_table)
    std::vector<unsigned> tab((size_t)mvm_batch_grid_words(d));
    const int words = mvm_batch_grid_table(d, tab.data());
    unsigned* dtab; CK(cudaMalloc(&dtab, (size_t)words * sizeof(unsigned))); CK(cudaMemcpy(dtab, tab.data(), (size_t)words * sizeof(unsigned), cudaMemcpyHostToDevice));
    d.gtab = dtab;
  }
  MvmBatchDesc* gd; CK(cudaMalloc(&gd, sizeof(MvmBatchDesc))); CK(cudaMemcpy(gd, &d, sizeof(MvmBatchDesc), cudaMemcpyHostToDevice));
  Timing tm{};
  CK(cudaMalloc(&tm.phase, MAXPH * sizeof(long long))); CK(cudaMalloc(&tm.gtime, 2 * sizeof(unsigned long long))); CK(cudaMalloc(&tm.smid, (size_t)d.members * sizeof(unsigned)));
  auto upload = [&]() {
    for (const MvmBatchRun& r : runs) {
      for (int l = 0; l < d.L; ++l) { CK(cudaMemcpy(r.w[l], data.W[l].data(), data.W[l].size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(r.b[l], data.b[l].data(), data.b[l].size() * 4, cudaMemcpyHostToDevice)); }
      CK(cudaMemcpy(r.wh, data.Wh.data(), data.Wh.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(r.emb, data.emb.data(), data.emb.size() * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(tm.phase, 0, MAXPH * sizeof(long long)));
  };
  auto launch = [&](int n) {
    int nn = n; void* args[] = {&gd, &nn, &tm};
    CK(cudaLaunchCooperativeKernel((void*)kernel, dim3(d.members), dim3(NT), args, smem_bytes, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  };
  // run g's model and losses after a launch, as one vector (weights, biases, head, embedding, then the first `k` losses)
  auto model_of = [&](int g, int k) {
    std::vector<float> out;
    auto get = [&](const float* p, size_t n) { const size_t o = out.size(); out.resize(o + n); CK(cudaMemcpy(out.data() + o, p, n * 4, cudaMemcpyDeviceToHost)); };
    for (int l = 0; l < d.L; ++l) { get(runs[g].w[l], data.W[l].size()); get(runs[g].b[l], d.H); }
    get(runs[g].wh, data.Wh.size()); get(runs[g].emb, data.emb.size());
    std::vector<float> pairs((size_t)k * 2);   // the first `k` steps' (mean, 1 / T) pairs: the means, as the dump has always held them
    if (k) CK(cudaMemcpy(pairs.data(), loss_store[g], pairs.size() * 4, cudaMemcpyDeviceToHost));
    for (int s = 0; s < k; ++s) out.push_back(pairs[(size_t)s * 2]);
    return out;
  };
  int bad = 0;
  if (check > 0) {   // the first `check` steps: run 0 against float64, every run against run 0
    upload(); launch(check);
    std::vector<float> loss(check), gW[mlp_ref::MAXL], gb[mlp_ref::MAXL], gWh(data.Wh.size()), gemb(data.emb.size());
    { std::vector<float> pairs((size_t)check * 2); CK(cudaMemcpy(pairs.data(), loss_store[0], pairs.size() * 4, cudaMemcpyDeviceToHost));
      for (int s = 0; s < check; ++s) loss[s] = pairs[(size_t)s * 2]; }
    for (int l = 0; l < d.L; ++l) { gW[l].resize(data.W[l].size()); gb[l].resize(d.H); CK(cudaMemcpy(gW[l].data(), runs[0].w[l], gW[l].size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gb[l].data(), runs[0].b[l], gb[l].size() * 4, cudaMemcpyDeviceToHost)); }
    CK(cudaMemcpy(gWh.data(), runs[0].wh, gWh.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gemb.data(), runs[0].emb, gemb.size() * 4, cudaMemcpyDeviceToHost));
    mlp_ref::Model m = mlp_ref::model_of(data, d.L);
    const std::vector<double> rl = mlp_ref::ref_steps(sh, m, data.ids, data.tgt, check, LR);
    double worst = 0;
    for (int st = 0; st < check; ++st) { worst = std::max(worst, std::fabs(loss[st] - rl[st]) / std::fabs(rl[st])); printf("check step %d: loss %.6f, fp64 %.6f\n", st, loss[st], rl[st]); }
    double eW = 0, eb = 0;
    for (int l = 0; l < d.L; ++l) { eW = std::max(eW, mlp_ref::rel_delta(gW[l], m.W[l], data.W[l])); eb = std::max(eb, mlp_ref::rel_delta(gb[l], m.b[l], data.b[l])); }
    const double eWh = mlp_ref::rel_delta(gWh, m.Wh, data.Wh), eemb = mlp_ref::rel_delta(gemb, m.emb, data.emb);
    printf("check after %d steps: loss max rel %.2e; update rel L2 (worst layer) W %.2e, b %.2e; Wh %.2e, emb %.2e (TF32 operands: a few 1e-3 expected)\n", check, worst, eW, eb, eWh, eemb);
    bad = !(worst <= 1e-3 && eW <= 2e-2 && eb <= 2e-2 && eWh <= 2e-2 && eemb <= 2e-2);   // a NaN fails too
    if (d.R > 1) {
      const std::vector<float> m0 = model_of(0, check);
      int differ = 0;
      for (int g = 1; g < d.R; ++g) { const std::vector<float> mg = model_of(g, check); differ += memcmp(mg.data(), m0.data(), m0.size() * 4) != 0; }
      printf("runs: %d of %d differ from run 0 (model and losses, bitwise) after %d steps\n", differ, d.R - 1, check);
      bad |= differ != 0;
    }
    printf("%s\n", bad ? "CHECK FAILED" : "check ok");
    if (!dump.empty()) {
      const std::vector<float> m0 = model_of(0, check);
      FILE* f = fopen(dump.c_str(), "wb");
      if (!f || fwrite(m0.data(), sizeof(float), m0.size(), f) != m0.size()) { fprintf(stderr, "cannot write %s\n", dump.c_str()); return 2; }
      fclose(f);
      printf("dumped run 0's model and %d losses: %zu floats to %s\n", check, m0.size(), dump.c_str());
    }
  }
  // timing: warm launches until warm_ms have passed (the SM clock ramps for about a second under sustained work, measured on the A100), then the timed one
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  {
    float warmed = 0.f; int launches = 0;
    while (warmed < (float)warm_ms || launches == 0) {
      upload(); CK(cudaEventRecord(e0)); launch(std::min(steps, 20)); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float t = 0.f; CK(cudaEventElapsedTime(&t, e0, e1)); warmed += t; ++launches;
    }
    printf("warm: %d launches, %.0f ms\n", launches, warmed);
  }
  upload();
  {   // the item clock counts the timed launch only
    long long z[CK_KINDS][3] = {}; int zn[CK_KINDS] = {}; long long zc[CK_KINDS][2][4] = {};
    CK(cudaMemcpyToSymbol(g_item_clk, z, sizeof(z))); CK(cudaMemcpyToSymbol(g_item_n, zn, sizeof(zn))); CK(cudaMemcpyToSymbol(g_chunk_clk, zc, sizeof(zc)));
  }
  CK(cudaEventRecord(e0)); launch(steps); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  long long hph[MAXPH]; CK(cudaMemcpy(hph, tm.phase, sizeof(hph), cudaMemcpyDeviceToHost));
  unsigned long long hgt[2]; CK(cudaMemcpy(hgt, tm.gtime, sizeof(hgt), cudaMemcpyDeviceToHost));
  long long tot = 0; for (int p = 0; p < np; ++p) tot += hph[p];
  const double kern_us = (hgt[1] - hgt[0]) / 1e3, rated = prop.clockRate / 1e3, eff = (double)tot / std::max(1e-9, kern_us);
  printf("timed: %d steps: %.2f us a step in the kernel (block 0's %%globaltimer over its loop), %.2f us by events (launch and host delays included)\n", steps, kern_us / steps, 1e3 * ms / steps);
  printf("block 0 by phase (cycles a step; each phase ends at its crossing, which its time includes):\n");
  for (int p = 0; p < np; ++p) {
    std::string name;
    if (p < d.L) name = "F" + std::to_string(p) + (p ? " hidden layer" : " layer 0 (gather)");
    else if (p == d.L) name = "HC head + cross-entropy";
    else if (p == d.L + 1) name = "BH head backward";
    else if (p <= 2 * d.L) name = "B" + std::to_string(2 * d.L + 1 - p) + " hidden backward";
    else if (p == 2 * d.L + 1) name = "BX dX0";
    else name = "BU W0, b0, embedding, loss";
    printf("  %-30s %3d items  %9.0f  (%4.1f%%)\n", name.c_str(), mvm_batch_items(d, p), (double)hph[p] / steps, 100.0 * hph[p] / std::max(1ll, tot));
  }
  printf("  %-30s            %9.0f cycles\n", "total", (double)tot / steps);
  {
    long long clk[CK_KINDS][3]; int nit[CK_KINDS];
    CK(cudaMemcpyFromSymbol(clk, g_item_clk, sizeof(clk))); CK(cudaMemcpyFromSymbol(nit, g_item_n, sizeof(nit)));
    const char* kinds[CK_KINDS] = {"F (forward tile)", "HC (head + CE)", "B (backward block)", "BX", "W0 tile", "embedding"};
    printf("block 0's items, cycles an item (thread 0's clock: operand loop | epilogue | end barrier):\n");
    for (int k = 0; k < CK_KINDS; ++k)
      if (nit[k]) printf("  %-20s %6d items  %9.0f | %9.0f | %9.0f\n", kinds[k], nit[k], (double)clk[k][0] / nit[k], (double)clk[k][1] / nit[k], (double)clk[k][2] / nit[k]);
    long long cc[CK_KINDS][2][4]; CK(cudaMemcpyFromSymbol(cc, g_chunk_clk, sizeof(cc)));
    printf("block 0's k chunks, cycles a chunk (warp 0 / warp 7: wait + barrier | copies | k-steps):\n");
    for (int k = 0; k < CK_KINDS; ++k)
      for (int w = 0; w < 2; ++w)
        if (cc[k][w][3]) printf("  %-20s warp %d %7lld chunks  %7.0f | %7.0f | %7.0f\n", kinds[k], 7 * w, cc[k][w][3], (double)cc[k][w][0] / cc[k][w][3],
                                (double)cc[k][w][1] / cc[k][w][3], (double)cc[k][w][2] / cc[k][w][3]);
  }
  {
    std::vector<unsigned> sids((size_t)d.members); CK(cudaMemcpy(sids.data(), tm.smid, sids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::sort(sids.begin(), sids.end());
    const int distinct = (int)(std::unique(sids.begin(), sids.end()) - sids.begin());
    printf("blocks ran on %d distinct SMs of %d blocks%s\n", distinct, d.members, distinct < d.members ? ": SOME SHARED AN SM" : " (one block a SM)");
  }
  printf("effective SM clock %.0f MHz (block 0's cycles over its kernel time), rated %.0f MHz\n", eff, rated);
  return bad;
}
