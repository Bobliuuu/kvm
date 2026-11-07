// kvm: "kittens"-style GPU virtual machine, v2 prototype (sm_80).
//
// ONE persistent kernel interprets an instruction tape. The tape encodes an ENTIRE training run
// (every step: fwd, loss, bwd, SGD) of a 2-layer MLP. No kernel boundaries between ops or steps:
// instructions synchronise through monotonic global counters (fine-grained, per row-block),
// so step s+1's first tiles can start while step s's tail is still running.
//
// Baselines execute the *same* tape with the classic model: one launch per op-phase, and the same
// launches captured in a CUDA graph.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 kvm.cu -o kvm
// Run:   ./kvm [--steps 1000] [--B 128] [--D 128] [--H 256] [--O 64] [--nb 8] [--lr 0.05] [--ref-steps 100]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

enum { OP_GEMM = 0, OP_SGD = 1, OP_SEL = 2 };
enum { F_TA = 1, F_TB = 2, F_RELU = 4, F_MSE = 8, F_RELUMASK = 16, F_SGD = 32 };

struct Buf { float* p; int ld; };
__constant__ Buf g_buf[512];


struct Instr {                 // 80 bytes
  int op, flags;
  int a, b, c, aux;            // buffer ids
  int m0, n0, K;
  float scale;
  int w0c, w0v, w1c, w1v;      // wait: counter id (-1 = none) >= value
  int s0, s1;                  // signal: counter ids to increment (-1 = none)
  int loss_idx, xflags, run;   // xflags: 1/2 = w0/w1 wait on previous step, 4 = a is X(step%NB), 8 = aux is T(step%NB)
};

struct KP { int SC, NB, perStep; };   // SC = counters per step (all runs), NB = data batches, perStep = template length
__constant__ KP g_kp;

// Loop opcode: the tape stores ONE step as a template; step s instantiates it on the device.
__device__ __forceinline__ Instr inst_at(const Instr* tape, int t, int s) {
  Instr in = tape[t];
  int xf = in.xflags;
  if (xf & 1) { in.w0c = s ? in.w0c + (s - 1) * g_kp.SC : -1; } else if (in.w0c >= 0) in.w0c += s * g_kp.SC;
  if (xf & 2) { in.w1c = s ? in.w1c + (s - 1) * g_kp.SC : -1; } else if (in.w1c >= 0) in.w1c += s * g_kp.SC;
  if (in.s0 >= 0) in.s0 += s * g_kp.SC;
  if (in.s1 >= 0) in.s1 += s * g_kp.SC;
  if (xf & 4) in.a += s % g_kp.NB;
  if (xf & 8) in.aux += s % g_kp.NB;
  in.loss_idx += s;
  return in;
}

#ifndef KTILE
#define KTILE 64
#endif
constexpr int TILE = KTILE, NT = 128, WT = TILE / 2, FR = WT / 16, VPT = TILE / 4;   // warp block, frags/dim, 16B vectors per TILE-wide row

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

// ---- pipelined 64x64 tile GEMM: STG-stage cp.async ring of K-chunks in dynamic smem.
// 4 warps in a 2x2 grid, each owning a 32x32 output block (2x2 wmma 16x16x8 TF32 fragments).
#ifndef KKC
#define KKC 32
#endif
#ifndef KSTG
#define KSTG 3
#endif
constexpr int KC = KKC, STG = KSTG, VPK = KC / 4, LDK = KC + 4, LDM = TILE + 4, SLOT = (TILE * LDK > KC * LDM ? TILE * LDK : KC * LDM), NV = TILE * KC / 4 / NT;
constexpr size_t DSMEM = (size_t)STG * 2 * SLOT * sizeof(float);
extern __shared__ __align__(128) float dsm[];
__shared__ unsigned long long g_stamp[2];
__device__ __forceinline__ unsigned long long gtime();
__device__ bool g_prof_on = false;

__device__ __forceinline__ void cp16(float* dst, const float* src) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_stg() { asm volatile("cp.async.wait_group %0;" :: "n"(STG - 2) : "memory"); }

template <bool TA, bool TB>
__device__ __forceinline__ void load_chunk(float* sA, float* sB, const Buf& A, const Buf& B, int m0, int n0, int k0) {
#pragma unroll
  for (int j = 0; j < NV; j++) {
    int i = threadIdx.x + j * NT;      // TILE*KC/4 16-byte vectors per operand chunk
    if (TA) { int r = i / VPT, c = (i % VPT) * 4; cp16(sA + r * LDM + c, A.p + (size_t)(k0 + r) * A.ld + m0 + c); }
    else    { int r = i / VPK, c = (i % VPK) * 4;    cp16(sA + r * LDK + c, A.p + (size_t)(m0 + r) * A.ld + k0 + c); }
    if (TB) { int r = i / VPK, c = (i % VPK) * 4;    cp16(sB + r * LDK + c, B.p + (size_t)(n0 + r) * B.ld + k0 + c); }
    else    { int r = i / VPT, c = (i % VPT) * 4; cp16(sB + r * LDM + c, B.p + (size_t)(k0 + r) * B.ld + n0 + c); }
  }
}

template <bool TA, bool TB>
__device__ void gemm_acc(const Buf A, const Buf B, int m0, int n0, int K, float* sm) {
  int warp = threadIdx.x >> 5, wm = warp >> 1, wn = warp & 1;
  using AL = typename std::conditional<TA, wmma::col_major, wmma::row_major>::type;
  using BL = typename std::conditional<TB, wmma::col_major, wmma::row_major>::type;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, AL> fa[FR];
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, BL> fb[FR];
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[FR][FR];
#pragma unroll
  for (int i = 0; i < FR; i++)
#pragma unroll
    for (int j = 0; j < FR; j++) wmma::fill_fragment(acc[i][j], 0.f);
  int nch = K / KC;
  for (int s = 0; s < STG - 1; s++) {
    if (s < nch) load_chunk<TA, TB>(dsm + s * 2 * SLOT, dsm + s * 2 * SLOT + SLOT, A, B, m0, n0, s * KC);
    cp_commit();
  }
  for (int c = 0; c < nch; c++) {
    cp_wait_stg();
    __syncthreads();
    if (c == 0 && threadIdx.x == 0 && g_prof_on) g_stamp[0] = gtime();
    int nx = c + STG - 1;
    if (nx < nch) { float* p = dsm + (nx % STG) * 2 * SLOT; load_chunk<TA, TB>(p, p + SLOT, A, B, m0, n0, nx * KC); }
    cp_commit();
    const float* sA = dsm + (c % STG) * 2 * SLOT; const float* sB = sA + SLOT;
#pragma unroll
    for (int kk = 0; kk < KC; kk += 8) {
#pragma unroll
      for (int i = 0; i < FR; i++) {
        wmma::load_matrix_sync(fa[i], TA ? sA + kk * LDM + wm * WT + i * 16 : sA + (wm * WT + i * 16) * LDK + kk, TA ? LDM : LDK);
#pragma unroll
        for (int e = 0; e < fa[i].num_elements; e++) fa[i].x[e] = wmma::__float_to_tf32(fa[i].x[e]);
      }
#pragma unroll
      for (int j = 0; j < FR; j++) {
        wmma::load_matrix_sync(fb[j], TB ? sB + (wn * WT + j * 16) * LDK + kk : sB + kk * LDM + wn * WT + j * 16, TB ? LDK : LDM);
#pragma unroll
        for (int e = 0; e < fb[j].num_elements; e++) fb[j].x[e] = wmma::__float_to_tf32(fb[j].x[e]);
      }
#pragma unroll
      for (int i = 0; i < FR; i++)
#pragma unroll
        for (int j = 0; j < FR; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
  }
  cp_wait_stg(); asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();   // ring reads done before the next instruction reuses it
  if (threadIdx.x == 0 && g_prof_on) g_stamp[1] = gtime();
#pragma unroll
  for (int i = 0; i < FR; i++)
#pragma unroll
    for (int j = 0; j < FR; j++) wmma::store_matrix_sync(sm + (wm * WT + i * 16) * TILE + wn * WT + j * 16, acc[i][j], TILE, wmma::mem_row_major);
}

__device__ float g_lr[64];   // per-run learning rate, changed in-kernel by PBT explore
// ---- in-kernel hyperparameter control: successive halving (SH). A rung boundary is a SEL slot at the end of step E*e-1:
// the block that claims it waits for every alive run's last step, ranks them by mean loss over the last `win` steps, and
// retires the worse half (g_alive[r]=0). Retired runs' later tiles are claimed, see the flag, and are skipped without waits.
struct SHP { int E, K, R, CPS, nRB, needDW1, needDW2, steps, win, mode; float target, divthr, prune;
             int useEval, evLen, needYV, evbase, bufW, nW1, nW2; };
__constant__ SHP g_sh;
__device__ volatile int g_alive[64];
__device__ int g_kill[64];   // rung index at which a run was retired (0 = never)
__device__ int g_srchist[64 * 64];   // PBT: g_srchist[e*64+r] = run whose weights r copied at event e (-1 = none)
// Rung decision (one thread). mode 0 successive halving, 1 retire rules, 2 PBT (exploit bottom quarter <- top quarter, explore lr x0.8/x1.2).
// Metric = mean train loss over the last `win` steps, or (useEval) the held-out loss computed in-kernel by the event's EV tiles.
__device__ void do_select(int e, int stp, const int* ctr, const float* loss, int* s_src) {
  const int R = g_sh.R, pitch = g_sh.nRB + 1;
  for (int r = 0; r < R; r++) if (g_alive[r]) {
    if (g_sh.useEval) { while (ld_acquire(ctr + g_sh.evbase + ((e - 1) * R + r) * 2 + 1) < g_sh.needYV) __nanosleep(32); }
    else {
      int base = (stp * R + r) * g_sh.CPS;
      while (ld_acquire(ctr + base + 4 * pitch + g_sh.nRB) < g_sh.needDW1) __nanosleep(32);
      while (ld_acquire(ctr + base + 3 * pitch + g_sh.nRB) < g_sh.needDW2) __nanosleep(32);
    }
  }
  float m[64]; int n = 0;
  for (int r = 0; r < R; r++) {
    float a = 0.f;
    if (g_sh.useEval) a = __ldcg(loss + (size_t)R * g_sh.steps + (e - 1) * R + r);
    else { for (int j = 0; j < g_sh.win; j++) a += __ldcg(loss + (size_t)r * g_sh.steps + stp - j); a /= g_sh.win; }
    m[r] = (a == a) ? a : 3.4e38f;   // NaN ranks last
    n += g_alive[r] ? 1 : 0;
    s_src[r] = -1;
  }
  int nal[64];
  if (g_sh.mode == 0) {
    int keep = n / 2 > 0 ? n / 2 : 1;
    for (int r = 0; r < R; r++) {
      int rank = 0;
      if (g_alive[r]) for (int j = 0; j < R; j++) if (g_alive[j] && (m[j] < m[r] || (m[j] == m[r] && j < r))) rank++;
      nal[r] = g_alive[r] && rank < keep;
    }
  } else if (g_sh.mode == 1) {
    float best = 3.4e38f;
    for (int r = 0; r < R; r++) if (g_alive[r] && m[r] < best) best = m[r];
    for (int r = 0; r < R; r++) nal[r] = g_alive[r] && m[r] < g_sh.divthr && m[r] >= g_sh.target && m[r] <= g_sh.prune * best;
  } else {                // PBT: everyone stays alive
    int rk[64], who[64];
    for (int r = 0; r < R; r++) { int rank = 0; for (int j = 0; j < R; j++) if (m[j] < m[r] || (m[j] == m[r] && j < r)) rank++; rk[r] = rank; who[rank] = r; nal[r] = 1; }
    int nb = R / 4 > 0 ? R / 4 : 1;
    for (int r = 0; r < R; r++) if (rk[r] >= R - nb) {
      int w = who[rk[r] - (R - nb)];
      s_src[r] = w; g_lr[r] = g_lr[w] * ((r & 1) ? 1.2f : 0.8f);
    }
    for (int r = 0; r < R; r++) g_srchist[e * 64 + r] = s_src[r];
  }
  for (int r = 0; r < R; r++) if (g_alive[r] && !nal[r]) g_kill[r] = e;
  for (int r = 0; r < R; r++) g_alive[r] = nal[r];
}

// Executes one instruction. All NT threads. Caller handles dependencies. Ends with __syncthreads-safe state.
__device__ void exec_instr(const Instr& in, float* sm, float* loss) {
  __shared__ float red[NT / 32];
  if (in.op == OP_GEMM) {
    Buf A = g_buf[in.a], B = g_buf[in.b], C = g_buf[in.c];
    int f = in.flags;
    if (f & F_TA) { if (f & F_TB) gemm_acc<true, true>(A, B, in.m0, in.n0, in.K, sm); else gemm_acc<true, false>(A, B, in.m0, in.n0, in.K, sm); }
    else          { if (f & F_TB) gemm_acc<false, true>(A, B, in.m0, in.n0, in.K, sm); else gemm_acc<false, false>(A, B, in.m0, in.n0, in.K, sm); }
    __syncthreads();
    float lsum = 0.f;
    Buf X = g_buf[in.aux >= 0 ? in.aux : 0];
    // batched float4 epilogue: issue all global loads first, then compute + store
    constexpr int EPI = TILE * TILE / 4 / NT;
    float4 v[EPI], xa[EPI], cu[EPI]; size_t gi[EPI], xi[EPI];
#pragma unroll
    for (int it = 0; it < EPI; it++) {
      int e4 = threadIdx.x + it * NT, r = e4 / VPT, c = (e4 % VPT) * 4;
      v[it] = *reinterpret_cast<const float4*>(sm + r * TILE + c);
      gi[it] = (size_t)(in.m0 + r) * C.ld + in.n0 + c;
      xi[it] = (size_t)(in.m0 + r) * X.ld + in.n0 + c;
      if (f & (F_RELUMASK | F_MSE)) xa[it] = __ldcg(reinterpret_cast<const float4*>(X.p + xi[it]));
      if (f & F_SGD) cu[it] = __ldcg(reinterpret_cast<const float4*>(C.p + gi[it]));
    }
#pragma unroll
    const float sgd_lr = (f & F_SGD) ? __ldcg(&g_lr[in.run]) : 0.f;
    for (int it = 0; it < EPI; it++) {
      float* q = &v[it].x; const float* x = &xa[it].x; const float* cc = &cu[it].x;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float t = q[j];
        if (f & F_SGD) t = cc[j] - sgd_lr * t;
        if (f & F_RELU) t = fmaxf(t, 0.f);
        if (f & F_RELUMASK) t = (x[j] > 0.f) ? t : 0.f;
        if (f & F_MSE) { float d = t - x[j]; lsum += d * d * in.scale * 0.5f; t = d * in.scale; }
        q[j] = t;
      }
      *reinterpret_cast<float4*>(C.p + gi[it]) = v[it];
    }
    if (f & F_MSE) {
      for (int o = 16; o; o >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
      if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lsum;
      __syncthreads();
      if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < NT / 32; i++) t += red[i]; atomicAdd(loss + in.loss_idx, t); }
    }
  } else {  // OP_SGD: W(c) -= scale * G(a)
    Buf W = g_buf[in.c], G = g_buf[in.a];
    for (int e = threadIdx.x; e < TILE * TILE; e += NT) {
      int r = e / TILE, c = e % TILE;
      float* wp = W.p + (size_t)(in.m0 + r) * W.ld + in.n0 + c;
      *wp = __ldcg(wp) - in.scale * __ldcg(G.p + (size_t)(in.m0 + r) * G.ld + in.n0 + c);
    }
  }
}

// ---- the VM: persistent, dynamic tape claim, counter-based deps
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void __launch_bounds__(NT) vm_kernel(const Instr* tape, const Instr* evtape, int n, int* head, int* ctr, float* loss, int* selctr) {
  __shared__ __align__(128) float sm[TILE * TILE];
  __shared__ int s_idx, s_dead, s_src[64];
  for (;;) {
    if (threadIdx.x == 0) s_idx = atomicAdd(head, 1);
    __syncthreads();
    int idx = s_idx;
    if (idx >= n) return;
    const int E = g_sh.E, K = g_sh.K, ps = g_kp.perStep, segLen = E * ps + g_sh.evLen;
    // tape layout: K x [E steps of the template, then one event (optional EV tiles + SEL)], then the remaining steps
    int stp = 0, epoch, ev = 0; Instr in;
    int seg = K > 0 ? idx / segLen : 0;
    if (seg < K) {
      int off = idx - seg * segLen; epoch = seg;
      if (off < E * ps) { stp = seg * E + off / ps; in = inst_at(tape, off % ps, stp); }
      else { ev = seg + 1; in = evtape[seg * g_sh.evLen + (off - E * ps)]; }
    } else {
      int off = K > 0 ? idx - K * segLen : idx; epoch = K; stp = K * E + off / ps; in = inst_at(tape, off % ps, stp);
    }
    if (in.op == OP_SEL) {
      if (threadIdx.x == 0) { if (ev > 1) while (ld_acquire(selctr + ev - 1) < 1) __nanosleep(32); do_select(ev, ev * E - 1, ctr, loss, s_src); }
      __syncthreads();
      if (g_sh.mode == 2) {
        for (int r = 0; r < g_sh.R; r++) if (s_src[r] >= 0) {
          const float4* a1 = reinterpret_cast<const float4*>(g_buf[g_sh.bufW + 5 * s_src[r]].p); float4* d1 = reinterpret_cast<float4*>(g_buf[g_sh.bufW + 5 * r].p);
          const float4* a2 = reinterpret_cast<const float4*>(g_buf[g_sh.bufW + 5 * s_src[r] + 1].p); float4* d2 = reinterpret_cast<float4*>(g_buf[g_sh.bufW + 5 * r + 1].p);
          for (int i = threadIdx.x; i < g_sh.nW1 / 4; i += NT) d1[i] = __ldcg(a1 + i);
          for (int i = threadIdx.x; i < g_sh.nW2 / 4; i += NT) d2[i] = __ldcg(a2 + i);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) { __threadfence(); atomicAdd(selctr + ev, 1); }
      continue;
    }
    if (threadIdx.x == 0) {
      int dead = 0;
      if (epoch > 0) { while (ld_acquire(selctr + epoch) < 1) __nanosleep(32); dead = !g_alive[in.run]; }
      if (!dead) {
        if (in.w0c >= 0) while (ld_acquire(ctr + in.w0c) < in.w0v) __nanosleep(32);
        if (in.w1c >= 0) while (ld_acquire(ctr + in.w1c) < in.w1v) __nanosleep(32);
        __threadfence();
      }
      s_dead = dead;
    }
    __syncthreads();
    if (s_dead) continue;
    exec_instr(in, sm, loss);
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      if (in.s0 >= 0) atomicAdd(ctr + in.s0, 1);
      if (in.s1 >= 0) atomicAdd(ctr + in.s1, 1);
    }
  }
}

// ================= host =================
struct Rng { uint64_t s; float next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 40) & 0xFFFFFF) / 16777216.f * 2.f - 1.f; } };

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }


static void dump(const std::string& dir, const char* name, const void* p, size_t bytes) {
  std::string f = dir + "/" + name; FILE* fp = fopen(f.c_str(), "wb"); if (!fp) { fprintf(stderr, "cannot write %s\n", f.c_str()); exit(1); } fwrite(p, 1, bytes, fp); fclose(fp);
}

int main(int argc, char** argv) {
  int steps = 300, B = 128, D = 128, Hd = 256, O = 64, NB = 4, R = 32, E = 75, K = 3, reps = 7;
  float lrlo = 0.01f, lrhi = 0.6f, target = 0.f, divthr = 1.f, prune = 1e9f; int mode = 0, useEval = 0; std::string out = "";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return atof(argv[++i]); };
    if (a == "--steps") steps = nx(); else if (a == "--B") B = nx(); else if (a == "--D") D = nx();
    else if (a == "--H") Hd = nx(); else if (a == "--O") O = nx(); else if (a == "--nb") NB = nx();
    else if (a == "--runs") R = nx(); else if (a == "--E") E = nx(); else if (a == "--K") K = nx(); else if (a == "--reps") reps = nx();
    else if (a == "--mode") mode = nx(); else if (a == "--eval") useEval = nx(); else if (a == "--target") target = nx(); else if (a == "--divthr") divthr = nx(); else if (a == "--prune") prune = nx();
    else if (a == "--lrlo") lrlo = nx(); else if (a == "--lrhi") lrhi = nx(); else if (a == "--out") out = argv[++i];
  }
  if (B % TILE || D % TILE || Hd % TILE || O % TILE || B % KC || D % KC || Hd % KC || O % KC) { fprintf(stderr, "dims must be multiples of max(TILE=%d, KC=%d)\n", TILE, KC); return 1; }
  if (R > 64 || E < NB || K * E >= steps) { fprintf(stderr, "need R<=64, E>=nb, K*E<steps\n"); return 1; }
  int nRB = B / TILE, nHC = Hd / TILE, nOC = O / TILE, nDC = D / TILE;
  std::vector<float> lrs(R); for (int r = 0; r < R; r++) lrs[r] = R == 1 ? lrlo : lrlo * powf(lrhi / lrlo, (float)r / (R - 1));

  Rng rng{12345};
  std::vector<float> W1(D * Hd), W2(Hd * O), Wt1(D * Hd), Wt2(Hd * O);
  for (auto& v : W1) v = rng.next() * sqrtf(3.f / D);
  for (auto& v : W2) v = rng.next() * sqrtf(3.f / Hd);
  for (auto& v : Wt1) v = rng.next() * sqrtf(3.f / D);
  for (auto& v : Wt2) v = rng.next() * sqrtf(3.f / Hd);
  std::vector<std::vector<float>> X(NB + 1, std::vector<float>(B * D)), T(NB + 1, std::vector<float>(B * O));   // index NB = held-out batch
  for (int k = 0; k <= NB; k++) {
    for (auto& v : X[k]) v = rng.next();
    std::vector<float> h(B * Hd, 0.f);
    for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) h[i * Hd + j] += x * Wt1[kk * Hd + j]; }
    for (auto& v : h) v = v > 0 ? v : 0;
    std::fill(T[k].begin(), T[k].end(), 0.f);
    for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) T[k][i * O + j] += x * Wt2[kk * O + j]; }
  }

  auto idX = [&](int k) { return k; };
  auto idT = [&](int k) { return NB + k; };
  auto idW1 = [&](int r) { return 2 * NB + 5 * r; };
  auto idW2 = [&](int r) { return 2 * NB + 5 * r + 1; };
  auto idH1 = [&](int r) { return 2 * NB + 5 * r + 2; };
  auto idDY = [&](int r) { return 2 * NB + 5 * r + 3; };
  auto idDH = [&](int r) { return 2 * NB + 5 * r + 4; };
  int bufEv = 2 * NB + 5 * R;   // XV, TV, then HV/YV per run
  auto idXV = [&]() { return bufEv; }; auto idTV = [&]() { return bufEv + 1; };
  auto idHV = [&](int r) { return bufEv + 2 + 2 * r; }; auto idYV = [&](int r) { return bufEv + 3 + 2 * r; };
  int nbuf = bufEv + 2 + 2 * R;
  if (nbuf > 512) { fprintf(stderr, "too many buffers\n"); return 1; }
  std::vector<Buf> hb(nbuf);
  auto dmalloc = [&](size_t n) { float* p; CK(cudaMalloc(&p, n * 4)); CK(cudaMemset(p, 0, n * 4)); return p; };
  for (int k = 0; k < NB; k++) {
    hb[idX(k)] = {dmalloc(B * D), D}; CK(cudaMemcpy(hb[idX(k)].p, X[k].data(), B * D * 4, cudaMemcpyHostToDevice));
    hb[idT(k)] = {dmalloc(B * O), O}; CK(cudaMemcpy(hb[idT(k)].p, T[k].data(), B * O * 4, cudaMemcpyHostToDevice));
  }
  hb[idXV()] = {dmalloc(B * D), D}; CK(cudaMemcpy(hb[idXV()].p, X[NB].data(), B * D * 4, cudaMemcpyHostToDevice));
  hb[idTV()] = {dmalloc(B * O), O}; CK(cudaMemcpy(hb[idTV()].p, T[NB].data(), B * O * 4, cudaMemcpyHostToDevice));
  for (int r = 0; r < R; r++) { hb[idHV(r)] = {dmalloc(B * Hd), Hd}; hb[idYV(r)] = {dmalloc(B * O), O}; }
  for (int r = 0; r < R; r++) {
    hb[idW1(r)] = {dmalloc(D * Hd), Hd}; hb[idW2(r)] = {dmalloc(Hd * O), O};
    hb[idH1(r)] = {dmalloc(B * Hd), Hd}; hb[idDY(r)] = {dmalloc(B * O), O}; hb[idDH(r)] = {dmalloc(B * Hd), Hd};
  }
  CK(cudaMemcpyToSymbol(g_buf, hb.data(), nbuf * sizeof(Buf)));

  constexpr int NPH = 5;
  enum { P_H1, P_Y, P_DH1, P_DW2, P_DW1 };
  int CPS = NPH * (nRB + 1);
  int SC = R * CPS;
  auto ctr = [&](int r, int s, int ph, int rb) { return (s * R + r) * CPS + ph * (nRB + 1) + (rb < 0 ? nRB : rb); };
  auto mk = [&](int r) { Instr i; memset(&i, 0, sizeof i); i.aux = -1; i.w0c = i.w1c = i.s0 = i.s1 = -1; i.run = r; return i; };
  auto emit_phase = [&](std::vector<Instr>& tape, int r, int ph) {
    const int s = 0; float rlr = lrs[r];
    if (ph == P_H1) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(r); i.op = OP_GEMM; i.flags = F_RELU; i.a = idX(0); i.b = idW1(r); i.c = idH1(r); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = D; i.xflags = 4 | 1 | 2;
      i.w0c = ctr(r, s, P_DW1, -1); i.w0v = nDC * nHC; i.w1c = ctr(r, s, P_DW2, -1); i.w1v = nHC * nOC;
      i.s0 = ctr(r, s, P_H1, rb); i.s1 = ctr(r, s, P_H1, -1); tape.push_back(i);
    }
    if (ph == P_Y) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nOC; cn++) {
      Instr i = mk(r); i.op = OP_GEMM; i.flags = F_MSE; i.a = idH1(r); i.b = idW2(r); i.c = idDY(r); i.aux = idT(0); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = Hd; i.xflags = 8 | 2;
      i.scale = 2.f / (B * O); i.loss_idx = r * steps;
      i.w0c = ctr(r, s, P_H1, rb); i.w0v = nHC; i.w1c = ctr(r, s, P_DW2, -1); i.w1v = nHC * nOC;
      i.s0 = ctr(r, s, P_Y, rb); i.s1 = ctr(r, s, P_Y, -1); tape.push_back(i);
    }
    if (ph == P_DH1) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(r); i.op = OP_GEMM; i.flags = F_TB | F_RELUMASK; i.a = idDY(r); i.b = idW2(r); i.c = idDH(r); i.aux = idH1(r); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = O;
      i.w0c = ctr(r, s, P_Y, rb); i.w0v = nOC; i.s1 = ctr(r, s, P_DH1, -1); tape.push_back(i);
    }
    if (ph == P_DW2) for (int m = 0; m < nHC; m++) for (int n = 0; n < nOC; n++) {
      Instr i = mk(r); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idH1(r); i.b = idDY(r); i.c = idW2(r); i.m0 = m * TILE; i.n0 = n * TILE; i.K = B; i.scale = rlr;
      i.w0c = ctr(r, s, P_Y, -1); i.w0v = nRB * nOC; i.w1c = ctr(r, s, P_DH1, -1); i.w1v = nRB * nHC;
      i.s1 = ctr(r, s, P_DW2, -1); tape.push_back(i);
    }
    if (ph == P_DW1) for (int m = 0; m < nDC; m++) for (int n = 0; n < nHC; n++) {
      Instr i = mk(r); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idX(0); i.b = idDH(r); i.c = idW1(r); i.m0 = m * TILE; i.n0 = n * TILE; i.K = B; i.scale = rlr; i.xflags = 4;
      i.w0c = ctr(r, s, P_DH1, -1); i.w0v = nRB * nHC; i.s1 = ctr(r, s, P_DW1, -1); tape.push_back(i);
    }
  };
  // step template: phase-major (phase p of every run, then phase p+1) then one SEL slot; keeps many runs in flight per phase
  std::vector<Instr> tapeP;
  for (int ph = 0; ph < NPH; ph++) for (int r = 0; r < R; r++) emit_phase(tapeP, r, ph);
  int perStep = (int)tapeP.size();
  int nctr = steps * SC;
  const int evbase = nctr;
  const int evLen = (useEval ? R * (nRB * nHC + nRB * nOC) : 0) + 1;
  std::vector<Instr> evT;
  for (int e = 1; e <= K; e++) {
    int stp = e * E - 1;
    if (useEval) {
      for (int r = 0; r < R; r++) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
        Instr i = mk(r); i.op = OP_GEMM; i.flags = F_RELU; i.a = idXV(); i.b = idW1(r); i.c = idHV(r); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = D;
        i.w0c = ctr(r, stp, P_DW1, -1); i.w0v = nDC * nHC; i.w1c = ctr(r, stp, P_DW2, -1); i.w1v = nHC * nOC; i.s1 = evbase + ((e - 1) * R + r) * 2; evT.push_back(i);
      }
      for (int r = 0; r < R; r++) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nOC; cn++) {
        Instr i = mk(r); i.op = OP_GEMM; i.flags = F_MSE; i.a = idHV(r); i.b = idW2(r); i.c = idYV(r); i.aux = idTV(); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = Hd;
        i.scale = 2.f / (B * O); i.loss_idx = R * steps + (e - 1) * R + r;
        i.w0c = evbase + ((e - 1) * R + r) * 2; i.w0v = nRB * nHC; i.s1 = evbase + ((e - 1) * R + r) * 2 + 1; evT.push_back(i);
      }
    }
    Instr i = mk(0); i.op = OP_SEL; evT.push_back(i);
  }
  if (evT.empty()) { Instr i = mk(0); i.op = OP_SEL; evT.push_back(i); }
  int nctrAll = nctr + (K + 1) * R * 2;
  auto ntape_for = [&](int k) { return perStep * steps + k * evLen; };
  int ntape = ntape_for(K);
  int tiles_per_runstep = perStep / R;
  printf("SH-KVM: B=%d D=%d H=%d O=%d runs=%d steps=%d rung every E=%d, K=%d rungs, %d tile instrs/run-step, tape %d executed instrs (%d-entry template), mode %d eval %d\n", B, D, Hd, O, R, steps, E, K, tiles_per_runstep, ntape, perStep, mode, useEval);

  Instr *dtapeP, *devtape; int *dhead, *dctr, *dsel; float* dloss;
  CK(cudaMalloc(&dtapeP, perStep * sizeof(Instr))); CK(cudaMemcpy(dtapeP, tapeP.data(), perStep * sizeof(Instr), cudaMemcpyHostToDevice));
  { KP kp{SC, NB, perStep}; CK(cudaMemcpyToSymbol(g_kp, &kp, sizeof kp)); }
  CK(cudaMalloc(&dhead, 4)); CK(cudaMalloc(&devtape, evT.size() * sizeof(Instr))); CK(cudaMemcpy(devtape, evT.data(), evT.size() * sizeof(Instr), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dctr, (size_t)nctrAll * 4)); CK(cudaMalloc(&dloss, ((size_t)R * steps + (K + 1) * R) * 4)); CK(cudaMalloc(&dsel, 64 * 4));
  CK(cudaFuncSetAttribute(vm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm_kernel, NT, DSMEM));
  int grid = prop.multiProcessorCount * occ;
  printf("%s  SMs=%d occ=%d  vm grid=%d\n", prop.name, prop.multiProcessorCount, occ, grid);

  auto set_sh = [&](int k) { SHP sh{E, k, R, CPS, nRB, nDC * nHC, nHC * nOC, steps, NB, mode, target, divthr, prune, useEval, evLen, nRB * nOC, evbase, 2 * NB, D * Hd, Hd * O}; CK(cudaMemcpyToSymbol(g_sh, &sh, sizeof sh)); };
  auto reset = [&]() {
    for (int r = 0; r < R; r++) {
      CK(cudaMemcpy(hb[idW1(r)].p, W1.data(), W1.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(hb[idW2(r)].p, W2.data(), W2.size() * 4, cudaMemcpyHostToDevice));
    }
    int al[64]; for (int r = 0; r < 64; r++) al[r] = 1; CK(cudaMemcpyToSymbol(g_alive, al, sizeof al)); int kz[64] = {0}; CK(cudaMemcpyToSymbol(g_kill, kz, sizeof kz)); { float lz[64] = {0}; for (int r = 0; r < R; r++) lz[r] = lrs[r]; CK(cudaMemcpyToSymbol(g_lr, lz, sizeof lz)); int sh[64 * 64]; for (int i = 0; i < 64 * 64; i++) sh[i] = -1; CK(cudaMemcpyToSymbol(g_srchist, sh, sizeof sh)); }
    CK(cudaMemset(dhead, 0, 4)); CK(cudaMemset(dsel, 0, 64 * 4)); CK(cudaMemset(dctr, 0, (size_t)nctrAll * 4)); CK(cudaMemset(dloss, 0, ((size_t)R * steps + (K + 1) * R) * 4));
    CK(cudaDeviceSynchronize());
  };
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto timed = [&](int k) { set_sh(k); reset(); cudaEventRecord(e0); vm_kernel<<<grid, NT, DSMEM>>>(dtapeP, devtape, ntape_for(k), dhead, dctr, dloss, dsel); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError()); float ms; cudaEventElapsedTime(&ms, e0, e1); return (double)ms; };
  struct Out { std::vector<float> L, w; int alive[64], kill[64], src[64 * 64]; float lr[64]; };
  auto fetch = [&]() {
    Out o; o.L.resize((size_t)R * steps + (K + 1) * R);
    CK(cudaMemcpy(o.L.data(), dloss, o.L.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpyFromSymbol(o.alive, g_alive, sizeof o.alive)); CK(cudaMemcpyFromSymbol(o.kill, g_kill, sizeof o.kill)); CK(cudaMemcpyFromSymbol(o.src, g_srchist, sizeof o.src)); CK(cudaMemcpyFromSymbol(o.lr, g_lr, sizeof o.lr));
    for (int r = 0; r < R; r++) { std::vector<float> a(W1.size()), b(W2.size());
      CK(cudaMemcpy(a.data(), hb[idW1(r)].p, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), hb[idW2(r)].p, b.size() * 4, cudaMemcpyDeviceToHost));
      o.w.insert(o.w.end(), a.begin(), a.end()); o.w.insert(o.w.end(), b.begin(), b.end()); }
    return o;
  };
  double best[2]; Out o[2]; std::vector<double> all[2];
  for (int mode = 0; mode < 2; mode++) {      // 0 = full population, no control (K=0); 1 = in-kernel successive halving
    int k = mode ? K : 0;
    timed(k);
    best[mode] = 1e30;
    for (int q = 0; q < reps; q++) { double t = timed(k); all[mode].push_back(t); best[mode] = std::min(best[mode], t); }
    o[mode] = fetch();
  }
  std::vector<double> s1 = all[1]; std::sort(s1.begin(), s1.end()); std::vector<double> s0 = all[0]; std::sort(s0.begin(), s0.end());
  printf("\nKVM full population (no halving): best %.3f ms  median %.3f ms\n", best[0], s0[s0.size() / 2]);
  printf("KVM in-kernel successive halving : best %.3f ms  median %.3f ms\n", best[1], s1[s1.size() / 2]);
  printf("halving saves %.2fx (best), %.2fx (median)\n", best[0] / best[1], s0[s0.size() / 2] / s1[s1.size() / 2]);
  printf("survivors:"); for (int r = 0; r < R; r++) if (o[1].alive[r]) printf(" %d(lr %.4f)", r, lrs[r]); printf("\nkill rung per run:"); for (int r = 0; r < R; r++) printf(" %d", o[1].kill[r]); printf("\n");
  int nl = 0; for (int r = 0; r < R; r++) { float l = o[1].L[(size_t)r * steps]; if (!(l == l)) nl++; }
  if (!out.empty()) {
    std::string cmd = "mkdir -p " + out; if (system(cmd.c_str())) return 1;
    FILE* m = fopen((out + "/meta.txt").c_str(), "w");
    fprintf(m, "B %d\nD %d\nH %d\nO %d\nNB %d\nR %d\nsteps %d\nE %d\nK %d\nwin %d\neval %d\nmode %d\ntarget %.9g\ndivthr %.9g\nprune %.9g\nkvm_full_ms %.4f\nkvm_sh_ms %.4f\n", B, D, Hd, O, NB, R, steps, E, K, NB, useEval, mode, target, divthr, prune, best[0], best[1]);
    for (int r = 0; r < R; r++) fprintf(m, "lr %.9g\n", lrs[r]); fclose(m);
    for (int k = 0; k < NB; k++) { dump(out, ("X" + std::to_string(k) + ".bin").c_str(), X[k].data(), X[k].size() * 4); dump(out, ("T" + std::to_string(k) + ".bin").c_str(), T[k].data(), T[k].size() * 4); }
    dump(out, "W1.bin", W1.data(), W1.size() * 4); dump(out, "W2.bin", W2.data(), W2.size() * 4);
    dump(out, "loss_sh.bin", o[1].L.data(), (size_t)R * steps * 4); dump(out, "evloss_sh.bin", o[1].L.data() + (size_t)R * steps, (size_t)(K + 1) * R * 4); dump(out, "src_sh.bin", o[1].src, sizeof o[1].src); dump(out, "lr_sh.bin", o[1].lr, sizeof o[1].lr);
    dump(out, "XV.bin", X[NB].data(), X[NB].size() * 4); dump(out, "TV.bin", T[NB].data(), T[NB].size() * 4); dump(out, "alive_sh.bin", o[1].alive, 64 * 4); dump(out, "kill_sh.bin", o[1].kill, 64 * 4); dump(out, "w_sh.bin", o[1].w.data(), o[1].w.size() * 4);
    dump(out, "loss_full.bin", o[0].L.data(), (size_t)R * steps * 4); dump(out, "w_full.bin", o[0].w.data(), o[0].w.size() * 4);
  }
  (void)nl;
  return 0;
}
