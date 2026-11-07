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
#include <type_traits>
#include <chrono>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

enum { OP_GEMM = 0, OP_SGD = 1 };
enum { F_TA = 1, F_TB = 2, F_RELU = 4, F_MSE = 8, F_RELUMASK = 16, F_SGD = 32 };

struct Buf { float* p; int ld; };
__constant__ Buf g_buf[256];


struct Instr {                 // 80 bytes
  int op, flags;
  int a, b, c, aux;            // buffer ids
  int m0, n0, K;
  float scale;
  int w0c, w0v, w1c, w1v;      // wait: counter id (-1 = none) >= value
  int s0, s1;                  // signal: counter ids to increment (-1 = none)
  int loss_idx, xflags, pad1;   // xflags: 1/2 = w0/w1 wait on previous step, 4 = a is X(step%NB), 8 = aux is T(step%NB)
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
    for (int it = 0; it < EPI; it++) {
      float* q = &v[it].x; const float* x = &xa[it].x; const float* cc = &cu[it].x;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float t = q[j];
        if (f & F_SGD) t = cc[j] - in.scale * t;
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

__global__ void __launch_bounds__(NT) vm_kernel(const Instr* tape, int n, int* head, int* ctr, float* loss, unsigned long long* prof) {   // n = steps * perStep
  __shared__ __align__(128) float sm[TILE * TILE];
  __shared__ int s_idx;
  for (;;) {
    if (threadIdx.x == 0) s_idx = atomicAdd(head, 1);
    __syncthreads();
    int idx = s_idx;
    if (idx >= n) return;
    int stp = idx / g_kp.perStep;
    Instr in = inst_at(tape, idx - stp * g_kp.perStep, stp);
    unsigned long long t0 = 0, t1 = 0, t2 = 0;
    if (prof) t0 = gtime();
    if (threadIdx.x == 0) {
      if (in.w0c >= 0) while (ld_acquire(ctr + in.w0c) < in.w0v) __nanosleep(32);
      if (in.w1c >= 0) while (ld_acquire(ctr + in.w1c) < in.w1v) __nanosleep(32);
      __threadfence();
    }
    __syncthreads();
    if (prof) t1 = gtime();
    exec_instr(in, sm, loss);
    __syncthreads();
    if (prof) t2 = gtime();
    if (threadIdx.x == 0) {
      __threadfence();   // cumulative over the block's writes via the bar.sync above
      if (in.s0 >= 0) atomicAdd(ctr + in.s0, 1);
      if (in.s1 >= 0) atomicAdd(ctr + in.s1, 1);
      if (prof) {   // 8 u64 per instruction: claim, deps ready, compute done, end, mma start, mma end, smid | block<<32, unused
        unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        unsigned long long* q = prof + (size_t)idx * 8;
        q[0] = t0; q[1] = t1; q[2] = t2; q[3] = gtime(); q[4] = g_stamp[0]; q[5] = g_stamp[1]; q[6] = (unsigned long long)smid | ((unsigned long long)blockIdx.x << 32); q[7] = 0;
      }
    }
  }
}

// ---- baseline: one launch per phase, block i runs tape[off+i], no waits
__global__ void __launch_bounds__(NT) phase_kernel(const Instr* tape, int off, int step, float* loss) {
  __shared__ __align__(128) float sm[TILE * TILE];
  exec_instr(inst_at(tape, off + blockIdx.x, step), sm, loss);
}

// ======================================================================================================================
// Fused column-parallel program (--prog fused): the 2-layer MLP step as TWO instructions per (run, hidden slice).
//
// Why: in the tile program one step is a chain of 4 dependent phases (H1 -> Y -> dH1 -> dW1 -> next H1), each hop ~5-6 us. Megatron-style
// tensor parallelism needs ONE cross-block exchange per step: slice W1 by columns and W2 by the same rows, then
//   FWD[c]: h_c = relu(X W1[:,c]);  yp_c = h_c W2[c,:]  (partial y, to global)
//   --- the one hop: every slice needs the sum of all slices' partials ---
//   BWD[c]: y = sum_c' yp_c' (fixed order);  dy = (y - t) 2/(B O);  dh_c = (dy W2[c,:]^T) * (h_c > 0);
//           W2[c,:] -= lr h_c^T dy;  W1[:,c] -= lr X^T dh_c        (all local to the slice)
// A slice's weights and h_c can stay in shared memory across steps ("resident") and its instruction queue is static (no claim atomic):
// the loop opcode (inst_at) re-instantiates the two instructions every step. The baselines run the very same two instructions as one
// launch per phase (weights and h_c round-trip through global memory, same device functions, same arithmetic), so the weights stay
// bitwise identical across the four modes. Sum order of the partials differs from the tile program, so tile-vs-fused agree to TF32 noise.
// ======================================================================================================================
enum { OP_FWD = 2, OP_BWD = 3 };
#ifndef KHS
#define KHS 32
#endif
constexpr int FB = 128, FD = 128, FH = 256, FO = 64, HS = KHS, NS = FH / HS, FNT = 256;
constexpr int LDX = FD + 4, LDH = HS + 4, LDY = FO + 4;
constexpr int SZ_X = FB * LDX, SZ_W1 = FD * LDH, SZ_H = FB * LDH, SZ_DY = FB * LDY, SZ_W2 = HS * LDY;
constexpr size_t FSMEM = (size_t)(SZ_X + SZ_W1 + SZ_H + SZ_DY + SZ_W2) * sizeof(float);
static_assert(HS % 32 == 0 && FH % HS == 0 && FB == 128 && FNT == 256, "fused program geometry");

using FragC = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;
__shared__ unsigned long long s_tready;
#ifdef KSTAGE
__device__ unsigned long long g_stage[24];
__shared__ long long s_last;
#define STAGE(i) do { if (blockIdx.x == 0 && threadIdx.x == 0) { long long n_ = clock64(); if ((i) > 0) atomicAdd(&g_stage[i], (unsigned long long)(n_ - s_last)); s_last = n_; } } while (0)
#else
#define STAGE(i) do {} while (0)
#endif

// acc[j] += A(16 x 8k) * B(8k x 16), per k-step of 8; A/B element order given by ACOL/BCOL (smem pointers, strides in floats)
template <bool ACOL, bool BCOL, int NF>
__device__ __forceinline__ void warp_mma(FragC (&acc)[NF], const float* pa, int lda, int astep, const float* pb, int ldb, int bstep, int bn, int ksteps) {
  using AL = typename std::conditional<ACOL, wmma::col_major, wmma::row_major>::type;
  using BL = typename std::conditional<BCOL, wmma::col_major, wmma::row_major>::type;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, AL> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, BL> fb;
#pragma unroll 4
  for (int k = 0; k < ksteps; k++) {
    wmma::load_matrix_sync(fa, pa + k * astep, lda);
#pragma unroll
    for (int e = 0; e < fa.num_elements; e++) fa.x[e] = wmma::__float_to_tf32(fa.x[e]);
#pragma unroll
    for (int j = 0; j < NF; j++) {
      wmma::load_matrix_sync(fb, pb + k * bstep + j * bn, ldb);
#pragma unroll
      for (int e = 0; e < fb.num_elements; e++) fb.x[e] = wmma::__float_to_tf32(fb.x[e]);
      wmma::mma_sync(acc[j], fa, fb, acc[j]);
    }
  }
}
template <int NF> __device__ __forceinline__ void zero_acc(FragC (&acc)[NF]) {
#pragma unroll
  for (int j = 0; j < NF; j++) wmma::fill_fragment(acc[j], 0.f);
}
__device__ __forceinline__ float lr_of(const Instr& in) { return __int_as_float(in.pad1); }   // fused instrs carry the learning rate bits in pad1

// async copy of a [rows x vecs*4] global block (row stride gld floats) into smem (row stride sld floats), all FNT threads
__device__ __forceinline__ void cp_block(float* s, int sld, const float* g, size_t gld, int rows, int vecs) {
  for (int i = threadIdx.x; i < rows * vecs; i += FNT) { int r = i / vecs, c4 = (i % vecs) * 4; cp16(s + r * sld + c4, g + r * gld + c4); }
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// FWD[c]. res: smem still holds this slice's weights from the previous step and BWD prefetched X, so only step 0 loads them.
__device__ void fwd_slice(const Instr& in, int stp, bool res, float* sm, int* ctr) {
  float *sX = sm, *sW1 = sX + SZ_X, *sH = sW1 + SZ_W1, *sDY = sH + SZ_H, *sW2 = sDY + SZ_DY;
  const Buf X = g_buf[in.a], W1 = g_buf[in.b], W2 = g_buf[in.c], T = g_buf[in.aux], HG = g_buf[in.n0], YP = g_buf[in.K];
  const int c = in.m0, w = threadIdx.x >> 5;
  if (!(res && stp > 0)) {
    cp_block(sX, LDX, X.p, X.ld, FB, FD / 4);
    cp_block(sW1, LDH, W1.p + c * HS, W1.ld, FD, HS / 4);
    cp_block(sW2, LDY, W2.p + (size_t)c * HS * W2.ld, W2.ld, HS, FO / 4);
  }
  STAGE(0);
  cp_commit(); cp_wait_all(); __syncthreads();
  STAGE(1);
  FragC a1[HS / 16]; zero_acc(a1);
  warp_mma<false, false, HS / 16>(a1, sX + 16 * w * LDX, LDX, 8, sW1, LDH, 8 * LDH, 16, FD / 8);
  float* hg = HG.p + (size_t)(16 * w) * HG.ld + c * HS;
#pragma unroll
  for (int j = 0; j < HS / 16; j++) {
#pragma unroll
    for (int e = 0; e < a1[j].num_elements; e++) a1[j].x[e] = fmaxf(a1[j].x[e], 0.f);
    wmma::store_matrix_sync(sH + 16 * w * LDH + j * 16, a1[j], LDH, wmma::mem_row_major);
    if (!res) wmma::store_matrix_sync(hg + j * 16, a1[j], HG.ld, wmma::mem_row_major);   // a non-resident BWD reads h back from here
  }
  __syncthreads(); STAGE(2);
  __syncwarp();                                               // F2 reads only this warp's own 16 rows of h
  FragC a2[FO / 16]; zero_acc(a2);
  warp_mma<false, false, FO / 16>(a2, sH + 16 * w * LDH, LDH, 8, sW2, LDY, 8 * LDY, 16, HS / 8);
  float* yp = YP.p + (size_t)((stp & 1) * NS + c) * FB * FO + (size_t)(16 * w) * FO;   // double-buffered by step parity
#pragma unroll
  for (int j = 0; j < FO / 16; j++) wmma::store_matrix_sync(yp + j * 16, a2[j], FO, wmma::mem_row_major);
  __syncthreads(); STAGE(3);
  if (ctr && threadIdx.x == 0) { __threadfence(); atomicAdd(ctr + in.s0, 1); }
  STAGE(4);
}

// BWD[c]. ctr == nullptr: launched as an ordinary kernel (launch order is the dependency). next_a: X buffer id of the next step (-1 = none).
__device__ void bwd_slice(const Instr& in, int stp, bool res, float* sm, float* loss, int* ctr, bool prof, int next_a) {
  __shared__ float red[FNT / 32];
  float *sX = sm, *sW1 = sX + SZ_X, *sH = sW1 + SZ_W1, *sDY = sH + SZ_H, *sW2 = sDY + SZ_DY;
  const Buf X = g_buf[in.a], W1 = g_buf[in.b], W2 = g_buf[in.c], T = g_buf[in.aux], HG = g_buf[in.n0], YP = g_buf[in.K];
  const int c = in.m0, w = threadIdx.x >> 5, tid = threadIdx.x;
  if (!res) {   // nothing resident: bring X, h, t and the current W2 slice back (independent of the hop, overlaps the wait)
    cp_block(sX, LDX, X.p, X.ld, FB, FD / 4); cp_block(sH, LDH, HG.p + c * HS, HG.ld, FB, HS / 4);
    cp_block(sW2, LDY, W2.p + (size_t)c * HS * W2.ld, W2.ld, HS, FO / 4);
    cp_commit();
  }
  STAGE(5);
  if (tid == 0) {   // the one hop: all slices' partial y of this step are in global memory
    if (ctr) { while (ld_acquire(ctr + in.w0c) < in.w0v) __nanosleep(16); __threadfence(); }
    if (prof) s_tready = gtime();
  }
  cp_wait_all(); __syncthreads(); STAGE(6);
  // ---- y = sum of the partials in slice order; dy = (y - t) * s; loss
  const float* ypb = YP.p + (size_t)(stp & 1) * NS * FB * FO;
  float lsum = 0.f;
#pragma unroll 1
  for (int g = 0; g < 4; g++) {
    float4 v[2][NS], tv[2];
#pragma unroll
    for (int u = 0; u < 2; u++)
#pragma unroll
      for (int p = 0; p < NS; p++) v[u][p] = __ldcg(reinterpret_cast<const float4*>(ypb + (size_t)p * FB * FO) + tid + FNT * (2 * g + u));
#pragma unroll
    for (int u = 0; u < 2; u++) tv[u] = __ldcg(reinterpret_cast<const float4*>(T.p) + tid + FNT * (2 * g + u));
#pragma unroll
    for (int u = 0; u < 2; u++) {
      int i4 = tid + FNT * (2 * g + u), r = i4 / (FO / 4), cc = (i4 % (FO / 4)) * 4;
      float4 y = v[u][0];
#pragma unroll
      for (int p = 1; p < NS; p++) { y.x += v[u][p].x; y.y += v[u][p].y; y.z += v[u][p].z; y.w += v[u][p].w; }
      float4 t = tv[u], o;
      const float* yy = &y.x; const float* tt = &t.x; float* oo = &o.x;
#pragma unroll
      for (int j = 0; j < 4; j++) { float d = yy[j] - tt[j]; lsum += d * d * in.scale * 0.5f; oo[j] = d * in.scale; }
      *reinterpret_cast<float4*>(sDY + r * LDY + cc) = o;
    }
  }
  if (c == 0) {   // slice 0 reports the loss: one writer per (run, step), fixed reduction order
    for (int o = 16; o; o >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
    if ((tid & 31) == 0) red[w] = lsum;
  }
  __syncthreads(); STAGE(7);   // dy complete (and red[])
  if (c == 0 && tid == 0) { float t = 0; for (int i = 0; i < FNT / 32; i++) t += red[i]; loss[in.loss_idx] = t; }
  const float lr = lr_of(in);
  // ---- gW2 = h^T dy (M = HS, N = FO, K = FB), 16x16 output tiles round-robin over the 8 warps
  constexpr int NT2 = HS / 32;   // tiles per warp: (HS/16)*(FO/16) / 8 warps
  FragC g2[NT2];
#pragma unroll
  for (int q = 0; q < NT2; q++) {
    int tile = w + 8 * q, mt = tile / (FO / 16), nt = tile % (FO / 16);
    FragC one[1]; zero_acc(one);
    warp_mma<true, false, 1>(one, sH + mt * 16, LDH, 8 * LDH, sDY + nt * 16, LDY, 8 * LDY, 16, FB / 8);
    g2[q] = one[0];
  }
  // ---- dh = (dy W2^T) masked by h > 0, from the OLD W2 slice (the update comes after every warp has read it)
  FragC dh[HS / 16]; zero_acc(dh);
  warp_mma<false, true, HS / 16>(dh, sDY + 16 * w * LDY, LDY, 8, sW2, LDY, 8, 16 * LDY, FO / 8);
  STAGE(8);
  __syncthreads();   // all warps done with h (gW2) and the old W2 slice
#pragma unroll
  for (int j = 0; j < HS / 16; j++) {
    FragC hm; wmma::load_matrix_sync(hm, sH + 16 * w * LDH + j * 16, LDH, wmma::mem_row_major);
#pragma unroll
    for (int e = 0; e < dh[j].num_elements; e++) dh[j].x[e] = (hm.x[e] > 0.f) ? dh[j].x[e] : 0.f;
    wmma::store_matrix_sync(sH + 16 * w * LDH + j * 16, dh[j], LDH, wmma::mem_row_major);   // dh replaces h in place (a lane owns the same elements)
  }
#pragma unroll
  for (int q = 0; q < NT2; q++) {   // W2 slice update, in smem and in global
    int tile = w + 8 * q, mt = tile / (FO / 16), nt = tile % (FO / 16);
    FragC cw; wmma::load_matrix_sync(cw, sW2 + mt * 16 * LDY + nt * 16, LDY, wmma::mem_row_major);
#pragma unroll
    for (int e = 0; e < cw.num_elements; e++) cw.x[e] = cw.x[e] - lr * g2[q].x[e];
    wmma::store_matrix_sync(sW2 + mt * 16 * LDY + nt * 16, cw, LDY, wmma::mem_row_major);
    wmma::store_matrix_sync(W2.p + (size_t)(c * HS + mt * 16) * W2.ld + nt * 16, cw, W2.ld, wmma::mem_row_major);
  }
  if (!res) { cp_block(sW1, LDH, W1.p + c * HS, W1.ld, FD, HS / 4); cp_commit(); }   // resident: sW1 already holds the current slice
  __syncthreads(); STAGE(9);   // dh complete for all 128 rows
  // ---- gW1 = X^T dh (M = FD, N = HS, K = FB); W1 slice update
  FragC g1[HS / 16]; zero_acc(g1);
  warp_mma<true, false, HS / 16>(g1, sX + 16 * w, LDX, 8 * LDX, sH, LDH, 8 * LDH, 16, FB / 8);
  cp_wait_all(); __syncthreads(); STAGE(10);   // (non-resident: the W1 slice has arrived)
#pragma unroll
  for (int j = 0; j < HS / 16; j++) {
    FragC cw; wmma::load_matrix_sync(cw, sW1 + 16 * w * LDH + j * 16, LDH, wmma::mem_row_major);
#pragma unroll
    for (int e = 0; e < cw.num_elements; e++) cw.x[e] = cw.x[e] - lr * g1[j].x[e];
    wmma::store_matrix_sync(sW1 + 16 * w * LDH + j * 16, cw, LDH, wmma::mem_row_major);
    wmma::store_matrix_sync(W1.p + (size_t)(16 * w) * W1.ld + c * HS + j * 16, cw, W1.ld, wmma::mem_row_major);
  }
  __syncthreads(); STAGE(11);   // sX is dead
  if (res && next_a >= 0) { const Buf Xn = g_buf[next_a]; cp_block(sX, LDX, Xn.p, Xn.ld, FB, FD / 4); cp_commit(); }   // next step's X streams in during the epilogue
  STAGE(12);
  if (!res) __threadfence();   // the next launch reads this slice's weights from global
}

// ---- the static-queue VM: block b owns a fixed list of tape entries and runs it once per step (loop opcode); res = one task per block
__global__ void __launch_bounds__(FNT) fvm_kernel(const Instr* tape, const int* qoff, const int* qlen, int steps, int* ctr, float* loss, unsigned long long* prof, int perStep, int res) {
  const int b = blockIdx.x, n = qlen[b], q0 = qoff[b];
  for (int s = 0; s < steps; s++)
    for (int i = 0; i < n; i++) {
      const int t = q0 + i;
      Instr in = inst_at(tape, t, s);
      unsigned long long t0 = 0, t1 = 0;
      if (prof) t0 = gtime();
      if (in.op == OP_FWD) { fwd_slice(in, s, res, dsm, ctr); t1 = t0; }
      else {
        int next_a = (s + 1 < steps) ? inst_at(tape, t, s + 1).a : -1;
        bwd_slice(in, s, res, dsm, loss, ctr, prof != nullptr, next_a);
        if (prof) t1 = s_tready;
      }
      __syncthreads();
      if (prof && threadIdx.x == 0) {
        unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        unsigned long long* q = prof + ((size_t)s * perStep + t) * 8; unsigned long long te = gtime();
        q[0] = t0; q[1] = t1; q[2] = te; q[3] = te; q[4] = t1; q[5] = te; q[6] = (unsigned long long)smid | ((unsigned long long)b << 32); q[7] = 0;
      }
    }
}
// ---- baseline: one launch per phase, block i runs tape[off + i], nothing resident
__global__ void __launch_bounds__(FNT) fphase_kernel(const Instr* tape, int off, int step, float* loss) {
  Instr in = inst_at(tape, off + blockIdx.x, step);
  if (in.op == OP_FWD) fwd_slice(in, step, false, dsm, nullptr); else bwd_slice(in, step, false, dsm, loss, nullptr, false, -1);
}

// ================= host =================
struct Rng { uint64_t s; float next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 40) & 0xFFFFFF) / 16777216.f * 2.f - 1.f; } };

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- self-test: every transpose combo of the tile GEMM vs CPU fp32 (TF32 tolerance)
static int selftest() {
  CK(cudaFuncSetAttribute(phase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  const int M = 128, N = 192;
  Rng r{777};
  int bad = 0;
  for (int K : {KC, 2 * KC, 3 * KC, 256}) for (int combo = 0; combo < 4; combo++) {
    bool ta = combo & 1, tb = combo & 2;
    std::vector<float> a(M * K), b(K * N), c(M * N), ref(M * N, 0.f);
    for (auto& v : a) v = r.next(); for (auto& v : b) v = r.next();
    // logical A is MxK, B is KxN; physical layouts depend on ta/tb
    auto Aat = [&](int m, int k) -> float& { return ta ? a[k * M + m] : a[m * K + k]; };
    auto Bat = [&](int k, int n) -> float& { return tb ? b[n * K + k] : b[k * N + n]; };
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) { float x = Aat(m, k); for (int n = 0; n < N; n++) ref[m * N + n] += x * Bat(k, n); }
    float *da, *db, *dc; CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, c.size() * 4));
    CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    Buf hb[3] = {{da, ta ? M : K}, {db, tb ? K : N}, {dc, N}};
    CK(cudaMemcpyToSymbol(g_buf, hb, sizeof hb));
    std::vector<Instr> t;
    for (int m = 0; m < M / TILE; m++) for (int n = 0; n < N / TILE; n++) {
      Instr i; memset(&i, 0, sizeof i); i.op = OP_GEMM; i.flags = (ta ? F_TA : 0) | (tb ? F_TB : 0); i.a = 0; i.b = 1; i.c = 2; i.aux = -1;
      i.w0c = i.w1c = i.s0 = i.s1 = -1; i.m0 = m * TILE; i.n0 = n * TILE; i.K = K; t.push_back(i);
    }
    Instr* dt; CK(cudaMalloc(&dt, t.size() * sizeof(Instr))); CK(cudaMemcpy(dt, t.data(), t.size() * sizeof(Instr), cudaMemcpyHostToDevice));
    { KP kp{0, 1, (int)t.size()}; CK(cudaMemcpyToSymbol(g_kp, &kp, sizeof kp)); }
    phase_kernel<<<(unsigned)t.size(), NT, DSMEM>>>(dt, 0, 0, nullptr); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost));
    double me = 0; for (int i = 0; i < M * N; i++) me = std::max(me, (double)fabs(c[i] - ref[i]));
    printf("selftest K=%d TA=%d TB=%d  max abs err %.3e [c0 %.3f ref0 %.3f]  %s\n", K, ta, tb, me, c[0], ref[0], me < 5e-2 ? "ok" : "BAD");
    if (me >= 5e-2) { bad++;
      if (K == KC && combo == 0) for (int bm = 0; bm < M / 16; bm++) { for (int bn = 0; bn < N / 16; bn++) { double e = 0; for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) e = std::max(e, (double)fabs(c[(bm * 16 + i) * N + bn * 16 + j] - ref[(bm * 16 + i) * N + bn * 16 + j])); printf("%c", e < 5e-2 ? '.' : 'X'); } printf("\n"); } }
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dt);
  }
  return bad;
}

#ifdef KVM_LIB
#define KVM_ENTRY extern "C" int kvm_main
#else
#define KVM_ENTRY static int kvm_main
#endif
KVM_ENTRY(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--selftest") return selftest() ? 2 : 0;
  int steps = 1000, B = 128, D = 128, Hd = 256, O = 64, NB = 4, ref_steps = 100, R = 1;
  float lr = 0.05f; int nreps = 5; std::string json_path, dump_dir, tl_path, prog = "tiles"; int tl_s0 = 0, tl_s1 = 4;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return atof(argv[++i]); };
    if (a == "--steps") steps = nx(); else if (a == "--B") B = nx(); else if (a == "--D") D = nx();
    else if (a == "--H") Hd = nx(); else if (a == "--O") O = nx(); else if (a == "--nb") NB = nx();
    else if (a == "--reps") nreps = nx(); else if (a == "--json") json_path = argv[++i]; else if (a == "--dump-data") dump_dir = argv[++i];
    else if (a == "--timeline") tl_path = argv[++i]; else if (a == "--tl-steps") { tl_s0 = nx(); tl_s1 = nx(); }
    else if (a == "--prog") prog = argv[++i];
    else if (a == "--runs") R = nx(); else if (a == "--lr") lr = nx(); else if (a == "--ref-steps") ref_steps = nx();
  }
  const bool fused = prog == "fused";
  if (!fused && prog != "tiles") { fprintf(stderr, "--prog tiles|fused\n"); return 1; }
  if (fused && (B != FB || D != FD || Hd != FH || O != FO)) { fprintf(stderr, "--prog fused is compiled for B=%d D=%d H=%d O=%d (KHS=%d)\n", FB, FD, FH, FO, HS); return 1; }
  if (B % TILE || D % TILE || Hd % TILE || O % TILE || B % KC || D % KC || Hd % KC || O % KC) { fprintf(stderr, "dims must be multiples of max(TILE=%d, KC=%d)\n", TILE, KC); return 1; }
  int nRB = B / TILE, nHC = Hd / TILE, nOC = O / TILE, nDC = D / TILE;
  // run r trains the same model from the same init with its own learning rate (an LR sweep)
  std::vector<float> lrs(R); for (int r = 0; r < R; r++) lrs[r] = lr * (R == 1 ? 1.f : 0.25f + 1.5f * r / (R - 1));

  // ---- data: teacher-generated regression targets
  Rng rng{12345};
  std::vector<float> W1(D * Hd), W2(Hd * O), Wt1(D * Hd), Wt2(Hd * O);
  for (auto& v : W1) v = rng.next() * sqrtf(3.f / D);
  for (auto& v : W2) v = rng.next() * sqrtf(3.f / Hd);
  for (auto& v : Wt1) v = rng.next() * sqrtf(3.f / D);
  for (auto& v : Wt2) v = rng.next() * sqrtf(3.f / Hd);
  std::vector<std::vector<float>> X(NB, std::vector<float>(B * D)), T(NB, std::vector<float>(B * O));
  for (int k = 0; k < NB; k++) {
    for (auto& v : X[k]) v = rng.next();
    std::vector<float> h(B * Hd, 0.f);
    for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) h[i * Hd + j] += x * Wt1[kk * Hd + j]; }
    for (auto& v : h) v = v > 0 ? v : 0;
    std::fill(T[k].begin(), T[k].end(), 0.f);
    for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) T[k][i * O + j] += x * Wt2[kk * O + j]; }
  }

  // ---- CPU reference (fp32) for a given lr
  int rs = std::min(steps, ref_steps);
  auto cpu_ref = [&](float rlr) {
    std::vector<float> L, rW1 = W1, rW2 = W2;
    std::vector<float> h(B * Hd), dY(B * O), dH(B * Hd), g1(D * Hd), g2(Hd * O);
    for (int s = 0; s < rs; s++) {
      int k = s % NB;
      std::fill(h.begin(), h.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) h[i * Hd + j] += x * rW1[kk * Hd + j]; }
      for (auto& v : h) v = v > 0 ? v : 0;
      std::fill(dY.begin(), dY.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) dY[i * O + j] += x * rW2[kk * O + j]; }
      double Ls = 0;
      for (int i = 0; i < B * O; i++) { float d = dY[i] - T[k][i]; Ls += (double)d * d; dY[i] = d * 2.f / (B * O); }
      L.push_back((float)(Ls / (B * O)));
      std::fill(g2.begin(), g2.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) g2[kk * O + j] += x * dY[i * O + j]; }
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float a = 0; for (int j = 0; j < O; j++) a += dY[i * O + j] * rW2[kk * O + j]; dH[i * Hd + kk] = h[i * Hd + kk] > 0 ? a : 0; }
      std::fill(g1.begin(), g1.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) g1[kk * Hd + j] += x * dH[i * Hd + j]; }
      for (int i = 0; i < D * Hd; i++) rW1[i] -= rlr * g1[i];
      for (int i = 0; i < Hd * O; i++) rW2[i] -= rlr * g2[i];
    }
    return L;
  };
  std::vector<float> ref0 = cpu_ref(lrs[0]), refN = R > 1 ? cpu_ref(lrs[R - 1]) : ref0;

  // ---- device buffers: shared data X_k, T_k; per-run {W1, W2, H1, dY, dH1}
  auto idX = [&](int k) { return k; };
  auto idT = [&](int k) { return NB + k; };
  auto idW1 = [&](int r) { return 2 * NB + 5 * r; };
  auto idW2 = [&](int r) { return 2 * NB + 5 * r + 1; };
  auto idH1 = [&](int r) { return 2 * NB + 5 * r + 2; };
  auto idDY = [&](int r) { return 2 * NB + 5 * r + 3; };
  auto idDH = [&](int r) { return 2 * NB + 5 * r + 4; };
  int nbuf = 2 * NB + 5 * R;
  if (nbuf > 256) { fprintf(stderr, "too many buffers\n"); return 1; }
  std::vector<Buf> hb(nbuf);
  auto dmalloc = [&](size_t n) { float* p; CK(cudaMalloc(&p, n * 4)); CK(cudaMemset(p, 0, n * 4)); return p; };
  for (int k = 0; k < NB; k++) {
    hb[idX(k)] = {dmalloc(B * D), D}; CK(cudaMemcpy(hb[idX(k)].p, X[k].data(), B * D * 4, cudaMemcpyHostToDevice));
    hb[idT(k)] = {dmalloc(B * O), O}; CK(cudaMemcpy(hb[idT(k)].p, T[k].data(), B * O * 4, cudaMemcpyHostToDevice));
  }
  for (int r = 0; r < R; r++) {
    hb[idW1(r)] = {dmalloc(D * Hd), Hd}; hb[idW2(r)] = {dmalloc(Hd * O), O};
    hb[idH1(r)] = {dmalloc(B * Hd), Hd}; hb[idDY(r)] = {dmalloc(fused ? (size_t)2 * NS * B * O : (size_t)B * O), O}; hb[idDH(r)] = {dmalloc(B * Hd), Hd};   // fused: DY holds the partial-y ring [parity][slice][B x O]
  }
  CK(cudaMemcpyToSymbol(g_buf, hb.data(), nbuf * sizeof(Buf)));

  // ---- tape builder. Per step and run: H1 -> Y -> DH1 -> DW2(+SGD) -> DW1(+SGD); counters give fine-grained deps.
  const int NPH = fused ? 2 : 5;
  enum { P_H1, P_Y, P_DH1, P_DW2, P_DW1 };
  int CPS = 5 * (nRB + 1);
  int SC = fused ? R : R * CPS;   // counters per step (fused: one "all partial y written" counter per run)
  auto ctr = [&](int r, int s, int ph, int rb) { return (s * R + r) * CPS + ph * (nRB + 1) + (rb < 0 ? nRB : rb); };
  auto mk = [&]() { Instr i; memset(&i, 0, sizeof i); i.aux = -1; i.w0c = i.w1c = i.s0 = i.s1 = -1; return i; };
  // Template for step 0 (waits on the previous step are flagged and dropped at s==0 on the device).
  auto emit_phase = [&](std::vector<Instr>& tape, int r, int ph) {
    const int s = 0; float rlr = lrs[r];
    if (ph == P_H1) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_RELU; i.a = idX(0); i.b = idW1(r); i.c = idH1(r); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = D; i.xflags = 4 | 1 | 2;
      i.w0c = ctr(r, s, P_DW1, -1); i.w0v = nDC * nHC; i.w1c = ctr(r, s, P_DW2, -1); i.w1v = nHC * nOC;
      i.s0 = ctr(r, s, P_H1, rb); i.s1 = ctr(r, s, P_H1, -1); tape.push_back(i);
    }
    if (ph == P_Y) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nOC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_MSE; i.a = idH1(r); i.b = idW2(r); i.c = idDY(r); i.aux = idT(0); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = Hd; i.xflags = 8 | 2;
      i.scale = 2.f / (B * O); i.loss_idx = r * steps;
      i.w0c = ctr(r, s, P_H1, rb); i.w0v = nHC; i.w1c = ctr(r, s, P_DW2, -1); i.w1v = nHC * nOC;
      i.s0 = ctr(r, s, P_Y, rb); i.s1 = ctr(r, s, P_Y, -1); tape.push_back(i);
    }
    if (ph == P_DH1) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TB | F_RELUMASK; i.a = idDY(r); i.b = idW2(r); i.c = idDH(r); i.aux = idH1(r); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = O;
      i.w0c = ctr(r, s, P_Y, rb); i.w0v = nOC; i.s1 = ctr(r, s, P_DH1, -1); tape.push_back(i);
    }
    if (ph == P_DW2) for (int m = 0; m < nHC; m++) for (int n = 0; n < nOC; n++) {   // W2 -= lr H1^T dY (after all dH1 tiles read W2)
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idH1(r); i.b = idDY(r); i.c = idW2(r); i.m0 = m * TILE; i.n0 = n * TILE; i.K = B; i.scale = rlr;
      i.w0c = ctr(r, s, P_Y, -1); i.w0v = nRB * nOC; i.w1c = ctr(r, s, P_DH1, -1); i.w1v = nRB * nHC;
      i.s1 = ctr(r, s, P_DW2, -1); tape.push_back(i);
    }
    if (ph == P_DW1) for (int m = 0; m < nDC; m++) for (int n = 0; n < nHC; n++) {   // W1 -= lr X^T dH1
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idX(0); i.b = idDH(r); i.c = idW1(r); i.m0 = m * TILE; i.n0 = n * TILE; i.K = B; i.scale = rlr; i.xflags = 4;
      i.w0c = ctr(r, s, P_DH1, -1); i.w0v = nRB * nHC; i.s1 = ctr(r, s, P_DW1, -1); tape.push_back(i);
    }
  };
  // fused program: FWD / BWD per (run, hidden slice); counter r counts the slices whose partial y of the step is written
  auto emit_fused = [&](std::vector<Instr>& tape, int r, int c, int ph) {
    Instr i = mk(); i.op = ph == 0 ? OP_FWD : OP_BWD; i.a = idX(0); i.b = idW1(r); i.c = idW2(r); i.aux = idT(0); i.m0 = c; i.n0 = idH1(r); i.K = idDY(r);
    i.scale = 2.f / (B * O); memcpy(&i.pad1, &lrs[r], 4); i.loss_idx = r * steps; i.xflags = 4 | 8;
    if (ph == 0) i.s0 = r; else { i.w0c = r; i.w0v = NS; }
    tape.push_back(i);
  };
  // one-step template, phase-major then run: [ph][r] segments. Batched baseline launches a whole phase (all runs);
  // sequential baseline launches one (run, phase) segment.
  std::vector<Instr> tapeP; std::vector<int> offP(NPH), nP(NPH), nPR(NPH);
  std::vector<std::vector<int>> offPR(NPH, std::vector<int>(R));
  for (int ph = 0; ph < NPH; ph++) {
    offP[ph] = (int)tapeP.size();
    for (int r = 0; r < R; r++) {
      offPR[ph][r] = (int)tapeP.size();
      if (fused) for (int c = 0; c < NS; c++) emit_fused(tapeP, r, c, ph); else emit_phase(tapeP, r, ph);
    }
    nP[ph] = (int)tapeP.size() - offP[ph]; nPR[ph] = nP[ph] / R;
  }
  int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  // The VM's own tape. tiles: the phase-major template, entries claimed dynamically. fused: a static queue per block (tasks round-robin over
  // min(tasks, SMs) blocks, each queue = all its FWDs then all its BWDs), run once per step by the loop opcode; resident iff one task per block.
  std::vector<Instr> tapeV; std::vector<int> tmplPhase, tmplRun, qoff, qlen; int res = 0, grid = 0, occ = 1;
  if (!fused) {
    tapeV = tapeP;
    for (int ph = 0; ph < NPH; ph++) for (int r = 0; r < R; r++) { tmplPhase.resize(offPR[ph][r] + nPR[ph], ph); tmplRun.resize(offPR[ph][r] + nPR[ph], r); }
    CK(cudaFuncSetAttribute(vm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
    CK(cudaFuncSetAttribute(phase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm_kernel, NT, DSMEM));
    grid = prop.multiProcessorCount * occ;
  } else {
    CK(cudaFuncSetAttribute(fvm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FSMEM));
    CK(cudaFuncSetAttribute(fphase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FSMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fvm_kernel, FNT, FSMEM));
    if (occ < 1) { fprintf(stderr, "fused kernel does not fit (%zu B smem)\n", FSMEM); return 1; }
    int ntask = R * NS; grid = std::min(ntask, prop.multiProcessorCount * occ); res = ntask <= grid;
    std::vector<std::vector<int>> tasks(grid);
    for (int t = 0; t < ntask; t++) tasks[t % grid].push_back(t);
    for (int b = 0; b < grid; b++) {
      qoff.push_back((int)tapeV.size());
      for (int ph = 0; ph < 2; ph++) for (int t : tasks[b]) { emit_fused(tapeV, t / NS, t % NS, ph); tmplPhase.push_back(ph); tmplRun.push_back(t / NS); }
      qlen.push_back((int)tapeV.size() - qoff.back());
    }
  }
  int perStep = (int)tapeV.size();
  int ntape = perStep * steps, nctr = steps * SC;
  if (fused) printf("model B=%d D=%d H=%d O=%d  runs=%d steps=%d  prog=fused hs=%d slices=%d  tape=%d instrs executed, %d-entry step template, %d static queues%s\n", B, D, Hd, O, R, steps, HS, NS, ntape, perStep, grid, res ? ", weights resident in smem" : ", multi-task blocks (not resident)");
  else printf("model B=%d D=%d H=%d O=%d  runs=%d steps=%d  prog=tiles tile=%d kc=%d stg=%d  tape=%d instrs executed, %d-entry step template (%.2f MB)\n", B, D, Hd, O, R, steps, TILE, KC, STG, ntape, perStep, perStep * sizeof(Instr) / 1e6);

  Instr *dtapeP, *dtapeV; int *dhead, *dctr, *dqoff = nullptr, *dqlen = nullptr; float* dloss;
  CK(cudaMalloc(&dtapeP, tapeP.size() * sizeof(Instr))); CK(cudaMemcpy(dtapeP, tapeP.data(), tapeP.size() * sizeof(Instr), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dtapeV, tapeV.size() * sizeof(Instr))); CK(cudaMemcpy(dtapeV, tapeV.data(), tapeV.size() * sizeof(Instr), cudaMemcpyHostToDevice));
  if (fused) {
    CK(cudaMalloc(&dqoff, grid * 4)); CK(cudaMemcpy(dqoff, qoff.data(), grid * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dqlen, grid * 4)); CK(cudaMemcpy(dqlen, qlen.data(), grid * 4, cudaMemcpyHostToDevice));
  }
  { KP kp{SC, NB, perStep}; CK(cudaMemcpyToSymbol(g_kp, &kp, sizeof kp)); }
  CK(cudaMalloc(&dhead, 4)); CK(cudaMalloc(&dctr, (size_t)nctr * 4)); CK(cudaMalloc(&dloss, (size_t)R * steps * 4));
  printf("%s  SMs=%d occ=%d  vm grid=%d\n", prop.name, prop.multiProcessorCount, occ, grid);

  auto reset = [&]() {
    for (int r = 0; r < R; r++) {
      CK(cudaMemcpy(hb[idW1(r)].p, W1.data(), W1.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(hb[idW2(r)].p, W2.data(), W2.size() * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(dhead, 0, 4)); CK(cudaMemset(dctr, 0, (size_t)nctr * 4)); CK(cudaMemset(dloss, 0, (size_t)R * steps * 4));
    CK(cudaDeviceSynchronize());
  };
  auto launch_seq = [&](cudaStream_t st, int rlo, int rhi) {   // one model at a time: run r, step s, phase ph
    for (int r = rlo; r < rhi; r++) for (int s = 0; s < steps; s++) for (int ph = 0; ph < NPH; ph++) { if (fused) fphase_kernel<<<nPR[ph], FNT, FSMEM, st>>>(dtapeP, offPR[ph][r], s, dloss); else phase_kernel<<<nPR[ph], NT, DSMEM, st>>>(dtapeP, offPR[ph][r], s, dloss); }
  };
  auto launch_batched = [&](cudaStream_t st) { for (int s = 0; s < steps; s++) for (int ph = 0; ph < NPH; ph++) { if (fused) fphase_kernel<<<nP[ph], FNT, FSMEM, st>>>(dtapeP, offP[ph], s, dloss); else phase_kernel<<<nP[ph], NT, DSMEM, st>>>(dtapeP, offP[ph], s, dloss); } };
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto timed = [&](auto fn) { reset(); cudaEventRecord(e0); fn(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError()); float ms; cudaEventElapsedTime(&ms, e0, e1); return (double)ms; };
  struct Out { std::vector<float> L, w; };
  auto fetch = [&]() {
    Out o; o.L.resize((size_t)R * steps);
    CK(cudaMemcpy(o.L.data(), dloss, o.L.size() * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < R; r++) { std::vector<float> a(W1.size()), b(W2.size());
      CK(cudaMemcpy(a.data(), hb[idW1(r)].p, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), hb[idW2(r)].p, b.size() * 4, cudaMemcpyDeviceToHost));
      o.w.insert(o.w.end(), a.begin(), a.end()); o.w.insert(o.w.end(), b.begin(), b.end()); }
    return o;
  };

  // graphs: (a) one run's whole training, replayed R times  (b) all runs batched per phase
  cudaGraph_t gA, gB; cudaGraphExec_t xA, xB;
  { cudaStream_t cs; cudaStreamCreate(&cs);
    cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal); launch_seq(cs, 0, 1); CK(cudaStreamEndCapture(cs, &gA)); CK(cudaGraphInstantiate(&xA, gA, 0));
    cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal); launch_batched(cs); CK(cudaStreamEndCapture(cs, &gB)); CK(cudaGraphInstantiate(&xB, gB, 0)); }
  // graph A replays run 0's phases only, so for R>1 build per-run graphs
  std::vector<cudaGraphExec_t> xRun(R);
  { cudaStream_t cs; cudaStreamCreate(&cs);
    for (int r = 0; r < R; r++) { cudaGraph_t g; cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal); launch_seq(cs, r, r + 1); CK(cudaStreamEndCapture(cs, &g)); CK(cudaGraphInstantiate(&xRun[r], g, 0)); } }

  auto vm_launch = [&](unsigned long long* prof) {
    if (fused) fvm_kernel<<<grid, FNT, FSMEM>>>(dtapeV, dqoff, dqlen, steps, dctr, dloss, prof, perStep, res);
    else vm_kernel<<<grid, NT, DSMEM>>>(dtapeV, ntape, dhead, dctr, dloss, prof);
  };
  auto run_vm = [&]() { vm_launch(nullptr); };
  auto run_seq = [&]() { launch_seq(0, 0, R); };
  auto run_seqgraph = [&]() { for (int r = 0; r < R; r++) CK(cudaGraphLaunch(xRun[r], 0)); };
  auto run_batched = [&]() { CK(cudaGraphLaunch(xB, 0)); };
  // each mode: 1 warm-up, then `reps` timed repetitions; best = min, spread = max/min (ratios below use the best)
  std::vector<double> ts[4]; Out o[4];
  auto measure = [&](auto fn, int idx) { timed(fn); for (int q = 0; q < nreps; q++) ts[idx].push_back(timed(fn)); o[idx] = fetch(); };
  measure(run_seq, 0); measure(run_seqgraph, 1); measure(run_batched, 2); measure(run_vm, 3);
  double t[4], tmax[4], tmed[4];
  for (int i = 0; i < 4; i++) { auto v = ts[i]; std::sort(v.begin(), v.end()); t[i] = v[0]; tmax[i] = v.back(); tmed[i] = v[v.size() / 2]; }

  double tot = (double)R * steps;
  const char* nm[4] = {"sequential, launch-per-op", "sequential, CUDA graph/run", "batched across runs (vmap-like graph)", "KVM: all runs, ONE kernel"};
  printf("\n%-40s %10s %14s %11s %9s %8s\n", "mode", "best ms", "us/step/run", "run-steps/s", "vs seq", "max/min");
  for (int i = 0; i < 4; i++) printf("%-40s %10.2f %14.2f %11.0f %8.2fx %7.2fx\n", nm[i], t[i], t[i] * 1000 / tot, tot / (t[i] / 1000), t[0] / t[i], tmax[i] / t[i]);
  printf("KVM vs best baseline: %.2fx   (%d reps per mode)\n", std::min(std::min(t[0], t[1]), t[2]) / t[3], nreps);

  // ---- correctness
  auto relerr = [&](const float* a, const std::vector<float>& ref) { double m = 0; for (int i = 0; i < rs; i++) m = std::max(m, (double)fabs(a[i] - ref[i]) / (fabs(ref[i]) + 1e-6)); return m; };
  double e_ref = 0; for (int i = 0; i < 4; i++) { e_ref = std::max(e_ref, relerr(&o[i].L[0], ref0)); e_ref = std::max(e_ref, relerr(&o[i].L[(size_t)(R - 1) * steps], refN)); }
  // weights must be BITWISE identical in all four modes (same tape, same per-tile reduction order): exact equality, not a tolerance
  size_t n_diff = 0; double e_mode = 0;
  for (int i = 0; i < 3; i++) for (size_t j = 0; j < o[3].w.size(); j++)
    if (memcmp(&o[3].w[j], &o[i].w[j], 4) != 0) { n_diff++; e_mode = std::max(e_mode, (double)fabs(o[3].w[j] - o[i].w[j])); }
  bool dec = true; for (int r = 0; r < R; r++) dec = dec && o[3].L[(size_t)r * steps + steps - 1] < o[3].L[(size_t)r * steps];
  printf("\nloss run0: %.5f -> %.5f (lr %.4f)", o[3].L[0], o[3].L[steps - 1], lrs[0]);
  if (R > 1) printf("   run%d: %.5f -> %.5f (lr %.4f)", R - 1, o[3].L[(size_t)(R - 1) * steps], o[3].L[(size_t)(R - 1) * steps + steps - 1], lrs[R - 1]);
  printf("\nmax rel loss err vs cpu fp32 (runs 0 and %d, first %d steps, all modes): %.2e\n", R - 1, rs, e_ref);
  printf("weights VM vs each baseline, all runs: %zu of %zu elements differ bitwise (max |dW| %.2e)\n", n_diff, 3 * o[3].w.size(), e_mode);
  bool ok = e_ref < 2e-2 && n_diff == 0 && dec;

  // ---- optional: raw data for external baselines (PyTorch), so every framework trains from the very same bits
  if (!dump_dir.empty()) {
    auto wr = [&](const char* name, const void* p, size_t bytes) { std::string f = dump_dir + "/" + name; FILE* fp = fopen(f.c_str(), "wb"); if (!fp) { fprintf(stderr, "cannot write %s\n", f.c_str()); exit(1); } fwrite(p, 1, bytes, fp); fclose(fp); };
    for (int k = 0; k < NB; k++) { wr(("X" + std::to_string(k) + ".f32").c_str(), X[k].data(), B * D * 4); wr(("T" + std::to_string(k) + ".f32").c_str(), T[k].data(), B * O * 4); }
    wr("W1.f32", W1.data(), W1.size() * 4); wr("W2.f32", W2.data(), W2.size() * 4); wr("lrs.f32", lrs.data(), R * 4);
    wr("vm_loss.f32", o[3].L.data(), o[3].L.size() * 4); wr("vm_weights.f32", o[3].w.data(), o[3].w.size() * 4);
    std::string f = dump_dir + "/meta.json"; FILE* fp = fopen(f.c_str(), "w");
    fprintf(fp, "{\"B\":%d,\"D\":%d,\"H\":%d,\"O\":%d,\"NB\":%d,\"R\":%d,\"steps\":%d}\n", B, D, Hd, O, NB, R, steps); fclose(fp);
  }

  // ---- optional timeline: one extra VM run with per-instruction timestamps (prof pointer set), CSV for the step window [tl_s0, tl_s1)
  double t_prof = 0;
  if (!tl_path.empty()) {
    unsigned long long* dprof; size_t pbytes = (size_t)ntape * 8 * 8; CK(cudaMalloc(&dprof, pbytes)); CK(cudaMemset(dprof, 0, pbytes));
    t_prof = timed([&]() { vm_launch(dprof); });
    std::vector<unsigned long long> hp((size_t)ntape * 8); CK(cudaMemcpy(hp.data(), dprof, pbytes, cudaMemcpyDeviceToHost)); cudaFree(dprof);
    unsigned long long tmin = ~0ull; for (int i = 0; i < ntape; i++) tmin = std::min(tmin, hp[(size_t)i * 8]);
    FILE* fp = fopen(tl_path.c_str(), "w"); if (!fp) { fprintf(stderr, "cannot write %s\n", tl_path.c_str()); exit(1); }
    fprintf(fp, "# idx,step,tmpl,phase,run,smid,block,claim_ns,ready_ns,compute_end_ns,end_ns,mma_start_ns,mma_end_ns (ns since the first claim of the whole run; prog=%s phases %s; profiled run %.3f ms)\n", prog.c_str(), fused ? "0=FWD 1=BWD" : "0=H1 1=Y 2=DH1 3=DW2 4=DW1", t_prof);
    for (int i = 0; i < ntape; i++) {
      int stp = i / perStep, tm = i % perStep; if (stp < tl_s0 || stp >= tl_s1) continue;
      const unsigned long long* q = &hp[(size_t)i * 8];
      fprintf(fp, "%d,%d,%d,%d,%d,%u,%u,%lld,%lld,%lld,%lld,%lld,%lld\n", i, stp, tm, tmplPhase[tm], tmplRun[tm], (unsigned)(q[6] & 0xffffffffu), (unsigned)(q[6] >> 32),
              (long long)(q[0] - tmin), (long long)(q[1] - tmin), (long long)(q[2] - tmin), (long long)(q[3] - tmin), (long long)(q[4] - tmin), (long long)(q[5] - tmin));
    }
    fclose(fp);
#ifdef KSTAGE
    { unsigned long long st[24]; CK(cudaMemcpyFromSymbol(st, g_stage, sizeof st)); int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
      printf("stage cycles per step (block 0, summed over instrs; %d kHz): FWD load/wait %.0f  F1 %.0f  F2+store %.0f  signal %.0f | BWD pre-wait %.0f  hop+load %.0f  reduce+dy %.0f  loss %.0f  gW2+dh %.0f  mask+W2upd %.0f  gW1 %.0f  W1upd %.0f  tail %.0f\n", clk,
             st[1] / (double)steps, st[2] / (double)steps, st[3] / (double)steps, st[4] / (double)steps, st[5] / (double)steps, st[6] / (double)steps, st[7] / (double)steps,
             (double)0, st[8] / (double)steps, st[9] / (double)steps, st[10] / (double)steps, st[11] / (double)steps, st[12] / (double)steps); }
#endif
    printf("timeline: profiled VM run %.3f ms (%.2fx the unprofiled best), steps [%d,%d) -> %s\n", t_prof, t_prof / t[3], tl_s0, tl_s1, tl_path.c_str());
  }

  if (!json_path.empty()) {
    FILE* fp = fopen(json_path.c_str(), "w");
    fprintf(fp, "{\"gpu\":\"%s\",\"sms\":%d,\"occ\":%d,\"grid\":%d,\"B\":%d,\"D\":%d,\"H\":%d,\"O\":%d,\"prog\":\"%s\",\"runs\":%d,\"steps\":%d,\"tile\":%d,\"kc\":%d,\"stg\":%d,\"reps\":%d,\n \"modes\":[", prop.name, prop.multiProcessorCount, occ, grid, B, D, Hd, O, prog.c_str(), R, steps, TILE, KC, STG, nreps);
    for (int i = 0; i < 4; i++) { fprintf(fp, "%s{\"name\":\"%s\",\"best_ms\":%.4f,\"median_ms\":%.4f,\"max_ms\":%.4f,\"us_per_step_run\":%.4f}", i ? "," : "", nm[i], t[i], tmed[i], tmax[i], t[i] * 1000 / tot); }
    fprintf(fp, "],\n \"weights_bitwise_differ\":%zu,\"max_rel_loss_err\":%.3e,\"pass\":%s,\"profiled_ms\":%.4f}\n", n_diff, e_ref, ok ? "true" : "false", t_prof);
    fclose(fp);
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}

#ifndef KVM_LIB
int main(int argc, char** argv) { return kvm_main(argc, argv); }
#endif
