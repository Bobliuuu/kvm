#pragma once
// The split fused MLP step, device side: one member's share of one step (mvm_fused_split.h has the design, the descriptor and the layout).
// Used by the op `fmlp_split` of the static schedule path (a gang op: its members run at once, placed by mvm_static_plan) and by
// bench/fused_mlp/split_bed.cu (the same code, timed alone). GEMMs are mvm_fused_mlp.cuh's (m16n8k8 TF32 mma); the barrier is mvm_gang.cuh's.
#include <cstdint>
#include "mvm_fused_split.h"
#include "mvm_fused_mlp.cuh"
#include "mvm_gang.cuh"

namespace mvm_fsplit {

using mvm_fmlp::NT;
constexpr int R = MVM_SPLIT_R;

__device__ __forceinline__ float4 ld_cg4(const float* p) { float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v; }

// rows x cols floats (cols a multiple of 4; 16-byte aligned rows on both sides) from global memory (L2) to shared memory by 16-byte cp.async,
// no wait: the caller commits and waits. The step's start issues every array at once (the share's weights, biases, head rows, the slice's embedding
// rows) behind one wait: plain loads there pay one round trip an array (measured on the 3080 Ti: the load phase 9.2k cycles as one group, 12.9k as
// plain loads, h576-d1).
__device__ __forceinline__ void copy_rows(float* dst, int dst_ld, const float* src, size_t src_ld, int rows, int cols) {
  mvm_fmlp::for_cells(rows, cols >> 2, [&](int r, int c4) { mvm_fmlp::cp_async16(dst + (size_t)r * dst_ld + 4 * c4, src + (size_t)r * src_ld + 4 * c4); });
}
// The same rows by plain L2 loads, 8 float4 a thread in flight before their stores. For one array read right after a barrier (an exchange),
// one cp.async group read 1.5x slower in the exchange read bed (xread_bed, 3080 Ti and A100); MVM_SPLIT_XREAD_ASYNC=1 takes cp.async for the exchanges too.
#ifndef MVM_SPLIT_XREAD_ASYNC
#define MVM_SPLIT_XREAD_ASYNC 0
#endif
// MVM_SPLIT_ODX: a streamed member computes its own columns of the slice's dX itself (split_run's backward, hidden layers), from every virtual
// share's dZ (the slice's, through one L2 slot after the barrier the reduce-scatter crossed) and its rows of W_l across every share: share t's
// product is the resident member's GEMM over that share's units, and the SV products are added in share order (P_0, then += P_t), as the
// reduce-scatter added the SV partials. So the bits are the same, and the SV full-width partials' L2 round trip (written by their shares'
// members, summed after the barrier) goes away (the split bed's sub-phases, Nov 13: at h232-d3 x8 the dX partials and their reduce-scatter held 25% of the step).
// MVM_SPLIT_ODX 2 (whole-row chunks, the default) takes the same sums over chunks of W's whole rows instead of a share's pieces (below, at the loop);
// 1 a share at a time; 0 the partials. On the A100's split bed (Nov 13; every bundle byte-equal to its run alone), whole-row chunks
// over a share at a time x1.086 at h232-d3 x8, Sv 18, and with the owner-side dW below, both over the partials x1.36-1.48 at Sv 18 and x1.55-1.63 at the
// share counts mvm_fuse_split_sv picks (h232-d3 x8, h184-d3 x8, h296-d3 x4). A run alone stays resident and takes neither. The default is
// set in mvm_fused_split.h, and the choice is the gang's, not a member's (mvm_split_odx: its members must take the same backward exchange).
// MVM_SPLIT_OWNDW: on a layer that takes the owner-side dX, the members of a share compute its dW_l themselves, each its own rows of W_l, from
// every slice's H_(l-1) (the forward's all-gather, XH[l - 1]) and dZ (the owner-side dX's exchange slot), slice by slice: a slice's product is
// the member partial's GEMM (gemm_m<true, false> over the slice's R rows: the same fragments, k-steps and rounding), and the products are
// added in slice order, as the owners added the Q partials. So the bits are the same, and the members' partials (their H_(l-1)^T dZ, [HR][HP]
// each, to L2) and the owners' Q-fold reads of them go away. The sum is stored where slice 0's partial sat; the owners read it alone.
__device__ __forceinline__ void load_rows(float* dst, int dst_ld, const float* src, size_t src_ld, int rows, int cols) {
  constexpr int U = 8;
  const int c4 = cols >> 2, n = rows * c4;
  for (int base = threadIdx.x; base < n; base += U * NT) {
    float4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) { const int i = base + u * NT; if (i < n) v[u] = ld_cg4(src + (size_t)(i / c4) * src_ld + 4 * (i % c4)); }
#pragma unroll
    for (int u = 0; u < U; u++) { const int i = base + u * NT; if (i < n) *reinterpret_cast<float4*>(dst + (size_t)(i / c4) * dst_ld + 4 * (i % c4)) = v[u]; }
  }
}

// The descriptor is copied to the arena's base before the task's waits (the worker knows its next record's descriptor), so its round trip
// overlaps them; split_run commits and waits for it.
__device__ __forceinline__ void prefetch_desc(const MvmSplitDesc* gd, float* sm) {
  const int chunks = (int)((sizeof(MvmSplitDesc) + 15) / 16);
  if (threadIdx.x < chunks) mvm_fmlp::cp_async16(sm + threadIdx.x * 4, reinterpret_cast<const float*>(gd) + threadIdx.x * 4);
}

// ---- Streamed weights (split_run<..., true>; the layout's two staging tiles, mvm_fused_split.h). A member holds no layer's weights: each
// GEMM that reads them takes its member's share kc rows at a time from L2, one tile loading while the other is read. The GEMMs below are the
// resident member's (mvm_fmlp::gemm) with their loops cut at the tiles: every warp issues the same mma on the same values in the same order,
// so a streamed member computes the resident member's bits.
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all_but_newest() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

struct Stage {
  float* buf[2];   // tile c lives in buf[c & 1]; the two are contiguous (buf[1] = buf[0] + kc * HP)
  int kc, HP;
  // Tile c of a share's weights, rows [c kc, min(rows, c kc + kc)) of `src` (pitch ld, already at the share's first column), w columns. Rows from
  // `real` on are written as zeros: the resident member holds zero rows past H (the layer's HR rows), and the backward reads them.
  __device__ __forceinline__ void issue(const float* src, size_t ld, int rows, int real, int w, int c) const {
    const int r0 = c * kc;
    if (r0 >= rows) return;
    const int n = min(kc, rows - r0), nr = max(0, min(n, real - r0));
    float* dst = buf[c & 1];
    mvm_fmlp::for_cells(nr, w >> 2, [&](int r, int c4) { mvm_fmlp::cp_async16(dst + (size_t)r * HP + 4 * c4, src + (size_t)(r0 + r) * ld + 4 * c4); });
    if (nr < n) mvm_fmlp::for_cells(n - nr, w, [&](int r, int col) { dst[(size_t)(nr + r) * HP + col] = 0.f; });
  }
  // Tiles 0 and 1, each its own cp.async group: a streamed GEMM waits for all groups but the newest, so tile c is complete when it reads it.
  __device__ __forceinline__ void prefetch(const float* src, size_t ld, int rows, int real, int w) const {
    issue(src, ld, rows, real, w, 0); cp_commit();
    issue(src, ld, rows, real, w, 1); cp_commit();
  }
  // Before reading tile c: it has landed, and every thread's zero rows are visible.
  __device__ __forceinline__ void wait() const { cp_wait_all_but_newest(); __syncthreads(); }
  // After reading tile c: every warp is done with its buffer, which takes tile c + 2 (an empty group when there is none, so wait() keeps its meaning).
  __device__ __forceinline__ void next(const float* src, size_t ld, int rows, int real, int w, int c) const {
    __syncthreads();
    issue(src, ld, rows, real, w, c + 2); cp_commit();
  }
};

// Z [16 x N] = A [16 x K] W with W streamed: mvm_fmlp::gemm<false, false, 1, false, 1> (the resident forward's gemm_r at R = 16) with its k-loop
// cut at the tiles. Warp w owns column block w, its accumulators stay in registers across tiles, and its k-steps run in the same order, so
// every output is the resident GEMM's. Every thread takes part in the tile waits; warps past N's column blocks issue no mma. Tiles 0 and 1 must
// already be issued (Stage::prefetch with the same src, ld and rows).
__device__ __forceinline__ void gemm_fwd_streamed(const float* A, int lda, const Stage& st, const float* src, size_t ld, int K, int w, float* C, int ldc, int N) {
  using namespace mvm_fmlp;
  const int ncb = (N + 15) >> 4, cb = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, la = lane & 7, b3 = (lane >> 3) & 1, b4 = lane >> 4;
  const bool act = cb < ncb;
  const int n0 = cb * 16 + g, ldb = st.HP, a_ld = (la + 8 * b3) * lda + 4 * b4;
  const int tlb = t * ldb, tlb4 = (t + 4) * ldb, ob0 = n0, ob1 = n0 + 8;
  float acc[2][4];
#pragma unroll
  for (int j = 0; j < 2; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  const int tiles = (K + st.kc - 1) / st.kc;
  for (int c = 0; c < tiles; ++c) {
    st.wait();
    if (act) {
      const float* B = st.buf[c & 1];
      const int k1 = min(K, c * st.kc + st.kc);
      int kb = 0;
      for (int kk = c * st.kc; kk < k1; kk += 8, kb += 8 * ldb) {
        const unsigned b00 = tf32_in(B[kb + tlb + ob0]), b01 = tf32_in(B[kb + tlb4 + ob0]), b10 = tf32_in(B[kb + tlb + ob1]), b11 = tf32_in(B[kb + tlb4 + ob1]);
        unsigned a[4];
        ldmatrix_x4(a, A + a_ld + kk); a[0] = tf32_round(a[0]); a[1] = tf32_round(a[1]); a[2] = tf32_round(a[2]); a[3] = tf32_round(a[3]);
        mma_tf32(acc[0], a, b00, b01);
        mma_tf32(acc[1], a, b10, b11);
      }
    }
    st.next(src, ld, K, K, w, c);
  }
  if (!act) return;
  float* c0 = C + (size_t)g * ldc + cb * 16 + 2 * t; float* c1 = C + (size_t)(g + 8) * ldc + cb * 16 + 2 * t;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    *reinterpret_cast<float2*>(c0 + j * 8) = make_float2(acc[j][0], acc[j][1]);
    *reinterpret_cast<float2*>(c1 + j * 8) = make_float2(acc[j][2], acc[j][3]);
  }
}

// One 16-column block of C = A B^T, B stored [n][k] with its 16 rows at B: the body that warp cb of mvm_fmlp::gemm<false, true, 1, false, 1> runs
// for its column block (B and C offset by that block), so the same values in the same mma order.
__device__ __forceinline__ void gemm_tb_block(const float* A, int lda, const float* B, int ldb, float* C, int ldc, int K) {
  using namespace mvm_fmlp;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, la = lane & 7, b3 = (lane >> 3) & 1, b4 = lane >> 4;
  const int a_ld = (la + 8 * b3) * lda + 4 * b4, ob0 = (la + 8 * b4) * ldb + 4 * b3;
  float acc[2][4];
#pragma unroll
  for (int j = 0; j < 2; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  for (int kk = 0; kk < K; kk += 8) {
    unsigned r[4];
    ldmatrix_x4(r, B + ob0 + kk);
    const unsigned b00 = tf32_round(r[0]), b01 = tf32_round(r[1]), b10 = tf32_round(r[2]), b11 = tf32_round(r[3]);
    unsigned a[4];
    ldmatrix_x4(a, A + a_ld + kk); a[0] = tf32_round(a[0]); a[1] = tf32_round(a[1]); a[2] = tf32_round(a[2]); a[3] = tf32_round(a[3]);
    mma_tf32(acc[0], a, b00, b01);
    mma_tf32(acc[1], a, b10, b11);
  }
  float* c0 = C + (size_t)g * ldc + 2 * t; float* c1 = C + (size_t)(g + 8) * ldc + 2 * t;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    *reinterpret_cast<float2*>(c0 + j * 8) = make_float2(acc[j][0], acc[j][1]);
    *reinterpret_cast<float2*>(c1 + j * 8) = make_float2(acc[j][2], acc[j][3]);
  }
}

// Each virtual share's full-width partial dZ_t W_t^T (columns [n0, n0 + N) of its slot), W streamed by rows (the backward's N): per tile, one item
// a (virtual share, 16-row block) shared out over the warps. An item is the resident member's call gemm_r<false, true>(dZ + c, HP, W + c, HP, out,
// ldo, R, N, wt) for one column block, so its outputs are the resident member's. `slot(t)` is virtual share t's output (columns from 0); `rows`
// are the weight rows (the output's columns, the zero pad rows past `real` included). Tiles 0 and 1 must already be issued.
template <class Slot>
__device__ __forceinline__ void gemm_dx_streamed(const float* dZ, int HP, const MvmSplitDesc& d, int t0, int t1, int h0, const Stage& st, const float* src, size_t ld, int rows, int real, int w, Slot slot, int ldo) {
  const int warp = threadIdx.x >> 5, tiles = (rows + st.kc - 1) / st.kc;
  for (int c = 0; c < tiles; ++c) {
    st.wait();
    const int n0 = c * st.kc, nb = (min(st.kc, rows - n0) + 15) >> 4, items = (t1 - t0) * nb;
    const float* B = st.buf[c & 1];
    for (int i = warp; i < items; i += NT / 32) {
      const int t = t0 + i / nb, b = i % nb, col = d.share0[t] - h0;
      gemm_tb_block(dZ + col, HP, B + (size_t)b * 16 * HP + col, HP, slot(t) + n0 + b * 16, ldo, d.share0[t + 1] - d.share0[t]);
    }
    st.next(src, ld, rows, real, w, c);
  }
}

struct NoPhases { __device__ __forceinline__ void operator()(int) const {} };
constexpr int NPH = 9;   // the phases `ph(i)` marks, once a step in this order (every thread calls it)
// 0 load, 1 forward (GEMMs, GELU, all-gathers), 2 logits partial, 3 barrier: logits, 4 logits summed + cross-entropy, 5 backward (GEMMs, reduce-scatters),
// 6 barrier: partials, 7 owners: parameters, 8 owners: embedding + the last barrier
// Sub-phases, for beds: the forward and the backward cut per layer at points every thread passes, as indices past NPH (the executor's NoPhases
// ignores every call, so its code is unchanged). In execution order: phase 0, then each forward layer l's sub_f(l, 0..3), phase 1, 2, 3, 4, then
// sub_head(), each backward layer's sub_b(p, 0..6) with p = 0 the top layer, phase 5, 6, 7, 8. A sub-phase a layer does not run is never stamped.
constexpr int NSUB_F = 4;   // forward layer l: 0 GEMM (the tile waits with it), 1 GELU, 2 all-gather out + barrier, 3 the slice's H read back
constexpr int NSUB_B = 7;   // backward layer: 0 gelu', 1 db partial, 2 the layer's input read back, 3 dW partial, 4 dX partials (tiles + writes), 5 barrier, 6 reduce-scatter
constexpr int NPH_SUB = MVM_SPLIT_MAXL * NSUB_F + 1 + MVM_SPLIT_MAXL * NSUB_B;
__host__ __device__ constexpr int sub_f(int l, int k) { return NPH + l * NSUB_F + k; }
__host__ __device__ constexpr int sub_head() { return NPH + MVM_SPLIT_MAXL * NSUB_F; }
__host__ __device__ constexpr int sub_b(int p, int k) { return sub_head() + 1 + p * NSUB_B + k; }

// One step of member `member` (= q * S + s) of a split run of SV virtual shares (the descriptor's Sv; its physical S is any count up to SV, mvm_fused_split.h),
// its weights resident for the step or, with STREAM, streamed through the layout's staging tiles (the same bits; mvm_fused_split.h):
// `ids` and `tgt` are the step's T tokens and targets, `abort` the run's error word
// (mvm_gang::cross: a failure elsewhere ends the step at its next crossing instead of hanging it; nullptr in a bed), `loss_out` the step's loss (member 0 writes loss_out[0] = sum / T and loss_out[1] = 1 / T, as softmax_ce_reduce does). The members
// cross 2 x L + 1 barriers; the last one is the step's end, so every update is visible to every member when any of them returns.
// ODX false compiles the owner-side backward out: the partials member, which a streamed gang that can't take the owner-side dX runs (its entry,
// mvm_static.cuh's split_stream_part), with the partials' own dealing of the owners' parameter groups (below).
template <int SV, int Q, bool STREAM = false, bool ODX = true, class PH>
__device__ __forceinline__ void split_run(int member, const int32_t* ids, const int32_t* tgt, float* loss_out, float* sm, const int* abort, PH ph) {
  mvm_fmlp::cp_commit_wait();
  __syncthreads();
  const MvmSplitDesc& d = *reinterpret_cast<const MvmSplitDesc*>(sm);
  const int S = d.S, q = member / S, s = member % S, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, members = Q * S;
  const int t0 = d.vs0[s], t1 = d.vs0[s + 1];   // this member's virtual shares
  const int h0 = d.share0[t0], w = d.share0[t1] - h0, E = d.E, V = d.V, VP = d.VP, HP = d.HP, HR = d.HR, H = d.H, T = d.T, L = d.L;
  float* Whs = sm + d.Whs; float* bs = sm + d.bs; float* X0 = sm + d.X0; float* Hin = sm + d.Hin;
  float* ZG = sm + d.ZG; float* Hs = sm + d.Hs; float* Lg = sm + d.Lg; float* dZ = sm + d.dZ; float* gtok = sm + d.gtok; float* rl = sm + d.rl;
  int* sid = reinterpret_cast<int*>(sm + d.ints);   // [T]: the step's token ids
  int* tg = sid + T;                                // [R]: the slice's targets
  int* mlist = tg + R;                              // [T]: positions of the tokens this member's embedding rows own, in token order
  int* wcnt = mlist + T;                            // [9]: per-warp match counts, then offsets
  const float inv_t = 1.f / (float)T, lr = d.lr;
  const int erow0 = (int)((long long)member * V / members), erow1 = (int)((long long)(member + 1) * V / members);
  unsigned* const ctl = d.ctl;
  unsigned long long at = mvm_gang::cross_start(ctl, member, members);   // this member's place in the gang's crossings (consecutive steps join member to member, mvm_fuse.h)
  static_assert(MVM_GANG_LINES_MAX <= 32 * (NT / 32), "mvm_gang::cross polls each flag line of 32 members with one warp of the block");
  // ---- load: the share's columns of every layer (zero past w, and past H in the hidden layers' rows), its head rows (zero past V), the slice's
  // embedding rows, the step's tokens and the slice's targets. The pads are written each step: the owners' scratch sits over the weights.
  for (int l = 0; l < L; ++l) {
    if constexpr (!STREAM) {
      float* Ws = sm + d.Ws[l];
      const int K = l ? H : E, KR = l ? HR : E;
      mvm_fmlp::for_cells(KR, HP, [&](int k, int c) { if (k >= K || c >= w) Ws[k * HP + c] = 0.f; });
    }
    for (int c = tid; c < HP; c += NT) if (c >= w) bs[l * HP + c] = 0.f;
  }
  mvm_fmlp::for_cells(HP, VP, [&](int r, int v) { if (r >= w || v >= V) Whs[r * VP + v] = 0.f; });
  for (int l = 0; l < L; ++l) {
    if constexpr (!STREAM) copy_rows(sm + d.Ws[l], HP, d.w[l] + h0, H, l ? H : E, w);   // W_l[:, h0 .. h0 + w)
    copy_rows(bs + l * HP, 0, d.b[l] + h0, 0, 1, w);
  }
  copy_rows(Whs, VP, d.wh + (size_t)h0 * V, V, w, V);            // Wh[h0 .. h0 + w, :]
  for (int r = 0; r < R; ++r) copy_rows(X0 + r * E, 0, d.we + (size_t)ids[q * R + r] * E, 0, 1, E);
  for (int k = tid; k < T; k += NT) sid[k] = ids[k];
  for (int r = tid; r < R; r += NT) tg[r] = tgt[q * R + r];
  const Stage st{{sm + d.Ws[0], sm + d.Ws[STREAM ? 1 : 0]}, STREAM ? mvm_split_stream_rows(d) : 0, HP};   // streamed: the staging tiles
  if constexpr (STREAM) st.prefetch(d.w[0] + h0, H, E, E, w);   // layer 0's first two tiles, behind the load's wait
  mvm_fmlp::cp_commit_wait();
  __syncthreads();
  ph(0);
  // ---- forward
  const float* A = X0; int lda = E, K = E;
  for (int l = 0; l < L; ++l) {
    float* ZGl = ZG + l * R * HP;
    if constexpr (STREAM) gemm_fwd_streamed(A, lda, st, d.w[l] + h0, H, K, w, ZGl, HP, HP);   // Z = A W_l,s, W_l,s by tiles
    else mvm_fmlp::gemm_r<false, false>(A, lda, sm + d.Ws[l], HP, ZGl, HP, R, HP, K);   // Z = A W_l,s
    __syncthreads();
    ph(sub_f(l, 0));
    mvm_fmlp::for_cells(R, HP, [&](int r, int c) { const int i = r * HP + c; float f, g; mvm_fmlp::gelu_fg(ZGl[i] + bs[l * HP + c], f, g); Hs[i] = c < w ? f : 0.f; ZGl[i] = c < w ? g : 0.f; });
    __syncthreads();
    ph(sub_f(l, 1));
    if (l + 1 < L) {   // the all-gather: this member's columns of H to L2, then the slice's whole H back as the next layer's input
      if constexpr (STREAM) st.prefetch(d.w[l + 1] + h0, H, H, H, w);   // the next layer's first tiles load across the barrier
      float* xh = d.XH[l] + (size_t)q * R * HR;
      mvm_fmlp::for_cells(R, w >> 2, [&](int r, int c4) { *reinterpret_cast<float4*>(xh + (size_t)r * HR + h0 + 4 * c4) = *reinterpret_cast<const float4*>(Hs + r * HP + 4 * c4); });
      if (!mvm_gang::cross(ctl, member, members, at, abort)) return;
      ph(sub_f(l, 2));
      if (MVM_SPLIT_XREAD_ASYNC) { copy_rows(Hin, HR, xh, HR, R, HR); mvm_fmlp::cp_commit_wait(); } else load_rows(Hin, HR, xh, HR, R, HR);   // pad columns past H stay zero in L2
      __syncthreads();
      ph(sub_f(l, 3));
      A = Hin; lda = HR; K = H;
    }
  }
  ph(1);
  for (int t = t0; t < t1; ++t) {   // each virtual share's part of the logits (zero past V), its own columns its whole K
    const int c = d.share0[t] - h0;
    mvm_fmlp::gemm_r<false, false>(Hs + c, HP, Whs + c * VP, VP, d.P + (size_t)(q * SV + t) * R * VP, VP, R, VP, d.share0[t + 1] - d.share0[t]);
  }
  ph(2);
  if (!mvm_gang::cross(ctl, member, members, at, abort)) return;
  ph(3);
  // ---- the slice's SV partials summed in virtual-share order, four logits a thread at a time; then the cross-entropy
  mvm_fmlp::for_cells(R, VP / 4, [&](int r, int v4) {
    float4 x[SV];
#pragma unroll
    for (int t = 0; t < SV; ++t) x[t] = ld_cg4(d.P + ((size_t)(q * SV + t) * R + r) * VP + 4 * v4);
    float4 a = x[0];
#pragma unroll
    for (int t = 1; t < SV; ++t) { a.x += x[t].x; a.y += x[t].y; a.z += x[t].z; a.w += x[t].w; }
    *reinterpret_cast<float4*>(Lg + r * VP + 4 * v4) = a;
  });
  __syncthreads();
  // 8 warps, two rows each. A row's VP <= 128 logits sit in registers, four a lane, and each exponential is taken once: the same operations in the same
  // order as three passes over shared memory with every exponential taken twice, so the same bits (the A100's split bed spent 17% of the step here).
  constexpr int VR = 128 / 32;
  for (int r = warp; r < R; r += NT / 32) {
    float* row = Lg + r * VP;
    float x[VR], m = -INFINITY;
#pragma unroll
    for (int j = 0; j < VR; ++j) { const int v = lane + 32 * j; x[j] = v < V ? row[v] : -INFINITY; m = fmaxf(m, x[j]); }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const int t = tg[r]; const float lt = row[t];   // every lane reads the target's logit before any lane overwrites the row
    float z = 0.f;
#pragma unroll
    for (int j = 0; j < VR; ++j) if (lane + 32 * j < V) { x[j] = expf(x[j] - m); z += x[j]; }
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    __syncwarp();
#pragma unroll
    for (int j = 0; j < VR; ++j) { const int v = lane + 32 * j; if (v < VP) row[v] = v < V ? (x[j] / z - (v == t ? 1.f : 0.f)) * inv_t : 0.f; }
    if (lane == 0) rl[r] = m + logf(z) - lt;
  }
  __syncthreads();
  if (s == 0 && tid == 0) { float sum = 0.f; for (int r = 0; r < R; ++r) sum += rl[r]; d.loss_part[q] = sum; }
  ph(4);
  // ---- backward: the head, then each layer from the top
  mvm_fmlp::gemm_r<false, true>(Lg, VP, Whs, VP, dZ, HP, R, HP, VP);                                 // dH_s = dlogits Wh_s^T
  mvm_fmlp::gemm_m<true, false>(Hs, HP, Lg, VP, d.dWh + (size_t)member * HP * VP, VP, HP, VP, R);    // dWh_s = H_s^T dlogits
  __syncthreads();
  ph(sub_head());
  // The owner-side dX's sizing (below, MVM_SPLIT_ODX) and the owner-side dW's (MVM_SPLIT_OWNDW), the same for every layer, so the owners'
  // phase reads the same choice: a layer above 0 takes the owner-side dX when odx_ok, and its dW from the owners when owndw_ok.
  int odx_wt = 0;
  for (int t = 0; t < SV; ++t) odx_wt = max(odx_wt, d.share0[t + 1] - d.share0[t]);
  const int w16 = (w + 15) & ~15;
  const int odx_p = odx_wt + 4, odx_slot = (w16 + R) * odx_p, odx_n = min(4, (2 * st.kc * HP) / odx_slot);
  // the streamed member's span from its staging tiles through its biases and head weights (contiguous in mvm_split_layout's order): the biases
  // feed only the forward and the head weights only the head, and the step's load writes both again, so it is free from the head's backward on
  const int own_cap = d.Whs + HP * VP - d.Ws[0];
  constexpr int ODXW_U = MVM_SPLIT_ODXW_U, ODXW_P = MVM_SPLIT_ODXW_P;
  const int odxw_slot = (w16 + R) * ODXW_P;
  const bool odx_ok = STREAM && ODX && mvm_split_odx(d);   // the gang's choice, the same in every member (mvm_fused_split.h)
  // The owner-side dW's tiles, every slice's at once, in that span: a slice's H_(l-1) rows [R][lda] (at most 3 row blocks of 16) and its dZ
  // [R][ldb], each pitch 8 mod 32 floats, so a lane's 32-bit fragment loads (gemm<true, false>'s) fall in distinct banks.
  const int own_lda = 72, own_ldb = ((w16 + 31) & ~31) + 8, own_slot = R * (own_lda + own_ldb);
  const bool owndw_ok = odx_ok && MVM_SPLIT_OWNDW && Q * own_slot <= own_cap;
  for (int l = L - 1; l >= 0; --l) {
    const int p = L - 1 - l;   // the sub-phases' layer position (0: the top layer, the first the backward runs)
    const float* ZGl = ZG + l * R * HP;
    float* Ws = sm + d.Ws[l];
    // owner-side dX (MVM_SPLIT_ODX): a hidden layer of a streamed member whose two staging tiles (contiguous, 2 kc HP floats) hold at least two
    // slots of one share's W block [w16][odx_p] and dZ block [R][odx_p]; any other keeps the partials and the reduce-scatter. The slots form a
    // ring of up to 4, so up to 3 shares' blocks load while one is multiplied.
    // The pitch is the widest virtual share's units + 4 (Nov 13: a fixed 20 overlapped h296-d3's 24-unit share and broke its bits).
    // Shares are whole 8-unit blocks, so a row is an odd number of 16-byte chunks: the 8 rows each ldmatrix matrix reads fall on 8 different
    // bank groups. A pitch of the widest share alone, 16 at h232-d3, is a 4-way conflict on every fragment (the split bed: +480 cycles a share,
    // more than the third slot it allowed hid).
    const bool odx = odx_ok && l > 0;
    if constexpr (STREAM) { if (!odx) st.prefetch(d.w[l] + h0, H, l ? HR : E, l ? H : E, w); }   // W_l,s's rows by tiles (zero rows past H)
    for (int i = tid; i < R * HP; i += NT) dZ[i] *= ZGl[i];                                         // dZ = dH_s gelu'(Z) (zero past w)
    __syncthreads();
    ph(sub_b(p, 0));
    for (int c = tid; c < HP; c += NT) { float a = 0.f; for (int r = 0; r < R; ++r) a += dZ[r * HP + c]; d.db[l][(size_t)member * HP + c] = a; }
    ph(sub_b(p, 1));
    if (l == 0) {
      mvm_fmlp::gemm_m<true, false>(X0, E, dZ, HP, d.dW[0] + (size_t)member * E * HP, HP, E, HP, R);          // dW_0,s = X0^T dZ
      ph(sub_b(p, 3));
      if constexpr (STREAM) gemm_dx_streamed(dZ, HP, d, t0, t1, h0, st, d.w[0] + h0, H, E, E, w, [&](int t) { return d.dX0 + (size_t)(q * SV + t) * R * E; }, E);
      else for (int t = t0; t < t1; ++t) {                                                                      // dX0 part of each virtual share = dZ_t W_0,t^T
        const int c = d.share0[t] - h0;
        mvm_fmlp::gemm_r<false, true>(dZ + c, HP, Ws + c, HP, d.dX0 + (size_t)(q * SV + t) * R * E, E, R, E, d.share0[t + 1] - d.share0[t]);
      }
      __syncthreads();
      ph(sub_b(p, 4));
    } else {
      const bool owndw = owndw_ok;   // (l > 0 here) the owners compute dW_l after the barrier: no read-back, no partial here
      if (!owndw) {
        const float* xin = d.XH[l - 1] + (size_t)q * R * HR;   // the layer's input, the slice's whole H_(l-1), read back from L2
        if (MVM_SPLIT_XREAD_ASYNC) { copy_rows(Hin, HR, xin, HR, R, HR); mvm_fmlp::cp_commit_wait(); } else load_rows(Hin, HR, xin, HR, R, HR);
        __syncthreads();
      }
      ph(sub_b(p, 2));
      if (!owndw)
        for (int m0 = 0; m0 < HR; m0 += 128)                                                                  // dW_l,s = H_(l-1)^T dZ
          mvm_fmlp::gemm_m<true, false>(Hin + m0, HR, dZ, HP, d.dW[l] + ((size_t)member * HR + m0) * HP, HP, min(128, HR - m0), HP, R);
      ph(sub_b(p, 3));
      if (odx) {
        // the slice's dZ through one L2 slot (slot (q, 0) of the layer's parity: no partial uses it here), each member its own columns
        float* const xe = d.XD[l & 1] + (size_t)(q * SV) * R * HR;
        mvm_fmlp::for_cells(R, w >> 2, [&](int r, int c4) { *reinterpret_cast<float4*>(xe + (size_t)r * HR + h0 + 4 * c4) = *reinterpret_cast<const float4*>(dZ + r * HP + 4 * c4); });   // as H's all-gather: the barrier's fence publishes it, and the reads are L2 (cp.async.cg)
        ph(sub_b(p, 4));
        if (!mvm_gang::cross(ctl, member, members, at, abort)) return;
        ph(sub_b(p, 5));
        // share t's blocks into ring slot t % odx_n: W_l rows [h0, h0 + w) x the share's units (zero rows past w and past H), and the slice's dZ
        // rows x the share's units; one cp.async group a share
        const float* const wl = d.w[l];
        float* const ring = st.buf[0];   // st.buf[1] follows st.buf[0]: 2 kc HP contiguous floats
        auto slot = [&](int t) { return ring + (size_t)(t % odx_n) * odx_slot; };
        auto issue = [&](int t) {
          float* const wb = slot(t); float* const zb = wb + (size_t)w16 * odx_p;
          const int u0 = d.share0[t], wt = d.share0[t + 1] - u0;
          mvm_fmlp::for_cells(w16, wt >> 2, [&](int r, int c4) {
            float* const dst = wb + (size_t)r * odx_p + 4 * c4;
            if (r < w && h0 + r < H) mvm_fmlp::cp_async16(dst, wl + (size_t)(h0 + r) * H + u0 + 4 * c4);
            else *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
          });
          mvm_fmlp::for_cells(R, wt >> 2, [&](int r, int c4) { mvm_fmlp::cp_async16(zb + (size_t)r * odx_p + 4 * c4, xe + (size_t)r * HR + u0 + 4 * c4); });
          cp_commit();
        };
        // wait until at most n of the issued groups are still in flight (cp.async.wait_group takes an immediate)
        auto wait_pending = [](int n) {
          if (n <= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
          else if (n == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
          else if (n == 2) asm volatile("cp.async.wait_group 2;" ::: "memory");
          else asm volatile("cp.async.wait_group 3;" ::: "memory");
        };
        const int depth = odx_n - 1;   // shares in flight ahead of the one being multiplied
        // P_t = dZ_t W_l[h0 .. h0 + w, share t]^T, the resident member's gemm_r<false, true> over the share's units, and dX = P_0 + P_1 + ...
        // in share order, as the reduce-scatter added the SV partials. Warp cb holds its 16 columns of dX (rows g and g + 8, two n-tiles: the
        // accumulators of gemm<false, true, 1, false, 1>) in registers across the shares and stores them once: share 0's product is copied,
        // every later one added (C + P, the rounding of the SGD epilogue's C - (-1) P that dZ took per share before; on the split bed that
        // read-modify-write of dZ at its pitch of HP floats was 8 wavefronts an access, 512 a share). The k-steps, fragments and rounding
        // are gemm's own, so the products are its bits.
        static_assert(R == 16, "the owner-side dX holds one row block of the slice's dX");
        const int lane = tid & 31, cb = tid >> 5, g = lane >> 2, lq = lane & 3, la = lane & 7, b3 = (lane >> 3) & 1, b4 = lane >> 4;
        const bool mine = cb < (w16 >> 4);   // a warp past the member's columns only loads and waits
        const int a_off = (la + 8 * b3) * odx_p + 4 * b4, b_off = (cb * 16 + la + 8 * b4) * odx_p + 4 * b3;   // the lanes' ldmatrix chunks, as gemm's
        float tot[2][4] = {};
        __syncthreads();   // every warp is done with dZ as the dW partial's operand: dX replaces it
        if constexpr (MVM_SPLIT_ODX == 2) {
          // Whole-row chunks: W_l's rows [h0, h0 + w) whole, ODXW_U units a chunk (zero rows past w and past H), and the slice's dZ rows over the same
          // units, through two slots of the free span. A share's pieces were 32-64 bytes a row, one L2 request each (the split bed: about 12
          // cycles a request per SM, 2,376 a layer at h232-d3 x8); a chunk's rows are 256 bytes. The k-steps walk the chunk, and at each
          // virtual share's first unit the share before it joins dX's sum (an empty share's zeros too), so each share's product is its own
          // k-steps from zero, added in share order, as the per-share loop added them.
          float* const span = sm + d.Ws[0];
          const int nch = (H + ODXW_U - 1) / ODXW_U;
          auto wissue = [&](int c) {
            float* const wb = span + (size_t)(c & 1) * odxw_slot; float* const zb = wb + (size_t)w16 * ODXW_P;
            const int u0 = c * ODXW_U, uc = min(ODXW_U, H - u0);
            mvm_fmlp::for_cells(w16, uc >> 2, [&](int r, int c4) {
              float* const dst = wb + (size_t)r * ODXW_P + 4 * c4;
              if (r < w && h0 + r < H) mvm_fmlp::cp_async16(dst, wl + (size_t)(h0 + r) * H + u0 + 4 * c4);
              else *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
            });
            mvm_fmlp::for_cells(R, uc >> 2, [&](int r, int c4) { mvm_fmlp::cp_async16(zb + (size_t)r * ODXW_P + 4 * c4, xe + (size_t)r * HR + u0 + 4 * c4); });
            cp_commit();
          };
          const int wa_off = (la + 8 * b3) * ODXW_P + 4 * b4, wb_off = (cb * 16 + la + 8 * b4) * ODXW_P + 4 * b3;
          float acc[2][4] = {};
          int tc = 0;   // the virtual share the k-steps are in
          auto flush = [&]() {
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
              for (int e = 0; e < 4; ++e) { tot[j][e] = tc == 0 ? acc[j][e] : tot[j][e] + acc[j][e]; acc[j][e] = 0.f; }
            ++tc;
          };
          wissue(0);
          for (int c = 0; c < nch; ++c) {
            if (c + 1 < nch) { wissue(c + 1); cp_wait_all_but_newest(); } else mvm_fmlp::cp_commit_wait();
            __syncthreads();   // chunk c has landed, and every thread's zero rows are visible
            if (mine) {
              const float* const wb = span + (size_t)(c & 1) * odxw_slot; const float* const zb = wb + (size_t)w16 * ODXW_P;
              const int u0 = c * ODXW_U, uc = min(ODXW_U, H - u0);
              for (int kk = 0; kk < uc; kk += 8) {
                while (tc + 1 < SV && u0 + kk >= d.share0[tc + 1]) flush();   // share tc ends here
                unsigned r[4], a[4];
                mvm_fmlp::ldmatrix_x4(r, wb + wb_off + kk);
                mvm_fmlp::ldmatrix_x4(a, zb + wa_off + kk);
#pragma unroll
                for (int i = 0; i < 4; ++i) { r[i] = mvm_fmlp::tf32_round(r[i]); a[i] = mvm_fmlp::tf32_round(a[i]); }
                mvm_fmlp::mma_tf32(acc[0], a, r[0], r[1]);
                mvm_fmlp::mma_tf32(acc[1], a, r[2], r[3]);
              }
            }
            __syncthreads();   // slot c & 1 is free for chunk c + 2
          }
          if (mine) while (tc < SV) flush();   // the last share's product, then any empty share past it
        } else {
        for (int t = 0; t < depth && t < SV; ++t) issue(t);
        for (int t = 0; t < SV; ++t) {
          wait_pending(min(SV, t + depth) - (t + 1));   // share t's group has landed (share t + depth's is issued after the barrier)
          __syncthreads();   // share t's blocks are visible to every warp, and every warp is done with share t - 1's slot
          if (t + depth < SV) issue(t + depth);   // into share t - 1's slot
          if (!mine) continue;
          const float* const wb = slot(t); const float* const zb = wb + (size_t)w16 * odx_p;
          const int wt = d.share0[t + 1] - d.share0[t];
          float acc[2][4] = {};
          for (int kk = 0; kk < wt; kk += 8) {
            unsigned r[4], a[4];
            mvm_fmlp::ldmatrix_x4(r, wb + b_off + kk);
            mvm_fmlp::ldmatrix_x4(a, zb + a_off + kk);
#pragma unroll
            for (int i = 0; i < 4; ++i) { r[i] = mvm_fmlp::tf32_round(r[i]); a[i] = mvm_fmlp::tf32_round(a[i]); }
            mvm_fmlp::mma_tf32(acc[0], a, r[0], r[1]);
            mvm_fmlp::mma_tf32(acc[1], a, r[2], r[3]);
          }
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) tot[j][e] = t == 0 ? acc[j][e] : tot[j][e] + acc[j][e];
        }
        }
        if (mine) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {   // gemm's stores: a lane's two adjacent columns of rows g and g + 8
            *reinterpret_cast<float2*>(dZ + (size_t)g * HP + cb * 16 + j * 8 + 2 * lq) = make_float2(tot[j][0], tot[j][1]);
            *reinterpret_cast<float2*>(dZ + (size_t)(g + 8) * HP + cb * 16 + j * 8 + 2 * lq) = make_float2(tot[j][2], tot[j][3]);
          }
        }
        __syncthreads();   // the next layer's elementwise dZ reads every warp's columns; the staging is free
        if (owndw) {
          // owner-side dW (MVM_SPLIT_OWNDW): row blocks [ob0, ob1) of share s's dW_l, at most 3 a chunk. Every slice's tiles at once: its
          // H_(l-1) rows [R][the chunk's rows] (XH[l - 1]) and its dZ [R][w] (slot (t, 0) of XD[l & 1], zero past w). Then each row block: for
          // each slice t in order, the member partial's GEMM over the slice's R rows (gemm<true, false, ...>'s fragments, k-steps and rounding)
          // into fresh accumulators, added to the block's sum (slice 0 copied). The sum goes where slice 0's partial sat (member s's rows of
          // d.dW[l]). Two accumulator sets of one row block a warp: a third set (all three blocks at once) spilled every streamed entry.
          float* const tiles = sm + d.Ws[0];
          const int nrb_all = HR >> 4, ob0 = q * nrb_all / Q, ob1 = (q + 1) * nrb_all / Q, n0 = cb * 16 + g;
          float* const dsum = d.dW[l] + (size_t)s * HR * HP;
          for (int c0 = ob0; c0 < ob1; c0 += 3) {
            const int nrb = min(3, ob1 - c0), m0 = 16 * c0;
            for (int t = 0; t < Q; ++t) {
              float* const ab = tiles + (size_t)t * own_slot; float* const bb = ab + (size_t)R * own_lda;
              const float* const ha = d.XH[l - 1] + (size_t)t * R * HR + m0;
              mvm_fmlp::for_cells(R, 4 * nrb, [&](int r, int c4) { mvm_fmlp::cp_async16(ab + (size_t)r * own_lda + 4 * c4, ha + (size_t)r * HR + 4 * c4); });
              const float* const zt = d.XD[l & 1] + (size_t)(t * SV) * R * HR + h0;
              mvm_fmlp::for_cells(R, w16 >> 2, [&](int r, int c4) {
                float* const dst = bb + (size_t)r * own_ldb + 4 * c4;
                if (4 * c4 < w) mvm_fmlp::cp_async16(dst, zt + (size_t)r * HR + 4 * c4); else *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
              });
            }
            mvm_fmlp::cp_commit_wait();
            __syncthreads();
            if (mine) {
              for (int i = 0; i < nrb; ++i) {
                float sum[2][4];
                for (int t = 0; t < Q; ++t) {
                  const float* const A = tiles + (size_t)t * own_slot + 16 * i; const float* const B = tiles + (size_t)t * own_slot + (size_t)R * own_lda;
                  float acc[2][4] = {};
#pragma unroll
                  for (int kk = 0; kk < R; kk += 8) {
                    const float* const b0 = B + (size_t)(kk + lq) * own_ldb + n0; const float* const b4 = b0 + 4 * own_ldb;
                    const float* const a0 = A + (size_t)(kk + lq) * own_lda + g; const float* const a4 = a0 + 4 * own_lda;
                    const unsigned b00 = mvm_fmlp::tf32_in(b0[0]), b01 = mvm_fmlp::tf32_in(b4[0]), b10 = mvm_fmlp::tf32_in(b0[8]), b11 = mvm_fmlp::tf32_in(b4[8]);
                    const unsigned a[4] = {mvm_fmlp::tf32_in(a0[0]), mvm_fmlp::tf32_in(a0[8]), mvm_fmlp::tf32_in(a4[0]), mvm_fmlp::tf32_in(a4[8])};
                    mvm_fmlp::mma_tf32(acc[0], a, b00, b01);
                    mvm_fmlp::mma_tf32(acc[1], a, b10, b11);
                  }
#pragma unroll
                  for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) sum[j][e] = t == 0 ? acc[j][e] : sum[j][e] + acc[j][e];
                }
                float* const r0 = dsum + (size_t)(m0 + 16 * i + g) * HP + cb * 16 + 2 * lq; float* const r8 = r0 + 8 * (size_t)HP;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                  *reinterpret_cast<float2*>(r0 + j * 8) = make_float2(sum[j][0], sum[j][1]);
                  *reinterpret_cast<float2*>(r8 + j * 8) = make_float2(sum[j][2], sum[j][3]);
                }
              }
            }
            __syncthreads();   // every warp is done with the tiles before the next chunk's land on them
          }
        }
        ph(sub_b(p, 6));
        continue;   // dZ holds this member's columns of dX, zero past w (the W block's zero rows), as the reduce-scatter left it
      }
      if constexpr (STREAM) gemm_dx_streamed(dZ, HP, d, t0, t1, h0, st, d.w[l] + h0, H, HR, H, w, [&](int t) { return d.XD[l & 1] + (size_t)(q * SV + t) * R * HR; }, HR);
      else for (int t = t0; t < t1; ++t) {   // each virtual share's full-width partial dZ_t W_l,t^T, its own columns its whole K
        const int c = d.share0[t] - h0, wt = d.share0[t + 1] - d.share0[t];
        float* xd = d.XD[l & 1] + (size_t)(q * SV + t) * R * HR;   // alternating with the layer: a slot is rewritten only after the next barrier (mvm_fused_split.h)
        for (int n0 = 0; n0 < HR; n0 += 128) mvm_fmlp::gemm_r<false, true>(dZ + c, HP, Ws + n0 * HP + c, HP, xd + n0, HR, R, min(128, HR - n0), wt);
      }
      ph(sub_b(p, 4));
      if (!mvm_gang::cross(ctl, member, members, at, abort)) return;
      ph(sub_b(p, 5));
      mvm_fmlp::for_cells(R, HP / 4, [&](int r, int c4) {   // the reduce-scatter: this member's columns of the slice's SV partials, in virtual-share order
        float4 x[SV];
#pragma unroll
        for (int t = 0; t < SV; ++t) x[t] = ld_cg4(d.XD[l & 1] + ((size_t)(q * SV + t) * R + r) * HR + h0 + 4 * c4);
        float4 a = x[0];
#pragma unroll
        for (int t = 1; t < SV; ++t) { a.x += x[t].x; a.y += x[t].y; a.z += x[t].z; a.w += x[t].w; }
        const int c = 4 * c4;   // columns past w belong to the next member: zero here
        a.x = c < w ? a.x : 0.f; a.y = c + 1 < w ? a.y : 0.f; a.z = c + 2 < w ? a.z : 0.f; a.w = c + 3 < w ? a.w : 0.f;
        *reinterpret_cast<float4*>(dZ + r * HP + c) = a;
      });
      __syncthreads();
      ph(sub_b(p, 6));
    }
  }
  ph(5);
  if (!mvm_gang::cross(ctl, member, members, at, abort)) return;
  ph(6);
  // ---- owners, parameters: the Q members of share s split its parameters in groups of four, every layer's and the head's in one index; each sums
  // the Q slices' partials in slice order (or reads the owner-side dW's one sum). The groups are dealt in chunks of NT, chunk c to member q = c mod Q,
  // so each member takes its share of every kind: contiguous ranges gave the last member all of the head's groups, which cost Q reads each where a
  // hidden layer's owner-summed group costs one (the split bed's sub-phases, Nov 13: at h232-d3 x8 the last slice's owners spent 40k cycles a step against a mean of
  // 20k, and every member waited for them at the step's last barrier). A group's arithmetic does not depend on who sums it.
  {
    const int w4 = w >> 2, V4 = V >> 2;
    int ng = 0, beg[MVM_SPLIT_MAXL + 1];
    for (int l = 0; l < L; ++l) { beg[l] = ng; ng += ((l ? H : E) + 1) * w4; }   // the layer's weight rows, then its bias
    beg[L] = ng; ng += w * V4;
    // The partials member (ODX false, streamed) is the streamed member as it was before the owner-side backward, its contiguous dealing included
    // (there every group costs Q reads): on the executor, its gangs ran 1.6% slower on the owner-side entry (h440-d2 x8, shares of different widths, Nov 15).
    constexpr bool CONTIG = STREAM && !ODX;
    const int g0 = CONTIG ? (int)((long long)q * ng / Q) : q * NT, g1 = CONTIG ? (int)((long long)(q + 1) * ng / Q) : ng, gstep = CONTIG ? NT : Q * NT;
    for (int g = g0 + tid; g < g1; g += gstep) {
      const float* src; size_t stride; float* dst;   // slice 0's partial of these four parameters, the distance to the next slice's, and the parameters
      bool summed = false;                           // a hidden layer's weights the owner-side dW summed (one sum, no partials)
      int l = 0; while (l < L && g >= beg[l + 1]) ++l;
      if (l < L) {
        const int Kl = l ? H : E, KR = l ? HR : E, j = g - beg[l], k = j / w4, c = 4 * (j % w4);
        if (k < Kl) { src = d.dW[l] + ((size_t)s * KR + k) * HP + c; stride = (size_t)S * KR * HP; dst = d.w[l] + (size_t)k * H + h0 + c; summed = owndw_ok && l > 0; }
        else { src = d.db[l] + (size_t)s * HP + c; stride = (size_t)S * HP; dst = d.b[l] + h0 + c; }
      } else {
        const int j = g - beg[L], r = j / V4, v = 4 * (j % V4);
        src = d.dWh + ((size_t)s * HP + r) * VP + v; stride = (size_t)S * HP * VP; dst = d.wh + (size_t)(h0 + r) * V + v;
      }
      float4 a;
      if (summed) a = ld_cg4(src);   // the owner-side dW's sum, in slice 0's place
      else {
        float4 x[Q];
#pragma unroll
        for (int t = 0; t < Q; ++t) x[t] = ld_cg4(src + (size_t)t * stride);
        a = x[0];
#pragma unroll
        for (int t = 1; t < Q; ++t) { a.x += x[t].x; a.y += x[t].y; a.z += x[t].z; a.w += x[t].w; }
      }
      float4 o = *reinterpret_cast<float4*>(dst);
      o.x -= lr * a.x; o.y -= lr * a.y; o.z -= lr * a.z; o.w -= lr * a.w;
      *reinterpret_cast<float4*>(dst) = o;
    }
  }
  __syncthreads();
  ph(7);
  // ---- owners, embedding rows [erow0, erow1): the step's tokens that fall in them, compacted in token order by warp ballots
  {
    for (int base = 0; base < T; base += NT) {
      const int k = base + tid, tok = k < T ? sid[k] : -1;
      const bool mine = tok >= erow0 && tok < erow1;
      const unsigned bal = __ballot_sync(0xffffffffu, mine);
      if (lane == 0) wcnt[warp] = __popc(bal);
      __syncthreads();
      if (tid == 0) { int off = base ? wcnt[8] : 0; for (int i = 0; i < NT / 32; ++i) { const int c = wcnt[i]; wcnt[i] = off; off += c; } wcnt[8] = off; }
      __syncthreads();
      if (mine) mlist[wcnt[warp] + __popc(bal & ((1u << lane) - 1u))] = k;
      __syncthreads();
    }
    const int nm = wcnt[8];
    for (int i = tid; i < nm * E; i += NT) {   // each matched token's gradient: its slice's SV parts in virtual-share order
      const int j = i / E, e = i % E, k = mlist[j], sq = k / R, r = k % R;
      float x[SV];
#pragma unroll
      for (int t = 0; t < SV; ++t) x[t] = __ldcg(d.dX0 + ((size_t)(sq * SV + t) * R + r) * E + e);   // other members' partials: L2, never a stale L1 line
      float a = x[0];
#pragma unroll
      for (int t = 1; t < SV; ++t) a += x[t];
      gtok[j * E + e] = a;
    }
    __syncthreads();
    for (int i = tid; i < (erow1 - erow0) * E; i += NT) {   // each owned row: its tokens' gradients in token order
      const int row = erow0 + i / E, e = i % E; float a = 0.f; bool any = false;
      for (int j = 0; j < nm; ++j) if (sid[mlist[j]] == row) { a += gtok[j * E + e]; any = true; }
      if (any) d.we[(size_t)row * E + e] -= lr * a;
    }
  }
  if (member == 0 && tid == 0) { float sum = 0.f; for (int t = 0; t < Q; ++t) sum += __ldcg(d.loss_part + t); if (loss_out) { loss_out[0] = sum * inv_t; loss_out[1] = inv_t; } }
  if (!mvm_gang::cross(ctl, member, members, at, abort)) return;
  ph(8);
}

}  // namespace mvm_fsplit
