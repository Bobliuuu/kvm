#pragma once
// The batched MLP step, device side: one member's part of one step of R runs (kvm_fused_batch.h has the design, the descriptor and the
// layout). Every GEMM streams its operands from global memory through a ring of KVM_BATCH_NST k chunks in shared memory (cp.async), and
// multiplies them with m16n8k8 TF32 mma, fp32 accumulators in registers, operands rounded to TF32 as the other fused steps round them
// (kvm_fused_mlp.cuh). Each output element is summed over its whole K in one fixed order by one item, so a run is deterministic and each
// run's result is bitwise the result it would have alone.
#include <cstdint>
#include "kvm_fused_batch.h"
#include "kvm_fused_mlp.cuh"
#include "kvm_gang.cuh"

// An item clock for beds: KVM_BATCH_CLOCK(kind, mark) runs at an item's start (mark 0), after its operand loop (1), after its epilogue (2) and at its
// end (3). A bed defines it before including this header; by default it compiles to nothing.
#ifndef KVM_BATCH_CLOCK
#define KVM_BATCH_CLOCK(kind, mark)
#endif

namespace kvm_fbatch {

using kvm_fmlp::NT;
using kvm_fmlp::tf32_in;
using kvm_fmlp::tf32_round;
constexpr int T = KVM_BATCH_T, BK = KVM_BATCH_BK, NST = KVM_BATCH_NST, RBT = T / 16;   // RBT: the step's 6 row blocks
constexpr int AP = BK + 4;    // pitch of a streamed [rows][BK] chunk read by ldmatrix: 36 floats, so 8 rows fall in 8 distinct bank quads
constexpr int E48 = 48, VP = 112;

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// rows x cols floats (cols a multiple of 4) into shared memory at pitch sp: element (r, c) from src(r)[c] when r < vr and c < vc (vc a
// multiple of 4, every row 16-byte aligned), by cp.async; every other element is zero (plain stores, visible after the next barrier).
template <class Src>
__device__ __forceinline__ void stage(float* dst, int sp, int rows, int cols, int vr, int vc, Src src) {
  const int c4n = cols >> 2;
  for (int i = threadIdx.x; i < rows * c4n; i += NT) {
    const int r = i / c4n, c = (i - r * c4n) * 4;
    float* t = dst + r * sp + c;
    if (r < vr && c < vc) kvm_fmlp::cp_async16(t, src(r) + c);
    else *reinterpret_cast<float4*>(t) = make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

// A k-chunk stream: the 16-byte pieces one copier thread issues of every chunk of one operand, their shared-memory offsets and chunk-0 sources
// computed once an item. TOTAL pieces a chunk spread over NC copier threads, the block's last NC: P = ceil(TOTAL / NC) a copier, all at
// compile time, so the indices are static and the offsets live in registers (read in the SASS: a loop over a run-time piece count spent a branch on every
// unused slot, and its fill wrote the arrays through local memory). Two operand shapes:
//   along<ROWS>(b, vr, sp, row): [ROWS][BK] chunks of an operand whose k runs along its rows, row(r) its row's first element at k = 0: piece
//                                (r, q) copies row(r) + 4 q + c BK, zero for r >= vr;
//   down<COLS>(b, vc, sp, ld):   [BK][COLS] chunks of an operand whose k runs down its rows (pitch ld): piece (kk, q) copies b + kk ld + 4 q
//                                + c BK ld, zero for 4 q >= vc.
// A piece past the chunk's last k (kv) is zero too; full chunks skip that test (issue<true>).
template <int TOTAL, int NC>
struct Stream {
  static_assert(NC % 32 == 0 && NC > 0 && NC <= NT, "copiers are whole warps of the block");
  static constexpr int P = (TOTAL + NC - 1) / NC;
  int soff[P], kq[P], goff[P];   // shared offset in a stage (-1: no piece), the piece's k in its chunk, its source at chunk 0 (-1: zero)
  const float* base;
  int step, kst;                   // a chunk's stride in the source (BK x kst), and one k's
  __device__ __forceinline__ static int copier() { return (int)threadIdx.x - (NT - NC); }   // this thread's index among the copiers (< 0: none)
  template <int ROWS, class Row>
  __device__ __forceinline__ void along(const float* b, int vr, int sp, Row row) {
    static_assert(ROWS * (BK / 4) == TOTAL, "a chunk of ROWS rows is TOTAL pieces");
    base = b; step = BK; kst = 1;
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const int i = copier() + p * NC, r = i / (BK / 4), q = i % (BK / 4);
      const bool on = i >= 0 && i < TOTAL;
      soff[p] = on ? r * sp + 4 * q : -1; kq[p] = 4 * q; goff[p] = on && r < vr ? row(r) + 4 * q : -1;
    }
  }
  template <int COLS>
  __device__ __forceinline__ void down(const float* b, int vc, int sp, int ld) {
    static_assert(BK * (COLS / 4) == TOTAL && COLS % 4 == 0, "a chunk of COLS columns is TOTAL pieces");
    base = b; step = BK * ld; kst = ld;
    constexpr int c4 = COLS / 4;
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const int i = copier() + p * NC, kk = i / c4, q = i % c4;
      const bool on = i >= 0 && i < TOTAL;
      soff[p] = on ? kk * sp + 4 * q : -1; kq[p] = kk; goff[p] = on && 4 * q < vc ? kk * ld + 4 * q : -1;
    }
  }
  // chunk c into the stage at stage_base (copiers only)
  template <bool FULL>
  __device__ __forceinline__ void issue(float* stage_base, int c, int kv) const {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      if (TOTAL % NC != 0 && soff[p] < 0) continue;
      float* t = stage_base + soff[p];
      if (goff[p] >= 0 && (FULL || kq[p] < kv)) kvm_fmlp::cp_async16(t, base + goff[p] + (size_t)c * step);
      else *reinterpret_cast<float4*>(t) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  // two pieces of 16 into one stage (copiers only): the stage's k 0-15 from k0a, its k 16-31 from k0b (absent: k0b < 0, zeros), va and vb of
  // each piece's k holding data; the grid path's walk, whose consecutive pieces may lie anywhere in k
  __device__ __forceinline__ void issue2(float* stage_base, int k0a, int va, int k0b, int vb) const {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      if (TOTAL % NC != 0 && soff[p] < 0) continue;
      float* t = stage_base + soff[p];
      const bool hi = kq[p] >= 16;
      const int k0 = hi ? k0b - 16 : k0a;
      if (goff[p] >= 0 && (hi ? k0b >= 0 && kq[p] - 16 < vb : kq[p] < va)) kvm_fmlp::cp_async16(t, base + goff[p] + (size_t)k0 * kst);
      else *reinterpret_cast<float4*>(t) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  // the chunk whose first k is k0 (any multiple of 4) into the stage, kv of its k holding data (copiers only): the grid path's walk
  template <bool FULL>
  __device__ __forceinline__ void issue_k(float* stage_base, int k0, int kv) const {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      if (TOTAL % NC != 0 && soff[p] < 0) continue;
      float* t = stage_base + soff[p];
      if (goff[p] >= 0 && (FULL || kq[p] < kv)) kvm_fmlp::cp_async16(t, base + goff[p] + (size_t)k0 * kst);
      else *reinterpret_cast<float4*>(t) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
};

// The copier threads of an item whose output tile gives MW warps work: when half the warps or more own nothing, they copy for the others, so an
// MMA warp's chunk is its barrier and its k-steps (read in the SASS: on a 32 x 32 tile every MMA warp also issued two pieces a chunk, about a third of
// its chunk's instructions); otherwise every thread copies.
template <int MW>
constexpr int kCopiers = MW <= 4 ? (8 - MW) * 32 : NT;

// A per-chunk clock for beds: KVM_BATCH_CHUNK_CLOCK(kind, t) is called by k_loop with the chunk's three intervals (t[0] the wait and barrier,
// t[1] the copies, t[2] the k-steps, in cycles summed over the item's chunks, and t[3] its chunk count). By default it compiles to nothing.
#ifndef KVM_BATCH_CHUNK_CLOCK
#define KVM_BATCH_CHUNK_CLOCK(kind, t)
#define KVM_BATCH_CHUNK_STAMPS 0
#else
#define KVM_BATCH_CHUNK_STAMPS 1
#endif

// The k-chunk loop of a streamed item: an NST-deep ring of [A chunk | B chunk] stages over K. The copiers issue chunk c + NST - 1 while every warp
// runs chunk c's k-steps (mma(As, Bs, kv, c), kv the chunk's k count: BK, a constant, on every chunk but a last partial one). With DATA, Kd is
// where the operands' data ends: past it the streams write zeros, so a chain may run on over zero columns to K, as the split step's head runs
// its chains to VP over the zero columns past V. A compile-time choice: its two ints made fwd_tile spill when every caller carried them.
template <int KIND, int NC, bool DATA = false, class SA, class SB, class MMA>
__device__ __forceinline__ void k_loop(float* sm, int stage_floats, int asz, int K, const SA& sa, const SB& sb, MMA mma, int Kd = 0) {
  const bool copier = (int)threadIdx.x >= NT - NC;
  const int nk = (K + BK - 1) / BK, nfull = K / BK;
  auto issue = [&](int c) {
    float* As = sm + (c % NST) * stage_floats;
    if constexpr (DATA) {   // the streams stop at Kd (at most K): zeros past it
      if (c < Kd / BK) { sa.template issue<true>(As, c, BK); sb.template issue<true>(As + asz, c, BK); }
      else { const int kv = Kd - c * BK > 0 ? Kd - c * BK : 0; sa.template issue<false>(As, c, kv); sb.template issue<false>(As + asz, c, kv); }
    } else {
      if (c < nfull) { sa.template issue<true>(As, c, BK); sb.template issue<true>(As + asz, c, BK); }
      else { const int kv = K - c * BK; sa.template issue<false>(As, c, kv); sb.template issue<false>(As + asz, c, kv); }
    }
  };
#pragma unroll
  for (int p = 0; p < NST - 1; ++p) { if (copier && p < nk) issue(p); cp_commit(); }
  [[maybe_unused]] long long tc[4] = {0, 0, 0, 0};
  for (int c = 0; c < nk; ++c) {
    [[maybe_unused]] long long t0 = 0, t1 = 0, t2 = 0;
    if constexpr (KVM_BATCH_CHUNK_STAMPS) t0 = clock64();
    cp_wait<NST - 2>();
    __syncthreads();   // chunk c landed everywhere; every warp is past chunk c - 1, whose stage the next issue refills
    if constexpr (KVM_BATCH_CHUNK_STAMPS) t1 = clock64();
    if (copier && c + NST - 1 < nk) issue(c + NST - 1);
    cp_commit();
    if constexpr (KVM_BATCH_CHUNK_STAMPS) t2 = clock64();
    const float* As = sm + (c % NST) * stage_floats;
    if (c < nfull) mma(As, As + asz, BK, c); else mma(As, As + asz, K - c * BK, c);
    if constexpr (KVM_BATCH_CHUNK_STAMPS) { const long long t3 = clock64(); tc[0] += t1 - t0; tc[1] += t2 - t1; tc[2] += t3 - t2; ++tc[3]; }
  }
  cp_wait<0>();
  KVM_BATCH_CHUNK_CLOCK(KIND, tc);
}

// Every item checks the shared memory its own layout uses against the arena the descriptor was sized for (kvm_batch_smem), so an item the
// sizing missed stops the step at once instead of writing over another buffer (found on the A100: the split dX tile was never counted, and large tiles
// overran the arena: wrong weights, and once an illegal address; layer 0's token ids sat in the last 128 floats against an FR-float reserve).
__device__ __forceinline__ void need_smem(const KvmBatchDesc& d, int floats) { if (floats > d.smem_floats) __trap(); }

// Which part of an (MRB x 16) x (NCB x 16) output a warp owns: column block cb (16 columns) and row blocks [rb0, rb0 + nrb). WPC warps share
// a column block, each RB row blocks of it; warps past NCB x WPC own nothing (on = false). One B fragment pair then serves RB A fragments.
template <int MRB, int NCB>
struct WarpMap {
  static_assert(NCB >= 1 && NCB <= 8 && MRB >= 1, "a tile is at most 128 columns");
  static constexpr int WPC = (8 / NCB) < MRB ? (8 / NCB) : MRB;
  static constexpr int RB = (MRB + WPC - 1) / WPC;
  int cb, rb0, nrb;
  bool on;
  __device__ __forceinline__ WarpMap() {
    const int w = threadIdx.x >> 5, rg = w / NCB;
    cb = w % NCB;
    on = rg < WPC;
    rb0 = rg * RB;
    nrb = on ? (MRB - rb0 < RB ? MRB - rb0 : RB) : 0;
  }
};

// One k8 step of a warp's part: acc[i][j] (row block rb0 + i, the j-th 8 columns of column block cb) += A[rows][kk, kk + 8) B[kk, kk + 8)[cols].
// A: row-major [m][k] at pitch lda, read by ldmatrix (lda a multiple of 4), or TA: stored [k][m], read by 32-bit loads.
// B: TB: stored [n][k] at pitch ldb, read by ldmatrix, or [k][n], read by 32-bit loads. Fragment layouts as kvm_fgrid::gemm_tall_n.
template <bool TA, bool TB, int RB>
__device__ __forceinline__ void mma_k8(float (&acc)[RB][2][4], const float* As, int lda, const float* Bs, int ldb, int rb0, int nrb, int cb, int kk) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, la = lane & 7, b3 = (lane >> 3) & 1, b4 = lane >> 4;
  unsigned b00, b01, b10, b11;
  if constexpr (TB) {
    unsigned r[4];
    kvm_fmlp::ldmatrix_x4(r, Bs + (cb * 16 + la + 8 * b4) * ldb + 4 * b3 + kk);
    b00 = tf32_round(r[0]); b01 = tf32_round(r[1]); b10 = tf32_round(r[2]); b11 = tf32_round(r[3]);
  } else {
    const float* b = Bs + (kk + t) * ldb + cb * 16 + g;
    b00 = tf32_in(b[0]); b01 = tf32_in(b[4 * ldb]); b10 = tf32_in(b[8]); b11 = tf32_in(b[4 * ldb + 8]);
  }
#pragma unroll
  for (int i = 0; i < RB; ++i) {
    if (i < nrb) {
      const int rb = rb0 + i;
      unsigned a[4];
      if constexpr (TA) {
        const float* p = As + (kk + t) * lda + rb * 16 + g;
        a[0] = tf32_in(p[0]); a[1] = tf32_in(p[8]); a[2] = tf32_in(p[4 * lda]); a[3] = tf32_in(p[4 * lda + 8]);
      } else {
        kvm_fmlp::ldmatrix_x4(a, As + (rb * 16 + la + 8 * b3) * lda + 4 * b4 + kk);
        a[0] = tf32_round(a[0]); a[1] = tf32_round(a[1]); a[2] = tf32_round(a[2]); a[3] = tf32_round(a[3]);
      }
      kvm_fmlp::mma_tf32(acc[i][0], a, b00, b01);
      kvm_fmlp::mma_tf32(acc[i][1], a, b10, b11);
    }
  }
}

// Calls fn(m, n, v) for each accumulator of a warp's part, (m, n) its output position.
template <int RB, class Fn>
__device__ __forceinline__ void for_acc(const float (&acc)[RB][2][4], int rb0, int nrb, int cb, Fn fn) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < RB; ++i)
    if (i < nrb)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int m = (rb0 + i) * 16 + g, n = cb * 16 + j * 8 + 2 * t;
        fn(m, n, acc[i][j][0]); fn(m, n + 1, acc[i][j][1]); fn(m + 8, n, acc[i][j][2]); fn(m + 8, n + 1, acc[i][j][3]);
      }
}

template <int RB>
__device__ __forceinline__ void zero(float (&acc)[RB][2][4]) {
#pragma unroll
  for (int i = 0; i < RB; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
}

// ---- The grid step's arithmetic (KvmBatchDesc::gHp > 0): a sum over the grid's blocks, as kvm_fused_grid.cuh takes it.
// n blocks over [0, Kp), block i from kvm_batch_goff(Kp, n, i) (each at least 16 wide), each block's partial one chain from zero in k8 steps
// from its start; the partials summed as the grid's sum_cg<8> sums them: partial i into chain i mod 8, each chain from 0.f (its first add turns
// -0 into +0), then ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)), an empty chain being 0.f. The walk visits the blocks chain by chain
// (0, 8, 16, ..., then 1, 9, ...), so each chain's partials are summed as they finish and each chain joins the tree once. It steps in pieces
// of 16 (every block is whole granules), and a ring stage of BK = 32 takes the walk's next two pieces wherever they lie in k: a stage is full
// whatever the blocks' widths, so a sum takes as many stages as a straight walk over Kp (with one piece a stage, 16-wide blocks took twice
// as many: h200-d2's 13-block sums ran x2.2-3.2 slower on the 5070). The walk is a table built on the host (kvm_batch_grid_walk), one word a
// piece: computing block edges in the loop cost a division each, several a stage. Past Kd (where the operands' data ends) the streams write
// zeros, as the grid's padded blocks hold zeros.
// Leaf u (chain u's sum, or 0.f for an empty chain) joins ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)): p1, p2 and p3 hold the pending
// left operands of each level, and leaf 7 leaves the total in p3. u is the item's, alike in every thread.
template <int RB>
__device__ __forceinline__ void tree_leaf(float (&p1)[RB][2][4], float (&p2)[RB][2][4], float (&p3)[RB][2][4], const float (&x)[RB][2][4], int nrb,
                                          int u, bool empty) {
#pragma unroll
  for (int i = 0; i < RB; ++i)
    if (i < nrb)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float leaf = empty ? 0.f : x[i][j][e];
          if (!(u & 1)) { p1[i][j][e] = leaf; continue; }
          const float y = p1[i][j][e] + leaf;
          if (!(u & 2)) { p2[i][j][e] = y; continue; }
          const float z = p2[i][j][e] + y;
          p3[i][j][e] = (u & 4) ? p3[i][j][e] + z : z;
        }
}
// The k-chunk loop of a grid sum: the ring of k_loop over the walk's stages of two pieces (issued NST - 1 ahead, across blocks). mma8(acc, As,
// Bs, kk) runs a warp's k8 step into a block's partial; chunk(As, Bs, kv, k0) is the item's other work on the staged columns (a weight's update
// and bias): once a stage when its two pieces are adjacent in k, else once a piece, any order being fine for it since every column is in
// exactly one piece. The sum lands in out; `on` and nrb are the warp's part.
// The sum's shape, a compile-time choice that keeps a warp's accumulator sets to what it needs: GSUM_ANY (a chain may hold several blocks: the
// block's partial, its chain and three tree nodes), GSUM_ONE (n <= 8: a chain is one block, 0.f + its partial, so no chain set) and
// GSUM_SINGLE (n = 1: the sum is 0.f + the block's partial exactly, since the tree only adds +0 leaves to it, so no tree either).
enum { GSUM_ANY = 0, GSUM_ONE = 1, GSUM_SINGLE = 2 };
template <int KIND, int NC, int RB, int SHAPE = GSUM_ANY, class SA, class SB, class MMA8, class CHUNK>
__device__ __forceinline__ void grid_k_loop(float* sm, int stage_floats, int asz, const unsigned* tab, int np, int n, int Kd, const SA& sa,
                                            const SB& sb, bool on, int nrb, float (&out)[RB][2][4], MMA8 mma8, CHUNK chunk) {
  const bool copier = (int)threadIdx.x >= NT - NC;
  if ((SHAPE == GSUM_ONE && n > 8) || (SHAPE == GSUM_SINGLE && n != 1)) __trap();
  constexpr int CRB = SHAPE == GSUM_ANY ? RB : 1, TRB = SHAPE == GSUM_SINGLE ? 1 : RB;   // the sets a shape keeps (the others are unused)
  float acc[RB][2][4], chain[CRB][2][4], p1[TRB][2][4], p2[TRB][2][4];
  zero(acc);
  zero(chain);
  const int ns = (np + 1) >> 1;   // stages: the table's np pieces two at a time
  auto data = [&](int k0) { const int v = Kd - k0; return v < 16 ? (v > 0 ? v : 0) : 16; };   // a piece's k that hold data
  auto issue = [&](int c) {   // pieces 2c and 2c + 1 into stage c % NST (the second absent at the sum's end: zeros)
    float* As = sm + (c % NST) * stage_floats;
    const int ka = KVM_BATCH_GP_K(__ldg(tab + 2 * c)), kb = 2 * c + 1 < np ? KVM_BATCH_GP_K(__ldg(tab + 2 * c + 1)) : -1;
    const int va = data(ka), vb = kb >= 0 ? data(kb) : 0;
    if (kb == ka + 16 && va == 16 && vb == 16) { sa.template issue_k<true>(As, ka, BK); sb.template issue_k<true>(As + asz, ka, BK); }
    else { sa.issue2(As, ka, va, kb, vb); sb.issue2(As + asz, ka, va, kb, vb); }
  };
  auto fold = [&](unsigned w) {   // a block's last piece: its partial joins its chain (the chain's first from 0.f), and a chain's last its leaf
    if constexpr (SHAPE == GSUM_ANY) {
      const bool first = KVM_BATCH_GP_FIRST(w);
#pragma unroll
      for (int i = 0; i < RB; ++i)
        if (i < nrb)
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { chain[i][j][e] = (first ? 0.f : chain[i][j][e]) + acc[i][j][e]; acc[i][j][e] = 0.f; }
      if (KVM_BATCH_GP_CHAIN_END(w)) tree_leaf(p1, p2, out, chain, nrb, KVM_BATCH_GP_CHAIN(w), false);
    } else {   // one block a chain: the leaf is 0.f + the partial (the single block's sum, at GSUM_SINGLE)
#pragma unroll
      for (int i = 0; i < RB; ++i)
        if (i < nrb)
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f + acc[i][j][e];
      if constexpr (SHAPE == GSUM_ONE) { tree_leaf(p1, p2, out, acc, nrb, KVM_BATCH_GP_CHAIN(w), false); zero(acc); }
      else {
#pragma unroll
        for (int i = 0; i < RB; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) out[i][j][e] = acc[i][j][e];
      }
    }
  };
#pragma unroll
  for (int p = 0; p < NST - 1; ++p) { if (copier && p < ns) issue(p); cp_commit(); }
  for (int c = 0; c < ns; ++c) {
    const unsigned wa = __ldg(tab + 2 * c), wb = 2 * c + 1 < np ? __ldg(tab + 2 * c + 1) : 0u;   // ahead of the wait: their latency hides there
    cp_wait<NST - 2>();
    __syncthreads();   // stage c landed everywhere; every warp is past stage c - 1, whose slot the next issue refills
    if (copier && c + NST - 1 < ns) issue(c + NST - 1);
    cp_commit();
    const float* As = sm + (c % NST) * stage_floats;
    const int ka = KVM_BATCH_GP_K(wa), kb = 2 * c + 1 < np ? KVM_BATCH_GP_K(wb) : -1;
    if (on) { mma8(acc, As, As + asz, 0); mma8(acc, As, As + asz, 8); }   // piece a: k8 steps 0-1
    if (on && KVM_BATCH_GP_BLOCK_END(wa)) fold(wa);
    // the item's other work: on the stage's columns at once when its pieces are adjacent, else on each piece
    if (kb == ka + 16) chunk(As, As + asz, data(ka) + data(kb), ka);
    else { chunk(As, As + asz, data(ka), ka); if (kb >= 0) chunk(As + 16, As + asz + 16, data(kb), kb); }
    if (kb >= 0) {   // piece b: k8 steps 2-3
      if (on) { mma8(acc, As, As + asz, 16); mma8(acc, As, As + asz, 24); }
      if (on && KVM_BATCH_GP_BLOCK_END(wb)) fold(wb);
    }
  }
  cp_wait<0>();
  if constexpr (SHAPE != GSUM_SINGLE)
    if (on)
      for (int u = n < 8 ? n : 8; u < 8; ++u) tree_leaf(p1, p2, out, acc, nrb, u, true);   // the empty chains' zeros (the leaf's value unread)
}
// No other work on a grid sum's chunks.
struct NoChunk { __device__ __forceinline__ void operator()(const float*, const float*, int, int) const {} };

// A chain over K taken in parts (the descriptor's arithmetic, KvmBatchDesc::Sv). end_of(i) is where part i ends: a multiple of 8, never
// decreasing (an empty part ends where the one before it did: the split step's empty virtual shares when Sv > H / 8, whose partials are zero),
// the last at K rounded up to 8, and -1 past the last part. A chain first calls fold_part at k_end = 0, which folds its leading empty parts
// (zeros: the first copied, as the split step's a = x[0] takes an empty share's zero). After the k8 step that ends at k_end, fold_part folds the part's accumulator into the total
// and zeroes it when that step ends the current part: the first part's is copied, never added to a zero (0 + -0 is +0), and each later one is
// added in part order, as the split step sums its partials (kvm_fused_split.cuh: a = x[0]; a += x[t]). The chain's last k8 step always ends its
// last part, so the total is the whole result. The total is a plain local array beside acc, with the part and its end in two ints: held in a
// struct with member functions, nvcc kept it in local memory (every dw_tile spilled exactly its RB x 8 floats).
template <int RB, class End>
__device__ __forceinline__ void fold_part(float (&tot)[RB][2][4], float (&acc)[RB][2][4], int nrb, int k_end, int& part, int& end, End end_of) {
  while (k_end == end) {   // a loop: an empty part (the split step's empty virtual shares, Sv > H / 8) ends where the one before it did
#pragma unroll
    for (int i = 0; i < RB; ++i)
      if (i < nrb)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) { tot[i][j][e] = part ? tot[i][j][e] + acc[i][j][e] : acc[i][j][e]; acc[i][j][e] = 0.f; }
    end = end_of(++part);
  }
}
// The chain over the step's rows (K = T, from operands already in shared memory), into tot: at Sv > 0 by the split step's 16-row slices, each
// slice's two k8 steps from zero and folded in slice order; at Sv = 0, one chain. mma8(acc, kk) runs the k8 step [kk, kk + 8). The slices are
// unrolled so the fold is a straight line: a fold under a run-time test inside a rolled loop kept tot in local memory, loaded and stored around
// every k8 step (dw_tile's SASS).
template <int RB, class MMA8>
__device__ __forceinline__ void row_chain(const KvmBatchDesc& d, float (&tot)[RB][2][4], float (&acc)[RB][2][4], int nrb, MMA8 mma8) {
  if (d.Sv > 0) {
#pragma unroll
    for (int q = 0; q < T / KVM_BATCH_SLICE; ++q) {
      zero(acc);
#pragma unroll
      for (int kk = q * KVM_BATCH_SLICE; kk < (q + 1) * KVM_BATCH_SLICE; kk += 8) mma8(acc, kk);
#pragma unroll
      for (int i = 0; i < RB; ++i)
        if (i < nrb)
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) tot[i][j][e] = q ? tot[i][j][e] + acc[i][j][e] : acc[i][j][e];
    }
  } else {
    zero(tot);
#pragma unroll 4
    for (int kk = 0; kk < T; kk += 8) mma8(tot, kk);
  }
}

// Where the parts end: a chain over the hidden units by the split step's virtual shares (one part, [0, H), at Sv = 0).
struct ShareEnds { const KvmBatchDesc& d; __device__ __forceinline__ int operator()(int i) const { return i < (d.Sv > 0 ? d.Sv : 1) ? d.share0[i + 1] : -1; } };
struct ColEnds {   // a chain over a weight's output columns: the hidden units by shares, or the head's padded vocabulary (VP) in one part
  const KvmBatchDesc& d; bool head;
  __device__ __forceinline__ int operator()(int i) const { return head ? (i ? -1 : VP) : ShareEnds{d}(i); }
};
// A sum over the step's rows of one column (a bias gradient): from 0.f in row order within a slice, the slices folded as Parts folds them, as the
// split step sums a slice's db and its owners the slices; at Sv = 0, one sum over the rows from 0.f.
template <class At>
__device__ __forceinline__ float row_sum(const KvmBatchDesc& d, At at) {
  if (d.Sv <= 0) { float s = 0.f; for (int t = 0; t < T; ++t) s += at(t); return s; }
  float s = 0.f;
  for (int q = 0; q < T / KVM_BATCH_SLICE; ++q) {
    float a = 0.f;
    for (int t = q * KVM_BATCH_SLICE; t < (q + 1) * KVM_BATCH_SLICE; ++t) a += at(t);
    s = q ? s + a : a;
  }
  return s;
}

// ---- F: a forward tile, layer l, rows [r0, r0 + 16 FM) and columns [n0, n0 + NW): Z = X W_l + b_l, Y = gelu(Z), G = gelu'(Z). Layer 0 gathers
// X's rows from the embedding by token id, and each row block's first tile also stores them (X0) for BU. Each output sums its whole K in one
// order, whatever FM and NW are, so the tiles' shape never changes a run's arithmetic.
// GRID: the grid step's arithmetic for a hidden layer (KvmBatchDesc::gHp > 0, l > 0), in instantiations of its own: a function allocates its own
// registers, so the grid path's accumulator sets never press the other paths' (in one function with it, BSA's share fold reloaded two spilled
// values after every k8 step, head_ce's SASS at 63f12a9 + the grid path).
template <int FM, int NCB, bool GRID = false>
__device__ __noinline__ void fwd_tile(const KvmBatchDesc& d, const KvmBatchRun& run, int l, int step, int r0, int n0, float* sm) {
  constexpr int NW = 16 * NCB, BP = NW + 8, FR = 16 * FM, ASZ = FR * AP, BSZ = BK * BP;
  using WM = WarpMap<FM, NCB>;
  need_smem(d, NST * (ASZ + BSZ) + 128 > 2 * FR * (NW + 4) ? NST * (ASZ + BSZ) + 128 : 2 * FR * (NW + 4));   // the ring and the ids; the Y and G tiles
  KVM_BATCH_CLOCK(CK_F, 0);
  const int K = l ? d.H : d.E, H = d.H, vc = H - n0 < NW ? H - n0 : NW;
  const float* X = l ? run.y[l - 1] : run.emb;
  const float* W = run.w[l];
  int* ids = reinterpret_cast<int*>(sm + d.smem_floats - 128);   // layer 0: the block's token ids, read once, at the arena's end
  if (l == 0) {
    for (int i = threadIdx.x; i < FR; i += NT) ids[i] = run.ids[step][r0 + i];
    __syncthreads();
  }
  const WM wm;
  float acc[WM::RB][2][4];
  zero(acc);
  constexpr int NC = kCopiers<NCB * WM::WPC>;
  Stream<FR * (BK / 4), NC> sa;
  Stream<BK * (NW / 4), NC> sb;
  if (l) sa.template along<FR>(X, FR, AP, [&](int r) { return (r0 + r) * K; });
  else sa.template along<FR>(X, FR, AP, [&](int r) { return ids[r] * E48; });
  sb.template down<NW>(W + n0, vc, BP, H);
  if constexpr (GRID) {   // the grid step's arithmetic: Z over the grid's gka K blocks of gHp (kvm_fused_grid.cuh l. 202, 209)
    static_assert(WM::RB <= KVM_BATCH_GRID_MAXRB, "the grid path keeps at most KVM_BATCH_GRID_MAXRB row blocks a warp (kvm_batch_layout)");
    grid_k_loop<CK_F, NC>(sm, ASZ + BSZ, ASZ, d.gtab + d.gtab_off[KVM_BATCH_GSUM_Z], d.gHp / 16, d.gka, K, sa, sb, wm.on, wm.nrb, acc,
                          [&](float (&a)[WM::RB][2][4], const float* As, const float* Bs, int kk) {
                            mma_k8<false, false, WM::RB>(a, As, AP, Bs, BP, wm.rb0, wm.nrb, wm.cb, kk); }, NoChunk{});
  } else
    k_loop<CK_F, NC>(sm, ASZ + BSZ, ASZ, K, sa, sb, [&](const float* As, const float* Bs, int kv, int) {
      if (wm.on)
        for (int kk = 0; kk < kv; kk += 8) mma_k8<false, false, WM::RB>(acc, As, AP, Bs, BP, wm.rb0, wm.nrb, wm.cb, kk);
    });
  KVM_BATCH_CLOCK(CK_F, 1);
  if (l == 0 && n0 == 0)   // the block's first tile keeps its X0 rows for BU, before the epilogue's tiles go over the ring (and the ids past it)
    for (int i = threadIdx.x; i < FR * E48 / 4; i += NT) {
      const int r = i / (E48 / 4), c = (i - r * (E48 / 4)) * 4;
      *reinterpret_cast<float4*>(run.x0 + (size_t)(r0 + r) * E48 + c) = *reinterpret_cast<const float4*>(X + (size_t)ids[r] * E48 + c);
    }
  __syncthreads();   // every warp is done with the ring
  // the epilogue through shared memory: each warp puts gelu(Z) and gelu'(Z) of its accumulators into [FR][NW + 4] tiles, then every thread
  // stores whole 16-byte pieces of the tile's rows: whole sectors, where storing the accumulators straight from the fragments writes 8 bytes
  // to each of 8 rows a warp instruction (the bed's item clock: 12-26k cycles an epilogue)
  constexpr int OP = NW + 4;
  float* const Ys = sm;
  float* const Gs = sm + FR * OP;
  if (wm.on) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const float* b = run.b[l];
    float bj[2][2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int n = wm.cb * 16 + j * 8 + 2 * t;
      bj[j][0] = n < vc ? b[n0 + n] : 0.f;
      bj[j][1] = n + 1 < vc ? b[n0 + n + 1] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < WM::RB; ++i)
      if (i < wm.nrb)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int m = (wm.rb0 + i) * 16 + g, n = wm.cb * 16 + j * 8 + 2 * t;
          float f0, g0, f1, g1, f2, g2, f3, g3;
          kvm_fmlp::gelu_fg(acc[i][j][0] + bj[j][0], f0, g0);
          kvm_fmlp::gelu_fg(acc[i][j][1] + bj[j][1], f1, g1);
          kvm_fmlp::gelu_fg(acc[i][j][2] + bj[j][0], f2, g2);
          kvm_fmlp::gelu_fg(acc[i][j][3] + bj[j][1], f3, g3);
          *reinterpret_cast<float2*>(Ys + m * OP + n) = make_float2(f0, f1);
          *reinterpret_cast<float2*>(Gs + m * OP + n) = make_float2(g0, g1);
          *reinterpret_cast<float2*>(Ys + (m + 8) * OP + n) = make_float2(f2, f3);
          *reinterpret_cast<float2*>(Gs + (m + 8) * OP + n) = make_float2(g2, g3);
        }
  }
  __syncthreads();
  {
    float* const Y = run.y[l];
    float* const G = run.g[l];
    constexpr int C4 = NW / 4;   // a compile-time divisor: a run-time one cost a full division a piece (the 12:10Z census); columns past vc skip
    for (int i = threadIdx.x; i < FR * C4; i += NT) {
      const int r = i / C4, c = (i % C4) * 4;
      if (c >= vc) continue;
      *reinterpret_cast<float4*>(Y + (size_t)(r0 + r) * H + n0 + c) = *reinterpret_cast<const float4*>(Ys + r * OP + c);
      *reinterpret_cast<float4*>(G + (size_t)(r0 + r) * H + n0 + c) = *reinterpret_cast<const float4*>(Gs + r * OP + c);
    }
  }
  KVM_BATCH_CLOCK(CK_F, 2);
  __syncthreads();   // the ring is free for the next item
  KVM_BATCH_CLOCK(CK_F, 3);
}

// ---- HC: the head and the cross-entropy for rows [t0, t0 + 16): logits = Y_(L - 1) Wh (no bias), each row's loss, dlog = (softmax - onehot) / T.
template <bool GRID = false>   // GRID: the grid step's arithmetic (KvmBatchDesc::gHp > 0), an instantiation of its own (fwd_tile has why)
__device__ __noinline__ void head_ce(const KvmBatchDesc& d, const KvmBatchRun& run, int step, int t0, float* sm) {
  constexpr int NCB = VP / 16, BP = VP + 8, ASZ = 16 * AP, BSZ = BK * BP, LP = VP + 4;
  using WM = WarpMap<1, NCB>;
  need_smem(d, NST * (ASZ + BSZ) > 16 * LP ? NST * (ASZ + BSZ) : 16 * LP);
  KVM_BATCH_CLOCK(CK_HC, 0);
  const int K = d.H, V = d.V;
  const float* Yin = run.y[d.L - 1] + (size_t)t0 * K;
  const WM wm;
  float acc[WM::RB][2][4];
  zero(acc);
  constexpr int NC = kCopiers<NCB * WM::WPC>;
  Stream<16 * (BK / 4), NC> sa;
  Stream<BK * (VP / 4), NC> sb;
  sa.along<16>(Yin, 16, AP, [&](int r) { return r * K; });
  sb.down<VP>(run.wh, V, BP, V);
  const ShareEnds ends{d};   // the logits by the split step's virtual shares (kvm_fused_split.cuh: each share's partial, summed in share order)
  float pt_tot[WM::RB][2][4]; int pt_part = 0, pt_end = ends(0);
  if constexpr (GRID)   // the grid step's arithmetic: the logits over its gkah head blocks of gHp (kvm_fused_grid.cuh l. 218, 226)
    grid_k_loop<CK_HC, NC>(sm, ASZ + BSZ, ASZ, d.gtab + d.gtab_off[KVM_BATCH_GSUM_LOGITS], d.gHp / 16, d.gkah, K, sa, sb, wm.on, wm.nrb, pt_tot,
                           [&](float (&a)[WM::RB][2][4], const float* As, const float* Bs, int kk) {
                             mma_k8<false, false, WM::RB>(a, As, AP, Bs, BP, wm.rb0, wm.nrb, wm.cb, kk); }, NoChunk{});
  else {
    fold_part(pt_tot, acc, wm.nrb, 0, pt_part, pt_end, ends);   // the leading empty shares, if any
    k_loop<CK_HC, NC>(sm, ASZ + BSZ, ASZ, K, sa, sb, [&](const float* As, const float* Bs, int kv, int c) {
      if (wm.on)
        for (int kk = 0; kk < kv; kk += 8) {
          mma_k8<false, false, WM::RB>(acc, As, AP, Bs, BP, wm.rb0, wm.nrb, wm.cb, kk);
          fold_part(pt_tot, acc, wm.nrb, c * BK + kk + 8, pt_part, pt_end, ends);
        }
    });
  }
  KVM_BATCH_CLOCK(CK_HC, 1);
  __syncthreads();   // every warp is done with the ring: the logits go over it
  float* Ls = sm;
  if (wm.on) for_acc(pt_tot, wm.rb0, wm.nrb, wm.cb, [&](int m, int n, float v) { Ls[m * LP + n] = v; });
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int32_t* tg = run.tgt[step] + t0;
  for (int r = warp; r < 16; r += NT / 32) {
    const float* lr_ = Ls + r * LP;
    float mx = -INFINITY;
    for (int v = lane; v < V; v += 32) mx = fmaxf(mx, lr_[v]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float z = 0.f;
    for (int v = lane; v < V; v += 32) z += expf(lr_[v] - mx);
#pragma unroll
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    const int y = tg[r];
    float* dl = run.dlog + (size_t)(t0 + r) * VP;
    const float inv_t = 1.f / (float)T;   // divided by z, as the split step's cross-entropy divides
    for (int v = lane; v < VP; v += 32) dl[v] = v < V ? (expf(lr_[v] - mx) / z - (v == y ? 1.f : 0.f)) * inv_t : 0.f;
    if (lane == 0) run.rloss[t0 + r] = mx + logf(z) - lr_[y];
  }
  KVM_BATCH_CLOCK(CK_HC, 2);
  __syncthreads();
  KVM_BATCH_CLOCK(CK_HC, 3);
}

// ---- B: a weight's backward by its row block [k0, k0 + 16 MRBW): with dZo the output side's gradient (T x Nd, pitch ldz; Nd = H, or the head's
// V with dlog's pitch VP), W the weight (rows of Nd at pitch ldw), Yi and Gi the input side's gelu and gelu' (T x H):
//   dZi[:, k0 + j] = (dZo W[k0 + j, :]^T) * Gi[:, k0 + j]   (K = N, streamed in chunks of BK columns: by the split step's virtual shares, or for
//                                                            the head over VP in one part, its columns past V zero, as the split step runs it)
//   W[k0 + j, :] -= lr Yi[:, k0 + j]^T dZo                    (K = T by 16-row slices, a chunk's columns at a time, from the same staged W chunk)
//   bias: b[c] -= lr sum_t dZo[t, c] when bias (the run's first row block only; by slices, row_sum)
template <int MRBW, bool HEAD, bool GRID = false>   // GRID: the grid step's arithmetic (KvmBatchDesc::gHp > 0), instantiations of their own
__device__ __noinline__ void bwd_block(const KvmBatchDesc& d, const float* dZo, int ldz, float* W, int ldw, float* bias, const float* Yi,
                          const float* Gi, float* dZi, int k0, float* sm) {
  const int N = HEAD ? VP : d.H, Nd = HEAD ? d.V : d.H;   // the chain's K, and where the output side's data ends
  constexpr int KW = 16 * MRBW, ASZ = T * AP, WSZ = KW * AP, YP = T + 4;
  using WX = WarpMap<RBT, MRBW>;   // dX: T x KW
  using WD = WarpMap<MRBW, BK / 16>;   // dW: KW x BK
  need_smem(d, NST * (ASZ + WSZ) + KW * YP + T * (KW + 4));   // the ring, then Ys and Gb at the arena's end
  KVM_BATCH_CLOCK(CK_B, 0);
  const int H = d.H, vr = H - k0 < KW ? H - k0 : KW;
  constexpr int GP = KW + 4;
  constexpr int TAIL = KW * YP + T * GP;   // Ys and Gb, at the arena's end
  float* Ys = sm + d.smem_floats - TAIL;    // Yi[:, k0 + j]^T: [KW][T + 4], transposed once
  float* Gb = Ys + KW * YP;             // Gi[:, k0 + j]: [T][KW + 4], for the epilogue, its own cp.async group ahead of the ring's
  stage(Gb, GP, T, KW, T, vr, [&](int r) { return Gi + (size_t)r * H + k0; });
  cp_commit();
  for (int i = threadIdx.x; i < KW * T; i += NT) {
    const int t = i / KW, j = i - t * KW;
    Ys[j * YP + t] = j < vr ? Yi[(size_t)t * H + k0 + j] : 0.f;
  }
  const WX wx;
  const WD wd;
  float ax[WX::RB][2][4];
  zero(ax);
  Stream<T * (BK / 4), NT> sz;   // every warp computes here (dX and dW), so every thread copies
  Stream<KW * (BK / 4), NT> sw;
  sz.template along<T>(dZo, T, AP, [&](int r) { return r * ldz; });
  sw.template along<KW>(W, vr, AP, [&](int r) { return (k0 + r) * ldw; });
  const ColEnds xends{d, HEAD};
  float px_tot[WX::RB][2][4]; int px_part = 0, px_end = xends(0);
  // the chunk's other work, from its first column c0: the weight's update and the bias, which every chunk order takes alike
  auto rest = [&](const float* Zs, const float* Ws, int cv, int c0) {
    const int cd = Nd - c0;   // cd: the chunk's columns that hold data (the head's past V do not)
    if (wd.on && cd > 0) {  // dW[block, chunk] = Yi^T dZo[:, chunk] (K = T, by slices), then W -= lr dW from the staged chunk
      float aw[WD::RB][2][4], tw[WD::RB][2][4];
      row_chain(d, tw, aw, wd.nrb, [&](float (&a)[WD::RB][2][4], int kk) { mma_k8<false, false, WD::RB>(a, Ys, YP, Zs, AP, wd.rb0, wd.nrb, wd.cb, kk); });
      const float lr = d.lr;
      for_acc(tw, wd.rb0, wd.nrb, wd.cb, [&](int j, int n, float v) {
        if (j < vr && n < cv && n < cd) W[(size_t)(k0 + j) * ldw + c0 + n] = Ws[j * AP + n] - lr * v;
      });
    }
    if (bias && threadIdx.x < cv && threadIdx.x < cd)   // the run's first row block: this chunk's bias columns, each over the rows by slices
      bias[c0 + threadIdx.x] -= d.lr * row_sum(d, [&](int t) { return Zs[t * AP + threadIdx.x]; });
  };
  if constexpr (GRID) {   // the grid step's arithmetic: dY over the grid's gnb N blocks of gHp, or the head's one block (kvm_fused_grid.cuh l. 259, 279)
    static_assert(WX::RB <= KVM_BATCH_GRID_MAXRB, "the grid path keeps at most KVM_BATCH_GRID_MAXRB row blocks a warp (kvm_batch_layout)");
    grid_k_loop<CK_B, NT, WX::RB, HEAD ? GSUM_SINGLE : GSUM_ONE>(sm, ASZ + WSZ, ASZ, d.gtab + d.gtab_off[HEAD ? KVM_BATCH_GSUM_HEAD : KVM_BATCH_GSUM_DY], (HEAD ? kvm_batch_pad16(d.V) : d.gHp) / 16,
                          HEAD ? 1 : d.gnb, Nd, sz, sw, wx.on, wx.nrb, px_tot,
                          [&](float (&a)[WX::RB][2][4], const float* Zs, const float* Ws, int kk) {
                            mma_k8<false, true, WX::RB>(a, Zs, AP, Ws, AP, wx.rb0, wx.nrb, wx.cb, kk); }, rest);
  } else {
    fold_part(px_tot, ax, wx.nrb, 0, px_part, px_end, xends);   // the leading empty shares, if any
    k_loop<CK_B, NT, HEAD>(sm, ASZ + WSZ, ASZ, N, sz, sw, [&](const float* Zs, const float* Ws, int cv, int c) {
      const int c0 = c * BK;
      if (wx.on)   // dX += dZo[:, chunk] W[block, chunk]^T: B[k = n][j] = W[j][n], stored [j][n], so TB
        for (int kk = 0; kk < cv; kk += 8) {
          mma_k8<false, true, WX::RB>(ax, Zs, AP, Ws, AP, wx.rb0, wx.nrb, wx.cb, kk);
          fold_part(px_tot, ax, wx.nrb, c0 + kk + 8, px_part, px_end, xends);
        }
      rest(Zs, Ws, cv, c0);
    }, Nd);
  }
  KVM_BATCH_CLOCK(CK_B, 1);
  __syncthreads();   // the ring is free and Gb has landed everywhere: dZi's block goes through a tile over the ring, then whole 16-byte pieces
  float* const Ds = sm;
  if (wx.on) for_acc(px_tot, wx.rb0, wx.nrb, wx.cb, [&](int t, int j, float v) { Ds[t * GP + j] = v * Gb[t * GP + j]; });
  __syncthreads();
  {
    constexpr int C4 = KW / 4;   // a compile-time divisor; columns past vr skip
    for (int i = threadIdx.x; i < T * C4; i += NT) {
      const int t = i / C4, c = (i % C4) * 4;
      if (c >= vr) continue;
      *reinterpret_cast<float4*>(dZi + (size_t)t * H + k0 + c) = *reinterpret_cast<const float4*>(Ds + t * GP + c);
    }
  }
  KVM_BATCH_CLOCK(CK_B, 2);
  __syncthreads();
  KVM_BATCH_CLOCK(CK_B, 3);
}

// ---- BX: dX0[t0, t0 + 16) = dZ0[rows] W0^T (K = H): B[k = j][e] = W0[e][j], stored [e][j], so TB.
template <bool GRID = false>   // GRID: the grid step's arithmetic (KvmBatchDesc::gHp > 0), an instantiation of its own (fwd_tile has why)
__device__ __noinline__ void dx0_rows(const KvmBatchDesc& d, const KvmBatchRun& run, int t0, float* sm) {
  constexpr int NCB = E48 / 16, ASZ = 16 * AP, BSZ = E48 * AP;
  using WM = WarpMap<1, NCB>;
  need_smem(d, NST * (ASZ + BSZ));
  const int K = d.H;
  const float* dZ = run.dz[0] + (size_t)t0 * K;   // layer 0's dZ: dz[0 % 3]
  const float* W0 = run.w[0];
  const WM wm;
  float acc[WM::RB][2][4];
  zero(acc);
  constexpr int NC = kCopiers<NCB * WM::WPC>;
  Stream<16 * (BK / 4), NC> sa;
  Stream<E48 * (BK / 4), NC> sb;
  sa.along<16>(dZ, 16, AP, [&](int r) { return r * K; });
  sb.along<E48>(W0, E48, AP, [&](int r) { return r * K; });
  const ShareEnds ends{d};   // dX0 by the split step's virtual shares (kvm_fused_split.cuh: each share's part, summed in share order)
  float pt_tot[WM::RB][2][4]; int pt_part = 0, pt_end = ends(0);
  if constexpr (GRID)   // the grid step's arithmetic: dX0 over its gnb0 column blocks of layer 0 (kvm_fused_grid.cuh l. 298, 315)
    grid_k_loop<CK_BX, NC>(sm, ASZ + BSZ, ASZ, d.gtab + d.gtab_off[KVM_BATCH_GSUM_DX0], d.gHp / 16, d.gnb0, K, sa, sb, wm.on, wm.nrb, pt_tot,
                           [&](float (&a)[WM::RB][2][4], const float* As, const float* Bs, int kk) {
                             mma_k8<false, true, WM::RB>(a, As, AP, Bs, AP, wm.rb0, wm.nrb, wm.cb, kk); }, NoChunk{});
  else {
    fold_part(pt_tot, acc, wm.nrb, 0, pt_part, pt_end, ends);   // the leading empty shares, if any
    k_loop<CK_BX, NC>(sm, ASZ + BSZ, ASZ, K, sa, sb, [&](const float* As, const float* Bs, int kv, int c) {
      if (wm.on)
        for (int kk = 0; kk < kv; kk += 8) {
          mma_k8<false, true, WM::RB>(acc, As, AP, Bs, AP, wm.rb0, wm.nrb, wm.cb, kk);
          fold_part(pt_tot, acc, wm.nrb, c * BK + kk + 8, pt_part, pt_end, ends);
        }
    });
  }
  if (wm.on) for_acc(pt_tot, wm.rb0, wm.nrb, wm.cb, [&](int m, int n, float v) { run.dx0[(size_t)(t0 + m) * E48 + n] = v; });
  __syncthreads();
}

// ---- BU, a column tile of W0: W0[:, n0, n0 + NW) -= lr X0^T dZ0[:, tile] (K = T), and b0 over the tile's columns.
template <int NCB>
__device__ __noinline__ void w0_tile(const KvmBatchDesc& d, const KvmBatchRun& run, int n0, float* sm) {
  constexpr int NW = 16 * NCB, XP = T + 4, ZP = NW + 8;
  using WM = WarpMap<E48 / 16, NCB>;
  need_smem(d, E48 * XP + T * ZP);
  const int H = d.H, vc = H - n0 < NW ? H - n0 : NW;
  float* Xs = sm;              // X0^T: [E][T + 4]
  float* Zs = sm + E48 * XP;   // dZ0[:, tile]: [T][NW + 8]
  for (int i = threadIdx.x; i < T * E48; i += NT) {
    const int t = i / E48, e = i - t * E48;
    Xs[e * XP + t] = run.x0[(size_t)t * E48 + e];
  }
  stage(Zs, ZP, T, NW, T, vc, [&](int r) { return run.dz[0] + (size_t)r * H + n0; });
  cp_commit();   // cp.async.wait_group waits only for committed groups: an uncommitted copy would still be in flight past the barrier
  cp_wait<0>();
  __syncthreads();
  const WM wm;
  float acc[WM::RB][2][4];
  zero(acc);
  if (wm.on) {
    float pt_tot[WM::RB][2][4];   // K = T by the split step's 16-row slices, summed in slice order (row_chain)
    row_chain(d, pt_tot, acc, wm.nrb, [&](float (&a)[WM::RB][2][4], int kk) { mma_k8<false, false, WM::RB>(a, Xs, XP, Zs, ZP, wm.rb0, wm.nrb, wm.cb, kk); });
    float* W0 = run.w[0];
    const float lr = d.lr;
    for_acc(pt_tot, wm.rb0, wm.nrb, wm.cb, [&](int e, int n, float v) {
      if (n < vc) W0[(size_t)e * H + n0 + n] -= lr * v;
    });
  }
  if (threadIdx.x < vc) run.b[0][n0 + threadIdx.x] -= d.lr * row_sum(d, [&](int t) { return Zs[t * ZP + threadIdx.x]; });
  __syncthreads();
}

// ---- BU, a run's last item: its embedding rows (each row's gradient summed over its tokens in token order, then applied once, as the
// float64 reference does) and the step's loss (the row losses summed in row order, over T).
__device__ __noinline__ void emb_loss(const KvmBatchDesc& d, const KvmBatchRun& run, int step, float* sm) {
  const int V = d.V;
  need_smem(d, V * E48 + T * E48 + V + T);
  float* gs = sm;                                        // [V][E]: each vocabulary row's gradient
  float* dxs = sm + V * E48;                             // [T][E]: the run's dX0 rows
  int* used = reinterpret_cast<int*>(dxs + T * E48);     // [V]
  int* ids = used + V;                                   // [T]
  stage(dxs, E48, T, E48, T, E48, [&](int r) { return run.dx0 + (size_t)r * E48; });
  cp_commit();
  for (int i = threadIdx.x; i < V * E48; i += NT) gs[i] = 0.f;
  for (int i = threadIdx.x; i < V; i += NT) used[i] = 0;
  for (int i = threadIdx.x; i < T; i += NT) ids[i] = run.ids[step][i];
  cp_wait<0>();
  __syncthreads();
  if (threadIdx.x < E48) {   // one thread a column, the tokens in order: a repeated token's rows add in token order, as the reference sums them
    const int e = threadIdx.x;
    for (int t = 0; t < T; ++t) gs[ids[t] * E48 + e] += dxs[t * E48 + e];
  } else if (threadIdx.x == E48) {
    for (int t = 0; t < T; ++t) used[ids[t]] = 1;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < V * E48; i += NT) {
    const int v = i / E48;
    if (used[v]) run.emb[i] -= d.lr * gs[i];
  }
  if (threadIdx.x == 0) {
    if (d.Sv > 0) {   // as the split step: each slice's row losses from 0.f in row order, the slice sums from 0.f in slice order, times 1 / T
      float s = 0.f;
      for (int q = 0; q < T / KVM_BATCH_SLICE; ++q) {
        float a = 0.f;
        for (int t = q * KVM_BATCH_SLICE; t < (q + 1) * KVM_BATCH_SLICE; ++t) a += run.rloss[t];
        s += a;
      }
      run.loss[step][0] = s * (1.f / (float)T);
    } else {
      float s = 0.f;
      for (int t = 0; t < T; ++t) s += run.rloss[t];
      run.loss[step][0] = s / (float)T;
    }
    run.loss[step][1] = 1.f / (float)T;   // the loss buffer's second float, as the split step and softmax_ce_reduce write it
  }
  __syncthreads();
}

// ---- The split backward (d.bsplit): a weight's dX tiles in one phase and its update tiles in the next, each output summed in the order
// bwd_block, dx0_rows and w0_tile sum it, so both modes give one arithmetic.
struct WeightView {   // weight w of a run, as the split backward reads it (kvm_batch_wrows / wcols)
  const float* dzo; int ldz;     // the output side's gradient: dlog (head) or dZ_w, its pitch
  float* w; int ldw, rows, cols; // the weight [rows][cols]
  const float* yi; int ldy;      // the input side's activations: Y_(w-1), or X0 for layer 0
  const float* gi;               // gelu'(Z_(w-1)) (null for layer 0: dX0 has no activation)
  float* dzi; int ldi;           // where dX through w goes: dZ_(w-1), or dX0
  float* bias;                   // the weight's bias (null for the head)
  bool head;                     // the head: dX through it runs over VP in one part, its columns past V zero
};
__device__ __forceinline__ WeightView weight_view(const KvmBatchDesc& d, const KvmBatchRun& run, int w) {
  WeightView v;
  const int L = d.L, H = d.H;
  v.head = w == L;
  if (w == L) { v.dzo = run.dlog; v.ldz = VP; v.w = run.wh; v.ldw = d.V; v.rows = H; v.cols = d.V; v.bias = nullptr; }
  else { v.dzo = run.dz[w % 3]; v.ldz = H; v.w = run.w[w]; v.ldw = H; v.rows = w ? H : d.E; v.cols = H; v.bias = run.b[w]; }
  if (w) { v.yi = run.y[w - 1]; v.ldy = H; v.gi = run.g[w - 1]; v.dzi = run.dz[(w - 1) % 3]; v.ldi = H; }
  else { v.yi = run.x0; v.ldy = E48; v.gi = nullptr; v.dzi = run.dx0; v.ldi = E48; }
  return v;
}

// dX tile through weight w: rows [r0, r0 + 16 FM) and the weight's rows [k0, k0 + 16 NCB): dzi = (dzo W^T) (* gi), K = cols in BK chunks, by the
// split step's virtual shares (the head: over VP in one part, its columns past V zero).
template <int FM, int NCB>
__device__ __noinline__ void dx_tile(const KvmBatchDesc& d, const WeightView& v, int r0, int k0, float* sm) {
  constexpr int NW = 16 * NCB, FR = 16 * FM, ASZ = FR * AP, BSZ = NW * AP, OP = NW + 4;
  using WM = WarpMap<FM, NCB>;
  need_smem(d, NST * (ASZ + BSZ) + FR * OP);   // the ring, then gi's tile at the arena's end
  if (d.gHp > 0) __trap();   // the grid path runs the fused backward only (kvm_batch_layout sets bsplit = 0 for it)
  KVM_BATCH_CLOCK(CK_B, 0);
  const int N = v.head ? VP : v.cols, vr = v.rows - k0 < NW ? v.rows - k0 : NW;
  float* const Gt = sm + d.smem_floats - FR * OP;   // gi's tile [FR][NW + 4] at the arena's end, its own cp.async group ahead of the ring's
  if (v.gi) { stage(Gt, OP, FR, NW, FR, vr, [&](int r) { return v.gi + (size_t)(r0 + r) * v.ldi + k0; }); }
  cp_commit();
  const WM wm;
  float acc[WM::RB][2][4];
  zero(acc);
  constexpr int NC = kCopiers<NCB * WM::WPC>;
  Stream<FR * (BK / 4), NC> sa;
  Stream<NW * (BK / 4), NC> sb;
  sa.template along<FR>(v.dzo, FR, AP, [&](int r) { return (r0 + r) * v.ldz; });
  sb.template along<NW>(v.w, vr, AP, [&](int r) { return (k0 + r) * v.ldw; });
  const ColEnds ends{d, v.head};
  float pt_tot[WM::RB][2][4]; int pt_part = 0, pt_end = ends(0);
  fold_part(pt_tot, acc, wm.nrb, 0, pt_part, pt_end, ends);   // the leading empty shares, if any
  k_loop<CK_B, NC, true>(sm, ASZ + BSZ, ASZ, N, sa, sb, [&](const float* As, const float* Bs, int cv, int c) {
    if (wm.on)
      for (int kk = 0; kk < cv; kk += 8) {
        mma_k8<false, true, WM::RB>(acc, As, AP, Bs, AP, wm.rb0, wm.nrb, wm.cb, kk);
        fold_part(pt_tot, acc, wm.nrb, c * BK + kk + 8, pt_part, pt_end, ends);
      }
  }, v.cols);
  KVM_BATCH_CLOCK(CK_B, 1);
  __syncthreads();   // the ring is free and gi's tile has landed: the tile goes through shared memory, then whole 16-byte pieces
  float* const Ds = sm;
  if (wm.on) for_acc(pt_tot, wm.rb0, wm.nrb, wm.cb, [&](int t, int j, float x) { Ds[t * OP + j] = v.gi ? x * Gt[t * OP + j] : x; });
  __syncthreads();
  {
    constexpr int C4 = NW / 4;   // a compile-time divisor; columns past vr skip
    for (int i = threadIdx.x; i < FR * C4; i += NT) {
      const int t = i / C4, c = (i % C4) * 4;
      if (c >= vr) continue;
      *reinterpret_cast<float4*>(v.dzi + (size_t)(r0 + t) * v.ldi + k0 + c) = *reinterpret_cast<const float4*>(Ds + t * OP + c);
    }
  }
  KVM_BATCH_CLOCK(CK_B, 2);
  __syncthreads();
  KVM_BATCH_CLOCK(CK_B, 3);
}

// Update tile of weight w: its rows [k0, k0 + 16 FM) and columns [n0, n0 + 16 NCB): W -= lr yi^T dzo (K = T by 16-row slices), and the bias over
// those columns when k0 == 0 (by slices, row_sum).
template <int FM, int NCB>
__device__ __noinline__ void dw_tile(const KvmBatchDesc& d, const WeightView& v, int k0, int n0, float* sm) {
  constexpr int NW = 16 * NCB, KR = 16 * FM, XP = T + 4, ZP = NW + 8;
  using WM = WarpMap<FM, NCB>;
  need_smem(d, KR * XP + T * ZP);
  KVM_BATCH_CLOCK(CK_W0, 0);
  const int vr = v.rows - k0 < KR ? v.rows - k0 : KR, vc = v.cols - n0 < NW ? v.cols - n0 : NW;
  float* Xs = sm;              // yi[:, k0 + j]^T: [KR][T + 4]
  float* Zs = sm + KR * XP;    // dzo[:, n0 + n]: [T][NW + 8]
  stage(Zs, ZP, T, NW, T, vc, [&](int r) { return v.dzo + (size_t)r * v.ldz + n0; });
  cp_commit();
  for (int i = threadIdx.x; i < T * KR; i += NT) {
    const int t = i / KR, j = i - t * KR;
    Xs[j * XP + t] = j < vr ? v.yi[(size_t)t * v.ldy + k0 + j] : 0.f;
  }
  cp_wait<0>();
  __syncthreads();
  KVM_BATCH_CLOCK(CK_W0, 1);
  const WM wm;
  float acc[WM::RB][2][4];
  zero(acc);
  if (wm.on) {
    float pt_tot[WM::RB][2][4];   // K = T by the split step's 16-row slices (row_chain)
    row_chain(d, pt_tot, acc, wm.nrb, [&](float (&a)[WM::RB][2][4], int kk) { mma_k8<false, false, WM::RB>(a, Xs, XP, Zs, ZP, wm.rb0, wm.nrb, wm.cb, kk); });
    const float lr = d.lr;
    for_acc(pt_tot, wm.rb0, wm.nrb, wm.cb, [&](int j, int n, float x) {
      if (j < vr && n < vc) { float* wp = v.w + (size_t)(k0 + j) * v.ldw + n0 + n; *wp = *wp - lr * x; }
    });
  }
  if (v.bias && k0 == 0 && threadIdx.x < vc) v.bias[n0 + threadIdx.x] -= d.lr * row_sum(d, [&](int t) { return Zs[t * ZP + threadIdx.x]; });
  KVM_BATCH_CLOCK(CK_W0, 2);
  __syncthreads();
  KVM_BATCH_CLOCK(CK_W0, 3);
}

template <int FM>
__device__ __forceinline__ void split_by_width(const KvmBatchDesc& d, const WeightView& v, bool dx, int a, int b, int width, float* sm) {
  switch (width) {
    case 16: if (dx) dx_tile<FM, 1>(d, v, a, b, sm); else dw_tile<FM, 1>(d, v, a, b, sm); break;
    case 32: if (dx) dx_tile<FM, 2>(d, v, a, b, sm); else dw_tile<FM, 2>(d, v, a, b, sm); break;
    case 64: if (dx) dx_tile<FM, 4>(d, v, a, b, sm); else dw_tile<FM, 4>(d, v, a, b, sm); break;
    default: if (dx) dx_tile<FM, 8>(d, v, a, b, sm); else dw_tile<FM, 8>(d, v, a, b, sm); break;
  }
}
// One split backward item: phase q = p - (L + 1) runs, a run at a time, the dX tiles through weight L - q, then the update tiles of weight
// L - q + 1, then (q = L + 1) the embedding item. `it` counts through them in that order (kvm_batch_items).
__device__ void split_item(const KvmBatchDesc& d, int q, int it, int step, float* sm) {
  const int L = d.L;
  const int nx = q <= L ? kvm_batch_dx_items(d, L - q) : 0, nwi = q >= 1 ? kvm_batch_dw_items(d, L - q + 1) : 0;
  const int per = nx + nwi + (q == L + 1 ? 1 : 0), r = it / per, i = it % per;
  const KvmBatchRun& run = d.runs[r];
  if (i < nx) {   // dX tile: (row block, block of the weight's rows)
    const WeightView v = weight_view(d, run, L - q);
    const int kb = (v.rows + d.xnw - 1) / d.xnw, r0 = (i / kb) * 16 * d.xfm, k0 = (i % kb) * d.xnw;
    switch (d.xfm) {
      case 1: split_by_width<1>(d, v, true, r0, k0, d.xnw, sm); break;
      case 2: split_by_width<2>(d, v, true, r0, k0, d.xnw, sm); break;
      case 3: split_by_width<3>(d, v, true, r0, k0, d.xnw, sm); break;
      default: split_by_width<6>(d, v, true, r0, k0, d.xnw, sm); break;
    }
  } else if (i < nx + nwi) {   // update tile: (block of the weight's rows, block of its columns)
    const WeightView v = weight_view(d, run, L - q + 1);
    const int j = i - nx, nb = (v.cols + d.wnw - 1) / d.wnw, k0 = (j / nb) * 16 * d.wfm, n0 = (j % nb) * d.wnw;
    switch (d.wfm) {
      case 1: split_by_width<1>(d, v, false, k0, n0, d.wnw, sm); break;
      case 2: split_by_width<2>(d, v, false, k0, n0, d.wnw, sm); break;
      case 3: split_by_width<3>(d, v, false, k0, n0, d.wnw, sm); break;
      default: split_by_width<6>(d, v, false, k0, n0, d.wnw, sm); break;
    }
  } else emb_loss(d, run, step, sm);
}

template <int FM>
__device__ __forceinline__ void fwd_by_width(const KvmBatchDesc& d, const KvmBatchRun& run, int l, int step, int r0, int n0, float* sm) {
  switch (d.nw) {
    case 16: fwd_tile<FM, 1>(d, run, l, step, r0, n0, sm); break;
    case 32: fwd_tile<FM, 2>(d, run, l, step, r0, n0, sm); break;
    case 64: fwd_tile<FM, 4>(d, run, l, step, r0, n0, sm); break;
    default: fwd_tile<FM, 8>(d, run, l, step, r0, n0, sm); break;
  }
}
// A hidden layer's forward tile with the grid step's arithmetic: only the tiles of at most KVM_BATCH_GRID_MAXRB row blocks a warp exist
// (WarpMap's RB; kvm_batch_layout gives no other).
template <int FM>
__device__ __forceinline__ void fwd_grid_by_width(const KvmBatchDesc& d, const KvmBatchRun& run, int l, int step, int r0, int n0, float* sm) {
  switch (d.nw) {
    case 16: if constexpr (WarpMap<FM, 1>::RB <= KVM_BATCH_GRID_MAXRB) { fwd_tile<FM, 1, true>(d, run, l, step, r0, n0, sm); return; } break;
    case 32: if constexpr (WarpMap<FM, 2>::RB <= KVM_BATCH_GRID_MAXRB) { fwd_tile<FM, 2, true>(d, run, l, step, r0, n0, sm); return; } break;
    case 64: if constexpr (WarpMap<FM, 4>::RB <= KVM_BATCH_GRID_MAXRB) { fwd_tile<FM, 4, true>(d, run, l, step, r0, n0, sm); return; } break;
    default: if constexpr (WarpMap<FM, 8>::RB <= KVM_BATCH_GRID_MAXRB) { fwd_tile<FM, 8, true>(d, run, l, step, r0, n0, sm); return; } break;
  }
  __trap();   // kvm_batch_layout gives the grid path RB <= KVM_BATCH_GRID_MAXRB
}

// One item of phase p (kvm_batch_items has the counts and the order). GRID: the grid step's arithmetic (KvmBatchDesc::gHp > 0), a call tree of
// its own: ptxas allocates a kernel's registers over its whole call tree, so with the grid path in the same tree, functions it never enters
// spilled in their loops (bwd_block<1, true>'s share total, reloaded and stored around every k8 step's fold, after the grid's walk changed).
// The split and own paths run on GRID = false and never see the grid functions; a kernel takes one (kvm_static.cuh, batch_bed).
template <bool GRID>
__device__ void run_item(const KvmBatchDesc& d, int p, int it, int step, float* sm) {
  const int L = d.L;
  if (p < L) {   // F_p: (run, row block, column tile)
    const int per = d.fr * d.nt, rem = it % per;
    const KvmBatchRun& run = d.runs[it / per];
    const int r0 = (rem / d.nt) * 16 * d.fm, n0 = (rem % d.nt) * d.nw;
    if constexpr (GRID) {
      if (p > 0) {   // a hidden layer with the grid step's arithmetic (layer 0 takes its K = E whole, as the grid step does)
        switch (d.fm) {
          case 1: fwd_grid_by_width<1>(d, run, p, step, r0, n0, sm); break;
          case 2: fwd_grid_by_width<2>(d, run, p, step, r0, n0, sm); break;
          case 3: fwd_grid_by_width<3>(d, run, p, step, r0, n0, sm); break;
          default: fwd_grid_by_width<6>(d, run, p, step, r0, n0, sm); break;
        }
        return;
      }
    }
    switch (d.fm) {
      case 1: fwd_by_width<1>(d, run, p, step, r0, n0, sm); break;
      case 2: fwd_by_width<2>(d, run, p, step, r0, n0, sm); break;
      case 3: fwd_by_width<3>(d, run, p, step, r0, n0, sm); break;
      default: fwd_by_width<6>(d, run, p, step, r0, n0, sm); break;
    }
  } else if (p == L) {   // HC
    head_ce<GRID>(d, d.runs[it / RBT], step, (it % RBT) * 16, sm);
  } else if (!GRID && d.bsplit) {   // the split backward (the grid path takes the fused one: kvm_batch_layout)
    split_item(d, p - (L + 1), it, step, sm);
  } else if (p <= 2 * L) {   // BH (p = L + 1), then B_l for l = 2L + 1 - p
    const KvmBatchRun& run = d.runs[it / d.kt];
    const int k0 = (it % d.kt) * d.kw;
    const bool head = p == L + 1;
    const int l = head ? L : 2 * L + 1 - p;   // the weight's layer (L: the head); its input side is layer l - 1
    const float* dZo = head ? run.dlog : run.dz[l % 3];
    const int ldz = head ? VP : d.H, ldw = head ? d.V : d.H;
    float* W = head ? run.wh : run.w[l];
    float* bias = (!head && k0 == 0) ? run.b[l] : nullptr;
    float* dZi = run.dz[(l - 1) % 3];
    if constexpr (GRID) {   // the grid step's arithmetic: kvm_batch_layout gives it the fused backward, kw 16, 32 or 64 (RB 1, 2 or 3)
      if (d.bsplit) __trap();
      switch (d.kw) {
        case 16: if (head) bwd_block<1, true, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm);
                 else bwd_block<1, false, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm); break;
        case 32: if (head) bwd_block<2, true, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm);
                 else bwd_block<2, false, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm); break;
        default: if (head) bwd_block<4, true, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm);
                 else bwd_block<4, false, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm); break;
      }
    } else switch (d.kw) {
      case 16: if (head) bwd_block<1, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm);
               else bwd_block<1, false>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm); break;
      case 32: if (head) bwd_block<2, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm);
               else bwd_block<2, false>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm); break;
      default: if (head) bwd_block<4, true>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm);
               else bwd_block<4, false>(d, dZo, ldz, W, ldw, bias, run.y[l - 1], run.g[l - 1], dZi, k0, sm); break;
    }
  } else if (p == 2 * L + 1) {   // BX
    dx0_rows<GRID>(d, d.runs[it / RBT], (it % RBT) * 16, sm);
  } else {   // BU
    const int ntile = d.R * d.nt;
    if (it < ntile) {
      const KvmBatchRun& run = d.runs[it / d.nt];
      const int n0 = (it % d.nt) * d.nw;
      switch (d.nw) {
        case 16: w0_tile<1>(d, run, n0, sm); break;
        case 32: w0_tile<2>(d, run, n0, sm); break;
        case 64: w0_tile<4>(d, run, n0, sm); break;
        default: w0_tile<8>(d, run, n0, sm); break;
      }
    } else emb_loss(d, d.runs[it - ntile], step, sm);
  }
}

// One step of every run: member `member` of d.members runs its items of each phase, then crosses with the gang. `at` is the member's place
// in the gang's crossings (kvm_gang::cross_start before its first step), `abort` the run's error word (null in a bed): a member that fails
// and never arrives ends the gang's step at the next crossing instead of hanging it, and the step returns false, alike in every thread of
// the block; the caller then returns without another crossing (kvm_gang.cuh's contract). ph(p) is called after phase p's crossing.
// GRID must be the descriptor's arithmetic (gHp > 0): the caller's kernel takes one call tree (run_item has why); a mismatch traps.
template <bool GRID, class PH>
__device__ bool batch_step(int member, const KvmBatchDesc& d, int step, float* sm, unsigned long long& at, const int* abort, PH ph) {
  if (GRID != (d.gHp > 0)) __trap();
  const int np = kvm_batch_phases(d.L);
  for (int p = 0; p < np; ++p) {
    const int n = kvm_batch_items(d, p);
    for (int it = member; it < n; it += d.members) run_item<GRID>(d, p, it, step, sm);
    if (!kvm_gang::cross(d.ctl, member, d.members, at, abort)) return false;
    ph(p);
  }
  return true;
}

}  // namespace kvm_fbatch
