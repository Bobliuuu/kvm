#pragma once
// F's device program: every step of a GPT training run as one member of a gang (kvm_f2d.h has the form, its descriptors, tilings,
// layouts, fit and builder; bench/fused_gpt/f2d_bed.cu runs it alone against the workload VM). In the order a step uses them:
//  * the in-order sums of the partials, the staging and the GEMM forms: streamed (cp.async stages, register accumulators), resident tiles
//    times streamed blocks (sgemm_rs, sgemm_sr), and the fused step's one-shot gemm for the attention items and the head;
//  * the attention half (attn_fwd, attn_bwd, attn_sums) and the MLP half (mlp_fwd, mlp_bwd, mlp_sums), each on its own layout;
//  * the head on vocabulary shards and the embedding's gradient, then the program (run).
// A phase that reads another member's output starts after a crossing (Cross). Every crossing is checked: a member that fails returns
// false up the call chain at once, and the run's error word makes every other member leave at its next crossing.
#include <cmath>
#include <cstdint>
#include "kvm_f2d.h"
#include "kvm_fused_gpt.cuh"
#include "kvm_gang.cuh"

namespace kvm_f2d {
using namespace kvm_fgpt;   // the fused step's helpers, on F's block through their thread-count parameter (FNT)

static_assert(KVM_GANG_LINES_MAX <= 32 * FNW, "the flag lines poll each line of 32 members with a warp of the block");
static_assert(ctl_words(1) == kvm_gang::line_words(1) && ctl_words(33) == kvm_gang::line_words(33) && ctl_words(108) == kvm_gang::line_words(108),
              "kvm_f2d.h's control-word count is kvm_gang's");

// ---- the gang's crossing --------------------------------------------------------------------------------------------------------------
// The crossing in the form the gang's size takes (kvm_gang::counted_form): the flag lines up to KVM_GANG_LINES_MAX members, the counted
// barrier past them. The program is instantiated per form (run<Counted>, the launch picking by G), as the fused step's is, so each
// crossing site carries one form's code. A member starts from the control words as the last run left them (they need no reset between
// runs) and holds only its place in the crossings; the words, the gang and the error word come from the descriptor at each crossing.
template <bool Counted> struct Cross;
template <> struct Cross<false> {
  unsigned epoch;
  __device__ Cross(const unsigned* ctl, int member, int) : epoch(kvm_gang::lines_epoch(ctl, member)) {}
  __device__ __forceinline__ bool operator()(unsigned* ctl, int G, const int* abort) { return kvm_gang::barrier_lines(ctl, blockIdx.x, G, epoch, abort); }
};
template <> struct Cross<true> {
  unsigned long long target;
  __device__ Cross(const unsigned* ctl, int, int G) : target(kvm_gang::counted_base(ctl, G)) {}
  __device__ __forceinline__ bool operator()(unsigned* ctl, int G, const int* abort) { return kvm_gang::barrier_counted(ctl, G, target, abort); }
};

// The sum over k < S of in[k stride] (four floats), in k order from zero, each element by one thread (the order every member's sum keeps).
__device__ __forceinline__ float4 sum_in_order4(const float* in, size_t stride, int S) {
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = 0; k < S; ++k) { const float4 v = ld4(in + (size_t)k * stride); a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
  return a;
}
// out[i] = sum over s < S of in[s stride + i] (in s order, each element its own sum), for i < n4 4, four elements a thread: the four
// arrays' sums of the step are one flat loop each, no branch an element.
__device__ __forceinline__ void sum_parts4(const float* in, size_t stride, int S, float* out, long long n4, long long i0, long long step) {
  for (long long i = i0; i < n4; i += step) reinterpret_cast<float4*>(out)[i] = sum_in_order4(in + 4 * i, stride, S);
}

// Every copy issued (16 B where the copy allows), no wait: the caller overlaps them with work and then waits.
template <int NC>
__device__ void stage_issue(const Copy (&cs)[NC]) {
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    const Copy& q = cs[i];
    if (q.vec())
      grid_loop<FNT>(q.prows, q.pcols >> 2, [&](int r, int c4) {
        const int c = c4 << 2;
        const bool in = r < q.rows && c < q.cols;
        cp16(q.dst + r * q.dld + c, in ? q.src + (size_t)r * q.sld + c : q.src, in);
      });
    else
      grid_loop<FNT>(q.prows, q.pcols, [&](int r, int c) {
        const bool in = r < q.rows && c < q.cols;
        cp4(q.dst + r * q.dld + c, in ? q.src + (size_t)r * q.sld + c : q.src, in);
      });
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void stage_wait() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); __syncthreads(); }

// One operand's chunk copies, decided once a GEMM: each of this thread's 16 B units (at most MAXU) keeps its shared-memory offset, its
// global offset from the chunk's base, its K index inside the chunk (a unit at or past the chunk's kc reads nothing) and whether its other
// index is inside the matrix. A chunk then issues its units with one add each: no division, no copy list.
template <int MAXU>
struct Units {
  int dst[MAXU], src[MAXU], kk[MAXU];
  bool use[MAXU], rin[MAXU];
  // rows x (cols / 4) units of a chunk laid out row-major in shared memory (row stride ld) and in global memory (row stride gld); the K
  // index is the row (krow) or the column; `lim` bounds the other index (the M or N extent)
  __device__ void plan(int rows, int cols, int ld, int gld, bool krow, int lim) {
    const int per = cols >> 2, total = rows * per;
    if (total > MAXU * FNT) __trap();                                       // a unit this thread cannot hold would be silently skipped
#pragma unroll
    for (int i = 0; i < MAXU; ++i) {                                       // unit i of this thread: threadIdx.x + i FNT (compile-time i: registers, not a stack array)
      const int u = threadIdx.x + i * FNT, r = u / per, c = (u % per) << 2;
      use[i] = u < total; dst[i] = r * ld + c; src[i] = r * gld + c; kk[i] = krow ? r : c; rin[i] = (krow ? c : r) < lim;
    }
  }
  __device__ __forceinline__ void issue(float* sm, const float* base, int kc) const {
#pragma unroll
    for (int i = 0; i < MAXU; ++i)
      if (use[i]) { const bool in = rin[i] && kk[i] < kc; cp16(sm + dst[i], in ? base + src[i] : base, in); }
  }
};

// The streamed GEMM's units a thread for a chunk of KC: its B chunk is KC x Fs, 3 units a thread at KC 32 and 4.5 at KC 48 when Fs is 96
// (d384 at Gc 16), so KC 48 takes five (four trapped there). Five unpacked units cost F's
// kernel nothing on sm_80 (255 registers, 48 B of stack, 4 B spilled, as with four); the same five packed into two words a unit spilled
// 452-488 B. KC 64 (six at Fs 96) is not built: five trap there, as four did at KC 48.
template <int KC> __host__ __device__ constexpr int stream_units() { return KC <= 32 ? 4 : 5; }

// The tensor core's TF32 operand from an fp32 bit pattern: rounded to nearest (cvt.rna: an FSETP and a predicated IADD on sm_80); in the
// precise build the raw bits, which mma_op splits.
__device__ __forceinline__ unsigned tfc(unsigned u) {
#if KVM_F2D_PRECISE
  return u;
#else
  return tf32_rna(__uint_as_float(u));
#endif
}

// A streamed operand's TF32 operand: its producer stored it rounded (every operand rounded where it is born), so its bits pass through;
// in the precise build tfc's raw bits.
__device__ __forceinline__ unsigned tfs(unsigned u) {
#if !KVM_F2D_PRECISE
  return u;
#else
  return tfc(u);
#endif
}

// The resident operands (At, the GELU'd tile, and dU_t) are GEMM operands only, except dU_t's db1 sum. In the 1xTF32 build they are stored
// TF32-rounded where they are produced (tfr: rounded, then the 13 dropped bits cleared, so the stored float is exactly the TF32 value and
// a second rounding would change nothing), the GEMMs read them with no rounding at use (tfres), and db1's partial sums dU from global
// memory, unrounded. Bitwise to rounding at every use: the tensor core reads the same 19 bits either way.
__device__ __forceinline__ float tfr(float x) {
#if !KVM_F2D_PRECISE
  return __uint_as_float(tf32_rna(x) & 0xffffe000u);
#else
  return x;
#endif
}
__device__ __forceinline__ unsigned tfres(unsigned u) {
#if !KVM_F2D_PRECISE
  return u;
#else
  return tfc(u);
#endif
}

// The streamed GEMMs' product. In the precise build every GEMM of F's runs in 3xTF32, the fused step's precise mode (matmul_tile's 3xTF32
// flag, mm_numerics 16): tfc passes each operand's fp32 bits through, and mma_op splits each into its TF32 value and the remainder and
// takes the product as lo x hi + hi x lo + hi x hi (kvm_fgpt::mma_frag, in its order); the one-shot GEMMs run precise by kPrec. Otherwise
// it is the 1xTF32 mma itself.
__device__ __forceinline__ void mma_op(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if KVM_F2D_PRECISE
  const unsigned b[2] = {b0, b1};
  mma_frag(c, a, b, true);
#else
  mma_tf32(c, a, b0, b1);
#endif
}

// The producers' rounding: every GEMM operand rounded once where it is born, so the GEMMs read it with no rounding at use and
// the tensor core sees the same 19 bits as when each GEMM rounds its own copy (bitwise outputs). pre_round_rows rounds rows r0 .. r1 of a
// d-wide array in place (a LayerNorm's output: only GEMMs read it); pre_round_tile a staged shared-memory tile (o's, dqkv's: their
// float32 originals stay in global memory for the float32 readers); copy_rows_rounded writes a rounded copy of rows r0 .. r1 beside the
// float32 original (a gradient that GEMMs and float32 sums both read), when its descriptor carries one. In the precise build the first two
// do nothing, and no descriptor carries a copy.
__device__ __forceinline__ void pre_round_rows(float* x, int r0, int r1, int d) {
#if !KVM_F2D_PRECISE
  __syncthreads();
  for (int e = threadIdx.x; e < (r1 - r0) * d; e += FNT) { float* const q = x + (size_t)r0 * d + e; *q = tfr(*q); }
  __syncthreads();
#else
  (void)x; (void)r0; (void)r1; (void)d;
#endif
}
__device__ __forceinline__ void pre_round_tile(float* t, int ld, int rows, int cols) {
#if !KVM_F2D_PRECISE
  for (int e = threadIdx.x; e < rows * cols; e += FNT) { const int i = e / cols, j = e % cols; t[i * ld + j] = tfr(t[i * ld + j]); }
  __syncthreads();
#else
  (void)t; (void)ld; (void)rows; (void)cols;
#endif
}
__device__ __forceinline__ void copy_rows_rounded(float* dst, const float* src, int r0, int r1, int d) {
  if (!dst) return;
  __syncthreads();
  for (int e = threadIdx.x; e < (r1 - r0) * d; e += FNT) dst[(size_t)r0 * d + e] = tfr(src[(size_t)r0 * d + e]);
}

// The column of tile q's local column x (0..7) among a warp's CPW 8-column tiles from `base`: tile q holds base + 8 q + x.
template <int CPW>
__device__ __forceinline__ int bcol(int base, int q, int x) {
  return base + 8 * q + x;
}

// Every tile's B fragment (raw bits) at k rows kr and kr + 4 of a K-major B (B(k, n) = B[k lsb + n], the untransposed form), the warp's
// tiles at columns bcol<CPW>(base, q, g): b[q][0] = B(kr, .), b[q][1] = B(kr + 4, .).
template <int CPW>
__device__ __forceinline__ void bfrag_k(unsigned (&b)[CPW][2], const float* B, int lsb, int kr, int base, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float* const row = B + (kr + 4 * h) * lsb + base;
#pragma unroll
    for (int q = 0; q < CPW; ++q) b[q][h] = __float_as_uint(row[8 * q + g]);
  }
}

// The same product with its accumulators in registers for the whole K and S chunks in flight: warp w owns row block w % TM16 (TM16 = M / 16,
// which must divide FNW) and exactly CPW of the block's 8-column tiles (N / 8 = CPW x FNW / TM16), so each K step loads its A fragment once
// and every tile's B fragment before their CPW products issue back to back, and no chunk touches an accumulator in shared memory. The
// stream region holds S stages of an A chunk and a B chunk; chunk k + S - 1 is issued as chunk k's products start (cp.async.wait_group
// S - 2 waits only for the oldest). Every dimension and leading dimension a multiple of 4 and every base 16 B aligned (the 16 B units).
// epi(m, n, v) runs once an element after the last chunk. Ends with __syncthreads.
template <bool TA, bool TB, int CPW, int S, int KC, int RB = 1, class Epi>
__device__ void sgemm_reg(const float* Ag, int lda, const float* Bg, int ldb, int M, int N, int K, float* region, Epi epi) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int M16 = kvm_fgpt_r16(M), N8 = kvm_fgpt_r8(N), KC8 = kvm_fgpt_r8(KC), tm = M16 / 16;
  const int gm = tm / RB, m0 = (warp % gm) * 16 * RB, c0 = (warp / gm) * CPW;   // RB row blocks and CPW column tiles a warp
  const int lsa = TA ? kvm_fgpt_ldmn(M) : kvm_fgpt_ldk(KC), lsb = TB ? kvm_fgpt_ldk(KC) : kvm_fgpt_ldmn(N);
  const int achunk = (int)kvm_fgpt_r4(TA ? (long long)KC8 * lsa : (long long)M16 * lsa), bchunk = (int)kvm_fgpt_r4(TB ? (long long)N8 * lsb : (long long)KC8 * lsb);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ak = (lane >> 4) * 4, br = lane & 7, bk = ((lane >> 3) & 1) * 4;
  Units<stream_units<KC>()> ua, ub;
  if (TA) ua.plan(KC8, M16, lsa, lda, true, M); else ua.plan(M16, KC8, lsa, lda, false, M);
  if (TB) ub.plan(N8, KC8, lsb, ldb, false, N); else ub.plan(KC8, N8, lsb, ldb, true, N);
  float c[RB][CPW][4];
#pragma unroll
  for (int i = 0; i < RB; ++i)
#pragma unroll
    for (int j = 0; j < CPW; ++j) c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.f;
  const int nk = (K + KC - 1) / KC;
  auto issue = [&](int k) {                                               // chunk k into stage k % S; past the last chunk an empty group
    if (k < nk) {
      float* const A = region + (k % S) * (achunk + bchunk);
      const int k0 = k * KC, kc = min(KC, K - k0);
      ua.issue(A, TA ? Ag + (size_t)k0 * lda : Ag + k0, kc);
      ub.issue(A + achunk, TB ? Bg + k0 : Bg + (size_t)k0 * ldb, kc);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
#pragma unroll
  for (int k = 0; k < S - 1; ++k) issue(k);
  for (int k = 0; k < nk; ++k) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2) : "memory");    // chunk k has landed (the S - 2 younger groups may not have)
    __syncthreads();                                                       // ... for every thread, and stage (k - 1) % S is free
    const bool full = (k + 1) * KC <= K;
    issue(k + S - 1);
    const float* const A = region + (k % S) * (achunk + bchunk);
    const float* const B = A + achunk;
    auto step = [&](int s8) {
      unsigned a[RB][4], b[CPW][2];
#pragma unroll
      for (int i = 0; i < RB; ++i) {
        const int mi = m0 + 16 * i;
        if (!TA) ldm_x4(a[i], A + (mi + ar) * lsa + s8 * 8 + ak);
        else { const float* pa = A + (s8 * 8 + t) * lsa + mi + g; a[i][0] = __float_as_uint(pa[0]); a[i][1] = __float_as_uint(pa[8]); a[i][2] = __float_as_uint(pa[4 * lsa]); a[i][3] = __float_as_uint(pa[4 * lsa + 8]); }
      }
      if (TB) {                                                            // every tile's B fragment first: CPW independent loads in flight
#pragma unroll
        for (int j = 0; j < CPW; ++j) ldm_x2(b[j], B + ((c0 + j) * 8 + br) * lsb + s8 * 8 + bk);
      } else bfrag_k<CPW>(b, B, lsb, s8 * 8 + t, c0 * 8, g);
      unsigned b2[CPW][2];
#pragma unroll
      for (int j = 0; j < CPW; ++j) { b2[j][0] = tfs(b[j][0]); b2[j][1] = tfs(b[j][1]); }
#pragma unroll
      for (int i = 0; i < RB; ++i) {
        const unsigned a2[4] = {tfs(a[i][0]), tfs(a[i][1]), tfs(a[i][2]), tfs(a[i][3])};
#pragma unroll
        for (int j = 0; j < CPW; ++j) mma_op(c[i][j], a2, b2[j][0], b2[j][1]);
      }
    };
    if (full) {                                                            // a full chunk: its KC / 8 steps unrolled, loads hoisted across them
#pragma unroll
      for (int s8 = 0; s8 < KC / 8; ++s8) {
        step(s8);
      }
    } else {
      const int ks = (K - k * KC + 7) >> 3;
      for (int s8 = 0; s8 < ks; ++s8) step(s8);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");                   // the empty groups past the last chunk
#pragma unroll
  for (int i = 0; i < RB; ++i)
#pragma unroll
    for (int j = 0; j < CPW; ++j) {
      const int ma = m0 + 16 * i + g, mb = ma + 8;
      const int ca = TB ? (c0 + j) * 8 + 2 * t : bcol<CPW>(c0 * 8, j, 2 * t), cb = TB ? ca + 1 : bcol<CPW>(c0 * 8, j, 2 * t + 1);
      if (ma < M && ca < N) epi(ma, ca, c[i][j][0]);
      if (ma < M && cb < N) epi(ma, cb, c[i][j][1]);
      if (mb < M && ca < N) epi(mb, ca, c[i][j][2]);
      if (mb < M && cb < N) epi(mb, cb, c[i][j][3]);
    }
  __syncthreads();
}

// C (M x N) = A (M x K) B (K x N) with A resident in shared memory (A(m, k) = As[TA ? k lsa + m : m lsa + k]) and B streamed from global
// memory by output tiles of TN columns (B(k, n) = Bg[TB ? n ldb + k : k ldb + n]), S tiles' B blocks in flight. Warp w owns row block
// w % TM16 and CPW of each tile's 8-column tiles (TN / 8 = CPW x FNW / TM16); it reloads its A fragments from shared memory at every
// tile's k steps (held across the tiles they would take KMAX / 8 x 4 more registers, and the kernel has none to spare: a compile that held them
// spilled 3.7 KB). epi(m, n, v) runs once an element as each output tile completes. K <= KMAX, K a multiple of 8, M and N
// multiples of 16 and of TN. Ends with __syncthreads.
template <bool TA, bool TB, int CPW, int S, int KMAX, class Epi>
__device__ void sgemm_rs(const float* As, int lsa, const float* Bg, int ldb, int M, int N, int K, float* region, Epi epi) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tm = kvm_fgpt_r16(M) / 16, m0 = (warp % tm) * 16, c0 = (warp / tm) * CPW, ks = K / 8, nt = N / TN;
  const int lsb = TB ? kvm_fgpt_ldk(K) : kvm_fgpt_ldmn(TN);
  const int bchunk = (int)kvm_fgpt_r4(TB ? (long long)TN * lsb : (long long)kvm_fgpt_r8(K) * lsb);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ak = (lane >> 4) * 4, br = lane & 7, bk = ((lane >> 3) & 1) * 4;
  auto afrag = [&](int s8, unsigned (&a)[4]) {                             // this warp's A fragment at k step s8, TF32-rounded, from shared memory
    unsigned r[4];
    if (!TA) ldm_x4(r, As + (m0 + ar) * lsa + s8 * 8 + ak);
    else { const float* pa = As + (s8 * 8 + t) * lsa + m0 + g; r[0] = __float_as_uint(pa[0]); r[1] = __float_as_uint(pa[8]); r[2] = __float_as_uint(pa[4 * lsa]); r[3] = __float_as_uint(pa[4 * lsa + 8]); }
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = tfres(r[i]);
  };
  Units<8> ub;
  if (TB) ub.plan(TN, K, lsb, ldb, false, TN); else ub.plan(kvm_fgpt_r8(K), TN, lsb, ldb, true, TN);
  auto issue = [&](int j) {                                                // output tile j's B block into stage j % S
    if (j < nt) ub.issue(region + (j % S) * bchunk, TB ? Bg + (size_t)j * TN * ldb : Bg + (size_t)j * TN, TB ? K : K);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
#pragma unroll
  for (int j = 0; j < S - 1; ++j) issue(j);
  for (int j = 0; j < nt; ++j) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2) : "memory");
    __syncthreads();
    issue(j + S - 1);
    const float* const B = region + (j % S) * bchunk;
    float c[CPW][4];
#pragma unroll
    for (int q = 0; q < CPW; ++q) c[q][0] = c[q][1] = c[q][2] = c[q][3] = 0.f;
#pragma unroll
    for (int s8 = 0; s8 < KMAX / 8; ++s8) {
      if (s8 >= ks) break;
      unsigned b[CPW][2];
      if (TB) {
#pragma unroll
        for (int q = 0; q < CPW; ++q) ldm_x2(b[q], B + ((c0 + q) * 8 + br) * lsb + s8 * 8 + bk);
      } else bfrag_k<CPW>(b, B, lsb, s8 * 8 + t, c0 * 8, g);
      unsigned a[4];
      afrag(s8, a);
#pragma unroll
      for (int q = 0; q < CPW; ++q) mma_op(c[q], a, tfs(b[q][0]), tfs(b[q][1]));
    }
#pragma unroll
    for (int q = 0; q < CPW; ++q) {
      const int ma = m0 + g, mb = ma + 8;
      const int ca = j * TN + (TB ? (c0 + q) * 8 + 2 * t : bcol<CPW>(c0 * 8, q, 2 * t)), cb = TB ? ca + 1 : j * TN + bcol<CPW>(c0 * 8, q, 2 * t + 1);
      epi(ma, ca, c[q][0]); epi(ma, cb, c[q][1]); epi(mb, ca, c[q][2]); epi(mb, cb, c[q][3]);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
}

// C (M x N) = A (M x K) B (K x N) with A streamed from global memory by output row tiles of TMS rows (A(m, k) = Ag[TA ? k lda + m :
// m lda + k]) and B resident in shared memory (B(k, n) = Bs[TB ? n lsb + k : k lsb + n]), S tiles' A blocks in flight: the mirror of
// sgemm_rs, for products whose wide dimension is the rows (dW1 = X^T dU over d's rows, dW2 computed as its transpose dY^T A). Warp w owns
// row block w % (TMS / 16) of each tile and CPW of its 8-column tiles (N / 8 = CPW x FNW / (TMS / 16)). epi(m, n, v) runs once an element
// as each row tile completes. K <= KMAX, K a multiple of 8, M a multiple of TMS. Ends with __syncthreads.
template <bool TA, bool TB, int CPW, int S, int KMAX, class Epi>
__device__ void sgemm_sr(const float* Ag, int lda, const float* Bs, int lsb, int M, int N, int K, float* region, Epi epi) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tm = TMS / 16, m0 = (warp % tm) * 16, c0 = (warp / tm) * CPW, ks = K / 8, nt = M / TMS;
  const int lsa = TA ? kvm_fgpt_ldmn(TMS) : kvm_fgpt_ldk(K);
  const int achunk = (int)kvm_fgpt_r4(TA ? (long long)kvm_fgpt_r8(K) * lsa : (long long)TMS * lsa);
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ak = (lane >> 4) * 4, br = lane & 7, bk = ((lane >> 3) & 1) * 4;
  Units<8> ua;
  if (TA) ua.plan(kvm_fgpt_r8(K), TMS, lsa, lda, true, TMS); else ua.plan(TMS, K, lsa, lda, false, TMS);
  auto issue = [&](int j) {                                                // row tile j's A block into stage j % S
    if (j < nt) ua.issue(region + (j % S) * achunk, TA ? Ag + (size_t)j * TMS : Ag + (size_t)j * TMS * lda, K);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
#pragma unroll
  for (int j = 0; j < S - 1; ++j) issue(j);
  for (int j = 0; j < nt; ++j) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2) : "memory");
    __syncthreads();
    issue(j + S - 1);
    const float* const A = region + (j % S) * achunk;
    float c[CPW][4];
#pragma unroll
    for (int q = 0; q < CPW; ++q) c[q][0] = c[q][1] = c[q][2] = c[q][3] = 0.f;
#pragma unroll
    for (int s8 = 0; s8 < KMAX / 8; ++s8) {
      if (s8 >= ks) break;
      unsigned a[4];
      if (!TA) ldm_x4(a, A + (m0 + ar) * lsa + s8 * 8 + ak);
      else { const float* pa = A + (s8 * 8 + t) * lsa + m0 + g; a[0] = __float_as_uint(pa[0]); a[1] = __float_as_uint(pa[8]); a[2] = __float_as_uint(pa[4 * lsa]); a[3] = __float_as_uint(pa[4 * lsa + 8]); }
      const unsigned a2[4] = {tfs(a[0]), tfs(a[1]), tfs(a[2]), tfs(a[3])};
      unsigned b[CPW][2];
      if (TB) {
#pragma unroll
        for (int q = 0; q < CPW; ++q) ldm_x2(b[q], Bs + ((c0 + q) * 8 + br) * lsb + s8 * 8 + bk);
      } else bfrag_k<CPW>(b, Bs, lsb, s8 * 8 + t, c0 * 8, g);
#pragma unroll
      for (int q = 0; q < CPW; ++q) mma_op(c[q], a2, tfres(b[q][0]), tfres(b[q][1]));
    }
#pragma unroll
    for (int q = 0; q < CPW; ++q) {
      const int ma = j * TMS + m0 + g, mb = ma + 8;
      const int ca = TB ? (c0 + q) * 8 + 2 * t : bcol<CPW>(c0 * 8, q, 2 * t), cb = TB ? ca + 1 : bcol<CPW>(c0 * 8, q, 2 * t + 1);
      epi(ma, ca, c[q][0]); epi(ma, cb, c[q][1]); epi(mb, ca, c[q][2]); epi(mb, cb, c[q][3]);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
}

// ---- the phase marks a caller's clock reads (run's `ph`): the attention half's, the MLP half's (MPh) and the step's own (SPh) ---------
enum APh { A_LN1, A_X0, A_QKV, A_X1, A_ITEMS, A_X2, A_PROJ, A_X3, A_RS, A_X4, B_DO, B_DWO, B_X1, B_ITEMSQ, B_X2, B_ITEMSK, B_X3, B_QKV, B_X4, B_LN1, B_X5,
            A_N };

// q|k|v's register GEMM, the instance the layout's tiling asks for: CPW 3 or 4 tiles a warp, RB 1 or 2 row blocks, KC 32 or 48.
template <class Epi>
__device__ __forceinline__ void qkv_gemm(const ADesc& p, const float* A, const float* B, float* region, Epi e) {
  const int R = p.R, Ns = p.Ns, d = p.d, ldb = p.ldw;
#define QKV_CALL(CPW_, RB_, KC_) sgemm_reg<false, false, CPW_, STAGES, KC_, RB_>(A, d, B, ldb, R, Ns, d, region, e)
  if (p.kc == 48) { if (p.cpw == 4) { if (p.rb == 2) QKV_CALL(4, 2, 48); else QKV_CALL(4, 1, 48); } else { if (p.rb == 2) QKV_CALL(3, 2, 48); else QKV_CALL(3, 1, 48); } }
  else { if (p.cpw == 4) { if (p.rb == 2) QKV_CALL(4, 2, 32); else QKV_CALL(4, 1, 32); } else { if (p.rb == 2) QKV_CALL(3, 2, 32); else QKV_CALL(3, 1, 32); } }
#undef QKV_CALL
}

// d_o's register GEMM: d_o[group rows, member columns] = dY[group rows] Wo[member rows, :]^T (B(k, n) = Wo[(k0 + n) d + k]); its N is
// dc = d / Gc, so one or two tiles a warp.
template <class Epi>
__device__ __forceinline__ void do_gemm(const ADesc& p, const float* A, const float* B, float* region, Epi e) {
  const int R = p.R, dc = p.dc, d = p.d;
#define DO_CALL(CPW_, RB_, KC_) sgemm_reg<false, true, CPW_, STAGES, KC_, RB_>(A, d, B, d, R, dc, d, region, e)
  if (p.kc == 48) { if (p.cpw_do == 2) { if (p.rb_do == 2) DO_CALL(2, 2, 48); else DO_CALL(2, 1, 48); } else { if (p.rb_do == 2) DO_CALL(1, 2, 48); else DO_CALL(1, 1, 48); } }
  else { if (p.cpw_do == 2) { if (p.rb_do == 2) DO_CALL(2, 2, 32); else DO_CALL(2, 1, 32); } else { if (p.rb_do == 2) DO_CALL(1, 2, 32); else DO_CALL(1, 1, 32); } }
#undef DO_CALL
}

// ---- the attention half of a layer, as functions a kernel calls: the forward (LN1 through the residual) and the backward (proj's through
// LN1's and the dW sums), each with its crossings. The gang has G members; the half's layout (p.gt x p.gc) is the first Gt Gc of them.
// Members past it take items (the items run over all G) and join every crossing; with G = Gt Gc every member is the layout's.
// Each half is inlined into its caller. A kernel passes its descriptors as __grid_constant__ parameters, so a half reads them in
// parameter space, not from a stack copy.
template <class XC, class PH>
__device__ __forceinline__ bool attn_fwd(const ADesc& p, float* smem, int G, int b, XC& n, PH& ph) {
  const int t = b / p.gc, c = b % p.gc, tid = threadIdx.x, Ga = p.gt * p.gc;
  const int R = p.R, d = p.d, hd = p.hd, H = p.H, T = p.T, QB = p.qb, TQ = p.TQ, r0 = t * R, n0 = c * p.Ns, k0 = c * p.dc;
  const int qa = c * R / p.gc, q0r = r0 + qa, rr = (c + 1) * R / p.gc - qa;
  const ALay Ly = alayout(R, p.Ns, p.kc, T, TQ, hd, p.dc);
  const bool lay = b < Ga;   // a member of the half's layout; the rest take items and join every crossing
  (void)tid; (void)k0; (void)n0; (void)q0r; (void)rr;
  // ---- LN1 on the member's own rows (the reduce-scatter's rows): h = (x - mean) rstd g1 + b1, the statistics kept for the backward
  if (lay) { ln_fwd<FNT>(p.X, p.g1, p.b1n, p.Hs, p.mean, p.rstd, q0r, q0r + rr, d, p.eps); pre_round_rows(p.Hs, q0r, q0r + rr, d); }
  ph(A_LN1);
  if (!n(p.count, G, p.abort)) return false;
  ph(A_X0);
  // ---- q|k|v column-parallel: QKV[group rows, n0 .. n0 + Ns) = H[group rows] Wqkv[:, n0 ..] + bqkv
  if (lay)
    qkv_gemm(p, p.Hs + (size_t)r0 * d, p.Wq3[n0 / d] + n0 % d, smem,
             [&](int m, int nn, float v) { p.QKV[(size_t)(r0 + m) * 3 * d + n0 + nn] = v + p.bq3[n0 / d][n0 % d + nn]; });
  ph(A_QKV);
  if (!n(p.count, G, p.abort)) return false;
  ph(A_X1);
  // ---- the items: (sequence, head, query block), item w on member w % G
  for (int w = b; w < p.W; w += G) {
    const int u = w / QB, qb = w % QB, sq = u / H, hh = u % H, ch = hh * hd, rq = sq * T + qb * TQ, q0 = qb * TQ;
    const int KE = q0 + TQ, KE16 = kvm_fgpt_r16(KE), TQ16 = kvm_fgpt_r16(TQ), ldh = Ly.ldh, ldS = Ly.ldS;
    float* const Q = smem;
    float* const K = Q + kvm_fgpt_r4((long long)TQ16 * ldh);
    float* const Vv = K + kvm_fgpt_r4((long long)kvm_fgpt_r16(T) * ldh);
    float* const S = Vv + kvm_fgpt_r4((long long)kvm_fgpt_r16(T) * ldh);
    {
      const size_t l3 = 3 * (size_t)d;
      const Copy cs[3] = {{Q, p.QKV + (size_t)rq * l3 + ch, ldh, l3, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                          {K, p.QKV + (size_t)sq * T * l3 + d + ch, ldh, l3, KE, hd, KE16, kvm_fgpt_r8(hd)},
                          {Vv, p.QKV + (size_t)sq * T * l3 + 2 * d + ch, ldh, l3, KE, hd, KE16, kvm_fgpt_r8(hd)}};
      zero<FNT>(S, ldS, TQ16, KE16);
      stage_all<FNT>(cs);
    }
    rope<FNT>(Q, ldh, TQ, q0, T, hd, p.rtab, 1.f);
    rope<FNT>(K, ldh, KE, 0, T, hd, p.rtab, 1.f);
    // the queries' (positions q0 + m) scores over the prefix's keys, 3xTF32 as the fused step's attention, causal; then each row's softmax
    gemm<false, true, FNT>(Q, ldh, K, ldh, TQ, KE, hd, true, [&](int m, int nn, float a) { S[m * ldS + nn] = nn <= q0 + m ? a * p.scale : 0.f; });
    {
      const int wp = tid >> 5, lane = tid & 31;
      for (int i = wp; i < TQ; i += FNW) {
        const int last = q0 + i;
        float mx = -INFINITY;
        for (int j = lane; j <= last; j += 32) mx = fmaxf(mx, S[i * ldS + j]);
        mx = warp_max(mx);
        float l1 = 0.f;
        for (int j = lane; j <= last; j += 32) { const float e = expf(S[i * ldS + j] - mx); S[i * ldS + j] = e; l1 += e; }
        l1 = warp_sum(l1);
        for (int j = lane; j <= last; j += 32) S[i * ldS + j] /= l1;
        if (lane == 0) p.LSE[(size_t)(sq * H + hh) * T + q0 + i] = mx + logf(l1);
      }
    }
    __syncthreads();
    gemm<false, false, FNT>(S, ldS, Vv, ldh, TQ, hd, KE, true, [&](int m, int nn, float a) { p.O[(size_t)(rq + m) * d + ch + nn] = a; });
  }
  ph(A_ITEMS);
  if (!n(p.count, G, p.abort)) return false;
  ph(A_X2);
  // ---- proj row-parallel: o's tile (the group's rows, columns k0 .. k0 + dc) resident, times Wo's rows k0 .., a partial of Y
  if (lay) {
    float* const Ot = smem;
    float* const region = smem + kvm_fgpt_r4((long long)kvm_fgpt_r16(R) * Ly.ldo);
    stage<FNT>(Ot, Ly.ldo, p.O + (size_t)r0 * d + k0, (size_t)d, R, p.dc, kvm_fgpt_r16(R), kvm_fgpt_r8(p.dc));
    pre_round_tile(Ot, Ly.ldo, kvm_fgpt_r16(R), kvm_fgpt_r8(p.dc));
    float* const Pt = p.P + ((size_t)(t * p.gc + c) * R) * d;
    auto e = [&](int m, int nn, float v) { Pt[(size_t)m * d + nn] = v; };
    if (p.cpw_rs == 2) sgemm_rs<false, false, 2, RS_STAGES, 96>(Ot, Ly.ldo, p.Wo + (size_t)k0 * d, d, R, d, p.dc, region, e);
    else if (p.cpw_rs == 3) sgemm_rs<false, false, 3, RS_STAGES, 96>(Ot, Ly.ldo, p.Wo + (size_t)k0 * d, d, R, d, p.dc, region, e);
    else sgemm_rs<false, false, 4, RS_STAGES, 96>(Ot, Ly.ldo, p.Wo + (size_t)k0 * d, d, R, d, p.dc, region, e);
  }
  ph(A_PROJ);
  if (!n(p.count, G, p.abort)) return false;
  ph(A_X3);
  // ---- the group's reduce-scatter: this member's rr rows, the Gc partials in member order, + bo
  if (lay) {
    for (int e = tid; e < rr * d / 4; e += FNT) {
      const int i = e / (d / 4), j = (e % (d / 4)) * 4;
      const float4 a = sum_in_order4(p.P + ((size_t)(t * p.gc) * R + qa + i) * d + j, (size_t)R * d, p.gc);
      const float4 bb = *reinterpret_cast<const float4*>(p.bo + j);
      const float4 y4 = make_float4(a.x + bb.x, a.y + bb.y, a.z + bb.z, a.w + bb.w), x4 = *reinterpret_cast<const float4*>(p.X + (size_t)(q0r + i) * d + j);
      *reinterpret_cast<float4*>(p.Y + (size_t)(q0r + i) * d + j) = y4;
      *reinterpret_cast<float4*>(p.XM + (size_t)(q0r + i) * d + j) = make_float4(x4.x + y4.x, x4.y + y4.y, x4.z + y4.z, x4.w + y4.w);
    }
  }
  ph(A_RS);
  if (!n(p.count, G, p.abort)) return false;
  ph(A_X4);
  return true;
}

template <class XC, class PH>
__device__ __forceinline__ bool attn_bwd(const ADesc& p, float* smem, int G, int b, XC& n, PH& ph) {
  const int t = b / p.gc, c = b % p.gc, tid = threadIdx.x, Ga = p.gt * p.gc;
  const int R = p.R, d = p.d, hd = p.hd, H = p.H, T = p.T, QB = p.qb, TQ = p.TQ, r0 = t * R, n0 = c * p.Ns, k0 = c * p.dc;
  const int qa = c * R / p.gc, q0r = r0 + qa, rr = (c + 1) * R / p.gc - qa;
  const ALay Ly = alayout(R, p.Ns, p.kc, T, TQ, hd, p.dc);
  const bool lay = b < Ga;   // a member of the half's layout; the rest take items and join every crossing
  (void)tid; (void)k0; (void)n0; (void)q0r; (void)rr;
  // ======== backward from dY ========
  // ---- proj's (row-parallel): d_o[group rows, k0 .. k0 + dc) = dY[group rows] Wo[k0 .., :]^T, the member's own columns of d_o
  const float* const dYg = p.dYg ? p.dYg : p.dY;                                      // the GEMMs' dY (PRE: rounded where born)
  if (lay) do_gemm(p, dYg + (size_t)r0 * d, p.Wo + (size_t)k0 * d, smem, [&](int m, int nn, float v) { p.dO[(size_t)(r0 + m) * d + k0 + nn] = v; });
  ph(B_DO);
  // dWo[k0 .., :] partial over the group's rows, computed as its transpose dY[group rows]^T O_t (o's tile resident, dY's rows streamed by
  // row tiles of TMS); dbo's partial over the group's rows (the member's dc columns)
  if (lay) {
    float* const Ot = smem;
    float* const region = smem + kvm_fgpt_r4((long long)kvm_fgpt_r16(R) * Ly.ldo);
    stage<FNT>(Ot, Ly.ldo, p.O + (size_t)r0 * d + k0, (size_t)d, R, p.dc, kvm_fgpt_r16(R), kvm_fgpt_r8(p.dc));
    pre_round_tile(Ot, Ly.ldo, kvm_fgpt_r16(R), kvm_fgpt_r8(p.dc));
    float* const Wp = p.dWop + (size_t)t * d * d;
    auto e = [&](int m, int nc, float v) { Wp[(size_t)(k0 + nc) * d + m] = v; };
    if (p.cpw_sro == 2) sgemm_sr<true, false, 2, RS_STAGES, 96>(dYg + (size_t)r0 * d, d, Ot, Ly.ldo, d, p.dc, R, region, e);
    else sgemm_sr<true, false, 4, RS_STAGES, 96>(dYg + (size_t)r0 * d, d, Ot, Ly.ldo, d, p.dc, R, region, e);
    for (int j = k0 + tid; j < k0 + p.dc; j += FNT) {
      float a = 0.f;
      for (int i = 0; i < R; ++i) a += p.dY[(size_t)(r0 + i) * d + j];
      p.dbop[(size_t)t * d + j] = a;
    }
  }
  ph(B_DWO);
  if (!n(p.count, G, p.abort)) return false;
  ph(B_X1);
  // ---- the items' backward as query blocks (the fused step's): do, delta, P, the dV and dK partials over the prefix, dS, dq (RoPE undone, into dQKV)
  for (int w = b; w < p.W; w += G) {
    const int u = w / QB, qb = w % QB, sq = u / H, hh = u % H, ch = hh * hd, rq = sq * T + qb * TQ, q0 = qb * TQ;
    const int KE = q0 + TQ, KE16 = kvm_fgpt_r16(KE), TQ16 = kvm_fgpt_r16(TQ), T16 = kvm_fgpt_r16(T), ldh = Ly.ldh, ldS = Ly.ldS;
    float* const DO = smem;
    float* const Os = DO + kvm_fgpt_r4((long long)TQ16 * ldh);
    float* const Q = Os + kvm_fgpt_r4((long long)TQ16 * ldh);
    float* const DQ = Q + kvm_fgpt_r4((long long)TQ16 * ldh);
    float* const K = DQ + kvm_fgpt_r4((long long)TQ16 * ldh);
    float* const Vv = K + kvm_fgpt_r4((long long)T16 * ldh);
    float* const DK = Vv + kvm_fgpt_r4((long long)T16 * ldh);
    float* const DV = DK + kvm_fgpt_r4((long long)T16 * ldh);
    float* const S = DV + kvm_fgpt_r4((long long)T16 * ldh);
    float* const delta = S + kvm_fgpt_r4((long long)TQ16 * ldS);
    float* const lse = delta + kvm_fgpt_r4(TQ);
    {
      const size_t l3 = 3 * (size_t)d;
      const Copy cs[6] = {{DO, p.dO + (size_t)rq * d + ch, ldh, (size_t)d, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                          {Os, p.O + (size_t)rq * d + ch, ldh, (size_t)d, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                          {Q, p.QKV + (size_t)rq * l3 + ch, ldh, l3, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                          {K, p.QKV + (size_t)sq * T * l3 + d + ch, ldh, l3, KE, hd, KE16, kvm_fgpt_r8(hd)},
                          {Vv, p.QKV + (size_t)sq * T * l3 + 2 * d + ch, ldh, l3, KE, hd, KE16, kvm_fgpt_r8(hd)},
                          {lse, p.LSE + (size_t)(sq * H + hh) * T + q0, 0, 0, 1, TQ, 1, TQ}};
      zero<FNT>(S, ldS, TQ16, KE16);
      zero<FNT>(DQ, ldh, TQ16, kvm_fgpt_r8(hd));
      zero<FNT>(DK, ldh, KE16, kvm_fgpt_r8(hd));
      zero<FNT>(DV, ldh, KE16, kvm_fgpt_r8(hd));
      stage_all<FNT>(cs);
    }
    rope<FNT>(Q, ldh, TQ, q0, T, hd, p.rtab, 1.f);
    rope<FNT>(K, ldh, KE, 0, T, hd, p.rtab, 1.f);
    for (int i = tid; i < TQ; i += FNT) { float a = 0.f; for (int cc = 0; cc < hd; ++cc) a += Os[i * ldh + cc] * DO[i * ldh + cc]; delta[i] = a; }
    __syncthreads();
    gemm<false, true, FNT>(Q, ldh, K, ldh, TQ, KE, hd, true, [&](int m, int nn, float a) { S[m * ldS + nn] = nn <= q0 + m ? expf(a * p.scale - lse[m]) : 0.f; });
    gemm<true, false, FNT>(S, ldS, DO, ldh, KE, hd, TQ, true, [&](int m, int nn, float a) { DV[m * ldh + nn] = a; });                       // dV partial = P^T do
    gemm<false, true, FNT>(DO, ldh, Vv, ldh, TQ, KE, hd, true, [&](int m, int nn, float a) { if (nn <= q0 + m) S[m * ldS + nn] *= a - delta[m]; });   // dS
    gemm<false, false, FNT>(S, ldS, K, ldh, TQ, hd, KE, true, [&](int m, int nn, float a) { DQ[m * ldh + nn] = a * p.scale; });           // dq
    gemm<true, false, FNT>(S, ldS, Q, ldh, KE, hd, TQ, true, [&](int m, int nn, float a) { DK[m * ldh + nn] = a * p.scale; });            // dK partial
    {
      float* const dkv = p.DKV + (size_t)w * T * 2 * hd;
      for (int e = tid; e < KE * (hd / 4); e += FNT) {
        const int j = e / (hd / 4), cc = (e % (hd / 4)) * 4;
        *reinterpret_cast<float4*>(dkv + (size_t)j * 2 * hd + cc) = *reinterpret_cast<const float4*>(DK + j * ldh + cc);
        *reinterpret_cast<float4*>(dkv + (size_t)j * 2 * hd + hd + cc) = *reinterpret_cast<const float4*>(DV + j * ldh + cc);
      }
    }
    rope<FNT>(DQ, ldh, TQ, q0, T, hd, p.rtab, -1.f);
    for (int e = tid; e < TQ * hd; e += FNT) { const int i = e / hd, cc = e % hd; p.dQKV[(size_t)(rq + i) * 3 * d + ch + cc] = DQ[i * ldh + cc]; }
    __syncthreads();
  }
  ph(B_ITEMSQ);
  if (!n(p.count, G, p.abort)) return false;
  ph(B_X2);
  // ---- as key blocks: each sums its keys' dK and dV partials over the query blocks at and after it, in order; dk's RoPE undone
  for (int w = b; w < p.W; w += G) {
    const int u = w / QB, kb = w % QB, sq = u / H, hh = u % H, ch = hh * hd, k0b = kb * TQ, rk = sq * T + k0b, ldh = Ly.ldh;
    float* const DK = smem;
    float* const DV = DK + kvm_fgpt_r4((long long)kvm_fgpt_r16(TQ) * ldh);
    for (int e = tid; e < TQ * 2 * hd; e += FNT) {
      const int j = e / (2 * hd), cc = e % (2 * hd);
      float a = 0.f;
      for (int qq = kb; qq < QB; ++qq) a += p.DKV[((size_t)(u * QB + qq) * T + k0b + j) * 2 * hd + cc];
      if (cc < hd) DK[j * ldh + cc] = a; else DV[j * ldh + cc - hd] = a;
    }
    __syncthreads();
    rope<FNT>(DK, ldh, TQ, k0b, T, hd, p.rtab, -1.f);
    for (int e = tid; e < TQ * hd; e += FNT) {
      const int j = e / hd, cc = e % hd;
      p.dQKV[(size_t)(rk + j) * 3 * d + d + ch + cc] = DK[j * ldh + cc];
      p.dQKV[(size_t)(rk + j) * 3 * d + 2 * d + ch + cc] = DV[j * ldh + cc];
    }
    __syncthreads();
  }
  ph(B_ITEMSK);
  if (!n(p.count, G, p.abort)) return false;
  ph(B_X3);
  // ---- q|k|v's (column-parallel): the dqkv tile (the group's rows, the member's Ns columns) resident; dWqkv[:, n0 ..] partial over the
  // group's rows = H[group rows]^T dqkv_t (H's rows streamed by row tiles of TMS); dbqkv's partial; dH's partial over the member's columns
  // = dqkv_t Wqkv[:, n0 ..]^T (B(k, n) = Wqkv[n 3d + n0 + k], by column tiles of TN)
  if (lay) {
    float* const Dq = smem;
    float* const region = smem + kvm_fgpt_r4((long long)kvm_fgpt_r16(R) * Ly.ldq);
    stage<FNT>(Dq, Ly.ldq, p.dQKV + (size_t)r0 * 3 * d + n0, (size_t)3 * d, R, p.Ns, kvm_fgpt_r16(R), kvm_fgpt_r8(p.Ns));
    pre_round_tile(Dq, Ly.ldq, kvm_fgpt_r16(R), kvm_fgpt_r8(p.Ns));
    float* const Wq = p.dWqkvp + (size_t)t * d * 3 * d;
    auto e1 = [&](int m, int nc, float v) { Wq[(size_t)m * 3 * d + n0 + nc] = v; };
    if (p.cpw_srq == 6) sgemm_sr<true, false, 6, RS_STAGES, 96>(p.Hs + (size_t)r0 * d, d, Dq, Ly.ldq, d, p.Ns, R, region, e1);
    else if (p.cpw_srq == 4) sgemm_sr<true, false, 4, RS_STAGES, 96>(p.Hs + (size_t)r0 * d, d, Dq, Ly.ldq, d, p.Ns, R, region, e1);
    else sgemm_sr<true, false, 3, RS_STAGES, 96>(p.Hs + (size_t)r0 * d, d, Dq, Ly.ldq, d, p.Ns, R, region, e1);
    for (int j = tid; j < p.Ns; j += FNT) {                                     // dbqkv sums the float32 dqkv: global memory's under PRE
      float a = 0.f;
#if !KVM_F2D_PRECISE
      for (int i = 0; i < R; ++i) a += p.dQKV[(size_t)(r0 + i) * 3 * d + n0 + j];
#else
      for (int i = 0; i < R; ++i) a += Dq[i * Ly.ldq + j];
#endif
      p.dbqkvp[(size_t)t * 3 * d + n0 + j] = a;
    }
    float* const Pt = p.P + ((size_t)(t * p.gc + c) * R) * d;
    auto e2 = [&](int m, int nn, float v) { Pt[(size_t)m * d + nn] = v; };
    if (p.cpw_rs == 2) sgemm_rs<false, true, 2, RS_STAGES, 96>(Dq, Ly.ldq, p.Wq3[n0 / d] + n0 % d, p.ldw, R, d, p.Ns, region, e2);
    else if (p.cpw_rs == 3) sgemm_rs<false, true, 3, RS_STAGES, 96>(Dq, Ly.ldq, p.Wq3[n0 / d] + n0 % d, p.ldw, R, d, p.Ns, region, e2);
    else sgemm_rs<false, true, 4, RS_STAGES, 96>(Dq, Ly.ldq, p.Wq3[n0 / d] + n0 % d, p.ldw, R, d, p.Ns, region, e2);
  }
  ph(B_QKV);
  if (!n(p.count, G, p.abort)) return false;
  ph(B_X4);
  // ---- dH's reduce-scatter in the group, the member's rows; LN1's backward on them: dx = dxm (the residual path) + LN1's backward of dH,
  // and the member's partials of dgamma1 and dbeta1
  if (lay) {
    for (int e = tid; e < rr * d / 4; e += FNT) {
      const int i = e / (d / 4), j = (e % (d / 4)) * 4;
      *reinterpret_cast<float4*>(p.dH + (size_t)(q0r + i) * d + j) = sum_in_order4(p.P + ((size_t)(t * p.gc) * R + qa + i) * d + j, (size_t)R * d, p.gc);
      *reinterpret_cast<float4*>(p.dX + (size_t)(q0r + i) * d + j) = *reinterpret_cast<const float4*>(p.dY + (size_t)(q0r + i) * d + j);
    }
    __syncthreads();
    ln_bwd<FNT>(p.X, p.dH, p.g1, p.mean, p.rstd, p.dX, true, p.dg1p + (size_t)b * d, p.db1np + (size_t)b * d, q0r, q0r + rr, d);
    copy_rows_rounded(p.dXg, p.dX, q0r, q0r + rr, d);                                // PRE: the layer below's GEMM copy of its dxo
  }
  ph(B_LN1);
  if (!n(p.count, G, p.abort)) return false;
  ph(B_X5);
  return true;
}

// The half's dW sums: every dW element's Gt row-group partials, and LN1's Gt Gc member partials, each summed in order by its owner over
// flat shares of all G members. The caller crosses after it (a block runs its halves' sums in one phase).
__device__ __forceinline__ void attn_sums(const ADesc& p, int G, int b) {
  const int d = p.d, Ga = p.gt * p.gc;
  const long long i0 = (long long)b * FNT + threadIdx.x, step = (long long)G * FNT;
  sum_parts4(p.dWop, (size_t)d * d, p.gt, p.dWo, (long long)d * d / 4, i0, step);
  sum_parts4(p.dbop, (size_t)d, p.gt, p.dbo, d / 4, i0, step);
  // q|k|v's: each element of the fused [d][3d] and [3d] partials into its block (four columns never straddle one: d % 4 == 0)
  for (long long i = i0; i < (long long)d * 3 * d / 4; i += step) {
    const long long e = 4 * i;
    const int row = (int)(e / (3 * d)), col = (int)(e % (3 * d));
    *reinterpret_cast<float4*>(p.dWq3[col / d] + (size_t)row * p.ldg + col % d) = sum_in_order4(p.dWqkvp + e, (size_t)d * 3 * d, p.gt);
  }
  for (long long i = i0; i < 3 * d / 4; i += step) {
    const int col = 4 * (int)i;
    *reinterpret_cast<float4*>(p.dbq3[col / d] + col % d) = sum_in_order4(p.dbqkvp + col, (size_t)3 * d, p.gt);
  }
  sum_parts4(p.dg1p, (size_t)d, Ga, p.dg1, d / 4, i0, step);
  sum_parts4(p.db1np, (size_t)d, Ga, p.db1n, d / 4, i0, step);
}

enum MPh { ML_LN2 = A_N, ML_X0, ML_FC1, ML_FC2, ML_X1, ML_RS, ML_X2, ML_DA, ML_DW2, ML_DXP, ML_DW1, ML_X3, ML_LN2B, ML_X4, K_N };

// fc1 and dA: the group's R rows by the member's Fs columns over K = d, streamed (sgemm_reg); B untransposed (fc1: W1's columns) or
// transposed (dA: W2's rows). CPW 3 or 4 tiles a warp, or 6 as CPW 3 x RB 2; KC 32 or 48.
template <bool TB, class Epi>
__device__ __forceinline__ void mlp_reg_gemm(const MDesc& p, const float* A, const float* B, int ldb, float* region, Epi e) {
  const int R = p.R, Fs = p.Fs, d = p.d;
#define MLP_CALL(CPW_, RB_, KC_) sgemm_reg<false, TB, CPW_, STAGES, KC_, RB_>(A, d, B, ldb, R, Fs, d, region, e)
  if (p.kc == 48) { if (p.cpw == 4) MLP_CALL(4, 1, 48); else if (p.cpw == 3) MLP_CALL(3, 1, 48); else MLP_CALL(3, 2, 48); }
  else { if (p.cpw == 4) MLP_CALL(4, 1, 32); else if (p.cpw == 3) MLP_CALL(3, 1, 32); else MLP_CALL(3, 2, 32); }
#undef MLP_CALL
}

template <class XC, class PH>
__device__ __forceinline__ bool mlp_fwd(const MDesc& p, float* smem, int G, int b, XC& n, PH& ph) {
  const int t = b / p.gc, c = b % p.gc, tid = threadIdx.x, R = p.R, Fs = p.Fs, d = p.d, F = p.F, r0 = t * R, f0 = c * Fs;
  const int qa = c * R / p.gc, q0r = r0 + qa, rr = (c + 1) * R / p.gc - qa;    // the member's rows of the group's reduce-scatters
  const bool lay = b < p.gt * p.gc;                                              // a member of the half's layout
  const Lay Ly = layout(R, Fs, p.kc);
  float* const acc = smem + Ly.acc;
  float* const At = smem + Ly.At;
  float* const region = smem + Ly.region;
  const int ldt = Ly.ldt, lacc = Ly.lacc;
  // ---- LN2 on the member's own rows: h2 = LN(xm), the statistics kept for the backward
  if (lay) { ln_fwd<FNT>(p.XM, p.g2, p.b2n, p.H2, p.mean2, p.rstd2, q0r, q0r + rr, d, p.eps); pre_round_rows(p.H2, q0r, q0r + rr, d); }
  ph(ML_LN2);
  if (!n(p.count, G, p.abort)) return false;
  ph(ML_X0);
  // ---- fc1 column-parallel: U = H2[group rows] W1[:, member columns] + b1, A = GELU(U); A's tile stays resident for fc2
  if (lay) {
    mlp_reg_gemm<false>(p, p.H2 + (size_t)r0 * d, p.W1 + f0, F, region, [&](int m, int nn, float v) { acc[m * lacc + nn] = v; });
    for (int e = tid; e < R * Fs / 4; e += FNT) {                                // four columns a thread: 16 B loads and stores
      const int i = e / (Fs / 4), j = (e % (Fs / 4)) * 4;
      const float4 a4 = *reinterpret_cast<const float4*>(acc + i * lacc + j), b4 = *reinterpret_cast<const float4*>(p.b1 + f0 + j);
      const float4 u = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w), a = make_float4(gelu(u.x), gelu(u.y), gelu(u.z), gelu(u.w));
      *reinterpret_cast<float4*>(p.U + (size_t)(r0 + i) * F + f0 + j) = u;
      *reinterpret_cast<float4*>(p.A + (size_t)(r0 + i) * F + f0 + j) = a;
      *reinterpret_cast<float4*>(At + i * ldt + j) = make_float4(tfr(a.x), tfr(a.y), tfr(a.z), tfr(a.w));
    }
    __syncthreads();
  }
  ph(ML_FC1);
  // ---- fc2 row-parallel: the partial of y2 over the group's rows, A's member columns (resident) times W2's member rows
  if (lay) {
    float* const Pt = p.P + ((size_t)(t * p.gc + c) * R) * d;
    auto e = [&](int m, int nn, float v) { Pt[(size_t)m * d + nn] = v; };
    if (p.cpw_rs == 2) sgemm_rs<false, false, 2, RS_STAGES, 96>(At, ldt, p.W2 + (size_t)f0 * d, d, R, d, Fs, region, e);
    else sgemm_rs<false, false, 4, RS_STAGES, 96>(At, ldt, p.W2 + (size_t)f0 * d, d, R, d, Fs, region, e);
  }
  ph(ML_FC2);
  if (!n(p.count, G, p.abort)) return false;
  ph(ML_X1);
  // ---- the group's reduce-scatter on the member's rows: y2 = the Gc partials in member order + b2; xo = xm + y2
  if (lay) {
    for (int e = tid; e < rr * d / 4; e += FNT) {
      const int i = e / (d / 4), j = (e % (d / 4)) * 4;
      const float4 a = sum_in_order4(p.P + ((size_t)(t * p.gc) * R + qa + i) * d + j, (size_t)R * d, p.gc);
      const float4 bb = *reinterpret_cast<const float4*>(p.b2 + j);
      const float4 y4 = make_float4(a.x + bb.x, a.y + bb.y, a.z + bb.z, a.w + bb.w), x4 = *reinterpret_cast<const float4*>(p.XM + (size_t)(q0r + i) * d + j);
      *reinterpret_cast<float4*>(p.Y2 + (size_t)(q0r + i) * d + j) = y4;
      *reinterpret_cast<float4*>(p.XO + (size_t)(q0r + i) * d + j) = make_float4(x4.x + y4.x, x4.y + y4.y, x4.z + y4.z, x4.w + y4.w);
    }
  }
  ph(ML_RS);
  if (!n(p.count, G, p.abort)) return false;
  ph(ML_X2);
  return true;
}

template <class XC, class PH>
__device__ __forceinline__ bool mlp_bwd(const MDesc& p, float* smem, int G, int b, XC& n, PH& ph) {
  const int t = b / p.gc, c = b % p.gc, tid = threadIdx.x, R = p.R, Fs = p.Fs, d = p.d, F = p.F, r0 = t * R, f0 = c * Fs;
  const int qa = c * R / p.gc, q0r = r0 + qa, rr = (c + 1) * R / p.gc - qa;
  const bool lay = b < p.gt * p.gc;
  const float* const dXOg = p.dXOg ? p.dXOg : p.dXO;                                     // the GEMMs' dxo (PRE: rounded where born)
  const Lay Ly = layout(R, Fs, p.kc);
  float* const acc = smem + Ly.acc;
  float* const At = smem + Ly.At;
  float* const dUt = smem + Ly.dUt;
  float* const region = smem + Ly.region;
  const int ldt = Ly.ldt, lacc = Ly.lacc;
  // ---- A's tile restaged from global memory (rounded as fc1 rounded it), then dA = dxo[group rows] W2[member rows]^T (B(k, n) =
  // W2[(f0 + n) d + k]) and dU = dA GELU'(U): its tile resident (rounded), its values in global memory
  if (lay) {
    const Copy cs[1] = {{At, p.A + (size_t)r0 * F + f0, ldt, (size_t)F, R, Fs, kvm_fgpt_r16(R), kvm_fgpt_r8(Fs)}};
    stage_all<FNT>(cs);
#if !KVM_F2D_PRECISE
    for (int e = tid; e < R * Fs; e += FNT) { const int i = e / Fs, j = e % Fs; At[i * ldt + j] = tfr(At[i * ldt + j]); }
    __syncthreads();
#endif
    mlp_reg_gemm<true>(p, dXOg + (size_t)r0 * d, p.W2 + (size_t)f0 * d, d, region, [&](int m, int nn, float v) { acc[m * lacc + nn] = v; });
    for (int e = tid; e < R * Fs / 4; e += FNT) {
      const int i = e / (Fs / 4), j = (e % (Fs / 4)) * 4;
      const float4 a4 = *reinterpret_cast<const float4*>(acc + i * lacc + j), u4 = *reinterpret_cast<const float4*>(p.U + (size_t)(r0 + i) * F + f0 + j);
      const float4 du = make_float4(a4.x * gelu_grad(u4.x), a4.y * gelu_grad(u4.y), a4.z * gelu_grad(u4.z), a4.w * gelu_grad(u4.w));
      *reinterpret_cast<float4*>(dUt + i * ldt + j) = make_float4(tfr(du.x), tfr(du.y), tfr(du.z), tfr(du.w));
      *reinterpret_cast<float4*>(p.dU + (size_t)(r0 + i) * F + f0 + j) = du;
    }
    __syncthreads();
  }
  ph(ML_DA);
  // ---- dW2[member rows, :] partial over the group's rows, as dW2^T = dxo[group rows]^T A_t (dxo's rows streamed by TMS, A_t resident);
  // db2's partial over the group's rows (the member's d / Gc columns)
  if (lay) {
    float* const W = p.dW2p + (size_t)t * F * d;
    auto e = [&](int m, int nc, float v) { W[(size_t)(f0 + nc) * d + m] = v; };
    if (p.cpw_sr == 4) sgemm_sr<true, false, 4, RS_STAGES, 96>(dXOg + (size_t)r0 * d, d, At, ldt, d, Fs, R, region, e);
    else if (p.cpw_sr == 3) sgemm_sr<true, false, 3, RS_STAGES, 96>(dXOg + (size_t)r0 * d, d, At, ldt, d, Fs, R, region, e);
    else sgemm_sr<true, false, 6, RS_STAGES, 96>(dXOg + (size_t)r0 * d, d, At, ldt, d, Fs, R, region, e);
    for (int j = c * (d / p.gc) + tid; j < (c + 1) * (d / p.gc); j += FNT) {
      float a = 0.f;
      for (int i = 0; i < R; ++i) a += p.dXO[(size_t)(r0 + i) * d + j];
      p.db2p[(size_t)t * d + j] = a;
    }
  }
  ph(ML_DW2);
  // ---- dh2's partial over the member's columns: dU_t W1[:, member columns]^T (B(k, n) = W1[n F + f0 + k]), by column tiles of TN
  if (lay) {
    float* const Pt = p.P + ((size_t)(t * p.gc + c) * R) * d;
    auto e = [&](int m, int nn, float v) { Pt[(size_t)m * d + nn] = v; };
    if (p.cpw_rs == 2) sgemm_rs<false, true, 2, RS_STAGES, 96>(dUt, ldt, p.W1 + f0, F, R, d, Fs, region, e);
    else sgemm_rs<false, true, 4, RS_STAGES, 96>(dUt, ldt, p.W1 + f0, F, R, d, Fs, region, e);
  }
  ph(ML_DXP);
  // ---- dW1[:, member columns] partial over the group's rows = H2[group rows]^T dU_t (H2's rows streamed, dU_t resident); db1's partial
  if (lay) {
    float* const W = p.dW1p + (size_t)t * d * F;
    auto e = [&](int m, int nc, float v) { W[(size_t)m * F + f0 + nc] = v; };
    if (p.cpw_sr == 4) sgemm_sr<true, false, 4, RS_STAGES, 96>(p.H2 + (size_t)r0 * d, d, dUt, ldt, d, Fs, R, region, e);
    else if (p.cpw_sr == 3) sgemm_sr<true, false, 3, RS_STAGES, 96>(p.H2 + (size_t)r0 * d, d, dUt, ldt, d, Fs, R, region, e);
    else sgemm_sr<true, false, 6, RS_STAGES, 96>(p.H2 + (size_t)r0 * d, d, dUt, ldt, d, Fs, R, region, e);
    for (int j = tid; j < Fs; j += FNT) {                                        // db1 sums dU unrounded, from global memory
      float a = 0.f;
#if !KVM_F2D_PRECISE
      for (int i = 0; i < R; ++i) a += p.dU[(size_t)(r0 + i) * F + f0 + j];
#else
      for (int i = 0; i < R; ++i) a += dUt[i * ldt + j];
#endif
      p.db1p[(size_t)t * F + f0 + j] = a;
    }
  }
  ph(ML_DW1);
  if (!n(p.count, G, p.abort)) return false;
  ph(ML_X3);
  // ---- dh2's reduce-scatter on the member's rows; LN2's backward on them: dxm = dxo (the residual path) + LN2's backward of dh2, and the
  // member's partials of dgamma2 and dbeta2
  if (lay) {
    for (int e = tid; e < rr * d / 4; e += FNT) {
      const int i = e / (d / 4), j = (e % (d / 4)) * 4;
      *reinterpret_cast<float4*>(p.dH2 + (size_t)(q0r + i) * d + j) = sum_in_order4(p.P + ((size_t)(t * p.gc) * R + qa + i) * d + j, (size_t)R * d, p.gc);
      *reinterpret_cast<float4*>(p.dXM + (size_t)(q0r + i) * d + j) = *reinterpret_cast<const float4*>(p.dXO + (size_t)(q0r + i) * d + j);
    }
    __syncthreads();
    ln_bwd<FNT>(p.XM, p.dH2, p.g2, p.mean2, p.rstd2, p.dXM, true, p.dg2p + (size_t)b * d, p.db2np + (size_t)b * d, q0r, q0r + rr, d);
    copy_rows_rounded(p.dXMg, p.dXM, q0r, q0r + rr, d);                              // PRE: the attention's GEMM copy of its dY
  }
  ph(ML_LN2B);
  if (!n(p.count, G, p.abort)) return false;
  ph(ML_X4);
  return true;
}

// The MLP half's dW sums: each dW and db element's Gt row-group partials, and LN2's Gt Gc member partials, summed in order by their owners
// over flat shares of all G members.
__device__ __forceinline__ void mlp_sums(const MDesc& p, int G, int b) {
  const int d = p.d, F = p.F, Gm = p.gt * p.gc;
  const long long i0 = (long long)b * FNT + threadIdx.x, step = (long long)G * FNT, n1 = (long long)d * F;
  sum_parts4(p.dW1p, (size_t)n1, p.gt, p.dW1, n1 / 4, i0, step);
  sum_parts4(p.dW2p, (size_t)n1, p.gt, p.dW2, n1 / 4, i0, step);
  sum_parts4(p.db1p, (size_t)F, p.gt, p.db1, F / 4, i0, step);
  sum_parts4(p.db2p, (size_t)d, p.gt, p.db2, d / 4, i0, step);
  sum_parts4(p.dg2p, (size_t)d, Gm, p.dg2, d / 4, i0, step);
  sum_parts4(p.db2np, (size_t)d, Gm, p.db2n, d / 4, i0, step);
}

enum SPh { S_INPUT = K_N, S_X0, S_LNF, S_X_HF, S_LOGITS, S_X_CE, S_CE, S_HEADB, S_X_DHF, S_LNFB, S_X_DXL, S_EMBB, S_X_EMBB, S_SUMS,
           S_X_SUMS, S_SQ, S_X_NORM, S_ADAMW, S_X_END, S_N };

// ---- the step's own phases ------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void own_rows(int N, int G, int b, int& r0, int& r1) { r0 = (int)((long long)b * N / G); r1 = (int)((long long)(b + 1) * N / G); }

// The head's logits over shard b's columns, a row chunk at a time (its columns staged with the first chunk and kept), then each row's
// softmax statistics over them (max, sum of exponentials) and each targeted row's logit when the target is the shard's.
__device__ void head_logits(const SDesc& s, float* smem, int b, long long tb) {
  const int N = s.N, D = s.d, V = s.V, v0 = b * s.vp, w = min(V, v0 + s.vp) - v0, ldd = kvm_fgpt_ldk(D), ldw = kvm_fgpt_ldmn(w), tid = threadIdx.x;
  float* const HF = smem;
  float* const WH = HF + kvm_fgpt_r4((long long)kvm_fgpt_r16(SRC) * ldd);
  for (int c0 = 0; c0 < N; c0 += SRC) {
    const int rc = min(SRC, N - c0), rc16 = kvm_fgpt_r16(rc);
    const Copy rows = {HF, s.hf + (size_t)c0 * D, ldd, (size_t)D, rc, D, rc16, kvm_fgpt_r8(D)};
    if (c0 == 0) { const Copy cs[2] = {rows, {WH, s.head + v0, ldw, (size_t)V, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}}; stage_all<FNT>(cs); }
    else { const Copy cs[1] = {rows}; stage_all<FNT>(cs); }
    gemm<false, false, FNT>(HF, ldd, WH, ldw, rc, w, D, kPrec, [&](int m, int nn, float a) { s.logits[(size_t)(c0 + m) * V + v0 + nn] = a; });
    __syncthreads();
  }
  for (int r = tid; r < N; r += FNT) {
    const float* const lr = s.logits + (size_t)r * V;
    float mx = -INFINITY;
    for (int c = v0; c < v0 + w; ++c) mx = fmaxf(mx, ld(lr + c));
    float se = 0.f;
    for (int c = v0; c < v0 + w; ++c) se += expf(ld(lr + c) - mx);
    s.ce[((size_t)b * N + r) * 2] = mx;
    s.ce[((size_t)b * N + r) * 2 + 1] = se;
    const int tg = s.tgt[tb + r];
    if (tg >= v0 && tg < v0 + w) s.tl[r] = ld(lr + tg);
  }
}

// Every row's log-sum-exp from the shards' statistics, by each shard's member into its own copy (its gradient reads it, so no crossing
// waits for one member's); member 0 also the loss, as softmax_ce_reduce writes it: the mean over targeted rows (summed in float64), then
// 1 / their count.
__device__ void head_loss(const SDesc& s, int b, long long tb, int step, double* red) {
  const int N = s.N, tid = threadIdx.x;
  float* const lse = s.lse + (size_t)b * N;
  for (int r = tid; r < N; r += FNT) {
    float M = -INFINITY;
    for (int j = 0; j < s.nv; ++j) M = fmaxf(M, ld(s.ce + ((size_t)j * N + r) * 2));
    float S = 0.f;
    for (int j = 0; j < s.nv; ++j) S += ld(s.ce + ((size_t)j * N + r) * 2 + 1) * expf(ld(s.ce + ((size_t)j * N + r) * 2) - M);
    lse[r] = M + logf(S);
  }
  if (b == 0) {
    __syncthreads();
    double sum = 0.0, cnt = 0.0;
    for (int r = tid; r < N; r += FNT)
      if (s.tgt[tb + r] != -1) { sum += (double)(lse[r] - ld(s.tl + r)); cnt += 1.0; }
    sum = block_sum<double, FNT>(sum, red);
    cnt = block_sum<double, FNT>(cnt, red);
    if (tid == 0) {
      float* const out = s.loss[step];
      out[0] = cnt > 0.0 ? (float)(sum / cnt) : 0.f;
      out[1] = cnt > 0.0 ? (float)(1.0 / cnt) : 0.f;
    }
  }
}

// The head's backward on shard b: DL = (softmax - onehot(target)) / N on targeted rows, a row chunk at a time; the head's gradient
// over the rows (first chunk writes, later chunks add, each element by the thread that wrote it); dhf's partial over the shard's columns.
__device__ void head_bwd(const SDesc& s, float* smem, int b, long long tb) {
  const int N = s.N, D = s.d, V = s.V, v0 = b * s.vp, w = min(V, v0 + s.vp) - v0, ldl = kvm_fgpt_ldk(w), tid = threadIdx.x;
  float* const DL = smem;
  float* const HF = DL + kvm_fgpt_r4((long long)kvm_fgpt_r16(SRC) * ldl);
  float* const WT = HF + kvm_fgpt_r4((long long)kvm_fgpt_r16(SRC) * kvm_fgpt_ldmn(D));
  float* const part = s.dhfp + (size_t)b * N * D;
  for (int c0 = 0; c0 < N; c0 += SRC) {
    const int rc = min(SRC, N - c0), rc16 = kvm_fgpt_r16(rc);
    const bool add = c0 > 0;
    zero<FNT>(DL, ldl, rc16, kvm_fgpt_r8(w));
    __syncthreads();
    const float* const lse = s.lse + (size_t)b * N;                       // the shard's own copy (head_loss)
    for (int e = tid; e < rc * w; e += FNT) {
      const int i = e / w, r = c0 + i, c = e % w, tg = s.tgt[tb + r];
      DL[i * ldl + c] = tg == -1 ? 0.f : (expf(ld(s.logits + (size_t)r * V + v0 + c) - lse[r]) - (v0 + c == tg ? 1.f : 0.f)) * s.inv_n;
    }
    const Copy rows = {HF, s.hf + (size_t)c0 * D, kvm_fgpt_ldmn(D), (size_t)D, rc, D, rc16, D};
    if (c0 == 0) { const Copy cs[2] = {rows, {WT, s.head + v0, ldl, (size_t)V, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}}; stage_all<FNT>(cs); }
    else { const Copy cs[1] = {rows}; stage_all<FNT>(cs); }
    gemm<true, false, FNT>(HF, kvm_fgpt_ldmn(D), DL, ldl, D, w, rc, kPrec, [&](int m, int nn, float a) {
      float* const o = s.ghead + (size_t)m * V + v0 + nn; *o = add ? ld(o) + a : a; });
    gemm<false, true, FNT>(DL, ldl, WT, ldl, rc, D, w, kPrec, [&](int m, int nn, float a) { part[(size_t)(c0 + m) * D + nn] = a; });
    __syncthreads();
  }
}

// The embedding's gradient on shard b's vocabulary rows: each element sums the gradients of its token's rows, in row order, from dx[0]
// staged a row chunk at a time (a later chunk resumes each running sum).
__device__ void emb_bwd(const SDesc& s, float* smem, int b, long long ib) {
  const int N = s.N, D = s.d, v0 = b * s.vp, w = min(s.V, v0 + s.vp) - v0, tid = threadIdx.x;
  float* const GXS = smem;
  const float* const gx = s.att[0].dX;
  for (int c0 = 0; c0 < N; c0 += SRC) {
    const int rc = min(SRC, N - c0);
    const Copy cs[1] = {{GXS, gx + (size_t)c0 * D, D, (size_t)D, rc, D, rc, D}};
    stage_all<FNT>(cs);
    for (int e = tid; e < w * D; e += FNT) {
      const int row = v0 + e / D, c = e % D;
      float* const o = s.gwte + (size_t)row * D + c;
      float a = c0 == 0 ? 0.f : ld(o);
      for (int r = 0; r < rc; ++r) if (s.ids[ib + c0 + r] == row) a += GXS[r * D + c];
      *o = a;
    }
    __syncthreads();
  }
}

// ---- the program -----------------------------------------------------------------------------------------------------------------------
// Every step of the run as member blockIdx.x of a gang of s.G blocks of FNT threads (the launch's grid must be the gang). `smem` is the
// member's dynamic shared memory, `smem_floats` floats: less than s.smem_floats fails the run (KVM_FGPT_E_SCRATCH) before any phase, as a
// grid other than the gang does (KVM_FGPT_E_SHAPE). A member that finds a fault reports it once, from thread 0, through `fail(code)` (a
// KvmFgptError) and returns; the caller's report must set the run's error word (s.abort; the executor's kvm_fail does), so every other
// member leaves at its next crossing. `ph(k)` is called at the end of each phase k (APh, MPh, SPh) after the block barrier every phase mark
// is (the program was verified with them), for a caller's phase clock; kvm_fgpt::NoPhases runs none. Returns false when the run failed.
// Every thread of the block calls it.
template <bool Counted, class FAIL, class PH>
__device__ bool run(const SDesc& s, float* smem, int smem_floats, FAIL fail, PH& ph) {
  const int G = gridDim.x, b = blockIdx.x, tid = threadIdx.x, N = s.N, D = s.d, L = s.L;
  auto mark = [&](int k) { __syncthreads(); ph(k); };
  if (G != s.G || smem_floats < s.smem_floats) {
    if (tid == 0) fail(G != s.G ? KVM_FGPT_E_SHAPE : KVM_FGPT_E_SCRATCH);
    return false;
  }
  Cross<Counted> n(s.count, b, G);                                          // this member's place in the gang's crossings
  __shared__ double red[FNW];
  __shared__ int bad;                                                        // a step's data fault, the block's
  // the run's step counter (the bundle's), and every step's AdamW bias corrections from it, as the fused step evaluates them (the float of
  // the double pow, subtracted from 1 in float); the first crossing orders them before any member's first AdamW
  const int t0 = __ldcg(s.step);
  if (b == 0)
    for (int k = tid; k < 2 * s.steps; k += FNT)
      s.bc[k] = __fsub_rn(1.f, (float)pow((double)((k & 1) ? s.beta2 : s.beta1), (double)((long long)t0 + (k >> 1) + 1)));
  if (b == 0) rope_table<FNT>(s.rtab, s.T, s.att[0].hd, s.theta);          // ordered before any item by the embedding's crossing
  for (int a = 0; a < s.na; ++a) {                                          // each GEMM weight's rounded shadow, over flat shares
    const SArr& A = s.arr[a];
    if (A.r) for (long long i = (long long)b * FNT + tid; i < A.n; i += (long long)G * FNT) A.r[i] = tfr(A.p[i]);
  }
  int r0, r1;
  own_rows(N, G, b, r0, r1);
  for (int step = 0; step < s.steps; ++step) {
    const long long t = (long long)t0 + step, ib = t * s.ids_stride + s.ids_off, tb = t * s.tgt_stride + s.tgt_off;
    // ---- the step's tokens and targets (the bundle's data windows), checked as the bundle's ops check them, with the fused step's codes
    if (tid == 0) bad = 0;
    __syncthreads();
    if (t + 1 < 1) { if (tid == 0) bad = KVM_FGPT_E_STEP; }
    else if (ib < 0 || ib + N > s.ids_len || tb < 0 || tb + N > s.tgt_len) { if (tid == 0) bad = KVM_FGPT_E_WINDOW; }
    else
      for (int r = tid; r < N; r += FNT) {
        const int id = __ldcg(s.ids + ib + r), tg = __ldcg(s.tgt + tb + r);
        if (id < 0 || id >= s.V) atomicCAS(&bad, 0, (int)KVM_FGPT_E_TOKEN);
        if (tg != -1 && (tg < 0 || tg >= s.V)) atomicCAS(&bad, 0, (int)KVM_FGPT_E_TARGET);
      }
    __syncthreads();
    if (bad) { if (tid == 0) fail(bad); return false; }
    // ---- forward: the embedding of own rows
    for (int e = tid; e < (r1 - r0) * D; e += FNT) {
      const int r = r0 + e / D, c = e % D;
      s.x0[(size_t)r * D + c] = ld(s.wte + (size_t)s.ids[ib + r] * D + c);
    }
    mark(S_INPUT);
    if (!n(s.count, G, s.abort)) return false;
    mark(S_X0);
    for (int l = 0; l < L; ++l) {
      if (!attn_fwd(s.att[l], smem, G, b, n, mark)) return false;
      if (!mlp_fwd(s.mlp[l], smem, G, b, n, mark)) return false;
    }
    // ---- the final LayerNorm on own rows, the head on the vocabulary shards, the loss
    ln_fwd<FNT>(s.xL, s.nfg, s.nfb, s.hf, s.mf, s.rf, r0, r1, D, s.ln_eps);
    mark(S_LNF);
    if (!n(s.count, G, s.abort)) return false;
    mark(S_X_HF);
    if (b < s.nv) head_logits(s, smem, b, tb);
    mark(S_LOGITS);
    if (!n(s.count, G, s.abort)) return false;
    mark(S_X_CE);
    if (b < s.nv) head_loss(s, b, tb, step, red);
    mark(S_CE);
    // ---- backward: the head on its shards, then the final LayerNorm's on own rows (dhf's shard partials summed in shard order)
    if (b < s.nv) head_bwd(s, smem, b, tb);
    mark(S_HEADB);
    if (!n(s.count, G, s.abort)) return false;
    mark(S_X_DHF);
    for (int e = tid; e < (r1 - r0) * D / 4; e += FNT) {
      const int i = e / (D / 4), j = (e % (D / 4)) * 4;
      *reinterpret_cast<float4*>(s.dyf + (size_t)(r0 + i) * D + j) = sum_in_order4(s.dhfp + (size_t)(r0 + i) * D + j, (size_t)N * D, s.nv);
    }
    __syncthreads();
    ln_bwd<FNT>(s.xL, s.dyf, s.nfg, s.mf, s.rf, s.dxL, false, s.lnfp + (size_t)b * 2 * D, s.lnfp + (size_t)b * 2 * D + D, r0, r1, D);
    copy_rows_rounded(s.dxLg, s.dxL, r0, r1, D);                            // the top MLP's GEMM copy of its dxo
    mark(S_LNFB);
    if (!n(s.count, G, s.abort)) return false;
    mark(S_X_DXL);
    for (int l = L - 1; l >= 0; --l) {
      if (!mlp_bwd(s.mlp[l], smem, G, b, n, mark)) return false;
      if (!attn_bwd(s.att[l], smem, G, b, n, mark)) return false;
    }
    // ---- the embedding's gradient on the vocabulary shards (dx[0] was gathered by the attention's last crossing)
    if (b < s.nv) emb_bwd(s, smem, b, ib);
    mark(S_EMBB);
    // ---- every layer's dW partials and the final LayerNorm's member partials, summed by owners over flat shares
    {
      const long long i0 = (long long)b * FNT + tid, st = (long long)G * FNT;
      for (int l = 0; l < L; ++l) { attn_sums(s.att[l], G, b); mlp_sums(s.mlp[l], G, b); }
      sum_parts4(s.lnfp, (size_t)2 * D, G, s.gnfg, D / 4, i0, st);
      sum_parts4(s.lnfp + D, (size_t)2 * D, G, s.gnfb, D / 4, i0, st);
    }
    mark(S_SUMS);
    if (!n(s.count, G, s.abort)) return false;
    mark(S_X_SUMS);
    // ---- the gradients' squares over flat shares of every array; the member's partial in float64, as the fused step's
    {
      float sq = 0.f;
      for (int a = 0; a < s.na; ++a) {
        const SArr& A = s.arr[a];
        for (long long i = (long long)b * FNT + tid; i < A.n; i += (long long)G * FNT) { const float g = ld(A.g + i); sq += g * g; }
      }
      const double tq = block_sum<double, FNT>((double)sq, red);
      if (tid == 0) s.gn[b] = (float)tq;
    }
    mark(S_SQ);
    if (!n(s.count, G, s.abort)) return false;
    mark(S_X_NORM);
    // ---- the grad norm from the G partials in member order, then AdamW over flat shares in the bundle's float32 operation order
    {
      float norm2 = 0.f;
      for (int m = 0; m < G; ++m) norm2 += ld(s.gn + m);
      const float norm = sqrtf(norm2), coef = s.max_norm > 0.f ? fminf(1.f, s.max_norm / (norm + 1e-6f)) : 1.f;
      const float b1 = s.beta1, b2 = s.beta2, lr = s.lr, eps = s.adam_eps, ob1 = __fsub_rn(1.f, b1), ob2 = __fsub_rn(1.f, b2);
      const float bc1 = ld(s.bc + 2 * step), bc2 = ld(s.bc + 2 * step + 1);
      for (int a = 0; a < s.na; ++a) {
        const SArr& A = s.arr[a];
        const float wd = A.wd;
        for (long long i = (long long)b * FNT + tid; i < A.n; i += (long long)G * FNT) {
          const float gi = __fmul_rn(ld(A.g + i), coef);
          const float mi = __fadd_rn(__fmul_rn(b1, A.m[i]), __fmul_rn(ob1, gi));
          const float vi = __fadd_rn(__fmul_rn(b2, A.v[i]), __fmul_rn(__fmul_rn(ob2, gi), gi));
          const float p0 = A.p[i], decay = __fmul_rn(__fmul_rn(lr, wd), p0);
          const float upd = __fdiv_rn(__fmul_rn(lr, __fdiv_rn(mi, bc1)), __fadd_rn(__fsqrt_rn(__fdiv_rn(vi, bc2)), eps));
          const float pn = __fsub_rn(__fsub_rn(p0, decay), upd);
          A.m[i] = mi; A.v[i] = vi; A.p[i] = pn;
          if (A.r) A.r[i] = tfr(pn);                                           // the GEMMs' shadow of the new weight
        }
      }
    }
    mark(S_ADAMW);
    if (!n(s.count, G, s.abort)) return false;                              // the step's end: every update visible
    mark(S_X_END);
  }
  if (b == 0 && tid == 0) *s.step = t0 + s.steps;                          // the run's counter, advanced by its steps
  return true;
}

}  // namespace kvm_f2d
