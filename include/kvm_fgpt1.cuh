// The fused GPT step on one block, its state on chip (kvm_fgpt1).
//
// kvm_fgpt::run spreads a GPT training step over a gang of G blocks. On a model as small as gpt-tiny (L 1, d 32, 32 rows) its step is
// latency: about 2.4 us a phase whatever the phase does, because each phase reads its inputs back from global memory and stages its
// weights again, and about 1 us a gang crossing through L2 even with one member (measured on an A100). This form runs the whole step in one
// block: every parameter resident in shared memory for the whole run (written back at its end), every activation on chip for the step,
// and phases separated by __syncthreads alone. AdamW's moments and the gradients stay in global memory (L2), read and written once a step.
//
// Its arithmetic is kvm_fgpt::run's at G 1, bit for bit: the same helpers (gemm, rope, gelu) and the same expressions in the same orders
// (LayerNorm a warp a row, the softmax a warp a row with lanes strided, the CE a thread a row in column order, the projection's per-head
// partials summed in head order, then the bias, then the residual). Attention is batched over heads and rows instead of run item by item:
// a masked score is an exact zero and the softmax never touches it, so a row's P V over all T keys adds only exact zeros to its prefix's sum.
//
// This file holds C2: the layout and the forward with the loss. The backward and the update are C3.
//
// C2r (Nov 13): relaxed numerics, each change one bit of KVM_FGPT1_FAST, each judged against float64 on its own line beside G 16's:
//   1  the attention softmax four lanes a row: two shuffle levels, __expf, one reciprocal a row, __logf;
//   2  the cross-entropy's row statistics eight lanes a row: three shuffle levels, __expf, __logf;
//   4  the LayerNorms eight lanes a row: three shuffle levels, the mean and variance through one reciprocal of d, rsqrtf; the residual
//      sums that feed LN2, the next layer's LN1 and the final norm run in the norm's own pass, each lane its own columns (their elements
//      are run's expressions, so their bits are run's);
//   8  GELU's tanh as 1 - 2 / (exp(2y) + 1), by __expf and __fdividef;
//  16  AdamW's update without IEEE division or square root (C3): the two bias corrections as reciprocals once a step, sqrt(v^) as
//      v^ rsqrtf(v^), one __fdividef. The moments keep run's float ops, so their bits; only the parameters' update moves.
// 0 keeps kvm_fgpt::run's bits at G 1, the oracle for every change meant to keep them. The structural changes beside them keep the bits
// in every form: the next step's tokens prefetched during this step, the head resident for the run, the loss's two sums in one pass.
#pragma once
#include <new>
#include "kvm_fused_gpt.cuh"
#include "kvm_fgpt1.h"

#ifndef KVM_FGPT1_FAST
#define KVM_FGPT1_FAST 0
#endif

namespace kvm_fgpt1 {
using namespace kvm_fgpt;
constexpr int kFast = KVM_FGPT1_FAST;
#ifndef KVM_FGPT1_C3
#define KVM_FGPT1_C3 0   // 1: each step also runs C3's backward and AdamW (relaxed); 0: C2's forward and loss only
#endif
constexpr bool kC3 = KVM_FGPT1_C3 != 0;
#ifndef KVM_FGPT1_ADAM_RING
#define KVM_FGPT1_ADAM_RING 4   // AdamW's ring slots a thread (adamw): gpt-tiny's activations' span holds 7, and 4 read the same
                                // (119.47 kcycles a step against 7's 119.60) in 12,288 of its 22,176 floats
#endif
constexpr int kAdamRing = KVM_FGPT1_ADAM_RING;
static_assert(kAdamRing >= 1 && kAdamRing <= 7, "KVM_FGPT1_ADAM_RING: 1 to 7 ring slots a thread");
static_assert((kFast & ~31) == 0, "KVM_FGPT1_FAST: a mask of 1 (softmax), 2 (CE), 4 (LayerNorm), 8 (GELU), 16 (AdamW's update)");
constexpr int kRowMax = KVM_FGPT1_ROW_MAX;

using Layout = KvmFgpt1Layout;   // kvm_fgpt1.h: the layout, host and device
// The form's GEMMs: whole tiles always (kvm_fgpt1_check_layout requires them), so their epilogues take no bounds branches (two
// epilogue edits of a few instructions an element moved 1.2 us each, the cost of elements a lane runs one at a time between branches).
// RND (kvm_fused_gpt.cuh's mma_frag): bit 1 when A's values were stored rounded to TF32 by their producer, bit 2 for B's. The GEMM-only
// operands are (each written rounded in a 1xTF32 run, exact in a 3xTF32 one): LN1's, LN2's and the final LN's outputs in NH (and their
// recomputations in the backward), GELU's output U, and the head's dL. Each is read by GEMMs alone, so the rounding moves from every use
// to its one store and every product keeps its bits.
template <bool TA, bool TB, int RND = 0, class Epi>
__device__ __forceinline__ GemmJob<TA, TB, Epi, true, RND> fjob(const float* A, int lda, const float* B, int ldb, int M, int N, int K, bool prec, Epi epi) {
  return {A, lda, B, ldb, M, N, K, prec, epi};
}
template <bool TA, bool TB, int RND = 0, class Epi>
__device__ __forceinline__ void fgemm(const float* A, int lda, const float* B, int ldb, int M, int N, int K, bool prec, Epi epi) {
  gemm<TA, TB, NT, true, RND>(A, lda, B, ldb, M, N, K, prec, epi);
}
// This instance's layout: its own mask's logits pitch.
KVM_FGPT_HD inline Layout kvm_fgpt1_layout(int N, int T, int d, int H, int F, int V, int L) {
  return ::kvm_fgpt1_layout(N, T, d, H, F, V, L, kFast);
}

// Two warp sums (max) at once: each value's own butterfly, exactly warp_sum's (warp_max's), the two trees interleaved.
__device__ __forceinline__ void warp_sum2(float& a, float& b) {
  for (int o = 16; o; o >>= 1) { const float u = __shfl_xor_sync(0xffffffffu, a, o), v = __shfl_xor_sync(0xffffffffu, b, o); a += u; b += v; }
}
__device__ __forceinline__ void warp_max2(float& a, float& b) {
  for (int o = 16; o; o >>= 1) { const float u = __shfl_xor_sync(0xffffffffu, a, o), v = __shfl_xor_sync(0xffffffffu, b, o); a = fmaxf(a, u); b = fmaxf(b, v); }
}

// LayerNorm forward of rows [0, N), kvm_fgpt::ln_fwd's arithmetic exactly (a warp a row, lanes strided over columns), with pitched input
// and output so the output is a padded GEMM operand. A warp runs two of its rows at once (row r and r + NW, each its own chain, the same
// expressions and shuffle trees), so their latencies overlap. Ends with __syncthreads.
template <int BT = NT>
__device__ void ln_fwd_p(const float* x, int ldx, const float* g, const float* be, float* y, int ldy, float* mean, float* rstd, int N, int d, float eps,
                         bool rnd = false) {   // rnd: the outputs rounded to TF32 (a 1xTF32 run's GEMM-only operand, fjob's RND)
  constexpr int NWB = BT / 32;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = w; r < N; r += 2 * NWB) {
    const int r2 = r + NWB;
    const bool two = r2 < N;
    const float* xa = x + (size_t)r * ldx;
    const float* xb = x + (size_t)(two ? r2 : r) * ldx;
    float sa = 0.f, sb = 0.f;
    for (int c = lane; c < d; c += 32) { sa += xa[c]; sb += xb[c]; }
    warp_sum2(sa, sb);
    const float mua = sa / (float)d, mub = sb / (float)d;
    float qa = 0.f, qb = 0.f;
    for (int c = lane; c < d; c += 32) { const float ta = xa[c] - mua, tb = xb[c] - mub; qa += ta * ta; qb += tb * tb; }
    warp_sum2(qa, qb);
    const float rsa = 1.f / sqrtf(qa / (float)d + eps), rsb = 1.f / sqrtf(qb / (float)d + eps);
    for (int c = lane; c < d; c += 32) {
      const float va = (xa[c] - mua) * rsa * g[c] + be[c];
      y[(size_t)r * ldy + c] = rnd ? tf32f(va) : va;
      if (two) { const float vb = (xb[c] - mub) * rsb * g[c] + be[c]; y[(size_t)r2 * ldy + c] = rnd ? tf32f(vb) : vb; }
    }
    if (lane == 0) { mean[r] = mua; rstd[r] = rsa; if (two) { mean[r2] = mub; rstd[r2] = rsb; } }
  }
  __syncthreads();
}

// The attention rows' softmax over R = B H T rows of T keys, causal (row i of a block keeps keys 0..i): run's code at G 1, a warp a row,
// two of a warp's rows at once (each its own chain and shuffle trees). S holds the B H blocks of T16 rows at pitch ldS; alse gets each
// row's log-sum-exp. No sync at the end (the caller's).
__device__ __forceinline__ void softmax_rows(float* S0, int ldS, int T, int T16, int R, float* alse) {
  const int tid = threadIdx.x, NW = NT / 32;
  const int wp = tid >> 5, lane = tid & 31;
  for (int rr = wp; rr < R; rr += 2 * NW) {
    const int r2 = rr + NW;
    const bool two = r2 < R;
    const int ia = rr % T, ib = (two ? r2 : rr) % T;
    float* const Sa = S0 + (size_t)(rr / T) * T16 * ldS + (size_t)ia * ldS;
    float* const Sb = S0 + (size_t)((two ? r2 : rr) / T) * T16 * ldS + (size_t)ib * ldS;
    float ma = -INFINITY, mb = -INFINITY;
    for (int j = lane; j <= ia; j += 32) ma = fmaxf(ma, Sa[j]);
    for (int j = lane; j <= ib; j += 32) mb = fmaxf(mb, Sb[j]);
    warp_max2(ma, mb);
    float la = 0.f, lb = 0.f;
    for (int j = lane; j <= ia; j += 32) { const float e = expf(Sa[j] - ma); Sa[j] = e; la += e; }
    if (two) for (int j = lane; j <= ib; j += 32) { const float e = expf(Sb[j] - mb); Sb[j] = e; lb += e; }
    warp_sum2(la, lb);
    for (int j = lane; j <= ia; j += 32) Sa[j] /= la;
    if (two) for (int j = lane; j <= ib; j += 32) Sb[j] /= lb;
    if (lane == 0) {
      alse[(size_t)(rr / T) * T + ia] = ma + logf(la);
      if (two) alse[(size_t)(r2 / T) * T + ib] = mb + logf(lb);
    }
  }
}

// The same softmax with a thread a row: the row's 32-lane partials held in registers and reduced by the warp butterfly's own pairs in its
// own order (level 16, then 8, 4, 2, 1: lane 0's sums, which every lane of warp_sum holds), so its bits are softmax_rows' with no
// shuffle on the chain. T at most 32 here (a lane's partial is one key); the caller checks. No sync at the end.
__device__ __forceinline__ void softmax_rows_t(float* S0, int ldS, int T, int T16, int R, float* alse) {
  for (int rr = threadIdx.x; rr < R; rr += NT) {
    const int i = rr % T;
    float* const Sr = S0 + (size_t)(rr / T) * T16 * ldS + (size_t)i * ldS;
    float v[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) v[l] = l <= i ? Sr[l] : -INFINITY;
    float mx = v[0];
#pragma unroll
    for (int l = 1; l < 32; ++l) mx = fmaxf(mx, v[l]);                        // a max is exact in any order
    float e[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) e[l] = l <= i ? expf(v[l] - mx) : 0.f;
    float s[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) s[l] = e[l];
#pragma unroll
    for (int o = 16; o; o >>= 1)                                               // warp_sum's pairs: lane l adds lane l ^ o
#pragma unroll
      for (int l = 0; l < o; ++l) s[l] = s[l] + s[l + o];
    const float sum = s[0];
#pragma unroll
    for (int l = 0; l < 32; ++l) if (l <= i) Sr[l] = e[l] / sum;
    alse[(size_t)(rr / T) * T + i] = mx + logf(sum);
  }
}

// The cross-entropy's row statistics of N rows of V logits: lse and the target's logit. A warp a row: the max in any order (exact), each
// exponential on a lane, their sum in column order on every lane (run at G 1: one thread walks the row, se += expf(l - mx) for c = 0, 1,
// ...; the same terms in the same order, so the same bits). No sync at the end (the caller's).
__device__ __forceinline__ void ce_rows(const float* logits, int N, int V, const int* tgt, float* lse, float* tl) {
  const int tid = threadIdx.x, NW = NT / 32;
  const int wp = tid >> 5, lane = tid & 31;
  for (int r = wp; r < N; r += NW) {
    const float* const lr = logits + (size_t)r * V;
    float mx = -INFINITY;
    for (int c = lane; c < V; c += 32) mx = fmaxf(mx, lr[c]);
    mx = warp_max(mx);
    float se = 0.f;
    for (int c0 = 0; c0 < V; c0 += 32) {
      const float e = c0 + lane < V ? expf(lr[c0 + lane] - mx) : 0.f;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const float t = __shfl_sync(0xffffffffu, e, j);
        if (c0 + j < V) se += t;
      }
    }
    if (lane == 0) {   // run: M = max over its one shard = mx; S = 0 + se * expf(mx - M) = se exactly; lse = M + logf(S)
      float M = -INFINITY;
      M = fmaxf(M, mx);
      float Ssum = 0.f;
      Ssum += se * expf(mx - M);
      lse[r] = M + logf(Ssum);
      const int tg = tgt[r];
      if (tg >= 0 && tg < V) tl[r] = lr[tg];
    }
  }
}

// The relaxed row phases put every row on a group of lanes of one warp (a power of two, lanes 0..LPR-1 of the group holding the row's slices),
// so a row's chain is a short local loop and log2(LPR) shuffle levels, and the rows spread over most of the block's warps (a
// phase is one warp's dependent chain; a row phase on 1-2 warps cost 2.5-7x its arithmetic). Reads and writes are 16 bytes a lane, laid out
// so 8 consecutive lanes touch 8 disjoint 4-bank groups.
template <int LPR> __device__ __forceinline__ float group_max(float v) {
#pragma unroll
  for (int o = 1; o < LPR; o <<= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
template <int LPR> __device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int o = 1; o < LPR; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// The attention rows' softmax (FAST & 1): four lanes a row, lane j holding keys 8j..8j+7 (the scores' 4 mod 8 pitch keeps two rows' eight
// lanes on disjoint bank groups). __expf, one reciprocal a row, __logf. A row's masked keys stay exact zeros. T at most 32 (the caller
// checks). Whole warps enter the loop, so every shuffle has its partners. No sync at the end.
__device__ __forceinline__ void softmax_rows_fast(float* S0, int ldS, int T, int T16, int R, float* alse) {
  constexpr int LPR = 4, KPL = 8;
  const int U = (LPR * R + 31) & ~31;
  for (int u = threadIdx.x; u < U; u += NT) {
    const int rr = u / LPR, j = u % LPR, k0 = j * KPL;
    const bool on = rr < R;
    const int i = on ? rr % T : 0;
    float* const Sr = S0 + (size_t)(on ? rr / T : 0) * T16 * ldS + (size_t)i * ldS + k0;
    float v[KPL];
#pragma unroll
    for (int l = 0; l < KPL; l += 4) {
      const float4 q = on && k0 + l < T16 ? *reinterpret_cast<const float4*>(Sr + l) : make_float4(0.f, 0.f, 0.f, 0.f);
      v[l] = q.x; v[l + 1] = q.y; v[l + 2] = q.z; v[l + 3] = q.w;
    }
    float mx = -INFINITY;
#pragma unroll
    for (int l = 0; l < KPL; ++l) if (on && k0 + l <= i) mx = fmaxf(mx, v[l]);
    mx = group_max<LPR>(mx);
    float se = 0.f;
#pragma unroll
    for (int l = 0; l < KPL; ++l) { v[l] = on && k0 + l <= i ? __expf(v[l] - mx) : 0.f; se += v[l]; }
    se = group_sum<LPR>(se);
    if (on) {
      const float inv = __frcp_rn(se);
#pragma unroll
      for (int l = 0; l < KPL; l += 4)
        if (k0 + l < T16) *reinterpret_cast<float4*>(Sr + l) = make_float4(v[l] * inv, v[l + 1] * inv, v[l + 2] * inv, v[l + 3] * inv);
      if (j == 0) alse[(size_t)(rr / T) * T + i] = mx + __logf(se);
    }
  }
}

// The cross-entropy's row statistics (FAST & 2): eight lanes a row, lane j holding the 16-byte groups j, j + 8, ... of the row (at the
// 4 mod 8 pitch ldV a row's eight lanes read disjoint bank groups). __expf, __logf. V at most 64 and a multiple of 4 (the caller checks).
// Whole warps enter the loop. No sync at the end.
__device__ __forceinline__ void ce_rows_fast(const float* logits, int ldV, int N, int V, const int* tgt, float* lse, float* tl) {
  constexpr int LPR = 8, G4 = 2;   // two 16-byte groups a lane: 64 logits a row at most
  const int U = (LPR * N + 31) & ~31;
  for (int u = threadIdx.x; u < U; u += NT) {
    const int r = u / LPR, j = u % LPR;
    const bool on = r < N;
    const float* const lr = logits + (size_t)(on ? r : 0) * ldV;
    float v[4 * G4];
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < G4; ++k) {
      const int c = 4 * (j + LPR * k);
      const bool in = on && c < V;
      const float4 q = in ? *reinterpret_cast<const float4*>(lr + c) : make_float4(0.f, 0.f, 0.f, 0.f);
      v[4 * k] = q.x; v[4 * k + 1] = q.y; v[4 * k + 2] = q.z; v[4 * k + 3] = q.w;
      if (in) mx = fmaxf(mx, fmaxf(fmaxf(q.x, q.y), fmaxf(q.z, q.w)));
    }
    mx = group_max<LPR>(mx);
    float se = 0.f;
#pragma unroll
    for (int k = 0; k < G4; ++k)
      if (on && 4 * (j + LPR * k) < V) { se += __expf(v[4 * k] - mx); se += __expf(v[4 * k + 1] - mx); se += __expf(v[4 * k + 2] - mx); se += __expf(v[4 * k + 3] - mx); }
    se = group_sum<LPR>(se);
    if (on && j == 0) {
      lse[r] = mx + __logf(se);
      const int tg = tgt[r];
      if (tg >= 0 && tg < V) tl[r] = lr[tg];
    }
  }
}

// LayerNorm forward, relaxed (FAST & 4): eight lanes a row, lane j holding the row's 16-byte groups j, j + 8, ... (an unpadded 32-wide row's
// eight lanes read disjoint bank groups); sums through three shuffle levels, the mean and variance through one reciprocal of d, rstd by
// rsqrtf. xv4(r, c) gives the four input values at columns c..c+3 (c a multiple of 4); a fused residual sum computes them from run's own
// expressions, element for element, and stores them there, each lane its own groups. xv4 must capture by value. d a multiple of 4, at
// most 128 (the caller checks). Whole warps enter the loop. Ends with __syncthreads.
template <class XV4>
__device__ void ln_rows_fast(XV4 xv4, const float* g, const float* be, float* y, int ldy, float* mean, float* rstd, int N, int d, float eps,
                             bool rnd = false) {   // rnd: the outputs rounded to TF32 (a 1xTF32 run's GEMM-only operand, fjob's RND)
  constexpr int LPR = 8, G4 = KVM_FGPT1_MAXD / 32;   // 16-byte groups a lane, LPR x 4 x G4 = KVM_FGPT1_MAXD columns at most
  const float inv_d = 1.f / (float)d;
  const int U = (LPR * N + 31) & ~31;
  for (int u = threadIdx.x; u < U; u += NT) {
    const int r = u / LPR, j = u % LPR;
    const bool on = r < N;
    float4 x[G4];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < G4; ++k) {
      const int c = 4 * (j + LPR * k);
      x[k] = on && c < d ? xv4(r, c) : make_float4(0.f, 0.f, 0.f, 0.f);
      s += (x[k].x + x[k].y) + (x[k].z + x[k].w);
    }
    const float mu = group_sum<LPR>(s) * inv_d;
    float q = 0.f;
#pragma unroll
    for (int k = 0; k < G4; ++k)
      if (on && 4 * (j + LPR * k) < d) {
        const float a = x[k].x - mu, b = x[k].y - mu, c2 = x[k].z - mu, e = x[k].w - mu;
        q += (a * a + b * b) + (c2 * c2 + e * e);
      }
    const float rs = rsqrtf(group_sum<LPR>(q) * inv_d + eps);
    if (on) {
#pragma unroll
      for (int k = 0; k < G4; ++k) {
        const int c = 4 * (j + LPR * k);
        if (c < d) {
          const float4 gg = *reinterpret_cast<const float4*>(g + c), bb = *reinterpret_cast<const float4*>(be + c);
          float4 o = make_float4((x[k].x - mu) * rs * gg.x + bb.x, (x[k].y - mu) * rs * gg.y + bb.y, (x[k].z - mu) * rs * gg.z + bb.z, (x[k].w - mu) * rs * gg.w + bb.w);
          if (rnd) o = make_float4(tf32f(o.x), tf32f(o.y), tf32f(o.z), tf32f(o.w));
          *reinterpret_cast<float4*>(y + (size_t)r * ldy + c) = o;
        }
      }
      if (j == 0) { mean[r] = mu; rstd[r] = rs; }
    }
  }
  __syncthreads();
}

// GELU with tanh as 1 - 2 / (exp(2y) + 1) (FAST & 8): exp overflowing to infinity gives 1, underflowing to 0 gives -1.
__device__ __forceinline__ float gelu_fast(float x) {
  const float y = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + (1.f - __fdividef(2.f, __expf(2.f * y) + 1.f)));
}
// kvm_fused_gpt.cuh's gelu and gelu_grad (the bundle's tanh form) from one tanh: their own expressions, so their bits. The forward's
// last layer stores gelu'(z) beside GELU's output, and the MLP backward's epilogue is then one product, not a tanhf an element (before it,
// db2, dW2 | dU read 10.0 us against dW1 | dh2's 5.3 at the same GEMM work).
__device__ __forceinline__ float gelu_tanh(float x) { return tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)); }
__device__ __forceinline__ float gelu_of(float x, float t) { return 0.5f * x * (1.f + t); }
__device__ __forceinline__ float gelu_grad_of(float x, float t) {
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * 0.7978845608028654f * (1.f + 3.f * 0.044715f * x * x);
}

// Two values' block sums in one pass: each value's own butterfly and its warps in order, exactly block_sum's, with one pair of barriers.
// `red` holds 2 NT / 32 values of the type.
template <class Tv>
__device__ void block_sum2(Tv& a, Tv& b, Tv* red) {
  for (int o = 16; o; o >>= 1) { const Tv u = __shfl_xor_sync(0xffffffffu, a, o), v = __shfl_xor_sync(0xffffffffu, b, o); a += u; b += v; }
  if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = a; red[(NT / 32) + (threadIdx.x >> 5)] = b; }
  __syncthreads();
  Tv s = 0, t = 0;
  for (int i = 0; i < NT / 32; ++i) { s += red[i]; t += red[(NT / 32) + i]; }
  __syncthreads();
  a = s; b = t;
}

// Zero `n` floats from `p` (padding a GEMM operand needs as exact zeros). No sync.
__device__ __forceinline__ void zero_n(float* p, long long n) { for (long long i = threadIdx.x; i < n; i += NT) p[i] = 0.f; }

// Copy a [rows x cols] row-major global matrix into shared memory at pitch ld, rows padded to prow and columns to pcol with zeros. No sync.
__device__ void load_padded(float* dst, int ld, const float* src, int rows, int cols, int prow, int pcol) {
  for (long long e = threadIdx.x; e < (long long)prow * pcol; e += NT) {
    const int r = (int)(e / pcol), c = (int)(e % pcol);
    dst[(size_t)r * ld + c] = r < rows && c < cols ? __ldcg(src + (size_t)r * cols + c) : 0.f;
  }
}
// Write a padded shared-memory matrix back to its global row-major home. No sync.
__device__ void store_back(float* dst, const float* src, int ld, int rows, int cols) {
  for (long long e = threadIdx.x; e < (long long)rows * cols; e += NT) {
    const int r = (int)(e / cols), c = (int)(e % cols);
    dst[(size_t)r * cols + c] = src[(size_t)r * ld + c];
  }
}

// Every parameter into its resident place: the q, k and v weights side by side as one [d x 3d] operand a layer, the rest at their own slots.
__device__ void load_params(const KvmFgptDesc& d, float* sm, const Layout& y) {
  const int D = d.d, F = d.F, V = d.V, L = d.L;
  for (int s = 0; s < kvm_fgpt_nparams(L); ++s) {
    if (y.p[s] < 0) continue;
    const int rows = d.prows[s], cols = d.pcols[s];
    if (y.pld[s] == cols) { for (long long e = threadIdx.x; e < (long long)rows * cols; e += NT) sm[y.p[s] + e] = __ldcg(d.p[s] + e); }
    else load_padded(sm + y.p[s], y.pld[s], d.p[s], rows, cols, kvm_fgpt_r8(rows), y.pld[s]);
  }
  const int ldw = kvm_fgpt_ldmn(3 * D);
  for (int l = 0; l < L; ++l) {
    float* const w = sm + y.wqkv + (size_t)l * y.wqkv_stride;
    float* const bq = sm + y.bqkv + (size_t)l * 3 * D;
    const int sq = kvm_fgpt_slot_layer(l, KVM_FGPT_R_WQ), sk = kvm_fgpt_slot_layer(l, KVM_FGPT_R_WK), sv = kvm_fgpt_slot_layer(l, KVM_FGPT_R_WV);
    for (long long e = threadIdx.x; e < (long long)kvm_fgpt_r8(D) * ldw; e += NT) {
      const int r = (int)(e / ldw), c = (int)(e % ldw);
      float v = 0.f;
      if (r < D && c < 3 * D) v = __ldcg(d.p[c < D ? sq : c < 2 * D ? sk : sv] + (size_t)r * D + c % D);
      w[e] = v;
    }
    for (int c = threadIdx.x; c < 3 * D; c += NT)
      bq[c] = __ldcg(d.p[kvm_fgpt_slot_layer(l, c < D ? KVM_FGPT_R_BQ : c < 2 * D ? KVM_FGPT_R_BK : KVM_FGPT_R_BV)] + c % D);
  }
  (void)F; (void)V;
  __syncthreads();
}

// The forward of one step and its loss, from the step's tokens to d.loss[step]. kvm_fgpt::run's forward at G 1, phase for phase.
template <class PH>
__device__ float forward(const KvmFgptDesc& d, float* sm, const Layout& y, int step, PH& ph) {
  const int tid = threadIdx.x, NW = NT / 32;
  const int N = d.N, T = d.T, B = d.B, D = d.d, H = d.H, hd = d.hd, F = d.F, V = d.V, L = d.L;
  const bool prec = (d.mm_numerics & 16) != 0;
  const int* const ids = reinterpret_cast<const int*>(sm + y.ids) + (step & 1) * N;   // this step's slot (run1 fills it)
  const int* const tgt = reinterpret_cast<const int*>(sm + y.tgt) + (step & 1) * N;
  const float* const rtab = sm + y.rtab;
  auto P = [&](int slot) { return sm + y.p[slot]; };
  auto lay = [&](int l, int role) { return kvm_fgpt_slot_layer(l, role); };
  // ---- the embedding: the step's rows of wte, from global (wte is not resident), as run reads them: a warp a row and lanes over its
  // columns (D at most KVM_FGPT1_MAXD, CJ a lane), every row's loads issued before any store, so the gather costs one trip to L2, not three,
  // and no division
  {
    constexpr int RW = 4, CJ = KVM_FGPT1_MAXD / 32;   // rows a warp at a time: 4 x 12 warps cover gpt-tiny's 32 rows in one pass
    const int w = tid >> 5, lane = tid & 31;
    float* const x0 = sm + y.x[0];
    for (int r0 = w; r0 < N; r0 += RW * NW) {
      float v[RW][CJ];
#pragma unroll
      for (int k = 0; k < RW; ++k) {
        const int r = r0 + k * NW;
        const float* const src = r < N ? d.p[0] + (size_t)ids[r] * D : nullptr;
#pragma unroll
        for (int j = 0; j < CJ; ++j) { const int c = lane + 32 * j; v[k][j] = r < N && c < D ? __ldcg(src + c) : 0.f; }
      }
#pragma unroll
      for (int k = 0; k < RW; ++k) {
        const int r = r0 + k * NW;
#pragma unroll
        for (int j = 0; j < CJ; ++j) { const int c = lane + 32 * j; if (r < N && c < D) x0[r * D + c] = v[k][j]; }
      }
    }
  }
  __syncthreads();
  ph(MK_INPUT);
  for (int l = 0; l < L; ++l) {
    float* const x = sm + y.x[l]; float* const h1 = sm + y.nh; float* const qkv = sm + y.qkv[l];
    // LN1, into NH (the padded A operand; its padding rows stay zero from the run's start). FAST & 4: a thread a row, and from layer 1 on
    // the previous layer's fc2 residual sum (run's expressions, stored into x) in the same pass
    if constexpr ((kFast & 4) != 0) {
      const float* const b2p = l > 0 ? P(lay(l - 1, KVM_FGPT_R_B2)) : nullptr; const float* const xmp = l > 0 ? sm + y.xm[l - 1] : nullptr;
      const float* const pt = sm + y.part;
      ln_rows_fast([=](int r, int c) {
        float4* const xr = reinterpret_cast<float4*>(x + r * D + c);
        if (l == 0) return *xr;
        const float4 pv = *reinterpret_cast<const float4*>(pt + r * D + c), bv = *reinterpret_cast<const float4*>(b2p + c), mv = *reinterpret_cast<const float4*>(xmp + r * D + c);
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);   // run's order, element for element: fc2's partial, the bias, the residual
        a.x += pv.x; a.y += pv.y; a.z += pv.z; a.w += pv.w;
        a.x += bv.x; a.y += bv.y; a.z += bv.z; a.w += bv.w;
        a.x += mv.x; a.y += mv.y; a.z += mv.z; a.w += mv.w;
        *xr = a;
        return a;
      }, P(lay(l, KVM_FGPT_R_N1G)), P(lay(l, KVM_FGPT_R_N1B)), h1, y.ldd, sm + y.st1[l], sm + y.st1[l] + N, N, D, d.ln_eps, !prec);
    } else {
      ln_fwd_p(x, D, P(lay(l, KVM_FGPT_R_N1G)), P(lay(l, KVM_FGPT_R_N1B)), h1, y.ldd, sm + y.st1[l], sm + y.st1[l] + N, N, D, d.ln_eps, !prec);
    }
    ph(MK_LN1);
    // q | k | v of every row and head: one GEMM over the layer's [d x 3d] weights, the bias in the epilogue (run: a + bias[n], per head)
    {
      const float* const bq = sm + y.bqkv + (size_t)l * 3 * D;
      const int ldq = y.ldq;   // epilogues capture plain locals by value: a reference capture puts the local on the stack (L2 here)
      fgemm<false, false, 1>(h1, y.ldd, sm + y.wqkv + (size_t)l * y.wqkv_stride, kvm_fgpt_ldmn(3 * D), N, 3 * D, D, prec,
                         [=](int m, int n, float a) { qkv[m * ldq + n] = a + bq[n]; });
    }
    ph(MK_UF_QKV);
    // RoPE on every sequence's q and k, head by head (run: rope(QKV, ldq, TQ, qb TQ, ...) on the item's q, then its k)
    for (int sq = 0; sq < B; ++sq)
      for (int hh = 0; hh < H; ++hh) {
        rope(qkv + (size_t)sq * T * y.ldq + hh * hd, y.ldq, T, 0, T, hd, rtab, 1.f);
        rope(qkv + (size_t)sq * T * y.ldq + D + hh * hd, y.ldq, T, 0, T, hd, rtab, 1.f);
      }
    ph(MK_UF_ROPE);
    // scores, causal, 3xTF32 as the attention ops: every (sequence, head) block of T x T, two blocks at once on disjoint warps
    const int T16 = kvm_fgpt_r16(T), nblk = B * H;
    auto scores_job = [&](int blk) {
      const int sq = blk / H, hh = blk % H;
      float* const S = sm + y.a + (size_t)blk * T16 * y.ldS;
      const float* const q = qkv + (size_t)sq * T * y.ldq + hh * hd;
      const float* const k = qkv + (size_t)sq * T * y.ldq + D + hh * hd;
      const int ldS = y.ldS; const float scale = d.scale;   // values, not the descriptor or the layout: a closure copies what it names
      return fjob<false, true>(q, y.ldq, k, y.ldq, T, T, hd, true, [=](int m, int n, float a) { S[m * ldS + n] = n <= m ? a * scale : 0.f; });
    };
    for (int blk = 0; blk < nblk; blk += 2) {
      if (blk + 1 < nblk) gemm_par(scores_job(blk), scores_job(blk + 1));
      else gemm_par(scores_job(blk));
    }
    ph(MK_UF_SCORES);
    if constexpr ((kFast & 1) != 0) softmax_rows_fast(sm + y.a, y.ldS, T, T16, B * H * T, sm + y.alse[l]);
    else softmax_rows(sm + y.a, y.ldS, T, T16, B * H * T, sm + y.alse[l]);
    __syncthreads();
    ph(MK_UF_SOFTMAX);
    // P V, every (sequence, head), into o's head columns, two at once
    auto pv_job = [&](int blk) {
      const int sq = blk / H, hh = blk % H;
      const float* const S = sm + y.a + (size_t)blk * T16 * y.ldS;
      const float* const v = qkv + (size_t)sq * T * y.ldq + 2 * D + hh * hd;
      float* const o = sm + y.o[l] + (size_t)sq * T * y.ldd + hh * hd;
      const int ldd = y.ldd;
      return fjob<false, false>(S, y.ldS, v, y.ldq, T, hd, T, true, [=](int m, int n, float a) { o[m * ldd + n] = a; });
    };
    for (int blk = 0; blk < nblk; blk += 2) {
      if (blk + 1 < nblk) gemm_par(pv_job(blk), pv_job(blk + 1));
      else gemm_par(pv_job(blk));
    }
    ph(MK_UF_PV);
    // the projection's per-head partials (run: an item's O times Wo's head rows, K hd), then their sum in head order, the bias, the residual
    auto proj_job = [&](int hh) {
      float* const pt = sm + y.part + (size_t)hh * N * D;
      const int ldwo = y.pld[lay(l, KVM_FGPT_R_WO)];
      return fjob<false, false>(sm + y.o[l] + hh * hd, y.ldd, P(lay(l, KVM_FGPT_R_WO)) + (size_t)hh * hd * ldwo, ldwo, N, D, hd, prec,
                               [=](int m, int n, float a) { pt[(size_t)m * D + n] = a; });
    };
    for (int hh = 0; hh < H; hh += 2) {
      if (hh + 1 < H) gemm_par(proj_job(hh), proj_job(hh + 1));
      else gemm_par(proj_job(hh));
    }
    ph(MK_UF_PROJ);
    if constexpr ((kFast & 4) != 0) {   // the projection's residual sum (run's order: partials by head, the bias, the residual) in LN2's pass
      const float* const bo = P(lay(l, KVM_FGPT_R_BO));
      float* const xm = sm + y.xm[l]; const float* const pt = sm + y.part;
      ln_rows_fast([=](int r, int c) {
        const int e = r * D + c;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);   // run's order, element for element: the partials by head, the bias, the residual
        for (int hh = 0; hh < H; ++hh) { const float4 pv = *reinterpret_cast<const float4*>(pt + (size_t)hh * N * D + e); a.x += pv.x; a.y += pv.y; a.z += pv.z; a.w += pv.w; }
        const float4 bv = *reinterpret_cast<const float4*>(bo + c), xv = *reinterpret_cast<const float4*>(x + e);
        a.x += bv.x; a.y += bv.y; a.z += bv.z; a.w += bv.w;
        a.x += xv.x; a.y += xv.y; a.z += xv.z; a.w += xv.w;
        *reinterpret_cast<float4*>(xm + e) = a;
        return a;
      }, P(lay(l, KVM_FGPT_R_N2G)), P(lay(l, KVM_FGPT_R_N2B)), sm + y.nh, y.ldd, sm + y.st2[l], sm + y.st2[l] + N, N, D, d.ln_eps, !prec);
    } else {
      {
        const float* const bo = P(lay(l, KVM_FGPT_R_BO));
        float* const xm = sm + y.xm[l];
        for (int e = tid; e < N * D; e += NT) {
          const int c = e % D;
          float a = 0.f;
          for (int hh = 0; hh < H; ++hh) a += sm[y.part + (size_t)hh * N * D + e];
          a += bo[c];
          a += x[e];
          xm[e] = a;
        }
      }
      __syncthreads();
      ln_fwd_p(sm + y.xm[l], D, P(lay(l, KVM_FGPT_R_N2G)), P(lay(l, KVM_FGPT_R_N2B)), sm + y.nh, y.ldd, sm + y.st2[l], sm + y.st2[l] + N, N, D, d.ln_eps, !prec);
    }
    ph(MK_LN2);
    // fc1 + b1, GELU; fc2 over all F (run at G 1: one owner of every column), then the bias and the residual (rs_sum of one partial)
    {
      float* const pre = sm + y.pre[l]; float* const u = sm + y.a;   // GELU's output in region A (the scores there are dead)
      const float* const b1 = P(lay(l, KVM_FGPT_R_B1));
      const int ldF = y.ldF;
      const bool kept = kC3 && l == L - 1;   // C3: the last layer's U stays in A for its backward, which needs only gelu'(z): pre keeps that
      const bool rnd = !prec;   // U is fc2's and dW2's operand alone: stored rounded in a 1xTF32 run
      fgemm<false, false, 1>(sm + y.nh, y.ldd, P(lay(l, KVM_FGPT_R_W1)), y.pld[lay(l, KVM_FGPT_R_W1)], N, F, D, prec, [=](int m, int n, float a) {
        const float z = a + b1[n];
        float g;
        if (kept) {
          const float t = gelu_tanh(z);
          pre[(size_t)m * ldF + n] = gelu_grad_of(z, t);
          if constexpr ((kFast & 8) != 0) g = gelu_fast(z); else g = gelu_of(z, t);
        } else {
          pre[(size_t)m * ldF + n] = z;
          if constexpr ((kFast & 8) != 0) g = gelu_fast(z); else g = gelu(z);
        }
        u[m * ldF + n] = rnd ? tf32f(g) : g;
      });
      float* const pt = sm + y.part;
      fgemm<false, false, 1>(u, y.ldF, P(lay(l, KVM_FGPT_R_W2)), y.pld[lay(l, KVM_FGPT_R_W2)], N, D, F, prec, [=](int m, int n, float a) { pt[(size_t)m * D + n] = a; });
    }
    ph(MK_MLP);
    if constexpr ((kFast & 4) == 0) {   // FAST & 4: the next norm's pass sums it
      const float* const b2 = P(lay(l, KVM_FGPT_R_B2));
      const float* const xm = sm + y.xm[l];
      float* const xn = sm + y.x[l + 1];
      for (int e = tid; e < N * D; e += NT) {
        float a = 0.f;
        a += sm[y.part + e];
        a += b2[e % D];
        a += xm[e];
        xn[e] = a;
      }
      __syncthreads();
    }
    ph(MK_RES);
  }
  // ---- the final norm, the logits, the cross-entropy (run at G 1: one vocabulary shard, so lse = mx + logf(se))
  const int nfg = kvm_fgpt_slot_nfg(L), nfb = kvm_fgpt_slot_nfb(L), head = kvm_fgpt_slot_head(L);
  // the head is resident for the run (run1 copies it in once)
  if constexpr ((kFast & 4) != 0) {   // the last layer's fc2 residual sum (run's order: fc2's partial, the bias, the residual) in this pass
    const float* const b2p = P(lay(L - 1, KVM_FGPT_R_B2)); const float* const xmp = sm + y.xm[L - 1]; const float* const pt = sm + y.part;
    float* const xn = sm + y.x[L];
    ln_rows_fast([=](int r, int c) {
      const float4 pv = *reinterpret_cast<const float4*>(pt + r * D + c), bv = *reinterpret_cast<const float4*>(b2p + c), mv = *reinterpret_cast<const float4*>(xmp + r * D + c);
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);   // run's order, element for element: fc2's partial, the bias, the residual
      a.x += pv.x; a.y += pv.y; a.z += pv.z; a.w += pv.w;
      a.x += bv.x; a.y += bv.y; a.z += bv.z; a.w += bv.w;
      a.x += mv.x; a.y += mv.y; a.z += mv.z; a.w += mv.w;
      *reinterpret_cast<float4*>(xn + r * D + c) = a;
      return a;
    }, P(nfg), P(nfb), sm + y.nh, y.ldd, sm + y.stf, sm + y.stf + N, N, D, d.ln_eps, !prec);
  } else {
    ln_fwd_p(sm + y.x[L], D, P(nfg), P(nfb), sm + y.nh, y.ldd, sm + y.stf, sm + y.stf + N, N, D, d.ln_eps, !prec);
  }
  ph(MK_LNF);
  float* const logits = sm + y.part;                                                  // the logits in region PART (fc2's partial is summed)
  const int ldV = y.ldV;
  fgemm<false, false, 1>(sm + y.nh, y.ldd, sm + y.head, y.pld[head], N, V, D, prec, [=](int m, int n, float a) { logits[(size_t)m * ldV + n] = a; });
  float* const lse = sm + y.lse; float* const tl = sm + y.tl;
  if constexpr ((kFast & 2) != 0) ce_rows_fast(logits, ldV, N, V, tgt, lse, tl);
  else ce_rows(logits, N, V, tgt, lse, tl);
  __syncthreads();
  ph(MK_LOGITS);
  {   // the loss, as softmax_ce_reduce writes it: the mean over targeted rows (summed in float64), then 1 / their count
    double* const red = reinterpret_cast<double*>(sm + y.red);
    double sum = 0.0, cnt = 0.0;
    for (int r = tid; r < N; r += NT)
      if (tgt[r] != -1) { sum += (double)(lse[r] - tl[r]); cnt += 1.0; }
    block_sum2(sum, cnt, red);
    if (tid == 0) {
      float* const out = d.loss[step];
      out[0] = cnt > 0.0 ? (float)(sum / cnt) : 0.f;
      out[1] = cnt > 0.0 ? (float)(1.0 / cnt) : 0.f;
    }
    ph(MK_CE);
    return cnt > 0.0 ? (float)(1.0 / cnt) : 0.f;   // every thread holds the block's sums
  }
}

// ---- C3: the backward and AdamW, relaxed (judged against a float64 reference and run_gpt.py's loss gate, not by
// run's bits). Gradients go to the bundle's gradient buffers (d.g, row-major, in L2), so the run leaves the last step's there as run does;
// their squares are summed in one pass after the backward, and AdamW then updates every parameter where it lives (in shared memory, wte in
// global) and its moments in global.

// Column sums of an R x C matrix at pitch ld (any memory), into out (global), four threads a column (rows split in four, one two-level
// shuffle), so a column's chain is R / 4 adds. Ends with __syncthreads.
__device__ void colsum_fast(const float* X, int ld, int R, int C, float* out, bool sync = true) {
  const int U = (4 * C + 31) & ~31;
  for (int u = threadIdx.x; u < U; u += NT) {
    const int c = u >> 2, q = u & 3;
    const bool on = c < C;
    float a = 0.f;
    if (on) for (int r = q; r < R; r += 4) a += X[(size_t)r * ld + c];
    a = group_sum<4>(a);
    if (on && q == 0) out[c] = a;
  }
  if (sync) __syncthreads();   // false: the caller's next pass neither writes what this reads nor reads what it writes
}

// The column sums of q, k and v's gradients in one pass (3d columns of dqkv at pitch ld), each third into its own bias gradient, four
// threads a column. No sync at the end: the caller's next pass shares its barrier.
__device__ void colsum_qkv(const float* X, int ld, int R, int D, float* oq, float* ok, float* ov) {
  const int C = 3 * D, U = (4 * C + 31) & ~31;
  for (int u = threadIdx.x; u < U; u += NT) {
    const int c = u >> 2, q = u & 3;
    const bool on = c < C;
    float a = 0.f;
    if (on) for (int r = q; r < R; r += 4) a += X[(size_t)r * ld + c];
    a = group_sum<4>(a);
    if (on && q == 0) { float* const o = c < D ? oq : c < 2 * D ? ok : ov; o[c < D ? c : c < 2 * D ? c - D : c - 2 * D] = a; }
  }
}

// LayerNorm backward, relaxed: eight lanes a row (ln_rows_fast's layout), dx = rstd (dy g - mean(dy g) - xh mean(dy g xh)) with xh =
// (x - mean) rstd, added to gx when acc, stored otherwise; and the gain's and bias's gradients (sums over the rows of dy xh and of dy) into
// gG and gB, four threads a column on the threads the rows leave free (or after them). x, dy and gx at pitch d. Ends with __syncthreads.
__device__ void ln_bwd_fast(const float* x, const float* dy, const float* g, const float* mean, const float* rstd, float* gx, bool acc,
                            float* gG, float* gB, int N, int d) {
  constexpr int LPR = 8, G4 = KVM_FGPT1_MAXD / 32;   // as ln_rows_fast's
  const float inv_d = 1.f / (float)d;
  const int U = (LPR * N + 31) & ~31, UC = (4 * d + 31) & ~31;
  const int c0 = NT - U >= UC ? U : 0;                                                 // the column sums' first thread
  for (int u = threadIdx.x; u < U; u += NT) {
    const int r = u / LPR, j = u % LPR;
    const bool on = r < N;
    const float mu = on ? mean[r] : 0.f, rs = on ? rstd[r] : 0.f;
    float4 gs[G4], xh[G4];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int k = 0; k < G4; ++k) {
      const int c = 4 * (j + LPR * k);
      if (on && c < d) {
        const float4 xv = *reinterpret_cast<const float4*>(x + (size_t)r * d + c), dv = *reinterpret_cast<const float4*>(dy + (size_t)r * d + c),
                     gv = *reinterpret_cast<const float4*>(g + c);
        gs[k] = make_float4(dv.x * gv.x, dv.y * gv.y, dv.z * gv.z, dv.w * gv.w);
        xh[k] = make_float4((xv.x - mu) * rs, (xv.y - mu) * rs, (xv.z - mu) * rs, (xv.w - mu) * rs);
        s1 += (gs[k].x + gs[k].y) + (gs[k].z + gs[k].w);
        s2 += (gs[k].x * xh[k].x + gs[k].y * xh[k].y) + (gs[k].z * xh[k].z + gs[k].w * xh[k].w);
      }
    }
    s1 = group_sum<LPR>(s1) * inv_d;
    s2 = group_sum<LPR>(s2) * inv_d;
#pragma unroll
    for (int k = 0; k < G4; ++k) {
      const int c = 4 * (j + LPR * k);
      if (on && c < d) {
        float4* const o = reinterpret_cast<float4*>(gx + (size_t)r * d + c);
        const float4 v = make_float4(rs * (gs[k].x - s1 - xh[k].x * s2), rs * (gs[k].y - s1 - xh[k].y * s2), rs * (gs[k].z - s1 - xh[k].z * s2),
                                     rs * (gs[k].w - s1 - xh[k].w * s2));
        if (acc) { const float4 w = *o; *o = make_float4(w.x + v.x, w.y + v.y, w.z + v.z, w.w + v.w); } else *o = v;
      }
    }
  }
  for (int u = (int)threadIdx.x - c0; u >= 0 && u < UC; u += NT) {                    // dgamma, dbeta: four threads a column
    const int c = u >> 2, q = u & 3;
    const bool on = c < d;
    float sg = 0.f, sb = 0.f;
    if (on)
      for (int r = q; r < N; r += 4) { const float dv = dy[(size_t)r * d + c]; sg += dv * (x[(size_t)r * d + c] - mean[r]) * rstd[r]; sb += dv; }
    sg = group_sum<4>(sg); sb = group_sum<4>(sb);
    if (on && q == 0) { gG[c] = sg; gB[c] = sb; }
  }
  __syncthreads();
}

// Where each parameter slot lives for the run: its first element and row pitch (shared memory, but wte in global; q, k and v's weights
// and biases inside the layer's joint q | k | v buffers). Filled once a run by thread 0.
struct ParamHome {
  float* base[KVM_FGPT_MAXP]; int pitch[KVM_FGPT_MAXP];
  float* g[KVM_FGPT_MAXP]; float* m[KVM_FGPT_MAXP]; float* v[KVM_FGPT_MAXP];   // the descriptor's tables, indexed here in shared memory
  int cols[KVM_FGPT_MAXP]; float inv_cols[KVM_FGPT_MAXP]; float wd[KVM_FGPT_MAXP]; float lrwd[KVM_FGPT_MAXP];   // lr x wd: run's first product
  bool vec4;   // every slot's size, columns and pitch a multiple of 4, every home 16-byte aligned: the update goes float4 a lane
  int off[KVM_FGPT_MAXP + 1];   // one flat index over every parameter element: slot s holds [off[s], off[s + 1])
};
__device__ void param_homes(const KvmFgptDesc& d, float* sm, const Layout& y, ParamHome& h) {
  const int D = d.d, L = d.L;
  h.off[0] = 0;
  for (int s = 0; s < kvm_fgpt_nparams(L); ++s) {
    h.base[s] = y.p[s] >= 0 ? sm + y.p[s] : nullptr; h.pitch[s] = y.pld[s];
    h.g[s] = d.g[s]; h.m[s] = d.m[s]; h.v[s] = d.v[s]; h.wd[s] = d.wd[s];
    h.cols[s] = d.pcols[s]; h.inv_cols[s] = __fdividef(1.f, (float)d.pcols[s]); h.lrwd[s] = __fmul_rn(d.lr, d.wd[s]);
    h.off[s + 1] = h.off[s] + d.prows[s] * d.pcols[s];
  }
  h.base[0] = d.p[0]; h.pitch[0] = d.pcols[0];                                        // wte stays in global
  const int hs = kvm_fgpt_slot_head(L);
  h.base[hs] = sm + y.head; h.pitch[hs] = y.pld[hs];
  for (int l = 0; l < L; ++l) {
    float* const w = sm + y.wqkv + (size_t)l * y.wqkv_stride;
    float* const b = sm + y.bqkv + (size_t)l * 3 * D;
    const int roles[3][2] = {{KVM_FGPT_R_WQ, KVM_FGPT_R_BQ}, {KVM_FGPT_R_WK, KVM_FGPT_R_BK}, {KVM_FGPT_R_WV, KVM_FGPT_R_BV}};
    for (int k = 0; k < 3; ++k) {
      const int sw = kvm_fgpt_slot_layer(l, roles[k][0]), sb = kvm_fgpt_slot_layer(l, roles[k][1]);
      h.base[sw] = w + k * D; h.pitch[sw] = kvm_fgpt_ldmn(3 * D);
      h.base[sb] = b + k * D; h.pitch[sb] = D;
    }
  }
  h.vec4 = true;
  for (int s = 0; s < kvm_fgpt_nparams(L); ++s) {
    const auto al = [](const void* q) { return (reinterpret_cast<size_t>(q) & 15) == 0; };
    h.vec4 &= h.cols[s] % 4 == 0 && h.pitch[s] % 4 == 0 && (h.off[s + 1] - h.off[s]) % 4 == 0 && al(h.base[s]) && al(h.g[s]) && al(h.m[s]) && al(h.v[s]);
  }
}

// One step's backward from the forward's state on chip, its gradients into d.g. inv_n: the loss's 1 / (targeted rows).
template <class PH>
__device__ void backward(const KvmFgptDesc& d, float* sm, const Layout& y, int step, float inv_n, PH& ph) {
  const int tid = threadIdx.x;
  const int N = d.N, T = d.T, B = d.B, D = d.d, H = d.H, hd = d.hd, F = d.F, V = d.V, L = d.L;
  const bool prec = (d.mm_numerics & 16) != 0, rnd = !prec;   // rnd: the GEMM-only operands are stored rounded (fjob's RND)
  const int* const tgt = reinterpret_cast<const int*>(sm + y.tgt) + (step & 1) * N;
  const int* const ids = reinterpret_cast<const int*>(sm + y.ids) + (step & 1) * N;
  const float* const rtab = sm + y.rtab;
  float* const gx = sm + y.gx; float* const dy = sm + y.dy; float* const nh = sm + y.nh;
  const int ldd = y.ldd, ldq = y.ldq, ldS = y.ldS, ldF = y.ldF, ldV = y.ldV, N16 = kvm_fgpt_r16(N), T16 = kvm_fgpt_r16(T);
  auto P = [sm, yp = &y](int slot) { return sm + yp->p[slot]; };
  auto G = [&d](int slot) { return d.g[slot]; };
  auto lay = [](int l, int role) { return kvm_fgpt_slot_layer(l, role); };
  // ---- the head: dL = (softmax - onehot) / n on targeted rows, 0 elsewhere (into A); dW_head = HF^T dL; dHF = dL W^T (into dy)
  {
    // dL in place of the logits (PART, at their pitch): each element reads only its own logit, and region A keeps GELU's output from the
    // forward for the last layer's MLP backward
    const int ldl = ldV, hs = kvm_fgpt_slot_head(L);
    float* const DL = sm + y.part;
    const float* const lse = sm + y.lse;
    grid_loop(N16, ldl, [=](int r, int c) {
      const int tg = r < N ? tgt[r] : -1;
      const float g = tg == -1 || c >= V ? 0.f : (__expf(DL[r * ldl + c] - lse[r]) - (c == tg ? 1.f : 0.f)) * inv_n;
      DL[r * ldl + c] = rnd ? tf32f(g) : g;
    });
    __syncthreads();
    float* const gh = G(hs); const float* const W = sm + y.head; const int ldw = y.pld[hs];
    gemm_par(fjob<true, false, 3>(nh, ldd, DL, ldl, D, V, N, prec, [=](int m, int n, float a) { gh[(size_t)m * V + n] = a; }),
             fjob<false, true, 1>(DL, ldl, W, ldw, N, D, V, prec, [=](int m, int n, float a) { dy[(size_t)m * D + n] = a; }));
    (void)hs;
  }
  ph(MK_HEADB);
  // ---- the final norm: gx = its backward (stored); nf's gradients
  {
    const int nfg = kvm_fgpt_slot_nfg(L), nfb = kvm_fgpt_slot_nfb(L);
    ln_bwd_fast(sm + y.x[L], dy, P(nfg), sm + y.stf, sm + y.stf + N, gx, false, G(nfg), G(nfb), N, D);
  }
  ph(MK_LNFB);
  for (int l = L - 1; l >= 0; --l) {
    // ---- the MLP: b2's gradient is gx's column sums; U = gelu(pre) into A; dW2 = U^T gx and DU = (gx W2^T) gelu'(pre) (into pre, in place)
    float* const pre = sm + y.pre[l]; float* const U = sm + y.a;
    colsum_fast(gx, D, N, D, G(lay(l, KVM_FGPT_R_B2)), false);
    if (l != L - 1) {   // the last layer's U is still in A from the forward (fc2's operand); an earlier layer's is recomputed
      grid_loop(N16, ldF, [=](int r, int c) {
        const float z = pre[r * ldF + c];
        float g;
        if constexpr ((kFast & 8) != 0) g = r < N && c < F ? gelu_fast(z) : 0.f; else g = r < N && c < F ? gelu(z) : 0.f;
        U[r * ldF + c] = rnd ? tf32f(g) : g;
      });
      __syncthreads();
    }
    {
      float* const gw2 = G(lay(l, KVM_FGPT_R_W2)); const float* const W2 = P(lay(l, KVM_FGPT_R_W2)); const int ldw2 = y.pld[lay(l, KVM_FGPT_R_W2)];
      gemm_par(fjob<true, false, 1>(U, ldF, gx, D, F, D, N, prec, [=](int m, int n, float a) { gw2[(size_t)m * D + n] = a; }),
               fjob<false, true>(gx, D, W2, ldw2, N, F, D, prec, [=](int m, int n, float a) {
                 const float q = pre[m * ldF + n];   // the last layer's pre holds gelu'(z) from the forward, an earlier layer's z
                 pre[m * ldF + n] = a * (l == L - 1 ? q : gelu_grad(q));
               }));
    }
    ph(MK_X_GFC2);   // the bed's finer marks (fgpt_bed.cu's kMarks1), in the enum's order: db2, dW2 | dU
    // b1's gradient is DU's column sums; LN2's output again into NH (from xm and its statistics, as the forward wrote it)
    colsum_fast(pre, ldF, N, F, G(lay(l, KVM_FGPT_R_B1)), false);   // shares the barrier below with LN2's recompute
    {
      const float* const xm = sm + y.xm[l]; const float* const st = sm + y.st2[l];
      const float* const g2 = P(lay(l, KVM_FGPT_R_N2G)); const float* const b2n = P(lay(l, KVM_FGPT_R_N2B));
      grid_loop(N, D, [=](int r, int c) { const float v = (xm[r * D + c] - st[r]) * st[N + r] * g2[c] + b2n[c]; nh[r * ldd + c] = rnd ? tf32f(v) : v; });
      __syncthreads();
      ph(MK_MLPB);   // db1, LN2's output again
      float* const gw1 = G(lay(l, KVM_FGPT_R_W1)); const float* const W1 = P(lay(l, KVM_FGPT_R_W1)); const int ldw1 = y.pld[lay(l, KVM_FGPT_R_W1)];
      gemm_par(fjob<true, false, 1>(nh, ldd, pre, ldF, D, F, N, prec, [=](int m, int n, float a) { gw1[(size_t)m * F + n] = a; }),
               fjob<false, true>(pre, ldF, W1, ldw1, N, D, F, prec, [=](int m, int n, float a) { dy[(size_t)m * D + n] = a; }));
      ph(MK_X_DH2);   // dW1 | dh2
      // LN2's backward, added to gx (the gradient at xm)
      ln_bwd_fast(xm, dy, g2, st, st + N, gx, true, G(lay(l, KVM_FGPT_R_N2G)), G(lay(l, KVM_FGPT_R_N2B)), N, D);
    }
    ph(MK_LN2B);   // LN2's backward
    // ---- attention: bo's gradient is gx's column sums; dWo = O^T gx; dO = gx Wo^T (into PART at pitch ldd)
    float* const dO = sm + y.part; const float* const O = sm + y.o[l]; float* const qkv = sm + y.qkv[l];
    float* const dqkv = pre;                                                            // pre (DU) is dead: dq | dk | dv of every row, at pitch ldq
    colsum_fast(gx, D, N, D, G(lay(l, KVM_FGPT_R_BO)), false);
    float* const gwo = G(lay(l, KVM_FGPT_R_WO)); const float* const Wo = P(lay(l, KVM_FGPT_R_WO)); const int ldwo = y.pld[lay(l, KVM_FGPT_R_WO)];
    // each (sequence, head): P again from q, k and the forward's log-sum-exps; dV = P^T dO; dP = dO V^T; dS = P (dP - delta);
    // dQ = dS K scale; dK = dS^T Q scale. Attention's products are 3xTF32 as the forward's.
    const float scale = d.scale; const float* const alse = sm + y.alse[l];
    const long long blkf = (long long)T16 * ldS;
    bool h1_ready = false;   // the pair path recomputes LN1's output inside its RoPE pass
    if (B * H == 2 && T <= 32 && 3 * blkf <= y.a_floats && blkf <= (long long)N16 * ldd) {
      // gpt-tiny's two blocks at once (one block after the other, with delta and dS on one warp, cost 25 us): P, dS in A and NH
      float* const P0 = sm + y.a; float* const P1 = P0 + blkf; float* const S0 = P1 + blkf; float* const S1 = nh;
      const float* q0 = qkv + (size_t)(0 / H) * T * ldq + (0 % H) * hd; const float* q1 = qkv + (size_t)(1 / H) * T * ldq + (1 % H) * hd;
      const float* const k0 = q0 + D; const float* const k1 = q1 + D; const float* const v0 = q0 + 2 * D; const float* const v1 = q1 + 2 * D;
      const float* const l0 = alse; const float* const l1 = alse + T;
      const float* const dO0 = dO + (size_t)(0 / H) * T * ldd + (0 % H) * hd; const float* const dO1 = dO + (size_t)(1 / H) * T * ldd + (1 % H) * hd;
      const float* const O0 = O + (size_t)(0 / H) * T * ldd + (0 % H) * hd; const float* const O1 = O + (size_t)(1 / H) * T * ldd + (1 % H) * hd;
      float* const dq0 = dqkv + (size_t)(0 / H) * T * ldq + (0 % H) * hd; float* const dq1 = dqkv + (size_t)(1 / H) * T * ldq + (1 % H) * hd;
      // dWo and dO, and P again for both blocks (it needs only q, k and the forward's log-sum-exps), in one pass
      gemm_par(fjob<true, false>(O, ldd, gx, D, D, D, N, prec, [=](int m, int n, float a) { gwo[(size_t)m * D + n] = a; }),
               fjob<false, true>(gx, D, Wo, ldwo, N, D, D, prec, [=](int m, int n, float a) { dO[m * ldd + n] = a; }),
               fjob<false, true>(q0, ldq, k0, ldq, T, T, hd, true, [=](int m, int n, float a) { P0[m * ldS + n] = n <= m ? __expf(a * scale - l0[m]) : 0.f; }),
               fjob<false, true>(q1, ldq, k1, ldq, T, T, hd, true, [=](int m, int n, float a) { P1[m * ldS + n] = n <= m ? __expf(a * scale - l1[m]) : 0.f; }));
      ph(MK_UB_STAGE);   // dbo, dWo | dO | P
      gemm_par(fjob<true, false>(P0, ldS, dO0, ldd, T, hd, T, true, [=](int m, int n, float a) { dq0[(size_t)m * ldq + 2 * D + n] = a; }),
               fjob<false, true>(dO0, ldd, v0, ldq, T, T, hd, true, [=](int m, int n, float a) { S0[m * ldS + n] = a; }),
               fjob<true, false>(P1, ldS, dO1, ldd, T, hd, T, true, [=](int m, int n, float a) { dq1[(size_t)m * ldq + 2 * D + n] = a; }),
               fjob<false, true>(dO1, ldd, v1, ldq, T, T, hd, true, [=](int m, int n, float a) { S1[m * ldS + n] = a; }));
      ph(MK_UB_DO);   // dV | dP
      {   // delta and dS on four lanes a row over both blocks' rows: delta_i = O_i . dO_i, then dS_ij = P_ij (dP_ij - delta_i) on the causal part
        constexpr int LPR = 4;
        const int KPL = T16 / LPR, U = (LPR * 2 * T + 31) & ~31;
        for (int u = tid; u < U; u += NT) {
          const int rr = u / LPR, j = u % LPR;
          const bool on = rr < 2 * T, one = rr >= T;
          const int i = on ? rr % T : 0;
          const float* const Ob = one ? O1 : O0; const float* const dOb = one ? dO1 : dO0;
          const float* const Pb = one ? P1 : P0; float* const Sb = one ? S1 : S0;
          float dl = 0.f;
          if (on) for (int c = j; c < hd; c += LPR) dl += Ob[(size_t)i * ldd + c] * dOb[(size_t)i * ldd + c];
          dl = group_sum<LPR>(dl);
          if (on) for (int jj = j * KPL; jj < (j + 1) * KPL; ++jj) Sb[i * ldS + jj] = jj <= i ? Pb[i * ldS + jj] * (Sb[i * ldS + jj] - dl) : 0.f;
        }
        __syncthreads();
      }
      ph(MK_UB_STAGE2);   // delta, dS
      gemm_par(fjob<false, false>(S0, ldS, k0, ldq, T, hd, T, true, [=](int m, int n, float a) { dq0[(size_t)m * ldq + n] = a * scale; }),
               fjob<true, false>(S0, ldS, q0, ldq, T, hd, T, true, [=](int m, int n, float a) { dq0[(size_t)m * ldq + D + n] = a * scale; }),
               fjob<false, false>(S1, ldS, k1, ldq, T, hd, T, true, [=](int m, int n, float a) { dq1[(size_t)m * ldq + n] = a * scale; }),
               fjob<true, false>(S1, ldS, q1, ldq, T, hd, T, true, [=](int m, int n, float a) { dq1[(size_t)m * ldq + D + n] = a * scale; }));
      ph(MK_UB_P);   // dQ | dK
      {   // RoPE's inverse on both blocks' dq and dk in one pass (rope()'s expressions, sign -1, positions 0..T-1)
        const int half = hd / 2, per = T * half;
        for (int e = tid; e < 4 * per; e += NT) {
          const int which = e / per, rem = e - which * per, r = rem / half, i = rem - r * half;
          float* const X = ((which >> 1) ? dq1 : dq0) + (which & 1) * D;
          const int at = r * half + i;
          const float cs = rtab[at], sn = -1.f * rtab[T * half + at];
          float* const pr = X + r * ldq + i;
          const float va = pr[0], vb = pr[half];
          pr[0] = va * cs - vb * sn;
          pr[half] = va * sn + vb * cs;
        }
        const float* const x = sm + y.x[l]; const float* const st = sm + y.st1[l];   // LN1's output again into NH, in the same pass
        const float* const g1 = P(lay(l, KVM_FGPT_R_N1G)); const float* const b1n = P(lay(l, KVM_FGPT_R_N1B));
        grid_loop(N, D, [=](int r, int c) { const float v = (x[r * D + c] - st[r]) * st[N + r] * g1[c] + b1n[c]; nh[r * ldd + c] = rnd ? tf32f(v) : v; });
        __syncthreads();
        h1_ready = true;
      }
      ph(MK_UB_DV_DS);   // RoPE back, LN1's output again
    } else {
      gemm_par(fjob<true, false>(O, ldd, gx, D, D, D, N, prec, [=](int m, int n, float a) { gwo[(size_t)m * D + n] = a; }),
               fjob<false, true>(gx, D, Wo, ldwo, N, D, D, prec, [=](int m, int n, float a) { dO[m * ldd + n] = a; }));
      ph(MK_UB_STAGE);   // dbo, dWo | dO
      float* const Pm = sm + y.a; float* const dS = sm + y.a + (size_t)T16 * ldS;
      for (int blk = 0; blk < B * H; ++blk) {
        const int sq = blk / H, hh = blk % H, c0 = hh * hd;
        const float* const q = qkv + (size_t)sq * T * ldq + c0;
        const float* const k = q + D;
        const float* const v = q + 2 * D;
        const float* const lsb = alse + (size_t)blk * T;
        const float* const dOb = dO + (size_t)sq * T * ldd + c0;
        const float* const Ob = O + (size_t)sq * T * ldd + c0;
        float* const dqb = dqkv + (size_t)sq * T * ldq + c0;
        fgemm<false, true>(q, ldq, k, ldq, T, T, hd, true, [=](int m, int n, float a) { Pm[m * ldS + n] = n <= m ? __expf(a * scale - lsb[m]) : 0.f; });
        gemm_par(fjob<true, false>(Pm, ldS, dOb, ldd, T, hd, T, true, [=](int m, int n, float a) { dqb[(size_t)m * ldq + 2 * D + n] = a; }),
                 fjob<false, true>(dOb, ldd, v, ldq, T, T, hd, true, [=](int m, int n, float a) { dS[m * ldS + n] = a; }));
        for (int i = tid; i < T; i += NT) {                                                 // delta and dS, a thread a row
          float dl = 0.f;
          for (int c = 0; c < hd; ++c) dl += Ob[(size_t)i * ldd + c] * dOb[(size_t)i * ldd + c];
          for (int j2 = 0; j2 < T; ++j2) dS[i * ldS + j2] = j2 <= i ? Pm[i * ldS + j2] * (dS[i * ldS + j2] - dl) : 0.f;
        }
        __syncthreads();
        gemm_par(fjob<false, false>(dS, ldS, k, ldq, T, hd, T, true, [=](int m, int n, float a) { dqb[(size_t)m * ldq + n] = a * scale; }),
                 fjob<true, false>(dS, ldS, q, ldq, T, hd, T, true, [=](int m, int n, float a) { dqb[(size_t)m * ldq + D + n] = a * scale; }));
        rope(dqb, ldq, T, 0, T, hd, rtab, -1.f);
        rope(dqb + D, ldq, T, 0, T, hd, rtab, -1.f);
      }
      ph(MK_UB_P);   // the other shapes: each block's P to RoPE back
    }
    // q, k and v's biases are dqkv's column sums; LN1's output again into NH; dWqkv = h1^T dqkv; dh1 = dqkv Wqkv^T (into dy)
    {
      const int lq = lay(l, KVM_FGPT_R_BQ), lk = lay(l, KVM_FGPT_R_BK), lv = lay(l, KVM_FGPT_R_BV);
      colsum_qkv(dqkv, ldq, N, D, G(lq), G(lk), G(lv));   // reads dqkv, writes the biases' gradients: no barrier of its own
      const float* const x = sm + y.x[l]; const float* const st = sm + y.st1[l];
      const float* const g1 = P(lay(l, KVM_FGPT_R_N1G)); const float* const b1n = P(lay(l, KVM_FGPT_R_N1B));
      if (!h1_ready) {
        grid_loop(N, D, [=](int r, int c) { const float v = (x[r * D + c] - st[r]) * st[N + r] * g1[c] + b1n[c]; nh[r * ldd + c] = rnd ? tf32f(v) : v; });
        __syncthreads();
      }
      float* const gq = G(lay(l, KVM_FGPT_R_WQ)); float* const gk = G(lay(l, KVM_FGPT_R_WK)); float* const gv = G(lay(l, KVM_FGPT_R_WV));
      const float* const Wqkv = sm + y.wqkv + (size_t)l * y.wqkv_stride; const int ldw = kvm_fgpt_ldmn(3 * D);
      gemm_par(fjob<true, false, 1>(nh, ldd, dqkv, ldq, D, 3 * D, N, prec, [=](int m, int n, float a) {
                 const int part = (n >= D) + (n >= 2 * D);   // q, k or v, by compares: no run-time modulo an element
                 float* const o = part == 0 ? gq : part == 1 ? gk : gv;
                 o[(size_t)m * D + (n - part * D)] = a;
               }),
               fjob<false, true>(dqkv, ldq, Wqkv, ldw, N, D, 3 * D, prec, [=](int m, int n, float a) { dy[(size_t)m * D + n] = a; }));
      ph(MK_UB_DQ_DK);   // dbq dbk dbv, (LN1's output again,) dWqkv | dh1
      ln_bwd_fast(x, dy, g1, st, st + N, gx, true, G(lay(l, KVM_FGPT_R_N1G)), G(lay(l, KVM_FGPT_R_N1B)), N, D);
    }
    ph(MK_ATTNB);
  }
  // ---- the embedding: wte's gradient row v is the sum of gx's rows r with ids[r] == v, in row order
  {   // a warp a vocabulary row at a time, lanes over the columns; each 32-row chunk's matches come from one ballot, taken in row order,
      // so a row's chain is its matches, not N compares (the compares cost 5 us). D at most KVM_FGPT1_MAXD, CJ a lane.
    float* const gw = G(0);
    const int w = tid >> 5, lane = tid & 31;
    for (int v = w; v < V; v += NT / 32) {
      constexpr int CJ = KVM_FGPT1_MAXD / 32;
      float acc[CJ];
#pragma unroll
      for (int k = 0; k < CJ; ++k) acc[k] = 0.f;
      for (int r0 = 0; r0 < N; r0 += 32) {
        unsigned m = __ballot_sync(0xffffffffu, r0 + lane < N && ids[r0 + lane] == v);
        while (m) {
          const int rr = r0 + __ffs(m) - 1;
          m &= m - 1;
#pragma unroll
          for (int k = 0; k < CJ; ++k) if (lane + 32 * k < D) acc[k] += gx[rr * D + lane + 32 * k];
        }
      }
#pragma unroll
      for (int k = 0; k < CJ; ++k) if (lane + 32 * k < D) gw[(size_t)v * D + lane + 32 * k] = acc[k];
    }
    __syncthreads();
  }
  ph(MK_EMBB);
}

// One element's AdamW in run's float ops (kvm_fused_gpt.cuh's update), with lr x wd from the slot's table (the product run takes first).
__device__ __forceinline__ void adam1(float g, float& m, float& v, float& p, float coef, float b1, float ob1, float b2, float ob2, float lr,
                                      float lrwd, float bc1, float bc2, float eps) {
  const float gi = __fmul_rn(g, coef);
  const float mi = __fadd_rn(__fmul_rn(b1, m), __fmul_rn(ob1, gi));
  const float vi = __fadd_rn(__fmul_rn(b2, v), __fmul_rn(__fmul_rn(ob2, gi), gi));
  const float decay = __fmul_rn(lrwd, p);
  float upd;
  if constexpr ((kFast & 16) != 0) {   // bc1, bc2 arrive as their reciprocals: no IEEE division's slow-path branch in the chain
    const float vh = vi * bc2, sq = vh > 0.f ? vh * rsqrtf(vh) : 0.f;
    upd = __fdividef(lr * (mi * bc1), sq + eps);
  } else {
    upd = __fdiv_rn(__fmul_rn(lr, __fdiv_rn(mi, bc1)), __fadd_rn(__fsqrt_rn(__fdiv_rn(vi, bc2)), eps));
  }
  m = mi; v = vi; p = __fsub_rn(__fsub_rn(p, decay), upd);
}

// AdamW over every parameter after the step's backward: the clip's norm from the squares of every gradient, then each element updated where
// it lives. When every slot is float4-able (h.vec4: gpt-tiny's case) and the ring fits, a lane takes float4 groups of one flat index over all
// the slots, KG of them, their gradients held in registers from the norm's pass to the update's (a gradient is read once) and their moments
// streamed through the ring. Otherwise a scalar form, AD elements in flight (a scalar element at a time cost 34 us of the step on
// one SM).
template <class PH>
__device__ void adamw(const KvmFgptDesc& d, const ParamHome& h, int step, float* red, float* ring, int ring_floats, PH& ph) {
  constexpr int AD = 8, KG = 12, RS = kAdamRing;   // KG x NT float4 groups: 4,608 a step, gpt-tiny's 4,216 fit
  const int np = kvm_fgpt_nparams(d.L), total = h.off[np];
  const float b1 = d.beta1, b2 = d.beta2, lr = d.lr, eps = d.adam_eps;
  float bc1 = __ldcg(d.bc + 2 * step), bc2 = __ldcg(d.bc + 2 * step + 1);
  if constexpr ((kFast & 16) != 0) { bc1 = __frcp_rn(bc1); bc2 = __frcp_rn(bc2); }   // adam1's relaxed form takes the reciprocals
  const float ob1 = __fsub_rn(1.f, b1), ob2 = __fsub_rn(1.f, b2);
  auto clip = [&](float sq) {
    double sd = sq, dummy = 0.0;
    block_sum2(sd, dummy, reinterpret_cast<double*>(red));
    const float norm = sqrtf((float)sd);
    return d.max_norm > 0.f ? fminf(1.f, d.max_norm / (norm + 1e-6f)) : 1.f;
  };
  if (h.vec4 && total / 4 <= KG * NT && ring_floats >= 8 * RS * NT) {
    const int t4 = total / 4, tid = threadIdx.x;
    // The ring: the activations' span, dead from the backward's last barrier to the next step's embedding. Each thread copies its own
    // groups' m and v into its own slots by cp.async (L2 to shared: no register held while they fly, and nothing through L1, so the stack
    // frame's lines and the descriptor's stay there) and reads back only what it copied, so the ring takes no barrier. RS groups fly from
    // the norm's pass on; each group the update consumes refills its slot with the group RS ahead, one commit group a group (empty when
    // none is left), so "group k has landed" is always cp.async.wait_group RS - 1 (the next step's token copies, committed earlier, are
    // older than every group the waits name). The moments held in registers instead, two batches beside the 12 gradients,
    // spilled 154 of the kernel's 499 local-memory instructions into the update, and its loads ran latency-bound at about 17 B a clock.
    // Each element's arithmetic is unchanged, so its bits are.
    float4* const rm = reinterpret_cast<float4*>(ring);   // the m plane: thread t's slot j at j NT + t (16-byte lanes: no bank conflict)
    float4* const rv = rm + RS * NT;                      // the v plane
    int si = 0;                                           // the copies' walk over the slots, in group order as the update's
    const auto issue = [&](int k) {
      const int i = tid + k * NT;
      if (i < t4) {
        while (4 * i >= h.off[si + 1]) ++si;
        const int le = 4 * i - h.off[si], q = (k % RS) * NT + tid;
        cp16(reinterpret_cast<float*>(rm + q), h.m[si] + le, true);
        cp16(reinterpret_cast<float*>(rv + q), h.v[si] + le, true);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    float4 gk[KG];
    float sq = 0.f;
    {
      int s = 0;
#pragma unroll
      for (int k = 0; k < KG; ++k) {
        const int i = tid + k * NT;
        if (i < t4) {
          while (4 * i >= h.off[s + 1]) ++s;
          gk[k] = __ldcg(reinterpret_cast<const float4*>(h.g[s] + (4 * i - h.off[s])));   // the block's own writes, after a barrier
        } else gk[k] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
#pragma unroll
      for (int k = 0; k < RS; ++k) issue(k);   // behind the gradients, which the clip needs first
#pragma unroll
      for (int k = 0; k < KG; ++k) sq += (gk[k].x * gk[k].x + gk[k].y * gk[k].y) + (gk[k].z * gk[k].z + gk[k].w * gk[k].w);
    }
    const float coef = clip(sq);
    ph(MK_SQUARES);
    int s = 0;
#pragma unroll
    for (int k = 0; k < KG; ++k) {
      const int i = tid + k * NT;
      asm volatile("cp.async.wait_group %0;\n" ::"n"(RS - 1) : "memory");   // group k's copies have landed
      if (i < t4) {
        while (4 * i >= h.off[s + 1]) ++s;
        const int le = 4 * i - h.off[s], cols = h.cols[s];
        const int r = div_small(le, cols, h.inv_cols[s]), c = le - r * cols;
        float* const pp = h.base[s] + (size_t)r * h.pitch[s] + c;
        const bool gl = s == 0;   // wte's home is global memory (param_homes); every other slot's is shared
        const int q = (k % RS) * NT + tid;
        float4 m = rm[q], v = rv[q];
        float4 p = gl ? __ldcg(reinterpret_cast<const float4*>(pp)) : *reinterpret_cast<const float4*>(pp);
        const float4 g = gk[k];
        const float lw = h.lrwd[s];
        adam1(g.x, m.x, v.x, p.x, coef, b1, ob1, b2, ob2, lr, lw, bc1, bc2, eps);
        adam1(g.y, m.y, v.y, p.y, coef, b1, ob1, b2, ob2, lr, lw, bc1, bc2, eps);
        adam1(g.z, m.z, v.z, p.z, coef, b1, ob1, b2, ob2, lr, lw, bc1, bc2, eps);
        adam1(g.w, m.w, v.w, p.w, coef, b1, ob1, b2, ob2, lr, lw, bc1, bc2, eps);
        __stcg(reinterpret_cast<float4*>(h.m[s] + le), m);
        __stcg(reinterpret_cast<float4*>(h.v[s] + le), v);
        if (gl) __stcg(reinterpret_cast<float4*>(pp), p); else *reinterpret_cast<float4*>(pp) = p;
      }
      if (k + RS < KG) issue(k + RS);   // into the slot just read, whose values the update above has consumed
      else asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    __syncthreads();
    ph(MK_ADAMW);
    return;
  }
  float sq = 0.f;
  {
    int s = 0;
    for (int e0 = threadIdx.x; e0 < total; e0 += AD * NT) {
      float x[AD];
#pragma unroll
      for (int u = 0; u < AD; ++u) {
        const int e = e0 + u * NT;
        if (e < total) { while (e >= h.off[s + 1]) ++s; x[u] = h.g[s][e - h.off[s]]; } else x[u] = 0.f;
      }
#pragma unroll
      for (int u = 0; u < AD; ++u) sq += x[u] * x[u];
    }
  }
  const float coef = clip(sq);
  ph(MK_SQUARES);
  int s = 0;
  for (int e0 = threadIdx.x; e0 < total; e0 += AD * NT) {
    float gx[AD], mx[AD], vx[AD], px[AD], lw[AD];
    float* pp[AD];
    float* mp[AD];
    float* vp[AD];
    bool ok[AD];
#pragma unroll
    for (int u = 0; u < AD; ++u) {
      const int e = e0 + u * NT;
      ok[u] = e < total;
      pp[u] = mp[u] = vp[u] = nullptr; gx[u] = lw[u] = 0.f;
      if (ok[u]) {
        while (e >= h.off[s + 1]) ++s;
        const int le = e - h.off[s], cols = h.cols[s];
        const int r = div_small(le, cols, h.inv_cols[s]), c = le - r * cols;
        pp[u] = h.base[s] + (size_t)r * h.pitch[s] + c; mp[u] = h.m[s] + le; vp[u] = h.v[s] + le;
        gx[u] = h.g[s][le]; lw[u] = h.lrwd[s];
      }
    }
#pragma unroll
    for (int u = 0; u < AD; ++u) { mx[u] = ok[u] ? *mp[u] : 0.f; vx[u] = ok[u] ? *vp[u] : 0.f; px[u] = ok[u] ? *pp[u] : 0.f; }
#pragma unroll
    for (int u = 0; u < AD; ++u) if (ok[u]) adam1(gx[u], mx[u], vx[u], px[u], coef, b1, ob1, b2, ob2, lr, lw[u], bc1, bc2, eps);
#pragma unroll
    for (int u = 0; u < AD; ++u)
      if (ok[u]) { *mp[u] = mx[u]; *vp[u] = vx[u]; *pp[u] = px[u]; }
  }
  __syncthreads();
  ph(MK_ADAMW);
}

// C2's run: the run's steps, each the forward and its loss, the parameters resident and unchanged (C3 adds the backward and AdamW). The
// step counter advances as run's does. False on a fault (fail(code) called once).
template <class FAIL, class PH>
__device__ bool run1(const KvmFgptDesc& d, float* sm, long long smem_floats, FAIL fail, PH& ph) {
  const int tid = threadIdx.x;
  // the layout once a block, in shared memory: raw storage, built in place by thread 0 (a __shared__ object of the layout's type would
  // need a dynamic initialization CUDA does not run, and making the type trivial instead cost the step about 4%: its stack frame grew 896 -> 1,056 bytes)
  __shared__ alignas(Layout) unsigned char ys_mem[sizeof(Layout)];
  Layout& ys = *reinterpret_cast<Layout*>(ys_mem);
  if (tid == 0) new (ys_mem) Layout(kvm_fgpt1_layout(d.N, d.T, d.d, d.H, d.F, d.V, d.L));
  __syncthreads();
  const Layout& y = ys;
  // the shape, mask and numerics this form takes, and its layout against the shared memory given (kvm_fgpt1.h: the pass and the executor ask
  // the same question before launching)
  const int bad = kvm_fgpt1_check_layout(y, d.N, d.T, d.B, d.d, d.H, d.hd, d.F, d.V, d.mm_numerics, kFast, smem_floats);
  if (bad) { if (tid == 0) fail(bad); return false; }
  zero_n(sm, y.total);                                                                // every pad an exact zero
  __syncthreads();
  rope_table(sm + y.rtab, d.T, d.hd, d.theta);
  load_params(d, sm, y);
  const int N = d.N, V = d.V;
  {   // the head, resident for the run like every other parameter (C3 updates it in place): its rows, 16 bytes a copy where every row starts
      // aligned (V a multiple of 4 floats), else 4 bytes
    const int hs = kvm_fgpt_slot_head(d.L), ldh = y.pld[hs];
    float* const hb = sm + y.head;
    const float* const src = d.p[hs];
    if ((V & 3) == 0 && (reinterpret_cast<size_t>(src) & 15) == 0)
      for (int e = tid; e < d.d * (V / 4); e += NT) { const int r = e / (V / 4), c = (e % (V / 4)) * 4; cp16(hb + (size_t)r * ldh + c, src + (size_t)r * V + c, true); }
    else
      for (int e = tid; e < d.d * V; e += NT) { const int r = e / V, c = e % V; cp4(hb + (size_t)r * ldh + c, src + (size_t)r * V + c, true); }
    asm volatile("cp.async.commit_group;\n" "cp.async.wait_all;\n" ::: "memory");
  }
  __shared__ ParamHome home;
  if constexpr (kC3) {   // each step's AdamW bias corrections, run's expression (the float of the double pow, from 1 in float); the homes
    for (int k = tid; k < 2 * d.steps; k += NT)
      d.bc[k] = __fsub_rn(1.f, (float)pow((double)((k & 1) ? d.beta2 : d.beta1), (double)((long long)__ldcg(d.step) + (k >> 1) + 1)));
    if (tid == 0) param_homes(d, sm, y, home);
    __syncthreads();
  }
  int* const ids0 = reinterpret_cast<int*>(sm + y.ids);
  int* const tgt0 = reinterpret_cast<int*>(sm + y.tgt);
  __shared__ int flag, next_flag;
  const int t0 = __ldcg(d.step);
  // A step's window: its error code, or 0 (run's checks: the step counter and the token and target windows within their streams)
  auto window_err = [&](long long t, long long& bi, long long& bt) -> int {
    bi = t * d.ids_stride + d.ids_off; bt = t * d.tgt_stride + d.tgt_off;
    if (t + 1 < 1) return (int)KVM_FGPT_E_STEP;
    if (bi < 0 || bi + N > d.ids_len || bt < 0 || bt + N > d.tgt_len) return (int)KVM_FGPT_E_WINDOW;
    return 0;
  };
  if (tid == 0) next_flag = 0;
  {   // step 0's tokens, read now; every later step's are prefetched during the step before it (cp.async), off the step's path
    long long bi, bt;
    const int e = window_err(t0, bi, bt);
    if (e) { if (tid == 0) next_flag = e; }
    else for (int r = tid; r < N; r += NT) { ids0[r] = __ldcg(d.ids_src + bi + r); tgt0[r] = __ldcg(d.tgt_src + bt + r); }
  }
  for (int step = 0; step < d.steps; ++step) {
    const long long t = (long long)t0 + step;
    int* const ids = ids0 + (step & 1) * N;
    int* const tgt = tgt0 + (step & 1) * N;
    asm volatile("cp.async.wait_all;\n" ::: "memory");                               // this step's prefetched tokens (a no-op at step 0)
    __syncthreads();
    if (tid == 0) { flag = next_flag; next_flag = 0; }
    __syncthreads();
    if (!flag)
      for (int r = tid; r < N; r += NT) {
        const int id = ids[r], tg = tgt[r];
        if (id < 0 || id >= V) atomicCAS(&flag, 0, KVM_FGPT_E_TOKEN);
        if (tg != -1 && (tg < 0 || tg >= V)) atomicCAS(&flag, 0, KVM_FGPT_E_TARGET);
      }
    __syncthreads();
    if (flag) { if (tid == 0) fail(flag); return false; }
    if (step + 1 < d.steps) {   // the next step's tokens into the other slot, waited for at its start
      long long bi, bt;
      const int e = window_err(t + 1, bi, bt);
      if (e) { if (tid == 0) next_flag = e; }
      else {
        int* const ni = ids0 + ((step + 1) & 1) * N; int* const nt = tgt0 + ((step + 1) & 1) * N;
        for (int r = tid; r < N; r += NT) {
          cp4(reinterpret_cast<float*>(ni + r), reinterpret_cast<const float*>(d.ids_src + bi + r), true);
          cp4(reinterpret_cast<float*>(nt + r), reinterpret_cast<const float*>(d.tgt_src + bt + r), true);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
      }
    }
    const float inv_n = forward(d, sm, y, step, ph);
    if constexpr (kC3) {
      __syncthreads();
      backward(d, sm, y, step, inv_n, ph);
      adamw(d, home, step, reinterpret_cast<float*>(sm + y.red), sm + y.ring, y.ring_floats, ph);
    } else {
      (void)inv_n;
    }
  }
  if constexpr (kC3) {   // the resident parameters back to their global homes (wte is already there)
    for (int s = 1; s < kvm_fgpt_nparams(d.L); ++s) store_back(d.p[s], home.base[s], home.pitch[s], d.prows[s], d.pcols[s]);
  }
  if (tid == 0) *d.step = t0 + d.steps;
  return true;
}

}  // namespace kvm_fgpt1
