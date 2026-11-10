#pragma once
// The fused GPT step, device side (design D2): one member's share of every step of a fused run. kvm_fused_gpt.h has the descriptor, the
// parameter slots and the scratch layout; bench/fused_gpt/d2_reference.py is the executable specification of the data flow below, checked
// against the workload VM, and bench/fused_gpt/fgpt_bed.cu runs this code alone against the VM.
//
// v0 is the correctness form. Activations and partials live in the global scratch (`xch`); parameters stay in their bundle buffers and are
// updated in place by their owners; GEMMs are FP32 FMA through one shared-memory-tiled block GEMM. Every value another member wrote is read
// through L2 (ld.global.cg), never through a possibly stale L1 line. Later forms keep the shards resident in shared memory and move the GEMMs
// to tensor cores; the phases, the ownership and the barriers stay.
//
// Ownership, for member b of a gang of G:
//   rows      [b N / G, (b + 1) N / G) of every N x d stream: LayerNorms, residual adds, the reduce-scatters' sums;
//   heads     head h belongs to member h % G: its q, k and v columns, RoPE, attention, and proj's rows of those columns;
//   fcols     fc1's columns [b P, (b + 1) P) with P = ceil(F / G), and fc2's rows, the same units;
//   vocab     the head's columns and wte's rows [b P, (b + 1) P) with P = ceil(V / G);
//   member 0  the replicated parameters: the LayerNorms' gains and biases, proj's and fc2's biases. Every member adds its own rows' partials
//             of their gradients; member 0 sums those in member order.
// Every element of every parameter has exactly one owner, which computes its gradient, counts it in the grad norm and applies AdamW to it.
// Exchanges: an all-gather is own rows written, then a barrier. A reduce-scatter is one partial a contributing member written, a barrier,
// then each member sums its own rows of the partials in member order. A step crosses 8 L + 6 barriers; the last one ends the step, so the
// next step's reads see every update.
#include <cmath>
#include <cstdint>
#include "kvm_fused_gpt.h"
#include "kvm_gang.cuh"

namespace kvm_fgpt {

constexpr int NT = KVM_FGPT_NT;
constexpr int NW = NT / 32;
constexpr int GK = KVM_FGPT_GK;
constexpr int GT = KVM_FGPT_GT;
constexpr int GLD = GT + 1;   // a staged slice's row, padded so that a warp's stores spread over the banks

__device__ __forceinline__ float ld(const float* p) { return __ldcg(p); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }

// The bundle's GELU (tanh form) and its derivative, as runtime/ref_gpt.py writes them.
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ float gelu_grad(float x) {
  const float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * 0.7978845608028654f * (1.f + 3.f * 0.044715f * x * x);
}

// The member's shared memory (kvm_fgpt_smem_floats(N, T) floats), carved once.
struct Smem {
  float* As; float* Bs;       // the block GEMM's staged K slices
  float* rows;                // two score rows of T a warp
  float* delta;               // attention backward's delta, T
  float* lse;                 // the cross-entropy's log-sum-exp a row, N
  int* ids; int* tgt;         // the step's tokens and targets, N each
  float* red;                 // a block reduction's NW partials
  int* flag;                  // a block-wide flag (an invalid input)
};
__device__ __forceinline__ Smem carve(float* sm, int N, int T) {
  Smem s;
  s.As = sm; s.Bs = s.As + GK * GLD; s.rows = s.Bs + GK * GLD; s.delta = s.rows + 2 * NW * T; s.lse = s.delta + T;
  s.ids = reinterpret_cast<int*>(s.lse + N); s.tgt = s.ids + N; s.red = reinterpret_cast<float*>(s.tgt + N); s.flag = reinterpret_cast<int*>(s.red + NW);
  return s;
}

// C(M x N) = epi(m, n, sum over k of A(m, k) B(k, n)) for every m < M and n < N, with A(m, k) = A[TA ? k lda + m : m lda + k] and B(k, n) =
// B[TB ? n ldb + k : k ldb + n]. Output tiles of GT x GT, K slices of GK staged in shared memory, a thread accumulating 4 x 4 outputs in K
// order: a result does not depend on the gang's size. Operands are read through L2. Every thread calls it; it ends with __syncthreads.
template <bool TA, bool TB, class Epi>
__device__ void gemm(const float* A, int lda, const float* B, int ldb, int M, int N, int K, const Smem& sm, Epi epi) {
  const int tid = threadIdx.x, tm = (tid >> 4) << 2, tn = (tid & 15) << 2;
  for (int m0 = 0; m0 < M; m0 += GT)
    for (int n0 = 0; n0 < N; n0 += GT) {
      float acc[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
      for (int k0 = 0; k0 < K; k0 += GK) {
        __syncthreads();   // the previous slice's readers are done
        for (int e = tid; e < GK * GT; e += NT) {
          // consecutive threads read consecutive addresses: k fastest where the operand is contiguous in k, m (or n) fastest otherwise
          const int ka = TA ? e / GT : e % GK, ma = TA ? e % GT : e / GK, m = m0 + ma, k = k0 + ka;
          sm.As[ka * GLD + ma] = (m < M && k < K) ? ld(A + (TA ? (size_t)k * lda + m : (size_t)m * lda + k)) : 0.f;
          const int kb = TB ? e % GK : e / GT, nb = TB ? e / GK : e % GT, n = n0 + nb, kk = k0 + kb;
          sm.Bs[kb * GLD + nb] = (n < N && kk < K) ? ld(B + (TB ? (size_t)n * ldb + kk : (size_t)kk * ldb + n)) : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GK; ++kk) {
          float a[4], bv[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) { a[i] = sm.As[kk * GLD + tm + i]; bv[i] = sm.Bs[kk * GLD + tn + i]; }
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int m = m0 + tm + i, n = n0 + tn + j;
          if (m < M && n < N) epi(m, n, acc[i][j]);
        }
    }
  __syncthreads();
}

// LayerNorm of rows [r0, r1) over d columns: y = (x - mean) rstd g + b, the statistics saved for the backward. One warp a row.
__device__ void ln_fwd(const float* x, const float* g, const float* be, float* y, float* mean, float* rstd, int r0, int r1, int d, float eps) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = r0 + w; r < r1; r += NW) {
    const float* xr = x + (size_t)r * d;
    float s = 0.f;
    for (int c = lane; c < d; c += 32) s += ld(xr + c);
    const float mu = warp_sum(s) / (float)d;
    float q = 0.f;
    for (int c = lane; c < d; c += 32) { const float t = ld(xr + c) - mu; q += t * t; }
    const float rs = 1.f / sqrtf(warp_sum(q) / (float)d + eps);
    for (int c = lane; c < d; c += 32) y[(size_t)r * d + c] = (ld(xr + c) - mu) * rs * ld(g + c) + ld(be + c);
    if (lane == 0) { mean[r] = mu; rstd[r] = rs; }
  }
}

// LayerNorm backward of rows [r0, r1): dx = rstd (dy g - mean(dy g) - xh mean(dy g xh)) with xh = (x - mean) rstd, added to gx (the
// residual-stream gradient); and this member's own-row partials of dgamma (dy xh summed over its rows in row order) into pg and of dbeta
// (dy) into pb. Ends with __syncthreads.
__device__ void ln_bwd(const float* x, const float* dy, const float* g, const float* mean, const float* rstd, float* gx, bool acc,
                       float* pg, float* pb, int r0, int r1, int d) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = r0 + w; r < r1; r += NW) {
    const size_t o = (size_t)r * d;
    const float mu = ld(mean + r), rs = ld(rstd + r);
    float s1 = 0.f, s2 = 0.f;
    for (int c = lane; c < d; c += 32) { const float gs = ld(dy + o + c) * ld(g + c), xh = (ld(x + o + c) - mu) * rs; s1 += gs; s2 += gs * xh; }
    s1 = warp_sum(s1) / (float)d; s2 = warp_sum(s2) / (float)d;
    for (int c = lane; c < d; c += 32) {
      const float gs = ld(dy + o + c) * ld(g + c), xh = (ld(x + o + c) - mu) * rs, v = rs * (gs - s1 - xh * s2);
      gx[o + c] = acc ? ld(gx + o + c) + v : v;
    }
  }
  for (int c = threadIdx.x; c < d; c += NT) {
    float sg = 0.f, sb = 0.f;
    for (int r = r0; r < r1; ++r) { const float dv = ld(dy + (size_t)r * d + c); sg += dv * (ld(x + (size_t)r * d + c) - ld(mean + r)) * ld(rstd + r); sb += dv; }
    pg[c] = sg; pb[c] = sb;
  }
  __syncthreads();
}

// The reduce-scatter's sum: out[r] = (sum of the P partials' row r, in member order) + bias + res[r], for rows [r0, r1). Ends with __syncthreads.
__device__ void rs_sum(const float* part, int P, size_t Nd, const float* bias, const float* res, float* out, int r0, int r1, int d) {
  for (int e = threadIdx.x; e < (r1 - r0) * d; e += NT) {
    const int c = e % d;
    const size_t o = (size_t)r0 * d + e;
    float a = 0.f;
    for (int p = 0; p < P; ++p) a += ld(part + p * Nd + o);
    if (bias) a += ld(bias + c);
    if (res) a += ld(res + o);
    out[o] = a;
  }
  __syncthreads();
}

// Own rows' column sums of an N x d stream (a replicated bias's gradient partial), in row order. Ends with __syncthreads.
__device__ void colsum_rows(const float* x, int r0, int r1, int d, float* out) {
  for (int c = threadIdx.x; c < d; c += NT) { float a = 0.f; for (int r = r0; r < r1; ++r) a += ld(x + (size_t)r * d + c); out[c] = a; }
  __syncthreads();
}

// RoPE of head hh's columns (pairs i and i + hd / 2) over every row, in place, at position r mod T; sign -1 rotates back (the backward).
// Ends with __syncthreads.
__device__ void rope(float* x, int N, int T, int d, int hd, int hh, float theta, float sign) {
  const int half = hd / 2;
  for (int e = threadIdx.x; e < N * half; e += NT) {
    const int r = e / half, i = e % half;
    const float ang = (float)(r % T) * powf(theta, -(float)(2 * i) / (float)hd);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    sn *= sign;
    float* p = x + (size_t)r * d + hh * hd + i;
    const float a = ld(p), b = ld(p + half);
    p[0] = a * cs - b * sn;
    p[half] = a * sn + b * cs;
  }
  __syncthreads();
}

// Causal attention of sequence sq, head hh: o_i = sum over j <= i of p_ij v_j / sum p_ij with p_ij = exp(scale q_i . k_j - max), and lse_i =
// max + log(sum), in the bundle's layout lse[(sq H + hh) T + i]. One warp a query row: lanes over keys for the scores (kept in the warp's
// score row), then over columns for the output. Ends with __syncthreads.
__device__ void attn_fwd(const float* q, const float* k, const float* v, float* o, float* lse, int sq, int hh, int T, int H, int d, int hd,
                         float scale, const Smem& sm) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* sc = sm.rows + w * T;
  for (int i = w; i < T; i += NW) {
    const float* qi = q + (size_t)(sq * T + i) * d + hh * hd;
    float m = -INFINITY;
    for (int j = lane; j <= i; j += 32) {
      const float* kj = k + (size_t)(sq * T + j) * d + hh * hd;
      float s = 0.f;
      for (int c = 0; c < hd; ++c) s += ld(qi + c) * ld(kj + c);
      s *= scale;
      sc[j] = s;
      m = fmaxf(m, s);
    }
    m = warp_max(m);
    float l = 0.f;
    for (int j = lane; j <= i; j += 32) { const float p = expf(sc[j] - m); sc[j] = p; l += p; }
    l = warp_sum(l);
    __syncwarp();
    for (int c = lane; c < hd; c += 32) {
      float a = 0.f;
      for (int j = 0; j <= i; ++j) a += sc[j] * ld(v + (size_t)(sq * T + j) * d + hh * hd + c);
      o[(size_t)(sq * T + i) * d + hh * hd + c] = a / l;
    }
    if (lane == 0) lse[(size_t)(sq * H + hh) * T + i] = m + logf(l);
    __syncwarp();
  }
  __syncthreads();
}

// The causal attention's backward for sequence sq, head hh, as the bundle's three ops compute it: delta_i = o_i . do_i; with p_ij =
// exp(scale q_i . k_j - lse_i) and ds_ij = p_ij (do_i . v_j - delta_i): dq_i = scale sum over j <= i of ds_ij k_j, dk_j = scale sum over i >=
// j of ds_ij q_i, dv_j = sum over i >= j of p_ij do_i. One warp a query row for dq, then one a key row for dk and dv: lanes over the other
// index for the scores (kept in the warp's two score rows), then over columns. Ends with __syncthreads.
__device__ void attn_bwd(const float* q, const float* k, const float* v, const float* o, const float* dout, const float* lse, float* dq,
                         float* dk, float* dv, int sq, int hh, int T, int H, int d, int hd, float scale, const Smem& sm) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  auto at = [&](int i) { return (size_t)(sq * T + i) * d + hh * hd; };   // row i's first column of the head
  auto dot = [&](const float* a, const float* b) { float s = 0.f; for (int c = 0; c < hd; ++c) s += ld(a + c) * ld(b + c); return s; };
  const float* L = lse + (size_t)(sq * H + hh) * T;
  for (int i = w; i < T; i += NW) {
    float s = 0.f;
    for (int c = lane; c < hd; c += 32) s += ld(o + at(i) + c) * ld(dout + at(i) + c);
    s = warp_sum(s);
    if (lane == 0) sm.delta[i] = s;
  }
  __syncthreads();
  float* p0 = sm.rows + w * T;              // dq: ds_ij of the warp's query row; dk, dv: p_ij of its key row
  float* p1 = sm.rows + (NW + w) * T;       // dk, dv: ds_ij of its key row
  for (int i = w; i < T; i += NW) {
    const float li = ld(L + i), di = sm.delta[i];
    for (int j = lane; j <= i; j += 32) p0[j] = expf(dot(q + at(i), k + at(j)) * scale - li) * (dot(dout + at(i), v + at(j)) - di);
    __syncwarp();
    for (int c = lane; c < hd; c += 32) {
      float a = 0.f;
      for (int j = 0; j <= i; ++j) a += p0[j] * ld(k + at(j) + c);
      dq[at(i) + c] = a * scale;
    }
    __syncwarp();
  }
  for (int j = w; j < T; j += NW) {
    for (int i = j + lane; i < T; i += 32) {
      const float p = expf(dot(q + at(i), k + at(j)) * scale - ld(L + i));
      p0[i] = p;
      p1[i] = p * (dot(dout + at(i), v + at(j)) - sm.delta[i]);
    }
    __syncwarp();
    for (int c = lane; c < hd; c += 32) {
      float ak = 0.f, av = 0.f;
      for (int i = j; i < T; ++i) { av += p0[i] * ld(dout + at(i) + c); ak += p1[i] * ld(q + at(i) + c); }
      dk[at(j) + c] = ak * scale;
      dv[at(j) + c] = av;
    }
    __syncwarp();
  }
  __syncthreads();
}

// A block's deterministic sum of one float a thread (warp butterflies, then the warps in order); every thread gets the result.
__device__ float block_sum(float v, const Smem& sm) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) sm.red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < NW; ++i) t += sm.red[i];
  __syncthreads();
  return t;
}

// Member b's share of the shape.
struct Plan {
  int b, G, N, T, B, d, H, hd, F, V, L;
  int r0, r1;            // own rows
  int fp, f0, f1, nf;    // fc1 columns: a member's count, its own [f0, f1), the members that own any
  int vp, v0, v1, nv;    // the vocabulary likewise
  int nh;                // members that own a head
};
__device__ __forceinline__ Plan plan_of(const KvmFgptDesc& d, int b) {
  Plan p;
  p.b = b; p.G = d.G; p.N = d.N; p.T = d.T; p.B = d.B; p.d = d.d; p.H = d.H; p.hd = d.hd; p.F = d.F; p.V = d.V; p.L = d.L;
  p.r0 = (int)((long long)b * d.N / d.G); p.r1 = (int)((long long)(b + 1) * d.N / d.G);
  p.fp = (d.F + d.G - 1) / d.G; p.f0 = min(d.F, b * p.fp); p.f1 = min(d.F, (b + 1) * p.fp); p.nf = (d.F + p.fp - 1) / p.fp;
  p.vp = (d.V + d.G - 1) / d.G; p.v0 = min(d.V, b * p.vp); p.v1 = min(d.V, (b + 1) * p.vp); p.nv = (d.V + p.vp - 1) / p.vp;
  p.nh = min(d.G, d.H);
  return p;
}

// The elements of parameter `slot` member b owns, as rectangles [ra, rb) x [ca, cb) of its row-major layout (a vector as one row): fn(ra, rb,
// ca, cb, cols). Over the gang these partition every parameter.
template <class Fn>
__device__ void for_owned(const KvmFgptDesc& d, const Plan& p, int slot, Fn fn) {
  const int rows = d.prows[slot], cols = d.pcols[slot];
  const bool vec = rows == 1 || cols == 1;
  const int R = vec ? 1 : rows, C = vec ? rows * cols : cols;
  if (slot == 0) { fn(p.v0, p.v1, 0, C, C); return; }                                   // wte: vocabulary rows
  if (slot == kvm_fgpt_slot_head(p.L)) { fn(0, R, p.v0, p.v1, C); return; }             // head: vocabulary columns
  if (slot == kvm_fgpt_slot_nfg(p.L) || slot == kvm_fgpt_slot_nfb(p.L)) { if (p.b == 0) fn(0, R, 0, C, C); return; }
  switch ((slot - 1) % KVM_FGPT_PER_LAYER) {
    case KVM_FGPT_R_N1G: case KVM_FGPT_R_N1B: case KVM_FGPT_R_BO: case KVM_FGPT_R_N2G: case KVM_FGPT_R_N2B: case KVM_FGPT_R_B2:
      if (p.b == 0) fn(0, R, 0, C, C);
      return;
    case KVM_FGPT_R_WQ: case KVM_FGPT_R_WK: case KVM_FGPT_R_WV: case KVM_FGPT_R_BQ: case KVM_FGPT_R_BK: case KVM_FGPT_R_BV:
      for (int h = p.b; h < p.H; h += p.G) fn(0, R, h * p.hd, (h + 1) * p.hd, C);
      return;
    case KVM_FGPT_R_WO: for (int h = p.b; h < p.H; h += p.G) fn(h * p.hd, (h + 1) * p.hd, 0, C, C); return;
    case KVM_FGPT_R_W1: case KVM_FGPT_R_B1: fn(0, R, p.f0, p.f1, C); return;
    case KVM_FGPT_R_W2: fn(p.f0, p.f1, 0, C, C); return;
  }
}

template <class Fn>
__device__ void for_owned_elements(const KvmFgptDesc& d, const Plan& p, int slot, Fn fn) {
  for_owned(d, p, slot, [&](int ra, int rb, int ca, int cb, int cols) {
    const int w = cb - ca;
    for (int e = threadIdx.x; e < (rb - ra) * w; e += NT) fn((size_t)(ra + e / w) * cols + ca + e % w);
  });
}


// Phase marks, for a bed's clock (fgpt_bed.cu): `ph(mark)` is called by every thread at the end of each phase and of each barrier's wait,
// once a layer for the per-layer ones. The production path passes NoPhases, which compiles to nothing.
enum KvmFgptMark : int {
  MK_INPUT, MK_LN1, MK_X_H, MK_ATTN, MK_X_PROJ, MK_LN2, MK_X_H2, MK_MLP, MK_X_FC2, MK_RES, MK_LNF, MK_X_HF, MK_LOGITS, MK_X_CE, MK_CE,
  MK_HEADB, MK_X_DHF, MK_LNFB, MK_X_GFC2, MK_MLPB, MK_X_DH2, MK_LN2B, MK_X_GPROJ, MK_ATTNB, MK_X_DH, MK_LN1B, MK_X_GEMB, MK_EMBB,
  MK_X_NORM, MK_ADAMW, MK_X_END, MK_COUNT
};
struct NoPhases { __device__ __forceinline__ void operator()(int) const {} };

// Every step of the run as member `member` of the gang. `smem` is the member's shared memory (kvm_fgpt_smem_floats(N, T) floats). `abort` is
// the run's error word, which the gang barrier only reads. A member that finds a fault reports it once, from thread 0, through
// `fail(code)` (a KvmFgptError), and returns; the caller's report must set the error word (the executor's kvm_fail does), so every other
// member leaves at its next crossing. Returns false when the run failed. Every thread of the block calls it.
template <class FAIL, class PH>
__device__ bool run(const KvmFgptDesc& d, int member, float* smem, const int* abort, FAIL fail, PH& ph) {
  const Plan p = plan_of(d, member);
  const Smem sm = carve(smem, d.N, d.T);
  const int tid = threadIdx.x, N = p.N, T = p.T, D = p.d, H = p.H, hd = p.hd, F = p.F, V = p.V, L = p.L, b = p.b, G = p.G;
  const size_t Nd = (size_t)N * D;
  const KvmFgptScratch s = kvm_fgpt_scratch(N, D, H, F, V, L, G);
  if ((size_t)s.total > d.xch_floats || hd > 128 || hd % 2 || L > KVM_FGPT_MAXL || L < 1 || N % T) {
    if (tid == 0) fail((size_t)s.total > d.xch_floats ? KVM_FGPT_E_SCRATCH : KVM_FGPT_E_SHAPE);
    return false;
  }
  float* const X = d.xch;
  auto P = [&](int slot) { return d.p[slot]; };
  auto Gr = [&](int slot) { return d.g[slot]; };
  auto lay = [&](int l, int role) { return kvm_fgpt_slot_layer(l, role); };
  float* const part = X + s.part + (size_t)b * Nd;   // this member's partial
  float* const gx = X + s.gx;
  float* const dy = X + s.dy;
  const float inv_n = (float)(1.0 / N);
  auto cross = [&]() { return kvm_gang::barrier(d.ctl, G, abort); };
  const int t0 = __ldcg(d.step);
  for (int step = 0; step < d.steps; ++step) {
    const long long t = (long long)t0 + step;
    // ---- the step's tokens and targets (the bundle's data windows), checked as the bundle's ops check them
    if (tid == 0) *sm.flag = 0;
    __syncthreads();
    {
      const long long bi = t * d.ids_stride + d.ids_off, bt = t * d.tgt_stride + d.tgt_off;
      if (t + 1 < 1) { if (tid == 0) *sm.flag = KVM_FGPT_E_STEP; }
      else if (bi < 0 || bi + N > d.ids_len || bt < 0 || bt + N > d.tgt_len) { if (tid == 0) *sm.flag = KVM_FGPT_E_WINDOW; }
      else
        for (int r = tid; r < N; r += NT) {
          const int id = __ldcg(d.ids_src + bi + r), tg = __ldcg(d.tgt_src + bt + r);
          sm.ids[r] = id; sm.tgt[r] = tg;
          if (id < 0 || id >= V) atomicCAS(sm.flag, 0, KVM_FGPT_E_TOKEN);
          if (tg != -1 && (tg < 0 || tg >= V)) atomicCAS(sm.flag, 0, KVM_FGPT_E_TARGET);
        }
    }
    __syncthreads();
    if (*sm.flag) { if (tid == 0) fail(*sm.flag); return false; }
    // ---- forward: the embedding of own rows
    for (int e = tid; e < (p.r1 - p.r0) * D; e += NT) {
      const int r = p.r0 + e / D, c = e % D;
      X[s.x[0] + (size_t)r * D + c] = ld(P(0) + (size_t)sm.ids[r] * D + c);
    }
    __syncthreads();
    ph(MK_INPUT);
    for (int l = 0; l < L; ++l) {
      float* const x = X + s.x[l]; float* const h = X + s.h[l]; float* const q = X + s.q[l]; float* const k = X + s.k[l];
      float* const v = X + s.v[l]; float* const o = X + s.o[l]; float* const lse = X + s.lse[l]; float* const xm = X + s.xm[l];
      float* const h2 = X + s.h2[l]; float* const pre = X + s.pre[l]; float* const u = X + s.u[l];
      ln_fwd(x, P(lay(l, KVM_FGPT_R_N1G)), P(lay(l, KVM_FGPT_R_N1B)), h, X + s.m1[l], X + s.r1[l], p.r0, p.r1, D, d.ln_eps);
      ph(MK_LN1);
      if (!cross()) return false;                                                     // all-gather h
      ph(MK_X_H);
      bool first = true;
      for (int hh = b; hh < H; hh += G) {                                             // head owners: q, k, v, RoPE, attention, proj's partial
        const int c0 = hh * hd;
        float* const outs[3] = {q, k, v};
        const int wr[3] = {KVM_FGPT_R_WQ, KVM_FGPT_R_WK, KVM_FGPT_R_WV}, br[3] = {KVM_FGPT_R_BQ, KVM_FGPT_R_BK, KVM_FGPT_R_BV};
        for (int i = 0; i < 3; ++i) {
          float* const out = outs[i]; const float* const bias = P(lay(l, br[i]));
          gemm<false, false>(h, D, P(lay(l, wr[i])) + c0, D, N, hd, D, sm,
                             [&](int m, int n, float a) { out[(size_t)m * D + c0 + n] = a + ld(bias + c0 + n); });
        }
        rope(q, N, T, D, hd, hh, d.theta, 1.f);
        rope(k, N, T, D, hd, hh, d.theta, 1.f);
        for (int sq = 0; sq < p.B; ++sq) attn_fwd(q, k, v, o, lse, sq, hh, T, H, D, hd, d.scale, sm);
        const bool f = first;
        gemm<false, false>(o + c0, D, P(lay(l, KVM_FGPT_R_WO)) + (size_t)c0 * D, D, N, D, hd, sm,
                           [&](int m, int n, float a) { const size_t i = (size_t)m * D + n; part[i] = f ? a : ld(part + i) + a; });
        first = false;
      }
      ph(MK_ATTN);
      if (!cross()) return false;                                                     // reduce-scatter proj
      ph(MK_X_PROJ);
      rs_sum(X + s.part, p.nh, Nd, P(lay(l, KVM_FGPT_R_BO)), x, xm, p.r0, p.r1, D);
      ln_fwd(xm, P(lay(l, KVM_FGPT_R_N2G)), P(lay(l, KVM_FGPT_R_N2B)), h2, X + s.m2[l], X + s.r2[l], p.r0, p.r1, D, d.ln_eps);
      ph(MK_LN2);
      if (!cross()) return false;                                                     // all-gather h2
      ph(MK_X_H2);
      if (p.f1 > p.f0) {                                                              // fc1 owners: their columns, GELU, fc2's partial
        const int f0 = p.f0, w = p.f1 - p.f0;
        const float* const b1 = P(lay(l, KVM_FGPT_R_B1));
        gemm<false, false>(h2, D, P(lay(l, KVM_FGPT_R_W1)) + f0, F, N, w, D, sm, [&](int m, int n, float a) {
          const size_t i = (size_t)m * F + f0 + n;
          const float z = a + ld(b1 + f0 + n);
          pre[i] = z;
          u[i] = gelu(z);
        });
        gemm<false, false>(u + f0, F, P(lay(l, KVM_FGPT_R_W2)) + (size_t)f0 * D, D, N, D, w, sm,
                           [&](int m, int n, float a) { part[(size_t)m * D + n] = a; });
      }
      ph(MK_MLP);
      if (!cross()) return false;                                                     // reduce-scatter fc2
      ph(MK_X_FC2);
      rs_sum(X + s.part, p.nf, Nd, P(lay(l, KVM_FGPT_R_B2)), xm, X + s.x[l + 1], p.r0, p.r1, D);
      ph(MK_RES);
    }
    // ---- the final LayerNorm, the head, the cross-entropy across vocabulary shards
    const int nfg = kvm_fgpt_slot_nfg(L), nfb = kvm_fgpt_slot_nfb(L), head = kvm_fgpt_slot_head(L);
    float* const hf = X + s.hf; float* const logits = X + s.logits; float* const dlog = X + s.dlog;
    ln_fwd(X + s.x[L], P(nfg), P(nfb), hf, X + s.mf, X + s.rf, p.r0, p.r1, D, d.ln_eps);
    ph(MK_LNF);
    if (!cross()) return false;                                                       // all-gather hf
    ph(MK_X_HF);
    if (p.v1 > p.v0) {
      const int v0 = p.v0, v1 = p.v1;
      gemm<false, false>(hf, D, P(head) + v0, V, N, v1 - v0, D, sm, [&](int m, int n, float a) { logits[(size_t)m * V + v0 + n] = a; });
      for (int r = tid; r < N; r += NT) {
        const float* const lr = logits + (size_t)r * V;
        float mx = -INFINITY;
        for (int c = v0; c < v1; ++c) mx = fmaxf(mx, ld(lr + c));
        float se = 0.f;
        for (int c = v0; c < v1; ++c) se += expf(ld(lr + c) - mx);
        X[s.ce + ((size_t)b * N + r) * 2] = mx;
        X[s.ce + ((size_t)b * N + r) * 2 + 1] = se;
        const int tg = sm.tgt[r];
        if (tg >= v0 && tg < v1) X[s.tl + r] = ld(lr + tg);
      }
    }
    ph(MK_LOGITS);
    if (!cross()) return false;                                                       // the cross-entropy's row statistics
    ph(MK_X_CE);
    for (int r = tid; r < N; r += NT) {
      float M = -INFINITY;
      for (int j = 0; j < p.nv; ++j) M = fmaxf(M, ld(X + s.ce + ((size_t)j * N + r) * 2));
      float S = 0.f;
      for (int j = 0; j < p.nv; ++j) S += ld(X + s.ce + ((size_t)j * N + r) * 2 + 1) * expf(ld(X + s.ce + ((size_t)j * N + r) * 2) - M);
      sm.lse[r] = M + logf(S);
    }
    __syncthreads();
    if (p.v1 > p.v0) {
      const int v0 = p.v0, w = p.v1 - p.v0;
      for (int e = tid; e < N * w; e += NT) {
        const int r = e / w, c = v0 + e % w, tg = sm.tgt[r];
        const size_t i = (size_t)r * V + c;
        dlog[i] = tg == -1 ? 0.f : (expf(ld(logits + i) - sm.lse[r]) - (c == tg ? 1.f : 0.f)) * inv_n;
      }
    }
    if (b == 0 && tid == 0) {   // the loss, as softmax_ce_reduce writes it: the mean over targeted rows, then 1 / their count
      double sum = 0.0; int cnt = 0;
      for (int r = 0; r < N; ++r)
        if (sm.tgt[r] != -1) { sum += (double)(sm.lse[r] - ld(X + s.tl + r)); ++cnt; }
      float* const out = d.loss[step];
      out[0] = cnt ? (float)(sum / cnt) : 0.f;
      out[1] = cnt ? (float)(1.0 / cnt) : 0.f;
    }
    __syncthreads();
    ph(MK_CE);
    // ---- backward: the head
    if (p.v1 > p.v0) {
      const int v0 = p.v0, w = p.v1 - p.v0;
      float* const gh = Gr(head);
      gemm<true, false>(hf, D, dlog + v0, V, D, w, N, sm, [&](int m, int n, float a) { gh[(size_t)m * V + v0 + n] = a; });
      gemm<false, true>(dlog + v0, V, P(head) + v0, V, N, D, w, sm, [&](int m, int n, float a) { part[(size_t)m * D + n] = a; });
    }
    ph(MK_HEADB);
    if (!cross()) return false;                                                       // reduce-scatter dhf
    ph(MK_X_DHF);
    rs_sum(X + s.part, p.nv, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
    ln_bwd(X + s.x[L], dy, P(nfg), X + s.mf, X + s.rf, gx, false, X + s.lnp + ((size_t)(2 * L) * G + b) * 2 * D,
           X + s.lnp + ((size_t)(2 * L) * G + b) * 2 * D + D, p.r0, p.r1, D);
    colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * (L - 1) + 1) * G + b) * D);   // b2 of the last layer: the gradient at x[L]
    ph(MK_LNFB);
    for (int l = L - 1; l >= 0; --l) {
      float* const x = X + s.x[l]; float* const h = X + s.h[l]; float* const q = X + s.q[l]; float* const k = X + s.k[l];
      float* const v = X + s.v[l]; float* const o = X + s.o[l]; float* const lse = X + s.lse[l]; float* const xm = X + s.xm[l];
      float* const h2 = X + s.h2[l]; float* const pre = X + s.pre[l]; float* const u = X + s.u[l];
      float* const du = X + s.du; float* const dq = X + s.dq; float* const dk = X + s.dk; float* const dv = X + s.dv; float* const dout = X + s.dout;
      if (!cross()) return false;                                                     // all-gather gx (fc2's output gradient)
      ph(MK_X_GFC2);
      if (p.f1 > p.f0) {                                                              // fc1 owners: fc2's rows and fc1's columns of their units
        const int f0 = p.f0, w = p.f1 - p.f0;
        float* const gw2 = Gr(lay(l, KVM_FGPT_R_W2)); float* const gw1 = Gr(lay(l, KVM_FGPT_R_W1)); float* const gb1 = Gr(lay(l, KVM_FGPT_R_B1));
        gemm<true, false>(u + f0, F, gx, D, w, D, N, sm, [&](int m, int n, float a) { gw2[(size_t)(f0 + m) * D + n] = a; });
        gemm<false, true>(gx, D, P(lay(l, KVM_FGPT_R_W2)) + (size_t)f0 * D, D, N, w, D, sm, [&](int m, int n, float a) {
          const size_t i = (size_t)m * F + f0 + n;
          du[i] = a * gelu_grad(ld(pre + i));
        });
        for (int c = tid; c < w; c += NT) { float a = 0.f; for (int r = 0; r < N; ++r) a += ld(du + (size_t)r * F + f0 + c); gb1[f0 + c] = a; }
        __syncthreads();
        gemm<true, false>(h2, D, du + f0, F, D, w, N, sm, [&](int m, int n, float a) { gw1[(size_t)m * F + f0 + n] = a; });
        gemm<false, true>(du + f0, F, P(lay(l, KVM_FGPT_R_W1)) + f0, F, N, D, w, sm, [&](int m, int n, float a) { part[(size_t)m * D + n] = a; });
      }
      ph(MK_MLPB);
      if (!cross()) return false;                                                     // reduce-scatter dh2
      ph(MK_X_DH2);
      rs_sum(X + s.part, p.nf, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
      ln_bwd(xm, dy, P(lay(l, KVM_FGPT_R_N2G)), X + s.m2[l], X + s.r2[l], gx, true, X + s.lnp + ((size_t)(2 * l + 1) * G + b) * 2 * D,
             X + s.lnp + ((size_t)(2 * l + 1) * G + b) * 2 * D + D, p.r0, p.r1, D);
      colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * l) * G + b) * D);  // bo: the gradient at xm
      ph(MK_LN2B);
      if (!cross()) return false;                                                     // all-gather gx (proj's output gradient)
      ph(MK_X_GPROJ);
      bool first = true;
      for (int hh = b; hh < H; hh += G) {                                             // head owners: proj's rows, attention, q, k, v of their heads
        const int c0 = hh * hd;
        float* const gwo = Gr(lay(l, KVM_FGPT_R_WO));
        gemm<true, false>(o + c0, D, gx, D, hd, D, N, sm, [&](int m, int n, float a) { gwo[(size_t)(c0 + m) * D + n] = a; });
        gemm<false, true>(gx, D, P(lay(l, KVM_FGPT_R_WO)) + (size_t)c0 * D, D, N, hd, D, sm,
                          [&](int m, int n, float a) { dout[(size_t)m * D + c0 + n] = a; });
        for (int sq = 0; sq < p.B; ++sq) attn_bwd(q, k, v, o, dout, lse, dq, dk, dv, sq, hh, T, H, D, hd, d.scale, sm);
        rope(dq, N, T, D, hd, hh, d.theta, -1.f);
        rope(dk, N, T, D, hd, hh, d.theta, -1.f);
        const float* const grads[3] = {dq, dk, dv};
        const int wr[3] = {KVM_FGPT_R_WQ, KVM_FGPT_R_WK, KVM_FGPT_R_WV}, br[3] = {KVM_FGPT_R_BQ, KVM_FGPT_R_BK, KVM_FGPT_R_BV};
        for (int i = 0; i < 3; ++i) {
          const float* const gi = grads[i];
          float* const gb = Gr(lay(l, br[i])); float* const gw = Gr(lay(l, wr[i]));
          for (int c = tid; c < hd; c += NT) { float a = 0.f; for (int r = 0; r < N; ++r) a += ld(gi + (size_t)r * D + c0 + c); gb[c0 + c] = a; }
          gemm<true, false>(h, D, gi + c0, D, D, hd, N, sm, [&](int m, int n, float a) { gw[(size_t)m * D + c0 + n] = a; });
          const bool f = first && i == 0;
          gemm<false, true>(gi + c0, D, P(lay(l, wr[i])) + c0, D, N, D, hd, sm,
                            [&](int m, int n, float a) { const size_t j = (size_t)m * D + n; part[j] = f ? a : ld(part + j) + a; });
        }
        first = false;
      }
      ph(MK_ATTNB);
      if (!cross()) return false;                                                     // reduce-scatter dh
      ph(MK_X_DH);
      rs_sum(X + s.part, p.nh, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
      ln_bwd(x, dy, P(lay(l, KVM_FGPT_R_N1G)), X + s.m1[l], X + s.r1[l], gx, true, X + s.lnp + ((size_t)(2 * l) * G + b) * 2 * D,
             X + s.lnp + ((size_t)(2 * l) * G + b) * 2 * D + D, p.r0, p.r1, D);
      if (l > 0) colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * (l - 1) + 1) * G + b) * D);   // b2 of layer l - 1: the gradient at x[l]
      ph(MK_LN1B);
    }
    if (!cross()) return false;                                                       // all-gather gx (the embedding's output gradient)
    ph(MK_X_GEMB);
    // ---- the embedding's gradient: each owned vocabulary row sums the gradients of its tokens' rows, in row order
    {
      float* const ge = Gr(0);
      const int v0 = p.v0, w = p.v1 - p.v0;
      for (int e = tid; e < w * D; e += NT) {
        const int row = v0 + e / D, c = e % D;
        float a = 0.f;
        for (int r = 0; r < N; ++r) if (sm.ids[r] == row) a += ld(gx + (size_t)r * D + c);
        ge[(size_t)row * D + c] = a;
      }
    }
    if (b == 0) {   // the replicated parameters' gradients: every member's own-row partials, summed in member order
      for (int j = 0; j <= 2 * L; ++j) {
        const int sg = j == 2 * L ? nfg : lay(j / 2, j % 2 ? KVM_FGPT_R_N2G : KVM_FGPT_R_N1G), sb = sg + 1;   // gain and bias are adjacent slots
        for (int c = tid; c < 2 * D; c += NT) {
          float a = 0.f;
          for (int m = 0; m < G; ++m) a += ld(X + s.lnp + ((size_t)j * G + m) * 2 * D + c);
          (c < D ? Gr(sg) : Gr(sb))[c % D] = a;
        }
      }
      for (int j = 0; j < 2 * L; ++j) {
        float* const gb = Gr(lay(j / 2, j % 2 ? KVM_FGPT_R_B2 : KVM_FGPT_R_BO));
        for (int c = tid; c < D; c += NT) { float a = 0.f; for (int m = 0; m < G; ++m) a += ld(X + s.rpart + ((size_t)j * G + m) * D + c); gb[c] = a; }
      }
    }
    __syncthreads();
    // ---- the grad-norm clip: each member's sum of squares of what it owns, one exchange, the members' sums in member order
    {
      float sq = 0.f;
      for (int slot = 0; slot < kvm_fgpt_nparams(L); ++slot) {
        const float* const g = Gr(slot);
        for_owned_elements(d, p, slot, [&](size_t e) { const float x = ld(g + e); sq += x * x; });
      }
      sq = block_sum(sq, sm);
      if (tid == 0) X[s.gn + b] = sq;
    }
    ph(MK_EMBB);
    if (!cross()) return false;                                                       // the grad norm's partials
    ph(MK_X_NORM);
    float norm2 = 0.f;
    for (int m = 0; m < G; ++m) norm2 += ld(X + s.gn + m);
    const float norm = sqrtf(norm2);
    const float coef = d.max_norm > 0.f ? fminf(1.f, d.max_norm / (norm + 1e-6f)) : 1.f;
    // ---- AdamW on the owned elements, in the bundle's float32 operation order, with the counter after its increment
    {
      const long long t1 = t + 1;
      const float b1 = d.beta1, b2 = d.beta2, lr = d.lr, eps = d.adam_eps;
      const float bc1 = __fsub_rn(1.f, (float)pow((double)b1, (double)t1)), bc2 = __fsub_rn(1.f, (float)pow((double)b2, (double)t1));
      const float ob1 = __fsub_rn(1.f, b1), ob2 = __fsub_rn(1.f, b2);
      for (int slot = 0; slot < kvm_fgpt_nparams(L); ++slot) {
        float* const pp = P(slot); float* const mm = d.m[slot]; float* const vv = d.v[slot]; const float* const gg = Gr(slot);
        const float lwd = __fmul_rn(lr, d.wd[slot]);
        for_owned_elements(d, p, slot, [&](size_t e) {
          const float gi = __fmul_rn(ld(gg + e), coef);
          const float mi = __fadd_rn(__fmul_rn(b1, ld(mm + e)), __fmul_rn(ob1, gi));
          const float vi = __fadd_rn(__fmul_rn(b2, ld(vv + e)), __fmul_rn(__fmul_rn(ob2, gi), gi));
          mm[e] = mi;
          vv[e] = vi;
          const float pe = ld(pp + e);
          const float decay = __fmul_rn(lwd, pe);
          const float upd = __fdiv_rn(__fmul_rn(lr, __fdiv_rn(mi, bc1)), __fadd_rn(__fsqrt_rn(__fdiv_rn(vi, bc2)), eps));
          pp[e] = __fsub_rn(__fsub_rn(pe, decay), upd);
        });
      }
    }
    ph(MK_ADAMW);
    if (!cross()) return false;                                                       // the step's end: every update visible
    ph(MK_X_END);
  }
  if (b == 0 && tid == 0) *d.step = t0 + d.steps;
  return true;
}

template <class FAIL>
__device__ __forceinline__ bool run(const KvmFgptDesc& d, int member, float* smem, const int* abort, FAIL fail) {
  NoPhases ph;
  return run(d, member, smem, abort, fail, ph);
}

}  // namespace kvm_fgpt
