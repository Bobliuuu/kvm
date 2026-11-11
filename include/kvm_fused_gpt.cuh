#pragma once
// The fused GPT step, device side (design D2): one member's share of every step of a fused run. kvm_fused_gpt.h has the descriptor, the
// parameter slots, the scratch layout and the shared-memory bound; bench/fused_gpt/d2_reference.py is the executable specification of the
// data flow, checked against the workload VM, and bench/fused_gpt/fgpt_bed.cu runs this code alone against the VM.
//
// Ownership, for member b of a gang of G:
//   rows      [b N / G, (b + 1) N / G) of every N x d stream: LayerNorms, residual adds, the reduce-scatters' sums;
//   units     attention runs as B H (sequence, head) units, unit u = s H + h on member u % G: q, k and v of its rows and head, RoPE,
//             attention, and proj's rows of the head, forward and backward. The head's parameters belong to member h % G, which sums the B
//             units' partials of their gradients, in sequence order, before the clip;
//   fcols     fc1's columns [b P, (b + 1) P) with P = ceil(F / G), and fc2's rows, the same units;
//   vocab     the head's columns and wte's rows [b P, (b + 1) P) with P = ceil(V / G);
//   member 0  the replicated parameters: the LayerNorms' gains and biases, proj's and fc2's biases. Every member adds its own rows' partials
//             of their gradients; member 0 sums those in member order.
// Every element of every parameter has exactly one owner, which computes (or sums) its gradient, counts it in the grad norm and applies AdamW.
// Exchanges: an all-gather is own rows written, then a barrier. A reduce-scatter is one partial a contributing member (or unit) written, a
// barrier, then each member sums its own rows of the partials in a fixed order. A step crosses 8 L + 6 barriers; the last one ends the step,
// so the next step's reads see every update. A value another member wrote is read through L2 (ld.global.cg), never a possibly stale L1 line.
//
// Compute: each phase stages its operands in shared memory and runs its products on the tensor cores (mma.sync m16n8k8 TF32, warps over
// 16 x 8 output tiles, so a narrow shard still spreads over all eight warps), in the bundle's numerics: matmul_tile's 1xTF32 for the linear
// layers (3xTF32 under its precise flag) and 3xTF32 for attention's products, as its attention ops run them. Softmax, LayerNorm, GELU, the
// cross-entropy and AdamW are FP32 as the bundle's ops write them. Parameters stay in their bundle buffers, updated in place by their owners.
#include <cmath>
#include <cstdint>
#include "kvm_fused_gpt.h"
#include "kvm_gang.cuh"

namespace kvm_fgpt {

constexpr int NT = KVM_FGPT_NT;
constexpr int NW = NT / 32;

__device__ __forceinline__ float ld(const float* p) { return __ldcg(p); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }

// The bundle's GELU (tanh form) and its derivative, as runtime/ref_gpt.py writes them.
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
__device__ __forceinline__ float gelu_grad(float x) {
  const float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * 0.7978845608028654f * (1.f + 3.f * 0.044715f * x * x);
}

// ---- shared memory: a bump allocator over the member's arena -------------------------------------------------------------------------
// Every thread runs the same takes, so every thread agrees on `over`; a phase takes its buffers, then checks `over` once before using any.
struct Arena {
  float* base;
  int cap, top;
  bool over;
  __device__ float* take(long long n) {
    const long long at = top, end = at + kvm_fgpt_r4(n);
    if (end > cap) { over = true; return base; }
    top = (int)end;
    return base + at;
  }
};

// ---- the tensor-core GEMM -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_tf32(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// C(M x N) = sum over k of A(m, k) B(k, n), with A(m, k) = A[TA ? k lda + m : m lda + k] and B(k, n) = B[TB ? n ldb + k : k ldb + n], both in
// shared memory and readable (finite, zero past K) over whole tiles: M rounded up to 16, N to 8, K to 8. `prec` takes 3xTF32 (lo hi + hi lo
// + hi hi), else 1xTF32. Warps take the 16 x 8 output tiles round robin, two at a time (two independent accumulators in flight); a tile
// accumulates in increasing k. An operand contiguous in k (A not
// transposed, B transposed) loads its fragments with ldmatrix (its rows 16-byte aligned: a row stride of 4 mod 8 floats keeps them
// conflict-free); the other kind by scalar loads. epi(m, n, value) runs once for every m < M, n < N. Every thread calls it; it ends with
// __syncthreads.
__device__ __forceinline__ unsigned tf32_rna(float x) { unsigned u; asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(u) : "f"(x)); return u; }
__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], const float* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldm_x2(unsigned (&r)[2], const float* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
// One tile's fragments at one k step, as raw float bits: A's four (rows g and g + 8, columns t and t + 4) and B's two (rows t and t + 4,
// column g). pa/pb: the lane's ldmatrix row address (k-contiguous operand) or its scalar base (the other kind).
template <bool TA, bool TB>
struct Frag {
  unsigned a[4], b[2];
  __device__ __forceinline__ void load(const float* pa, const float* pb, int lda, int ldb) {
    if (!TA) ldm_x4(a, pa);
    else { a[0] = __float_as_uint(pa[0]); a[1] = __float_as_uint(pa[8]); a[2] = __float_as_uint(pa[4 * lda]); a[3] = __float_as_uint(pa[4 * lda + 8]); }
    if (TB) ldm_x2(b, pb);
    else { b[0] = __float_as_uint(pb[0]); b[1] = __float_as_uint(pb[4 * ldb]); }
  }
};
__device__ __forceinline__ void mma_frag(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2], bool prec) {
  if (prec) {
    unsigned ah[4], al[4], bh[2], bl[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float x = __uint_as_float(a[i]); ah[i] = tf32_rna(x); al[i] = __float_as_uint(x - __uint_as_float(ah[i])); }
#pragma unroll
    for (int i = 0; i < 2; ++i) { const float x = __uint_as_float(b[i]); bh[i] = tf32_rna(x); bl[i] = __float_as_uint(x - __uint_as_float(bh[i])); }
    mma_tf32(c, al, bh[0], bh[1]);
    mma_tf32(c, ah, bl[0], bl[1]);
    mma_tf32(c, ah, bh[0], bh[1]);
  } else {
    const unsigned a2[4] = {tf32_rna(__uint_as_float(a[0])), tf32_rna(__uint_as_float(a[1])), tf32_rna(__uint_as_float(a[2])), tf32_rna(__uint_as_float(a[3]))};
    mma_tf32(c, a2, tf32_rna(__uint_as_float(b[0])), tf32_rna(__uint_as_float(b[1])));
  }
}
template <bool TA, bool TB, class Epi>
__device__ void gemm(const float* A, int lda, const float* B, int ldb, int M, int N, int K, bool prec, Epi epi) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tn = (N + 7) >> 3, tiles = ((M + 15) >> 4) * tn, ks = (K + 7) >> 3;
  const int sa = TA ? 8 * lda : 8, sb = TB ? 8 : 8 * ldb;              // pointer steps a k8 step
  // the lane's ldmatrix rows: A's x4 (rows 0-7 and 8-15 of the tile at k and k + 4), B's x2 (rows 0-7 of the tile at k and k + 4)
  const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ak = (lane >> 4) * 4, br = lane & 7, bk = ((lane >> 3) & 1) * 4;
  auto base_a = [&](int m0) { return A + (TA ? t * lda + m0 + g : (m0 + ar) * lda + ak); };
  auto base_b = [&](int n0) { return B + (TB ? (n0 + br) * ldb + bk : t * ldb + n0 + g); };
  for (int tile = warp; tile < tiles; tile += 2 * NW) {
    const int t1 = tile + NW;
    const bool two = t1 < tiles;
    const int m0 = (tile / tn) << 4, n0 = (tile % tn) << 3, m1 = two ? (t1 / tn) << 4 : m0, n1 = two ? (t1 % tn) << 3 : n0;
    const float* pa0 = base_a(m0); const float* pb0 = base_b(n0);
    const float* pa1 = base_a(m1); const float* pb1 = base_b(n1);
    float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ks; ++k) {
      Frag<TA, TB> f0;
      f0.load(pa0, pb0, lda, ldb);
      if (two) {
        Frag<TA, TB> f1;
        f1.load(pa1, pb1, lda, ldb);
        mma_frag(c1, f1.a, f1.b, prec);
        pa1 += sa; pb1 += sb;
      }
      mma_frag(c0, f0.a, f0.b, prec);
      pa0 += sa; pb0 += sb;
    }
    const int ma = m0 + g, mb = ma + 8, cc = n0 + 2 * t;
    if (ma < M && cc < N) epi(ma, cc, c0[0]);
    if (ma < M && cc + 1 < N) epi(ma, cc + 1, c0[1]);
    if (mb < M && cc < N) epi(mb, cc, c0[2]);
    if (mb < M && cc + 1 < N) epi(mb, cc + 1, c0[3]);
    if (two) {
      const int na = m1 + g, nb = na + 8, cd = n1 + 2 * t;
      if (na < M && cd < N) epi(na, cd, c1[0]);
      if (na < M && cd + 1 < N) epi(na, cd + 1, c1[1]);
      if (nb < M && cd < N) epi(nb, cd, c1[2]);
      if (nb < M && cd + 1 < N) epi(nb, cd + 1, c1[3]);
    }
  }
  __syncthreads();
}

// One staging copy: rows x cols floats of global memory (row stride sld) into shared memory (row stride dld), and zeros over the rest of
// [0, prows) x [0, pcols).
struct Copy {
  float* dst; const float* src; int dld; size_t sld; int rows, cols, prows, pcols;
  __device__ bool vec() const {   // 16-byte units: every column count, stride and address a multiple of 4 floats
    return (cols & 3) == 0 && (pcols & 3) == 0 && (dld & 3) == 0 && (sld & 3) == 0 && ((reinterpret_cast<size_t>(src) | reinterpret_cast<size_t>(dst)) & 15) == 0;
  }
  __device__ int units() const { return vec() ? prows * (pcols >> 2) : prows * pcols; }
};
// Asynchronous global-to-shared copies through L2 (cp.async; Ampere and later): `in` false writes zeros instead of reading (src-size 0).
__device__ __forceinline__ void cp16(float* dst, const float* src, bool in) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(in ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp4(float* dst, const float* src, bool in) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(in ? 4 : 0) : "memory");
}
// Every copy of a phase in one pass, so its operands cost one round trip: each thread issues an asynchronous copy for each of its units of
// every copy (16 bytes where the copy allows it, 4 otherwise; zeros over the padding), holding no registers for them, then one wait and one
// __syncthreads. The copies are unrolled at compile time, so the list stays in registers.
template <int NC>
__device__ void stage_all(const Copy (&cs)[NC]) {
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    const Copy& q = cs[i];
    if (q.vec()) {
      const int w = q.pcols >> 2, n = q.prows * w;
      for (int x = threadIdx.x; x < n; x += NT) {
        const int r = x / w, c = (x % w) << 2;
        const bool in = r < q.rows && c < q.cols;
        cp16(q.dst + r * q.dld + c, in ? q.src + (size_t)r * q.sld + c : q.src, in);
      }
    } else {
      const int n = q.prows * q.pcols;
      for (int x = threadIdx.x; x < n; x += NT) {
        const int r = x / q.pcols, c = x % q.pcols;
        const bool in = r < q.rows && c < q.cols;
        cp4(q.dst + r * q.dld + c, in ? q.src + (size_t)r * q.sld + c : q.src, in);
      }
    }
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
}
// One copy alone (the same pass).
__device__ __forceinline__ void stage(float* dst, int dld, const float* src, size_t sld, int rows, int cols, int prows, int pcols) {
  const Copy cs[1] = {{dst, src, dld, sld, rows, cols, prows, pcols}};
  stage_all(cs);
}
// Zeros a rows x cols block of shared memory (row stride ld): a GEMM's operand that an epilogue fills only in part. No sync.
__device__ __forceinline__ void zero(float* p, int ld, int rows, int cols) { for (int e = threadIdx.x; e < rows * cols; e += NT) p[(e / cols) * ld + e % cols] = 0.f; }

// Column sums of an R x C block of shared memory (row stride ld) into out[c] (global). With C < NT, NT / C threads share a column, each
// summing a contiguous chunk of rows in order, then one thread a column sums the chunks in order (`scr`: NT floats of shared memory), so the
// order is fixed for a given R and C. Ends with __syncthreads.
__device__ void colsum_smem(const float* X, int ld, int R, int C, float* out, float* scr) {
  if (C >= NT) {
    for (int c = threadIdx.x; c < C; c += NT) { float a = 0.f; for (int r = 0; r < R; ++r) a += X[r * ld + c]; out[c] = a; }
    __syncthreads();
    return;
  }
  const int K = min(NT / C, R), c = threadIdx.x % C, k = threadIdx.x / C;
  if (k < K) {
    const int r0 = k * R / K, r1 = (k + 1) * R / K;
    float a = 0.f;
    for (int r = r0; r < r1; ++r) a += X[r * ld + c];
    scr[k * C + c] = a;
  }
  __syncthreads();
  if (threadIdx.x < C) { float a = 0.f; for (int j = 0; j < K; ++j) a += scr[j * C + threadIdx.x]; out[threadIdx.x] = a; }
  __syncthreads();
}

// ---- row phases (own rows, global memory) ---------------------------------------------------------------------------------------------
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
  __syncthreads();
}

// LayerNorm backward of rows [r0, r1): dx = rstd (dy g - mean(dy g) - xh mean(dy g xh)) with xh = (x - mean) rstd, added to gx (the
// residual-stream gradient) when acc, stored otherwise; and this member's own-row partials of dgamma (dy xh, rows in order) into pg and of
// dbeta (dy) into pb. Ends with __syncthreads.
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

// The reduce-scatter's sum for rows [r0, r1): out[r] = (the partials of row r, in order) + bias + res[r]. With `units`, the partials are the
// H (sequence, head) units' T x d blocks of row r's sequence (unit s H + h at part + (s H + h) T d), else P members' N x d partials. Ends with
// __syncthreads.
__device__ void rs_sum(const float* part, bool units, int P, int T, size_t Nd, const float* bias, const float* res, float* out, int r0, int r1, int d) {
  for (int e = threadIdx.x; e < (r1 - r0) * d; e += NT) {
    const int c = e % d, r = r0 + e / d;
    const size_t o = (size_t)r * d + c;
    float a = 0.f;
    if (units) {
      const int s = r / T, i = r % T;
      for (int h = 0; h < P; ++h) a += ld(part + ((size_t)(s * P + h) * T + i) * d + c);
    } else {
      for (int m = 0; m < P; ++m) a += ld(part + m * Nd + o);
    }
    if (bias) a += ld(bias + c);
    if (res) a += ld(res + o);
    out[o] = a;
  }
  __syncthreads();
}

// Own rows' column sums of an N x d stream (a replicated bias's gradient partial), rows in order. Ends with __syncthreads.
__device__ void colsum_rows(const float* x, int r0, int r1, int d, float* out) {
  for (int c = threadIdx.x; c < d; c += NT) { float a = 0.f; for (int r = r0; r < r1; ++r) a += ld(x + (size_t)r * d + c); out[c] = a; }
  __syncthreads();
}

// RoPE's angles, once a run: cos and sin of position r times theta^(-2 i / hd), for r < T and i < hd / 2, into tab (cos) and tab + T hd / 2
// (sin), the expressions the bundle's rope ops evaluate. Ends with __syncthreads.
__device__ void rope_table(float* tab, int T, int hd, float theta) {
  const int half = hd / 2;
  for (int e = threadIdx.x; e < T * half; e += NT) {
    const int r = e / half, i = e % half;
    float sn, cs;
    sincosf((float)r * powf(theta, -(float)(2 * i) / (float)hd), &sn, &cs);
    tab[e] = cs;
    tab[T * half + e] = sn;
  }
  __syncthreads();
}
// RoPE of a T x hd block in shared memory (row stride ld; row i at position i), pairs i and i + hd / 2, in place; sign -1 rotates back.
// Ends with __syncthreads.
__device__ void rope(float* X, int ld_, int T, int hd, const float* tab, float sign) {
  const int half = hd / 2;
  for (int e = threadIdx.x; e < T * half; e += NT) {
    const int r = e / half, i = e % half;
    const float cs = tab[e], sn = sign * tab[T * half + e];
    float* p = X + r * ld_ + i;
    const float a = p[0], b = p[half];
    p[0] = a * cs - b * sn;
    p[half] = a * sn + b * cs;
  }
  __syncthreads();
}

// A block's deterministic sum of one value a thread (warp butterflies, then the warps in order); every thread gets the result. `red` holds
// NW values of the type.
template <class Tv>
__device__ Tv block_sum(Tv v, Tv* red) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  Tv t = 0;
  for (int i = 0; i < NW; ++i) t += red[i];
  __syncthreads();
  return t;
}

// ---- ownership ---------------------------------------------------------------------------------------------------------------------------
struct Plan {
  int b, G, N, T, B, d, H, hd, F, V, L;
  int r0, r1;            // own rows
  int fp, f0, f1, nf;    // fc1 columns: a member's count, its own [f0, f1), the members that own any
  int vp, v0, v1, nv;    // the vocabulary likewise
};
__device__ __forceinline__ Plan plan_of(const KvmFgptDesc& d, int b) {
  Plan p;
  p.b = b; p.G = d.G; p.N = d.N; p.T = d.T; p.B = d.B; p.d = d.d; p.H = d.H; p.hd = d.hd; p.F = d.F; p.V = d.V; p.L = d.L;
  p.r0 = (int)((long long)b * d.N / d.G); p.r1 = (int)((long long)(b + 1) * d.N / d.G);
  p.fp = (d.F + d.G - 1) / d.G; p.f0 = min(d.F, b * p.fp); p.f1 = min(d.F, (b + 1) * p.fp); p.nf = (d.F + p.fp - 1) / p.fp;
  p.vp = (d.V + d.G - 1) / d.G; p.v0 = min(d.V, b * p.vp); p.v1 = min(d.V, (b + 1) * p.vp); p.nv = (d.V + p.vp - 1) / p.vp;
  return p;
}

// Ownership of the parameters, three kinds, which together partition every element of every parameter over the gang:
//  * blocks: the MLP's (fc1's columns and bias, fc2's rows of the member's units), wte's vocabulary rows and the head's vocabulary columns,
//    as rectangles [ra, rb) x [ca, cb) of the slot's row-major layout (a vector as one row): rect(d, p, slot, fn(ra, rb, ca, cb, cols));
//  * attention: a slice of one flat index over every layer's every head's unit-gradient layout (kvm_fgpt_unit_grads), [a0, a1);
//  * replicated: a slice of one flat index over the 2 L + 1 LayerNorms' gains and biases, then the 2 L row-parallel biases, [q0, q1).
// Slices keep every member's share of the summing and of AdamW near equal, whichever members ran the units.
template <class Fn>
__device__ void rect(const KvmFgptDesc& d, const Plan& p, int slot, Fn fn) {
  const int rows = d.prows[slot], cols = d.pcols[slot];
  const bool vec = rows == 1 || cols == 1;
  const int R = vec ? 1 : rows, C = vec ? rows * cols : cols;
  if (slot == 0) { fn(p.v0, p.v1, 0, C, C); return; }                                   // wte: vocabulary rows
  if (slot == kvm_fgpt_slot_head(p.L)) { fn(0, R, p.v0, p.v1, C); return; }             // head: vocabulary columns
  if (slot >= kvm_fgpt_slot_nfg(p.L)) return;                                           // the final LayerNorm: replicated
  switch ((slot - 1) % KVM_FGPT_PER_LAYER) {
    case KVM_FGPT_R_W1: case KVM_FGPT_R_B1: fn(0, R, p.f0, p.f1, C); return;
    case KVM_FGPT_R_W2: fn(p.f0, p.f1, 0, C, C); return;
    default: return;                                                                    // attention or replicated
  }
}
__device__ __forceinline__ void slice_of(long long total, int b, int G, long long& lo, long long& hi) { lo = total * b / G; hi = total * (b + 1) / G; }
// Attention's flat index: layer, head, then the unit-gradient layout (dWq, dWk, dWv [d][hd]; dbq, dbk, dbv [hd]; dWo [hd][d]) to the slot
// and the element offset in it; `src` is the offset of the element in a unit's partial (the same layout).
__device__ __forceinline__ void attn_elem(const Plan& p, long long a, int& slot, size_t& off, int& l, int& hh, int& src) {
  const int D = p.d, hd = p.hd, hg = kvm_fgpt_unit_grads(D, hd);
  l = (int)(a / ((long long)p.H * hg));
  const int rem = (int)(a % ((long long)p.H * hg)), e = rem % hg, c0 = (rem / hg) * hd;
  hh = rem / hg;
  src = e;
  if (e < 3 * D * hd) { const int i = e / (D * hd), r = (e % (D * hd)) / hd, c = e % hd; slot = kvm_fgpt_slot_layer(l, KVM_FGPT_R_WQ + 2 * i); off = (size_t)r * D + c0 + c; }
  else if (e < 3 * D * hd + 3 * hd) { const int i = (e - 3 * D * hd) / hd, c = (e - 3 * D * hd) % hd; slot = kvm_fgpt_slot_layer(l, KVM_FGPT_R_BQ + 2 * i); off = (size_t)c0 + c; }
  else { const int f = e - 3 * D * hd - 3 * hd, r = f / D, c = f % D; slot = kvm_fgpt_slot_layer(l, KVM_FGPT_R_WO); off = (size_t)(c0 + r) * D + c; }
}
// The replicated flat index to the slot, the element offset, and its partials: in lnp (ln: a LayerNorm's gain or bias) or rpart (a bias),
// member 0's at `src`, the next member's `stride` floats on.
__device__ __forceinline__ void repl_elem(const Plan& p, long long q, int& slot, size_t& off, size_t& src, size_t& stride, bool& ln) {
  const int D = p.d, L = p.L, G = p.G;
  if (q < (long long)(2 * L + 1) * 2 * D) {
    const int j = (int)(q / (2 * D)), c = (int)(q % (2 * D));
    const int sg = j == 2 * L ? kvm_fgpt_slot_nfg(L) : kvm_fgpt_slot_layer(j / 2, j % 2 ? KVM_FGPT_R_N2G : KVM_FGPT_R_N1G);   // gain, then bias
    slot = c < D ? sg : sg + 1; off = (size_t)(c % D); src = (size_t)j * G * 2 * D + c; stride = (size_t)2 * D; ln = true;
  } else {
    const long long q1 = q - (long long)(2 * L + 1) * 2 * D;
    const int j = (int)(q1 / D), c = (int)(q1 % D);
    slot = kvm_fgpt_slot_layer(j / 2, j % 2 ? KVM_FGPT_R_B2 : KVM_FGPT_R_BO); off = (size_t)c; src = (size_t)j * G * D + c; stride = (size_t)D; ln = false;
  }
}

// Phase marks, for a bed's clock (fgpt_bed.cu): `ph(mark)` is called by every thread at the end of each phase and of each barrier's wait,
// once a layer for the per-layer ones. The production path passes NoPhases, which compiles to nothing.
enum KvmFgptMark : int {
  MK_INPUT, MK_LN1, MK_X_H, MK_UF_STAGE, MK_UF_QKV, MK_UF_ROPE, MK_UF_SCORES, MK_UF_SOFTMAX, MK_UF_PV, MK_UF_PROJ, MK_ATTN, MK_X_PROJ, MK_LN2,
  MK_X_H2, MK_MLP, MK_X_FC2, MK_RES, MK_LNF, MK_X_HF, MK_LOGITS, MK_X_CE, MK_CE, MK_HEADB, MK_X_DHF, MK_LNFB, MK_X_GFC2, MK_MLPB, MK_X_DH2,
  MK_LN2B, MK_X_GPROJ, MK_UB_STAGE, MK_UB_DO, MK_UB_STAGE2, MK_UB_P, MK_UB_DV_DS, MK_UB_DQ_DK, MK_UB_DB, MK_UB_DW, MK_UB_DH, MK_ATTNB, MK_X_DH,
  MK_LN1B, MK_X_GEMB, MK_EMBB, MK_SUMS, MK_SQUARES, MK_X_NORM, MK_ADAMW, MK_X_END, MK_COUNT
};
struct NoPhases { __device__ __forceinline__ void operator()(int) const {} };

// ---- the program -------------------------------------------------------------------------------------------------------------------------
// Every step of the run as member `member` of the gang. `smem` is the member's shared memory (d.smem_floats floats). `abort` is the run's
// error word, which the gang barrier only reads. A member that finds a fault reports it once, from thread 0, through `fail(code)` (a
// KvmFgptError), and returns; the caller's report must set the error word (the executor's kvm_fail does), so every other member leaves at its
// next crossing. Returns false when the run failed. Every thread of the block calls it.
template <class FAIL, class PH>
__device__ bool run(const KvmFgptDesc& d, int member, float* smem, const int* abort, FAIL fail, PH& ph) {
  const Plan p = plan_of(d, member);
  const int tid = threadIdx.x, N = p.N, T = p.T, D = p.d, H = p.H, hd = p.hd, F = p.F, V = p.V, L = p.L, b = p.b, G = p.G, B = p.B;
  const int T16 = kvm_fgpt_r16(T), N16 = kvm_fgpt_r16(N), U = B * H, hgsz = kvm_fgpt_unit_grads(D, hd);
  const size_t Nd = (size_t)N * D;
  const KvmFgptScratch s = kvm_fgpt_scratch(N, T, D, H, F, V, L, G);
  const bool prec = (d.mm_numerics & 16) != 0;                                        // matmul_tile's 3xTF32 flag
  const bool shape_ok = hd <= 128 && hd % 8 == 0 && D == H * hd && L >= 1 && L <= KVM_FGPT_MAXL && T > 0 && N % T == 0 && B == N / T &&
                        (d.mm_numerics & ~16) == 0 && D % 4 == 0;
  if ((size_t)s.total > d.xch_floats || !shape_ok) {
    if (tid == 0) fail((size_t)s.total > d.xch_floats ? KVM_FGPT_E_SCRATCH : KVM_FGPT_E_SHAPE);
    return false;
  }
  // the fixed head of shared memory (kvm_fgpt_smem_head), then one phase's buffers at a time from `base`
  int* const ids = reinterpret_cast<int*>(smem);
  int* const tgt = ids + kvm_fgpt_r4(N);
  float* const lse_ce = reinterpret_cast<float*>(tgt + kvm_fgpt_r4(N));
  float* const rtab = lse_ce + kvm_fgpt_r4(N);                                          // RoPE's cos and sin, T x hd / 2 each
  double* const red = reinterpret_cast<double*>(rtab + kvm_fgpt_r4((long long)T * hd)); // NW partials (8-byte aligned: the offsets are multiples of 4 floats, the arena 16-byte aligned)
  int* const flag = reinterpret_cast<int*>(red + NW);
  float* const scr = reinterpret_cast<float*>(flag + 4);                                // one float a thread
  float** const ptab = reinterpret_cast<float**>(scr + NT);                             // [4][MAXP]: p, m, v, g of every slot
  float* const wdt = reinterpret_cast<float*>(ptab + 4 * KVM_FGPT_MAXP);                // [MAXP] weight decay
  int* const blk = reinterpret_cast<int*>(wdt + kvm_fgpt_r4(KVM_FGPT_MAXP));            // [MAXBLK][8]: slot, first row, first column, width, row length, count, first index
  const int base = (int)kvm_fgpt_smem_head(N, T, hd);
  Arena ar{smem, d.smem_floats, base, d.smem_floats < base};
  auto phase = [&]() { ar.top = base; };
  auto check = [&]() {   // after a phase's takes: every thread agrees
    if (!ar.over) return true;
    if (tid == 0) fail(KVM_FGPT_E_SCRATCH);
    return false;
  };
  if (!check()) return false;
  float* const X = d.xch;
  auto P = [&](int slot) { return d.p[slot]; };
  auto Gr = [&](int slot) { return d.g[slot]; };
  auto lay = [&](int l, int role) { return kvm_fgpt_slot_layer(l, role); };
  float* const part = X + s.part + (size_t)b * Nd;   // this member's N x d partial (MLP, head)
  float* const gx = X + s.gx;
  float* const dy = X + s.dy;
  const float inv_n = (float)(1.0 / N);
  auto cross = [&]() { return kvm_gang::barrier(d.ctl, G, abort); };
  // row strides: q | k | v, a d-wide block, the T x T scores and a head-wide block (the last two are read transposed: whole 16-row tiles)
  const int ldq = kvm_fgpt_ldk(3 * hd), ldd = kvm_fgpt_ldk(D), ldT = kvm_fgpt_ldk(T16), ldh = kvm_fgpt_ldk(kvm_fgpt_r16(hd));
  rope_table(rtab, T, hd, d.theta);
  long long a0, a1, q0, q1;                                                             // this member's attention and replicated slices
  slice_of((long long)L * H * hgsz, b, G, a0, a1);
  slice_of((long long)(2 * L + 1) * 2 * D + (long long)2 * L * D, b, G, q0, q1);
  // every slot's pointers and weight decay, and this member's block rectangles in one flat index (fc1's columns, b1, fc2's rows a layer,
  // then wte's rows and the head's columns), so the end of a step reads no descriptor field per element
  const int np = kvm_fgpt_nparams(L);
  for (int i = tid; i < np; i += NT) { ptab[i] = d.p[i]; ptab[KVM_FGPT_MAXP + i] = d.m[i]; ptab[2 * KVM_FGPT_MAXP + i] = d.v[i]; ptab[3 * KVM_FGPT_MAXP + i] = d.g[i]; wdt[i] = d.wd[i]; }
  if (tid == 0) {
    int k = 0, at = 0;
    auto add = [&](int slot) {
      rect(d, p, slot, [&](int ra, int rb, int ca, int cb, int cols) {
        if (rb <= ra || cb <= ca) return;
        int* e = blk + 8 * k++;
        e[0] = slot; e[1] = ra; e[2] = ca; e[3] = cb - ca; e[4] = cols; e[5] = (rb - ra) * (cb - ca); e[6] = at;
        at += e[5];
      });
    };
    for (int l = 0; l < L; ++l) { add(kvm_fgpt_slot_layer(l, KVM_FGPT_R_W1)); add(kvm_fgpt_slot_layer(l, KVM_FGPT_R_B1)); add(kvm_fgpt_slot_layer(l, KVM_FGPT_R_W2)); }
    add(0);
    add(kvm_fgpt_slot_head(L));
    flag[1] = k; flag[2] = at;
  }
  __syncthreads();
  const int nblk = flag[1], nb = flag[2], na = (int)(a1 - a0), nq = (int)(q1 - q0);
  // element i of this member's owned list: its blocks [0, nb), its attention slice [nb, nb + na), its replicated slice after
  auto own = [&](int i, int& slot, size_t& off) {
    if (i < nb) {
      int k = 0;
      while (k + 1 < nblk && blk[8 * (k + 1) + 6] <= i) ++k;
      const int* e = blk + 8 * k, x = i - e[6];
      slot = e[0]; off = (size_t)(e[1] + x / e[3]) * e[4] + e[2] + x % e[3];
    } else if (i < nb + na) {
      int ll, hh, src;
      attn_elem(p, a0 + (i - nb), slot, off, ll, hh, src);
    } else {
      size_t src, stride; bool lnq;
      repl_elem(p, q0 + (i - nb - na), slot, off, src, stride, lnq);
    }
  };
  const int t0 = __ldcg(d.step);
  for (int step = 0; step < d.steps; ++step) {
    const long long t = (long long)t0 + step;
    // ---- the step's tokens and targets (the bundle's data windows), checked as the bundle's ops check them
    if (tid == 0) *flag = 0;
    __syncthreads();
    {
      const long long bi = t * d.ids_stride + d.ids_off, bt = t * d.tgt_stride + d.tgt_off;
      if (t + 1 < 1) { if (tid == 0) *flag = KVM_FGPT_E_STEP; }
      else if (bi < 0 || bi + N > d.ids_len || bt < 0 || bt + N > d.tgt_len) { if (tid == 0) *flag = KVM_FGPT_E_WINDOW; }
      else
        for (int r = tid; r < N; r += NT) {
          const int id = __ldcg(d.ids_src + bi + r), tg = __ldcg(d.tgt_src + bt + r);
          ids[r] = id; tgt[r] = tg;
          if (id < 0 || id >= V) atomicCAS(flag, 0, KVM_FGPT_E_TOKEN);
          if (tg != -1 && (tg < 0 || tg >= V)) atomicCAS(flag, 0, KVM_FGPT_E_TARGET);
        }
    }
    __syncthreads();
    if (*flag) { if (tid == 0) fail(*flag); return false; }
    // ---- forward: the embedding of own rows
    for (int e = tid; e < (p.r1 - p.r0) * D; e += NT) {
      const int r = p.r0 + e / D, c = e % D;
      X[s.x[0] + (size_t)r * D + c] = ld(P(0) + (size_t)ids[r] * D + c);
    }
    __syncthreads();
    ph(MK_INPUT);
    for (int l = 0; l < L; ++l) {
      float* const x = X + s.x[l]; float* const h = X + s.h[l]; float* const xm = X + s.xm[l]; float* const h2 = X + s.h2[l];
      ln_fwd(x, P(lay(l, KVM_FGPT_R_N1G)), P(lay(l, KVM_FGPT_R_N1B)), h, X + s.m1[l], X + s.r1[l], p.r0, p.r1, D, d.ln_eps);
      ph(MK_LN1);
      if (!cross()) return false;                                                     // all-gather h
      ph(MK_X_H);
      for (int u = b; u < U; u += G) {                                                // (sequence, head) units: q, k, v, RoPE, attention, proj's partial
        const int sq = u / H, hh = u % H, c0 = hh * hd, row0 = sq * T;
        phase();
        float* const QKV = ar.take((long long)T16 * ldq);                             // q | k | v, T x 3 hd
        float* const bias = ar.take(3 * hd);
        float* const lse = ar.take(T);
        float* const HS = ar.take((long long)T16 * ldd);
        float* const WC = ar.take((long long)kvm_fgpt_r8(D) * kvm_fgpt_ldmn(3 * hd));
        float* const S = ar.take((long long)T16 * ldT);
        float* const O = ar.take((long long)T16 * ldh);
        float* const WR = ar.take((long long)kvm_fgpt_r8(hd) * kvm_fgpt_ldmn(D));
        if (!check()) return false;
        {
          const int ldw = kvm_fgpt_ldmn(3 * hd);
          const Copy cs[8] = {{HS, h + (size_t)row0 * D, ldd, (size_t)D, T, D, T16, kvm_fgpt_r8(D)},
                              {WC, P(lay(l, KVM_FGPT_R_WQ)) + c0, ldw, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {WC + hd, P(lay(l, KVM_FGPT_R_WK)) + c0, ldw, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {WC + 2 * hd, P(lay(l, KVM_FGPT_R_WV)) + c0, ldw, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {bias, P(lay(l, KVM_FGPT_R_BQ)) + c0, 0, 0, 1, hd, 1, hd},
                              {bias + hd, P(lay(l, KVM_FGPT_R_BK)) + c0, 0, 0, 1, hd, 1, hd},
                              {bias + 2 * hd, P(lay(l, KVM_FGPT_R_BV)) + c0, 0, 0, 1, hd, 1, hd},
                              {WR, P(lay(l, KVM_FGPT_R_WO)) + (size_t)c0 * D, kvm_fgpt_ldmn(D), (size_t)D, hd, D, kvm_fgpt_r8(hd), D}};
          zero(QKV, ldq, T16, kvm_fgpt_r8(3 * hd));
          zero(S, ldT, T16, kvm_fgpt_r8(T));
          zero(O, ldh, T16, kvm_fgpt_r8(hd));
          stage_all(cs);
        }
        ph(MK_UF_STAGE);
        gemm<false, false>(HS, ldd, WC, kvm_fgpt_ldmn(3 * hd), T, 3 * hd, D, prec, [&](int m, int n, float a) { QKV[m * ldq + n] = a + bias[n]; });
        ph(MK_UF_QKV);
        rope(QKV, ldq, T, hd, rtab, 1.f);
        rope(QKV + hd, ldq, T, hd, rtab, 1.f);
        ph(MK_UF_ROPE);
        {                                                                              // q, k, v of the unit, kept for the backward
          float* const dsts[3] = {X + s.q[l], X + s.k[l], X + s.v[l]};
          const int q4 = 3 * hd / 4;
          for (int e = tid; e < T * q4; e += NT) {
            const int i = e / q4, c = (e % q4) * 4;
            *reinterpret_cast<float4*>(dsts[c / hd] + (size_t)(row0 + i) * D + c0 + c % hd) = *reinterpret_cast<const float4*>(QKV + i * ldq + c);
          }
        }
        // scores, 3xTF32 as the attention ops; then the softmax of each row over its causal keys, normalized in place
        gemm<false, true>(QKV, ldq, QKV + hd, ldq, T, T, hd, true, [&](int m, int n, float a) { S[m * ldT + n] = n <= m ? a * d.scale : 0.f; });
        ph(MK_UF_SCORES);
        {
          const int w = tid >> 5, lane = tid & 31;
          for (int i = w; i < T; i += NW) {
            float mx = -INFINITY;
            for (int j = lane; j <= i; j += 32) mx = fmaxf(mx, S[i * ldT + j]);
            mx = warp_max(mx);
            float l1 = 0.f;
            for (int j = lane; j <= i; j += 32) { const float e = expf(S[i * ldT + j] - mx); S[i * ldT + j] = e; l1 += e; }
            l1 = warp_sum(l1);
            for (int j = lane; j <= i; j += 32) S[i * ldT + j] /= l1;
            if (lane == 0) { lse[i] = mx + logf(l1); X[s.lse[l] + (size_t)(sq * H + hh) * T + i] = mx + logf(l1); }
          }
        }
        __syncthreads();
        ph(MK_UF_SOFTMAX);
        gemm<false, false>(S, ldT, QKV + 2 * hd, ldq, T, hd, T, true, [&](int m, int n, float a) {
          O[m * ldh + n] = a;
          X[s.o[l] + (size_t)(row0 + m) * D + c0 + n] = a;
        });
        ph(MK_UF_PV);
        float* const pu = X + s.part + (size_t)u * T * D;                             // the unit's partial of its sequence's rows
        gemm<false, false>(O, ldh, WR, kvm_fgpt_ldmn(D), T, D, hd, prec, [&](int m, int n, float a) { pu[(size_t)m * D + n] = a; });
        ph(MK_UF_PROJ);
      }
      ph(MK_ATTN);
      if (!cross()) return false;                                                     // reduce-scatter proj
      ph(MK_X_PROJ);
      rs_sum(X + s.part, true, H, T, Nd, P(lay(l, KVM_FGPT_R_BO)), x, xm, p.r0, p.r1, D);
      ln_fwd(xm, P(lay(l, KVM_FGPT_R_N2G)), P(lay(l, KVM_FGPT_R_N2B)), h2, X + s.m2[l], X + s.r2[l], p.r0, p.r1, D, d.ln_eps);
      ph(MK_LN2);
      if (!cross()) return false;                                                     // all-gather h2
      ph(MK_X_H2);
      if (p.f1 > p.f0) {                                                              // fc1 owners: their columns, GELU, fc2's partial
        const int f0 = p.f0, w = p.f1 - p.f0, ldw = kvm_fgpt_ldmn(w), ldu = kvm_fgpt_ldk(w);
        phase();
        float* const H2 = ar.take((long long)N16 * ldd);
        float* const W1 = ar.take((long long)kvm_fgpt_r8(D) * ldw);
        float* const Us = ar.take((long long)N16 * ldu);
        float* const W2 = ar.take((long long)kvm_fgpt_r8(w) * kvm_fgpt_ldmn(D));
        float* const b1 = ar.take(w);
        if (!check()) return false;
        {
          const Copy cs[4] = {{H2, h2, ldd, (size_t)D, N, D, N16, kvm_fgpt_r8(D)},
                              {W1, P(lay(l, KVM_FGPT_R_W1)) + f0, ldw, (size_t)F, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)},
                              {W2, P(lay(l, KVM_FGPT_R_W2)) + (size_t)f0 * D, kvm_fgpt_ldmn(D), (size_t)D, w, D, kvm_fgpt_r8(w), D},
                              {b1, P(lay(l, KVM_FGPT_R_B1)) + f0, 0, 0, 1, w, 1, w}};
          zero(Us, ldu, N16, kvm_fgpt_r8(w));
          stage_all(cs);
        }
        float* const pre = X + s.pre[l];
        gemm<false, false>(H2, ldd, W1, ldw, N, w, D, prec, [&](int m, int n, float a) {
          const float z = a + b1[n];
          pre[(size_t)m * F + f0 + n] = z;
          Us[m * ldu + n] = gelu(z);
        });
        gemm<false, false>(Us, ldu, W2, kvm_fgpt_ldmn(D), N, D, w, prec, [&](int m, int n, float a) { part[(size_t)m * D + n] = a; });
      }
      ph(MK_MLP);
      if (!cross()) return false;                                                     // reduce-scatter fc2
      ph(MK_X_FC2);
      rs_sum(X + s.part, false, p.nf, T, Nd, P(lay(l, KVM_FGPT_R_B2)), xm, X + s.x[l + 1], p.r0, p.r1, D);
      ph(MK_RES);
    }
    // ---- the final LayerNorm, the head, the cross-entropy across vocabulary shards
    const int nfg = kvm_fgpt_slot_nfg(L), nfb = kvm_fgpt_slot_nfb(L), head = kvm_fgpt_slot_head(L);
    float* const hf = X + s.hf; float* const logits = X + s.logits;
    ln_fwd(X + s.x[L], P(nfg), P(nfb), hf, X + s.mf, X + s.rf, p.r0, p.r1, D, d.ln_eps);
    ph(MK_LNF);
    if (!cross()) return false;                                                       // all-gather hf
    ph(MK_X_HF);
    if (p.v1 > p.v0) {
      const int v0 = p.v0, w = p.v1 - p.v0, ldw = kvm_fgpt_ldmn(w);
      phase();
      float* const HF = ar.take((long long)N16 * ldd);
      float* const WH = ar.take((long long)kvm_fgpt_r8(D) * ldw);
      if (!check()) return false;
      {
        const Copy cs[2] = {{HF, hf, ldd, (size_t)D, N, D, N16, kvm_fgpt_r8(D)}, {WH, P(head) + v0, ldw, (size_t)V, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}};
        stage_all(cs);
      }
      gemm<false, false>(HF, ldd, WH, ldw, N, w, D, prec, [&](int m, int n, float a) { logits[(size_t)m * V + v0 + n] = a; });
      for (int r = tid; r < N; r += NT) {
        const float* const lr = logits + (size_t)r * V;
        float mx = -INFINITY;
        for (int c = v0; c < p.v1; ++c) mx = fmaxf(mx, ld(lr + c));
        float se = 0.f;
        for (int c = v0; c < p.v1; ++c) se += expf(ld(lr + c) - mx);
        X[s.ce + ((size_t)b * N + r) * 2] = mx;
        X[s.ce + ((size_t)b * N + r) * 2 + 1] = se;
        const int tg = tgt[r];
        if (tg >= v0 && tg < p.v1) X[s.tl + r] = ld(lr + tg);
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
      lse_ce[r] = M + logf(S);
    }
    __syncthreads();
    if (b == 0) {   // the loss, as softmax_ce_reduce writes it: the mean over targeted rows (summed in float64), then 1 / their count
      double sum = 0.0, cnt = 0.0;
      for (int r = tid; r < N; r += NT)
        if (tgt[r] != -1) { sum += (double)(lse_ce[r] - ld(X + s.tl + r)); cnt += 1.0; }
      sum = block_sum(sum, red);
      cnt = block_sum(cnt, red);
      if (tid == 0) {
        float* const out = d.loss[step];
        out[0] = cnt > 0.0 ? (float)(sum / cnt) : 0.f;
        out[1] = cnt > 0.0 ? (float)(1.0 / cnt) : 0.f;
      }
    }
    ph(MK_CE);
    // ---- backward: the head
    if (p.v1 > p.v0) {
      const int v0 = p.v0, w = p.v1 - p.v0, ldl = kvm_fgpt_ldk(w);
      phase();
      float* const DL = ar.take((long long)N16 * ldl);
      float* const HF = ar.take((long long)N16 * kvm_fgpt_ldmn(D));
      float* const WT = ar.take((long long)kvm_fgpt_r8(D) * ldl);
      if (!check()) return false;
      zero(DL, ldl, N16, kvm_fgpt_r8(w));
      __syncthreads();
      for (int e = tid; e < N * w; e += NT) {
        const int r = e / w, c = e % w, tg = tgt[r];
        DL[r * ldl + c] = tg == -1 ? 0.f : (expf(ld(logits + (size_t)r * V + v0 + c) - lse_ce[r]) - (v0 + c == tg ? 1.f : 0.f)) * inv_n;
      }
      {
        const Copy cs[2] = {{HF, hf, kvm_fgpt_ldmn(D), (size_t)D, N, D, N16, D}, {WT, P(head) + v0, ldl, (size_t)V, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}};
        stage_all(cs);
      }
      float* const gh = Gr(head);
      gemm<true, false>(HF, kvm_fgpt_ldmn(D), DL, ldl, D, w, N, prec, [&](int m, int n, float a) { gh[(size_t)m * V + v0 + n] = a; });
      gemm<false, true>(DL, ldl, WT, ldl, N, D, w, prec, [&](int m, int n, float a) { part[(size_t)m * D + n] = a; });
    }
    ph(MK_HEADB);
    if (!cross()) return false;                                                       // reduce-scatter dhf
    ph(MK_X_DHF);
    rs_sum(X + s.part, false, p.nv, T, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
    ln_bwd(X + s.x[L], dy, P(nfg), X + s.mf, X + s.rf, gx, false, X + s.lnp + ((size_t)(2 * L) * G + b) * 2 * D,
           X + s.lnp + ((size_t)(2 * L) * G + b) * 2 * D + D, p.r0, p.r1, D);
    colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * (L - 1) + 1) * G + b) * D);   // b2 of the last layer: the gradient at x[L]
    ph(MK_LNFB);
    for (int l = L - 1; l >= 0; --l) {
      float* const x = X + s.x[l]; float* const h = X + s.h[l]; float* const xm = X + s.xm[l]; float* const h2 = X + s.h2[l];
      if (!cross()) return false;                                                     // all-gather gx (fc2's output gradient)
      ph(MK_X_GFC2);
      if (p.f1 > p.f0) {                                                              // fc1 owners: fc2's rows and fc1's columns of their units
        const int f0 = p.f0, w = p.f1 - p.f0, ldu = kvm_fgpt_ldk(w), ldum = kvm_fgpt_ldmn(w);
        phase();
        float* const PRE = ar.take((long long)N16 * ldu);
        float* const Ut = ar.take((long long)N16 * ldum);                              // gelu(pre), the A operand of dW2 (transposed)
        float* const DU = ar.take((long long)N16 * ldu);
        float* const XS = ar.take((long long)N16 * (ldd > kvm_fgpt_ldmn(D) ? ldd : kvm_fgpt_ldmn(D)));   // gx, then h2
        float* const WS = ar.take((long long)kvm_fgpt_r8(w) * ldd > (long long)kvm_fgpt_r8(D) * ldu ? (long long)kvm_fgpt_r8(w) * ldd
                                                                                                     : (long long)kvm_fgpt_r8(D) * ldu);
        if (!check()) return false;
        {
          const Copy cs[3] = {{PRE, X + s.pre[l] + f0, ldu, (size_t)F, N, w, N16, kvm_fgpt_r8(w)},
                              {XS, gx, ldd, (size_t)D, N, D, N16, kvm_fgpt_r8(D)},
                              {WS, P(lay(l, KVM_FGPT_R_W2)) + (size_t)f0 * D, ldd, (size_t)D, w, D, kvm_fgpt_r8(w), kvm_fgpt_r8(D)}};
          zero(DU, ldu, N16, kvm_fgpt_r8(w));
          stage_all(cs);
        }
        for (int e = tid; e < N16 * kvm_fgpt_r8(w); e += NT) { const int r = e / kvm_fgpt_r8(w), c = e % kvm_fgpt_r8(w); Ut[r * ldum + c] = r < N && c < w ? gelu(PRE[r * ldu + c]) : 0.f; }
        __syncthreads();
        float* const gw2 = Gr(lay(l, KVM_FGPT_R_W2)); float* const gw1 = Gr(lay(l, KVM_FGPT_R_W1)); float* const gb1 = Gr(lay(l, KVM_FGPT_R_B1));
        gemm<true, false>(Ut, ldum, XS, ldd, w, D, N, prec, [&](int m, int n, float a) { gw2[(size_t)(f0 + m) * D + n] = a; });
        gemm<false, true>(XS, ldd, WS, ldd, N, w, D, prec, [&](int m, int n, float a) { DU[m * ldu + n] = a * gelu_grad(PRE[m * ldu + n]); });
        colsum_smem(DU, ldu, N, w, gb1 + f0, scr);
        {
          const Copy cs[2] = {{XS, h2, kvm_fgpt_ldmn(D), (size_t)D, N, D, N16, D},
                              {WS, P(lay(l, KVM_FGPT_R_W1)) + f0, ldu, (size_t)F, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}};
          stage_all(cs);
        }
        gemm<true, false>(XS, kvm_fgpt_ldmn(D), DU, ldu, D, w, N, prec, [&](int m, int n, float a) { gw1[(size_t)m * F + f0 + n] = a; });
        gemm<false, true>(DU, ldu, WS, ldu, N, D, w, prec, [&](int m, int n, float a) { part[(size_t)m * D + n] = a; });
      }
      ph(MK_MLPB);
      if (!cross()) return false;                                                     // reduce-scatter dh2
      ph(MK_X_DH2);
      rs_sum(X + s.part, false, p.nf, T, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
      ln_bwd(xm, dy, P(lay(l, KVM_FGPT_R_N2G)), X + s.m2[l], X + s.r2[l], gx, true, X + s.lnp + ((size_t)(2 * l + 1) * G + b) * 2 * D,
             X + s.lnp + ((size_t)(2 * l + 1) * G + b) * 2 * D + D, p.r0, p.r1, D);
      colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * l) * G + b) * D);  // bo: the gradient at xm
      ph(MK_LN2B);
      if (!cross()) return false;                                                     // all-gather gx (proj's output gradient)
      ph(MK_X_GPROJ);
      for (int u = b; u < U; u += G) {                                                // (sequence, head) units: proj's rows, attention, q, k, v
        const int sq = u / H, hh = u % H, c0 = hh * hd, row0 = sq * T;
        float* const hg = X + s.hgp + ((size_t)l * U + u) * hgsz;                     // the unit's partial of its head's gradients
        phase();
        float* const DO = ar.take((long long)T16 * ldh);
        float* const delta = ar.take(T);
        float* const DQKV = ar.take((long long)T16 * ldq);
        float* const QKV = ar.take((long long)T16 * ldq);
        float* const lse = ar.take(T);
        float* const S = ar.take((long long)T16 * ldT);
        float* const O = ar.take((long long)T16 * ldh);
        const int mark = ar.top;                                                       // gx and Wo's rows, then h and W's columns
        float* const GS = ar.take((long long)T16 * ldd);
        float* const WO = ar.take((long long)kvm_fgpt_r8(hd) * ldd);
        if (!check()) return false;
        {
          const Copy cs[7] = {{GS, gx + (size_t)row0 * D, ldd, (size_t)D, T, D, T16, kvm_fgpt_r8(D)},
                              {WO, P(lay(l, KVM_FGPT_R_WO)) + (size_t)c0 * D, ldd, (size_t)D, hd, D, kvm_fgpt_r8(hd), kvm_fgpt_r8(D)},
                              {O, X + s.o[l] + (size_t)row0 * D + c0, ldh, (size_t)D, T, hd, T16, kvm_fgpt_r8(hd)},
                              {QKV, X + s.q[l] + (size_t)row0 * D + c0, ldq, (size_t)D, T, hd, T16, hd},
                              {QKV + hd, X + s.k[l] + (size_t)row0 * D + c0, ldq, (size_t)D, T, hd, T16, hd},
                              {QKV + 2 * hd, X + s.v[l] + (size_t)row0 * D + c0, ldq, (size_t)D, T, hd, T16, hd},
                              {lse, X + s.lse[l] + (size_t)(sq * H + hh) * T, 0, 0, 1, T, 1, T}};
          zero(DO, ldh, T16, kvm_fgpt_r8(hd));
          zero(S, ldT, T16, kvm_fgpt_r8(T));
          zero(DQKV, ldq, T16, kvm_fgpt_r8(3 * hd));
          stage_all(cs);
        }
        ph(MK_UB_STAGE);
        gemm<true, false>(O, ldh, GS, ldd, hd, D, T, prec, [&](int m, int n, float a) { hg[3 * D * hd + 3 * hd + m * D + n] = a; });   // dWo rows
        gemm<false, true>(GS, ldd, WO, ldd, T, hd, D, prec, [&](int m, int n, float a) { DO[m * ldh + n] = a; });                         // do
        for (int i = tid; i < T; i += NT) { float a = 0.f; for (int c = 0; c < hd; ++c) a += O[i * ldh + c] * DO[i * ldh + c]; delta[i] = a; }
        __syncthreads();
        ph(MK_UB_DO);
        ph(MK_UB_STAGE2);
        gemm<false, true>(QKV, ldq, QKV + hd, ldq, T, T, hd, true, [&](int m, int n, float a) { S[m * ldT + n] = n <= m ? expf(a * d.scale - lse[m]) : 0.f; });
        ph(MK_UB_P);
        gemm<true, false>(S, ldT, DO, ldh, T, hd, T, true, [&](int m, int n, float a) { DQKV[m * ldq + 2 * hd + n] = a; });                 // dv = P^T do
        gemm<false, true>(DO, ldh, QKV + 2 * hd, ldq, T, T, hd, true, [&](int m, int n, float a) { if (n <= m) S[m * ldT + n] *= a - delta[m]; });   // dS
        ph(MK_UB_DV_DS);
        gemm<false, false>(S, ldT, QKV + hd, ldq, T, hd, T, true, [&](int m, int n, float a) { DQKV[m * ldq + n] = a * d.scale; });           // dq
        gemm<true, false>(S, ldT, QKV, ldq, T, hd, T, true, [&](int m, int n, float a) { DQKV[m * ldq + hd + n] = a * d.scale; });             // dk
        ph(MK_UB_DQ_DK);
        rope(DQKV, ldq, T, hd, rtab, -1.f);
        rope(DQKV + hd, ldq, T, hd, rtab, -1.f);
        colsum_smem(DQKV, ldq, T, 3 * hd, hg + 3 * D * hd, scr);                                                                                  // dbq, dbk, dbv
        ph(MK_UB_DB);
        ar.top = mark;
        float* const HS = ar.take((long long)T16 * kvm_fgpt_ldmn(D));
        float* const WC = ar.take((long long)kvm_fgpt_r8(D) * ldq);
        if (!check()) return false;
        {
          const Copy cs[4] = {{HS, h + (size_t)row0 * D, kvm_fgpt_ldmn(D), (size_t)D, T, D, T16, D},
                              {WC, P(lay(l, KVM_FGPT_R_WQ)) + c0, ldq, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {WC + hd, P(lay(l, KVM_FGPT_R_WK)) + c0, ldq, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {WC + 2 * hd, P(lay(l, KVM_FGPT_R_WV)) + c0, ldq, (size_t)D, D, hd, kvm_fgpt_r8(D), hd}};
          stage_all(cs);
        }
        gemm<true, false>(HS, kvm_fgpt_ldmn(D), DQKV, ldq, D, 3 * hd, T, prec, [&](int m, int n, float a) { hg[(n / hd) * D * hd + m * hd + n % hd] = a; });   // dWq, dWk, dWv
        ph(MK_UB_DW);
        float* const pu = X + s.part + (size_t)u * T * D;
        gemm<false, true>(DQKV, ldq, WC, ldq, T, D, 3 * hd, prec, [&](int m, int n, float a) { pu[(size_t)m * D + n] = a; });               // dh partial
        ph(MK_UB_DH);
      }
      ph(MK_ATTNB);
      if (!cross()) return false;                                                     // reduce-scatter dh
      ph(MK_X_DH);
      rs_sum(X + s.part, true, H, T, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
      ln_bwd(x, dy, P(lay(l, KVM_FGPT_R_N1G)), X + s.m1[l], X + s.r1[l], gx, true, X + s.lnp + ((size_t)(2 * l) * G + b) * 2 * D,
             X + s.lnp + ((size_t)(2 * l) * G + b) * 2 * D + D, p.r0, p.r1, D);
      if (l > 0) colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * (l - 1) + 1) * G + b) * D);   // b2 of layer l - 1: the gradient at x[l]
      ph(MK_LN1B);
    }
    if (!cross()) return false;                                                       // all-gather gx (the embedding's output gradient)
    ph(MK_X_GEMB);
    // ---- the embedding's gradient: each owned vocabulary row sums the gradients of its tokens' rows, in row order, from gx staged once
    {
      const int v0 = p.v0, w = p.v1 - p.v0;
      phase();
      float* const GXS = ar.take((long long)N * D);
      if (!check()) return false;
      if (w > 0) {
        const Copy cs[1] = {{GXS, gx, D, (size_t)D, N, D, N, D}};
        stage_all(cs);
      }
      float* const ge = ptab[3 * KVM_FGPT_MAXP + 0];
      for (int e = tid; e < w * D; e += NT) {
        const int row = v0 + e / D, c = e % D;
        float a = 0.f;
        for (int r = 0; r < N; ++r) if (ids[r] == row) a += GXS[r * D + c];
        ge[(size_t)row * D + c] = a;                                                   // its square is counted with the blocks' below
      }
      __syncthreads();
    }
    ph(MK_EMBB);
    // ---- the gradients' squares over this member's blocks (written by its epilogues) and its attention slice (each element the sum of its B
    // units' partials, in sequence order, stored first), one flat index, four elements a thread in flight
    float sq = 0.f;
    for (int i0 = tid; i0 < nb + na; i0 += 4 * NT) {
      float v[4];
      int sl[4];
      size_t off[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int i = i0 + u * NT;
        v[u] = 0.f;
        if (i >= nb + na) continue;
        if (i < nb) { own(i, sl[u], off[u]); v[u] = ld(ptab[3 * KVM_FGPT_MAXP + sl[u]] + off[u]); }
        else {
          int ll, hh, src;
          attn_elem(p, a0 + (i - nb), sl[u], off[u], ll, hh, src);
          const float* const part0 = X + s.hgp + ((size_t)ll * U + hh) * hgsz + src;
          for (int sqi = 0; sqi < B; ++sqi) v[u] += ld(part0 + (size_t)sqi * H * hgsz);
          ptab[3 * KVM_FGPT_MAXP + sl[u]][off[u]] = v[u];
        }
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) sq += v[u] * v[u];
    }
    // ---- the replicated slice: each element's G member partials, summed by 8 threads over fixed member chunks, the chunks in order
    {
      const int lane8 = tid & 7, grp = tid >> 3, ngrp = NT / 8;
      const int m0 = (int)((long long)lane8 * G / 8), m1 = (int)((long long)(lane8 + 1) * G / 8);
      for (long long q = q0 + grp; q - grp < q1; q += ngrp) {                            // every thread of a group runs the same trips
        const bool okq = q < q1;
        int sl = 0; size_t off = 0, src = 0, stride = 0; bool lnq = false;
        if (okq) repl_elem(p, q, sl, off, src, stride, lnq);
        const float* const base0 = X + (lnq ? s.lnp : s.rpart) + src;
        float a = 0.f;
        if (okq) for (int m = m0; m < m1; ++m) a += ld(base0 + (size_t)m * stride);
        for (int o = 1; o < 8; o <<= 1) a += __shfl_xor_sync(0xffffffffu, a, o);       // the 8 chunks, the same tree on every group
        if (okq && lane8 == 0) { ptab[3 * KVM_FGPT_MAXP + sl][off] = a; sq += a * a; }
      }
    }
    ph(MK_SUMS);
    {
      const double tq = block_sum((double)sq, red);
      if (tid == 0) X[s.gn + b] = (float)tq;
    }
    ph(MK_SQUARES);
    if (!cross()) return false;                                                       // the grad norm's partials
    ph(MK_X_NORM);
    float norm2 = 0.f;
    for (int m = 0; m < G; ++m) norm2 += ld(X + s.gn + m);
    const float norm = sqrtf(norm2);
    const float coef = d.max_norm > 0.f ? fminf(1.f, d.max_norm / (norm + 1e-6f)) : 1.f;
    // ---- AdamW on the owned elements, in the bundle's float32 operation order, with the counter after its increment: one flat index over
    // the blocks, the attention slice and the replicated slice, four elements a thread in flight
    {
      const long long t1 = t + 1;
      const float b1 = d.beta1, b2 = d.beta2, lr = d.lr, eps = d.adam_eps;
      const float bc1 = __fsub_rn(1.f, (float)pow((double)b1, (double)t1)), bc2 = __fsub_rn(1.f, (float)pow((double)b2, (double)t1));
      const float ob1 = __fsub_rn(1.f, b1), ob2 = __fsub_rn(1.f, b2);
      const int n_own = nb + na + nq;
      for (int i0 = tid; i0 < n_own; i0 += 4 * NT) {
        int sl[4];
        size_t off[4];
        bool ok[4];
        float g4[4], m4[4], v4[4], p4[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int i = i0 + u * NT;
          ok[u] = i < n_own;
          if (!ok[u]) continue;
          own(i, sl[u], off[u]);
          g4[u] = ld(ptab[3 * KVM_FGPT_MAXP + sl[u]] + off[u]);
          m4[u] = ld(ptab[KVM_FGPT_MAXP + sl[u]] + off[u]);
          v4[u] = ld(ptab[2 * KVM_FGPT_MAXP + sl[u]] + off[u]);
          p4[u] = ld(ptab[sl[u]] + off[u]);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          if (!ok[u]) continue;
          const float gi = __fmul_rn(g4[u], coef);
          const float mi = __fadd_rn(__fmul_rn(b1, m4[u]), __fmul_rn(ob1, gi));
          const float vi = __fadd_rn(__fmul_rn(b2, v4[u]), __fmul_rn(__fmul_rn(ob2, gi), gi));
          const float decay = __fmul_rn(__fmul_rn(lr, wdt[sl[u]]), p4[u]);
          const float upd = __fdiv_rn(__fmul_rn(lr, __fdiv_rn(mi, bc1)), __fadd_rn(__fsqrt_rn(__fdiv_rn(vi, bc2)), eps));
          ptab[KVM_FGPT_MAXP + sl[u]][off[u]] = mi;
          ptab[2 * KVM_FGPT_MAXP + sl[u]][off[u]] = vi;
          ptab[sl[u]][off[u]] = __fsub_rn(__fsub_rn(p4[u], decay), upd);
        }
      }
    }
    __syncthreads();
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
