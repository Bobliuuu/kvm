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

// A load another member may have written, from L2 (a strong load: L1 never serves it stale). A volatile access rather than __ldcg, whose
// `asm volatile` the compiler must treat as a write to any memory: after each one it read every value it keeps in shared memory again.
__device__ __forceinline__ float ld(const float* p) { return *reinterpret_cast<const volatile float*>(p); }
// ld's four-wide form: the four floats at p (16-byte aligned) in one request, from L2 as ld's. It is an `asm volatile`, so after each one
// the compiler reads shared memory again: a loop around it takes the constants it uses as values first (KVM_FGPT_PHASE_CONSTS).
__device__ __forceinline__ float4 ld4(const float* p) {
  float4 r;
  asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(__cvta_generic_to_global(p)));
  return r;
}
// a + p(0) + p(1) + ... + p(n-1), the terms read from L2 and added in that order (the rolled loop's float result, bit for bit), with up to
// MAXN loads in flight before the first add: a sum waits for about one round trip to L2 per MAXN terms, where a rolled loop of dependent
// adds waits for each load in turn (ptxas unrolls one at most 4 deep). `addr(m)` is term m's address. Terms past n are never added.
template <int MAXN, class Addr>
__device__ __forceinline__ float sum_l2(int n, const Addr& addr, float a = 0.f) {
  for (int m0 = 0; m0 < n; m0 += MAXN) {
    float v[MAXN];
#pragma unroll
    for (int k = 0; k < MAXN; ++k) v[k] = m0 + k < n ? ld(addr(m0 + k)) : 0.f;
#pragma unroll
    for (int k = 0; k < MAXN; ++k)
      if (m0 + k < n) a += v[k];
  }
  return a;
}
// sum_l2's sum, bit for bit, of n <= 32 MAXK terms at consecutive words, for a sum every thread needs: lane l loads terms l, l + 32, ...,
// so a warp makes ceil(n / 32) requests of a line each where every thread loading every term made n of one word, and every lane adds the
// terms in index order through shuffles, so each lane returns the rolled loop's sum.
template <int MAXK, class Addr>
__device__ __forceinline__ float warp_sum_l2(int n, const Addr& addr) {
  const int lane = threadIdx.x & 31;
  float v[MAXK];
#pragma unroll
  for (int k = 0; k < MAXK; ++k) v[k] = 32 * k + lane < n ? ld(addr(32 * k + lane)) : 0.f;
  float a = 0.f;
#pragma unroll
  for (int k = 0; k < MAXK; ++k)
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      const float t = __shfl_sync(0xffffffffu, v[k], l);                            // every lane runs every shuffle: the warp stays converged
      if (32 * k + l < n) a += t;
    }
  return a;
}
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
#pragma unroll 1
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

// f(r, c) over a rows x units grid, unit x = r * units + c taken by thread x % NT in turn (x = threadIdx.x, + NT, ...), the order a flat
// loop takes them, with (r, c) stepped instead of divided out: integer division by a runtime value is a ~20-instruction sequence, and these
// loops ran one a unit (one division a call instead).
template <class F>
__device__ __forceinline__ void grid_loop(int rows, int units, F&& f) {
  if (rows <= 0 || units <= 0) return;
  const int dr = NT / units, dc = NT % units;
  for (int r = threadIdx.x / units, c = threadIdx.x % units; r < rows;) {
    f(r, c);
    r += dr; c += dc;
    if (c >= units) { c -= units; ++r; }
  }
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
    if (q.vec())
      grid_loop(q.prows, q.pcols >> 2, [&](int r, int c4) {
        const int c = c4 << 2;
        const bool in = r < q.rows && c < q.cols;
        cp16(q.dst + r * q.dld + c, in ? q.src + (size_t)r * q.sld + c : q.src, in);
      });
    else
      grid_loop(q.prows, q.pcols, [&](int r, int c) {
        const bool in = r < q.rows && c < q.cols;
        cp4(q.dst + r * q.dld + c, in ? q.src + (size_t)r * q.sld + c : q.src, in);
      });
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
__device__ __forceinline__ void zero(float* p, int ld, int rows, int cols) {
  if (((cols | ld) & 3) == 0 && (reinterpret_cast<size_t>(p) & 15) == 0)   // 16 bytes a store where every row allows it
    grid_loop(rows, cols >> 2, [&](int r, int c4) { *reinterpret_cast<float4*>(p + r * ld + (c4 << 2)) = make_float4(0.f, 0.f, 0.f, 0.f); });
  else
    grid_loop(rows, cols, [&](int r, int c) { p[r * ld + c] = 0.f; });
}

// Column sums of an R x C block of shared memory (row stride ld) into out[c] (global). With C < NT, NT / C threads share a column, each
// summing a contiguous chunk of rows in order, then one thread a column sums the chunks in order (`scr`: NT floats of shared memory), so the
// order is fixed for a given R and C. With `add`, each sum is added to out[c] (a later row chunk's, by the thread that wrote it before).
// Ends with __syncthreads.
__device__ void colsum_smem(const float* X, int ld, int R, int C, float* out, float* scr, bool add = false) {
  if (C >= NT) {
    for (int c = threadIdx.x; c < C; c += NT) { float a = 0.f; for (int r = 0; r < R; ++r) a += X[r * ld + c]; out[c] = add ? out[c] + a : a; }
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
  if (threadIdx.x < C) { float a = 0.f; for (int j = 0; j < K; ++j) a += scr[j * C + threadIdx.x]; out[threadIdx.x] = add ? out[threadIdx.x] + a : a; }
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

// The reduce-scatter's sum for rows [r0, r1): out[r] = (the partials of row r, in order) + bias + res[r]. With `units` (QB query blocks of TQ
// rows), the partials are the H attention items' TQ x d blocks of row r's block, heads in order (item (s H + h) QB + qb at part + item TQ
// d), else P members' N x d partials. Ends with __syncthreads.
__device__ void rs_sum(const float* part, bool units, int P, int T, int QB, size_t Nd, const float* bias, const float* res, float* out, int r0, int r1, int d) {
  for (int e = threadIdx.x; e < (r1 - r0) * d; e += NT) {
    const int c = e % d, r = r0 + e / d;
    const size_t o = (size_t)r * d + c;
    const float bv = bias ? ld(bias + c) : 0.f, rv = res ? ld(res + o) : 0.f;
    float a;
    if (units) {
      const int s = r / T, i = r % T, TQ = T / QB, qb = i / TQ, ii = i % TQ;
      a = sum_l2<8>(P, [&](int h) { return part + ((size_t)((s * P + h) * QB + qb) * TQ + ii) * d + c; });
    } else {
      a = sum_l2<16>(P, [&](int m) { return part + m * Nd + o; });
    }
    if (bias) a += bv;
    if (res) a += rv;
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
// RoPE of R rows x hd columns in shared memory (row stride ld; row r at position pos0 + r of a sequence of T), pairs i and i + hd / 2, in
// place; sign -1 rotates back. Ends with __syncthreads.
__device__ void rope(float* X, int ld_, int R, int pos0, int T, int hd, const float* tab, float sign) {
  const int half = hd / 2;
  for (int e = threadIdx.x; e < R * half; e += NT) {
    const int r = e / half, i = e % half, at = (pos0 + r) * half + i;
    const float cs = tab[at], sn = sign * tab[T * half + at];
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
  MK_INPUT, MK_LN1, MK_X_H, MK_UF_STAGE, MK_UF_QKV, MK_UF_ROPE, MK_X_KV, MK_UF_SCORES, MK_UF_SOFTMAX, MK_UF_PV, MK_UF_PROJ, MK_ATTN, MK_X_PROJ, MK_LN2,
  MK_X_H2, MK_MLP, MK_X_FC2, MK_RES, MK_LNF, MK_X_HF, MK_LOGITS, MK_X_CE, MK_CE, MK_HEADB, MK_X_DHF, MK_LNFB, MK_X_GFC2, MK_MLPB, MK_X_DH2,
  MK_LN2B, MK_X_GPROJ, MK_UB_STAGE, MK_UB_DO, MK_UB_STAGE2, MK_UB_P, MK_UB_DV_DS, MK_UB_DQ_DK, MK_UB_DB, MK_UB_ZBWA, MK_X_DKV, MK_UB_DW, MK_UB_DH, MK_UB_ZBWB, MK_ATTNB, MK_X_DH,
  MK_LN1B, MK_X_GEMB, MK_EMBB, MK_SUMS, MK_SQUARES, MK_X_NORM, MK_ADAMW, MK_X_END, MK_COUNT
};
struct NoPhases { __device__ __forceinline__ void operator()(int) const {} };

// ---- the gang's barrier -------------------------------------------------------------------------------------------------------------------
// The gang's crossing on its control words (d.ctl), in the form its size takes (kvm_gang::counted_form): the flag lines up to
// KVM_GANG_LINES_MAX members, the counted barrier past them. The program is instantiated per form (run<Counted>, the launch picking by G),
// so each crossing site carries one form's code: through kvm_gang::cross, both forms inline at every site behind a run-time branch added
// 1,020 instructions to this program and cost 16- and 32-member gangs 0.6-0.9% on the A100. A run starts from the words as the last run left
// them, so they need no reset between runs.
static_assert(KVM_GANG_LINES_MAX <= 32 * (KVM_FGPT_NT / 32), "the flag lines poll each line of 32 members with a warp of the block");

// The run's constants: thread 0 builds one copy a block after the scratch layout, and the program reads them by reference (run, below).
struct Ctx {
  Plan p;
  int QB, TQ, W, TQ16, hgsz, RC, RC16, base, ldq, ldd, ldh, np;
  int zbw, o0, o1, zg, zdb;                  // ZBW (the MLP backward): engaged; a doer's owners [o0, o1); owners a pass; gx's or h2's columns a block
  int zbwa;                                  // ZBW for the attention (KVM_FGPT_ZBW_ATTN): engaged; member W + w is item w's doer
  int nblk, nb, na, nq, n_own;                // the member's owned elements: its blocks, then its attention and replicated slices
  bool prec, ok;
  float inv_n;
  long long a0, a1, q0, q1, ostride;
  size_t Nd;
  int *ids, *tgt, *flag, *blk;
  float *lse_ce, *rtab, *scr, *wdt;
  double* red;
  float** ptab;
  float *X, *part, *gx, *dy;
  uint32_t *olist, *osrc;
};
static_assert(sizeof(Ctx) <= KVM_FGPT_CTX_FLOATS * sizeof(float) && alignof(Ctx) <= 16, "the run's constants outgrow their head slot");
static_assert(KVM_FGPT_LAYOUT_FLOATS % 4 == 0, "the constants after the layout are 16-byte aligned");
// A phase that loops (an attention item, a row chunk) takes the sizes, strides, plan and pointers it uses as values for its own body, shadowing run()'s
// references to the block's copy: read there at every use, each one was read again after every store to shared memory (gpt-small's step
// 5% slower on an RTX 5070), and held for the whole program they crowded sm_80's registers. Those a body does not use cost nothing.
#define KVM_FGPT_PHASE_CONSTS                                                                                                       \
  [[maybe_unused]] const int N = c.p.N, T = c.p.T, D = c.p.d, H = c.p.H, hd = c.p.hd, F = c.p.F, V = c.p.V, L = c.p.L, G = c.p.G,      \
      B = c.p.B, b = c.p.b, QB = c.QB, TQ = c.TQ, W = c.W, TQ16 = c.TQ16, hgsz = c.hgsz, RC = c.RC, RC16 = c.RC16, ldq = c.ldq,         \
      ldd = c.ldd, ldh = c.ldh;                                                                                                       \
  [[maybe_unused]] const size_t Nd = c.Nd;                                                                                           \
  [[maybe_unused]] const bool prec = c.prec;                                                                                         \
  [[maybe_unused]] const Plan p = c.p;                                                                                               \
  [[maybe_unused]] float* const X = c.X;                                                                                             \
  [[maybe_unused]] float* const part = c.part;                                                                                       \
  [[maybe_unused]] float* const gx = c.gx;                                                                                           \
  [[maybe_unused]] float* const dy = c.dy

// ---- the program -------------------------------------------------------------------------------------------------------------------------
// Every step of the run as member `member` of the gang. `smem` is the member's shared memory, `smem_floats` floats (at least
// kvm_fgpt_smem_floats of the shape: every phase checks, and fails the run with KVM_FGPT_E_SCRATCH rather than overrun). `abort` is the run's
// error word, which the gang barrier only reads. A member that finds a fault reports it once, from thread 0, through `fail(code)` (a
// KvmFgptError), and returns; the caller's report must set the error word (the executor's kvm_fail does), so every other member leaves at its
// next crossing. Returns false when the run failed. Every thread of the block calls it.
template <bool Counted, class FAIL, class PH>
__device__ bool run(const KvmFgptDesc& d, int member, float* smem, int smem_floats, const int* abort, FAIL fail, PH& ph) {
  const int tid = threadIdx.x;
  // the scratch layout and the run's constants, one copy a block at the head of shared memory, written once by thread 0
  if (smem_floats < KVM_FGPT_LAYOUT_FLOATS + KVM_FGPT_CTX_FLOATS) {
    if (tid == 0) fail(KVM_FGPT_E_SCRATCH);
    return false;
  }
  KvmFgptScratch& s = *reinterpret_cast<KvmFgptScratch*>(smem);
  Ctx& c = *reinterpret_cast<Ctx*>(smem + KVM_FGPT_LAYOUT_FLOATS);
  if (tid == 0) {
    const Plan pl = plan_of(d, member);
    c.p = pl;
    const int N = pl.N, T = pl.T, D = pl.d, H = pl.H, hd = pl.hd, F = pl.F, V = pl.V, L = pl.L, b = pl.b, G = pl.G, B = pl.B;
    // attention's work items: (sequence, head, query block), QB blocks of TQ rows a unit, item w = (s H + h) QB + qb on member w % G
    c.QB = d.qb; c.TQ = c.QB > 0 ? T / c.QB : 0; c.W = B * H * c.QB; c.TQ16 = kvm_fgpt_r16(c.TQ);
    c.hgsz = kvm_fgpt_unit_grads(D, hd); c.Nd = (size_t)N * D;
    // the N-wide phases' row chunk (kvm_fgpt_fit): the MLP's, the logits' and the head's backward walk N in chunks of RC rows, their
    // weight slices resident; RC is N where the whole fits, and then each phase runs as one chunk
    c.RC = d.rows < N ? d.rows : N; c.RC16 = kvm_fgpt_r16(c.RC);
    c.prec = (d.mm_numerics & 16) != 0;                                                  // matmul_tile's 3xTF32 flag
    c.ok = hd <= 128 && hd % 8 == 0 && D == H * hd && L >= 1 && L <= KVM_FGPT_MAXL && T > 0 && N % T == 0 && B == N / T &&
           (d.mm_numerics & ~16) == 0 && D % 4 == 0 && d.qb >= 1 && T % d.qb == 0 && (d.qb == 1 || (T / d.qb) % 16 == 0) &&
           d.rows >= 1 && d.rows <= N && (d.rows == N || d.rows % 16 == 0);
    if (c.ok) kvm_fgpt_scratch_into(s, N, T, D, H, F, V, L, G, d.qb);                     // only a shape it holds: L indexes its arrays
    else s.total = 0;
    // the fixed head of shared memory (kvm_fgpt_smem_head), then one phase's buffers at a time from `base`
    c.ids = reinterpret_cast<int*>(smem + KVM_FGPT_LAYOUT_FLOATS + KVM_FGPT_CTX_FLOATS);  // after the layout and these constants (16-byte aligned)
    c.tgt = c.ids + kvm_fgpt_r4(N);
    c.lse_ce = reinterpret_cast<float*>(c.tgt + kvm_fgpt_r4(N));
    c.rtab = c.lse_ce + kvm_fgpt_r4(N);                                                 // RoPE's cos and sin, T x hd / 2 each
    c.red = reinterpret_cast<double*>(c.rtab + kvm_fgpt_r4((long long)T * hd));        // NW partials (8-byte aligned: the offsets are multiples of 4 floats)
    c.flag = reinterpret_cast<int*>(c.red + NW);
    c.scr = reinterpret_cast<float*>(c.flag + 4);                                       // one float a thread
    c.ptab = reinterpret_cast<float**>(c.scr + NT);                                     // [4][MAXP]: p, m, v, g of every slot
    c.wdt = reinterpret_cast<float*>(c.ptab + 4 * KVM_FGPT_MAXP);                       // [MAXP] weight decay
    c.blk = reinterpret_cast<int*>(c.wdt + kvm_fgpt_r4(KVM_FGPT_MAXP));                 // [MAXBLK][8]: slot, first row, first column, width, row length, count, first index
    c.base = (int)kvm_fgpt_smem_head(N, T, hd);
    c.X = d.xch;
    c.part = c.X + s.part + (size_t)b * c.Nd;                                          // this member's N x d partial (MLP, head)
    c.gx = c.X + s.gx;
    c.dy = c.X + s.dy;
    c.inv_n = (float)(1.0 / N);
    // row strides: q | k | v, a d-wide block and a head-wide block (read transposed: whole 16-row tiles)
    c.ldq = kvm_fgpt_ldk(3 * hd); c.ldd = kvm_fgpt_ldk(D); c.ldh = kvm_fgpt_ldk(kvm_fgpt_r16(hd));
    slice_of((long long)L * H * c.hgsz, b, G, c.a0, c.a1);                              // this member's attention and replicated slices
    slice_of((long long)(2 * L + 1) * 2 * D + (long long)2 * L * D, b, G, c.q0, c.q1);
    c.np = kvm_fgpt_nparams(L);
    c.ostride = kvm_fgpt_own_stride(B, T, D, H, F, V, L, G);
    c.olist = reinterpret_cast<uint32_t*>(d.xch + s.olist) + (size_t)b * c.ostride;
    c.osrc = reinterpret_cast<uint32_t*>(d.xch + s.osrc) + (size_t)b * c.ostride;
    // ZBW (the MLP backward, kvm_fgpt_zbw_plan): a member without an attention item (b >= W) is a doer of a contiguous run of the nf owners
    const KvmFgptZbw z = kvm_fgpt_zbw_plan(G, c.W, pl.nf, pl.fp, D, c.RC16, (long long)smem_floats - c.base);
    c.zbw = z.zg > 0; c.zg = z.zg; c.zdb = z.zdb; c.o0 = c.o1 = 0;
    if (c.zbw && b >= c.W) { c.o0 = (int)((long long)(b - c.W) * pl.nf / z.doers); c.o1 = (int)((long long)(b - c.W + 1) * pl.nf / z.doers); }
    c.zbwa = KVM_FGPT_ZBW_ATTN && c.QB > 1 && c.W > 0 && 2 * c.W <= G;
  }
  __syncthreads();
  // Every constant below is a reference into that copy, so a phase reads what it uses there: a value read through shared memory is not held
  // across a __syncthreads, where held in registers for the whole program they rode through every phase (sm_120: 114 registers of values a
  // loop did not use; sm_80, whose phases need more of their own, spilled hundreds a pass).
  const Plan& p = c.p;
  const int &N = p.N, &T = p.T, &D = p.d, &H = p.H, &hd = p.hd, &V = p.V, &L = p.L, &b = p.b, &G = p.G;
  const int &QB = c.QB, &W = c.W, &hgsz = c.hgsz, &RC = c.RC, &RC16 = c.RC16, &base = c.base;
  const size_t& Nd = c.Nd;
  if ((size_t)s.total > d.xch_floats || !c.ok) {
    if (tid == 0) fail((size_t)s.total > d.xch_floats ? KVM_FGPT_E_SCRATCH : KVM_FGPT_E_SHAPE);
    return false;
  }
  int* const& ids = c.ids; int* const& tgt = c.tgt; float* const& lse_ce = c.lse_ce; float* const& rtab = c.rtab; double* const& red = c.red;
  int* const& flag = c.flag; float* const& scr = c.scr; float** const& ptab = c.ptab; float* const& wdt = c.wdt; int* const& blk = c.blk;
  Arena ar{smem, smem_floats, base, smem_floats < base};
  auto phase = [&]() { ar.top = base; };
  auto check = [&]() {   // after a phase's takes: every thread agrees
    if (!ar.over) return true;
    if (tid == 0) fail(KVM_FGPT_E_SCRATCH);
    return false;
  };
  if (!check()) return false;
  float* const& X = c.X;
  auto P = [&](int slot) { return d.p[slot]; };
  auto Gr = [&](int slot) { return d.g[slot]; };
  auto lay = [&](int l, int role) { return kvm_fgpt_slot_layer(l, role); };
  float* const& gx = c.gx;
  float* const& dy = c.dy;
  const float& inv_n = c.inv_n;
  [[maybe_unused]] unsigned epoch = 0;                                               // the flag lines' place in the gang's crossings, or
  [[maybe_unused]] unsigned long long target = 0;                                    // the count's: read before this member's first arrival
  if constexpr (Counted) target = kvm_gang::counted_base(d.ctl, G); else epoch = kvm_gang::lines_epoch(d.ctl, member);
  auto cross = [&]() {
    if constexpr (Counted) return kvm_gang::barrier_counted(d.ctl, G, target, abort);
    else return kvm_gang::barrier_lines(d.ctl, member, G, epoch, abort);
  };
  // row strides: q | k | v, a d-wide block, the T x T scores and a head-wide block (the last two are read transposed: whole 16-row tiles)
  const int& ldd = c.ldd;
  rope_table(rtab, T, hd, d.theta);
  const long long &a0 = c.a0, &q0 = c.q0, &q1 = c.q1;                                   // this member's attention slice's start, its replicated slice
  // every slot's pointers and weight decay, and this member's block rectangles in one flat index (fc1's columns, b1, fc2's rows a layer,
  // then wte's rows and the head's columns), so the end of a step reads no descriptor field per element
  const int& np = c.np;
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
    c.nblk = k; c.nb = at; c.na = (int)(c.a1 - c.a0); c.nq = (int)(c.q1 - c.q0); c.n_own = c.nb + c.na + c.nq;
  }
  __syncthreads();
  const int &nblk = c.nblk, &nb = c.nb, &na = c.na;
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
  // the owned elements decoded once a run: slot << 25 | offset, and an attention element's first item partial in hgp
  const int& n_own = c.n_own;
  const long long& ostride = c.ostride;
  uint32_t* const& olist = c.olist;
  uint32_t* const& osrc = c.osrc;
  {
    bool fits = n_own <= ostride;
    for (int sl = 0; sl < np; ++sl) fits = fits && (long long)d.prows[sl] * d.pcols[sl] < (1LL << 25);
    if (!fits) { if (tid == 0) fail(KVM_FGPT_E_SHAPE); return false; }
  }
  for (int i = tid; i < n_own; i += NT) {
    int slot; size_t off;
    own(i, slot, off);
    olist[i] = ((uint32_t)slot << 25) | (uint32_t)off;
    if (i >= nb && i < nb + na) {
      int ll, hh, src;
      attn_elem(p, a0 + (i - nb), slot, off, ll, hh, src);
      osrc[i] = (uint32_t)(((size_t)ll * W + (size_t)hh * QB) * hgsz + src);
    }
  }
  __syncthreads();
  const int t0 = __ldcg(d.step);
  // every step's AdamW bias corrections, in the expression the step used to evaluate (the float of the double pow, subtracted from 1 in
  // float): one pow a thread, all at once, by member 0; the gang's first crossing orders them before any member's first AdamW
  if (b == 0)
    for (int k = tid; k < 2 * d.steps; k += NT)
      d.bc[k] = __fsub_rn(1.f, (float)pow((double)((k & 1) ? d.beta2 : d.beta1), (double)((long long)t0 + (k >> 1) + 1)));
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
      // ---- attention, forward. Each item first projects its own TQ rows to q, k and v (RoPE'd) and keeps them; one exchange; then its queries
      // attend over the gathered keys and values of their prefix, and it writes proj's partial of its rows.
      for (int w = b; w < W; w += G) {
        KVM_FGPT_PHASE_CONSTS;
        const int u = w / QB, qb = w % QB, sq = u / H, hh = u % H, c0 = hh * hd, r0 = sq * T + qb * TQ;
        phase();
        float* const HS = ar.take((long long)TQ16 * ldd);
        float* const WC = ar.take((long long)kvm_fgpt_r8(D) * kvm_fgpt_ldmn(3 * hd));
        float* const bias = ar.take(3 * hd);
        float* const QKV = ar.take((long long)TQ16 * ldq);
        if (!check()) return false;
        {
          const int ldw = kvm_fgpt_ldmn(3 * hd);
          const Copy cs[7] = {{HS, h + (size_t)r0 * D, ldd, (size_t)D, TQ, D, TQ16, kvm_fgpt_r8(D)},
                              {WC, P(lay(l, KVM_FGPT_R_WQ)) + c0, ldw, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {WC + hd, P(lay(l, KVM_FGPT_R_WK)) + c0, ldw, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {WC + 2 * hd, P(lay(l, KVM_FGPT_R_WV)) + c0, ldw, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {bias, P(lay(l, KVM_FGPT_R_BQ)) + c0, 0, 0, 1, hd, 1, hd},
                              {bias + hd, P(lay(l, KVM_FGPT_R_BK)) + c0, 0, 0, 1, hd, 1, hd},
                              {bias + 2 * hd, P(lay(l, KVM_FGPT_R_BV)) + c0, 0, 0, 1, hd, 1, hd}};
          zero(QKV, ldq, TQ16, kvm_fgpt_r8(3 * hd));
          stage_all(cs);
        }
        ph(MK_UF_STAGE);
        gemm<false, false>(HS, ldd, WC, kvm_fgpt_ldmn(3 * hd), TQ, 3 * hd, D, prec, [&](int m, int n, float a) { QKV[m * ldq + n] = a + bias[n]; });
        ph(MK_UF_QKV);
        rope(QKV, ldq, TQ, qb * TQ, T, hd, rtab, 1.f);
        rope(QKV + hd, ldq, TQ, qb * TQ, T, hd, rtab, 1.f);
        {                                                                              // q, k and v of the item's rows, kept
          float* const dsts[3] = {X + s.q[l], X + s.k[l], X + s.v[l]};
          const int q4 = 3 * hd / 4;
          for (int e = tid; e < TQ * q4; e += NT) {
            const int i = e / q4, c = (e % q4) * 4;
            *reinterpret_cast<float4*>(dsts[c / hd] + (size_t)(r0 + i) * D + c0 + c % hd) = *reinterpret_cast<const float4*>(QKV + i * ldq + c);
          }
        }
        ph(MK_UF_ROPE);
      }
      if (QB > 1) { if (!cross()) return false; }                                       // all-gather q, k, v (one block a unit: the member's own)
      else __syncthreads();
      ph(MK_X_KV);
      for (int w = b; w < W; w += G) {
        KVM_FGPT_PHASE_CONSTS;
        const int u = w / QB, qb = w % QB, sq = u / H, hh = u % H, c0 = hh * hd, r0 = sq * T + qb * TQ, q0 = qb * TQ;
        const int KE = q0 + TQ, KE16 = kvm_fgpt_r16(KE), ldS = kvm_fgpt_ldk(KE16);   // the prefix's keys
        phase();
        float* const Q = ar.take((long long)TQ16 * ldh);
        float* const K = ar.take((long long)KE16 * ldh);
        float* const Vv = ar.take((long long)KE16 * ldh);
        float* const S = ar.take((long long)TQ16 * ldS);
        float* const O = ar.take((long long)TQ16 * ldh);
        float* const WR = ar.take((long long)kvm_fgpt_r8(hd) * kvm_fgpt_ldmn(D));
        if (!check()) return false;
        {
          const Copy cs[4] = {{Q, X + s.q[l] + (size_t)r0 * D + c0, ldh, (size_t)D, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                              {K, X + s.k[l] + (size_t)sq * T * D + c0, ldh, (size_t)D, KE, hd, KE16, kvm_fgpt_r8(hd)},
                              {Vv, X + s.v[l] + (size_t)sq * T * D + c0, ldh, (size_t)D, KE, hd, KE16, kvm_fgpt_r8(hd)},
                              {WR, P(lay(l, KVM_FGPT_R_WO)) + (size_t)c0 * D, kvm_fgpt_ldmn(D), (size_t)D, hd, D, kvm_fgpt_r8(hd), D}};
          zero(S, ldS, TQ16, KE16);
          zero(O, ldh, TQ16, kvm_fgpt_r8(hd));
          stage_all(cs);
        }
        // the queries' (positions q0 + m) scores over the prefix's keys, 3xTF32 as the attention ops, causal; then each row's softmax
        gemm<false, true>(Q, ldh, K, ldh, TQ, KE, hd, true, [&](int m, int n, float a) { S[m * ldS + n] = n <= q0 + m ? a * d.scale : 0.f; });
        ph(MK_UF_SCORES);
        {
          const int wp = tid >> 5, lane = tid & 31;
          for (int i = wp; i < TQ; i += NW) {
            const int last = q0 + i;
            float mx = -INFINITY;
            for (int j = lane; j <= last; j += 32) mx = fmaxf(mx, S[i * ldS + j]);
            mx = warp_max(mx);
            float l1 = 0.f;
            for (int j = lane; j <= last; j += 32) { const float e = expf(S[i * ldS + j] - mx); S[i * ldS + j] = e; l1 += e; }
            l1 = warp_sum(l1);
            for (int j = lane; j <= last; j += 32) S[i * ldS + j] /= l1;
            if (lane == 0) X[s.lse[l] + (size_t)(sq * H + hh) * T + q0 + i] = mx + logf(l1);
          }
        }
        __syncthreads();
        ph(MK_UF_SOFTMAX);
        gemm<false, false>(S, ldS, Vv, ldh, TQ, hd, KE, true, [&](int m, int n, float a) {
          O[m * ldh + n] = a;
          X[s.o[l] + (size_t)(r0 + m) * D + c0 + n] = a;
        });
        ph(MK_UF_PV);
        float* const pu = X + s.part + (size_t)w * TQ * D;                            // the item's partial of its rows
        gemm<false, false>(O, ldh, WR, kvm_fgpt_ldmn(D), TQ, D, hd, prec, [&](int m, int n, float a) { pu[(size_t)m * D + n] = a; });
        ph(MK_UF_PROJ);
      }
      ph(MK_ATTN);
      if (!cross()) return false;                                                     // reduce-scatter proj
      ph(MK_X_PROJ);
      rs_sum(X + s.part, true, H, T, QB, Nd, P(lay(l, KVM_FGPT_R_BO)), x, xm, p.r0, p.r1, D);
      ln_fwd(xm, P(lay(l, KVM_FGPT_R_N2G)), P(lay(l, KVM_FGPT_R_N2B)), h2, X + s.m2[l], X + s.r2[l], p.r0, p.r1, D, d.ln_eps);
      ph(MK_LN2);
      if (!cross()) return false;                                                     // all-gather h2
      ph(MK_X_H2);
      if (p.f1 > p.f0) {                                                              // fc1 owners: their columns, GELU, fc2's partial
        const int f0 = p.f0, w = p.f1 - p.f0, ldw = kvm_fgpt_ldmn(w), ldu = kvm_fgpt_ldk(w);
        phase();
        float* const H2 = ar.take((long long)RC16 * ldd);
        float* const W1 = ar.take((long long)kvm_fgpt_r8(D) * ldw);
        float* const Us = ar.take((long long)RC16 * ldu);
        float* const W2 = ar.take((long long)kvm_fgpt_r8(w) * kvm_fgpt_ldmn(D));
        float* const b1 = ar.take(w);
        if (!check()) return false;
        float* const pre = X + s.pre[l];
        for (int c0 = 0; c0 < N; c0 += RC) {                                          // the weights stage with the first chunk and stay
          KVM_FGPT_PHASE_CONSTS;
          const int rc = N - c0 < RC ? N - c0 : RC, rc16 = kvm_fgpt_r16(rc);
          const Copy rows = {H2, h2 + (size_t)c0 * D, ldd, (size_t)D, rc, D, rc16, kvm_fgpt_r8(D)};
          zero(Us, ldu, rc16, kvm_fgpt_r8(w));
          if (c0 == 0) {
            const Copy cs[4] = {rows,
                                {W1, P(lay(l, KVM_FGPT_R_W1)) + f0, ldw, (size_t)F, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)},
                                {W2, P(lay(l, KVM_FGPT_R_W2)) + (size_t)f0 * D, kvm_fgpt_ldmn(D), (size_t)D, w, D, kvm_fgpt_r8(w), D},
                                {b1, P(lay(l, KVM_FGPT_R_B1)) + f0, 0, 0, 1, w, 1, w}};
            stage_all(cs);
          } else {
            const Copy cs[1] = {rows};
            stage_all(cs);
          }
          gemm<false, false>(H2, ldd, W1, ldw, rc, w, D, prec, [&](int m, int n, float a) {
            const float z = a + b1[n];
            pre[(size_t)(c0 + m) * F + f0 + n] = z;
            Us[m * ldu + n] = gelu(z);
          });
          gemm<false, false>(Us, ldu, W2, kvm_fgpt_ldmn(D), rc, D, w, prec, [&](int m, int n, float a) { part[(size_t)(c0 + m) * D + n] = a; });
        }
      }
      ph(MK_MLP);
      if (!cross()) return false;                                                     // reduce-scatter fc2
      ph(MK_X_FC2);
      rs_sum(X + s.part, false, p.nf, T, 1, Nd, P(lay(l, KVM_FGPT_R_B2)), xm, X + s.x[l + 1], p.r0, p.r1, D);
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
      float* const HF = ar.take((long long)RC16 * ldd);
      float* const WH = ar.take((long long)kvm_fgpt_r8(D) * ldw);
      if (!check()) return false;
      for (int c0 = 0; c0 < N; c0 += RC) {                                            // the head's columns stage with the first chunk and stay
        KVM_FGPT_PHASE_CONSTS;
        const int rc = N - c0 < RC ? N - c0 : RC, rc16 = kvm_fgpt_r16(rc);
        const Copy rows = {HF, hf + (size_t)c0 * D, ldd, (size_t)D, rc, D, rc16, kvm_fgpt_r8(D)};
        if (c0 == 0) {
          const Copy cs[2] = {rows, {WH, P(head) + v0, ldw, (size_t)V, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}};
          stage_all(cs);
        } else {
          const Copy cs[1] = {rows};
          stage_all(cs);
        }
        gemm<false, false>(HF, ldd, WH, ldw, rc, w, D, prec, [&](int m, int n, float a) { logits[(size_t)(c0 + m) * V + v0 + n] = a; });
      }
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
      float* const DL = ar.take((long long)RC16 * ldl);
      float* const HF = ar.take((long long)RC16 * kvm_fgpt_ldmn(D));
      float* const WT = ar.take((long long)kvm_fgpt_r8(D) * ldl);
      if (!check()) return false;
      float* const gh = Gr(head);
      for (int c0 = 0; c0 < N; c0 += RC) {                                            // the head's gradient sums over the rows: first chunk writes, later add
        KVM_FGPT_PHASE_CONSTS;
        const int rc = N - c0 < RC ? N - c0 : RC, rc16 = kvm_fgpt_r16(rc);
        const bool add = c0 > 0;
        zero(DL, ldl, rc16, kvm_fgpt_r8(w));
        __syncthreads();
        for (int e = tid; e < rc * w; e += NT) {
          const int i = e / w, r = c0 + i, c = e % w, tg = tgt[r];
          DL[i * ldl + c] = tg == -1 ? 0.f : (expf(ld(logits + (size_t)r * V + v0 + c) - lse_ce[r]) - (v0 + c == tg ? 1.f : 0.f)) * inv_n;
        }
        const Copy rows = {HF, hf + (size_t)c0 * D, kvm_fgpt_ldmn(D), (size_t)D, rc, D, rc16, D};
        if (c0 == 0) {
          const Copy cs[2] = {rows, {WT, P(head) + v0, ldl, (size_t)V, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}};
          stage_all(cs);
        } else {
          const Copy cs[1] = {rows};
          stage_all(cs);
        }
        gemm<true, false>(HF, kvm_fgpt_ldmn(D), DL, ldl, D, w, rc, prec, [&](int m, int n, float a) { float* const o = gh + (size_t)m * V + v0 + n; *o = add ? ld(o) + a : a; });
        gemm<false, true>(DL, ldl, WT, ldl, rc, D, w, prec, [&](int m, int n, float a) { part[(size_t)(c0 + m) * D + n] = a; });
      }
    }
    ph(MK_HEADB);
    if (!cross()) return false;                                                       // reduce-scatter dhf
    ph(MK_X_DHF);
    rs_sum(X + s.part, false, p.nv, T, 1, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
    ln_bwd(X + s.x[L], dy, P(nfg), X + s.mf, X + s.rf, gx, false, X + s.lnp + ((size_t)(2 * L) * G + b) * 2 * D,
           X + s.lnp + ((size_t)(2 * L) * G + b) * 2 * D + D, p.r0, p.r1, D);
    colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * (L - 1) + 1) * G + b) * D);   // b2 of the last layer: the gradient at x[L]
    ph(MK_LNFB);
    // ZBW, a doer's pass over its owners' fc1 columns [o0 fp, o1 fp) of layer l, in the attention backward's waits: pass 0 dW2 and db1
    // (before the dK dV exchange), pass 1 dW1 (before reduce-scatter dh). Its inputs are what the owners would have used: gelu(pre), fc2's
    // output gradient (gxk, kept before ln_bwd rewrote gx), h2 and the owners' dU (du). Every element sums as the owner's would: the same
    // row chunks in order, and a GEMM tile's sum runs a k step at a time whatever the tile's place, so wider passes and gx's or h2's
    // columns in blocks change no bit; db1 sums each owner's columns alone (colsum_smem's order depends on the width).
    auto zbw_pass = [&](int l, int pass) -> bool {
      const int fp = p.fp, Fm = p.F, ca0 = min(Fm, c.o0 * fp), cb0 = min(Fm, c.o1 * fp), wmax = c.zg * fp, db = c.zdb;
      phase();
      float* const XS = ar.take((long long)RC16 * (kvm_fgpt_ldk(db) > kvm_fgpt_ldmn(db) ? kvm_fgpt_ldk(db) : kvm_fgpt_ldmn(db)));
      float* const PRE = ar.take((long long)RC16 * kvm_fgpt_ldk(wmax));
      float* const Ut = ar.take((long long)RC16 * kvm_fgpt_ldmn(wmax));
      float* const DUs = ar.take((long long)RC16 * kvm_fgpt_ldk(wmax));
      if (!check()) return false;
      float* const gw2 = Gr(lay(l, KVM_FGPT_R_W2)); float* const gw1 = Gr(lay(l, KVM_FGPT_R_W1)); float* const gb1 = Gr(lay(l, KVM_FGPT_R_B1));
      const float* const rows = pass == 0 ? X + s.gxk : X + s.h2[l];
      for (int c0 = 0; c0 < N; c0 += RC) {
        KVM_FGPT_PHASE_CONSTS;
        const int rc = N - c0 < RC ? N - c0 : RC, rc16 = kvm_fgpt_r16(rc);
        const bool add = c0 > 0;
        for (int ca = ca0; ca < cb0; ca += wmax) {
          const int cb = min(cb0, ca + wmax), wb = cb - ca, ldw = kvm_fgpt_ldk(wb), ldwm = kvm_fgpt_ldmn(wb);
          if (pass == 0) {
            const Copy cs[2] = {{PRE, X + s.pre[l] + (size_t)c0 * F + ca, ldw, (size_t)F, rc, wb, rc16, kvm_fgpt_r8(wb)},
                                {DUs, X + s.du + (size_t)c0 * F + ca, ldw, (size_t)F, rc, wb, rc16, kvm_fgpt_r8(wb)}};
            stage_all(cs);
            grid_loop(rc16, kvm_fgpt_r8(wb), [&](int r, int cc) { Ut[r * ldwm + cc] = r < rc && cc < wb ? gelu(PRE[r * ldw + cc]) : 0.f; });
            __syncthreads();
            for (int f0 = ca; f0 < cb; f0 += fp)                                       // db1, an owner at a time
              colsum_smem(DUs + (f0 - ca), ldw, rc, min(F, f0 + fp) - f0, gb1 + f0, scr, add);
          } else {
            stage(DUs, ldw, X + s.du + (size_t)c0 * F + ca, (size_t)F, rc, wb, rc16, kvm_fgpt_r8(wb));
          }
          for (int n0 = 0; n0 < D; n0 += db) {                                         // gx's (dW2) or h2's (dW1) columns, a block at a time
            const int dn = min(D, n0 + db) - n0;
            if (pass == 0) {
              stage(XS, kvm_fgpt_ldk(dn), rows + (size_t)c0 * D + n0, (size_t)D, rc, dn, rc16, kvm_fgpt_r8(dn));
              gemm<true, false>(Ut, ldwm, XS, kvm_fgpt_ldk(dn), wb, dn, rc, prec, [&](int m, int n, float a) {
                float* const o = gw2 + (size_t)(ca + m) * D + n0 + n; *o = add ? ld(o) + a : a; });
            } else {
              stage(XS, kvm_fgpt_ldmn(dn), rows + (size_t)c0 * D + n0, (size_t)D, rc, dn, rc16, dn);
              gemm<true, false>(XS, kvm_fgpt_ldmn(dn), DUs, ldw, dn, wb, rc, prec, [&](int m, int n, float a) {
                float* const o = gw1 + (size_t)(n0 + m) * F + ca + n; *o = add ? ld(o) + a : a; });
            }
          }
        }
      }
      return true;
    };
    // ZBW for the attention: item w's weight gradients that wait on nothing it computes later, run by its doer (member W + w) as the item
    // would have: pass 0, before the dK dV exchange, dWo from o's and gx's rows; pass 1, after it, dWq from h's rows and the dq the item
    // stored (after RoPE's backward), and dbq. The same copies, strides and GEMMs as the item's, into the item's own head-gradient slot,
    // so every element sums in the same order.
    auto zbwa_pass = [&](int l, int w, int pass) -> bool {
      KVM_FGPT_PHASE_CONSTS;
      const int u = w / QB, qb = w % QB, sq = u / H, hh = u % H, c0 = hh * hd, r0 = sq * T + qb * TQ;
      float* const hg = X + s.hgp + ((size_t)l * W + w) * hgsz;
      phase();
      if (pass == 0) {
        float* const O = ar.take((long long)TQ16 * ldh);
        float* const GS = ar.take((long long)TQ16 * ldd);
        if (!check()) return false;
        const Copy cs[2] = {{GS, gx + (size_t)r0 * D, ldd, (size_t)D, TQ, D, TQ16, kvm_fgpt_r8(D)},
                            {O, X + s.o[l] + (size_t)r0 * D + c0, ldh, (size_t)D, TQ, hd, TQ16, kvm_fgpt_r8(hd)}};
        stage_all(cs);
        gemm<true, false>(O, ldh, GS, ldd, hd, D, TQ, prec, [&](int m, int n, float a) { hg[3 * D * hd + 3 * hd + m * D + n] = a; });   // dWo rows
      } else {
        float* const DQ = ar.take((long long)TQ16 * ldh);
        float* const HQ = ar.take((long long)TQ16 * kvm_fgpt_ldmn(D));
        if (!check()) return false;
        const Copy cs[2] = {{DQ, X + s.dqk + (size_t)r0 * D + c0, ldh, (size_t)D, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                            {HQ, X + s.h[l] + (size_t)r0 * D, kvm_fgpt_ldmn(D), (size_t)D, TQ, D, TQ16, D}};
        stage_all(cs);
        colsum_smem(DQ, ldh, TQ, hd, hg + 3 * D * hd, scr);                                                                         // dbq
        gemm<true, false>(HQ, kvm_fgpt_ldmn(D), DQ, ldh, D, hd, TQ, prec, [&](int m, int n, float a) { hg[m * hd + n] = a; });          // dWq
      }
      return true;
    };
    for (int l = L - 1; l >= 0; --l) {
      float* const x = X + s.x[l]; float* const h = X + s.h[l]; float* const xm = X + s.xm[l]; float* const h2 = X + s.h2[l];
      if (!cross()) return false;                                                     // all-gather gx (fc2's output gradient)
      ph(MK_X_GFC2);
      if (p.f1 > p.f0 && c.zbw) {
        // ZBW (Zero Bubble's B/W split, inside the gang): the members without an attention item run the weight gradients (dW2, db1, dW1)
        // in the attention backward's waits (zbw_pass), so an owner runs only the input gradients the next crossing needs: dU, kept in du
        // for its doer, and fc2's dX partial. W2's rows, then W1's columns, in one buffer as below.
        const int f0 = p.f0, w = p.f1 - p.f0, ldu = kvm_fgpt_ldk(w);
        phase();
        float* const PRE = ar.take((long long)RC16 * ldu);
        float* const DU = ar.take((long long)RC16 * ldu);
        float* const XS = ar.take((long long)RC16 * ldd);                              // gx
        float* const WS = ar.take((long long)kvm_fgpt_r8(w) * ldd > (long long)kvm_fgpt_r8(D) * ldu ? (long long)kvm_fgpt_r8(w) * ldd
                                                                                                     : (long long)kvm_fgpt_r8(D) * ldu);
        if (!check()) return false;
        float* const du = X + s.du;
        for (int c0 = 0; c0 < N; c0 += RC) {
          KVM_FGPT_PHASE_CONSTS;
          const int rc = N - c0 < RC ? N - c0 : RC, rc16 = kvm_fgpt_r16(rc);
          {
            const Copy cs[3] = {{PRE, X + s.pre[l] + (size_t)c0 * F + f0, ldu, (size_t)F, rc, w, rc16, kvm_fgpt_r8(w)},
                                {XS, gx + (size_t)c0 * D, ldd, (size_t)D, rc, D, rc16, kvm_fgpt_r8(D)},
                                {WS, P(lay(l, KVM_FGPT_R_W2)) + (size_t)f0 * D, ldd, (size_t)D, w, D, kvm_fgpt_r8(w), kvm_fgpt_r8(D)}};
            zero(DU, ldu, rc16, kvm_fgpt_r8(w));
            stage_all(cs);
          }
          gemm<false, true>(XS, ldd, WS, ldd, rc, w, D, prec, [&](int m, int n, float a) {
            const float v = a * gelu_grad(PRE[m * ldu + n]);
            DU[m * ldu + n] = v;
            du[(size_t)(c0 + m) * F + f0 + n] = v;
          });
          stage(WS, ldu, P(lay(l, KVM_FGPT_R_W1)) + f0, (size_t)F, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w));
          gemm<false, true>(DU, ldu, WS, ldu, rc, D, w, prec, [&](int m, int n, float a) { part[(size_t)(c0 + m) * D + n] = a; });
        }
      } else if (p.f1 > p.f0) {                                                       // fc1 owners: fc2's rows and fc1's columns of their units
        const int f0 = p.f0, w = p.f1 - p.f0, ldu = kvm_fgpt_ldk(w), ldum = kvm_fgpt_ldmn(w);
        phase();
        float* const PRE = ar.take((long long)RC16 * ldu);
        float* const Ut = ar.take((long long)RC16 * ldum);                             // gelu(pre), the A operand of dW2 (transposed)
        float* const DU = ar.take((long long)RC16 * ldu);
        float* const XS = ar.take((long long)RC16 * (ldd > kvm_fgpt_ldmn(D) ? ldd : kvm_fgpt_ldmn(D)));   // gx, then h2
        float* const WS = ar.take((long long)kvm_fgpt_r8(w) * ldd > (long long)kvm_fgpt_r8(D) * ldu ? (long long)kvm_fgpt_r8(w) * ldd
                                                                                                     : (long long)kvm_fgpt_r8(D) * ldu);
        if (!check()) return false;
        float* const gw2 = Gr(lay(l, KVM_FGPT_R_W2)); float* const gw1 = Gr(lay(l, KVM_FGPT_R_W1)); float* const gb1 = Gr(lay(l, KVM_FGPT_R_B1));
        // the rows in chunks: dW2, db1 and dW1 sum over the rows, so the first chunk writes them and each later one adds, in order (each element
        // by the thread that wrote it: a GEMM's tiles map to threads by its M and N, the same every chunk)
        for (int c0 = 0; c0 < N; c0 += RC) {
          KVM_FGPT_PHASE_CONSTS;
          const int rc = N - c0 < RC ? N - c0 : RC, rc16 = kvm_fgpt_r16(rc);
          const bool add = c0 > 0;
          {
            const Copy cs[3] = {{PRE, X + s.pre[l] + (size_t)c0 * F + f0, ldu, (size_t)F, rc, w, rc16, kvm_fgpt_r8(w)},
                                {XS, gx + (size_t)c0 * D, ldd, (size_t)D, rc, D, rc16, kvm_fgpt_r8(D)},
                                {WS, P(lay(l, KVM_FGPT_R_W2)) + (size_t)f0 * D, ldd, (size_t)D, w, D, kvm_fgpt_r8(w), kvm_fgpt_r8(D)}};
            zero(DU, ldu, rc16, kvm_fgpt_r8(w));
            stage_all(cs);
          }
          grid_loop(rc16, kvm_fgpt_r8(w), [&](int r, int c) { Ut[r * ldum + c] = r < rc && c < w ? gelu(PRE[r * ldu + c]) : 0.f; });
          __syncthreads();
          gemm<true, false>(Ut, ldum, XS, ldd, w, D, rc, prec, [&](int m, int n, float a) { float* const o = gw2 + (size_t)(f0 + m) * D + n; *o = add ? ld(o) + a : a; });
          gemm<false, true>(XS, ldd, WS, ldd, rc, w, D, prec, [&](int m, int n, float a) { DU[m * ldu + n] = a * gelu_grad(PRE[m * ldu + n]); });
          colsum_smem(DU, ldu, rc, w, gb1 + f0, scr, add);
          {
            const Copy cs[2] = {{XS, h2 + (size_t)c0 * D, kvm_fgpt_ldmn(D), (size_t)D, rc, D, rc16, D},
                                {WS, P(lay(l, KVM_FGPT_R_W1)) + f0, ldu, (size_t)F, D, w, kvm_fgpt_r8(D), kvm_fgpt_r8(w)}};
            stage_all(cs);
          }
          gemm<true, false>(XS, kvm_fgpt_ldmn(D), DU, ldu, D, w, rc, prec, [&](int m, int n, float a) { float* const o = gw1 + (size_t)m * F + f0 + n; *o = add ? ld(o) + a : a; });
          gemm<false, true>(DU, ldu, WS, ldu, rc, D, w, prec, [&](int m, int n, float a) { part[(size_t)(c0 + m) * D + n] = a; });
        }
      }
      ph(MK_MLPB);
      if (!cross()) return false;                                                     // reduce-scatter dh2
      ph(MK_X_DH2);
      rs_sum(X + s.part, false, p.nf, T, 1, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
      if (c.zbw) {   // ZBW: fc2's output gradient, kept for the doers before ln_bwd turns this member's rows of gx into the gradient at xm
        float* const gxk = X + s.gxk;
        for (size_t i = (size_t)p.r0 * D + tid; i < (size_t)p.r1 * D; i += NT) gxk[i] = gx[i];
        __syncthreads();
      }
      ln_bwd(xm, dy, P(lay(l, KVM_FGPT_R_N2G)), X + s.m2[l], X + s.r2[l], gx, true, X + s.lnp + ((size_t)(2 * l + 1) * G + b) * 2 * D,
             X + s.lnp + ((size_t)(2 * l + 1) * G + b) * 2 * D + D, p.r0, p.r1, D);
      colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * l) * G + b) * D);  // bo: the gradient at xm
      ph(MK_LN2B);
      if (!cross()) return false;                                                     // all-gather gx (proj's output gradient)
      ph(MK_X_GPROJ);
      // ---- attention, backward. Each item as a query block: dWo, do, P, dS, dQ, the dK and dV partials over its prefix's keys, dbq, dWq and
      // q's dh rows; one exchange; then each item as a key block sums its keys' dK and dV partials over the query blocks at and after it, in
      // order, and adds dbk, dbv, dWk, dWv and k's and v's dh rows. An item's head-gradient partial is whole after both.
      for (int w = b; w < W; w += G) {
        KVM_FGPT_PHASE_CONSTS;
        const int u = w / QB, qb = w % QB, sq = u / H, hh = u % H, c0 = hh * hd, r0 = sq * T + qb * TQ, q0 = qb * TQ;
        const int KE = q0 + TQ, KE16 = kvm_fgpt_r16(KE), ldS = kvm_fgpt_ldk(KE16);
        float* const hg = X + s.hgp + ((size_t)l * W + w) * hgsz;
        float* const dkv = X + s.dkv + (size_t)w * T * 2 * hd;
        phase();
        float* const DO = ar.take((long long)TQ16 * ldh);
        float* const delta = ar.take(TQ);
        float* const Q = ar.take((long long)TQ16 * ldh);
        float* const K = ar.take((long long)KE16 * ldh);
        float* const Vv = ar.take((long long)KE16 * ldh);
        float* const lse = ar.take(TQ);
        float* const S = ar.take((long long)TQ16 * ldS);
        float* const DQ = ar.take((long long)TQ16 * ldh);
        float* const DK = ar.take((long long)KE16 * ldh);
        float* const DV = ar.take((long long)KE16 * ldh);
        float* const O = ar.take((long long)TQ16 * ldh);
        const int mark = ar.top;                                                       // gx's rows and Wo's rows, then h's rows and Wq's columns
        float* const GS = ar.take((long long)TQ16 * ldd);
        float* const WO = ar.take((long long)kvm_fgpt_r8(hd) * ldd);
        if (!check()) return false;
        {
          const Copy cs[7] = {{GS, gx + (size_t)r0 * D, ldd, (size_t)D, TQ, D, TQ16, kvm_fgpt_r8(D)},
                              {WO, P(lay(l, KVM_FGPT_R_WO)) + (size_t)c0 * D, ldd, (size_t)D, hd, D, kvm_fgpt_r8(hd), kvm_fgpt_r8(D)},
                              {O, X + s.o[l] + (size_t)r0 * D + c0, ldh, (size_t)D, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                              {Q, X + s.q[l] + (size_t)r0 * D + c0, ldh, (size_t)D, TQ, hd, TQ16, kvm_fgpt_r8(hd)},
                              {K, X + s.k[l] + (size_t)sq * T * D + c0, ldh, (size_t)D, KE, hd, KE16, kvm_fgpt_r8(hd)},
                              {Vv, X + s.v[l] + (size_t)sq * T * D + c0, ldh, (size_t)D, KE, hd, KE16, kvm_fgpt_r8(hd)},
                              {lse, X + s.lse[l] + (size_t)(sq * H + hh) * T + q0, 0, 0, 1, TQ, 1, TQ}};
          zero(DO, ldh, TQ16, kvm_fgpt_r8(hd));
          zero(S, ldS, TQ16, KE16);
          zero(DQ, ldh, TQ16, kvm_fgpt_r8(hd));
          zero(DK, ldh, KE16, kvm_fgpt_r8(hd));
          zero(DV, ldh, KE16, kvm_fgpt_r8(hd));
          stage_all(cs);
        }
        ph(MK_UB_STAGE);
        if (!c.zbwa)                                                                   // dWo rows (with ZBW for the attention, the item's doer's)
          gemm<true, false>(O, ldh, GS, ldd, hd, D, TQ, prec, [&](int m, int n, float a) { hg[3 * D * hd + 3 * hd + m * D + n] = a; });
        gemm<false, true>(GS, ldd, WO, ldd, TQ, hd, D, prec, [&](int m, int n, float a) { DO[m * ldh + n] = a; });                       // do
        for (int i = tid; i < TQ; i += NT) { float a = 0.f; for (int c = 0; c < hd; ++c) a += O[i * ldh + c] * DO[i * ldh + c]; delta[i] = a; }
        __syncthreads();
        ph(MK_UB_DO);
        gemm<false, true>(Q, ldh, K, ldh, TQ, KE, hd, true, [&](int m, int n, float a) { S[m * ldS + n] = n <= q0 + m ? expf(a * d.scale - lse[m]) : 0.f; });
        ph(MK_UB_P);
        gemm<true, false>(S, ldS, DO, ldh, KE, hd, TQ, true, [&](int m, int n, float a) { DV[m * ldh + n] = a; });                       // dV partial = P^T do
        gemm<false, true>(DO, ldh, Vv, ldh, TQ, KE, hd, true, [&](int m, int n, float a) { if (n <= q0 + m) S[m * ldS + n] *= a - delta[m]; });   // dS
        ph(MK_UB_DV_DS);
        gemm<false, false>(S, ldS, K, ldh, TQ, hd, KE, true, [&](int m, int n, float a) { DQ[m * ldh + n] = a * d.scale; });              // dq
        gemm<true, false>(S, ldS, Q, ldh, KE, hd, TQ, true, [&](int m, int n, float a) { DK[m * ldh + n] = a * d.scale; });               // dK partial
        for (int e = tid; e < KE * (hd / 4); e += NT) {                                // the dK and dV partials of the prefix's keys, for the key blocks
          const int j = e / (hd / 4), c = (e % (hd / 4)) * 4;
          *reinterpret_cast<float4*>(dkv + (size_t)j * 2 * hd + c) = *reinterpret_cast<const float4*>(DK + j * ldh + c);
          *reinterpret_cast<float4*>(dkv + (size_t)j * 2 * hd + hd + c) = *reinterpret_cast<const float4*>(DV + j * ldh + c);
        }
        ph(MK_UB_DQ_DK);
        rope(DQ, ldh, TQ, q0, T, hd, rtab, -1.f);
        if (!c.zbwa) colsum_smem(DQ, ldh, TQ, hd, hg + 3 * D * hd, scr);                                                            // dbq
        else {   // ZBW for the attention: dq (after RoPE's backward) for the item's doer, which runs dWq and dbq after the dK dV exchange
          float* const dqk = X + s.dqk;
          for (int e = tid; e < TQ * hd; e += NT) { const int i = e / hd, cc = e % hd; dqk[(size_t)(r0 + i) * D + c0 + cc] = DQ[i * ldh + cc]; }
          __syncthreads();
        }
        ph(MK_UB_DB);
        {   // h's rows and Wq's columns: below DQ where they fit (every buffer before DQ is dead now), else where gx's rows and Wo's rows were
          const long long need = kvm_fgpt_r4((long long)TQ16 * kvm_fgpt_ldmn(D)) + kvm_fgpt_r4((long long)kvm_fgpt_r8(D) * ldh);
          const int lo = (int)(DO - ar.base);
          ar.top = (long long)(DQ - ar.base) - lo >= need ? lo : mark;
        }
        float* const HQ = ar.take((long long)TQ16 * kvm_fgpt_ldmn(D));
        float* const WQ = ar.take((long long)kvm_fgpt_r8(D) * ldh);
        if (!check()) return false;
        if (!c.zbwa) {
          const Copy cs[2] = {{HQ, h + (size_t)r0 * D, kvm_fgpt_ldmn(D), (size_t)D, TQ, D, TQ16, D},
                              {WQ, P(lay(l, KVM_FGPT_R_WQ)) + c0, ldh, (size_t)D, D, hd, kvm_fgpt_r8(D), kvm_fgpt_r8(hd)}};
          stage_all(cs);
          gemm<true, false>(HQ, kvm_fgpt_ldmn(D), DQ, ldh, D, hd, TQ, prec, [&](int m, int n, float a) { hg[m * hd + n] = a; });        // dWq
        } else {   // h's rows are dWq's alone, which the item's doer runs
          stage(WQ, ldh, P(lay(l, KVM_FGPT_R_WQ)) + c0, (size_t)D, D, hd, kvm_fgpt_r8(D), kvm_fgpt_r8(hd));
        }
        float* const pu = X + s.part + (size_t)w * TQ * D;
        gemm<false, true>(DQ, ldh, WQ, ldh, TQ, D, hd, prec, [&](int m, int n, float a) { pu[(size_t)m * D + n] = a; });                // q's dh rows
        ph(MK_UB_STAGE2);
      }
      if (c.zbw && c.o1 > c.o0 && !zbw_pass(l, 0)) return false;                       // a doer: dW2 and db1 of its owners
      if (c.zbwa && b >= W && b < 2 * W && !zbwa_pass(l, b - W, 0)) return false;     // item b - W's doer: its dWo
      ph(MK_UB_ZBWA);
      if (QB > 1) { if (!cross()) return false; }                                       // the dK and dV partials (one block a unit: its own)
      else __syncthreads();
      ph(MK_X_DKV);
      for (int w = b; w < W; w += G) {                                                // each item as a key block
        KVM_FGPT_PHASE_CONSTS;
        const int u = w / QB, kb = w % QB, sq = u / H, hh = u % H, c0 = hh * hd, r0 = sq * T + kb * TQ, k0 = kb * TQ, ldv = kvm_fgpt_ldk(2 * hd);
        float* const hg = X + s.hgp + ((size_t)l * W + w) * hgsz;
        phase();
        float* const DKV = ar.take((long long)TQ16 * ldv);
        float* const HK = ar.take((long long)TQ16 * kvm_fgpt_ldmn(D));
        float* const WKV = ar.take((long long)kvm_fgpt_r8(D) * ldv);
        if (!check()) return false;
        {
          const Copy cs[3] = {{HK, h + (size_t)r0 * D, kvm_fgpt_ldmn(D), (size_t)D, TQ, D, TQ16, D},
                              {WKV, P(lay(l, KVM_FGPT_R_WK)) + c0, ldv, (size_t)D, D, hd, kvm_fgpt_r8(D), hd},
                              {WKV + hd, P(lay(l, KVM_FGPT_R_WV)) + c0, ldv, (size_t)D, D, hd, kvm_fgpt_r8(D), hd}};
          zero(DKV, ldv, TQ16, kvm_fgpt_r8(2 * hd));
          stage_all(cs);
        }
        for (int e = tid; e < TQ * 2 * hd; e += NT) {                                  // the keys' dK and dV: the query blocks at and after this one, in order
          const int j = e / (2 * hd), c = e % (2 * hd);
          float a = 0.f;
          for (int qq = kb; qq < QB; ++qq) a += ld(X + s.dkv + ((size_t)(u * QB + qq) * T + k0 + j) * 2 * hd + c);
          DKV[j * ldv + c] = a;
        }
        __syncthreads();
        rope(DKV, ldv, TQ, k0, T, hd, rtab, -1.f);                                     // dK back through RoPE
        colsum_smem(DKV, ldv, TQ, 2 * hd, hg + 3 * D * hd + hd, scr);                  // dbk, dbv
        ph(MK_UB_DW);
        gemm<true, false>(HK, kvm_fgpt_ldmn(D), DKV, ldv, D, 2 * hd, TQ, prec, [&](int m, int n, float a) { hg[(1 + n / hd) * D * hd + m * hd + n % hd] = a; });   // dWk, dWv
        float* const pu = X + s.part + (size_t)w * TQ * D;
        gemm<false, true>(DKV, ldv, WKV, ldv, TQ, D, 2 * hd, prec, [&](int m, int n, float a) { const size_t i = (size_t)m * D + n; pu[i] = ld(pu + i) + a; });   // k's and v's dh rows
        ph(MK_UB_DH);
      }
      if (c.zbw && c.o1 > c.o0 && !zbw_pass(l, 1)) return false;                       // a doer: dW1 of its owners
      if (c.zbwa && b >= W && b < 2 * W && !zbwa_pass(l, b - W, 1)) return false;     // item b - W's doer: its dWq and dbq
      ph(MK_UB_ZBWB);
      ph(MK_ATTNB);
      if (!cross()) return false;                                                     // reduce-scatter dh
      ph(MK_X_DH);
      rs_sum(X + s.part, true, H, T, QB, Nd, nullptr, nullptr, dy, p.r0, p.r1, D);
      ln_bwd(x, dy, P(lay(l, KVM_FGPT_R_N1G)), X + s.m1[l], X + s.r1[l], gx, true, X + s.lnp + ((size_t)(2 * l) * G + b) * 2 * D,
             X + s.lnp + ((size_t)(2 * l) * G + b) * 2 * D + D, p.r0, p.r1, D);
      if (l > 0) colsum_rows(gx, p.r0, p.r1, D, X + s.rpart + ((size_t)(2 * (l - 1) + 1) * G + b) * D);   // b2 of layer l - 1: the gradient at x[l]
      ph(MK_LN1B);
    }
    if (!cross()) return false;                                                       // all-gather gx (the embedding's output gradient)
    ph(MK_X_GEMB);
    // ---- the embedding's gradient: each owned vocabulary row sums the gradients of its tokens' rows, in row order, from gx staged a row
    // chunk at a time; a later chunk resumes each running sum from the one the earlier left, so the order (and every bit) is the whole N's
    {
      const int v0 = p.v0, w = p.v1 - p.v0;
      phase();
      float* const GXS = ar.take((long long)RC * D);
      if (!check()) return false;
      float* const ge = ptab[3 * KVM_FGPT_MAXP + 0];
      if (w > 0)
        for (int c0 = 0; c0 < N; c0 += RC) {
          KVM_FGPT_PHASE_CONSTS;
          const int rc = N - c0 < RC ? N - c0 : RC;
          const Copy cs[1] = {{GXS, gx + (size_t)c0 * D, D, (size_t)D, rc, D, rc, D}};
          stage_all(cs);
          for (int e = tid; e < w * D; e += NT) {
            const int row = v0 + e / D, c = e % D;
            float* const o = ge + (size_t)row * D + c;                                 // its square is counted with the blocks' below
            float a = c0 == 0 ? 0.f : ld(o);
            for (int r = 0; r < rc; ++r) if (ids[c0 + r] == row) a += GXS[r * D + c];
            *o = a;
          }
          __syncthreads();
        }
      __syncthreads();
    }
    ph(MK_EMBB);
    // ---- the attention slice's gradients, then the squares of every gradient this member owns, in two passes a barrier apart.
    // Pass 1: each attention element is the sum of its nu = B QB item partials (sequence, then query block) from hgp, added in that order
    // from +0 (sum_l2's float, bit for bit), stored to its gradient. A run of 4 flat attention indices from a multiple of 4 is 4 consecutive
    // floats of each item's partial and 4 consecutive gradients of one slot, both at offsets that are multiples of 4: every block of the
    // unit-gradient layout (dWq, dWk, dWv rows of hd; the biases' hd; dWo's [hd][d]) holds a multiple of 4 floats (c.ok: hd % 8, d % 4), so no
    // run crosses one. A run loads each partial as one 16-byte request (ld4) where its elements made four, and reads one olist and one osrc
    // word where they read four of each; its sums leave as one 16-byte store where the gradient buffer is 16-byte aligned. A thread holds two
    // runs' partials in flight: on an RTX 5070 four ran slower on both bed shapes and took the kernel from 144 registers to 213 on sm_120
    // (from 182, the one loop's, to 198 on sm_80). The slice's ragged ends (at most 3 elements each) are summed one element at a time.
    // Pass 2: the squares in the order each thread always added them, its elements i = tid, tid + NT, ... in index order, each read from its
    // gradient (a block's written by this member's epilogues, an attention element's by pass 1), so every thread's sq keeps every bit. With
    // no partials to hold, a thread reads 16 elements before its adds where the one loop read 4: any count keeps the order, and on an RTX
    // 5070 16 cut a d128's phase by 3.3 us against 4 (32, or pass 1's words loaded a trip ahead, bought nothing more).
    float sq = 0.f;
    {
      KVM_FGPT_PHASE_CONSTS;                                                          // values: ld4 would make each a shared-memory read again
      const int nu = B * QB, hq = H * QB, nbv = nb, nav = na, n = nbv + nav;           // partials an attention element; owned block + attention
      const long long a0v = a0, a1v = a0v + nav;
      const long long up = (a0v + 3) & ~3LL, r0 = up < a1v ? up : a1v, dn = a1v & ~3LL, r1 = dn > r0 ? dn : r0;   // the runs: [r0, r1)
      const int ng = (int)((r1 - r0) >> 2), ir = nbv + (int)(r0 - a0v);               // runs, and the owned index of the first one's first element
      const int nhead = (int)(r0 - a0v), nedge = nhead + (int)(a1v - r1);            // the ragged ends: [a0, r0) and [r1, a1)
      const uint32_t* const ol = olist;
      const uint32_t* const os = osrc;
      float* const* const gt = ptab + 3 * KVM_FGPT_MAXP;                              // every slot's gradient
      const float* const hg = X + s.hgp;                                              // 16-byte aligned, as every array of the layout
      constexpr uint32_t OFF = (1u << 25) - 1;
      constexpr int KG = 2, CH = 8;                                                   // runs a thread has in flight; partials a run loads before its adds
      for (int g0 = tid; g0 < ng; g0 += KG * NT) {
        float* gp[KG];
        const float* src[KG];
        bool ok[KG];
#pragma unroll
        for (int u = 0; u < KG; ++u) {
          const int g = g0 + u * NT, i = ir + 4 * g;
          ok[u] = g < ng;
          const uint32_t e = ok[u] ? ol[i] : 0u;
          gp[u] = gt[e >> 25] + (e & OFF);
          src[u] = hg + (ok[u] ? os[i] : 0u);                                         // the head's items: sequence, then query block
        }
        float4 v[KG];
#pragma unroll
        for (int u = 0; u < KG; ++u) v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int m0 = 0, bq = 0, qq = 0; m0 < nu; m0 += CH) {                          // partial m is sequence bq, query block qq
          size_t offk[CH];                                                            // the same for every run: one chunk's offsets
#pragma unroll
          for (int k = 0; k < CH; ++k) {
            offk[k] = ((size_t)bq * hq + qq) * hgsz;
            if (++qq == QB) { qq = 0; ++bq; }
          }
          float4 pv[KG][CH];
#pragma unroll
          for (int u = 0; u < KG; ++u)
#pragma unroll
            for (int k = 0; k < CH; ++k) pv[u][k] = ok[u] && m0 + k < nu ? ld4(src[u] + offk[k]) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
          for (int u = 0; u < KG; ++u)
#pragma unroll
            for (int k = 0; k < CH; ++k)
              if (ok[u] && m0 + k < nu) { v[u].x += pv[u][k].x; v[u].y += pv[u][k].y; v[u].z += pv[u][k].z; v[u].w += pv[u][k].w; }
        }
#pragma unroll
        for (int u = 0; u < KG; ++u)                                                  // after the trip's sums: no store between their loads
          if (ok[u]) {
            if ((reinterpret_cast<size_t>(gp[u]) & 15) == 0) *reinterpret_cast<float4*>(gp[u]) = v[u];
            else { gp[u][0] = v[u].x; gp[u][1] = v[u].y; gp[u][2] = v[u].z; gp[u][3] = v[u].w; }
          }
      }
      if (NT - 1 - tid < nedge) {                                                     // a ragged end's element, on the last threads
        const int j = NT - 1 - tid, i = j < nhead ? nbv + j : ir + 4 * ng + (j - nhead);
        const uint32_t e = ol[i];
        const float* const src = hg + os[i];
        gt[e >> 25][e & OFF] = sum_l2<8>(nu, [&](int m) { return src + ((size_t)(m / QB) * hq + m % QB) * hgsz; });
      }
      __syncthreads();                                                                // pass 2 reads sums other threads stored
      constexpr int SQ = 16;                                                          // elements a thread reads before its adds
      for (int i0 = tid; i0 < n; i0 += SQ * NT) {
        float v[SQ];
        bool ok[SQ];
#pragma unroll
        for (int u = 0; u < SQ; ++u) {
          const int i = i0 + u * NT;
          ok[u] = i < n;
          const uint32_t e = ok[u] ? ol[i] : 0u;
          v[u] = ok[u] ? gt[e >> 25][e & OFF] : 0.f;
        }
#pragma unroll
        for (int u = 0; u < SQ; ++u) if (ok[u]) sq += v[u] * v[u];
      }
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
        if (okq) a = sum_l2<(KVM_FGPT_MAX_GANG + 7) / 8>(m1 - m0, [&](int k) { return base0 + (size_t)(m0 + k) * stride; });
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
    // The G partials in index order, read once a warp, a line a request (warp_sum_l2): with every thread reading every partial, each
    // member sent 8 G one-word requests to the same few lines, so the gang's grew as G squared.
    const float norm2 = warp_sum_l2<(KVM_FGPT_MAX_GANG + 31) / 32>(G, [&](int m) { return X + s.gn + m; });
    const float norm = sqrtf(norm2);
    const float coef = d.max_norm > 0.f ? fminf(1.f, d.max_norm / (norm + 1e-6f)) : 1.f;
    // ---- AdamW on the owned elements, in the bundle's float32 operation order, with the counter after its increment: one flat index over
    // the blocks, the attention slice and the replicated slice, four elements a thread in flight. Each element's pointers are taken from
    // the shared table once, before any store, so no store waits on the table again, and every element's math runs before the first
    // store. A slot past n_own loads nothing (predicated, not branched, so the trip's loads still issue together), skips the math and
    // stores nothing: a member with fewer elements than a trip's slots (gpt-tiny's ~530 against 2,048) pays only for its own. All four
    // arrays are this member's own.
    {
      const float b1 = d.beta1, b2 = d.beta2, lr = d.lr, eps = d.adam_eps;
      const float bc1 = ld(d.bc + 2 * step), bc2 = ld(d.bc + 2 * step + 1);          // member 0's table (above), for t + 1
      const float ob1 = __fsub_rn(1.f, b1), ob2 = __fsub_rn(1.f, b2);
      constexpr int AD = 4;                                                         // the A100 reads 8 slower than 4 (x1.085 against x1.070 on a d128)
      for (int i0 = tid; i0 < n_own; i0 += AD * NT) {
        float* pp[AD];
        float* mp[AD];
        float* vp[AD];
        float g4[AD], m4[AD], v4[AD], p4[AD], w4[AD];
        bool ok[AD];
#pragma unroll
        for (int u = 0; u < AD; ++u) {
          const int i = i0 + u * NT;
          ok[u] = i < n_own;
          const uint32_t e = ok[u] ? olist[i] : 0u;                                 // a slot past n_own loads nothing
          const int sl = (int)(e >> 25);
          const size_t off = e & ((1u << 25) - 1);
          pp[u] = ptab[sl] + off; mp[u] = ptab[KVM_FGPT_MAXP + sl] + off; vp[u] = ptab[2 * KVM_FGPT_MAXP + sl] + off;
          g4[u] = ok[u] ? *(ptab[3 * KVM_FGPT_MAXP + sl] + off) : 0.f;
          w4[u] = wdt[sl];
        }
#pragma unroll
        for (int u = 0; u < AD; ++u) { m4[u] = ok[u] ? *mp[u] : 0.f; v4[u] = ok[u] ? *vp[u] : 0.f; p4[u] = ok[u] ? *pp[u] : 0.f; }
#pragma unroll
        for (int u = 0; u < AD; ++u) {
          if (!ok[u]) continue;                                                        // nor any math
          const float gi = __fmul_rn(g4[u], coef);
          const float mi = __fadd_rn(__fmul_rn(b1, m4[u]), __fmul_rn(ob1, gi));
          const float vi = __fadd_rn(__fmul_rn(b2, v4[u]), __fmul_rn(__fmul_rn(ob2, gi), gi));
          const float decay = __fmul_rn(__fmul_rn(lr, w4[u]), p4[u]);
          const float upd = __fdiv_rn(__fmul_rn(lr, __fdiv_rn(mi, bc1)), __fadd_rn(__fsqrt_rn(__fdiv_rn(vi, bc2)), eps));
          m4[u] = mi; v4[u] = vi; p4[u] = __fsub_rn(__fsub_rn(p4[u], decay), upd);
        }
#pragma unroll
        for (int u = 0; u < AD; ++u)
          if (ok[u]) { *mp[u] = m4[u]; *vp[u] = v4[u]; *pp[u] = p4[u]; }
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

template <bool Counted, class FAIL>
__device__ __forceinline__ bool run(const KvmFgptDesc& d, int member, float* smem, int smem_floats, const int* abort, FAIL fail) {
  NoPhases ph;
  return run<Counted>(d, member, smem, smem_floats, abort, fail, ph);
}

}  // namespace kvm_fgpt
