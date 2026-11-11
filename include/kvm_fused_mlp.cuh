#pragma once
// The fused MLP step, device side: a slice's forward and backward as one program of a 256-thread block with the weights and activations in shared memory, and the reduction of the partial gradients.
// Used by the ops `fmlp_slice` and `fmlp_sgd` of the static schedule path, and by bench/fused_mlp (the same code, timed alone). Layout and descriptor: kvm_fused_mlp.h.
//
// An SM issues about one instruction a cycle across its warps, so what a loop costs is its dynamic instruction count: everything that does not change along K is computed once per GEMM, a
// thread's element loops walk (row, column) with carries instead of dividing, and the accumulators stay in registers (an array indexed at run time would live in memory).
#include "kvm_fused_mlp.h"

#ifndef KVM_FMLP_RZ
#define KVM_FMLP_RZ 0   // 1: TF32 by truncation (the MMA's own reading of raw fp32 bits); 0: round to nearest
#endif

namespace kvm_fmlp {

constexpr int NT = 256;   // threads of the block (the static worker's KVM_THREADS)

__device__ __forceinline__ void cp_async16(float* dst, const float* src) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void mma_tf32(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const float* p) {   // four 8x8 b16 matrices == four 8x4 tf32 matrices (16 bytes per lane)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ unsigned tf32_round(unsigned u) { return KVM_FMLP_RZ ? u : u + 0x1000u; }   // operands are finite: half an ulp of TF32 added to the raw bits
__device__ __forceinline__ unsigned tf32_in(float x) { return tf32_round(__float_as_uint(x)); }

// C[M x N] = A B over shared-memory operands, m16n8k8 TF32 mma with fp32 accumulators held in registers. A: row-major, pitch lda (TA: stored [k][m], A^T); B: pitch ldb (TB: stored [n][k], B^T).
// M, N multiples of 16 (N at most 128), K a multiple of 8. Warp w owns column block w (16 columns) and every row block of it (at most MAXI row blocks): no division, and pairs are loop indices, so the accumulators
// stay in registers (an array indexed at run time would live in memory). K-contiguous operands (A as [m][k], B as [n][k]) are read with ldmatrix (their pitch must be a multiple of 4), the others with 32-bit loads.
// NRB > 0 fixes the number of row blocks at compile time (M = 16 NRB, MAXI = NRB): the k-loop then has no per-block predicate and branch (measured: at one row block those cost a
// third of a k-step); NRB = 0 takes M at run time, up to MAXI blocks. SGD = true makes the epilogue C -= lr * (A B): a weight block updated in place by
// its own gradient (the grid step's members), with no gradient array.
template <bool TA, bool TB, int MAXI, bool ALIAS = false, int NRB = 0, bool SGD = false>
__device__ __forceinline__ void gemm(const float* A, int lda, const float* B, int ldb, float* C, int ldc, int M, int N, int K, float lr = 0.f) {
  static_assert(NRB == 0 || NRB == MAXI, "a fixed row-block count is the array bound");
  const int nrb = NRB ? NRB : (M + 15) >> 4, ncb = (N + 15) >> 4, cb = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, la = lane & 7, b3 = (lane >> 3) & 1, b4 = lane >> 4;
  if (cb >= ncb) { if constexpr (ALIAS) __syncthreads(); return; }   // this warp has no column block (it still takes part in the barrier below)
  const int n0 = cb * 16 + g, nl = cb * 16 + la + 8 * b4;
  int a_ld[MAXI], oa1[MAXI];
  float acc[MAXI][2][4];
  float* cout0[MAXI], *cout1[MAXI];                     // this lane's two output rows of the block's accumulators
#pragma unroll
  for (int i = 0; i < MAXI; ++i) {
    if (!NRB && i >= nrb) continue;
    const int m0 = i * 16 + g;
    if constexpr (TA) { a_ld[i] = m0; oa1[i] = m0 + 8; } else a_ld[i] = (i * 16 + la + 8 * b3) * lda + 4 * b4;   // ldmatrix: the lane's chunk (row of its 8x8 block, k + 0 or 4)
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    cout0[i] = C + (size_t)m0 * ldc + cb * 16 + 2 * t; cout1[i] = C + (size_t)(m0 + 8) * ldc + cb * 16 + 2 * t;
  }
  const int ob0 = TB ? nl * ldb + 4 * b3 : n0, ob1 = TB ? ob0 : n0 + 8;      // B: the lane's ldmatrix chunk (row of its 8x8 block of the column block, k + 0 or 4), or the columns of the two fragments
  const int tla = t * lda, tla4 = (t + 4) * lda, tlb = t * ldb, tlb4 = (t + 4) * ldb;
  int ka = 0, kb = 0;                                   // the K row of this k-step in elements (TA: of A; !TB: of B)
  const int dka = TA ? 8 * lda : 0, dkb = TB ? 0 : 8 * ldb;
  for (int kk = 0; kk < K; kk += 8, ka += dka, kb += dkb) {
    unsigned r[4], b00 = 0, b01 = 0, b10 = 0, b11 = 0;
    if constexpr (TB) { ldmatrix_x4(r, B + ob0 + kk); b00 = tf32_round(r[0]); b01 = tf32_round(r[1]); b10 = tf32_round(r[2]); b11 = tf32_round(r[3]); }
    else { b00 = tf32_in(B[kb + tlb + ob0]); b01 = tf32_in(B[kb + tlb4 + ob0]); b10 = tf32_in(B[kb + tlb + ob1]); b11 = tf32_in(B[kb + tlb4 + ob1]); }
#pragma unroll
    for (int i = 0; i < MAXI; ++i) {
      if (!NRB && i >= nrb) continue;
      unsigned a[4];
      if constexpr (TA) { const int i0 = ka + tla, i4 = ka + tla4; a[0] = tf32_in(A[i0 + a_ld[i]]); a[1] = tf32_in(A[i0 + oa1[i]]); a[2] = tf32_in(A[i4 + a_ld[i]]); a[3] = tf32_in(A[i4 + oa1[i]]); }
      else { ldmatrix_x4(a, A + a_ld[i] + kk); a[0] = tf32_round(a[0]); a[1] = tf32_round(a[1]); a[2] = tf32_round(a[2]); a[3] = tf32_round(a[3]); }
      mma_tf32(acc[i][0], a, b00, b01);
      mma_tf32(acc[i][1], a, b10, b11);
    }
  }
  if constexpr (ALIAS) __syncthreads();                 // the output replaces the input: every warp's reads end before any warp's stores
#pragma unroll
  for (int i = 0; i < MAXI; ++i) {
    if (!NRB && i >= nrb) continue;
    float* c0 = cout0[i], *c1 = cout1[i];
#pragma unroll
    for (int j = 0; j < 2; ++j) {   // a lane's two adjacent columns as one 8-byte store (every ldc is even and every C 8-byte aligned): half the store instructions, and global partials fill whole sectors
      if constexpr (SGD) {
        float2* p0 = reinterpret_cast<float2*>(c0 + j * 8); float2* p1 = reinterpret_cast<float2*>(c1 + j * 8);
        float2 u = *p0, v = *p1;
        u.x -= lr * acc[i][j][0]; u.y -= lr * acc[i][j][1]; v.x -= lr * acc[i][j][2]; v.y -= lr * acc[i][j][3];
        *p0 = u; *p1 = v;
      } else {
        *reinterpret_cast<float2*>(c0 + j * 8) = make_float2(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<float2*>(c1 + j * 8) = make_float2(acc[i][j][2], acc[i][j][3]);
      }
    }
  }
}

// A GEMM whose M is the slice's R rows (forward, head, activation gradients): R is 16 or 32, so the row-block count is a compile-time 1 or 2.
template <bool TA, bool TB, bool ALIAS = false>
__device__ __forceinline__ void gemm_r(const float* A, int lda, const float* B, int ldb, float* C, int ldc, int R, int N, int K) {
  if (R == 16) gemm<TA, TB, 1, ALIAS, 1>(A, lda, B, ldb, C, ldc, 16, N, K);
  else if (R == 32) gemm<TA, TB, 2, ALIAS, 2>(A, lda, B, ldb, C, ldc, 32, N, K);
  else gemm<TA, TB, 4, ALIAS>(A, lda, B, ldb, C, ldc, R, N, K);
}

// A GEMM whose M is a layer's width (the weight gradients, A^T dZ: one row block per 16 of the K side): the same compile-time row-block count, one instantiation per count up to 8.
template <bool TA, bool TB, bool SGD = false>
__device__ __forceinline__ void gemm_m(const float* A, int lda, const float* B, int ldb, float* C, int ldc, int M, int N, int K, float lr = 0.f) {
  switch ((M + 15) >> 4) {
    case 1: gemm<TA, TB, 1, false, 1, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;
    case 2: gemm<TA, TB, 2, false, 2, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;
    case 3: gemm<TA, TB, 3, false, 3, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;
    case 4: gemm<TA, TB, 4, false, 4, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;
    case 5: gemm<TA, TB, 5, false, 5, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;
    case 6: gemm<TA, TB, 6, false, 6, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;
    case 7: gemm<TA, TB, 7, false, 7, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;
    default: gemm<TA, TB, 8, false, 8, SGD>(A, lda, B, ldb, C, ldc, M, N, K, lr); break;   // M <= 128: the pass's widths are at most 128
  }
}

// Thread loop over (row, column) cells of a rows x width grid (cell index i = row * width + column), each thread its own cells every NT: no division per cell.
template <class Fn>
__device__ __forceinline__ void for_cells(int rows, int width, Fn fn) {
  const int q = NT / width, s = NT % width, n = rows * width;
  int r = threadIdx.x / width, c = threadIdx.x % width;
  for (int i = threadIdx.x; i < n; i += NT) { fn(r, c); r += q; c += s; if (c >= width) { c -= width; ++r; } }
}

__device__ __forceinline__ float gelu_y(float x) { return 0.7978845608028654f * (x + 0.044715f * x * x * x); }
__device__ __forceinline__ float gelu_f(float x) { return x / (1.f + expf(-2.f * gelu_y(x))); }               // 0.5 x (1 + tanh y) = x / (1 + e^{-2y})
__device__ __forceinline__ float gelu_g(float x) {                                                                  // d/dx of the above, with s = (1 + tanh y) / 2 and 1 - tanh^2 y = 4 s (1 - s)
  const float s = 1.f / (1.f + expf(-2.f * gelu_y(x)));
  return s + 2.f * x * s * (1.f - s) * 0.7978845608028654f * (1.f + 3.f * 0.044715f * x * x);
}
// gelu_f(x) and gelu_g(x) from one exp: the same expressions as the two functions above, so each result is bitwise theirs.
__device__ __forceinline__ void gelu_fg(float x, float& f, float& g) {
  const float den = 1.f + expf(-2.f * gelu_y(x));
  f = x / den;
  const float s = 1.f / den;
  g = s + 2.f * x * s * (1.f - s) * 0.7978845608028654f * (1.f + 3.f * 0.044715f * x * x);
}

// The descriptor is copied to the base of shared memory by a prefetch issued before the task's waits (the worker knows its next record's descriptor from the record), so the round trip overlaps them; the op body then
// commits and waits for it. The bed calls prefetch_desc just before the op.
__device__ __forceinline__ void prefetch_desc(const KvmFusedDesc* gd, float* sm) {
  const int chunks = (int)((sizeof(KvmFusedDesc) + 15) / 16);
  if (threadIdx.x < chunks) cp_async16(sm + threadIdx.x * 4, reinterpret_cast<const float*>(gd) + threadIdx.x * 4);
}
__device__ __forceinline__ const KvmFusedDesc* wait_desc(float* sm) {
  cp_commit_wait();
  __syncthreads();
  return reinterpret_cast<const KvmFusedDesc*>(sm);
}

struct NoFail { __device__ __forceinline__ void operator()(int, int) const {} };
// Accumulates the cycles of each phase of a slice into ph[i], thread 0 only; `last` is a shared-memory slot holding the previous mark. A null `ph` does nothing.
struct PhaseTimer {
  long long* ph; long long* last;
  __device__ __forceinline__ void operator()(int i) const { if (ph && threadIdx.x == 0) { const long long n = clock64(); if (i >= 0) ph[i] += n - *last; *last = n; } }
};
constexpr int ERR_TOKEN = 1005;   // a token id or target outside [0, V) (= KVM_GPT_E_TOKEN of the executor's op pack)

// One slice of one step: forward and backward over rows [sc * R, sc * R + R) of the step's batch (ids and tgt: the step's rows); the partial gradients go to global memory, one copy per granule of
// KVM_FMLP_GRANULE rows (kvm_fused_mlp.h: the step's arithmetic does not depend on the slice size). Each granule's sum of row losses goes to `loss_slot[g]`, or to d.loss_part[sc * R / 16 + g] when it is
// null. `ph(i)` marks the end of phase i (the bed times them; the executor passes a null timer unless --fuse-phases). Ends with its stores issued, not yet visible GPU-wide:
// the caller's publish fences them.
template <class PH, class FAIL>
__device__ __forceinline__ void slice_run(int sc, const int32_t* ids, const int32_t* tgt, float* loss_slot, float* sm, PH ph, FAIL fail) {
  const KvmFusedDesc& d = *wait_desc(sm);
  const int R = d.R, L = d.L, E = d.E, V = d.V, VP = d.VP, XP = d.XP, maxHP = d.maxHP, bt = d.T;
  ids += (size_t)sc * R; tgt += (size_t)sc * R;
  constexpr int G = KVM_FMLP_GRANULE;
  if (!loss_slot) loss_slot = d.loss_part + sc * (d.R / G);
  const int VN = kvm_fmlp_pad16(V), VK = kvm_fmlp_pad8(V);
  float* const sX = sm + d.X, *const sY = sm + d.Y, *const sH = sm + d.Htmp;
  int32_t* const s_ids = reinterpret_cast<int32_t*>(sm + d.ids), *const s_tgt = reinterpret_cast<int32_t*>(sm + d.tgt);
  float* const s_ce = sm + d.ce, *const s_rl = sm + d.rl;
  float* const dcur = sm + d.D;                    // the backward's activation gradient (each GEMM's output replaces its input)
  ph(-1);
  // ---- operand loads: every weight matrix, the biases, this slice's tokens and targets (one pipelined pass)
  for (int l = 0; l < L; ++l) {
    float* sW = sm + d.W[l]; const float* gw = d.w[l]; const int K = l ? d.H[l - 1] : E, Hl = d.H[l], WPl = d.WP[l], c4 = Hl / 4;
    for_cells(K, c4, [&](int r, int c) { cp_async16(sW + (size_t)r * WPl + c * 4, gw + (size_t)r * Hl + c * 4); });
    float* sB = sm + d.B[l]; const float* gb = d.b[l];
    for (int i = threadIdx.x; i < c4; i += NT) cp_async16(sB + i * 4, gb + i * 4);
  }
  const int WhP = d.WhP;
  { float* sW = sm + d.Wh; const float* gwh = d.wh; const int K = d.H[L - 1], c4 = V / 4;
    for_cells(K, c4, [&](int r, int c) { cp_async16(sW + (size_t)r * WhP + c * 4, gwh + (size_t)r * V + c * 4); });
    if (VK > V) for_cells(K, VK - V, [&](int r, int c) { sW[(size_t)r * WhP + V + c] = 0.f; });      // the columns the head's backward GEMM reads past V
  }
  for (int i = threadIdx.x; i < R / 4; i += NT) {
    cp_async16(reinterpret_cast<float*>(s_ids) + i * 4, reinterpret_cast<const float*>(ids) + i * 4);
    cp_async16(reinterpret_cast<float*>(s_tgt) + i * 4, reinterpret_cast<const float*>(tgt) + i * 4);
  }
  cp_commit_wait();
  __syncthreads();
  for (int i = threadIdx.x; i < R; i += NT) if ((unsigned)s_ids[i] >= (unsigned)V || (unsigned)s_tgt[i] >= (unsigned)V) fail(ERR_TOKEN, (unsigned)s_ids[i] >= (unsigned)V ? s_ids[i] : s_tgt[i]);
  ph(0);
  // ---- embedding gather: X[r] = we[ids[r]]
  { const float* we = d.we; const int c4 = E / 4;
    for_cells(R, c4, [&](int r, int c) { cp_async16(sX + (size_t)r * XP + c * 4, we + (size_t)s_ids[r] * E + c * 4); }); }
  cp_commit_wait();
  __syncthreads();
  ph(1);
  // ---- forward: Z[l] = A W[l] + b[l] (pre-GELU, kept for the backward), H = gelu(Z[l]) in sH is the next A
  {
    const float* A = sX; int lda = XP;
    for (int l = 0; l < L; ++l) {
      float* sZ = sm + d.Z[l]; const float* sB = sm + d.B[l];
      const int HPl = d.HP[l], Hl = d.H[l];
      float* const sHl = d.keepH && l + 1 < L ? sm + d.Hk[l] : sH;   // the last layer's H stays in sH (the head and its gradient read it there)
      gemm_r<false, false>(A, lda, sm + d.W[l], d.WP[l], sZ, HPl, R, kvm_fmlp_pad16(Hl), l ? d.H[l - 1] : E);
      __syncthreads();
      if (d.keepH)   // the backward needs Z[l] only for gelu'(Z[l]): store that instead, so its pass is one multiply
        for_cells(R, Hl, [&](int r, int c) { const float z = sZ[r * HPl + c] + sB[c]; float h, g; gelu_fg(z, h, g); sZ[r * HPl + c] = g; sHl[r * maxHP + c] = h; });
      else
        for_cells(R, Hl, [&](int r, int c) { const float z = sZ[r * HPl + c] + sB[c]; sZ[r * HPl + c] = z; sHl[r * maxHP + c] = gelu_f(z); });
      __syncthreads();
      A = sHl; lda = maxHP;
    }
    ph(2);
    gemm_r<false, false>(A, lda, sm + d.Wh, WhP, sY, VP, R, VN, d.H[L - 1]);
    __syncthreads();
    ph(3);
  }
  // ---- softmax CE, one row per 8 threads: the row's max, then e[c] = exp(Y[c] - max) overwrites Y and sums to S; the row's share of the loss is log(S) + max - Y[target], and d(loss)/dY[c] = (e[c] / S - (c == target)) / T.
  {
    const int t = threadIdx.x, r = t >> 3, l = t & 7;
    const float inv_bt = 1.f / bt;
    if (r < R) {
      float mx = -INFINITY;
      for (int c = l; c < V; c += 8) mx = fmaxf(mx, sY[r * VP + c]);
      s_ce[r * 8 + l] = mx;
    }
    __syncthreads();
    if (r < R) {
      float M = -INFINITY;
      for (int j = 0; j < 8; ++j) M = fmaxf(M, s_ce[r * 8 + j]);
      const int tg = s_tgt[r];
      float ytg = 0.f;            // Y[target] before e[] replaces it (only the thread holding that column)
      if (tg % 8 == l) ytg = sY[r * VP + tg];
      float sum = 0.f;
      for (int c = l; c < V; c += 8) { const float e = expf(sY[r * VP + c] - M); sY[r * VP + c] = e; sum += e; }
      s_ce[r * 8 + l] = sum;
      __syncthreads();
      float S = 0.f;
      for (int j = 0; j < 8; ++j) S += s_ce[r * 8 + j];
      const float inv_s = 1.f / S;
      if (tg % 8 == l) s_rl[r] = logf(S) + M - ytg;
      for (int c = l; c < V; c += 8) { const float p = sY[r * VP + c] * inv_s; sY[r * VP + c] = (p - (c == tg ? 1.f : 0.f)) * inv_bt; }
    } else __syncthreads();
    __syncthreads();
    if (VK > V) for_cells(R, VK - V, [&](int rr, int c) { sY[rr * VP + V + c] = 0.f; });   // columns V..VK-1 of dY, read by the head's backward GEMM
    if (threadIdx.x < R / G) { float s = 0; for (int j = threadIdx.x * G; j < threadIdx.x * G + G; ++j) s += s_rl[j]; loss_slot[threadIdx.x] = s; }   // a granule's rows in order
  }
  __syncthreads();
  ph(4);
  // ---- backward, last layer first. sH still holds gelu(Z[L-1]) of the last forward layer.
  {
    const int KH = d.H[L - 1];
    for (int g = 0; g < R / G; ++g)   // one partial per granule: K over the granule's rows
      gemm_m<true, false>(sH + (size_t)g * G * maxHP, maxHP, sY + (size_t)g * G * VP, VP, d.gh + (size_t)(sc * (R / G) + g) * kvm_fmlp_pad16(KH) * d.ghld, d.ghld,
                          kvm_fmlp_pad16(KH), VN, G);
    gemm_r<false, true>(sY, VP, sm + d.Wh, WhP, dcur, maxHP, R, kvm_fmlp_pad16(KH), VK);
    __syncthreads();
    ph(5);
    for (int l = L - 1; l >= 0; --l) {
      // dcur is dH[l]'s gradient: dZ[l] = dcur * gelu'(Z[l]), the bias gradient, the layer's weight gradient, then the next activation gradient
      const float* sZ = sm + d.Z[l];
      const int HPl = d.HP[l], Hl = d.H[l], KL = l ? d.H[l - 1] : E;
      if (d.keepH) for_cells(R, Hl, [&](int r, int c) { dcur[r * maxHP + c] *= sZ[r * HPl + c]; });   // the forward stored gelu'(Z[l])
      else for_cells(R, Hl, [&](int r, int c) { dcur[r * maxHP + c] *= gelu_g(sZ[r * HPl + c]); });
      __syncthreads();
      ph(6);
      if (threadIdx.x < Hl) {   // the bias gradient: one thread per column sums each granule's rows (a few warps' worth of work: a reduction across lanes would cost more instructions than it saves)
        const int c = threadIdx.x;
        for (int g = 0; g < R / G; ++g) {
          float v = 0.f;
          for (int r = g * G; r < g * G + G; ++r) v += dcur[r * maxHP + c];
          d.gb[l][(size_t)(sc * (R / G) + g) * Hl + c] = v;
        }
      }
      ph(7);
      const size_t gw_copy = (size_t)kvm_fmlp_pad16(KL) * d.gwld[l];
      float* const gw = d.gw[l] + (size_t)sc * (R / G) * gw_copy;   // granule g's partial at gw + g * gw_copy
      if (l > 0) {
        const float* Hp_ = sm + (d.keepH ? d.Hk[l - 1] : d.Htmp);
        if (!d.keepH) {   // recompute H[l-1] = gelu(Z[l-1]) into sH (the forward overwrote it)
          const float* sZp = sm + d.Z[l - 1]; const int HPp = d.HP[l - 1], Hp = d.H[l - 1];
          for_cells(R, Hp, [&](int r, int c) { sH[r * maxHP + c] = gelu_f(sZp[r * HPp + c]); });
          __syncthreads();
        }
        ph(8);
        for (int g = 0; g < R / G; ++g)   // weight gradient partials: H[l-1]^T dZ[l], K over each granule's rows
          gemm_m<true, false>(Hp_ + (size_t)g * G * maxHP, maxHP, dcur + (size_t)g * G * maxHP, maxHP, gw + g * gw_copy, d.gwld[l], kvm_fmlp_pad16(KL), kvm_fmlp_pad16(Hl), G);
      } else {
        for (int g = 0; g < R / G; ++g)
          gemm_m<true, false>(sX + (size_t)g * G * XP, XP, dcur + (size_t)g * G * maxHP, maxHP, gw + g * gw_copy, d.gwld[0], kvm_fmlp_pad16(KL), kvm_fmlp_pad16(Hl), G);
      }
      ph(9);
      gemm_r<false, true, true>(dcur, maxHP, sm + d.W[l], d.WP[l], dcur, maxHP, R, kvm_fmlp_pad16(KL), Hl);       // the next activation gradient, in place: dZ[l] W[l]^T
      __syncthreads();
      ph(10);
    }
  }
  // embedding gradient partials, one per granule: rows of one granule can name one token, and the panel's copies must agree bitwise, so atomics (whose order rounds differently run to run) are out. Each
  // row's token chains to the next row of its granule naming it; a (token, 4-float chunk) cell sums its rows along the chain, in row order, and stores once into its granule's partial. The partials are
  // all zero here (the previous step's owners cleared them).
  {
    int32_t* const s_next = reinterpret_cast<int32_t*>(s_rl);          // the next row of the granule naming this row's token, or -1
    int32_t* const s_first = reinterpret_cast<int32_t*>(s_ce);         // 1 when the row is the first of its granule naming its token
    for (int r = threadIdx.x; r < R; r += NT) {
      const int id = s_ids[r], g0 = r / G * G; int nx = -1, first = 1;
      for (int q = r + 1; q < g0 + G; ++q) if (s_ids[q] == id) { nx = q; break; }
      for (int q = g0; q < r; ++q) if (s_ids[q] == id) { first = 0; break; }
      s_next[r] = nx; s_first[r] = first;
    }
    __syncthreads();
    float4* const ge = reinterpret_cast<float4*>(d.ge + (size_t)sc * (R / G) * V * E);
    for_cells(R, E / 4, [&](int r, int ct) {
      if (!s_first[r]) return;
      float4 acc = *reinterpret_cast<const float4*>(dcur + (size_t)r * maxHP + ct * 4);
      for (int q = s_next[r]; q >= 0; q = s_next[q]) { const float4 v = *reinterpret_cast<const float4*>(dcur + (size_t)q * maxHP + ct * 4); acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
      ge[((size_t)(r / G) * V + s_ids[r]) * (E / 4) + ct] = acc;
    });
  }
}

// One owner's share of one parameter, rows [first, last): a warp takes the rows congruent to its index minus `lead` mod 8 (warp `lead` takes row `first`), a lane its own 4-float chunks of each row: no
// division and no per-thread walk, and each chunk's operands (every granule's partial and the old weight) are loaded together before the first use, so a chunk costs one round trip however many granules
// there are. (A cp.async of 16 bytes per operand into shared memory measured slower: R-10i, 32.7k cycles against 19.1k: request-bound at about 11 GB/s.)
constexpr int MAX_SLICES = KVM_FMLP_MAX_SLICES;

__device__ __forceinline__ void sgd_share(float* w, const float* gg, int gld, size_t stride, int cols, int first, int last, int lead, float lr, int nsl, bool clear) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c4 = cols / 4;
  for (int r = first + ((warp - lead) & 7); r < last; r += 8) {
    for (int c = lane; c < c4; c += 32) {
      float4 v[MAX_SLICES];
      const float* gp = gg + (size_t)r * gld + c * 4;
#pragma unroll
      for (int sl = 0; sl < MAX_SLICES; ++sl) if (sl < nsl) v[sl] = *reinterpret_cast<const float4*>(gp + (size_t)sl * stride);
      float4* wp = reinterpret_cast<float4*>(w + (size_t)r * cols + c * 4);
      const float4 w0 = *wp;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int sl = 0; sl < MAX_SLICES; ++sl) if (sl < nsl) { acc.x += v[sl].x; acc.y += v[sl].y; acc.z += v[sl].z; acc.w += v[sl].w; }
      *wp = make_float4(w0.x - lr * acc.x, w0.y - lr * acc.y, w0.z - lr * acc.z, w0.w - lr * acc.w);
      if (clear) {
#pragma unroll
        for (int sl = 0; sl < MAX_SLICES; ++sl) if (sl < nsl) *reinterpret_cast<float4*>(const_cast<float*>(gp) + (size_t)sl * stride) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }
}

// One owner's share of the reduction: every parameter's rows split evenly over the d.groups owners (the inverse in `d.inv_groups` replaces a division per parameter; the shares meet exactly). Each warp runs
// its rows of the shares one after another, so an owner costs one round trip per row its busiest warp holds. A bias (owner l % groups updates bias l) and the loss (owner 0) therefore go to the first warp
// past the rows of the share beside them, which holds none of them while the share has fewer than 8 rows: their round trips overlap that share's instead of following it (measured: owner 0 ran 8.3 us
// against a mean of 5.5 with both on its warp 0). The loss is loss_out[0] = sum / T, loss_out[1] = 1 / T, as softmax_ce_reduce writes them, summed in granule (row) order.
__device__ __forceinline__ void sgd_run(int owner, float* loss_out, float* sm) {
  const KvmFusedDesc& d = *wait_desc(sm);
  const int L = d.L, E = d.E, V = d.V, nsl = d.T / KVM_FMLP_GRANULE, warp = threadIdx.x >> 5; const float lr = d.lr;   // nsl: the step's granules, whose partials every sum adds in row order
  const unsigned long long inv = d.inv_groups;
  const auto first = [&](int o, int rows) { return (int)(((unsigned long long)(unsigned)o * (unsigned)rows * inv) >> 32); };
  const int ef = first(owner, V), el = first(owner + 1, V);
  if (owner == 0 && loss_out && warp == (el & 7)) {   // lane i loads granule i's sum (one round trip for all), lane 0 adds them in row order
    const int lane = threadIdx.x & 31;
    const float v = lane < nsl ? d.loss_part[lane] : 0.f;
    float s = 0.f;
    for (int i = 0; i < nsl; ++i) s += __shfl_sync(0xffffffffu, v, i);
    if (lane == 0) { loss_out[0] = s / d.T; loss_out[1] = 1.f / d.T; }
  }
  sgd_share(d.we, d.ge, E, (size_t)V * E, E, ef, el, ef & 7, lr, nsl, true);
  for (int l = 0; l < L; ++l) {
    const int Hl = d.H[l], K = l ? d.H[l - 1] : E, f = first(owner, K), e = first(owner + 1, K);
    sgd_share(d.w[l], d.gw[l], d.gwld[l], (size_t)kvm_fmlp_pad16(K) * d.gwld[l], Hl, f, e, f & 7, lr, nsl, false);
    if (owner == l % d.groups) sgd_share(d.b[l], d.gb[l], Hl, (size_t)Hl, Hl, 0, 1, e & 7, lr, nsl, false);
  }
  { const int K = d.H[L - 1], f = first(owner, K), e = first(owner + 1, K); sgd_share(d.wh, d.gh, d.ghld, (size_t)kvm_fmlp_pad16(K) * d.ghld, V, f, e, f & 7, lr, nsl, false); }
}

}  // namespace kvm_fmlp
