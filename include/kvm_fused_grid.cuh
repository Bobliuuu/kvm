#pragma once
// The grid fused MLP step, device side: one member's part of one step (kvm_fused_grid.h has the design, the descriptor and the layout).
// GEMMs are kvm_fused_mlp.cuh's (m16n8k8 TF32 mma over shared-memory operands, partials stored straight to L2, weight blocks updated in place by
// the SGD epilogue); the barrier is kvm_gang.cuh's; row copies and the phase-clock hook are the split step's.
#include <cstdint>
#include "kvm_fused_grid.h"
#include "kvm_fused_mlp.cuh"
#include "kvm_fused_split.cuh"
#include "kvm_gang.cuh"

namespace kvm_fgrid {

using kvm_fmlp::NT;
using kvm_fsplit::NoPhases;
constexpr int T96 = 96, NRB = T96 / 16;   // rows a step: the GEMMs over all rows take 6 row blocks at compile time
constexpr int NPH = 8;   // the phases `ph(i)` marks, once a step in this order (every thread calls it)
// 0 load, 1 forward (layer 0 and the hidden layers), 2 head partials, 3 cross-entropy, 4 head backward, 5 hidden layers backward, 6 layer 0 backward,
// 7 embedding rows, write-back and the last barrier

// rows x cols floats between global memory and shared memory (cols a multiple of 4, 16-byte aligned rows): plain stores back to global
__device__ __forceinline__ void store_rows(float* dst, size_t dst_ld, const float* src, int src_ld, int rows, int cols) {
  kvm_fmlp::for_cells(rows, cols >> 2, [&](int r, int c4) {
    *reinterpret_cast<float4*>(dst + (size_t)r * dst_ld + 4 * c4) = *reinterpret_cast<const float4*>(src + (size_t)r * src_ld + 4 * c4); });
}

// The sum of n values p[0], p[stride], ... through L2 in a fixed order: value i joins chain i % U and the U chains are added in a fixed tree, so a run
// is deterministic while U loads are in flight instead of one (a sum over 64 partials as one chain is 64 dependent L2 round trips).
template <int U = 8>
__device__ __forceinline__ float sum_cg(const float* p, size_t stride, int n) {
  float acc[U];
#pragma unroll
  for (int u = 0; u < U; ++u) acc[u] = 0.f;
  int i = 0;
  for (; i + U <= n; i += U) {
#pragma unroll
    for (int u = 0; u < U; ++u) acc[u] += __ldcg(p + (size_t)(i + u) * stride);
  }
#pragma unroll
  for (int u = 0; u < U; ++u) if (i + u < n) acc[u] += __ldcg(p + (size_t)(i + u) * stride);
#pragma unroll
  for (int w = 1; w < U; w <<= 1)
#pragma unroll
    for (int u = 0; u + w < U; u += 2 * w) acc[u] += acc[u + w];
  return acc[0];
}

// The descriptor is copied to the arena's base before the task's waits, so its round trip overlaps them; grid_run commits and waits for it.
__device__ __forceinline__ void prefetch_desc(const KvmGridDesc* gd, float* sm) {
  const int chunks = (int)((sizeof(KvmGridDesc) + 15) / 16);
  for (int i = threadIdx.x; i < chunks; i += NT) kvm_fmlp::cp_async16(sm + i * 4, reinterpret_cast<const float*>(gd) + i * 4);
}

// One step of member `member` of a grid run: `ids` and `tgt` are the step's T tokens and targets, `abort` the run's error word (kvm_gang::barrier;
// nullptr in a bed), `loss_out` the step's loss (member 0 writes loss_out[0] = sum / T and loss_out[1] = 1 / T, as softmax_ce_reduce does). The
// descriptor sits at the arena's base, its cp.async issued by the caller. The members cross 4 L + 2 barriers; the last one ends the step, so every
// update is visible to every member when any of them returns.
template <class PH>
__device__ __forceinline__ void grid_run(int member, const int32_t* ids, const int32_t* tgt, float* loss_out, float* sm, const int* abort, PH ph) {
  kvm_fmlp::cp_commit_wait();
  __syncthreads();
  const KvmGridDesc& d = *reinterpret_cast<const KvmGridDesc*>(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, m = member, P = d.P, T = T96, E = d.E, H = d.H, V = d.V, VP = d.VP, L = d.L;
  const int pA = d.pA, pB = d.pB;
  const float lr = d.lr, inv_t = 1.f / (float)T;
  unsigned* const ctl = d.ctl;
  int* sid = reinterpret_cast<int*>(sm + d.sints);
  int* stg = sid + T;
  float* const A = sm + d.sA;   // [T][pA]: a GEMM's row operand (the embedding rows, a layer's input columns)
  float* const B = sm + d.sB;   // [T][pB]: the other one (a layer's dZ columns, the logit gradients)
  // ---- this member's roles: layer 0's column block, the hidden layers' block (a, b) with its reduce rows, the head's row block, embedding rows
  const bool in0 = m < d.nb0;
  const int c0 = in0 ? kvm_grid_off(H, d.nb0, m) : 0, w0 = in0 ? kvm_grid_off(H, d.nb0, m + 1) - c0 : 0;
  const bool inh = L > 1 && m < d.ka * d.nb;
  const int ha = inh ? m / d.nb : 0, hb = inh ? m % d.nb : 0;
  const int k0 = inh ? kvm_grid_off(H, d.ka, ha) : 0, wk = inh ? kvm_grid_off(H, d.ka, ha + 1) - k0 : 0;
  const int n0 = inh ? kvm_grid_off(H, d.nb, hb) : 0, wn = inh ? kvm_grid_off(H, d.nb, hb + 1) - n0 : 0;
  const int r0 = inh ? kvm_grid_off(T, d.ka, ha, 1) : 0, wr = inh ? kvm_grid_off(T, d.ka, ha + 1, 1) - r0 : 0;
  const bool inhd = m < d.kah;
  const int kh0 = inhd ? kvm_grid_off(H, d.kah, m) : 0, wkh = inhd ? kvm_grid_off(H, d.kah, m + 1) - kh0 : 0;
  const int e0 = (int)((long long)m * V / P), e1 = (int)((long long)(m + 1) * V / P);
  auto cross = [&]() { return kvm_gang::barrier(ctl, P, abort); };

  // ---- load: the step's tokens and targets, then every block and bias this member holds and (layer 0) the step's embedding rows, behind one wait
  for (int i = tid; i < T; i += NT) { sid[i] = ids[i]; stg[i] = tgt[i]; }
  __syncthreads();
  if (in0) {
    kvm_fsplit::copy_rows(sm + d.sW0, d.pW0, d.w[0] + c0, H, E, w0);
    kvm_fsplit::copy_rows(sm + d.sb0, w0, d.b[0] + c0, H, 1, w0);
    kvm_fmlp::for_cells(T, E >> 2, [&](int r, int c4) { kvm_fmlp::cp_async16(A + r * pA + 4 * c4, d.emb + (size_t)sid[r] * E + 4 * c4); });
  }
  if (inh)
    for (int l = 1; l < L; ++l) {
      kvm_fsplit::copy_rows(sm + d.sW[l], d.pW, d.w[l] + (size_t)k0 * H + n0, H, wk, wn);
      kvm_fsplit::copy_rows(sm + d.sb[l], wn, d.b[l] + n0, H, 1, wn);
    }
  if (inhd) kvm_fsplit::copy_rows(sm + d.sWh, d.pWh, d.wh + (size_t)kh0 * V, V, wkh, V);
  kvm_fmlp::cp_commit_wait();
  if (inhd && VP > V) kvm_fmlp::for_cells(wkh, VP - V, [&](int r, int c) { sm[d.sWh + r * d.pWh + V + c] = 0.f; });   // the head's pad columns: zero logits, zero gradients
  __syncthreads();
  ph(0);

  // ---- forward, layer 0: its columns for every row, kept as gelu'(Z), written as H
  if (in0) {
    kvm_fmlp::gemm<false, false, NRB, false, NRB>(A, pA, sm + d.sW0, d.pW0, B, pB, T, w0, E);
    __syncthreads();
    const float* b0 = sm + d.sb0; float* g0 = sm + d.sg0; float* Y0 = d.Y[0];
    kvm_fmlp::for_cells(T, w0, [&](int r, int c) { float f, g; kvm_fmlp::gelu_fg(B[r * pB + c] + b0[c], f, g); g0[r * w0 + c] = g; Y0[(size_t)r * H + c0 + c] = f; });
  }
  if (!cross()) return;
  // ---- forward, hidden layers: the partial to L2, then this member's piece summed over the Ka partials
  for (int l = 1; l < L; ++l) {
    if (inh) {
      kvm_fsplit::load_rows(A, pA, d.Y[l - 1] + k0, H, T, wk);
      __syncthreads();
      kvm_fmlp::gemm<false, false, NRB, false, NRB>(A, pA, sm + d.sW[l], d.pW, d.part + (size_t)ha * T * H + n0, H, T, wn, wk);
    }
    if (!cross()) return;
    if (inh) {
      const float* bl = sm + d.sb[l]; float* gl = sm + d.sg[l]; float* Yl = d.Y[l];
      kvm_fmlp::for_cells(wr, wn, [&](int r, int c) {
        const size_t i = (size_t)(r0 + r) * H + n0 + c;
        float f, g; kvm_fmlp::gelu_fg(sum_cg(d.part + i, (size_t)T * H, d.ka) + bl[c], f, g); gl[r * wn + c] = g; Yl[i] = f; });
    }
    if (!cross()) return;
  }
  ph(1);
  // ---- head: partial logits over this member's rows of the head (A keeps the top layer's columns for the head's dW below)
  if (inhd) {
    kvm_fsplit::load_rows(A, pA, d.Y[L - 1] + kh0, H, T, wkh);
    __syncthreads();
    kvm_fmlp::gemm<false, false, NRB, false, NRB>(A, pA, sm + d.sWh, d.pWh, d.hpart + (size_t)m * T * VP, VP, T, VP, wkh);
  }
  if (!cross()) return;
  ph(2);
  // ---- cross-entropy: rows m, m + P, ..., one warp a row; the logits summed over the Kah partials in row-block order
  for (int r = m + warp * P; r < T; r += (NT / 32) * P) {
    float x[KVM_GRID_MAXN / 32];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) {
      const int v = lane + 32 * j;
      float s = -INFINITY;
      if (v < V) s = sum_cg(d.hpart + (size_t)r * VP + v, (size_t)T * VP, d.kah);
      x[j] = s; mx = fmaxf(mx, s);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float z = 0.f;
#pragma unroll
    for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) if (lane + 32 * j < V) z += expf(x[j] - mx);
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    const int t = stg[r];
    float lt = 0.f;
#pragma unroll
    for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) if (lane + 32 * j == t) lt = x[j];
    for (int o = 16; o; o >>= 1) lt += __shfl_xor_sync(0xffffffffu, lt, o);   // only the target's lane holds a non-zero term
#pragma unroll
    for (int j = 0; j < KVM_GRID_MAXN / 32; ++j) {
      const int v = lane + 32 * j;
      if (v < VP) d.dlog[(size_t)r * VP + v] = v < V ? (expf(x[j] - mx) / z - (v == t ? 1.f : 0.f)) * inv_t : 0.f;
    }
    if (lane == 0) d.rloss[r] = mx + logf(z) - lt;
  }
  if (!cross()) return;
  if (m == 0 && tid == 0) { float sum = 0.f; for (int r = 0; r < T; ++r) sum += __ldcg(d.rloss + r); loss_out[0] = sum / (float)T; loss_out[1] = inv_t; }
  ph(3);
  // ---- backward, head: the top layer's output gradient from the block before its update, then the block's own update
  if (inhd) {
    kvm_fsplit::load_rows(B, pB, d.dlog, VP, T, VP);
    __syncthreads();
    kvm_fmlp::gemm<false, true, NRB, false, NRB>(B, pB, sm + d.sWh, d.pWh, d.dyp + kh0, H, T, wkh, VP);      // dlogits Wh_block^T
    __syncthreads();
    kvm_fmlp::gemm_m<true, false, true>(A, pA, B, pB, sm + d.sWh, d.pWh, wkh, VP, T, lr);                    // Wh_block -= lr H^T dlogits
  }
  if (!cross()) return;
  ph(4);
  // ---- backward, hidden layers from the top: this member's piece of dZ, then the input-gradient partial and the block's update
  for (int l = L - 1; l >= 1; --l) {
    const int nparts = l == L - 1 ? 1 : d.nb;   // partials of this layer's output gradient: the head's one, or layer l + 1's column blocks
    if (inh) {
      const float* gl = sm + d.sg[l];
      kvm_fmlp::for_cells(wr, wn, [&](int r, int c) {
        const size_t i = (size_t)(r0 + r) * H + n0 + c;
        d.dz[i] = sum_cg(d.dyp + i, (size_t)T * H, nparts) * gl[r * wn + c]; });
    }
    if (!cross()) return;
    if (inh) {
      kvm_fsplit::load_rows(B, pB, d.dz + n0, H, T, wn);
      kvm_fsplit::load_rows(A, pA, d.Y[l - 1] + k0, H, T, wk);
      __syncthreads();
      kvm_fmlp::gemm<false, true, NRB, false, NRB>(B, pB, sm + d.sW[l], d.pW, d.dyp + (size_t)hb * T * H + k0, H, T, wk, wn);   // dZ W_block^T
      __syncthreads();
      kvm_fmlp::gemm_m<true, false, true>(A, pA, B, pB, sm + d.sW[l], d.pW, wk, wn, T, lr);                                  // W_block -= lr X^T dZ
      float* bl = sm + d.sb[l];
      for (int c = tid; c < wn; c += NT) { float s = 0.f; for (int r = 0; r < T; ++r) s += B[r * pB + c]; bl[c] -= lr * s; }   // each of the column's Ka members alike
    }
    if (!cross()) return;
  }
  ph(5);
  // ---- backward, layer 0: dZ for every row of its columns (local), the embedding's input-gradient partial, the block's update
  if (in0) {
    const int nparts = L == 1 ? 1 : d.nb;
    const float* g0 = sm + d.sg0;
    kvm_fmlp::for_cells(T, w0, [&](int r, int c) {
      const size_t i = (size_t)r * H + c0 + c;
      B[r * pB + c] = sum_cg(d.dyp + i, (size_t)T * H, nparts) * g0[r * w0 + c]; });
    kvm_fmlp::for_cells(T, E >> 2, [&](int r, int c4) { kvm_fmlp::cp_async16(A + r * pA + 4 * c4, d.emb + (size_t)sid[r] * E + 4 * c4); });   // X0 again: rows not yet updated
    kvm_fmlp::cp_commit_wait();
    __syncthreads();
    kvm_fmlp::gemm<false, true, NRB, false, NRB>(B, pB, sm + d.sW0, d.pW0, d.dx0 + (size_t)m * T * E, E, T, E, w0);   // dZ W0_block^T
    __syncthreads();
    kvm_fmlp::gemm_m<true, false, true>(A, pA, B, pB, sm + d.sW0, d.pW0, E, w0, T, lr);                              // W0_block -= lr X0^T dZ
    float* b0 = sm + d.sb0;
    for (int c = tid; c < w0; c += NT) { float s = 0.f; for (int r = 0; r < T; ++r) s += B[r * pB + c]; b0[c] -= lr * s; }
  }
  if (!cross()) return;
  ph(6);
  // ---- the embedding rows this member owns: each row's gradient sums its tokens' rows of dX0 in token order, each a fixed-order sum over layer 0's
  // members; the rows go to the warps, a row's columns to the lanes
  for (int v = e0 + warp; v < e1; v += NT / 32)
    for (int c = lane; c < E; c += 32) {
      float g = 0.f;
      for (int t = 0; t < T; ++t)
        if (sid[t] == v) g += sum_cg(d.dx0 + (size_t)t * E + c, (size_t)T * E, d.nb0);
      d.emb[(size_t)v * E + c] -= lr * g;
    }
  // ---- every block and bias back to the model (a hidden bias by its column's first member)
  if (in0) { store_rows(d.w[0] + c0, H, sm + d.sW0, d.pW0, E, w0); store_rows(d.b[0] + c0, H, sm + d.sb0, w0, 1, w0); }
  if (inh)
    for (int l = 1; l < L; ++l) {
      store_rows(d.w[l] + (size_t)k0 * H + n0, H, sm + d.sW[l], d.pW, wk, wn);
      if (ha == 0) store_rows(d.b[l] + n0, H, sm + d.sb[l], wn, 1, wn);
    }
  if (inhd) store_rows(d.wh + (size_t)kh0 * V, V, sm + d.sWh, d.pWh, wkh, V);
  if (!cross()) return;
  ph(7);
}

}  // namespace kvm_fgrid
