#pragma once
// The split fused MLP step, device side: one member's share of one step (kvm_fused_split.h has the design, the descriptor and the layout).
// Used by the op `fmlp_split` of the static schedule path (a gang op: its members run at once, placed by kvm_static_plan) and by
// bench/fused_mlp/split_bed.cu (the same code, timed alone). GEMMs are kvm_fused_mlp.cuh's (m16n8k8 TF32 mma); the barrier is kvm_gang.cuh's.
#include <cstdint>
#include "kvm_fused_split.h"
#include "kvm_fused_mlp.cuh"
#include "kvm_gang.cuh"

namespace kvm_fsplit {

using kvm_fmlp::NT;
constexpr int R = KVM_SPLIT_R;

__device__ __forceinline__ float4 ld_cg4(const float* p) { float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v; }

// rows x cols floats (cols a multiple of 4; 16-byte aligned rows on both sides) from global memory (L2) to shared memory by 16-byte cp.async,
// no wait: the caller commits and waits. The step's start issues every array at once (the share's weights, biases, head rows, the slice's embedding
// rows) behind one wait: plain loads there pay one round trip an array (measured on the 3080 Ti: the load phase 9.2k cycles as one group, 12.9k as
// plain loads, h576-d1).
__device__ __forceinline__ void copy_rows(float* dst, int dst_ld, const float* src, size_t src_ld, int rows, int cols) {
  kvm_fmlp::for_cells(rows, cols >> 2, [&](int r, int c4) { kvm_fmlp::cp_async16(dst + (size_t)r * dst_ld + 4 * c4, src + (size_t)r * src_ld + 4 * c4); });
}
// The same rows by plain L2 loads, 8 float4 a thread in flight before their stores. For one array read right after a barrier (an exchange),
// one cp.async group read 1.5x slower in the exchange read bed (xread_bed, 3080 Ti and A100); KVM_SPLIT_XREAD_ASYNC=1 takes cp.async for the exchanges too.
#ifndef KVM_SPLIT_XREAD_ASYNC
#define KVM_SPLIT_XREAD_ASYNC 0
#endif
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
__device__ __forceinline__ void prefetch_desc(const KvmSplitDesc* gd, float* sm) {
  const int chunks = (int)((sizeof(KvmSplitDesc) + 15) / 16);
  if (threadIdx.x < chunks) kvm_fmlp::cp_async16(sm + threadIdx.x * 4, reinterpret_cast<const float*>(gd) + threadIdx.x * 4);
}

struct NoPhases { __device__ __forceinline__ void operator()(int) const {} };
constexpr int NPH = 9;   // the phases `ph(i)` marks, once a step in this order (every thread calls it)
// 0 load, 1 forward (GEMMs, GELU, all-gathers), 2 logits partial, 3 barrier: logits, 4 logits summed + cross-entropy, 5 backward (GEMMs, reduce-scatters),
// 6 barrier: partials, 7 owners: parameters, 8 owners: embedding + the last barrier

// One step of member `member` (= q * S + s) of a split run: `ids` and `tgt` are the step's T tokens and targets, `abort` the run's error word
// (kvm_gang::barrier: a failure elsewhere ends the step at its next crossing instead of hanging it; nullptr in a bed), `loss_out` the step's loss (member 0 writes loss_out[0] = sum / T and loss_out[1] = 1 / T, as softmax_ce_reduce does). The members
// cross 2 x L + 1 barriers; the last one is the step's end, so every update is visible to every member when any of them returns.
template <int S, int Q, class PH>
__device__ __forceinline__ void split_run(int member, const int32_t* ids, const int32_t* tgt, float* loss_out, float* sm, const int* abort, PH ph) {
  kvm_fmlp::cp_commit_wait();
  __syncthreads();
  const KvmSplitDesc& d = *reinterpret_cast<const KvmSplitDesc*>(sm);
  const int q = member / S, s = member % S, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, members = Q * S;
  const int h0 = d.share0[s], w = d.share0[s + 1] - h0, E = d.E, V = d.V, VP = d.VP, HP = d.HP, HR = d.HR, H = d.H, T = d.T, L = d.L;
  float* Whs = sm + d.Whs; float* bs = sm + d.bs; float* X0 = sm + d.X0; float* Hin = sm + d.Hin;
  float* ZG = sm + d.ZG; float* Hs = sm + d.Hs; float* Lg = sm + d.Lg; float* dZ = sm + d.dZ; float* gtok = sm + d.gtok; float* rl = sm + d.rl;
  int* sid = reinterpret_cast<int*>(sm + d.ints);   // [T]: the step's token ids
  int* tg = sid + T;                                // [R]: the slice's targets
  int* mlist = tg + R;                              // [T]: positions of the tokens this member's embedding rows own, in token order
  int* wcnt = mlist + T;                            // [9]: per-warp match counts, then offsets
  const float inv_t = 1.f / (float)T, lr = d.lr;
  const int erow0 = (int)((long long)member * V / members), erow1 = (int)((long long)(member + 1) * V / members);
  unsigned* const ctl = d.ctl;
  // ---- load: the share's columns of every layer (zero past w, and past H in the hidden layers' rows), its head rows (zero past V), the slice's
  // embedding rows, the step's tokens and the slice's targets. The pads are written each step: the owners' scratch sits over the weights.
  for (int l = 0; l < L; ++l) {
    float* Ws = sm + d.Ws[l];
    const int K = l ? H : E, KR = l ? HR : E;
    kvm_fmlp::for_cells(KR, HP, [&](int k, int c) { if (k >= K || c >= w) Ws[k * HP + c] = 0.f; });
    for (int c = tid; c < HP; c += NT) if (c >= w) bs[l * HP + c] = 0.f;
  }
  kvm_fmlp::for_cells(HP, VP, [&](int r, int v) { if (r >= w || v >= V) Whs[r * VP + v] = 0.f; });
  for (int l = 0; l < L; ++l) {
    copy_rows(sm + d.Ws[l], HP, d.w[l] + h0, H, l ? H : E, w);   // W_l[:, h0 .. h0 + w)
    copy_rows(bs + l * HP, 0, d.b[l] + h0, 0, 1, w);
  }
  copy_rows(Whs, VP, d.wh + (size_t)h0 * V, V, w, V);            // Wh[h0 .. h0 + w, :]
  for (int r = 0; r < R; ++r) copy_rows(X0 + r * E, 0, d.we + (size_t)ids[q * R + r] * E, 0, 1, E);
  for (int k = tid; k < T; k += NT) sid[k] = ids[k];
  for (int r = tid; r < R; r += NT) tg[r] = tgt[q * R + r];
  kvm_fmlp::cp_commit_wait();
  __syncthreads();
  ph(0);
  // ---- forward
  const float* A = X0; int lda = E, K = E;
  for (int l = 0; l < L; ++l) {
    float* ZGl = ZG + l * R * HP;
    kvm_fmlp::gemm_r<false, false>(A, lda, sm + d.Ws[l], HP, ZGl, HP, R, HP, K);   // Z = A W_l,s
    __syncthreads();
    kvm_fmlp::for_cells(R, HP, [&](int r, int c) { const int i = r * HP + c; float f, g; kvm_fmlp::gelu_fg(ZGl[i] + bs[l * HP + c], f, g); Hs[i] = c < w ? f : 0.f; ZGl[i] = c < w ? g : 0.f; });
    __syncthreads();
    if (l + 1 < L) {   // the all-gather: this member's columns of H to L2, then the slice's whole H back as the next layer's input
      float* xh = d.XH[l] + (size_t)q * R * HR;
      kvm_fmlp::for_cells(R, w >> 2, [&](int r, int c4) { *reinterpret_cast<float4*>(xh + (size_t)r * HR + h0 + 4 * c4) = *reinterpret_cast<const float4*>(Hs + r * HP + 4 * c4); });
      if (!kvm_gang::barrier(ctl, members, abort)) return;
      if (KVM_SPLIT_XREAD_ASYNC) { copy_rows(Hin, HR, xh, HR, R, HR); kvm_fmlp::cp_commit_wait(); } else load_rows(Hin, HR, xh, HR, R, HR);   // pad columns past H stay zero in L2
      __syncthreads();
      A = Hin; lda = HR; K = H;
    }
  }
  ph(1);
  kvm_fmlp::gemm_r<false, false>(Hs, HP, Whs, VP, d.P + (size_t)member * R * VP, VP, R, VP, HP);   // this share's part of the logits (zero past V)
  ph(2);
  if (!kvm_gang::barrier(ctl, members, abort)) return;
  ph(3);
  // ---- the slice's S partials summed in share order, four logits a thread at a time; then the cross-entropy
  kvm_fmlp::for_cells(R, VP / 4, [&](int r, int v4) {
    float4 x[S];
#pragma unroll
    for (int t = 0; t < S; ++t) x[t] = ld_cg4(d.P + ((size_t)(q * S + t) * R + r) * VP + 4 * v4);
    float4 a = x[0];
#pragma unroll
    for (int t = 1; t < S; ++t) { a.x += x[t].x; a.y += x[t].y; a.z += x[t].z; a.w += x[t].w; }
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
  kvm_fmlp::gemm_r<false, true>(Lg, VP, Whs, VP, dZ, HP, R, HP, VP);                                 // dH_s = dlogits Wh_s^T
  kvm_fmlp::gemm_m<true, false>(Hs, HP, Lg, VP, d.dWh + (size_t)member * HP * VP, VP, HP, VP, R);    // dWh_s = H_s^T dlogits
  __syncthreads();
  for (int l = L - 1; l >= 0; --l) {
    const float* ZGl = ZG + l * R * HP;
    float* Ws = sm + d.Ws[l];
    for (int i = tid; i < R * HP; i += NT) dZ[i] *= ZGl[i];                                         // dZ = dH_s gelu'(Z) (zero past w)
    __syncthreads();
    for (int c = tid; c < HP; c += NT) { float a = 0.f; for (int r = 0; r < R; ++r) a += dZ[r * HP + c]; d.db[l][(size_t)member * HP + c] = a; }
    if (l == 0) {
      kvm_fmlp::gemm_m<true, false>(X0, E, dZ, HP, d.dW[0] + (size_t)member * E * HP, HP, E, HP, R);          // dW_0,s = X0^T dZ
      kvm_fmlp::gemm_r<false, true>(dZ, HP, Ws, HP, d.dX0 + (size_t)member * R * E, E, R, E, HP);             // dX0 part = dZ W_0,s^T
      __syncthreads();
    } else {
      const float* xin = d.XH[l - 1] + (size_t)q * R * HR;   // the layer's input, the slice's whole H_(l-1), read back from L2
      if (KVM_SPLIT_XREAD_ASYNC) { copy_rows(Hin, HR, xin, HR, R, HR); kvm_fmlp::cp_commit_wait(); } else load_rows(Hin, HR, xin, HR, R, HR);
      __syncthreads();
      for (int m0 = 0; m0 < HR; m0 += 128)                                                                    // dW_l,s = H_(l-1)^T dZ
        kvm_fmlp::gemm_m<true, false>(Hin + m0, HR, dZ, HP, d.dW[l] + ((size_t)member * HR + m0) * HP, HP, min(128, HR - m0), HP, R);
      float* xd = d.XD[l & 1] + (size_t)member * R * HR;   // alternating with the layer: a slot is rewritten only after the next barrier (kvm_fused_split.h)
      for (int n0 = 0; n0 < HR; n0 += 128)                                                                    // the full-width partial dZ W_l,s^T
        kvm_fmlp::gemm_r<false, true>(dZ, HP, Ws + n0 * HP, HP, xd + n0, HR, R, min(128, HR - n0), HP);
      if (!kvm_gang::barrier(ctl, members, abort)) return;
      kvm_fmlp::for_cells(R, HP / 4, [&](int r, int c4) {   // the reduce-scatter: this share's columns of the slice's S partials, in share order
        float4 x[S];
#pragma unroll
        for (int t = 0; t < S; ++t) x[t] = ld_cg4(d.XD[l & 1] + ((size_t)(q * S + t) * R + r) * HR + h0 + 4 * c4);
        float4 a = x[0];
#pragma unroll
        for (int t = 1; t < S; ++t) { a.x += x[t].x; a.y += x[t].y; a.z += x[t].z; a.w += x[t].w; }
        const int c = 4 * c4;   // columns past w belong to the next share: zero here
        a.x = c < w ? a.x : 0.f; a.y = c + 1 < w ? a.y : 0.f; a.z = c + 2 < w ? a.z : 0.f; a.w = c + 3 < w ? a.w : 0.f;
        *reinterpret_cast<float4*>(dZ + r * HP + c) = a;
      });
      __syncthreads();
    }
  }
  ph(5);
  if (!kvm_gang::barrier(ctl, members, abort)) return;
  ph(6);
  // ---- owners, parameters: the Q members of share s split its parameters in groups of four, every layer's and the head's in one index; each sums
  // the Q slices' partials in slice order
  {
    const int w4 = w >> 2, V4 = V >> 2;
    int ng = 0, beg[KVM_SPLIT_MAXL + 1];
    for (int l = 0; l < L; ++l) { beg[l] = ng; ng += ((l ? H : E) + 1) * w4; }   // the layer's weight rows, then its bias
    beg[L] = ng; ng += w * V4;
    const int g0 = (int)((long long)q * ng / Q), g1 = (int)((long long)(q + 1) * ng / Q);
    for (int g = g0 + tid; g < g1; g += NT) {
      const float* src; size_t stride; float* dst;   // slice 0's partial of these four parameters, the distance to the next slice's, and the parameters
      int l = 0; while (l < L && g >= beg[l + 1]) ++l;
      if (l < L) {
        const int Kl = l ? H : E, KR = l ? HR : E, j = g - beg[l], k = j / w4, c = 4 * (j % w4);
        if (k < Kl) { src = d.dW[l] + ((size_t)s * KR + k) * HP + c; stride = (size_t)S * KR * HP; dst = d.w[l] + (size_t)k * H + h0 + c; }
        else { src = d.db[l] + (size_t)s * HP + c; stride = (size_t)S * HP; dst = d.b[l] + h0 + c; }
      } else {
        const int j = g - beg[L], r = j / V4, v = 4 * (j % V4);
        src = d.dWh + ((size_t)s * HP + r) * VP + v; stride = (size_t)S * HP * VP; dst = d.wh + (size_t)(h0 + r) * V + v;
      }
      float4 x[Q];
#pragma unroll
      for (int t = 0; t < Q; ++t) x[t] = ld_cg4(src + (size_t)t * stride);
      float4 a = x[0];
#pragma unroll
      for (int t = 1; t < Q; ++t) { a.x += x[t].x; a.y += x[t].y; a.z += x[t].z; a.w += x[t].w; }
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
    for (int i = tid; i < nm * E; i += NT) {   // each matched token's gradient: its slice's S parts in share order
      const int j = i / E, e = i % E, k = mlist[j], sq = k / R, r = k % R;
      float x[S];
#pragma unroll
      for (int t = 0; t < S; ++t) x[t] = __ldcg(d.dX0 + ((size_t)(sq * S + t) * R + r) * E + e);   // other members' partials: L2, never a stale L1 line
      float a = x[0];
#pragma unroll
      for (int t = 1; t < S; ++t) a += x[t];
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
  if (!kvm_gang::barrier(ctl, members, abort)) return;
  ph(8);
}

}  // namespace kvm_fsplit
