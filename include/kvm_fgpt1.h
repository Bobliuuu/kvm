#pragma once
// The fused GPT step on one block (kvm_fgpt1.cuh) as the host sees it: its shared-memory layout, whether it runs a shape, and the relaxed
// numerics the executor's instance is built with. Host and device: the GPT pass (kvm_fuse_gpt.h) picks the form with it, the executor sizes
// the launch with it, and run1 checks its own run with the same function, so the three cannot disagree.
#include "kvm_fused_gpt.h"

// The executor's one-block instance (kvm_fgpt_kernel.cuh's kvm_fgpt1_kernel): C3 at this KVM_FGPT1_FAST mask. 1 the attention softmax four
// lanes a row, 2 the CE's row statistics eight lanes a row, 4 the LayerNorms eight lanes a row, 16 AdamW's update without IEEE division:
// each judged against a float64 reference at x1.000 the gang's. 8 (GELU's tanh by __expf) is left out: it read slower.
#define KVM_FGPT1_EXEC_FAST 23
#define KVM_FGPT1_ROW_MAX 32   // the most keys a softmax row (T) the relaxed softmax takes, and half the logits the relaxed CE takes
#ifndef KVM_FGPT1_MAXD
#define KVM_FGPT1_MAXD 32      // the widest model (d) the form takes: its row helpers unroll this many columns a row and no more (LNAN, Nov 13:
#endif                         // three unrolled column groups of four sitting empty at d 32 cost the LayerNorm 29%); a wider run takes its gang
static_assert(KVM_FGPT1_MAXD % 32 == 0 && KVM_FGPT1_MAXD >= 32 && KVM_FGPT1_MAXD <= 128, "KVM_FGPT1_MAXD: 32, 64, 96 or 128 columns");

// Shared-memory layout of one shape, in floats from the block's start, computed alike on host and device. Every GEMM operand is padded as
// kvm_fgpt pads it: rows to whole 16-row tiles, k-contiguous rows to a 4 mod 8 pitch (kvm_fgpt_ldk), n-contiguous to 8 mod 16 (ldmn).
// Buffers share space by liveness (every activation kept apart is 45,432 floats for gpt-tiny, over the A100's 41,524):
//   - wte stays in global memory (L2) like the moments: the step gathers its 32 wte rows; the head is resident like every other
//     parameter, at HEAD, after the activations;
//   - the norms' outputs are recomputed from x and their stored row statistics, into region NH, where a GEMM needs them as an operand
//     (the same expression from the same values, so the same bits);
//   - region A holds, one phase at a time: the scores (the backward recomputes P, as run does), GELU's output for fc2 and the backward's
//     temporaries;
//   - region PART holds the projection's per-head partials, fc2's partial, then the logits, then the backward's dh partials;
//   - the activations' span (x[0] through dy) is AdamW's ring: every buffer in it is dead from the backward's last barrier to the next
//     step's embedding, and the update streams the moments through it (kvm_fgpt1.cuh's adamw).
struct KvmFgpt1Layout {
  long long total = 0;       // floats used; 0 when the shape is not one this form runs
  int ids, tgt, lse, tl, rtab, red;   // tokens and targets, two steps' slots each (step t in slot t & 1, the next prefetched into the other);
                                      // the CE's per-row values; RoPE's table; block_sum's doubles (two values a warp)
  int p[KVM_FGPT_MAXP], pld[KVM_FGPT_MAXP];   // resident parameters (wte -1, in global; the head -1, at `head`): offset and padded pitch
  int wqkv, wqkv_stride, bqkv;                // a layer's q | k | v weights as one [d x 3d] B operand, and their [3d] biases
  int x[KVM_FGPT_MAXL + 1];  // the residual stream into layer l (x[L]: into the final norm), N x d, unpadded
  int xm[KVM_FGPT_MAXL];     // x + proj, N x d
  int st1[KVM_FGPT_MAXL], st2[KVM_FGPT_MAXL], stf;   // the norms' row statistics: mean [N], then rstd [N]
  int qkv[KVM_FGPT_MAXL];    // q | k | v of every row and head, after RoPE: r16(N) rows at ldk(3d)
  int alse[KVM_FGPT_MAXL];   // the attention rows' log-sum-exp, [B H T]
  int o[KVM_FGPT_MAXL];      // attention out (before proj), every head side by side: r16(N) rows at ldk(d)
  int pre[KVM_FGPT_MAXL];    // fc1 + b1 before GELU: r16(N) rows at ldk(F) (the backward's dU overwrites it in place)
  int nh;                    // NH: a norm's output, recomputed where a GEMM reads it: r16(N) rows at ldk(d)
  int a, a_floats;           // region A
  int part, part_floats;     // region PART
  int gx, dy;                // the backward's residual-stream gradient and a norm's output gradient, N x d each
  int ring, ring_floats;     // AdamW's ring: the activations' span, x[0] through dy
  int head;                  // the head, resident for the run (loaded at its start, updated in place, stored back at its end): r8(d) rows
                             // at ldmn(V), its padding zero
  int ldd, ldq, ldS, ldF;    // pitches: d-wide, 3d-wide, the scores', F-wide A operands
  int ldV;                   // the logits' pitch in PART: V (run's), or a 4 mod 8 pitch where the CE reads a row a thread pair (FAST & 2)
};

// Where every buffer goes for a shape at relaxed-numerics mask `fast` (or total 0 when this form can't run it).
KVM_FGPT_HD inline KvmFgpt1Layout kvm_fgpt1_layout(int N, int T, int d, int H, int F, int V, int L, int fast) {
  KvmFgpt1Layout y;
  const int hd = H > 0 ? d / H : 0, B = T > 0 ? N / T : 0;
  if (L < 1 || L > KVM_FGPT_MAXL || T <= 0 || N % T || hd <= 0 || hd * H != d || hd % 8 || d % 4 || F <= 0 || V <= 0) return y;
  long long at = 0;
  auto take = [&](long long n) { const int o = (int)at; at += (n + 3) & ~3LL; return o; };   // 16-byte aligned
  auto mx = [](long long u, long long v) { return u > v ? u : v; };
  const int N16 = kvm_fgpt_r16(N), T16 = kvm_fgpt_r16(T);
  y.ldd = kvm_fgpt_ldk(d); y.ldq = kvm_fgpt_ldk(3 * d); y.ldS = kvm_fgpt_ldk(T16); y.ldF = kvm_fgpt_ldk(F);
  y.ldV = (fast & 2) ? kvm_fgpt_ldk(V) : V;
  y.ids = take(2LL * N); y.tgt = take(2LL * N); y.lse = take(N); y.tl = take(N);
  y.rtab = take((long long)T * hd);
  y.red = take(4 * (KVM_FGPT_NT / 32));
  for (int s = 0; s < KVM_FGPT_MAXP; ++s) { y.p[s] = -1; y.pld[s] = 0; }
  y.wqkv_stride = (int)(((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(3 * d) + 3) & ~3LL);
  y.wqkv = take((long long)L * y.wqkv_stride);
  y.bqkv = take((long long)L * 3 * d);
  for (int l = 0; l < L; ++l) {
    const int s0 = kvm_fgpt_slot_layer(l, 0);
    for (int r : {KVM_FGPT_R_N1G, KVM_FGPT_R_N1B, KVM_FGPT_R_N2G, KVM_FGPT_R_N2B, KVM_FGPT_R_BO, KVM_FGPT_R_B2}) { y.p[s0 + r] = take(d); y.pld[s0 + r] = d; }
    y.p[s0 + KVM_FGPT_R_B1] = take(F); y.pld[s0 + KVM_FGPT_R_B1] = F;
    y.pld[s0 + KVM_FGPT_R_WO] = kvm_fgpt_ldmn(d);  y.p[s0 + KVM_FGPT_R_WO] = take((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(d));
    y.pld[s0 + KVM_FGPT_R_W1] = kvm_fgpt_ldmn(F);  y.p[s0 + KVM_FGPT_R_W1] = take((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(F));
    y.pld[s0 + KVM_FGPT_R_W2] = kvm_fgpt_ldmn(d);  y.p[s0 + KVM_FGPT_R_W2] = take((long long)kvm_fgpt_r8(F) * kvm_fgpt_ldmn(d));
  }
  y.p[kvm_fgpt_slot_nfg(L)] = take(d); y.pld[kvm_fgpt_slot_nfg(L)] = d;
  y.p[kvm_fgpt_slot_nfb(L)] = take(d); y.pld[kvm_fgpt_slot_nfb(L)] = d;
  y.pld[kvm_fgpt_slot_head(L)] = kvm_fgpt_ldmn(V);   // the head's staged pitch in region A; it is not resident (y.p stays -1)
  for (int l = 0; l <= L; ++l) y.x[l] = take((long long)N * d);
  for (int l = 0; l < L; ++l) {
    y.xm[l] = take((long long)N * d);
    y.st1[l] = take(2LL * N); y.st2[l] = take(2LL * N);
    y.qkv[l] = take((long long)N16 * y.ldq);
    y.alse[l] = take((long long)B * H * T);
    y.o[l] = take((long long)N16 * y.ldd);
    y.pre[l] = take((long long)N16 * y.ldF);
  }
  y.stf = take(2LL * N);
  y.nh = take((long long)N16 * y.ldd);
  // region A: the largest of its tenants
  const long long a_scores = (long long)B * H * T16 * y.ldS;
  const long long a_gelu = (long long)N16 * y.ldF;
  const long long a_attnb = 4LL * T16 * kvm_fgpt_ldk(kvm_fgpt_r16(hd)) + (long long)T16 * y.ldS + (long long)T16 * kvm_fgpt_ldk(2 * hd) + T;
  y.a_floats = (int)((mx(mx(a_scores, a_gelu), a_attnb) + 3) & ~3LL);
  y.a = take(y.a_floats);
  y.part_floats = (int)((mx((long long)(H > 1 ? H : 1) * N * d, (long long)N * y.ldV) + 3) & ~3LL);
  y.part = take(y.part_floats);
  y.gx = take((long long)N * d); y.dy = take((long long)N * d);
  y.ring = y.x[0]; y.ring_floats = (int)(at - y.x[0]);
  y.head = take((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(V));
  y.total = at;
  return y;
}

// Whether the one-block form runs a shape, given its layout: 0, KVM_FGPT_E_SHAPE (a shape, mask or GEMM numerics it does not take) or
// KVM_FGPT_E_SCRATCH (its layout over `smem_floats`). run1 checks itself with this; kvm_fgpt1_check is the same from the shape alone.
KVM_FGPT_HD inline int kvm_fgpt1_check_layout(const KvmFgpt1Layout& y, int N, int T, int B, int d, int H, int hd, int F, int V, int mm_numerics,
                                              int fast, long long smem_floats) {
  const bool fast_ok = ((fast & 1) == 0 || T <= KVM_FGPT1_ROW_MAX) && ((fast & 2) == 0 || (V <= 2 * KVM_FGPT1_ROW_MAX && V % 4 == 0)) &&
                       ((fast & 4) == 0 || d % 4 == 0) && d <= KVM_FGPT1_MAXD;
  // whole GEMM tiles in every product of the step (rows and d, F, T a multiple of 16; V and the head width of 8): its GEMMs run without
  // bounds branches in their epilogues (kvm_fused_gpt.cuh's gemm_on<..., Full>)
  const bool whole = N % 16 == 0 && T % 16 == 0 && d % 16 == 0 && F % 16 == 0 && V % 8 == 0 && hd % 8 == 0;
  if (y.total == 0 || !fast_ok || !whole || T <= 0 || B != N / T || hd * H != d || (mm_numerics & ~16) != 0) return KVM_FGPT_E_SHAPE;
  if (y.total > smem_floats) return KVM_FGPT_E_SCRATCH;
  return 0;
}
KVM_FGPT_HD inline int kvm_fgpt1_check(int N, int T, int B, int d, int H, int F, int V, int L, int mm_numerics, int fast, long long smem_floats) {
  const int hd = H > 0 ? d / H : 0;
  return kvm_fgpt1_check_layout(kvm_fgpt1_layout(N, T, d, H, F, V, L, fast), N, T, B, d, H, hd, F, V, mm_numerics, fast, smem_floats);
}
// kvm_fgpt1_check's code in words, for a log (host).
inline const char* kvm_fgpt1_why(int code) {
  return code == 0 ? "it runs" : code == KVM_FGPT_E_SCRATCH ? "its state is over one block's shared memory"
                                                            : "a shape, numerics mask or GEMM numerics the one-block form does not take";
}
