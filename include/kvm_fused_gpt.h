#pragma once
// The fused GPT step (design D2): the descriptor one fused run's gang reads, and the gang's sizing (host and device; host-only
// callers include it with plain g++).
//
// A fused run trains one pre-norm GPT (the template of kvm_fuse_gpt.h) for all of its steps as one gang of G blocks (`fgpt_step`, a gang op:
// the static planner starts its G members together). Each member holds a shard of every parameter with its gradient and AdamW moments in
// shared memory across the steps: column shards of q, k, v (by head), fc1 and the head; row shards of proj and fc2; rows of every N x d
// stream. Activations cross members through global memory at two-barrier exchanges (reduce-scatter, all-gather). The member reads the
// step's tokens and targets from the run's streams itself, as the bundle's data windows did. Device side: kvm_fused_gpt.cuh.
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define KVM_FGPT_HD __host__ __device__
#else
#define KVM_FGPT_HD
#endif

#define KVM_FGPT_MAXL 4                       // transformer layers a fused run may have (the shared-memory fit bounds it before this does)
#define KVM_FGPT_PER_LAYER 16                 // parameters a layer: n1 (g, b), q, k, v, proj (w, b each), n2 (g, b), fc1, fc2 (w, b each)
#define KVM_FGPT_MAXP (4 + KVM_FGPT_PER_LAYER * KVM_FGPT_MAXL)   // + wte, nf (g, b), head
#define KVM_FGPT_SMEM_FLOATS 38400            // shared memory a member may use (the static worker's own arena stays out of it, as for the MLP step)
#define KVM_FGPT_GRANULE 8                    // columns of a shard's unit (the mma n8 width): every column shard is a whole number of granules

// Parameter slots, in one fixed order: 0 wte; 1 + 16 l + r for layer l's role r (KVM_FGPT_R_*); then nf's g and b; then the head.
enum KvmFgptRole {
  KVM_FGPT_R_N1G = 0, KVM_FGPT_R_N1B, KVM_FGPT_R_WQ, KVM_FGPT_R_BQ, KVM_FGPT_R_WK, KVM_FGPT_R_BK, KVM_FGPT_R_WV, KVM_FGPT_R_BV,
  KVM_FGPT_R_WO, KVM_FGPT_R_BO, KVM_FGPT_R_N2G, KVM_FGPT_R_N2B, KVM_FGPT_R_W1, KVM_FGPT_R_B1, KVM_FGPT_R_W2, KVM_FGPT_R_B2
};
KVM_FGPT_HD inline int kvm_fgpt_slot_layer(int l, int role) { return 1 + KVM_FGPT_PER_LAYER * l + role; }
KVM_FGPT_HD inline int kvm_fgpt_slot_nfg(int L) { return 1 + KVM_FGPT_PER_LAYER * L; }
KVM_FGPT_HD inline int kvm_fgpt_slot_nfb(int L) { return 2 + KVM_FGPT_PER_LAYER * L; }
KVM_FGPT_HD inline int kvm_fgpt_slot_head(int L) { return 3 + KVM_FGPT_PER_LAYER * L; }
KVM_FGPT_HD inline int kvm_fgpt_nparams(int L) { return 4 + KVM_FGPT_PER_LAYER * L; }

struct alignas(16) KvmFgptDesc {
  // shape: rows (B sequences of T), width, heads x head width, MLP width, vocabulary, layers; the gang's size; the run's steps
  int N, T, B, d, H, hd, F, V, L, G, steps;
  int mm_numerics;                            // the bundle's GEMM numerics bits (matmul_tile flags: precise, trunc, exact), which the fused GEMMs reproduce
  float ln_eps, theta, scale;                 // LayerNorm epsilon, RoPE base, attention scale
  float lr, beta1, beta2, adam_eps, max_norm; // AdamW and the grad-norm clip
  float wd[KVM_FGPT_MAXP];                    // weight decay of every parameter slot
  int prows[KVM_FGPT_MAXP], pcols[KVM_FGPT_MAXP];
  // the step's data, read the way the bundle's data windows read it: element i of step s's tokens is ids_src[step * stride + off + i]
  int ids_stride, ids_off, tgt_stride, tgt_off;
  long long ids_len, tgt_len;                 // elements of the two streams: a window must lie inside its stream
  // global buffers (device pointers, resolved by the executor from bundle buffer indices)
  const int32_t* ids_src; const int32_t* tgt_src;
  int32_t* step;                              // the run's step counter: read once, advanced once a step, written back at the end
  float* const* loss;                         // [steps]: each step's loss buffer (the bundle writes one a step), a device array of pointers
  float* p[KVM_FGPT_MAXP]; float* m[KVM_FGPT_MAXP]; float* v[KVM_FGPT_MAXP];   // parameters and AdamW moments, row-major, contiguous
  float* g[KVM_FGPT_MAXP];                    // the bundle's gradient buffers: written once at the end, so a reader after the run sees the last step's
  float* xch;                                 // the global scratch (kvm_fgpt_scratch's layout, zeroed or not: every element is written before it is read)
  size_t xch_floats;
  unsigned* ctl;                              // the gang barrier's two words (kvm_gang.cuh), zeroed once
};

// The gang's size for a shape: the smallest G whose member state fits `smem_floats`, among the G that split every column-sharded phase into
// whole granules (q, k and v go by head, so G is also checked against the heads); 0 when no G up to `max_g` fits (the run is not fused).
// A member holds 4 floats a parameter of its shard (w, g, m, v) plus its own rows of every stored N x d stream, its heads' q, k, v, o and lse,
// and its fc1 columns' pre-activation and output, plus two staging buffers for the streamed GEMM operands.
KVM_FGPT_HD inline long long kvm_fgpt_member_floats(int N, int d, int H, int F, int V, int L, int G) {
  const long long P = (long long)V * d * 2 + (long long)L * (4LL * d * d + 2LL * d * F + 9LL * d + F) + 2LL * d;   // wte, head; per layer q,k,v,proj, fc1, fc2, biases, norms; nf
  const long long shard = (P + G - 1) / G;
  const int heads_of_member = (H + G - 1) / G;                   // a member owns at most this many heads (0 or 1 when G >= H)
  const int hd = d / H;
  const long long own_rows = (N + G - 1) / G;
  const long long streams = own_rows * d * (6LL * L + 3);        // x, h, x_mid, h2 (and their gradients) a layer, own rows; the final ones
  const long long attn = (long long)heads_of_member * (long long)N * hd * 5 + (long long)heads_of_member * N;   // q, k, v, o, dq/dk/dv reuse; lse
  const long long fc1 = (long long)N * ((F + G - 1) / G) * 2 * L; // pre and u of its fc1 columns, every layer
  const long long staging = 2LL * 16 * 128;                       // two 16 x 128 operand stages of the streamed A operand
  return 4 * shard + streams + attn + fc1 + staging;
}
KVM_FGPT_HD inline int kvm_fgpt_gang(int N, int d, int H, int F, int V, int L, int smem_floats, int max_g) {
  for (int G = 1; G <= max_g; ++G) {
    if (F % KVM_FGPT_GRANULE || V % KVM_FGPT_GRANULE || d % KVM_FGPT_GRANULE) return 0;
    if ((F / KVM_FGPT_GRANULE) % G && G < F / KVM_FGPT_GRANULE) continue;   // fc1's granules split evenly (or one at most a member)
    if (kvm_fgpt_member_floats(N, d, H, F, V, L, G) <= smem_floats) return G;
  }
  return 0;
}

// The device program's global scratch (v0 keeps every activation and partial in global memory; later forms keep the shards in shared memory):
// offsets in floats into KvmFgptDesc::xch, laid out by the host for the run's shape and gang, every array 16-byte aligned.
//  * Saved for the backward, a layer l: its input x[l] (x[L] is the last layer's output), LN1's output h[l] and row statistics m1, r1, q, k
//    and v after RoPE, attention's output o and log-sum-exp lse, the residual after attention xm, LN2's output h2 and statistics m2, r2, fc1's
//    pre-activation pre and output u. Then the final LayerNorm's output hf and statistics mf, rf, the logits and their gradient dlog.
//  * Exchanges: ce, the cross-entropy's per-member row statistics (max, sum of exp) [G][N][2]; tl, each row's target logit; gx, the
//    residual-stream gradient (N x d, own rows written); part, one N x d partial a contributing member, read by the reduce-scatter (one buffer:
//    an all-gather's barrier always separates a reduce-scatter's reads from the next one's writes); lnp, each LayerNorm's own-row partials of
//    dgamma and dbeta [2 L + 1][G][2 d] (LN1 of layer l is 2 l, LN2 2 l + 1, the final one 2 L); rpart, the own-row partials of proj's and
//    fc2's bias gradients [2 L][G][d] (bo of layer l is 2 l, b2 2 l + 1); gn, each member's sum of squares of the gradients it owns.
//  * A member's own work arrays: dy, the reduce-scatter's sum of its rows; du, dq, dk, dv and dout (attention's output gradient).
struct KvmFgptScratch {
  long long x[KVM_FGPT_MAXL + 1], h[KVM_FGPT_MAXL], m1[KVM_FGPT_MAXL], r1[KVM_FGPT_MAXL], q[KVM_FGPT_MAXL], k[KVM_FGPT_MAXL], v[KVM_FGPT_MAXL],
      o[KVM_FGPT_MAXL], lse[KVM_FGPT_MAXL], xm[KVM_FGPT_MAXL], h2[KVM_FGPT_MAXL], m2[KVM_FGPT_MAXL], r2[KVM_FGPT_MAXL], pre[KVM_FGPT_MAXL],
      u[KVM_FGPT_MAXL];
  long long hf, mf, rf, logits, dlog, ce, tl, gx, part, lnp, rpart, gn, dy, du, dq, dk, dv, dout, total;
};
KVM_FGPT_HD inline KvmFgptScratch kvm_fgpt_scratch(int N, int d, int H, int F, int V, int L, int G) {
  KvmFgptScratch s{};
  long long at = 0;
  const long long Nd = (long long)N * d, NF = (long long)N * F;
  auto take = [&at](long long n) { const long long o = at; at += (n + 3) & ~3LL; return o; };
  for (int l = 0; l <= L; ++l) s.x[l] = take(Nd);
  for (int l = 0; l < L; ++l) {
    s.h[l] = take(Nd); s.m1[l] = take(N); s.r1[l] = take(N); s.q[l] = take(Nd); s.k[l] = take(Nd); s.v[l] = take(Nd); s.o[l] = take(Nd);
    s.lse[l] = take((long long)N * H); s.xm[l] = take(Nd); s.h2[l] = take(Nd); s.m2[l] = take(N); s.r2[l] = take(N); s.pre[l] = take(NF);
    s.u[l] = take(NF);
  }
  s.hf = take(Nd); s.mf = take(N); s.rf = take(N); s.logits = take((long long)N * V); s.dlog = take((long long)N * V);
  s.ce = take((long long)G * N * 2); s.tl = take(N); s.gx = take(Nd); s.part = take((long long)G * Nd);
  s.lnp = take((long long)(2 * L + 1) * G * 2 * d); s.rpart = take((long long)2 * L * G * d); s.gn = take(G);
  s.dy = take(Nd); s.du = take(NF); s.dq = take(Nd); s.dk = take(Nd); s.dv = take(Nd); s.dout = take(Nd);
  s.total = at;
  return s;
}

// The v0 program's shared memory, in floats: the block GEMM's two staged K slices (KVM_FGPT_GK x (KVM_FGPT_GT + 1) each), two score rows of
// T a warp (attention), the attention backward's delta (T), each row's log-sum-exp of the cross-entropy (N), the step's tokens and targets
// (N ints each), and a block reduction's partials and flags.
#define KVM_FGPT_NT 256                       // threads of a member's block (the static worker's KVM_THREADS)
#define KVM_FGPT_GK 16                        // the block GEMM's K slice
#define KVM_FGPT_GT 64                        // its output tile (KVM_FGPT_GT x KVM_FGPT_GT), 4 x 4 outputs a thread
KVM_FGPT_HD inline long long kvm_fgpt_smem_floats(int N, int T) {
  return 2LL * KVM_FGPT_GK * (KVM_FGPT_GT + 1) + 2LL * (KVM_FGPT_NT / 32) * T + T + N + 2LL * N + 16;
}

// The faults a member reports through its caller's `fail(code)` (kvm_fused_gpt.cuh's run; the executor reports one as KVM_ERR_BOUNDS with the
// code and the member as its arguments, which sets the error word every member's barrier watches): a token or a target outside the vocabulary,
// a data window outside its stream, a step counter AdamW refuses (below 1 after the increment), a scratch smaller than the layout, or a shape
// the program does not take.
enum KvmFgptError : int { KVM_FGPT_E_TOKEN = 1, KVM_FGPT_E_TARGET, KVM_FGPT_E_WINDOW, KVM_FGPT_E_STEP, KVM_FGPT_E_SCRATCH, KVM_FGPT_E_SHAPE };
