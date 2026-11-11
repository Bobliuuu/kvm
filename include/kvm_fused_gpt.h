#pragma once
// The fused GPT step: the descriptor one fused run's gang reads, and the gang's sizing (host and device; host-only callers include it with
// plain g++).
//
// A fused run trains one pre-norm GPT (the template of kvm_fuse_gpt.h) for all of its steps as one gang of G blocks (its G `fgpt_step`
// members, launched together as one kernel: kvm_fgpt_kernel.cuh). Each member holds a shard of every parameter with its gradient and AdamW moments in
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
#define KVM_FGPT_SMEM_FLOATS 38400            // the pass's default budget of shared memory a member, in floats (the executor passes the device's)
#define KVM_FGPT_MAX_GANG 32                  // the widest gang the policy picks (kvm_fgpt_gang)

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
  int qb;                                     // attention's query blocks a (sequence, head) unit (kvm_fgpt_qblocks gives the default); T a multiple of 16 qb
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
  float* bc;                                  // [2 steps]: each step's AdamW bias corrections, 1 - beta1^t and 1 - beta2^t, filled by member 0
                                              // at the run's start (a double pow is microseconds where FP64 is slow; once a run, not a step)
  unsigned* ctl;                              // the gang barrier's flag line: KVM_FGPT_MAX_GANG words, one a member, 128-byte aligned
                                              // (kvm_fused_gpt.cuh's cross_line), zeroed once
};

// The device program's global scratch: offsets in floats into KvmFgptDesc::xch, laid out by the host for the run's shape and gang, every
// array 16-byte aligned. Every element is written before it is read.
//  * Saved for the backward, a layer l: its input x[l] (x[L] is the last layer's output), LN1's output h[l] and row statistics m1, r1; q, k
//    (after RoPE), v and attention's output o, each written by the (sequence, head) unit that owns its rows and columns, and the log-sum-exp
//    lse [(sequence H + head) T + position]; the residual after attention xm, LN2's output h2 and statistics m2, r2; fc1's pre-activation pre
//    (its output is recomputed from it). Then the final LayerNorm's output hf and statistics mf, rf, and the logits.
//  * Exchanges: ce, the cross-entropy's per-member row statistics (max, sum of exp) [G][N][2]; tl, each row's target logit; gx, the
//    residual-stream gradient (N x d, own rows written); part, the reduce-scatters' partials: one N x d a contributing member (MLP, head), or
//    one T x d a (sequence, head) unit (attention), max(G, H) N x d floats. One buffer: an all-gather's barrier always separates a
//    reduce-scatter's reads from the next one's writes. lnp, each LayerNorm's own-row partials of dgamma and dbeta [2 L + 1][G][2 d] (LN1
//    of layer l is 2 l, LN2 2 l + 1, the final one 2 L); rpart, the own-row partials of proj's and fc2's bias gradients [2 L][G][d] (bo of
//    layer l is 2 l, b2 2 l + 1); hgp, each (layer, unit)'s partial of its head's gradients [L][B H][kvm_fgpt_unit_grads(d, hd)], summed by
//    the head's owner in sequence order after the backward (a unit is QB items: one partial an item); dkv, each item's partials of dK and
//    dV over its prefix's keys [B H QB][T][2 hd], summed by key block; gn, each member's sum of squares of the gradients it owns.
//  * olist, osrc: each member's owned parameter elements, decoded once a run ([G][kvm_fgpt_own_stride]): the element as slot << 25 | offset,
//    and for an attention element the offset of its first item partial in hgp (the end of a step reads one word an element, not a decode).
//  * dy: a member's own rows of the reduce-scatter's sum.
KVM_FGPT_HD inline int kvm_fgpt_unit_grads(int d, int hd) { return 4 * d * hd + 3 * hd; }   // dWq, dWk, dWv [d][hd]; dbq, dbk, dbv [hd]; dWo [hd][d]
// Attention's work items: each (sequence, head) unit cut into QB query blocks of T / QB rows, so that up to G members share it. Blocks cost
// two more gang barriers a layer (q, k, v gathered; dK and dV partials summed); on the flag-line barrier they pay wherever a block is whole
// mma row tiles. The default is the largest power of two with B H QB <= G and T a multiple of 16 QB. On an RTX 5070, gpt-tiny (T = 32) ran
// x1.06 faster with 2 blocks than with 1 (and 10% slower under the counter barrier, whose crossings cost 1.7x as much).
KVM_FGPT_HD inline int kvm_fgpt_qblocks(int B, int H, int T, int G) {
  int q = 1;
  while ((long long)B * H * q * 2 <= G && T % (16 * q * 2) == 0) q *= 2;
  return q;
}
// A bound on one member's owned parameter elements: its blocks (fc1's columns, b1 and fc2's rows a layer; wte's rows; the head's
// columns), its attention slice and its replicated slice.
KVM_FGPT_HD inline long long kvm_fgpt_own_stride(int B, int T, int d, int H, int F, int V, int L, int G) {
  (void)B; (void)T;
  const long long fp = (F + G - 1) / G, vp = (V + G - 1) / G, hd = H > 0 ? d / H : 0;
  const long long att = (long long)L * H * (4 * d * hd + 3 * hd), rep = (long long)(2 * L + 1) * 2 * d + 2LL * L * d;
  return L * (2 * d * fp + fp) + 2 * d * vp + (att + G - 1) / G + (rep + G - 1) / G + 4;
}
struct KvmFgptScratch {
  long long x[KVM_FGPT_MAXL + 1], h[KVM_FGPT_MAXL], m1[KVM_FGPT_MAXL], r1[KVM_FGPT_MAXL], q[KVM_FGPT_MAXL], k[KVM_FGPT_MAXL], v[KVM_FGPT_MAXL],
      o[KVM_FGPT_MAXL], lse[KVM_FGPT_MAXL], xm[KVM_FGPT_MAXL], h2[KVM_FGPT_MAXL], m2[KVM_FGPT_MAXL], r2[KVM_FGPT_MAXL], pre[KVM_FGPT_MAXL];
  long long hf, mf, rf, logits, ce, tl, gx, part, lnp, rpart, hgp, dkv, olist, osrc, gn, dy, total;
};
KVM_FGPT_HD inline KvmFgptScratch kvm_fgpt_scratch(int N, int T, int d, int H, int F, int V, int L, int G, int QB) {
  KvmFgptScratch s{};
  long long at = 0;
  const long long Nd = (long long)N * d, NF = (long long)N * F, B = T > 0 ? N / T : 0;
  auto take = [&at](long long n) { const long long o = at; at += (n + 3) & ~3LL; return o; };
  for (int l = 0; l <= L; ++l) s.x[l] = take(Nd);
  for (int l = 0; l < L; ++l) {
    s.h[l] = take(Nd); s.m1[l] = take(N); s.r1[l] = take(N); s.q[l] = take(Nd); s.k[l] = take(Nd); s.v[l] = take(Nd); s.o[l] = take(Nd);
    s.lse[l] = take((long long)N * H); s.xm[l] = take(Nd); s.h2[l] = take(Nd); s.m2[l] = take(N); s.r2[l] = take(N); s.pre[l] = take(NF);
  }
  s.hf = take(Nd); s.mf = take(N); s.rf = take(N); s.logits = take((long long)N * V);
  s.ce = take((long long)G * N * 2); s.tl = take(N); s.gx = take(Nd); s.part = take((long long)(G > H ? G : H) * Nd);
  s.lnp = take((long long)(2 * L + 1) * G * 2 * d); s.rpart = take((long long)2 * L * G * d);
  const int hd = H > 0 ? d / H : 0, items = (int)B * H * QB;
  s.hgp = take((long long)L * items * kvm_fgpt_unit_grads(d, hd)); s.dkv = take((long long)items * T * 2 * hd);
  const long long os = kvm_fgpt_own_stride(T > 0 ? N / T : 0, T, d, H, F, V, L, G);
  s.olist = take((long long)G * os); s.osrc = take((long long)G * os); s.gn = take(G); s.dy = take(Nd);
  s.total = at;
  return s;
}

// Shared memory. The tensor-core GEMM reads its operands from shared memory, padded to whole mma tiles (16 rows of A, 8 columns of B, k in
// 8s) and to conflict-free row strides: 4 mod 8 floats for an operand contiguous in k, 8 mod 16 for one contiguous in m or n.
KVM_FGPT_HD inline int kvm_fgpt_ldk(int cols) { return ((cols + 7) & ~7) + 4; }
KVM_FGPT_HD inline int kvm_fgpt_ldmn(int cols) { return ((cols + 15) & ~15) + 8; }
KVM_FGPT_HD inline int kvm_fgpt_r16(int n) { return (n + 15) & ~15; }
KVM_FGPT_HD inline int kvm_fgpt_r8(int n) { return (n + 7) & ~7; }
KVM_FGPT_HD inline long long kvm_fgpt_r4(long long n) { return (n + 3) & ~3LL; }
#define KVM_FGPT_NT 256                       // threads of a member's block
#define KVM_FGPT_MAXBLK 16                    // a member's block rectangles: fc1's columns, b1, fc2's rows a layer, wte's rows, the head's columns
// The fixed head of a member's shared memory: the step's tokens and targets and the cross-entropy's log-sum-exp a row (N each), RoPE's cos
// and sin table (T x hd / 2 each), a block reduction's partials (one double a warp), a flag, a scratch of one float a thread, every
// parameter slot's four pointers (p, m, v, g) and weight decay, and the member's block table (8 ints an entry).
KVM_FGPT_HD inline long long kvm_fgpt_smem_head(int N, int T, int hd) {
  return 3 * kvm_fgpt_r4(N) + kvm_fgpt_r4((long long)T * hd) + 2 * (KVM_FGPT_NT / 32) + 4 + KVM_FGPT_NT + 4LL * KVM_FGPT_MAXP * 2 +
         kvm_fgpt_r4(KVM_FGPT_MAXP) + 8LL * KVM_FGPT_MAXBLK;
}
// A bound on the program's shared memory, in floats: a fixed head (the step's tokens and targets, the cross-entropy's log-sum-exp a row, a
// block reduction's partials and a flag), then the largest phase's buffers, as kvm_fused_gpt.cuh takes them. The device checks every phase
// against the shared memory its caller gives it and fails the run (KVM_FGPT_E_SCRATCH) instead of overrunning, so a bound too small shows
// at once.
KVM_FGPT_HD inline long long kvm_fgpt_smem_floats(int N, int T, int d, int H, int F, int V, int G, int QB) {
  const int hd = H > 0 ? d / H : 0, T16 = kvm_fgpt_r16(T), N16 = kvm_fgpt_r16(N), wf = (F + G - 1) / G, wv = (V + G - 1) / G;
  const int ldq = kvm_fgpt_ldk(3 * hd), ldd = kvm_fgpt_ldk(d), ldh = kvm_fgpt_ldk(kvm_fgpt_r16(hd)), ldm = kvm_fgpt_ldmn(d);
  auto r4 = [](long long n) { return kvm_fgpt_r4(n); };
  auto mx = [](long long a, long long b) { return a > b ? a : b; };
  // attention items of TQ query rows (KE = T keys at most): their q | k | v rows (h's rows, W's columns, the biases); then attention (q, k,
  // v, lse, the scores, o, proj's rows); the backward's query side (do, delta, q, k, v, lse, P, dq, dk, dv, o, then gx's rows and Wo's
  // rows or h's rows and Wq's columns); its key side (dk | dv, h's rows, Wk | Wv's columns)
  const int TQ = QB > 0 ? T / QB : T, TQ16 = kvm_fgpt_r16(TQ), ldS = kvm_fgpt_ldk(T16), ldkv = kvm_fgpt_ldk(2 * hd);
  const long long qrow = r4((long long)TQ16 * ldh), krow = r4((long long)T16 * ldh);
  const long long attn_qkv = r4((long long)TQ16 * ldd) + r4((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(3 * hd)) + r4(3 * hd) + r4((long long)TQ16 * ldq);
  const long long attn_f = qrow + 2 * krow + r4(TQ) + r4((long long)TQ16 * ldS) + qrow + r4((long long)kvm_fgpt_r8(hd) * ldm);
  const long long attn_b = qrow + r4(TQ) + qrow + 2 * krow + r4(TQ) + r4((long long)TQ16 * ldS) + qrow + 2 * krow + qrow +
                           mx(r4((long long)TQ16 * ldd) + r4((long long)kvm_fgpt_r8(hd) * ldd), r4((long long)TQ16 * ldm) + r4((long long)kvm_fgpt_r8(d) * ldh));
  const long long attn_k = r4((long long)TQ16 * ldkv) + r4((long long)TQ16 * ldm) + r4((long long)kvm_fgpt_r8(d) * ldkv);
  // the MLP, forward: h2, W1's columns, u, W2's rows, b1; backward: pre, gelu(pre), du, gx then h2, W2's rows then W1's columns
  const long long mlp_f = r4((long long)N16 * ldd) + r4((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(wf)) + r4((long long)N16 * kvm_fgpt_ldk(wf)) +
                          r4((long long)kvm_fgpt_r8(wf) * ldm) + r4(wf);
  const long long mlp_b = 2 * r4((long long)N16 * kvm_fgpt_ldk(wf)) + r4((long long)N16 * kvm_fgpt_ldmn(wf)) + r4((long long)N16 * mx(ldd, ldm)) +
                          mx(r4((long long)kvm_fgpt_r8(wf) * ldd), r4((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldk(wf)));
  // the head: hf and its columns (logits); dlogits, hf, its columns again (backward)
  const long long logit = r4((long long)N16 * ldd) + r4((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(wv));
  const long long headb = r4((long long)N16 * kvm_fgpt_ldk(wv)) + r4((long long)N16 * ldm) + r4((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldk(wv));
  return kvm_fgpt_smem_head(N, T, hd) + mx(mx(mx(mx(attn_qkv, attn_f), mx(attn_b, attn_k)), mx(mlp_f, mlp_b)), mx(logit, headb));
}

// The gang's size for a shape: the largest power of two up to min(KVM_FGPT_MAX_GANG, max_g, N) whose members fit `smem_floats` (the fast
// form's phases, kvm_fgpt_smem_floats, with the default query blocks); 0 when none does (the run is not fused). On an RTX 5070, gpt-small ran
// faster at 32 members than at 16, and gpt-tiny was flat from 8 to 32 (20 steps each, bench/fused_gpt/fgpt_bed.cu).
KVM_FGPT_HD inline int kvm_fgpt_gang(int N, int T, int d, int H, int F, int V, int smem_floats, int max_g) {
  if (T <= 0 || N % T) return 0;
  int G = 1;
  while (G * 2 <= KVM_FGPT_MAX_GANG && G * 2 <= max_g && G * 2 <= N) G *= 2;
  for (; G >= 1; G /= 2)
    if (kvm_fgpt_smem_floats(N, T, d, H, F, V, G, kvm_fgpt_qblocks(N / T, H, T, G)) <= smem_floats) return G;
  return 0;
}
// Why the device program cannot run a matched shape, or "" when it can (the conditions kvm_fused_gpt.cuh's run checks; its layer count is a
// question of fit, which the pass asks separately).
inline const char* kvm_fgpt_unrunnable(int d, int H, int L, int mm_numerics) {
  const int hd = H > 0 ? d / H : 0;
  if (hd <= 0 || d != H * hd || hd % 8 || hd > 128) return "a head width the fused step does not take (a multiple of 8, at most 128)";
  if (d % 4) return "a width that is not a multiple of 4";
  if (L < 1) return "no layer";
  if (mm_numerics & ~16) return "GEMM numerics the fused step does not reproduce (truncated or exact tiles)";
  return "";
}

// The faults a member reports through its caller's `fail(code)` (kvm_fused_gpt.cuh's run; the executor reports one as KVM_ERR_BOUNDS with the
// code and the member as its arguments, which sets the error word every member's barrier watches): a token or a target outside the vocabulary,
// a data window outside its stream, a step counter AdamW refuses (below 1 after the increment), a scratch smaller than the layout, or a shape
// the program does not take.
enum KvmFgptError : int { KVM_FGPT_E_TOKEN = 1, KVM_FGPT_E_TARGET, KVM_FGPT_E_WINDOW, KVM_FGPT_E_STEP, KVM_FGPT_E_SCRATCH, KVM_FGPT_E_SHAPE };
