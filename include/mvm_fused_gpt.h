#pragma once
// The fused GPT step: the descriptor one fused run's gang reads, and the gang's sizing (host and device; host-only callers include it with
// plain g++).
//
// A fused run trains one pre-norm GPT (the template of mvm_fuse_gpt.h) for all of its steps as one gang of G blocks (its G `fgpt_step`
// members, launched together as one kernel: mvm_fgpt_kernel.cuh). Each member holds a shard of every parameter with its gradient and AdamW moments in
// shared memory across the steps: column shards of q, k, v (by head), fc1 and the head; row shards of proj and fc2; rows of every N x d
// stream. Activations cross members through global memory at two-barrier exchanges (reduce-scatter, all-gather). The member reads the
// step's tokens and targets from the run's streams itself, as the bundle's data windows did. Device side: mvm_fused_gpt.cuh.
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define MVM_FGPT_HD __host__ __device__
#else
#define MVM_FGPT_HD
#endif

#define MVM_FGPT_MAXL 4                       // transformer layers a fused run may have (the shared-memory fit bounds it before this does)
#define MVM_FGPT_PER_LAYER 16                 // parameters a layer: n1 (g, b), q, k, v, proj (w, b each), n2 (g, b), fc1, fc2 (w, b each)
#define MVM_FGPT_MAXP (4 + MVM_FGPT_PER_LAYER * MVM_FGPT_MAXL)   // + wte, nf (g, b), head
#define MVM_FGPT_SMEM_FLOATS 38400            // the pass's default budget of shared memory a member, in floats (the executor passes the device's)
#define MVM_FGPT_MAX_GANG 64                  // the widest gang the policy picks (mvm_fgpt_fit); an A100's 108 SMs seat 64 members
#define MVM_FGPT_QUARTER_MAX 48               // the widest gang of half a hidden tile a member the policy prefers (mvm_fgpt_fit)

// Parameter slots, in one fixed order: 0 wte; 1 + 16 l + r for layer l's role r (MVM_FGPT_R_*); then nf's g and b; then the head.
enum MvmFgptRole {
  MVM_FGPT_R_N1G = 0, MVM_FGPT_R_N1B, MVM_FGPT_R_WQ, MVM_FGPT_R_BQ, MVM_FGPT_R_WK, MVM_FGPT_R_BK, MVM_FGPT_R_WV, MVM_FGPT_R_BV,
  MVM_FGPT_R_WO, MVM_FGPT_R_BO, MVM_FGPT_R_N2G, MVM_FGPT_R_N2B, MVM_FGPT_R_W1, MVM_FGPT_R_B1, MVM_FGPT_R_W2, MVM_FGPT_R_B2
};
MVM_FGPT_HD inline int mvm_fgpt_slot_layer(int l, int role) { return 1 + MVM_FGPT_PER_LAYER * l + role; }
MVM_FGPT_HD inline int mvm_fgpt_slot_nfg(int L) { return 1 + MVM_FGPT_PER_LAYER * L; }
MVM_FGPT_HD inline int mvm_fgpt_slot_nfb(int L) { return 2 + MVM_FGPT_PER_LAYER * L; }
MVM_FGPT_HD inline int mvm_fgpt_slot_head(int L) { return 3 + MVM_FGPT_PER_LAYER * L; }
MVM_FGPT_HD inline int mvm_fgpt_nparams(int L) { return 4 + MVM_FGPT_PER_LAYER * L; }

struct alignas(16) MvmFgptDesc {
  // shape: rows (B sequences of T), width, heads x head width, MLP width, vocabulary, layers; the gang's size; the run's steps
  int N, T, B, d, H, hd, F, V, L, G, steps;
  int mm_numerics;                            // the bundle's GEMM numerics bits (matmul_tile flags: precise, trunc, exact), which the fused GEMMs reproduce
  int rows;                                   // the N-wide phases' row chunk (the MLP's, the logits', the head's backward): N, or a multiple
                                              // of 16 below it (mvm_fgpt_fit)
  int qb;                                     // attention's query blocks a (sequence, head) unit (mvm_fgpt_qblocks gives the default); T a multiple of 16 qb
  float ln_eps, theta, scale;                 // LayerNorm epsilon, RoPE base, attention scale
  float lr, beta1, beta2, adam_eps, max_norm; // AdamW and the grad-norm clip
  float wd[MVM_FGPT_MAXP];                    // weight decay of every parameter slot
  int prows[MVM_FGPT_MAXP], pcols[MVM_FGPT_MAXP];
  // the step's data, read the way the bundle's data windows read it: element i of step s's tokens is ids_src[step * stride + off + i]
  int ids_stride, ids_off, tgt_stride, tgt_off;
  long long ids_len, tgt_len;                 // elements of the two streams: a window must lie inside its stream
  // global buffers (device pointers, resolved by the executor from bundle buffer indices)
  const int32_t* ids_src; const int32_t* tgt_src;
  int32_t* step;                              // the run's step counter: read once, advanced once a step, written back at the end
  float* const* loss;                         // [steps]: each step's loss buffer (the bundle writes one a step), a device array of pointers
  float* p[MVM_FGPT_MAXP]; float* m[MVM_FGPT_MAXP]; float* v[MVM_FGPT_MAXP];   // parameters and AdamW moments, row-major, contiguous
  float* g[MVM_FGPT_MAXP];                    // the bundle's gradient buffers: written once at the end, so a reader after the run sees the last step's
  float* xch;                                 // the global scratch (mvm_fgpt_scratch's layout, zeroed or not: every element is written before it is read)
  size_t xch_floats;
  float* bc;                                  // [2 steps]: each step's AdamW bias corrections, 1 - beta1^t and 1 - beta2^t, filled by member 0
                                              // at the run's start (a double pow is microseconds where FP64 is slow; once a run, not a step)
  unsigned* ctl;                              // the gang's control words (mvm_gang::cross): mvm_gang::line_words(MVM_FGPT_MAX_GANG) words,
                                              // 128-byte aligned, zeroed once
};

// The device program's global scratch: offsets in floats into MvmFgptDesc::xch, laid out by the host for the run's shape and gang, every
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
//    layer l is 2 l, b2 2 l + 1); hgp, each (layer, unit)'s partial of its head's gradients [L][B H][mvm_fgpt_unit_grads(d, hd)], summed by
//    the head's owner in sequence order after the backward (a unit is QB items: one partial an item); dkv, each item's partials of dK and
//    dV over its prefix's keys [B H QB][T][2 hd], summed by key block; gn, each member's sum of squares of the gradients it owns.
//  * olist, osrc: each member's owned parameter elements, decoded once a run ([G][mvm_fgpt_own_stride]): the element as slot << 25 | offset,
//    and for an attention element the offset of its first item partial in hgp (the end of a step reads one word an element, not a decode).
//  * dy: a member's own rows of the reduce-scatter's sum.
MVM_FGPT_HD inline int mvm_fgpt_unit_grads(int d, int hd) { return 4 * d * hd + 3 * hd; }   // dWq, dWk, dWv [d][hd]; dbq, dbk, dbv [hd]; dWo [hd][d]
// Attention's work items: each (sequence, head) unit cut into QB query blocks of T / QB rows, so that up to G members share it. Blocks cost
// two more gang barriers a layer (q, k, v gathered; dK and dV partials summed); on the flag-line barrier they pay wherever a block is whole
// mma row tiles. The default is the largest QB with B H QB <= G and T a multiple of 16 QB, any count, not only a power of two: T = 96
// takes 3 blocks of 32 rows where powers of two stopped at 2 of 48. On an RTX 5070, gpt-tiny (T = 32) ran x1.06 faster with 2 blocks
// than with 1 (and 10% slower under the counter barrier, whose crossings cost 1.7x as much); on an A100 the hold-out's d128 draw (T 96,
// B 2, H 4) ran x1.07 faster with 3 than with 2, in 140,208 B of shared memory a member instead of 164,400.
MVM_FGPT_HD inline int mvm_fgpt_qblocks(int B, int H, int T, int G) {
  int best = 1;
  for (int q = 2; (long long)B * H * q <= G && 16 * q <= T; ++q)
    if (T % (16 * q) == 0) best = q;
  return best;
}
// A bound on one member's owned parameter elements: its blocks (fc1's columns, b1 and fc2's rows a layer; wte's rows; the head's
// columns), its attention slice and its replicated slice.
MVM_FGPT_HD inline long long mvm_fgpt_own_stride(int B, int T, int d, int H, int F, int V, int L, int G) {
  (void)B; (void)T;
  const long long fp = (F + G - 1) / G, vp = (V + G - 1) / G, hd = H > 0 ? d / H : 0;
  const long long att = (long long)L * H * (4 * d * hd + 3 * hd), rep = (long long)(2 * L + 1) * 2 * d + 2LL * L * d;
  return L * (2 * d * fp + fp) + 2 * d * vp + (att + G - 1) / G + (rep + G - 1) / G + 4;
}
struct MvmFgptScratch {
  long long x[MVM_FGPT_MAXL + 1], h[MVM_FGPT_MAXL], m1[MVM_FGPT_MAXL], r1[MVM_FGPT_MAXL], q[MVM_FGPT_MAXL], k[MVM_FGPT_MAXL], v[MVM_FGPT_MAXL],
      o[MVM_FGPT_MAXL], lse[MVM_FGPT_MAXL], xm[MVM_FGPT_MAXL], h2[MVM_FGPT_MAXL], m2[MVM_FGPT_MAXL], r2[MVM_FGPT_MAXL], pre[MVM_FGPT_MAXL];
  long long hf, mf, rf, logits, ce, tl, lsec, gx, gxk, du, dqk, part, lnp, rpart, hgp, dkv, olist, osrc, gn, dy, total;
};
// The layout written into `s` in place (the program's thread 0 builds the block's shared copy this way, with no local copy to spill); every
// offset is a long long, so the zeroing loop clears the arrays past L as MvmFgptScratch{} would.
MVM_FGPT_HD inline void mvm_fgpt_scratch_into(MvmFgptScratch& s, int N, int T, int d, int H, int F, int V, int L, int G, int QB) {
  static_assert(sizeof(MvmFgptScratch) % sizeof(long long) == 0, "the layout is long long offsets");
  for (size_t i = 0; i < sizeof(MvmFgptScratch) / sizeof(long long); ++i) reinterpret_cast<long long*>(&s)[i] = 0;
  long long at = 0;
  const long long Nd = (long long)N * d, NF = (long long)N * F, B = T > 0 ? N / T : 0;
  auto take = [&at](long long n) { const long long o = at; at += (n + 3) & ~3LL; return o; };
  for (int l = 0; l <= L; ++l) s.x[l] = take(Nd);
  for (int l = 0; l < L; ++l) {
    s.h[l] = take(Nd); s.m1[l] = take(N); s.r1[l] = take(N); s.q[l] = take(Nd); s.k[l] = take(Nd); s.v[l] = take(Nd); s.o[l] = take(Nd);
    s.lse[l] = take((long long)N * H); s.xm[l] = take(Nd); s.h2[l] = take(Nd); s.m2[l] = take(N); s.r2[l] = take(N); s.pre[l] = take(NF);
  }
  s.hf = take(Nd); s.mf = take(N); s.rf = take(N); s.logits = take((long long)N * V);
  s.ce = take((long long)G * N * 2); s.tl = take(N); s.lsec = take(N); s.gx = take(Nd);   // lsec: each row's log-sum-exp, by its owner
  s.gxk = take(Nd); s.du = take(NF);   // ZBW's (mvm_fused_gpt.cuh, the MLP backward): fc2's output gradient kept, and the owners' dU
  s.dqk = take(Nd);                    // ZBW's for the attention: the items' dq after RoPE's backward, for their doers
  s.part = take((long long)(G > H ? G : H) * Nd);
  s.lnp = take((long long)(2 * L + 1) * G * 2 * d); s.rpart = take((long long)2 * L * G * d);
  const int hd = H > 0 ? d / H : 0, items = (int)B * H * QB;
  s.hgp = take((long long)L * items * mvm_fgpt_unit_grads(d, hd)); s.dkv = take((long long)items * T * 2 * hd);
  const long long os = mvm_fgpt_own_stride(T > 0 ? N / T : 0, T, d, H, F, V, L, G);
  s.olist = take((long long)G * os); s.osrc = take((long long)G * os); s.gn = take(G); s.dy = take(Nd);
  s.total = at;
}
MVM_FGPT_HD inline MvmFgptScratch mvm_fgpt_scratch(int N, int T, int d, int H, int F, int V, int L, int G, int QB) {
  MvmFgptScratch s;
  mvm_fgpt_scratch_into(s, N, T, d, H, F, V, L, G, QB);
  return s;
}

// Shared memory. The tensor-core GEMM reads its operands from shared memory, padded to whole mma tiles (16 rows of A, 8 columns of B, k in
// 8s) and to conflict-free row strides: 4 mod 8 floats for an operand contiguous in k, 8 mod 16 for one contiguous in m or n.
MVM_FGPT_HD inline int mvm_fgpt_ldk(int cols) { return ((cols + 7) & ~7) + 4; }
MVM_FGPT_HD inline int mvm_fgpt_ldmn(int cols) { return ((cols + 15) & ~15) + 8; }
MVM_FGPT_HD inline int mvm_fgpt_r16(int n) { return (n + 15) & ~15; }
MVM_FGPT_HD inline int mvm_fgpt_r8(int n) { return (n + 7) & ~7; }
MVM_FGPT_HD inline long long mvm_fgpt_r4(long long n) { return (n + 3) & ~3LL; }
// The words the embedding gradient's row lists take from a block of NT threads' shared memory before its sums (mvm_fused_gpt.cuh's
// emb_grad_rows): a chunk's listed token rows and their vocabulary rows (NT ints each) and each warp's count, rounded to a 16-byte line.
template <int NT>
MVM_FGPT_HD constexpr int mvm_fgpt_emb_list_floats() { return (2 * NT + NT / 32 + 3) / 4 * 4; }
// ZBW (mvm_fused_gpt.cuh, the MLP backward): a member without an attention item runs the MLP's weight gradients of up to
// MVM_FGPT_ZBW_OWNERS owners' fc1 columns in the attention backward's waits (0 turns it off). Four: on an RTX 5070 one pass over four
// owners' columns cost 2.13x one owner's weight gradients (d 128, w 8 a member), which the hold-out's d128 at G 64 (16 such members for
// 64 owners) needs to fit their waits; a doer of more owners would overrun them.
// ZBW for the attention (mvm_fused_gpt.cuh, the attention backward), its own switch (0 turns it off): when the gang has at least as many
// members without an item as items (2 W <= G) and query blocks to exchange (QB > 1), member W + w runs item w's dWo in the first wait and
// its dWq and dbq after the dK dV exchange, from the dq the item stores. gpt-tiny at G 16 (4 items, 12 such members) takes it; the
// hold-out's d128 at G 64 (48 items, 16 such members, already full of the MLP's) does not.
#ifndef MVM_FGPT_ZBW_ATTN
#define MVM_FGPT_ZBW_ATTN 1
#endif
#ifndef MVM_FGPT_ZBW_OWNERS
#define MVM_FGPT_ZBW_OWNERS 4
#endif
// A doer pass's shared memory: wb fc1 columns' pre, gelu(pre) and dU over a row chunk of RC16 rows, and db columns of gx or h2's rows.
MVM_FGPT_HD inline long long mvm_fgpt_zbw_floats(int RC16, int wb, int db) {
  const int ldx = mvm_fgpt_ldk(db) > mvm_fgpt_ldmn(db) ? mvm_fgpt_ldk(db) : mvm_fgpt_ldmn(db);
  return 2 * mvm_fgpt_r4((long long)RC16 * mvm_fgpt_ldk(wb)) + mvm_fgpt_r4((long long)RC16 * mvm_fgpt_ldmn(wb)) + mvm_fgpt_r4((long long)RC16 * ldx);
}
// ZBW's plan for a gang of G members whose attention has W items (the run decides with it at its start; a bed prints it): engaged when
// G - W members have no item and none of them covers more than MVM_FGPT_ZBW_OWNERS of the nf owners (fc1 columns fp an owner), in passes
// of zg owners' columns and blocks of zdb of gx's or h2's d columns, the widest that fit `room` floats of a member's arena (more owners a
// pass first: one wide pass costs less than two narrow ones). zg 0: not engaged.
struct MvmFgptZbw { int zg = 0, zdb = 0, doers = 0; };
MVM_FGPT_HD inline MvmFgptZbw mvm_fgpt_zbw_plan(int G, int W, int nf, int fp, int d, int RC16, long long room) {
  MvmFgptZbw z;
  const int idle = G - W, most = idle > 0 ? (nf + idle - 1) / idle : 0;
  if (idle <= 0 || most < 1 || most > MVM_FGPT_ZBW_OWNERS) return z;
  for (int g = most; g >= 1; --g)
    for (int db = d; db >= 8; db = db > 8 ? mvm_fgpt_r8(db / 2) : 0)
      if (mvm_fgpt_zbw_floats(RC16, g * fp, db) <= room) { z.zg = g; z.zdb = db; z.doers = idle; return z; }
  return z;
}
#define MVM_FGPT_NT 384                       // threads of a member's block: 12 warps, three to each SM sub-partition, which hide the step's latency-bound tile loops better than two
#define MVM_FGPT_MAXBLK 16                    // a member's block rectangles: fc1's columns, b1, fc2's rows a layer, wte's rows, the head's columns
// The scratch layout in floats, rounded to 16 bytes: one copy a block heads its shared memory, so a thread never holds the offsets itself.
// Read with a runtime layer index, a thread's own copy would live in local memory, 592 bytes a thread (on an A100, beside this kernel's
// shared memory, its L1 keeps little of that), and on sm_80 the struct's 74 offsets crowded every phase's registers.
#define MVM_FGPT_LAYOUT_FLOATS ((long long)((sizeof(MvmFgptScratch) + 15) / 16 * 4))
// The run's constants (mvm_fused_gpt.cuh's Ctx) after it, in a fixed slot of 384 bytes (a static_assert there checks it fits).
#define MVM_FGPT_CTX_FLOATS 96LL
// The fixed head of a member's shared memory: the scratch layout, the run's constants, the step's tokens and targets and the cross-entropy's log-sum-exp a row (N each), RoPE's cos
// and sin table (T x hd / 2 each), a block reduction's partials (one double a warp), a flag, a scratch of one float a thread, every
// parameter slot's four pointers (p, m, v, g) and weight decay, and the member's block table (8 ints an entry).
MVM_FGPT_HD inline long long mvm_fgpt_smem_head(int N, int T, int hd) {
  return MVM_FGPT_LAYOUT_FLOATS + MVM_FGPT_CTX_FLOATS + 3 * mvm_fgpt_r4(N) + mvm_fgpt_r4((long long)T * hd) + 2 * (MVM_FGPT_NT / 32) + 4 + MVM_FGPT_NT + 4LL * MVM_FGPT_MAXP * 2 +
         mvm_fgpt_r4(MVM_FGPT_MAXP) + 8LL * MVM_FGPT_MAXBLK;
}
// A bound on the program's shared memory, in floats: a fixed head (the step's tokens and targets, the cross-entropy's log-sum-exp a row, a
// block reduction's partials and a flag), then the largest phase's buffers, as mvm_fused_gpt.cuh takes them. The device checks every phase
// against the shared memory its caller gives it and fails the run (MVM_FGPT_E_SCRATCH) instead of overrunning, so a bound too small shows
// at once.
MVM_FGPT_HD inline long long mvm_fgpt_smem_floats(int N, int T, int d, int H, int F, int V, int G, int QB, int R) {
  const int hd = H > 0 ? d / H : 0, T16 = mvm_fgpt_r16(T), N16 = mvm_fgpt_r16(R < N ? R : N), wf = (F + G - 1) / G, wv = (V + G - 1) / G;   // N16: a row chunk's
  const int ldq = mvm_fgpt_ldk(3 * hd), ldd = mvm_fgpt_ldk(d), ldh = mvm_fgpt_ldk(mvm_fgpt_r16(hd)), ldm = mvm_fgpt_ldmn(d);
  auto r4 = [](long long n) { return mvm_fgpt_r4(n); };
  auto mx = [](long long a, long long b) { return a > b ? a : b; };
  // attention items of TQ query rows (KE = T keys at most): their q | k | v rows (h's rows, W's columns, the biases); then attention (q, k,
  // v, lse, the scores, o, proj's rows); the backward's query side (do, delta, q, k, v, lse, P, dq, dk, dv, o, then gx's rows and Wo's
  // rows or h's rows and Wq's columns); its key side (dk | dv, h's rows, Wk | Wv's columns)
  const int TQ = QB > 0 ? T / QB : T, TQ16 = mvm_fgpt_r16(TQ), ldS = mvm_fgpt_ldk(T16), ldkv = mvm_fgpt_ldk(2 * hd);
  const long long qrow = r4((long long)TQ16 * ldh), krow = r4((long long)T16 * ldh);
  const long long attn_qkv = r4((long long)TQ16 * ldd) + r4((long long)mvm_fgpt_r8(d) * mvm_fgpt_ldmn(3 * hd)) + r4(3 * hd) + r4((long long)TQ16 * ldq);
  const long long attn_f = qrow + 2 * krow + r4(TQ) + r4((long long)TQ16 * ldS) + qrow + r4((long long)mvm_fgpt_r8(hd) * ldm);
  // (h's rows and Wq's columns go below DQ, over the dead do..scores, where they fit there)
  const long long below_dq = qrow + r4(TQ) + qrow + 2 * krow + r4(TQ) + r4((long long)TQ16 * ldS);
  const long long gx_wo = r4((long long)TQ16 * ldd) + r4((long long)mvm_fgpt_r8(hd) * ldd), h_wq = r4((long long)TQ16 * ldm) + r4((long long)mvm_fgpt_r8(d) * ldh);
  const long long attn_b = below_dq + qrow + 2 * krow + qrow + (h_wq <= below_dq ? gx_wo : mx(gx_wo, h_wq));
  const long long attn_k = r4((long long)TQ16 * ldkv) + r4((long long)TQ16 * ldm) + r4((long long)mvm_fgpt_r8(d) * ldkv);
  // the MLP, forward: h2, W1's columns, u, W2's rows, b1; backward: pre, gelu(pre), du, gx then h2, W2's rows then W1's columns
  const long long mlp_f = r4((long long)N16 * ldd) + r4((long long)mvm_fgpt_r8(d) * mvm_fgpt_ldmn(wf)) + r4((long long)N16 * mvm_fgpt_ldk(wf)) +
                          r4((long long)mvm_fgpt_r8(wf) * ldm) + r4(wf);
  const long long mlp_b = 2 * r4((long long)N16 * mvm_fgpt_ldk(wf)) + r4((long long)N16 * mvm_fgpt_ldmn(wf)) + r4((long long)N16 * mx(ldd, ldm)) +
                          mx(r4((long long)mvm_fgpt_r8(wf) * ldd), r4((long long)mvm_fgpt_r8(d) * mvm_fgpt_ldk(wf)));
  // the head: hf, its columns and the chunk's logits (forward); dlogits, hf, its columns again (backward)
  const long long logit = r4((long long)N16 * ldd) + r4((long long)mvm_fgpt_r8(d) * mvm_fgpt_ldmn(wv)) + r4((long long)N16 * wv);   // + the chunk's logits
  const long long headb = r4((long long)N16 * mvm_fgpt_ldk(wv)) + r4((long long)N16 * ldm) + r4((long long)mvm_fgpt_r8(d) * mvm_fgpt_ldk(wv));
  // the embedding's gradient: gx, a row chunk of it (unpadded)
  const long long emb = r4((long long)(R < N ? R : N) * d);
  return mvm_fgpt_smem_head(N, T, hd) + mx(mx(mx(mx(attn_qkv, attn_f), mx(attn_b, attn_k)), mx(mlp_f, mlp_b)), mx(mx(logit, headb), emb));
}

// The row chunk of the N-wide phases at gang G: the whole N when its buffers fit `smem_floats`, else 128 rows, else 64 (whole mma row tiles a
// warp); 0 when none fits. A chunk keeps the phase's weight slices resident and walks the rows, so its buffers no longer grow with N; a
// weight gradient (a GEMM over the rows) is written by the first chunk and added to by each later one, in order.
MVM_FGPT_HD inline int mvm_fgpt_rows_for(int N, int T, int d, int H, int F, int V, int G, int QB, long long smem_floats) {
  const int cand[3] = {N, 128, 64};
  for (int i = 0; i < 3; ++i)
    if ((i == 0 || cand[i] < N) && mvm_fgpt_smem_floats(N, T, d, H, F, V, G, QB, cand[i]) <= smem_floats) return cand[i];
  return 0;
}
// ZBW's plan for a run (mvm_fused_gpt.cuh): the MLP's doers (mvm_fgpt_zbw_plan) and the attention's, from the run's shape, its row chunk and
// the launch's dynamic shared memory in floats. The host picks the step kernel's instance with it (the one that carries ZBW's code serves
// only a run it engages: the code alone, idle, cost the A100's step 1.4-2.0% where ZBW was off), and that instance plans with it, so the
// two agree by construction.
struct MvmFgptZbwRun {
  MvmFgptZbw mlp;
  bool attn = false;
  MVM_FGPT_HD bool any() const { return mlp.zg > 0 || attn; }
};
// Whether ZBW's doers fit the waits they fill. A doer's weight gradients (dW1 and dW2 of `most` owners' fp hidden units over the N rows,
// 2 most fp d N multiply-adds) are set against one attention item's backward, which the members with an item run meanwhile (its TQ query
// rows against the T keys in 5 products of width hd, and its projections' 7 of width d: TQ hd (5 T + 7 d)); and a doer whose owners take
// more than one pass (zg < most) stages every row chunk once a pass, which the count leaves out. On an A100 bed (fifteen
// hold-out-shaped GPTs at the executor's G, ZBW on against off), ZBW won only in one pass at ratios of 0.42 to 1.41 (its step
// x1.029-1.057); it lost in two or more passes at 1.41 to 2.77 (x0.81-0.93) and in one at 1.76 to 2.23 (x0.93-1.00). Every output is the
// same bits either way (a doer sums each owner's gradient in the owner's order), so the rule decides only the step's time.
MVM_FGPT_HD inline bool mvm_fgpt_zbw_fits(int N, int T, int d, int hd, int QB, int fp, int most) {
  const long long TQ = QB > 0 ? T / QB : T;
  const long long doer = 2LL * most * fp * d * N, item = TQ * hd * (5LL * T + 7LL * d);
  return 10 * doer <= 16 * item;
}
MVM_FGPT_HD inline MvmFgptZbwRun mvm_fgpt_zbw_of(int N, int T, int d, int H, int F, int G, int QB, int rows, long long smem_floats) {
  MvmFgptZbwRun z;
  const int B = T > 0 ? N / T : 0, W = B * H * QB, fp = G > 0 ? (F + G - 1) / G : 0, nf = fp > 0 ? (F + fp - 1) / fp : 0;
  const int RC = rows < N ? rows : N, hd = H > 0 ? d / H : 0;
  z.mlp = mvm_fgpt_zbw_plan(G, W, nf, fp, d, mvm_fgpt_r16(RC), smem_floats - mvm_fgpt_smem_head(N, T, hd));
  if (z.mlp.zg > 0) {
    const int most = (nf + z.mlp.doers - 1) / z.mlp.doers;
    if (z.mlp.zg < most || !mvm_fgpt_zbw_fits(N, T, d, hd, QB, fp, most)) z.mlp = MvmFgptZbw{};
  }
  z.attn = MVM_FGPT_ZBW_ATTN && QB > 1 && W > 0 && 2 * W <= G;
  return z;
}
// A shape's fit: the gang width G whose members fit `smem_floats` (the fast form's phases, mvm_fgpt_smem_floats) with the default query
// blocks and the largest row chunk that fits; G 0 when none does (the run is not fused). The preferred width gives each member two rows at
// least (G <= N / 2) and one 8-wide MMA tile of the MLP's hidden units (G <= F / 8): below either, a member's GEMM tiles run part empty while
// each barrier and reduce-scatter pays for every member. On an A100 bed (20 steps, 5 paired rounds): gpt-tiny (N 32, F 128) 199.0k cycles a
// step at G 16 against 201.9k at 32; gpt-small (N 128, F 256) 314.7 us at 32 against 357.3 at 64; a d128 (N 128, F 512) 963.9 us at 64
// against 1,037.1 at 32, and the hold-out's d128 (N 192) 1,201.9 against 1,328.9. So G = F / 8 comes first, at any width up to
// MVM_FGPT_MAX_GANG: where F / 8 is no power of two, the powers below it leave a member's second tile part empty. An A100 bed
// (2 rounds): d96 (F 384) at G 48 against the power of two the search took, x1.306 at N 128 (G 32), x1.165 at N 384 (G 32) and x1.084
// at N 512 (G 64); d48 (F 192) at N 384 x1.163 at G 24 against G 16. Then the powers of two up to min(MVM_FGPT_MAX_GANG, max_g, N): where
// the preferred width does not fit, a wider one may (its weight slices are smaller), then a narrower one.
struct MvmFgptFit { int G = 0, qb = 1, rows = 0; };
MVM_FGPT_HD inline bool mvm_fgpt_fit_at(int G, int N, int T, int d, int H, int F, int V, int smem_floats, MvmFgptFit& f) {
  const int qb = mvm_fgpt_qblocks(N / T, H, T, G), rows = mvm_fgpt_rows_for(N, T, d, H, F, V, G, qb, smem_floats);
  if (rows <= 0) return false;
  f.G = G; f.qb = qb; f.rows = rows;
  return true;
}
MVM_FGPT_HD inline MvmFgptFit mvm_fgpt_fit(int N, int T, int d, int H, int F, int V, int smem_floats, int max_g) {
  MvmFgptFit f;
  if (T <= 0 || N % T) return f;
  // half a tile a member (G = F / 4) wherever that width is at most MVM_FGPT_QUARTER_MAX and still seats two rows a member. On an A100 bed
  // (20 steps, 2 rounds a point) d48 (F 192) at G 48 against its F / 8 of 24 gains at every N that seats it: x1.115 and x1.121 at N 96,
  // x1.053 and x1.057 at N 128, x1.065 to x1.146 at N 192 to 384. gpt-small (F 256) at G 64 against 32
  // reads level (x0.996), so 64 members is past the gain and the panel's rows keep their width and their bits; d96's and d128's
  // F / 4 lie past MVM_FGPT_MAX_GANG, where the same bed found 96 members no faster. Then one whole 8-wide tile of hidden units a member.
  const int gq = F / 4, gw = F / 8;
  if (F % 4 == 0 && gq >= 1 && gq <= MVM_FGPT_QUARTER_MAX && gq <= max_g && 2 * gq <= N && mvm_fgpt_fit_at(gq, N, T, d, H, F, V, smem_floats, f)) return f;
  if (F % 8 == 0 && gw >= 1 && gw <= MVM_FGPT_MAX_GANG && gw <= max_g && 2 * gw <= N && mvm_fgpt_fit_at(gw, N, T, d, H, F, V, smem_floats, f)) return f;
  int gmax = 1;
  while (gmax * 2 <= MVM_FGPT_MAX_GANG && gmax * 2 <= max_g && gmax * 2 <= N) gmax *= 2;
  int g0 = gmax;
  while (g0 > 1 && (g0 * 2 > N || g0 * 8 > F)) g0 /= 2;                             // two rows and one 8-wide tile of hidden units a member
  for (int G = g0; G <= gmax; G *= 2)
    if (mvm_fgpt_fit_at(G, N, T, d, H, F, V, smem_floats, f)) return f;
  for (int G = g0 / 2; G >= 1; G /= 2)
    if (mvm_fgpt_fit_at(G, N, T, d, H, F, V, smem_floats, f)) return f;
  return f;
}
// Why the device program cannot run a matched shape, or "" when it can (the conditions mvm_fused_gpt.cuh's run checks; its layer count is a
// question of fit, which the pass asks separately).
inline const char* mvm_fgpt_unrunnable(int d, int H, int L, int mm_numerics) {
  const int hd = H > 0 ? d / H : 0;
  if (hd <= 0 || d != H * hd || hd % 8 || hd > 128) return "a head width the fused step does not take (a multiple of 8, at most 128)";
  if (d % 4) return "a width that is not a multiple of 4";
  if (L < 1) return "no layer";
  if (mm_numerics & ~16) return "GEMM numerics the fused step does not reproduce (truncated or exact tiles)";
  return "";
}

// The faults a member reports through its caller's `fail(code)` (mvm_fused_gpt.cuh's run; the executor reports one as MVM_ERR_BOUNDS with the
// code and the member as its arguments, which sets the error word every member's barrier watches): a token or a target outside the vocabulary,
// a data window outside its stream, a step counter AdamW refuses (below 1 after the increment), a scratch smaller than the layout, or a shape
// the program does not take.
enum MvmFgptError : int { MVM_FGPT_E_TOKEN = 1, MVM_FGPT_E_TARGET, MVM_FGPT_E_WINDOW, MVM_FGPT_E_STEP, MVM_FGPT_E_SCRATCH, MVM_FGPT_E_SHAPE };
