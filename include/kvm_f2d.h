#pragma once
// F, the fused GPT step's 2-D form: one GPT training run as one gang, each half of every layer on a layout of its own. This header holds its
// descriptors, the halves' tilings and shared-memory layouts, the fit and the descriptor builder (host and device; the pass includes it with
// plain g++). Device side: kvm_f2d.cuh; the executor's kernel: kvm_f2d_kernel.cuh; the bed: bench/fused_gpt/f2d_bed.cu.
//
// The fused step (kvm_fused_gpt.h) gives each member a column shard of every weight and every member all N rows, so a member's phases hold
// N-wide tiles, and a GPT of d 384 (N 384, F 1536) fits no gang in an A100's shared memory. F cuts each half of a layer on a Gt x Gc
// layout: Gt row groups of R = N / Gt rows, Gc column shards in each.
//  * The attention half (LN1, q|k|v, the attention items, proj and the residual) on Gt_a x Gc_a members. Member c of a group computes
//    q|k|v's columns c (Ns = 3d / Gc of them, all in one of the bundle's q, k, v slots: Gc a multiple of 3) for the group's rows and proj's
//    partial from its rows of o (dc = d / Gc), reduce-scattered in the group. The attention items ((sequence, head, query block) units) run
//    over the whole gang.
//  * The MLP half (LN2, fc1 + GELU, fc2 and the residual) on Gt_m x Gc_m members: Megatron's pair inside a group, fc1 column-parallel
//    (Fs = F / Gc columns a member), fc2 row-parallel and reduce-scattered in the group.
//  * The gang is the larger layout. A member past a half's layout only crosses with that half (and takes attention items). Each half runs
//    its row phases on its own row partition, so the hand-off between the halves is the residual stream, through global memory.
//  * Each weight gradient is one partial a row group, summed by its owners at the step's end, with the other phases the fused step runs
//    around the layers: the embedding and the final LayerNorm on a row partition, the head on vocabulary shards, the cross-entropy, the
//    squares, the grad norm and AdamW over flat shares of every parameter, in the bundle's float32 operation order.
// Numerics: matmul_tile's 1xTF32 for the linear layers, each operand rounded once where it is born (bitwise to rounding at every use: the
// tensor core reads the same 19 bits), and 3xTF32 for attention's products, as the bundle's attention ops run them. KVM_F2D_PRECISE=1
// builds the linear layers in 3xTF32 too, the numerics of bundles with mm_numerics 16; the executor carries the default build.
#include <cstddef>
#include <cstdint>
#include "kvm_fused_gpt.h"

#ifndef KVM_F2D_PRECISE
#define KVM_F2D_PRECISE 0
#endif
#ifndef KVM_F2D_STAGES
#define KVM_F2D_STAGES 3                       // the streamed GEMMs' chunks in flight + 1 (2 fits a laptop part's shared memory)
#endif
#define KVM_F2D_NT 256                         // threads of a member's block: 8 warps at up to 255 registers. On an A100, 12 warps (168
                                               // registers) spilled about 650 B and ran the MLP block 20% slower

namespace kvm_f2d {

constexpr int FNT = KVM_F2D_NT, FNW = FNT / 32;
constexpr bool kPrec = KVM_F2D_PRECISE != 0;   // the linear layers in 3xTF32
constexpr bool kPre = !kPrec;                  // every 1xTF32 GEMM operand rounded once where it is born
constexpr int kNumerics = kPrec ? 16 : 0;      // the bundles' mm_numerics this build reproduces
constexpr int MAXL = KVM_FGPT_MAXL;            // layers a run's descriptor holds
constexpr int MAXA = KVM_FGPT_MAXP;            // parameter arrays: one a bundle slot (4 + 16 L)
// Tile constants: sgemm_rs's column tiles (TN), the one-shot accumulator's row tiles of d (TMW), sgemm_sr's row tiles (TMS), the streamed
// GEMMs' stages, the resident forms' stages, the head's and the embedding gradient's row chunk (SRC), and the streamed GEMMs' K chunk.
constexpr int TN = 64, TMW = 96, TMS = 64, STAGES = KVM_F2D_STAGES, RS_STAGES = 2, SRC = 32, KC = 32;
static_assert(STAGES >= 2, "the streamed GEMMs need a chunk in flight");

KVM_FGPT_HD inline long long mx2(long long a, long long b) { return a > b ? a : b; }
// The gang's control words for G members: kvm_gang::line_words(G), both crossing forms' (the counted barrier uses the first two). Here
// for the host, which cannot include kvm_gang.cuh; kvm_f2d.cuh checks the two agree.
KVM_FGPT_HD constexpr int ctl_words(int members) { return (members + 31) / 32 * 32; }

// ---- the attention half ---------------------------------------------------------------------------------------------------------------
struct ADesc {
  int N, T, B, d, H, hd, gt, gc, kc, R, Ns, dc, qb, TQ, W, cpw, rb, cpw_rs;           // Ns = 3d / Gc, dc = d / Gc; W = B H QB items
  int cpw_do, rb_do, cpw_sro, cpw_srq;                                                // d_o's register GEMM; dWo's and dWqkv's sgemm_sr
  float scale, eps;                                                                    // 1 / sqrt(hd); LN1's epsilon
  const float *X, *g1, *b1n, *Wo, *bo, *rtab;                                          // rtab: RoPE's cos then sin, T x hd / 2 each
  // q|k|v's weight and bias as three d-wide blocks, q's, k's, v's: column j of the fused 3d is block j / d's column j % d, rows ldw floats
  // apart (the bundle's three slots: ldw = d, and a member's q|k|v columns lie in one block). Their gradients take the same form (ldg).
  const float *Wq3[3], *bq3[3];
  float *dWq3[3], *dbq3[3];
  int ldw, ldg;
  float *Hs, *mean, *rstd, *XM, *dX, *dg1p, *db1np, *dg1, *db1n;                       // h = LN1(x); xm = x + y; LN1's gradients, partials [G][d]
  float *QKV, *O, *LSE, *P, *Y;                                                        // P: [Gt][Gc][R][d] proj's partials, then dH's
  const float* dY;
  const float* dYg;                                                                    // dY's rounded copy for the GEMMs (null: dY)
  float* dXg;                                                                          // where dx's rounded copy goes (null: none)
  float *dO, *dQKV, *DKV, *dWop, *dbop, *dWqkvp, *dbqkvp, *dWo, *dbo, *dH;            // DKV: [W][T][2 hd] the items' dK dV partials
  unsigned* count;                                                                     // the gang's control words (kvm_gang)
  const int* abort;                                                                    // the run's error word
};

// The member's shared memory for the attention half, in floats: the largest of the q|k|v GEMM's stream region (STAGES stages of an R x KC
// chunk of H and a KC x Ns chunk of Wqkv), an item's tiles (q, k and v of the prefix, the scores), proj's o tile beside sgemm_rs's region
// (RS_STAGES blocks of Wo's rows by column tiles of TN), and the backward's: d_o's register GEMM (the q|k|v region's form), dWo's sgemm_sr
// (dY by row tiles of TMS beside o's tile), the query blocks (do, o, q, dq of TQ rows; k, v, dk, dv of the prefix; the scores, delta,
// lse), the key blocks (dk, dv) and q|k|v's backward (the dqkv tile beside sgemm_sr's and sgemm_rs's regions).
struct ALay { long long qkv, items, proj, itemsb, keyb, qkvb, total; int ldh, ldS, ldo, ldq; };
KVM_FGPT_HD inline ALay alayout(int R, int Ns, int KC_, int T, int TQ, int hd, int dc) {
  ALay L{};
  const long long R16 = kvm_fgpt_r16(R), KC8 = kvm_fgpt_r8(KC_), Ns8 = kvm_fgpt_r8(Ns), T16 = kvm_fgpt_r16(T), TQ16 = kvm_fgpt_r16(TQ);
  L.qkv = STAGES * (kvm_fgpt_r4(R16 * kvm_fgpt_ldk(KC_)) + mx2(kvm_fgpt_r4(KC8 * kvm_fgpt_ldmn(Ns)), kvm_fgpt_r4(Ns8 * kvm_fgpt_ldk(KC_))));
  L.ldh = kvm_fgpt_ldk(kvm_fgpt_r16(hd)); L.ldS = kvm_fgpt_ldk((int)T16);
  L.items = kvm_fgpt_r4(TQ16 * L.ldh) + 2 * kvm_fgpt_r4(T16 * L.ldh) + kvm_fgpt_r4(TQ16 * L.ldS);
  L.ldo = kvm_fgpt_ldk(dc);
  const long long rsb = mx2(kvm_fgpt_r4((long long)kvm_fgpt_r8(dc) * kvm_fgpt_ldmn(TN)), kvm_fgpt_r4((long long)TN * kvm_fgpt_ldk(dc)));
  L.proj = kvm_fgpt_r4(R16 * L.ldo) + RS_STAGES * rsb;
  const long long srb = (long long)RS_STAGES * kvm_fgpt_r4((long long)kvm_fgpt_r8(R) * kvm_fgpt_ldmn(TMS));
  L.proj = mx2(L.proj, kvm_fgpt_r4(R16 * L.ldo) + srb);
  L.itemsb = 4 * kvm_fgpt_r4(TQ16 * L.ldh) + 4 * kvm_fgpt_r4(T16 * L.ldh) + kvm_fgpt_r4(TQ16 * L.ldS) + 2 * kvm_fgpt_r4(TQ);
  L.keyb = 2 * kvm_fgpt_r4(TQ16 * L.ldh);
  L.ldq = kvm_fgpt_ldmn(Ns);
  const long long rsq = mx2(kvm_fgpt_r4((long long)Ns8 * kvm_fgpt_ldmn(TN)), kvm_fgpt_r4((long long)TN * kvm_fgpt_ldk(Ns)));
  L.qkvb = kvm_fgpt_r4(R16 * L.ldq) + mx2(srb, RS_STAGES * rsq);
  L.total = mx2(mx2(mx2(L.qkv, L.items), L.proj), mx2(mx2(L.itemsb, L.keyb), L.qkvb));
  return L;
}

// The attention half's GEMM forms on its layout (Gt x Gc over N rows, width d, chunk KC): nullptr when every GEMM tiles, else what fails.
struct ATiling { int R = 0, Ns = 0, dc = 0, cpw = 0, rb = 0, cpw_rs = 0, cpw_do = 0, rb_do = 0, cpw_sro = 0, cpw_srq = 0; };
inline const char* attn_tiling(int N, int d, int gt, int gc, int kc, ATiling& a) {
  a = ATiling{};
  if (gt <= 0 || gc <= 0 || N % gt || (3 * d) % gc || d % gc) return "the layout must divide evenly (N % Gt, 3d % Gc, d % Gc)";
  a.R = N / gt; a.Ns = 3 * d / gc; a.dc = d / gc;
  const int R = a.R, Ns = a.Ns, dc = a.dc;
  if (R % 16 || R < gc) return "the layout needs R % 16 == 0 and at least a row a member in each group";
  if (d % 64 || R > 96 || R % 8) return "the register GEMM's streamed rows need d % 64 == 0 and R <= 96, a multiple of 8";
  if (kc != 32 && kc != 48) return "the chunk KC is 32 or 48";
  if (dc % 8 || dc > 96 || d % TN) return "proj (sgemm_rs) needs dc = d / Gc a multiple of 8 and at most 96, and d a multiple of TN";
  // q|k|v's tiling: R / 16 row blocks over FNW warps, RB of them a warp, CPW 8-column tiles each: Ns / 8 = CPW x FNW / (R / 16 / RB)
  const int tm = R / 16;
  for (int rb_ = 1; rb_ <= 2; ++rb_)
    for (int cpw_ = 3; cpw_ <= 4; ++cpw_)
      if (!a.cpw && tm % rb_ == 0 && FNW % (tm / rb_) == 0 && Ns / 8 == cpw_ * (FNW / (tm / rb_))) { a.cpw = cpw_; a.rb = rb_; }
  if (!a.cpw) return "q|k|v's register GEMM has no tiling for this layout: Ns / 8 must be 3 or 4 x FNW / (R / 16 / RB)";
  a.cpw_rs = (TN / 8) * tm / FNW;
  if (a.cpw_rs < 2 || a.cpw_rs > 4 || (TN / 8) * tm % FNW) return "proj's sgemm_rs has no tiling for this layout (TN / 8 x R / 16 over the warps: 2, 3 or 4)";
  // d_o's register GEMM (N = dc), like q|k|v's but one or two tiles a warp
  for (int rb_ = 1; rb_ <= 2; ++rb_)
    for (int cpw_ = 1; cpw_ <= 2; ++cpw_)
      if (!a.cpw_do && tm % rb_ == 0 && FNW % (tm / rb_) == 0 && dc / 8 == cpw_ * (FNW / (tm / rb_))) { a.cpw_do = cpw_; a.rb_do = rb_; }
  if (!a.cpw_do) return "d_o's register GEMM has no tiling for this layout: dc / 8 must be 1 or 2 x FNW / (R / 16 / RB)";
  // sgemm_sr streams its rows by tiles of TMS: TMS / 16 row blocks over the warps, N / 8 = CPW x FNW / (TMS / 16)
  const int wsr = FNW / (TMS / 16);
  a.cpw_sro = (dc / 8) % wsr == 0 ? dc / 8 / wsr : 0; a.cpw_srq = (Ns / 8) % wsr == 0 ? Ns / 8 / wsr : 0;
  if ((a.cpw_sro != 2 && a.cpw_sro != 4) || (a.cpw_srq != 3 && a.cpw_srq != 4 && a.cpw_srq != 6) || d % TMS)
    return "dWo's or dWqkv's sgemm_sr has no tiling for this layout (dc / 8 / 2 in {2, 4}, Ns / 8 / 2 in {3, 4, 6}, d % TMS == 0)";
  return nullptr;
}
inline void attn_desc_tiling(ADesc& p, const ATiling& a) {
  p.R = a.R; p.Ns = a.Ns; p.dc = a.dc; p.cpw = a.cpw; p.rb = a.rb; p.cpw_rs = a.cpw_rs; p.cpw_do = a.cpw_do; p.rb_do = a.rb_do;
  p.cpw_sro = a.cpw_sro; p.cpw_srq = a.cpw_srq;
}

// ---- the MLP half ---------------------------------------------------------------------------------------------------------------------
struct MDesc {
  int N, d, F, gt, gc, kc, R, Fs, cpw, cpw_rs, cpw_sr;   // R = N / Gt rows a group, Fs = F / Gc columns a member; each form's tiles a warp
  float eps;
  const float *XM, *g2, *b2n, *W1, *b1, *W2, *b2, *dXO;  // XM: the attention half's output (the residual stream); dXO: the layer output's gradient
  const float* dXOg;                                     // dXO's rounded copy for the GEMMs (null: dXO)
  float* dXMg;                                           // where dxm's rounded copy goes (the attention's dY for its GEMMs; null: none)
  float *H2, *mean2, *rstd2, *U, *A, *P, *Y2, *XO;        // h2 = LN2(xm); U = h2 W1 + b1, A = GELU(U); P: [Gt][Gc][R][d] partials; xo = xm + y2
  float *dU, *dH2, *dXM;                                  // dU = dA GELU'(U); dh2 = dU W1^T; dxm = dxo + LN2's backward of dh2 (the attention's dY)
  float *dW1p, *dW2p, *db1p, *db2p, *dg2p, *db2np;        // partials: [Gt] row groups (dW, db), [Gm] members (LN2's)
  float *dW1, *dW2, *db1, *db2, *dg2, *db2n;
  unsigned* count;                                        // the gang's control words (kvm_gang)
  const int* abort;                                       // the run's error word
};

// The member's shared memory for the MLP half, in floats: the one-shot accumulator (R x Fs, or R x TMW for dW1's row tiles), the resident
// A and dU tiles (R x Fs), and a region for the streamed GEMMs' STAGES chunks, the one-shot forms' operands and sgemm_rs's B blocks.
struct Lay { long long acc, At, dUt, region, rsize, total; int lacc, ldt; };
KVM_FGPT_HD inline Lay layout(int R, int Fs, int KC_) {
  Lay L{};
  L.lacc = kvm_fgpt_ldmn(Fs > TN ? Fs : TN); L.ldt = kvm_fgpt_ldmn(Fs);
  const long long R16 = kvm_fgpt_r16(R), R8 = kvm_fgpt_r8(R), KC8 = kvm_fgpt_r8(KC_), Fs8 = kvm_fgpt_r8(Fs);
  const long long acc = kvm_fgpt_r4(mx2(R16 * L.lacc, R8 * kvm_fgpt_ldmn(TMW)));
  const long long t = kvm_fgpt_r4(R16 * L.ldt);
  const long long achunk = kvm_fgpt_r4(R16 * kvm_fgpt_ldk(KC_));                               // the two streamed GEMMs: A untransposed
  const long long bchunk = mx2(kvm_fgpt_r4(KC8 * kvm_fgpt_ldmn(Fs)), kvm_fgpt_r4(Fs8 * kvm_fgpt_ldk(KC_)));   // B either form
  const long long oneshot = mx2(mx2(kvm_fgpt_r4(Fs8 * kvm_fgpt_ldmn(TN)), kvm_fgpt_r4(R8 * kvm_fgpt_ldmn(TN))), kvm_fgpt_r4((long long)kvm_fgpt_r8(TN) * kvm_fgpt_ldk(Fs)));
  const long long rsb = mx2(kvm_fgpt_r4((long long)kvm_fgpt_r8(Fs) * kvm_fgpt_ldmn(TN)), kvm_fgpt_r4((long long)TN * kvm_fgpt_ldk(Fs)));   // sgemm_rs's B block
  L.rsize = mx2(mx2(STAGES * (achunk + bchunk), oneshot), RS_STAGES * rsb);
  L.acc = 0; L.At = acc; L.dUt = L.At + t; L.region = L.dUt + t; L.total = L.region + L.rsize;
  return L;
}

// The MLP half's GEMM forms on its layout: nullptr when every GEMM tiles, else what fails.
struct MTiling { int R = 0, Fs = 0, cpw = 0, cpw_rs = 0, cpw_sr = 0; };
inline const char* mlp_tiling(int N, int d, int F, int gt, int gc, int kc, MTiling& m) {
  m = MTiling{};
  if (gt <= 0 || gc <= 0 || N % gt || F % gc || d % gc) return "the MLP layout must divide evenly (N % Gt, F % Gc, d % Gc)";
  m.R = N / gt; m.Fs = F / gc;
  const int R = m.R, Fs = m.Fs;
  if (R < gc) return "the MLP layout needs at least a row a member in each group";
  if (d % TMS || R > 96 || R % 8) return "the MLP's streamed-rows GEMMs need d % TMS == 0 and R <= 96, a multiple of 8";
  if (d % TN || Fs > 96 || Fs % 8) return "the MLP's resident GEMMs need d % TN == 0 and F / Gc <= 96, a multiple of 8";
  if (kc != 32 && kc != 48) return "the chunk KC is 32 or 48";
  const int tm = kvm_fgpt_r16(R) / 16, wpb = tm ? FNW / tm : 0;          // R / 16 row blocks, wpb warps each
  m.cpw = (tm && FNW % tm == 0 && (Fs / 8) % wpb == 0) ? (Fs / 8) / wpb : 0;
  if (m.cpw != 3 && m.cpw != 4 && m.cpw != 6) return "fc1 and dA need R / 16 dividing the warps and F / Gc / 8 = 3, 4 or 6 tiles a warp";
  m.cpw_rs = (TN / 8) / wpb;
  if ((m.cpw_rs != 2 && m.cpw_rs != 4) || (TN / 8) % wpb) return "fc2 and dX (sgemm_rs) need 2 or 4 tiles a warp";
  const int wsr = FNW / (TMS / 16);
  m.cpw_sr = (Fs / 8) % wsr == 0 ? (Fs / 8) / wsr : 0;
  if (m.cpw_sr != 3 && m.cpw_sr != 4 && m.cpw_sr != 6) return "dW1 and dW2 (sgemm_sr) need 3, 4 or 6 tiles a warp";
  return nullptr;
}

// ---- the step -------------------------------------------------------------------------------------------------------------------------
// One parameter array (a bundle slot, read and updated in place) for the sums, the squares and AdamW: parameters, moments, gradient, the
// GEMMs' rounded shadow (a GEMM weight's, which the halves' GEMMs read; null elsewhere and in the precise build), elements, weight decay.
struct SArr { float *p, *m, *v, *g, *r; long long n; float wd; };

struct SDesc {
  ADesc att[MAXL];
  MDesc mlp[MAXL];
  SArr arr[MAXA];                                        // the layers' slots in slot order, then wte, the final LayerNorm's g and b, the head
  int na, N, T, d, V, L, G, steps, nv, vp;               // nv vocabulary shards of vp columns (members 0 .. nv - 1)
  long long smem_floats;                                 // the shared memory a member needs (floats): the device refuses less
  int32_t* step;                                         // the run's step counter: read once at the start, written back at the end
  long long ids_len, tgt_len;                            // elements of the two streams: a step's window must lie inside its stream
  float ln_eps, lr, beta1, beta2, adam_eps, max_norm, inv_n;
  const int32_t *ids, *tgt;                              // the bundle's token and target streams
  int ids_stride, ids_off, tgt_stride, tgt_off;
  const float *wte, *nfg, *nfb, *head;                   // the parameters outside the layers (their slots, updated in place)
  float *gwte, *gnfg, *gnfb, *ghead;
  float *x0, *xL, *dxL, *hf, *mf, *rf, *logits, *ce, *tl, *lse, *dhfp, *dyf, *lnfp;   // ce [nv][N][2]; lse [nv][N]; dhfp [nv][N][d]; lnfp [G][2d]
  float* rtab;                                           // RoPE's table, written once a run by member 0
  float* dxLg;                                           // dx[L]'s rounded copy, the top MLP's GEMM dxo (null: none)
  float theta;
  float* const* loss;                                    // [steps]: each step's loss buffer (loss, 1 / count), a device array of pointers
  float *gn, *bc;                                        // gn [G] the squares' partials; bc [2 steps] the bias corrections
  unsigned* count;                                       // the gang's control words (kvm_gang): line_words(G) words, zeroed once
  const int* abort;                                      // the run's error word, which every crossing watches
};

// ---- the fit ---------------------------------------------------------------------------------------------------------------------------
// A run's form: each half's layout, the gang, the chunk, attention's query blocks, the vocabulary shards and the shared memory a member.
struct Fit { int gta = 0, gca = 0, gtm = 0, gcm = 0, G = 0, kc = KC, qb = 1, nv = 0, vp = 0; long long smem_floats = 0; };

// The shared memory a member needs (floats): the larger half's, or the head's phases (hf's row chunk and the shard's columns; then the
// logits' gradient chunk, hf's chunk and the columns again), or the embedding gradient's row chunk.
inline long long smem_floats(const ATiling& ta, const MTiling& tm, int T, int TQ, int d, int hd, int vp, int kc) {
  const ALay La = alayout(ta.R, ta.Ns, kc, T, TQ, hd, ta.dc);
  const Lay Lm = layout(tm.R, tm.Fs, kc);
  const long long head_f = kvm_fgpt_r4((long long)kvm_fgpt_r16(SRC) * kvm_fgpt_ldk(d)) + kvm_fgpt_r4((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldmn(vp));
  const long long headb_f = kvm_fgpt_r4((long long)kvm_fgpt_r16(SRC) * kvm_fgpt_ldk(vp)) + kvm_fgpt_r4((long long)kvm_fgpt_r16(SRC) * kvm_fgpt_ldmn(d)) +
                            kvm_fgpt_r4((long long)kvm_fgpt_r8(d) * kvm_fgpt_ldk(vp));
  return mx2(mx2(La.total, Lm.total), mx2(mx2(head_f, headb_f), (long long)SRC * d));
}

// The largest layout of a half with at most `max_g` members whose GEMMs tile (the attention's Gc a multiple of 3), the fewest row groups
// among equals; false when none does.
inline bool attn_layout(int N, int d, int kc, int max_g, int& gt, int& gc) {
  ATiling t;
  for (int a = max_g; a >= 1; --a)
    for (int g = 1; g <= a; ++g)
      if (a % g == 0 && (a / g) % 3 == 0 && !attn_tiling(N, d, g, a / g, kc, t)) { gt = g; gc = a / g; return true; }
  return false;
}
inline bool mlp_layout(int N, int d, int F, int kc, int max_g, int& gt, int& gc) {
  MTiling t;
  for (int a = max_g; a >= 1; --a)
    for (int g = 1; g <= a; ++g)
      if (a % g == 0 && !mlp_tiling(N, d, F, g, a / g, kc, t)) { gt = g; gc = a / g; return true; }
  return false;
}

// F's form for a GPT run (the fused step's template: kvm_fuse_gpt.h's KvmGptStep) on a device whose member may take `smem_budget` floats
// of shared memory and whose gang may have `max_g` members: nullptr with `out` filled, or why F cannot run it (`out` then empty, G 0).
// Each half takes its largest layout within the gang (on an A100, d 384: attention 6 x 12, MLP 6 x 16, a gang of 96).
inline const char* fit(int N, int T, int B, int d, int H, int F, int V, int L, int mm_numerics, long long smem_budget, int max_g, Fit& out) {
  out = Fit{};                                                             // a run F cannot take leaves no form behind
  if (mm_numerics != kNumerics) return kPrec ? "GEMM numerics this build does not reproduce (it is F's precise build: mm_numerics 16)"
                                             : "GEMM numerics F's executor build does not reproduce (it runs 1xTF32 bundles, mm_numerics 0)";
  if (L < 1 || L > MAXL) return "a layer count outside 1 .. KVM_FGPT_MAXL";
  if (T <= 0 || B <= 0 || N != B * T || H <= 0 || d % H) return "a shape whose rows or heads do not divide (N = B T, d = H hd)";
  const int hd = d / H;
  if (hd % 8 || hd > 128) return "a head width F's attention items do not take (a multiple of 8, at most 128)";
  if (d % 4 || V < 1) return "a width that is not a multiple of 4, or no vocabulary";
  Fit f;
  if (!attn_layout(N, d, KC, max_g, f.gta, f.gca)) return "no attention layout tiles within the gang (q|k|v, proj and their backward GEMMs; Gc a multiple of 3)";
  if (!mlp_layout(N, d, F, KC, max_g, f.gtm, f.gcm)) return "no MLP layout tiles within the gang (fc1, fc2 and their backward GEMMs)";
  f.G = (int)mx2(f.gta * f.gca, f.gtm * f.gcm);
  f.qb = kvm_fgpt_qblocks(B, H, T, f.G);
  f.nv = f.G < (V + 15) / 16 ? f.G : (V + 15) / 16;                      // vocabulary shards of at least 16 columns
  f.vp = (V + f.nv - 1) / f.nv;
  ATiling ta; MTiling tm;
  attn_tiling(N, d, f.gta, f.gca, KC, ta);
  mlp_tiling(N, d, F, f.gtm, f.gcm, KC, tm);
  f.smem_floats = smem_floats(ta, tm, T, T / f.qb, d, hd, f.vp, KC);
  if (f.smem_floats > smem_budget) return "a member's phases need more shared memory than the device gives a block";
  out = f;
  return nullptr;
}

// ---- the descriptor ---------------------------------------------------------------------------------------------------------------------
// Run d's descriptor on fit f: the shape, the hyperparameters, the parameter slots (each with its gradient, moments and weight decay), the
// streams, the step counter and the losses come from d (the executor's KvmFgptDesc; its gang, chunk, scratch, bias-correction and control
// fields are the fused step's and are not read). `alloc(n)` returns n zeroed device floats that outlive the run; every other buffer F
// keeps comes from it: the saved activations of every layer, the exchanges, the partial gradients, the GEMMs' rounded weight shadows and
// the gang's control words. The run's error word is set apart (set_abort), since the executor allocates it at launch. Returns nullptr, or
// what in d contradicts f or the GPT's slots.
template <class Alloc>
inline const char* build(const KvmFgptDesc& d, const Fit& f, Alloc&& alloc, SDesc& s) {
  s = SDesc{};
  const int N = d.N, T = d.T, B = d.B, D = d.d, H = d.H, hd = d.hd, F = d.F, V = d.V, L = d.L, G = f.G;
  if (L < 1 || L > MAXL || N != B * T || H * hd != D) return "the run's shape is not a GPT F holds";
  ATiling ta; MTiling tm;
  if (const char* why = attn_tiling(N, D, f.gta, f.gca, f.kc, ta)) return why;
  if (const char* why = mlp_tiling(N, D, F, f.gtm, f.gcm, f.kc, tm)) return why;
  if (f.gca % 3) return "the attention layout's Gc is not a multiple of 3, so a member's q|k|v columns straddle two slots";
  const int Ga = f.gta * f.gca, Gm = f.gtm * f.gcm;
  if (G < Ga || G < Gm || f.qb < 1 || T % f.qb || f.nv < 1 || f.nv > G || (long long)f.nv * f.vp < V) return "the fit does not hold the run";
  // the slots: their shapes are the GPT's roles (kvm_fused_gpt.h's slot order)
  const int np = kvm_fgpt_nparams(L);
  auto shape_is = [&](int k, int r, int c) { return d.prows[k] == r && d.pcols[k] == c; };
  if (!shape_is(0, V, D) || !shape_is(kvm_fgpt_slot_nfg(L), 1, D) || !shape_is(kvm_fgpt_slot_nfb(L), 1, D) || !shape_is(kvm_fgpt_slot_head(L), D, V))
    return "a slot outside the layers has another shape than the GPT's (wte V x d, nf 1 x d, head d x V)";
  for (int l = 0; l < L; ++l) {
    const int sh[KVM_FGPT_PER_LAYER][2] = {{1, D}, {1, D}, {D, D}, {1, D}, {D, D}, {1, D}, {D, D}, {1, D}, {D, D}, {1, D}, {1, D}, {1, D}, {D, F}, {1, F}, {F, D}, {1, D}};
    for (int r = 0; r < KVM_FGPT_PER_LAYER; ++r)
      if (!shape_is(kvm_fgpt_slot_layer(l, r), sh[r][0], sh[r][1])) return "a layer's slot has another shape than its role";
  }
  for (int k = 0; k < np; ++k)
    if (!d.p[k] || !d.m[k] || !d.v[k] || !d.g[k]) return "a parameter slot without its buffers";
  // the arrays, in the order the flat shares walk them (the squares' sums follow it): the layers' slots, then wte, nf's g and b, the head
  int arr_of[MAXA];
  s.na = 0;
  auto add = [&](int k) { arr_of[k] = s.na; s.arr[s.na++] = SArr{d.p[k], d.m[k], d.v[k], d.g[k], nullptr, (long long)d.prows[k] * d.pcols[k], d.wd[k]}; };
  for (int k = 1; k <= KVM_FGPT_PER_LAYER * L; ++k) add(k);
  add(0); add(kvm_fgpt_slot_nfg(L)); add(kvm_fgpt_slot_nfb(L)); add(kvm_fgpt_slot_head(L));
  auto A_ = [&](int l, int role) -> SArr& { return s.arr[arr_of[kvm_fgpt_slot_layer(l, role)]]; };
  if (kPre)   // every GEMM weight (q, k, v, proj, fc1, fc2) gets the rounded shadow the halves' GEMMs read, written at the run's start and by AdamW
    for (int l = 0; l < L; ++l)
      for (int role : {KVM_FGPT_R_WQ, KVM_FGPT_R_WK, KVM_FGPT_R_WV, KVM_FGPT_R_WO, KVM_FGPT_R_W1, KVM_FGPT_R_W2}) A_(l, role).r = alloc((size_t)A_(l, role).n);
  auto gw = [&](int l, int role) -> const float* { const SArr& a = A_(l, role); return a.r ? a.r : a.p; };   // a GEMM's weight: its shadow, else itself
  // the layers: per-layer activations kept for the backward; the exchanges and the backward's scratch shared by the layers
  const size_t Nd = (size_t)N * D;
  const int TQ = T / f.qb, W = B * H * f.qb;
  float* xs[MAXL + 1]; float* dxs[MAXL + 1]; float* dxsg[MAXL + 1];       // dxsg: dx[1 .. L]'s rounded copies, the MLPs' GEMM dxo
  for (int l = 0; l <= L; ++l) { xs[l] = alloc(Nd); dxs[l] = alloc(Nd); dxsg[l] = kPre && l >= 1 ? alloc(Nd) : nullptr; }
  float* const aP = alloc((size_t)Ga * ta.R * D); float* const aDKV = alloc((size_t)W * T * 2 * hd); float* const adO = alloc(Nd);
  float* const adQKV = alloc(3 * Nd); float* const adH = alloc(Nd); float* const aY = alloc(Nd);
  float* const mP = alloc((size_t)Gm * tm.R * D); float* const mdU = alloc((size_t)N * F); float* const mdH2 = alloc(Nd);
  float* const mdXM = alloc(Nd); float* const mY2 = alloc(Nd);
  float* const mdXMg = kPre ? alloc(Nd) : nullptr;                         // dxm's GEMM copy, the attention's dY
  float* const rtab = alloc((size_t)T * hd);                               // filled on the device (rope_table)
  unsigned* const count = reinterpret_cast<unsigned*>(alloc((size_t)ctl_words(G)));
  for (int l = 0; l < L; ++l) {
    ADesc& pa = s.att[l];
    pa.N = N; pa.T = T; pa.B = B; pa.d = D; pa.H = H; pa.hd = hd; pa.gt = f.gta; pa.gc = f.gca; pa.kc = f.kc; pa.qb = f.qb; pa.TQ = TQ; pa.W = W;
    pa.scale = d.scale; pa.eps = d.ln_eps;
    attn_desc_tiling(pa, ta);
    pa.X = xs[l]; pa.g1 = A_(l, KVM_FGPT_R_N1G).p; pa.b1n = A_(l, KVM_FGPT_R_N1B).p; pa.Wo = gw(l, KVM_FGPT_R_WO); pa.bo = A_(l, KVM_FGPT_R_BO).p; pa.rtab = rtab;
    for (int j = 0; j < 3; ++j) {                                           // q, k, v: the three slots, read in place
      const int rw = KVM_FGPT_R_WQ + 2 * j, rb = KVM_FGPT_R_BQ + 2 * j;
      pa.Wq3[j] = gw(l, rw); pa.bq3[j] = A_(l, rb).p; pa.dWq3[j] = A_(l, rw).g; pa.dbq3[j] = A_(l, rb).g;
    }
    pa.ldw = pa.ldg = D;
    pa.Hs = alloc(Nd); pa.mean = alloc(N); pa.rstd = alloc(N); pa.XM = alloc(Nd); pa.dX = dxs[l];
    pa.dg1p = alloc((size_t)Ga * D); pa.db1np = alloc((size_t)Ga * D); pa.dg1 = A_(l, KVM_FGPT_R_N1G).g; pa.db1n = A_(l, KVM_FGPT_R_N1B).g;
    pa.QKV = alloc(3 * Nd); pa.O = alloc(Nd); pa.LSE = alloc((size_t)B * H * T); pa.P = aP; pa.Y = aY;
    pa.dY = mdXM; pa.dYg = mdXMg; pa.dXg = l >= 1 ? dxsg[l] : nullptr; pa.dO = adO; pa.dQKV = adQKV; pa.DKV = aDKV;
    pa.dWop = alloc((size_t)f.gta * D * D); pa.dbop = alloc((size_t)f.gta * D); pa.dWqkvp = alloc((size_t)f.gta * D * 3 * D); pa.dbqkvp = alloc((size_t)f.gta * 3 * D);
    pa.dWo = A_(l, KVM_FGPT_R_WO).g; pa.dbo = A_(l, KVM_FGPT_R_BO).g; pa.dH = adH;
    pa.count = count;
    MDesc& pm = s.mlp[l];
    pm.N = N; pm.d = D; pm.F = F; pm.gt = f.gtm; pm.gc = f.gcm; pm.kc = f.kc; pm.R = tm.R; pm.Fs = tm.Fs; pm.cpw = tm.cpw; pm.cpw_rs = tm.cpw_rs; pm.cpw_sr = tm.cpw_sr;
    pm.eps = d.ln_eps;
    pm.XM = pa.XM; pm.g2 = A_(l, KVM_FGPT_R_N2G).p; pm.b2n = A_(l, KVM_FGPT_R_N2B).p; pm.W1 = gw(l, KVM_FGPT_R_W1); pm.b1 = A_(l, KVM_FGPT_R_B1).p;
    pm.W2 = gw(l, KVM_FGPT_R_W2); pm.b2 = A_(l, KVM_FGPT_R_B2).p;
    pm.dXO = dxs[l + 1]; pm.dXOg = dxsg[l + 1]; pm.dXMg = mdXMg;
    pm.H2 = alloc(Nd); pm.mean2 = alloc(N); pm.rstd2 = alloc(N); pm.U = alloc((size_t)N * F); pm.A = alloc((size_t)N * F);
    pm.P = mP; pm.Y2 = mY2; pm.XO = xs[l + 1];
    pm.dU = mdU; pm.dH2 = mdH2; pm.dXM = mdXM;
    pm.dW1p = alloc((size_t)f.gtm * D * F); pm.dW2p = alloc((size_t)f.gtm * F * D); pm.db1p = alloc((size_t)f.gtm * F); pm.db2p = alloc((size_t)f.gtm * D);
    pm.dg2p = alloc((size_t)Gm * D); pm.db2np = alloc((size_t)Gm * D);
    pm.dW1 = A_(l, KVM_FGPT_R_W1).g; pm.dW2 = A_(l, KVM_FGPT_R_W2).g; pm.db1 = A_(l, KVM_FGPT_R_B1).g; pm.db2 = A_(l, KVM_FGPT_R_B2).g;
    pm.dg2 = A_(l, KVM_FGPT_R_N2G).g; pm.db2n = A_(l, KVM_FGPT_R_N2B).g;
    pm.count = count;
  }
  // the step: around the layers, the streams and the run's buffers
  s.N = N; s.T = T; s.d = D; s.V = V; s.L = L; s.G = G; s.steps = d.steps; s.nv = f.nv; s.vp = f.vp; s.smem_floats = f.smem_floats;
  s.step = d.step; s.ids_len = d.ids_len; s.tgt_len = d.tgt_len;
  s.ln_eps = d.ln_eps; s.lr = d.lr; s.beta1 = d.beta1; s.beta2 = d.beta2; s.adam_eps = d.adam_eps; s.max_norm = d.max_norm; s.inv_n = (float)(1.0 / N);
  s.ids = d.ids_src; s.tgt = d.tgt_src; s.ids_stride = d.ids_stride; s.ids_off = d.ids_off; s.tgt_stride = d.tgt_stride; s.tgt_off = d.tgt_off;
  const SArr &Awte = s.arr[arr_of[0]], &Anfg = s.arr[arr_of[kvm_fgpt_slot_nfg(L)]], &Anfb = s.arr[arr_of[kvm_fgpt_slot_nfb(L)]], &Ahead = s.arr[arr_of[kvm_fgpt_slot_head(L)]];
  s.wte = Awte.p; s.nfg = Anfg.p; s.nfb = Anfb.p; s.head = Ahead.p;
  s.gwte = Awte.g; s.gnfg = Anfg.g; s.gnfb = Anfb.g; s.ghead = Ahead.g;
  s.x0 = xs[0]; s.xL = xs[L]; s.dxL = dxs[L]; s.hf = alloc(Nd); s.mf = alloc(N); s.rf = alloc(N); s.logits = alloc((size_t)N * V);
  s.ce = alloc((size_t)f.nv * N * 2); s.tl = alloc(N); s.lse = alloc((size_t)f.nv * N); s.dhfp = alloc((size_t)f.nv * Nd); s.dyf = alloc(Nd);
  s.lnfp = alloc((size_t)G * 2 * D); s.gn = alloc(G); s.bc = alloc((size_t)2 * d.steps);
  s.rtab = rtab; s.dxLg = dxsg[L]; s.theta = d.theta; s.loss = d.loss;
  s.count = count;
  return nullptr;
}
// The run's error word, in every descriptor that crosses: the step's and each half's.
inline void set_abort(SDesc& s, const int* abort) {
  s.abort = abort;
  for (int l = 0; l < MAXL; ++l) { s.att[l].abort = abort; s.mlp[l].abort = abort; }
}

}  // namespace kvm_f2d
