#pragma once
// The split fused MLP step: the descriptor and shared-memory layout of `fmlp_split`, host and device (host-only callers include it with plain g++).
//
// A split run trains one MLP whose step one block cannot hold (kvm_fused_mlp.h's slices, past their budget), as one gang op a step: Q x S
// members (kvm_static_plan's gangs), member q * S + s taking row slice q (KVM_SPLIT_R rows) and share s of every layer's hidden units: those
// columns of each layer's weights and bias, and those rows of the head. The members meet at kvm_gang::cross, 2 x L + 1 times a step:
//  * an all-gather of each hidden layer below the top (its columns written to XH, then the slice's whole layer read back);
//  * the logits' partials (P);
//  * a reduce-scatter of each activation gradient above layer 0 (full-width partials in XD, each member summing its own columns);
//  * the partial gradients, for the owners;
//  * the owners' updates, so the next step reads the new parameters.
// The owners: the Q members of share s split its parameters and sum the Q slices' partials in slice order, and every member owns a few
// embedding rows. Fixed orders throughout: a run is deterministic. Device side: kvm_fused_split.cuh. Measured as bench/fused_mlp/split_bed.cu
// (on the A100 and an RTX 3080 Ti).
//
// Virtual shares: three sums run over shares, the logits' partials (P), the reduce-scatter's (XD) and the embedding rows' (dX0), so the share
// count is part of a run's arithmetic. It is fixed by the run's shape: Sv virtual shares, virtual share t the hidden units [share0[t],
// share0[t + 1]), whole 8-unit blocks. The S physical shares of a gang (1 <= S <= Sv, chosen per bundle so that several runs' gangs can run at
// once) each carry a contiguous range of them, [vs0[s], vs0[s + 1]). A member computes each of its virtual shares' partials on its own, with
// that share's columns as its whole K, and every sum reads all Sv partials in virtual-share order. So a run computes the same bits at any S:
// the partial of a virtual share is one fixed chain of k8 steps over its columns whichever member computes it (kvm_fused_mlp.cuh's gemm keeps a
// k-loop per warp, no split K), and where a member of today's S = Sv layout ran its share's padded columns, the pad's chunks add zeros.
#include <cstddef>
#ifdef __CUDACC__
#define KVM_SPLIT_HD __host__ __device__
#else
#define KVM_SPLIT_HD
#endif
#define KVM_SPLIT_MAXL 4     // hidden layers
#define KVM_SPLIT_MAXS 18    // shares of a slice (the gang op's template instantiations: kvm_fused_split.cuh)
#define KVM_SPLIT_R 16       // rows of a slice (one m16 row block)
#define KVM_SPLIT_Q 6        // slices of a run: the op is instantiated for 6 (T = 96, the panel's and the hold-out's MLP runs)
// The share counts the op is instantiated for (its partial-sum loops are unrolled over S): the pass chooses among these, the static worker dispatches
// on them, and a run that needs another stays unfused. X(S) once each. 16 is the panel's mlp256 (h256-d4): its 32 blocks of 8 units split 2 a share,
// so a member holds 23,452 floats; at 8 to 15 shares the widest share is 24 units, padded to 32 columns, and a member needs 39,900 (budget 38,400).
#define KVM_SPLIT_SHARE_LIST X(2) X(3) X(4) X(5) X(6) X(8) X(9) X(10) X(12) X(16) X(18)
struct alignas(16) KvmSplitDesc {   // copied to shared memory with 16-byte cp.async
  // shape: vocabulary (VP rounded up to 16), embedding width, hidden layers, hidden width (HR rounded up to 16), the widest physical share rounded
  // up to 16, physical shares, slices, rows a step, virtual shares (the run's arithmetic: see above)
  int V, VP, E, L, H, HR, HP, S, Q, T, Sv;
  int share0[KVM_SPLIT_MAXS + 1];    // virtual share t: hidden units [share0[t], share0[t + 1]) of every layer (whole 8-unit blocks)
  int vs0[KVM_SPLIT_MAXS + 1];       // physical share s: virtual shares [vs0[s], vs0[s + 1]), so hidden units [share0[vs0[s]], share0[vs0[s + 1]])
  // shared-memory offsets (floats from the arena's base; the descriptor sits below them)
  int Ws[KVM_SPLIT_MAXL], bs, Whs, X0, Hin, ZG, Hs, Lg, dZ, rl, ints, gtok, total;
  float lr;
  // the model in global memory, row-major: embedding [V][E], W[0] [E][H], W[l > 0] [H][H], b[l] [H], head [H][V]
  float *we, *w[KVM_SPLIT_MAXL], *b[KVM_SPLIT_MAXL], *wh;
  // per member (slot q * S + s): the partial gradients dW[l] [E or HR][HP], db[l] [HP], dWh [HP][VP]; per virtual share (slot q * Sv + t): the
  // logits' partial P [R][VP] and the embedding rows' dX0 [R][E]
  float *P, *dW[KVM_SPLIT_MAXL], *db[KVM_SPLIT_MAXL], *dWh, *dX0;
  // the exchanges, zeroed once (the pad columns past H are never written):
  //  * XH[l] [Q][R][HR]: layer l's activations (l < L - 1);
  //  * XD[2] [virtual slot q * Sv + t][R][HR]: the backward's full-width partials, layer l's in XD[l & 1].
  // Two buffers, because a member overwrites its own slot at the next reduce-scatter while a slower member of its slice may still be reading it
  // from this one. With layers alternating, a slot is rewritten only after another reduce-scatter's barrier, which every reader must reach first.
  float *XH[KVM_SPLIT_MAXL], *XD[2];
  float* loss_part;                  // [Q]: each slice's sum of row losses
  // the gang's control words (kvm_gang::cross: line_words(Q x S) words, 128-byte aligned), zeroed once when the run's buffers are made. One
  // set serves every step: the words need no reset, and a run's gangs never overlap in time (a step's members start after the last step's final
  // barrier, and member m's next step waits on its last, so it reads its own word as that step left it)
  unsigned* ctl;
};
KVM_SPLIT_HD inline constexpr int kvm_split_pad16(int x) { return (x + 15) & ~15; }
// Floats of shared memory the descriptor occupies (it is copied to the arena's base at the start of each task).
KVM_SPLIT_HD inline constexpr int kvm_split_desc_floats() { return (int)((sizeof(KvmSplitDesc) + 63) / 64 * 16); }
// The virtual shares of H hidden units in whole 8-unit blocks (every share starts 32-byte aligned), and each physical share's range of them;
// returns the widest physical share, or -1 when the counts do not fit the table. Sv = 0 means Sv = S (one virtual share a member).
KVM_SPLIT_HD inline int kvm_split_shares(KvmSplitDesc& d) {
  if (!d.Sv) d.Sv = d.S;
  if (d.S < 1 || d.Sv < d.S || d.Sv > KVM_SPLIT_MAXS || d.H % 8) return -1;
  for (int t = 0; t <= d.Sv; ++t) d.share0[t] = 8 * (int)((long long)t * (d.H / 8) / d.Sv);
  for (int s = 0; s <= d.S; ++s) d.vs0[s] = (int)((long long)s * d.Sv / d.S);
  int wmax = 0;
  for (int s = 0; s < d.S; ++s) { const int w = d.share0[d.vs0[s + 1]] - d.share0[d.vs0[s]]; wmax = wmax > w ? wmax : w; }
  return wmax;
}
// Fills the derived shape (VP, HR, HP, T), the shares and the shared-memory offsets of `d` from V, E, L, H, S, Sv (0: S) and Q; returns the
// floats of shared memory a member needs, or -1 for a shape the op does not take:
//  * widths: H a multiple of 8, E a multiple of 16 and at most 128, V a multiple of 4 with VP at most 128;
//  * the widest physical share at most 128 (one GEMM's N);
//  * 1 <= L <= KVM_SPLIT_MAXL.
// The embedding owners' scratch [T][E] sits over the weights when they are at least as large (they are dead from the owners to the next step's load).
//
// Streamed weights (kc > 0, the member split_run<..., true> runs): a member holds no layer's weights for the step. Ws[0] and Ws[1] are two
// staging tiles of kc rows x HP, which carry each layer's share kc rows at a time (cp.async, one tile loading while the other is read), in the
// forward as the GEMM's K rows and in the backward as its N rows. Each GEMM keeps its chains (kvm_fused_split.cuh), so a member computes the
// resident member's bits; what changes is its memory, which no longer grows with L x HR x HP, so a gang can take fewer physical shares and
// several runs' gangs fit the workers at once. Ws[l >= 2] are unused (-1). kc is a multiple of 16 (the backward's N), at most 128.
KVM_SPLIT_HD inline int kvm_split_layout(KvmSplitDesc& d, int kc = 0) {
  if (d.L < 1 || d.L > KVM_SPLIT_MAXL || d.E % 16 || d.E > 128 || d.V % 4 || d.Q < 1) return -1;
  if (kc < 0 || kc % 16 || kc > 128) return -1;
  d.VP = kvm_split_pad16(d.V); d.HR = kvm_split_pad16(d.H); d.T = KVM_SPLIT_R * d.Q;
  const int wmax = kvm_split_shares(d);
  if (wmax < 0 || d.VP > 128) return -1;
  d.HP = kvm_split_pad16(wmax);
  if (d.HP > 128) return -1;
  const int R = KVM_SPLIT_R;
  int off = kvm_split_desc_floats();
  if (kc) { for (int l = 0; l < KVM_SPLIT_MAXL; ++l) d.Ws[l] = -1; for (int b = 0; b < 2; ++b) { d.Ws[b] = off; off += kc * d.HP; } }
  else for (int l = 0; l < d.L; ++l) { d.Ws[l] = off; off += (l ? d.HR : d.E) * d.HP; }
  d.bs = off; off += d.L * d.HP;
  d.Whs = off; off += d.HP * d.VP;
  const int weights = off - d.Ws[0];
  d.X0 = off; off += R * d.E;
  d.Hin = off; off += d.L > 1 ? R * d.HR : 0;
  d.ZG = off; off += d.L * R * d.HP;
  d.Hs = off; off += R * d.HP;
  d.Lg = off; off += R * d.VP;
  d.dZ = off; off += R * d.HP;
  d.rl = off; off += R;
  d.ints = off; off += (d.T + R + d.T + 12 + 3) & ~3;   // sid [T], tg [R], mlist [T], wcnt [9]
  if (d.T * d.E <= weights) d.gtok = d.Ws[0]; else { d.gtok = off; off += d.T * d.E; }
  d.total = off;
  return off;
}
// The streamed member's staging rows for `d` within `budget` floats: the largest of 64, 32 and 16 whose layout fits (more rows a tile, fewer
// tile waits a GEMM), with `d` laid out at it; 0 (and `d` laid out resident) when none fits. The device reads the rows back from the layout:
// kvm_split_stream_rows.
KVM_SPLIT_HD inline int kvm_split_stream_kc(KvmSplitDesc& d, int budget) {
  for (int kc = 64; kc >= 16; kc -= 16) {
    if (kc == 48) continue;
    const int f = kvm_split_layout(d, kc);
    if (f > 0 && f <= budget) return kc;
  }
  kvm_split_layout(d);
  return 0;
}
KVM_SPLIT_HD inline int kvm_split_stream_rows(const KvmSplitDesc& d) { return (d.Ws[1] - d.Ws[0]) / d.HP; }
// A descriptor laid out for the streamed member: two staging tiles of 16, 32 or 64 rows, back to back, and no weight arrays past them.
KVM_SPLIT_HD inline bool kvm_split_is_streamed(const KvmSplitDesc& d) {
  const int kc = d.HP > 0 ? kvm_split_stream_rows(d) : 0;
  return (kc == 16 || kc == 32 || kc == 64) && (d.Ws[1] - d.Ws[0]) % d.HP == 0 && d.Ws[2] == -1 && d.Ws[3] == -1 && d.bs == d.Ws[1] + kc * d.HP;
}
// The virtual share counts the streamed member is built for (the static path's split_stream entries, one kernel a count): 18, the count every
// split shape the hold-out's drawer makes takes on the A100 (widths 80-640 at depths 1-4, kvm_fuse_split_fits). Each entry costs ptxas
// minutes, so the list holds what runs; a run at another count keeps the resident member. Add a count here to stream it.
#define KVM_SPLIT_STREAM_LIST X(18)
// The fold by which streaming must cut a bundle's waves before the pass takes it (kvm_fuse.h): a streamed wave of narrow members cost x2.65 of
// a resident one on the A100's split bed, so a 3-fold cut is the measured break-even with a margin.
#define KVM_SPLIT_STREAM_WAVE_CUT 3
KVM_SPLIT_HD inline bool kvm_split_stream_built(int Sv) {
  switch (Sv) {
#define X(S_) case S_: return true;
    KVM_SPLIT_STREAM_LIST
#undef X
    default: return false;
  }
}
