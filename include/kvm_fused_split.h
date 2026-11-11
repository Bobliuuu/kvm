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
KVM_SPLIT_HD inline int kvm_split_layout(KvmSplitDesc& d) {
  if (d.L < 1 || d.L > KVM_SPLIT_MAXL || d.E % 16 || d.E > 128 || d.V % 4 || d.Q < 1) return -1;
  d.VP = kvm_split_pad16(d.V); d.HR = kvm_split_pad16(d.H); d.T = KVM_SPLIT_R * d.Q;
  const int wmax = kvm_split_shares(d);
  if (wmax < 0 || d.VP > 128) return -1;
  d.HP = kvm_split_pad16(wmax);
  if (d.HP > 128) return -1;
  const int R = KVM_SPLIT_R;
  int off = kvm_split_desc_floats();
  for (int l = 0; l < d.L; ++l) { d.Ws[l] = off; off += (l ? d.HR : d.E) * d.HP; }
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
