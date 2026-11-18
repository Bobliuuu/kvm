#pragma once
// The batched MLP step (host and device; host-only callers include it with plain g++): one gang carries one step of R independent runs of
// the same MLP, for the draws whose runs the split and grid steps can only take in turns (R gangs of their size past the workers).
//
// Every weight and activation stays in global memory (L2, or HBM when R runs' weights do not fit L2) and streams through shared memory, so
// the gang's size never depends on the model's: its members are the workers. A step is a fixed list of phases separated by gang crossings
// (mvm_gang::cross); a phase is a set of independent items, each one run's output tile with the whole of its K, so no element has two
// writers or a partial sum, a run is deterministic, and each run's result is bitwise the one it would have alone. Member m takes items
// m, m + members, ... of each phase. The phases (T = 96 rows, L hidden layers, the panel's MLP: embedding V x E, linear layers with bias
// then tanh GELU, a head H x V without bias, softmax cross-entropy over the step's T tokens, SGD):
//   F0          layer 0, by column tiles: Z0 = X0 W0 + b0, Y0 = gelu(Z0), G0 = gelu'(Z0); X0 is gathered from the embedding rows of the
//               step's tokens by every tile, and the run's first tile also stores it for BU
//   F1..F(L - 1)  the hidden layers, by column tiles: Z_l = Y_(l-1) W_l + b_l, Y_l, G_l                                         (K = H)
//   HC          the head and the cross-entropy, by blocks of 16 rows (an item owns whole rows of the logits): each row's loss and
//               its logits' gradient dlog = (softmax - onehot) / T                                                               (K = H)
//   BH          the head's backward, by row blocks of Wh: dZ_(L - 1)[:, k] = (dlog Wh[k, :]^T) * G_(L - 1)[:, k], and from the same
//               read of Wh[k, :], Wh[k, :] -= lr Y_(L - 1)[:, k]^T dlog                                                           (K = V, then T)
//   B(L - 1)..B1  each hidden layer's backward the same way: dZ_(l-1)[:, k] and W_l[k, :] -= lr Y_(l-1)[:, k]^T dZ_l; the run's first row
//               block also takes b_l -= lr colsum(dZ_l)                                                                         (K = H, then T)
//   BX          dX0 = dZ_0 W0^T, by blocks of 16 rows                                                                           (K = H)
//   BU          W0 -= lr X0^T dZ_0 and b0, by column tiles; then one item a run: its embedding rows (each row's gradient summed in
//               token order, then applied once) and the step's loss (the row losses summed in order, over T)                 (K = T)
// 2 L + 3 phases and crossings a step: the last ends the step, so the next step's gather reads the updated embedding. Each weight is read
// once forward and read and written once backward: 12 bytes a parameter a step.
// Device side: mvm_fused_batch.cuh. Measured as bench/fused_mlp/batch_bed.cu.
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define MVM_BATCH_HD __host__ __device__
#else
#define MVM_BATCH_HD
#endif

#define MVM_BATCH_MAXL 4      // hidden layers
#define MVM_BATCH_T 96        // rows a step (6 row blocks of 16 at compile time)
#define MVM_BATCH_MAXR 64     // runs a gang carries
#define MVM_BATCH_BK 32       // the streamed GEMMs' k chunk
#define MVM_BATCH_NST 3       // stages of the k-chunk ring (measured: a deeper ring changed nothing, and a larger arena cost ~10% of L1)
#define MVM_BATCH_MAXS 18     // virtual shares of the split step's arithmetic (mvm_fused_split.h: MVM_SPLIT_MAXS)
#define MVM_BATCH_SLICE 16    // rows of the split step's slices (mvm_fused_split.h: MVM_SPLIT_R)
#define MVM_BATCH_GRID_MAXRB 3   // row blocks of 16 a warp owns on the grid path: each holds up to five accumulator sets of 8 floats (a block's
                                 // partial, its chain and three tree nodes), so 3 keeps them at 120 floats, the widest tiles the own path takes
                                 // at R > 1 (96 x 64 forward, 64-row backward blocks)

// The row blocks of 16 each warp of 8 owns in an (mrb x 16) x (ncb x 16) tile: mvm_fused_batch.cuh's WarpMap<mrb, ncb>::RB.
MVM_BATCH_HD inline int mvm_batch_warp_rb(int mrb, int ncb) {
  const int wpc = 8 / ncb < mrb ? 8 / ncb : mrb;
  return (mrb + wpc - 1) / wpc;
}

struct MvmBatchRun {   // one run's model and scratch, row-major, as the bundle holds the model
  float* emb;                      // [V][E]
  float* w[MVM_BATCH_MAXL];        // w[0] [E][H], w[l > 0] [H][H]
  float* b[MVM_BATCH_MAXL];        // [H]
  float* wh;                       // [H][V]
  const int32_t* const* ids;       // [steps]: each step's T tokens (the bundle's own buffer of that step)
  const int32_t* const* tgt;       // [steps]: each step's T targets
  float* const* loss;              // [steps]: each step's loss, two floats: the mean, then 1 / T (as the split step and softmax_ce_reduce write it)
  // scratch, every element written before it is read:
  float* x0;                       // [T][E] the gathered embedding rows
  float* y[MVM_BATCH_MAXL];        // [T][H] gelu(Z_l)
  float* g[MVM_BATCH_MAXL];        // [T][H] gelu'(Z_l)
  float* dlog;                     // [T][VP] the logits' gradient (columns past V are zero)
  float* dz[3];                    // [T][H] dZ, layer l's in dz[l % 3]: a split backward phase reads two layers' and writes a third's
  float* dx0;                      // [T][E]
  float* rloss;                    // [T] each row's loss
};

struct MvmBatchDesc {
  int R, E, V, VP, H, L, T, members;
  int steps;                       // the steps each run's ids, tgt and loss hold a pointer for (the static entry refuses a step past them)
  int nw, nt;                      // forward column tiles: nt a run, each nw columns (16, 32, 64 or 128; the last may be narrower)
  int fm, fr;                      // forward row blocks: fr a run, each fm x 16 rows (fm 1, 2, 3 or 6; fr = 6 / fm)
  int bsplit;                      // the backward's items: 0 fused (a weight's row block: its dX and its update together, kw x kt),
                                   // 1 split (dX tiles of fm x 16 rows by nw of the weight's rows, then the weight's update tiles of fm x 16 of
                                   // its rows by nw of its columns, one phase later). Each output's sum runs in one order either way: the
                                   // same arithmetic, so the bed can force either and compare them bit for bit
  int kw, kt;                      // backward row blocks of a weight: kt a run, each kw rows (16, 32 or 64)
  int xfm, xnw;                    // the split backward's dX tiles: xfm x 16 rows by xnw of the weight's rows (by default the forward's fm and nw)
  int wfm, wnw;                    // its update tiles: wfm x 16 of the weight's rows by wnw of its columns (by default fm and nw)
  int smem_floats;                 // shared memory a member needs (floats)
  // The arithmetic: the one the shape's single-run form computes (a run's arithmetic depends on its shape alone, so that its copies in one
  // bundle are bitwise the run alone; a bundle may take this op by its run count only when the op computes the single-run form's bits). Sv > 0: the split step's (mvm_fused_split.cuh): the
  // logits, each hidden layer's dX and dX0 as one chain a virtual share, share t the hidden units [share0[t], share0[t + 1]), summed in share
  // order; every sum over the step's rows (the weights' and biases' gradients, the loss) as one chain a 16-row slice, summed in slice order.
  // Sv = 0: each output over its whole K in one chain, the op's own arithmetic, for a shape whose single-run form is this op.
  int Sv;
  int share0[MVM_BATCH_MAXS + 1];  // filled by mvm_batch_layout from H and Sv (Sv = 0: one part, [0, H))
  // gHp > 0 (Sv = 0): the grid step's arithmetic (mvm_fused_grid.cuh at the grid form's P). Five sums run over the grid's blocks, each block's
  // partial one chain from zero, the partials summed as its sum_cg<8> sums them (partial i into chain i mod 8, each chain from 0.f, then a fixed
  // tree): the hidden layers' Z over K in gka blocks, the logits over K in gkah, each hidden dY over N in gnb, dX0 over layer 0's columns in
  // gnb0, and the top layer's dY from the head as one block. gHp is H padded to 16, the blocks' range, its padding zeros; block i of n over a
  // range Kp starts at mvm_batch_goff(Kp, n, i). Copied from mvm_grid_layout's own (the pass and the beds), never derived here.
  int gHp, gka, gnb, gnb0, gkah;
  // Each grid sum's walk as a table of pieces of 16 (mvm_batch_grid_table, built on the host): the device reads one word a piece instead of
  // computing block edges (a division each) in its loop. gtab_off[s] is sum s's first piece in gtab: MVM_BATCH_GSUM_*.
  const unsigned* gtab;
  int gtab_off[5];
  float lr;
  const MvmBatchRun* runs;         // [R]
  unsigned* ctl;                   // the gang's control words (mvm_gang::cross: mvm_gang::line_words(members) words, zeroed once, never reset)
};

// The item kinds a bed's item clock tells apart (mvm_fused_batch.cuh: MVM_BATCH_CLOCK).
enum { CK_F = 0, CK_HC, CK_B, CK_BX, CK_W0, CK_EMB, CK_KINDS };

// The grid sums (MvmBatchDesc::gtab_off's index): the hidden layers' Z over K, the logits, the hidden dY over N, dX0, the head's dY.
enum { MVM_BATCH_GSUM_Z = 0, MVM_BATCH_GSUM_LOGITS, MVM_BATCH_GSUM_DY, MVM_BATCH_GSUM_DX0, MVM_BATCH_GSUM_HEAD, MVM_BATCH_GSUMS };
// A piece's word in the table: its first k (bits 0-15), whether it ends its block (16), whether its block is its chain's first (17), whether it
// ends its chain (18), and its chain (20-22).
#define MVM_BATCH_GP_K(w) ((int)((w) & 0xffffu))
#define MVM_BATCH_GP_BLOCK_END(w) (((w) >> 16) & 1u)
#define MVM_BATCH_GP_FIRST(w) (((w) >> 17) & 1u)
#define MVM_BATCH_GP_CHAIN_END(w) (((w) >> 18) & 1u)
#define MVM_BATCH_GP_CHAIN(w) ((int)(((w) >> 20) & 7u))

// Block i of n over a range of Kp (a multiple of 16) starts here: mvm_fused_grid.h's mvm_grid_off, in granules of 16.
MVM_BATCH_HD inline int mvm_batch_goff(int Kp, int n, int i) { return (int)(((long long)i * (Kp / 16)) / n) * 16; }
MVM_BATCH_HD inline int mvm_batch_pad16(int x) { return (x + 15) & ~15; }

// One grid sum's walk into a table: n blocks over [0, Kp), visited chain by chain (chain u: blocks u, u + 8, ..., as the grid's sum_cg<8>
// sums its partials), each block in pieces of 16. Appends Kp / 16 words (MVM_BATCH_GP_*) to out at *len and advances it; host only.
inline void mvm_batch_grid_walk(int Kp, int n, unsigned* out, int* len) {
  for (int u = 0; u < (n < 8 ? n : 8); ++u)
    for (int i = u; i < n; i += 8) {
      const int k0 = mvm_batch_goff(Kp, n, i), k1 = mvm_batch_goff(Kp, n, i + 1);
      for (int k = k0; k < k1; k += 16) {
        const bool end = k + 16 >= k1;
        out[(*len)++] = (unsigned)k | (end ? 1u << 16 : 0u) | (i == u ? 1u << 17 : 0u) | (end && i + 8 >= n ? 1u << 18 : 0u) | ((unsigned)u << 20);
      }
    }
}
// The five sums' tables of a descriptor with the grid's arithmetic (gHp > 0), into out (at least mvm_batch_grid_words(d) words), and their
// first pieces into d.gtab_off. Returns the words written. The caller uploads them and points d.gtab at the copy.
MVM_BATCH_HD inline int mvm_batch_grid_words(const MvmBatchDesc& d) { return 4 * (d.gHp / 16) + mvm_batch_pad16(d.V) / 16; }
inline int mvm_batch_grid_table(MvmBatchDesc& d, unsigned* out) {
  int len = 0;
  d.gtab_off[MVM_BATCH_GSUM_Z] = len; if (d.L > 1) mvm_batch_grid_walk(d.gHp, d.gka, out, &len);
  d.gtab_off[MVM_BATCH_GSUM_LOGITS] = len; mvm_batch_grid_walk(d.gHp, d.gkah, out, &len);
  d.gtab_off[MVM_BATCH_GSUM_DY] = len; if (d.L > 1) mvm_batch_grid_walk(d.gHp, d.gnb, out, &len);
  d.gtab_off[MVM_BATCH_GSUM_DX0] = len; mvm_batch_grid_walk(d.gHp, d.gnb0, out, &len);
  d.gtab_off[MVM_BATCH_GSUM_HEAD] = len; mvm_batch_grid_walk(mvm_batch_pad16(d.V), 1, out, &len);
  return len;
}

// The step's phases in order: F0, F1..F(L - 1), HC, BH, B(L - 1)..B1, BX, BU.
MVM_BATCH_HD inline constexpr int mvm_batch_phases(int L) { return 2 * L + 3; }

// The split backward's weights: w = L is the head (rows H, columns V), 1..L - 1 the hidden layers (H x H), 0 layer 0 (rows E, columns H).
MVM_BATCH_HD inline int mvm_batch_wrows(const MvmBatchDesc& d, int w) { return w == 0 ? d.E : d.H; }
MVM_BATCH_HD inline int mvm_batch_wcols(const MvmBatchDesc& d, int w) { return w == d.L ? d.V : d.H; }
// Split backward items a run: dX tiles through weight w (the step's rows in xfm-row-block tiles x the weight's rows in xnw tiles), and w's
// update tiles (wfm x 16 of its rows by wnw of its columns). The tiles' shapes spread the work, never change its arithmetic (each output sums
// its whole K in one order), so they need not be the forward's.
MVM_BATCH_HD inline int mvm_batch_dx_items(const MvmBatchDesc& d, int w) { return (6 / d.xfm) * ((mvm_batch_wrows(d, w) + d.xnw - 1) / d.xnw); }
MVM_BATCH_HD inline int mvm_batch_dw_items(const MvmBatchDesc& d, int w) {
  return ((mvm_batch_wrows(d, w) + 16 * d.wfm - 1) / (16 * d.wfm)) * ((mvm_batch_wcols(d, w) + d.wnw - 1) / d.wnw);
}

// Items of phase p (0-based, in the order above). A split backward's phase L + 1 + q holds the dX tiles through weight L - q (q <= L) and the
// update tiles of weight L - q + 1 (q >= 1), then (q = L + 1) one embedding item a run.
MVM_BATCH_HD inline int mvm_batch_items(const MvmBatchDesc& d, int p) {
  const int L = d.L, rb = d.T / 16;
  if (p < L) return d.R * d.fr * d.nt;             // F0..F(L - 1): (run, row block, column tile)
  if (p == L) return d.R * rb;                     // HC
  if (d.bsplit) {
    const int q = p - (L + 1);
    return d.R * ((q <= L ? mvm_batch_dx_items(d, L - q) : 0) + (q >= 1 ? mvm_batch_dw_items(d, L - q + 1) : 0) + (q == L + 1 ? 1 : 0));
  }
  if (p <= 2 * L) return d.R * d.kt;               // BH, B(L - 1)..B1 (the head's row blocks are the hidden width's, like a layer's)
  if (p == 2 * L + 1) return d.R * rb;             // BX
  return d.R * d.nt + d.R;                         // BU: W0's column tiles, then one item a run
}

// Shared memory (floats) the device code needs for the descriptor's tiles: the k-chunk ring of the largest item, plus the backward's
// resident Y block and the head's logits. See mvm_fused_batch.cuh for each item's use.
MVM_BATCH_HD inline int mvm_batch_smem(int nw, int fm, int kw, int xfm, int xnw, int wfm, int wnw) {
  const int BK = MVM_BATCH_BK, T = MVM_BATCH_T, FR = 16 * fm;
  int fwd = MVM_BATCH_NST * (FR * (BK + 4) + BK * (nw + 8)) + 128;                     // F: A [16 fm][36], B [32][nw + 8]; layer 0's token ids (the last 128)
  fwd = fwd > 2 * FR * (nw + 4) ? fwd : 2 * FR * (nw + 4);                              // F's epilogue: Y and G tiles [16 fm][nw + 4] over the ring
  const int hc = MVM_BATCH_NST * (16 * (BK + 4) + BK * (112 + 8)) + 16 * (112 + 4);    // HC: A [16][36], B [32][120]; logits [16][116]
  const int bwd = MVM_BATCH_NST * (T * (BK + 4) + kw * (BK + 4)) + kw * (T + 4) + T * (kw + 4);   // B: dZ [96][36], W [kw][36]; Y^T [kw][100]; G [96][kw + 4]
  const int bx = MVM_BATCH_NST * (16 * (BK + 4) + 48 * (BK + 4));                      // BX: dZ0 [16][36], W0 [48][36] (E = 48)
  const int bu = 48 * (T + 4) + T * (nw + 8);                                          // BU: X0^T [E = 48][100], dZ0 [96][nw + 8]
  const int dw = 16 * wfm * (T + 4) + T * (wnw + 8);                                   // split update tile: Y^T [16 wfm][100], dZ [96][wnw + 8]
  const int dx = MVM_BATCH_NST * (16 * xfm * (BK + 4) + xnw * (BK + 4)) + 16 * xfm * (xnw + 4);   // split dX tile: dZ [16 xfm][36], W [xnw][36]; gi [16 xfm][xnw + 4]
  const int emb = 112 * 48 + T * 48 + 112 + T;                                          // BU's embedding item: [V][E] sums, [T][E] dX0 rows, [V] marks, [T] ids
  int m = fwd;
  m = m > hc ? m : hc; m = m > bwd ? m : bwd; m = m > bx ? m : bx; m = m > bu ? m : bu; m = m > emb ? m : emb; m = m > dw ? m : dw; m = m > dx ? m : dx;
  return m;
}

// The split step's share boundaries for H and Sv (mvm_fused_split.h, mvm_split_shares: whole 8-unit blocks), or one part for Sv = 0.
MVM_BATCH_HD inline void mvm_batch_shares(MvmBatchDesc& d) {
  if (d.Sv <= 0) { d.share0[0] = 0; d.share0[1] = d.H; return; }
  for (int t = 0; t <= d.Sv; ++t) d.share0[t] = 8 * (int)((long long)t * (d.H / 8) / d.Sv);
}

// Fills the derived shape and the tiles for `members` workers; returns the shared-memory floats a member needs, or -1 for a shape the step
// does not take (H a multiple of 8 up to 4096, E 48 (the op's compile-time embedding width), V up to 112 (padded to VP = 112), T 96,
// 1 <= L <= MVM_BATCH_MAXL, 1 <= R <= MVM_BATCH_MAXR). The tiles minimise a phase's rounds times its tile's size: a forward tile of nw columns
// streams the run's whole input (96 x H) for nw outputs a row, so narrow tiles pay in bytes what wide ones pay in rounds.
MVM_BATCH_HD inline int mvm_batch_layout(MvmBatchDesc& d) {
  if (d.L < 1 || d.L > MVM_BATCH_MAXL || d.H < 8 || d.H % 8 || d.H > 4096 || d.E != 48 || d.V < 1 || d.V > 112 || d.V % 4 || d.T != MVM_BATCH_T ||
      d.R < 1 || d.R > MVM_BATCH_MAXR || d.members < 1 || d.Sv < 0 || d.Sv > MVM_BATCH_MAXS)
    return -1;
  const bool grid = d.gHp > 0;   // the grid step's arithmetic: its partition must be the grid layout's for this shape
  if (grid) {
    const int gran = d.gHp / 16;
    if (d.Sv != 0 || d.gHp % 16 || d.gHp != mvm_batch_pad16(d.H) || d.gnb0 < 1 || d.gnb0 > gran || d.gkah < 1 || d.gkah > gran ||
        (d.L > 1 && (d.gka < 1 || d.gka > gran || d.gnb < 1 || d.gnb > gran || d.gnb > 8)) || mvm_batch_pad16(d.V) > 112)   // gnb <= 8: one block a chain (H <= 1024)
      return -1;
  }   // Sv > H / 8 leaves some virtual shares empty, as the split step's do (their partials are zeros): the chains fold them too
  d.VP = 112;
  mvm_batch_shares(d);
  // forward: an item of fm x 16 rows and nw columns over K = H streams its rows of the input and its columns of the weight, 4 (16 fm + nw) H
  // bytes at ~28 B/cycle a member (measured on the A100), and multiplies 2 (16 fm) nw H FLOPs at ~300 a cycle (30% of an SM's TF32 peak); with ~6,000 cycles
  // of fixed cost an item (measured by the bed's item clock: 3-7k for the smallest tiles), a phase costs its rounds x the item's time
  // R > 1: whole-row tiles by rounds x (nw + 48), the shapes that read best at R > 1 on the A100 (the cost model below picked smaller tiles there and lost);
  // R = 1: the model below. The choice spreads the step differently, never changes its arithmetic (the bed's dumps: byte-identical across them)
  // The grid path keeps a block's partial, its chain and three tree nodes for every output it owns, so its tiles give a warp at most
  // MVM_BATCH_GRID_MAXRB row blocks (WarpMap's RB): forward tiles up to 96 x 64, backward blocks up to 64 rows, the fused backward. One row
  // block a warp (48 x 32 tiles, 16-row blocks) streamed each operand up to 4x as often, and on the A100 at 108 members its steps cost
  // x1.84-2.05 of the own path's (measured on the A100, Nov 12).
  const int maxnw = grid ? 64 : 128;   // whole-row tiles (fm 6) up to 64 columns keep RB at 3 (mvm_batch_warp_rb(6, 4))
  double bestf = -1.0;
  if (d.R > 1) {
    long long bestr = -1;
    for (int nw = 16; nw <= maxnw; nw *= 2) {
      const int nt = (d.H + nw - 1) / nw, items = d.R * nt, rounds = (items + d.members - 1) / d.members;
      const long long cost = (long long)rounds * (nw + 48);
      if (bestr < 0 || cost < bestr) { bestr = cost; d.nw = nw; d.nt = nt; d.fm = 6; d.fr = 1; }
    }
    bestf = 0.0;
  }
  for (int fm = 1; fm <= 6 && d.R == 1; ++fm) {
    if (6 % fm || fm == 5 || fm == 4) continue;   // 1, 2, 3 or 6 row blocks an item
    for (int nw = 16; nw <= 128; nw *= 2) {
      if (grid && mvm_batch_warp_rb(fm, nw / 16) > MVM_BATCH_GRID_MAXRB) continue;
      const int nt = (d.H + nw - 1) / nw, items = d.R * (6 / fm) * nt, rounds = (items + d.members - 1) / d.members;
      const double bytes = 4.0 * (16 * fm + nw) * d.H, flop = 2.0 * 16 * fm * nw * d.H;
      const double item = 6000.0 + (bytes / 28.0 > flop / 300.0 ? bytes / 28.0 : flop / 300.0);
      const double cost = rounds * item;
      if (bestf < 0 || cost < bestf) { bestf = cost; d.nw = nw; d.nt = nt; d.fm = fm; d.fr = 6 / fm; }
    }
  }
  long long best = -1;
  for (int kw = 16; kw <= 64; kw *= 2) {   // the grid path too: a 64-row block's dX is 6 x 4 blocks of 16, RB 3 (mvm_batch_warp_rb(6, 4))
    const int kt = (d.H + kw - 1) / kw, items = d.R * kt, rounds = (items + d.members - 1) / d.members;
    const long long cost = (long long)rounds * (kw + 48);
    if (best < 0 || cost < best) { best = cost; d.kw = kw; d.kt = kt; }
  }
  // the backward: fused items unless they would leave more than half the gang idle, then the split tiles (the same arithmetic; the bed can
  // force either). Measured: split items at R = 8 each paid 3-7k fixed cycles, and h400 x8 ran at half the fused mode's speed; at R > 1 the
  // fused mode, as the A100 read it
  d.bsplit = !grid && d.R == 1 && 2 * d.R * d.kt < d.members ? 1 : 0;
  d.xfm = d.fm; d.xnw = d.nw; d.wfm = d.fm; d.wnw = d.nw;
  d.smem_floats = mvm_batch_smem(d.nw, d.fm, d.kw, d.xfm, d.xnw, d.wfm, d.wnw);
  return d.smem_floats;
}
