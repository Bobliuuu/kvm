#pragma once
// The grid MLP step (host and device; host-only callers include it with plain g++): an MLP training step too wide for the split step
// (mvm_fused_split.h, whose members each hold the whole input width of every layer: mlp1024 needs 99,292 floats a member at best) as one gang of
// P blocks over ONE copy of the model. Each weight matrix is cut into a Ka x Nb grid of blocks, its own grid a layer, and member a * Nb + b holds
// block (a, b): rows [k(a), k(a + 1)) and columns [n(b), n(b + 1)). Every member works on all T rows, so a block's gradient is complete on its
// member and SGD updates it in place: no owner phase. A step (device code: mvm_fused_grid.cuh):
//   * forward, layer 0 (K = E, not cut): member b of Nb0 computes Z = X0 W0[:, N_b] + b0 and H = gelu(Z) for every row, keeps gelu'(Z) and writes its
//     columns of H; a barrier.
//   * forward, hidden layer l > 0: member (a, b) writes the partial X[:, K_a] W_l[K_a, N_b] to L2; after a barrier it sums rows R_a of its columns
//     over the Ka partials in a order, adds the bias, keeps gelu'(Z) of its piece and writes its piece of H; a barrier.
//   * head (K = H cut into Kah blocks, N = V whole): partial logits to L2; after a barrier, member r takes rows r, r + P, ...: the sum of the Kah
//     partials, the cross-entropy, the row's logit gradient and loss term; a barrier.
//   * backward, from the head down: a layer's input gradient is a sum over its N blocks of dZ[:, N_b] W[K_a, N_b]^T. Each member writes its
//     partial (from the block before its update) and the next layer down sums its own piece of them in b order, times gelu'. The member's
//     dW = X[:, K_a]^T dZ[:, N_b] goes straight into its block (W -= lr dW), and a column's bias gradient is summed by each of its Ka members
//     alike, so their copies agree. Two barriers a hidden layer (the dZ piece, then the partials), one for the head and one for layer 0.
//   * the embedding: layer 0's members write their partials of dX0; after a barrier the owners of the vocabulary rows sum their tokens' rows
//     in token order, then member order, and update them; every member writes its blocks back; a last barrier ends the step.
// 4 L + 2 barriers a step, and every sum in a fixed order, so a run is deterministic and independent of the GEMMs' warp count.
// Blocks are whole granules of 16 (the GEMM's column block and a multiple of its k-step). A hidden width H that is a multiple of 8 but not of 16
// is cut as Hp = H + 8: the step loads the padding's weights and biases as zeros and never writes them back, so its units hold exact zeros
// through every sum and contribute nothing (gelu(0) = 0, and the next layer's padded rows are zero). V is padded to VP (a multiple of 16).
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define MVM_GRID_HD __host__ __device__
#else
#define MVM_GRID_HD
#endif

#define MVM_GRID_MAXL 4     // hidden layers
#define MVM_GRID_G 16       // granule of a block's rows and columns
#define MVM_GRID_MAXN 128   // widest column block (the GEMM: one 16-column block a warp, 8 warps)
#define MVM_GRID_MAXP 256   // members of a gang, at most (mvm_gang::cross itself takes any size)

struct alignas(16) MvmGridDesc {   // copied to shared memory with 16-byte cp.async
  int V, VP, E, H, Hp, L, T, P;    // vocabulary (padded to VP), embedding width, hidden width (the model's: its arrays' leading dimension) and it
                                   // padded to the granule (the cuts' and the step's scratch's), hidden layers, rows a step, the gang's members
  int nb0, ka, nb, kah;            // grids: layer 0 is 1 x nb0; hidden layers 1..L-1 are ka x nb; the head is kah x 1
  // shared-memory offsets of one member (floats from the arena's base; the descriptor itself sits at 0)
  int sW0, sW[MVM_GRID_MAXL], sWh, sb0, sb[MVM_GRID_MAXL], sg0, sg[MVM_GRID_MAXL], sA, sB, sints, total;
  int pA, pB, pW0, pW, pWh;        // pitches (floats) of the staging and weight blocks: width + 4, so ldmatrix rows fall in distinct banks
  float lr;
  // the virtual member (mvm_grid_layout_chunked; phys 0 is the resident member): its blocks, the regions of a chunk's weight block (sS[0]) and
  // biases (sSb[0]); sA2, sS[1] and sSb[1] unused (-1); gqs the floats of each block's gelu' factors in L2 (gq)
  int phys, sA2, sS[2], sSb[2], gqs;
  int G;                           // the chunked member (mvm_grid_layout_chunked): hidden blocks of a column a load carries, at most
  // the model, row-major as the bundle holds it: embedding [V][E], w[0] [E][H], w[l > 0] [H][H], b[l] [H], head [H][V]
  float* emb; float* w[MVM_GRID_MAXL]; float* b[MVM_GRID_MAXL]; float* wh;
  // global scratch (zeroed or not: every element is written before it is read): Y[l] [T][H] each layer's output; part [ka][T][H] the hidden
  // layers' forward partials; hpart [kah][T][VP] the head's; dlog [T][VP]; dyp [max(nb, 1)][T][H] input-gradient partials over a layer's N
  // blocks; dz [T][H]; dx0 [nb0][T][E]; rloss [T] each row's loss term
  float* Y[MVM_GRID_MAXL]; float* part; float* hpart; float* dlog; float* dyp; float* dz; float* dx0; float* rloss;
  unsigned* ctl;                   // the gang's control words (mvm_gang::cross: line_words(P) words, 128-byte aligned), zero before the first step
  float* gq;                       // the virtual member's: [phys][gqs] each block's gelu'(Z) of its chunk (mvm_grid_layout_chunked)
};

MVM_GRID_HD inline constexpr int mvm_grid_desc_floats() { return (int)((sizeof(MvmGridDesc) + 63) / 64 * 16); }
MVM_GRID_HD inline int mvm_grid_pad16(int x) { return (x + 15) & ~15; }
// Block i of n over D elements, in granules of g (D a multiple of g): its first element. Blocks differ by at most one granule.
MVM_GRID_HD inline int mvm_grid_off(int D, int n, int i, int g = MVM_GRID_G) { return (int)(((long long)i * (D / g)) / n) * g; }
MVM_GRID_HD inline int mvm_grid_widest(int D, int n, int g = MVM_GRID_G) { return ((D / g + n - 1) / n) * g; }

// The grids for P members and the member's shared-memory layout. Returns the floats a member needs, or -1 for a shape the step does not take.
// Layer 0 takes nb0 = min(P, Hp / 16) column blocks. The hidden layers take the fewest column blocks that keep a block at most 128 wide, and as
// many row blocks as the members then allow. The head takes kah = min(P, Hp / 16) row blocks.
MVM_GRID_HD inline int mvm_grid_layout(MvmGridDesc& d) {
  // V % 4: the head's rows are copied in 16-byte pieces; T = 96: the GEMMs over every row take 6 row blocks at compile time
  if (d.L < 1 || d.L > MVM_GRID_MAXL || d.H % 8 || d.E % MVM_GRID_G || d.E > MVM_GRID_MAXN || d.V < 1 || d.V % 4 || d.T != 96 || d.P < 1 || d.P > MVM_GRID_MAXP) return -1;
  d.VP = mvm_grid_pad16(d.V);
  d.Hp = mvm_grid_pad16(d.H);
  if (d.VP > MVM_GRID_MAXN) return -1;
  const int gran = d.Hp / MVM_GRID_G;
  d.nb0 = d.P < gran ? d.P : gran;
  d.nb = (d.Hp + MVM_GRID_MAXN - 1) / MVM_GRID_MAXN;
  d.ka = d.L > 1 ? (d.P / d.nb < gran ? d.P / d.nb : gran) : 0;
  if (d.L > 1 && d.ka < 1) return -1;
  d.kah = d.P < gran ? d.P : gran;
  const int w0 = mvm_grid_widest(d.Hp, d.nb0), wn = mvm_grid_widest(d.Hp, d.nb), wk = d.L > 1 ? mvm_grid_widest(d.Hp, d.ka) : 0,
            wkh = mvm_grid_widest(d.Hp, d.kah), rows = d.L > 1 ? (d.T + d.ka - 1) / d.ka : 0;
  d.pW0 = w0 + 4; d.pW = wn + 4; d.pWh = d.VP + 4;
  const int wA = d.E > wk ? (d.E > wkh ? d.E : wkh) : (wk > wkh ? wk : wkh), wB = wn > d.VP ? (wn > w0 ? wn : w0) : (d.VP > w0 ? d.VP : w0);
  d.pA = wA + 4; d.pB = wB + 4;
  int off = mvm_grid_desc_floats();
  auto take = [&off](int n) { const int o = off; off += (n + 3) & ~3; return o; };
  d.sW0 = take(d.E * d.pW0);
  for (int l = 1; l < d.L; ++l) d.sW[l] = take(wk * d.pW);
  d.sWh = take(wkh * d.pWh);
  d.sb0 = take(w0);
  for (int l = 1; l < d.L; ++l) d.sb[l] = take(wn);
  d.sg0 = take(d.T * w0);
  for (int l = 1; l < d.L; ++l) d.sg[l] = take(rows * wn);
  d.sA = take(d.T * d.pA);
  d.sB = take(d.T * d.pB);
  d.sints = take(2 * d.T);
  d.total = off;
  return off;
}

// The chunked virtual member (mvm_fused_grid_virtual.cuh: grid_run_chunked): the grid step of P members, its arithmetic and every sum in its
// order, run by phys < P physical members, each owning a contiguous chunk of every role family: the hidden layers' blocks in column-major order
// (i = hb ka + ha), layer 0's column blocks, the head's row blocks, the cross-entropy's rows and the embedding's rows. Chunk p of n is
// [p n / phys, (p + 1) n / phys). A chunk's hidden blocks of one column are consecutive in ha, so their K rows are contiguous: a group of up to G
// of them loads as one A slice and one weight block, its GEMMs run back to back from shared memory, and its reduce rows are contiguous.
MVM_GRID_HD inline int mvm_grid_chunk(int n, int phys, int p) { return (int)((long long)p * n / phys); }
struct MvmGridChunkShape {   // the widest a chunk can be in each family, and the regions' pitches
  int hmax, kc, wn, rows, c0max, w0, khmax, kh, pA, pW, pWh, pW0, pB, bias;
};
MVM_GRID_HD inline MvmGridChunkShape mvm_grid_chunk_shape(const MvmGridDesc& d, int G) {
  MvmGridChunkShape c;
  const int nh = d.L > 1 ? d.ka * d.nb : 0, wk = d.L > 1 ? mvm_grid_widest(d.Hp, d.ka) : 0;
  c.hmax = nh ? (nh + d.phys - 1) / d.phys : 0;
  c.kc = (G < c.hmax ? G : c.hmax) * wk;                                   // the K rows of a group of hidden blocks
  c.wn = d.L > 1 ? mvm_grid_widest(d.Hp, d.nb) : 0;
  c.rows = c.hmax * ((d.T + (d.ka > 0 ? d.ka : 1) - 1) / (d.ka > 0 ? d.ka : 1));   // a chunk's reduce rows of one column, at most
  c.c0max = (d.nb0 + d.phys - 1) / d.phys; c.w0 = c.c0max * mvm_grid_widest(d.Hp, d.nb0);
  c.khmax = (d.kah + d.phys - 1) / d.phys; c.kh = c.khmax * mvm_grid_widest(d.Hp, d.kah);
  int ka = c.kc > c.kh ? c.kc : c.kh; ka = ka > d.E ? ka : d.E;
  c.pA = ka + 4; c.pW = c.wn + 4; c.pWh = d.VP + 4; c.pW0 = c.w0 + 4;
  int b = c.wn > d.VP ? c.wn : d.VP; b = b > c.w0 ? b : c.w0; c.pB = b + 4;
  c.bias = c.wn > c.w0 ? c.wn : c.w0;
  return c;
}
// The chunked member's layout: the largest G (hidden blocks a load) whose regions fit `budget` floats. A: a group's A slice, the head's or X0;
// W: a group's weight block, the head's or layer 0's; B: the output or the other operand ([T][pB]); bias; the tokens. gelu'(Z) goes to L2:
// gqs floats a physical member (layer 0's chunk [T][w0], then each hidden layer's chunk rows [rows][wn] by column run). Returns the floats a
// member needs, or -1 for a shape or count it does not take.
MVM_GRID_HD inline int mvm_grid_layout_chunked(MvmGridDesc& d, int budget) {
  if (d.phys < 1 || d.phys >= d.P || mvm_grid_layout(d) < 0) return -1;
  for (int G = (d.L > 1 ? (d.ka * d.nb + d.phys - 1) / d.phys : 1); G >= 1; --G) {
    const MvmGridChunkShape c = mvm_grid_chunk_shape(d, G);
    int off = mvm_grid_desc_floats();
    auto take = [&off](int n) { const int o = off; off += (n + 3) & ~3; return o; };
    int w = c.kc * c.pW; w = w > c.kh * c.pWh ? w : c.kh * c.pWh; w = w > d.E * c.pW0 ? w : d.E * c.pW0;
    const int sA = take(d.T * c.pA), sW = take(w), sB = take(d.T * c.pB), sBias = take(c.bias), sints = take(2 * d.T + 1);
    if (off > budget && G > 1) continue;
    d.sW0 = d.sWh = d.sb0 = d.sg0 = -1;
    for (int l = 0; l < MVM_GRID_MAXL; ++l) d.sW[l] = d.sb[l] = d.sg[l] = -1;
    d.sA = sA; d.sA2 = -1; d.sS[0] = sW; d.sS[1] = -1; d.sB = sB; d.sSb[0] = sBias; d.sSb[1] = -1; d.sints = sints;
    d.G = G; d.gqs = d.T * c.w0 + (d.L > 1 ? (d.L - 1) * c.rows * c.wn : 0);
    d.total = off;
    return off;
  }
  return -1;
}
