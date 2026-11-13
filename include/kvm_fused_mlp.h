#pragma once
// The fused MLP step: the descriptor one run's fused tasks read and its shared-memory layout (host and device; host-only callers include it with plain g++).
//
// A fused run trains one MLP (embedding, L GELU layers, linear head, softmax CE, SGD) as one program per row slice: `fmlp_slice` runs a slice's forward and backward with the weights and
// every activation in shared memory and writes its partial gradients to global memory; `fmlp_sgd` reduces the partials and updates a share of the parameters. Device side: kvm_fused_mlp.cuh.
#include <cstddef>

#ifdef __CUDACC__
#define KVM_FMLP_HD __host__ __device__
#else
#define KVM_FMLP_HD
#endif

#define KVM_FMLP_MAXL 6              // hidden layers
#define KVM_FMLP_MAX_SLICE_ROWS 32   // rows per slice (the mma fragment height is 16: a slice is a multiple of 16; the cross-entropy gives each row 8 of the 256 threads, so 32 rows at most)
#define KVM_FMLP_MAX_SLICES 8        // slices of one run, and gradient granules of one step (fmlp_sgd loads every granule's partial together in registers)
// A step's cross-row sums (weight and bias gradients, the embedding rows' gradients, the loss) are taken over granules of 16 rows: each slice writes one
// partial per granule it holds, and fmlp_sgd sums the T / 16 granules' partials in row order. The slice size then decides only how a step is spread over
// workers, never its arithmetic, so a run's result is the same whatever the pass chose for it (the MLP bench's gate 2 asserts four replicated copies of a
// run bitwise equal to the run alone, and the pass picks 32-row slices when 16-row ones would not all get a worker).
#define KVM_FMLP_GRANULE 16
#define KVM_FMLP_NPHASES 16          // phase timers of the slice body (--fuse-phases)
#define KVM_FMLP_MAX_GROUPS 16       // reduction owners of one run: a next-step slice waits for all of them, and one record holds 16 waits inline
#define KVM_FMLP_SMEM_FLOATS 38400   // shared memory a fused block may use (153.6 KB of the A100's 160 KB: the static worker's own arena stays out of it)

struct alignas(16) KvmFusedDesc {   // copied to shared memory with 16-byte cp.async: an array of them keeps every element 16-byte aligned
  // shape: rows per slice, vocabulary, embedding width, hidden layers, slices of this run, rows per step; groups: the reduction's owners (each updates 1 / groups of every parameter)
  int R, V, E, L, nsl, T, groups;
  int H[KVM_FMLP_MAXL];
  // row pitches in shared memory (floats). Activations and their GEMM outputs: HP[l] (layer l), XP (embedding), VP (logits), maxHP (the backward's activation gradient). Weights are only read, so their row needs no padding
  // past its own width (a GEMM reads a weight's pad columns only for its pad output columns): WP[l], WhP. The head's weights keep the columns up to the K of its backward GEMM (V rounded up to 8).
  int HP[KVM_FMLP_MAXL], XP, VP, maxHP, WP[KVM_FMLP_MAXL], WhP;
  // shared-memory offsets (floats from the arena's base): weights W, biases B, head Wh, activations, the slice's tokens and targets, cross-entropy scratch
  int W[KVM_FMLP_MAXL], B[KVM_FMLP_MAXL], Wh, X, Z[KVM_FMLP_MAXL], Y, D, Htmp, ids, tgt, ce, rl, total;
  // keepH: the forward keeps H[l] = gelu(Z[l]) of every layer but the last in its own buffer at Hk[l] (when the layout still fits the budget with them), so the backward's weight
  // gradients read it instead of recomputing gelu(Z[l]); 0: one buffer (Htmp) holds the current layer's H and the backward recomputes the ones below
  int keepH, Hk[KVM_FMLP_MAXL];
  float lr;
  unsigned long long inv_groups;     // ceil(2^32 / groups): fmlp_sgd splits every parameter over the owners with one multiply and shift (exact for rows < 2^16)
  // parameters in global memory, row-major and contiguous
  float* we, *w[KVM_FMLP_MAXL], *b[KVM_FMLP_MAXL], *wh;
  // per-granule partial gradients in global memory (T / KVM_FMLP_GRANULE copies each, in row order: slice sc's granule g is copy sc * R / 16 + g)
  float* ge, *gw[KVM_FMLP_MAXL], *gb[KVM_FMLP_MAXL], *gh;
  int gwld[KVM_FMLP_MAXL], ghld;     // row pitch of gw[l] and gh
  float* loss_part;                  // [T / KVM_FMLP_GRANULE]: each granule's sum of row losses
};

KVM_FMLP_HD inline constexpr int kvm_fmlp_pad8(int x) { return (x + 7) & ~7; }
KVM_FMLP_HD inline constexpr int kvm_fmlp_pad16(int x) { return (x + 15) & ~15; }
// A row pitch: at least the padded width, and congruent to 4 mod 16 elements, so the 8 rows of a warp's load land on different banks of shared memory (16 banks of 4 bytes).
KVM_FMLP_HD inline constexpr int kvm_fmlp_pitch(int width) { const int p = kvm_fmlp_pad16(width); return (p & ~15) + 4 >= p ? (p & ~15) + 4 : (p & ~15) + 20; }
// The pitch of a matrix that is only read: the smallest multiple of 4 at or above the width that spreads a warp's rows over the banks (congruent to 4 or 12 mod 16).
KVM_FMLP_HD inline constexpr int kvm_fmlp_read_pitch(int width) { int p = (width + 3) & ~3; while ((p & 7) != 4) p += 4; return p; }

// Floats of shared memory the descriptor occupies (it is copied to the arena's base at the start of each task).
KVM_FMLP_HD inline constexpr int kvm_fmlp_desc_floats() { return (int)((sizeof(KvmFusedDesc) + 63) / 64 * 16); }

// Fills the pitches and offsets of `d` from its shape (R, V, E, L, H); returns the floats of shared memory a slice needs. The tokens, targets and CE scratch sit after the matrices; the kept
// activations (keepH) after them, only when the whole still fits `budget`. The minimal layout (no kept activations) decides whether a shape fits at all: kvm_fmlp_min_floats.
inline int kvm_fmlp_layout(KvmFusedDesc& d, int budget = KVM_FMLP_SMEM_FLOATS) {
  int off = kvm_fmlp_desc_floats(), maxHP = 0;
  d.XP = kvm_fmlp_pitch(d.E);
  d.VP = kvm_fmlp_pitch(d.V);
  for (int l = 0; l < d.L; ++l) { d.HP[l] = kvm_fmlp_pitch(d.H[l]); d.WP[l] = kvm_fmlp_read_pitch(d.H[l]); maxHP = maxHP > d.HP[l] ? maxHP : d.HP[l]; }
  d.WhP = kvm_fmlp_read_pitch(kvm_fmlp_pad8(d.V));
  d.maxHP = maxHP > d.XP ? maxHP : d.XP;
  for (int l = 0; l < d.L; ++l) { d.W[l] = off; off += (l ? d.H[l - 1] : d.E) * d.WP[l]; d.B[l] = off; off += d.WP[l]; }
  d.Wh = off; off += d.H[d.L - 1] * d.WhP;
  d.X = off; off += d.R * d.XP;
  for (int l = 0; l < d.L; ++l) { d.Z[l] = off; off += d.R * d.HP[l]; }
  d.Y = off; off += d.R * d.VP;
  d.D = off; off += d.R * d.maxHP;
  d.Htmp = off; off += d.R * d.maxHP;
  d.ids = off; off += d.R;
  d.tgt = off; off += d.R;
  d.ce = off; off += d.R * 8;
  d.rl = off; off += d.R;
  d.keepH = 0;
  if (d.L > 1 && off + (d.L - 1) * d.R * d.maxHP <= budget) { d.keepH = 1; for (int l = 0; l + 1 < d.L; ++l) { d.Hk[l] = off; off += d.R * d.maxHP; } }
  d.total = off;
  d.inv_groups = (0x100000000ULL + d.groups - 1) / d.groups;
  return off;
}
