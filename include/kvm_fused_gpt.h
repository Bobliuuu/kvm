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
  // global buffers (device pointers, resolved by the executor from bundle buffer indices)
  const int32_t* ids_src; const int32_t* tgt_src;
  int32_t* step;                              // the run's step counter: read once, advanced once a step, written back at the end
  float* const* loss;                         // [steps]: each step's loss buffer (the bundle writes one a step), a device array of pointers
  float* p[KVM_FGPT_MAXP]; float* m[KVM_FGPT_MAXP]; float* v[KVM_FGPT_MAXP];   // parameters and AdamW moments, row-major, contiguous
  float* g[KVM_FGPT_MAXP];                    // the bundle's gradient buffers: written once at the end, so a reader after the run sees the last step's
  float* xch;                                 // exchange scratch in global memory: G slots of the largest partial, plus the gathered streams
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
