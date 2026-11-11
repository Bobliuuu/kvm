#pragma once
// KVM fusion pass, host side: recognize whole MLP training steps in a closed bundle and rewrite each into per-slice fused tasks.
//
// ===== Why =====
// A training step of a small MLP is 51 tasks a step on the task graph, and each task pays a round trip to global memory for operands the previous task just wrote. A step fused
// into one program per row slice keeps the weights and activations in shared memory and only touches global memory to load the weights, write the partial gradients and reduce them
// (fused_mlp_bed.cu measures it).
//
// ===== What it decides, and from what =====
// The pass decides from the bundle's DAG and its op arguments alone: no model name, no row id, no width or depth constant. For each connected component of the DAG (an independent
// run) it parses the tasks in bundle order against one template (embedding -> L GEMMs with GELU -> head GEMM -> softmax CE -> the mirrored backward -> SGD on every parameter) and
// checks every buffer identity the template implies. A run whose steps do not all match stays exactly as it was: its tasks are copied through and the report says why.
//
// ===== Output =====
// A rewritten bundle (the replaced tasks of each matched step are gone; `fmlp_slice` and `fmlp_sgd` tasks take their place, and edges from outside a replaced set are redirected to
// its slices and groups) and, per fused run, what the device op needs. Host only: tests/static/test_fuse_pass.cpp drives it with plain g++ on real bundles.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "kvm_fused_mlp.h"
#include "kvm_static.h"
#include "kvm_task_abi.h"

#define KVM_FUSE_MAXL KVM_FMLP_MAXL   // hidden layers the template accepts

struct KvmFuseBuf { int rows = 0, cols = 0, ld = 0, dtype = 0; };   // what the pass needs of a bundle buffer (a parameter must be contiguous: ld == cols)

struct KvmFuseOps {   // operator ids (from the registry, by name); a missing op leaves its id at -1 and the template never matches it
  int noop = -1, select_window = -1, embedding_fwd = -1, matmul_tile = -1, softmax_ce_fwd = -1, softmax_ce_reduce = -1, softmax_ce_bwd = -1, softmax_ce_fused = -1, gelu_bwd = -1, colsum = -1, embedding_bwd = -1, sgd = -1, sgd_range = -1;
  int fmlp_slice = -1, fmlp_sgd = -1;   // the fused ops the pass emits
};

struct KvmFuseOptions {
  int smem_floats = KVM_FMLP_SMEM_FLOATS;   // shared memory a fused block may use
  int workers = 108;                  // blocks the run has: more slices per run shorten each slice, but every slice needs a worker of its own
  std::vector<int> keep;              // buffers read after the run (a dump): a step whose interior holds one is not fused
};
inline constexpr int kvm_fuse_slice_choices[] = {16, 32};   // rows per slice, multiples of 16 (the mma fragment height), smallest first, at most KVM_FMLP_MAX_SLICE_ROWS
static_assert(kvm_fuse_slice_choices[1] <= KVM_FMLP_MAX_SLICE_ROWS, "a slice choice exceeds what the fused body handles");

// One fused run (what the device op needs, as bundle buffer indices; the executor resolves them to pointers).
struct KvmFuseRun {
  int L = 0, E = 0, V = 0, T = 0, slice_rows = 0, nsl = 0, groups = 0;   // groups: the fmlp_sgd owners of each step
  int smem_floats = KVM_FMLP_SMEM_FLOATS;   // the shared-memory budget the pass sized this run for: the device layout (kept activations) is computed against the same one
  float lr = 0.f;
  std::array<int, KVM_FUSE_MAXL> H{}, w{}, b{};
  int wte = -1, head = -1;
  int steps = 0;
};

struct KvmFuseResult {
  std::vector<KvmTaskDesc> tasks;
  std::vector<int32_t> succ, args;
  std::vector<KvmFuseRun> runs;
  std::vector<std::string> report;     // one line per component (and per reason it did not fire)
  std::vector<std::string> warnings;   // the unchanged runs whose steps matched the template's model (embedding, GELU layers, head) and then
                                       // missed: a run the fused step was written for, falling silently to the per-op path (a lowering the
                                       // matcher does not read yet). The caller logs them as warnings; a shape the template refuses is not one.
  int fused_runs = 0, unchanged_runs = 0, replaced_tasks = 0, new_tasks = 0;
  std::set<int> interior;              // buffers no longer written by a fused run (its activations and gradients): a reader of them after the run reads their initial values
};

namespace kvm_fuse {

constexpr int F_TRUNC = 256, F_DGELU = 512;   // matmul_tile flags: operands truncated to TF32 by the tensor core (a numerics preset of the tile; the fused GEMM has its own); out *= gelu'(pre) in the tile's epilogue

struct Gemm { int a = -1, b = -1, out = -1, bias = -1, res = -1, pre = -1, M = 0, N = 0, K = 0, flags = 0; std::vector<int> tasks; };

// Rows of a fragment grid: the width a block of m16 rows or n8 columns must be padded to.
inline int pad16(int x) { return (x + 15) & ~15; }
inline int pad8(int x) { return (x + 7) & ~7; }

class Parser {
 public:
  Parser(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& args, const std::vector<KvmFuseBuf>& bufs, const KvmFuseOps& ops)
      : tasks_(tasks), args_(args), bufs_(bufs), ops_(ops) {}

  // Parses one step starting at comp[pos]; on success advances pos past it and fills `out`. On failure leaves `why` set to the first mismatch and returns false.
  struct Step {
    int ids = -1, tgt = -1, loss = -1;
    int E = 0, V = 0, T = 0, L = 0;
    std::array<int, KVM_FUSE_MAXL> H{}, w{}, b{};
    int wte = -1, head = -1;
    float lr = 0.f;
    std::vector<int> replaced;     // every task of this step the fused tasks take over
    std::set<int> interior;        // buffers only this step's replaced tasks may touch
    int barrier_end = -1;          // the noop that ends the step, if the bundle has one: kept as the join into the next step
    bool model = false;            // its embedding, GELU layers and head matched the template (set before the loss is read)
  };

  bool parse(const std::vector<int>& comp, size_t& pos, Step& s, std::string& why) const {
    size_t c = pos;
    // tasks before the embedding (the step's data windows) are not part of the fused region; skip them
    while (c < comp.size() && op(comp[c]) != ops_.embedding_fwd) { if (!is_prefix_task(comp[c])) { why = "step " + std::to_string(c) + ": task before the embedding is not a data window (" + std::to_string(op(comp[c])) + ")"; return false; } ++c; }
    if (c == comp.size()) { why = "no embedding task left"; return false; }
    // ---- embedding: ids, wte, wpe, out, r0, r1, T, d
    const int* a = argp(comp[c]);
    const int x = a[3];
    s.ids = a[0]; s.wte = a[1]; s.E = a[7]; s.T = a[5];
    if (a[2] != -1 || a[4] != 0) { why = "embedding with position table or partial rows"; return false; }
    if (!buf_is(s.wte, -1, s.E) || !buf_is(x, s.T, s.E)) { why = "embedding buffer shapes"; return false; }
    s.replaced.push_back(comp[c]); s.interior.insert(x); ++c;
    // ---- forward GEMMs with GELU (flag 8): layer l multiplies the previous activation by w[l], adds bias b[l], writes the pre-GELU copy to pre
    int prev = x, K = s.E;
    std::array<int, KVM_FUSE_MAXL> z{}, pre{};
    Gemm g;
    while (true) {
      if (c >= comp.size() || op(comp[c]) != ops_.matmul_tile) { why = "expected a GEMM after the embedding or a forward layer"; return false; }
      if (!read_gemm(comp, c, g, why)) return false;
      if (g.flags != 8) break;     // the head GEMM (flags 0) ends the forward layers; `g` holds it
      if (s.L >= KVM_FUSE_MAXL) { why = "more hidden layers than the template holds"; return false; }
      if (g.a != prev || g.M != s.T || g.K != K || g.res != -1) { why = "forward GEMM " + std::to_string(s.L) + " does not chain from its input"; return false; }
      if (!buf_is(g.b, g.K, g.N) || !buf_is(g.bias, 1, g.N) || !buf_is(g.out, s.T, g.N) || !buf_is(g.pre, s.T, g.N)) { why = "forward GEMM buffer shapes"; return false; }
      const int l = s.L++;
      s.H[l] = g.N; s.w[l] = g.b; s.b[l] = g.bias; z[l] = g.out; pre[l] = g.pre;
      s.interior.insert(g.out); s.interior.insert(g.pre);
      s.replaced.insert(s.replaced.end(), g.tasks.begin(), g.tasks.end());
      prev = g.out; K = g.N;
    }
    if (s.L == 0) { why = "no hidden layer (flag 8) before the head"; return false; }
    // ---- head GEMM
    if (g.flags != 0 || g.a != prev || g.M != s.T || g.K != K || g.res != -1 || g.bias != -1 || g.pre != -1) { why = "the head GEMM does not chain from the last layer"; return false; }
    s.V = g.N; s.head = g.b;
    if (!buf_is(s.head, K, s.V) || !buf_is(g.out, s.T, s.V)) { why = "head GEMM buffer shapes"; return false; }
    const int Y = g.out; s.interior.insert(Y);
    s.replaced.insert(s.replaced.end(), g.tasks.begin(), g.tasks.end());
    s.model = true;   // the step's model matched: a mismatch from here on is a near miss (KvmFuseResult::warnings)
    // ---- softmax CE: fwd (logits, tgt, rowloss, lse, r0, r1, V, ignore), reduce (rowloss, tgt, loss, N, ignore), bwd (logits, tgt, lse, loss, dlogits, r0, r1, V, ignore, scale).
    // A tiling may lower the fwd, bwd and fused ops in row blocks (the frontend's Tiling.ce_rows: rows [r0, r1) a task, each row's arithmetic
    // unchanged); the reduce is one task over every row. `row_blocks` reads such a run of tasks; the first block's arguments stand for all.
    int dY = -1;
    if (c < comp.size() && op(comp[c]) == ops_.softmax_ce_fused) {       // softmax_ce_fused(logits, tgt, rowloss, lse, dlogits, r0, r1, V, ignore, N, scale) blocks, then the reduce
      const int nf = row_blocks(comp, c, ops_.softmax_ce_fused, 11, 5, s.T);
      if (nf == 0) { why = "the fused softmax CE's row blocks do not cover the step's rows once, in order, with one set of arguments"; return false; }
      if (c + nf >= comp.size() || op(comp[c + nf]) != ops_.softmax_ce_reduce) { why = "expected the reduce after the fused softmax CE"; return false; }
      const int* f = argp(comp[c]), *r = argp(comp[c + nf]);
      s.tgt = f[1]; s.loss = r[2]; dY = f[4];
      if (f[0] != Y || f[7] != s.V || f[8] != -1 || f[9] != s.T || afloat(f[10]) != 1.f || r[0] != f[2] || r[1] != s.tgt || r[3] != s.T || r[4] != -1) { why = "fused softmax CE shape"; return false; }
      s.interior.insert(f[2]); s.interior.insert(f[3]); s.interior.insert(dY);
      for (int i = 0; i <= nf; ++i) s.replaced.push_back(comp[c + i]);
      c += nf + 1;
    } else {
      const int nw = row_blocks(comp, c, ops_.softmax_ce_fwd, 8, 4, s.T);
      const size_t cr = c + nw;   // the reduce
      const int nb = nw ? row_blocks(comp, cr + 1, ops_.softmax_ce_bwd, 10, 5, s.T) : 0;
      if (nw == 0 || cr >= comp.size() || op(comp[cr]) != ops_.softmax_ce_reduce || nb == 0) { why = "expected softmax CE after the head (fwd row blocks, the reduce, bwd row blocks)"; return false; }
      const int* f = argp(comp[c]), *r = argp(comp[cr]), *bwd = argp(comp[cr + 1]);
      s.tgt = f[1]; s.loss = r[2]; dY = bwd[4];
      if (f[0] != Y || f[6] != s.V || f[7] != -1 || r[0] != f[2] || r[1] != s.tgt || r[3] != s.T || r[4] != -1) { why = "softmax CE fwd or reduce shape"; return false; }
      if (bwd[0] != Y || bwd[1] != s.tgt || bwd[2] != f[3] || bwd[3] != s.loss || bwd[7] != s.V || bwd[8] != -1 || afloat(bwd[9]) != 1.f) { why = "softmax CE bwd shape"; return false; }
      s.interior.insert(f[2]); s.interior.insert(f[3]); s.interior.insert(dY);
      for (size_t i = c; i < cr + 1 + (size_t)nb; ++i) s.replaced.push_back(comp[i]);
      c = cr + 1 + nb;
    }
    if (!buf_is(s.loss, 1, 2)) { why = "loss buffer shape"; return false; }
    // ---- backward: the head's weight gradient (A^T dY), the gradient of the last activation (dY W^T)
    if (!read_gemm(comp, c, g, why)) return false;
    if (g.flags != 1 || g.a != z[s.L - 1] || g.b != dY || g.M != K || g.N != s.V || g.K != s.T) { why = "head weight gradient GEMM"; return false; }
    int dW_head = g.out; s.interior.insert(g.out); s.replaced.insert(s.replaced.end(), g.tasks.begin(), g.tasks.end());
    if (!read_gemm(comp, c, g, why)) return false;
    if ((g.flags != 2 && g.flags != (2 | kvm_fuse::F_DGELU)) || g.a != dY || g.b != s.head || g.M != s.T || g.N != K || g.K != s.V) { why = "head activation gradient GEMM"; return false; }
    if ((g.flags & kvm_fuse::F_DGELU) && g.pre != pre[s.L - 1]) { why = "the head's gelu-backward epilogue does not read the last layer's pre-activation"; return false; }
    int D = g.out; bool D_has_gelu = (g.flags & kvm_fuse::F_DGELU) != 0;     // D is dZ[L-1] already when the epilogue applied gelu'
    s.interior.insert(g.out); s.replaced.insert(s.replaced.end(), g.tasks.begin(), g.tasks.end());
    std::array<int, KVM_FUSE_MAXL> dW{}, db{};
    for (int l = s.L - 1; l >= 0; --l) {
      const int Hl = s.H[l], Kl = l ? s.H[l - 1] : s.E, Al = l ? z[l - 1] : x;
      // gelu_bwd (x, dy, dx, e0, e1, acc), unless the GEMM before it applied gelu' in its epilogue
      int dZ = D;
      if (!D_has_gelu) {
        if (c >= comp.size() || op(comp[c]) != ops_.gelu_bwd) { why = "expected gelu_bwd of layer " + std::to_string(l); return false; }
        const int* gb = argp(comp[c]);
        if (gb[0] != pre[l] || gb[1] != D || gb[3] != 0 || gb[4] != s.T * Hl || gb[5] != 0) { why = "gelu_bwd of layer " + std::to_string(l) + " does not chain"; return false; }
        dZ = gb[2]; s.replaced.push_back(comp[c]); s.interior.insert(dZ); ++c;
      }
      // colsum (x, out, R, c0, c1, acc): the bias gradient
      if (c >= comp.size() || op(comp[c]) != ops_.colsum) { why = "expected colsum of layer " + std::to_string(l); return false; }
      const int* cs = argp(comp[c]);
      if (cs[0] != dZ || cs[2] != s.T || cs[3] != 0 || cs[4] != Hl || cs[5] != 0) { why = "colsum of layer " + std::to_string(l); return false; }
      db[l] = cs[1]; s.replaced.push_back(comp[c]); s.interior.insert(cs[1]); ++c;
      // the layer's weight gradient (A^T dZ) and the next activation gradient (dZ W^T)
      if (!read_gemm(comp, c, g, why)) return false;
      if (g.flags != 1 || g.a != Al || g.b != dZ || g.M != Kl || g.N != Hl || g.K != s.T) { why = "weight gradient GEMM of layer " + std::to_string(l); return false; }
      dW[l] = g.out; s.interior.insert(g.out); s.replaced.insert(s.replaced.end(), g.tasks.begin(), g.tasks.end());
      if (!read_gemm(comp, c, g, why)) return false;
      if ((g.flags != 2 && !(l > 0 && g.flags == (2 | kvm_fuse::F_DGELU))) || g.a != dZ || g.b != s.w[l] || g.M != s.T || g.N != Kl || g.K != Hl) { why = "activation gradient GEMM of layer " + std::to_string(l); return false; }
      if ((g.flags & kvm_fuse::F_DGELU) && g.pre != pre[l - 1]) { why = "a gelu-backward epilogue does not read the previous layer's pre-activation"; return false; }
      D = g.out; D_has_gelu = (g.flags & kvm_fuse::F_DGELU) != 0;
      s.interior.insert(g.out); s.replaced.insert(s.replaced.end(), g.tasks.begin(), g.tasks.end());
    }
    // ---- embedding_bwd (ids, dout, dtab, N, mod, v0, v1, d, acc)
    if (c >= comp.size() || op(comp[c]) != ops_.embedding_bwd) { why = "expected embedding_bwd"; return false; }
    const int* eb = argp(comp[c]);
    if (eb[0] != s.ids || eb[1] != D || eb[3] != s.T || eb[4] != 0 || eb[5] != 0 || eb[6] != s.V || eb[7] != s.E || eb[8] != 0) { why = "embedding_bwd shape"; return false; }
    const int dE = eb[2]; s.replaced.push_back(comp[c]); s.interior.insert(dE); ++c;
    // ---- the barrier before the updates, one SGD per parameter, the barrier that ends the step
    if (c < comp.size() && op(comp[c]) == ops_.noop) { s.replaced.push_back(comp[c]); ++c; }       // the barrier before the updates (a pass that dissolves ordering noops removes it)
    std::map<int, int> want;
    want[s.wte] = dE; want[s.head] = dW_head;
    for (int l = 0; l < s.L; ++l) { want[s.w[l]] = dW[l]; want[s.b[l]] = db[l]; }
    for (size_t n = want.size(); n > 0; --n, ++c) {
      if (c >= comp.size() || (op(comp[c]) != ops_.sgd && op(comp[c]) != ops_.sgd_range)) { why = "expected an SGD update"; return false; }
      const int* u = argp(comp[c]);
      auto it = want.find(u[0]);
      if (it == want.end() || it->second != u[1]) { why = "SGD does not pair a parameter with its gradient"; return false; }
      const int whole = bufs_[u[0]].rows * bufs_[u[0]].cols;         // sgd (w, g, lr, n), or sgd_range (w, g, lr, e0, e1) over the whole parameter
      if (op(comp[c]) == ops_.sgd ? u[3] != whole : (u[3] != 0 || u[4] != whole)) { why = "SGD does not cover its whole parameter"; return false; }
      if (s.lr == 0.f) s.lr = afloat(u[2]); else if (afloat(u[2]) != s.lr) { why = "SGD learning rates differ"; return false; }
      want.erase(it);
      s.replaced.push_back(comp[c]);
    }
    if (c < comp.size() && op(comp[c]) == ops_.noop) { s.barrier_end = comp[c]; ++c; }           // the barrier that ends the step (absent if ordering noops were dissolved: the next step's tasks hang directly off the updates)
    pos = c;
    return true;
  }

 private:
  int op(int t) const { return tasks_[t].op_id; }
  const int* argp(int t) const { return args_.data() + tasks_[t].args_off; }
  static float afloat(int32_t w) { float f; memcpy(&f, &w, 4); return f; }
  bool is_prefix_task(int t) const { return op(t) == ops_.select_window; }   // the data windows that feed a step stay as they are
  bool buf_is(int b, int rows, int cols) const {
    if (b < 0 || b >= (int)bufs_.size() || bufs_[b].dtype != 0 || (bufs_[b].ld != bufs_[b].cols && bufs_[b].rows > 1)) return false;
    return (rows < 0 || bufs_[b].rows == rows) && bufs_[b].cols == cols;
  }
  // A loss op lowered as one task or over consecutive row blocks: the tasks of op `id` from comp[c] on, rows [r0, r1) in argument words ri
  // and ri + 1. They must start at row 0, each block where the one before ended, and end at row T, with at least `arity` argument words,
  // and every word but the rows equal to the first block's. Returns how many tasks that is; 0 when comp[c] is not that op or its blocks
  // are not such a cover (a gap, an overlap, a short or long cover, another argument).
  int row_blocks(const std::vector<int>& comp, size_t c, int id, int arity, int ri, int T) const {
    if (c >= comp.size() || op(comp[c]) != id) return 0;
    const int words = tasks_[comp[c]].args_words;
    if (words < arity || ri + 1 >= words) return 0;
    const int* a0 = argp(comp[c]);
    int k = 0;
    for (int next = 0; next < T; ++k) {
      if (c + k >= comp.size() || op(comp[c + k]) != id || tasks_[comp[c + k]].args_words != words) return 0;
      const int* a = argp(comp[c + k]);
      for (int i = 0; i < words; ++i) if (i != ri && i != ri + 1 && a[i] != a0[i]) return 0;
      if (a[ri] != next || a[ri + 1] <= next || a[ri + 1] > T) return 0;
      next = a[ri + 1];
    }
    return k;
  }
  // Consumes the contiguous tiles of one GEMM at comp[c]. All tiles must name the same GEMM and cover its M x N grid exactly once, with the base 64 x 64 tile.
  bool read_gemm(const std::vector<int>& comp, size_t& c, Gemm& g, std::string& why) const {
    g = Gemm{};
    if (c >= comp.size() || op(comp[c]) != ops_.matmul_tile) { why = "expected a GEMM"; return false; }
    std::set<std::pair<int, int>> tiles;
    const int* a0 = argp(comp[c]);
    const int fl = a0[11], M = a0[8], N = a0[9];
    const size_t first = c;
    for (; c < comp.size() && op(comp[c]) == ops_.matmul_tile; ++c) {
      const int* a = argp(comp[c]);
      if (a[0] != a0[0] || a[1] != a0[1] || a[2] != a0[2] || a[3] != a0[3] || a[4] != a0[4] || a[5] != a0[5] || a[8] != M || a[9] != N || a[10] != a0[10] || a[11] != fl) break;   // the next GEMM starts here
      if (afloat(a[12]) != 1.f) { why = "a GEMM with alpha != 1"; return false; }
      if (!tiles.insert({a[6], a[7]}).second) { why = "a tile listed twice in a GEMM"; return false; }
      g.tasks.push_back(comp[c]);
    }
    const int fl_kept = fl & ~kvm_fuse::F_TRUNC;
    if (fl_kept & ~(15 | kvm_fuse::F_DGELU)) { why = "a GEMM with flags " + std::to_string(fl) + " the template does not hold"; return false; }
    // the 64 x 64 grid of the op, clipped to the GEMM's edge by the op itself
    for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < N; n0 += 64) if (!tiles.count({m0, n0})) { why = "a GEMM's tile set does not cover its grid"; return false; }
    if ((int)tiles.size() != ((M + 63) / 64) * ((N + 63) / 64)) { why = "a GEMM has tiles outside its 64 x 64 grid"; return false; }
    (void)first;
    g.a = a0[0]; g.b = a0[1]; g.out = a0[2]; g.bias = a0[3]; g.res = a0[4]; g.pre = a0[5]; g.M = M; g.N = N; g.K = a0[10]; g.flags = fl_kept;
    return true;
  }
  const std::vector<KvmTaskDesc>& tasks_;
  const std::vector<int32_t>& args_;
  const std::vector<KvmFuseBuf>& bufs_;
  KvmFuseOps ops_;
};

}  // namespace kvm_fuse

// Connected components of the DAG (successor CSR plus version-variable edges): independent runs. Tasks of a component stay in bundle order.
inline std::vector<std::vector<int>> kvm_fuse_components(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& succ) {
  const int nt = (int)tasks.size();
  std::vector<int> root(nt);
  for (int i = 0; i < nt; ++i) root[i] = i;
  std::function<int(int)> find = [&](int x) { while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; } return x; };
  auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) root[b] = a; };
  for (int i = 0; i < nt; ++i) for (int k = 0; k < tasks[i].succ_cnt; ++k) unite(i, succ[tasks[i].succ_off + k]);
  // version variables join their publishers and waiters (a run's chain across steps)
  std::map<int, int> rep;
  for (int i = 0; i < nt; ++i) for (int v : {tasks[i].pub_var, tasks[i].wait_var}) if (v >= 0) { auto it = rep.find(v); if (it == rep.end()) rep[v] = i; else unite(i, it->second); }
  std::map<int, std::vector<int>> by_root;
  for (int i = 0; i < nt; ++i) by_root[find(i)].push_back(i);
  std::vector<std::vector<int>> comps;
  for (auto& kv : by_root) comps.push_back(std::move(kv.second));
  return comps;
}

// The shared-memory floats a fused slice of R rows needs for the shape (the device op's own layout code).
inline int kvm_fuse_layout_floats(int L, int E, int V, const std::array<int, KVM_FUSE_MAXL>& H, int T, int R) {
  KvmFusedDesc d{};
  d.R = R; d.V = V; d.E = E; d.L = L; d.T = T; d.nsl = T / R; d.groups = 1;
  for (int l = 0; l < L; ++l) d.H[l] = H[l];
  return kvm_fmlp_layout(d, 0);   // the minimal layout: kept activations are added only where they still fit
}

// The pass: rewrite every run (connected component) whose steps all match the MLP template into fused tasks; the others stay as they are. `op_of` gives each op's signature (to find its buffer arguments).
inline KvmFuseResult kvm_fuse_mlp(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& succ, const std::vector<int32_t>& args, const std::vector<KvmFuseBuf>& bufs, const KvmFuseOps& ops, const KvmSOpLookup& op_of, const KvmFuseOptions& opts = {}) {
  using Step = kvm_fuse::Parser::Step;
  KvmFuseResult R;
  const int nt = (int)tasks.size();
  // every reference to each buffer, so an interior buffer can be checked to be touched by its step alone
  std::vector<std::vector<int>> refs(bufs.size());
  for (int i = 0; i < nt; ++i) {
    const KvmSOpInfo o = op_of(tasks[i].op_id);
    for (int j = 0; j < tasks[i].args_words && j < o.arity; ++j) {
      const int ty = (int)((o.types >> (3 * j)) & 7);
      const int v = args[tasks[i].args_off + j];
      if ((ty == 0 || ty == 3) && v >= 0 && v < (int)bufs.size()) refs[v].push_back(i);
    }
  }
  struct Plan { KvmFuseRun run; std::vector<Step> steps; };
  std::vector<Plan> plans;
  std::vector<int> plan_comp;   // the component each plan came from (for the report)
  const kvm_fuse::Parser parser(tasks, args, bufs, ops);
  const auto comps = kvm_fuse_components(tasks, succ);
  // Why the template cannot take a step of this model, from its shapes alone (empty: it can). A refusal for this reason is by design.
  const auto uncovered = [&](const Step& s) -> std::string {
    for (int l = 0; l < s.L; ++l) if (s.H[l] % 8 || s.H[l] > 128) return "layer width not a multiple of 8 up to 128";
    if (s.E % 8 || s.E == 0) return "embedding width not a multiple of 8";
    if (s.V % 4 || s.V == 0 || s.V > 128) return "vocabulary not a multiple of 4 up to 128";
    bool any = false;
    for (int r : kvm_fuse_slice_choices) any |= s.T % r == 0 && r <= s.T && s.T / r <= KVM_FMLP_MAX_SLICES && kvm_fuse_layout_floats(s.L, s.E, s.V, s.H, s.T, r) <= opts.smem_floats;
    if (!any) return "no slice size fits (" + std::to_string(kvm_fuse_layout_floats(s.L, s.E, s.V, s.H, s.T, kvm_fuse_slice_choices[0])) + " floats of shared memory at the smallest slice, at most " + std::to_string(KVM_FMLP_MAX_SLICES) + " slices, limit " + std::to_string(opts.smem_floats) + ")";
    return "";
  };
  for (size_t ci = 0; ci < comps.size(); ++ci) {
    const std::vector<int>& comp = comps[ci];
    Plan pl; std::string why;
    size_t pos = 0;
    // A refusal of a run the fused step was written for (a step matched the template's model, the embedding, the GELU layers and the head,
    // in shapes the template covers) is a silent fall to the per-op path and goes to KvmFuseResult::warnings. A shape the template leaves
    // out, or a buffer the caller keeps, is a refusal by design, whichever check meets it first.
    bool near = false, intended = false;
    while (pos < comp.size()) {
      Step st;
      if (!parser.parse(comp, pos, st, why)) {
        const Step* m = st.model ? &st : pl.steps.empty() ? nullptr : &pl.steps[0];   // a step whose model matched, if any
        near = m && uncovered(*m).empty();
        why = "step " + std::to_string(pl.steps.size()) + ": " + why; break;
      }
      pl.steps.push_back(std::move(st));
    }
    if (why.empty() && pl.steps.empty()) why = "no training step found";
    if (why.empty()) near = uncovered(pl.steps[0]).empty();   // every step matched: a near miss only in shapes the template covers
    // Every step's closing barrier but the last is replaced too: the fused tasks' own edges order the steps, and a join in the middle would sit on the critical path. The last one stays: it publishes the run's final version.
    if (why.empty()) for (size_t si = 0; si + 1 < pl.steps.size(); ++si) if (pl.steps[si].barrier_end >= 0) { pl.steps[si].replaced.push_back(pl.steps[si].barrier_end); pl.steps[si].barrier_end = -1; }
    if (why.empty()) {
      const Step& s0 = pl.steps[0];
      for (const Step& s : pl.steps) if (s.E != s0.E || s.V != s0.V || s.T != s0.T || s.L != s0.L || s.H != s0.H || s.w != s0.w || s.b != s0.b || s.wte != s0.wte || s.head != s0.head || s.lr != s0.lr) { why = "steps do not share one model"; break; }
    }
    if (why.empty()) {   // a step's interior buffers are written and read by its replaced tasks alone: fusing them away must not be visible to anyone else
      for (size_t si = 0; si < pl.steps.size() && why.empty(); ++si) {
        const Step& s = pl.steps[si];
        const std::set<int> mine(s.replaced.begin(), s.replaced.end());
        for (int b : s.interior) { for (int t : refs[b]) if (!mine.count(t)) { why = "step " + std::to_string(si) + ": interior buffer " + std::to_string(b) + " is used outside the step"; break; } if (!why.empty()) break; }
      }
    }
    if (why.empty()) {   // the shapes the template covers (a refusal here is by design)
      const Step& s = pl.steps[0];
      why = uncovered(s);
      if (why.empty()) { pl.run.L = s.L; pl.run.E = s.E; pl.run.V = s.V; pl.run.T = s.T; pl.run.lr = s.lr; pl.run.H = s.H; pl.run.w = s.w; pl.run.b = s.b; pl.run.wte = s.wte; pl.run.head = s.head; pl.run.steps = (int)pl.steps.size(); pl.run.smem_floats = opts.smem_floats; }
      else intended = true;
    }
    if (why.empty()) {   // a buffer the caller reads after the run may not be interior
      for (size_t si = 0; si < pl.steps.size() && why.empty(); ++si) for (int b : opts.keep) if (pl.steps[si].interior.count(b)) { why = "buffer " + std::to_string(b) + " is read after the run and interior to step " + std::to_string(si); break; }
      intended = !why.empty();
    }
    if (!why.empty()) {
      ++R.unchanged_runs; R.report.push_back("component " + std::to_string(ci) + " (" + std::to_string(comp.size()) + " tasks): unchanged: " + why);
      if (near && !intended) R.warnings.push_back("component " + std::to_string(ci) + " is an MLP training run the fused step was written for, but stays unfused (" + why + "): it runs per op, correct and slower");
      continue;
    }
    ++R.fused_runs; plans.push_back(std::move(pl)); plan_comp.push_back((int)ci);
  }
  if (plans.empty()) { R.tasks = tasks; R.succ = succ; R.args = args; return R; }
  // ---- slice size: one size for all runs, the smallest that fits every run and gives each slice a worker; if none does, the largest that fits every run (slices then share workers);
  // runs that fit no common size each take their own smallest
  {
    auto fits = [&](const KvmFuseRun& r, int rows) { return r.T % rows == 0 && rows <= r.T && r.T / rows <= KVM_FMLP_MAX_SLICES && kvm_fuse_layout_floats(r.L, r.E, r.V, r.H, r.T, rows) <= opts.smem_floats; };
    int common = 0, widest_common = 0;
    for (int rows : kvm_fuse_slice_choices) {
      bool all = true; int total = 0;
      for (const Plan& p : plans) { all &= fits(p.run, rows); total += p.run.T / rows; }
      if (!all) continue;
      widest_common = rows;
      if (total <= opts.workers) { common = rows; break; }
    }
    if (!common) common = widest_common;
    for (Plan& p : plans) {
      if (common) { p.run.slice_rows = common; continue; }
      for (int rows : kvm_fuse_slice_choices) if (fits(p.run, rows)) { p.run.slice_rows = rows; break; }
    }
    for (size_t i = 0; i < plans.size(); ++i) {
      KvmFuseRun& r = plans[i].run; r.nsl = r.T / r.slice_rows;
      // the reduction runs after every slice of a step has finished: the slices' workers are free for it, and any workers no slice uses take a share too (the reduce is bandwidth-bound per SM)
      { int total = 0; for (const Plan& p : plans) total += p.run.T / p.run.slice_rows;
        const int spare = std::max(0, opts.workers - total) / (int)plans.size();
        r.groups = std::min(KVM_FMLP_MAX_GROUPS, std::max(r.nsl, r.nsl + spare)); }
      std::string shape = std::to_string(r.E);
      for (int l = 0; l < r.L; ++l) shape += "x" + std::to_string(r.H[l]);
      R.report.push_back("component " + std::to_string(plan_comp[i]) + ": fused: " + std::to_string(plans[i].steps.size()) + " steps, T=" + std::to_string(r.T) + " V=" + std::to_string(r.V) + " widths " + shape + ", " + std::to_string(r.nsl) + " slices of " + std::to_string(r.slice_rows) + " rows, " + std::to_string(r.groups) + " reduction owners");
    }
  }
  // ---- emission: which step (of which plan) replaces each old task
  struct Node { int plan, step; };
  std::vector<int> run_of(nt, -1), step_of(nt, -1);
  for (size_t p = 0; p < plans.size(); ++p) for (size_t si = 0; si < plans[p].steps.size(); ++si) for (int t : plans[p].steps[si].replaced) { run_of[t] = (int)p; step_of[t] = (int)si; ++R.replaced_tasks; }
  // new task ids: kept tasks in order, each step's fused tasks where the step's first replaced task stood
  std::vector<int> new_of(nt, -1);
  std::map<std::pair<int, int>, int> first_new;
  int nnew = 0;
  std::vector<std::pair<int, int>> order;                    // (kind: -1 kept with old id, else plan<<20|step, old id or slot)
  for (int t = 0; t < nt; ++t) {
    if (run_of[t] < 0) { new_of[t] = nnew++; order.push_back({-1, t}); continue; }
    const auto key = std::make_pair(run_of[t], step_of[t]);
    if (!first_new.count(key)) {
      first_new[key] = nnew;
      const int nodes = plans[key.first].run.nsl + plans[key.first].run.groups;
      for (int i = 0; i < nodes; ++i) order.push_back({key.first << 20 | key.second, i});
      nnew += nodes;
    }
  }
  R.new_tasks = nnew;
  const auto slice_id = [&](int pl, int st, int i) { return first_new.at({pl, st}) + i; };
  const auto group_id = [&](int pl, int st, int j) { return first_new.at({pl, st}) + plans[pl].run.nsl + j; };
  std::vector<std::set<int>> out(nnew);
  const auto old_edges = kvm_static::build_dag(tasks, succ);
  for (int u = 0; u < nt; ++u) for (int v : old_edges[u]) {
    const bool ru = run_of[u] >= 0, rv = run_of[v] >= 0;
    if (ru && rv && run_of[u] == run_of[v] && step_of[u] == step_of[v]) continue;   // inside one replaced step
    std::vector<int> from, to;
    if (ru) for (int j = 0; j < plans[run_of[u]].run.groups; ++j) from.push_back(group_id(run_of[u], step_of[u], j)); else from.push_back(new_of[u]);
    if (rv) for (int i = 0; i < plans[run_of[v]].run.nsl; ++i) to.push_back(slice_id(run_of[v], step_of[v], i)); else to.push_back(new_of[v]);
    for (int f : from) for (int g : to) out[f].insert(g);
  }
  for (size_t p = 0; p < plans.size(); ++p) for (size_t st = 0; st < plans[p].steps.size(); ++st) for (int i = 0; i < plans[p].run.nsl; ++i) for (int j = 0; j < plans[p].run.groups; ++j) out[slice_id((int)p, (int)st, i)].insert(group_id((int)p, (int)st, j));
  // ---- the new CSR and argument arrays
  R.tasks.resize(nnew);
  std::vector<int> indeg(nnew, 0);
  for (const auto& e : out) for (int v : e) ++indeg[v];
  for (const auto& o : order) {
    KvmTaskDesc d{};
    d.deadline_us = -1; d.wait_var = -1; d.pub_var = -1;
    int id;
    if (o.first == -1) {
      const KvmTaskDesc& old = tasks[o.second];
      d = old; id = new_of[o.second];
      d.args_off = (int)R.args.size();
      R.args.insert(R.args.end(), args.begin() + old.args_off, args.begin() + old.args_off + old.args_words);
    } else {
      const int pl = o.first >> 20, st = o.first & 0xFFFFF, nsl = plans[pl].run.nsl;
      const Step& s = plans[pl].steps[st];
      d.workload_id = tasks[s.replaced[0]].workload_id;
      if (o.second < nsl) {
        id = slice_id(pl, st, o.second);
        d.op_id = ops.fmlp_slice; d.args_off = (int)R.args.size(); d.args_words = 4;
        R.args.insert(R.args.end(), {pl, o.second, s.ids, s.tgt});
      } else {
        id = group_id(pl, st, o.second - nsl);
        d.op_id = ops.fmlp_sgd; d.args_off = (int)R.args.size(); d.args_words = 3;
        R.args.insert(R.args.end(), {pl, o.second - nsl, s.loss});
      }
    }
    d.succ_off = (int)R.succ.size(); d.succ_cnt = (int)out[id].size(); d.indegree = indeg[id];
    R.succ.insert(R.succ.end(), out[id].begin(), out[id].end());
    R.tasks[id] = d;
  }
  for (const Plan& p : plans) { R.runs.push_back(p.run); for (const Step& st : p.steps) R.interior.insert(st.interior.begin(), st.interior.end()); }
  return R;
}
