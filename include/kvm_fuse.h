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
//
// ===== The split form =====
// A run that the one-block form cannot take (a width past 128, or no slice size inside the budget) takes the split form when kvm_fused_split.h does:
//  * one hidden width, at most KVM_SPLIT_MAXL layers, T = KVM_SPLIT_R x KVM_SPLIT_Q rows a step, and a share count S in KVM_SPLIT_SHARE_LIST whose
//    member layout fits the budget;
//  * each step becomes one gang of Q x S `fmlp_split` members (kvm_static_plan places a gang's members together);
//  * the edges into a step go to every member, and the edges out of it leave from member 0: a gang finishes together, its last barrier being the
//    step's end;
//  * consecutive steps of one run are joined member to member, so the planner can keep a member's chain on one worker.
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
#include "kvm_fused_split.h"
#include "kvm_fused_grid.h"
#include "kvm_static.h"
#include "kvm_task_abi.h"

#define KVM_FUSE_MAXL KVM_FMLP_MAXL   // hidden layers the template accepts

struct KvmFuseBuf { int rows = 0, cols = 0, ld = 0, dtype = 0; };   // what the pass needs of a bundle buffer (a parameter must be contiguous: ld == cols)

struct KvmFuseOps {   // operator ids (from the registry, by name); a missing op leaves its id at -1 and the template never matches it
  int noop = -1, select_window = -1, embedding_fwd = -1, matmul_tile = -1, softmax_ce_fwd = -1, softmax_ce_reduce = -1, softmax_ce_bwd = -1, softmax_ce_fused = -1, gelu_bwd = -1, colsum = -1, embedding_bwd = -1, sgd = -1, sgd_range = -1;
  int fmlp_slice = -1, fmlp_sgd = -1;   // the fused ops the pass emits
  int fmlp_split = -1;                  // the split form's gang op (-1: the split form is off)
  int fmlp_grid = -1;                   // the grid form's gang op (-1: the grid form is off)
};

struct KvmFuseOptions {
  int smem_floats = KVM_FMLP_SMEM_FLOATS;   // shared memory a fused block may use (a split member's too)
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
  int split = 0, S = 0, Q = 0;          // the split form: S shares x Q slices a gang (slice_rows = KVM_SPLIT_R; nsl and groups unused)
  int grid = 0, P = 0;                  // the grid form: one gang of P members a step over one copy of the model (nsl and groups unused)
};
// A run whose steps run as gangs (the split or the grid form), and the size of its gangs.
inline bool kvm_fuse_gang(const KvmFuseRun& r) { return r.split || r.grid; }
inline int kvm_fuse_members(const KvmFuseRun& r) { return r.split ? r.Q * r.S : r.grid ? r.P : 0; }

// The static kernel entry a rewritten bundle runs on (kvm_static.cuh). Every fused form runs on an entry of its own: ptxas sets an
// out-of-line function's registers by what every other function of its kernel holds across calls, and beside the one-block form's
// functions the grid member lost 35 of its 219 registers and reloaded in its MMA loops (sm_80 SASS, Nov 11). So a fused bundle holds one
// form (the one-block form, or one gang form at one share count) and, besides its fused tasks, only the steps' data windows and ordering
// noops; `mixed` names what else it holds (the executor then runs the pass again with fewer forms). Nothing fused: the generic entry,
// which holds no fused function and refuses the fused ops.
struct KvmFuseEntry {
  enum Kind { generic, block, split, grid, mixed } kind = generic;
  int S = 0;           // the split entry's share count
  std::string why;     // mixed: what keeps the bundle off a gang entry
};
struct KvmFuseResult {
  std::vector<KvmTaskDesc> tasks;
  std::vector<int32_t> succ, args;
  std::vector<KvmFuseRun> runs;
  std::vector<std::string> report;     // one line per component (and per reason it did not fire)
  int fused_runs = 0, unchanged_runs = 0, replaced_tasks = 0, new_tasks = 0;
  std::set<int> interior;              // buffers no longer written by a fused run (its activations and gradients): a reader of them after the run reads their initial values
  KvmFuseEntry entry;                  // the static kernel entry the rewritten bundle runs on (kvm_fuse_entry)
  std::string dropped;                 // non-empty: runs fused, but no single entry runs them with the rest of the bundle, so nothing was (why)
};

inline KvmFuseEntry kvm_fuse_entry(const KvmFuseResult& r, const KvmFuseOps& ops) {
  KvmFuseEntry e;
  int nsplit = 0, ngrid = 0, nblock = 0;
  for (const KvmFuseRun& run : r.runs) {
    if (run.split) { if (nsplit++ && run.S != e.S) { e.kind = KvmFuseEntry::mixed; e.why = "split runs at " + std::to_string(e.S) + " and " + std::to_string(run.S) + " shares"; return e; } e.S = run.S; }
    else if (run.grid) ++ngrid;
    else ++nblock;
  }
  if (!nsplit && !ngrid && !nblock) return e;
  if ((nsplit > 0) + (ngrid > 0) + (nblock > 0) > 1) {
    e.kind = KvmFuseEntry::mixed; e.why = std::to_string(nblock) + " one-block, " + std::to_string(nsplit) + " split and " + std::to_string(ngrid) + " grid run(s)"; return e;
  }
  const int a = nsplit ? ops.fmlp_split : ngrid ? ops.fmlp_grid : ops.fmlp_slice, b = nblock ? ops.fmlp_sgd : a;
  for (size_t t = 0; t < r.tasks.size(); ++t) {
    const int op = r.tasks[t].op_id;
    if (op != a && op != b && op != ops.select_window && op != ops.noop) {
      e.kind = KvmFuseEntry::mixed; e.why = "task " + std::to_string(t) + " (op " + std::to_string(op) + ") beside the fused runs"; return e;
    }
  }
  e.kind = nsplit ? KvmFuseEntry::split : ngrid ? KvmFuseEntry::grid : KvmFuseEntry::block;
  return e;
}
inline std::string kvm_fuse_entry_name(const KvmFuseEntry& e) {
  switch (e.kind) {
    case KvmFuseEntry::generic: return "generic";
    case KvmFuseEntry::block: return "one-block";
    case KvmFuseEntry::split: return "split, " + std::to_string(e.S) + " shares";
    case KvmFuseEntry::grid: return "grid";
    default: return "mixed (" + e.why + ")";
  }
}

namespace kvm_fuse {

constexpr int F_TRUNC = 256, F_DGELU = 512;   // matmul_tile flags: operands truncated to TF32 by the tensor core (a numerics preset of the tile; the fused GEMM has its own); out *= gelu'(pre) in the tile's epilogue
constexpr int F_BIG = 32, F_HUGE = 64;        // the tile's size (128 x 128, 256 x 128): a layout of the lowering, not of the math, so the template takes any

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
    // ---- softmax CE: fwd (logits, tgt, rowloss, lse, r0, r1, V, ignore), reduce (rowloss, tgt, loss, N, ignore), bwd (logits, tgt, lse, loss, dlogits, r0, r1, V, ignore, scale)
    int dY = -1;
    if (c + 2 <= comp.size() && op(comp[c]) == ops_.softmax_ce_fused) {       // softmax_ce_fused(logits, tgt, rowloss, lse, dlogits, r0, r1, V, ignore, N, scale), then the reduce
      if (op(comp[c + 1]) != ops_.softmax_ce_reduce) { why = "expected the reduce after the fused softmax CE"; return false; }
      const int* f = argp(comp[c]), *r = argp(comp[c + 1]);
      s.tgt = f[1]; s.loss = r[2]; dY = f[4];
      if (f[0] != Y || f[5] != 0 || f[6] != s.T || f[7] != s.V || f[8] != -1 || f[9] != s.T || afloat(f[10]) != 1.f || r[0] != f[2] || r[1] != s.tgt || r[3] != s.T || r[4] != -1) { why = "fused softmax CE shape"; return false; }
      s.interior.insert(f[2]); s.interior.insert(f[3]); s.interior.insert(dY);
      s.replaced.push_back(comp[c]); s.replaced.push_back(comp[c + 1]);
      c += 2;
    } else {
      if (c + 3 > comp.size() || op(comp[c]) != ops_.softmax_ce_fwd || op(comp[c + 1]) != ops_.softmax_ce_reduce || op(comp[c + 2]) != ops_.softmax_ce_bwd) { why = "expected softmax CE after the head"; return false; }
      const int* f = argp(comp[c]), *r = argp(comp[c + 1]), *bwd = argp(comp[c + 2]);
      s.tgt = f[1]; s.loss = r[2]; dY = bwd[4];
      if (f[0] != Y || f[4] != 0 || f[5] != s.T || f[6] != s.V || f[7] != -1 || r[0] != f[2] || r[1] != s.tgt || r[3] != s.T || r[4] != -1) { why = "softmax CE fwd or reduce shape"; return false; }
      if (bwd[0] != Y || bwd[1] != s.tgt || bwd[2] != f[3] || bwd[3] != s.loss || bwd[5] != 0 || bwd[6] != s.T || bwd[7] != s.V || bwd[8] != -1 || afloat(bwd[9]) != 1.f) { why = "softmax CE bwd shape"; return false; }
      s.interior.insert(f[2]); s.interior.insert(f[3]); s.interior.insert(dY);
      for (int i = 0; i < 3; ++i) s.replaced.push_back(comp[c + i]);
      c += 3;
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
      // colsum (x, out, R, c0, c1, acc): the bias gradient, one task or several over consecutive column ranges (a wide layer's lowering splits it)
      if (c >= comp.size() || op(comp[c]) != ops_.colsum) { why = "expected colsum of layer " + std::to_string(l); return false; }
      const int* cs = argp(comp[c]);
      if (cs[0] != dZ || cs[2] != s.T || cs[3] != 0 || cs[5] != 0) { why = "colsum of layer " + std::to_string(l); return false; }
      db[l] = cs[1]; s.interior.insert(cs[1]);
      for (int covered = 0; covered < Hl; ++c) {
        if (c >= comp.size() || op(comp[c]) != ops_.colsum) { why = "colsum of layer " + std::to_string(l) + " stops at column " + std::to_string(covered) + " of " + std::to_string(Hl); return false; }
        const int* cr = argp(comp[c]);
        if (cr[0] != dZ || cr[1] != db[l] || cr[2] != s.T || cr[3] != covered || cr[4] <= covered || cr[4] > Hl || cr[5] != 0) { why = "colsum of layer " + std::to_string(l) + ": its column ranges do not tile [0, " + std::to_string(Hl) + ")"; return false; }
        covered = cr[4]; s.replaced.push_back(comp[c]);
      }
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
    std::map<int, int> want, covered;
    want[s.wte] = dE; want[s.head] = dW_head;
    for (int l = 0; l < s.L; ++l) { want[s.w[l]] = dW[l]; want[s.b[l]] = db[l]; }
    // every parameter's update: one sgd (w, g, lr, n) over it, or sgd_range (w, g, lr, e0, e1) tasks over consecutive element ranges that tile it
    // (a large parameter's lowering splits its update), the ranges of one parameter in order; parameters may interleave
    for (size_t done = 0; done < want.size(); ++c) {
      if (c < comp.size() && op(comp[c]) == ops_.noop) {   // an ordering noop between updates (the lowering bounds an update's fan-in with them): part of the step
        size_t n = c;
        while (n < comp.size() && op(comp[n]) == ops_.noop) ++n;
        if (n < comp.size() && (op(comp[n]) == ops_.sgd || op(comp[n]) == ops_.sgd_range)) { for (; c < n; ++c) s.replaced.push_back(comp[c]); }
      }
      if (c >= comp.size() || (op(comp[c]) != ops_.sgd && op(comp[c]) != ops_.sgd_range)) { why = "expected an SGD update (" + std::to_string(want.size() - done) + " parameters not yet covered)"; return false; }
      const int* u = argp(comp[c]);
      auto it = want.find(u[0]);
      if (it == want.end() || it->second != u[1]) { why = "SGD does not pair a parameter with its gradient"; return false; }
      const int whole = bufs_[u[0]].rows * bufs_[u[0]].cols;
      int& have = covered[u[0]];
      if (have == whole) { why = "a parameter updated twice in one step"; return false; }
      if (op(comp[c]) == ops_.sgd) { if (have != 0 || u[3] != whole) { why = "SGD does not cover its whole parameter"; return false; } have = whole; }
      else { if (u[3] != have || u[4] <= have || u[4] > whole) { why = "SGD ranges of a parameter do not tile it in order"; return false; } have = u[4]; }
      if (s.lr == 0.f) s.lr = afloat(u[2]); else if (afloat(u[2]) != s.lr) { why = "SGD learning rates differ"; return false; }
      if (have == whole) ++done;
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
  // Consumes the contiguous tiles of one GEMM at comp[c]. All tiles must name the same GEMM and cover its M x N grid exactly once, with the base 64 x 64 tile.
  bool read_gemm(const std::vector<int>& comp, size_t& c, Gemm& g, std::string& why) const {
    g = Gemm{};
    if (c >= comp.size() || op(comp[c]) != ops_.matmul_tile) { why = "expected a GEMM"; return false; }
    std::set<std::pair<int, int>> tiles;
    const int* a0 = argp(comp[c]);
    const int fl = a0[11], M = a0[8], N = a0[9];
    const size_t first = c;
    const auto same = [&](int t) { const int* a = argp(t); return op(t) == ops_.matmul_tile && a[0] == a0[0] && a[1] == a0[1] && a[2] == a0[2] && a[3] == a0[3] && a[4] == a0[4] && a[5] == a0[5] && a[8] == M && a[9] == N && a[10] == a0[10] && a[11] == fl; };
    for (; c < comp.size(); ++c) {
      // an ordering noop between two tiles of this GEMM (the lowering's join of a split update, placed before the first tile that reads the parameter) is part
      // of the GEMM's span: it does no work, and the rewrite redirects its outside edges to the fused step like any replaced task's
      if (op(comp[c]) == ops_.noop) {
        size_t n = c;
        while (n < comp.size() && op(comp[n]) == ops_.noop) ++n;
        if (n < comp.size() && same(comp[n])) { for (; c < n; ++c) g.tasks.push_back(comp[c]); }
        else break;
      }
      if (op(comp[c]) != ops_.matmul_tile) break;
      const int* a = argp(comp[c]);
      if (!same(comp[c])) break;   // the next GEMM starts here
      if (afloat(a[12]) != 1.f) { why = "a GEMM with alpha != 1"; return false; }
      if (!tiles.insert({a[6], a[7]}).second) { why = "a tile listed twice in a GEMM"; return false; }
      g.tasks.push_back(comp[c]);
    }
    const int fl_kept = fl & ~(kvm_fuse::F_TRUNC | kvm_fuse::F_BIG | kvm_fuse::F_HUGE);
    if (fl_kept & ~(15 | kvm_fuse::F_DGELU)) { why = "a GEMM with flags " + std::to_string(fl) + " the template does not hold"; return false; }
    // the op's grid at its tile size (64 x 64; 128 x 128 with F_BIG; 256 x 128 with F_HUGE), clipped to the GEMM's edge by the op itself
    const int tm = (fl & kvm_fuse::F_HUGE) ? 256 : (fl & kvm_fuse::F_BIG) ? 128 : 64, tn = (fl & (kvm_fuse::F_BIG | kvm_fuse::F_HUGE)) ? 128 : 64;
    for (int m0 = 0; m0 < M; m0 += tm) for (int n0 = 0; n0 < N; n0 += tn) if (!tiles.count({m0, n0})) { why = "a GEMM's tile set does not cover its " + std::to_string(tm) + " x " + std::to_string(tn) + " grid (" + std::to_string(M) + " x " + std::to_string(N) + ", flags " + std::to_string(fl) + ")"; return false; }
    if ((int)tiles.size() != ((M + tm - 1) / tm) * ((N + tn - 1) / tn)) { why = "a GEMM has tiles outside its " + std::to_string(tm) + " x " + std::to_string(tn) + " grid"; return false; }
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

// The share counts (ascending, from KVM_SPLIT_SHARE_LIST) whose split member fits `opts` for the shape: the layout kvm_fused_split.h takes, within the
// shared-memory budget, and one gang of Q x S members inside the workers. Empty, with the reason in `why`, when none does.
inline std::vector<int> kvm_fuse_split_fits(int L, int E, int V, const std::array<int, KVM_FUSE_MAXL>& H, int T, const KvmFuseOps& ops, const KvmFuseOptions& opts, std::string& why) {
  std::vector<int> fit;
  if (ops.fmlp_split < 0) { why = "off (no fmlp_split op)"; return fit; }
  if (L > KVM_SPLIT_MAXL) { why = std::to_string(L) + " hidden layers, at most " + std::to_string(KVM_SPLIT_MAXL); return fit; }
  for (int l = 1; l < L; ++l) if (H[l] != H[0]) { why = "the hidden widths differ (the op takes one)"; return fit; }
  if (T != KVM_SPLIT_R * KVM_SPLIT_Q) { why = std::to_string(T) + " rows a step (the op is built for " + std::to_string(KVM_SPLIT_R * KVM_SPLIT_Q) + ")"; return fit; }
  // the smallest member among the share counts the shape takes and the workers allow: how far the best candidate is from the budget
  int takes = 0, best = -1, best_s = 0;
#define X(S_) { KvmSplitDesc d{}; d.V = V; d.E = E; d.L = L; d.H = H[0]; d.S = S_; d.Q = KVM_SPLIT_Q; const int f = kvm_split_layout(d); \
                takes += f > 0; \
                if (f > 0 && KVM_SPLIT_Q * S_ <= opts.workers && (best < 0 || f < best)) { best = f; best_s = S_; } \
                if (f > 0 && f <= opts.smem_floats && KVM_SPLIT_Q * S_ <= opts.workers) fit.push_back(S_); }
  KVM_SPLIT_SHARE_LIST
#undef X
  if (fit.empty())
    why = !takes ? "no share count takes the shape (E, V, H or a share's width outside the op's bounds)"
        : best < 0 ? "no gang fits " + std::to_string(opts.workers) + " workers (" + std::to_string(KVM_SPLIT_Q) + " x S members)"
                   : "no share count fits (at best " + std::to_string(best) + " floats a member, at " + std::to_string(best_s) + " shares, limit " + std::to_string(opts.smem_floats) +
                     ", gangs of at most " + std::to_string(opts.workers) + " workers)";
  return fit;
}

// The grid form's gang for the shape: every worker a member (more members make every member's blocks smaller), up to the KVM_GRID_MAXP members
// the flag-line barrier's polling warps cover, when that member fits the budget. Returns the member count, or 0 with the reason in `why`. The
// pass offers it to a run the split form refused, and to every run of a bundle where one needs it (one form a bundle).
inline int kvm_fuse_grid_fits(int L, int E, int V, const std::array<int, KVM_FUSE_MAXL>& H, int T, const KvmFuseOps& ops, const KvmFuseOptions& opts, std::string& why) {
  if (ops.fmlp_grid < 0) { why = "off (no fmlp_grid op)"; return 0; }
  if (L > KVM_GRID_MAXL) { why = std::to_string(L) + " hidden layers, at most " + std::to_string(KVM_GRID_MAXL); return 0; }
  for (int l = 1; l < L; ++l) if (H[l] != H[0]) { why = "the hidden widths differ (the op takes one)"; return 0; }
  KvmGridDesc d{}; d.V = V; d.E = E; d.L = L; d.H = H[0]; d.T = T; d.P = std::min(opts.workers, KVM_GRID_MAXP);
  const int f = kvm_grid_layout(d);
  if (f < 0) { why = "the shape is outside the op's bounds (H a multiple of 8, E of 16, V of 4 up to 128, 96 rows a step)"; return 0; }
  if (f > opts.smem_floats) { why = std::to_string(f) + " floats a member at " + std::to_string(d.P) + " members, limit " + std::to_string(opts.smem_floats); return 0; }
  return d.P;
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
  struct Plan { KvmFuseRun run; std::vector<Step> steps; std::vector<int> split_fit; std::string split_why; };   // split_fit: the share counts that fit (the split form only)
  std::vector<Plan> plans;
  std::vector<int> plan_comp;   // the component each plan came from (for the report)
  const kvm_fuse::Parser parser(tasks, args, bufs, ops);
  const auto comps = kvm_fuse_components(tasks, succ);
  for (size_t ci = 0; ci < comps.size(); ++ci) {
    const std::vector<int>& comp = comps[ci];
    Plan pl; std::string why;
    size_t pos = 0;
    while (pos < comp.size()) {
      Step st;
      if (!parser.parse(comp, pos, st, why)) { why = "step " + std::to_string(pl.steps.size()) + ": " + why; break; }
      pl.steps.push_back(std::move(st));
    }
    if (why.empty() && pl.steps.empty()) why = "no training step found";
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
    if (why.empty()) {
      const Step& s = pl.steps[0];
      std::string one;   // why the one-block form cannot take the run (empty: it can)
      for (int l = 0; l < s.L; ++l) if (s.H[l] % 8 || s.H[l] > 128) { one = "layer width not a multiple of 8 up to 128"; break; }
      if (one.empty() && (s.E % 8 || s.E == 0)) one = "embedding width not a multiple of 8";
      if (one.empty() && (s.V % 4 || s.V == 0 || s.V > 128)) one = "vocabulary not a multiple of 4 up to 128";
      if (one.empty()) {
        bool any = false;
        for (int r : kvm_fuse_slice_choices) any |= s.T % r == 0 && r <= s.T && s.T / r <= KVM_FMLP_MAX_SLICES && s.T / KVM_FMLP_GRANULE <= KVM_FMLP_MAX_SLICES && kvm_fuse_layout_floats(s.L, s.E, s.V, s.H, s.T, r) <= opts.smem_floats;
        if (!any) one = "no slice size fits (" + std::to_string(kvm_fuse_layout_floats(s.L, s.E, s.V, s.H, s.T, kvm_fuse_slice_choices[0])) + " floats of shared memory at the smallest slice, at most " + std::to_string(KVM_FMLP_MAX_SLICES) + " slices, limit " + std::to_string(opts.smem_floats) + ")";
      }
      const auto take = [&]() { pl.run.L = s.L; pl.run.E = s.E; pl.run.V = s.V; pl.run.T = s.T; pl.run.lr = s.lr; pl.run.H = s.H; pl.run.w = s.w; pl.run.b = s.b; pl.run.wte = s.wte; pl.run.head = s.head; pl.run.steps = (int)pl.steps.size(); pl.run.smem_floats = opts.smem_floats; };
      if (one.empty()) take();
      else {
        std::string sw;
        const std::vector<int> fit = kvm_fuse_split_fits(s.L, s.E, s.V, s.H, s.T, ops, opts, sw);
        if (!fit.empty()) { take(); pl.run.split = 1; pl.run.Q = KVM_SPLIT_Q; pl.run.S = fit.back(); pl.split_fit = fit; pl.split_why = one; }
        else {   // the grid form: for a run the split form cannot hold
          std::string gw;
          const int P = kvm_fuse_grid_fits(s.L, s.E, s.V, s.H, s.T, ops, opts, gw);
          if (P) { take(); pl.run.grid = 1; pl.run.P = P; pl.split_why = one + "; split form: " + sw; }
          else why = one + "; split form: " + sw + "; grid form: " + gw;
        }
      }
    }
    if (why.empty()) {   // a buffer the caller reads after the run may not be interior
      for (size_t si = 0; si < pl.steps.size() && why.empty(); ++si) for (int b : opts.keep) if (pl.steps[si].interior.count(b)) { why = "buffer " + std::to_string(b) + " is read after the run and interior to step " + std::to_string(si); break; }
    }
    if (!why.empty()) { ++R.unchanged_runs; R.report.push_back("component " + std::to_string(ci) + " (" + std::to_string(comp.size()) + " tasks): unchanged: " + why); continue; }
    ++R.fused_runs; plans.push_back(std::move(pl)); plan_comp.push_back((int)ci);
  }
  if (plans.empty()) { R.tasks = tasks; R.succ = succ; R.args = args; return R; }
  // ---- one form for the bundle: each fused form runs on a static-kernel entry of its own (kvm_fuse_entry), so runs that took different forms
  // move to the widest form among them when each fits it (a one-block run to the split or the grid form, a split run to the grid form)
  {
    bool any_block = false, any_split = false, any_grid = false;
    for (const Plan& p : plans) { any_split |= p.run.split != 0; any_grid |= p.run.grid != 0; any_block |= !kvm_fuse_gang(p.run); }
    if (any_block + any_split + any_grid > 1)
      for (Plan& p : plans) {
        const Step& s = p.steps[0];
        std::string w;
        if (any_grid && !p.run.grid) {
          const int P = kvm_fuse_grid_fits(s.L, s.E, s.V, s.H, s.T, ops, opts, w);
          if (P) { p.run.split = p.run.S = p.run.Q = 0; p.split_fit.clear(); p.run.grid = 1; p.run.P = P; p.split_why = "one form for the bundle: grid"; }
        } else if (!any_grid && !p.run.split) {
          const std::vector<int> fit = kvm_fuse_split_fits(s.L, s.E, s.V, s.H, s.T, ops, opts, w);
          if (!fit.empty()) { p.run.split = 1; p.run.Q = KVM_SPLIT_Q; p.run.S = fit.back(); p.split_fit = fit; p.split_why = "one form for the bundle: split"; }
        }
      }
  }
  // ---- slice size: one size for all runs, the smallest that fits every run and gives each slice a worker; if none does, the largest that fits every run (slices then share workers);
  // runs that fit no common size each take their own smallest
  {
    // a slice holds whole granules, and the owners sum the step's T / 16 granule partials at most KVM_FMLP_MAX_SLICES at a time (kvm_fused_mlp.h)
    auto fits = [&](const KvmFuseRun& r, int rows) { return r.T % rows == 0 && rows <= r.T && r.T / rows <= KVM_FMLP_MAX_SLICES && r.T / KVM_FMLP_GRANULE <= KVM_FMLP_MAX_SLICES && kvm_fuse_layout_floats(r.L, r.E, r.V, r.H, r.T, rows) <= opts.smem_floats; };
    int common = 0, widest_common = 0, nblock = 0;
    for (const Plan& p : plans) nblock += !kvm_fuse_gang(p.run);
    for (int rows : kvm_fuse_slice_choices) {
      bool all = nblock > 0; int total = 0;
      for (const Plan& p : plans) if (!kvm_fuse_gang(p.run)) { all &= fits(p.run, rows); total += p.run.T / rows; }
      if (!all) continue;
      widest_common = rows;
      if (total <= opts.workers) { common = rows; break; }
    }
    if (!common) common = widest_common;
    for (Plan& p : plans) {
      if (kvm_fuse_gang(p.run)) continue;
      if (common) { p.run.slice_rows = common; continue; }
      for (int rows : kvm_fuse_slice_choices) if (fits(p.run, rows)) { p.run.slice_rows = rows; break; }
    }
    // the split runs' share count: each run's own, the largest its shape takes (kvm_fuse_split_fits: a member within the budget, one gang of
    // Q x S members within the workers; wider shares ran faster, measured: h576-d1 on the A100, 52.1 us a step at S = 5 against 34.2 at S = 12).
    // Never chosen by how many runs share the bundle: the logits' and the reduce-scatter's sums run over a run's S shares, so S is part of the
    // run's arithmetic, and a run's arithmetic must depend on its shape alone (the MLP bench's gate 2, bench/exec_vs_torch/run_mlp.py, asserts
    // four replicated copies of a run bitwise equal to the run alone). Runs of one shape take one S; their gangs take turns past the workers.
    // Runs whose counts differ leave the bundle unfused (kvm_fuse_entry: an entry runs one share count).
    for (Plan& p : plans) if (p.run.split) p.run.S = p.split_fit.back();
    for (size_t i = 0; i < plans.size(); ++i) {
      KvmFuseRun& r = plans[i].run;
      if (r.grid) {
        KvmGridDesc d{}; d.V = r.V; d.E = r.E; d.L = r.L; d.H = r.H[0]; d.T = r.T; d.P = r.P;
        const int floats = kvm_grid_layout(d);
        std::string shape = std::to_string(r.E);
        for (int l = 0; l < r.L; ++l) shape += "x" + std::to_string(r.H[l]);
        R.report.push_back("component " + std::to_string(plan_comp[i]) + ": fused, grid form (" + plans[i].split_why + "): " + std::to_string(plans[i].steps.size()) + " steps, T=" + std::to_string(r.T) +
                           " V=" + std::to_string(r.V) + " widths " + shape + ", gangs of " + std::to_string(r.P) + " members (layer 0 1 x " + std::to_string(d.nb0) + ", hidden " + std::to_string(d.ka) + " x " +
                           std::to_string(d.nb) + ", head " + std::to_string(d.kah) + " x 1), " + std::to_string(floats) + " floats a member");
        continue;
      }
      if (r.split) {
        r.slice_rows = KVM_SPLIT_R;
        KvmSplitDesc d{}; d.V = r.V; d.E = r.E; d.L = r.L; d.H = r.H[0]; d.S = r.S; d.Q = r.Q;
        const int floats = kvm_split_layout(d);
        std::string shape = std::to_string(r.E);
        for (int l = 0; l < r.L; ++l) shape += "x" + std::to_string(r.H[l]);
        R.report.push_back("component " + std::to_string(plan_comp[i]) + ": fused, split form (" + plans[i].split_why + "): " + std::to_string(plans[i].steps.size()) + " steps, T=" + std::to_string(r.T) + " V=" + std::to_string(r.V) + " widths " + shape + ", gangs of " + std::to_string(r.Q) + " slices x " + std::to_string(r.S) + " shares = " + std::to_string(r.Q * r.S) + " members, " + std::to_string(floats) + " floats a member");
        continue;
      }
      r.nsl = r.T / r.slice_rows;
      // the reduction runs after every slice of a step has finished: the slices' workers are free for it, and any workers no slice uses take a share too (the reduce is bandwidth-bound per SM)
      { int total = 0; for (const Plan& p : plans) if (!kvm_fuse_gang(p.run)) total += p.run.T / p.run.slice_rows;
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
      const KvmFuseRun& kr = plans[key.first].run;
      const int nodes = kvm_fuse_gang(kr) ? kvm_fuse_members(kr) : kr.nsl + kr.groups;
      for (int i = 0; i < nodes; ++i) order.push_back({key.first << 20 | key.second, i});
      nnew += nodes;
    }
  }
  R.new_tasks = nnew;
  const auto slice_id = [&](int pl, int st, int i) { return first_new.at({pl, st}) + i; };
  const auto group_id = [&](int pl, int st, int j) { return first_new.at({pl, st}) + plans[pl].run.nsl + j; };
  const auto member_id = [&](int pl, int st, int m) { return first_new.at({pl, st}) + m; };   // the gang forms: member m of a step's gang
  std::vector<std::set<int>> out(nnew);
  const auto old_edges = kvm_static::build_dag(tasks, succ);
  for (int u = 0; u < nt; ++u) for (int v : old_edges[u]) {
    const bool ru = run_of[u] >= 0, rv = run_of[v] >= 0;
    if (ru && rv && run_of[u] == run_of[v] && step_of[u] == step_of[v]) continue;   // inside one replaced step
    const bool su = ru && kvm_fuse_gang(plans[run_of[u]].run), sv = rv && kvm_fuse_gang(plans[run_of[v]].run);
    if (su && sv && run_of[u] == run_of[v]) {   // two steps of one gang run: member to member (a gang finishes together, so any member's end is the step's)
      const int G = kvm_fuse_members(plans[run_of[u]].run);
      for (int m = 0; m < G; ++m) out[member_id(run_of[u], step_of[u], m)].insert(member_id(run_of[v], step_of[v], m));
      continue;
    }
    std::vector<int> from, to;
    if (su) from.push_back(member_id(run_of[u], step_of[u], 0));   // a gang step's end is any member's: member 0 stands for it
    else if (ru) for (int j = 0; j < plans[run_of[u]].run.groups; ++j) from.push_back(group_id(run_of[u], step_of[u], j));
    else from.push_back(new_of[u]);
    if (sv) for (int m = 0; m < kvm_fuse_members(plans[run_of[v]].run); ++m) to.push_back(member_id(run_of[v], step_of[v], m));
    else if (rv) for (int i = 0; i < plans[run_of[v]].run.nsl; ++i) to.push_back(slice_id(run_of[v], step_of[v], i));
    else to.push_back(new_of[v]);
    for (int f : from) for (int g : to) out[f].insert(g);
  }
  for (size_t p = 0; p < plans.size(); ++p) if (!kvm_fuse_gang(plans[p].run)) for (size_t st = 0; st < plans[p].steps.size(); ++st) for (int i = 0; i < plans[p].run.nsl; ++i) for (int j = 0; j < plans[p].run.groups; ++j) out[slice_id((int)p, (int)st, i)].insert(group_id((int)p, (int)st, j));
  // gang ids: one per gang step, unique in the bundle (kvm_static_plan groups a gang's members by it)
  std::map<std::pair<int, int>, int> gang_of_step;
  for (size_t p = 0; p < plans.size(); ++p) if (kvm_fuse_gang(plans[p].run)) for (size_t st = 0; st < plans[p].steps.size(); ++st) gang_of_step[{(int)p, (int)st}] = (int)gang_of_step.size();
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
      if (kvm_fuse_gang(plans[pl].run)) {   // fmlp_split and fmlp_grid take the same arguments
        id = member_id(pl, st, o.second);
        d.op_id = plans[pl].run.split ? ops.fmlp_split : ops.fmlp_grid; d.args_off = (int)R.args.size(); d.args_words = 7;
        R.args.insert(R.args.end(), {pl, gang_of_step.at({pl, st}), o.second, kvm_fuse_members(plans[pl].run), s.ids, s.tgt, s.loss});
      } else if (o.second < nsl) {
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
  R.entry = kvm_fuse_entry(R, ops);
  if (R.entry.kind == KvmFuseEntry::mixed) {   // fused runs beside other work, or forms that would not unify: no entry runs the bundle
    KvmFuseResult U;
    U.tasks = tasks; U.succ = succ; U.args = args; U.report = R.report;
    U.unchanged_runs = R.fused_runs + R.unchanged_runs;
    U.dropped = R.entry.why;
    U.report.push_back("every run left unchanged: " + R.entry.why + " (each fused form runs on a static-kernel entry of its own)");
    return U;
  }
  return R;
}
