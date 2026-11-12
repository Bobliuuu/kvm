#pragma once
// KVM fusion pass for GPT training steps, host side: recognize whole pre-norm GPT training steps in a closed bundle and describe each matched
// run for the fused GPT step (Megatron column and row shards over a gang of blocks, AdamW in place).
//
// ===== What it decides, and from what =====
// From the bundle's DAG and its op arguments alone, like kvm_fuse_mlp: no model name, no task name, no width, depth, head or vocabulary
// constant. Each connected component (an independent run) is parsed in bundle order against one template:
//   data windows -> embedding -> L x [LayerNorm -> q, k, v GEMMs with bias, RoPE on q and k -> causal attention -> proj GEMM with bias and
//   the residual -> LayerNorm -> fc1 GEMM with bias and GELU -> fc2 GEMM with bias and the residual] -> LayerNorm -> head GEMM -> softmax
//   cross-entropy -> the mirrored backward -> grad-norm partials and finish -> the step counter -> AdamW on every parameter,
// and every buffer identity the template implies is checked: each GEMM chains from its producer, the residual-stream gradient accumulates
// through the layers in place, every update pairs a parameter with its own gradient and moments, every row, column, tile and attention block
// set covers its whole extent exactly once. A run whose steps do not all match, or whose steps differ in shape or parameters, stays as written,
// and the report says which step and why.
//
// ===== Output =====
// A description of each matched run (KvmFuseGptRun: the shapes, every parameter with its gradient and moments, the step's tasks), and the
// rewrite (kvm_fuse_gpt) of each run that fits into the G members of one fgpt_step gang, which the executor launches as one kernel
// (KvmExec::run_fgpt). A run takes the fused step's form (kvm_fused_gpt.h) where a gang of it fits; where none does (its N-wide phases
// outgrow a member's shared memory, as a d 384 GPT's do on an A100), F's 2-D form (kvm_f2d.h), each half of a layer on a layout of its
// own. Host only: tests/static/test_fuse_gpt.cpp drives it with plain g++ on real bundles.
#include <set>
#include <string>
#include <vector>
#include "kvm_fuse.h"
#include "kvm_f2d.h"
#include "kvm_fused_gpt.h"

struct KvmFuseGptOps {   // operator ids (from the registry, by name); a missing op leaves its id at -1 and the template never matches it
  int select_window = -1, embedding_fwd = -1, layernorm_fwd = -1, matmul_tile = -1, rope_fwd = -1, attn_fwd = -1, softmax_ce_fused = -1,
      softmax_ce_reduce = -1, layernorm_bwd_dx = -1, norm_bwd_params = -1, colsum = -1, attn_bwd_delta = -1, attn_bwd_dq = -1, attn_bwd_dkv = -1,
      rope_bwd = -1, embedding_bwd = -1, gradnorm_partial = -1, gradnorm_finish = -1, counter_inc = -1, adamw = -1, noop = -1;
};

struct KvmGptLayer {   // a transformer block's parameters (bundle buffer indices)
  int n1g = -1, n1b = -1, wq = -1, bq = -1, wk = -1, bk = -1, wv = -1, bv = -1, wo = -1, bo = -1, n2g = -1, n2b = -1, w1 = -1, b1 = -1, w2 = -1, b2 = -1;
};

struct KvmGptUpdate { int p = -1, g = -1, m = -1, v = -1; float wd = 0.f; };   // one AdamW update: a parameter, its gradient, its moments

// One step of a matched run.
struct KvmGptStep {
  int N = 0, T = 0, B = 0, d = 0, H = 0, hd = 0, F = 0, V = 0, L = 0;     // rows (B sequences of T tokens), width, heads, head width, MLP width, vocabulary, layers
  int mm_numerics = 0;                    // the GEMM tiles' numerics bits (precise, trunc, exact), the same on every tile of the step
  float ln_eps = 0.f, theta = 0.f, scale = 0.f;   // LayerNorm epsilon, RoPE base, attention scale
  float lr = 0.f, beta1 = 0.f, beta2 = 0.f, adam_eps = 0.f, max_norm = 0.f;
  int ids = -1, tgt = -1, loss = -1, step_buf = -1, coef = -1;            // the step's inputs and outputs that live outside it
  int ids_src = -1, ids_stride = 0, ids_off = 0, tgt_src = -1, tgt_stride = 0, tgt_off = 0;   // the data windows: element i of ids is ids_src[step * stride + off + i]
  int wte = -1, nfg = -1, nfb = -1, head = -1;
  std::vector<KvmGptLayer> layers;
  std::vector<KvmGptUpdate> updates;      // in the bundle's AdamW order
  std::vector<int> replaced;              // every task of this step the fused step takes over
  std::set<int> interior;                 // buffers only this step's replaced tasks may touch (activations and gradients)
  int barrier_end = -1;                   // the noop that ends the step, kept as the join into the next one
  // A version the step publishes for readers of its weights (the bundle's barriers do): at most one before the updates (once every gradient
  // is final) and one at the step's end (once every weight is updated); -1 none. No task of a step may wait on a version.
  int pub_var = -1, pub_pre = -1, pub_end = -1;
};

struct KvmFuseGptRun {
  KvmGptStep shape; int steps = 0; int component = -1;
  std::vector<int> losses;                // each step's loss buffer, in step order
  int pub_first = -1;                     // the first version the run publishes (step 0's first), then one more at every publish point of every step
};
struct KvmFuseGptResult {
  std::vector<KvmFuseGptRun> runs;
  std::vector<std::string> report;        // one line per component
  int matched_runs = 0, unchanged_runs = 0;
};

namespace kvm_fuse {

constexpr int G_TA = 1, G_TB = 2, G_ACC = 4, G_GELU = 8, G_PRECISE = 16, G_BIG = 32, G_HUGE = 64, G_TRUNC = 256, G_DGELU = 512, G_EXACT = 1024;
constexpr int G_ROLE = G_TA | G_TB | G_ACC | G_GELU | G_DGELU, G_NUMERICS = G_PRECISE | G_TRUNC | G_EXACT, G_TILE = G_BIG | G_HUGE;

class GptParser {
 public:
  GptParser(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& args, const std::vector<KvmFuseBuf>& bufs, const KvmFuseGptOps& ops)
      : tasks_(tasks), args_(args), bufs_(bufs), ops_(ops) {}

  struct Gemm { int a = -1, b = -1, out = -1, bias = -1, res = -1, pre = -1, M = 0, N = 0, K = 0, role = 0, numerics = 0; std::vector<int> tasks; };

  // Parses one step starting at comp[pos]; on success advances pos past it and fills `s`. On failure leaves `why` set to the first mismatch.
  bool parse(const std::vector<int>& comp, size_t& pos, KvmGptStep& s, std::string& why) const {
    s = KvmGptStep{};
    size_t c = pos;
    numerics_ = -1;
    joins_.clear();
    // ---- data windows: select_window (src, step, dst, i0, i1, stride, off), blocks of the step's tokens and targets; matched to the embedding's
    // ids and the loss's targets once those are read, and to the step counter once that is
    struct Win { int src = -1, step = -1, stride = 0, off = 0; std::vector<std::pair<int, int>> rows; };
    std::map<int, Win> wins;                              // by destination buffer
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.select_window; ++c) {
      const int* w = argp(comp[c]);
      Win& x = wins[w[2]];
      if (x.src == -1) { x.src = w[0]; x.step = w[1]; x.stride = w[5]; x.off = w[6]; }
      else if (x.src != w[0] || x.step != w[1] || x.stride != w[5] || x.off != w[6]) { why = "the data windows of one buffer disagree"; return false; }
      x.rows.push_back({w[3], w[4]}); take(s, comp[c]);
    }
    // ---- embedding_fwd (ids, wte, wpe, out, r0, r1, T, d): row blocks of one lookup, covering every row of its output
    if (!at(comp, c, ops_.embedding_fwd, "the embedding", why)) return false;
    const int* e = argp(comp[c]);
    s.ids = e[0]; s.wte = e[1]; s.T = e[6]; s.d = e[7];
    const int x0 = e[3];
    if (e[2] != -1) { why = "an embedding with a position table"; return false; }
    {
      std::vector<std::pair<int, int>> rows;
      for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.embedding_fwd; ++c) {
        const int* a = argp(comp[c]);
        if (a[0] != e[0] || a[1] != e[1] || a[2] != e[2] || a[3] != e[3] || a[6] != e[6] || a[7] != e[7]) break;
        rows.push_back({a[4], a[5]}); take(s, comp[c]);
      }
      if (x0 < 0 || x0 >= (int)bufs_.size()) { why = "embedding output buffer"; return false; }
      s.N = bufs_[x0].rows;
      if (!covers(rows, s.N)) { why = "the embedding's row blocks do not cover its output"; return false; }
    }
    if (s.T <= 0 || s.N % s.T) { why = "embedding rows are not whole sequences"; return false; }
    s.B = s.N / s.T;
    if (!buf_is(s.wte, -1, s.d) || !buf_is(x0, s.N, s.d)) { why = "embedding buffer shapes"; return false; }
    s.V = bufs_[s.wte].rows;
    s.interior.insert(x0);

    // ---- forward layers
    struct Fwd { int x, h, mean1, rstd1, q, qr, k, kr, v, o, lse, xm, h2, mean2, rstd2, u, pre, xn; };
    std::vector<Fwd> fw;
    int x = x0;
    for (;;) {
      // a layer starts with a LayerNorm whose next GEMM has three siblings (q, k, v); the final LayerNorm is followed by the head GEMM alone
      const size_t jmark = joins_.size();                // the lookahead's joins: kept up to the LayerNorm's end for a layer, dropped for the head
      size_t look = c;
      Ln ln;
      if (!read_ln(comp, look, x, ln, why)) return false;
      look = skip(comp, look);
      const size_t jln = joins_.size();
      if (look >= comp.size() || op(comp[look]) != ops_.matmul_tile) { why = "expected a GEMM after a LayerNorm"; return false; }
      Gemm g1;
      size_t look2 = look;
      if (!read_gemm(comp, look2, g1, why)) return false;
      look2 = skip(comp, look2);
      const bool layer = look2 < comp.size() && op(comp[look2]) == ops_.rope_fwd;
      joins_.resize(layer ? jln : jmark);                // a layer resumes after its LayerNorm; the head re-reads the final LayerNorm from c
      if (!layer) break;                                  // the final LayerNorm and the head: handled below from c
      Fwd f{};
      f.x = x;
      const int l = (int)s.layers.size();
      KvmGptLayer P;
      P.n1g = ln.gamma; P.n1b = ln.beta; f.h = ln.out; f.mean1 = ln.mean; f.rstd1 = ln.rstd;
      if (!ln_params(s, ln, l == 0, why)) return false;
      take_all(s, ln.tasks); s.interior.insert({ln.out, ln.mean, ln.rstd});
      c = look;
      // q, then RoPE on it; k, then RoPE; v
      Gemm gq, gk, gv;
      if (!read_gemm(comp, c, gq, why) || !lin(s, gq, f.h, s.d, s.d, true, -1, 0, "q projection", why)) return false;
      Rope rq;
      if (!read_rope(comp, c, ops_.rope_fwd, gq.out, s, rq, why)) return false;
      if (!read_gemm(comp, c, gk, why) || !lin(s, gk, f.h, s.d, s.d, true, -1, 0, "k projection", why)) return false;
      Rope rk;
      if (!read_rope(comp, c, ops_.rope_fwd, gk.out, s, rk, why)) return false;
      if (!read_gemm(comp, c, gv, why) || !lin(s, gv, f.h, s.d, s.d, true, -1, 0, "v projection", why)) return false;
      P.wq = gq.b; P.bq = gq.bias; P.wk = gk.b; P.bk = gk.bias; P.wv = gv.b; P.bv = gv.bias;
      f.q = gq.out; f.qr = rq.out; f.k = gk.out; f.kr = rk.out; f.v = gv.out;
      for (const Gemm* g : {&gq, &gk, &gv}) { take_all(s, g->tasks); s.interior.insert(g->out); }
      take_all(s, rq.tasks); take_all(s, rk.tasks); s.interior.insert({rq.out, rk.out});
      // causal attention, one task per (sequence, head, query block)
      Attn att;
      if (!read_attn_fwd(comp, c, f.qr, f.kr, f.v, s, att, why)) return false;
      f.o = att.o; f.lse = att.lse;
      take_all(s, att.tasks); s.interior.insert({att.o, att.lse});
      // proj with bias and the residual x
      Gemm go;
      if (!read_gemm(comp, c, go, why) || !lin(s, go, f.o, s.d, s.d, true, x, 0, "attention output projection", why)) return false;
      P.wo = go.b; P.bo = go.bias; f.xm = go.out;
      take_all(s, go.tasks); s.interior.insert(go.out);
      // LayerNorm, fc1 with bias and GELU (its pre-activation kept), fc2 with bias and the residual
      Ln ln2;
      if (!read_ln(comp, c, f.xm, ln2, why) || !ln_params(s, ln2, false, why)) return false;
      P.n2g = ln2.gamma; P.n2b = ln2.beta; f.h2 = ln2.out; f.mean2 = ln2.mean; f.rstd2 = ln2.rstd;
      take_all(s, ln2.tasks); s.interior.insert({ln2.out, ln2.mean, ln2.rstd});
      Gemm g1f, g2f;
      if (!read_gemm(comp, c, g1f, why)) return false;
      if (l == 0) s.F = g1f.N;
      if (!lin(s, g1f, f.h2, s.d, s.F, true, -1, G_GELU, "fc1", why)) return false;
      if (!buf_is(g1f.pre, s.N, s.F)) { why = "fc1's pre-activation buffer"; return false; }
      if (!read_gemm(comp, c, g2f, why) || !lin(s, g2f, g1f.out, s.F, s.d, true, f.xm, 0, "fc2", why)) return false;
      P.w1 = g1f.b; P.b1 = g1f.bias; P.w2 = g2f.b; P.b2 = g2f.bias; f.u = g1f.out; f.pre = g1f.pre; f.xn = g2f.out;
      for (const Gemm* g : {&g1f, &g2f}) { take_all(s, g->tasks); s.interior.insert(g->out); }
      s.interior.insert(g1f.pre);
      s.layers.push_back(P); fw.push_back(f);
      x = f.xn;
    }
    s.L = (int)s.layers.size();
    if (s.L == 0) { why = "no transformer layer before the head"; return false; }
    // ---- final LayerNorm and the head (no bias, no residual)
    Ln lnf;
    if (!read_ln(comp, c, x, lnf, why) || !ln_params(s, lnf, false, why)) return false;
    s.nfg = lnf.gamma; s.nfb = lnf.beta;
    take_all(s, lnf.tasks); s.interior.insert({lnf.out, lnf.mean, lnf.rstd});
    Gemm gh;
    if (!read_gemm(comp, c, gh, why) || !lin(s, gh, lnf.out, s.d, s.V, false, -1, 0, "the head", why)) return false;
    s.head = gh.b;
    take_all(s, gh.tasks); s.interior.insert(gh.out);
    const int logits = gh.out;
    // ---- softmax CE, fused: (logits, tgt, rowloss, lse, dlogits, r0, r1, V, ignore, N, scale) row blocks covering [0, N), then the reduce (rowloss, tgt, loss, N, ignore)
    if (!at(comp, c, ops_.softmax_ce_fused, "the fused softmax cross-entropy", why)) return false;
    const int* ce = argp(comp[c]);
    const int dlogits = ce[4];
    {
      std::vector<std::pair<int, int>> rows;
      for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.softmax_ce_fused; ++c) {
        const int* a = argp(comp[c]);
        bool same = a[7] == ce[7] && a[8] == ce[8] && a[9] == ce[9] && a[10] == ce[10];
        for (int jj = 0; jj < 5; ++jj) same = same && a[jj] == ce[jj];
        if (!same) break;
        rows.push_back({a[5], a[6]}); take(s, comp[c]);
      }
      if (ce[0] != logits || ce[7] != s.V || ce[8] != -1 || ce[9] != s.N || afloat(ce[10]) != 1.f || !covers(rows, s.N)) { why = "fused softmax cross-entropy shape"; return false; }
    }
    if (!at(comp, c, ops_.softmax_ce_reduce, "the reduce after the fused softmax cross-entropy", why)) return false;
    const int* rd = argp(comp[c]);
    s.tgt = ce[1]; s.loss = rd[2];
    if (rd[0] != ce[2] || rd[1] != s.tgt || rd[3] != s.N || rd[4] != -1) { why = "cross-entropy reduce shape"; return false; }
    if (!buf_is(dlogits, s.N, s.V) || !buf_is(s.loss, 1, -1)) { why = "cross-entropy buffer shapes"; return false; }
    take(s, comp[c]); s.interior.insert({ce[2], ce[3], dlogits});
    ++c;

    // ---- backward: the head
    std::vector<int> grad_of_param(bufs_.size(), -1);    // parameter buffer -> its gradient buffer, filled as the backward is read
    auto gradient = [&](int param, int g, const char* what) {
      if (param < 0 || g < 0 || grad_of_param[param] != -1) { why = std::string("a second gradient for ") + what; return false; }
      grad_of_param[param] = g; s.interior.insert(g); return true;
    };
    Gemm gdh, gdx;
    if (!read_gemm(comp, c, gdh, why) || !grad_w(s, gdh, lnf.out, dlogits, s.d, s.V, "the head's weight gradient", why) || !gradient(s.head, gdh.out, "the head")) return false;
    take_all(s, gdh.tasks);
    if (!read_gemm(comp, c, gdx, why) || !grad_x(s, gdx, dlogits, s.head, s.V, s.d, false, -1, 0, "the head's input gradient", why)) return false;
    take_all(s, gdx.tasks); s.interior.insert(gdx.out);
    // the final LayerNorm's backward writes the residual-stream gradient G (acc 0); every later LayerNorm backward adds into it (acc 1)
    LnBwd bf;
    if (!read_ln_bwd(comp, c, x, gdx.out, lnf, -1, 0, s, bf, why) || !gradient(s.nfg, bf.dgamma, "the final LayerNorm's gain") || !gradient(s.nfb, bf.dbeta, "the final LayerNorm's bias")) return false;
    const int G = bf.dx;
    take_all(s, bf.tasks); s.interior.insert(G);
    for (int l = s.L - 1; l >= 0; --l) {
      const Fwd& f = fw[l];
      const KvmGptLayer& P = s.layers[l];
      const std::string lw = " of layer " + std::to_string(l);
      // fc2: its bias gradient (colsum of G), its weight gradient (u^T G), its input gradient through GELU' (G W2^T * gelu'(pre))
      int db2 = -1, db1 = -1;
      if (!read_colsum(comp, c, G, s.d, s, db2, why) || !gradient(P.b2, db2, ("fc2's bias" + lw).c_str())) return false;
      Gemm gw2, gdu, gw1, gdh2;
      if (!read_gemm(comp, c, gw2, why) || !grad_w(s, gw2, f.u, G, s.F, s.d, ("fc2's weight gradient" + lw).c_str(), why) || !gradient(P.w2, gw2.out, "fc2")) return false;
      if (!read_gemm(comp, c, gdu, why) || !grad_x(s, gdu, G, P.w2, s.d, s.F, false, f.pre, G_DGELU, ("fc2's input gradient" + lw).c_str(), why)) return false;
      take_all(s, gw2.tasks); take_all(s, gdu.tasks); s.interior.insert(gdu.out);
      // fc1: bias, weight, input gradients
      if (!read_colsum(comp, c, gdu.out, s.F, s, db1, why) || !gradient(P.b1, db1, ("fc1's bias" + lw).c_str())) return false;
      if (!read_gemm(comp, c, gw1, why) || !grad_w(s, gw1, f.h2, gdu.out, s.d, s.F, ("fc1's weight gradient" + lw).c_str(), why) || !gradient(P.w1, gw1.out, "fc1")) return false;
      if (!read_gemm(comp, c, gdh2, why) || !grad_x(s, gdh2, gdu.out, P.w1, s.F, s.d, false, -1, 0, ("fc1's input gradient" + lw).c_str(), why)) return false;
      take_all(s, gw1.tasks); take_all(s, gdh2.tasks); s.interior.insert(gdh2.out);
      // the second LayerNorm's backward adds into G
      LnBwd b2;
      Ln ln2{}; ln2.gamma = P.n2g; ln2.mean = f.mean2; ln2.rstd = f.rstd2;
      if (!read_ln_bwd(comp, c, f.xm, gdh2.out, ln2, G, 1, s, b2, why) || !gradient(P.n2g, b2.dgamma, "a LayerNorm gain") || !gradient(P.n2b, b2.dbeta, "a LayerNorm bias")) return false;
      take_all(s, b2.tasks);
      // proj: bias, weight, input gradients; then attention's backward
      int dbo = -1;
      if (!read_colsum(comp, c, G, s.d, s, dbo, why) || !gradient(P.bo, dbo, ("proj's bias" + lw).c_str())) return false;
      Gemm gwo, gdo;
      if (!read_gemm(comp, c, gwo, why) || !grad_w(s, gwo, f.o, G, s.d, s.d, ("proj's weight gradient" + lw).c_str(), why) || !gradient(P.wo, gwo.out, "proj")) return false;
      if (!read_gemm(comp, c, gdo, why) || !grad_x(s, gdo, G, P.wo, s.d, s.d, false, -1, 0, ("proj's input gradient" + lw).c_str(), why)) return false;
      take_all(s, gwo.tasks); take_all(s, gdo.tasks); s.interior.insert(gdo.out);
      AttnBwd ab;
      if (!read_attn_bwd(comp, c, f, gdo.out, s, ab, why)) return false;
      take_all(s, ab.tasks); s.interior.insert({ab.delta, ab.dq, ab.dk, ab.dv});
      // v: bias, weight, and the first term of dh (fresh); k and q: RoPE's backward, bias, weight, and dh += (accumulate)
      int dbv = -1, dbk = -1, dbq = -1;
      Gemm gwv, gdhv, gwk, gdhk, gwq, gdhq;
      if (!read_colsum(comp, c, ab.dv, s.d, s, dbv, why) || !gradient(P.bv, dbv, ("v's bias" + lw).c_str())) return false;
      if (!read_gemm(comp, c, gwv, why) || !grad_w(s, gwv, f.h, ab.dv, s.d, s.d, ("v's weight gradient" + lw).c_str(), why) || !gradient(P.wv, gwv.out, "v")) return false;
      if (!read_gemm(comp, c, gdhv, why) || !grad_x(s, gdhv, ab.dv, P.wv, s.d, s.d, false, -1, 0, ("v's input gradient" + lw).c_str(), why)) return false;
      const int dh = gdhv.out;
      take_all(s, gwv.tasks); take_all(s, gdhv.tasks); s.interior.insert(dh);
      Rope rbk, rbq;
      if (!read_rope(comp, c, ops_.rope_bwd, ab.dk, s, rbk, why)) return false;
      if (!read_colsum(comp, c, rbk.out, s.d, s, dbk, why) || !gradient(P.bk, dbk, ("k's bias" + lw).c_str())) return false;
      if (!read_gemm(comp, c, gwk, why) || !grad_w(s, gwk, f.h, rbk.out, s.d, s.d, ("k's weight gradient" + lw).c_str(), why) || !gradient(P.wk, gwk.out, "k")) return false;
      if (!read_gemm(comp, c, gdhk, why) || !grad_x(s, gdhk, rbk.out, P.wk, s.d, s.d, true, dh, 0, ("k's input gradient" + lw).c_str(), why)) return false;
      take_all(s, rbk.tasks); take_all(s, gwk.tasks); take_all(s, gdhk.tasks); s.interior.insert(rbk.out);
      if (!read_rope(comp, c, ops_.rope_bwd, ab.dq, s, rbq, why)) return false;
      if (!read_colsum(comp, c, rbq.out, s.d, s, dbq, why) || !gradient(P.bq, dbq, ("q's bias" + lw).c_str())) return false;
      if (!read_gemm(comp, c, gwq, why) || !grad_w(s, gwq, f.h, rbq.out, s.d, s.d, ("q's weight gradient" + lw).c_str(), why) || !gradient(P.wq, gwq.out, "q")) return false;
      if (!read_gemm(comp, c, gdhq, why) || !grad_x(s, gdhq, rbq.out, P.wq, s.d, s.d, true, dh, 0, ("q's input gradient" + lw).c_str(), why)) return false;
      take_all(s, rbq.tasks); take_all(s, gwq.tasks); take_all(s, gdhq.tasks); s.interior.insert(rbq.out);
      // the first LayerNorm's backward adds into G
      LnBwd b1n;
      Ln ln1{}; ln1.gamma = P.n1g; ln1.mean = f.mean1; ln1.rstd = f.rstd1;
      if (!read_ln_bwd(comp, c, f.x, dh, ln1, G, 1, s, b1n, why) || !gradient(P.n1g, b1n.dgamma, "a LayerNorm gain") || !gradient(P.n1b, b1n.dbeta, "a LayerNorm bias")) return false;
      take_all(s, b1n.tasks);
    }
    // ---- embedding_bwd (ids, dout, dtab, N, mod, v0, v1, d, acc), vocabulary blocks covering [0, V)
    {
      std::vector<std::pair<int, int>> blocks;
      int dtab = -1;
      for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.embedding_bwd; ++c) {
        const int* b = argp(comp[c]);
        if (dtab == -1) dtab = b[2];
        if (b[0] != s.ids || b[1] != G || b[2] != dtab || b[3] != s.N || b[4] != 0 || b[7] != s.d || b[8] != 0) { why = "embedding_bwd shape"; return false; }
        blocks.push_back({b[5], b[6]}); take(s, comp[c]);
      }
      if (dtab == -1) { why = "expected embedding_bwd"; return false; }
      if (!covers(blocks, s.V)) { why = "embedding_bwd's vocabulary blocks do not cover the table"; return false; }
      if (!buf_is(dtab, s.V, s.d) || !gradient(s.wte, dtab, "the embedding")) return false;
    }
    // ---- the grad-norm clip: gradnorm_partial (g, partial, slot, e0, e1) over every gradient, then gradnorm_finish (partial, nslots, out, max_norm)
    int partial = -1, slots = 0;
    std::map<int, std::vector<std::pair<int, int>>> norm_ranges;   // gradient -> the element ranges its partials cover
    std::set<int> slot_seen;
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.gradnorm_partial; ++c) {
      const int* p = argp(comp[c]);
      if (partial == -1) partial = p[1];
      if (p[1] != partial || !slot_seen.insert(p[2]).second) { why = "grad-norm partials do not share one partial buffer with distinct slots"; return false; }
      norm_ranges[p[0]].push_back({p[3], p[4]}); ++slots; take(s, comp[c]);
    }
    if (!at(comp, c, ops_.gradnorm_finish, "the grad-norm finish", why)) return false;
    {
      const int* fn = argp(comp[c]);
      if (fn[0] != partial || fn[1] != slots) { why = "the grad-norm finish does not read every partial"; return false; }
      for (int sl = 0; sl < slots; ++sl) if (!slot_seen.count(sl)) { why = "grad-norm slots are not 0..n-1"; return false; }
      s.coef = fn[2]; s.max_norm = afloat(fn[3]);
      take(s, comp[c]); s.interior.insert(partial); ++c;
    }
    // ---- the step counter (x, i0, i1, delta): one word, +1
    if (!at(comp, c, ops_.counter_inc, "the step counter", why)) return false;
    {
      const int* k = argp(comp[c]);
      if (k[1] != 0 || k[2] != 1 || k[3] != 1) { why = "the step counter is not one word advanced by one"; return false; }
      s.step_buf = k[0]; take(s, comp[c]); ++c;
    }
    const size_t pre_upd = joins_.size();
    c = skip(comp, c);                                   // the barrier before the updates (absent if ordering noops were dissolved)
    const size_t pre_upd_end = joins_.size();
    // ---- AdamW (p, g, m, v, gscale, step, e0, e1, lr, b1, b2, eps, wd): every parameter once, over its whole extent, scaled by the clip coefficient
    std::set<int> params;
    for (int l = 0; l < s.L; ++l) {
      const KvmGptLayer& P = s.layers[l];
      for (int b : {P.n1g, P.n1b, P.wq, P.bq, P.wk, P.bk, P.wv, P.bv, P.wo, P.bo, P.n2g, P.n2b, P.w1, P.b1, P.w2, P.b2}) params.insert(b);
    }
    for (int b : {s.wte, s.nfg, s.nfb, s.head}) params.insert(b);
    std::map<int, std::vector<std::pair<int, int>>> upd_ranges;
    std::map<int, KvmGptUpdate> upd;
    std::vector<int> order;
    for (;; ++c) {
      size_t k = c;                                       // joins inside the update block are the step's; a noop run that no update follows ends it
      while (k < comp.size() && is_join(comp[k])) ++k;
      if (k >= comp.size() || op(comp[k]) != ops_.adamw) break;
      while (c < k) joins_.push_back(comp[c++]);
      const int* a = argp(comp[c]);
      const int p = a[0];
      if (!params.count(p)) { why = "AdamW on a buffer that is not one of the model's parameters"; return false; }
      if (a[1] != grad_of_param[p]) { why = "AdamW does not pair a parameter with its gradient"; return false; }
      if (a[4] != s.coef || a[5] != s.step_buf) { why = "AdamW does not read the clip coefficient and the step counter"; return false; }
      const float lr = afloat(a[8]), b1 = afloat(a[9]), b2 = afloat(a[10]), eps = afloat(a[11]), wd = afloat(a[12]);
      if (s.lr == 0.f) { s.lr = lr; s.beta1 = b1; s.beta2 = b2; s.adam_eps = eps; }
      else if (lr != s.lr || b1 != s.beta1 || b2 != s.beta2 || eps != s.adam_eps) { why = "AdamW hyperparameters differ between parameters"; return false; }
      auto it = upd.find(p);
      if (it == upd.end()) { upd[p] = KvmGptUpdate{p, a[1], a[2], a[3], wd}; order.push_back(p); }
      else if (it->second.m != a[2] || it->second.v != a[3] || it->second.wd != wd) { why = "AdamW chunks of one parameter disagree"; return false; }
      upd_ranges[p].push_back({a[6], a[7]}); take(s, comp[c]);
    }
    if (upd.size() != params.size()) { why = "AdamW updates " + std::to_string(upd.size()) + " of the model's " + std::to_string(params.size()) + " parameters"; return false; }
    std::set<int> moments;
    for (int p : order) {
      const KvmGptUpdate& u = upd[p];
      const int n = bufs_[p].rows * bufs_[p].cols;
      if (!covers(upd_ranges[p], n)) { why = "an AdamW update does not cover its whole parameter"; return false; }
      if (!covers(norm_ranges[u.g], n)) { why = "the grad-norm partials do not cover a gradient exactly once"; return false; }
      if (!same_shape(u.m, p) || !same_shape(u.v, p) || !moments.insert(u.m).second || !moments.insert(u.v).second) { why = "AdamW moments are not distinct buffers shaped like their parameter"; return false; }
      s.updates.push_back(u);
    }
    if (norm_ranges.size() != params.size()) { why = "the grad-norm partials read a buffer that is not a gradient"; return false; }
    // the data windows: exactly the tokens and the targets, every row of each, at the step the counter holds
    if (wins.size() != 2 || !wins.count(s.ids) || !wins.count(s.tgt)) { why = "the data windows are not exactly the step's tokens and targets"; return false; }
    for (const auto& kv : wins)
      if (kv.second.step != s.step_buf || !covers(kv.second.rows, s.N)) { why = "a data window does not cover its rows at the step counter"; return false; }
    s.ids_src = wins[s.ids].src; s.ids_stride = wins[s.ids].stride; s.ids_off = wins[s.ids].off;
    s.tgt_src = wins[s.tgt].src; s.tgt_stride = wins[s.tgt].stride; s.tgt_off = wins[s.tgt].off;
    {                                                     // the trailing noops: every one but the last is a join; the last is the barrier that ends the step
      size_t k = c;
      while (k < comp.size() && is_join(comp[k])) ++k;
      if (k > c) { while (c + 1 < k) joins_.push_back(comp[c++]); s.barrier_end = comp[c++]; }
    }
    s.mm_numerics = numerics_ < 0 ? 0 : numerics_;
    {
      std::set<int> pre(joins_.begin() + pre_upd, joins_.begin() + pre_upd_end);
      auto pub = [&](int t, int& slot) {
        if (tasks_[t].pub_var == KVM_NO_VAR) return true;
        if (s.pub_var != -1 && tasks_[t].pub_var != s.pub_var) { why = "the step publishes two version variables"; return false; }
        if (slot != -1) { why = "the step publishes twice at one point"; return false; }
        s.pub_var = tasks_[t].pub_var; slot = tasks_[t].pub_ver; return true;
      };
      for (int t : pre) if (!pub(t, s.pub_pre)) return false;
      if (s.barrier_end >= 0 && !pub(s.barrier_end, s.pub_end)) return false;
      std::vector<int> all(s.replaced); all.insert(all.end(), joins_.begin(), joins_.end()); if (s.barrier_end >= 0) all.push_back(s.barrier_end);
      for (int t : all) {
        if (tasks_[t].wait_var != KVM_NO_VAR) { why = "a task of the step waits on a version"; return false; }
        if (tasks_[t].pub_var != KVM_NO_VAR && !pre.count(t) && t != s.barrier_end) { why = "a task other than the step's barriers publishes a version"; return false; }
      }
      if (s.pub_pre != -1 && s.pub_end != -1 && s.pub_end != s.pub_pre + 1) { why = "the step's two versions are not consecutive"; return false; }
    }
    take_all(s, joins_);
    pos = c;
    return true;
  }

 private:
  struct Ln { int x = -1, gamma = -1, beta = -1, out = -1, mean = -1, rstd = -1; float eps = 0.f; std::vector<int> tasks; };
  struct LnBwd { int dx = -1, dgamma = -1, dbeta = -1; std::vector<int> tasks; };
  struct Rope { int out = -1; std::vector<int> tasks; };
  struct Attn { int o = -1, lse = -1; std::vector<int> tasks; };
  struct AttnBwd { int delta = -1, dq = -1, dk = -1, dv = -1; std::vector<int> tasks; };

  int op(int t) const { return tasks_[t].op_id; }
  // An ordering noop the frontend puts between tasks of a step (a join of dependencies, no arguments): part of the step, taken with it.
  bool is_join(int t) const { return op(t) == ops_.noop && tasks_[t].args_words == 0; }
  size_t skip(const std::vector<int>& comp, size_t c) const { while (c < comp.size() && is_join(comp[c])) joins_.push_back(comp[c++]); return c; }
  const int* argp(int t) const { return args_.data() + tasks_[t].args_off; }
  static float afloat(int32_t w) { float f; memcpy(&f, &w, 4); return f; }
  static void take(KvmGptStep& s, int t) { s.replaced.push_back(t); }
  static void take_all(KvmGptStep& s, const std::vector<int>& ts) { s.replaced.insert(s.replaced.end(), ts.begin(), ts.end()); }
  bool at(const std::vector<int>& comp, size_t& c, int opid, const char* what, std::string& why) const {
    c = skip(comp, c);
    if (c < comp.size() && op(comp[c]) == opid) return true;
    why = std::string("expected ") + what;
    return false;
  }
  bool buf_is(int b, int rows, int cols) const {   // a dense f32 buffer of that shape (-1: any)
    if (b < 0 || b >= (int)bufs_.size() || bufs_[b].dtype != 0 || (bufs_[b].ld != bufs_[b].cols && bufs_[b].rows > 1)) return false;
    return (rows < 0 || bufs_[b].rows == rows) && (cols < 0 || bufs_[b].cols == cols);
  }
  bool same_shape(int a, int b) const { return buf_is(a, bufs_[b].rows, bufs_[b].cols); }
  // [lo, hi) intervals that tile [0, n) exactly once, in any order
  static bool covers(std::vector<std::pair<int, int>> iv, int n) {
    std::sort(iv.begin(), iv.end());
    int at = 0;
    for (auto& r : iv) { if (r.first != at || r.second <= r.first) return false; at = r.second; }
    return at == n;
  }
  // the GEMM numerics of the step: one setting on every tile, or the template does not hold (the fused step runs one)
  bool numerics(int bits, std::string& why) const {
    if (numerics_ < 0) numerics_ = bits;
    else if (numerics_ != bits) { why = "GEMM tiles with different numerics in one step"; return false; }
    return true;
  }
  // The contiguous tiles of one GEMM at comp[c]: every tile names the same GEMM, and their origins form the full grid of the tile size its flags select.
  bool read_gemm(const std::vector<int>& comp, size_t& c, Gemm& g, std::string& why) const {
    g = Gemm{};
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != ops_.matmul_tile) { why = "expected a GEMM"; return false; }
    const int* a0 = argp(comp[c]);
    const int fl = a0[11], M = a0[8], N = a0[9];
    std::set<std::pair<int, int>> tiles;
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.matmul_tile; ++c) {
      const int* a = argp(comp[c]);
      bool same = a[8] == M && a[9] == N && a[10] == a0[10] && a[11] == fl;
      for (int j = 0; j < 6; ++j) same = same && a[j] == a0[j];
      if (!same) break;                                   // the next GEMM starts here
      if (afloat(a[12]) != 1.f) { why = "a GEMM with alpha != 1"; return false; }
      if (!tiles.insert({a[6], a[7]}).second) { why = "a tile listed twice in a GEMM"; return false; }
      g.tasks.push_back(comp[c]);
    }
    if (fl & ~(G_ROLE | G_NUMERICS | G_TILE)) { why = "a GEMM with flags " + std::to_string(fl) + " the template does not know"; return false; }
    const int tm = (fl & G_HUGE) ? 256 : (fl & G_BIG) ? 128 : 64, tn = (fl & (G_HUGE | G_BIG)) ? 128 : 64;
    for (int m0 = 0; m0 < M; m0 += tm) for (int n0 = 0; n0 < N; n0 += tn) if (!tiles.count({m0, n0})) { why = "a GEMM's tiles do not cover its grid"; return false; }
    if ((int)tiles.size() != ((M + tm - 1) / tm) * ((N + tn - 1) / tn)) { why = "a GEMM has tiles outside its grid"; return false; }
    g.a = a0[0]; g.b = a0[1]; g.out = a0[2]; g.bias = a0[3]; g.res = a0[4]; g.pre = a0[5]; g.M = M; g.N = N; g.K = a0[10];
    g.role = fl & G_ROLE; g.numerics = fl & G_NUMERICS;
    return numerics(g.numerics, why);
  }
  // A forward linear layer: out = act(a W + bias) + res, a of N x K, W of K x n.
  bool lin(const KvmGptStep& s, const Gemm& g, int a, int K, int n, bool bias, int res, int role, const char* what, std::string& why) const {
    if (g.a != a || g.M != s.N || g.K != K || g.N != n || g.role != role || g.res != res || (g.pre != -1) != ((role & G_GELU) != 0)) { why = std::string(what) + " does not chain from its input"; return false; }
    if (!buf_is(g.b, K, n) || !buf_is(g.out, s.N, n) || (bias ? !buf_is(g.bias, 1, n) : g.bias != -1)) { why = std::string(what) + " buffer shapes"; return false; }
    return true;
  }
  // A weight gradient A^T dY: A of N x k, dY of N x n, written fresh (k x n).
  bool grad_w(const KvmGptStep& s, const Gemm& g, int a, int dy, int k, int n, const char* what, std::string& why) const {
    if (g.role != G_TA || g.a != a || g.b != dy || g.M != k || g.N != n || g.K != s.N || g.bias != -1 || g.res != -1 || g.pre != -1 || !buf_is(g.out, k, n)) { why = std::string(what) + " does not pair its operands"; return false; }
    return true;
  }
  // An input gradient dY W^T: dY of N x k, W of n x k, into out (N x n); `acc` adds into an earlier term `into`; `role` adds GELU'(pre).
  bool grad_x(const KvmGptStep& s, const Gemm& g, int dy, int w, int k, int n, bool acc, int into_or_pre, int role, const char* what, std::string& why) const {
    const int want = G_TB | (acc ? G_ACC : 0) | role;
    if (g.role != want || g.a != dy || g.b != w || g.M != s.N || g.N != n || g.K != k || g.bias != -1 || g.res != -1) { why = std::string(what) + " does not chain"; return false; }
    if ((role & G_DGELU) ? g.pre != into_or_pre : g.pre != -1) { why = std::string(what) + ": its GELU' epilogue does not read the layer's pre-activation"; return false; }
    if (acc && g.out != into_or_pre) { why = std::string(what) + " does not accumulate into the term before it"; return false; }
    if (!buf_is(g.out, s.N, n)) { why = std::string(what) + " buffer shape"; return false; }
    return true;
  }
  // LayerNorm forward (x, gamma, beta, out, mean, rstd, r0, r1, d, eps): row blocks of one norm, covering [0, N)
  bool read_ln(const std::vector<int>& comp, size_t& c, int x, Ln& ln, std::string& why) const {
    ln = Ln{};
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != ops_.layernorm_fwd) { why = "expected a LayerNorm"; return false; }
    const int* a0 = argp(comp[c]);
    std::vector<std::pair<int, int>> rows;
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.layernorm_fwd; ++c) {
      const int* a = argp(comp[c]);
      bool same = a[8] == a0[8] && a[9] == a0[9];
      for (int j = 0; j < 6; ++j) same = same && a[j] == a0[j];
      if (!same) break;
      rows.push_back({a[6], a[7]}); ln.tasks.push_back(comp[c]);
    }
    ln.x = a0[0]; ln.gamma = a0[1]; ln.beta = a0[2]; ln.out = a0[3]; ln.mean = a0[4]; ln.rstd = a0[5]; ln.eps = afloat(a0[9]);
    const int N = bufs_[x].rows, d = bufs_[x].cols;
    if (ln.x != x || a0[8] != d || !covers(rows, N)) { why = "a LayerNorm does not read its input over every row"; return false; }
    if (!buf_is(ln.gamma, 1, d) || !buf_is(ln.beta, 1, d) || !buf_is(ln.out, N, d) || !buf_is(ln.mean, N, 1) || !buf_is(ln.rstd, N, 1)) { why = "LayerNorm buffer shapes"; return false; }
    return true;
  }
  bool ln_params(KvmGptStep& s, const Ln& ln, bool first, std::string& why) const {
    if (first) s.ln_eps = ln.eps;
    else if (ln.eps != s.ln_eps) { why = "LayerNorm epsilons differ"; return false; }
    return true;
  }
  // LayerNorm backward: layernorm_bwd_dx (x, dy, gamma, mean, rstd, dx, r0, r1, d, acc) row blocks, then norm_bwd_params
  // (x, dy, mean, rstd, dgamma, dbeta, R, d, c0, c1, acc) column blocks. dx is `into` when acc is 1 (the residual-stream gradient), any buffer when 0.
  bool read_ln_bwd(const std::vector<int>& comp, size_t& c, int x, int dy, const Ln& ln, int into, int acc, const KvmGptStep& s, LnBwd& o, std::string& why) const {
    o = LnBwd{};
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != ops_.layernorm_bwd_dx) { why = "expected a LayerNorm backward"; return false; }
    std::vector<std::pair<int, int>> rows, cols;
    const int* a0 = argp(comp[c]);
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.layernorm_bwd_dx; ++c) {
      const int* a = argp(comp[c]);
      bool same = a[8] == a0[8] && a[9] == a0[9];
      for (int j = 0; j < 6; ++j) same = same && a[j] == a0[j];
      if (!same) break;
      rows.push_back({a[6], a[7]}); o.tasks.push_back(comp[c]);
    }
    o.dx = a0[5];
    if (a0[0] != x || a0[1] != dy || a0[2] != ln.gamma || a0[3] != ln.mean || a0[4] != ln.rstd || a0[8] != s.d || a0[9] != acc || (acc && o.dx != into) || !covers(rows, s.N)) {
      why = "a LayerNorm backward does not chain from its forward"; return false;
    }
    if (!buf_is(o.dx, s.N, s.d)) { why = "LayerNorm backward dx shape"; return false; }
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != ops_.norm_bwd_params) { why = "expected the LayerNorm's parameter gradients"; return false; }
    const int* p0 = argp(comp[c]);
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.norm_bwd_params; ++c) {
      const int* p = argp(comp[c]);
      bool same = p[6] == p0[6] && p[7] == p0[7] && p[10] == p0[10];
      for (int j = 0; j < 6; ++j) same = same && p[j] == p0[j];
      if (!same) break;
      cols.push_back({p[8], p[9]}); o.tasks.push_back(comp[c]);
    }
    o.dgamma = p0[4]; o.dbeta = p0[5];
    if (p0[0] != x || p0[1] != dy || p0[2] != ln.mean || p0[3] != ln.rstd || p0[6] != s.N || p0[7] != s.d || p0[10] != 0 || !covers(cols, s.d)) { why = "LayerNorm parameter gradients do not chain"; return false; }
    if (!buf_is(o.dgamma, 1, s.d) || !buf_is(o.dbeta, 1, s.d)) { why = "LayerNorm parameter gradient shapes"; return false; }
    return true;
  }
  // RoPE forward or backward (x, out, pos, r0, r1, T, H, hd, theta): row blocks covering [0, N); fixes H, hd and theta at the first one
  bool read_rope(const std::vector<int>& comp, size_t& c, int opid, int x, KvmGptStep& s, Rope& r, std::string& why) const {
    r = Rope{};
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != opid) { why = "expected RoPE"; return false; }
    const int* a0 = argp(comp[c]);
    std::vector<std::pair<int, int>> rows;
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == opid; ++c) {
      const int* a = argp(comp[c]);
      bool same = a[0] == a0[0] && a[1] == a0[1] && a[2] == a0[2] && a[5] == a0[5] && a[6] == a0[6] && a[7] == a0[7] && a[8] == a0[8];
      if (!same) break;
      rows.push_back({a[3], a[4]}); r.tasks.push_back(comp[c]);
    }
    r.out = a0[1];
    if (s.H == 0) { s.H = a0[6]; s.hd = a0[7]; s.theta = afloat(a0[8]); }
    if (a0[0] != x || a0[2] != -1 || a0[5] != s.T || a0[6] != s.H || a0[7] != s.hd || afloat(a0[8]) != s.theta || s.H * s.hd != s.d || !covers(rows, s.N)) { why = "RoPE does not cover its input with the model's heads"; return false; }
    if (!buf_is(r.out, s.N, s.d)) { why = "RoPE output shape"; return false; }
    return true;
  }
  // (b, h, block) triples of one attention op: every sequence, every head, query (or key) blocks tiling [0, T)
  bool attn_cover(const std::vector<std::array<int, 3>>& u, const KvmGptStep& s, std::string& why, const char* what) const {
    std::map<std::pair<int, int>, std::vector<int>> starts;
    for (auto& t : u) {
      if (t[0] < 0 || t[0] >= s.B || t[1] < 0 || t[1] >= s.H || t[2] < 0 || t[2] >= s.T) { why = std::string(what) + " outside the batch, the heads or the sequence"; return false; }
      starts[{t[0], t[1]}].push_back(t[2]);
    }
    if ((int)starts.size() != s.B * s.H) { why = std::string(what) + " does not cover every (sequence, head)"; return false; }
    std::vector<int> ref;
    for (auto& kv : starts) {
      std::vector<int> v = kv.second; std::sort(v.begin(), v.end());
      if (v.empty() || v[0] != 0 || std::adjacent_find(v.begin(), v.end()) != v.end()) { why = std::string(what) + " blocks of a (sequence, head) do not start at 0 once each"; return false; }
      if (ref.empty()) ref = v; else if (v != ref) { why = std::string(what) + " blocks differ between (sequence, head) pairs"; return false; }
    }
    for (size_t i = 1; i < ref.size(); ++i) if (ref[i] - ref[i - 1] != ref[1] - ref[0]) { why = std::string(what) + " blocks are not evenly spaced"; return false; }
    if (ref.size() > 1 && ref.back() + (ref[1] - ref[0]) < s.T) { why = std::string(what) + " blocks stop before the sequence ends"; return false; }
    return true;
  }
  // attn_fwd (q, k, v, o, lse, b, h, q0, T, H, Hkv, hd, scale): causal, one KV head per head
  bool read_attn_fwd(const std::vector<int>& comp, size_t& c, int q, int k, int v, KvmGptStep& s, Attn& at, std::string& why) const {
    at = Attn{};
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != ops_.attn_fwd) { why = "expected causal attention"; return false; }
    const int* a0 = argp(comp[c]);
    std::vector<std::array<int, 3>> u;
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.attn_fwd; ++c) {
      const int* a = argp(comp[c]);
      bool same = a[8] == a0[8] && a[9] == a0[9] && a[10] == a0[10] && a[11] == a0[11] && a[12] == a0[12];
      for (int j = 0; j < 5; ++j) same = same && a[j] == a0[j];
      if (!same) break;
      u.push_back({a[5], a[6], a[7]}); at.tasks.push_back(comp[c]);
    }
    at.o = a0[3]; at.lse = a0[4];
    if (s.scale == 0.f) s.scale = afloat(a0[12]);
    if (a0[0] != q || a0[1] != k || a0[2] != v || a0[8] != s.T || a0[9] != s.H || a0[10] != s.H || a0[11] != s.hd || afloat(a0[12]) != s.scale) { why = "attention does not read the layer's q, k and v with the model's heads"; return false; }
    if (!buf_is(at.o, s.N, s.d) || !buf_is(at.lse, -1, -1) || bufs_[at.lse].rows * bufs_[at.lse].cols != s.N * s.H) { why = "attention output shapes"; return false; }
    return attn_cover(u, s, why, "attention");
  }
  // the attention backward: attn_bwd_delta (o, do, delta, r0, r1, T, H, hd) row blocks, attn_bwd_dq (q, k, v, do, lse, delta, dq, b, h, q0, T, H, Hkv, hd, scale),
  // attn_bwd_dkv (q, k, v, do, lse, delta, dk, dv, b, kvh, k0, T, H, hd, scale)
  template <class F>
  bool read_attn_bwd(const std::vector<int>& comp, size_t& c, const F& f, int dout, const KvmGptStep& s, AttnBwd& o, std::string& why) const {
    o = AttnBwd{};
    std::vector<std::pair<int, int>> rows;
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != ops_.attn_bwd_delta) { why = "expected the attention backward's delta"; return false; }
    const int* d0 = argp(comp[c]);
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.attn_bwd_delta; ++c) {
      const int* a = argp(comp[c]);
      if (a[0] != d0[0] || a[1] != d0[1] || a[2] != d0[2] || a[5] != d0[5] || a[6] != d0[6] || a[7] != d0[7]) break;
      rows.push_back({a[3], a[4]}); o.tasks.push_back(comp[c]);
    }
    o.delta = d0[2];
    if (d0[0] != f.o || d0[1] != dout || d0[5] != s.T || d0[6] != s.H || d0[7] != s.hd || !covers(rows, s.N)) {
      why = "the attention backward's delta does not chain (o " + std::to_string(d0[0]) + " for " + std::to_string(f.o) + ", do " + std::to_string(d0[1]) + " for " +
            std::to_string(dout) + ", rows cover " + (covers(rows, s.N) ? "all" : "not all") + ")";
      return false;
    }
    // dq (q, k, v, do, lse, delta, dq, b, h, q0, T, H, Hkv, hd, scale) and dkv (q, k, v, do, lse, delta, dk, dv, b, kvh, k0, T, H, hd, scale) tasks, which the
    // frontend may interleave (per sequence: its dq blocks, then its dk/dv blocks): one group, each op with its own identity and its own coverage
    const int* q0 = nullptr, *k0 = nullptr;
    std::vector<std::array<int, 3>> uq, ukv;
    for (;;) {
      c = skip(comp, c);
      if (c >= comp.size()) break;
      const int* a = argp(comp[c]);
      if (op(comp[c]) == ops_.attn_bwd_dq) {
        if (!q0) q0 = a;
        bool same = a[10] == q0[10] && a[11] == q0[11] && a[12] == q0[12] && a[13] == q0[13] && a[14] == q0[14];
        for (int j = 0; j < 7; ++j) same = same && a[j] == q0[j];
        if (!same) break;
        uq.push_back({a[7], a[8], a[9]});
      } else if (op(comp[c]) == ops_.attn_bwd_dkv) {
        if (!k0) k0 = a;
        bool same = a[11] == k0[11] && a[12] == k0[12] && a[13] == k0[13] && a[14] == k0[14];
        for (int j = 0; j < 8; ++j) same = same && a[j] == k0[j];
        if (!same) break;
        ukv.push_back({a[8], a[9], a[10]});
      } else {
        break;
      }
      o.tasks.push_back(comp[c]); ++c;
    }
    if (!q0 || !k0) { why = q0 ? "expected the attention backward's dk and dv" : "expected the attention backward's dq"; return false; }
    o.dq = q0[6]; o.dk = k0[6]; o.dv = k0[7];
    if (q0[0] != f.qr || q0[1] != f.kr || q0[2] != f.v || q0[3] != dout || q0[4] != f.lse || q0[5] != o.delta || q0[10] != s.T || q0[11] != s.H || q0[12] != s.H || q0[13] != s.hd || afloat(q0[14]) != s.scale) {
      why = "the attention backward's dq does not chain"; return false;
    }
    if (k0[0] != f.qr || k0[1] != f.kr || k0[2] != f.v || k0[3] != dout || k0[4] != f.lse || k0[5] != o.delta || k0[11] != s.T || k0[12] != s.H || k0[13] != s.hd || afloat(k0[14]) != s.scale) {
      why = "the attention backward's dk and dv do not chain"; return false;
    }
    if (!attn_cover(uq, s, why, "the attention backward's dq") || !attn_cover(ukv, s, why, "the attention backward's dk and dv")) return false;
    for (int b : {o.dq, o.dk, o.dv}) if (!buf_is(b, s.N, s.d)) { why = "attention gradient shapes"; return false; }
    return true;
  }
  // colsum (x, out, R, c0, c1, acc): a bias gradient, column blocks covering [0, cols), written fresh
  bool read_colsum(const std::vector<int>& comp, size_t& c, int x, int cols, KvmGptStep& s, int& out, std::string& why) const {
    c = skip(comp, c);
    if (c >= comp.size() || op(comp[c]) != ops_.colsum) { why = "expected a bias gradient (colsum)"; return false; }
    const int* a0 = argp(comp[c]);
    std::vector<std::pair<int, int>> blocks;
    std::vector<int> ts;
    for (; (c = skip(comp, c)) < comp.size() && op(comp[c]) == ops_.colsum; ++c) {
      const int* a = argp(comp[c]);
      if (a[0] != a0[0] || a[1] != a0[1] || a[2] != a0[2] || a[5] != a0[5]) break;
      blocks.push_back({a[3], a[4]}); ts.push_back(comp[c]);
    }
    if (a0[0] != x || a0[2] != s.N || a0[5] != 0 || !covers(blocks, cols) || !buf_is(a0[1], 1, cols)) { why = "a bias gradient does not sum its gradient's columns"; return false; }
    out = a0[1];
    take_all(s, ts);
    return true;
  }

  const std::vector<KvmTaskDesc>& tasks_;
  const std::vector<int32_t>& args_;
  const std::vector<KvmFuseBuf>& bufs_;
  KvmFuseGptOps ops_;
  mutable int numerics_ = -1;
  mutable std::vector<int> joins_;   // the step's ordering noops, taken into it at the end of parse
};

}  // namespace kvm_fuse

// The scan: describe every run (connected component) whose steps all match the GPT template, every step the same model on the same
// parameters; the others are reported with the first reason they did not match.
inline KvmFuseGptResult kvm_fuse_gpt_scan(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& succ, const std::vector<int32_t>& args,
                                          const std::vector<KvmFuseBuf>& bufs, const KvmFuseGptOps& ops) {
  KvmFuseGptResult R;
  const kvm_fuse::GptParser parser(tasks, args, bufs, ops);
  const auto comps = kvm_fuse_components(tasks, succ);
  for (size_t ci = 0; ci < comps.size(); ++ci) {
    const std::vector<int>& comp = comps[ci];
    KvmFuseGptRun run; run.component = (int)ci;
    std::string why;
    size_t pos = 0;
    bool ok = true;
    while (pos < comp.size()) {
      KvmGptStep s;
      const size_t at = pos;
      if (!parser.parse(comp, pos, s, why)) { why = "step " + std::to_string(run.steps) + " (task " + std::to_string(comp[at]) + "): " + why; ok = false; break; }
      run.losses.push_back(s.loss);
      {
        const int first = s.pub_pre != -1 ? s.pub_pre : s.pub_end;
        const int per = (s.pub_pre != -1) + (s.pub_end != -1);
        if (run.steps == 0) run.pub_first = first;
        else if (s.pub_var != run.shape.pub_var || (s.pub_pre != -1) != (run.shape.pub_pre != -1) || (s.pub_end != -1) != (run.shape.pub_end != -1) ||
                 (per && first != run.pub_first + per * run.steps)) {
          why = "step " + std::to_string(run.steps) + "'s versions do not continue step 0's"; ok = false; break;
        }
      }
      if (run.steps == 0) run.shape = s;
      else {
        const KvmGptStep& f = run.shape;
        bool same = s.N == f.N && s.T == f.T && s.d == f.d && s.H == f.H && s.F == f.F && s.V == f.V && s.L == f.L && s.wte == f.wte && s.head == f.head &&
                    s.nfg == f.nfg && s.nfb == f.nfb && s.updates.size() == f.updates.size() && s.lr == f.lr && s.max_norm == f.max_norm && s.mm_numerics == f.mm_numerics &&
                    s.ids_src == f.ids_src && s.ids_stride == f.ids_stride && s.ids_off == f.ids_off && s.tgt_src == f.tgt_src && s.tgt_stride == f.tgt_stride &&
                    s.tgt_off == f.tgt_off && s.step_buf == f.step_buf;
        for (size_t i = 0; same && i < s.updates.size(); ++i) same = s.updates[i].p == f.updates[i].p && s.updates[i].m == f.updates[i].m && s.updates[i].v == f.updates[i].v && s.updates[i].wd == f.updates[i].wd;
        if (!same) { why = "step " + std::to_string(run.steps) + " differs from step 0 in shape or parameters"; ok = false; break; }
      }
      ++run.steps;
    }
    const int ntasks = (int)comp.size();
    if (ok && run.steps > 0) {
      const KvmGptStep& s = run.shape;
      char line[400];
      snprintf(line, sizeof line, "component %zu: GPT run of %d steps, %d tasks: L %d, d %d, heads %d x %d, MLP %d, vocabulary %d, %d sequences of %d (N %d), %zu parameters, %zu tasks a step, GEMM numerics %d",
               ci, run.steps, ntasks, s.L, s.d, s.H, s.hd, s.F, s.V, s.B, s.T, s.N, s.updates.size(), s.replaced.size(), s.mm_numerics);
      R.report.push_back(line); R.runs.push_back(run); ++R.matched_runs;
    } else {
      R.report.push_back("component " + std::to_string(ci) + ": unchanged (" + std::to_string(ntasks) + " tasks): " + (why.empty() ? "no step" : why));
      ++R.unchanged_runs;
    }
  }
  return R;
}

struct KvmFuseGptOptions {
  int smem_floats = KVM_FGPT_SMEM_FLOATS;   // shared memory a gang member may use (the device's budget for the fused step's kernel)
  int f2d_smem_floats = KVM_FGPT_SMEM_FLOATS;   // the same for F's kernel (its own static shared memory), so each form is judged by its own
  int workers = 108;                        // workers the run has: a gang wider than them cannot start together
  std::vector<int> keep;                    // buffers read after the run (a dump): a run that would leave one of them stale is not fused
};

// The rewritten bundle, and what each fused run's gang needs (runs[i] is the run of descriptor i, the `desc` argument of its members).
struct KvmFuseGptPlan {
  std::vector<KvmTaskDesc> tasks;
  std::vector<int32_t> succ, args;
  std::vector<KvmFuseGptRun> runs;
  std::vector<int> gang;                    // each fused run's member count G
  std::vector<int> rows;                    // each fused run's row chunk for its N-wide phases (kvm_fgpt_fit: N, or 128 or 64 below it; 0 in F's form)
  std::vector<kvm_f2d::Fit> f2d;            // each fused run's 2-D form (kvm_f2d::fit); its G is 0 for a run in the fused step's form
  std::vector<std::string> report;          // one line per component
  int fused_runs = 0, unchanged_runs = 0, replaced_tasks = 0, new_tasks = 0;
};

// The pass: every matched run that fits a gang is replaced, all of its tasks (its data windows included: the gang reads the streams), by the G
// members of one fgpt_step gang; every other task is copied through with its arguments, its successors renumbered. Runs are connected
// components, so no edge crosses a fused run's boundary and the members need no predecessors.
inline KvmFuseGptPlan kvm_fuse_gpt(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& succ, const std::vector<int32_t>& args,
                                   const std::vector<KvmFuseBuf>& bufs, const KvmFuseGptOps& ops, int fgpt_step_op, const KvmFuseGptOptions& opt) {
  KvmFuseGptPlan P;
  const KvmFuseGptResult scan = kvm_fuse_gpt_scan(tasks, succ, args, bufs, ops);
  const auto comps = kvm_fuse_components(tasks, succ);
  std::vector<int> run_of_comp(comps.size(), -1);
  std::vector<std::string> line(comps.size());
  for (size_t ci = 0; ci < comps.size() && ci < scan.report.size(); ++ci) line[ci] = scan.report[ci];
  const std::set<int> keep(opt.keep.begin(), opt.keep.end());
  for (const KvmFuseGptRun& r : scan.runs) {
    const KvmGptStep& s = r.shape;
    std::string why;
    const KvmFgptFit fit = fgpt_step_op < 0 ? KvmFgptFit{} : kvm_fgpt_fit(s.N, s.T, s.d, s.H, s.F, s.V, opt.smem_floats, opt.workers);
    const std::string unrun = kvm_fgpt_unrunnable(s.d, s.H, s.L, s.mm_numerics);
    // where no gang of the fused step's form fits, F's 2-D form may: each half's own layout keeps a member's tiles small
    kvm_f2d::Fit f2d;
    const char* f2d_why = nullptr;
    if (fgpt_step_op >= 0 && unrun.empty() && s.L <= KVM_FGPT_MAXL && !fit.G)
      f2d_why = kvm_f2d::fit(s.N, s.T, s.B, s.d, s.H, s.F, s.V, s.L, s.mm_numerics, opt.f2d_smem_floats, opt.workers, f2d);
    const int G = fit.G ? fit.G : f2d.G;
    if (fgpt_step_op < 0) why = "the registry has no fgpt_step op";
    else if (!unrun.empty()) why = "the fused step cannot run it: " + unrun;
    else if (s.L > KVM_FGPT_MAXL) why = "does not fit a gang: " + std::to_string(s.L) + " layers, more than the " + std::to_string(KVM_FGPT_MAXL) + " the fused step's scratch holds";
    else if (!G) why = "does not fit a gang: no gang of at most " + std::to_string(opt.workers) + " members fits its phases in " + std::to_string(opt.smem_floats) +
                       " floats of shared memory a member, nor F's 2-D form (" + (f2d_why ? f2d_why : "not tried") + ")";
    if (why.empty()) {   // the gang writes the parameters, moments, gradients, step counter and losses; activations and the windows' copies go stale
      std::set<int> written;
      for (const KvmGptUpdate& u : s.updates) written.insert({u.p, u.g, u.m, u.v});
      written.insert(s.step_buf); written.insert(r.losses.begin(), r.losses.end());
      std::set<int> stale(s.interior.begin(), s.interior.end());
      stale.insert({s.ids, s.tgt, s.coef});
      for (int b : keep) if (stale.count(b) && !written.count(b)) { why = "buffer " + std::to_string(b) + " is read after the run, and the fused step leaves it stale"; break; }
    }
    char head[96];
    snprintf(head, sizeof head, "component %d: ", r.component);
    if (!why.empty()) { line[r.component] = head + std::string("unchanged: a GPT run, but ") + why; continue; }
    run_of_comp[r.component] = (int)P.runs.size();
    P.runs.push_back(r); P.gang.push_back(G); P.rows.push_back(fit.G ? fit.rows : 0); P.f2d.push_back(fit.G ? kvm_f2d::Fit{} : f2d);
    line[r.component] += ", fused as a gang of " + std::to_string(G);
    if (fit.G && fit.rows < s.N) line[r.component] += " (its N-wide phases in chunks of " + std::to_string(fit.rows) + " of " + std::to_string(s.N) + " rows)";
    if (!fit.G) {
      char form[160];
      snprintf(form, sizeof form, " in F's 2-D form (attention %d x %d, MLP %d x %d, %d query block(s) a unit, %lld floats of shared memory a member)",
               f2d.gta, f2d.gca, f2d.gtm, f2d.gcm, f2d.qb, f2d.smem_floats);
      line[r.component] += form;
    }
    P.replaced_tasks += (int)comps[r.component].size(); P.new_tasks += G;
  }
  for (size_t ci = 0; ci < comps.size(); ++ci) {
    P.report.push_back(line[ci]);
    if (run_of_comp[ci] >= 0) ++P.fused_runs; else ++P.unchanged_runs;
  }
  // the new bundle: every kept task in its order, then each fused run's members
  std::vector<int> new_of(tasks.size(), -1);
  std::vector<char> kept(tasks.size(), 1);
  for (size_t ci = 0; ci < comps.size(); ++ci) if (run_of_comp[ci] >= 0) for (int t : comps[ci]) kept[t] = 0;
  int n = 0;
  for (size_t t = 0; t < tasks.size(); ++t) if (kept[t]) new_of[t] = n++;
  for (size_t t = 0; t < tasks.size(); ++t) {
    if (!kept[t]) continue;
    KvmTaskDesc d = tasks[t];
    d.args_off = (int)P.args.size();
    P.args.insert(P.args.end(), args.begin() + tasks[t].args_off, args.begin() + tasks[t].args_off + tasks[t].args_words);
    d.succ_off = (int)P.succ.size();
    for (int k = 0; k < tasks[t].succ_cnt; ++k) P.succ.push_back(new_of[succ[tasks[t].succ_off + k]]);   // a kept task's successors are kept: same component
    P.tasks.push_back(d);
  }
  for (size_t ri = 0; ri < P.runs.size(); ++ri) {
    const int G = P.gang[ri];
    const int wl = tasks[comps[P.runs[ri].component][0]].workload_id;
    for (int m = 0; m < G; ++m) {
      KvmTaskDesc d{};
      d.op_id = fgpt_step_op; d.deadline_us = -1; d.wait_var = KVM_NO_VAR; d.pub_var = KVM_NO_VAR; d.workload_id = wl;
      const KvmFuseGptRun& run = P.runs[ri];
      if (m == 0 && run.shape.pub_var != -1) {   // the run's last version: the fused path takes a bundle's final versions from its publishers
        const int per = (run.shape.pub_pre != -1) + (run.shape.pub_end != -1);
        d.pub_var = run.shape.pub_var; d.pub_ver = run.pub_first + per * run.steps - 1;
      }
      d.args_off = (int)P.args.size(); d.args_words = 4;
      P.args.insert(P.args.end(), {(int32_t)ri, (int32_t)ri, (int32_t)m, (int32_t)G});   // desc, gang, member, members
      d.succ_off = (int)P.succ.size(); d.succ_cnt = 0; d.indegree = 0;
      P.tasks.push_back(d);
    }
  }
  return P;
}
