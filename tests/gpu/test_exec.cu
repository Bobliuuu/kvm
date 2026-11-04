// GPU executor tests. Build (tests need per-op IEEE arithmetic to compare bitwise against the CPU reference):
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -DKVM_EXEC_NO_MAIN tests/gpu/test_exec.cu -o test_exec
//   ./test_exec [suite...]     suites: ops errors dag sched admission versions checker bench smoke   (default: all but bench)
//   ./test_exec --expect-fail versions    # negative test: exits 0 only if the suite DETECTS a violation (use with -DKVM_BREAK_FENCE)
#include "../../kvm_exec.cu"
#include <array>
#include <functional>
#include <random>

static int g_fail = 0, g_checks = 0;
static bool g_mute = false;  // the checker suite injects faults on purpose
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; if (!g_mute) fprintf(stderr, "  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #c); if (!g_mute) { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)
using Bufs = std::vector<std::vector<float>>;
static bool close_rel(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol * std::max(1.f, std::fabs(b)); }
static bool same_bits(float a, float b) { return memcmp(&a, &b, 4) == 0; }
static float asf(int32_t w) { float f; memcpy(&f, &w, 4); return f; }

// ---------------- CPU reference of every operator (replicates the device accumulation order) ----------------
static void cpu_op(const KvmBundle& bd, int ti, Bufs& B) {
  const KvmTaskDesc& t = bd.tasks[ti];
  const int32_t* a = bd.args.data() + t.args_off;
  auto buf = [&](int j) -> std::vector<float>& { return B[a[j]]; };
  switch (t.op_id) {
    case KVM_OP_noop: case KVM_OP_spin: break;
    case KVM_OP_copy: for (int i = 0; i < a[2]; ++i) buf(1)[i] = buf(0)[i]; break;
    case KVM_OP_add: for (int i = 0; i < a[3]; ++i) buf(2)[i] = buf(0)[i] + buf(1)[i]; break;
    case KVM_OP_scale: for (int i = 0; i < a[3]; ++i) buf(1)[i] = asf(a[2]) * buf(0)[i]; break;
    case KVM_OP_relu: for (int i = 0; i < a[2]; ++i) buf(1)[i] = buf(0)[i] > 0.f ? buf(0)[i] : 0.f; break;
    case KVM_OP_sgd: for (int i = 0; i < a[3]; ++i) buf(0)[i] = buf(0)[i] - asf(a[2]) * buf(1)[i]; break;
    case KVM_OP_predict: for (int i = 0; i < a[3]; ++i) buf(2)[i] = buf(1)[0] * buf(0)[i]; break;
    case KVM_OP_mse_grad: {
      int n = a[5]; float pl[KVM_THREADS] = {0}, pg[KVM_THREADS] = {0};
      for (int i = 0; i < n; ++i) { float d = buf(0)[i] - buf(1)[i]; pl[i % KVM_THREADS] += d * d; pg[i % KVM_THREADS] += (2.f * d) * buf(2)[i]; }
      for (int s = KVM_THREADS / 2; s > 0; s >>= 1) for (int k = 0; k < s; ++k) { pl[k] += pl[k + s]; pg[k] += pg[k + s]; }
      buf(3)[0] = pl[0] / (float)n; buf(4)[0] = pg[0] / (float)n; break;
    }
    case KVM_OP_matmul: break;  // needs buffer shapes: the ops suite computes its matmul references itself
    case KVM_OP_fill: for (int i = 0; i < a[2]; ++i) buf(0)[i] = asf(a[1]); break;
    default: fprintf(stderr, "cpu_op: no reference for op %d\n", t.op_id); ++g_fail;
  }
}

// ---------------- helpers ----------------
struct Rng { std::mt19937 g; explicit Rng(unsigned s) : g(s) {} int in(int lo, int hi) { return lo + (int)(g() % (unsigned)(hi - lo + 1)); } bool p(int pct) { return in(0, 99) < pct; } };
static KvmConfig small_cfg(int blocks = 4) { KvmConfig c; c.blocks = blocks; c.max_tasks = 1 << 15; c.live_cap = 1 << 12; c.ring_words = 1 << 14; c.timeout_us = 20ull * 1000000; return c; }
static void upload(float* d, const std::vector<float>& v) { CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); }
static std::vector<float> download(const float* d, size_t n) { std::vector<float> v(n); CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v; }
static std::vector<float> rand_ints(Rng& r, size_t n, int lo = -8, int hi = 8) { std::vector<float> v(n); for (auto& x : v) x = (float)r.in(lo, hi); return v; }
static std::vector<float> rand_fp(Rng& r, size_t n) { std::vector<float> v(n); for (auto& x : v) x = (float)r.in(-1000, 1000) / 128.f + 0.00137f * r.in(0, 9); return v; }
static KvmArg Bf(int i) { return KvmArg::buf(i); }
static KvmArg I(int v) { return KvmArg::i32(v); }
static KvmArg F(float v) { return KvmArg::f32(v); }

static bool run_ok(KvmExec& ex, const std::vector<KvmBundle>& bs, int timeout_ms = 10000) {
  ex.start();
  for (auto& b : bs) { KvmSubmit s = ex.submit(b, timeout_ms); if (s != KvmSubmit::OK) { CHECK(false, "submit: %s %s", kvm_submit_name(s), ex.last_error().c_str()); ex.close(); ex.finish(); return false; } }
  const KvmResult& r = ex.finish();
  CHECK(r.ok(), "device error %s", kvm_err_name(r.err_code));
  return r.ok();
}

// Exactly-once, count, and ordering checks over a finished run. Bundles are identified by workload_id (== index in `bs`).
static void check_trace(const KvmResult& r, const std::vector<KvmBundle>& bs, bool topo = true) {
  size_t total = 0; for (auto& b : bs) total += b.tasks.size();
  CHECK(r.done == (int)total && r.admitted == (int)total, "done %d admitted %d expected %zu", r.done, r.admitted, total);
  int bad_once = 0, bad_time = 0;
  for (size_t i = 0; i < r.trace.size(); ++i) {
    if (r.trace[i].exec_count != 1) ++bad_once;
    if (r.trace[i].end_ns < r.trace[i].start_ns || r.trace[i].start_ns < r.trace[i].ready_ns) ++bad_time;
  }
  CHECK(bad_once == 0, "%d tasks with exec_count != 1", bad_once);
  CHECK(bad_time == 0, "%d tasks with ready > start or start > end", bad_time);
  if (!topo) return;
  std::vector<int> base(bs.size(), -1);
  for (size_t i = 0; i < r.tasks.size(); ++i) { int w = r.tasks[i].workload_id; if (base[w] < 0) base[w] = (int)i; }
  int viol = 0;
  for (size_t w = 0; w < bs.size(); ++w)
    for (size_t i = 0; i < bs[w].tasks.size(); ++i)
      for (int k = 0; k < bs[w].tasks[i].succ_cnt; ++k) {
        int u = base[w] + (int)i, v = base[w] + bs[w].succ[bs[w].tasks[i].succ_off + k];
        if (r.trace[u].end_ns > r.trace[v].start_ns) ++viol;
      }
  CHECK(viol == 0, "%d dependency edges ran out of order", viol);
}

// ================= ops =================
static bool suite_ops() {
  Rng rg(7);
  KvmExec ex(small_cfg());
  KvmBundleBuilder bb; Bufs host;
  std::vector<std::pair<float*, std::vector<float>>> inits;
  auto newbuf = [&](int rows, int cols, std::vector<float> init) {
    float* p = ex.arena((size_t)rows * cols); int idx = ex.add_view(p, rows, cols);
    if (init.empty()) init.assign((size_t)rows * cols, 0.f);
    inits.push_back({p, init}); host.push_back(init); return idx;
  };
  auto task = [&](const char* op, std::vector<KvmArg> a) { return bb.add(op, a); };
  struct Pending { const char* name; int out; bool exact; };
  std::vector<Pending> pend;
  // elementwise ops at awkward sizes, integer-valued data (bitwise) and fp data (1e-5 rel)
  for (bool fp : {false, true})
    for (int n : {1, 31, 256, 257, 1000, 5000}) {
      auto gen = [&](size_t k) { return fp ? rand_fp(rg, k) : rand_ints(rg, k); };
      int x = newbuf(1, n, gen(n)), y = newbuf(1, n, gen(n)), w1 = newbuf(1, 1, gen(1));
      float al = fp ? 0.37f : 3.f;
      int o_copy = newbuf(1, n, {}), o_add = newbuf(1, n, {}), o_scale = newbuf(1, n, {}), o_relu = newbuf(1, n, {}), o_pred = newbuf(1, n, {});
      int wsgd = newbuf(1, n, gen(n)), o_fill = newbuf(1, n, {});
      task("copy", {Bf(x), Bf(o_copy), I(n)}); task("add", {Bf(x), Bf(y), Bf(o_add), I(n)}); task("scale", {Bf(x), Bf(o_scale), F(al), I(n)});
      task("relu", {Bf(x), Bf(o_relu), I(n)}); task("predict", {Bf(x), Bf(w1), Bf(o_pred), I(n)});
      task("sgd", {Bf(wsgd), Bf(y), F(fp ? 0.125f : 2.f), I(n)}); task("fill", {Bf(o_fill), F(-2.5f), I(n)});
      int pn = newbuf(1, n, gen(n)), tn = newbuf(1, n, gen(n)), loss = newbuf(1, 1, {}), grad = newbuf(1, 1, {});
      task("mse_grad", {Bf(pn), Bf(tn), Bf(x), Bf(loss), Bf(grad), I(n)});
      for (int o : {o_copy, o_add, o_scale, o_relu, o_pred, wsgd, o_fill, loss, grad}) pend.push_back({fp ? "fp" : "int", o, !fp});
    }
  // matmul: any m,n,k
  for (auto [m, n, k] : std::vector<std::array<int, 3>>{{1, 1, 1}, {16, 16, 16}, {17, 15, 33}, {50, 3, 1}, {2, 64, 129}, {31, 47, 16}}) {
    for (bool fp : {false, true}) {
      int A = newbuf(m, k, fp ? rand_fp(rg, (size_t)m * k) : rand_ints(rg, (size_t)m * k)), Bm = newbuf(k, n, fp ? rand_fp(rg, (size_t)k * n) : rand_ints(rg, (size_t)k * n));
      int O = newbuf(m, n, {});
      task("matmul", {Bf(A), Bf(Bm), Bf(O), I(m), I(n), I(k)});
      pend.push_back({fp ? "matmul-fp" : "matmul-int", O, !fp});
      // CPU reference (naive, k ascending)
      for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) { float acc = 0; for (int kk = 0; kk < k; ++kk) acc += host[A][(size_t)i * k + kk] * host[Bm][(size_t)kk * n + j]; host[O][(size_t)i * n + j] = acc; }
    }
  }
  int spin_t = task("spin", {I(300)}); task("noop", {});
  KvmBundle b = bb.build();
  // CPU reference for every non-matmul task, in order (matmul already filled above)
  for (size_t i = 0; i < b.tasks.size(); ++i) if (b.tasks[i].op_id != KVM_OP_matmul) cpu_op(b, (int)i, host);
  for (auto& [p, v] : inits) upload(p, v);
  std::vector<float*> ptrs; for (auto& [p, v] : inits) ptrs.push_back(p);
  // matmul outputs must start zero on the device (host mirror already holds the result) -> re-zero
  for (size_t i = 0; i < b.tasks.size(); ++i) if (b.tasks[i].op_id == KVM_OP_matmul) { int o = b.args[b.tasks[i].args_off + 2]; upload(ptrs[o], std::vector<float>(host[o].size(), 0.f)); }
  bool ok = run_ok(ex, {b});
  if (!ok) return false;
  const KvmResult& r = ex.finish();
  int exact_bad = 0, fp_bad = 0; size_t exact_n = 0, fp_n = 0; double max_rel = 0;
  for (auto& pd : pend) {
    auto got = download(ptrs[pd.out], host[pd.out].size());
    for (size_t i = 0; i < got.size(); ++i) {
      if (pd.exact) { ++exact_n; if (!same_bits(got[i], host[pd.out][i])) { if (exact_bad++ < 3) fprintf(stderr, "  %s buf %d[%zu]: got %g want %g\n", pd.name, pd.out, i, got[i], host[pd.out][i]); } }
      else { ++fp_n; max_rel = std::max(max_rel, (double)std::fabs(got[i] - host[pd.out][i]) / std::max(1.0, (double)std::fabs(host[pd.out][i]))); if (!close_rel(got[i], host[pd.out][i])) ++fp_bad; }
    }
  }
  CHECK(exact_bad == 0, "%d/%zu integer-valued elements differ bitwise", exact_bad, exact_n);
  CHECK(fp_bad == 0, "%d/%zu fp elements beyond 1e-5 rel (max %.3g)", fp_bad, fp_n, max_rel);
  const KvmTrace& st = r.trace[spin_t];
  CHECK(st.end_ns - st.start_ns >= 300000, "spin(300us) lasted %.1f us", (st.end_ns - st.start_ns) / 1e3);
  check_trace(r, {b});
  printf("  ops: %zu integer elements bitwise equal, %zu fp elements within 1e-5 rel (max rel err %.3g); spin(300us) ran %.1f us\n", exact_n, fp_n, max_rel, (st.end_ns - st.start_ns) / 1e3);
  return g_fail == 0;
}

// ================= device-side error codes (host validation bypassed) =================
static void err_case(const char* name, int want, KvmConfig cfg, const std::function<KvmBundle(KvmExec&)>& make, int timeout_ms = 5000, std::function<void(KvmExec&)> after = nullptr) {
  KvmExec ex(cfg);
  float* a = ex.arena(64); int b0 = ex.add_view(a, 1, 64); int b1 = ex.add_view(a, 1, 8);
  (void)b0; (void)b1;
  ex.start();
  KvmBundle b = make(ex);
  KvmSubmit s = ex.submit(b, timeout_ms, false);
  if (after) after(ex);
  ex.close();
  const KvmResult& r = ex.finish();
  (void)s;
  CHECK(r.err_code == want, "%s: got %s want %s", name, kvm_err_name(r.err_code), kvm_err_name(want));
  printf("  error %-18s -> %s (slot %d a=%d b=%d)\n", name, kvm_err_name(r.err_code), r.err_slot, r.err_a, r.err_b);
}
static bool suite_errors() {
  auto one = [](const char* op, std::vector<KvmArg> a, KvmTaskOpts o = {}) { KvmBundleBuilder bb; bb.add(op, a, o); return bb.build(); };
  KvmConfig c = small_cfg(3);
  err_case("bad op id", KVM_ERR_BAD_OP, c, [&](KvmExec&) { KvmBundle b = one("noop", {}); b.tasks[0].op_id = 99; return b; });
  err_case("bad arity", KVM_ERR_BAD_ARITY, c, [&](KvmExec&) { KvmBundle b = one("noop", {}); b.tasks[0].args_words = 2; b.args = {0, 0}; return b; });
  err_case("bad buffer index", KVM_ERR_BAD_BUF_INDEX, c, [&](KvmExec&) { return one("fill", {Bf(7), F(1), I(4)}); });
  err_case("n beyond buffer", KVM_ERR_BOUNDS, c, [&](KvmExec&) { return one("fill", {Bf(1), F(1), I(9)}); });
  err_case("negative n", KVM_ERR_BOUNDS, c, [&](KvmExec&) { return one("copy", {Bf(0), Bf(1), I(-1)}); });
  err_case("bad priority", KVM_ERR_BAD_TASK, c, [&](KvmExec&) { return one("noop", {}, [] { KvmTaskOpts o; o.priority = 40; return o; }()); });
  err_case("bad version var", KVM_ERR_BAD_VAR, c, [&](KvmExec&) { return one("noop", {}, [] { KvmTaskOpts o; o.pub_var = 999; o.pub_ver = 1; return o; }()); });
  err_case("version gap", KVM_ERR_VERSION_GAP, c, [&](KvmExec&) { return one("noop", {}, [] { KvmTaskOpts o; o.pub_var = 3; o.pub_ver = 5; return o; }()); });
  err_case("bad successor", KVM_ERR_BAD_TASK, c, [&](KvmExec&) { KvmBundle b = one("noop", {}); b.tasks[0].succ_cnt = 1; b.succ = {5}; return b; });
  { KvmConfig s = c; s.max_tasks = 8; err_case("slot pool exhausted", KVM_ERR_SLOTS, s, [&](KvmExec&) { KvmBundleBuilder bb; for (int i = 0; i < 9; ++i) bb.add("noop", {}); return bb.build(); }); }
  { KvmConfig s = c; s.live_cap = 8; err_case("batch > live_cap", KVM_ERR_BATCH_TOO_LARGE, s, [&](KvmExec&) { KvmBundleBuilder bb; for (int i = 0; i < 9; ++i) bb.add("noop", {}); return bb.build(); }); }
  { KvmConfig s = c; s.args_cap = 2; s.succ_cap = 4; err_case("args pool exhausted", KVM_ERR_POOL, s, [&](KvmExec&) { return one("fill", {Bf(0), F(1), I(4)}); }); }
  { KvmConfig s = c; s.timeout_us = 300000; err_case("unsatisfied wait -> timeout", KVM_ERR_TIMEOUT, s, [&](KvmExec&) { return one("noop", {}, [] { KvmTaskOpts o; o.wait_var = 2; o.wait_ver = 1; return o; }()); }); }
  // host-side validation rejects what the device would (and the executor stays usable)
  { KvmExec ex(c); float* a = ex.arena(8); ex.add_view(a, 1, 8); ex.start();
    KvmBundleBuilder bb;
    bb.add("noop", {}, [] { KvmTaskOpts o; o.priority = 99; return o; }());
    CHECK(ex.submit(bb.build()) == KvmSubmit::INVALID, "host should reject priority 99");
    KvmBundleBuilder cy; int x = cy.add("noop", {}), y = cy.add("noop", {}); cy.edge(x, y); cy.edge(y, x);
    CHECK(ex.submit(cy.build()) == KvmSubmit::INVALID, "host should reject a cycle");
    KvmBundleBuilder tb; for (int i = 0; i < c.live_cap + 1; ++i) tb.add("noop", {});
    CHECK(ex.submit(tb.build()) == KvmSubmit::TOO_BIG, "host should reject bundle > live_cap");
    bool threw = false; try { KvmBundleBuilder e; e.add("fill", {I(0), F(1), I(1)}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "builder should reject i32 for a buf argument");
    KvmBundleBuilder ok; ok.add("fill", {Bf(0), F(1), I(8)}); CHECK(ex.submit(ok.build()) == KvmSubmit::OK, "valid bundle after rejections");
    ex.close(); CHECK(ex.submit(ok.build()) == KvmSubmit::CLOSED, "submit after close");
    CHECK(ex.finish().ok(), "run ok after rejected bundles");
  }
  return g_fail == 0;
}

// ================= random DAG stress =================
struct DagSpec { int n, layers, maxctl; int prio_lo, prio_hi; };
// Node i in layer(i). Data preds (<=2) and extra control preds come from strictly earlier layers. Value of node: fill / scale(1.5) / add.
static KvmBundle gen_dag(Rng& r, const DagSpec& sp, int workload, int buf_base, int& dupes_removed) {
  KvmBundleBuilder bb;
  std::vector<int> layer(sp.n);
  for (int i = 0; i < sp.n; ++i) layer[i] = (int)((long long)i * sp.layers / sp.n);
  std::vector<std::vector<int>> by_layer(sp.layers);
  for (int i = 0; i < sp.n; ++i) by_layer[layer[i]].push_back(i);
  dupes_removed = 0;
  std::vector<std::vector<int>> data(sp.n);
  for (int i = 0; i < sp.n; ++i) {
    if (layer[i] > 0) {
      int nd = r.in(0, 2);
      for (int k = 0; k < nd; ++k) {
        int l = r.p(70) ? layer[i] - 1 : r.in(0, layer[i] - 1);
        int p = by_layer[l][r.in(0, (int)by_layer[l].size() - 1)];
        if (std::find(data[i].begin(), data[i].end(), p) == data[i].end()) data[i].push_back(p);
      }
    }
    KvmTaskOpts o; o.priority = r.in(sp.prio_lo, sp.prio_hi); o.workload = workload;
    int out = buf_base + i;
    if (data[i].empty()) bb.add("fill", {Bf(out), F((float)(1 + i % 7)), I(1)}, o);
    else if (data[i].size() == 1) bb.add("scale", {Bf(buf_base + data[i][0]), Bf(out), F(1.5f), I(1)}, o);
    else bb.add("add", {Bf(buf_base + data[i][0]), Bf(buf_base + data[i][1]), Bf(out), I(1)}, o);
  }
  for (int i = 0; i < sp.n; ++i) {
    for (int p : data[i]) bb.edge(p, i);
    int nc = layer[i] ? r.in(0, sp.maxctl) : 0;
    for (int k = 0; k < nc; ++k) {  // control edges, often duplicates of data edges: the builder removes them
      int l = r.in(0, layer[i] - 1); int p = by_layer[l][r.in(0, (int)by_layer[l].size() - 1)];
      if (std::find(data[i].begin(), data[i].end(), p) != data[i].end()) ++dupes_removed;
      bb.edge(p, i);
    }
  }
  return bb.build();
}

static bool dag_run(unsigned seed, int nbundles, int n_each, int layers, KvmConfig cfg, int prio_hi, long long* total_tasks, int* dupes) {
  Rng r(seed);
  KvmExec ex(cfg);
  size_t N = (size_t)nbundles * n_each;
  float* arena = ex.arena(N);
  for (size_t i = 0; i < N; ++i) ex.add_view(arena + i, 1, 1);
  std::vector<KvmBundle> bs;
  *dupes = 0;
  for (int b = 0; b < nbundles; ++b) { int d; bs.push_back(gen_dag(r, {n_each, layers, 3, 0, prio_hi}, b, b * n_each, d)); *dupes += d; }
  Bufs want(N, std::vector<float>(1, 0.f));
  for (int b = 0; b < nbundles; ++b) {  // sequential CPU evaluation (nodes are topologically numbered)
    for (size_t i = 0; i < bs[b].tasks.size(); ++i) cpu_op(bs[b], (int)i, want);
  }
  *total_tasks += (long long)N;
  if (!run_ok(ex, bs)) return false;
  const KvmResult& res = ex.finish();
  check_trace(res, bs);
  auto got = download(arena, N);
  int bad = 0; for (size_t i = 0; i < N; ++i) if (!same_bits(got[i], want[i][0])) ++bad;
  CHECK(bad == 0, "seed %u: %d/%zu node values differ from the CPU reference", seed, bad, N);
  return true;
}

static bool suite_dag(bool quick) {
  long long total = 0; int dupes = 0, d;
  int before = g_fail;
  struct Cfg { int blocks; const char* what; };
  // many seeds x grid sizes; 2 blocks = a single worker
  for (unsigned seed = 1; seed <= (quick ? 3u : 24u); ++seed)
    for (int blocks : {2, 3, 16, 0}) {
      KvmConfig c = small_cfg(blocks); c.live_cap = 1 << 13; c.ring_words = 1 << 17;
      int nb = 1 + seed % 3, n_each = quick ? 150 : 900 + 300 * (seed % 4);
      dag_run(seed * 7919, nb, n_each, 8 + seed % 30, c, 31, &total, &d); dupes += d;
    }
  // queue wraparound: live_cap 64 with thousands of tasks all in priority 0 (one ring wraps ~100 times) and a spread version
  { KvmConfig c = small_cfg(8); c.live_cap = 64; c.max_tasks = 1 << 16; c.ring_words = 4096;
    long long t = 0; dag_run(4242, quick ? 20 : 150, 40, 10, c, 0, &t, &d); total += t; dupes += d;
    printf("  wraparound: %lld tasks through 32 rings of 64 cells (one ring took all of them: %lld wraps)\n", t, t / 64); }
  { KvmConfig c = small_cfg(8); c.live_cap = 64; c.max_tasks = 1 << 16; c.ring_words = 4096; long long t = 0; dag_run(99, quick ? 10 : 60, 40, 10, c, 31, &t, &d); total += t; }
  printf("  dag: %lld tasks over seeds x grids {1,2,15,auto workers}, all exactly-once, edges ordered by timestamps, values bitwise vs CPU; %d duplicate edges removed by the builder\n", total, dupes);
  return g_fail == before;
}

// ================= priority, FIFO, aging, deadlines =================
static bool suite_sched() {
  int before = g_fail;
  // (1) strict priority and FIFO with ONE worker: a long gate task keeps the worker busy while everything else queues up
  {
    KvmConfig c = small_cfg(2);
    KvmExec ex(c); ex.start();
    std::vector<KvmBundle> bs;
    { KvmBundleBuilder g; KvmTaskOpts o; o.priority = 31; o.workload = 0; g.add("spin", {I(20000)}, o); bs.push_back(g.build()); }
    Rng r(5); std::vector<int> prio;
    { KvmBundleBuilder m; for (int i = 0; i < 300; ++i) { KvmTaskOpts o; o.priority = r.in(0, 30); o.workload = 1; prio.push_back(o.priority); m.add("noop", {}, o); } bs.push_back(m.build()); }
    int fifo_n = 40;  // 40 one-task bundles, same priority, submitted in order: admission order = ticket order
    for (int i = 0; i < fifo_n; ++i) { KvmBundleBuilder f; KvmTaskOpts o; o.priority = 15; o.workload = 2 + i; f.add("noop", {}, o); bs.push_back(f.build()); }
    for (auto& b : bs) CHECK(ex.submit(b) == KvmSubmit::OK, "submit");
    const KvmResult& res = ex.finish();
    CHECK(res.ok(), "device error");
    // execution order of everything but the gate
    std::vector<int> order; for (size_t i = 1; i < res.trace.size(); ++i) order.push_back((int)i);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return res.trace[a].start_ns < res.trace[b].start_ns; });
    int inversions = 0; int last = 31;
    for (int s : order) { int p = res.tasks[s].priority; if (p > last) ++inversions; last = p; }
    CHECK(inversions == 0, "%d priority inversions in execution order", inversions);
    // FIFO among the 40 single-task bundles: slots 301.. start in slot order
    int fifo_bad = 0; for (int i = 0; i + 1 < fifo_n; ++i) if (res.trace[301 + i].start_ns > res.trace[302 + i].start_ns) ++fifo_bad;
    CHECK(fifo_bad == 0, "%d FIFO violations within priority 15", fifo_bad);
    printf("  priority: 340 queued tasks ran in non-increasing priority order (%d inversions), FIFO within a priority (%d violations)\n", inversions, fifo_bad);
  }
  // (2) aging: a priority-0 task vs a long chain of priority-10 tasks, one worker
  for (int aging_us : {0, 2000}) {
    KvmConfig c = small_cfg(2); c.aging_us = aging_us;
    KvmExec ex(c); ex.start();
    KvmBundleBuilder bb; const int H = 400;
    int prev = -1;
    for (int i = 0; i < H; ++i) { KvmTaskOpts o; o.priority = 10; int t = bb.add("spin", {I(100)}, o); if (prev >= 0) bb.edge(prev, t); prev = t; }
    KvmTaskOpts lo; lo.priority = 0; int low = bb.add("noop", {}, lo);
    KvmBundle b = bb.build();
    CHECK(ex.submit(b) == KvmSubmit::OK, "submit");
    const KvmResult& res = ex.finish(); CHECK(res.ok(), "device error");
    int pos = 0; for (int i = 0; i < H; ++i) if (res.trace[i].start_ns < res.trace[low].start_ns) ++pos;
    double waited_ms = (res.trace[low].start_ns - res.trace[low].admit_ns) / 1e6, chain_ms = (res.trace[H - 1].end_ns - res.trace[0].start_ns) / 1e6;
    if (aging_us == 0) CHECK(pos == H, "no aging: low-priority task must wait for the whole chain (ran after %d of %d)", pos, H);
    else {
      CHECK(pos < H - 20, "aging: low-priority task should overtake the chain (ran after %d of %d)", pos, H);
      CHECK(waited_ms > 0.8 * 10 * aging_us / 1e3, "aging must not be instant: waited %.2f ms (10 levels x %d us)", waited_ms, aging_us);
    }
    printf("  aging %4d us/level: low-priority task ran after %d of %d high-priority chain tasks, waited %.2f ms of a %.2f ms chain\n", aging_us, pos, H, waited_ms, chain_ms);
  }
  // (3) deadlines and per-task trace fields
  {
    KvmConfig c = small_cfg(4); KvmExec ex(c); ex.start();
    KvmBundleBuilder bb; KvmTaskOpts miss; miss.deadline_us = 1; KvmTaskOpts meet; meet.deadline_us = 5000000; KvmTaskOpts none;
    int a = bb.add("spin", {I(400)}, miss), b2 = bb.add("spin", {I(10)}, meet), c2 = bb.add("noop", {}, none);
    CHECK(ex.submit(bb.build()) == KvmSubmit::OK, "submit");
    const KvmResult& res = ex.finish(); CHECK(res.ok(), "device error");
    CHECK(res.trace[a].flags & KVM_TRACE_MISSED, "1 us deadline on a 400 us task must be flagged missed");
    CHECK(!(res.trace[b2].flags & KVM_TRACE_MISSED), "5 s deadline must not be flagged");
    CHECK(!(res.trace[c2].flags & KVM_TRACE_MISSED), "no deadline must not be flagged");
    for (int s : {a, b2, c2}) CHECK(res.trace[s].smid >= 0 && res.trace[s].smid < ex.sms() && res.trace[s].block >= 1 && res.trace[s].block < res.grid, "smid %d block %d", res.trace[s].smid, res.trace[s].block);
    kvm_write_trace_csv(res, "trace_sched.csv");
    printf("  deadlines: missed flag set only for the 1 us deadline; smid/block recorded; CSV written\n");
  }
  return g_fail == before;
}

// ================= admission while the kernel runs =================
static bool suite_admission() {
  int before = g_fail;
  // (1) ring wrap + backpressure: tiny ring, many bundles with enough work to build a backlog, one host thread
  {
    KvmConfig c = small_cfg(4); c.ring_words = 512; c.live_cap = 256; c.max_tasks = 1 << 14;
    KvmExec ex(c);
    const int NB = 300, T = 12;
    float* arena = ex.arena(NB * T);
    for (int i = 0; i < NB * T; ++i) ex.add_view(arena + i, 1, 1);
    ex.start();
    std::vector<KvmBundle> bs; Rng r(11);
    for (int b = 0; b < NB; ++b) { int d; bs.push_back(gen_dag(r, {T, 4, 2, 0, 31}, b, b * T, d)); }
    for (auto& b : bs) CHECK(ex.submit(b) == KvmSubmit::OK, "submit");
    ex.close();
    const KvmResult& res = ex.finish(); CHECK(res.ok(), "device error");
    check_trace(res, bs);
    Bufs want(NB * T, std::vector<float>(1, 0.f)); for (int b = 0; b < NB; ++b) for (size_t i = 0; i < bs[b].tasks.size(); ++i) cpu_op(bs[b], (int)i, want);
    auto got = download(arena, NB * T); int bad = 0; for (int i = 0; i < NB * T; ++i) if (!same_bits(got[i], want[i][0])) ++bad;
    CHECK(bad == 0, "%d values differ", bad);
    CHECK(ex.stats().pads > 0, "ring never wrapped (pads %d)", ex.stats().pads);
    printf("  admission ring: %d bundles / %d tasks through a %d-word ring: %d wraps (pad records), %d backpressure waits, values bitwise vs CPU\n", NB, NB * T, c.ring_words, ex.stats().pads, ex.stats().backpressure_waits);
  }
  // (2) backpressure with a slow device: spin tasks (1 ms) on one worker, ring of 2 records
  {
    KvmConfig c = small_cfg(2); c.ring_words = 256; c.live_cap = 64; KvmExec ex(c); ex.start();
    int ok = 0;
    for (int b = 0; b < 30; ++b) { KvmBundleBuilder bb; for (int i = 0; i < 3; ++i) { KvmTaskOpts o; o.workload = b; bb.add("spin", {I(300)}, o); } ok += ex.submit(bb.build(), 10000) == KvmSubmit::OK; }
    ex.close(); const KvmResult& res = ex.finish();
    CHECK(ok == 30 && res.ok() && res.done == 90, "ok %d done %d", ok, res.done);
    CHECK(ex.stats().backpressure_waits > 0, "expected the host to hit backpressure");
    printf("  backpressure: 30 bundles x 3 x 300 us on one worker, 256-word ring: host waited %d times, nothing dropped (%d/90 done)\n", ex.stats().backpressure_waits, res.done);
  }
  // (3) explicit errors: record bigger than the ring; backpressure timeout when admission is stalled by waiting tasks
  {
    KvmConfig c = small_cfg(2); c.ring_words = 128; c.live_cap = 16; KvmExec ex(c); ex.start();
    KvmBundleBuilder big; for (int i = 0; i < 12; ++i) big.add("noop", {});
    CHECK(ex.submit(big.build()) == KvmSubmit::TOO_BIG, "record larger than the ring must be TOO_BIG");
    ex.close(); CHECK(ex.finish().ok(), "idle run");
  }
  {
    KvmConfig c = small_cfg(2); c.ring_words = 512; c.live_cap = 16; c.timeout_us = 1500000; KvmExec ex(c); ex.start();
    // 16 tasks waiting on a version nobody publishes fill the live window, so the admitter stops consuming the ring
    KvmBundleBuilder w; KvmTaskOpts o; o.wait_var = 1; o.wait_ver = 1; for (int i = 0; i < 16; ++i) w.add("noop", {}, o);
    KvmBundleBuilder one; one.add("noop", {});
    KvmBundle sb = one.build();
    CHECK(ex.submit(w.build()) == KvmSubmit::OK, "waiting bundle");
    KvmSubmit s = KvmSubmit::OK; int n = 0;
    for (; n < 100 && s == KvmSubmit::OK; ++n) s = ex.submit(sb, 300);
    CHECK(s == KvmSubmit::BACKPRESSURE_TIMEOUT, "stalled admission must surface as BACKPRESSURE_TIMEOUT, got %s", kvm_submit_name(s));
    ex.close();
    const KvmResult& res = ex.finish();
    CHECK(ex.submit(sb) == KvmSubmit::CLOSED, "submit after close");
    CHECK(res.err_code == KVM_ERR_TIMEOUT, "device must end with TIMEOUT, got %s", kvm_err_name(res.err_code));
    printf("  explicit failures: TOO_BIG for an oversized record; %s after %d submits against a stalled live window; device ended with %s (admitted %d, done %d)\n", kvm_submit_name(s), n, kvm_err_name(res.err_code), res.err_a, res.err_b);
  }
  // (4) close() with nothing submitted, and close() while work is queued: kernel exits, everything admitted finishes
  {
    KvmExec e0(small_cfg(3)); e0.start(); e0.close(); const KvmResult& r0 = e0.finish(); CHECK(r0.ok() && r0.done == 0, "empty close");
    KvmExec e1(small_cfg(3)); e1.start(); KvmBundleBuilder bb; for (int i = 0; i < 50; ++i) bb.add("spin", {I(100)}); CHECK(e1.submit(bb.build()) == KvmSubmit::OK, "submit");
    e1.close(); const KvmResult& r1 = e1.finish(); CHECK(r1.ok() && r1.done == 50, "drain: done %d", r1.done);
    printf("  close(): empty launch exits; close() right after submit drains all 50 tasks\n");
  }
  // (5) concurrent producers (host threads) while the kernel runs
  {
    KvmConfig c = small_cfg(6); c.ring_words = 2048; KvmExec ex(c);
    const int TH = 4, PER = 60, T = 10; float* arena = ex.arena(TH * PER * T);
    for (int i = 0; i < TH * PER * T; ++i) ex.add_view(arena + i, 1, 1);
    ex.start();
    std::vector<KvmBundle> bs(TH * PER); Rng r(77);
    for (int b = 0; b < TH * PER; ++b) { int d; bs[b] = gen_dag(r, {T, 4, 2, 0, 31}, b, b * T, d); }
    std::atomic<int> okc{0};
    std::vector<std::thread> th;
    for (int t = 0; t < TH; ++t) th.emplace_back([&, t] { for (int b = t; b < TH * PER; b += TH) { okc += ex.submit(bs[b]) == KvmSubmit::OK; std::this_thread::sleep_for(std::chrono::microseconds(50 * (b % 5))); } });
    for (auto& x : th) x.join();
    ex.close(); const KvmResult& res = ex.finish(); CHECK(res.ok() && okc == TH * PER, "ok %d", okc.load());
    check_trace(res, bs);
    Bufs want(TH * PER * T, std::vector<float>(1, 0.f)); for (int b = 0; b < TH * PER; ++b) for (size_t i = 0; i < bs[b].tasks.size(); ++i) cpu_op(bs[b], (int)i, want);
    auto got = download(arena, TH * PER * T); int bad = 0; for (size_t i = 0; i < got.size(); ++i) if (!same_bits(got[i], want[i][0])) ++bad;
    CHECK(bad == 0, "%d values differ", bad);
    printf("  %d producer threads x %d bundles concurrently: %d tasks, exactly-once, values bitwise vs CPU\n", TH, PER, res.done);
  }
  return g_fail == before;
}

// ================= weight versions: train graph concurrent with serve requests =================
// Golden for the vector test: w_k[j] = w_0 - lr * sum_{s<k} G_s (identical for every element).
struct Golden { std::vector<float> w; };
static float step_g(int s) { return (float)(s % 7 + 1) * 0.25f; }

// Checker used both on real traces and on hand-made bad ones (see suite_checker).
struct ReaderObs { int exact_ver;  // >=0: exact reader of that snapshot; -1: at_least
                   int need; int ver0, ver1; std::vector<float> seen; };
static int check_reader(const ReaderObs& o, const Golden& g) {
  int bad = 0;
  if (o.ver0 < o.need) ++bad;                                   // ran before its version existed
  if (o.exact_ver >= 0) {
    if (o.ver0 != o.exact_ver || o.ver1 != o.exact_ver) ++bad;  // observed a different version than its snapshot
    for (float v : o.seen) if (!same_bits(v, g.w[o.exact_ver])) { ++bad; break; }
  } else {
    for (float v : o.seen) {
      bool allowed = false;
      for (int k = o.ver0; k <= std::min<int>(o.ver1 + 1, (int)g.w.size() - 1); ++k) allowed |= same_bits(v, g.w[k]);
      if (!allowed) { ++bad; break; }
    }
  }
  return bad;
}

// Vector test: NW-element weight vector, fill+sgd train steps, exact readers (copy) gated by a reader-done variable.
static bool versions_vector(unsigned seed, int steps, int NW, int blocks, bool spins, int* nex, int* nal, int* violations) {
  Rng r(seed);
  KvmConfig c = small_cfg(blocks); c.timeout_us = 30ull * 1000000;
  KvmExec ex(c);
  float* arena = ex.arena((size_t)NW * 2 + 1);
  const int W = ex.add_view(arena, 1, NW), Gb = ex.add_view(arena + NW, 1, NW);
  std::vector<std::vector<int>> outs;  // reader output buffers
  struct Rd { int snap; int need; std::vector<int> out; bool exact; int first_slot; };
  const float lr = 0.5f, w0 = 1.0f;
  Golden gold; gold.w.push_back(w0); for (int s = 0; s < steps; ++s) gold.w.push_back(gold.w.back() - lr * step_g(s));
  upload(arena, std::vector<float>(NW, w0));
  // reader plan: each snapshot gets an exact-reader bundle with prob 60%; at_least readers sprinkled
  std::vector<int> exact_snaps; for (int s = 0; s < steps; ++s) if (r.p(60)) exact_snaps.push_back(s);
  int nread_bufs = 0; std::vector<Rd> rds; std::vector<KvmBundle> serve; std::vector<float*> outp;
  const int DONE_VAR_BASE = 1;
  for (int s : exact_snaps) {
    KvmBundleBuilder bb; int K = r.in(1, 3); std::vector<int> copies;
    for (int k = 0; k < K; ++k) {
      float* o = ex.arena(NW); outp.push_back(o); int ob = ex.add_view(o, 1, NW);
      KvmTaskOpts t; t.wait_var = 0; t.wait_ver = s; t.workload = 1 + (int)serve.size();
      int prev = -1;
      if (spins && r.p(60)) { KvmTaskOpts so; so.workload = t.workload; prev = bb.add("spin", {I(r.in(1, 150))}, so); }
      int cp = bb.add("copy", {Bf(W), Bf(ob), I(NW)}, t); if (prev >= 0) bb.edge(prev, cp);
      copies.push_back(cp); rds.push_back({s, s, {ob}, true, 0}); ++nread_bufs;
    }
    KvmTaskOpts jo; jo.pub_var = DONE_VAR_BASE + s; jo.pub_ver = 1; jo.workload = 1 + (int)serve.size();
    int join = bb.add("noop", {}, jo); for (int cp : copies) bb.edge(cp, join);
    serve.push_back(bb.build());
  }
  int n_at = r.in(2, 6);
  for (int i = 0; i < n_at; ++i) {
    KvmBundleBuilder bb; float* o = ex.arena(NW); outp.push_back(o); int ob = ex.add_view(o, 1, NW);
    KvmTaskOpts t; t.wait_var = 0; t.wait_ver = r.in(0, steps); t.workload = 1 + (int)serve.size();
    bb.add("copy", {Bf(W), Bf(ob), I(NW)}, t); rds.push_back({-1, t.wait_ver, {ob}, false, 0}); ++nread_bufs;
    serve.push_back(bb.build());
  }
  // train bundle: fill(g) -> sgd(w,g) pub W; sgd_s waits for the readers of snapshot s (if any); sgd_s -> fill_{s+1}
  KvmBundleBuilder tb; int prev = -1;
  for (int s = 0; s < steps; ++s) {
    KvmTaskOpts to; to.workload = 0; to.priority = 3;
    int sp = -1; if (spins && r.p(30)) sp = tb.add("spin", {I(r.in(1, 100))}, to);
    int f = tb.add("fill", {Bf(Gb), F(step_g(s)), I(NW)}, to);
    KvmTaskOpts so = to; so.pub_var = 0; so.pub_ver = s + 1;
    if (std::find(exact_snaps.begin(), exact_snaps.end(), s) != exact_snaps.end()) { so.wait_var = DONE_VAR_BASE + s; so.wait_ver = 1; }
    int g = tb.add("sgd", {Bf(W), Bf(Gb), F(lr), I(NW)}, so);
    if (sp >= 0) tb.edge(sp, f);
    if (prev >= 0) tb.edge(prev, sp >= 0 ? sp : f);
    tb.edge(f, g); prev = g;
  }
  std::vector<KvmBundle> all; all.push_back(tb.build());
  for (auto& b : serve) all.push_back(b);
  ex.start();
  // host: train first or serve first at random; serve bundles arrive at random times while the kernel runs
  std::vector<int> order(serve.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i + 1;
  std::shuffle(order.begin(), order.end(), r.g);
  size_t train_pos = r.in(0, (int)order.size());
  order.insert(order.begin() + train_pos, 0);
  for (int bi : order) { CHECK(ex.submit(all[bi]) == KvmSubmit::OK, "submit"); if (r.p(30)) std::this_thread::sleep_for(std::chrono::microseconds(r.in(5, 300))); }
  ex.close();
  const KvmResult& res = ex.finish();
  if (!res.ok()) { CHECK(false, "device error %s seed %u", kvm_err_name(res.err_code), seed); return false; }
  check_trace(res, all);
  int bad = 0, nex_ = 0, nal_ = 0;
  // locate reader copy tasks in the trace: ops copy with workload >=1 in admission order, same order as rds
  // match each copy task to its reader by its output buffer index (args word 1)
  std::map<int, int> by_out; for (size_t k = 0; k < rds.size(); ++k) by_out[rds[k].out[0]] = (int)k;
  std::vector<int> seen_task(rds.size(), -1);
  for (size_t i = 0; i < res.tasks.size(); ++i) if (res.tasks[i].op_id == KVM_OP_copy && res.tasks[i].workload_id >= 1) {
    // output buffer index is in the args pool, which we kept host-side per bundle
    const KvmBundle& b = all[res.tasks[i].workload_id]; int base = -1;
    for (size_t j = 0; j < res.tasks.size(); ++j) if (res.tasks[j].workload_id == res.tasks[i].workload_id) { base = (int)j; break; }
    int local = (int)i - base; int ob = b.args[b.tasks[local].args_off + 1];
    seen_task[by_out[ob]] = (int)i;
  }
  for (size_t k = 0; k < rds.size(); ++k) {
    CHECK(seen_task[k] >= 0, "reader %zu never ran", k);
    if (seen_task[k] < 0) continue;
    const KvmTrace& t = res.trace[seen_task[k]];
    ReaderObs o{rds[k].exact ? rds[k].snap : -1, rds[k].need, t.ver0, t.ver1, download(outp[k], NW)};
    int vb = check_reader(o, gold);
    if (vb) fprintf(stderr, "  seed %u reader %zu (%s snap %d need %d): ver0 %d ver1 %d first %g last %g golden %g\n", seed, k, rds[k].exact ? "exact" : "at_least", rds[k].snap, rds[k].need, t.ver0, t.ver1, o.seen[0], o.seen.back(), rds[k].exact ? gold.w[rds[k].snap] : gold.w[std::max(0, rds[k].need)]);
    bad += vb; (rds[k].exact ? nex_ : nal_)++;
  }
  auto wf = download(arena, NW);
  bool fin = true; for (float v : wf) fin = fin && same_bits(v, gold.w[steps]);
  CHECK(fin, "final weights differ from golden w_%d", steps);
  CHECK(res.version[0] == steps, "final version %d != %d", res.version[0], steps);
  CHECK(bad == 0, "seed %u: %d readers saw something other than their snapshot", seed, bad);
  *nex += nex_; *nal += nal_; *violations += bad;
  return bad == 0;
}

// Scalar test with the real train graph predict -> mse_grad -> sgd, exact + at_least serve requests (predict).
static bool versions_scalar(unsigned seed, int steps, int blocks, bool spins, int* nex, int* nal, int* violations) {
  Rng r(seed);
  const int N = 64; const float lr = 1.f / 64.f;
  KvmConfig c = small_cfg(blocks); c.timeout_us = 30ull * 1000000;
  KvmExec ex(c);
  std::vector<float> x = rand_ints(r, N, 1, 5), t = rand_ints(r, N, -20, 20);
  float* ar = ex.arena(N * 3 + 8);  // x, t, pred, then scalars w, loss, grad
  upload(ar, x); upload(ar + N, t);
  const int bx = ex.add_view(ar, 1, N), bt = ex.add_view(ar + N, 1, N), bp = ex.add_view(ar + 2 * N, 1, N), bw = ex.add_view(ar + 3 * N, 1, 1),
            bl = ex.add_view(ar + 3 * N + 1, 1, 1), bg = ex.add_view(ar + 3 * N + 2, 1, 1);
  // golden via the CPU reference ops: w_0 = 0, then one train step at a time
  std::vector<float> wk{0.f};
  { KvmBundleBuilder gb; gb.add("predict", {Bf(0), Bf(3), Bf(2), I(N)}); gb.add("mse_grad", {Bf(2), Bf(1), Bf(0), Bf(4), Bf(5), I(N)}); gb.add("sgd", {Bf(3), Bf(5), F(lr), I(1)});
    KvmBundle b = gb.build(); Bufs B{x, t, std::vector<float>(N), {0.f}, {0.f}, {0.f}};
    for (int s = 0; s < steps; ++s) { for (int i = 0; i < 3; ++i) cpu_op(b, i, B); wk.push_back(B[3][0]); } }
  struct Rd { int snap, need; int ob; bool exact; };
  std::vector<Rd> rds; std::vector<KvmBundle> serve; std::vector<float*> outp; std::vector<int> exact_snaps;
  for (int s = 0; s < steps; ++s) if (r.p(50)) exact_snaps.push_back(s);
  std::vector<std::vector<float>> xs;
  auto mkreq = [&](int ver, bool exact, KvmBundleBuilder& bb, std::vector<int>* joins) {
    float* o = ex.arena(N); outp.push_back(o); int ob = ex.add_view(o, 1, N);
    std::vector<float> xr = rand_ints(r, N, -6, 6); float* xp = ex.arena(N); upload(xp, xr); xs.push_back(xr); int bxr = ex.add_view(xp, 1, N);
    KvmTaskOpts to; to.wait_var = 0; to.wait_ver = ver; to.workload = 1 + (int)serve.size(); to.priority = 20;
    int prev = -1; if (spins && r.p(50)) { KvmTaskOpts so; so.workload = to.workload; prev = bb.add("spin", {I(r.in(1, 120))}, so); }
    int p = bb.add("predict", {Bf(bxr), Bf(bw), Bf(ob), I(N)}, to); if (prev >= 0) bb.edge(prev, p);
    if (joins) joins->push_back(p);
    rds.push_back({exact ? ver : -1, ver, ob, exact});
  };
  for (int s : exact_snaps) {
    KvmBundleBuilder bb; std::vector<int> ps; int K = r.in(1, 3); for (int k = 0; k < K; ++k) mkreq(s, true, bb, &ps);
    KvmTaskOpts jo; jo.pub_var = 1 + s; jo.pub_ver = 1; jo.workload = 1 + (int)serve.size(); int j = bb.add("noop", {}, jo); for (int p : ps) bb.edge(p, j);
    serve.push_back(bb.build());
  }
  for (int i = 0; i < r.in(2, 5); ++i) { KvmBundleBuilder bb; mkreq(r.in(0, steps), false, bb, nullptr); serve.push_back(bb.build()); }
  KvmBundleBuilder tb; int prev = -1;
  for (int s = 0; s < steps; ++s) {
    KvmTaskOpts to; to.workload = 0; to.priority = 3;
    int sp = -1; if (spins && r.p(30)) sp = tb.add("spin", {I(r.in(1, 80))}, to);
    int f = tb.add("predict", {Bf(bx), Bf(bw), Bf(bp), I(N)}, to), m = tb.add("mse_grad", {Bf(bp), Bf(bt), Bf(bx), Bf(bl), Bf(bg), I(N)}, to);
    KvmTaskOpts so = to; so.pub_var = 0; so.pub_ver = s + 1;
    if (std::find(exact_snaps.begin(), exact_snaps.end(), s) != exact_snaps.end()) { so.wait_var = 1 + s; so.wait_ver = 1; }
    int g = tb.add("sgd", {Bf(bw), Bf(bg), F(lr), I(1)}, so);
    // the train predict reads w: it must precede sgd_s (edge), and sgd_{s-1} must precede it (edge)
    tb.edge(f, m); tb.edge(m, g); if (sp >= 0) tb.edge(sp, f); if (prev >= 0) tb.edge(prev, sp >= 0 ? sp : f); prev = g;
  }
  std::vector<KvmBundle> all; all.push_back(tb.build()); for (auto& b : serve) all.push_back(b);
  ex.start();
  std::vector<int> order(serve.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i + 1;
  std::shuffle(order.begin(), order.end(), r.g); order.insert(order.begin() + r.in(0, (int)order.size()), 0);
  for (int bi : order) { CHECK(ex.submit(all[bi]) == KvmSubmit::OK, "submit"); if (r.p(30)) std::this_thread::sleep_for(std::chrono::microseconds(r.in(5, 300))); }
  ex.close();
  const KvmResult& res = ex.finish();
  if (!res.ok()) { CHECK(false, "device error %s seed %u", kvm_err_name(res.err_code), seed); return false; }
  check_trace(res, all);
  std::map<int, int> by_out; for (size_t k = 0; k < rds.size(); ++k) by_out[rds[k].ob] = (int)k;
  std::vector<int> task_of(rds.size(), -1);
  for (size_t i = 0; i < res.tasks.size(); ++i) if (res.tasks[i].op_id == KVM_OP_predict && res.tasks[i].workload_id >= 1) {
    const KvmBundle& b = all[res.tasks[i].workload_id]; int base = -1;
    for (size_t j = 0; j < res.tasks.size(); ++j) if (res.tasks[j].workload_id == res.tasks[i].workload_id) { base = (int)j; break; }
    task_of[by_out[b.args[b.tasks[(int)i - base].args_off + 2]]] = (int)i;
  }
  int bad = 0, nex_ = 0, nal_ = 0;
  for (size_t k = 0; k < rds.size(); ++k) {
    CHECK(task_of[k] >= 0, "reader %zu never ran", k); if (task_of[k] < 0) continue;
    const KvmTrace& tr = res.trace[task_of[k]];
    auto out = download(outp[k], N);
    int vb = 0;
    if (tr.ver0 < rds[k].need) ++vb;
    if (rds[k].exact) {
      if (tr.ver0 != rds[k].snap || tr.ver1 != rds[k].snap) ++vb;
      for (int i = 0; i < N; ++i) if (!same_bits(out[i], wk[rds[k].snap] * xs[k][i])) { ++vb; break; }   // predict = w * x_req, bitwise
    } else {  // at_least: output equals w_v * x for some v in [ver0, ver1]
      for (int i = 0; i < N; ++i) { bool okv = false; for (int v = tr.ver0; v <= tr.ver1 && v <= steps; ++v) okv |= same_bits(out[i], wk[v] * xs[k][i]); if (!okv) { ++vb; break; } }
    }
    if (vb) fprintf(stderr, "  seed %u scalar reader %zu (%s snap %d need %d): ver0 %d ver1 %d\n", seed, k, rds[k].exact ? "exact" : "at_least", rds[k].snap, rds[k].need, tr.ver0, tr.ver1);
    bad += vb; (rds[k].exact ? nex_ : nal_)++;
  }
  auto wf = download(ar + 3 * N, 1);
  CHECK(same_bits(wf[0], wk[steps]), "final w %g != golden %g", wf[0], wk[steps]);
  CHECK(bad == 0, "seed %u: %d scalar readers violated their snapshot", seed, bad);
  *nex += nex_; *nal += nal_; *violations += bad;
  return bad == 0;
}

static bool suite_versions(bool quick) {
  int before = g_fail, nex = 0, nal = 0, viol = 0, runs = 0;
  unsigned seeds = quick ? 4 : 30;
  for (unsigned s = 1; s <= seeds; ++s) {
    for (int blocks : {2, 4, 24}) {
      versions_vector(1000 + s * 13 + blocks, 12 + s % 10, 1 << 16, blocks, s % 2, &nex, &nal, &viol); ++runs;
      versions_scalar(5000 + s * 17 + blocks, 12 + s % 10, blocks, s % 2, &nex, &nal, &viol); ++runs;
    }
  }
  printf("  versions: %d runs (vector 64Ki-float weights and the real predict->mse_grad->sgd graph): %d exact + %d at_least readers checked, %d violations\n", runs, nex, nal, viol);
  return g_fail == before;
}

// The checker itself: hand-made bad observations must be flagged, good ones accepted.
static bool suite_checker() {
  int before = g_fail;
  Golden g; g.w = {1.f, 2.f, 3.f, 4.f};
  ReaderObs good{2, 2, 2, 2, {3.f, 3.f}}, wrongver{2, 2, 3, 3, {4.f, 4.f}}, torn{2, 2, 2, 2, {3.f, 4.f}}, early{-1, 3, 2, 2, {3.f}}, stale{2, 2, 2, 2, {2.f}}, al_ok{-1, 1, 1, 3, {2.f, 3.f, 4.f}}, al_bad{-1, 1, 1, 2, {1.f}};
  CHECK(check_reader(good, g) == 0, "good exact reader flagged");
  CHECK(check_reader(wrongver, g) > 0, "exact reader that observed another version not flagged");
  CHECK(check_reader(torn, g) > 0, "torn exact read not flagged");
  CHECK(check_reader(early, g) > 0, "reader that ran before its version not flagged");
  CHECK(check_reader(stale, g) > 0, "stale value under a fresh version not flagged");
  CHECK(check_reader(al_ok, g) == 0, "good at_least reader flagged");
  CHECK(check_reader(al_bad, g) > 0, "at_least value older than its observed versions not flagged");
  // trace checker: fabricate a run with a dependency inversion and a double execution
  KvmResult r; r.done = r.admitted = 2; r.tasks.resize(2); r.trace.resize(2);
  r.tasks[0].workload_id = r.tasks[1].workload_id = 0;
  r.trace[0] = {10, 20, 30, 50, 0, 1, -1, -1, 0, 1}; r.trace[1] = {10, 20, 40, 60, 0, 1, -1, -1, 0, 2};
  KvmBundleBuilder bb; int a = bb.add("noop", {}), b = bb.add("noop", {}); bb.edge(a, b); KvmBundle bd = bb.build();
  int f0 = g_fail, c0 = g_checks; g_mute = true; check_trace(r, {bd}); g_mute = false;
  int flagged = g_fail - f0; g_fail = f0; g_checks = c0;  // the expected failures must not count
  CHECK(flagged >= 2, "check_trace flagged %d of 2 injected faults (order inversion + exec_count 2)", flagged);
  printf("  checker: 7 hand-made reader observations classified correctly; trace checker flagged %d/2 injected faults\n", flagged);
  return g_fail == before;
}

// ================= dispatch cost =================
static bool suite_bench() {
  printf("  %-34s %8s %10s %12s %10s\n", "case", "blocks", "tasks", "us/task", "tasks/s");
  auto indep = [&](int blocks, int total, int bundle_sz) {
    KvmConfig c = small_cfg(blocks); c.live_cap = 1 << 14; c.max_tasks = total + 64; c.ring_words = 1 << 18; KvmExec ex(c); ex.start();
    for (int done = 0; done < total; done += bundle_sz) { KvmBundleBuilder bb; for (int i = 0; i < bundle_sz; ++i) bb.add("noop", {}); ex.submit(bb.build()); }
    ex.close(); const KvmResult& r = ex.finish();
    unsigned long long t0 = ~0ull, t1 = 0; for (auto& t : r.trace) { t0 = std::min(t0, t.start_ns); t1 = std::max(t1, t.end_ns); }
    double us = (t1 - t0) / 1e3; printf("  %-34s %8d %10d %12.4f %10.3g\n", "independent noop", r.grid, r.done, us / r.done, r.done / (us / 1e6));
  };
  for (int b : {2, 8, 32, 108, 0}) indep(b, 64 * 1024, 4096);
  auto chain = [&](int blocks, int len, const char* op) {
    KvmConfig c = small_cfg(blocks); c.live_cap = 1 << 13; c.ring_words = 1 << 17; KvmExec ex(c); ex.start();
    KvmBundleBuilder bb; int prev = -1; for (int i = 0; i < len; ++i) { int t = bb.add(op, op[0] == 's' ? std::vector<KvmArg>{I(0)} : std::vector<KvmArg>{}); if (prev >= 0) bb.edge(prev, t); prev = t; }
    ex.submit(bb.build()); ex.close(); const KvmResult& r = ex.finish();
    unsigned long long t0 = r.trace.front().start_ns, t1 = r.trace.back().end_ns; double us = (t1 - t0) / 1e3;
    printf("  %-34s %8d %10d %12.4f %10.3g\n", "dependent chain (hop latency)", r.grid, r.done, us / r.done, r.done / (us / 1e6));
  };
  for (int b : {2, 32, 0}) chain(b, 4000, "noop");
  return true;
}

// Small cases for compute-sanitizer (memcheck/racecheck): every code path, tiny sizes.
static bool suite_smoke() {
  int before = g_fail;
  { KvmConfig c = small_cfg(3); c.max_tasks = 4096; c.live_cap = 256; c.ring_words = 1024; long long t = 0; int d; dag_run(3, 3, 60, 6, c, 31, &t, &d); }
  { int a = 0, b = 0, v = 0; versions_vector(21, 5, 1024, 3, true, &a, &b, &v); versions_scalar(22, 5, 3, true, &a, &b, &v); }
  { KvmConfig c = small_cfg(2); c.ring_words = 256; c.live_cap = 32; KvmExec ex(c); float* a = ex.arena(300); int x = ex.add_view(a, 1, 300), y = ex.add_view(a, 1, 300);
    ex.start(); KvmBundleBuilder bb; bb.add("fill", {Bf(x), F(2), I(300)}); bb.add("mse_grad", {Bf(x), Bf(y), Bf(x), Bf(x), Bf(y), I(300)}); bb.add("matmul", {Bf(x), Bf(y), Bf(x), I(3), I(3), I(3)});
    for (int i = 0; i < 20; ++i) ex.submit(bb.build()); ex.close(); CHECK(ex.finish().ok(), "smoke ops"); }
  return g_fail == before;
}

int main(int argc, char** argv) {
  bool expect_fail = false; std::vector<std::string> want;
  for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--expect-fail") expect_fail = true; else if (a == "--quiet") g_kvm_log = 0; else if (a == "--log") g_kvm_log = atoi(argv[++i]); else want.push_back(a); }
  bool quick = getenv("KVM_TEST_QUICK") != nullptr;
  if (want.empty()) want = {"checker", "ops", "errors", "dag", "sched", "admission", "versions"};
  int dev; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p, dev));
  int drv, rt; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("GPU %s, %d SMs, sm_%d%d, driver CUDA %d.%d, runtime %d.%d%s\n", p.name, p.multiProcessorCount, p.major, p.minor, drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10, quick ? " [quick]" : "");
#ifdef KVM_BREAK_FENCE
  printf("BUILT WITH -DKVM_BREAK_FENCE (version published before the op's stores): version checks are EXPECTED to fail\n");
#endif
  int failed_suites = 0;
  for (auto& s : want) {
    int f0 = g_fail; auto t0 = std::chrono::steady_clock::now(); bool ok = true;
    printf("== %s\n", s.c_str()); fflush(stdout);
    try {
      if (s == "checker") ok = suite_checker(); else if (s == "ops") ok = suite_ops(); else if (s == "errors") ok = suite_errors();
      else if (s == "dag") ok = suite_dag(quick); else if (s == "sched") ok = suite_sched(); else if (s == "admission") ok = suite_admission();
      else if (s == "versions") ok = suite_versions(quick); else if (s == "bench") ok = suite_bench(); else if (s == "smoke") ok = suite_smoke();
      else { fprintf(stderr, "unknown suite %s\n", s.c_str()); return 2; }
    } catch (const std::exception& e) { fprintf(stderr, "  EXCEPTION: %s\n", e.what()); ++g_fail; ok = false; }
    ok = ok && g_fail == f0;
    printf("== %s: %s (%.1fs, %d new failures)\n", s.c_str(), ok ? "PASS" : "FAIL", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), g_fail - f0);
    failed_suites += !ok;
  }
  printf("%d checks, %d failed, %d suites failed\n", g_checks, g_fail, failed_suites);
  if (expect_fail) { printf(failed_suites ? "NEGATIVE TEST PASS: the harness detected the injected violation\n" : "NEGATIVE TEST FAIL: no violation detected\n"); return failed_suites ? 0 : 1; }
  printf(failed_suites ? "FAIL\n" : "PASS\n");
  return failed_suites ? 1 : 0;
}
