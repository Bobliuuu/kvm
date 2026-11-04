// End-to-end adversarial scenarios against the real executor (host driver + persistent kernel), independent of tests/gpu.
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude -DKVM_EXEC_NO_MAIN tests/litmus/e2e.cu -o e2e       (-DKVM_BREAK_FENCE: negative control)
//   ./e2e dag|versions|ring|bigrecord|probe_timeout|probe_deadlock|probe_aging|probe_spin|probe_closerace|all  [scale]
// Every scenario prints "E2E <name> ... PASS|FAIL|FINDING" with counts. FINDING = the executor behaved contrary to its documented
// contract (not a crash of the harness).
#ifndef KVM_EXEC_CU
#define KVM_EXEC_CU "../../kvm_exec.cu"      // -DKVM_EXEC_CU='"/path/kvm_exec.cu"' -DKVM_NEW to build against the slot-recycling executor (feat/executor 5b869aa)
#endif
#include KVM_EXEC_CU
#include <thread>
#include <random>

static int g_fail = 0, g_find = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++g_fail; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static KvmArg Bf(int i) { return KvmArg::buf(i); }
static KvmArg I(int v) { return KvmArg::i32(v); }
static KvmArg F(float v) { return KvmArg::f32(v); }

// ---------- random layered DAGs with exact-integer dataflow, concurrent producers, tiny ring/live_cap ----------
#ifdef KVM_NEW
#define SLOTS_FOR(i) (((i) % 2) ? (1 << 17) : 128)    /* tiny slot pool: constant recycling under load */
#else
#define SLOTS_FOR(i) (1 << 17)
#endif
struct Node { int op; int p0, p1; float v; };
static void s_dag(int scale) {
  const int SZ = 256, N = 24 * 1;       // nodes per bundle
  long long total_tasks = 0, bad_val = 0, bad_exec = 0, bad_edge = 0, edges = 0; int runs = 0;
  for (int cfgi = 0; cfgi < 4 * scale; ++cfgi) {
    std::mt19937 rng(1000 + cfgi);
    KvmConfig c; c.blocks = (cfgi % 3 == 0) ? 0 : 2 + (cfgi % 9); c.live_cap = (cfgi % 2) ? 32 : 64; c.max_tasks = SLOTS_FOR(cfgi); c.ring_words = (cfgi % 4 == 3) ? 576 : 1024;   // record ~520 words: 576 forces a pad on nearly every wrap
    c.num_vars = 4; c.aging_us = (cfgi % 5 == 0) ? 3 : 0; c.timeout_us = 120ull * 1000000;
    const int B = 600;
    KvmExec ex(c);
    float* arena = ex.arena((size_t)B * N * SZ);
    std::vector<std::vector<Node>> g(B); std::vector<std::vector<float>> val(B);
    for (int b = 0; b < B; ++b) for (int i = 0; i < N; ++i) ex.add_view(arena + ((size_t)b * N + i) * SZ, 1, SZ);
    ex.start();
    std::vector<std::vector<std::vector<int>>> succs(B);
    std::vector<KvmBundle> bundles(B);
    for (int b = 0; b < B; ++b) {
      KvmBundleBuilder bb; g[b].resize(N); val[b].resize(N); succs[b].assign(N, {});
      for (int i = 0; i < N; ++i) {
        Node& n = g[b][i]; int layer = i / 4;      // 4 nodes per layer, preds from the previous layer
        int kind = layer == 0 ? 0 : (int)(rng() % 3);
        KvmTaskOpts o; o.priority = rng() % KVM_NBUCKETS; o.workload = b;
        int base = b * N;
        if (kind == 0) { n.op = 0; n.v = (float)(1 + rng() % 63); bb.add("fill", {Bf(base + i), F(n.v), I(SZ)}, o); val[b][i] = n.v; }
        else if (kind == 1) { n.op = 1; n.p0 = (layer - 1) * 4 + rng() % 4; bb.add("scale", {Bf(base + n.p0), Bf(base + i), F(1.0f), I(SZ)}, o); val[b][i] = val[b][n.p0]; bb.edge(n.p0, i); succs[b][n.p0].push_back(i); }
        else { n.op = 2; n.p0 = (layer - 1) * 4 + rng() % 4; do n.p1 = (layer - 1) * 4 + rng() % 4; while (n.p1 == n.p0);
          bb.add("add", {Bf(base + n.p0), Bf(base + n.p1), Bf(base + i), I(SZ)}, o); val[b][i] = val[b][n.p0] + val[b][n.p1];
          bb.edge(n.p0, i); bb.edge(n.p1, i); succs[b][n.p0].push_back(i); succs[b][n.p1].push_back(i); }
      }
      bundles[b] = bb.build(); for (int i = 0; i < N; ++i) bundles[b].tasks[i].reserved0 = i;
    }
    const int T = 1 + cfgi % 4; std::atomic<int> next{0}; std::atomic<int> bad_submit{0};
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back([&] { for (int b; (b = next++) < B;) if (ex.submit(bundles[b], 60000) != KvmSubmit::OK) ++bad_submit; });
    for (auto& t : th) t.join();
    const KvmResult& r = ex.finish();
    EXPECT(r.ok(), "cfg %d device error %s", cfgi, r.ok() ? "" : kvm_err_name(r.err_code)); EXPECT(bad_submit == 0, "submit failures %d", bad_submit.load());
    EXPECT(r.done == B * N && r.admitted == B * N, "done %d admitted %d", r.done, r.admitted);
    std::vector<float> all((size_t)B * N * SZ); CK(cudaMemcpy(all.data(), arena, all.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<std::vector<int>> slot_of(B, std::vector<int>(N, -1));
#ifndef KVM_NEW   // slots are unique per task only without recycling (the new executor reuses slots and overwrites reserved0 with the window id)
    for (size_t s = 0; s < r.tasks.size(); ++s) { slot_of[r.tasks[s].workload_id][r.tasks[s].reserved0] = (int)s; if (r.trace[s].exec_count != 1) ++bad_exec; }
#endif
    for (int b = 0; b < B; ++b) for (int i = 0; i < N; ++i) {
      for (int k = 0; k < SZ; ++k) if (all[((size_t)b * N + i) * SZ + k] != val[b][i]) { ++bad_val; break; }
#ifndef KVM_NEW
      int s = slot_of[b][i]; if (s < 0) { ++bad_exec; continue; }
      for (int sn : succs[b][i]) { int t = slot_of[b][sn]; ++edges; if (t >= 0 && r.trace[t].start_ns < r.trace[s].end_ns) ++bad_edge; }
#else
      edges += (long long)succs[b][i].size();
#endif
    }
    total_tasks += r.done; ++runs;
  }
  printf("E2E dag: %d configs, %lld tasks, %lld edges: wrong_values=%lld bad_exec_count=%lld edge_order_violations=%lld  %s\n", runs, total_tasks, edges, bad_val, bad_exec, bad_edge, (bad_val || bad_exec || bad_edge || g_fail) ? "FAIL" : "PASS");
  g_fail += (int)(bad_val + bad_exec + bad_edge);
}

// ---------- version pipelines ordered ONLY by version variables (no graph edges), readers admitted before writers ----------
//   W_k: wait done>=k-1, fill(weights, k), publish weight=k.   R_k: wait weight>=k, copy(weights -> out_k), publish done=k.
static void s_versions(int scale) {
  const int P = 24, K = 120, SZ = 2048;
  long long stale = 0, runs = 0, tasks = 0;
  for (int rep = 0; rep < 3 * scale; ++rep) {
    KvmConfig c; c.blocks = (rep % 2) ? 0 : 6; c.live_cap = 8192; c.max_tasks = 1 << 16; c.num_vars = 2 * P; c.ring_words = 1 << 14; c.timeout_us = 120ull * 1000000; c.aging_us = rep == 2 ? 2 : 0;
    KvmExec ex(c);
    float* wa = ex.arena((size_t)P * SZ); float* oa = ex.arena((size_t)P * K * SZ);
    std::vector<int> wb(P), ob((size_t)P * K);
    for (int p = 0; p < P; ++p) wb[p] = ex.add_view(wa + (size_t)p * SZ, 1, SZ);
    for (int p = 0; p < P; ++p) for (int k = 0; k < K; ++k) ob[(size_t)p * K + k] = ex.add_view(oa + ((size_t)p * K + k) * SZ, 1, SZ);
    ex.start();
    // readers first (they park), then writers, in separate bundles, per pipeline
    std::vector<KvmBundle> rb, wbund;
    for (int p = 0; p < P; ++p) {
      KvmBundleBuilder r, w;
      for (int k = 1; k <= K; ++k) { KvmTaskOpts o; o.wait_var = 2 * p; o.wait_ver = k; o.pub_var = 2 * p + 1; o.pub_ver = k; o.priority = k % 7; r.add("copy", {Bf(wb[p]), Bf(ob[(size_t)p * K + k - 1]), I(SZ)}, o); }
      for (int k = 1; k <= K; ++k) { KvmTaskOpts o; o.wait_var = 2 * p + 1; o.wait_ver = k - 1; o.pub_var = 2 * p; o.pub_ver = k; o.priority = 20; w.add("fill", {Bf(wb[p]), F((float)k), I(SZ)}, o); }
      rb.push_back(r.build()); wbund.push_back(w.build());
    }
    std::thread t1([&] { for (int p = 0; p < P; ++p) ex.submit(rb[p]); });
    t1.join();
    std::thread t2([&] { for (int p = 0; p < P; ++p) ex.submit(wbund[p]); });
    t2.join();
    const KvmResult& r = ex.finish();
    EXPECT(r.ok(), "rep %d device error %s (slot %d a=%d b=%d)", rep, kvm_err_name(r.err_code), r.err_slot, r.err_a, r.err_b);
    if (!r.ok()) continue;
    std::vector<float> o((size_t)P * K * SZ); CK(cudaMemcpy(o.data(), oa, o.size() * 4, cudaMemcpyDeviceToHost));
    for (int p = 0; p < P; ++p) for (int k = 1; k <= K; ++k) for (int i = 0; i < SZ; ++i) if (o[((size_t)p * K + k - 1) * SZ + i] != (float)k) { ++stale; break; }
    for (int v : {0, 1}) EXPECT(r.version[v] == K, "final version[%d]=%d", v, r.version[v]);
    tasks += r.done; ++runs;
  }
  printf("E2E versions: %lld runs, %lld tasks, %d pipelines x %d versions, %d floats/version: reads_not_equal_exact_version=%lld  %s\n", runs, tasks, P, K, SZ, stale, (stale || g_fail) ? "FAIL" : "PASS");
#ifdef KVM_BREAK_FENCE
  printf("  (built with -DKVM_BREAK_FENCE: FAIL above is the EXPECTED outcome of the negative control)\n");
#endif
  g_fail += (int)stale;
}

// ---------- admission ring: wraparound, pads, minimum ring, concurrent producers, one huge record ----------
static void s_ring(int scale) {
  long long bad = 0, n = 0;
  for (int rw : {64, 72, 96, 256, 4096}) {
    KvmConfig c; c.blocks = 4; c.live_cap = 64; c.max_tasks = 1 << 17; c.ring_words = rw; c.num_vars = 2; c.timeout_us = 120ull * 1000000;
    const int NB = 12000 * scale / 4 + 1000; KvmExec ex(c);
    float* a = ex.arena(NB); for (int i = 0; i < NB; ++i) ex.add_view(a + i, 1, 1);
    ex.start(); std::atomic<int> next{0}; std::vector<std::thread> th; std::atomic<int> sf{0};
    for (int t = 0; t < 3; ++t) th.emplace_back([&] { for (int b; (b = next++) < NB;) {
      KvmBundleBuilder bb; int k = 1 + b % 2;     // 1 or 2 tasks: record = 24..40 words, so rw=64 wraps constantly and needs pads
      bb.add("fill", {Bf(b), F((float)(b % 1000 + 1)), I(1)}); if (k == 2) bb.add("noop", {}, {});
      if (ex.submit(bb.build(), 60000) != KvmSubmit::OK) ++sf; } });
    for (auto& t : th) t.join();
    const KvmResult& r = ex.finish(); EXPECT(r.ok() && sf == 0, "rw=%d err=%s submit_fail=%d", rw, kvm_err_name(r.err_code), sf.load());
    std::vector<float> h(NB); CK(cudaMemcpy(h.data(), a, NB * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NB; ++i) if (h[i] != (float)(i % 1000 + 1)) ++bad;
    n += NB; printf("  ring_words=%-5d bundles=%d pads=%d backpressure_waits=%d\n", rw, NB, ex.stats().pads, ex.stats().backpressure_waits);
  }
  printf("E2E ring: %lld bundles through rings of 64..4096 words, 3 producer threads: wrong_or_missing=%lld  %s\n", n, bad, (bad || g_fail) ? "FAIL" : "PASS"); g_fail += (int)bad;
}
static void s_bigrecord(int scale) {   // > glibc's non-temporal memcpy threshold? a 20 MB record
  KvmConfig c; c.blocks = 0; c.live_cap = 1 << 17; c.max_tasks = 1 << 18; c.ring_words = 1 << 23; c.num_vars = 2; c.timeout_us = 120ull * 1000000;
#ifndef KVM_NEW
  c.succ_cap = 1 << 20; c.args_cap = 1 << 20;
#else
  c.succ_cap = 1 << 20;
#endif
 
  KvmExec ex(c); const int NT = 100000;
  float* a = ex.arena(NT); for (int i = 0; i < NT; ++i) ex.add_view(a + i, 1, 1);
  ex.start(); long long bad = 0; (void)scale;
  { KvmBundleBuilder bb; for (int i = 0; i < NT; ++i) bb.add("fill", {Bf(i), F(1.0f), I(1)});
    if (ex.submit(bb.build(), 60000) != KvmSubmit::OK) { printf("E2E bigrecord submit failed: %s\n", ex.last_error().c_str()); ++g_fail; } }   // one record: slots are not recycled in 083513e
  const KvmResult& r = ex.finish(); std::vector<float> h(NT); CK(cudaMemcpy(h.data(), a, NT * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < NT; ++i) if (h[i] != 1.0f) ++bad;
  printf("E2E bigrecord: %d tasks in one %.1f MB record, err=%s wrong=%lld  %s\n", r.done, (100000 * 64 + 100000 * 4) / 1e6, kvm_err_name(r.err_code), bad, (bad || !r.ok()) ? "FAIL" : "PASS"); g_fail += (int)bad + !r.ok();
}

// ---------- probes for suspected contract violations ----------
static void s_probe_timeout() {  // healthy but IDLE for longer than timeout_s: a stall detector must not fire; a total-lifetime cap does
  KvmConfig c; c.blocks = 4; c.live_cap = 64; c.max_tasks = 1024; c.timeout_us = 2ull * 1000000; c.num_vars = 2; c.ring_words = 1024;
  KvmExec ex(c); int b = ex.add_buffer(1, 1); ex.start();
  KvmBundleBuilder bb; bb.add("fill", {Bf(b), F(1), I(1)}); KvmSubmit s0 = ex.submit(bb.build());
  std::this_thread::sleep_for(std::chrono::milliseconds(3500));
  KvmSubmit s1 = ex.submit(bb.build(), 500);
  const KvmResult& r = ex.finish();
  bool find = !r.ok() && r.err_code == KVM_ERR_TIMEOUT;
  printf("E2E probe_timeout: first submit %s, healthy executor idle 3.5s with timeout_s=2, second submit %s, final err=%s done=%d/%d  %s\n",
         kvm_submit_name(s0), kvm_submit_name(s1), kvm_err_name(r.err_code), r.done, r.admitted, find ? "FINDING (timeout is a lifetime cap, not a stall detector)" : "ok");
  g_find += find;
}
static void s_probe_timeout_busy() {  // a healthy long job: tasks keep completing for > timeout_s
  KvmConfig c; c.blocks = 4; c.live_cap = 64; c.max_tasks = 4096; c.timeout_us = 2ull * 1000000; c.num_vars = 2; c.ring_words = 4096;
  KvmExec ex(c); ex.start(); int ok = 0; double t0 = now_s(); KvmSubmit last = KvmSubmit::OK;
  for (int i = 0; i < 40 && last == KvmSubmit::OK; ++i) { KvmBundleBuilder bb; bb.add("spin", {I(100000)}); last = ex.submit(bb.build(), 2000); ok += last == KvmSubmit::OK; std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
  const KvmResult& r = ex.finish();
  bool find = !r.ok();
  printf("E2E probe_timeout_busy: steady stream (a 100ms task every 100ms, always progressing), timeout_s=2: submitted %d/40 over %.1fs, err=%s done=%d/%d  %s\n", ok, now_s() - t0, kvm_err_name(r.err_code), r.done, r.admitted, find ? "FINDING (healthy progressing run killed)" : "ok");
  g_find += find;
}
static void s_probe_deadlock() {  // cross-bundle wait: parked tasks occupy live_cap, so the publisher's bundle can never be admitted
  KvmConfig c; c.blocks = 4; c.live_cap = 8; c.max_tasks = 1024; c.timeout_us = 3ull * 1000000; c.num_vars = 2; c.ring_words = 1024;
  KvmExec ex(c); int b = ex.add_buffer(1, 1), o = ex.add_buffer(1, 1); ex.start();
  KvmBundleBuilder waiters; for (int i = 0; i < 8; ++i) { KvmTaskOpts t; t.wait_var = 0; t.wait_ver = 1; waiters.add("copy", {Bf(b), Bf(o), I(1)}, t); }
  KvmBundleBuilder pub; { KvmTaskOpts t; t.pub_var = 0; t.pub_ver = 1; pub.add("fill", {Bf(b), F(1), I(1)}, t); }
  double t0 = now_s(); ex.submit(waiters.build()); ex.submit(pub.build()); const KvmResult& r = ex.finish();
  bool find = !r.ok();
  printf("E2E probe_deadlock: 8 waiters (one full live_cap) then the publisher bundle, live_cap=8: err=%s done=%d/%d after %.1fs  %s\n", kvm_err_name(r.err_code), r.done, r.admitted, now_s() - t0,
         find ? "FINDING (valid cross-bundle dependency deadlocks until the device timeout)" : "ok");
  g_find += find;
}
static void s_probe_aging() {   // one worker, 12 ready tasks of priorities 0..11: the run order shows what the aging flag does
  for (int aging : {0, 1000000, -5, -2000000}) {
    KvmConfig c; c.blocks = 2; c.live_cap = 64; c.max_tasks = 1024; c.timeout_us = 3ull * 1000000; c.num_vars = 2; c.ring_words = 1024; c.aging_us = aging;
    KvmExec ex(c); ex.start(); double t0 = now_s();
    KvmBundleBuilder bb; int gate = bb.add("spin", {I(2000)}, [] { KvmTaskOpts o; o.priority = 31; return o; }());   // occupies the worker while the rest become ready
    for (int i = 0; i < 12; ++i) { KvmTaskOpts o; o.priority = i; o.workload = i; int t = bb.add("spin", {I(50)}, o); bb.edge(gate, t); }
    ex.submit(bb.build());
    const KvmResult& r = ex.finish();
    std::vector<std::pair<unsigned long long, int>> ord;
    for (size_t s = 1; s < r.trace.size(); ++s) ord.push_back({r.trace[s].start_ns, r.tasks[s].priority});
    std::sort(ord.begin(), ord.end());
    std::string o; for (auto& p : ord) o += std::to_string(p.second) + " ";
    printf("E2E probe_aging: aging_us=%d: err=%s done=%d/%d in %.2fs, run order of priorities 0..11: %s\n", aging, kvm_err_name(r.err_code), r.done, r.admitted, now_s() - t0, o.c_str());
  }
}
static void s_probe_spin() {  // a single long task: is the device timeout enforced at all?
  KvmConfig c; c.blocks = 4; c.live_cap = 64; c.max_tasks = 1024; c.timeout_us = 2ull * 1000000; c.num_vars = 2; c.ring_words = 1024;
  KvmExec ex(c); ex.start(); double t0 = now_s();
  KvmBundleBuilder bb; bb.add("spin", {I(9000000)}); ex.submit(bb.build()); const KvmResult& r = ex.finish();   // 9 s task, 2 s timeout
  double dt = now_s() - t0;
  printf("E2E probe_spin: one spin task of 9 s with timeout_s=2: finish() returned after %.1f s, err=%s  %s\n", dt, kvm_err_name(r.err_code), dt > 5 ? "FINDING (no per-task watchdog: timeout cannot bound a long task; spin arg is unvalidated, INT_MAX us = 35 min)" : "ok");
  g_find += dt > 5;
}
static void s_probe_closerace() {  // close() racing submits from several threads: every OK-submitted bundle must run, none after CLOSED
  long long bad = 0, ran = 0, tot = 0;
  for (int rep = 0; rep < 200; ++rep) {
    KvmConfig c; c.blocks = 3; c.live_cap = 64; c.max_tasks = 1 << 14; c.ring_words = 256; c.num_vars = 2; c.timeout_us = 30ull * 1000000;
    KvmExec ex(c); float* a = ex.arena(2000); for (int i = 0; i < 2000; ++i) ex.add_view(a + i, 1, 1); ex.start();
    std::atomic<int> okc{0}; std::atomic<int> next{0}; std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t) th.emplace_back([&] { for (int b; (b = next++) < 2000;) { KvmBundleBuilder bb; bb.add("fill", {Bf(b), F(7), I(1)}); if (ex.submit(bb.build(), 2000) == KvmSubmit::OK) ++okc; } });
    std::this_thread::sleep_for(std::chrono::microseconds(100 + rep * 37)); ex.close();
    for (auto& t : th) t.join();
    const KvmResult& r = ex.finish(); std::vector<float> h(2000); CK(cudaMemcpy(h.data(), a, 8000, cudaMemcpyDeviceToHost));
    int filled = 0; for (float x : h) filled += x == 7.f;
    if (!r.ok() || r.admitted != okc || filled != okc) { ++bad; if (bad < 4) printf("  rep %d: err=%s admitted=%d ok_submits=%d filled=%d\n", rep, kvm_err_name(r.err_code), r.admitted, okc.load(), filled); }
    ran += okc; ++tot;
  }
  printf("E2E probe_closerace: %lld close-vs-submit races, %lld bundles accepted and run: mismatches=%lld  %s\n", tot, ran, bad, bad ? "FAIL" : "PASS"); g_fail += (int)bad;
}


// ---------- device-side admission checks with HOST validation bypassed (defense in depth against a corrupt ring record) ----------
// One case per process (a sticky CUDA error would poison later cases). Usage: ./e2e devfuzz <case_seed>   exit: 0 clean, 3 CUDA error / crash
static void s_devfuzz(unsigned seed) {
  std::mt19937 rng(seed); static const int32_t V[] = {0, 1, 2, -1, -2, 7, 31, 32, 63, 64, 255, 256, 65536, 1 << 24, 2147483647, 2147483646, (int32_t)0x80000000, (int32_t)0x80000001};
  KvmConfig c; c.blocks = 4; c.live_cap = 64; c.max_tasks = 1024; c.num_vars = 4; c.ring_words = 1024; c.timeout_us = 2ull * 1000000;
  KvmExec ex(c); int a = ex.add_buffer(8, 8), b = ex.add_buffer(8, 8), o = ex.add_buffer(8, 8), w = ex.add_buffer(1, 1); ex.start();
  KvmBundleBuilder bb;
  int t0 = bb.add("fill", {Bf(a), F(2), I(64)}); int t1 = bb.add("fill", {Bf(b), F(3), I(64)});
  KvmTaskOpts op; op.pub_var = 0; op.pub_ver = 1; int t2 = bb.add("add", {Bf(a), Bf(b), Bf(o), I(64)}, op);
  KvmTaskOpts ow; ow.wait_var = 0; ow.wait_ver = 1; int t3 = bb.add("copy", {Bf(o), Bf(w), I(1)}, ow);
  bb.edge(t0, t2); bb.edge(t1, t2); bb.edge(t2, t3);
  KvmBundle bu = bb.build();
  static const char* fn[] = {"op_id", "succ_off", "succ_cnt", "indegree", "priority", "deadline_us", "args_off", "args_words", "wait_var", "wait_ver", "pub_var", "pub_ver", "workload", "flags", "r0", "r1"};
  std::string desc; int kind = rng() % 4;
  if (seed >= 1000) {   // deterministic boundary sweep: task i, field in {succ_off, succ_cnt, args_off, args_words}, value in {INT_MAX, INT_MAX-1, INT_MIN}
    int k = seed - 1000, vi = k % 3, f = (k / 3) % 4, i = (k / 12) % 4; static const int FI[] = {1, 2, 6, 7}; static const int32_t BV[] = {2147483647, 2147483646, (int32_t)0x80000000};
    ((int32_t*)&bu.tasks[i])[FI[f]] = BV[vi]; desc = "task[" + std::to_string(i) + "]." + fn[FI[f]] + "=" + std::to_string(BV[vi]);
  } else if (kind <= 1) { int i = rng() % bu.tasks.size(), f = rng() % 16; int32_t v = V[rng() % (sizeof V / 4)]; ((int32_t*)&bu.tasks[i])[f] = v; desc = "task[" + std::to_string(i) + "]." + fn[f] + "=" + std::to_string(v); }
  else if (kind == 2) { int k = rng() % bu.succ.size(); int32_t v = V[rng() % (sizeof V / 4)]; bu.succ[k] = v; desc = "succ[" + std::to_string(k) + "]=" + std::to_string(v); }
  else { int k = rng() % bu.args.size(); int32_t v = V[rng() % (sizeof V / 4)]; bu.args[k] = v; desc = "args[" + std::to_string(k) + "]=" + std::to_string(v); }
  printf("DEVFUZZ-PLAN seed=%u %s\n", seed, desc.c_str()); fflush(stdout);
  KvmSubmit st = ex.submit(bu, 2000, /*validate=*/false);
  const KvmResult& r = ex.finish();
  printf("DEVFUZZ seed=%u %-34s submit=%s err=%s done=%d/%d\n", seed, desc.c_str(), kvm_submit_name(st), kvm_err_name(r.err_code), r.done, r.admitted);
}

int main(int argc, char** argv) {
  std::string t = argc > 1 ? argv[1] : "all"; int scale = argc > 2 ? atoi(argv[2]) : 1; g_kvm_log = 1;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); printf("GPU %s, %d SMs\n", p.name, p.multiProcessorCount);
  try {
    if (t == "devfuzz") { s_devfuzz((unsigned)scale); return 0; }
    bool all = t == "all";
    if (all || t == "dag") s_dag(scale);
    if (all || t == "versions") s_versions(scale);
    if (all || t == "ring") s_ring(scale);
    if (all || t == "bigrecord") s_bigrecord(scale);
    if (all || t == "probe_closerace") s_probe_closerace();
    if (all || t == "probe_timeout") s_probe_timeout();
    if (all || t == "probe_timeout_busy") s_probe_timeout_busy();
    if (all || t == "probe_deadlock") s_probe_deadlock();
    if (all || t == "probe_aging") s_probe_aging();
    if (all || t == "probe_spin") s_probe_spin();
  } catch (const std::exception& e) { printf("EXCEPTION %s\n", e.what()); return 3; }
  printf("E2E summary: failures=%d findings=%d\n", g_fail, g_find);
  return g_fail ? 1 : 0;
}
