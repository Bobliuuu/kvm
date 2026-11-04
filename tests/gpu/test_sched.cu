// Scheduler tests and measurements for the KVM executor (include/kvm_sched.h, include/kvm_sched.cuh).
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN tests/gpu/test_sched.cu -o test_sched
//   ./test_sched [suite...]   functional: cost share reserve edf inherit rand      (default)
//                             measurements: arbiter rebalance baselines            (explicit; they print the numbers quoted in results/)
//                             sanitizer smoke: smoke
//   ./test_sched --expect-fail starve_neg inherit_neg    # negative tests: exit 0 only if the harness DETECTS the violation
// Everything here is workload-agnostic: the scheduler sees workload_id, priority, deadline_us and the op cost table. "train",
// "serve", "stream" and "chain" below are only names for task streams the test generates.
#define main exec_suite_main
#include "test_exec.cu"
#undef main
#include "../../include/kvm_sched.h"

// ---------------- trace analysis ----------------
struct TV { int cls, op, flags; double ready, start, end; };   // microseconds since the first admission
static void check_once(const KvmResult& r, int total) {
  int bad = 0; for (auto& t : r.trace) bad += t.exec_count != 1;
  CHECK(r.done == total && r.admitted == total && bad == 0, "done %d admitted %d (want %d), %d tasks with exec_count != 1", r.done, r.admitted, total, bad);
}
static std::vector<TV> tview(const KvmResult& r) {
  unsigned long long t0 = ~0ull; for (auto& t : r.trace) if (t.admit_ns) t0 = std::min(t0, t.admit_ns);
  std::vector<TV> v(r.trace.size());
  for (size_t i = 0; i < v.size(); ++i) {
    const KvmTrace& t = r.trace[i];
    v[i] = {r.tasks[i].workload_id, r.tasks[i].op_id, t.flags, (t.ready_ns - t0) / 1e3, (t.start_ns - t0) / 1e3, (t.end_ns - t0) / 1e3};
  }
  return v;
}
static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p / 100.0 * v.size()))]; }
static double median(std::vector<double> v) { return pct(v, 50); }
struct Spread { double lo, med, hi; };
static Spread spread(std::vector<double> v) { std::sort(v.begin(), v.end()); return {v.front(), v[v.size() / 2], v.back()}; }

static int g_sms = 0;
static KvmConfig sched_cfg(int blocks, int nc, int mode = KVM_SCHED_DYNAMIC) {
  KvmConfig c = small_cfg(blocks); c.sched.on = true; c.sched.mode = mode; c.sched.num_classes = nc;
  return c;
}
static KvmBundle spin_bundle(int n, int us, int cls, int prio = 0, int deadline_us = -1) {
  KvmBundleBuilder b; KvmTaskOpts o; o.workload = cls; o.priority = prio; o.deadline_us = deadline_us;
  for (int i = 0; i < n; ++i) b.add("spin", {I(us)}, o);
  return b.build();
}
// Submit bundles alternating between the given lists so that every class keeps a backlog.
static void submit_all(KvmExec& ex, const std::vector<KvmBundle>& bs) {
  for (auto& b : bs) { KvmSubmit s = ex.submit(b, 30000); CHECK(s == KvmSubmit::OK, "submit %s %s", kvm_submit_name(s), ex.last_error().c_str()); }
}

// ================= cost table =================
static bool suite_cost() {
  int before = g_fail;
  CHECK(kvm_cost_names_registered(), "a name in kvm_ops_cost.inc is not a registered operator");
  int32_t a[8] = {0, 1, 100000, 0, 0, 0, 0, 0};
  KvmCost c = kvm_cost(KVM_OP_copy, a);
  CHECK(c.res == KVM_RES_BW && c.bytes == 800000ull, "copy 1e5 floats: res %d bytes %llu", c.res, (unsigned long long)c.bytes);
  int32_t s[8] = {0, 1, 0, 100, 50, 30, 0, 0};  // matmul a,b,out,m,n,k
  c = kvm_cost(KVM_OP_matmul, s);
  CHECK(c.res == KVM_RES_MIXED && c.bytes == 4ull * (100 * 30 + 30 * 50 + 100 * 50) && c.flops == 2ull * 100 * 50 * 30, "matmul bytes %llu flops %llu", (unsigned long long)c.bytes, (unsigned long long)c.flops);
  int32_t small[8] = {0, 1, 256, 0, 0, 0, 0, 0};
  c = kvm_cost(KVM_OP_copy, small);
  CHECK(c.res == KVM_RES_LAT, "a 2 KiB copy must be latency-bound, got res %d", c.res);
  c = kvm_cost(KVM_OP_spin, a); CHECK(c.res == KVM_RES_LAT && c.bytes == 0, "spin is latency-bound");
  c = kvm_cost(9999, a); CHECK(c.res == KVM_RES_MIXED && c.bytes == 0, "unknown op must get the default (MIXED, 0 bytes)");
  int listed = 0; for (const KvmOpInfo& o : kKvmOps) { bool in = false; for (const char* n : kKvmCostNames) in = in || !strcmp(n, o.name); listed += in; }
  printf("  cost table: %zu entries, all registered; %d of %d registered ops listed, the rest use the default (MIXED, 0 bytes); copy/matmul/default estimates checked\n",
         sizeof(kKvmCostNames) / sizeof(kKvmCostNames[0]), listed, KVM_NUM_OPS);
  return g_fail == before;
}

// ================= weighted fair sharing and work conservation =================
// Window = [first start, min over classes of its last start]: while every class still has a backlog.
static std::vector<double> window_share(const std::vector<TV>& v, int nc, double* span_us) {
  std::vector<double> last(nc, 0), first(nc, 1e30); double w0 = 1e30;
  for (auto& t : v) { last[t.cls] = std::max(last[t.cls], t.start); first[t.cls] = std::min(first[t.cls], t.start); w0 = std::min(w0, t.start); }
  double w1 = 1e30; for (int c = 0; c < nc; ++c) if (last[c] > 0) w1 = std::min(w1, last[c]);
  std::vector<double> cta(nc, 0);
  for (auto& t : v) { double a = std::max(t.start, w0), b = std::min(t.end, w1); if (b > a) cta[t.cls] += b - a; }
  *span_us = w1 - w0; return cta;
}
static bool suite_share() {
  int before = g_fail;
  const int W = 32;
  struct Case { const char* name; std::vector<int> weights; };
  for (const Case& cs : std::vector<Case>{{"3:1", {3, 1}}, {"1:2:4:1", {1, 2, 4, 1}}}) {
    const int nc = (int)cs.weights.size();
    KvmConfig c = sched_cfg(W + 2, nc); c.max_tasks = 1 << 16; c.live_cap = 1 << 13;
    for (int k = 0; k < nc; ++k) c.sched.p.cls[k].weight = cs.weights[k];
    KvmExec ex(c); ex.start();
    std::vector<KvmBundle> bs;
    for (int rep = 0; rep < 12; ++rep) for (int k = 0; k < nc; ++k) bs.push_back(spin_bundle(300, 200, k));
    submit_all(ex, bs);
    const KvmResult& r = ex.finish();
    CHECK(r.ok(), "device error %s", kvm_err_name(r.err_code));
    check_once(r, 12 * nc * 300);
    auto v = tview(r); double span; auto cta = window_share(v, nc, &span);
    double tot = 0, wt = 0; for (int k = 0; k < nc; ++k) { tot += cta[k]; wt += cs.weights[k]; }
    printf("  weights %s: CTA-time shares", cs.name);
    for (int k = 0; k < nc; ++k) {
      double want = cs.weights[k] / wt, got = cta[k] / tot;
      printf(" %.3f(want %.3f)", got, want);
      CHECK(std::fabs(got - want) < 0.03, "class %d share %.3f vs %.3f", k, got, want);
    }
    printf(", window %.1f ms, occupancy %.1f%%\n", span / 1e3, 100 * tot / (span * W));
    { std::vector<std::vector<std::pair<double,double>>> byb(c.blocks); for (size_t i = 0; i < r.trace.size(); ++i) byb[r.trace[i].block].push_back({v[i].start, v[i].end});
      std::vector<double> gaps; for (auto& b : byb) { std::sort(b.begin(), b.end()); for (size_t i = 1; i < b.size(); ++i) gaps.push_back(b[i].first - b[i-1].second); }
      printf("    gap between tasks on one CTA: p50 %.2f us p90 %.2f us p99 %.2f us\n", pct(gaps, 50), pct(gaps, 90), pct(gaps, 99)); }
    CHECK(tot / (span * W) > 0.9, "backlogged classes must keep >= 90%% of the workers busy, got %.1f%%", 100 * tot / (span * W));
  }
  // Work conservation: an idle class's share is lent out at once, and taken back at the next task boundary.
  {
    KvmConfig c = sched_cfg(W + 2, 2); c.max_tasks = 1 << 16; c.live_cap = 1 << 13;
    c.sched.p.cls[0].weight = 1; c.sched.p.cls[1].weight = 1; c.sched.p.cls[1].min_cta = 16;
    KvmExec ex(c); ex.start();
    for (int i = 0; i < 20; ++i) CHECK(ex.submit(spin_bundle(400, 200, 0)) == KvmSubmit::OK, "submit");   // class 0 alone for ~50 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (int i = 0; i < 4; ++i) CHECK(ex.submit(spin_bundle(200, 200, 1)) == KvmSubmit::OK, "submit");    // class 1 arrives mid-run
    const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
    auto v = tview(r);
    double t1 = 1e30; for (auto& t : v) if (t.cls == 1) t1 = std::min(t1, t.ready);
    // utilisation before class 1 arrived: class 0 alone must have used all workers (lending)
    double busy = 0, a0 = 2000, b0 = t1 - 2000;  // skip ramp-up
    for (auto& t : v) if (t.cls == 0) { double a = std::max(t.start, a0), b = std::min(t.end, b0); if (b > a) busy += b - a; }
    double util = busy / ((b0 - a0) * W);
    // reclaim: first 16 class-1 starts after it arrived
    std::vector<double> st; for (auto& t : v) if (t.cls == 1) st.push_back(t.start - t1);
    std::sort(st.begin(), st.end());
    printf("  work conservation: class 0 alone kept %.1f%% of the %d workers busy; class 1 (reservation 16) arrived and its 16th CTA started %.0f us later (task length 200 us)\n", 100 * util, W, st.size() > 15 ? st[15] : -1.0);
    CHECK(util > 0.95, "idle share not lent out: class 0 alone used %.1f%%", 100 * util);
    CHECK(st.size() > 15 && st[15] < 200 + 150, "reservation not reclaimed within one task length: 16th start after %.0f us", st.size() > 15 ? st[15] : -1.0);
  }
  return g_fail == before;
}

// ================= reservations: no starvation, measured reclaim latency; negative test =================
struct ResvReport { int violations = 0; double max_wait_us = 0, longest_other_us = 0, grace_us = 0; int n = 0; double avg_running = 0; };
// A violation is a task of class `cls` that waited longer than `grace` while its class had fewer than `mn` running CTAs
// for the whole wait. grace = the longest task of any other class (non-preemptive reclaim bound) + `slack_us` of dispatch.
static ResvReport check_reservation(const std::vector<TV>& v, int cls, int mn, double slack_us) {
  ResvReport rep; for (auto& t : v) if (t.cls != cls) rep.longest_other_us = std::max(rep.longest_other_us, t.end - t.start);
  rep.grace_us = rep.longest_other_us + slack_us;
  struct Ev { double t; int d; }; std::vector<Ev> ev;
  for (auto& t : v) if (t.cls == cls) { ev.push_back({t.start, +1}); ev.push_back({t.end, -1}); }
  std::sort(ev.begin(), ev.end(), [](const Ev& a, const Ev& b) { return a.t < b.t || (a.t == b.t && a.d < b.d); });
  std::vector<double> et; std::vector<int> run; int cur = 0;
  for (auto& e : ev) { cur += e.d; et.push_back(e.t); run.push_back(cur); }
  for (auto& t : v) if (t.cls == cls) {
    ++rep.n; double w = t.start - t.ready; rep.max_wait_us = std::max(rep.max_wait_us, w);
    if (w <= rep.grace_us) continue;
    // max running count over [ready, start]: below mn throughout => the reservation was not honoured
    int mx = 0; size_t i = std::upper_bound(et.begin(), et.end(), t.ready) - et.begin();
    mx = i ? run[i - 1] : 0;
    for (; i < et.size() && et[i] < t.start; ++i) mx = std::max(mx, run[i]);
    if (mx < mn) ++rep.violations;
  }
  return rep;
}
static bool reserve_scenario(bool guarantee, ResvReport* out) {
  const int W = 32, MN = 6;
  KvmConfig c = sched_cfg(W + 2, 2); c.max_tasks = 1 << 16; c.live_cap = 1 << 13;
  c.sched.p.cls[0].weight = 1000; c.sched.p.cls[1].weight = 1; c.sched.p.cls[1].min_cta = MN;
  if (!guarantee) c.sched.p.flags &= ~KVM_SF_GUARANTEE;
  KvmExec exs(c); KvmExec* ex = &exs; ex->start();
  std::vector<KvmBundle> bs;
  for (int rep = 0; rep < 16; ++rep) bs.push_back(spin_bundle(250, 300, 0));
  // class 1 arrives in pulses of MN tasks with gaps (so every pulse needs a reclaim), class 0 saturates all along
  std::thread prod([&] { submit_all(*ex, bs); });
  for (int k = 0; k < 40; ++k) { std::this_thread::sleep_for(std::chrono::microseconds(700)); CHECK(ex->submit(spin_bundle(MN, 200, 1)) == KvmSubmit::OK, "submit"); }
  prod.join();
  const KvmResult& r = ex->finish();
  if (!r.ok()) { CHECK(false, "device error %s", kvm_err_name(r.err_code)); return false; }
  auto v = tview(r);
  *out = check_reservation(v, 1, MN, 150);
  double span = 0, w0 = 1e30, w1 = 0; for (auto& t : v) if (t.cls == 1) { w0 = std::min(w0, t.start); w1 = std::max(w1, t.end); }
  double busy = 0; for (auto& t : v) if (t.cls == 1) busy += t.end - t.start;
  out->avg_running = busy / (w1 - w0); (void)span;
  return true;
}
static bool suite_reserve() {
  int before = g_fail;
  ResvReport r; if (!reserve_scenario(true, &r)) return false;
  printf("  reservation 6 CTAs for a weight-1 class against a saturating weight-1000 class, 40 pulses of 6 tasks: max reclaim wait %.0f us "
         "(longest competing task %.0f us, bound %.0f us), %d violations of %d tasks\n", r.max_wait_us, r.longest_other_us, r.grace_us, r.violations, r.n);
  CHECK(r.violations == 0, "%d tasks waited beyond the reclaim bound while below the reservation", r.violations);
  CHECK(r.max_wait_us <= r.grace_us, "worst reclaim wait %.0f us exceeds longest task + slack %.0f us", r.max_wait_us, r.grace_us);
  return g_fail == before;
}
// Negative: the same scenario with the guarantee disabled. The harness must report starvation, so this suite FAILS (exit 0 under --expect-fail).
static bool suite_starve_neg() {
  int before = g_fail;
  ResvReport r; if (!reserve_scenario(false, &r)) return false;
  printf("  guarantee DISABLED: max wait %.0f us, %d violations\n", r.max_wait_us, r.violations);
  CHECK(r.violations == 0, "STARVATION DETECTED: %d tasks waited beyond %.0f us below the reservation", r.violations, r.grace_us);
  CHECK(r.max_wait_us <= r.grace_us, "STARVATION DETECTED: worst wait %.0f us exceeds the reclaim bound %.0f us", r.max_wait_us, r.grace_us);
  return g_fail == before;
}

// ================= deadline boost (EDF among urgent heads) =================
static bool suite_edf() {
  int before = g_fail;
  for (int urgent_us : {0, 100000}) {
    KvmConfig c = sched_cfg(3, 1); c.sched.p.cls[0].urgent_us = urgent_us;   // one worker
    KvmExec ex(c); ex.start();
    KvmBundleBuilder bb; KvmTaskOpts g; g.priority = 31; bb.add("spin", {I(3000)}, g);   // gate: keeps the single worker busy while the rest queue
    std::vector<int> hi, dl;   // 20 no-deadline tasks at priority 20; 6 deadline tasks at priority 0..3 with deadlines in reverse order
    for (int i = 0; i < 20; ++i) { KvmTaskOpts o; o.priority = 20; hi.push_back(bb.add("spin", {I(50)}, o)); }
    for (int i = 0; i < 6; ++i) { KvmTaskOpts o; o.priority = i; o.deadline_us = 20000 - 1000 * i;   // one task per bucket: EDF is exact across bucket heads (FIFO inside a bucket)
 dl.push_back(bb.add("spin", {I(50)}, o)); }
    CHECK(ex.submit(bb.build()) == KvmSubmit::OK, "submit");
    const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
    int dl_first = 0; double last_hi = 0; for (int s : hi) last_hi = std::max(last_hi, r.trace[s].start_ns / 1e3);
    for (int s : dl) dl_first += r.trace[s].start_ns / 1e3 < last_hi;
    std::vector<int> ord = dl; std::sort(ord.begin(), ord.end(), [&](int a, int b) { return r.trace[a].start_ns < r.trace[b].start_ns; });
    int edf_bad = 0; for (size_t i = 0; i + 1 < ord.size(); ++i) if (r.tasks[ord[i]].deadline_us > r.tasks[ord[i + 1]].deadline_us) ++edf_bad;
    int urgent_flag = 0; for (int s : dl) urgent_flag += (r.trace[s].flags & KVM_TRACE_URGENT) != 0;
    printf("  urgent_us %6d: %d of 6 low-priority deadline tasks overtook the 20 high-priority tasks, EDF order violations among them: %d, started by the deadline tier: %d\n", urgent_us, dl_first, edf_bad, urgent_flag);
    if (urgent_us == 0) CHECK(dl_first == 0 && urgent_flag == 0, "deadline boost off: deadline tasks must not overtake priority");
    else CHECK(dl_first == 6 && edf_bad == 0 && urgent_flag == 6, "deadline boost: all 6 overtake (%d), EDF order (%d bad), flagged %d", dl_first, edf_bad, urgent_flag);
  }
  return g_fail == before;
}

// ================= priority inheritance across version waits =================
// Serve (class 0) has a task waiting for version 1 of variable 0. Train (class 1) publishes it, behind a chain of 3 train tasks,
// and train is fully throttled (max_cta 0) with filler tasks queued ahead. A scheduler that honours the throttle never runs the
// publisher and the waiter parks forever; with inheritance the publisher's class is boosted past the throttle.
static int wait_done(KvmExec& ex, int cls, unsigned long long want, int timeout_ms) {
  auto t0 = std::chrono::steady_clock::now();
  while (ex.telemetry()->cls[cls].done < want) {
    if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) return -1;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}
static void release_throttle(KvmExec& ex, int cls) {
  KvmSchedParams p = ex.config().sched.p; for (int c = 0; c < ex.num_classes(); ++c) p.cls[c].max_cta = KVM_UNLIMITED; (void)cls;
  p.flags = ex.config().sched.p.flags | KVM_SF_INHERIT | KVM_SF_GUARANTEE; ex.set_sched_params(p);
}
static bool inherit_scenario(bool inherit, int shape, int* waited_ms, int* boosted_runs) {
  KvmConfig c = sched_cfg(6, 3);
  c.sched.p.cls[1].max_cta = 0; c.sched.p.cls[2].max_cta = 0;     // classes 1 and 2 are fully throttled
  c.sched.p.cls[0].min_cta = 1;                                    // the waiter's class holds a reservation
  if (!inherit) c.sched.p.flags &= ~KVM_SF_INHERIT;
  KvmExec ex(c); ex.start();
  std::vector<KvmBundle> bs; const int NFILL = 30;
  // shape 0: serve waits v0 <- train.  shape 1: serve waits v0 <- class 1 task that waits v1 <- class 2 (transitive).
  // shape 2: the waiter is admitted and parked BEFORE the publisher's bundle exists.
  KvmBundleBuilder tr; KvmTaskOpts f; f.workload = 1;
  for (int i = 0; i < NFILL; ++i) tr.add("spin", {I(20)}, f);
  KvmTaskOpts pub; pub.workload = 1; pub.pub_var = 0; pub.pub_ver = 1; if (shape == 1) { pub.wait_var = 1; pub.wait_ver = 1; }
  int a = tr.add("spin", {I(20)}, f), b2 = tr.add("spin", {I(20)}, f), p = tr.add("spin", {I(20)}, pub);
  tr.edge(a, b2); tr.edge(b2, p);
  KvmBundleBuilder sv; KvmTaskOpts w; w.workload = 0; w.wait_var = 0; w.wait_ver = 1;
  sv.add("spin", {I(20)}, w); KvmTaskOpts pl; pl.workload = 0; for (int i = 0; i < 5; ++i) sv.add("spin", {I(20)}, pl);
  KvmBundleBuilder d2; KvmTaskOpts f2; f2.workload = 2;
  if (shape == 1) { KvmTaskOpts q; q.workload = 2; q.pub_var = 1; q.pub_ver = 1; for (int i = 0; i < 10; ++i) d2.add("spin", {I(20)}, f2); d2.add("spin", {I(20)}, q); }
  if (shape == 2) { CHECK(ex.submit(sv.build()) == KvmSubmit::OK, "submit"); std::this_thread::sleep_for(std::chrono::milliseconds(5)); CHECK(ex.submit(tr.build()) == KvmSubmit::OK, "submit"); }
  else { CHECK(ex.submit(tr.build()) == KvmSubmit::OK, "submit"); if (shape == 1) CHECK(ex.submit(d2.build()) == KvmSubmit::OK, "submit"); CHECK(ex.submit(sv.build()) == KvmSubmit::OK, "submit"); }
  int ms = wait_done(ex, 0, 6, 400);   // all 6 serve tasks done?
  *waited_ms = ms;
  // let train finish cleanly (also exercises a run-time parameter update)
  release_throttle(ex, 1);
  const KvmResult& r = ex.finish();
  CHECK(r.ok(), "device error %s", kvm_err_name(r.err_code));
  int boosted = 0; for (size_t i = 0; i < r.trace.size(); ++i) boosted += (r.trace[i].flags & KVM_TRACE_BOOST) != 0;
  *boosted_runs = boosted;
  return ms >= 0;
}
static bool suite_inherit() {
  int before = g_fail;
  const char* nm[] = {"serve <- train publisher, train throttled to 0", "serve <- class 1 <- class 2 (transitive), both throttled", "waiter parked before the publisher is admitted"};
  for (int shape = 0; shape < 3; ++shape) {
    int ms, boosted; bool ok = inherit_scenario(true, shape, &ms, &boosted);
    printf("  %s: serve finished in %d ms with inheritance, %d tasks started under a boost\n", nm[shape], ms, boosted);
    CHECK(ok, "deadlock: the waiter never finished although inheritance is on");
    CHECK(boosted > 0, "no task was started under inheritance");
  }
  return g_fail == before;
}
// Negative: inheritance disabled => the throttled publisher never runs => the harness must report the stall (this suite FAILS).
static bool suite_inherit_neg() {
  int before = g_fail;
  int ms, boosted; bool ok = inherit_scenario(false, 0, &ms, &boosted);
  printf("  inheritance DISABLED: serve finished within 400 ms: %s\n", ok ? "yes" : "no (stalled, released by the host afterwards)");
  CHECK(ok, "DEADLOCK DETECTED: the waiter did not finish in 400 ms behind a throttled publisher");
  return g_fail == before;
}

// ================= random multi-class DAGs through every mode =================
static bool suite_rand(bool quick) {
  int before = g_fail; long long total = 0; int dupes = 0, runs = 0;
  for (unsigned seed = 1; seed <= (quick ? 2u : 10u); ++seed)
    for (int mode : {KVM_SCHED_GLOBAL, KVM_SCHED_STATIC, KVM_SCHED_DYNAMIC})
      for (int blocks : {4, 18, 0}) {
        if (mode == KVM_SCHED_STATIC && blocks != 0) continue;   // a static split needs workers on every class's SMs: whole grid only
        if (mode != KVM_SCHED_STATIC && blocks == 0) continue;
        Rng rg(seed * 131 + mode);
        int nb = 2 + (int)(seed % 4);                          // workload ids 0..nb-1 = classes
        KvmConfig c = sched_cfg(blocks, nb, mode); c.live_cap = 1 << 13; c.ring_words = 1 << 17;
        for (int k = 0; k < nb; ++k) {
          KvmClassParams& q = c.sched.p.cls[k];
          q.weight = rg.in(1, 8); q.min_cta = rg.p(50) ? rg.in(0, 2) : 0; q.max_cta = rg.p(30) ? rg.in(std::max(1, q.min_cta), 3) : KVM_UNLIMITED;
          q.bw_cta = rg.p(30) ? rg.in(1, 2) : KVM_UNLIMITED; q.urgent_us = rg.p(40) ? 5000 : 0;
        }
        if (mode == KVM_SCHED_STATIC) kvm_static_split(c.sched.p, std::vector<double>(nb, 1.0), g_sms);
        int d; dag_run(seed * 6151 + mode, nb, 500 + 200 * (seed % 3), 6 + seed % 20, c, 31, &total, &d); dupes += d; ++runs;
      }
  printf("  %d runs (GLOBAL / STATIC / DYNAMIC, 2 grid sizes, random weights, reservations, caps, BW caps, deadline windows), %lld tasks: exactly-once, edges ordered by timestamps, node values bitwise vs CPU\n", runs, total);
  return g_fail == before;
}
static bool suite_smoke_sched() {
  int before = g_fail; long long t = 0; int d;
  for (int mode : {KVM_SCHED_GLOBAL, KVM_SCHED_STATIC, KVM_SCHED_DYNAMIC}) {
    KvmConfig c = sched_cfg(4, 3, mode); c.max_tasks = 4096; c.live_cap = 256; c.ring_words = 4096;
    c.sched.p.cls[1].min_cta = 1; c.sched.p.cls[2].max_cta = 1;
    if (mode == KVM_SCHED_STATIC) kvm_static_split(c.sched.p, {1, 1, 1}, g_sms);
    dag_run(3 + mode, 3, 60, 6, c, 31, &t, &d);
  }
  { int a, b; inherit_scenario(true, 0, &a, &b); }
  return g_fail == before;
}

#ifndef KVM_SCHED_NO_BENCH
#include "sched_bench.inc"
#endif

int main(int argc, char** argv) {
  bool expect_fail = false; std::vector<std::string> want;
  for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--expect-fail") expect_fail = true; else if (a == "--quiet") g_kvm_log = 0; else if (a == "--log") g_kvm_log = atoi(argv[++i]); else want.push_back(a); }
  bool quick = getenv("KVM_TEST_QUICK") != nullptr;
  if (want.empty()) want = {"cost", "share", "reserve", "edf", "inherit", "rand"};
  int dev; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p, dev));
  g_sms = p.multiProcessorCount;
  int drv, rt; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("GPU %s, %d SMs, sm_%d%d, driver CUDA %d.%d, runtime %d.%d%s\n", p.name, p.multiProcessorCount, p.major, p.minor, drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10, quick ? " [quick]" : "");
  int failed_suites = 0;
  for (auto& s : want) {
    int f0 = g_fail; auto t0 = std::chrono::steady_clock::now(); bool ok = true;
    printf("== %s\n", s.c_str()); fflush(stdout);
    try {
      if (s == "cost") ok = suite_cost(); else if (s == "share") ok = suite_share(); else if (s == "reserve") ok = suite_reserve();
      else if (s == "edf") ok = suite_edf(); else if (s == "inherit") ok = suite_inherit(); else if (s == "rand") ok = suite_rand(quick);
      else if (s == "starve_neg") ok = suite_starve_neg(); else if (s == "inherit_neg") ok = suite_inherit_neg(); else if (s == "smoke") ok = suite_smoke_sched();
#ifndef KVM_SCHED_NO_BENCH
      else if (sched_bench_suite(s, ok)) {}
#endif
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
