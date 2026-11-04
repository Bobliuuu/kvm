// Closed-loop scheduling, stress and limits (include/kvm_adapt.h on top of include/kvm_sched.*). Synthetic tasks only (spin, add, copy): the
// scheduler sees workload_id, priority, deadline_us and the op cost table, so what is measured here is the policy, not a model.
//   nvcc ... -DKVM_EXEC_NO_MAIN -DKVM_SCHED_NO_BENCH tests/gpu/test_adaptive.cu -o test_adaptive      (tests/gpu/build_adaptive.sh)
//   ./test_adaptive [suite...]        functional: telemetry loop floor
//                                     measurements (explicit): granularity hol
//   ./test_adaptive --expect-fail floor_neg   # negative control: exit 0 only if the harness DETECTS the injected failure
#define KVM_SCHED_NO_BENCH
#define KVM_SCHED_NO_MAIN
#include "test_sched.cu"
#include <deque>
#include "../../include/kvm_adapt.h"

static std::string env_or(const char* k, const char* d) { const char* v = getenv(k); return v ? v : d; }
static double now_us_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); }

// A saturating producer for one class: keeps up to `budget` tasks of bundles (n tasks of `us` each) in flight through a KvmAdmissionGate, so the
// shared admission ring is never filled by one class (an ungated flood starves every other class at admission, see suite_hol).
static std::thread flood_producer(KvmExec& ex, KvmAdmissionGate& gate, int cls, int n, int us, std::atomic<bool>& stop) {
  return std::thread([&ex, &gate, cls, n, us, &stop] {
    std::deque<unsigned> win;
    while (!stop.load()) {
      while (!win.empty() && ex.window_done(win.front())) { gate.release(cls, n); win.pop_front(); }
      if (!gate.acquire(cls, n, 5)) continue;
      unsigned w; if (ex.submit(spin_bundle(n, us, cls), 30000, false, &w) != KvmSubmit::OK) break;
      win.push_back(w);
    }
    for (unsigned w : win) while (!ex.window_done(w)) std::this_thread::sleep_for(std::chrono::microseconds(50));
  });
}

// ================= telemetry: the new device counters are exact =================
// dl_n / near_miss / missed / wait_over must equal the same quantities recomputed from the per-task trace.
static bool suite_telemetry() {
  int before = g_fail;
  KvmConfig c = sched_cfg(6, 2); c.max_tasks = 1 << 14; c.live_cap = 1 << 12;
  c.sched.p.cls[1].target_wait_us = 300;
  KvmExec ex(c); ex.start();
  // class 0: tasks of 400 us with a 1000 us budget: (queue + run) mostly lands in the last quarter -> near misses and some misses
  // class 1: no deadline, target wait 300 us: the backlog makes most of them wait longer
  CHECK(ex.submit(spin_bundle(200, 400, 0, 0, 1000)) == KvmSubmit::OK, "submit");
  CHECK(ex.submit(spin_bundle(400, 100, 1)) == KvmSubmit::OK, "submit");
  CHECK(ex.submit(spin_bundle(100, 400, 0, 0, 5000)) == KvmSubmit::OK, "submit");
  const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
  unsigned long long dl = 0, near = 0, miss = 0, over[2] = {0, 0};
  for (size_t i = 0; i < r.trace.size(); ++i) {
    const KvmTrace& t = r.trace[i]; const KvmTaskDesc& d = r.tasks[i];
    if (d.deadline_us >= 0) { const long long budget = (long long)d.deadline_us * 1000, left = budget - (long long)(t.end_ns - t.admit_ns); ++dl; near += left < budget / 4; miss += (t.flags & KVM_TRACE_MISSED) != 0; }
    if (c.sched.p.cls[d.workload_id].target_wait_us > 0 && t.start_ns - t.ready_ns > (unsigned long long)c.sched.p.cls[d.workload_id].target_wait_us * 1000ull) ++over[d.workload_id];
  }
  const KvmSchedTele* T = ex.telemetry();
  // the exporter runs every ~20 us while the policy block lives; finish() ends it, so compare against the device accounting copied at the end
  const KvmClsDev& k0 = r.sched.dev.cls[0]; const KvmClsDev& k1 = r.sched.dev.cls[1];
  printf("  trace: %llu deadline tasks, %llu near miss, %llu missed, %llu waited over target; device: dl_n %llu near %llu missed %llu wait_over(cls1) %llu; mapped telemetry at exit: near %llu\n",
         dl, near, miss, over[1], (unsigned long long)k0.dl_n, (unsigned long long)k0.near_miss, (unsigned long long)k0.missed, (unsigned long long)k1.wait_over, (unsigned long long)T->cls[0].near_miss);
  CHECK(k0.dl_n == dl && k0.near_miss == near && k0.missed == miss, "device counters differ from the trace");
  CHECK(k1.wait_over == over[1] && over[1] > 50, "wait_over %llu vs trace %llu (the scenario must produce waits over target)", (unsigned long long)k1.wait_over, over[1]);
  CHECK(near > 0 && near >= miss, "scenario must produce near misses (%llu) including every miss (%llu)", near, miss);
  CHECK(T->cls[0].near_miss <= near, "exported counter must never run ahead of the device");
  return g_fail == before;
}

// ================= closed loop against the real device: burst attack, release, floor =================
struct LoopOut { double attack_ms = -1, release_ms = -1; int peak_min = 0, final_min = 0, train_floor_min = 1 << 30; double train_share_burst = 0; int changes = 0; };
static LoopOut run_loop(bool floor_on, bool controller) {
  const int W = 62; LoopOut out;
  KvmConfig c = sched_cfg(W + 2, 2); c.max_tasks = 1 << 18; c.live_cap = 1 << 14; c.ring_words = 1 << 17;
  c.sched.p.cls[0].weight = 8; c.sched.p.cls[1].weight = 1; c.sched.p.cls[0].target_wait_us = 500;
  KvmExec ex(c); ex.start();
  KvmAdaptConfig ac; ac.workers = W; ac.period_us = 500; ac.release_periods = 40;
  KvmAdaptClass s; s.cls = 0; s.role = KvmRole::LATENCY; s.min_hi = W; s.w_lo = 8; s.w_hi = 16; s.target_wait_us = 500;
  KvmAdaptClass t; t.cls = 1; t.role = KvmRole::BATCH; t.floor_cta = floor_on ? W / 5 : 0;
  ac.classes = {s, t};
  KvmAdaptive ctl(ex.sched_params(), ex.telemetry(), c.sched.p, ac);
  if (controller) ctl.start();
  std::atomic<bool> stop{false};
  KvmAdmissionGate gate({-1, 6000});
  std::thread train = flood_producer(ex, gate, 1, 400, 300, stop);   // saturating batch class
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  const auto t0 = std::chrono::steady_clock::now();
  const unsigned long long sm1_0 = ex.telemetry()->cls[1].sm_ns, sm0_0 = ex.telemetry()->cls[0].sm_ns;
  // burst: 60 ms of serve work at ~2.6 x W workers worth of demand, as bundles of 40 x 200 us every 150 us
  std::thread serve([&] {
    const auto b0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - b0 < std::chrono::milliseconds(60)) { if (ex.submit(spin_bundle(40, 200, 0), 30000) != KvmSubmit::OK) break; std::this_thread::sleep_for(std::chrono::microseconds(150)); }
  });
  while (now_us_since(t0) < 60e3) {
    const KvmClassParams& q = ex.sched_params()->cls[0]; out.peak_min = std::max(out.peak_min, q.min_cta);
    if (out.attack_ms < 0 && q.min_cta >= 10) out.attack_ms = now_us_since(t0) / 1e3;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  const unsigned long long sm1_1 = ex.telemetry()->cls[1].sm_ns, sm0_1 = ex.telemetry()->cls[0].sm_ns;
  out.train_share_burst = (double)(sm1_1 - sm1_0) / std::max(1.0, (double)((sm1_1 - sm1_0) + (sm0_1 - sm0_0)));
  serve.join();
  const auto t1 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() < 400) {   // calm: the reservation must come back down
    const int m = ex.sched_params()->cls[0].min_cta;
    if (out.release_ms < 0 && m == 0) out.release_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  out.final_min = ex.sched_params()->cls[0].min_cta; out.changes = ctl.changes();
  stop = true; train.join(); ctl.stop();
  const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error %s", kvm_err_name(r.err_code));
  return out;
}
static bool suite_loop() {
  int before = g_fail;
  LoopOut a = run_loop(true, true);
  printf("  burst of serve demand beside saturating batch (62 workers): reservation reached %d CTAs (10 CTAs reached after %.1f ms), released to %d after %.0f ms; batch got %.0f%% of the SM time during the burst; %d parameter writes\n",
         a.peak_min, a.attack_ms, a.final_min, a.release_ms, 100 * a.train_share_burst, a.changes);
  CHECK(a.peak_min >= 10 && a.attack_ms >= 0 && a.attack_ms < 10, "reservation must follow a burst within 10 ms, took %.1f", a.attack_ms);
  CHECK(a.release_ms >= 0 && a.final_min == 0, "reservation must be released after the burst (final %d)", a.final_min);
  CHECK(a.train_share_burst >= 0.16, "the batch floor (20%% of the workers) must hold during the burst: batch share %.3f", a.train_share_burst);
  CHECK(a.changes < 200, "controller churn: %d parameter writes", a.changes);
  return g_fail == before;
}

// ================= starvation floor; negative control without the floor =================
static double floor_scenario(bool floor_on) {
  const int W = 30;
  KvmConfig c = sched_cfg(W + 2, 2); c.max_tasks = 1 << 17; c.live_cap = 1 << 13; c.ring_words = 1 << 17;
  c.sched.p.cls[0].weight = 100; c.sched.p.cls[1].weight = 1;
  KvmExec ex(c); ex.start();
  KvmAdaptConfig ac; ac.workers = W; ac.period_us = 500;
  KvmAdaptClass s; s.cls = 0; s.role = KvmRole::LATENCY; s.min_hi = W; s.w_lo = 100; s.w_hi = 100;
  KvmAdaptClass t; t.cls = 1; t.role = KvmRole::BATCH; t.floor_cta = floor_on ? W / 5 : 0;
  ac.classes = {s, t};
  KvmAdaptive ctl(ex.sched_params(), ex.telemetry(), c.sched.p, ac); ctl.start();
  std::atomic<bool> stop{false};
  KvmAdmissionGate gate({6000, 1500});
  std::thread hi = flood_producer(ex, gate, 0, 300, 200, stop);   // sustained overload by the favoured class (4x the batch class's backlog)
  std::thread lo = flood_producer(ex, gate, 1, 100, 200, stop);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const unsigned long long a1 = ex.telemetry()->cls[1].sm_ns, a0 = ex.telemetry()->cls[0].sm_ns;
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const unsigned long long b1 = ex.telemetry()->cls[1].sm_ns, b0 = ex.telemetry()->cls[0].sm_ns;
  stop = true; hi.join(); lo.join(); ctl.stop();
  const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
  return (double)(b1 - a1) / std::max(1.0, (double)((b1 - a1) + (b0 - a0)));
}
static bool suite_floor() {
  int before = g_fail;
  const double with = floor_scenario(true);
  printf("  sustained overload by a weight-100 class against a weight-1 batch class: batch SM-time share %.3f with the controller's 20%% floor\n", with);
  CHECK(with >= 0.17, "the floor must hold: batch share %.3f (want >= 0.17 of 0.20)", with);
  return g_fail == before;
}
static bool suite_floor_neg() {
  int before = g_fail;
  const double without = floor_scenario(false);
  printf("  controller floor DISABLED: batch SM-time share %.3f\n", without);
  CHECK(without >= 0.17, "STARVATION DETECTED: batch share %.3f is below the 17%% floor", without);
  return g_fail == before;
}

// ================= measurements =================
// Preemption granularity. Tasks are not preemptible: a ready latency task starts when ANY CTA finishes its current task. With W workers
// running competing tasks of length L at random phases, the wait is the minimum of W residuals, P(wait > t) = (1 - t/L)^W, so
// p99 = L (1 - 0.01^(1/W)) ~ 4.6 L / W; the worst case (all CTAs started together) is L. Measured against that.
static bool suite_granularity() {
  int before = g_fail;
  printf("  latency task (200 us, 2 ms period) beside saturating competitors of length L; reclaim = queue wait of the latency task\n");
  printf("  %-9s %-4s | %-30s | %-12s | %s\n", "mode", "W", "L(us): p50 / p99 / max wait (us)", "model p99", "latency tasks past a 5 ms budget");
  for (int W : {16, 62}) for (int mode : {KVM_SCHED_GLOBAL, KVM_SCHED_DYNAMIC}) {
    std::string row;
    for (int L : {100, 1000, 5000, 20000}) {
      KvmConfig c = sched_cfg(W + 2, 2, mode); c.max_tasks = 1 << 17; c.live_cap = 1 << 13; c.ring_words = 1 << 17;
      c.sched.p.cls[0].min_cta = mode == KVM_SCHED_DYNAMIC ? 1 : 0; c.sched.p.cls[0].weight = 4;
      KvmExec ex(c); ex.start();
      Rng rg(L);
      const int per = std::max(4, 600000 / L);   // ~600 ms of competitor work per worker
      for (int b = 0; b < per * W / 64 + 1; ++b) {
        KvmBundleBuilder bb; KvmTaskOpts o; o.workload = mode == KVM_SCHED_GLOBAL ? 0 : 1;
        for (int i = 0; i < 64; ++i) bb.add("spin", {I(std::max(10, L / 2 + rg.in(0, L)))}, o);   // lengths uniform in [L/2, 3L/2]: random phases
        CHECK(ex.submit(bb.build(), 60000) == KvmSubmit::OK, "submit");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(std::min(100, 2 * L / 1000 + 20)));
      const int nt = 150; std::vector<int> ids;
      for (int k = 0; k < nt; ++k) {
        KvmBundleBuilder bb; KvmTaskOpts o; o.workload = 0; o.deadline_us = 5000; bb.add("spin", {I(200)}, o);
        CHECK(ex.submit(bb.build(), 60000) == KvmSubmit::OK, "submit");
        std::this_thread::sleep_for(std::chrono::microseconds(2000));
      }
      const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
      std::vector<double> w; int late = 0;
      for (size_t i = 0; i < r.trace.size(); ++i) if (r.tasks[i].op_id == KVM_OP_spin && r.tasks[i].deadline_us == 5000) { w.push_back((r.trace[i].start_ns - r.trace[i].ready_ns) / 1e3); late += (r.trace[i].flags & KVM_TRACE_MISSED) != 0; }
      char b[160]; snprintf(b, sizeof b, "%s%d: %.0f/%.0f/%.0f (model %.0f)%s", row.empty() ? "" : "; ", L, pct(w, 50), pct(w, 99), pct(w, 100), L * (1 - std::pow(0.01, 1.0 / W)), late ? "" : ""); row += b;
      if (L == 20000) { printf("  %-9s %-4d | %s | past budget at L=20000: %d of %zu\n", mode == KVM_SCHED_GLOBAL ? "global" : "dynamic", W, row.c_str(), late, w.size()); }
    }
  }
  return g_fail == before;
}

// Admission head-of-line blocking. The admitter is one FIFO and the live window is shared: while a flood of one class fills it, a bundle of
// another class behind it waits for room before the scheduler can rank it at all. Gate = KvmAdmissionGate with the flood limited to half the window.
static bool suite_hol() {
  int before = g_fail;
  printf("  flood: bundles of 256 x 200 us tasks submitted as fast as they are accepted (class 1); latency bundles (1 x 200 us, class 0, reservation 1) every 2 ms; live window 4096 tasks\n");
  for (int gated : {0, 1}) {
    const int W = 30;
    KvmConfig c = sched_cfg(W + 2, 2); c.max_tasks = 1 << 17; c.live_cap = 1 << 12; c.ring_words = 1 << 18;
    c.sched.p.cls[0].min_cta = 1; c.sched.p.cls[0].weight = 4;
    KvmExec ex(c); ex.start();
    KvmAdmissionGate gate({-1, gated ? (long long)c.live_cap / 2 : -1});
    std::atomic<bool> stop{false};
    std::thread fl([&] {
      std::deque<std::pair<unsigned, int>> fl_win;
      while (!stop.load()) {
        while (!fl_win.empty() && ex.window_done(fl_win.front().first)) { gate.release(1, fl_win.front().second); fl_win.pop_front(); }
        KvmBundle b = spin_bundle(256, 200, 1);
        if (!gate.acquire(1, 256, 20)) continue;
        unsigned w; if (ex.submit(b, 30000, false, &w) != KvmSubmit::OK) break;
        fl_win.push_back({w, 256});
      }
      for (auto& f : fl_win) { while (!ex.window_done(f.first)) std::this_thread::sleep_for(std::chrono::microseconds(50)); }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::vector<double> lat; const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 150; ++k) {
      const double due = k * 2000.0; while (now_us_since(t0) < due) std::this_thread::yield();
      unsigned w; const double s = now_us_since(t0);
      CHECK(ex.submit(spin_bundle(1, 200, 0), 30000, false, &w) == KvmSubmit::OK, "submit");
      while (!ex.window_done(w)) std::this_thread::yield();
      lat.push_back(now_us_since(t0) - s);
    }
    stop = true; fl.join();
    const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
    printf("  %-8s: latency bundle end-to-end p50 %.0f us, p99 %.0f us, max %.0f us (task alone: ~210 us)\n", gated ? "gated" : "ungated", pct(lat, 50), pct(lat, 99), pct(lat, 100));
    if (gated) CHECK(pct(lat, 99) < 2000, "with the flood gated the latency class must not wait for window room: p99 %.0f us", pct(lat, 99));
  }
  return g_fail == before;
}

int main(int argc, char** argv) {
  bool expect_fail = false; std::vector<std::string> want;
  for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--expect-fail") expect_fail = true; else if (a == "--quiet") g_kvm_log = 0; else want.push_back(a); }
  if (want.empty()) want = {"telemetry", "loop", "floor"};
  int dev; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p, dev));
  g_sms = p.multiProcessorCount;
  printf("GPU %s, %d SMs, sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
  int failed_suites = 0;
  for (auto& s : want) {
    int f0 = g_fail; auto t0 = std::chrono::steady_clock::now(); bool ok = true;
    printf("== %s\n", s.c_str()); fflush(stdout);
    try {
      if (s == "telemetry") ok = suite_telemetry(); else if (s == "loop") ok = suite_loop(); else if (s == "floor") ok = suite_floor();
      else if (s == "floor_neg") ok = suite_floor_neg(); else if (s == "granularity") ok = suite_granularity(); else if (s == "hol") ok = suite_hol();
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
