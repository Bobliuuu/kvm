// Measurement suites (included by test_sched.cu): dispatch, arbiter, rebalance, baselines. They print the numbers quoted in results/.
struct GapStats { double p50, p90, p99, occ; };
static GapStats gap_stats(const KvmResult& r, int workers) {
  auto v = tview(r); std::vector<std::vector<std::pair<double, double>>> byb(r.grid);
  double w0 = 1e30, w1 = 0, busy = 0;
  for (size_t i = 0; i < r.trace.size(); ++i) { byb[r.trace[i].block].push_back({v[i].start, v[i].end}); w0 = std::min(w0, v[i].start); w1 = std::max(w1, v[i].end); busy += v[i].end - v[i].start; }
  std::vector<double> gaps; for (auto& b : byb) { std::sort(b.begin(), b.end()); for (size_t i = 1; i < b.size(); ++i) gaps.push_back(b[i].first - b[i - 1].second); }
  return {pct(gaps, 50), pct(gaps, 90), pct(gaps, 99), busy / ((w1 - w0) * workers)};
}

static GapStats gap_stats_cls(const KvmResult& r, int workers, int cls) {   // gaps between consecutive tasks of one class on one CTA
  auto v = tview(r); std::vector<std::vector<std::pair<double, double>>> byb(r.grid);
  for (size_t i = 0; i < r.trace.size(); ++i) if (v[i].cls == cls) byb[r.trace[i].block].push_back({v[i].start, v[i].end});
  std::vector<double> gaps; for (auto& b : byb) { std::sort(b.begin(), b.end()); for (size_t i = 1; i < b.size(); ++i) gaps.push_back(b[i].first - b[i - 1].second); }
  (void)workers; return {pct(gaps, 50), pct(gaps, 90), pct(gaps, 99), 0};
}
// Dispatch cost: the price of per-class queues and accounting, with synchronized equal tasks (the worst case for head contention).
static bool suite_dispatch() {
  int before = g_fail; const int W = 32;
  struct Cfg { const char* name; bool on; int mode, nc; };
  for (int us : {200, 20}) for (const Cfg& cf : std::vector<Cfg>{{"sched off (original)", false, 0, 1}, {"GLOBAL", true, KVM_SCHED_GLOBAL, 4}, {"DYNAMIC 1 class", true, KVM_SCHED_DYNAMIC, 1}, {"DYNAMIC 4 classes", true, KVM_SCHED_DYNAMIC, 4}}) {
    std::vector<double> occ, p50;
    for (int rep = 0; rep < 3; ++rep) {
      KvmConfig c = sched_cfg(W + (cf.on ? 2 : 1), cf.nc, cf.mode); c.sched.on = cf.on; c.max_tasks = 1 << 17; c.live_cap = 1 << 14; c.ring_words = 1 << 17;
      KvmExec ex(c); ex.start();
      const int nt = us == 200 ? 400 : 3000;
      for (int b = 0; b < 12; ++b) CHECK(ex.submit(spin_bundle(nt, us, b % cf.nc), 30000) == KvmSubmit::OK, "submit");
      const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
      GapStats g = gap_stats(r, W); occ.push_back(g.occ); p50.push_back(g.p50);
    }
    Spread so = spread(occ), sg = spread(p50);
    printf("  %3d us tasks, %-22s: occupancy %.1f%% [%.1f, %.1f], gap p50 %.2f us [%.2f, %.2f]\n", us, cf.name, 100 * so.med, 100 * so.lo, 100 * so.hi, sg.med, sg.lo, sg.hi);
  }
  return g_fail == before;
}

#ifdef KVM_PROF_GAP
// Where the dispatch gap goes (build with -DKVM_PROF_GAP): mean/p50 SM-clock time of each phase between the end of one op and the start
// of the next on the same block (see KVM_PG in kvm_exec.cuh). 32 workers, equal spin tasks, no successors.
static bool suite_gapprof() {
  int before = g_fail; const int W = 32;
  static const char* ph[] = {"end bookkeeping+publish", "fence+succ release", "sched_done", "window credit", "free slot", "done count", "POP", "desc copy", "start bookkeeping"};
  struct Cfg { const char* name; bool on; int mode, nc; };
  for (int us : {200, 20}) for (const Cfg& cf : std::vector<Cfg>{{"sched off", false, 0, 1}, {"DYNAMIC 4 classes", true, KVM_SCHED_DYNAMIC, 4}}) {
    KvmConfig c = sched_cfg(W + (cf.on ? 2 : 1), cf.nc, cf.mode); c.sched.on = cf.on; c.max_tasks = 1 << 17; c.live_cap = 1 << 14; c.ring_words = 1 << 17;
    KvmExec ex(c); ex.start();
    const int nt = us == 200 ? 400 : 3000;
    for (int b = 0; b < 12; ++b) CHECK(ex.submit(spin_bundle(nt, us, b % cf.nc), 30000) == KvmSubmit::OK, "submit");
    const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
    std::vector<std::vector<double>> v(9); double gaps = 0; int n = 0;
    for (auto& t : r.trace) if (t.gp[15] == 1 && t.gp[6] < 200000) { for (int k = 0; k < 9; ++k) v[k].push_back(t.gp[k] / 1.41); ++n; }
    printf("  %3d us tasks, %-18s: %d samples (ns, SM clock 1.41 GHz)  ", us, cf.name, n);
    for (int k = 0; k < 9; ++k) { double m = 0; for (double x : v[k]) m += x; m /= std::max<size_t>(1, v[k].size()); gaps += m; printf("| %s %.0f/%.0f ", ph[k], m, pct(v[k], 50)); }
    printf("| TOTAL mean %.0f ns\n", gaps);
  }
  return g_fail == before;
}
#endif

// Admission: how long the admitter takes per record (the next record's admission starts when this one is done), and how long a
// 1-task record of another class waits behind a big one (head-of-line blocking).
static KvmBundle noop_bundle(int n, int cls, bool chain = false) {
  KvmBundleBuilder b; KvmTaskOpts o; o.workload = cls; int prev = -1;
  for (int i = 0; i < n; ++i) { int t = b.add("noop", {}, o); if (chain && prev >= 0) b.edge(prev, t); prev = t; }
  return b.build();
}
static bool suite_admit() {
  int before = g_fail; const int W = 64;
  auto cfg = [&](bool on) {
    KvmConfig c = sched_cfg(W + 2, 2, KVM_SCHED_DYNAMIC); c.sched.on = on; c.max_tasks = 1 << 18; c.live_cap = 1 << 17; c.succ_cap = 1 << 18; c.ring_words = 1 << 21; c.win_cap = 1 << 10;
    return c;
  };
  printf("  admission time per record = admit instant of the next record minus this one's (2 records back to back, 4 repeats, median [min,max])\n");
  for (bool chain : {false, true}) for (int n : {1024, 16384, 65536}) {
    std::vector<double> dur;
    for (int rep = 0; rep < 4; ++rep) {
      KvmExec ex(cfg(true)); ex.start();
      CHECK(ex.submit(noop_bundle(n, 0, chain), 30000) == KvmSubmit::OK, "submit"); CHECK(ex.submit(noop_bundle(n, 0, chain), 30000) == KvmSubmit::OK, "submit");
      const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
      unsigned long long a0 = ~0ull, a1 = 0; for (auto& t : r.trace) if (t.admit_ns) { a0 = std::min(a0, t.admit_ns); a1 = std::max(a1, t.admit_ns); }
      dur.push_back((a1 - a0) / 1e3);
    }
    Spread sd = spread(dur);
    printf("  %-11s record of %6d tasks: admitted in %8.0f us [%.0f,%.0f] = %.3f us/task (%.2f M tasks/s)\n", chain ? "chain" : "independent", n, sd.med, sd.lo, sd.hi, sd.med / n, n / sd.med);
  }
  printf("  head-of-line: a 1-task record of class 0 submitted right behind a record of N independent tasks of class 1; wait = small.ready - big.admit\n");
  for (int n : {1024, 16384, 65536}) {
    std::vector<double> wt;
    for (int rep = 0; rep < 4; ++rep) {
      KvmExec ex(cfg(true)); ex.start();
      CHECK(ex.submit(noop_bundle(n, 1), 30000) == KvmSubmit::OK, "submit"); CHECK(ex.submit(noop_bundle(1, 0), 30000) == KvmSubmit::OK, "submit");
      const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
      unsigned long long big0 = ~0ull, small_ready = 0; for (size_t i = 0; i < r.trace.size(); ++i) { if (r.tasks[i].workload_id == 1) big0 = std::min(big0, r.trace[i].admit_ns); else small_ready = r.trace[i].ready_ns; }
      wt.push_back((small_ready - big0) / 1e3);
    }
    Spread sd = spread(wt);
    printf("  big record %6d tasks: 1-task record ready %8.0f us [%.0f,%.0f] after the big one's admission began\n", n, sd.med, sd.lo, sd.hi);
  }
  return g_fail == before;
}

// Latency probes (1 noop task of class 0 with a deadline, every ~1 ms) while a 65,536-task flood of 20 us tasks of class 1 is being admitted.
// direct = ex.submit of the flood as one record (the ring is FIFO); front = the KvmFront (chunks of 2048 tasks, ring budget 4096 tasks).
static bool suite_hol() {
  int before = g_fail; const int W = 64, NF = 65536, NP = 40;
  printf("  flood: %d x spin(20 us) of class 1 (%d workers); probes: 1 noop of class 0, 200 us deadline, end-to-end host latency submit -> window done\n", NF, W);
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<double> lat, wall; 
    for (int rep = 0; rep < 3; ++rep) {
      KvmConfig c = sched_cfg(W + 2, 2, KVM_SCHED_DYNAMIC); c.max_tasks = 1 << 18; c.live_cap = 1 << 17; c.succ_cap = 1 << 18; c.ring_words = 1 << 21; c.sched.p.cls[0].urgent_us = KVM_URGENT_ALL;
      KvmExec ex(c); ex.start();
      std::unique_ptr<KvmFront> fr; if (mode == 1) { KvmFront::Cfg fc; fc.classes = 2; fr.reset(new KvmFront(ex, fc)); }
      KvmBundle flood = spin_bundle(NF, 20, 1), probe; { KvmBundleBuilder b; KvmTaskOpts o; o.workload = 0; o.deadline_us = 200; b.add("noop", {}, o); probe = b.build(); }
      auto t0 = std::chrono::steady_clock::now();
      if (mode == 0) CHECK(ex.submit(flood, 30000) == KvmSubmit::OK, "flood submit"); else fr->submit(flood, 1);
      std::this_thread::sleep_for(std::chrono::microseconds(300));
      for (int k = 0; k < NP; ++k) {
        auto ts = std::chrono::steady_clock::now(); bool ok;
        if (mode == 0) { unsigned wid = 0; ok = ex.submit(probe, 30000, true, &wid) == KvmSubmit::OK; while (ok && !ex.window_done(wid) && std::chrono::steady_clock::now() - ts < std::chrono::seconds(10)) {} }
        else { KvmFront::Ticket t = fr->submit(probe, 0); ok = fr->wait(t, 10000); }
        double l = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ts).count();
        CHECK(ok, "probe %d did not complete", k); lat.push_back(l);
        std::this_thread::sleep_for(std::chrono::microseconds(700));
      }
      if (fr) { CHECK(fr->flush(30000), "front flush"); }
      ex.close(); const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
      wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      int bad = 0; for (auto& t : r.trace) bad += t.exec_count != 1; CHECK(bad == 0 && r.done == NF + NP, "done %d (want %d), %d tasks with exec_count != 1", r.done, NF + NP, bad);
    }
    Spread sw = spread(wall);
    printf("  %-6s probe latency p50 %8.0f p99 %8.0f max %8.0f us (%zu probes) | flood+probes wall %.0f ms [%.0f,%.0f]\n", mode == 0 ? "direct" : "front", pct(lat, 50), pct(lat, 99), *std::max_element(lat.begin(), lat.end()), lat.size(), sw.med, sw.lo, sw.hi);
  }
  return g_fail == before;
}

// Per-op running time of the generator's task types on an otherwise idle GPU (one wave of 30 CTAs), to size the loads.
static bool suite_calib() {
  KvmConfig c = sched_cfg(32, 1); c.sched.on = false; c.max_tasks = 1 << 14; c.live_cap = 1 << 12; c.ring_words = 1 << 17;
  KvmExec ex(c);
  const int SL = 64, N = 65536;
  float* big = ex.arena((size_t)SL * 3 * N); std::vector<int> v;
  for (int i = 0; i < SL * 3; ++i) v.push_back(ex.add_view(big + (size_t)i * N, 1, N));
  int m64a = ex.add_view(ex.arena(64 * 64), 64, 64), m64b = ex.add_view(ex.arena(64 * 64), 64, 64), m64c = ex.add_view(ex.arena(64 * 64), 64, 64);
  int m128a = ex.add_view(ex.arena(128 * 128), 128, 128), m128b = ex.add_view(ex.arena(128 * 128), 128, 128), m128c = ex.add_view(ex.arena(128 * 128), 128, 128);
  ex.start();
  KvmBundleBuilder bb; std::vector<std::vector<int>> ids(4);
  for (int i = 0; i < 30; ++i) {
    ids[0].push_back(bb.add("add", {Bf(v[3 * i]), Bf(v[3 * i + 1]), Bf(v[3 * i + 2]), I(N)}));
    ids[1].push_back(bb.add("matmul", {Bf(m64a), Bf(m64b), Bf(m64c), I(64), I(64), I(64)}));
    ids[2].push_back(bb.add("matmul", {Bf(m128a), Bf(m128b), Bf(m128c), I(128), I(128), I(128)}));
    ids[3].push_back(bb.add("copy", {Bf(v[3 * i]), Bf(v[3 * i + 1]), I(4096)}));
  }
  CHECK(ex.submit(bb.build()) == KvmSubmit::OK, "submit");
  const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
  const char* nm[] = {"add 65536 floats (786 KB moved)", "matmul 64^3", "matmul 128^3", "copy 4096 floats (32 KB moved)"};
  for (int k = 0; k < 4; ++k) { std::vector<double> t; for (int id : ids[k]) t.push_back((r.trace[id].end_ns - r.trace[id].start_ns) / 1e3); printf("  %-34s median %.1f us (30 concurrent)\n", nm[k], median(t)); }
  return g_fail == 0;
}

// ================= load generator: a latency-sensitive request stream next to a saturating batch stream =================
// Both are just task streams with a workload id. "serve" (class 0): open-loop arrivals, each a bundle of `fan` independent
// chains of `hops` dependent tasks. "train" (class 1): bundles of independent tasks, resubmitted so that a fixed number is
// always outstanding (a saturating backlog).
enum { SK_HOT = 0, SK_COLD = 1, SK_MATMUL = 2 };
struct Phase { double ms, rate; };   // serve arrivals per second during this phase
struct LoadSpec {
  int hops = 16, fan = 1, serve_kind = SK_HOT; double slo_us = 1000;
  std::vector<Phase> phases;
  int serve_prio = 0, train_prio = 0, train_outstanding = 3000, train_bundle = 256; bool train_bw_only = false; double train_matmul_frac = 0.5;
  unsigned seed = 1;
};
struct Metrics {
  double req_p50 = 0, req_p99 = 0, req_max = 0, hop_wait_p50 = 0, hop_wait_p99 = 0, miss_frac = 0;
  int nreq = 0; double train_ktasks_s = 0, train_gbs = 0, train_burst_ktasks_s = 0, util = 0, wall_ms = 0, serve_cta_ms = 0;
  double reclaim_max_us = 0, hop_run_p50 = 0, hop_run_p99 = 0, train_gap_p50 = 0, train_gap_p90 = 0; int err = 0;
};
struct PhaseM { double ms = 0, rate = 0, serve_p99 = 0, serve_p50 = 0, train_busy = 0, serve_busy = 0; int nreq = 0; double ctl_lo = 0, ctl_hi = 0; };
struct RunOut { Metrics m; std::vector<KvmCtlSample> ctl; std::vector<PhaseM> ph; double ctl_lead_ms = 0; int ctl_changes = 0; };

static RunOut run_load(KvmConfig cfg, const LoadSpec& L, bool with_ctl = false, KvmCtlConfig ctlcfg = {}) {
  RunOut out; Rng rg(L.seed);
  cfg.max_tasks = 1 << 20; cfg.live_cap = 1 << 14; cfg.ring_words = 1 << 18; cfg.timeout_us = 120ull * 1000000;
  KvmExec ex(cfg);
  const int SL = 128, N = 524288;                      // train slices: 3 x 128 x 2 MB = 768 MB, far beyond the 40 MB L2; an `add` moves 6.3 MB (~1 ms)
  float* big = ex.arena((size_t)SL * 3 * N); std::vector<int> tv;
  for (int i = 0; i < SL * 3; ++i) tv.push_back(ex.add_view(big + (size_t)i * N, 1, N));
  const int CS = 2048, CN = 4096;                      // cold serve slices
  float* cold = ex.arena((size_t)CS * 2 * CN); std::vector<int> cv;
  for (int i = 0; i < CS * 2; ++i) cv.push_back(ex.add_view(cold + (size_t)i * CN, 1, CN));
  int hx = ex.add_view(ex.arena(1024), 1, 1024), hy = ex.add_view(ex.arena(1024), 1, 1024), ho = ex.add_view(ex.arena(1024), 1, 1024);
  int ma = ex.add_view(ex.arena(64 * 64), 64, 64), mb = ex.add_view(ex.arena(64 * 64), 64, 64), mc = ex.add_view(ex.arena(64 * 64), 64, 64);
  int Ma = ex.add_view(ex.arena(128 * 128), 128, 128), Mb = ex.add_view(ex.arena(128 * 128), 128, 128), Mc = ex.add_view(ex.arena(128 * 128), 128, 128);
  ex.start();
  // serve request pool
  auto make_req = [&](int variant) {
    KvmBundleBuilder b; KvmTaskOpts o; o.workload = 0; o.priority = L.serve_prio; o.deadline_us = (int)L.slo_us;
    for (int f = 0; f < L.fan; ++f) {
      int prev = -1;
      for (int h = 0; h < L.hops; ++h) {
        int t;
        if (L.serve_kind == SK_HOT) t = b.add("add", {Bf(hx), Bf(hy), Bf(ho), I(1024)}, o);
        else if (L.serve_kind == SK_COLD) { int sl = (variant * 131 + f * 17 + h) % CS; t = b.add("copy", {Bf(cv[2 * sl]), Bf(cv[2 * sl + 1]), I(CN)}, o); }
        else t = b.add("matmul", {Bf(Ma), Bf(Mb), Bf(Mc), I(128), I(128), I(128)}, o);
        if (prev >= 0) b.edge(prev, t);
        prev = t;
      }
    }
    return b.build();
  };
  std::vector<KvmBundle> pool; for (int i = 0; i < 64; ++i) pool.push_back(make_req(i));
  const int req_tasks = L.fan * L.hops;
  // train bundle: independent tasks over distinct slices
  auto make_train = [&](int base) {
    KvmBundleBuilder b; KvmTaskOpts o; o.workload = 1; o.priority = L.train_prio;
    for (int i = 0; i < L.train_bundle; ++i) {
      int sl = (base + i) % SL;
      if (!L.train_bw_only && rg.in(0, 99) < (int)(L.train_matmul_frac * 100)) b.add("matmul", {Bf(Ma), Bf(Mb), Bf(Mc), I(128), I(128), I(128)}, o);
      else b.add("add", {Bf(tv[3 * sl]), Bf(tv[3 * sl + 1]), Bf(tv[3 * sl + 2]), I(N)}, o);
    }
    return b.build();
  };
  std::atomic<bool> stop{false}; std::atomic<long long> train_sub{0};
  std::thread tp([&] {
    int base = 0;
    while (!stop.load()) {
      long long done = (long long)ex.telemetry()->cls[1].done;
      if (train_sub.load() - done < L.train_outstanding) { KvmBundle b = make_train(base); base += L.train_bundle; if (ex.submit(b, 5000) != KvmSubmit::OK) break; train_sub += L.train_bundle; }
      else std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  std::unique_ptr<KvmController> ctl;
  if (with_ctl) { ctl.reset(new KvmController(ex.sched_params(), ex.telemetry(), cfg.sched.p, ctlcfg)); ctl->start(); }
  // serve arrivals: Poisson within each phase, open loop (the producer busy-waits to each arrival time)
  std::vector<int> req_phase; std::vector<double> req_arr; auto t0 = std::chrono::steady_clock::now(); double t_phase0 = 0;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));   // let the train backlog build
  t0 = std::chrono::steady_clock::now();
  if (ctl) out.ctl_lead_ms = 20;   // the controller started ~20 ms before t0 (the sleep above)
  int variant = 0;
  for (size_t ph = 0; ph < L.phases.size(); ++ph) {
    double t = t_phase0, tend = t_phase0 + L.phases[ph].ms * 1e3;
    while (true) {
      if (L.phases[ph].rate > 0) t += -std::log(1.0 - (rg.in(0, 999999) / 1e6)) / L.phases[ph].rate * 1e6; else t = tend;
      if (t >= tend) break;
      while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < t) {}
      if (ex.submit(pool[variant++ % pool.size()], 5000) != KvmSubmit::OK) { CHECK(false, "serve submit failed"); break; }
      req_phase.push_back((int)ph); req_arr.push_back(t);
    }
    while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < tend) std::this_thread::sleep_for(std::chrono::microseconds(50));
    t_phase0 = tend;
  }
  stop.store(true); tp.join(); if (ctl) { ctl->stop(); out.ctl = ctl->log(); out.ctl_changes = ctl->changes(); }
  const KvmResult& r = ex.finish();
  Metrics& m = out.m; m.err = r.err_code;
  if (!r.ok()) { CHECK(false, "device error %s", kvm_err_name(r.err_code)); return out; }
  auto v = tview(r);
  // serve requests = consecutive groups of req_tasks workload-0 slots
  std::vector<double> lat, hopw, hopr; std::vector<std::vector<double>> by_phase_t0(L.phases.size());
  std::vector<double> rq0, rq1; int miss = 0;
  { int k = 0; double mn = 1e30, mx = 0; int cnt = 0;
    for (size_t i = 0; i < v.size(); ++i) if (v[i].cls == 0) {
      mn = std::min(mn, (r.trace[i].admit_ns - r.trace[0].admit_ns) / 1e3); mx = std::max(mx, v[i].end); hopw.push_back(v[i].start - v[i].ready); hopr.push_back(v[i].end - v[i].start);
      if (++cnt == req_tasks) { double l = mx - mn; lat.push_back(l); miss += l > L.slo_us; rq0.push_back(mn); rq1.push_back(mx); cnt = 0; mn = 1e30; mx = 0; ++k; }
    } }
  m.nreq = (int)lat.size(); m.req_p50 = pct(lat, 50); m.req_p99 = pct(lat, 99); m.req_max = lat.empty() ? 0 : *std::max_element(lat.begin(), lat.end());
  m.hop_run_p50 = pct(hopr, 50); m.hop_run_p99 = pct(hopr, 99);
  { GapStats g = gap_stats_cls(r, ex.workers(), 1); m.train_gap_p50 = g.p50; m.train_gap_p90 = g.p90; }
  m.hop_wait_p50 = pct(hopw, 50); m.hop_wait_p99 = pct(hopw, 99); m.miss_frac = m.nreq ? (double)miss / m.nreq : 0;
  // time base: serve requests' first admission to the last end
  double w0 = rq0.empty() ? 0 : *std::min_element(rq0.begin(), rq0.end()), w1 = rq1.empty() ? 1 : *std::max_element(rq1.begin(), rq1.end());
  double train_n = 0, train_b = 0, busy = 0, burst_n = 0, burst_t = 0, serve_busy = 0;
  std::vector<std::pair<double, double>> burst;   // device-time windows of phases with the highest rate
  double maxrate = 0; for (auto& p : L.phases) maxrate = std::max(maxrate, p.rate);
  for (size_t ph = 0; ph < L.phases.size(); ++ph) if (L.phases[ph].rate >= maxrate * 0.99) {
    double a = 1e30, b = 0; for (size_t k = 0; k < rq0.size(); ++k) if (req_phase[k] == (int)ph) { a = std::min(a, rq0[k]); b = std::max(b, rq0[k]); }
    if (b > a) { burst.push_back({a, b}); burst_t += b - a; }
  }
  for (size_t i = 0; i < v.size(); ++i) {
    double a = std::max(v[i].start, w0), b = std::min(v[i].end, w1); if (b > a) { busy += b - a; if (v[i].cls == 0) serve_busy += b - a; }
    if (v[i].cls == 1 && v[i].end >= w0 && v[i].end <= w1) { train_n += 1; train_b += (double)r.info[i].bytes; }
    if (v[i].cls == 1) for (auto& bw : burst) if (v[i].end >= bw.first && v[i].end <= bw.second) burst_n += 1;
  }
  double span = w1 - w0;
  m.train_ktasks_s = train_n / span * 1e3; m.train_gbs = train_b / (span * 1e3); m.train_burst_ktasks_s = burst_t > 0 ? burst_n / burst_t * 1e3 : 0;
  m.util = busy / (span * ex.workers()); m.wall_ms = span / 1e3; m.serve_cta_ms = serve_busy / 1e3;
  // reclaim latency: worst wait of a task started under its class's reservation
  for (size_t i = 0; i < v.size(); ++i) if (v[i].flags & KVM_TRACE_RESV) m.reclaim_max_us = std::max(m.reclaim_max_us, v[i].start - v[i].ready);
  // per-phase device-time windows, anchored by the lowest observed (arrival -> admission) offset
  double off = 1e30; for (size_t k = 0; k < rq0.size() && k < req_arr.size(); ++k) off = std::min(off, rq0[k] - req_arr[k]);
  double ps = 0;
  for (size_t ph = 0; ph < L.phases.size(); ++ph) {
    PhaseM pm; pm.ms = L.phases[ph].ms; pm.rate = L.phases[ph].rate; double a = ps + off, b = ps + L.phases[ph].ms * 1e3 + off; ps += L.phases[ph].ms * 1e3;
    std::vector<double> pl; for (size_t k = 0; k < rq0.size(); ++k) if (req_phase[k] == (int)ph) pl.push_back(rq1[k] - rq0[k]);
    pm.nreq = (int)pl.size(); pm.serve_p99 = pct(pl, 99); pm.serve_p50 = pct(pl, 50);
    for (auto& t : v) { double x = std::max(t.start, a), y = std::min(t.end, b); if (y > x) (t.cls == 1 ? pm.train_busy : pm.serve_busy) += y - x; }
    pm.train_busy /= (b - a) * ex.workers(); pm.serve_busy /= (b - a) * ex.workers();
    out.ph.push_back(pm);
  }
  return out;
}

static void print_metrics(const char* name, const Metrics& m) {
  printf("  hop run p50 %.1f p99 %.1f us, train gap p50 %.1f p90 %.1f us\n", m.hop_run_p50, m.hop_run_p99, m.train_gap_p50, m.train_gap_p90);
  printf("  %-28s serve: %5d req p50 %7.0f p99 %8.0f max %8.0f us, hop wait p50 %6.1f p99 %8.1f us, SLO miss %5.1f%% | train %6.1f ktask/s %6.1f GB/s (burst %6.1f ktask/s) | util %4.1f%%\n",
         name, m.nreq, m.req_p50, m.req_p99, m.req_max, m.hop_wait_p50, m.hop_wait_p99, 100 * m.miss_frac, m.train_ktasks_s, m.train_gbs, m.train_burst_ktasks_s, 100 * m.util);
}

static KvmConfig mode_cfg(int mode, int workers_blocks) {
  KvmConfig c = sched_cfg(workers_blocks, 2, mode); c.sched.p.cls[0].weight = 4; c.sched.p.cls[1].weight = 1;
  return c;
}


// ---- repeated runs, median [min, max] ----
struct Rep { std::vector<Metrics> m; };
static Rep repeat(const KvmConfig& cfg, const LoadSpec& L, int n, bool ctl = false, const KvmCtlConfig& cc = {}) {
  Rep r; for (int i = 0; i < n; ++i) { LoadSpec l = L; l.seed = L.seed + i; RunOut o = run_load(cfg, l, ctl, cc); r.m.push_back(o.m); }
  return r;
}
template <class F> static Spread sp(const Rep& r, F f) { std::vector<double> v; for (auto& m : r.m) v.push_back(f(m)); return spread(v); }
static void print_rep(const char* name, const Rep& r) {
  Spread hr = sp(r, [](const Metrics& m) { return m.hop_run_p50; }), gp = sp(r, [](const Metrics& m) { return m.train_gap_p50; });
  printf("    [%s] chain hop run p50 %.1f us, stream CTA gap p50 %.1f us\n", name, hr.med, gp.med);
  Spread p50 = sp(r, [](const Metrics& m) { return m.req_p50; }), p99 = sp(r, [](const Metrics& m) { return m.req_p99; }), hw = sp(r, [](const Metrics& m) { return m.hop_wait_p99; });
  Spread ms = sp(r, [](const Metrics& m) { return 100 * m.miss_frac; }), kt = sp(r, [](const Metrics& m) { return m.train_ktasks_s; }), gb = sp(r, [](const Metrics& m) { return m.train_gbs; });
  Spread kb = sp(r, [](const Metrics& m) { return m.train_burst_ktasks_s; }), ut = sp(r, [](const Metrics& m) { return 100 * m.util; });
  printf("  %-26s req p50 %7.0f [%.0f,%.0f] p99 %8.0f [%.0f,%.0f] us | hop-wait p99 %7.1f [%.1f,%.1f] | miss %5.1f%% [%.1f,%.1f] | train %6.1f [%.1f,%.1f] kt/s %6.1f [%.1f,%.1f] GB/s burst %6.1f [%.1f,%.1f] kt/s | util %4.1f%% [%.1f,%.1f]\n",
         name, p50.med, p50.lo, p50.hi, p99.med, p99.lo, p99.hi, hw.med, hw.lo, hw.hi, ms.med, ms.lo, ms.hi, kt.med, kt.lo, kt.hi, gb.med, gb.lo, gb.hi, kb.med, kb.lo, kb.hi, ut.med, ut.lo, ut.hi);
}
static bool reps_ok(const Rep& r) { for (auto& m : r.m) if (m.err || m.nreq == 0) return false; return true; }

// ================= memory-bandwidth arbiter: streaming copies vs a dependent latency chain =================
static bool suite_arbiter() {
  int before = g_fail; const int N = 5;
  LoadSpec L; L.serve_kind = SK_COLD; L.hops = 16; L.fan = 1; L.slo_us = 250; L.phases = {{300, 4000}};
  L.train_bw_only = true; L.train_outstanding = 1200; L.train_bundle = 128;
  printf("  chain request = 16 dependent cold 32 KB copies (5 us each when idle), 4000 req/s, SLO %.0f us; stream = 6.3 MB `add` tasks (~1 ms), always 1200 outstanding; whole GPU\n", L.slo_us);
  struct Cfg { const char* name; int mode; int serve_prio; bool resv; bool arb; };
  std::vector<std::pair<std::string, Rep>> res;
  for (const Cfg& cf : std::vector<Cfg>{{"global queue, flat", KVM_SCHED_GLOBAL, 0, false, false}, {"global queue, chain prio 20", KVM_SCHED_GLOBAL, 20, false, false},
                                         {"dynamic, arbiter OFF", KVM_SCHED_DYNAMIC, 0, true, false}, {"dynamic, arbiter ON (60us)", KVM_SCHED_DYNAMIC, 0, true, true}, {"dynamic, arbiter ON (25us)", KVM_SCHED_DYNAMIC, 0, true, true}}) {
    KvmConfig c = mode_cfg(cf.mode, 0); LoadSpec l = L; l.serve_prio = cf.serve_prio;
    if (cf.resv) { c.sched.p.cls[0].min_cta = 8; c.sched.p.cls[0].urgent_us = 150; c.sched.p.cls[0].weight = 4; }
    if (cf.arb) { c.sched.p.flags |= KVM_SF_ARBITER; c.sched.p.cls[0].protect = 1; c.sched.p.cls[0].target_wait_us = strstr(cf.name, "25us") ? 25 : 60; c.sched.p.bw_limit_init = 1 << 20; c.sched.p.bw_limit_min = 16; c.sched.p.window_us = 200; }
    Rep r = repeat(c, l, N); CHECK(reps_ok(r), "%s: device error or no requests", cf.name); print_rep(cf.name, r); res.push_back({cf.name, r});
  }
  printf("  sweep of a static cap on concurrent stream (BW-bound) CTAs, arbiter flag off (the curve the adaptive limit searches):\n");
  for (int cap : {320, 256, 192, 128, 96, 64, 32}) {
    KvmConfig c = mode_cfg(KVM_SCHED_DYNAMIC, 0); c.sched.p.cls[0].min_cta = 8; c.sched.p.cls[0].urgent_us = 150; c.sched.p.cls[0].weight = 4; c.sched.p.cls[1].bw_cta = cap;
    char nm[64]; snprintf(nm, sizeof nm, "stream BW cap %d", cap); Rep r = repeat(c, L, 3); CHECK(reps_ok(r), "%s", nm); print_rep(nm, r);
  }
  for (int k : {3, 4}) {
    double off = sp(res[2].second, [](const Metrics& m) { return m.train_gbs; }).med, on = sp(res[k].second, [](const Metrics& m) { return m.train_gbs; }).med;
    double p_off = sp(res[2].second, [](const Metrics& m) { return m.req_p99; }).med, p_on = sp(res[k].second, [](const Metrics& m) { return m.req_p99; }).med;
    printf("  %s vs OFF: chain request p99 %.0f -> %.0f us (%.2fx), stream throughput retained %.1f%% (%.1f -> %.1f GB/s)\n", res[k].first.c_str(), p_off, p_on, p_off / std::max(p_on, 1.0), 100 * on / std::max(off, 1e-9), off, on);
  }
  return g_fail == before;
}

// Idle-CTA headroom for a latency chain next to a stream: the chain's next hop can only start on a CTA that is free NOW (tasks cannot be
// preempted), so the lever is the number of CTAs the stream may hold (max_cta = workers - k), not the bytes it moves.
static bool suite_headroom() {
  int before = g_fail; const int N = 5;
  LoadSpec L; L.serve_kind = SK_COLD; L.hops = 16; L.fan = 1; L.slo_us = 250; L.phases = {{300, 4000}};
  L.train_bw_only = true; L.train_outstanding = 1200; L.train_bundle = 128;
  printf("  chain request = 16 dependent cold 32 KB copies, 4000 req/s; stream = 6.3 MB adds always outstanding; stream max_cta = workers - k\n");
  for (int blocks : {108, 0}) for (int k : {0, 1, 2, 4, 8, 16}) {
    KvmConfig c = mode_cfg(KVM_SCHED_DYNAMIC, blocks); c.sched.p.cls[0].weight = 4; c.sched.p.cls[0].urgent_us = KVM_URGENT_ALL;
    int w = blocks ? blocks - 2 : 0; if (!w) { KvmExec probe(c); w = probe.workers(); }
    c.sched.p.cls[1].max_cta = w - k;
    char nm[80]; snprintf(nm, sizeof nm, "%d workers, headroom %d", w, k); Rep r = repeat(c, L, N); CHECK(reps_ok(r), "%s", nm); print_rep(nm, r);
  }
  return g_fail == before;
}

// Bandwidth lever: cap on the stream's concurrent BW-bound CTAs at the build's full grid, chain p99 against the stream throughput kept.
static bool suite_bwcap() {
  int before = g_fail; const int N = 5;
  LoadSpec L; L.serve_kind = SK_COLD; L.hops = 16; L.fan = 1; L.slo_us = 250; L.phases = {{300, 4000}};
  L.train_bw_only = true; L.train_outstanding = 1200; L.train_bundle = 128;
  KvmConfig c0 = mode_cfg(KVM_SCHED_DYNAMIC, 0); int w; { KvmExec probe(c0); w = probe.workers(); }
  printf("  full grid: %d workers; chain = 16 dependent cold 32 KB copies at 4000 req/s; stream = 6.3 MB adds; stream bw_cta = cap\n", w);
  for (int cap : {w, w * 3 / 4, w / 2, w * 3 / 8, w / 4, w / 8}) {
    KvmConfig c = mode_cfg(KVM_SCHED_DYNAMIC, 0); c.sched.p.cls[0].weight = 4; c.sched.p.cls[0].urgent_us = KVM_URGENT_ALL; c.sched.p.cls[1].bw_cta = cap;
    char nm[80]; snprintf(nm, sizeof nm, "stream BW cap %d of %d", cap, w); Rep r = repeat(c, L, N); CHECK(reps_ok(r), "%s", nm); print_rep(nm, r);
  }
  return g_fail == before;
}

// ================= rebalancing: controller on a time-varying load =================
static std::vector<Phase> trace_phases() { return {{200, 0}, {200, 200}, {300, 2500}, {200, 200}, {200, 0}, {300, 2500}, {200, 0}}; }
static LoadSpec rebalance_load() {
  LoadSpec L; L.serve_kind = SK_MATMUL; L.hops = 1; L.fan = 48; L.slo_us = 3000; L.phases = trace_phases(); L.train_outstanding = 1500; L.train_bundle = 128; L.train_matmul_frac = 0.5;
  return L;
}
static KvmCtlConfig ctl_for_serve(int hi) { KvmCtlConfig cc; cc.classes = {{0, 0, hi}}; cc.period_us = 500; cc.alpha = 0.3; cc.slack = 2; cc.deadband = 2; cc.release_periods = 8; return cc; }
static bool suite_rebalance() {
  int before = g_fail; const int W = 106;   // 108 blocks = one per SM: 1 admitter + 1 policy + 106 workers, identical in every mode
  LoadSpec L = rebalance_load();
  KvmConfig c = mode_cfg(KVM_SCHED_DYNAMIC, 108); c.sched.p.cls[0].weight = 1; c.sched.p.cls[0].urgent_us = 1500; c.sched.p.cls[0].target_wait_us = 500; c.sched.p.cls[0].protect = 1;
  RunOut o = run_load(c, L, true, ctl_for_serve(W / 2));
  CHECK(o.m.err == 0 && o.m.nreq > 100, "run failed");
  printf("  serve: bursts of 48-task requests (matmul 128^3); train: saturating mix of 6.3 MB adds and matmuls; %d workers; controller period 500 us, deadband 2, slow release 8 periods\n", W);
  printf("  phase  rate/s   ms  reqs  serve p50  p99 us  train busy  serve busy  reservation(min_cta) min..max over the 2nd half   changes in 2nd half\n");
  double ps = 0;
  for (size_t i = 0; i < o.ph.size(); ++i) {
    const PhaseM& p = o.ph[i]; double a = o.ctl_lead_ms + ps, b = a + p.ms, mid = (a + b) / 2; ps += p.ms;
    int lo = 1 << 30, hi = -1, ch = 0, prev = -1;
    for (auto& cs : o.ctl) if (cs.t_ms >= mid && cs.t_ms <= b) { lo = std::min(lo, cs.min_cta); hi = std::max(hi, cs.min_cta); if (prev >= 0 && cs.min_cta != prev) ++ch; prev = cs.min_cta; }
    printf("  %5zu %7.0f %4.0f %5d %10.0f %7.0f %10.1f%% %10.1f%%  %6d..%-6d %26d\n", i, p.rate, p.ms, p.nreq, p.serve_p50, p.serve_p99, 100 * p.train_busy, 100 * p.serve_busy, lo == (1 << 30) ? -1 : lo, hi, ch);
    if (p.rate == 0 && i > 0 && i + 1 < o.ph.size()) CHECK(p.train_busy > 0.85, "phase %zu: serve idle but train used only %.1f%% of the workers (work conservation)", i, 100 * p.train_busy);
    if (p.rate >= 2000) {
      CHECK(hi - lo <= 8, "phase %zu: reservation not stable, ranged %d..%d in the second half", i, lo, hi);
      CHECK(ch <= 4, "phase %zu: %d reservation changes in the second half (oscillation)", i, ch);
    }
  }
  // convergence: in the first burst (phase 2), time from the phase start until the reservation first reaches 90%% of its settled median
  { double a = o.ctl_lead_ms + 400, b = a + 300; std::vector<int> tail; for (auto& cs : o.ctl) if (cs.t_ms >= (a + b) / 2 && cs.t_ms <= b) tail.push_back(cs.min_cta);
    std::sort(tail.begin(), tail.end()); int med = tail.empty() ? 0 : tail[tail.size() / 2]; double t_hit = -1;
    for (auto& cs : o.ctl) if (cs.t_ms >= a && cs.min_cta >= 0.9 * med) { t_hit = cs.t_ms - a; break; }
    printf("  convergence: settled reservation %d CTAs; first reached 90%% of it %.2f ms after the burst began (controller period 0.5 ms); %d controller updates in %.1f s\n", med, t_hit, o.ctl_changes, 1.6);
    CHECK(med > 0 && t_hit >= 0 && t_hit < 20, "reservation did not converge quickly (%.2f ms)", t_hit); }
  return g_fail == before;
}

// ================= baselines under the scripted load =================
static bool suite_baselines() {
  int before = g_fail; const int W = 106, N = 5;   // 108 blocks, one per SM, in every mode (a static split by SM id is then a split by worker count)
  LoadSpec L = rebalance_load();
  printf("  scripted load: %zu phases (idle/low/BURST/low/idle/BURST/idle) over %d ms, %d workers (108 blocks: 1 admitter, 1 policy, the rest workers), best-of-%d, median [min,max]\n", L.phases.size(), 1600, W, N);
  struct Cfg { const char* name; int mode; int serve_prio; double share; bool ctl; bool arb; bool nolf; };
  std::vector<std::pair<std::string, Rep>> res;
  for (const Cfg& cf : std::vector<Cfg>{{"global queue, flat", KVM_SCHED_GLOBAL, 0, 0, false, false}, {"global queue, serve prio 20", KVM_SCHED_GLOBAL, 20, 0, false, false},
                                         {"static split 25% serve", KVM_SCHED_STATIC, 0, 0.25, false, false}, {"static split 50% serve", KVM_SCHED_STATIC, 0, 0.5, false, false},
                                         {"dynamic + controller (LATFIRST off)", KVM_SCHED_DYNAMIC, 0, 0, true, false, true}, {"dynamic + controller", KVM_SCHED_DYNAMIC, 0, 0, true, false},
                                         {"dynamic + ctl + arbiter", KVM_SCHED_DYNAMIC, 0, 0, true, true}}) {
    KvmConfig c = mode_cfg(cf.mode, 108); LoadSpec l = L; l.serve_prio = cf.serve_prio;
    if (cf.mode == KVM_SCHED_STATIC) { kvm_static_split(c.sched.p, {cf.share, 1 - cf.share}, g_sms); }
    c.sched.p.cls[0].weight = 1; c.sched.p.cls[0].urgent_us = cf.mode == KVM_SCHED_DYNAMIC ? KVM_URGENT_ALL : 0; c.sched.p.cls[0].protect = 1; c.sched.p.cls[0].target_wait_us = 500;
    if (cf.arb) c.sched.p.flags |= KVM_SF_ARBITER;
    if (cf.nolf) c.sched.p.flags &= ~KVM_SF_LATFIRST;
    Rep r = repeat(c, l, N, cf.ctl, ctl_for_serve(W / 2)); CHECK(reps_ok(r), "%s: device error or no requests", cf.name); print_rep(cf.name, r); res.push_back({cf.name, r});
  }
  auto med = [&](int i, auto f) { return sp(res[i].second, f).med; };
  auto p99 = [](const Metrics& m) { return m.req_p99; }; auto kt = [](const Metrics& m) { return m.train_ktasks_s; }; auto ut = [](const Metrics& m) { return m.util; };
  printf("  ratios (dynamic + controller, LATFIRST on, vs): serve p99 / train ktask/s / utilization\n");
  for (int i : {0, 1, 2, 3, 4}) printf("    vs %-36s p99 %.2fx lower, train %.2fx, utilization %.2fx\n", res[i].first.c_str(), med(i, p99) / med(5, p99), med(5, kt) / med(i, kt), med(5, ut) / med(i, ut));
  CHECK(med(5, p99) <= 1.15 * med(1, p99), "dynamic serve p99 %.0f us must not be worse than the global queue with serve priority (%.0f us) by more than 15%%", med(5, p99), med(1, p99));
  return g_fail == before;
}

// ================= serve tail latency: dynamic sharing against a hand-set priority queue =================
// Same scripted load as `baselines`; every row is one policy. `urgent` = urgent_us of the serve class (the width of its EDF window before the 3000 us deadline).
static bool suite_p99() {
  int before = g_fail; const int W = 106, N = 5;
  LoadSpec L = rebalance_load();
  printf("  scripted load as in `baselines` (SLO %.0f us), %d workers, %d reps, median [min,max]\n", L.slo_us, W, N);
  struct Cfg { const char* name; int mode; int prio; int urgent; int serve_min; bool ctl; bool arb; };
  const int ALWAYS = KVM_URGENT_ALL;
  std::vector<std::pair<std::string, Rep>> res;
  for (const Cfg& cf : std::vector<Cfg>{{"global queue, serve prio 20", KVM_SCHED_GLOBAL, 20, 0, 0, false, false},
                                         {"dyn+ctl, urgent 1500 (as before)", KVM_SCHED_DYNAMIC, 0, 1500, 0, true, false},
                                         {"dyn+ctl, urgent always", KVM_SCHED_DYNAMIC, 0, ALWAYS, 0, true, false},
                                         {"dyn static resv 53, urgent 1500", KVM_SCHED_DYNAMIC, 0, 1500, 53, false, false},
                                         {"dyn static resv 53, urgent always", KVM_SCHED_DYNAMIC, 0, ALWAYS, 53, false, false},
                                         {"dyn no resv, urgent always", KVM_SCHED_DYNAMIC, 0, ALWAYS, 0, false, false}}) {
    KvmConfig c = mode_cfg(cf.mode, 108); LoadSpec l = L; l.serve_prio = cf.prio;
    c.sched.p.cls[0].weight = 1; c.sched.p.cls[0].urgent_us = cf.urgent; c.sched.p.cls[0].protect = 1; c.sched.p.cls[0].target_wait_us = 500; c.sched.p.cls[0].min_cta = cf.serve_min;
    if (cf.arb) c.sched.p.flags |= KVM_SF_ARBITER;
    Rep r = repeat(c, l, N, cf.ctl, ctl_for_serve(W / 2)); CHECK(reps_ok(r), "%s: device error or no requests", cf.name); print_rep(cf.name, r); res.push_back({cf.name, r});
  }
  return g_fail == before;
}

// ================= task affinity: run a chain task on the block that ran its predecessor =================
static bool suite_affinity() {
  int before = g_fail; const int W = 64, HOPS = 300, N = 32768;
  printf("  chain of %d dependent in-place adds over %d KB, 64 workers kept busy by 100 us tasks of another class; hop = end(i) - end(i-1)\n", HOPS, N * 4 / 1024);
  double same_off = 0, same_on = 0;
  for (int aff : {0, 20, 100}) {
    std::vector<double> hop, same;
    for (int rep = 0; rep < 5; ++rep) {
      KvmConfig c = sched_cfg(W + 2, 2); c.max_tasks = 1 << 15; c.live_cap = 1 << 13; c.ring_words = 1 << 17; c.sched.p.affinity_us = aff;
      KvmExec ex(c); float* arena = ex.arena(2 * N); int x = ex.add_view(arena, 1, N), y = ex.add_view(arena + N, 1, N); ex.start();
      for (int b = 0; b < 20; ++b) CHECK(ex.submit(spin_bundle(300, 100, 1)) == KvmSubmit::OK, "submit");
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      KvmBundleBuilder bb; KvmTaskOpts o; o.workload = 0; int prev = -1;
      for (int h = 0; h < HOPS; ++h) { int t = bb.add("add", {Bf(x), Bf(y), Bf(x), I(N)}, o); if (prev >= 0) bb.edge(prev, t); prev = t; }
      CHECK(ex.submit(bb.build()) == KvmSubmit::OK, "submit");
      const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
      std::vector<int> ids; for (size_t i = 0; i < r.tasks.size(); ++i) if (r.tasks[i].workload_id == 0) ids.push_back((int)i);
      int sb = 0; for (size_t k = 1; k < ids.size(); ++k) { hop.push_back((r.trace[ids[k]].end_ns - r.trace[ids[k - 1]].end_ns) / 1e3); sb += r.trace[ids[k]].block == r.trace[ids[k - 1]].block; }
      same.push_back((double)sb / (ids.size() - 1));
    }
    printf("  affinity %3d us: hop p50 %.1f p90 %.1f us, successor on the same block %.0f%%\n", aff, pct(hop, 50), pct(hop, 90), 100 * median(same));
    if (aff == 0) same_off = median(same); if (aff == 20) same_on = median(same);
  }
  CHECK(same_on > 0.9 && same_on > same_off + 0.3, "affinity must keep the successor on the same block (off %.2f, on %.2f)", same_off, same_on);
  return g_fail == before;
}

// ================= overload: a serve flood that alone would take every CTA; who gets starved? =================
static bool suite_overload() {
  int before = g_fail; const int N = 5;
  LoadSpec L = rebalance_load(); L.phases = {{300, 9000}}; L.slo_us = 5000;
  printf("  serve flood: 9000 req/s x 48 matmul tasks (~115 CTAs of demand on 106 workers) for 300 ms, train saturating; train reservation 26 CTAs (25%%) in the dynamic row\n");
  struct Cfg { const char* name; int mode; int prio; int train_min; };
  std::vector<std::pair<std::string, Rep>> res;
  for (const Cfg& cf : std::vector<Cfg>{{"global queue, serve prio 20", KVM_SCHED_GLOBAL, 20, 0}, {"dynamic, train min 26 CTAs", KVM_SCHED_DYNAMIC, 0, 26}}) {
    KvmConfig c = mode_cfg(cf.mode, 108); c.sched.p.cls[0].weight = 4; c.sched.p.cls[0].urgent_us = 0; c.sched.p.cls[1].min_cta = cf.train_min;
    LoadSpec l = L; l.serve_prio = cf.prio; Rep r = repeat(c, l, N); CHECK(reps_ok(r), "%s", cf.name); print_rep(cf.name, r); res.push_back({cf.name, r});
  }
  auto kt = [](const Metrics& m) { return m.train_ktasks_s; };
  double g = sp(res[0].second, kt).med, d = sp(res[1].second, kt).med;
  printf("  train throughput under the flood: global-prio %.1f ktask/s, dynamic with reservation %.1f ktask/s (%.1fx); \n", g, d, d / std::max(g, 1e-9));
  return g_fail == before;
}


// ================= mix: serve + train + a Doom-rate frame stream on real GPT-pack ops =================
// serve (class 0): open-loop requests, each a chain of dependent `layernorm_fwd` hops over cold 16 x 1024 slabs (memory-bound, latency-critical: the decode/norm
//   side of inference).  train (class 1): a saturating backlog of `matmul_tile` (128x128 tiles, K = 1024, L2-resident operands: compute-bound), and two
//   memory-bound sweeps, `layernorm_fwd` over 256 x 1024 and `gelu_fwd` over 512K elements, from distinct cold slices.  frames (class 2): a Doom-RATE stream, 35 Hz,
//   frame = 1 `spin` tick of 5.4 ms (the measured real tick) then 4 parallel `spin` presents of 100 us, deadline = frame period. This is a stand-in: spin moves no
//   memory; the real Doom engine through the scheduler is doom/tests/run_sched.sh. All three use the SAME cost table path as production (kvm_cost()).
struct MixSpec { double ms = 600, serve_rate = 3000; int hops = 12, serve_prio = 0, train_prio = 0, frame_prio = 0, outstanding = 1500, bundle = 128; double mm_frac = 0.5, ln_frac = 0.25; bool frames = true, serve = true; unsigned seed = 1; double slo_us = 600; };
struct MixOut {
  int err = 0, nreq = 0, nframes = 0, frame_miss = 0; double req_p50 = 0, req_p99 = 0, frame_p99 = 0, frame_max = 0, mm_tflops = 0, sweep_gbs = 0, ktask_s = 0, util = 0, hop_run_p50 = 0, hop_wait_p99 = 0;
  std::vector<KvmBwArbSample> arb; int arb_changes = 0, cap1_min = KVM_UNLIMITED;
};
static MixOut run_mix(KvmConfig cfg, const MixSpec& L, bool ctl, bool hostarb, KvmBwArbConfig ac = {}) {
  MixOut out; Rng rg(L.seed);
  cfg.max_tasks = 1 << 20; cfg.live_cap = 1 << 14; cfg.ring_words = 1 << 18; cfg.timeout_us = 120ull * 1000000;
  KvmExec ex(cfg);
  auto zf = [&](size_t n) { float* p = ex.arena(n); CK(cudaMemset(p, 0, n * 4)); return p; };
  const int PS = 64, R = 16, D = 1024;                      // serve pool: 64 request variants x 12 hops, each hop its own cold slab (x, out)
  const int NSL = 2048; float* sx = zf((size_t)NSL * R * D); float* so = zf((size_t)NSL * R * D);
  int vg = ex.add_view(zf(D), 1, D), vb = ex.add_view(zf(D), 1, D);
  std::vector<int> sxv, sov, smean, srstd;
  for (int i = 0; i < NSL; ++i) { sxv.push_back(ex.add_view(sx + (size_t)i * R * D, R, D)); sov.push_back(ex.add_view(so + (size_t)i * R * D, R, D)); }
  for (int i = 0; i < PS; ++i) { smean.push_back(ex.add_view(zf(R), 1, R)); srstd.push_back(ex.add_view(zf(R), 1, R)); }
  const int TR = 256, SL = 96, GN = 524288;                 // train sweeps: layernorm 256 x 1024 (2 MB in, 2 MB out); gelu over 2 MB in + 2 MB out
  float* tx = zf((size_t)SL * TR * D); float* to = zf((size_t)SL * TR * D);
  float* gx = zf((size_t)SL * GN); float* go = zf((size_t)SL * GN);
  std::vector<int> txv, tov, gxv, gov;
  for (int i = 0; i < SL; ++i) { txv.push_back(ex.add_view(tx + (size_t)i * TR * D, TR, D)); tov.push_back(ex.add_view(to + (size_t)i * TR * D, TR, D)); gxv.push_back(ex.add_view(gx + (size_t)i * GN, 1, GN)); gov.push_back(ex.add_view(go + (size_t)i * GN, 1, GN)); }
  int tmean = ex.add_view(zf(TR), 1, TR), trstd = ex.add_view(zf(TR), 1, TR);
  const int MK = 1024; int ma = ex.add_view(zf((size_t)128 * MK), 128, MK), mb = ex.add_view(zf((size_t)MK * 128), MK, 128), mc = ex.add_view(zf(128 * 128), 128, 128);
  ex.start();
  auto make_req = [&](int var) {
    KvmBundleBuilder b; KvmTaskOpts o; o.workload = 0; o.priority = L.serve_prio; o.deadline_us = (int)L.slo_us; int prev = -1;
    for (int h = 0; h < L.hops; ++h) {
      int sl = (var * 131 + h * 17) % NSL;
      int t = b.add("layernorm_fwd", {Bf(sxv[sl]), Bf(vg), Bf(vb), Bf(sov[sl]), Bf(smean[var]), Bf(srstd[var]), I(0), I(R), I(D), F(1e-5f)}, o);
      if (prev >= 0) b.edge(prev, t); prev = t;
    }
    return b.build();
  };
  std::vector<KvmBundle> pool; for (int i = 0; i < PS; ++i) pool.push_back(make_req(i));
  auto make_train = [&](int base) {
    KvmBundleBuilder b; KvmTaskOpts o; o.workload = 1; o.priority = L.train_prio;
    for (int i = 0; i < L.bundle; ++i) {
      int sl = (base + i) % SL; int u = rg.in(0, 99);
      if (u < (int)(L.mm_frac * 100)) b.add("matmul_tile", {Bf(ma), Bf(mb), Bf(mc), KvmArg::none(), KvmArg::none(), KvmArg::none(), I(0), I(0), I(128), I(128), I(MK), I(32), F(1.f)}, o);
      else if (u < (int)((L.mm_frac + L.ln_frac) * 100)) b.add("layernorm_fwd", {Bf(txv[sl]), Bf(vg), Bf(vb), Bf(tov[sl]), Bf(tmean), Bf(trstd), I(0), I(TR), I(D), F(1e-5f)}, o);
      else b.add("gelu_fwd", {Bf(gxv[sl]), Bf(gov[sl]), I(0), I(GN)}, o);
    }
    return b.build();
  };
  std::atomic<bool> stop{false}; std::atomic<long long> train_sub{0};
  std::thread tp([&] {
    int base = 0;
    while (!stop.load()) {
      long long done = (long long)ex.telemetry()->cls[1].done;
      if (train_sub.load() - done < L.outstanding) { KvmBundle b = make_train(base); base += L.bundle; if (ex.submit(b, 5000) != KvmSubmit::OK) break; train_sub += L.bundle; }
      else std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  std::unique_ptr<KvmController> kc; std::unique_ptr<KvmBwArbiter> ba;
  KvmCtlConfig cc = ctl_for_serve(40);
  if (ctl) { kc.reset(new KvmController(ex.sched_params(), ex.telemetry(), cfg.sched.p, cc)); kc->start(); }
  if (hostarb) { ba.reset(new KvmBwArbiter(ex.sched_params(), ex.telemetry(), cfg.sched.p, cfg.sched.num_classes, ac)); ba->start(); }
  std::thread fp;
  if (L.frames) fp = std::thread([&] {
    auto f0 = std::chrono::steady_clock::now();
    for (int f = 0; (double)f * 1e3 / 35.0 < L.ms; ++f) {
      while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - f0).count() < f * 1e6 / 35.0) {}
      KvmBundleBuilder b; KvmTaskOpts o; o.workload = 2; o.priority = L.frame_prio; o.deadline_us = (int)(1e6 / 35.0);
      int tk = b.add("spin", {I(5400)}, o);
      for (int k = 0; k < 4; ++k) { int pr = b.add("spin", {I(100)}, o); b.edge(tk, pr); }
      if (ex.submit(b.build(), 5000) != KvmSubmit::OK) break;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  auto t0 = std::chrono::steady_clock::now(); int variant = 0;
  if (L.serve) {
    double t = 0, tend = L.ms * 1e3;
    while (true) {
      t += -std::log(1.0 - (rg.in(0, 999999) / 1e6)) / L.serve_rate * 1e6; if (t >= tend) break;
      while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < t) {}
      if (ex.submit(pool[variant++ % pool.size()], 5000) != KvmSubmit::OK) { CHECK(false, "serve submit failed"); break; }
    }
  } else std::this_thread::sleep_for(std::chrono::milliseconds((int)L.ms));
  while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < L.ms) std::this_thread::sleep_for(std::chrono::microseconds(50));
  stop.store(true); tp.join(); if (fp.joinable()) fp.join();
  if (kc) kc->stop(); if (ba) { ba->stop(); out.arb = ba->log(); out.arb_changes = ba->changes(); }
  const KvmResult& r = ex.finish(); out.err = r.err_code;
  if (!r.ok()) { CHECK(false, "device error %s", kvm_err_name(r.err_code)); return out; }
  auto v = tview(r);
  std::vector<double> lat, hopr, hopw, fl; { int cnt = 0; double mn = 1e30, mx = 0;
    for (size_t i = 0; i < v.size(); ++i) if (v[i].cls == 0) {
      mn = std::min(mn, (r.trace[i].admit_ns - r.trace[0].admit_ns) / 1e3); mx = std::max(mx, v[i].end); hopr.push_back(v[i].end - v[i].start); hopw.push_back(v[i].start - v[i].ready);
      if (++cnt == L.hops) { lat.push_back(mx - mn); cnt = 0; mn = 1e30; mx = 0; } } }
  { int cnt = 0; double mn = 1e30, mx = 0;
    for (size_t i = 0; i < v.size(); ++i) if (v[i].cls == 2) {
      mn = std::min(mn, (r.trace[i].admit_ns - r.trace[0].admit_ns) / 1e3); mx = std::max(mx, v[i].end);
      if (++cnt == 5) { fl.push_back(mx - mn); cnt = 0; mn = 1e30; mx = 0; } } }
  out.nreq = (int)lat.size(); out.req_p50 = pct(lat, 50); out.req_p99 = pct(lat, 99); out.hop_run_p50 = pct(hopr, 50); out.hop_wait_p99 = pct(hopw, 99);
  out.nframes = (int)fl.size(); out.frame_p99 = pct(fl, 99); out.frame_max = fl.empty() ? 0 : *std::max_element(fl.begin(), fl.end());
  for (double x : fl) out.frame_miss += x > 1e6 / 35.0;
  double w0 = 1e30, w1 = 0; for (size_t i = 0; i < v.size(); ++i) if (v[i].cls == 0) { w0 = std::min(w0, v[i].start); w1 = std::max(w1, v[i].end); }
  if (!L.serve) { w0 = 1e30; w1 = 0; for (size_t i = 0; i < v.size(); ++i) if (v[i].cls == 1) { w0 = std::min(w0, v[i].start); w1 = std::max(w1, v[i].end); } }
  double mmn = 0, swb = 0, tn = 0, busy = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    double a = std::max(v[i].start, w0), b = std::min(v[i].end, w1); if (b > a) busy += b - a;
    if (v[i].cls == 1 && v[i].end >= w0 && v[i].end <= w1) { tn += 1; if (v[i].op == KVM_OP_matmul_tile) mmn += 1; else if (v[i].op == KVM_OP_layernorm_fwd) swb += 8.0 * TR * D + 8.0 * TR; else swb += 8.0 * GN; }   // analytic, independent of the cost table so both builds are comparable
  }
  double span = std::max(w1 - w0, 1.0);
  out.mm_tflops = mmn * 2.0 * 128 * 128 * MK / (span * 1e6); out.sweep_gbs = swb / (span * 1e3); out.ktask_s = tn / span * 1e3; out.util = busy / (span * ex.workers());
  return out;
}
static void print_mix(const char* name, const std::vector<MixOut>& rs) {
  auto med = [&](auto f) { std::vector<double> x; for (auto& m : rs) x.push_back(f(m)); return spread(x); };
  Spread p50 = med([](const MixOut& m) { return m.req_p50; }), p99 = med([](const MixOut& m) { return m.req_p99; }), fp = med([](const MixOut& m) { return m.frame_p99; }), fm = med([](const MixOut& m) { return (double)m.frame_miss; });
  Spread tf = med([](const MixOut& m) { return m.mm_tflops; }), sg = med([](const MixOut& m) { return m.sweep_gbs; }), ut = med([](const MixOut& m) { return 100 * m.util; }), hr = med([](const MixOut& m) { return m.hop_run_p50; }), hw = med([](const MixOut& m) { return m.hop_wait_p99; });
  printf("  %-34s serve p50 %6.0f p99 %7.0f [%.0f,%.0f] us (hop run p50 %.1f, wait p99 %.0f) | frames p99 %.1f ms, missed %.0f/%d | train: matmul %.2f TFLOP/s, sweeps %.1f GB/s | util %.1f%%\n",
         name, p50.med, p99.med, p99.lo, p99.hi, hr.med, hw.med, fp.med / 1e3, fm.med, rs[0].nframes, tf.med, sg.med, ut.med);
}
static bool suite_mix() {
  int before = g_fail; const int N = 5, W = 106;
  MixSpec L;
  // 1. measured peak: train sweeps alone (no matmul, no serve, no frames), cost-table GB/s of the whole GPU
  double peak = 0;
  { MixSpec p = L; p.serve = false; p.frames = false; p.mm_frac = 0; p.ln_frac = 0.5; p.ms = 400; std::vector<double> g;
    for (int i = 0; i < 3; ++i) { p.seed = 100 + i; MixOut o = run_mix(mode_cfg(KVM_SCHED_GLOBAL, 108), p, false, false); CHECK(o.err == 0, "peak run"); g.push_back(o.sweep_gbs); }
    peak = spread(g).med; printf("  measured peak: memory-bound train sweeps alone move %.1f GB/s by the cost table (3 runs: %.1f..%.1f)\n", peak, spread(g).lo, spread(g).hi); }
  { MixSpec p = L; p.serve = false; p.frames = false; p.mm_frac = 1; p.ln_frac = 0; p.ms = 400; MixOut o = run_mix(mode_cfg(KVM_SCHED_GLOBAL, 108), p, false, false); CHECK(o.err == 0, "mm run");
    printf("  measured: matmul_tile 128x128xK1024 tiles alone sustain %.2f TFLOP/s\n", o.mm_tflops); }
  { MixSpec p = L; p.serve = true; p.frames = false; p.mm_frac = 0; p.ln_frac = 0; p.outstanding = 0; MixOut o = run_mix(mode_cfg(KVM_SCHED_GLOBAL, 108), p, false, false); CHECK(o.err == 0, "serve alone");
    printf("  serve alone (no train, no frames): p50 %.0f p99 %.0f us, hop run p50 %.1f us\n", o.req_p50, o.req_p99, o.hop_run_p50); }
  printf("  mix: serve %.0f req/s x %d dependent layernorm hops (SLO %.0f us); train backlog %d: %.0f%% matmul_tile / %.0f%% layernorm 256x1024 / %.0f%% gelu 512K; frames 35 Hz stand-in; %d workers; median [min,max] of %d\n",
         L.serve_rate, L.hops, L.slo_us, L.outstanding, 100 * L.mm_frac, 100 * L.ln_frac, 100 * (1 - L.mm_frac - L.ln_frac), W, N);
  struct Cfg { const char* name; int mode; bool prio; double share_serve, share_frame; bool ctl; int arb; };   // arb: 0 off, 1 device AIMD, 2 host measured
  for (const Cfg& cf : std::vector<Cfg>{{"global queue, flat", KVM_SCHED_GLOBAL, false, 0, 0, false, 0}, {"global queue, prio frames>serve>train", KVM_SCHED_GLOBAL, true, 0, 0, false, 0},
                                         {"static 10%/20%/70%", KVM_SCHED_STATIC, false, 0.2, 0.1, false, 0}, {"dynamic + controller", KVM_SCHED_DYNAMIC, false, 0, 0, true, 0},
                                         {"dynamic + ctl + device arbiter", KVM_SCHED_DYNAMIC, false, 0, 0, true, 1}, {"dynamic + ctl + host measured arb", KVM_SCHED_DYNAMIC, false, 0, 0, true, 2}}) {
    std::vector<MixOut> rs; std::vector<KvmBwArbSample> lastlog;
    for (int i = 0; i < N; ++i) {
      MixSpec l = L; l.seed = 1 + i; if (cf.prio) { l.serve_prio = 20; l.frame_prio = 30; }
      KvmConfig c = mode_cfg(cf.mode, 108); c.sched.num_classes = 3; c.sched.p.cls[0].weight = 1; c.sched.p.cls[1].weight = 1; c.sched.p.cls[2].weight = 1;
      if (cf.mode == KVM_SCHED_STATIC) kvm_static_split(c.sched.p, {cf.share_serve, 1 - cf.share_serve - cf.share_frame, cf.share_frame}, g_sms);
      if (cf.mode == KVM_SCHED_DYNAMIC) { c.sched.p.cls[0].urgent_us = 300; c.sched.p.cls[2].min_cta = 1; c.sched.p.cls[2].urgent_us = 20000; }
      c.sched.p.cls[0].protect = 1; c.sched.p.cls[0].target_wait_us = 100;
      if (cf.arb == 1) { c.sched.p.flags |= KVM_SF_ARBITER; c.sched.p.bw_limit_min = 16; c.sched.p.window_us = 200; }
      KvmBwArbConfig ac; ac.peak_gbs = peak;
      MixOut o = run_mix(c, l, cf.ctl, cf.arb == 2, ac); CHECK(o.err == 0 && o.nreq > 100, "%s: device error or no requests", cf.name); rs.push_back(o);
    }
    print_mix(cf.name, rs);
    if (cf.arb == 2) { const MixOut& o = rs[rs.size() / 2]; int ncap = 0; double smin = 1; for (auto& s : o.arb) { ncap += s.cap[1] != KVM_UNLIMITED; smin = std::min(smin, s.sat); }
      printf("    host arbiter (one run): %zu periods, %d cap writes, train capped in %d periods, min sat %.2f, total measured GB/s p50 over periods:", o.arb.size(), o.arb_changes, ncap, smin);
      std::vector<double> tg; for (auto& s : o.arb) tg.push_back(s.total_gbs); printf(" %.1f (peak %.1f)\n", pct(tg, 50), peak); }
  }
  return g_fail == before;
}

static bool sched_bench_suite(const std::string& s, bool& ok) {
#ifdef KVM_PROF_GAP
  if (s == "gapprof") { ok = suite_gapprof(); return true; }
#endif
  if (s == "calib") { ok = suite_calib(); return true; }
  if (s == "dispatch") { ok = suite_dispatch(); return true; }
  if (s == "hol") { ok = suite_hol(); return true; }
  if (s == "admit") { ok = suite_admit(); return true; }
  if (s == "bwcap") { ok = suite_bwcap(); return true; }
  if (s == "headroom") { ok = suite_headroom(); return true; }
  if (s == "p99") { ok = suite_p99(); return true; }
  if (s == "overload") { ok = suite_overload(); return true; }
  if (s == "affinity") { ok = suite_affinity(); return true; }
  if (s == "arbiter") { ok = suite_arbiter(); return true; }
  if (s == "rebalance") { ok = suite_rebalance(); return true; }
  if (s == "baselines") { ok = suite_baselines(); return true; }
  if (s == "mix") { ok = suite_mix(); return true; }
  if (s == "loadsmoke") {
    LoadSpec L = rebalance_load(); L.phases = {{150, 2500}, {100, 0}};
    KvmConfig c = mode_cfg(KVM_SCHED_DYNAMIC, 132); RunOut o = run_load(c, L); print_metrics("dynamic", o.m); ok = o.m.nreq > 100; return true;
  }
  return false;
}
