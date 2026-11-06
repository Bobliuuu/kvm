// Admission-path measurements and tests (include/kvm_exec.cuh, the admitter block).
//   tests/gpu/build_admit.sh ; ./test_admit [suite...]    suites: rate (tasks/s + per-phase breakdown), hol (latency bundle behind a big record)
// Build with -DKVM_ADMIT_PROF for the per-phase breakdown of the admitter (globaltimer sums in g_adm_prof).
#define KVM_SCHED_NO_BENCH
#define KVM_SCHED_NO_MAIN
#include "test_sched.cu"

#ifdef KVM_ADMIT_PROF
static void prof_reset() { unsigned long long z[16] = {0}; CK(cudaMemcpyToSymbol(g_adm_prof, z, sizeof z)); }
static void prof_print(const char* tag) {
  unsigned long long p[16]; CK(cudaMemcpyFromSymbol(p, g_adm_prof, sizeof p));
  const char* nm[6] = {"room/validate", "slot pop", "task copy+decode", "edges", "sched+commit", "release(enqueue)"};
  double tot = 0; for (int i = 0; i < 6; ++i) tot += p[i];
  printf("    [%s] %llu tasks in %llu records; admitter ns/task:", tag, p[6], p[7]);
  for (int i = 0; i < 6; ++i) printf(" %s %.0f |", nm[i], p[6] ? (double)p[i] / p[6] : 0.0);
  printf(" total %.0f\n", p[6] ? tot / p[6] : 0.0);
}
#else
static void prof_reset() {}
static void prof_print(const char*) {}
#endif

// Admission rate: back-to-back records of independent noops. A record is fully admitted when its last task is ready.
static double run_rate(int nrec, int K, int chain, bool sched, int blocks, double* span_us) {
  KvmConfig c = sched ? sched_cfg(blocks, 3) : small_cfg(blocks);
  c.max_tasks = 1 << 17; c.live_cap = 1 << 16; c.ring_words = 1 << 22;
  KvmExec ex(c); ex.start();
  for (int r = 0; r < nrec; ++r) {
    KvmBundleBuilder b; KvmTaskOpts o;
    for (int i = 0; i < K; ++i) { b.add("noop", {}, o); if (chain && i >= 1) b.edge(i - 1, i); }
    CHECK(ex.submit(b.build(), 30000) == KvmSubmit::OK, "submit");
  }
  const KvmResult& res = ex.finish(); CHECK(res.ok() && res.done == nrec * K, "done %d", res.done);
  unsigned long long a0 = ~0ull, r1 = 0; for (auto& t : res.trace) if (t.admit_ns) { a0 = std::min(a0, t.admit_ns); r1 = std::max(r1, t.ready_ns); }
  *span_us = (r1 - a0) / 1e3; return *span_us / ((double)nrec * K);
}
static bool suite_rate() {
  int before = g_fail;
  printf("  admission rate: noop records; us/task = (first admit .. last task ready) / tasks, best of 3\n");
  for (int sched : {0, 1}) for (int chain : {0, 1}) {
    for (int K : {1024, 8192, 32768}) {
      double best = 1e9, span = 0;
      prof_reset();
      for (int rep = 0; rep < 3; ++rep) { double s; double v = run_rate(std::max(1, 65536 / K), K, chain, sched, 0, &s); if (v < best) { best = v; span = s; } }
      char tag[64]; snprintf(tag, sizeof tag, "sched %d chain %d K %d", sched, chain, K);
      printf("  %-28s %.3f us/task  (%.2f Mtasks/s)\n", tag, best, 1.0 / best);
      prof_print(tag);
    }
  }
  return g_fail == before;
}

// Head-of-line: a 1-task class-0 bundle submitted right behind a K-task class-1 record. Reports, from the executor's own timestamps
// (microseconds from the big record's first admission): when the big record finished admitting (last ready), when the small task was
// admitted / ready / started / ended, and the end-to-end latency seen by the host.
static double us_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); }
static bool suite_hol(int blocks = 34) {
  int before = g_fail;
  printf("  small bundle behind a K-task record, %d blocks; times in us from the big record's admission\n", blocks);
  printf("  %8s %10s %10s %10s %10s %10s %10s %12s\n", "K", "big_ready", "sm_admit", "sm_ready", "sm_start", "sm_end", "host_lat", "big_ready/task");
  for (int K : {2000, 8000, 32000, 64000}) {
    const int REPS = std::min(6, (1 << 18) / (K + 1));   // slots are never recycled below, so the trace is complete
    KvmConfig c = sched_cfg(blocks, 2); c.max_tasks = 1 << 18; c.live_cap = 1 << 17; c.ring_words = 1 << 22;
    std::vector<double> lat, bigr, sa, sr, ss, se;
    KvmExec ex(c); ex.start();
    for (int rep = 0; rep < REPS; ++rep) {
      KvmBundleBuilder big; KvmTaskOpts o; o.workload = 1; for (int i = 0; i < K; ++i) big.add("noop", {}, o);
      unsigned w1, w2;
      CHECK(ex.submit(big.build(), 30000, false, &w1) == KvmSubmit::OK, "submit");
      const auto t0 = std::chrono::steady_clock::now();
      KvmBundleBuilder sm; KvmTaskOpts o0; o0.workload = 0; sm.add("noop", {}, o0);
      CHECK(ex.submit(sm.build(), 30000, false, &w2) == KvmSubmit::OK, "submit");
      while (!ex.window_done(w2)) std::this_thread::yield();
      lat.push_back(us_since(t0));
      while (!ex.window_done(w1)) std::this_thread::yield();
    }
    const KvmResult& r = ex.finish(); CHECK(r.ok(), "device error");
    // slots are not recycled (max_tasks > total), so slot order == admission order: record rep = tasks [rep*(K+1), rep*(K+1)+K), its small bundle next
    for (int rep = 0; rep < REPS; ++rep) {
      const size_t b0 = (size_t)rep * (K + 1), s0 = b0 + K;
      unsigned long long a0 = r.trace[b0].admit_ns, lr = 0; for (size_t i = b0; i < s0; ++i) lr = std::max(lr, r.trace[i].ready_ns);
      bigr.push_back((lr - a0) / 1e3); sa.push_back(((double)r.trace[s0].admit_ns - a0) / 1e3); sr.push_back(((double)r.trace[s0].ready_ns - a0) / 1e3);
      ss.push_back(((double)r.trace[s0].start_ns - a0) / 1e3); se.push_back(((double)r.trace[s0].end_ns - a0) / 1e3);
    }
    printf("  %8d %10.0f %10.0f %10.0f %10.0f %10.0f %10.0f %12.3f\n", K, median(bigr), median(sa), median(sr), median(ss), median(se), median(lat), median(bigr) / K);
  }
  return g_fail == before;
}

int main(int argc, char** argv) {
  std::vector<std::string> want; for (int i = 1; i < argc; ++i) want.push_back(argv[i]);
  if (want.empty()) want = {"rate"};
  int dev; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p, dev)); g_sms = p.multiProcessorCount;
  g_kvm_log = 1;
  printf("GPU %s, %d SMs, sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
  int failed = 0;
  for (auto& s : want) {
    int f0 = g_fail; printf("== %s\n", s.c_str()); fflush(stdout); bool ok = true;
    try { if (s == "rate") ok = suite_rate(); else if (s == "hol") ok = suite_hol(); else { fprintf(stderr, "unknown suite %s\n", s.c_str()); return 2; } }
    catch (const std::exception& e) { fprintf(stderr, "  EXCEPTION: %s\n", e.what()); ++g_fail; ok = false; }
    ok = ok && g_fail == f0; printf("== %s: %s\n", s.c_str(), ok ? "PASS" : "FAIL"); failed += !ok;
  }
  printf("%d checks, %d failed\n%s\n", g_checks, g_fail, failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
