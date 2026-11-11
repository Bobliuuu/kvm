// Where a dependent level's time goes in the executor: chains and layered DAGs of tiny tasks through the real ready queue, split by phase
// (trace timestamps and, in a -DKVM_PROF_GAP build, the SM-clock phases between two tasks on one block).
//   nvcc -O3 -arch=sm_80 -std=c++17 -I<core include dir> --fmad=false -Xcompiler -ffp-contract=off -DKVM_EXEC_NO_MAIN -DKVM_PROF_GAP bench/hop/chainprof.cu -o chainprof
#include "kvm_exec.cu"
#include <cinttypes>

static double median_of(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double mean_of(const std::vector<double>& v) { if (v.empty()) return 0; double s = 0; for (double x : v) s += x; return s / v.size(); }

static KvmConfig cfg_for(int blocks, int ntasks) {
  KvmConfig c; c.blocks = blocks;
#ifdef KVM_CHAIN_CFG
  c.chain = KVM_CHAIN_CFG;
#endif
  c.max_tasks = 1 << 17; c.live_cap = 1 << 13; c.ring_words = 1 << 20; c.succ_cap = 1u << 21; c.timeout_us = 60ull * 1000000; (void)ntasks;
#ifdef KVM_FIFO_CFG   // the panel's CLI setting: a single bundle as one record, one FIFO ring, no recycling, no version waits
  c.max_tasks = 1 << 18; c.live_cap = 1 << 17; c.ring_words = 1 << 22; c.succ_cap = 1u << 22; c.fifo_only = 1; c.no_recycle = 1; c.no_waits = 1;
#endif
  return c;
}

// Chain: every task depends on the one before it (the executor's dependent hop, nothing else).
static KvmBundle chain_bundle(int len, bool real_op, int bx, int by) {
  KvmBundleBuilder bb; int prev = -1;
  for (int i = 0; i < len; ++i) {
    int t = real_op ? bb.add("copy", {KvmArg::buf(bx), KvmArg::buf(by), KvmArg::i32(256)}) : bb.add("noop", {});
    if (prev >= 0) bb.edge(prev, t);
    prev = t;
  }
  return bb.build();
}

// Layered: W tasks per level, each depending on every task of the previous level (a join of W edges, as a dense GPT level has).
static KvmBundle layered_bundle(int width, int levels, int bx, int by) {
  KvmBundleBuilder bb; std::vector<int> prev;
  for (int l = 0; l < levels; ++l) {
    std::vector<int> cur;
    for (int w = 0; w < width; ++w) {
      int t = bb.add("copy", {KvmArg::buf(bx), KvmArg::buf(by), KvmArg::i32(256)});
      for (int p : prev) bb.edge(p, t);
      cur.push_back(t);
    }
    prev = cur;
  }
  return bb.build();
}

static void run_case(const char* name, const KvmConfig& c, const KvmBundle& b) {
  KvmExec ex(c);
  float* a = ex.arena(1024); int bx = ex.add_view(a, 1, 512), by = ex.add_view(a + 512, 1, 512); (void)bx; (void)by;
  ex.start();
  KvmSubmit st = (int)b.tasks.size() > c.live_cap ? ex.submit_stream(b, 2048, 60000) : ex.submit(b, 60000);
  if (st != KvmSubmit::OK) { printf("{\"case\":\"%s\",\"error\":\"%s\"}\n", name, kvm_submit_name(st)); return; }
  ex.close();
  const KvmResult& r = ex.finish();
  if (!r.ok()) { printf("{\"case\":\"%s\",\"error\":\"%s\"}\n", name, kvm_err_name(r.err_code)); return; }
  std::vector<const KvmTrace*> tr; for (auto& t : r.trace) if (t.start_ns) tr.push_back(&t);
  std::sort(tr.begin(), tr.end(), [](const KvmTrace* a, const KvmTrace* b) { return a->start_ns < b->start_ns; });
  std::vector<double> start_gap, end_to_ready, ready_to_start, end_to_start, run_ns; int cross = 0;
  for (size_t i = 0; i + 1 < tr.size(); ++i) {
    const KvmTrace& t0 = *tr[i]; const KvmTrace& t1 = *tr[i + 1];
    start_gap.push_back((double)(t1.start_ns - t0.start_ns));
    if (t1.ready_ns >= t0.end_ns) { end_to_ready.push_back((double)(t1.ready_ns - t0.end_ns)); ready_to_start.push_back((double)(t1.start_ns - t1.ready_ns)); end_to_start.push_back((double)(t1.start_ns - t0.end_ns)); }
    run_ns.push_back((double)(t0.end_ns - t0.start_ns));
    if (t1.block != t0.block) ++cross;
  }
  double span_us = (tr.back()->end_ns - tr.front()->start_ns) / 1e3;
  printf("{\"case\":\"%s\",\"blocks\":%d,\"tasks\":%d,\"kernel_ms\":%.3f,\"span_us\":%.1f,\"us_per_task\":%.3f,\"tasks_per_s_M\":%.3f,\"start_gap_ns_p50\":%.0f,\"start_gap_ns_mean\":%.0f,\"end_to_ready_ns_p50\":%.0f,\"ready_to_start_ns_p50\":%.0f,\"end_to_start_ns_p50\":%.0f,\"run_ns_p50\":%.0f,\"cross_block_frac\":%.3f",
         name, r.grid, r.done, r.kernel_ms, span_us, span_us / r.done, r.done / span_us, median_of(start_gap), mean_of(start_gap), median_of(end_to_ready), median_of(ready_to_start), median_of(end_to_start), median_of(run_ns), (double)cross / std::max<size_t>(1, tr.size() - 1));
#ifdef KVM_PROF_GAP
  static const char* ph[] = {"op_end_book+publish", "release", "sched_release", "p3", "p4", "p5", "pop_wait", "desc", "start_book"};
  printf(",\"gap_ns\":{");
  for (int k = 0; k < 9; ++k) {
    std::vector<double> v; for (auto& t : r.trace) if (t.gp[15] == 1 && t.gp[6] < 1000000) v.push_back(t.gp[k] / 1.41);
    printf("%s\"%s_mean\":%.0f,\"%s_p50\":%.0f", k ? "," : "", ph[k], mean_of(v), ph[k], median_of(v));
  }
  printf("}");
#endif
  printf("}\n"); fflush(stdout);
}

// Admission time per task, by tape size and shape: two identical records back to back; a record's tasks share one admission timestamp, taken after its slots were popped, so
// (timestamp of record 2 - timestamp of record 1) is the time the admitter spent on record 1 up to the same point of its work.
static void admit_case(const char* name, int width, int levels, int nargs_case) {
  (void)nargs_case;
  KvmBundle b = layered_bundle(width, levels, 0, 1);
  const int n = (int)b.tasks.size();
  KvmConfig c; c.blocks = 0; c.max_tasks = std::max(1 << 18, n * 4); { int p = 1 << 18; while (p < n * 4) p <<= 1; c.max_tasks = p; c.live_cap = p / 2; }
  c.ring_words = 1 << 26; c.succ_cap = 1u << 26; c.timeout_us = 120ull * 1000000;
  KvmExec ex(c);
  float* a = ex.arena(1024); int bx = ex.add_view(a, 1, 512), by = ex.add_view(a + 512, 1, 512); (void)bx; (void)by;
  ex.start();
  for (int rec = 0; rec < 2; ++rec) { KvmSubmit st = ex.submit(b, 120000, false); if (st != KvmSubmit::OK) { printf("{\"case\":\"%s\",\"error\":\"%s\"}\n", name, kvm_submit_name(st)); return; } }
  ex.close();
  const KvmResult& r = ex.finish();
  if (!r.ok()) { printf("{\"case\":\"%s\",\"error\":\"%s\"}\n", name, kvm_err_name(r.err_code)); return; }
  unsigned long long a0 = ~0ull, a1 = 0;
  for (auto& t : r.trace) if (t.admit_ns) { a0 = std::min(a0, t.admit_ns); a1 = std::max(a1, t.admit_ns); }
  double us = (a1 - a0) / 1e3;
  printf("{\"case\":\"%s\",\"tasks_per_record\":%d,\"edges_per_record\":%zu,\"admit_us\":%.1f,\"admit_ns_per_task\":%.1f,\"kernel_ms\":%.3f}\n", name, n, b.succ.size(), us, us * 1e3 / n, r.kernel_ms);
  fflush(stdout);
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;
  if (getenv("CHAINPROF_ADMIT")) {
    for (int n : {4096, 20480, 131072}) { char nm[64]; snprintf(nm, sizeof nm, "admit_layered_w32_n%d", n); admit_case(nm, 32, n / 32, 0); }
    for (int n : {4096, 20480, 131072}) { char nm[64]; snprintf(nm, sizeof nm, "admit_chain_n%d", n); admit_case(nm, 1, n, 0); }
    return 0;
  }
  KvmExec* probe = nullptr; (void)probe;
  for (int blocks : {2, 0}) {
    { KvmConfig c = cfg_for(blocks, 4000); KvmBundle b = chain_bundle(4000, false, 0, 1); char n[64]; snprintf(n, sizeof n, "chain_noop_b%d", blocks); run_case(n, c, b); }
    { KvmConfig c = cfg_for(blocks, 4000); KvmBundle b = chain_bundle(4000, true, 0, 1); char n[64]; snprintf(n, sizeof n, "chain_copy256_b%d", blocks); run_case(n, c, b); }
  }
  { const int n = 65536; KvmConfig c = cfg_for(0, n); c.max_tasks = 1 << 18; c.live_cap = 1 << 17; c.ring_words = 1 << 22; c.succ_cap = 1u << 22; KvmBundleBuilder bb; for (int i = 0; i < n; ++i) bb.add("noop", {}); run_case("indep_noop_65536", c, bb.build()); }
  for (int width : {4, 16}) { KvmConfig c = cfg_for(0, 8000); KvmBundle b = layered_bundle(width, 1000, 0, 1); char n[64]; snprintf(n, sizeof n, "layered_w%d_copy256", width); run_case(n, c, b); }
  return 0;
}
