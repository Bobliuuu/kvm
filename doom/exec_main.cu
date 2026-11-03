// Real Doom through the generic executor: loads a compiled bundle (runtime/compile_workload.py, ABI v2) that holds doom_tick/present/status
// tasks (and optionally a competing matmul load), runs it on the persistent scheduler, then writes frames.bin + summary.json (same
// format as cpu_run.py / doomtest, so compare.py diffs them), the per-task trace and the cadence / SM-usage metrics.
//   nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off doom/exec_main.cu doom/build/doom_dev.o -o doom_exec
//   doom_exec <bundle_dir> --wad doom1.wad --out DIR [--blocks N] [--stack BYTES] [--timeout-s S]
// kvm_exec.cu's own CLI main is renamed so its bundle reader and KvmExec class are reused unchanged.
#define main kvm_exec_cli_main
#include "kvm_exec.cu"
#undef main
#include <sys/stat.h>

static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; }

int main(int argc, char** argv) {
  KvmConfig cfg; cfg.stack_bytes = KVM_DOOM_STACK_BYTES; cfg.live_cap = 1 << 14;
  std::string dir, wad, out;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto sval = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "--wad") wad = sval(); else if (a == "--out") out = sval();
    else if (a == "--blocks") cfg.blocks = atoi(sval().c_str()); else if (a == "--stack") cfg.stack_bytes = (size_t)atol(sval().c_str());
    else if (a == "--timeout-s") cfg.timeout_us = (unsigned long long)atoll(sval().c_str()) * 1000000ull;
    else if (a == "--live-cap") cfg.live_cap = atoi(sval().c_str());
    else if (a[0] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; } else dir = a;
  }
  if (dir.empty() || wad.empty() || out.empty()) { fprintf(stderr, "usage: doom_exec <bundle_dir> --wad doom1.wad --out DIR [--blocks N] [--stack BYTES] [--timeout-s S]\n"); return 2; }
  try {
    mkdir(out.c_str(), 0755);
    if (!kvm_doom_host_init(wad.c_str(), (int)cfg.stack_bytes)) return 1;
    Json m = JsonParser(read_file(dir + "/metadata.json")).parse();
    if ((int)m.num("abi_version", -1) != KVM_ABI_VERSION) throw std::runtime_error("abi_version mismatch");
    const int nt = (int)m.num("task_count", -1), ne = (int)m.num("edge_count", -1), nw = (int)m.num("args_words", -1);
    KvmBundle b;
    b.tasks = read_bin<KvmTaskDesc>(dir + "/tasks.bin", nt); b.succ = read_bin<int32_t>(dir + "/successors.bin", ne); b.args = read_bin<int32_t>(dir + "/args.bin", nw);
    cfg.ring_words = (int)((std::max<size_t>(cfg.ring_words, 2 * b.words()) + 7) / 8 * 8);      // the whole bundle is one admission record
    if (b.tasks.size() > (size_t)cfg.live_cap) { cfg.live_cap = 1; while ((size_t)cfg.live_cap < b.tasks.size()) cfg.live_cap <<= 1; }
    KvmExec ex(cfg);
    std::map<std::string, int> buf_idx;
    for (auto& jb : m.get("buffers")->a) {
      const int idx = (int)jb.num("index", -1), rows = (int)jb.num("rows", 0), cols = (int)jb.num("cols", 0);
      std::vector<float> init;
      if (auto* in = jb.get("init")) {
        const Json* kind = in->get("kind");
        if (kind->s == "fill") init.assign((size_t)rows * cols, (float)in->num("value", 0)); else for (auto& v : in->get("values")->a) init.push_back((float)v.n);
      }
      if (ex.add_buffer(rows, cols, init) != idx) throw std::logic_error("buffer index mismatch");
      buf_idx[jb.get("name")->s] = idx;
    }
    KLOG(2, "bundle %s: %d tasks, %d edges, %d buffers; grid %d blocks, stack %zu B/thread", dir.c_str(), nt, ne, ex.nbufs(), ex.grid(), cfg.stack_bytes);
    ex.start();
    const KvmSubmit st = ex.submit(b, 60000);
    if (st != KvmSubmit::OK) { KLOG(0, "submit: %s (%s)", kvm_submit_name(st), ex.last_error().c_str()); ex.close(); ex.finish(); return 1; }
    const KvmResult& r = ex.finish();
    if (!r.ok()) return 1;
    kvm_write_trace_csv(r, (out + "/trace.csv").c_str());

    // ---- Doom outputs: per-frame hashes + state from d_state, raw frames from fb_<f>
    std::vector<float> state = ex.read_buffer(buf_idx.at("d_state"));
    int frames = 0; while (buf_idx.count("fb_" + std::to_string(frames))) ++frames;
    static const char* names[14] = {"booted", "tics", "gamestate", "health", "armor", "kills", "monsters", "items", "secrets", "deaths", "level_finished_tic", "x", "y", "waypoint"};
    FILE* fr = fopen((out + "/frames.bin").c_str(), "wb");
    for (int f = 0; f < frames; ++f) { std::vector<float> fb = ex.read_buffer(buf_idx.at("fb_" + std::to_string(f))); fwrite(fb.data(), 1, KVM_DOOM_FB_BYTES, fr); }
    fclose(fr);
    FILE* js = fopen((out + "/summary.json").c_str(), "w");
    fprintf(js, "{\"frames\": %d, \"hashes\": [", frames);
    for (int f = 0; f < frames; ++f) { uint32_t x, y; memcpy(&x, &state[KVM_DOOM_ST_HASH + KVM_DOOM_ST_REC * f], 4); memcpy(&y, &state[KVM_DOOM_ST_HASH + KVM_DOOM_ST_REC * f + 1], 4); fprintf(js, "%s[%u, %u]", f ? ", " : "", x, y); }
    fprintf(js, "], \"states\": [");
    for (int f = 0; f < frames; ++f) { fprintf(js, "%s[", f ? ", " : ""); for (int i = 0; i < 14; ++i) { int v; memcpy(&v, &state[KVM_DOOM_ST_HASH + KVM_DOOM_ST_REC * f + 2 + i], 4); fprintf(js, "%s%d", i ? ", " : "", v); } fprintf(js, "]"); }
    fprintf(js, "], \"state_names\": [");
    for (int i = 0; i < 14; ++i) fprintf(js, "%s\"%s\"", i ? ", " : "", names[i]);
    fprintf(js, "]}\n"); fclose(js);

    // ---- competing compute load: every chain output must equal A@B
    int mm_ok = 0, mm_bad = 0; double mm_worst = 0;
    if (buf_idx.count("mm_a")) {
      const std::vector<float> A = ex.read_buffer(buf_idx["mm_a"]), B = ex.read_buffer(buf_idx["mm_b"]);
      const int n = (int)std::lround(std::sqrt((double)A.size())); std::vector<double> ref((size_t)n * n);
      for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { double s = 0; for (int k = 0; k < n; ++k) s += (double)A[i * n + k] * B[k * n + j]; ref[i * n + j] = s; }
      for (int c = 0; buf_idx.count("mm_out_" + std::to_string(c)); ++c) {
        const std::vector<float> O = ex.read_buffer(buf_idx["mm_out_" + std::to_string(c)]); bool ok = true;
        for (size_t i = 0; i < O.size(); ++i) { const double e = fabs(O[i] - ref[i]) / std::max(1.0, fabs(ref[i])); mm_worst = std::max(mm_worst, e); if (e > 1e-5) ok = false; }
        (ok ? mm_ok : mm_bad)++;
      }
      KLOG(2, "compute check: %d/%d matmul outputs match A@B (worst rel err %.2g)", mm_ok, mm_ok + mm_bad, mm_worst);
    }

    // ---- cadence and SM usage from the executor's trace
    std::vector<double> tick_s(frames, -1), tick_e(frames, -1), frame_done(frames, -1), tick_wait;
    unsigned long long t0 = ~0ull; for (auto& t : r.trace) if (t.admit_ns) t0 = std::min(t0, t.admit_ns);
    auto us = [&](unsigned long long x) { return (double)(x - t0) / 1e3; };
    const int op_tick = kvm_op_by_name("doom_tick");
    std::map<int, double> busy_doom, busy_other; std::set<int> doom_sms, all_sms; double t_end = 0, other_run = 0, doom_run = 0;
    for (size_t i = 0; i < r.trace.size(); ++i) {
      const KvmTrace& t = r.trace[i]; const int op = r.tasks[i].op_id;
      if (!t.end_ns) continue;
      const double run = (double)(t.end_ns - t.start_ns) / 1e3; t_end = std::max(t_end, us(t.end_ns)); all_sms.insert(t.smid);
      const bool is_doom = op == op_tick || op == kvm_op_by_name("doom_present") || op == kvm_op_by_name("doom_status");
      (is_doom ? busy_doom : busy_other)[t.smid] += run; (is_doom ? doom_run : other_run) += run; if (is_doom) doom_sms.insert(t.smid);
    }
    // frame index of a task = its 'frame'/ordinal: ticks appear in frame order in the bundle
    { int f = 0; for (size_t i = 0; i < r.trace.size(); ++i) if (r.tasks[i].op_id == op_tick && f < frames) { const KvmTrace& t = r.trace[i]; tick_s[f] = us(t.start_ns); tick_e[f] = us(t.end_ns); tick_wait.push_back((double)(t.start_ns - t.ready_ns) / 1e3); ++f; } }
    { int f = -1; for (size_t i = 0; i < r.trace.size(); ++i) { const int op = r.tasks[i].op_id; if (op == op_tick) ++f; if (f >= 0 && f < frames && (op == op_tick || op == kvm_op_by_name("doom_present") || op == kvm_op_by_name("doom_status"))) frame_done[f] = std::max(frame_done[f], us(r.trace[i].end_ns)); } }
    std::vector<double> gap, lat; for (int f = 2; f < frames; ++f) gap.push_back(tick_s[f] - tick_s[f - 1]); for (int f = 1; f < frames; ++f) lat.push_back(frame_done[f] - tick_s[f]);
    const double span_s = frames > 2 ? (tick_s[frames - 1] - tick_s[1]) / 1e6 : 0;
    const int workers = r.grid - 1; const double busy_total = doom_run + other_run;
    FILE* mj = fopen((out + "/metrics.json").c_str(), "w");
    fprintf(mj, "{\"device\": \"%s\", \"sms\": %d, \"grid\": %d, \"frames\": %d, \"kernel_ms\": %.3f, \"boot_tick_ms\": %.3f, \"achieved_fps\": %.2f, "
                "\"tick_gap_us\": {\"p50\": %.1f, \"p99\": %.1f, \"max\": %.1f}, \"frame_latency_us\": {\"p50\": %.1f, \"p99\": %.1f, \"max\": %.1f}, "
                "\"tick_queue_wait_us\": {\"p50\": %.1f, \"p99\": %.1f, \"max\": %.1f}, \"sms_used\": %zu, \"sms_used_by_doom\": %zu, "
                "\"busy_us\": {\"doom\": %.0f, \"other\": %.0f}, \"worker_blocks\": %d, \"mean_block_utilization\": %.3f, \"matmul_ok\": %d, \"matmul_bad\": %d}\n",
            ex.device_name().c_str(), ex.sms(), r.grid, frames, r.kernel_ms, (tick_e[0] - tick_s[0]) / 1e3, span_s > 0 ? (frames - 2) / span_s : 0.0,
            pct(gap, .5), pct(gap, .99), pct(gap, 1), pct(lat, .5), pct(lat, .99), pct(lat, 1), pct(tick_wait, .5), pct(tick_wait, .99), pct(tick_wait, 1),
            all_sms.size(), doom_sms.size(), doom_run, other_run, workers, t_end > 0 ? busy_total / (t_end * workers) : 0.0, mm_ok, mm_bad);
    fclose(mj);
    KLOG(2, "frames %d, achieved %.1f fps (35 needed), tick gap p50/p99 %.2f/%.2f ms, frame latency p50/p99 %.2f/%.2f ms, tick queue wait p99 %.0f us, doom ran on %zu of %zu SMs, block utilization %.1f%%",
         frames, span_s > 0 ? (frames - 2) / span_s : 0.0, pct(gap, .5) / 1e3, pct(gap, .99) / 1e3, pct(lat, .5) / 1e3, pct(lat, .99) / 1e3, pct(tick_wait, .99), doom_sms.size(), all_sms.size(),
         t_end > 0 ? 100.0 * busy_total / (t_end * workers) : 0.0);
    printf("%s: %d/%d tasks, %.3f ms\n", mm_bad ? "FAIL" : "OK", r.done, r.admitted, r.kernel_ms);
    return mm_bad ? 1 : 0;
  } catch (const std::exception& e) { KLOG(0, "%s", e.what()); return 1; }
}
