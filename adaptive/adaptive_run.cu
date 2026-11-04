// adaptive_run: real Doom at a paced 35 Hz, GPT training runs and GPT serving requests on ONE persistent kernel under a chosen scheduler
// policy, with load phases that change while it runs. Measures what each tenant sees (frame deadlines, request latency, training
// throughput) and what the machine does (SM time per class, idle workers while work was ready).
//
//   nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off \
//        adaptive/adaptive_run.cu doom/build/doom_dev.o [doom/build/doom_dev_i*.o] -o adaptive_run
//   adaptive_run SCENARIO_DIR --out DIR --wad doom1.wad [--sched off|global|static|dynamic] [--ctl none|v1|adaptive] [--duration-s 20] ...
//
// Workload ids (= scheduler classes): 0 Doom, 1 serve, 2 train, 3 sweep (optional bandwidth-bound co-runner: AdamW over --sweep-gb of parameters). Only these ids are known
// here; the scheduler below sees workload_id, priority, deadline_us and the op cost table, never "Doom" or "GPT".
//   Doom   per instance a real-time loop: frame f is released at t0 + f/35 s (or when frame f-1 finished, if later) as a 6-task bundle
//          (tick, 4 present slices, status), deadline_us = one period. The frame is on time when it completed before t0 + (f+1)/35 s.
//   Serve  open-loop Poisson arrivals (piecewise rate: base, burst); a request = one bundle from a pool (adaptive/make_scenario.py).
//   Train  --train-runs independent runs; each admits its one-step window bundle again as soon as the previous one completed; can be paused.
// Every Doom frame is compared with the CPU oracle afterwards (adaptive/run_adaptive.sh): the scheduler must never change a frame.
#define main kvm_exec_cli_main
#include "../kvm_exec.cu"
#undef main
#include "../include/kvm_adapt.h"
#include <sys/stat.h>
#include <random>
#include <deque>
#include <sstream>
#include <iomanip>

namespace adp {
using Clock = std::chrono::steady_clock;
static double us_since(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
enum Cls { CLS_DOOM = 0, CLS_SERVE = 1, CLS_TRAIN = 2, CLS_SWEEP = 3 };

struct Opt {
  std::string scn, out, wad, sched = "dynamic", ctl = "none", tag;
  int doom_instances = 1, train_runs = -1, serve_slots = -1;
  double duration_s = 20, fps = 35, serve_rate = 30;
  double burst0 = -1, burst1 = -1, burst_rate = 0, pause0 = -1, pause1 = -1, scene0 = -1, scene1 = -1; int scene_extra_us = 0;
  int ctl_period_us = 1000, target_wait_us = 1000, floor_pct = 10, doom_min_hi = 8, serve_min_hi = 0, release_periods = 50, slack = 0;
  int sample_us = 2000, seed = 1, log_level = 2;
  double sweep_gb = 0; int sweep_chunk_k = 256, sweep_inflight = 2;   // co-runner: AdamW over 4 tensors of sweep_gb/4 GB each, chunks of sweep_chunk_k*1024 elements
  long long gate_serve = -1, gate_train = -1;   // admission budgets (tasks in flight per class); -1 = none
  bool gate_tune = false, arbiter_assist = false, no_doom = false, no_serve = false, no_train = false, dump_frames = true;
  KvmConfig cfg;
};

// ------------------------------------------------------------ bundles
struct Meta { Json j; std::string dir; };
static KvmInit parse_init(const Json& jb, const std::string& dir, int dtype) {
  KvmInit init;
  if (auto* in = jb.get("init")) {
    const std::string k = in->get("kind") ? in->get("kind")->s : "";
    if (k == "fill") { init.kind = KvmInit::FILL; init.value = (float)in->num("value", 0); }
    else if (k == "values") { init.kind = KvmInit::VALUES; for (auto& v : in->need("values").a) init.values.push_back(dtype == 1 ? [&] { int iv = (int)v.n; float f; memcpy(&f, &iv, 4); return f; }() : (float)v.n); }
    else if (k == "seeded") { init.kind = KvmInit::SEEDED; init.seed = in->get("seed") ? strtoull(in->get("seed")->s.c_str(), nullptr, 10) : 0; init.draws = (int)in->num("draws", 1); init.bias = (float)in->num("bias", 0); init.coef = (float)in->num("coef", 1); }
    else if (k == "file") { init.kind = KvmInit::FILE; init.path = dir + "/" + in->need("path").s; }
    else throw std::runtime_error(dir + ": unknown buffer init kind '" + k + "'");
  }
  return init;
}
struct Pool {   // the executor's global buffer table, filled by name
  KvmExec& ex; std::map<std::string, int> idx;
  explicit Pool(KvmExec& e) : ex(e) {}
  int add(const std::string& name, int rows, int cols, const KvmInit& init, int dtype = 0) {
    auto it = idx.find(name); if (it != idx.end()) return it->second;
    return idx[name] = ex.add_buffer(rows, cols, init, dtype);
  }
  int get(const std::string& n) const { auto it = idx.find(n); if (it == idx.end()) throw std::runtime_error("no buffer " + n); return it->second; }
};
// Reads a compiled bundle, registers its buffers by name and rewrites the local buffer indices of its arguments to the global table.
static KvmBundle load_bundle(const std::string& dir, Pool& pool, int cls) {
  Json m = JsonParser(read_file(dir + "/metadata.json")).parse();
  if ((int)m.num("abi_version", -1) != KVM_ABI_VERSION) throw std::runtime_error(dir + ": abi_version");
  const int nt = (int)m.num("task_count", -1), ne = (int)m.num("edge_count", -1), nw = (int)m.num("args_words", -1);
  KvmBundle b; b.tasks = read_bin<KvmTaskDesc>(dir + "/tasks.bin", nt); b.succ = read_bin<int32_t>(dir + "/successors.bin", ne); b.args = read_bin<int32_t>(dir + "/args.bin", nw);
  std::map<int, int> g;
  for (auto& jb : m.need("buffers").a) {
    const int dtype = (int)jb.num("dtype", 0);
    g[(int)jb.num("index", -1)] = pool.add(jb.need("name").s, (int)jb.num("rows", 0), (int)jb.num("cols", 0), parse_init(jb, dir, dtype), dtype);
  }
  for (int i = 0; i < nt; ++i) {
    KvmTaskDesc& t = b.tasks[i]; const KvmOpInfo* op = kvm_op(t.op_id);
    if (!op) throw std::runtime_error(dir + ": unknown op");
    t.workload_id = cls;
    for (int j = 0; j < op->arity; ++j) {
      const int ty = (int)((op->types >> (3 * j)) & 7u); int32_t& w = b.args[t.args_off + j];
      if ((ty == KVM_A_BUF || ty == KVM_A_BUF_OPT) && w >= 0) { auto it = g.find(w); if (it == g.end()) throw std::runtime_error(dir + ": argument names a buffer missing from metadata"); w = it->second; }
    }
  }
  return b;
}

// ------------------------------------------------------------ Doom
#ifdef KVM_WITH_GAME
struct DoomInst { int inst, input, state; std::vector<int> fb; };
static KvmBundle doom_frame(const DoomInst& d, int f, int deadline_us, int extra_us, bool indexed) {
  KvmBundleBuilder bb; KvmTaskOpts o; o.workload = CLS_DOOM; o.deadline_us = deadline_us;
  auto A = [&](std::vector<KvmArg> a) { if (indexed) a.push_back(KvmArg::i32(d.inst)); return a; };
  const char* sfx = indexed ? "_n" : "";
  const int tick = bb.add((std::string("doom_tick") + sfx).c_str(), A({KvmArg::i32(f), KvmArg::buf(d.input), KvmArg::buf(d.state)}), o);
  int gate = tick;
  if (extra_us > 0) { gate = bb.add("spin", {KvmArg::i32(extra_us)}, o); bb.edge(tick, gate); }
  for (int k = 0; k < 4; ++k) bb.edge(gate, bb.add((std::string("doom_present") + sfx).c_str(), A({KvmArg::buf(d.fb[f]), KvmArg::i32(k * 50), KvmArg::i32(50)}), o));
  bb.edge(gate, bb.add((std::string("doom_status") + sfx).c_str(), A({KvmArg::i32(f), KvmArg::buf(d.state)}), o));
  return bb.build();
}
#endif

// ------------------------------------------------------------ events
struct DoomEv { int inst, f; double rel_us, sub_us, ret_us, done_us; };
struct ServeEv { double arr_us, sub_us, done_us; int slot; };
struct TrainEv { int run, step; double sub_us, done_us; float loss; };

class Run {
 public:
  explicit Run(Opt o) : o_(std::move(o)) {}
  int go();
 private:
  Opt o_;
  std::unique_ptr<KvmExec> ex_;
  Clock::time_point t0_;
  std::atomic<bool> stop_{false}, failed_{false};
  std::mutex mu_;
  std::unique_ptr<KvmAdmissionGate> gate_; std::unique_ptr<KvmGateTuner> tuner_, tuner2_;
  std::vector<DoomEv> doom_ev_; std::vector<ServeEv> serve_ev_; std::vector<TrainEv> train_ev_;
  int serve_dropped_ = 0, serve_arrivals_ = 0;
  std::vector<KvmBundle> serve_b_, train_b_; std::vector<int> train_loss_buf_;
  double sweep_elems_ = 0;
  double now_us() const { return us_since(t0_, Clock::now()); }

  bool wait_window(unsigned wid, double limit_s = 120) {
    const auto t = Clock::now();
    while (!ex_->window_done(wid)) {
      if (failed_.load()) return false;
      if (us_since(t, Clock::now()) > limit_s * 1e6) { fprintf(stderr, "window %u did not complete in %.0f s\n", wid, limit_s); failed_ = true; return false; }
      std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
    return true;
  }
  bool submit(const KvmBundle& b, unsigned* wid) {
    const KvmSubmit s = ex_->submit(b, 20000, false, wid);
    if (s != KvmSubmit::OK) { fprintf(stderr, "submit: %s (%s)\n", kvm_submit_name(s), ex_->last_error().c_str()); failed_ = true; return false; }
    return true;
  }
  void sleep_until_us(double target_us) {   // coarse sleep, then spin the last 150 us
    for (;;) {
      const double d = target_us - now_us();
      if (d <= 0 || stop_.load()) return;
      if (d > 400) std::this_thread::sleep_for(std::chrono::microseconds((long long)(d - 300))); else if (d > 150) std::this_thread::yield();
    }
  }
#ifdef KVM_WITH_GAME
  void doom_loop(const DoomInst& d, int frames);
#endif
  void serve_loop(const std::vector<double>& arrivals);
  void train_loop(int run);
  void sweep_loop();
  std::vector<KvmBundle> sweep_b_; double sweep_bytes_ = 0;
  void sampler(std::ofstream& tl);
  std::vector<double> make_arrivals();
};

std::vector<double> Run::make_arrivals() {
  std::mt19937_64 rng(o_.seed * 7919 + 13); std::uniform_real_distribution<double> U(1e-12, 1.0);
  std::vector<double> a; double t = 0.2;   // seconds
  while (t < o_.duration_s) {
    const double rate = (o_.burst0 >= 0 && t >= o_.burst0 && t < o_.burst1) ? o_.burst_rate : o_.serve_rate;
    t += -std::log(U(rng)) / std::max(rate, 1e-6);
    if (t < o_.duration_s) a.push_back(t * 1e6);
  }
  return a;
}

#ifdef KVM_WITH_GAME
void Run::doom_loop(const DoomInst& d, int frames) {
  const double period = 1e6 / o_.fps;
  for (int f = 1; f < frames && !stop_.load() && !failed_.load(); ++f) {
    const double rel = f * period;
    sleep_until_us(rel);
    if (stop_.load()) break;
    const bool scene = o_.scene0 >= 0 && rel >= o_.scene0 * 1e6 && rel < o_.scene1 * 1e6;
    KvmBundle b = doom_frame(d, f, (int)period, scene ? o_.scene_extra_us : 0, o_.doom_instances > 1);
    unsigned wid; const double sub = now_us();
    if (!submit(b, &wid)) return;
    const double ret = now_us();
    if (!wait_window(wid)) return;
    const double done = now_us();
    if (tuner_) { tuner_->report_slack((rel + period - done) / period); tuner2_->report_slack((rel + period - done) / period); }
    std::lock_guard<std::mutex> lk(mu_); doom_ev_.push_back({d.inst, f, rel, sub, ret, done});
  }
}
#endif

void Run::serve_loop(const std::vector<double>& arrivals) {
  const int pool = (int)serve_b_.size();
  std::vector<int> freel; for (int i = 0; i < pool; ++i) freel.push_back(i);
  struct Fl { int slot; unsigned wid; double arr, sub; };
  std::vector<Fl> inflight;
  auto reap = [&]() {
    for (size_t i = 0; i < inflight.size();) {
      if (ex_->window_done(inflight[i].wid)) {
        const double done = now_us();
        { std::lock_guard<std::mutex> lk(mu_); serve_ev_.push_back({inflight[i].arr, inflight[i].sub, done, inflight[i].slot}); }
        if (gate_) gate_->release(CLS_SERVE, (long long)serve_b_[inflight[i].slot].tasks.size());
        freel.push_back(inflight[i].slot); inflight[i] = inflight.back(); inflight.pop_back();
      } else ++i;
    }
  };
  size_t next = 0;
  while ((!stop_.load() || !inflight.empty()) && !failed_.load()) {
    reap();
    if (next < arrivals.size() && !stop_.load() && now_us() >= arrivals[next]) {
      ++serve_arrivals_;
      if (freel.empty() || (gate_ && gate_->budget(CLS_SERVE) >= 0 && gate_->in_flight(CLS_SERVE) + (long long)serve_b_[0].tasks.size() > gate_->budget(CLS_SERVE))) { ++serve_dropped_; ++next; continue; }
      const int slot = freel.back(); freel.pop_back();
      if (gate_) gate_->acquire(CLS_SERVE, (long long)serve_b_[slot].tasks.size(), 1000);
      unsigned wid; const double sub = now_us();
      if (!submit(serve_b_[slot], &wid)) return;
      inflight.push_back({slot, wid, arrivals[next], sub}); ++next; continue;
    }
    if (stop_.load() && next >= arrivals.size() && inflight.empty()) break;
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
}

void Run::train_loop(int run) {
  int step = 0;
  while (!stop_.load() && !failed_.load()) {
    const double t = now_us();
    if (o_.pause0 >= 0 && t >= o_.pause0 * 1e6 && t < o_.pause1 * 1e6) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
    if (gate_ && !gate_->acquire(CLS_TRAIN, (long long)train_b_[run].tasks.size(), 1000)) continue;
    unsigned wid; const double sub = now_us();
    if (!submit(train_b_[run], &wid) || !wait_window(wid)) return;
    const double done = now_us();
    if (gate_) gate_->release(CLS_TRAIN, (long long)train_b_[run].tasks.size());
    float loss = -1;
    { std::lock_guard<std::mutex> lk(mu_); std::vector<float> l = ex_->read_buffer(train_loss_buf_[run]); loss = l[0]; train_ev_.push_back({run, step, sub, done, loss}); }
    ++step;
  }
}

// AdamW sweep: each bundle updates the whole parameter set once in chunks (28 bytes moved per element), sweep_inflight bundles in flight.
void Run::sweep_loop() {
  std::deque<std::pair<unsigned, double>> fl;
  while (!stop_.load() && !failed_.load()) {
    while (!fl.empty() && ex_->window_done(fl.front().first)) { sweep_bytes_ += fl.front().second; fl.pop_front(); }
    if ((int)fl.size() >= o_.sweep_inflight) { std::this_thread::sleep_for(std::chrono::microseconds(100)); continue; }
    unsigned wid; if (!submit(sweep_b_[0], &wid)) return;
    fl.push_back({wid, 28.0 * sweep_elems_});
  }
  for (auto& f : fl) { wait_window(f.first); sweep_bytes_ += f.second; }
}

void Run::sampler(std::ofstream& tl) {
  const KvmSchedTele* T = ex_->telemetry(); const KvmSchedParams* P = ex_->sched_params(); const int nc = ex_->num_classes();
  tl << "t_ms"; for (int c = 0; c < nc; ++c) tl << ",c" << c << "_running,c" << c << "_ready,c" << c << "_sm_ns,c" << c << "_done,c" << c << "_missed,c" << c << "_near,c" << c << "_min,c" << c << "_max,c" << c << "_w,c" << c << "_urg"; tl << "\n";
  while (!stop_.load() && !failed_.load()) {
    tl << std::fixed << std::setprecision(3) << now_us() / 1e3;
    for (int c = 0; c < nc; ++c) {
      const KvmClassTele& x = T->cls[c]; const KvmClassParams& q = P->cls[c];
      tl << "," << x.running << "," << x.ready << "," << x.sm_ns << "," << x.done << "," << x.missed << "," << x.near_miss << "," << q.min_cta << "," << (q.max_cta == KVM_UNLIMITED ? -1 : q.max_cta) << "," << q.weight << "," << q.urgent_us;
    }
    tl << "\n";
    std::this_thread::sleep_for(std::chrono::microseconds(o_.sample_us));
  }
}

static double pctl(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; }

int Run::go() {
  Json scn = JsonParser(read_file(o_.scn + "/scenario.json")).parse();
  const int s_runs = (int)scn.need("train").num("runs", 0), s_pool = (int)scn.need("serve").num("pool", 0);
  const int runs = o_.no_train ? 0 : (o_.train_runs < 0 ? s_runs : std::min(o_.train_runs, s_runs));
  const int pool_n = o_.no_serve ? 0 : (o_.serve_slots < 0 ? s_pool : std::min(o_.serve_slots, s_pool));
  const int max_new = (int)scn.need("serve").num("max_new", 0);
  mkdir(o_.out.c_str(), 0755);
  const int nc = o_.cfg.sched.on ? std::max(o_.cfg.sched.num_classes, o_.sweep_gb > 0 ? 4 : 3) : 1;
  if (o_.cfg.sched.on) o_.cfg.sched.num_classes = nc;
  ex_.reset(new KvmExec(o_.cfg));
  Pool pool(*ex_);
  if (o_.sweep_gb > 0) {
    const long long n = (long long)(o_.sweep_gb * 1e9 / 4 / 4), chunk = (long long)o_.sweep_chunk_k * 1024;
    int t[4]; const char* nm[4] = {"p", "g", "m", "v"};
    for (int k = 0; k < 4; ++k) t[k] = pool.add(std::string("sw_") + nm[k], 1, (int)n, {});
    KvmInit one; one.kind = KvmInit::FILL; one.value = 1; const int step = pool.add("sw_step", 1, 1, one, 1);
    KvmBundleBuilder bb; KvmTaskOpts o; o.workload = CLS_SWEEP;
    for (long long e0 = 0; e0 < n; e0 += chunk)
      bb.add("adamw", {KvmArg::buf(t[0]), KvmArg::buf(t[1]), KvmArg::buf(t[2]), KvmArg::buf(t[3]), KvmArg::none(), KvmArg::buf(step), KvmArg::i32((int)e0), KvmArg::i32((int)std::min(n, e0 + chunk)),
                  KvmArg::f32(1e-3f), KvmArg::f32(0.9f), KvmArg::f32(0.999f), KvmArg::f32(1e-8f), KvmArg::f32(0.01f)}, o);
    sweep_b_.push_back(bb.build()); sweep_elems_ = (double)n;
  }
  const int frames = o_.no_doom ? 0 : (int)(o_.duration_s * o_.fps) + 3;
  for (int r = 0; r < runs; ++r) train_b_.push_back(load_bundle(o_.scn + "/train_r" + std::to_string(r), pool, CLS_TRAIN));
  for (int r = 0; r < runs; ++r) train_loss_buf_.push_back(pool.get("t" + std::to_string(r) + ".loss.0"));
  for (int i = 0; i < pool_n; ++i) serve_b_.push_back(load_bundle(o_.scn + "/serve_" + std::to_string(i), pool, CLS_SERVE));
#ifdef KVM_WITH_GAME
  std::vector<DoomInst> doom;
  if (frames) {
    if (!kvm_doom_host_init(o_.wad.c_str(), (int)o_.cfg.stack_bytes)) return 1;
    for (int k = 0; k < o_.doom_instances; ++k) {
      DoomInst d; d.inst = k; const std::string s = "d" + std::to_string(k);
      d.input = pool.add(s + ".input", 1, 1, {}); d.state = pool.add(s + ".state", 1, KVM_DOOM_STATE_WORDS(frames), {});
      for (int f = 0; f < frames; ++f) d.fb.push_back(pool.add(s + ".fb" + std::to_string(f), 1, KVM_DOOM_FB_WORDS, {}));
      doom.push_back(std::move(d));
    }
  }
#else
  if (frames) { fprintf(stderr, "built without the game pack (-DKVM_WITH_GAME): use --no-doom\n"); return 2; }
#endif
  if (ex_->nbufs() == 0) return 2;
  KLOG(2, "%d buffers; %d train runs (%d tasks/window), %d serve slots (%d tasks/request), %d doom instances x %d frames; grid %d, workers %d", ex_->nbufs(), runs,
       runs ? (int)train_b_[0].tasks.size() : 0, pool_n, pool_n ? (int)serve_b_[0].tasks.size() : 0, frames ? o_.doom_instances : 0, frames, ex_->grid(), ex_->workers());

  if (o_.gate_serve >= 0 || o_.gate_train >= 0) { gate_.reset(new KvmAdmissionGate({-1, o_.gate_serve, o_.gate_train, -1})); 
    if (o_.gate_tune) {   // closed loop: the budgets given are the maxima; Doom's end-to-end slack shrinks them under pressure
      const long long hs = o_.gate_serve > 0 ? o_.gate_serve : 1 << 30, ht = o_.gate_train > 0 ? o_.gate_train : 1 << 30;
      tuner_.reset(new KvmGateTuner(*gate_, {CLS_SERVE}, std::max(1LL, hs / 10), hs)); tuner2_.reset(new KvmGateTuner(*gate_, {CLS_TRAIN}, std::max(1LL, ht / 10), ht));
    } }
  std::unique_ptr<KvmController> v1; std::unique_ptr<KvmAdaptive> ad;
  ex_->start();
  // ---- boot Doom (frame 0 loads the level, 170-330 ms) before the clock starts, so the measured window starts on a running game
#ifdef KVM_WITH_GAME
  { std::vector<std::thread> boot;
    for (auto& d : doom) boot.emplace_back([&, pd = &d] { unsigned wid; KvmBundle b = doom_frame(*pd, 0, -1, 0, o_.doom_instances > 1); if (submit(b, &wid)) wait_window(wid, 60); });
    for (auto& t : boot) t.join(); }
  if (failed_) { ex_->close(); ex_->finish(); return 1; }
#endif
  const KvmSchedParams base = o_.cfg.sched.p;
  const int W = ex_->workers();
  if (o_.cfg.sched.on && o_.cfg.sched.mode == KVM_SCHED_DYNAMIC) {
    if (o_.ctl == "v1") {
      KvmCtlConfig cc; cc.period_us = o_.ctl_period_us;
      cc.classes = {{CLS_DOOM, 0, o_.doom_min_hi}, {CLS_SERVE, 0, std::max(1, W / 2)}};
      v1.reset(new KvmController(ex_->sched_params(), ex_->telemetry(), base, cc));
    } else if (o_.ctl == "adaptive") {
      KvmAdaptConfig ac; ac.workers = W; ac.period_us = o_.ctl_period_us; ac.arbiter_assist = o_.arbiter_assist; ac.release_periods = o_.release_periods; ac.slack = o_.slack;
      KvmAdaptClass d; d.cls = CLS_DOOM; d.role = KvmRole::DEADLINE; d.min_hi = o_.doom_min_hi; d.w_lo = 2; d.w_hi = 16; d.urgent_max_us = (int)(1e6 / o_.fps); d.boost_max = o_.doom_min_hi;
      KvmAdaptClass s; s.cls = CLS_SERVE; s.role = KvmRole::LATENCY; s.min_hi = o_.serve_min_hi ? o_.serve_min_hi : W; s.w_lo = 1; s.w_hi = 8; s.target_wait_us = o_.target_wait_us;
      KvmAdaptClass t; t.cls = CLS_TRAIN; t.role = KvmRole::BATCH; t.floor_cta = std::max(1, W * o_.floor_pct / 100);
      ac.classes = {d, s, t};
      ad.reset(new KvmAdaptive(ex_->sched_params(), ex_->telemetry(), base, ac));
    }
  }
  std::ofstream tl(o_.out + "/timeline.csv");
  t0_ = Clock::now();
  std::thread th_sample; if (o_.cfg.sched.on) th_sample = std::thread([&] { sampler(tl); });
  if (v1) v1->start();
  if (ad) ad->start();
  std::vector<std::thread> th;
#ifdef KVM_WITH_GAME
  for (auto& d : doom) th.emplace_back([&, pd = &d] { doom_loop(*pd, frames); });
#endif
  std::vector<double> arrivals; if (pool_n) arrivals = make_arrivals();
  if (pool_n) th.emplace_back([&] { serve_loop(arrivals); });
  for (int r = 0; r < runs; ++r) th.emplace_back([&, r] { train_loop(r); });
  if (o_.sweep_gb > 0) th.emplace_back([&] { sweep_loop(); });
  std::this_thread::sleep_for(std::chrono::duration<double>(o_.duration_s));
  stop_ = true;
  for (auto& t : th) t.join();
  const double t_end = now_us();
  if (v1) v1->stop();
  if (ad) ad->stop();
  if (th_sample.joinable()) th_sample.join();
  tl.close();
  ex_->close();
  const KvmResult& r = ex_->finish();
  if (!r.ok() || failed_) { KLOG(0, "run failed (device error %s)", kvm_err_name(r.err_code)); return 1; }
  kvm_print_class_report(r);

  // ---------------------------------------------------------------- outputs
  { std::ofstream f(o_.out + "/doom_frames.csv"); f << "inst,frame,release_us,submit_us,done_us,submit_ret_us\n"; for (auto& e : doom_ev_) f << e.inst << "," << e.f << "," << e.rel_us << "," << e.sub_us << "," << e.done_us << "," << e.ret_us << "\n"; }
  { std::ofstream f(o_.out + "/serve.csv"); f << "arrival_us,submit_us,done_us,slot\n"; for (auto& e : serve_ev_) f << e.arr_us << "," << e.sub_us << "," << e.done_us << "," << e.slot << "\n"; }
  { std::ofstream f(o_.out + "/train.csv"); f << "run,step,submit_us,done_us,loss\n"; for (auto& e : train_ev_) f << e.run << "," << e.step << "," << e.sub_us << "," << e.done_us << "," << e.loss << "\n"; }
  if (ad) { std::ofstream f(o_.out + "/controller.csv"); f << "t_ms,cls,running,ready,min_cta,max_cta,weight,urgent_us,boost,share,near_miss,missed\n"; for (auto& s : ad->log()) f << s.t_ms << "," << s.cls << "," << s.running << "," << s.ready << "," << s.min_cta << "," << s.max_cta << "," << s.weight << "," << s.urgent_us << "," << s.boost << "," << s.share << "," << s.near_miss << "," << s.missed << "\n"; }
  if (v1) { std::ofstream f(o_.out + "/controller.csv"); f << "t_ms,cls,running,ready,min_cta,wait_max_us\n"; for (auto& s : v1->log()) f << s.t_ms << "," << s.cls << "," << s.running << "," << s.ready << "," << s.min_cta << "," << s.wait_max_us << "\n"; }
  { std::ofstream f(o_.out + "/doom_tasks.csv"); f << "op,inst_frame_slot,admit_us,ready_us,start_us,end_us,smid\n";   // class 0 only: the full trace is millions of rows
    unsigned long long t0d = ~0ull; for (auto& t : r.trace) if (t.admit_ns) t0d = std::min(t0d, t.admit_ns);
    for (size_t i = 0; i < r.trace.size(); ++i) if (r.tasks[i].workload_id == CLS_DOOM && r.trace[i].end_ns) {
      const KvmTrace& t = r.trace[i]; auto u = [&](unsigned long long x) { return (double)(x - t0d) / 1e3; };
      f << kvm_op(r.tasks[i].op_id)->name << "," << i << "," << u(t.admit_ns) << "," << u(t.ready_ns) << "," << u(t.start_ns) << "," << u(t.end_ns) << "," << t.smid << "\n"; } }

  // ---- correctness: Doom frames per instance (summary.json + frames.bin, same format as cpu_run.py), serve tokens, train losses
  int serve_ok = 1; std::vector<float> ref_tok;
  for (int i = 0; i < pool_n; ++i) {
    bool used = false; for (auto& e : serve_ev_) used = used || e.slot == i;
    if (!used) continue;
    std::vector<float> tok; for (int j = 1; j <= max_new; ++j) { auto v = ex_->read_buffer(pool.get("sv" + std::to_string(i) + ".cur" + std::to_string(j))); tok.push_back(v[0]); }
    if (ref_tok.empty()) ref_tok = tok; else if (memcmp(ref_tok.data(), tok.data(), tok.size() * 4) != 0) { serve_ok = 0; KLOG(0, "serve slot %d produced different tokens", i); }
  }
  int doom_done = 0;
#ifdef KVM_WITH_GAME
  for (auto& d : doom) {
    int n = 0; { std::set<int> fr; for (auto& e : doom_ev_) if (e.inst == d.inst) fr.insert(e.f); n = (int)fr.size() + 1; }   // frames 0..n-1 completed (frame 0 = boot)
    doom_done += n;
    const std::string dir = o_.out + "/doom" + std::to_string(d.inst); mkdir(dir.c_str(), 0755);
    std::vector<float> state = ex_->read_buffer(d.state);
    static const char* names[14] = {"booted", "tics", "gamestate", "health", "armor", "kills", "monsters", "items", "secrets", "deaths", "level_finished_tic", "x", "y", "waypoint"};
    FILE* js = fopen((dir + "/summary.json").c_str(), "w");
    fprintf(js, "{\"frames\": %d, \"hashes\": [", n);
    for (int f = 0; f < n; ++f) { uint32_t x, y; memcpy(&x, &state[KVM_DOOM_ST_HASH + KVM_DOOM_ST_REC * f], 4); memcpy(&y, &state[KVM_DOOM_ST_HASH + KVM_DOOM_ST_REC * f + 1], 4); fprintf(js, "%s[%u, %u]", f ? ", " : "", x, y); }
    fprintf(js, "], \"states\": [");
    for (int f = 0; f < n; ++f) { fprintf(js, "%s[", f ? ", " : ""); for (int i = 0; i < 14; ++i) { int v; memcpy(&v, &state[KVM_DOOM_ST_HASH + KVM_DOOM_ST_REC * f + 2 + i], 4); fprintf(js, "%s%d", i ? ", " : "", v); } fprintf(js, "]"); }
    fprintf(js, "], \"state_names\": [");
    for (int i = 0; i < 14; ++i) fprintf(js, "%s\"%s\"", i ? ", " : "", names[i]);
    fprintf(js, "]}\n"); fclose(js);
    if (o_.dump_frames) { FILE* fr = fopen((dir + "/frames.bin").c_str(), "wb"); for (int f = 0; f < n; ++f) { std::vector<float> fb = ex_->read_buffer(d.fb[f]); fwrite(fb.data(), 1, KVM_DOOM_FB_BYTES, fr); } fclose(fr); }
  }
#endif
  // ---- metrics
  const double period = 1e6 / o_.fps;
  std::vector<double> flat, late, scall; int miss = 0; for (auto& e : doom_ev_) { flat.push_back(e.done_us - e.sub_us); scall.push_back(e.ret_us - e.sub_us); const double over = e.done_us - (e.rel_us + period); late.push_back(over); miss += over > 0; }
  std::vector<double> slat; for (auto& e : serve_ev_) slat.push_back(e.done_us - e.arr_us);
  long long steps = train_ev_.size();
  const double span_s = t_end / 1e6;
  FILE* mj = fopen((o_.out + "/summary.json").c_str(), "w");
  fprintf(mj, "{\"tag\": \"%s\", \"device\": \"%s\", \"sms\": %d, \"grid\": %d, \"workers\": %d, \"sched\": \"%s\", \"ctl\": \"%s\", \"duration_s\": %.3f, \"kernel_ms\": %.3f,\n", o_.tag.c_str(), ex_->device_name().c_str(), ex_->sms(), r.grid, W, o_.sched.c_str(), o_.ctl.c_str(), span_s, r.kernel_ms);
  fprintf(mj, " \"doom\": {\"instances\": %d, \"frames\": %zu, \"deadline_misses\": %d, \"frame_time_us\": {\"p50\": %.0f, \"p99\": %.0f, \"p999\": %.0f, \"max\": %.0f}, \"lateness_us\": {\"p99\": %.0f, \"p999\": %.0f, \"max\": %.0f}, \"submit_call_us\": {\"p50\": %.0f, \"p99\": %.0f, \"max\": %.0f}, \"fps\": %.2f},\n",
          o_.doom_instances, doom_ev_.size(), miss, pctl(flat, .5), pctl(flat, .99), pctl(flat, .999), pctl(flat, 1), pctl(late, .99), pctl(late, .999), pctl(late, 1), pctl(scall, .5), pctl(scall, .99), pctl(scall, 1), doom_ev_.empty() ? 0.0 : doom_ev_.size() / span_s / o_.doom_instances);
  fprintf(mj, " \"serve\": {\"arrivals\": %d, \"completed\": %zu, \"dropped\": %d, \"latency_us\": {\"p50\": %.0f, \"p99\": %.0f, \"p999\": %.0f, \"max\": %.0f}, \"tokens_consistent\": %d},\n",
          serve_arrivals_, serve_ev_.size(), serve_dropped_, pctl(slat, .5), pctl(slat, .99), pctl(slat, .999), pctl(slat, 1), serve_ok);
  fprintf(mj, " \"train\": {\"runs\": %d, \"steps\": %lld, \"steps_per_s\": %.3f},\n", runs, steps, steps / span_s);
  fprintf(mj, " \"sweep\": {\"gb\": %.2f, \"gbytes_moved\": %.2f, \"gb_per_s\": %.1f},\n", o_.sweep_gb, sweep_bytes_ / 1e9, sweep_bytes_ / 1e9 / span_s);
  fprintf(mj, " \"tasks\": {\"admitted\": %d, \"done\": %d}, \"doom_frames_done\": %d}\n", r.admitted, r.done, doom_done);
  fclose(mj);
  KLOG(2, "doom: %zu frames, %d past their deadline, frame time p50/p99/max %.1f/%.1f/%.1f ms | serve: %zu done, p50/p99 %.1f/%.1f ms, %d dropped | train: %lld steps (%.2f/s)",
       doom_ev_.size(), miss, pctl(flat, .5) / 1e3, pctl(flat, .99) / 1e3, pctl(flat, 1) / 1e3, serve_ev_.size(), pctl(slat, .5) / 1e3, pctl(slat, .99) / 1e3, serve_dropped_, steps, steps / span_s);
  return 0;
}
}  // namespace adp

int main(int argc, char** argv) {
  adp::Opt o; o.cfg.stack_bytes = 16384; o.cfg.live_cap = 1 << 16; o.cfg.max_tasks = 1 << 24; o.cfg.ring_words = 1 << 22; o.cfg.timeout_us = 120ull * 1000000;
  o.cfg.sched.on = true; o.cfg.sched.mode = KVM_SCHED_DYNAMIC; o.cfg.sched.num_classes = 3;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto sv = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    auto dv = [&]() { return atof(sv().c_str()); }; auto iv = [&]() { return atoi(sv().c_str()); };
    if (a == "--out") o.out = sv(); else if (a == "--wad") o.wad = sv(); else if (a == "--tag") o.tag = sv();
    else if (a == "--ctl") o.ctl = sv(); else if (a == "--duration-s") o.duration_s = dv(); else if (a == "--fps") o.fps = dv();
    else if (a == "--doom-instances") o.doom_instances = iv(); else if (a == "--train-runs") o.train_runs = iv(); else if (a == "--serve-slots") o.serve_slots = iv();
    else if (a == "--serve-rate") o.serve_rate = dv();
    else if (a == "--burst") { o.burst0 = dv(); o.burst1 = dv(); o.burst_rate = dv(); }            // --burst T0 T1 RATE
    else if (a == "--pause-train") { o.pause0 = dv(); o.pause1 = dv(); }                          // --pause-train T0 T1
    else if (a == "--scene") { o.scene0 = dv(); o.scene1 = dv(); o.scene_extra_us = iv(); }       // --scene T0 T1 EXTRA_US (spin added to each frame)
    else if (a == "--ctl-period-us") o.ctl_period_us = iv(); else if (a == "--target-wait-us") o.target_wait_us = iv();
    else if (a == "--floor-pct") o.floor_pct = iv(); else if (a == "--doom-min-hi") o.doom_min_hi = iv(); else if (a == "--serve-min-hi") o.serve_min_hi = iv();
    else if (a == "--release-periods") o.release_periods = iv(); else if (a == "--slack") o.slack = iv(); else if (a == "--sample-us") o.sample_us = iv(); else if (a == "--seed") o.seed = iv();
    else if (a == "--sweep-gb") o.sweep_gb = dv(); else if (a == "--sweep-chunk-k") o.sweep_chunk_k = iv(); else if (a == "--sweep-inflight") o.sweep_inflight = iv();
    else if (a == "--gate-serve") o.gate_serve = atoll(sv().c_str()); else if (a == "--gate-train") o.gate_train = atoll(sv().c_str());
    else if (a == "--log") o.log_level = iv();
    else if (a == "--live-cap") o.cfg.live_cap = iv(); else if (a == "--ring-words") o.cfg.ring_words = iv();
    else if (a == "--gate-tune") o.gate_tune = true;
    else if (a == "--arbiter-assist") o.arbiter_assist = true;
    else if (a == "--no-doom") o.no_doom = true; else if (a == "--no-serve") o.no_serve = true; else if (a == "--no-train") o.no_train = true; else if (a == "--no-dump-frames") o.dump_frames = false;
    else if (a == "--sched") {
      o.sched = sv();
      if (o.sched == "off") o.cfg.sched.on = false;
      else { o.cfg.sched.on = true; o.cfg.sched.mode = o.sched == "global" ? KVM_SCHED_GLOBAL : o.sched == "static" ? KVM_SCHED_STATIC : KVM_SCHED_DYNAMIC; }
    }
    else if (int r = kvm_cli_sched_option(o.cfg, i, argc, argv)) { if (r < 0) { fprintf(stderr, "missing value for %s\n", a.c_str()); return 2; } }
    else if (a[0] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    else o.scn = a;
  }
  if (o.cfg.sched.p.flags & KVM_SF_ARBITER) {   // the arbiter throttles bandwidth-bound tasks while a PROTECTED class waits longer than its target
    for (int c : {adp::CLS_DOOM, adp::CLS_SERVE}) { o.cfg.sched.p.cls[c].protect |= KVM_CF_PROTECT; o.cfg.sched.p.cls[c].target_wait_us = c == adp::CLS_DOOM ? 500 : o.target_wait_us; }
  }
  if (o.scn.empty() || o.out.empty()) { fprintf(stderr, "usage: adaptive_run SCENARIO_DIR --out DIR --wad doom1.wad [options]\n"); return 2; }
  g_kvm_log = o.log_level;
  try { adp::Run r(o); return r.go(); } catch (const std::exception& e) { KLOG(0, "%s", e.what()); return 1; }
}
