// KVM GPU executor, host side: bundle building/validation, the admission ring producer, launch, result decoding, and the
// CLI (compiled out with -DKVM_EXEC_NO_MAIN so tests/gpu can include this file).
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude kvm_exec.cu -o kvm_exec
//   KVM_HOST=... ./run_exec_tests.sh        # tests on a GPU box
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "kvm_exec.cuh"

// ---------------- logging (levels 0 error .. 3 debug; everything is shown by default) ----------------
static int g_kvm_log = 3;
static void kvm_log(int lvl, const char* fmt, ...) {
  if (lvl > g_kvm_log) return;
  static const char* nm[] = {"E", "W", "I", "D"};
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[kvm_exec][%s] ", nm[lvl]); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
  va_end(ap);
}
#define KLOG(lvl, ...) kvm_log(lvl, __VA_ARGS__)
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  char m_[512]; snprintf(m_, sizeof m_, "%s:%d: %s -> %s", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); throw std::runtime_error(m_); } } while (0)

static const char* kvm_err_name(int c) {
  static const char* nm[] = {
#define E(n, s) #n,
      KVM_ERRORS(E)
#undef E
  };
  return c >= 0 && c < (int)(sizeof nm / sizeof nm[0]) ? nm[c] : "UNKNOWN";
}
static const char* kvm_err_text(int c) {
  static const char* tx[] = {
#define E(n, s) s,
      KVM_ERRORS(E)
#undef E
  };
  return c >= 0 && c < (int)(sizeof tx / sizeof tx[0]) ? tx[c] : "unknown error code";
}
static int kvm_op_by_name(const std::string& n) {
  for (const KvmOpInfo& o : kKvmOps) if (n == o.name) return o.id;
  return -1;
}

// ---------------- bundles ----------------
struct KvmArg {
  int32_t w; int type;
  static KvmArg buf(int idx) { return {idx, KVM_A_BUF}; }
  static KvmArg i32(int v) { return {v, KVM_A_I32}; }
  static KvmArg f32(float v) { int32_t w; memcpy(&w, &v, 4); return {w, KVM_A_F32}; }
};
struct KvmTaskOpts { int priority = 0, deadline_us = -1, wait_var = -1, wait_ver = 0, pub_var = -1, pub_ver = 0, workload = 0; };

struct KvmBundle {
  std::vector<KvmTaskDesc> tasks;
  std::vector<int32_t> succ, args;
  // "" if well formed, else the first problem. nbufs/nvars < 0 skip those range checks.
  std::string validate(int nbufs, int nvars, int live_cap) const {
    std::ostringstream e;
    const int nt = (int)tasks.size();
    if (nt == 0) return "empty bundle";
    if (live_cap > 0 && nt > live_cap) { e << nt << " tasks > live_cap " << live_cap; return e.str(); }
    std::vector<int> in(nt, 0);
    for (int i = 0; i < nt; ++i) {
      const KvmTaskDesc& t = tasks[i];
      const KvmOpInfo* op = kvm_op(t.op_id);
      if (!op) { e << "task " << i << ": bad op_id " << t.op_id; return e.str(); }
      if (t.args_words != op->arity) { e << "task " << i << ": " << op->name << " takes " << op->arity << " args, got " << t.args_words; return e.str(); }
      if (t.args_off < 0 || t.args_off + t.args_words > (int)args.size()) { e << "task " << i << ": args range"; return e.str(); }
      if (t.succ_off < 0 || t.succ_cnt < 0 || t.succ_off + t.succ_cnt > (int)succ.size()) { e << "task " << i << ": successor range"; return e.str(); }
      if (t.priority < 0 || t.priority >= KVM_NBUCKETS) { e << "task " << i << ": priority " << t.priority << " outside [0," << KVM_NBUCKETS - 1 << "]"; return e.str(); }
      if (t.deadline_us < -1) { e << "task " << i << ": deadline_us"; return e.str(); }
      for (int v : {t.wait_var, t.pub_var}) if (v < -1 || (nvars >= 0 && v >= nvars)) { e << "task " << i << ": version variable " << v << " out of range"; return e.str(); }
      if ((t.wait_var >= 0 && t.wait_ver < 0) || (t.pub_var >= 0 && t.pub_ver < 1)) { e << "task " << i << ": wait_ver >= 0 and pub_ver >= 1 (0 is the initial state)"; return e.str(); }
      for (int j = 0; j < t.args_words; ++j)
        if (((op->types >> (2 * j)) & 3u) == KVM_A_BUF) {
          int b = args[t.args_off + j];
          if (b < 0 || (nbufs >= 0 && b >= nbufs)) { e << "task " << i << ": " << op->name << " arg " << j << " buffer index " << b << " out of range"; return e.str(); }
        }
      for (int k = 0; k < t.succ_cnt; ++k) {
        int s = succ[t.succ_off + k];
        if (s < 0 || s >= nt || s == i) { e << "task " << i << ": successor " << s << " out of range"; return e.str(); }
        ++in[s];
      }
    }
    for (int i = 0; i < nt; ++i) if (in[i] != tasks[i].indegree) { e << "task " << i << ": indegree " << tasks[i].indegree << " != in-edges " << in[i]; return e.str(); }
    std::vector<int> q, rem = in;  // Kahn: any cycle would deadlock the device
    for (int i = 0; i < nt; ++i) if (!rem[i]) q.push_back(i);
    for (size_t h = 0; h < q.size(); ++h) {
      const KvmTaskDesc& t = tasks[q[h]];
      for (int k = 0; k < t.succ_cnt; ++k) if (--rem[succ[t.succ_off + k]] == 0) q.push_back(succ[t.succ_off + k]);
    }
    if ((int)q.size() != nt) return "dependency cycle";
    return "";
  }
  size_t words() const { return (KVM_REC_HDR + tasks.size() * KVM_TASK_WORDS + succ.size() + args.size() + 7) & ~(size_t)7; }
};

class KvmBundleBuilder {
 public:
  int add(const char* op, std::vector<KvmArg> args, KvmTaskOpts o = {}) {
    int id = kvm_op_by_name(op);
    if (id < 0) throw std::invalid_argument(std::string("unknown operator ") + op);
    const KvmOpInfo* info = kvm_op(id);
    if ((int)args.size() != info->arity) throw std::invalid_argument(std::string(op) + ": wrong argument count");
    for (size_t j = 0; j < args.size(); ++j)
      if (args[j].type != (int)((info->types >> (2 * j)) & 3u)) throw std::invalid_argument(std::string(op) + ": argument " + std::to_string(j) + " has the wrong type");
    KvmTaskDesc t{};
    t.op_id = id; t.priority = o.priority; t.deadline_us = o.deadline_us; t.wait_var = o.wait_var; t.wait_ver = o.wait_ver;
    t.pub_var = o.pub_var; t.pub_ver = o.pub_ver; t.workload_id = o.workload; t.args_words = (int)args.size();
    t.args_off = (int)args_.size();
    for (auto& a : args) args_.push_back(a.w);
    tasks_.push_back(t); out_.emplace_back();
    return (int)tasks_.size() - 1;
  }
  void edge(int from, int to) {  // duplicate edges are removed here (the device counts every in-edge)
    if (from < 0 || to < 0 || from >= (int)tasks_.size() || to >= (int)tasks_.size() || from == to) throw std::invalid_argument("bad edge");
    out_[from].insert(to);
  }
  KvmBundle build() const {
    KvmBundle b; b.tasks = tasks_; b.args = args_;
    for (size_t i = 0; i < tasks_.size(); ++i) {
      b.tasks[i].succ_off = (int)b.succ.size();
      b.tasks[i].succ_cnt = (int)out_[i].size();
      for (int s : out_[i]) { b.succ.push_back(s); ++b.tasks[s].indegree; }
    }
    return b;
  }
 private:
  std::vector<KvmTaskDesc> tasks_; std::vector<int32_t> args_; std::vector<std::set<int>> out_;
};

// ---------------- executor ----------------
struct KvmConfig {
  int blocks = 0;               // 0: occupancy-sized (SMs * resident blocks); >= 2 (block 0 admits, the rest work)
  int max_tasks = 1 << 18;      // slot pool for the whole launch (no recycling)
  int live_cap = 1 << 14;       // max admitted-but-unfinished tasks; ready rings hold this many each; power of two
  int succ_cap = 0, args_cap = 0;  // 0: 8 * max_tasks
  int num_vars = 64;
  int ring_words = 1 << 16;     // admission ring, multiple of 8
  int aging_us = 0;             // 0: strict priority; else one priority level per aging_us of waiting
  unsigned long long timeout_us = 60ull * 1000000;
  size_t stack_bytes = 0;       // per-thread stack (cudaLimitStackSize); ops with deep device calls (game ops) need a large one
  KvmSchedCfg sched;            // SCHED: scheduling policy (off by default: the original single-queue executor)
};
enum class KvmSubmit { OK, BACKPRESSURE_TIMEOUT, TOO_BIG, DEVICE_ERROR, CLOSED, INVALID };
static const char* kvm_submit_name(KvmSubmit s) {
  static const char* n[] = {"OK", "BACKPRESSURE_TIMEOUT", "TOO_BIG", "DEVICE_ERROR", "CLOSED", "INVALID"};
  return n[(int)s];
}
struct KvmSchedStats {  // end-of-run copy of the device accounting (SCHED)
  bool valid = false;
  KvmSchedDev dev;
};
struct KvmResult {
  KvmSchedStats sched;
  std::vector<KvmSlotInfo> info;
  int err_code = 0, err_slot = -1, err_a = 0, err_b = 0, slots_used = 0, admitted = 0, done = 0, grid = 0;
  float kernel_ms = 0;
  std::vector<KvmTaskDesc> tasks;
  std::vector<KvmTrace> trace;
  std::vector<int> version;
  bool ok() const { return err_code == 0; }
};
struct KvmStats { int bundles = 0, backpressure_waits = 0, pads = 0; };

class KvmExec {
 public:
  explicit KvmExec(KvmConfig c) : cfg_(c) {
    if (cfg_.succ_cap == 0) cfg_.succ_cap = 8 * cfg_.max_tasks;
    if (cfg_.args_cap == 0) cfg_.args_cap = 8 * cfg_.max_tasks;
    if (cfg_.ring_words % 8 || cfg_.ring_words < 64) throw std::invalid_argument("ring_words must be a multiple of 8 and >= 64");
    if (cfg_.live_cap < 2 || (cfg_.live_cap & (cfg_.live_cap - 1))) throw std::invalid_argument("live_cap must be a power of two");
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    sms_ = p.multiProcessorCount; name_ = p.name;
    if (cfg_.stack_bytes) CK(cudaDeviceSetLimit(cudaLimitStackSize, cfg_.stack_bytes));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kvm_exec_kernel, KVM_THREADS, 0));
    grid_ = cfg_.blocks ? cfg_.blocks : sms_ * occ;
    const KvmSchedCfg& sc = cfg_.sched;
    const int nsys = 1 + (sc.on ? 1 : 0);  // admitter (+ policy block)
    if (sc.on && (sc.num_classes < 1 || sc.num_classes > KVM_MAX_CLS)) throw std::invalid_argument("sched.num_classes must be 1..8");
    if (grid_ < nsys + 1) throw std::invalid_argument("need an admitter" + std::string(sc.on ? ", a policy block" : "") + " and at least one worker block");
    nc_ = sc.on ? sc.num_classes : 1;
    KLOG(2, "device %s, %d SMs, %d resident blocks/SM -> grid %d (1 admitter%s + %d workers)", name_.c_str(), sms_, occ, grid_, sc.on ? " + 1 policy" : "", grid_ - nsys);
    const size_t T = cfg_.max_tasks;
    CK(cudaMalloc(&S_.tasks, T * sizeof(KvmTaskDesc)));
    CK(cudaMalloc(&S_.indeg, T * 4)); CK(cudaMalloc(&S_.state, T * 4)); CK(cudaMalloc(&S_.park_next, T * 4));
    CK(cudaMalloc(&S_.trace, T * sizeof(KvmTrace)));
    CK(cudaMalloc(&S_.succ, (size_t)cfg_.succ_cap * 4)); CK(cudaMalloc(&S_.args, (size_t)cfg_.args_cap * 4));
    CK(cudaMalloc(&S_.version, cfg_.num_vars * 4)); CK(cudaMalloc(&S_.park_head, cfg_.num_vars * 4));
    CK(cudaMalloc(&S_.cells, (size_t)nc_ * KVM_NBUCKETS * cfg_.live_cap * sizeof(KvmCell)));
    CK(cudaMalloc(&S_.qctl, nc_ * KVM_NBUCKETS * sizeof(KvmQCtl)));
    CK(cudaMalloc(&S_.cnt, sizeof(KvmCounters)));
    CK(cudaMemset(S_.tasks, 0, T * sizeof(KvmTaskDesc))); CK(cudaMemset(S_.trace, 0, T * sizeof(KvmTrace)));
    CK(cudaMemset(S_.version, 0, cfg_.num_vars * 4)); CK(cudaMemset(S_.qctl, 0, nc_ * KVM_NBUCKETS * sizeof(KvmQCtl)));
    CK(cudaMemset(S_.cnt, 0, sizeof(KvmCounters)));
    // host-shared control block: ring + doorbell/ring_head/closed on separate lines + error mirror
    CK(cudaHostAlloc((void**)&ctl_, 4096, cudaHostAllocMapped)); memset(ctl_, 0, 4096);
    init_sched();
    CK(cudaHostAlloc((void**)&ring_, (size_t)cfg_.ring_words * 4, cudaHostAllocMapped)); memset(ring_, 0, (size_t)cfg_.ring_words * 4);
    doorbell_ = (unsigned long long*)(ctl_ + 0); ring_head_ = (unsigned long long*)(ctl_ + 128);
    closed_ = (unsigned*)(ctl_ + 256); herr_ = (KvmHostErr*)(ctl_ + 384);
    void* d;
    CK(cudaHostGetDevicePointer(&d, ring_, 0)); S_.ring = (int*)d;
    CK(cudaHostGetDevicePointer(&d, doorbell_, 0)); S_.doorbell = (unsigned long long*)d;
    CK(cudaHostGetDevicePointer(&d, ring_head_, 0)); S_.ring_head = (unsigned long long*)d;
    CK(cudaHostGetDevicePointer(&d, closed_, 0)); S_.closed = (unsigned*)d;
    CK(cudaHostGetDevicePointer(&d, herr_, 0)); S_.herr = (KvmHostErr*)d;
    S_.max_tasks = cfg_.max_tasks; S_.live_cap = cfg_.live_cap; S_.succ_cap = cfg_.succ_cap; S_.args_cap = cfg_.args_cap;
    S_.num_vars = cfg_.num_vars; S_.ring_words = cfg_.ring_words;
    S_.aging_ns = cfg_.aging_us ? (long long)cfg_.aging_us * 1000 : (1ll << 50);
    S_.timeout_ns = cfg_.timeout_us * 1000ull;
    CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CK(cudaEventCreate(&e0_)); CK(cudaEventCreate(&e1_));
  }
  ~KvmExec() {
    if (started_ && !finished_) { try { close(); finish(); } catch (...) {} }
    for (size_t i = 0; i < bufs_.size(); ++i) if (!std::count(views_.begin(), views_.end(), i)) cudaFree(bufs_[i].p);
    for (float* p : arenas_) cudaFree(p);
    for (void* p : {(void*)S_.tasks, (void*)S_.indeg, (void*)S_.state, (void*)S_.park_next, (void*)S_.trace, (void*)S_.succ, (void*)S_.args,
                    (void*)S_.version, (void*)S_.park_head, (void*)S_.cells, (void*)S_.qctl, (void*)S_.cnt, (void*)S_.bufs, (void*)S_.sp.dev, (void*)S_.sp.info, (void*)S_.sp.pv, (void*)S_.sp.pubcls}) cudaFree(p);
    cudaFreeHost(ctl_); cudaFreeHost(ring_);
    cudaStreamDestroy(stream_); cudaEventDestroy(e0_); cudaEventDestroy(e1_);
  }
  KvmExec(const KvmExec&) = delete; KvmExec& operator=(const KvmExec&) = delete;

  int add_buffer(int rows, int cols, const std::vector<float>& init = {}) {
    if (started_) throw std::logic_error("buffers are fixed at start()");
    if (!init.empty() && (long long)init.size() != (long long)rows * cols) throw std::invalid_argument("buffer init size");
    KvmBuf b{}; CK(cudaMalloc(&b.p, (size_t)rows * cols * 4));
    b.rows = rows; b.cols = cols; b.ld = cols; b.dtype = 0;
    if (init.empty()) CK(cudaMemset(b.p, 0, (size_t)rows * cols * 4)); else CK(cudaMemcpy(b.p, init.data(), init.size() * 4, cudaMemcpyHostToDevice));
    bufs_.push_back(b);
    return (int)bufs_.size() - 1;
  }
  void set_version(int var, int v) {
    if (started_) throw std::logic_error("versions are fixed at start()");
    if (var < 0 || var >= cfg_.num_vars) throw std::invalid_argument("version variable out of range");
    CK(cudaMemcpy(S_.version + var, &v, 4, cudaMemcpyHostToDevice));
  }
  // Pooled device memory for many small buffers: arena() then add_view(). Zero-initialised, freed with the executor.
  float* arena(size_t floats) {
    float* p; CK(cudaMalloc(&p, floats * 4)); CK(cudaMemset(p, 0, floats * 4)); arenas_.push_back(p); return p;
  }
  int add_view(float* p, int rows, int cols) {
    if (started_) throw std::logic_error("buffers are fixed at start()");
    KvmBuf b{}; b.p = p; b.rows = rows; b.cols = cols; b.ld = cols; b.dtype = 0; views_.push_back(bufs_.size());
    bufs_.push_back(b); return (int)bufs_.size() - 1;
  }
  int nbufs() const { return (int)bufs_.size(); }
  int grid() const { return grid_; }
  const KvmConfig& config() const { return cfg_; }
  const KvmStats& stats() const { return stats_; }

  // ---- SCHED: runtime policy control through the mapped control block (no relaunch) ----
  KvmSchedParams* sched_params() { return hp_; }                // controllers write with KvmController::write_params
  void set_sched_params(const KvmSchedParams& p) { if (!cfg_.sched.on) throw std::logic_error("scheduler is off"); validate_sched(p); KvmController::write_params(hp_, p); }
  const KvmSchedTele* telemetry() const { return ht_; }
  int num_classes() const { return nc_; }
  int workers() const { return grid_ - 1 - (cfg_.sched.on ? 1 : 0); }
  void static_split(KvmSchedParams& p, const std::vector<double>& shares) const { kvm_static_split(p, shares, sms_); }

  void start() {
    if (started_) throw std::logic_error("already started");
    KvmBuf* d; size_t n = std::max<size_t>(bufs_.size(), 1);
    CK(cudaMalloc(&d, n * sizeof(KvmBuf)));
    if (!bufs_.empty()) CK(cudaMemcpy(d, bufs_.data(), bufs_.size() * sizeof(KvmBuf), cudaMemcpyHostToDevice));
    S_.bufs = d; S_.nbufs = (int)bufs_.size();
    if (cfg_.sched.on) {  // the policy mirror starts as the launch parameters, so workers never see a zero weight
      KvmSchedDev init{}; const KvmSchedParams& p = cfg_.sched.p;
      init.flags = p.flags; init.affinity_ns = p.affinity_us * 1000; init.bw_limit = std::max(p.bw_limit_min, std::min(p.bw_limit_max, p.bw_limit_init));
      for (int c = 0; c < nc_; ++c) {
        KvmClsDev& k = init.cls[c]; const KvmClassParams& q = p.cls[c];
        k.weight = q.weight; k.min_cta = q.min_cta; k.max_cta = q.max_cta; k.bw_cta = q.bw_cta; k.urgent_us = q.urgent_us;
        k.protect = q.protect; k.target_wait_us = q.target_wait_us; k.sm_hi = q.sm_hi; k.vt_mul = (1 << 20) / std::max(q.weight, 1);
      }
      CK(cudaMemcpy(S_.sp.dev, &init, sizeof init, cudaMemcpyHostToDevice));
    }
    kvm_init_kernel<<<64, KVM_THREADS, 0, stream_>>>(S_);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e0_, stream_));
    kvm_exec_kernel<<<grid_, KVM_THREADS, 0, stream_>>>(S_);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1_, stream_));
    started_ = true;
    KLOG(2, "kernel launched: grid %d x %d threads, live_cap %d, max_tasks %d, ring %d words, aging %d us, sched %s", grid_, KVM_THREADS, cfg_.live_cap, cfg_.max_tasks, cfg_.ring_words, cfg_.aging_us,
         !cfg_.sched.on ? "off" : cfg_.sched.mode == KVM_SCHED_GLOBAL ? "global" : cfg_.sched.mode == KVM_SCHED_STATIC ? "static" : "dynamic");
  }

  // Thread-safe producer. Blocks (up to timeout_ms) while the ring is full: that is the backpressure; it never drops.
  KvmSubmit submit(const KvmBundle& b, int timeout_ms = 10000, bool validate = true) {  // validate=false: tests of the device-side checks
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_) throw std::logic_error("submit before start()");
    if (closed_flag_) { KLOG(1, "submit after close()"); return KvmSubmit::CLOSED; }
    std::string why = validate ? b.validate(nbufs(), cfg_.num_vars, cfg_.live_cap) : std::string();
    if (!why.empty()) { KLOG(0, "bundle rejected: %s", why.c_str()); last_error_ = why; return why.find("live_cap") != std::string::npos ? KvmSubmit::TOO_BIG : KvmSubmit::INVALID; }
    const size_t words = b.words();
    if (words > (size_t)cfg_.ring_words) { last_error_ = "bundle record larger than the admission ring"; KLOG(0, "%s (%zu > %d words)", last_error_.c_str(), words, cfg_.ring_words); return KvmSubmit::TOO_BIG; }
    auto t0 = std::chrono::steady_clock::now();
    bool waited = false;
    // A record is contiguous. If it does not fit before the ring end, a pad record fills the tail and is published ON ITS OWN
    // (waiting for pad + record at once could need more than the whole ring); then the record waits for space at lap start.
    for (;;) {
      unsigned long long head = __atomic_load_n(ring_head_, __ATOMIC_ACQUIRE);
      const size_t rem = cfg_.ring_words - wpos_ % cfg_.ring_words, need = words > rem ? rem : words;
      if (cfg_.ring_words - (wpos_ - head) >= need) {
        if (need == words) break;
        write_header(rem, 0, 0, 0, KVM_REC_PAD); wpos_ += rem; ++stats_.pads;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(doorbell_, wpos_, __ATOMIC_RELEASE);
        KLOG(3, "ring wrap: %zu pad words", rem);
        continue;
      }
      if (herr_->code) { last_error_ = "device error"; return KvmSubmit::DEVICE_ERROR; }
      if (!waited) { waited = true; ++stats_.backpressure_waits; KLOG(3, "ring full (%llu/%d words used): backpressure", (unsigned long long)(wpos_ - head), cfg_.ring_words); }
      if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
        last_error_ = "admission ring full for " + std::to_string(timeout_ms) + " ms";
        KLOG(1, "%s", last_error_.c_str()); return KvmSubmit::BACKPRESSURE_TIMEOUT;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    int* r = ring_ + wpos_ % cfg_.ring_words;
    write_header(words, (int)b.tasks.size(), (int)b.succ.size(), (int)b.args.size(), KVM_REC_BUNDLE);
    r += KVM_REC_HDR;
    memcpy(r, b.tasks.data(), b.tasks.size() * sizeof(KvmTaskDesc)); r += b.tasks.size() * KVM_TASK_WORDS;
    if (!b.succ.empty()) memcpy(r, b.succ.data(), b.succ.size() * 4);
    r += b.succ.size();
    if (!b.args.empty()) memcpy(r, b.args.data(), b.args.size() * 4);
    wpos_ += words;
    // Publication: record bytes first (seq_cst fence also orders non-temporal stores), then the doorbell with release.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(doorbell_, wpos_, __ATOMIC_RELEASE);
    ++stats_.bundles;
    return KvmSubmit::OK;
  }
  const std::string& last_error() const { return last_error_; }

  // No more bundles: the kernel exits once everything admitted has finished.
  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_flag_) return;
    closed_flag_ = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(closed_, 1u, __ATOMIC_RELEASE);
  }

  const KvmResult& finish() {
    if (!started_) throw std::logic_error("finish before start()");
    if (finished_) return res_;
    close();
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
    KvmCounters c; CK(cudaMemcpy(&c, S_.cnt, sizeof c, cudaMemcpyDeviceToHost));
    res_.err_code = c.err_code; res_.err_slot = c.err_slot; res_.err_a = c.err_a; res_.err_b = c.err_b;
    res_.slots_used = c.slots_used; res_.admitted = c.admitted; res_.done = c.done; res_.grid = grid_;
    CK(cudaEventElapsedTime(&res_.kernel_ms, e0_, e1_));
    res_.tasks.resize(c.slots_used); res_.trace.resize(c.slots_used); res_.version.resize(cfg_.num_vars);
    if (c.slots_used) {
      CK(cudaMemcpy(res_.tasks.data(), S_.tasks, c.slots_used * sizeof(KvmTaskDesc), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(res_.trace.data(), S_.trace, c.slots_used * sizeof(KvmTrace), cudaMemcpyDeviceToHost));
    }
    CK(cudaMemcpy(res_.version.data(), S_.version, cfg_.num_vars * 4, cudaMemcpyDeviceToHost));
    if (cfg_.sched.on) {
      CK(cudaMemcpy(&res_.sched.dev, S_.sp.dev, sizeof(KvmSchedDev), cudaMemcpyDeviceToHost)); res_.sched.valid = true;
      res_.info.resize(c.slots_used);
      if (c.slots_used) CK(cudaMemcpy(res_.info.data(), S_.sp.info, c.slots_used * sizeof(KvmSlotInfo), cudaMemcpyDeviceToHost));
    }
    finished_ = true;
    if (res_.ok()) KLOG(2, "kernel done: %d/%d tasks in %.3f ms, %d bundles, %d ring wraps, %d backpressure waits", res_.done, res_.admitted, res_.kernel_ms, stats_.bundles, stats_.pads, stats_.backpressure_waits);
    else KLOG(0, "DEVICE ERROR %s (%s): slot %d, a=%d, b=%d; done %d/%d", kvm_err_name(res_.err_code), kvm_err_text(res_.err_code), res_.err_slot, res_.err_a, res_.err_b, res_.done, res_.admitted);
    return res_;
  }

  std::vector<float> read_buffer(int idx) const {
    if (idx < 0 || idx >= nbufs()) throw std::out_of_range("buffer index");
    std::vector<float> v((size_t)bufs_[idx].rows * bufs_[idx].cols);
    CK(cudaMemcpy(v.data(), bufs_[idx].p, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
  }
  int sms() const { return sms_; }
  const std::string& device_name() const { return name_; }

 private:
  void validate_sched(const KvmSchedParams& p) const {
    for (int c = 0; c < nc_; ++c) {
      const KvmClassParams& q = p.cls[c];
      if (q.weight < 1 || q.min_cta < 0 || q.max_cta < 0 || q.bw_cta < 0 || q.urgent_us < 0) throw std::invalid_argument("class " + std::to_string(c) + ": weight >= 1, min/max/bw caps and urgent_us >= 0");
      if (q.max_cta < q.min_cta) throw std::invalid_argument("class " + std::to_string(c) + ": max_cta < min_cta");
    }
    if (p.bw_limit_min < 0 || p.bw_limit_max < p.bw_limit_min) throw std::invalid_argument("bw_limit_min <= bw_limit_max");
  }
  void init_sched() {
    const KvmSchedCfg& sc = cfg_.sched;
    hp_ = (KvmSchedParams*)(ctl_ + 512); ht_ = (KvmSchedTele*)(ctl_ + 1024);
    S_.sp = KvmSchedPtrs{}; S_.sp.nc = nc_; S_.sp.on = sc.on ? 1 : 0; S_.sp.mode = sc.on ? sc.mode : KVM_SCHED_GLOBAL;
    if (!sc.on) return;
    validate_sched(sc.p);
    void* d;
    CK(cudaHostGetDevicePointer(&d, hp_, 0)); S_.sp.hp = (KvmSchedParams*)d;
    CK(cudaHostGetDevicePointer(&d, ht_, 0)); S_.sp.ht = (KvmSchedTele*)d;
    CK(cudaMalloc(&S_.sp.dev, sizeof(KvmSchedDev))); CK(cudaMemset(S_.sp.dev, 0, sizeof(KvmSchedDev)));
    CK(cudaMalloc(&S_.sp.info, (size_t)cfg_.max_tasks * sizeof(KvmSlotInfo))); CK(cudaMemset(S_.sp.info, 0, (size_t)cfg_.max_tasks * sizeof(KvmSlotInfo)));
    CK(cudaMalloc(&S_.sp.pv, (size_t)cfg_.num_vars * KVM_MAX_CLS * 4)); CK(cudaMemset(S_.sp.pv, 0, (size_t)cfg_.num_vars * KVM_MAX_CLS * 4));
    CK(cudaMalloc(&S_.sp.pubcls, (size_t)cfg_.num_vars * 4)); CK(cudaMemset(S_.sp.pubcls, 0, (size_t)cfg_.num_vars * 4));
    KvmController::write_params(hp_, sc.p);   // generation 1: the policy block's first tick re-reads the launch parameters
  }
  void write_header(size_t total, int nt, int ne, int na, int kind) {
    int* h = ring_ + wpos_ % cfg_.ring_words;
    h[0] = KVM_REC_MAGIC; h[1] = (int)total; h[2] = nt; h[3] = ne; h[4] = na; h[5] = stats_.bundles; h[6] = kind; h[7] = 0;
  }
  KvmConfig cfg_;
  KvmDevState S_{};
  std::vector<KvmBuf> bufs_;
  std::vector<size_t> views_;
  std::vector<float*> arenas_;
  char* ctl_ = nullptr; int* ring_ = nullptr;
  unsigned long long *doorbell_ = nullptr, *ring_head_ = nullptr; unsigned* closed_ = nullptr; KvmHostErr* herr_ = nullptr;
  unsigned long long wpos_ = 0;
  cudaStream_t stream_{}; cudaEvent_t e0_{}, e1_{};
  int sms_ = 0, grid_ = 0, nc_ = 1;
  KvmSchedParams* hp_ = nullptr; KvmSchedTele* ht_ = nullptr;
  std::string name_, last_error_;
  std::mutex mu_;
  bool started_ = false, finished_ = false, closed_flag_ = false;
  KvmResult res_;
  KvmStats stats_;
};

// Timeline-ready CSV: times in microseconds since the first admission.
static void kvm_write_trace_csv(const KvmResult& r, const char* path) {
  FILE* f = fopen(path, "w");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  unsigned long long t0 = ~0ull;
  for (auto& t : r.trace) if (t.admit_ns) t0 = std::min(t0, t.admit_ns);
  fprintf(f, "slot,op,workload,priority,deadline_us,admit_us,ready_us,start_us,end_us,queue_wait_us,run_us,smid,block,ver_start,ver_end,missed,exec_count,bytes,res,resv,boost,urgent\n");
  for (size_t i = 0; i < r.trace.size(); ++i) {
    const KvmTrace& t = r.trace[i]; const KvmTaskDesc& d = r.tasks[i];
    auto us = [&](unsigned long long x) { return x ? (double)(x - t0) / 1e3 : -1.0; };
    const KvmSlotInfo si = i < r.info.size() ? r.info[i] : KvmSlotInfo{0, 0, 0};
    fprintf(f, "%zu,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%llu,%d,%d,%d,%d\n", i, kvm_op(d.op_id)->name, d.workload_id, d.priority, d.deadline_us,
            us(t.admit_ns), us(t.ready_ns), us(t.start_ns), us(t.end_ns), t.start_ns ? (double)(t.start_ns - t.ready_ns) / 1e3 : -1.0,
            t.end_ns ? (double)(t.end_ns - t.start_ns) / 1e3 : -1.0, t.smid, t.block, t.ver0, t.ver1, (t.flags & KVM_TRACE_MISSED) ? 1 : 0, t.exec_count,
            si.bytes, si.res, (t.flags & KVM_TRACE_RESV) ? 1 : 0, (t.flags & KVM_TRACE_BOOST) ? 1 : 0, (t.flags & KVM_TRACE_URGENT) ? 1 : 0);
  }
  fclose(f);
  KLOG(2, "trace: %zu tasks -> %s", r.trace.size(), path);
}

// Per-workload telemetry from the trace: tasks, CTA-seconds, queue-wait percentiles, deadline misses, bytes moved and bandwidth.
static void kvm_print_class_report(const KvmResult& r) {
  int maxc = 0; for (auto& d : r.tasks) maxc = std::max(maxc, d.workload_id);
  unsigned long long t0 = ~0ull, t1 = 0; for (auto& t : r.trace) if (t.admit_ns) { t0 = std::min(t0, t.admit_ns); t1 = std::max(t1, t.end_ns); }
  const double span_us = t1 > t0 ? (t1 - t0) / 1e3 : 1.0;
  for (int c = 0; c <= maxc; ++c) {
    std::vector<double> w; double cta_us = 0, bytes = 0; int n = 0, miss = 0;
    for (size_t i = 0; i < r.trace.size(); ++i) if (r.tasks[i].workload_id == c) {
      const KvmTrace& t = r.trace[i]; ++n; w.push_back((t.start_ns - t.ready_ns) / 1e3); cta_us += (t.end_ns - t.start_ns) / 1e3;
      miss += (t.flags & KVM_TRACE_MISSED) != 0; if (i < r.info.size()) bytes += (double)r.info[i].bytes;
    }
    if (!n) continue;
    std::sort(w.begin(), w.end());
    auto q = [&](double p) { return w[std::min(w.size() - 1, (size_t)(p * w.size()))]; };
    KLOG(2, "workload %d: %d tasks, %.3f CTA-s, queue wait p50 %.1f p99 %.1f max %.1f us, %d deadline misses, %.2f GB moved (%.1f GB/s over the run)",
         c, n, cta_us / 1e6, q(0.5), q(0.99), w.back(), miss, bytes / 1e9, bytes / (span_us * 1e3));
  }
}

// ======================= CLI =======================
#ifndef KVM_EXEC_NO_MAIN
// Minimal JSON reader (objects, arrays, numbers, strings, true/false/null): enough for bundle metadata.json.
struct Json {
  enum T { Z, B, N, S, A, O } t = Z;
  double n = 0; std::string s; std::vector<Json> a; std::vector<std::pair<std::string, Json>> o;
  const Json* get(const char* k) const { for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
  double num(const char* k, double dflt) const { auto* j = get(k); return j && j->t == N ? j->n : dflt; }
};
struct JsonParser {
  const std::string& s; size_t i = 0;
  explicit JsonParser(const std::string& str) : s(str) {}
  void ws() { while (i < s.size() && isspace((unsigned char)s[i])) ++i; }
  [[noreturn]] void bad(const char* m) { throw std::runtime_error(std::string("json: ") + m + " at byte " + std::to_string(i)); }
  Json parse() { Json j = value(); ws(); if (i != s.size()) bad("trailing data"); return j; }
  Json value() {
    ws(); if (i >= s.size()) bad("eof");
    Json j; char c = s[i];
    if (c == '{') { j.t = Json::O; ++i; ws(); if (s[i] == '}') { ++i; return j; }
      for (;;) { ws(); std::string k = str(); ws(); if (s[i++] != ':') bad("expected ':'"); j.o.emplace_back(k, value()); ws();
        if (s[i] == ',') { ++i; continue; } if (s[i] == '}') { ++i; return j; } bad("expected ',' or '}'"); } }
    if (c == '[') { j.t = Json::A; ++i; ws(); if (s[i] == ']') { ++i; return j; }
      for (;;) { j.a.push_back(value()); ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == ']') { ++i; return j; } bad("expected ',' or ']'"); } }
    if (c == '"') { j.t = Json::S; j.s = str(); return j; }
    if (!s.compare(i, 4, "true")) { i += 4; j.t = Json::B; j.n = 1; return j; }
    if (!s.compare(i, 5, "false")) { i += 5; j.t = Json::B; return j; }
    if (!s.compare(i, 4, "null")) { i += 4; return j; }
    char* e; j.n = strtod(s.c_str() + i, &e); if (e == s.c_str() + i) bad("bad value");
    i = e - s.c_str(); j.t = Json::N; return j;
  }
  std::string str() {
    if (s[i] != '"') bad("expected string");
    std::string r; ++i;
    while (i < s.size() && s[i] != '"') { if (s[i] == '\\') { ++i; char c = s[i]; r += c == 'n' ? '\n' : c == 't' ? '\t' : c; } else r += s[i]; ++i; }
    if (i >= s.size()) bad("unterminated string");
    ++i; return r;
  }
};
static std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + p);
  std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
template <class T> static std::vector<T> read_bin(const std::string& p, size_t count) {
  std::string raw = read_file(p);
  if (raw.size() != count * sizeof(T)) throw std::runtime_error(p + ": expected " + std::to_string(count * sizeof(T)) + " bytes, got " + std::to_string(raw.size()));
  std::vector<T> v(count); if (count) memcpy(v.data(), raw.data(), raw.size()); return v;
}

static void usage() {
  fprintf(stderr, "usage: kvm_exec [options] bundle_dir...\n"
    "  bundle_dir: tasks.bin successors.bin args.bin metadata.json (ABI v2)\n"
    "  --blocks N --live-cap N --max-tasks N --ring-words N --aging-us N --timeout-s N\n"
    "  --sched MODE        global|static|dynamic: per-workload scheduling (workload_id = class, up to 8)  --classes N\n"
    "  --weights a,b,..    --min a,b,.. (reserved CTAs)  --max a,b,.. (CTA caps)  --bw-cta a,b,..  --urgent-us a,b,..\n"
    "  --shares a,b (static SM shares)  --arbiter  --bw-limit N   adaptive concurrent bandwidth-bound CTA limit   --affinity-us N   --no-guarantee  --no-inherit\n"
    "  --golden            compare with <bundle_dir>/golden.json (single bundle): final buffers, versions, observed versions\n"
    "  --trace-csv FILE   per-task timeline  --dump-buffers   print buffers   --quiet   errors only   --check-ops   print registry\n");
}

int main(int argc, char** argv) {
  KvmConfig cfg; std::vector<std::string> dirs; const char* csv = nullptr; bool dump = false, golden = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> long long { if (i + 1 >= argc) { usage(); exit(2); } return atoll(argv[++i]); };
    if (a == "--blocks") cfg.blocks = (int)val();
    else if (a == "--live-cap") cfg.live_cap = (int)val();
    else if (a == "--max-tasks") cfg.max_tasks = (int)val();
    else if (a == "--ring-words") cfg.ring_words = (int)val();
    else if (a == "--aging-us") cfg.aging_us = (int)val();
    else if (a == "--timeout-s") cfg.timeout_us = (unsigned long long)val() * 1000000ull;
    else if (a == "--sched") { std::string m = i + 1 < argc ? argv[++i] : ""; cfg.sched.on = true; cfg.sched.mode = m == "global" ? KVM_SCHED_GLOBAL : m == "static" ? KVM_SCHED_STATIC : KVM_SCHED_DYNAMIC; }
    else if (a == "--classes") cfg.sched.num_classes = (int)val();
    else if (a == "--weights" || a == "--min" || a == "--max" || a == "--bw-cta" || a == "--urgent-us") {
      if (i + 1 >= argc) { usage(); return 2; }
      std::stringstream ss(argv[++i]); std::string tok; int c = 0;
      while (std::getline(ss, tok, ',') && c < KVM_MAX_CLS) { KvmClassParams& q = cfg.sched.p.cls[c++]; int v = atoi(tok.c_str());
        if (a == "--weights") q.weight = v; else if (a == "--min") q.min_cta = v; else if (a == "--max") q.max_cta = v; else if (a == "--bw-cta") q.bw_cta = v; else q.urgent_us = v; }
    }
    else if (a == "--shares") {   // STATIC mode: SM share per class, e.g. 1,3
      if (i + 1 >= argc) { usage(); return 2; }
      std::stringstream ss(argv[++i]); std::string tok; std::vector<double> sh; while (std::getline(ss, tok, ',')) sh.push_back(atof(tok.c_str()));
      cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); kvm_static_split(cfg.sched.p, sh, pr.multiProcessorCount);
    }
    else if (a == "--arbiter") cfg.sched.p.flags |= KVM_SF_ARBITER;
    else if (a == "--bw-limit") cfg.sched.p.bw_limit_init = (int)val();
    else if (a == "--affinity-us") cfg.sched.p.affinity_us = (int)val();
    else if (a == "--no-guarantee") cfg.sched.p.flags &= ~KVM_SF_GUARANTEE;
    else if (a == "--no-inherit") cfg.sched.p.flags &= ~KVM_SF_INHERIT;
    else if (a == "--trace-csv") { if (i + 1 >= argc) { usage(); return 2; } csv = argv[++i]; }
    else if (a == "--dump-buffers") dump = true;
    else if (a == "--golden") golden = true;
    else if (a == "--quiet") g_kvm_log = 0;
    else if (a == "--check-ops") { for (auto& o : kKvmOps) printf("%2d %-9s arity %d  (%s)\n", o.id, o.name, o.arity, o.sig); return 0; }
    else if (a[0] == '-') { usage(); return 2; }
    else dirs.push_back(a);
  }
  if (dirs.empty()) { usage(); return 2; }
  try {
    KvmExec ex(cfg);
    std::vector<KvmBundle> bundles;
    std::vector<std::pair<int, int>> shapes;
    std::vector<std::string> buf_names, var_names; std::map<int, std::string> var_by_id;
    Json meta0;
    for (auto& dir : dirs) {
      Json m = JsonParser(read_file(dir + "/metadata.json")).parse();
      if ((int)m.num("abi_version", -1) != KVM_ABI_VERSION) throw std::runtime_error(dir + ": abi_version " + std::to_string((int)m.num("abi_version", -1)) + " != " + std::to_string(KVM_ABI_VERSION));
      int nt = (int)m.num("task_count", -1), ne = (int)m.num("edge_count", -1), nw = (int)m.num("args_words", -1);
      if (nt < 1 || ne < 0 || nw < 0) throw std::runtime_error(dir + ": task_count/edge_count/args_words missing");
      KvmBundle b;
      b.tasks = read_bin<KvmTaskDesc>(dir + "/tasks.bin", nt);
      b.succ = read_bin<int32_t>(dir + "/successors.bin", ne);
      b.args = read_bin<int32_t>(dir + "/args.bin", nw);
      if (dirs.size() == 1) meta0 = m;
      if (auto* bs = m.get("buffers")) {
        for (auto& jb : bs->a) {
          if (const Json* nm = jb.get("name")) buf_names.push_back(nm->s);
          int idx = (int)jb.num("index", -1), rows = (int)jb.num("rows", 0), cols = (int)jb.num("cols", 0);
          if (idx < 0 || rows < 1 || cols < 1) throw std::runtime_error(dir + ": bad buffer entry");
          if (idx < (int)shapes.size()) { if (shapes[idx] != std::make_pair(rows, cols)) throw std::runtime_error(dir + ": buffer " + std::to_string(idx) + " shape differs from an earlier bundle"); continue; }
          if (idx != (int)shapes.size()) throw std::runtime_error(dir + ": buffer indices must be dense and ascending");
          std::vector<float> init;
          if (auto* in = jb.get("init")) {
            const Json* kind = in->get("kind");
            if (kind && kind->s == "fill") init.assign((size_t)rows * cols, (float)in->num("value", 0));
            else if (kind && kind->s == "values") { for (auto& v : in->get("values")->a) init.push_back((float)v.n); }
            else throw std::runtime_error(dir + ": unknown buffer init kind");
          }
          if (ex.add_buffer(rows, cols, init) != idx) throw std::logic_error("buffer index mismatch");
          shapes.push_back({rows, cols});
        }
      }
      if (auto* vs = m.get("variables")) for (auto& jv : vs->a) ex.set_version((int)jv.num("id", -1), (int)jv.num("base_version", 0));
      KLOG(2, "%s: %d tasks, %d edges, %d arg words", dir.c_str(), nt, ne, nw);
      bundles.push_back(std::move(b));
    }
    ex.start();
    for (size_t i = 0; i < bundles.size(); ++i) {
      KvmSubmit st = ex.submit(bundles[i]);
      if (st != KvmSubmit::OK) { KLOG(0, "bundle %zu: %s (%s)", i, kvm_submit_name(st), ex.last_error().c_str()); ex.close(); ex.finish(); return 1; }
    }
    const KvmResult& r = ex.finish();
    if (r.ok()) kvm_print_class_report(r);
    if (csv) kvm_write_trace_csv(r, csv);
    int golden_bad = 0;
    if (golden) {
      if (dirs.size() != 1) throw std::runtime_error("--golden needs exactly one bundle directory");
      Json g = JsonParser(read_file(dirs[0] + "/golden.json")).parse();
      if (!r.ok()) { KLOG(0, "golden: skipped, the run failed"); return 1; }
      int nb = 0, exact = 0; float worst = 0;
      if (const Json* gb = g.get("buffers")) for (auto& kv : gb->o) {
        auto it = std::find(buf_names.begin(), buf_names.end(), kv.first);
        if (it == buf_names.end()) { KLOG(0, "golden: unknown buffer %s", kv.first.c_str()); ++golden_bad; continue; }
        std::vector<float> got = ex.read_buffer((int)(it - buf_names.begin()));
        if (got.size() != kv.second.a.size()) { KLOG(0, "golden: buffer %s has %zu values, golden %zu", kv.first.c_str(), got.size(), kv.second.a.size()); ++golden_bad; continue; }
        for (size_t i = 0; i < got.size(); ++i) {
          float want = (float)kv.second.a[i].n; ++nb;
          if (memcmp(&got[i], &want, 4) == 0) { ++exact; continue; }
          float rel = std::fabs(got[i] - want) / std::max(1e-30f, std::fabs(want)); worst = std::max(worst, rel);
          if (rel > 1e-6f) { KLOG(0, "golden: %s[%zu] = %.9g, golden %.9g", kv.first.c_str(), i, got[i], want); ++golden_bad; }
        }
      }
      KLOG(2, "golden: %d buffer values compared, %d bitwise equal, worst relative difference %.3g", nb, exact, worst);
      std::map<std::string, int> var_id;
      if (const Json* vs = meta0.get("variables")) for (auto& jv : vs->a) var_id[jv.get("name")->s] = (int)jv.num("id", -1);
      if (const Json* gv = g.get("versions")) for (auto& kv : gv->o) {
        int want = (int)kv.second.n, got = var_id.count(kv.first) ? r.version[var_id[kv.first]] : -1;
        if (got != want) { KLOG(0, "golden: version %s = %d, golden %d", kv.first.c_str(), got, want); ++golden_bad; }
      }
      int nreads = 0;
      if (const Json* rs = meta0.get("reads")) for (auto& jr : rs->a) {
        const Json* gate = jr.get("gate");
        int slot = (int)jr.num("task", -1); if (gate && gate->t == Json::N) slot = (int)gate->n;  // the task carrying the wait
        const Json* pol = jr.get("policy"); int need = (int)jr.num("version", 0);
        const KvmTrace& t = r.trace[slot]; ++nreads;
        bool ok = t.ver0 >= need;
        if (pol && pol->s == "exact") ok = ok && t.ver0 == need && t.ver1 == need;
        KLOG(3, "golden: read %s (%s v%d): observed ver_start %d ver_end %d -> %s", jr.get("name")->s.c_str(), pol->s.c_str(), need, t.ver0, t.ver1, ok ? "ok" : "VIOLATION");
        golden_bad += !ok;
      }
      KLOG(2, "golden: %d reads checked, final versions compared; %d mismatches", nreads, golden_bad);
    }
    if (dump) for (int i = 0; i < ex.nbufs(); ++i) { auto v = ex.read_buffer(i); printf("buffer %d:", i); for (float x : v) printf(" %.9g", x); printf("\n"); }
    bool pass = r.ok() && golden_bad == 0;
    printf("%s: %d/%d tasks, %.3f ms%s\n", pass ? "OK" : "FAIL", r.done, r.admitted, r.kernel_ms, golden ? (golden_bad ? ", golden MISMATCH" : ", golden match") : "");
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    KLOG(0, "%s", e.what());
    return 1;
  }
}
#endif
