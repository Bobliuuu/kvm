#pragma once
// KVM scheduling policy: plain data shared by the host and the device (no CUDA here), and the host-side controller.
// The device side is include/kvm_sched.cuh; the integration points in kvm_exec.cuh are marked "SCHED".
//
// Vocabulary. A "class" is a workload_id (0..KVM_MAX_CLS-1): the scheduler sees nothing about a workload except that id,
// the task's priority and deadline_us, and per-op cost metadata (include/kvm_ops_cost.inc). A "CTA" is one worker block;
// it runs one task at a time and tasks are NOT preemptible, so every share below is enforced at task boundaries.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#define KVM_MAX_CLS    8
#define KVM_UNLIMITED  0x7fffffff
#define KVM_OP_SLOTS   256           // per-op running-time estimates are kept for op ids below this
#define KVM_LAT_BYTES  16384         // a "BW" task that moves less than this is latency-bound

enum KvmSchedMode { KVM_SCHED_GLOBAL = 0, KVM_SCHED_STATIC = 1, KVM_SCHED_DYNAMIC = 2 };
// GLOBAL : one queue for all classes (the original executor). STATIC: a worker serves only the class that owns its SM
// (the "static split" baseline). DYNAMIC: per-class queues, weighted fair sharing, reservations, caps, deadline boost.
enum KvmRes { KVM_RES_COMPUTE = 0, KVM_RES_BW = 1, KVM_RES_LAT = 2, KVM_RES_MIXED = 3 };
// Policy switches (KvmSchedParams.flags), changeable at run time.
#define KVM_SF_GUARANTEE 1   // reservations (min_cta) are enforced; clear = negative test of the starvation check
#define KVM_SF_INHERIT   2   // priority inheritance across version waits; clear = the deadlock the design prevents
#define KVM_SF_ARBITER   4   // global concurrent-BW-task limit, adapted online by the policy block
#define KVM_SF_LATFIRST  8   // a protected class (KVM_CF_PROTECT) with ready work is served before fair-share classes (below reservation/boost, EDF-urgent first); caps still apply
#define KVM_SF_DEFAULT   (KVM_SF_GUARANTEE | KVM_SF_INHERIT | KVM_SF_LATFIRST)
#define KVM_CF_PROTECT 1
#define KVM_CF_OPAFF   2     // a ready task is offered first to the block that last ran a task of the same op (warm code/state), for affinity_us
// Bits the device records in KvmTrace.flags for how a task was started (bit0 is KVM_TRACE_MISSED).
#define KVM_TRACE_RESV    2    // started while its class was below its reservation
#define KVM_TRACE_BOOST   4    // started under priority inheritance
#define KVM_TRACE_URGENT  8    // started by the deadline (EDF) tier

struct KvmClassParams {        // 32 B; written by the host at any time through the mapped control block
  int weight;                  // >= 1: relative share of CTA-time among classes that compete
  int min_cta;                 // reservation: a class with ready work and fewer running CTAs is served first (0 = none)
  int max_cta;                 // concurrency cap; KVM_UNLIMITED = none, 0 = fully throttled (reservations/inheritance override)
  int bw_cta;                  // cap on this class's concurrent bandwidth-bound tasks (KVM_UNLIMITED = none)
  int urgent_us;               // a ready task whose deadline is within this many us is served by EDF first (0 = off)
  int protect;                 // bit0 (KVM_CF_PROTECT): this class's queue wait feeds the arbiter's pressure signal; bit1 (KVM_CF_OPAFF): op affinity
  int target_wait_us;          // queue-wait target of a protected class (arbiter and controller)
  int sm_hi;                   // STATIC mode: this class owns SM ids below this bound (cumulative)
};
struct KvmSchedParams {        // seqlock: writer bumps gen_a, writes, then sets gen_b = gen_a
  volatile int gen_a;
  int flags, bw_limit_init, bw_limit_min, bw_limit_max, sat_pct, window_us;
  int affinity_us;             // a ready task whose last predecessor ran on block B is offered only to B for this long (warm caches); 0 = off
  KvmClassParams cls[KVM_MAX_CLS];
  volatile int gen_b, pad1[7];
};
static_assert(sizeof(KvmSchedParams) == 4 * (8 + 8 * 8 + 8), "KvmSchedParams layout");

struct KvmClassTele {          // 128 B, written by the policy block (about every 20 us). Cumulative counters are lossless for a host that
                               // samples slower than the device window: the host differences them (KvmAdaptive does).
  volatile unsigned long long sm_ns, done, bytes, missed;
  volatile unsigned long long dl_n, near_miss, wait_over, wait_sum_ns;   // dl_n: finished tasks with a deadline; near_miss: of those, finished
                                                                          // with less than 1/4 of their deadline budget left (misses included);
                                                                          // wait_over: tasks that waited longer than the class's target_wait_us in the queue
  volatile unsigned long long bw_ns, bw_bytes;   // running time and (cost-table) bytes of finished bandwidth-bound tasks: bw_bytes / bw_ns = what one such CTA streams, in bytes/ns = GB/s
  volatile int running, ready, win_wait_avg_us, win_wait_max_us, win_n, resv_max_us, pad[6];
};
struct KvmSchedTele {
  volatile unsigned long long ticks;
  volatile int bw_mbps, bw_peak_mbps, bw_limit, boost_mask, params_gen, bw_running, pad[8];
  KvmClassTele cls[KVM_MAX_CLS];
};
static_assert(sizeof(KvmClassTele) == 128 && sizeof(KvmSchedTele) == 64 + 8 * 128, "telemetry layout");

// Device-memory state (allocated by the host, zero-initialized, then the policy mirror is written before launch).
struct KvmSlotInfo { unsigned long long bytes; int res, est_ns, pref, pad; };   // per task slot; pref = block that ran the releasing predecessor (-1 none)
struct alignas(128) KvmClsDev {
  // line 0, policy mirror: written only by the policy block (and the host before launch); read by every worker in one load
  int weight, min_cta, max_cta, bw_cta, urgent_us, protect, target_wait_us, vt_mul;
  int sm_hi, pad0[23];
  // line 1, hot (atomics): kvm_pop reads the whole line with one coalesced load
  alignas(128) int running;                                // (alignas applies per declarator: one per cache line)
  int bw_running, ready, dl_ready;                         // dl_ready: ready tasks that carry a deadline
  unsigned long long vtime;                                // weighted CTA-ns consumed (ns * 2^20 / weight), charged ahead of time
  int pad1[24];
  // statistics (queue-wait distributions and per-op detail come from the trace on the host)
  alignas(128) unsigned long long sm_ns;
  unsigned long long bytes, tasks, missed, resv_starts, resv_wait_max_ns;
  unsigned long long w_wait_sum, w_n, w_wait_max, w_miss;   // window accumulators, drained by the policy block
  unsigned long long dl_n, near_miss, wait_over, wait_sum;  // cumulative, exported as is (KvmClassTele)
  unsigned long long bw_ns, bw_bytes;                       // BW-class tasks only (the arbiter's measured per-class bandwidth)
};

static_assert(offsetof(KvmClsDev, running) == 128 && offsetof(KvmClsDev, sm_ns) == 256, "KvmClsDev lines");
struct alignas(128) KvmSchedDev {
  int flags, bw_limit, boost_mask, affinity_ns, pad0[28];
  alignas(128) int bw_total;                   // concurrently running bandwidth-bound CTAs
  alignas(128) unsigned long long bw_bytes;    // cumulative bytes of finished bandwidth-bound tasks
  alignas(128) unsigned ema_ns[KVM_OP_SLOTS];  // running-time estimate per op id
  int op_blk[KVM_OP_SLOTS];                    // block that last finished a task of this op (-1 none)
  KvmClsDev cls[KVM_MAX_CLS];
};
// Pointers and launch-time constants carried in KvmDevState (kernel parameter space).
struct KvmSchedPtrs {
  KvmSchedDev* dev; KvmSlotInfo* info;
  int* pv;                                     // [var][class] parked-waiter counts
  int* pubcls;                                 // [var] bitmask of classes that publish it
  KvmSchedParams* hp; KvmSchedTele* ht;        // mapped host memory
  int on, mode, nc, pad;
};

// ---------------- host-side launch configuration ----------------
struct KvmSchedCfg {
  bool on = false;                  // false: the original executor (one queue, no accounting, block 1 is a worker)
  int mode = KVM_SCHED_DYNAMIC;
  int num_classes = 2;
  KvmSchedParams p{};
  KvmSchedCfg() {
    p.flags = KVM_SF_DEFAULT; p.bw_limit_init = 1 << 20; p.bw_limit_min = 8; p.bw_limit_max = 1 << 20; p.sat_pct = 90; p.window_us = 200;
    for (auto& c : p.cls) c = {1, 0, KVM_UNLIMITED, KVM_UNLIMITED, 0, 0, 0, KVM_UNLIMITED};
  }
};

// STATIC mode helper: split SM ids among classes in proportion to `shares` (cumulative bounds; the last class takes the rest).
static inline void kvm_static_split(KvmSchedParams& p, const std::vector<double>& shares, int sms) {
  double tot = 0, acc = 0; for (double x : shares) tot += x;
  for (size_t c = 0; c < shares.size() && c < KVM_MAX_CLS; ++c) { acc += shares[c]; p.cls[c].sm_hi = (c + 1 == shares.size()) ? sms : (int)(acc / tot * sms + 0.5); }
}

// ---------------- host controller: rebalancing from telemetry, no relaunch ----------------
// Demand-tracking reservation with hysteresis. For each controlled class, every period:
//   demand = running + ready (CTAs it could use right now), smoothed:  s += (demand - s) * alpha
//   target = clamp(ceil(s) + slack, lo, hi); the reservation moves to the target only outside a +-deadband, rises at once
//   (fast attack: a burst must get CTAs at the next task boundary) and falls by one per `release_periods` calm periods.
//   SLO feedback: if the class's last-window queue wait exceeds target_wait_us the reservation gets +1 on top.
// The other classes need no explicit action: work conservation lends the idle share and the reservation reclaims it.
struct KvmCtlClass { int cls; int lo, hi; };
struct KvmCtlConfig {
  std::vector<KvmCtlClass> classes;
  int period_us = 500; double alpha = 0.3; int slack = 1, deadband = 1, release_periods = 40;
};
struct KvmCtlSample { double t_ms; int cls, running, ready, min_cta, wait_max_us; };
class KvmController {
 public:
  KvmController(KvmSchedParams* hp, const KvmSchedTele* ht, KvmSchedParams base, KvmCtlConfig cfg) : hp_(hp), ht_(ht), p_(base), cfg_(cfg) {
    for (auto& c : cfg_.classes) smooth_[c.cls] = 0;
  }
  ~KvmController() { stop(); }
  void start() {
    stop_.store(false); t0_ = std::chrono::steady_clock::now();
    th_ = std::thread([this] { loop(); });
  }
  void stop() { stop_.store(true); if (th_.joinable()) th_.join(); }
  const std::vector<KvmCtlSample>& log() const { return log_; }
  int changes() const { return changes_; }
  // Seqlock writer, usable without a controller: the policy block picks the new set up within a few us.
  static void write_params(KvmSchedParams* hp, const KvmSchedParams& v) {
    int g = hp->gen_a + 1;
    __atomic_store_n(&hp->gen_a, g, __ATOMIC_RELEASE);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    hp->flags = v.flags; hp->bw_limit_init = v.bw_limit_init; hp->bw_limit_min = v.bw_limit_min; hp->bw_limit_max = v.bw_limit_max;
    hp->sat_pct = v.sat_pct; hp->window_us = v.window_us; hp->affinity_us = v.affinity_us;
    for (int i = 0; i < KVM_MAX_CLS; ++i) hp->cls[i] = v.cls[i];
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(&hp->gen_b, g, __ATOMIC_RELEASE);
  }
 private:
  void loop() {
    std::vector<int> calm(KVM_MAX_CLS, 0);
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(cfg_.period_us));
      bool dirty = false;
      double now_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
      for (auto& c : cfg_.classes) {
        const KvmClassTele& t = ht_->cls[c.cls];
        int running = t.running, ready = t.ready < 0 ? 0 : t.ready, wait = t.win_wait_max_us;
        double& s = smooth_[c.cls];
        s += ((running + ready) - s) * cfg_.alpha;
        int want = (int)std::ceil(s) + cfg_.slack;
        if (p_.cls[c.cls].target_wait_us > 0 && t.win_n > 0 && wait > p_.cls[c.cls].target_wait_us) want = p_.cls[c.cls].min_cta + std::max(1, cfg_.deadband);
        want = std::max(c.lo, std::min(c.hi, want));
        int cur = p_.cls[c.cls].min_cta, nxt = cur;
        if (want >= cur + std::max(1, cfg_.deadband)) { nxt = want; calm[c.cls] = 0; }                                                                       // fast attack
        else if (want < cur - cfg_.deadband) { if (++calm[c.cls] >= cfg_.release_periods) { nxt = cur - 1; calm[c.cls] = 0; } }  // slow release
        else calm[c.cls] = 0;
        if (nxt != cur) { p_.cls[c.cls].min_cta = nxt; dirty = true; ++changes_; }
        log_.push_back({now_ms, c.cls, running, ready, p_.cls[c.cls].min_cta, wait});
      }
      if (dirty) { for (int i = 0; i < KVM_MAX_CLS; ++i) p_.cls[i].bw_cta = hp_->cls[i].bw_cta; write_params(hp_, p_); }   // the bandwidth arbiter owns bw_cta
    }
  }
  KvmSchedParams* hp_; const KvmSchedTele* ht_; KvmSchedParams p_; KvmCtlConfig cfg_;
  double smooth_[KVM_MAX_CLS] = {};
  std::vector<KvmCtlSample> log_; int changes_ = 0;
  std::atomic<bool> stop_{false}; std::thread th_; std::chrono::steady_clock::time_point t0_;
};

// ---------------- host bandwidth arbiter: per-class caps from MEASURED per-class bandwidth ----------------
// The device-side arbiter (KVM_SF_ARBITER) AIMDs one global count of concurrent bandwidth-bound tasks and cannot tell a 6 MB sweep from a
// 20 KB gather. This one works in bytes/s: telemetry gives, per class, the bytes (cost table, so only as good as kvm_ops_cost.inc) and
// the running time of finished BW-class tasks (KvmClassTele.bw_bytes / bw_ns), from which it derives, every period,
//   rate_c   = d(bw_bytes_c) / dt                       what class c streams now                   [GB/s]
//   per_cta  = d(bw_bytes_c) / d(bw_ns_c)               what ONE BW task of class c streams        [GB/s per CTA]
// and the memory system's budget  B = sat * peak  (peak = configured, or the largest total rate seen). Protected classes (KVM_CF_PROTECT)
// are never capped. The others share  max(0, B - sum(protected rate))  in proportion to weight, and a class's cap is its share divided by
// per_cta, written into bw_cta (never below min_cap, never above its current running demand's headroom). `sat` is tightened by
// `tighten` when a protected class's queue wait exceeds its target and relaxed by `relax` per calm period back to `sat_max`, so the
// feed-forward budget is corrected by the latency signal rather than relying on it alone.
struct KvmBwArbConfig {
  double peak_gbs = 0;        // 0 = learn: the largest total rate seen (decays 0.5% per period)
  double sat_max = 0.95, sat_min = 0.35, tighten = 0.85, relax = 0.02;
  int period_us = 500, min_cap = 2, calm_periods = 4;
};
struct KvmBwArbSample { double t_ms, total_gbs, peak_gbs, sat; int cap[KVM_MAX_CLS]; double rate[KVM_MAX_CLS], per_cta[KVM_MAX_CLS]; };
class KvmBwArbiter {
 public:
  KvmBwArbiter(KvmSchedParams* hp, const KvmSchedTele* ht, KvmSchedParams base, int nc, KvmBwArbConfig cfg) : hp_(hp), ht_(ht), p_(base), nc_(nc), cfg_(cfg), sat_(cfg.sat_max) {}
  ~KvmBwArbiter() { stop(); }
  void start() { stop_.store(false); t0_ = std::chrono::steady_clock::now(); th_ = std::thread([this] { loop(); }); }
  void stop() { stop_.store(true); if (th_.joinable()) th_.join(); }
  const std::vector<KvmBwArbSample>& log() const { return log_; }
  int changes() const { return changes_; }
 private:
  void loop() {
    unsigned long long pb[KVM_MAX_CLS] = {}, pn[KVM_MAX_CLS] = {}; auto pt = std::chrono::steady_clock::now(); int calm = 0;
    double per_cta[KVM_MAX_CLS] = {};
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(cfg_.period_us));
      auto now = std::chrono::steady_clock::now(); double dt = std::chrono::duration<double>(now - pt).count(); pt = now;
      KvmBwArbSample sm{}; sm.t_ms = std::chrono::duration<double, std::milli>(now - t0_).count();
      double rate[KVM_MAX_CLS] = {}, total = 0, prot_rate = 0, wsum = 0; bool pressure = false;
      for (int c = 0; c < nc_; ++c) {
        const KvmClassTele& t = ht_->cls[c];
        unsigned long long b = t.bw_bytes, n = t.bw_ns, db = b - pb[c], dn = n - pn[c]; pb[c] = b; pn[c] = n;
        rate[c] = dt > 0 ? (double)db / dt / 1e9 : 0;
        if (dn > 0 && db > 0) per_cta[c] = per_cta[c] > 0 ? 0.7 * per_cta[c] + 0.3 * ((double)db / (double)dn) : (double)db / (double)dn;   // bytes/ns = GB/s
        total += rate[c];
        const bool prot = (p_.cls[c].protect & KVM_CF_PROTECT) != 0;
        if (prot) { prot_rate += rate[c]; if (t.win_n > 0 && p_.cls[c].target_wait_us > 0 && t.win_wait_max_us > p_.cls[c].target_wait_us) pressure = true; }
        else wsum += std::max(p_.cls[c].weight, 1);
        sm.rate[c] = rate[c]; sm.per_cta[c] = per_cta[c];
      }
      peak_ = cfg_.peak_gbs > 0 ? cfg_.peak_gbs : std::max(peak_ * 0.995, total);
      if (pressure) { sat_ = std::max(cfg_.sat_min, sat_ * cfg_.tighten); calm = 0; }
      else if (++calm >= cfg_.calm_periods) { sat_ = std::min(cfg_.sat_max, sat_ + cfg_.relax); calm = 0; }
      const double budget = std::max(0.0, sat_ * peak_ - prot_rate);
      bool dirty = false;
      for (int c = 0; c < nc_; ++c) {
        sm.cap[c] = p_.cls[c].bw_cta;
        if ((p_.cls[c].protect & KVM_CF_PROTECT) || per_cta[c] <= 0 || wsum <= 0) continue;
        const double share = budget * std::max(p_.cls[c].weight, 1) / wsum;
        int want = (int)std::floor(share / per_cta[c]);
        want = std::max(cfg_.min_cap, want);
        // while the whole system is far below its budget, do not cap at all (a cap only exists to hold the total near the budget)
        if (total < 0.6 * sat_ * peak_ && !pressure) want = KVM_UNLIMITED;
        if (want != p_.cls[c].bw_cta) { p_.cls[c].bw_cta = want; dirty = true; ++changes_; }
        sm.cap[c] = p_.cls[c].bw_cta;
      }
      sm.total_gbs = total; sm.peak_gbs = peak_; sm.sat = sat_;
      if (dirty) { for (int i = 0; i < KVM_MAX_CLS; ++i) p_.cls[i].min_cta = hp_->cls[i].min_cta; KvmController::write_params(hp_, p_); }   // the rebalancing controller owns min_cta
      log_.push_back(sm);
    }
  }
  KvmSchedParams* hp_; const KvmSchedTele* ht_; KvmSchedParams p_; int nc_; KvmBwArbConfig cfg_;
  double sat_, peak_ = 0; std::vector<KvmBwArbSample> log_; int changes_ = 0;
  std::atomic<bool> stop_{false}; std::thread th_; std::chrono::steady_clock::time_point t0_;
};

