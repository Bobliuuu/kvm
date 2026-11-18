#pragma once
// Magni VM scheduling policy: plain data shared by the host and the device (no CUDA here), and the host-side controller.
// The device side is include/mvm_sched.cuh; the integration points in mvm_exec.cuh are marked "SCHED".
//
// Vocabulary. A "class" is a workload_id (0..MVM_MAX_CLS-1): the scheduler sees nothing about a workload except that id,
// the task's priority and deadline_us, and per-op cost metadata (include/mvm_ops_cost.inc). A "CTA" is one worker block;
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

#define MVM_MAX_CLS    8
#define MVM_UNLIMITED  0x7fffffff
#define MVM_OP_SLOTS   256           // per-op running-time estimates are kept for op ids below this
#define MVM_LAT_BYTES  16384         // a "BW" task that moves less than this is latency-bound

enum MvmSchedMode { MVM_SCHED_GLOBAL = 0, MVM_SCHED_STATIC = 1, MVM_SCHED_DYNAMIC = 2 };
// GLOBAL : one queue for all classes (the original executor). STATIC: a worker serves only the class that owns its SM
// (the "static split" baseline). DYNAMIC: per-class queues, weighted fair sharing, reservations, caps, deadline boost.
enum MvmRes { MVM_RES_COMPUTE = 0, MVM_RES_BW = 1, MVM_RES_LAT = 2, MVM_RES_MIXED = 3 };
// Policy switches (MvmSchedParams.flags), changeable at run time.
#define MVM_SF_GUARANTEE 1   // reservations (min_cta) are enforced; clear = negative test of the starvation check
#define MVM_SF_INHERIT   2   // priority inheritance across version waits; clear = the deadlock the design prevents
#define MVM_SF_ARBITER   4   // global concurrent-BW-task limit, adapted online by the policy block
#define MVM_SF_DEFAULT   (MVM_SF_GUARANTEE | MVM_SF_INHERIT)
#define MVM_CF_PROTECT 1
#define MVM_CF_OPAFF   2     // a ready task is offered first to the block that last ran a task of the same op (warm code/state), for affinity_us
// Bits the device records in MvmTrace.flags for how a task was started (bit0 is MVM_TRACE_MISSED).
#define MVM_TRACE_RESV    2    // started while its class was below its reservation
#define MVM_TRACE_BOOST   4    // started under priority inheritance
#define MVM_TRACE_URGENT  8    // started by the deadline (EDF) tier
#define MVM_HOW_BW        0x100  // mvm_pop only: the claimed task is bandwidth-bound (its BW counters were incremented); never stored in the trace
#define MVM_URGENT_ALL    (1 << 30)  // MvmClassParams.urgent_us: every deadline task of the class is in the EDF tier (pure EDF, no window)

struct MvmClassParams {        // 32 B; written by the host at any time through the mapped control block
  int weight;                  // >= 1: relative share of CTA-time among classes that compete
  int min_cta;                 // reservation: a class with ready work and fewer running CTAs is served first (0 = none)
  int max_cta;                 // concurrency cap; MVM_UNLIMITED = none, 0 = fully throttled (reservations/inheritance override)
  int bw_cta;                  // cap on this class's concurrent bandwidth-bound tasks (MVM_UNLIMITED = none)
  int urgent_us;               // a ready task whose deadline is within this many us is served by EDF first (0 = off)
  int protect;                 // bit0 (MVM_CF_PROTECT): this class's queue wait feeds the arbiter's pressure signal; bit1 (MVM_CF_OPAFF): op affinity
  int target_wait_us;          // queue-wait target of a protected class (arbiter and controller)
  int sm_hi;                   // STATIC mode: this class owns SM ids below this bound (cumulative)
};
struct MvmSchedParams {        // seqlock: writer bumps gen_a, writes, then sets gen_b = gen_a
  volatile int gen_a;
  int flags, bw_limit_init, bw_limit_min, bw_limit_max, sat_pct, window_us;
  int affinity_us;             // a ready task whose last predecessor ran on block B is offered only to B for this long (warm caches); 0 = off
  MvmClassParams cls[MVM_MAX_CLS];
  volatile int gen_b, pad1[7];
};
static_assert(sizeof(MvmSchedParams) == 4 * (8 + 8 * 8 + 8), "MvmSchedParams layout");

struct MvmClassTele {          // 128 B, written by the policy block (about every 20 us). Cumulative counters are lossless for a host that
                               // samples slower than the device window: the host differences them (MvmAdaptive does).
  volatile unsigned long long sm_ns, done, bytes, missed;
  volatile unsigned long long dl_n, near_miss, wait_over, wait_sum_ns;   // dl_n: finished tasks with a deadline; near_miss: of those, finished
                                                                          // with less than 1/4 of their deadline budget left (misses included);
                                                                          // wait_over: tasks that waited longer than the class's target_wait_us in the queue
  volatile int running, ready, win_wait_avg_us, win_wait_max_us, win_n, resv_max_us, pad[10];
};
struct MvmSchedTele {
  volatile unsigned long long ticks;
  volatile int bw_mbps, bw_peak_mbps, bw_limit, boost_mask, params_gen, bw_running, pad[8];
  MvmClassTele cls[MVM_MAX_CLS];
};
static_assert(sizeof(MvmClassTele) == 128 && sizeof(MvmSchedTele) == 64 + 8 * 128, "telemetry layout");

// Device-memory state (allocated by the host, zero-initialized, then the policy mirror is written before launch).
struct MvmSlotInfo { unsigned long long bytes; int res, est_ns, pref, pad; };   // per task slot; pref = block that ran the releasing predecessor (-1 none)
struct alignas(128) MvmClsDev {
  // line 0, policy mirror: written only by the policy block (and the host before launch); read by every worker in one load
  int weight, min_cta, max_cta, bw_cta, urgent_us, protect, target_wait_us, vt_mul;
  int sm_hi, pad0[23];
  // line 1, hot (atomics): mvm_pop reads the whole line with one coalesced load
  alignas(128) int running;                                // (alignas applies per declarator: one per cache line)
  int bw_running, ready, dl_ready;                         // dl_ready: ready tasks that carry a deadline
  unsigned long long vtime;                                // weighted CTA-ns consumed (ns * 2^20 / weight), charged ahead of time
  int pad1[24];
  // statistics (queue-wait distributions and per-op detail come from the trace on the host)
  alignas(128) unsigned long long sm_ns;
  unsigned long long bytes, tasks, missed, resv_starts, resv_wait_max_ns;
  unsigned long long w_wait_sum, w_n, w_wait_max, w_miss;   // window accumulators, drained by the policy block
  unsigned long long dl_n, near_miss, wait_over, wait_sum;  // cumulative, exported as is (MvmClassTele)
};

static_assert(offsetof(MvmClsDev, running) == 128 && offsetof(MvmClsDev, sm_ns) == 256, "MvmClsDev lines");
struct alignas(128) MvmSchedDev {
  int flags, bw_limit, boost_mask, affinity_ns, pad0[28];
  alignas(128) int bw_total;                   // concurrently running bandwidth-bound CTAs
  alignas(128) unsigned long long bw_bytes;    // cumulative bytes of finished bandwidth-bound tasks
  alignas(128) unsigned ema_ns[MVM_OP_SLOTS];  // running-time estimate per op id
  int op_blk[MVM_OP_SLOTS];                    // block that last finished a task of this op (-1 none)
  MvmClsDev cls[MVM_MAX_CLS];
};
// Pointers and launch-time constants carried in MvmDevState (kernel parameter space).
struct MvmSchedPtrs {
  MvmSchedDev* dev; MvmSlotInfo* info;
  int* pv;                                     // [var][class] parked-waiter counts
  int* pubcls;                                 // [var] bitmask of classes that publish it
  MvmSchedParams* hp; MvmSchedTele* ht;        // mapped host memory
  int on, mode, nc, pad;
};

// ---------------- host-side launch configuration ----------------
struct MvmSchedCfg {
  bool on = false;                  // false: the original executor (one queue, no accounting, block 1 is a worker)
  int mode = MVM_SCHED_DYNAMIC;
  int num_classes = 2;
  MvmSchedParams p{};
  MvmSchedCfg() {
    p.flags = MVM_SF_DEFAULT; p.bw_limit_init = 1 << 20; p.bw_limit_min = 8; p.bw_limit_max = 1 << 20; p.sat_pct = 90; p.window_us = 200;
    for (auto& c : p.cls) c = {1, 0, MVM_UNLIMITED, MVM_UNLIMITED, 0, 0, 0, MVM_UNLIMITED};
  }
};

// STATIC mode helper: split SM ids among classes in proportion to `shares` (cumulative bounds; the last class takes the rest).
static inline void mvm_static_split(MvmSchedParams& p, const std::vector<double>& shares, int sms) {
  double tot = 0, acc = 0; for (double x : shares) tot += x;
  for (size_t c = 0; c < shares.size() && c < MVM_MAX_CLS; ++c) { acc += shares[c]; p.cls[c].sm_hi = (c + 1 == shares.size()) ? sms : (int)(acc / tot * sms + 0.5); }
}

// ---------------- host controller: rebalancing from telemetry, no relaunch ----------------
// Demand-tracking reservation with hysteresis. For each controlled class, every period:
//   demand = running + ready (CTAs it could use right now), smoothed:  s += (demand - s) * alpha
//   target = clamp(ceil(s) + slack, lo, hi); the reservation moves to the target only outside a +-deadband, rises at once
//   (fast attack: a burst must get CTAs at the next task boundary) and falls by one per `release_periods` calm periods.
//   SLO feedback: if the class's last-window queue wait exceeds target_wait_us the reservation gets +1 on top.
// The other classes need no explicit action: work conservation lends the idle share and the reservation reclaims it.
struct MvmCtlClass { int cls; int lo, hi; };
struct MvmCtlConfig {
  std::vector<MvmCtlClass> classes;
  int period_us = 500; double alpha = 0.3; int slack = 1, deadband = 1, release_periods = 40;
};
struct MvmCtlSample { double t_ms; int cls, running, ready, min_cta, wait_max_us; };
class MvmController {
 public:
  MvmController(MvmSchedParams* hp, const MvmSchedTele* ht, MvmSchedParams base, MvmCtlConfig cfg) : hp_(hp), ht_(ht), p_(base), cfg_(cfg) {
    for (auto& c : cfg_.classes) smooth_[c.cls] = 0;
  }
  ~MvmController() { stop(); }
  void start() {
    stop_.store(false); t0_ = std::chrono::steady_clock::now();
    th_ = std::thread([this] { loop(); });
  }
  void stop() { stop_.store(true); if (th_.joinable()) th_.join(); }
  const std::vector<MvmCtlSample>& log() const { return log_; }
  int changes() const { return changes_; }
  // Seqlock writer, usable without a controller: the policy block picks the new set up within a few us.
  static void write_params(MvmSchedParams* hp, const MvmSchedParams& v) {
    int g = hp->gen_a + 1;
    __atomic_store_n(&hp->gen_a, g, __ATOMIC_RELEASE);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    hp->flags = v.flags; hp->bw_limit_init = v.bw_limit_init; hp->bw_limit_min = v.bw_limit_min; hp->bw_limit_max = v.bw_limit_max;
    hp->sat_pct = v.sat_pct; hp->window_us = v.window_us; hp->affinity_us = v.affinity_us;
    for (int i = 0; i < MVM_MAX_CLS; ++i) hp->cls[i] = v.cls[i];
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(&hp->gen_b, g, __ATOMIC_RELEASE);
  }
 private:
  void loop() {
    std::vector<int> calm(MVM_MAX_CLS, 0);
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(cfg_.period_us));
      bool dirty = false;
      double now_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
      for (auto& c : cfg_.classes) {
        const MvmClassTele& t = ht_->cls[c.cls];
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
      if (dirty) write_params(hp_, p_);
    }
  }
  MvmSchedParams* hp_; const MvmSchedTele* ht_; MvmSchedParams p_; MvmCtlConfig cfg_;
  double smooth_[MVM_MAX_CLS] = {};
  std::vector<MvmCtlSample> log_; int changes_ = 0;
  std::atomic<bool> stop_{false}; std::thread th_; std::chrono::steady_clock::time_point t0_;
};
