#pragma once
// Closed-loop adaptive rebalancing (host side). Builds on kvm_sched.h: the device already enforces weights, reservations, caps and
// EDF at task boundaries and exports per-class counters; this controller reads the counters every `period_us` and rewrites the
// parameters through the seqlock block, no relaunch, no hard SM partition.
//
// Signals (all differences of the device's cumulative counters, so a slow host loses nothing):
//   pressure    near-miss tasks (finished in the last quarter of their deadline budget, misses included) and tasks that waited longer
//               than the class's target_wait_us in the queue. Pressure is acted on BEFORE a miss: the first near miss raises the class.
//   demand      running + ready CTAs, smoothed (how many workers the class could use right now)
//   share       achieved SM-time = d(sm_ns) / (dt * workers)
// Per role:
//   DEADLINE / LATENCY  reservation = ceil(smoothed demand) + slack + boost. `boost` doubles (+1) on pressure (fast attack) and halves
//                       after `release_periods` calm periods (slow release); a DEADLINE class also opens its EDF window (urgent_us)
//                       while boosted; the weight follows the boost between w_lo and w_hi.
//   BATCH               holds `floor_cta` workers in reserve while it has work (anti-starvation: no latency class can take them), takes
//                       whatever the others leave (the device lends idle CTAs, no explicit action).
// Invariant: sum of reservations <= workers - floor_cta of the batch classes, and each latency class's cap is what the others leave, so
// a flood of requests can not starve training below its floor and an idle reservation costs nothing (the device lends it).
#include "kvm_sched.h"
#include <condition_variable>
#include <mutex>

enum class KvmRole { DEADLINE, LATENCY, BATCH };
struct KvmAdaptClass {
  int cls = 0; KvmRole role = KvmRole::LATENCY;
  int min_lo = 0, min_hi = KVM_UNLIMITED;   // reservation bounds (DEADLINE / LATENCY)
  int w_lo = 1, w_hi = 1;                   // weight range: w_lo when calm, up to w_hi under full boost
  int floor_cta = 0;                        // BATCH: workers kept for it while it has work
  int urgent_max_us = 0;                    // DEADLINE: EDF window while boosted (0 = never open it)
  int target_wait_us = 0;                   // LATENCY: queue-wait SLO; counted on the device as wait_over
  int boost_max = 1 << 20;                  // bound on the extra reservation
};
struct KvmAdaptConfig {
  std::vector<KvmAdaptClass> classes;
  int workers = 0;                          // worker blocks (KvmExec::workers())
  int period_us = 1000;
  double alpha = 0.3;                       // smoothing of demand
  int slack = 0;                            // CTAs reserved above the smoothed demand
  int release_periods = 50;                 // calm periods before the boost halves
  double wait_frac = 0.02;                  // a LATENCY class is under pressure when more than this fraction of its tasks waited over target
  bool arbiter_assist = false;              // switch the device's bandwidth arbiter (KVM_SF_ARBITER) on while a protected class is under pressure
                                            // AND the memory system is saturated; off again after `release_periods` calm periods
  int bw_sat_pct = 70;                      // saturated = achieved bandwidth above this percentage of the observed peak
};
struct KvmAdaptGlobal { int bw_mbps = 0, bw_peak_mbps = 0; };
struct KvmAdaptSnap {                       // one class's cumulative counters + instantaneous queue state (a copy of KvmClassTele)
  unsigned long long sm_ns = 0, done = 0, missed = 0, dl_n = 0, near_miss = 0, wait_over = 0;
  int running = 0, ready = 0;
};
struct KvmAdaptSample { double t_ms; int cls, running, ready, min_cta, max_cta, weight, urgent_us, boost; double share; unsigned long long near_miss, missed; };

class KvmAdaptive {
 public:
  KvmAdaptive(KvmSchedParams* hp, const KvmSchedTele* ht, KvmSchedParams base, KvmAdaptConfig cfg) : hp_(hp), ht_(ht), p_(base), cfg_(cfg) { init(); }
  KvmAdaptive(KvmSchedParams base, KvmAdaptConfig cfg) : p_(base), cfg_(cfg) { init(); }   // no device: drive update() directly (tests)
  ~KvmAdaptive() { stop(); }
  void start() {
    if (!hp_ || !ht_) throw std::logic_error("KvmAdaptive: no device attached");
    stop_.store(false); t0_ = std::chrono::steady_clock::now(); last_ = t0_;
    for (auto& s : st_) s.valid = false;
    th_ = std::thread([this] { loop(); });
  }
  void stop() { stop_.store(true); if (th_.joinable()) th_.join(); }
  const std::vector<KvmAdaptSample>& log() const { return log_; }
  int changes() const { return changes_; }
  const KvmSchedParams& params() const { return p_; }

  // One control step. `snaps[i]` belongs to cfg.classes[i]; dt_us is the time since the previous step. Returns true when the
  // parameters changed (the caller writes them). Pure with respect to the device: this is what the unit tests drive.
  bool update(const std::vector<KvmAdaptSnap>& snaps, double dt_us, double t_ms = 0, KvmAdaptGlobal g = {}) {
    bool dirty = false;
    bool any_pressure = false;
    const int n = (int)cfg_.classes.size();
    std::vector<int> want_min(n, 0);
    int batch_floor = 0;
    for (int i = 0; i < n; ++i) if (cfg_.classes[i].role == KvmRole::BATCH && snaps[i].running + snaps[i].ready > 0) batch_floor += cfg_.classes[i].floor_cta;
    const int budget = std::max(0, cfg_.workers - batch_floor);   // reservations of the latency classes must fit in what the floor leaves
    for (int i = 0; i < n; ++i) {
      const KvmAdaptClass& c = cfg_.classes[i]; State& s = st_[i]; const KvmAdaptSnap& now = snaps[i];
      KvmClassParams& q = p_.cls[c.cls];
      double share = 0;
      bool pressure = false;
      if (s.valid) {
        const double dsm = (double)(now.sm_ns - s.prev.sm_ns), dtn = std::max(dt_us, 1.0) * 1e3;
        share = dsm / (dtn * std::max(cfg_.workers, 1));
        const unsigned long long ddone = now.done - s.prev.done, dnear = now.near_miss - s.prev.near_miss, dwait = now.wait_over - s.prev.wait_over;
        pressure = dnear > 0 || (c.target_wait_us > 0 && ddone > 0 && (double)dwait > cfg_.wait_frac * (double)ddone);
      }
      s.prev = now; s.valid = true; s.share = share;
      any_pressure |= pressure && c.role != KvmRole::BATCH;
      s.demand += ((now.running + now.ready) - s.demand) * cfg_.alpha;
      if (c.role == KvmRole::BATCH) {
        const int want = (now.running + now.ready > 0) ? c.floor_cta : 0;
        dirty |= set(q.min_cta, want);
        dirty |= set(q.max_cta, (int)KVM_UNLIMITED);
        want_min[i] = 0;
        continue;
      }
      if (pressure) { s.boost = std::min(c.boost_max, s.boost * 2 + 1); s.calm = 0; }
      else if (s.boost > 0 && ++s.calm >= cfg_.release_periods) { s.boost /= 2; s.calm = 0; }
      const int base = (int)std::ceil(s.demand) + cfg_.slack;
      want_min[i] = std::max(c.min_lo, std::min(std::min(c.min_hi, budget), base + s.boost));
      // weight: w_lo calm, w_hi at the boost level that has used up the whole reservation range
      const int span = std::max(1, std::min(c.min_hi, budget) - c.min_lo);
      const int wt = c.w_lo + (int)std::lround((double)(c.w_hi - c.w_lo) * std::min(1.0, (double)s.boost / span));
      dirty |= set(q.weight, std::max(1, wt));
      dirty |= set(q.urgent_us, (c.urgent_max_us > 0 && s.boost > 0) ? c.urgent_max_us : 0);
      dirty |= set(q.target_wait_us, c.target_wait_us);
    }
    // Reservations never exceed the budget in sum. A DEADLINE class keeps what it asked for (a game tick needs its CTA whatever else is
    // going on); the LATENCY classes share the rest proportionally.
    int tot = 0, dl = 0, lat = 0;
    for (int i = 0; i < n; ++i) { tot += want_min[i]; (cfg_.classes[i].role == KvmRole::DEADLINE ? dl : lat) += want_min[i]; }
    if (tot > budget) {
        const int room = std::max(0, budget - dl);
        for (int i = 0; i < n; ++i) if (cfg_.classes[i].role == KvmRole::LATENCY) want_min[i] = lat > 0 ? (int)((long long)want_min[i] * room / lat) : 0;
        tot = 0; for (int i = 0; i < n; ++i) tot += want_min[i];
        if (tot > budget) for (int i = 0; i < n; ++i) want_min[i] = (int)((long long)want_min[i] * budget / tot);   // deadline classes alone exceed the budget: scale all
    }
    tot = 0; for (int i = 0; i < n; ++i) tot += want_min[i];
    for (int i = 0; i < n; ++i) {
      const KvmAdaptClass& c = cfg_.classes[i]; if (c.role == KvmRole::BATCH) continue;
      KvmClassParams& q = p_.cls[c.cls];
      dirty |= set(q.min_cta, want_min[i]);
      dirty |= set(q.max_cta, std::max(want_min[i], budget - (tot - want_min[i])));   // the cap above the reservation: what the other reservations leave
    }
    if (cfg_.arbiter_assist) {
      const bool saturated = g.bw_peak_mbps > 0 && (long long)g.bw_mbps * 100 >= (long long)cfg_.bw_sat_pct * g.bw_peak_mbps;
      if (any_pressure && saturated) arb_calm_ = 0; else if (arb_calm_ < cfg_.release_periods) ++arb_calm_;
      const int flags = arb_calm_ < cfg_.release_periods ? (p_.flags | KVM_SF_ARBITER) : (p_.flags & ~KVM_SF_ARBITER);
      dirty |= set(p_.flags, flags);
      for (int i = 0; i < n; ++i) if (cfg_.classes[i].role != KvmRole::BATCH) dirty |= set(p_.cls[cfg_.classes[i].cls].protect, p_.cls[cfg_.classes[i].cls].protect | KVM_CF_PROTECT);
    }
    if (dirty) ++changes_;
    for (int i = 0; i < n; ++i) {
      const KvmAdaptClass& c = cfg_.classes[i]; const KvmClassParams& q = p_.cls[c.cls];
      log_.push_back({t_ms, c.cls, snaps[i].running, snaps[i].ready, q.min_cta, q.max_cta, q.weight, q.urgent_us, st_[i].boost, st_[i].share, snaps[i].near_miss, snaps[i].missed});
    }
    return dirty;
  }

 private:
  struct State { KvmAdaptSnap prev; bool valid = false; double demand = 0, share = 0; int boost = 0, calm = 0; };
  static bool set(int& field, int v) { if (field == v) return false; field = v; return true; }
  void init() {
    if (cfg_.workers < 1) throw std::invalid_argument("KvmAdaptConfig.workers must be >= 1");
    if (cfg_.classes.empty()) throw std::invalid_argument("KvmAdaptConfig needs at least one class");
    std::vector<bool> seen(KVM_MAX_CLS, false);
    for (auto& c : cfg_.classes) {
      if (c.cls < 0 || c.cls >= KVM_MAX_CLS || seen[c.cls]) throw std::invalid_argument("KvmAdaptConfig: bad or duplicate class id");
      if (c.w_lo < 1 || c.w_hi < c.w_lo || c.min_lo < 0 || c.min_hi < c.min_lo) throw std::invalid_argument("KvmAdaptConfig: weight / reservation range");
      seen[c.cls] = true;
    }
    st_.assign(cfg_.classes.size(), State{});
  }
  void loop() {
    std::vector<KvmAdaptSnap> snaps(cfg_.classes.size());
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(cfg_.period_us));
      const auto t = std::chrono::steady_clock::now();
      const double dt_us = std::chrono::duration<double, std::micro>(t - last_).count(), t_ms = std::chrono::duration<double, std::milli>(t - t0_).count();
      last_ = t;
      for (size_t i = 0; i < snaps.size(); ++i) {
        const KvmClassTele& x = ht_->cls[cfg_.classes[i].cls];
        snaps[i] = {x.sm_ns, x.done, x.missed, x.dl_n, x.near_miss, x.wait_over, x.running, x.ready < 0 ? 0 : x.ready};
      }
      if (update(snaps, dt_us, t_ms, {ht_->bw_mbps, ht_->bw_peak_mbps})) KvmController::write_params(hp_, p_);
    }
  }
  KvmSchedParams* hp_ = nullptr; const KvmSchedTele* ht_ = nullptr; KvmSchedParams p_; KvmAdaptConfig cfg_;
  std::vector<State> st_; std::vector<KvmAdaptSample> log_; int changes_ = 0, arb_calm_ = 1 << 20;
  std::atomic<bool> stop_{false}; std::thread th_; std::chrono::steady_clock::time_point t0_, last_;
};

// Host-side admission budget per class (tasks in flight). The executor's admitter is ONE FIFO ring and the live window (live_cap) is
// shared: a flood of one class fills the window and every bundle behind it waits for room however the scheduler would rank it. The gate
// keeps each class's admitted-but-unfinished tasks under a budget, so a latency class's bundle finds room at once. release() when the
// bundle's window completes. budget < 0 = unlimited.
class KvmAdmissionGate {
 public:
  explicit KvmAdmissionGate(std::vector<long long> budget) : budget_(std::move(budget)), used_(budget_.size(), 0) {}
  // Blocks until `tasks` fit in the class's budget (a bundle larger than the whole budget is admitted alone). False on timeout.
  bool acquire(int cls, long long tasks, int timeout_ms = 60000) {
    std::unique_lock<std::mutex> lk(mu_);
    const long long b = budget_.at(cls);
    auto fits = [&] { return b < 0 || used_[cls] == 0 || used_[cls] + tasks <= b; };
    if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), fits)) return false;
    used_[cls] += tasks; return true;
  }
  void release(int cls, long long tasks) { { std::lock_guard<std::mutex> lk(mu_); used_.at(cls) -= tasks; } cv_.notify_all(); }
  long long in_flight(int cls) { std::lock_guard<std::mutex> lk(mu_); return used_.at(cls); }
 private:
  std::vector<long long> budget_, used_; std::mutex mu_; std::condition_variable cv_;
};
