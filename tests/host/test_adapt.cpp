// Unit tests of the closed-loop controller (include/kvm_adapt.h) against a scripted plant: no GPU, no threads.
//   c++ -std=c++17 -O1 -Iinclude tests/host/test_adapt.cpp -o test_adapt && ./test_adapt
#include "../../include/kvm_adapt.h"
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static KvmAdaptConfig cfg3() {   // class 0 DEADLINE (a game tick), class 1 LATENCY (serve), class 2 BATCH (training)
  KvmAdaptConfig c; c.workers = 100; c.release_periods = 5; c.alpha = 1.0;
  KvmAdaptClass d; d.cls = 0; d.role = KvmRole::DEADLINE; d.min_hi = 16; d.w_lo = 2; d.w_hi = 16; d.urgent_max_us = 20000;
  KvmAdaptClass s; s.cls = 1; s.role = KvmRole::LATENCY; s.min_hi = 90; s.w_lo = 1; s.w_hi = 8; s.target_wait_us = 500;
  KvmAdaptClass t; t.cls = 2; t.role = KvmRole::BATCH; t.floor_cta = 20;
  c.classes = {d, s, t}; return c;
}
struct Plant {   // cumulative counters the way the device exports them
  std::vector<KvmAdaptSnap> s = std::vector<KvmAdaptSnap>(3);
  void run(int cls, int running, int ready, int done, int near, int wait_over) {
    auto& x = s[cls]; x.running = running; x.ready = ready; x.done += done; x.near_miss += near; x.wait_over += wait_over; x.dl_n += done;
  }
};
static void drive(KvmAdaptive& a, Plant& p, int steps, double& t) { for (int i = 0; i < steps; ++i) { a.update(p.s, 1000, t); t += 1; } }

static void test_tracks_demand_and_releases() {
  KvmAdaptive a(KvmSchedParams{}, cfg3()); Plant p; double t = 0;
  p.run(1, 30, 20, 100, 0, 0); drive(a, p, 3, t);
  CHECK(a.params().cls[1].min_cta == 50, "reservation follows demand 30+20, got %d", a.params().cls[1].min_cta);
  p.run(1, 0, 0, 0, 0, 0); drive(a, p, 3, t);
  CHECK(a.params().cls[1].min_cta == 0, "an idle class releases its reservation at once (the device lends it anyway), got %d", a.params().cls[1].min_cta);
  CHECK(a.params().cls[1].weight == 1, "calm weight is w_lo, got %d", a.params().cls[1].weight);
}
static void test_near_miss_attack_and_slow_release() {
  KvmAdaptive a(KvmSchedParams{}, cfg3()); Plant p; double t = 0;
  p.run(0, 1, 0, 10, 0, 0); drive(a, p, 3, t);
  CHECK(a.params().cls[0].min_cta == 1 && a.params().cls[0].urgent_us == 0, "calm deadline class: reservation 1 (its demand), no EDF window; got min %d urgent %d", a.params().cls[0].min_cta, a.params().cls[0].urgent_us);
  p.run(0, 1, 0, 10, 2, 0); drive(a, p, 1, t);               // one period with near misses
  CHECK(a.params().cls[0].min_cta >= 2, "first near miss raises the reservation in ONE period, got %d", a.params().cls[0].min_cta);
  CHECK(a.params().cls[0].urgent_us == 20000, "EDF window opens under pressure, got %d", a.params().cls[0].urgent_us);
  CHECK(a.params().cls[0].weight > 2, "weight rises with the boost, got %d", a.params().cls[0].weight);
  int hi = a.params().cls[0].min_cta;
  p.run(0, 1, 0, 10, 0, 0); drive(a, p, 4, t);               // calm, but release_periods = 5 not yet reached
  CHECK(a.params().cls[0].min_cta == hi, "release is slow: still %d after 4 calm periods, got %d", hi, a.params().cls[0].min_cta);
  drive(a, p, 40, t);
  CHECK(a.params().cls[0].min_cta <= 1 + 2 && a.params().cls[0].urgent_us == 0, "boost fully released (within the deadband of the demand) after a long calm: min %d urgent %d", a.params().cls[0].min_cta, a.params().cls[0].urgent_us);
}
static void test_flood_cannot_starve_batch() {
  KvmAdaptive a(KvmSchedParams{}, cfg3()); Plant p; double t = 0;
  for (int i = 0; i < 20; ++i) { p.run(1, 100, 5000, 200, 0, 200); p.run(2, 0, 500, 0, 0, 0); p.run(0, 1, 0, 10, 3, 0); a.update(p.s, 1000, t); t += 1; }
  int tot = a.params().cls[0].min_cta + a.params().cls[1].min_cta;
  CHECK(tot <= 100 - 20, "reservations %d must leave the batch floor (20 of 100), got sum %d", 80, tot);
  CHECK(a.params().cls[2].min_cta == 20, "batch holds its floor while it has work, got %d", a.params().cls[2].min_cta);
  CHECK(a.params().cls[1].max_cta == KvmSchedParams{}.cls[1].max_cta, "the controller never touches caps (reservations carry the guarantee), got %d", a.params().cls[1].max_cta);
  CHECK(a.params().cls[0].min_cta >= 1, "the deadline class keeps its reservation under a flood");
}
static void test_idle_batch_releases_floor() {
  KvmAdaptive a(KvmSchedParams{}, cfg3()); Plant p; double t = 0;
  p.run(1, 95, 100, 100, 0, 100); p.run(2, 0, 0, 0, 0, 0); drive(a, p, 4, t);
  CHECK(a.params().cls[2].min_cta == 0, "no batch work: no floor reserved, got %d", a.params().cls[2].min_cta);
  CHECK(a.params().cls[1].min_cta > 80, "serve may use the freed CTAs, got %d", a.params().cls[1].min_cta);
}
static void test_no_churn_in_steady_state() {
  KvmAdaptive a(KvmSchedParams{}, cfg3()); Plant p; double t = 0;
  p.run(1, 10, 2, 50, 0, 0); p.run(0, 1, 0, 10, 0, 0); p.run(2, 60, 400, 3000, 0, 0); drive(a, p, 5, t);
  const int before = a.changes();
  drive(a, p, 200, t);
  CHECK(a.changes() == before, "steady state must not rewrite parameters (%d writes in 200 periods)", a.changes() - before);
}
static void test_gate_tuner() {
  KvmAdmissionGate g({-1, 1000, 1000}); KvmGateTuner t(g, {1, 2}, 100, 1000, 0.25, 0.5, 0.5, 0.1, 3);
  CHECK(g.budget(1) == 1000, "starts at the maximum");
  t.report_slack(0.1); CHECK(g.budget(1) == 500 && g.budget(2) == 500, "pressure halves every tuned class at once, got %lld", g.budget(1));
  t.report_slack(0.0); t.report_slack(0.0); t.report_slack(0.0); CHECK(g.budget(1) == 100, "floor holds, got %lld", g.budget(1));
  t.report_slack(0.7); t.report_slack(0.7); CHECK(g.budget(1) == 100, "growth needs 3 calm reports");
  t.report_slack(0.7); CHECK(g.budget(1) == 200, "additive increase by 10%% of max, got %lld", g.budget(1));
  for (int i = 0; i < 100; ++i) t.report_slack(0.9);
  CHECK(g.budget(1) == 1000, "never above the maximum, got %lld", g.budget(1));
  CHECK(g.in_flight(0) == 0 && g.acquire(0, 1 << 30, 10), "untuned classes are untouched");
  bool threw = false; try { KvmGateTuner bad(g, {1}, 0, 10); } catch (const std::invalid_argument&) { threw = true; } CHECK(threw, "min 0 rejected");
}
static void test_config_errors() {
  auto bad = [](auto f) { KvmAdaptConfig c = cfg3(); f(c); try { KvmAdaptive a(KvmSchedParams{}, c); return false; } catch (const std::invalid_argument&) { return true; } };
  CHECK(bad([](KvmAdaptConfig& c) { c.workers = 0; }), "workers 0 rejected");
  CHECK(bad([](KvmAdaptConfig& c) { c.classes[1].cls = 0; }), "duplicate class rejected");
  CHECK(bad([](KvmAdaptConfig& c) { c.classes[0].w_hi = 0; }), "w_hi < w_lo rejected");
  CHECK(bad([](KvmAdaptConfig& c) { c.classes.clear(); }), "empty rejected");
}
int main() {
  struct { const char* n; void (*f)(); } T[] = {{"tracks demand, releases when idle", test_tracks_demand_and_releases}, {"near-miss attack, slow release", test_near_miss_attack_and_slow_release},
    {"flood cannot starve batch", test_flood_cannot_starve_batch}, {"idle batch releases its floor", test_idle_batch_releases_floor},
    {"no churn in steady state", test_no_churn_in_steady_state}, {"gate tuner", test_gate_tuner}, {"config errors", test_config_errors}};
  for (auto& t : T) { int b = g_fail; t.f(); printf("%s %s\n", g_fail == b ? "ok  " : "FAIL", t.n); }
  printf(g_fail ? "FAILED: %d checks\n" : "ALL PASS\n", g_fail);
  return g_fail != 0;
}
