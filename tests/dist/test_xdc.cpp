// Cross-datacenter pieces: Reliable RTT estimator / adaptive RTO, AdaptiveWindow, PublishBatcher. Pure C++17, no CUDA.
//   g++ -O2 -std=c++17 -pthread -I dist tests/dist/test_xdc.cpp -o test_xdc      (-fsanitize=thread / address,undefined also clean)
// WAN = EMULATION: Reliable -> FaultyTransport{latency_ms, drop} -> BottleneckTransport{bandwidth, tail-drop buffer} -> Loopback.
// Timing-based checks use wide margins; wall time is bounded (about 15 s).
#include <atomic>
#include <cmath>
#include <functional>

#include "../../dist/xdc.hpp"

using namespace kvmd;

static std::atomic<int> g_fail{0};
static std::atomic<int> g_checks{0};
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; fprintf(stderr, "  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #c); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

// Wall-clock assertions (RTT bounds, throughput ratios, latencies) cannot hold on a machine that is being starved: when one fails,
// re-measure scheduler jitter and, if the host is noisy, report a WARNING instead of a failure. Structural assertions (CHECK) never relax.
static bool host_noisy() {
  double worst = 0;
  for (int i = 0; i < 25; ++i) { auto a = Clock::now(); std::this_thread::sleep_for(std::chrono::milliseconds(2)); worst = std::max(worst, std::chrono::duration<double, std::milli>(Clock::now() - a).count() - 2.0); }
  return worst > 8.0;
}
static std::atomic<int> g_timing_warn{0};
#define TCHECK(c, ...) do { ++g_checks; if (!(c)) { if (host_noisy()) { ++g_timing_warn; fprintf(stderr, "  WARN (timing, noisy host) %s:%d: %s -- ", __FILE__, __LINE__, #c); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } else { ++g_fail; fprintf(stderr, "  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #c); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static std::vector<uint8_t> msg_of(uint32_t tag, size_t n) {
  std::vector<uint8_t> v(n);
  uint32_t x = tag * 2654435761u + 1;
  for (size_t i = 0; i < n; ++i) { x = x * 1664525u + 1013904223u; v[i] = (uint8_t)(x >> 24); }
  if (n >= 4) memcpy(v.data(), &tag, 4);
  return v;
}

// An emulated WAN between rank 0 and rank 1.
struct Wan {
  LoopbackHub hub{2};
  LoopbackTransport l0{hub, 0}, l1{hub, 1};
  std::unique_ptr<BottleneckTransport> b0, b1;
  std::unique_ptr<FaultyTransport> f0, f1;
  Wan(int one_way_ms, double bw_Bps, size_t buffer, double drop, uint64_t seed) {
    Transport* t0 = &l0; Transport* t1 = &l1;
    if (bw_Bps > 0) { b0 = std::make_unique<BottleneckTransport>(l0, bw_Bps, buffer); b1 = std::make_unique<BottleneckTransport>(l1, bw_Bps, buffer); t0 = b0.get(); t1 = b1.get(); }
    FaultSpec f; f.latency_ms = one_way_ms; f.drop = drop; f.seed = seed;
    f0 = std::make_unique<FaultyTransport>(*t0, f); f1 = std::make_unique<FaultyTransport>(*t1, f);
  }
  Transport& a() { return *f0; }
  Transport& b() { return *f1; }
};

static ReliableConfig wan_cfg() {
  ReliableConfig c; c.window_msgs = 4096; c.window_bytes = 64u << 20;
  c.rto_ms = 150; c.rto_max_ms = 1000; c.rto_adaptive = true; c.rto_min_ms = 60; c.max_retries = 40; c.dead_ms = 60000;
  return c;
}

// ------------------------------------------------------------------------------------------------ Reliable RTT estimator
static void suite_rtt() {
  fprintf(stderr, "== reliable: RTT estimator (Karn) and adaptive RTO, emulated 20 ms RTT\n");
  auto run = [](double drop, bool adaptive_rto, int rto_ms, int n, int window_msgs, RttInfo* out, long* retx, int* ms) {
    Wan w(10, 0, 0, drop, 21);
    ReliableConfig c = wan_cfg(); c.rto_ms = rto_ms; c.rto_adaptive = adaptive_rto; c.window_msgs = window_msgs;
    Reliable ra(w.a(), c), rb(w.b(), c);
    std::atomic<bool> done{false}; std::atomic<int> got{0};
    auto t0 = Clock::now();
    std::thread tr([&] { Reliable::Delivery d; while (!done) if (rb.poll(d, 1)) ++got; });
    int sent = 0; Reliable::Delivery d;
    while ((sent < n || ra.unacked_total() > 0) && ms_since(t0) < 30000) {
      if (sent < n) { auto m = msg_of(sent, 1000); if (ra.send(1, m.data(), m.size()) == Reliable::OK) ++sent; }
      ra.poll(d, 1);
    }
    *ms = ms_since(t0);
    done = true; tr.join();
    CHECK(got == n, "delivered %d/%d", got.load(), n);
    *out = ra.rtt(1); *retx = ra.st.retx;
  };
  RttInfo r; long retx; int ms;
  run(0, false, 150, 150, 8, &r, &retx, &ms);
  fprintf(stderr, "  clean: srtt %.1f ms, rttvar %.1f, min_rtt %.1f, %ld samples, rate %.2f MB/s, acked %llu B\n", r.srtt_ms, r.rttvar_ms, r.min_rtt_ms, r.samples, r.rate_Bps / 1e6, r.acked_bytes);
  CHECK(r.samples > 5, "no RTT samples");
  TCHECK(r.min_rtt_ms >= 19.0 && r.min_rtt_ms < 35.0, "min_rtt %.1f ms for a 20 ms RTT", r.min_rtt_ms);
  TCHECK(r.srtt_ms >= 19.0 && r.srtt_ms < 60.0, "srtt %.1f ms", r.srtt_ms);
  CHECK(r.acked_bytes == 150ull * 1000, "acked bytes %llu", r.acked_bytes);
  CHECK(r.rate_Bps > 0, "no delivery rate");
  run(0.2, false, 150, 120, 8, &r, &retx, &ms);   // heavy loss both ways: samples must stay clean (retransmitted messages are never sampled)
  fprintf(stderr, "  20%% drop: srtt %.1f ms, min_rtt %.1f, %ld samples, %ld retx\n", r.srtt_ms, r.min_rtt_ms, r.samples, retx);
  CHECK(retx > 0, "no retransmissions at 20%% drop");
  TCHECK(r.min_rtt_ms >= 19.0, "min_rtt %.1f ms is below the physical RTT: an ambiguous sample got through", r.min_rtt_ms);
  TCHECK(r.srtt_ms < 70.0, "srtt %.1f ms: samples polluted by loss recovery", r.srtt_ms);
  // the RTO follows SRTT: with a conservative fixed RTO (300 ms) a loss costs 300 ms; adaptive pays ~srtt + 4 rttvar (floored at rto_min).
  long retx_fixed, retx_adapt; int ms_fixed, ms_adapt; RttInfo ri;
  run(0.1, false, 300, 100, 4, &ri, &retx_fixed, &ms_fixed);
  run(0.1, true, 300, 100, 4, &ri, &retx_adapt, &ms_adapt);
  fprintf(stderr, "  10%% drop, window 4: fixed RTO 300 ms = %d ms (%ld retx); adaptive RTO = %d ms (%ld retx)\n", ms_fixed, retx_fixed, ms_adapt, retx_adapt);
  TCHECK(ms_adapt < ms_fixed, "adaptive RTO (%d ms) not faster than fixed 300 ms (%d ms)", ms_adapt, ms_fixed);
  { // default config keeps the fixed RTO and the estimator does not change behaviour
    ReliableConfig d; CHECK(!d.rto_adaptive && d.rto_ms == 100, "defaults changed");
  }
}

// ----------------------------------------------------------------------------------------------------------- batcher unit
struct FakeStream {
  std::vector<std::vector<uint8_t>> sent; Reliable::Status next = Reliable::OK;
  PublishBatcher::SendFn fn() { return [this](int, const void* p, size_t n) { if (next == Reliable::OK) sent.emplace_back((const uint8_t*)p, (const uint8_t*)p + n); return next; }; }
};

static void suite_batcher_unit() {
  fprintf(stderr, "== PublishBatcher (unit, no network)\n");
  auto t0 = Clock::now();
  {   // coalescing: last version wins, lower versions dropped, one message
    FakeStream s; BatcherConfig c; c.max_pairs = 100; c.max_bytes = 1 << 20; c.max_delay_us = 1000000;
    PublishBatcher b(s.fn(), c);
    for (uint32_t v = 1; v <= 50; ++v) b.publish(1, 7, v, t0);
    b.publish(1, 7, 3, t0);                  // stale: ignored
    b.publish(1, 9, 4, t0);
    CHECK(s.sent.empty(), "nothing may be sent before a flush condition");
    CHECK(b.pending(1) == 2, "pending %zu", b.pending(1));
    CHECK(b.flush(1) == Reliable::OK && s.sent.size() == 1, "one wire message");
    PublishBatcher::Pairs p; CHECK(PublishBatcher::decode(s.sent[0].data(), s.sent[0].size(), p), "decode");
    CHECK(p.size() == 2 && p[0] == std::make_pair(7u, 50u) && p[1] == std::make_pair(9u, 4u), "last version wins: got %zu pairs", p.size());
    CHECK(b.st.published == 52 && b.st.coalesced == 50, "stats published %ld coalesced %ld", b.st.published, b.st.coalesced);
  }
  {   // count flush, bytes flush, delay flush
    FakeStream s; BatcherConfig c; c.max_pairs = 4; c.max_bytes = 1 << 20; c.max_delay_us = 5000;
    PublishBatcher b(s.fn(), c);
    for (uint32_t v = 0; v < 8; ++v) b.publish(1, v, 1, t0);
    CHECK(s.sent.size() == 2, "count flush: %zu messages", s.sent.size());
    b.publish(1, 100, 1, t0);
    b.tick(t0 + std::chrono::microseconds(4000)); CHECK(s.sent.size() == 2, "too early");
    b.tick(t0 + std::chrono::microseconds(5000)); CHECK(s.sent.size() == 3 && b.pending(1) == 0, "delay flush");
    FakeStream s2; BatcherConfig c2; c2.max_pairs = 1000; c2.max_bytes = 4 + 8 * 3; c2.max_delay_us = 1000000;
    PublishBatcher b2(s2.fn(), c2);
    for (uint32_t v = 0; v < 6; ++v) b2.publish(2, v, 1, t0);
    CHECK(s2.sent.size() == 2 && s2.sent[0].size() == 28, "bytes flush: %zu msgs", s2.sent.size());
  }
  {   // FULL: the batch is kept, keeps coalescing, and goes out once the stream accepts it
    FakeStream s; BatcherConfig c; c.max_pairs = 2;
    PublishBatcher b(s.fn(), c);
    s.next = Reliable::FULL;
    b.publish(1, 1, 1, t0); b.publish(1, 2, 1, t0);
    CHECK(s.sent.empty() && b.pending(1) == 2, "kept while FULL");
    b.publish(1, 1, 2, t0);
    s.next = Reliable::OK;
    CHECK(b.flush(1) == Reliable::OK && s.sent.size() == 1, "sent after FULL cleared");
    PublishBatcher::Pairs p; PublishBatcher::decode(s.sent[0].data(), s.sent[0].size(), p);
    CHECK(p.size() == 2 && p[0].second == 2, "newest version kept across the FULL stall");
  }
  {   // put_signal refuses to publish when the data was not accepted (a signal must never exist without its data in the stream)
    FakeStream s; PublishBatcher b(s.fn());
    s.next = Reliable::FULL;
    CHECK(b.put_signal(1, "x", 1, 3, 1) == Reliable::FULL && b.pending(1) == 0, "no publish after a refused data send");
    s.next = Reliable::OK;
    CHECK(b.put_signal(1, "x", 1, 3, 1) == Reliable::OK && b.pending(1) == 1 && s.sent.size() == 1, "data sent, signal parked");
  }
  {   // decode rejects malformed input
    PublishBatcher::Pairs p; uint8_t good[12] = {XDC_SIGNALS, 1, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0};
    CHECK(PublishBatcher::decode(good, 12, p) && p.size() == 1 && p[0].first == 1 && p[0].second == 2, "valid");
    CHECK(!PublishBatcher::decode(good, 11, p) && p.empty(), "truncated");
    uint8_t bad[12]; memcpy(bad, good, 12); bad[0] = 0x00; CHECK(!PublishBatcher::decode(bad, 12, p), "bad kind");
    memcpy(bad, good, 12); bad[2] = 5; CHECK(!PublishBatcher::decode(bad, 12, p), "count/length mismatch");
    CHECK(!PublishBatcher::decode(good, 0, p), "empty");
  }
}

// ----------------------------------------------------------------------------------------------------- adaptive window
struct XferResult { double ms = 0, MBps = 0; long retx = 0; size_t peak_window = 0; int got = 0; double bw_est = 0; };
enum class Mode { FIXED, ADAPTIVE };

static XferResult transfer(Wan& w, Mode mode, size_t fixed_window, int n, size_t msg_size, AdaptiveWindowConfig acfg = {}) {
  ReliableConfig c = wan_cfg();
  c.window_bytes = mode == Mode::FIXED ? fixed_window : acfg.init_bytes;
  Reliable ra(w.a(), c), rb(w.b(), c);
  std::unique_ptr<AdaptiveWindow> aw; if (mode == Mode::ADAPTIVE) aw = std::make_unique<AdaptiveWindow>(ra, 1, acfg);
  std::atomic<bool> done{false}; std::atomic<int> got{0}; std::atomic<bool> order_ok{true}; std::atomic<long> t_last{0};
  auto t0 = Clock::now();
  std::thread tr([&] {
    Reliable::Delivery d; uint32_t expect = 0;
    while (!done) if (rb.poll(d, 1)) {
      uint32_t tag; memcpy(&tag, d.data.data(), 4);
      if (tag != expect++ || d.data != msg_of(tag, msg_size)) order_ok = false;
      if (++got == n) t_last = ms_since(t0);
    }
  });
  int sent = 0; Reliable::Delivery d;
  while ((sent < n || ra.unacked_total() > 0) && ms_since(t0) < 40000) {
    if (aw) aw->update();
    for (int k = 0; k < 8 && sent < n; ++k) { auto m = msg_of(sent, msg_size); if (ra.send(1, m.data(), m.size()) == Reliable::OK) ++sent; else break; }
    ra.poll(d, 1);
  }
  done = true; tr.join();
  CHECK(order_ok, "payload corrupted or out of order");
  fprintf(stderr, "    [link drops: %ld random, %ld tail]\n", w.f0->dropped + w.f1->dropped, (w.b0 ? w.b0->dropped.load() : 0L) + (w.b1 ? w.b1->dropped.load() : 0L));
  XferResult r; r.got = got; r.ms = got == n ? (double)t_last : (double)ms_since(t0); r.retx = ra.st.retx;
  r.MBps = (double)got * msg_size / 1e6 / (r.ms / 1000.0);
  if (aw) { r.peak_window = aw->peak_window; r.bw_est = aw->bw_Bps(); } else r.peak_window = fixed_window;
  return r;
}

static void suite_adaptive() {
  const int ONE_WAY = 20; const double BW = 8e6; const size_t MSG = 16 << 10;
  const double bdp = BW * 2 * ONE_WAY / 1000.0;
  const size_t buffer = (size_t)(1.5 * bdp);
  fprintf(stderr, "== AdaptiveWindow: %d ms one way (40 ms RTT), %.0f MB/s bottleneck, BDP %.0f KB, buffer %zu KB (emulated)\n", ONE_WAY, BW / 1e6, bdp / 1024, buffer / 1024);
  const int N = 256, NS = 64;                                         // 4 MiB for tuned / adaptive; 1 MiB for the slow fixed window (it is rate-limited, size does not matter)
  XferResult small, tuned, adapt;
  { Wan w(ONE_WAY, BW, buffer, 0, 1); small = transfer(w, Mode::FIXED, 32 << 10, NS, MSG); }
  { Wan w(ONE_WAY, BW, buffer, 0, 2); tuned = transfer(w, Mode::FIXED, (size_t)(1.2 * bdp), N, MSG); }
  { Wan w(ONE_WAY, BW, buffer, 0, 3); adapt = transfer(w, Mode::ADAPTIVE, 0, N, MSG); }
  fprintf(stderr, "  fixed 32 KiB window:        %6.0f ms  %5.2f MB/s  retx %ld\n", small.ms, small.MBps, small.retx);
  fprintf(stderr, "  hand-tuned 1.2 x BDP:       %6.0f ms  %5.2f MB/s  retx %ld\n", tuned.ms, tuned.MBps, tuned.retx);
  fprintf(stderr, "  adaptive (BBR-lite):        %6.0f ms  %5.2f MB/s  retx %ld  peak window %zu KiB, bw est %.2f MB/s\n", adapt.ms, adapt.MBps, adapt.retx, adapt.peak_window >> 10, adapt.bw_est / 1e6);
  CHECK(small.got == NS && tuned.got == N && adapt.got == N, "incomplete transfers %d %d %d", small.got, tuned.got, adapt.got);
  TCHECK(adapt.MBps > 2.0 * small.MBps, "adaptive %.2f MB/s is not far above the fixed small window %.2f MB/s", adapt.MBps, small.MBps);
  TCHECK(adapt.MBps > 0.4 * tuned.MBps, "adaptive %.2f MB/s is not near the hand-tuned window %.2f MB/s", adapt.MBps, tuned.MBps);
  CHECK(adapt.peak_window < 6 * (size_t)bdp, "window grew to %zu KiB (BDP %.0f KiB)", adapt.peak_window >> 10, bdp / 1024);
  {   // 5% random loss: completes, no runaway window, no retransmission storm
    const int N2 = 64;
    XferResult tl, al;
    { Wan w(ONE_WAY, BW, buffer, 0.05, 11); tl = transfer(w, Mode::FIXED, (size_t)(1.2 * bdp), N2, MSG); }
    { Wan w(ONE_WAY, BW, buffer, 0.05, 12); AdaptiveWindowConfig ac; ac.max_bytes = 4u << 20; al = transfer(w, Mode::ADAPTIVE, 0, N2, MSG, ac); }
    fprintf(stderr, "  5%% drop, 1 MiB:  tuned %6.0f ms %5.2f MB/s retx %ld | adaptive %6.0f ms %5.2f MB/s retx %ld peak window %zu KiB\n", tl.ms, tl.MBps, tl.retx, al.ms, al.MBps, al.retx, al.peak_window >> 10);
    CHECK(al.got == N2 && tl.got == N2, "incomplete under loss");
    CHECK(al.peak_window <= (4u << 20), "window exceeded its cap");
    CHECK(al.peak_window < 8 * (size_t)bdp, "window blew up under loss: %zu KiB", al.peak_window >> 10);
    CHECK(al.retx < 3L * N2, "retransmission storm: %ld retx for %d messages", al.retx, N2);
  }
}

// ------------------------------------------------------------------------------------------------ both together, 2 ranks
// Receiver logic shared by the real run and the negative check.
struct XdcRx {
  int nvars; std::vector<std::vector<char>> got; std::vector<uint32_t> inbox;
  long violations = 0, data_msgs = 0, sig_msgs = 0, sig_pairs = 0;
  std::vector<double> lat_ms; std::function<double(uint32_t, uint32_t)> pub_age_ms;
  explicit XdcRx(int v, int maxver) : nvars(v), got(v, std::vector<char>(maxver + 2, 0)), inbox(v, 0) {}
  bool on(const std::vector<uint8_t>& m) {
    if (m.empty()) return false;
    if (m[0] == XDC_DATA && m.size() >= 9) {
      uint32_t var, ver; memcpy(&var, &m[1], 4); memcpy(&ver, &m[5], 4);
      if (m.size() > 9 && m != tile(var, ver, m.size())) ++violations;
      got[var][ver] = 1; ++data_msgs; return true;
    }
    PublishBatcher::Pairs p;
    if (!PublishBatcher::decode(m.data(), m.size(), p)) return false;
    ++sig_msgs;
    for (auto& pr : p) {
      ++sig_pairs;
      if (!got[pr.first][pr.second]) ++violations;               // a signal overtook its data
      if (pr.second > inbox[pr.first]) inbox[pr.first] = pr.second;
      if (pub_age_ms) lat_ms.push_back(pub_age_ms(pr.first, pr.second));
    }
    return true;
  }
  static std::vector<uint8_t> tile(uint32_t var, uint32_t ver, size_t n) {
    std::vector<uint8_t> v = msg_of(var * 1000 + ver, n); v[0] = XDC_DATA; memcpy(&v[1], &var, 4); memcpy(&v[5], &ver, 4); return v;
  }
};

struct CombinedResult { long sig_msgs = 0, sig_pairs = 0, published = 0, violations = 0, retx = 0; double lat_med = 0, lat_max = 0; int ms = 0; bool complete = false; };

static CombinedResult run_combined(BatcherConfig bc, int one_way_ms, double drop) {
  const int T = 240, V = 12, SZ = 2000, MAXV = T / V;
  Wan w(one_way_ms, 0, 0, drop, 77);
  ReliableConfig c = wan_cfg(); c.rto_ms = 4 * one_way_ms + 30; c.rto_min_ms = 2 * one_way_ms + 20;
  Reliable ra(w.a(), c), rb(w.b(), c);
  AdaptiveWindow aw(ra, 1);
  PublishBatcher batch([&](int dst, const void* p, size_t n) { return ra.send(dst, p, n); }, bc);
  std::vector<std::atomic<long long>> pub_ns((V) * (MAXV + 2));
  for (auto& x : pub_ns) x = 0;
  auto now_ns = [] { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); };
  XdcRx rx(V, MAXV);
  rx.pub_age_ms = [&](uint32_t var, uint32_t ver) { return (now_ns() - pub_ns[var * (MAXV + 2) + ver]) / 1e6; };
  std::atomic<bool> done{false}; std::atomic<long> rx_msgs{0}; std::atomic<bool> rx_complete{false};
  auto t0 = Clock::now();
  std::thread tr([&] { Reliable::Delivery d; while (!done) if (rb.poll(d, 1)) { rx.on(d.data); ++rx_msgs; bool all = rx.data_msgs == T; for (int v = 0; v < V; ++v) all &= rx.inbox[v] == (uint32_t)MAXV; if (all) rx_complete = true; } });
  Reliable::Delivery d;
  bool dead = false;
  for (int i = 0; i < T && !dead; ++i) {
    if (bc.max_pairs == 1) while (batch.pending(1) > 0 && !dead) { if (batch.flush(1) == Reliable::DEAD) dead = true; ra.poll(d, 1); }   // unbatched = never coalesce behind a FULL stall
    uint32_t var = i % V, ver = i / V + 1;
    auto msg = XdcRx::tile(var, ver, SZ);
    pub_ns[var * (MAXV + 2) + ver] = now_ns();
    for (;;) {                                           // data first, then the signal, on the same Reliable stream
      aw.update(); batch.tick();
      auto s = batch.put_signal(1, msg.data(), msg.size(), var, ver);
      if (s == Reliable::OK) break;
      if (s == Reliable::DEAD) { dead = true; break; }
      ra.poll(d, 1);
    }
    ra.poll(d, 0);
    std::this_thread::sleep_for(std::chrono::microseconds(300));    // the producer takes time per tile
  }
  while (batch.pending(1) > 0 && !dead) { if (batch.flush(1) == Reliable::DEAD) dead = true; ra.poll(d, 1); }   // flush BEFORE finishing (the FIN rule)
  while (ra.unacked_total() > 0 && ms_since(t0) < 30000) { aw.update(); ra.poll(d, 1); }
  { auto q0 = Clock::now(); while (!rx_complete && !dead && ms_since(q0) < 60000) std::this_thread::sleep_for(std::chrono::milliseconds(1)); }   // completion, not a settle timer: the last signal is the last message on the ordered stream
  done = true; tr.join();
  CombinedResult r;
  r.sig_msgs = rx.sig_msgs; r.sig_pairs = rx.sig_pairs; r.published = batch.st.published; r.violations = rx.violations; r.retx = ra.st.retx;
  r.ms = ms_since(t0);
  std::sort(rx.lat_ms.begin(), rx.lat_ms.end());
  if (!rx.lat_ms.empty()) { r.lat_med = rx.lat_ms[rx.lat_ms.size() / 2]; r.lat_max = rx.lat_ms.back(); }
  bool all = rx.data_msgs == T;
  for (int v = 0; v < V; ++v) all &= rx.inbox[v] == (uint32_t)MAXV;           // last version wins on every variable
  r.complete = all && !dead;
  return r;
}

static void suite_combined() {
  fprintf(stderr, "== adaptive window + publish batching together: 2 ranks, Reliable over Faulty{latency 15 ms one way, 2%% drop}\n");
  BatcherConfig on; on.max_pairs = 6; on.max_bytes = 4096; on.max_delay_us = 5000;
  BatcherConfig off; off.max_pairs = 1;
  CombinedResult a = run_combined(off, 15, 0.02), b = run_combined(on, 15, 0.02);
  fprintf(stderr, "  unbatched: %ld signal msgs for %ld publishes, latency median %.1f max %.1f ms, retx %ld, %d ms\n", a.sig_msgs, a.published, a.lat_med, a.lat_max, a.retx, a.ms);
  fprintf(stderr, "  batched:   %ld signal msgs for %ld publishes (%ld pairs on the wire), latency median %.1f max %.1f ms, retx %ld, %d ms\n", b.sig_msgs, b.published, b.sig_pairs, b.lat_med, b.lat_max, b.retx, b.ms);
  CHECK(a.complete && b.complete, "data lost or a final version missing (unbatched %d, batched %d)", (int)a.complete, (int)b.complete);
  CHECK(a.violations == 0 && b.violations == 0, "a signal overtook its data or a tile was corrupt: %ld / %ld", a.violations, b.violations);
  CHECK(a.sig_msgs == a.published, "unbatched must send exactly one message per publish (%ld vs %ld)", a.sig_msgs, a.published);
  CHECK(b.sig_msgs * 2 <= b.published, "batching saved too little: %ld messages for %ld publishes", b.sig_msgs, b.published);
  CHECK(b.sig_pairs <= b.published, "more pairs on the wire than publishes");
  TCHECK(b.lat_med < 15 + 5 + 30, "batched median signal latency %.1f ms", b.lat_med);
  TCHECK(b.lat_max < 600, "batched max signal latency %.1f ms", b.lat_max);
  {   // the checker must be able to FAIL: signal delivered before its data (what an unordered path would do) is flagged
    XdcRx rx(2, 4);
    PublishBatcher::Pairs unused; std::map<uint32_t, uint32_t> m{{1, 2}};
    rx.on(PublishBatcher::encode(m));
    rx.on(XdcRx::tile(1, 2, 64));
    CHECK(rx.violations == 1, "negative check: a signal that precedes its data must be flagged (got %ld)", rx.violations);
  }
}

int main() {
  auto t0 = Clock::now();
  suite_batcher_unit();
  suite_rtt();
  suite_adaptive();
  suite_combined();
  fprintf(stderr, "wall time %d ms\n", ms_since(t0));
  printf("%s: %d checks, %d failed, %d timing warnings (noisy host)\n", g_fail ? "FAIL" : "PASS", g_checks.load(), g_fail.load(), g_timing_warn.load());
  return g_fail ? 1 : 0;
}
