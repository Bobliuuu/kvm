// Transport / reliable layer / proxy tests. Pure C++17, no CUDA: runs on a laptop and on the GPU box.
//   g++ -O2 -std=c++17 -pthread -I dist tests/dist/test_transport.cpp -o test_transport      (-fsanitize=thread for TSAN)
//   ./test_transport                      all in-process suites
//   ./test_transport proc RANK WORLD PORT0,PORT1,..[,drop=0.1,dup=0.1,delay=0.2]   one rank of a multi-PROCESS run
#include <atomic>
#include <cassert>
#include <cmath>
#include <functional>
#include <set>
#include <sstream>

#include "../../dist/proxy.hpp"

using namespace kvmd;

static std::atomic<int> g_fail{0};
static int g_checks = 0;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; fprintf(stderr, "  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #c); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static std::vector<uint8_t> msg_of(uint32_t tag, size_t n) {
  std::vector<uint8_t> v(n);
  uint32_t x = tag * 2654435761u + 1;
  for (size_t i = 0; i < n; ++i) { x = x * 1664525u + 1013904223u; v[i] = (uint8_t)(x >> 24); }
  if (n >= 4) memcpy(v.data(), &tag, 4);
  return v;
}

// Two reliable endpoints exchange `count` messages each way; checks exactly-once in-order delivery of exact bytes.
static void exchange(const char* label, Transport& a, Transport& b, int count, size_t size, ReliableConfig rc, bool expect_retx = false) {
  Reliable ra(a, rc), rb(b, rc);
  std::atomic<int> finished{0};
  auto pump = [&](Reliable& r, int me, int n, std::vector<uint32_t>& got, std::atomic<bool>& stop, bool& ok) {
    int sent = 0; bool mine = false;
    while (!stop) {
      if (mine && finished == 2) return;      // keep acknowledging the peer until it has finished too
      if (sent < n) {
        auto m = msg_of(me * 1000000u + sent, size);
        auto s = r.send(1 - me, m.data(), m.size());
        if (s == Reliable::OK) ++sent;
        else if (s == Reliable::DEAD) { ok = false; return; }
      }
      Reliable::Delivery d;
      if (r.poll(d, sent < n ? 0 : 2)) {
        uint32_t tag; memcpy(&tag, d.data.data(), 4);
        got.push_back(tag);
        if (d.data != msg_of(tag, size)) ok = false;
      }
      if (!mine && sent == n && (int)got.size() == count && r.unacked_total() == 0) { mine = true; ++finished; }
    }
  };
  std::vector<uint32_t> ga, gb; std::atomic<bool> stop{false}; bool oka = true, okb = true;
  auto t0 = Clock::now();
  std::thread ta([&] { pump(ra, 0, count, ga, stop, oka); });
  std::thread tb([&] { pump(rb, 1, count, gb, stop, okb); });
  std::thread wd([&] { while (!stop && ms_since(t0) < 60000) std::this_thread::sleep_for(std::chrono::milliseconds(5)); stop = true; });
  ta.join(); tb.join(); stop = true; wd.join();
  bool inorder = true;
  for (size_t i = 0; i < gb.size(); ++i) inorder &= gb[i] == 0 * 1000000u + (uint32_t)i;
  for (size_t i = 0; i < ga.size(); ++i) inorder &= ga[i] == 1 * 1000000u + (uint32_t)i;
  CHECK(oka && okb, "%s: payload corrupted or peer declared dead", label);
  CHECK((int)ga.size() == count && (int)gb.size() == count, "%s: delivered %zu/%zu of %d", label, ga.size(), gb.size(), count);
  CHECK(inorder, "%s: not exactly-once in-order", label);
  if (expect_retx) CHECK(ra.st.retx + rb.st.retx > 0, "%s: faults injected but no retransmission", label);
  fprintf(stderr, "  %-34s %d x %zu B: retx %ld/%ld dups %ld/%ld reordered %ld/%ld  %d ms\n", label, count, size, ra.st.retx, rb.st.retx, ra.st.dups,
          rb.st.dups, ra.st.reordered, rb.st.reordered, ms_since(t0));
}

static ReliableConfig fast_cfg() { ReliableConfig c; c.rto_ms = 20; c.rto_max_ms = 200; c.max_retries = 30; c.dead_ms = 20000; return c; }

static void suite_crc() {
  fprintf(stderr, "== crc32\n");
  CHECK(crc32("123456789", 9) == 0xCBF43926u, "crc32 check value");
  CHECK(crc32("", 0) == 0, "empty");
  auto a = msg_of(1, 1000); auto c = a; c[500] ^= 1;
  CHECK(crc32(a.data(), a.size()) != crc32(c.data(), c.size()), "single bit flip detected");
}

static void suite_loopback() {
  fprintf(stderr, "== reliable over loopback (+ faults)\n");
  for (int i = 0; i < 4; ++i) {
    LoopbackHub hub(2);
    LoopbackTransport a(hub, 0), b(hub, 1);
    if (i == 0) { exchange("loopback clean", a, b, 2000, 100, fast_cfg()); continue; }
    FaultSpec f; f.seed = 7 + i;
    if (i == 1) f.drop = 0.25;
    if (i == 2) { f.dup = 0.3; f.delay = 0.4; f.delay_ms_max = 15; }
    if (i == 3) { f.drop = 0.2; f.dup = 0.2; f.delay = 0.3; f.delay_ms_max = 30; }
    FaultyTransport fa(a, f), fb(b, f);
    const char* names[] = {"", "loopback drop 25%", "loopback dup 30% delay 40%", "loopback drop+dup+delay"};
    exchange(names[i], fa, fb, 1500, 64, fast_cfg(), i != 2);
    CHECK(fa.dropped + fb.dropped + fa.duplicated + fb.duplicated + fa.delayed + fb.delayed > 0, "injector idle");
  }
}

static std::vector<std::unique_ptr<TcpTransport>> tcp_pair(unsigned corrupt_ppm = 0) {
  auto ports = pick_free_ports(2);
  std::vector<std::unique_ptr<TcpTransport>> t(2);
  std::vector<std::thread> th;
  for (int r = 0; r < 2; ++r) th.emplace_back([&, r] {
    TcpConfig c; c.rank = r; c.hosts = {"127.0.0.1", "127.0.0.1"}; c.ports = ports; c.corrupt_ppm = corrupt_ppm; c.cluster_id = 42;
    t[r] = std::make_unique<TcpTransport>(c);
  });
  for (auto& x : th) x.join();
  return t;
}

static void suite_tcp() {
  fprintf(stderr, "== reliable over TCP (framing, CRC, faults)\n");
  { auto t = tcp_pair(); exchange("tcp clean", *t[0], *t[1], 3000, 200, fast_cfg()); }
  { auto t = tcp_pair(); exchange("tcp 1 MiB messages", *t[0], *t[1], 20, 1 << 20, fast_cfg()); }
  { auto t = tcp_pair(); exchange("tcp 4 MiB messages", *t[0], *t[1], 4, 4 << 20, fast_cfg()); }
  {
    auto t = tcp_pair(200000);   // 20% of frames get a flipped bit AFTER the CRC: the receiver must drop them, ARQ recovers
    exchange("tcp 20% corrupted frames", *t[0], *t[1], 800, 300, fast_cfg(), true);
    CHECK(t[0]->crc_errors + t[1]->crc_errors > 0, "CRC never fired");
  }
  {
    auto t = tcp_pair();
    FaultSpec f; f.drop = 0.15; f.dup = 0.15; f.delay = 0.3; f.seed = 99;
    FaultyTransport a(*t[0], f), b(*t[1], f);
    exchange("tcp drop+dup+delay", a, b, 1500, 128, fast_cfg(), true);
  }
  {   // wrong cluster id: connection must be refused, the dialer must give up with PEER_DOWN
    auto ports = pick_free_ports(2);
    TcpConfig c0; c0.rank = 0; c0.hosts = {"127.0.0.1", "127.0.0.1"}; c0.ports = ports; c0.cluster_id = 1; c0.connect_timeout_ms = 1500;
    TcpConfig c1 = c0; c1.rank = 1; c1.cluster_id = 2;
    TcpTransport a(c0), b(c1);
    Event e; bool down = false;
    for (int i = 0; i < 40 && !down; ++i) if (a.poll(e, 100) && e.type == EvType::PEER_DOWN) down = true;   // rank 0 dials rank 1
    CHECK(down, "mismatched cluster_id was accepted or never reported");
    CHECK(!a.send(1, "x", 1), "send to a rejected peer must fail");
  }
}

static void suite_failure() {
  fprintf(stderr, "== failure detection (no hangs)\n");
  {   // loopback: peer crashes
    LoopbackHub hub(2); LoopbackTransport a(hub, 0), b(hub, 1);
    ReliableConfig c = fast_cfg(); c.max_retries = 4; Reliable ra(a, c), rb(b, c);
    auto m = msg_of(1, 10);
    CHECK(ra.send(1, m.data(), m.size()) == Reliable::OK, "first send");
    Reliable::Delivery d; CHECK(rb.poll(d, 200), "delivered before the crash"); ra.poll(d, 50);
    hub.kill(1);
    auto t0 = Clock::now(); int dead = -1;
    while (dead < 0 && ms_since(t0) < 3000) { ra.poll(d, 5); dead = ra.dead_peer(); }
    CHECK(dead == 1, "crashed peer not reported (got %d)", dead);
    CHECK(ra.send(1, m.data(), m.size()) == Reliable::DEAD, "send to a dead peer must say DEAD");
    CHECK(ms_since(t0) < 1000, "detection took %d ms", ms_since(t0));
  }
  {   // tcp: a peer that stays silent forever (all its frames dropped): retransmit limit ends the wait
    auto t = tcp_pair();
    FaultSpec f; f.drop = 1.0; FaultyTransport silent(*t[1], f);
    ReliableConfig c = fast_cfg(); c.max_retries = 5; c.rto_max_ms = 40;
    Reliable ra(*t[0], c), rb(silent, c);
    auto m = msg_of(2, 10);
    ra.send(1, m.data(), m.size());
    auto t0 = Clock::now(); int dead = -1; Reliable::Delivery d;
    while (dead < 0 && ms_since(t0) < 5000) { ra.poll(d, 5); rb.poll(d, 0); dead = ra.dead_peer(); }
    CHECK(dead == 1, "silent peer not declared dead");
    fprintf(stderr, "  silent peer declared dead after %d ms\n", ms_since(t0));
  }
  {   // tcp: connection closed under us
    auto t = tcp_pair(); Reliable ra(*t[0], fast_cfg()), rb(*t[1], fast_cfg());
    auto m = msg_of(3, 10); Reliable::Delivery d;
    ra.send(1, m.data(), m.size());                      // connections are established lazily: wait until one is really up
    for (int i = 0; i < 400 && !rb.poll(d, 5); ++i) {}
    t[1]->close();
    auto t0 = Clock::now(); int dead = -1;
    while (dead < 0 && ms_since(t0) < 5000) { ra.poll(d, 5); dead = ra.dead_peer(); }
    CHECK(dead == 1, "closed connection not reported");
  }
}

static void suite_window() {
  fprintf(stderr, "== flow control: tiny window, FULL back-pressure, bounded memory\n");
  LoopbackHub hub(2); LoopbackTransport a(hub, 0), b(hub, 1);
  ReliableConfig c = fast_cfg(); c.window_msgs = 3; c.window_bytes = 1 << 20;
  Reliable ra(a, c), rb(b, c);
  int sent = 0, fulls = 0, got = 0; size_t max_unacked = 0;
  while (got < 500) {
    for (int burst = 0; burst < 10 && sent < 500; ++burst) {      // a burst larger than the window must hit FULL
      auto m = msg_of(sent, 50);
      auto s = ra.send(1, m.data(), m.size());
      if (s == Reliable::OK) ++sent; else if (s == Reliable::FULL) ++fulls;
    }
    max_unacked = std::max(max_unacked, ra.unacked(1));
    Reliable::Delivery d;
    if (rb.poll(d, 0)) ++got;
    ra.poll(d, 0);
  }
  CHECK(fulls > 0, "window never filled");
  CHECK(max_unacked <= 3, "window exceeded: %zu", max_unacked);
  fprintf(stderr, "  500 messages through a window of 3: %d FULL stalls, max in flight %zu\n", fulls, max_unacked);
  ReliableConfig cb = fast_cfg(); cb.window_msgs = 1000; cb.window_bytes = 1000;   // byte window: second 600 B message must wait
  LoopbackHub h2(2); LoopbackTransport x(h2, 0), y(h2, 1); Reliable rx(x, cb);
  auto big = msg_of(1, 600);
  CHECK(rx.send(1, big.data(), big.size()) == Reliable::OK, "first");
  CHECK(rx.send(1, big.data(), big.size()) == Reliable::FULL, "byte window");
}

static void suite_wan() {
  fprintf(stderr, "== latency hiding: the send window is the pipeline depth (emulated 10 ms RTT)\n");
  auto run = [](int window_msgs, int n) {
    LoopbackHub hub(2); LoopbackTransport a(hub, 0), b(hub, 1);
    FaultSpec f; f.latency_ms = 5;                                   // 5 ms each way
    FaultyTransport fa(a, f), fb(b, f);
    ReliableConfig c = fast_cfg(); c.window_msgs = window_msgs; c.rto_ms = 200; c.rto_max_ms = 400;
    Reliable ra(fa, c), rb(fb, c);
    auto t0 = Clock::now(); int sent = 0, got = 0;
    while (got < n && ms_since(t0) < 60000) {
      for (int k = 0; k < 64 && sent < n; ++k) { auto m = msg_of(sent, 1024); auto s = ra.send(1, m.data(), m.size()); if (s == Reliable::OK) ++sent; else break; }
      Reliable::Delivery d; if (rb.poll(d, 0)) ++got;
      ra.poll(d, 0);
    }
    CHECK(got == n, "window %d: %d/%d delivered", window_msgs, got, n);
    return (double)ms_since(t0);
  };
  double w1 = run(1, 60), w8 = run(8, 60), w64 = run(64, 60);
  fprintf(stderr, "  60 x 1 KiB over a 10 ms RTT: window 1 = %.0f ms, window 8 = %.0f ms, window 64 = %.0f ms\n", w1, w8, w64);
  CHECK(w1 > 3.0 * w8, "a window of 8 must hide latency (%.0f vs %.0f ms)", w1, w8);
  CHECK(w8 > 1.5 * w64 || w64 < 80, "a deeper window must not be slower (%.0f vs %.0f ms)", w8, w64);
}

// ---------------------------------------------------------------------------------------------- proxy end to end
// A thread plays the DEVICE: it takes tickets, writes the payload, then publishes the slot with release semantics (exactly the
// protocol of kvm_ops_comm.cuh send_proxy). Two such "devices" with HostSinks and TCP/faulty transports run the proxy loops.
struct FakeDevice {
  std::vector<uint8_t> qmem;
  unsigned long long tail = 0;                    // plays the device-memory ticket counter
  KvmSqHdr* hdr() { return (KvmSqHdr*)qmem.data(); }
  FakeDevice(uint32_t slots, uint32_t slot_bytes) : qmem(kvm_sq_bytes(slots, slot_bytes) + 64) { SendQ::init(qmem.data(), slots, slot_bytes, 5ull * 1000000000ull, &tail); }
  // returns false if the queue stayed full past the timeout (the device-side SENDQ_TIMEOUT)
  bool produce(int dst, int dst_buf, int dst_off, int rows, int cols, int var, int ver, const float* data, int timeout_ms = 5000) {
    KvmSqHdr* h = hdr();
    unsigned long long t = __atomic_fetch_add(h->tail_dev, 1, __ATOMIC_RELAXED);
    auto t0 = Clock::now();
    while (t - __atomic_load_n(&h->head, __ATOMIC_ACQUIRE) >= h->slots) { if (ms_since(t0) > timeout_ms) return false; std::this_thread::yield(); }
    uint8_t* slot = qmem.data() + sizeof(KvmSqHdr) + (t % h->slots) * kvm_sq_stride(h->slot_bytes);
    memcpy(slot + KVM_SQ_SLOT_HDR, data, (size_t)rows * cols * 4);
    KvmSqSlot* s = (KvmSqSlot*)slot;
    s->dst_rank = dst; s->dst_buf = dst_buf; s->dst_off = dst_off; s->dst_ld = cols; s->rows = rows; s->cols = cols; s->var = var; s->ver = ver; s->nbytes = rows * cols * 4;
    __atomic_store_n(&s->seq, t + 1, __ATOMIC_RELEASE);
    return true;
  }
};

static void suite_proxy(const char* label, bool tcp, FaultSpec fs, uint32_t slots, int producers, int msgs_per_producer) {
  fprintf(stderr, "== proxy end to end: %s\n", label);
  const int W = 2, NB = 1, ROWS = 512, COLS = 16, NV = 8;
  std::unique_ptr<LoopbackHub> hub; std::vector<std::unique_ptr<Transport>> base(W), wrapped(W);
  if (tcp) { auto t = tcp_pair(); base[0] = std::move(t[0]); base[1] = std::move(t[1]); }
  else { hub = std::make_unique<LoopbackHub>(W); for (int r = 0; r < W; ++r) base[r] = std::make_unique<LoopbackTransport>(*hub, r); }
  bool faulty = fs.drop > 0 || fs.dup > 0 || fs.delay > 0;
  for (int r = 0; r < W; ++r) wrapped[r] = faulty ? std::unique_ptr<Transport>(new FaultyTransport(*base[r], fs)) : std::move(base[r]);
  std::vector<std::vector<float>> mem(W, std::vector<float>(ROWS * COLS * NB, -1.f));
  std::vector<std::vector<std::atomic<unsigned>>> inbox(W);
  for (auto& v : inbox) v = std::vector<std::atomic<unsigned>>(NV);
  for (auto& v : inbox) for (auto& x : v) x = 0;
  std::vector<std::unique_ptr<FakeDevice>> dev; std::vector<std::unique_ptr<Reliable>> rel; std::vector<std::unique_ptr<HostSink>> sink; std::vector<std::unique_ptr<Proxy>> px;
  for (int r = 0; r < W; ++r) {
    dev.push_back(std::make_unique<FakeDevice>(slots, 4096));
    rel.push_back(std::make_unique<Reliable>(*wrapped[r], fast_cfg()));
    sink.push_back(std::make_unique<HostSink>(std::vector<HostBuf>{{mem[r].data(), ROWS, COLS}}, inbox[r].data(), NV));
    px.push_back(std::make_unique<Proxy>(r, W, SendQ(dev[r]->qmem.data()), *rel[r], *sink[r]));
  }
  for (auto& p : px) p->start();
  // each producer thread p of each device sends msgs_per_producer rows-blocks of 4 rows into its own region of the PEER buffer
  const int rows_per = 4;
  CHECK(producers * msgs_per_producer * rows_per <= ROWS, "test sizing");
  auto t0 = Clock::now();
  std::vector<std::thread> prod;
  std::atomic<int> produce_fail{0};
  for (int r = 0; r < W; ++r)
    for (int p = 0; p < producers; ++p)
      prod.emplace_back([&, r, p] {
        for (int m = 0; m < msgs_per_producer; ++m) {
          std::vector<float> data(rows_per * COLS);
          int blk = p * msgs_per_producer + m;
          for (int i = 0; i < rows_per * COLS; ++i) data[i] = (float)(r * 100000 + blk * 100 + i);
          if (!dev[r]->produce(1 - r, 0, blk * rows_per * COLS, rows_per, COLS, p % NV, m + 1, data.data())) ++produce_fail;
        }
      });
  for (auto& t : prod) t.join();
  CHECK(produce_fail == 0, "device-side send queue timed out %d times", produce_fail.load());
  for (auto& p : px) p->kernel_done();
  bool clean = true;
  for (auto& p : px) clean &= p->wait_finished(60000);
  CHECK(clean, "proxy shutdown protocol did not complete");
  for (auto& p : px) p->stop();
  int wrong = 0;
  for (int r = 0; r < W; ++r) {
    int src = 1 - r;                   // r's memory holds what src produced
    for (int p = 0; p < producers; ++p)
      for (int m = 0; m < msgs_per_producer; ++m) {
        int blk = p * msgs_per_producer + m;
        for (int i = 0; i < rows_per * COLS; ++i) wrong += mem[r][blk * rows_per * COLS + i] != (float)(src * 100000 + blk * 100 + i);
      }
    for (int v = 0; v < std::min(producers, NV); ++v) CHECK(inbox[r][v] == (unsigned)msgs_per_producer, "inbox[%d] = %u, want %d", v, inbox[r][v].load(), msgs_per_producer);
    CHECK(sink[r]->fail_code == 0, "sink failure %d", sink[r]->fail_code.load());
  }
  CHECK(wrong == 0, "%d wrong elements", wrong);
  long msgs = (long)W * producers * msgs_per_producer;
  CHECK(px[0]->st.sent + px[1]->st.sent == msgs, "sent %ld of %ld", px[0]->st.sent + px[1]->st.sent, msgs);
  fprintf(stderr, "  %ld messages, queue slots %u, %d ms, retx %ld/%ld\n", msgs, slots, ms_since(t0), rel[0]->st.retx, rel[1]->st.retx);
}

static void suite_proxy_failure() {
  fprintf(stderr, "== proxy: peer dies mid-run -> the device is told (no hang)\n");
  LoopbackHub hub(2); LoopbackTransport a(hub, 0), b(hub, 1);
  std::vector<float> mem(64, 0); std::vector<std::atomic<unsigned>> inbox(4); for (auto& x : inbox) x = 0;
  FakeDevice dev(8, 1024);
  ReliableConfig c = fast_cfg(); c.max_retries = 3;
  Reliable ra(a, c);
  HostSink sink({{mem.data(), 8, 8}}, inbox.data(), 4);
  Proxy px(0, 2, SendQ(dev.qmem.data()), ra, sink);
  px.start();
  float d[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  hub.kill(1);                                           // rank 1 is gone before it ever answers
  dev.produce(1, 0, 0, 1, 8, 0, 1, d);
  auto t0 = Clock::now();
  while (sink.fail_code == 0 && ms_since(t0) < 5000) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  CHECK(sink.fail_code == KVM_ERR_DIST_PEER_DOWN && sink.fail_arg == 1, "device not told about the dead peer (code %d)", sink.fail_code.load());
  px.kernel_done();
  CHECK(px.wait_finished(5000), "proxy must finish even though a peer is dead");
  px.stop();
  fprintf(stderr, "  reported after %d ms\n", ms_since(t0));
}

static void suite_three_ranks() {
  fprintf(stderr, "== 3 ranks over TCP, all-to-all\n");
  auto ports = pick_free_ports(3);
  std::vector<std::unique_ptr<TcpTransport>> t(3); std::vector<std::thread> th;
  for (int r = 0; r < 3; ++r) th.emplace_back([&, r] {
    TcpConfig c; c.rank = r; c.hosts = {"127.0.0.1", "127.0.0.1", "127.0.0.1"}; c.ports = ports; t[r] = std::make_unique<TcpTransport>(c);
  });
  for (auto& x : th) x.join();
  std::vector<std::unique_ptr<Reliable>> rel; for (int r = 0; r < 3; ++r) rel.push_back(std::make_unique<Reliable>(*t[r], fast_cfg()));
  std::vector<std::vector<uint32_t>> got(3); std::atomic<int> done{0};
  std::vector<std::thread> w;
  for (int r = 0; r < 3; ++r) w.emplace_back([&, r] {
    int sent = 0; const int N = 200;
    auto t0 = Clock::now();
    while (ms_since(t0) < 30000) {
      if (sent < 2 * N) { int dst = (r + 1 + sent % 2) % 3; auto m = msg_of(r * 100000 + sent, 40); if (rel[r]->send(dst, m.data(), m.size()) == Reliable::OK) ++sent; }
      Reliable::Delivery d; if (rel[r]->poll(d, 0)) { uint32_t tag; memcpy(&tag, d.data.data(), 4); got[r].push_back(tag); }
      if (sent == 2 * N && (int)got[r].size() == 2 * N && rel[r]->unacked_total() == 0) { ++done; break; }
    }
  });
  for (auto& x : w) x.join();
  CHECK(done == 3, "only %d/3 ranks completed", done.load());
  for (int r = 0; r < 3; ++r) {   // per source, tags must arrive in increasing order
    std::map<int, uint32_t> last; bool ok = true;
    for (uint32_t tag : got[r]) { int src = tag / 100000; ok &= !last.count(src) || tag > last[src]; last[src] = tag; }
    CHECK(ok, "rank %d: per-source order violated", r);
  }
}

// -------------------------------------------------------------------------------------------------- multi-process mode
static int proc_main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: proc RANK WORLD PORTS[,drop=..,dup=..,delay=..]\n"); return 2; }
  int rank = atoi(argv[2]), world = atoi(argv[3]);
  std::vector<int> ports; FaultSpec f; std::stringstream ss(argv[4]); std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.rfind("drop=", 0) == 0) f.drop = atof(tok.c_str() + 5);
    else if (tok.rfind("dup=", 0) == 0) f.dup = atof(tok.c_str() + 4);
    else if (tok.rfind("delay=", 0) == 0) f.delay = atof(tok.c_str() + 6);
    else ports.push_back(atoi(tok.c_str()));
  }
  const char* hosts_env = getenv("KVM_PEER_HOSTS");     // comma separated, one per rank; default all loopback
  std::vector<std::string> hosts(world, "127.0.0.1");
  if (hosts_env) { std::stringstream hs(hosts_env); for (int i = 0; i < world && std::getline(hs, tok, ','); ++i) hosts[i] = tok; }
  TcpConfig c; c.rank = rank; c.hosts = hosts; c.ports = ports; c.cluster_id = 7;
  TcpTransport tcp(c);
  FaultyTransport faulty(tcp, f);
  Transport& t = (f.drop > 0 || f.dup > 0 || f.delay > 0) ? (Transport&)faulty : (Transport&)tcp;
  Reliable r(t, fast_cfg());
  const int N = 500, SZ = 256;
  int next = (rank + 1) % world, prev = (rank + world - 1) % world;
  int sent = 0; std::vector<uint32_t> got; auto t0 = Clock::now();
  while (ms_since(t0) < 60000) {
    if (sent < N) { auto m = msg_of(rank * 1000000u + sent, SZ); if (r.send(next, m.data(), m.size()) == Reliable::OK) ++sent; }
    Reliable::Delivery d;
    if (r.poll(d, sent < N ? 0 : 2)) { uint32_t tag; memcpy(&tag, d.data.data(), 4); if (d.data != msg_of(tag, SZ)) { fprintf(stderr, "rank %d: corrupt payload\n", rank); return 1; } got.push_back(tag); }
    if (sent == N && (int)got.size() == N && r.unacked_total() == 0) break;
  }
  bool ok = (int)got.size() == N;
  for (size_t i = 0; i < got.size(); ++i) ok &= got[i] == (uint32_t)prev * 1000000u + (uint32_t)i;
  // linger so our last ACKs reach the peer before the sockets close
  for (auto end = Clock::now() + std::chrono::milliseconds(300); Clock::now() < end;) { Reliable::Delivery d; r.poll(d, 5); }
  fprintf(stderr, "proc rank %d/%d: %s (%d msgs in order, retx %ld, dups %ld)\n", rank, world, ok ? "PASS" : "FAIL", (int)got.size(), r.st.retx, r.st.dups);
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "proc") return proc_main(argc, argv);
  suite_crc();
  suite_loopback();
  suite_tcp();
  suite_failure();
  suite_window();
  suite_wan();
  { FaultSpec none; suite_proxy("loopback, 3 producer threads, 4-slot queue (wraps)", false, none, 4, 3, 10); }
  { FaultSpec f; f.drop = 0.1; f.dup = 0.1; f.delay = 0.3; f.seed = 5; suite_proxy("tcp + drop/dup/delay, 16-slot queue", true, f, 16, 4, 8); }
  suite_proxy_failure();
  suite_three_ranks();
  printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail.load());
  return g_fail ? 1 : 0;
}
