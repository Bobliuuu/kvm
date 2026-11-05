#pragma once
// Cross-datacenter (long fat pipe) protocol pieces on top of Reliable. Pure C++17, no CUDA.
//
//   AdaptiveWindow       sets Reliable's send window to a multiple of the measured bandwidth-delay product.
//   PublishBatcher       coalesces version publishes (var, ver) per destination into one wire message.
//   BottleneckTransport  TEST/EMULATION decorator: a finite-bandwidth egress link with a tail-drop buffer. Together with
//                        FaultyTransport{latency_ms} it emulates a WAN link (latency x bandwidth x buffer). It is an emulation:
//                        no congestion from other flows, no jitter, one queue.
//
// ------------------------------------------------------------------------------------------------------ AdaptiveWindow
// Algorithm: "BBR-lite" window control driven by Reliable's measurements (reliable.hpp RttInfo).
//   * Every round (max(min_rtt, 5 ms)) it computes the delivery rate = payload bytes newly acknowledged / round time, and keeps the
//     MAX over the last `bw_rounds` rounds (bw estimate). min_rtt is Reliable's windowed minimum RTT over acks of never-
//     retransmitted messages (Karn), so queueing delay does not inflate the BDP.
//   * window = clamp(gain * bw * min_rtt * loss_scale, min_bytes, max_bytes), and never more than 2x the previous window per round.
//     Startup (slow start) begins at init_bytes; because the delivery rate while window-limited is window / RTT, the target
//     2 * bw * min_rtt DOUBLES the window each RTT until the rate stops growing (< 25 % growth for 3 rounds): that is the
//     bottleneck. Then the steady gain (default 1.25) applies: a small standing queue, no big overshoot.
//   * Loss: a round with retransmissions multiplies loss_scale by loss_beta (floor loss_floor); a clean round adds 0.05 back. Loss
//     therefore shrinks the window at most to loss_floor x BDP, it does not collapse it (a lossy long link still needs depth).
//   * The message-count window is set to ~2 x window / average message size, clamped [min_msgs, max_msgs].
// Call update() often (every protocol-loop iteration; it is a few comparisons when no round has ended). Same thread as Reliable.
// Limits: the estimator is only as good as the acks: a receiver that acks late (Reliable acks per received batch, so fine) or an
// app-limited sender reports a low rate and the window then shrinks until traffic resumes (it re-ramps within a few RTTs).
//
// ----------------------------------------------------------------------------------------------------- PublishBatcher
// Ordering rule (why batching is safe): version signals are monotone per variable (DISTRIBUTED.md: single writer), and the data a
// signal announces is sent on the same ordered, exactly-once Reliable stream BEFORE the signal. Coalescing keeps only the highest
// version per var (a lower version is implied by a higher one) and only DELAYS signals; it can never move one in front of its data
// provided that
//    (1) publish(dst, var, ver) is called only AFTER the data message(s) for (var, ver) were accepted (Reliable::OK) for the SAME
//        dst. put_signal() below does exactly that and refuses to publish if the data send returned FULL / DEAD;
//    (2) the batch goes out on that same stream, so it is delivered after every data message accepted before the flush;
//    (3) a batch is flushed (flush(dst)/flush_all()) before anything that must not overtake signals, e.g. the shutdown FIN of the
//        proxy protocol: a FIN sent while signals are still parked would let the peer finish without them.
// A signal parked in a batch for at most max_delay_us (tick() enforces it; call it from the loop) delays the consumer by at most that.
// If Reliable returns FULL for the batch the batch stays parked and is retried by tick(); newer publishes keep coalescing into it.
// Wire format of a batch: u8 kind(0xB5) | u8 version(1) | u16 npairs | npairs x (u32 var, u32 ver), little-endian hosts.
#include <deque>
#include <functional>
#include <map>

#include "reliable.hpp"

namespace kvmd {

constexpr uint8_t XDC_DATA = 0xD1, XDC_SIGNALS = 0xB5;

// ------------------------------------------------------------------------------------------------------ AdaptiveWindow
struct AdaptiveWindowConfig {
  size_t init_bytes = 32u << 10, min_bytes = 16u << 10, max_bytes = 256u << 20;
  double startup_gain = 2.0, gain = 1.25;
  int bw_rounds = 10;
  double loss_beta = 0.8, loss_floor = 0.5;
  int min_msgs = 8, max_msgs = 4096;
};

class AdaptiveWindow {
 public:
  AdaptiveWindow(Reliable& r, int peer, AdaptiveWindowConfig c = {}) : r_(r), peer_(peer), c_(c), window_(c.init_bytes) { apply(); }
  void update() {
    RttInfo ri = r_.rtt(peer_);
    if (ri.samples == 0) return;
    auto now = Clock::now();
    if (!started_) { started_ = true; t0_ = now; acked0_ = ri.acked_bytes; retx0_ = r_.st.retx; sent_msgs0_ = r_.st.sent; sent_bytes0_ = r_.st.bytes_sent; return; }
    double round_s = std::max(ri.min_rtt_ms, 5.0) / 1000.0;
    double dt = std::chrono::duration<double>(now - t0_).count();
    if (dt < round_s) return;
    double rate = (double)(ri.acked_bytes - acked0_) / dt;
    rates_.push_back(rate); if ((int)rates_.size() > c_.bw_rounds) rates_.pop_front();
    bw_ = *std::max_element(rates_.begin(), rates_.end());
    long retx = r_.st.retx - retx0_;
    if (startup_) {
      if (bw_ > full_bw_ * 1.25) { full_bw_ = bw_; stall_ = 0; } else if (++stall_ >= 3) { startup_ = false; KD_LOG(3, "xdc", "adaptive window: startup over, bw %.2f MB/s", bw_ / 1e6); }
    }
    loss_scale_ = retx > 0 ? std::max(c_.loss_floor, loss_scale_ * c_.loss_beta) : std::min(1.0, loss_scale_ + 0.05);
    double target = (startup_ ? c_.startup_gain : c_.gain) * bw_ * (ri.min_rtt_ms / 1000.0) * loss_scale_;
    double capped = std::min(target, 2.0 * (double)window_);
    window_ = (size_t)std::max<double>((double)c_.min_bytes, std::min<double>((double)c_.max_bytes, capped));
    long dm = r_.st.sent - sent_msgs0_; unsigned long long db = r_.st.bytes_sent - sent_bytes0_;
    if (dm > 0) avg_msg_ = std::max<double>(1.0, (double)db / (double)dm);
    ++rounds;
    apply();
    t0_ = now; acked0_ = ri.acked_bytes; retx0_ = r_.st.retx; sent_msgs0_ = r_.st.sent; sent_bytes0_ = r_.st.bytes_sent;
  }
  size_t window_bytes() const { return window_; }
  double bw_Bps() const { return bw_; }
  bool in_startup() const { return startup_; }
  double loss_scale() const { return loss_scale_; }
  size_t peak_window = 0;
  long rounds = 0;
 private:
  void apply() {
    int msgs = (int)std::max<double>(c_.min_msgs, std::min<double>(c_.max_msgs, 2.0 * (double)window_ / avg_msg_ + 4));
    r_.set_window(window_, msgs);
    peak_window = std::max(peak_window, window_);
  }
  Reliable& r_; int peer_; AdaptiveWindowConfig c_;
  size_t window_;
  bool started_ = false, startup_ = true;
  Clock::time_point t0_;
  unsigned long long acked0_ = 0, sent_bytes0_ = 0; long retx0_ = 0, sent_msgs0_ = 0;
  std::deque<double> rates_; double bw_ = 0, full_bw_ = 0, loss_scale_ = 1.0, avg_msg_ = 4096; int stall_ = 0;
};

// ----------------------------------------------------------------------------------------------------- PublishBatcher
struct BatcherConfig {
  size_t max_pairs = 64;      // flush when this many distinct vars are parked for one destination
  size_t max_bytes = 4096;    // ... or when the wire message reaches this size
  int max_delay_us = 2000;    // ... or when the oldest parked signal is this old (tick())
};

class PublishBatcher {
 public:
  using SendFn = std::function<Reliable::Status(int dst, const void* p, size_t n)>;
  using Pairs = std::vector<std::pair<uint32_t, uint32_t>>;
  PublishBatcher(SendFn send, BatcherConfig c = {}) : send_(std::move(send)), c_(c) {}

  // Park (var, ver) for dst. A lower or equal version than the one already parked is dropped (monotone: implied).
  // Precondition: the data for (var, ver) was already accepted by the same stream (see the ordering rule above).
  void publish(int dst, uint32_t var, uint32_t ver, Clock::time_point now = Clock::now()) {
    Batch& b = batch_[dst];
    ++st.published;
    auto it = b.pairs.find(var);
    if (it != b.pairs.end()) { ++st.coalesced; if (ver > it->second) it->second = ver; }
    else { if (b.pairs.empty()) b.first = now; b.pairs.emplace(var, ver); }
    if (b.pairs.size() >= c_.max_pairs || wire_bytes(b.pairs.size()) >= c_.max_bytes) { if (flush(dst) == Reliable::OK) ++st.flush_full; }
  }
  // Data message then signal, in that order on one stream. FULL / DEAD from the data send: nothing is published.
  Reliable::Status put_signal(int dst, const void* msg, size_t n, uint32_t var, uint32_t ver, Clock::time_point now = Clock::now()) {
    Reliable::Status s = send_(dst, msg, n);
    if (s != Reliable::OK) return s;
    publish(dst, var, ver, now);
    return Reliable::OK;
  }
  // Send the parked batch for dst. FULL: kept, retry later. DEAD: dropped.
  Reliable::Status flush(int dst) {
    auto it = batch_.find(dst);
    if (it == batch_.end() || it->second.pairs.empty()) return Reliable::OK;
    Batch& b = it->second;
    std::vector<uint8_t> w = encode(b.pairs);
    Reliable::Status s = send_(dst, w.data(), w.size());
    if (s == Reliable::FULL) { ++st.full_retries; return s; }
    if (s == Reliable::OK) { ++st.msgs; st.pairs += b.pairs.size(); st.wire_bytes += w.size(); }
    b.pairs.clear();
    return s;
  }
  Reliable::Status flush_all() {
    Reliable::Status worst = Reliable::OK;
    for (auto& kv : batch_) { Reliable::Status s = flush(kv.first); if (s != Reliable::OK) worst = s; }
    return worst;
  }
  // Flush every batch whose oldest signal reached max_delay (and retry batches that were FULL).
  void tick(Clock::time_point now = Clock::now()) {
    for (auto& kv : batch_) {
      Batch& b = kv.second;
      if (b.pairs.empty()) continue;
      if (now - b.first >= std::chrono::microseconds(c_.max_delay_us)) { if (flush(kv.first) == Reliable::OK) ++st.flush_delay; }
    }
  }
  size_t pending(int dst) const { auto it = batch_.find(dst); return it == batch_.end() ? 0 : it->second.pairs.size(); }

  static std::vector<uint8_t> encode(const std::map<uint32_t, uint32_t>& pairs) {
    std::vector<uint8_t> w(4 + 8 * pairs.size());
    w[0] = XDC_SIGNALS; w[1] = 1; uint16_t n = (uint16_t)pairs.size(); memcpy(&w[2], &n, 2);
    size_t o = 4;
    for (auto& kv : pairs) { memcpy(&w[o], &kv.first, 4); memcpy(&w[o + 4], &kv.second, 4); o += 8; }
    return w;
  }
  // Receiving side. false: not a well-formed batch (out is left empty).
  static bool decode(const void* p, size_t n, Pairs& out) {
    out.clear();
    const uint8_t* b = (const uint8_t*)p;
    if (n < 4 || b[0] != XDC_SIGNALS || b[1] != 1) return false;
    uint16_t cnt; memcpy(&cnt, b + 2, 2);
    if (n != 4 + 8u * cnt) return false;
    for (uint16_t i = 0; i < cnt; ++i) { uint32_t v, e; memcpy(&v, b + 4 + 8 * i, 4); memcpy(&e, b + 8 + 8 * i, 4); out.emplace_back(v, e); }
    return true;
  }
  struct Stats { long published = 0, coalesced = 0, msgs = 0, pairs = 0, flush_full = 0, flush_delay = 0, full_retries = 0; unsigned long long wire_bytes = 0; } st;

 private:
  struct Batch { std::map<uint32_t, uint32_t> pairs; Clock::time_point first; };
  static size_t wire_bytes(size_t pairs) { return 4 + 8 * pairs; }
  SendFn send_; BatcherConfig c_;
  std::map<int, Batch> batch_;
};

// ------------------------------------------------------------------------------------------------ BottleneckTransport
// EMULATION. Egress link of `rate_Bps` bytes/s: a message departs when the link has serialised everything before it, and is passed
// to the inner transport then. If more than `buffer_bytes` are queued the message is DROPPED (tail drop; 0 = unbounded buffer).
// Wire overhead of the inner transport is not modelled.
class BottleneckTransport : public Transport {
 public:
  BottleneckTransport(Transport& inner, double rate_Bps, size_t buffer_bytes = 0) : in_(inner), rate_(rate_Bps), buf_(buffer_bytes) {}
  int rank() const override { return in_.rank(); }
  int world() const override { return in_.world(); }
  const char* name() const override { return "bottleneck"; }
  bool send(int dst, const void* p, size_t n) override {
    std::lock_guard<std::mutex> l(m_);
    auto now = Clock::now();
    if (free_at_ < now) free_at_ = now;
    double queued = std::chrono::duration<double>(free_at_ - now).count() * rate_;
    if (buf_ && queued + (double)n > (double)buf_) { ++dropped; return true; }
    free_at_ += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>((double)n / rate_));
    held_.push_back({free_at_, dst, std::vector<uint8_t>((const uint8_t*)p, (const uint8_t*)p + n)});
    return true;
  }
  bool poll(Event& ev, int timeout_ms) override {
    auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
      bool any = flush();
      int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
      if (in_.poll(ev, std::max(0, std::min(any ? 1 : timeout_ms, left)))) return true;
      if (Clock::now() >= end) return false;
    }
  }
  void close() override { in_.close(); }
  std::atomic<long> dropped{0};
 private:
  struct Held { Clock::time_point due; int dst; std::vector<uint8_t> data; };
  bool flush() {            // true when messages are still held
    std::lock_guard<std::mutex> l(m_);
    while (!held_.empty() && held_.front().due <= Clock::now()) { in_.send(held_.front().dst, held_.front().data.data(), held_.front().data.size()); held_.pop_front(); }
    return !held_.empty();
  }
  Transport& in_; double rate_; size_t buf_;
  std::mutex m_; std::deque<Held> held_; Clock::time_point free_at_{};
};

}  // namespace kvmd
