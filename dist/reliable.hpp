#pragma once
// Reliable, ordered, flow-controlled delivery over any (possibly lossy) Transport, with failure detection.
//
//   * Per peer, messages carry a sequence number (1, 2, ...). The receiver delivers strictly in order, exactly once: duplicates
//     are dropped (and re-acknowledged), out-of-order messages wait in a reorder buffer. Order is therefore deterministic per
//     (source, destination) pair, which is what makes landing buffers and one-shot version signals reproducible.
//   * Cumulative ACKs (piggybacked on data, plus an explicit ACK per received batch). The sender retransmits the oldest
//     unacknowledged message after an RTO that doubles up to rto_max_ms; more than max_retries retransmissions of one message
//     declare the peer DEAD.
//   * Flow control = a send window (messages and bytes) per peer. send() returns FULL instead of blocking: the proxy then stops
//     draining the device send queue, the queue fills, and the producing task waits with a timeout on the device
//     (KVM_ERR_DIST_SENDQ_TIMEOUT). Nothing buffers without bound anywhere. For a WAN link, set window_bytes >= the
//     bandwidth-delay product (bandwidth x RTT) or the link idles waiting for ACKs: that IS the pipeline-depth parameter.
//   * Liveness: idle links exchange PINGs; a peer silent for dead_ms after first contact, or never heard from within
//     startup_grace_ms while we have data for it, is DEAD. The first dead peer is reported once via dead_peer().
//
// Single-threaded protocol engine: send() and poll() must be called from the same thread (the proxy loop).
#include <algorithm>
#include <cmath>
#include <map>

#include "transport.hpp"

namespace kvmd {

struct ReliableConfig {
  int window_msgs = 256;
  size_t window_bytes = 64u << 20;
  int rto_ms = 100, rto_max_ms = 2000, max_retries = 12;
  int ping_ms = 500, dead_ms = 10000, startup_grace_ms = 60000;
  // Optional (default off = the fixed rto_ms above, i.e. unchanged behaviour): once RTT samples exist, the base RTO follows
  // SRTT + 4 * RTTVAR (RFC 6298 form), clamped to [rto_min_ms, rto_max_ms]. Needed on a WAN, where a fixed 100 ms RTO is wrong.
  bool rto_adaptive = false;
  int rto_min_ms = 10;
};

struct ReliableStats { long sent = 0, retx = 0, dups = 0, reordered = 0, delivered = 0, acks_in = 0, bad = 0; unsigned long long bytes_sent = 0; };

// Per-peer transport measurements. RTT samples come only from acks that cover NO retransmitted message (Karn's algorithm), taking the
// newest covered message (the one with the least queueing). rate_Bps = payload bytes newly acknowledged per second over the last
// completed measurement interval (>= max(20 ms, SRTT)); it is a delivery rate, not a send rate.
struct RttInfo { double srtt_ms = 0, rttvar_ms = 0, min_rtt_ms = 0, rate_Bps = 0; long samples = 0; unsigned long long acked_bytes = 0; };

class Reliable {
 public:
  enum Status { OK, FULL, DEAD };
  struct Delivery { int src; std::vector<uint8_t> data; };

  Reliable(Transport& t, ReliableConfig c) : t_(t), c_(c), peers_(t.world()), start_(Clock::now()) {}

  Status send(int dst, const void* p, size_t n) {
    Peer& q = peers_.at(dst);
    if (q.dead) return DEAD;
    if ((int)q.unacked.size() >= c_.window_msgs || (q.unacked_bytes + n > c_.window_bytes && !q.unacked.empty())) return FULL;
    Pending m;
    m.seq = q.next_seq++;
    m.pkt = encode(DATA, m.seq, q.rx_next, p, n);
    m.sent = m.last = Clock::now();
    q.unacked_bytes += n;
    q.unacked.push_back(std::move(m));
    q.last_tx = Clock::now();
    ++st.sent; st.bytes_sent += n;
    if (!t_.send(dst, q.unacked.back().pkt.data(), q.unacked.back().pkt.size())) declare_dead(dst, "transport refused send");
    return q.dead ? DEAD : OK;
  }

  // Runs the protocol for up to timeout_ms; true when a message was delivered (in order, exactly once).
  bool poll(Delivery& d, int timeout_ms) {
    auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
      if (!ready_.empty()) { d = std::move(ready_.front()); ready_.pop_front(); ++st.delivered; return true; }
      Event ev;
      int slice = std::max(0, std::min(5, (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count()));
      bool got = t_.poll(ev, slice);
      while (got) {
        handle(ev);
        got = t_.poll(ev, 0);
      }
      flush_acks();
      timers();
      if (!ready_.empty()) continue;
      if (Clock::now() >= end) return false;
    }
  }

  bool dead(int peer) const { return peers_.at(peer).dead; }
  int dead_peer() { int r = reported_; reported_ = -1; return r; }       // each dead peer is reported once
  size_t unacked(int peer) const { return peers_.at(peer).unacked.size(); }
  size_t unacked_total() const { size_t n = 0; for (auto& p : peers_) n += p.unacked.size(); return n; }
  const ReliableConfig& config() const { return c_; }
  // Change the send window at run time (all peers; call from the protocol thread). msgs <= 0 keeps the message window.
  void set_window(size_t bytes, int msgs = 0) { c_.window_bytes = bytes; if (msgs > 0) c_.window_msgs = msgs; }
  // Diagnostic snapshot of one peer's protocol state.
  std::string debug(int peer) const {
    const Peer& q = peers_.at(peer); char b[256];
    snprintf(b, sizeof b, "peer %d: next_seq %u rx_next %u unacked %zu (first seq %u, retries %d) unacked_bytes %zu reorder %zu dead %d",
             peer, q.next_seq, q.rx_next, q.unacked.size(), q.unacked.empty() ? 0u : q.unacked.front().seq, q.unacked.empty() ? 0 : q.unacked.front().retries, q.unacked_bytes, q.reorder.size(), (int)q.dead);
    return b;
  }
  RttInfo rtt(int peer) const {
    const Peer& q = peers_.at(peer); RttInfo r;
    r.srtt_ms = q.srtt_us / 1000.0; r.rttvar_ms = q.rttvar_us / 1000.0; r.min_rtt_ms = q.min_rtt_us / 1000.0;
    r.rate_Bps = q.rate_Bps; r.samples = q.rtt_samples; r.acked_bytes = q.acked_bytes;
    return r;
  }
  ReliableStats st;

 private:
  enum Type : uint8_t { DATA = 1, ACK = 2, PING = 3 };
  struct Hdr { uint8_t type, pad[3]; uint32_t src, seq, ack; };  // 16 bytes, little-endian hosts
  struct Pending { uint32_t seq; std::vector<uint8_t> pkt; Clock::time_point sent, last; int retries = 0; };
  struct Peer {
    uint32_t next_seq = 1, rx_next = 1;
    std::deque<Pending> unacked; size_t unacked_bytes = 0;
    std::map<uint32_t, std::vector<uint8_t>> reorder;
    bool dead = false, contacted = false, ack_due = false;
    Clock::time_point last_rx, last_tx;
    // measurements (see RttInfo)
    double srtt_us = 0, rttvar_us = 0, min_rtt_us = 0, rate_Bps = 0; long rtt_samples = 0; unsigned long long acked_bytes = 0, rate_b0 = 0;
    Clock::time_point min_rtt_at, rate_t0;
  };

  std::vector<uint8_t> encode(Type ty, uint32_t seq, uint32_t ack, const void* p, size_t n) {
    std::vector<uint8_t> b(sizeof(Hdr) + n);
    Hdr h{}; h.type = ty; h.src = (uint32_t)t_.rank(); h.seq = seq; h.ack = ack;
    memcpy(b.data(), &h, sizeof h);
    if (n) memcpy(b.data() + sizeof h, p, n);
    return b;
  }
  void declare_dead(int peer, const char* why) {
    Peer& q = peers_[peer];
    if (q.dead) return;
    q.dead = true;
    if (reported_ < 0) reported_ = peer;
    KD_LOG(0, "reliable", "rank %d: peer %d DEAD (%s); %zu unacked", t_.rank(), peer, why, q.unacked.size());
  }
  void ack_upto(Peer& q, uint32_t ack) {
    auto now = Clock::now();
    bool retransmitted = false; double sample_us = -1;
    while (!q.unacked.empty() && q.unacked.front().seq < ack) {
      const Pending& m = q.unacked.front();
      size_t pay = m.pkt.size() - sizeof(Hdr);
      q.unacked_bytes -= pay; q.acked_bytes += pay;
      if (m.retries > 0) retransmitted = true;
      else sample_us = std::chrono::duration<double, std::micro>(now - m.sent).count();      // ends as the newest covered message
      q.unacked.pop_front();
      ++st.acks_in;
    }
    if (sample_us >= 0 && !retransmitted) rtt_sample(q, sample_us, now);       // Karn: an ack covering a retransmission is ambiguous
    if (q.rate_t0 == Clock::time_point()) { q.rate_t0 = now; q.rate_b0 = q.acked_bytes; }
    double dt = std::chrono::duration<double>(now - q.rate_t0).count();
    if (dt >= std::max(0.020, q.srtt_us * 1e-6)) { q.rate_Bps = (double)(q.acked_bytes - q.rate_b0) / dt; q.rate_t0 = now; q.rate_b0 = q.acked_bytes; }
  }
  void rtt_sample(Peer& q, double us, Clock::time_point now) {
    if (q.rtt_samples++ == 0) { q.srtt_us = us; q.rttvar_us = us / 2; }
    else { q.rttvar_us = 0.75 * q.rttvar_us + 0.25 * std::abs(q.srtt_us - us); q.srtt_us = 0.875 * q.srtt_us + 0.125 * us; }
    // windowed minimum: a new minimum, or the old one is older than 10 s (routes change)
    if (q.min_rtt_us == 0 || us <= q.min_rtt_us || std::chrono::duration_cast<std::chrono::seconds>(now - q.min_rtt_at).count() > 10) { q.min_rtt_us = us; q.min_rtt_at = now; }
  }
  int base_rto_ms(const Peer& q) const {
    if (!c_.rto_adaptive || q.rtt_samples == 0) return c_.rto_ms;
    int ms = (int)std::ceil((q.srtt_us + 4 * q.rttvar_us) / 1000.0);
    return std::max(c_.rto_min_ms, std::min(c_.rto_max_ms, ms));
  }
  void handle(const Event& ev) {
    if (ev.type == EvType::PEER_DOWN) { declare_dead(ev.peer, "transport reported the connection lost"); return; }
    if (ev.data.size() < sizeof(Hdr)) { ++st.bad; return; }
    Hdr h; memcpy(&h, ev.data.data(), sizeof h);
    if ((int)h.src != ev.peer || h.src >= peers_.size()) { ++st.bad; return; }
    Peer& q = peers_[h.src];
    q.last_rx = Clock::now(); q.contacted = true;
    ack_upto(q, h.ack);
    if (h.type == PING) { q.ack_due = true; return; }
    if (h.type != DATA) return;
    q.ack_due = true;
    if (h.seq < q.rx_next || q.reorder.count(h.seq)) { ++st.dups; KD_LOG(4, "reliable", "rank %d: duplicate seq %u from %u", t_.rank(), h.seq, h.src); return; }
    std::vector<uint8_t> body(ev.data.begin() + sizeof(Hdr), ev.data.end());
    if (h.seq != q.rx_next) { ++st.reordered; q.reorder.emplace(h.seq, std::move(body)); return; }
    ready_.push_back({(int)h.src, std::move(body)}); ++q.rx_next;
    for (auto it = q.reorder.find(q.rx_next); it != q.reorder.end(); it = q.reorder.find(q.rx_next)) {
      ready_.push_back({(int)h.src, std::move(it->second)}); q.reorder.erase(it); ++q.rx_next;
    }
  }
  void flush_acks() {
    for (size_t i = 0; i < peers_.size(); ++i) {
      Peer& q = peers_[i];
      if (!q.ack_due || q.dead) continue;
      q.ack_due = false;
      auto b = encode(ACK, 0, q.rx_next, nullptr, 0);
      t_.send((int)i, b.data(), b.size());
    }
  }
  void timers() {
    auto now = Clock::now();
    for (size_t i = 0; i < peers_.size(); ++i) {
      if ((int)i == t_.rank()) continue;
      Peer& q = peers_[i];
      if (q.dead) continue;
      for (auto& m : q.unacked) {
        int rto = std::min(c_.rto_max_ms, base_rto_ms(q) << std::min(m.retries, 10));
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - m.last).count() < rto) break;   // oldest first
        if (m.retries >= c_.max_retries) { declare_dead((int)i, "retransmit limit"); break; }
        // refresh the piggybacked ack, resend, back off
        uint32_t ack = q.rx_next; memcpy(m.pkt.data() + offsetof(Hdr, ack), &ack, 4);
        ++m.retries; ++st.retx; m.last = now;
        KD_LOG(3, "reliable", "rank %d: retransmit seq %u to %zu (try %d)", t_.rank(), m.seq, i, m.retries);
        if (!t_.send((int)i, m.pkt.data(), m.pkt.size())) { declare_dead((int)i, "transport refused retransmit"); break; }
      }
      if (q.dead) continue;
      if (q.unacked.empty() && q.contacted && std::chrono::duration_cast<std::chrono::milliseconds>(now - q.last_tx).count() > c_.ping_ms) {
        auto b = encode(PING, 0, q.rx_next, nullptr, 0); q.last_tx = now; t_.send((int)i, b.data(), b.size());
      }
      if (q.contacted && std::chrono::duration_cast<std::chrono::milliseconds>(now - q.last_rx).count() > c_.dead_ms) declare_dead((int)i, "silent past dead_ms");
      else if (!q.contacted && !q.unacked.empty() && ms_since(start_) > c_.startup_grace_ms) declare_dead((int)i, "never answered");
    }
  }

  Transport& t_;
  ReliableConfig c_;
  std::vector<Peer> peers_;
  std::deque<Delivery> ready_;
  Clock::time_point start_;
  int reported_ = -1;
};

}  // namespace kvmd
