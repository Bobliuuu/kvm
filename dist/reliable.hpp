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
#include <map>

#include "transport.hpp"

namespace kvmd {

struct ReliableConfig {
  int window_msgs = 256;
  size_t window_bytes = 64u << 20;
  int rto_ms = 100, rto_max_ms = 2000, max_retries = 12;
  int ping_ms = 500, dead_ms = 10000, startup_grace_ms = 60000;
};

struct ReliableStats { long sent = 0, retx = 0, dups = 0, reordered = 0, delivered = 0, acks_in = 0, bad = 0; };

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
    ++st.sent;
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
    while (!q.unacked.empty() && q.unacked.front().seq < ack) {
      q.unacked_bytes -= q.unacked.front().pkt.size() - sizeof(Hdr);
      q.unacked.pop_front();
      ++st.acks_in;
    }
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
        int rto = std::min(c_.rto_max_ms, c_.rto_ms << std::min(m.retries, 10));
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
