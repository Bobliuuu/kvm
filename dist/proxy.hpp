#pragma once
// Host proxy: one thread per device. Drains the device-written send queue (wire.hpp), moves messages over a Reliable channel,
// and applies received messages to the local device through a Sink (data first, then the version signal).
//
// Pure C++ (the CUDA Sink lives in cuda_dist.hpp), so the whole path is testable on a laptop with a HostSink and a thread that
// emulates the device producers with the same queue protocol.
//
// Shutdown protocol (no rank leaves while a peer may still send to it): the owner calls kernel_done() once its kernel has ended
// (so no more send-queue tickets). The proxy then drains the queue, waits until every message it sent is acknowledged, sends a
// FIN to each peer, and finishes when it has received a FIN from every peer. FINs ride the same in-order reliable stream as the
// data, so receiving peer P's FIN means all of P's data for us has been delivered to the Sink. With a dead peer the wait ends
// at the peer's deadline instead: the proxy never waits forever.
#include <atomic>

#include "reliable.hpp"
#include "wire.hpp"

namespace kvmd {

class Sink {
 public:
  virtual ~Sink() = default;
  // Apply one message: write the tile, then (only after the data is visible) publish (var, ver). Returns 0 or a KVM_ERR_DIST_*.
  virtual int deliver(int src, const KvmMsgHdr& h, const uint8_t* payload) = 0;
  virtual void flush() = 0;                         // everything delivered so far is now visible on the device
  virtual void fail(int code, int arg) = 0;         // tell the device (its pump raises the error)
  virtual void heartbeat() {}
};

// Consumer side of the device send queue. `base` points at a KvmSqHdr in mapped pinned host memory.
class SendQ {
 public:
  explicit SendQ(void* base = nullptr) : base_((uint8_t*)base) {}
  static void init(void* base, uint32_t slots, uint32_t slot_bytes, uint64_t timeout_ns) {
    if (slot_bytes % 64) throw std::invalid_argument("slot_bytes must be a multiple of 64");
    memset(base, 0, kvm_sq_bytes(slots, slot_bytes));
    KvmSqHdr* h = (KvmSqHdr*)base;
    h->slots = slots; h->slot_bytes = slot_bytes; h->version = 1; h->timeout_ns = timeout_ns;
    __atomic_store_n(&h->magic, KVM_SQ_MAGIC, __ATOMIC_RELEASE);
  }
  bool valid() const { return base_ != nullptr; }
  // The next slot in ticket order, if its producer has published it. Acquire pairs with the device's st.release.sys.
  bool peek(const KvmSqSlot*& s, const uint8_t*& payload) {
    KvmSqHdr* h = hdr();
    const KvmSqSlot* sl = slot(head_ % h->slots);
    if (__atomic_load_n(&sl->seq, __ATOMIC_ACQUIRE) != head_ + 1) return false;
    s = sl; payload = (const uint8_t*)sl + KVM_SQ_SLOT_HDR;
    return true;
  }
  void pop() { ++head_; __atomic_store_n(&hdr()->head, head_, __ATOMIC_RELEASE); }
  uint64_t tickets() const { return __atomic_load_n(&hdr()->tail, __ATOMIC_ACQUIRE); }
  uint64_t head() const { return head_; }
  uint32_t slot_bytes() const { return hdr()->slot_bytes; }
  KvmSqHdr* hdr() const { return (KvmSqHdr*)base_; }
 private:
  const KvmSqSlot* slot(uint64_t i) const { return (const KvmSqSlot*)(base_ + sizeof(KvmSqHdr) + i * kvm_sq_stride(hdr()->slot_bytes)); }
  uint8_t* base_;
  uint64_t head_ = 0;
};

struct ProxyStats { long sent = 0, delivered = 0, dropped_dead = 0, rejected = 0, full_stalls = 0; unsigned long long bytes_out = 0, bytes_in = 0; };

class Proxy {
 public:
  Proxy(int rank, int world, SendQ q, Reliable& rel, Sink& sink) : rank_(rank), world_(world), q_(q), rel_(rel), sink_(sink), fin_seen_(world, false) { fin_seen_[rank] = true; }
  ~Proxy() { stop(); }

  void start() { th_ = std::thread([this] { run(); }); }
  void kernel_done() { kernel_done_ = true; }
  bool finished() const { return finished_; }
  // Waits for the shutdown protocol to finish (or the timeout). true = clean.
  bool wait_finished(int timeout_ms) {
    auto t0 = Clock::now();
    while (!finished_ && ms_since(t0) < timeout_ms) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return finished_;
  }
  void stop() { stop_ = true; if (th_.joinable()) th_.join(); }
  ProxyStats st;

 private:
  static constexpr uint8_t kData = 1, kFin = 2;

  void run() {
    KD_LOG(2, "proxy", "rank %d proxy started (%s, world %d)", rank_, "reliable", world_);
    bool fin_sent = false;
    while (!stop_) {
      sink_.heartbeat();
      bool busy = drain_queue();
      Reliable::Delivery d;
      if (rel_.poll(d, busy ? 0 : 1)) {
        busy = true;
        receive(d);
        for (int i = 0; i < 64 && rel_.poll(d, 0); ++i) receive(d);
        sink_.flush();
      }
      int dead = rel_.dead_peer();
      if (dead >= 0) { KD_LOG(0, "proxy", "rank %d: reporting dead peer %d to the device", rank_, dead); sink_.fail(KVM_ERR_DIST_PEER_DOWN, dead); fin_seen_[dead] = true; }
      if (kernel_done_ && !fin_sent && q_empty() && unacked_alive() == 0) {
        for (int p = 0; p < world_; ++p) if (p != rank_ && !rel_.dead(p)) { uint8_t f = kFin; send_blocking(p, &f, 1); }
        fin_sent = true;
        KD_LOG(3, "proxy", "rank %d: FIN sent", rank_);
      }
      if (fin_sent && all_fin() && unacked_alive() == 0 && !finished_) {
        finished_ = true;
        KD_LOG(2, "proxy", "rank %d proxy done: sent %ld delivered %ld rejected %ld dropped(dead) %ld full-stalls %ld, retx %ld dups %ld", rank_,
               st.sent, st.delivered, st.rejected, st.dropped_dead, st.full_stalls, rel_.st.retx, rel_.st.dups);
      }
      if (!busy && finished_) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }
  bool q_empty() { return !q_.valid() || q_.head() == q_.tickets(); }
  int unacked_alive() { int n = 0; for (int p = 0; p < world_; ++p) if (p != rank_ && !rel_.dead(p)) n += (int)rel_.unacked(p); return n; }
  bool all_fin() { for (int p = 0; p < world_; ++p) if (!fin_seen_[p] && !rel_.dead(p)) return false; return true; }

  void send_blocking(int dst, const void* p, size_t n) {
    while (!stop_) {
      auto s = rel_.send(dst, p, n);
      if (s != Reliable::FULL) return;
      Reliable::Delivery d;
      if (rel_.poll(d, 1)) { receive(d); sink_.flush(); }
    }
  }

  bool drain_queue() {
    if (!q_.valid()) return false;
    bool any = false;
    const KvmSqSlot* s; const uint8_t* payload;
    while (q_.peek(s, payload)) {
      int dst = s->dst_rank;
      if (dst < 0 || dst >= world_ || dst == rank_ || s->nbytes < 0 || (uint32_t)s->nbytes > q_.slot_bytes() ||
          (size_t)s->rows * s->cols * 4 != (size_t)s->nbytes) {
        KD_LOG(0, "proxy", "rank %d: malformed send-queue slot (dst %d, %d bytes)", rank_, dst, s->nbytes);
        sink_.fail(KVM_ERR_DIST_BAD_MSG, dst); ++st.rejected; q_.pop(); any = true; continue;
      }
      std::vector<uint8_t> msg(1 + sizeof(KvmMsgHdr) + (size_t)s->nbytes);
      msg[0] = kData;
      KvmMsgHdr h{s->dst_buf, s->dst_off, s->dst_ld, s->rows, s->cols, s->var, s->ver, s->nbytes};
      memcpy(msg.data() + 1, &h, sizeof h);
      memcpy(msg.data() + 1 + sizeof h, payload, (size_t)s->nbytes);
      auto r = rel_.send(dst, msg.data(), msg.size());
      if (r == Reliable::FULL) { ++st.full_stalls; return any; }          // window full: leave the slot, the device backs up
      if (r == Reliable::DEAD) ++st.dropped_dead; else { ++st.sent; st.bytes_out += (unsigned long long)s->nbytes; }
      KD_LOG(4, "proxy", "rank %d -> %d: %d bytes var %d ver %d", rank_, dst, s->nbytes, s->var, s->ver);
      q_.pop();
      any = true;
    }
    return any;
  }

  void receive(const Reliable::Delivery& d) {
    if (d.data.empty()) return;
    if (d.data[0] == kFin) { fin_seen_[d.src] = true; KD_LOG(3, "proxy", "rank %d: FIN from %d", rank_, d.src); return; }
    if (d.data[0] != kData || d.data.size() < 1 + sizeof(KvmMsgHdr)) { ++st.rejected; sink_.fail(KVM_ERR_DIST_BAD_MSG, d.src); return; }
    KvmMsgHdr h; memcpy(&h, d.data.data() + 1, sizeof h);
    if (h.nbytes < 0 || d.data.size() != 1 + sizeof h + (size_t)h.nbytes) { ++st.rejected; sink_.fail(KVM_ERR_DIST_BAD_MSG, d.src); return; }
    int rc = sink_.deliver(d.src, h, d.data.data() + 1 + sizeof h);
    if (rc) { ++st.rejected; KD_LOG(0, "proxy", "rank %d: rejected message from %d (code %d)", rank_, d.src, rc); sink_.fail(rc, d.src); return; }
    ++st.delivered; st.bytes_in += (unsigned long long)h.nbytes;
  }

  int rank_, world_;
  SendQ q_;
  Reliable& rel_;
  Sink& sink_;
  std::vector<bool> fin_seen_;
  std::atomic<bool> stop_{false}, kernel_done_{false}, finished_{false};
  std::thread th_;
};

// ---------------------------------------------------------------------------------------------------------- host-memory Sink
// Applies messages to plain host arrays (tests, and a reference for the CUDA sink). inbox[var] = max(inbox[var], ver).
struct HostBuf { float* p; int rows, cols; };
class HostSink : public Sink {
 public:
  HostSink(std::vector<HostBuf> bufs, std::atomic<unsigned>* inbox, int nvars) : bufs_(std::move(bufs)), inbox_(inbox), nvars_(nvars) {}
  int deliver(int, const KvmMsgHdr& h, const uint8_t* payload) override {
    if (h.dst_buf < 0 || h.dst_buf >= (int)bufs_.size() || h.rows < 1 || h.cols < 1 || h.dst_ld < h.cols || h.dst_off < 0) return KVM_ERR_DIST_BAD_MSG;
    HostBuf& b = bufs_[h.dst_buf];
    if ((long long)h.dst_off + (long long)(h.rows - 1) * h.dst_ld + h.cols > (long long)b.rows * b.cols) return KVM_ERR_DIST_BAD_MSG;
    if (h.var >= nvars_) return KVM_ERR_DIST_BAD_SIGNAL;
    for (int r = 0; r < h.rows; ++r) memcpy(b.p + h.dst_off + (size_t)r * h.dst_ld, payload + (size_t)r * h.cols * 4, (size_t)h.cols * 4);
    if (h.var >= 0) {
      unsigned cur = inbox_[h.var].load(std::memory_order_relaxed);
      while ((unsigned)h.ver > cur && !inbox_[h.var].compare_exchange_weak(cur, (unsigned)h.ver, std::memory_order_release)) {}
    }
    return 0;
  }
  void flush() override {}
  void fail(int code, int arg) override { fail_code = code; fail_arg = arg; }
  std::atomic<int> fail_code{0}, fail_arg{0};
 private:
  std::vector<HostBuf> bufs_;
  std::atomic<unsigned>* inbox_;
  int nvars_;
};

}  // namespace kvmd
