#pragma once
// RDMA transport over a small `Verbs` abstraction, plus the RDMA version inbox (write-with-immediate signalling).
//
// STATUS (read this first)
//   * ShimVerbs (always compiled) runs everything against verbs_shim.hpp, a SOFTWARE EMULATION in one process. All tests in
//     tests/dist/test_rdma.cpp use it. They test the protocol logic (credits, ring slots, imm signalling, error handling).
//   * IbvVerbs (below, only under `#ifdef HAVE_IBVERBS`) is a real libibverbs backend written from memory of the public
//     ibverbs API. It is UNTESTED ON REAL NICs and NOT COMPILED in the environment this was written in (no libibverbs on the
//     laptop). Treat it as a starting point that needs a first bring-up on hardware, not as working code.
//
// Layers
//   Verbs             abstract: reg_mr, create_qp (->INIT), connect_qp (->RTR->RTS), post_send, post_recv, poll_cq, wait_cq,
//                     poll_async. Types are the ibverbs-shaped ones from verbs_shim.hpp (SendWr, WorkCompletion, ...).
//   Rendezvous        out-of-band exchange of {qpn, psn, rkey, ring addr, ...} blobs: InProcRendezvous (tests) and
//                     TransportRendezvous (over any Transport; for real NICs use a TcpTransport, which already authenticates the job).
//   RdmaTransport     a `Transport` (send/poll/PEER_DOWN) on one RC QP per peer. See "Design" below.
//   RdmaVersionInbox  the point of the RDMA design: a registered u32 version array per rank, updated by RDMA_WRITE_WITH_IMM that
//                     is posted on the SAME QP right after the data RDMA_WRITE, so RC in-order delivery gives data-before-signal.
//
// RdmaTransport design
//   * Each rank registers, per peer, a receive RING of `slots` fixed-size slots (slot = 16 B header + max_msg) with REMOTE_WRITE,
//     and a registered tx bounce area of `slots` slots. send() copies [hdr|payload] into a bounce slot and posts ONE
//     RDMA_WRITE_WITH_IMM into the peer's ring slot (imm = type(2 bits) | slot index). The write-with-imm consumes a posted receive
//     on the peer; the receives are zero-length "tokens" (the data goes to the ring, not to the receive buffer), `3*slots` of them
//     are kept posted and re-posted as completions are consumed.
//   * Credit-based flow control: the sender may only write slot s of the peer's ring when that slot is free. Slots are used and
//     freed strictly in order, so a counter suffices: credits = slots at start, -1 per data message, + n when the peer reports n
//     slots consumed. A slot is "consumed" when the application's poll() has COPIED the message out (not when it arrived), so a
//     slow consumer applies end-to-end back-pressure instead of an unbounded event queue. Credits ride (a) in the header of every
//     data message in the reverse direction and (b) in an explicit imm-only credit message (type 1, count in the imm, no ring slot)
//     sent by poll() when >= slots/4 are freed, and on an idle poll. Credit messages use only receive tokens, never a ring slot,
//     so credit return can never deadlock behind data. Explicit credit messages in flight are capped at `slots` per peer;
//     if the peer's CQ thread lags enough to run out of tokens anyway, the sender just sees RNR and retries (rnr_retry = 7), it cannot overrun the ring.
//   * Out of credits: send() does NOT block (two ranks blocking on each other's credits while neither polls would deadlock) and
//     does NOT return false (false means "peer is down" and Reliable would declare it dead). It queues a copy, up to
//     max_pending_msgs / max_pending_bytes; beyond that it DROPS the message and returns true, which the Transport contract allows
//     (lossy) and Reliable repairs by retransmission. Reliable's window_msgs should be <= slots + max_pending_msgs to avoid drops.
//     Small messages (<= ctrl_bytes: Reliable's 16-byte ACK/PING) skip the data queue and may use the last max(1, slots/8) credits, which
//     data may not take: found by TSAN stress, ACKs queued behind megabytes of data caused RTOs and Reliable declared a live peer DEAD.
//     The queue drains from the CQ thread when credits or bounce slots come back. So the sender never has more messages in flight
//     than the peer has free ring slots (asserted in the tests), and RNR (no receive posted) cannot happen in normal operation;
//     rnr_retry is still set to 7 (infinite) as a safety net.
//   * A CQ thread polls one CQ (send + recv completions, told apart by a tag in wr_id; never by wc.opcode, which ibverbs leaves
//     undefined on error). Receive completions become descriptors (peer, slot, len); poll() turns them into Event::RECV. A failed
//     completion (flush, retry-exceeded, RNR-exceeded, access error) or an async QP-fatal event becomes ONE Event::PEER_DOWN per peer.
//     PEER_DOWN is queued behind messages that had already arrived, so ordering with delivered data is preserved.
//   * close() destroys the QPs; the peer sees the QP error (flush) and reports PEER_DOWN, like a closed TCP connection.
//
// GPUDirect RDMA: DESIGN NOTES ONLY, NOTHING IN THIS FILE IMPLEMENTS OR TESTS ANY OF IT
//   Goal: the NIC reads tile data straight from GPU memory and writes straight into the owner GPU's landing buffer and inbox,
//   removing the pinned-host staging copy that the TCP path needs.
//   1. Registering device memory. Two mechanisms:
//        a) nvidia-peermem (kernel module, formerly nv_peer_mem): ibv_reg_mr() on a cudaMalloc'd pointer works once the module is
//           loaded; it calls nvidia_p2p_get_pages() to pin and map the GPU pages for the NIC.
//        b) dma-buf: cuMemGetHandleForAddressRange(&fd, dptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0) gives an fd, then
//           ibv_reg_dmabuf_mr(pd, offset, len, iova, fd, access). Preferred on recent drivers/kernels (open kernel modules, no
//           out-of-tree peermem). Needs a driver/rdma-core/kernel combination that supports it: CHECK on the target box.
//      The DMA source must be allocated with cudaMalloc / cuMemCreate (not cudaMallocManaged). GPU BAR pages are 64 KiB: register
//      64 KiB-aligned ranges, pad small landing buffers and the inbox to 64 KiB, and expect registration to cover whole pages, so
//      a sub-page inbox shares a page with whatever else is there unless padded. BAR1 aperture size limits how much can be mapped
//      (small on some GPUs unless resizable BAR is on): CHECK `nvidia-smi -q -d MEMORY` BAR1 usage.
//   2. Device-initiated posting (IBGDA / DOCA GPUNetIO): the GPU writes WQEs into a send queue that lives in GPU memory and rings
//      the NIC doorbell itself (via a mapped doorbell register), so the host proxy and the send queue copy go away. This needs the
//      QP/CQ/WQ buffers allocated in GPU memory (mlx5dv / DOCA APIs, not plain libibverbs) and a NIC that supports it (ConnectX-6 Dx
//      and newer in DOCA's list). Not designed further here: it replaces post_send, not the credit / inbox protocol above.
//   3. Ordering of NIC writes vs GPU consumers. The device inbox is polled with ld.acquire.sys (kvm_dist.cuh). The argument we
//      would rely on: RC delivers the data RDMA_WRITE before the later RDMA_WRITE(_WITH_IMM) that carries the version, so a GPU
//      thread that observes the new version can read the data. CAVEATS TO VERIFY ON HARDWARE, not facts established here:
//        * "NIC write complete" is a statement about the PCIe/BAR write being posted, and whether the data is visible to SMs by the
//          time the later inbox write is visible depends on PCIe ordering rules and on the GPU's handling of incoming writes (relaxed
//          ordering on the NIC / PCIe bridge settings can break it). NVIDIA documents a GPUDirect "flush"/consistency requirement:
//          after a CQE for a write to GPU memory, a GPU kernel launched afterwards sees it (cudaDeviceFlushGPUDirectRDMAWrites,
//          or CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_*), but that is for visibility to later kernels, NOT necessarily for a running
//          kernel that is spinning on the inbox while the write arrives.
//        * Safe fallback if the ordering is not guaranteed for a spinning kernel: have the host CQ thread (as in
//          RdmaVersionInbox here) observe the write-with-imm CQE and then perform the inbox store itself (cudaMemcpy / a
//          GDRCopy-mapped store, or a tiny kernel) after a GPUDirect flush. That costs a CQE round trip but restores the
//          argument. A NIC-written inbox (no host in the path) should only be enabled after a litmus test on the target hardware.
//        * Atomics: the version is a plain 4 B write, not an RDMA atomic; monotonicity relies on the single-writer-per-variable rule
//          of DISTRIBUTED.md and on in-order RC delivery. Writes may be max()-ed only by a host or device agent, not by the NIC.
//   4. What could NOT be verified here: everything above. No RDMA NIC, no GPU, no libibverbs, no nvidia-peermem, no dma-buf export,
//      no IBGDA; IbvVerbs is uncompiled; PCIe ordering and flush behaviour are unmeasured; performance of any of it is unknown.
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <tuple>

#include "transport.hpp"
#include "verbs_shim.hpp"

namespace kvmd {

using shim::AsyncEvent;
using shim::QpAttr;
using shim::QpCaps;
using shim::RecvWr;
using shim::SendWr;
using shim::Sge;
using shim::WcOp;
using shim::WcStatus;
using shim::WorkCompletion;
using shim::WrOp;

struct VMr { uint64_t addr = 0; size_t length = 0; uint32_t lkey = 0, rkey = 0; int id = -1; };

// --------------------------------------------------------------------------------------------------------- Verbs interface
class Verbs {
 public:
  virtual ~Verbs() = default;
  virtual const char* name() const = 0;
  virtual bool reg_mr(void* p, size_t n, int access, VMr& out) = 0;
  virtual void dereg_mr(VMr& m) = 0;
  virtual int create_cq(int cap) = 0;                                   // handle >= 0, -1 on failure
  virtual void destroy_cq(int cq) = 0;
  virtual int create_qp(int send_cq, int recv_cq, const QpCaps& caps) = 0;   // RC QP, returned in INIT
  virtual uint32_t qpn(int qp) = 0;
  virtual uint32_t local_psn(int qp) = 0;
  virtual std::vector<uint8_t> local_address() = 0;                     // opaque: LID/GID for real NICs, empty for the shim
  virtual int connect_qp(int qp, uint32_t remote_qpn, uint32_t remote_psn, uint32_t local_psn, const std::vector<uint8_t>& remote_addr, int rnr_retry) = 0;  // INIT->RTR->RTS
  virtual void error_qp(int qp) = 0;                                    // -> ERR (test hook / orderly teardown)
  virtual void destroy_qp(int qp) = 0;                                  // peer sees the QP die
  virtual int post_send(int qp, const SendWr& wr) = 0;
  virtual int post_recv(int qp, const RecvWr& wr) = 0;
  virtual int poll_cq(int cq, WorkCompletion* wc, int n) = 0;
  virtual bool wait_cq(int cq, int timeout_ms) = 0;                     // true: completions may be available (spurious allowed)
  virtual void interrupt_cq(int cq) = 0;
  virtual bool poll_async(AsyncEvent& ev) = 0;
};

// ------------------------------------------------------------------------------------------------------ ShimVerbs backend
class ShimVerbs : public Verbs {
 public:
  explicit ShimVerbs(shim::ShimFabric& f, const char* name = "shim0") : dev_(f, name), pd_(dev_.alloc_pd()) {}
  const char* name() const override { return "shim"; }
  bool reg_mr(void* p, size_t n, int access, VMr& out) override {
    auto m = pd_->reg_mr(p, n, access);
    out.addr = m->addr; out.length = n; out.lkey = m->lkey; out.rkey = m->rkey;
    std::lock_guard<std::mutex> l(m_);
    out.id = (int)mrs_.size(); mrs_.push_back(std::move(m));
    return true;
  }
  void dereg_mr(VMr& v) override { std::lock_guard<std::mutex> l(m_); if (v.id >= 0 && v.id < (int)mrs_.size()) mrs_[v.id].reset(); v.id = -1; }
  int create_cq(int cap) override { std::lock_guard<std::mutex> l(m_); cqs_.push_back(dev_.create_cq(cap)); return (int)cqs_.size() - 1; }
  void destroy_cq(int cq) override { std::lock_guard<std::mutex> l(m_); cqs_[cq].reset(); }
  int create_qp(int scq, int rcq, const QpCaps& caps) override {
    std::lock_guard<std::mutex> l(m_);
    auto q = dev_.create_qp(*pd_, *cqs_[scq], *cqs_[rcq], caps);
    QpAttr a; a.state = shim::QpState::INIT; a.access_flags = shim::ACCESS_LOCAL_WRITE | shim::ACCESS_REMOTE_WRITE;
    if (q->modify(a, shim::QP_STATE | shim::QP_ACCESS) != 0) return -1;
    qps_.push_back(std::move(q)); return (int)qps_.size() - 1;
  }
  uint32_t qpn(int qp) override { return Q(qp)->qpn(); }
  uint32_t local_psn(int qp) override { return 0x1000 + Q(qp)->qpn(); }
  std::vector<uint8_t> local_address() override { return {}; }
  int connect_qp(int qp, uint32_t rqpn, uint32_t rpsn, uint32_t lpsn, const std::vector<uint8_t>&, int rnr_retry) override {
    shim::ShimQP* q = Q(qp);
    QpAttr a; a.state = shim::QpState::RTR; a.dest_qpn = rqpn; a.rq_psn = rpsn;
    int rc = q->modify(a, shim::QP_STATE | shim::QP_DEST_QPN | shim::QP_RQ_PSN);
    if (rc) return rc;
    a.state = shim::QpState::RTS; a.sq_psn = lpsn; a.rnr_retry = rnr_retry;
    return q->modify(a, shim::QP_STATE | shim::QP_SQ_PSN | shim::QP_RNR_RETRY);
  }
  void error_qp(int qp) override { QpAttr a; a.state = shim::QpState::ERR; Q(qp)->modify(a, shim::QP_STATE); }
  void destroy_qp(int qp) override { std::unique_ptr<shim::ShimQP> q; { std::lock_guard<std::mutex> l(m_); q = std::move(qps_[qp]); } }
  int post_send(int qp, const SendWr& wr) override { return Q(qp)->post_send(wr); }
  int post_recv(int qp, const RecvWr& wr) override { return Q(qp)->post_recv(wr); }
  int poll_cq(int cq, WorkCompletion* wc, int n) override { return C(cq)->poll(n, wc); }
  bool wait_cq(int cq, int ms) override { return C(cq)->wait(ms); }
  void interrupt_cq(int cq) override { C(cq)->interrupt(); }
  bool poll_async(AsyncEvent& ev) override { return dev_.poll_async(ev); }
  shim::ShimDevice& device() { return dev_; }
 private:
  shim::ShimQP* Q(int h) { std::lock_guard<std::mutex> l(m_); return qps_.at(h).get(); }
  shim::ShimCQ* C(int h) { std::lock_guard<std::mutex> l(m_); return cqs_.at(h).get(); }
  shim::ShimDevice dev_; std::unique_ptr<shim::ShimPD> pd_;
  std::mutex m_;
  std::vector<std::unique_ptr<shim::ShimMR>> mrs_;
  std::vector<std::unique_ptr<shim::ShimCQ>> cqs_;
  std::vector<std::unique_ptr<shim::ShimQP>> qps_;
};

// ---------------------------------------------------------------------------------------------------------- rendezvous
// Out-of-band exchange of connection blobs. publish() never blocks; fetch() waits for the blob (from -> to, channel).
class Rendezvous {
 public:
  virtual ~Rendezvous() = default;
  virtual void publish(int channel, int from, int to, const std::vector<uint8_t>& blob) = 0;
  virtual bool fetch(int channel, int from, int to, std::vector<uint8_t>& blob, int timeout_ms) = 0;
};

class InProcRendezvous : public Rendezvous {
 public:
  void publish(int ch, int from, int to, const std::vector<uint8_t>& b) override {
    { std::lock_guard<std::mutex> l(m_); box_[{ch, from, to}] = b; }
    cv_.notify_all();
  }
  bool fetch(int ch, int from, int to, std::vector<uint8_t>& b, int timeout_ms) override {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::milliseconds(timeout_ms), [&] { return box_.count({ch, from, to}) > 0; })) return false;
    b = box_[{ch, from, to}]; return true;
  }
 private:
  std::mutex m_; std::condition_variable cv_;
  std::map<std::tuple<int, int, int>, std::vector<uint8_t>> box_;
};

// Rendezvous over any Transport (for real NICs: a TcpTransport used only for bootstrap). The Transport must be dedicated to
// this rendezvous (it consumes its events); blobs for other channels / peers are stashed, not lost. Single user thread.
class TransportRendezvous : public Rendezvous {
  struct Hdr { uint32_t magic, channel, from, to; };
  static constexpr uint32_t kMagic = 0x5a444e52;
 public:
  explicit TransportRendezvous(Transport& t) : t_(t) {}
  void publish(int ch, int from, int to, const std::vector<uint8_t>& b) override {
    std::vector<uint8_t> m(sizeof(Hdr) + b.size());
    Hdr h{kMagic, (uint32_t)ch, (uint32_t)from, (uint32_t)to};
    memcpy(m.data(), &h, sizeof h); if (!b.empty()) memcpy(m.data() + sizeof h, b.data(), b.size());
    t_.send(to, m.data(), m.size());
  }
  bool fetch(int ch, int from, int to, std::vector<uint8_t>& b, int timeout_ms) override {
    auto key = std::make_tuple(ch, from, to);
    auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
      auto it = stash_.find(key);
      if (it != stash_.end()) { b = it->second; return true; }
      int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
      if (left <= 0) return false;
      Event ev;
      if (!t_.poll(ev, std::min(left, 20))) continue;
      if (ev.type != EvType::RECV || ev.data.size() < sizeof(Hdr)) continue;
      Hdr h; memcpy(&h, ev.data.data(), sizeof h);
      if (h.magic != kMagic) continue;
      stash_[std::make_tuple((int)h.channel, (int)h.from, (int)h.to)] = std::vector<uint8_t>(ev.data.begin() + sizeof h, ev.data.end());
    }
  }
 private:
  Transport& t_;
  std::map<std::tuple<int, int, int>, std::vector<uint8_t>> stash_;
};

// Tiny POD blob helpers.
struct BlobW { std::vector<uint8_t> b; template <class T> BlobW& put(const T& v) { const uint8_t* p = (const uint8_t*)&v; b.insert(b.end(), p, p + sizeof v); return *this; }
  BlobW& bytes(const std::vector<uint8_t>& v) { put<uint32_t>((uint32_t)v.size()); b.insert(b.end(), v.begin(), v.end()); return *this; } };
struct BlobR { const std::vector<uint8_t>& b; size_t o = 0; bool ok = true;
  template <class T> T get() { T v{}; if (o + sizeof v > b.size()) { ok = false; return v; } memcpy(&v, b.data() + o, sizeof v); o += sizeof v; return v; }
  std::vector<uint8_t> bytes() { uint32_t n = get<uint32_t>(); std::vector<uint8_t> v; if (!ok || o + n > b.size()) { ok = false; return v; } v.assign(b.begin() + o, b.begin() + o + n); o += n; return v; } };

// ================================================================================================== RdmaTransport
struct RdmaConfig {
  int slots = 64;                         // receive-ring slots per peer = max messages in flight per peer
  size_t max_msg = (256u << 10) + 256;    // largest message (Reliable header + proxy header + 256 KiB payload)
  int max_pending_msgs = 0;               // queued when out of credits; 0 = 2 * slots. Beyond it: dropped (see header comment)
  size_t ctrl_bytes = 64;                 // messages of at most this size (Reliable ACK / PING) bypass the data queue and may use the reserved slots
  size_t max_pending_bytes = 64u << 20;
  int cq_poll_ms = 2;
  int connect_timeout_ms = 10000;
  int channel = 1;                        // rendezvous channel (this transport uses channel and channel + 1)
};

class RdmaTransport : public Transport {
  static constexpr uint32_t kMagic = 0x4b524d31;
  struct SlotHdr { uint32_t magic, len, credits, pad; };
  enum Kind : uint64_t { K_RECV = 1, K_SEND = 2, K_CREDIT = 3 };
  static constexpr uint32_t T_DATA = 0, T_CREDIT = 1;
  static uint64_t wid(Kind k, int peer, uint32_t idx) { return ((uint64_t)k << 56) | ((uint64_t)peer << 32) | idx; }
  struct Desc { int peer; uint32_t slot, len; bool down; };
  struct Peer {
    int qp = -1; uint32_t qpn = 0;
    std::vector<uint8_t> ring, bounce; VMr ring_mr, bounce_mr;
    uint64_t r_ring = 0; uint32_t r_rkey = 0, r_slots = 0, r_slot_bytes = 0;
    std::mutex m;                                   // credits, next_tx, freed, free_bounce, pending
    uint32_t credits = 0, next_tx = 0, freed = 0;
    std::vector<uint32_t> free_bounce;
    std::deque<std::vector<uint8_t>> pending, pending_ctrl; size_t pending_bytes = 0;
    uint32_t credit_inflight = 0;                   // explicit credit messages not yet completed (bounded by slots)
    std::atomic<bool> down{false};
  };

 public:
  // Creates local resources and publishes the connection blobs; call connect() (concurrently on all ranks) before use.
  RdmaTransport(Verbs& v, int rank, int world, Rendezvous& rdz, RdmaConfig c = {}) : v_(v), rank_(rank), world_(world), rdz_(rdz), c_(c) {
    if (c_.slots < 2) throw std::invalid_argument("rdma: slots < 2");
    if (c_.max_pending_msgs <= 0) c_.max_pending_msgs = 2 * c_.slots;
    reserve_ = std::max(1, c_.slots / 8);
    slot_bytes_ = (uint32_t)((sizeof(SlotHdr) + c_.max_msg + 63) & ~(size_t)63);
    cq_ = v_.create_cq(8 * c_.slots * std::max(1, world_) + 64);
    peers_.resize(world_);
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      peers_[p] = std::make_unique<Peer>();
      Peer& q = *peers_[p];
      q.ring.assign((size_t)slot_bytes_ * c_.slots, 0);
      q.bounce.assign((size_t)slot_bytes_ * c_.slots, 0);
      v_.reg_mr(q.ring.data(), q.ring.size(), shim::ACCESS_LOCAL_WRITE | shim::ACCESS_REMOTE_WRITE, q.ring_mr);
      v_.reg_mr(q.bounce.data(), q.bounce.size(), shim::ACCESS_LOCAL_WRITE, q.bounce_mr);
      QpCaps caps; caps.max_send_wr = 4 * c_.slots + 16; caps.max_recv_wr = 3 * c_.slots + 8; caps.max_inline = 0;
      q.qp = v_.create_qp(cq_, cq_, caps);
      if (q.qp < 0) throw std::runtime_error("rdma: create_qp failed");
      q.qpn = v_.qpn(q.qp);
      qp_to_peer_[q.qpn] = p;
      for (int i = 0; i < c_.slots; ++i) q.free_bounce.push_back(i);
      for (int i = 0; i < 3 * c_.slots; ++i) v_.post_recv(q.qp, RecvWr{wid(K_RECV, p, i), {}, 0});
      BlobW w; w.put<uint32_t>(q.qpn).put<uint32_t>(v_.local_psn(q.qp)).put<uint64_t>(q.ring_mr.addr).put<uint32_t>(q.ring_mr.rkey)
          .put<uint32_t>(c_.slots).put<uint32_t>(slot_bytes_).bytes(v_.local_address());
      rdz_.publish(c_.channel, rank_, p, w.b);
    }
  }
  ~RdmaTransport() override { close(); }

  // Fetch every peer's blob, bring the QPs to RTS, then wait until every peer is also at RTS (ready barrier). false: timeout/error.
  bool connect() {
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      Peer& q = *peers_[p];
      std::vector<uint8_t> b;
      if (!rdz_.fetch(c_.channel, p, rank_, b, c_.connect_timeout_ms)) { KD_LOG(0, "rdma", "rank %d: no connection info from %d", rank_, p); return false; }
      BlobR r{b};
      uint32_t rqpn = r.get<uint32_t>(), rpsn = r.get<uint32_t>(); q.r_ring = r.get<uint64_t>(); q.r_rkey = r.get<uint32_t>();
      q.r_slots = r.get<uint32_t>(); q.r_slot_bytes = r.get<uint32_t>(); auto addr = r.bytes();
      if (!r.ok || q.r_slots == 0) { KD_LOG(0, "rdma", "rank %d: bad connection info from %d", rank_, p); return false; }
      q.credits = q.r_slots;
      if (v_.connect_qp(q.qp, rqpn, rpsn, v_.local_psn(q.qp), addr, 7) != 0) { KD_LOG(0, "rdma", "rank %d: QP connect to %d failed", rank_, p); return false; }
    }
    for (int p = 0; p < world_; ++p) if (p != rank_) rdz_.publish(c_.channel + 1, rank_, p, {1});
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      std::vector<uint8_t> b;
      if (!rdz_.fetch(c_.channel + 1, p, rank_, b, c_.connect_timeout_ms)) { KD_LOG(0, "rdma", "rank %d: peer %d never became ready", rank_, p); return false; }
    }
    connected_ = true;
    cq_thread_ = std::thread([this] { cq_loop(); });
    KD_LOG(2, "rdma", "rank %d connected (%s backend, %d slots x %u B)", rank_, v_.name(), c_.slots, slot_bytes_);
    return true;
  }

  int rank() const override { return rank_; }
  int world() const override { return world_; }
  const char* name() const override { return "rdma"; }

  // Never blocks. false: peer down / not connected / message larger than max_msg. Out of credits: queued (bounded), then dropped.
  bool send(int dst, const void* p, size_t n) override {
    if (dst < 0 || dst >= world_ || dst == rank_ || !connected_ || closing_) return false;
    if (n > c_.max_msg) { KD_LOG(0, "rdma", "rank %d: message of %zu B exceeds max_msg %zu", rank_, n, c_.max_msg); return false; }
    Peer& q = *peers_[dst];
    std::lock_guard<std::mutex> l(q.m);
    if (q.down) return false;
    if (n <= c_.ctrl_bytes) {                        // small control messages never wait behind queued data
      if (q.pending_ctrl.empty() && try_post(q, dst, (const uint8_t*)p, n, true)) return true;
      if ((int)q.pending_ctrl.size() >= c_.max_pending_msgs) { ++overflow_dropped; return true; }
      q.pending_ctrl.emplace_back((const uint8_t*)p, (const uint8_t*)p + n); ++credit_stalls;
      return true;
    }
    if (q.pending.empty() && try_post(q, dst, (const uint8_t*)p, n, false)) return true;
    if ((int)q.pending.size() >= c_.max_pending_msgs || q.pending_bytes + n > c_.max_pending_bytes) { ++overflow_dropped; return true; }
    q.pending.emplace_back((const uint8_t*)p, (const uint8_t*)p + n); q.pending_bytes += n; ++credit_stalls;
    return true;
  }

  bool poll(Event& ev, int timeout_ms) override {
    Desc d;
    {
      std::unique_lock<std::mutex> l(rxm_);
      if (!rxcv_.wait_for(l, std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms), [&] { return !rxq_.empty(); })) { l.unlock(); flush_credits(); return false; }
      d = rxq_.front(); rxq_.pop_front();
    }
    ev.peer = d.peer; ev.data.clear();
    if (d.down) { ev.type = EvType::PEER_DOWN; return true; }
    Peer& q = *peers_[d.peer];
    const uint8_t* s = q.ring.data() + (size_t)d.slot * slot_bytes_ + sizeof(SlotHdr);
    ev.type = EvType::RECV; ev.data.assign(s, s + d.len);
    ++msgs_in;
    std::lock_guard<std::mutex> l(q.m);          // slot consumed: it may be reused by the sender once the credit reaches it
    ++q.freed;
    if (q.freed >= std::max<uint32_t>(1, c_.slots / 4)) send_credit(q, d.peer);
    return true;
  }

  void close() override {
    if (closing_.exchange(true)) return;
    if (connected_) { v_.interrupt_cq(cq_); if (cq_thread_.joinable()) cq_thread_.join(); }
    for (auto& q : peers_) if (q) { v_.destroy_qp(q->qp); }            // peers observe the QP error and report PEER_DOWN
    for (auto& q : peers_) if (q) { v_.dereg_mr(q->ring_mr); v_.dereg_mr(q->bounce_mr); }
    v_.destroy_cq(cq_);
  }

  uint32_t peer_qpn(int peer) const { return peers_.at(peer)->qpn; }          // test hook (kill a QP in the shim fabric)
  // Diagnostic snapshot of one peer's flow-control state.
  std::string debug(int peer) {
    Peer& q = *peers_.at(peer); std::lock_guard<std::mutex> l(q.m);
    size_t rx; { std::lock_guard<std::mutex> l2(rxm_); rx = rxq_.size(); }
    char b[256]; snprintf(b, sizeof b, "peer %d: credits %u/%u freed %u credit_inflight %u free_bounce %zu pending %zu (%zu B) rxq %zu down %d dropped %ld",
                          peer, q.credits, q.r_slots, q.freed, q.credit_inflight, q.free_bounce.size(), q.pending.size() + q.pending_ctrl.size(), q.pending_bytes, rx, (int)q.down, overflow_dropped.load());
    return b;
  }
  std::atomic<long> msgs_in{0}, msgs_out{0}, credit_msgs{0}, credit_stalls{0}, overflow_dropped{0}, max_inflight{0}, bad_msgs{0};

 private:
  // q.m held. Posts [hdr|payload] if a credit and a bounce slot are available.
  bool try_post(Peer& q, int dst, const uint8_t* p, size_t n, bool ctrl) {
    if (q.credits <= (ctrl ? 0u : reserve_) || q.free_bounce.empty()) return false;     // data may not take the last `reserve_` slots
    uint32_t b = q.free_bounce.back();
    uint8_t* buf = q.bounce.data() + (size_t)b * slot_bytes_;
    SlotHdr h{kMagic, (uint32_t)n, q.freed, 0};
    memcpy(buf, &h, sizeof h); if (n) memcpy(buf + sizeof h, p, n);
    uint32_t slot = q.next_tx % q.r_slots;
    SendWr wr; wr.wr_id = wid(K_SEND, dst, b); wr.opcode = WrOp::RDMA_WRITE_WITH_IMM; wr.send_flags = shim::SEND_SIGNALED;
    wr.imm_data = (T_DATA << 30) | slot;
    wr.sge = {q.bounce_mr.addr + (uint64_t)b * slot_bytes_, (uint32_t)(sizeof h + n), q.bounce_mr.lkey}; wr.num_sge = 1;
    wr.remote_addr = q.r_ring + (uint64_t)slot * q.r_slot_bytes; wr.rkey = q.r_rkey;
    int rc = v_.post_send(q.qp, wr);
    if (rc != 0) { if (rc == -ENOMEM) return false; mark_down(dst, "post_send failed"); return false; }
    q.free_bounce.pop_back(); q.freed = 0; ++q.next_tx; --q.credits; ++msgs_out;
    long infl = (long)q.r_slots - (long)q.credits, cur = max_inflight.load();
    while (infl > cur && !max_inflight.compare_exchange_weak(cur, infl)) {}
    return true;
  }
  void drain(Peer& q, int dst) {                    // q.m held
    while (!q.pending_ctrl.empty() && !q.down) {
      auto& f = q.pending_ctrl.front();
      if (!try_post(q, dst, f.data(), f.size(), true)) break;
      q.pending_ctrl.pop_front();
    }
    while (!q.pending.empty() && !q.down) {
      auto& f = q.pending.front();
      if (!try_post(q, dst, f.data(), f.size(), false)) break;
      q.pending_bytes -= f.size(); q.pending.pop_front();
    }
  }
  void send_credit(Peer& q, int dst) {              // q.m held; imm-only write, uses a receive token on the peer but no ring slot
    if (q.freed == 0 || q.down || q.credit_inflight >= (uint32_t)c_.slots) return;   // bounded: control messages also need receive tokens
    SendWr wr; wr.wr_id = wid(K_CREDIT, dst, 0); wr.opcode = WrOp::RDMA_WRITE_WITH_IMM; wr.send_flags = shim::SEND_SIGNALED;
    wr.imm_data = (T_CREDIT << 30) | (q.freed & 0x3fffffff); wr.num_sge = 0; wr.remote_addr = q.r_ring; wr.rkey = q.r_rkey;
    int rc = v_.post_send(q.qp, wr);
    if (rc == 0) { q.freed = 0; ++q.credit_inflight; ++credit_msgs; } else if (rc != -ENOMEM) mark_down(dst, "credit post failed");
  }
  void flush_credits() {
    for (int p = 0; p < world_; ++p) if (p != rank_) { Peer& q = *peers_[p]; std::lock_guard<std::mutex> l(q.m); if (q.freed > 0) send_credit(q, p); }
  }
  void mark_down(int peer, const char* why) {
    Peer& q = *peers_[peer];
    if (q.down.exchange(true) || closing_) return;
    KD_LOG(1, "rdma", "rank %d: peer %d down (%s)", rank_, peer, why);
    { std::lock_guard<std::mutex> l(rxm_); rxq_.push_back({peer, 0, 0, true}); }
    rxcv_.notify_one();
  }
  void cq_loop() {
    WorkCompletion wc[32];
    while (!closing_) {
      int n = v_.poll_cq(cq_, wc, 32);
      AsyncEvent ae;
      while (v_.poll_async(ae)) { auto it = qp_to_peer_.find(ae.qpn); if (it != qp_to_peer_.end()) mark_down(it->second, "QP fatal event"); }
      if (n <= 0) { v_.wait_cq(cq_, c_.cq_poll_ms); continue; }
      for (int i = 0; i < n; ++i) on_wc(wc[i]);
    }
  }
  void on_wc(const WorkCompletion& wc) {
    Kind k = (Kind)(wc.wr_id >> 56); int peer = (int)((wc.wr_id >> 32) & 0xffffff); uint32_t idx = (uint32_t)wc.wr_id;
    if (peer < 0 || peer >= world_ || !peers_[peer]) return;
    Peer& q = *peers_[peer];
    if (wc.status != WcStatus::SUCCESS) { mark_down(peer, shim::to_string(wc.status)); return; }
    if (k == K_SEND) { std::lock_guard<std::mutex> l(q.m); q.free_bounce.push_back(idx); drain(q, peer); return; }
    if (k == K_CREDIT) { std::lock_guard<std::mutex> l(q.m); if (q.credit_inflight) --q.credit_inflight; if (q.freed >= std::max<uint32_t>(1, c_.slots / 4)) send_credit(q, peer); return; }
    if (k != K_RECV) return;
    uint32_t imm = wc.imm_data, type = imm >> 30, val = imm & 0x3fffffff;
    if (type == T_CREDIT) {
      std::lock_guard<std::mutex> l(q.m); q.credits += val; drain(q, peer);
    } else if (type == T_DATA && val < (uint32_t)c_.slots) {
      SlotHdr h; memcpy(&h, q.ring.data() + (size_t)val * slot_bytes_, sizeof h);
      if (h.magic != kMagic || h.len + sizeof h != wc.byte_len || h.len > c_.max_msg) { ++bad_msgs; KD_LOG(0, "rdma", "rank %d: bad slot header from %d", rank_, peer); }
      else {
        { std::lock_guard<std::mutex> l(q.m); q.credits += h.credits; if (q.credits) drain(q, peer); }
        { std::lock_guard<std::mutex> l(rxm_); rxq_.push_back({peer, val, h.len, false}); }
        rxcv_.notify_one();
      }
    } else ++bad_msgs;
    v_.post_recv(q.qp, RecvWr{wc.wr_id, {}, 0});        // re-post the token (fails harmlessly if the QP is already dead)
  }

  Verbs& v_; int rank_, world_; Rendezvous& rdz_; RdmaConfig c_;
  uint32_t slot_bytes_ = 0, reserve_ = 1; int cq_ = -1;
  std::vector<std::unique_ptr<Peer>> peers_;
  std::map<uint32_t, int> qp_to_peer_;
  std::mutex rxm_; std::condition_variable rxcv_; std::deque<Desc> rxq_;
  std::thread cq_thread_;
  std::atomic<bool> connected_{false}, closing_{false};
};

// ============================================================================================== RdmaVersionInbox
// Per rank: a registered u32 array `inbox[nvars]` (in tests host memory; on a GPU box the device inbox or a mapped mirror) and
// landing buffers registered for REMOTE_WRITE. put_signal(dst, landing, off, data, n, var, ver) posts, on the SAME QP and in this
// order: (1) RDMA_WRITE of the tile into the peer's landing buffer (unsignalled), (2) RDMA_WRITE_WITH_IMM whose imm = var and whose
// 4-byte payload (inline) = ver, written to a per-source staging array on the peer. RC executes a QP's work requests in order, so the
// peer's CQE for (2) is generated after (1) is placed. The peer's CQ thread then does inbox[var] = max(inbox[var], stage[src][var])
// with a RELEASE store and wakes waiters. A reader that sees inbox[var] >= ver with an ACQUIRE load therefore sees the whole tile.
// Why the staging array and max(): two signals for the same var may be in flight; the CQE of the first may be processed after the
// second signal's payload landed, and stage[src][var] then holds the NEWER version. That is fine: its data was placed before it
// (same QP order), and versions are monotone single-writer per var (DISTRIBUTED.md), so applying max() is exactly "latest version".
// Imm-only variant (publish_remote_imm): imm = 0x80000000 | var(15 bits) << 16 | ver(16 bits); no payload, no staging; ver < 65536.
// Direct readers of `inbox` must use acquire loads (RdmaVersionInbox::load, or __atomic_load_n(..., ACQUIRE), or ld.acquire.sys).
struct RdmaInboxConfig { int bounce = 64; size_t max_put = 64u << 10; int tokens = 512; int cq_poll_ms = 1; int connect_timeout_ms = 10000; int channel = 20; };

class RdmaVersionInbox {
 public:
  struct Landing { void* ptr; size_t len; };
  struct RemoteMr { uint64_t addr = 0; uint32_t rkey = 0; size_t len = 0; };
  using Config = RdmaInboxConfig;

  RdmaVersionInbox(Verbs& v, int rank, int world, uint32_t* inbox, int nvars, std::vector<Landing> landings, Rendezvous& rdz, Config c = {})
      : v_(v), rank_(rank), world_(world), inbox_(inbox), nvars_(nvars), land_(std::move(landings)), rdz_(rdz), c_(c) {
    cq_ = v_.create_cq(world_ * (c_.tokens + 4 * c_.bounce) + 64);
    land_mr_.resize(land_.size());
    for (size_t i = 0; i < land_.size(); ++i) v_.reg_mr(land_[i].ptr, land_[i].len, shim::ACCESS_LOCAL_WRITE | shim::ACCESS_REMOTE_WRITE, land_mr_[i]);
    peers_.resize(world_);
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      peers_[p] = std::make_unique<Peer>();
      Peer& q = *peers_[p];
      q.stage.assign((size_t)nvars_, 0);
      q.bounce.assign(c_.max_put * c_.bounce, 0);
      v_.reg_mr(q.stage.data(), q.stage.size() * 4, shim::ACCESS_LOCAL_WRITE | shim::ACCESS_REMOTE_WRITE, q.stage_mr);
      v_.reg_mr(q.bounce.data(), q.bounce.size(), shim::ACCESS_LOCAL_WRITE, q.bounce_mr);
      QpCaps caps; caps.max_send_wr = 4 * c_.bounce + 16; caps.max_recv_wr = c_.tokens + 8; caps.max_inline = 64;
      q.qp = v_.create_qp(cq_, cq_, caps);
      if (q.qp < 0) throw std::runtime_error("inbox: create_qp failed");
      q.qpn = v_.qpn(q.qp); qp_to_peer_[q.qpn] = p;
      for (int i = 0; i < c_.bounce; ++i) q.free_b.push_back(i);
      for (int i = 0; i < c_.tokens; ++i) v_.post_recv(q.qp, RecvWr{(uint64_t)p << 32 | (uint32_t)i, {}, 0});
      BlobW w; w.put<uint32_t>(q.qpn).put<uint32_t>(v_.local_psn(q.qp)).put<uint64_t>(q.stage_mr.addr).put<uint32_t>(q.stage_mr.rkey).put<uint32_t>(nvars_)
          .put<uint32_t>((uint32_t)land_.size());
      for (auto& m : land_mr_) w.put<uint64_t>(m.addr).put<uint32_t>(m.rkey).put<uint64_t>(m.length);
      w.bytes(v_.local_address());
      rdz_.publish(c_.channel, rank_, p, w.b);
    }
  }
  ~RdmaVersionInbox() { close(); }

  bool connect() {
    for (int p = 0; p < world_; ++p) {
      if (p == rank_) continue;
      Peer& q = *peers_[p];
      std::vector<uint8_t> b;
      if (!rdz_.fetch(c_.channel, p, rank_, b, c_.connect_timeout_ms)) return false;
      BlobR r{b};
      uint32_t rqpn = r.get<uint32_t>(), rpsn = r.get<uint32_t>(); q.r_stage = r.get<uint64_t>(); q.r_stage_rkey = r.get<uint32_t>();
      uint32_t rn = r.get<uint32_t>(), nl = r.get<uint32_t>();
      if (!r.ok || rn != (uint32_t)nvars_ || nl > 4096) return false;
      for (uint32_t i = 0; i < nl; ++i) { RemoteMr m; m.addr = r.get<uint64_t>(); m.rkey = r.get<uint32_t>(); m.len = r.get<uint64_t>(); q.r_land.push_back(m); }
      auto addr = r.bytes();
      if (!r.ok || v_.connect_qp(q.qp, rqpn, rpsn, v_.local_psn(q.qp), addr, 7) != 0) return false;
    }
    for (int p = 0; p < world_; ++p) if (p != rank_) rdz_.publish(c_.channel + 1, rank_, p, {1});
    for (int p = 0; p < world_; ++p) { if (p == rank_) continue; std::vector<uint8_t> b; if (!rdz_.fetch(c_.channel + 1, p, rank_, b, c_.connect_timeout_ms)) return false; }
    connected_ = true;
    thr_ = std::thread([this] { loop(); });
    return true;
  }

  RemoteMr remote_landing(int dst, size_t idx) const { return peers_.at(dst)->r_land.at(idx); }

  // data -> landing[off..off+n) on dst, then version `ver` for `var` into dst's inbox, in that order on one QP.
  bool put_signal(int dst, const RemoteMr& landing, size_t off, const void* data, size_t n, uint32_t var, uint32_t ver) {
    if (!ok_to_send(dst) || var >= (uint32_t)nvars_ || n > c_.max_put || off + n > landing.len) return false;
    Peer& q = *peers_[dst];
    int b = -1;
    if (n) { b = acquire_bounce(q, dst); if (b < 0) return false; }
    std::lock_guard<std::mutex> l(q.sm);                     // data WR and signal WR adjacent on the SQ (not required for correctness)
    if (n) {
      uint8_t* buf = q.bounce.data() + (size_t)b * c_.max_put;
      memcpy(buf, data, n);
      SendWr w; w.wr_id = tag(dst, b, false); w.opcode = WrOp::RDMA_WRITE; w.send_flags = 0;
      w.sge = {q.bounce_mr.addr + (uint64_t)b * c_.max_put, (uint32_t)n, q.bounce_mr.lkey}; w.num_sge = 1;
      w.remote_addr = landing.addr + off; w.rkey = landing.rkey;
      if (!post(q, dst, w)) { release_b(q, b); return false; }
    }
    return post_signal(q, dst, var, ver, b);
  }
  bool put_signal(int dst, size_t landing_idx, size_t off, const void* data, size_t n, uint32_t var, uint32_t ver) {
    if (dst < 0 || dst >= world_ || !peers_[dst] || landing_idx >= peers_[dst]->r_land.size()) return false;
    return put_signal(dst, peers_[dst]->r_land[landing_idx], off, data, n, var, ver);
  }
  // Signal only. NOTE: ordering against earlier data is only guaranteed for data posted on this inbox's QP to the same dst.
  bool publish_remote(int dst, uint32_t var, uint32_t ver) {
    if (!ok_to_send(dst) || var >= (uint32_t)nvars_) return false;
    Peer& q = *peers_[dst]; std::lock_guard<std::mutex> l(q.sm);
    return post_signal(q, dst, var, ver, -1);
  }
  bool publish_remote_imm(int dst, uint32_t var, uint32_t ver) {
    if (!ok_to_send(dst) || var >= 32768 || ver >= 65536) return false;
    Peer& q = *peers_[dst]; std::lock_guard<std::mutex> l(q.sm);
    SendWr w; w.wr_id = tag(dst, 0, true); w.opcode = WrOp::RDMA_WRITE_WITH_IMM; w.send_flags = shim::SEND_SIGNALED;
    w.imm_data = 0x80000000u | (var << 16) | ver; w.num_sge = 0; w.remote_addr = q.r_stage; w.rkey = q.r_stage_rkey;
    return post(q, dst, w);
  }

  uint32_t load(uint32_t var) const { return __atomic_load_n(&inbox_[var], __ATOMIC_ACQUIRE); }
  bool wait_ge(uint32_t var, uint32_t ver, int timeout_ms) {
    auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
      if (load(var) >= ver) return true;
      if (Clock::now() >= end) return false;
      std::unique_lock<std::mutex> l(wm_);
      wcv_.wait_for(l, std::chrono::milliseconds(1));
    }
  }
  bool peer_down(int p) const { return peers_.at(p) && peers_.at(p)->down; }
  uint32_t peer_qpn(int p) const { return peers_.at(p)->qpn; }
  void on_peer_down(std::function<void(int)> f) { down_cb_ = std::move(f); }
  void close() {
    if (closing_.exchange(true)) return;
    if (connected_) { v_.interrupt_cq(cq_); if (thr_.joinable()) thr_.join(); }
    for (auto& q : peers_) if (q) v_.destroy_qp(q->qp);
    for (auto& q : peers_) if (q) { v_.dereg_mr(q->stage_mr); v_.dereg_mr(q->bounce_mr); }
    for (auto& m : land_mr_) v_.dereg_mr(m);
    v_.destroy_cq(cq_);
  }
  std::atomic<long> signals_applied{0}, bad{0};

 private:
  struct Peer {
    int qp = -1; uint32_t qpn = 0;
    std::vector<uint32_t> stage; std::vector<uint8_t> bounce; VMr stage_mr, bounce_mr;
    uint64_t r_stage = 0; uint32_t r_stage_rkey = 0; std::vector<RemoteMr> r_land;
    std::mutex sm;                      // serialises posting on the QP
    std::mutex bm; std::condition_variable bcv; std::vector<int> free_b;   // bounce free list
    std::atomic<bool> down{false};
  };
  static uint64_t tag(int peer, int b, bool nob) { return ((uint64_t)(nob ? 2 : 1) << 56) | ((uint64_t)peer << 32) | (uint32_t)(b < 0 ? 0 : b); }
  bool ok_to_send(int dst) const { return connected_ && !closing_ && dst >= 0 && dst < world_ && dst != rank_ && peers_[dst] && !peers_[dst]->down; }
  int acquire_bounce(Peer& q, int dst) {
    std::unique_lock<std::mutex> l(q.bm);
    if (!q.bcv.wait_for(l, std::chrono::seconds(10), [&] { return !q.free_b.empty() || q.down || closing_; }) || q.free_b.empty()) return -1;
    int b = q.free_b.back(); q.free_b.pop_back(); (void)dst; return b;
  }
  void release_b(Peer& q, int b) { if (b < 0) return; { std::lock_guard<std::mutex> l(q.bm); q.free_b.push_back(b); } q.bcv.notify_one(); }
  bool post(Peer& q, int dst, const SendWr& w) {
    for (int tries = 0; tries < 20000; ++tries) {
      int rc = v_.post_send(q.qp, w);
      if (rc == 0) return true;
      if (rc != -ENOMEM) { mark_down(dst); return false; }
      std::this_thread::sleep_for(std::chrono::microseconds(50));        // SQ full: completions free entries
    }
    return false;
  }
  bool post_signal(Peer& q, int dst, uint32_t var, uint32_t ver, int b) {
    SendWr w; w.wr_id = b >= 0 ? tag(dst, b, false) : tag(dst, 0, true); w.opcode = WrOp::RDMA_WRITE_WITH_IMM; w.send_flags = shim::SEND_SIGNALED | shim::SEND_INLINE;
    w.imm_data = var;
    w.sge = {(uint64_t)(uintptr_t)&ver, 4, 0}; w.num_sge = 1;
    w.remote_addr = q.r_stage + 4ull * var; w.rkey = q.r_stage_rkey;
    if (!post(q, dst, w)) { release_b(q, b); return false; }
    return true;
  }
  void mark_down(int p) {
    if (p < 0 || !peers_[p]) return;
    if (peers_[p]->down.exchange(true) || closing_) return;
    peers_[p]->bcv.notify_all();
    KD_LOG(1, "inbox", "rank %d: peer %d down", rank_, p);
    if (down_cb_) down_cb_(p);
  }
  void apply(uint32_t var, uint32_t ver) {
    if (var >= (uint32_t)nvars_) { ++bad; return; }
    if (ver > __atomic_load_n(&inbox_[var], __ATOMIC_RELAXED)) __atomic_store_n(&inbox_[var], ver, __ATOMIC_RELEASE);   // single writer: this thread
    ++signals_applied;
    { std::lock_guard<std::mutex> l(wm_); }
    wcv_.notify_all();
  }
  void loop() {
    WorkCompletion wc[32];
    while (!closing_) {
      int n = v_.poll_cq(cq_, wc, 32);
      AsyncEvent ae;
      while (v_.poll_async(ae)) { auto it = qp_to_peer_.find(ae.qpn); if (it != qp_to_peer_.end()) mark_down(it->second); }
      if (n <= 0) { v_.wait_cq(cq_, c_.cq_poll_ms); continue; }
      for (int i = 0; i < n; ++i) {
        const WorkCompletion& w = wc[i];
        int kind = (int)(w.wr_id >> 56); int peer = (int)((w.wr_id >> 32) & 0xffffff);
        if (peer < 0 || peer >= world_ || !peers_[peer]) continue;
        Peer& q = *peers_[peer];
        if (w.status != WcStatus::SUCCESS) { mark_down(peer); continue; }
        if (kind == 0) {                                                      // receive token (wr_id = peer << 32 | i)
          uint32_t imm = w.imm_data;
          if (imm & 0x80000000u) apply((imm >> 16) & 0x7fff, imm & 0xffff);
          else if (imm < (uint32_t)nvars_) apply(imm, __atomic_load_n(&q.stage[imm], __ATOMIC_RELAXED));   // CQE ordering makes the staged word visible
          else ++bad;
          v_.post_recv(q.qp, RecvWr{w.wr_id, {}, 0});
        } else if (kind == 1) release_b(q, (int)(uint32_t)w.wr_id);          // signal completed => its data (earlier on the QP) completed
      }
    }
  }

  Verbs& v_; int rank_, world_; uint32_t* inbox_; int nvars_;
  std::vector<Landing> land_; std::vector<VMr> land_mr_;
  Rendezvous& rdz_; Config c_; int cq_ = -1;
  std::vector<std::unique_ptr<Peer>> peers_;
  std::map<uint32_t, int> qp_to_peer_;
  std::mutex wm_; std::condition_variable wcv_;
  std::function<void(int)> down_cb_;
  std::thread thr_;
  std::atomic<bool> connected_{false}, closing_{false};
};

}  // namespace kvmd

// ===================================================================================================== IbvVerbs (real NIC)
// UNTESTED ON REAL NICs. NOT COMPILED unless HAVE_IBVERBS is defined (needs <infiniband/verbs.h>, link with -libverbs).
// Written from memory of the public libibverbs API; expect compile fixes and a bring-up on hardware. Known gaps / things to check:
//   * wc.imm_data is in NETWORK byte order in ibverbs (htonl on post, ntohl on completion): done below, verify.
//   * On a failed completion only wr_id, status (and qp_num) are valid; opcode is not. This code never relies on it.
//   * RoCE (Ethernet) needs a GID/GRH in the address handle; InfiniBand needs the LID. local_address() returns {lid, gid}; the
//     gid index (default 0) may need to be chosen to be the RoCEv2 one (`show_gids`).
//   * wait_cq uses a completion channel with ibv_req_notify_cq re-armed per call; completions racing the arming are caught by the
//     caller's next poll (the CQ threads wake at least every cq_poll_ms). Not tuned.
//   * max_inline / inline sends depend on the device (ibv_query_qp reports the granted value). QP caps are requested, not verified.
//   * Bootstrap: run a TcpTransport (rank/hosts/ports/cluster_id) among the ranks and use TransportRendezvous over it; the blobs carry
//     {qpn, psn, ring addr, rkey, slots, slot_bytes, lid/gid}. The TcpTransport can be closed after connect().
//   * Memory registration of the rings uses plain host memory. GPUDirect (device memory) is design-only, see the header comment.
//   * Timeouts: QP timeout=14 (~67 ms * 2^... = 4.096us*2^14 = 67 ms), retry_cnt=7, rnr_retry=7 (infinite): a dead peer takes about
//     retry_cnt * timeout ~ 0.5 s to surface as RETRY_EXC_ERR. Tune for WAN.
#ifdef HAVE_IBVERBS
#include <arpa/inet.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <poll.h>

namespace kvmd {
class IbvVerbs : public Verbs {
  static_assert((int)IBV_ACCESS_LOCAL_WRITE == shim::ACCESS_LOCAL_WRITE && (int)IBV_ACCESS_REMOTE_WRITE == shim::ACCESS_REMOTE_WRITE, "access flag values");
 public:
  // dev_name: e.g. "mlx5_0" ("" = first device). port: usually 1. gid_index: RoCE GID index.
  explicit IbvVerbs(const std::string& dev_name = "", int port = 1, int gid_index = 0) : port_(port), gid_index_(gid_index) {
    int n = 0; ibv_device** list = ibv_get_device_list(&n);
    if (!list || n == 0) throw std::runtime_error("ibv: no RDMA devices");
    ibv_device* d = nullptr;
    for (int i = 0; i < n; ++i) if (dev_name.empty() || dev_name == ibv_get_device_name(list[i])) { d = list[i]; break; }
    if (!d) { ibv_free_device_list(list); throw std::runtime_error("ibv: device not found"); }
    ctx_ = ibv_open_device(d); ibv_free_device_list(list);
    if (!ctx_) throw std::runtime_error("ibv: open_device failed");
    pd_ = ibv_alloc_pd(ctx_);
    if (!pd_ || ibv_query_port(ctx_, (uint8_t)port_, &pattr_) != 0) throw std::runtime_error("ibv: alloc_pd/query_port failed");
    if (pattr_.link_layer == IBV_LINK_LAYER_ETHERNET && ibv_query_gid(ctx_, (uint8_t)port_, gid_index_, &gid_) != 0) throw std::runtime_error("ibv: query_gid failed");
    int fl = fcntl(ctx_->async_fd, F_GETFL); fcntl(ctx_->async_fd, F_SETFL, fl | O_NONBLOCK);   // poll_async must not block
  }
  ~IbvVerbs() override {
    for (auto* q : qps_) if (q) ibv_destroy_qp(q);
    for (auto& c : cqs_) { if (c.cq) ibv_destroy_cq(c.cq); if (c.ch) ibv_destroy_comp_channel(c.ch); }
    for (auto* m : mrs_) if (m) ibv_dereg_mr(m);
    if (pd_) ibv_dealloc_pd(pd_);
    if (ctx_) ibv_close_device(ctx_);
  }
  const char* name() const override { return "ibverbs"; }
  bool reg_mr(void* p, size_t n, int access, VMr& out) override {
    ibv_mr* m = ibv_reg_mr(pd_, p, n, access);
    if (!m) return false;
    std::lock_guard<std::mutex> l(mu_); mrs_.push_back(m);
    out.addr = (uint64_t)(uintptr_t)p; out.length = n; out.lkey = m->lkey; out.rkey = m->rkey; out.id = (int)mrs_.size() - 1;
    return true;
  }
  void dereg_mr(VMr& v) override { std::lock_guard<std::mutex> l(mu_); if (v.id >= 0 && mrs_[v.id]) { ibv_dereg_mr(mrs_[v.id]); mrs_[v.id] = nullptr; } v.id = -1; }
  int create_cq(int cap) override {
    Cq c; c.ch = ibv_create_comp_channel(ctx_);
    if (!c.ch) return -1;
    c.cq = ibv_create_cq(ctx_, cap, nullptr, c.ch, 0);
    if (!c.cq) return -1;
    fcntl(c.ch->fd, F_SETFL, fcntl(c.ch->fd, F_GETFL) | O_NONBLOCK);
    std::lock_guard<std::mutex> l(mu_); cqs_.push_back(c); return (int)cqs_.size() - 1;
  }
  void destroy_cq(int h) override { std::lock_guard<std::mutex> l(mu_); if (cqs_[h].cq) { ibv_destroy_cq(cqs_[h].cq); ibv_destroy_comp_channel(cqs_[h].ch); cqs_[h] = Cq{}; } }
  int create_qp(int scq, int rcq, const QpCaps& caps) override {
    ibv_qp_init_attr ia{};
    ia.send_cq = cq(scq); ia.recv_cq = cq(rcq); ia.qp_type = IBV_QPT_RC; ia.sq_sig_all = 0;
    ia.cap.max_send_wr = caps.max_send_wr; ia.cap.max_recv_wr = caps.max_recv_wr; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1; ia.cap.max_inline_data = caps.max_inline;
    ibv_qp* q = ibv_create_qp(pd_, &ia);
    if (!q) return -1;
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = (uint8_t)port_; a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(q, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) { ibv_destroy_qp(q); return -1; }
    std::lock_guard<std::mutex> l(mu_); qps_.push_back(q); return (int)qps_.size() - 1;
  }
  uint32_t qpn(int h) override { return qp(h)->qp_num; }
  uint32_t local_psn(int h) override { return (0x7a1000u + qp(h)->qp_num * 7919u) & 0xffffff; }
  std::vector<uint8_t> local_address() override {
    BlobW w; w.put<uint16_t>(pattr_.lid).put<uint8_t>(pattr_.link_layer == IBV_LINK_LAYER_ETHERNET ? 1 : 0);
    w.b.insert(w.b.end(), gid_.raw, gid_.raw + 16); return w.b;
  }
  int connect_qp(int h, uint32_t rqpn, uint32_t rpsn, uint32_t lpsn, const std::vector<uint8_t>& raddr, int rnr_retry) override {
    if (raddr.size() < 3 + 16) return -EINVAL;
    uint16_t dlid; uint8_t roce; memcpy(&dlid, raddr.data(), 2); roce = raddr[2];
    ibv_qp* q = qp(h);
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_RTR; a.path_mtu = pattr_.active_mtu; a.dest_qp_num = rqpn; a.rq_psn = rpsn;
    a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    a.ah_attr.port_num = (uint8_t)port_; a.ah_attr.sl = 0; a.ah_attr.src_path_bits = 0; a.ah_attr.dlid = dlid;
    if (roce) {
      a.ah_attr.is_global = 1; a.ah_attr.grh.hop_limit = 64; a.ah_attr.grh.sgid_index = (uint8_t)gid_index_; a.ah_attr.grh.traffic_class = 0;
      memcpy(a.ah_attr.grh.dgid.raw, raddr.data() + 3, 16);
    }
    int rc = ibv_modify_qp(q, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc) return -rc;
    ibv_qp_attr b{};
    b.qp_state = IBV_QPS_RTS; b.timeout = 14; b.retry_cnt = 7; b.rnr_retry = (uint8_t)rnr_retry; b.sq_psn = lpsn; b.max_rd_atomic = 1;
    rc = ibv_modify_qp(q, &b, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    return rc ? -rc : 0;
  }
  void error_qp(int h) override { ibv_qp_attr a{}; a.qp_state = IBV_QPS_ERR; ibv_modify_qp(qp(h), &a, IBV_QP_STATE); }
  void destroy_qp(int h) override { std::lock_guard<std::mutex> l(mu_); if (qps_[h]) { ibv_destroy_qp(qps_[h]); qps_[h] = nullptr; } }
  int post_send(int h, const SendWr& w) override {
    ibv_sge sg{}; ibv_send_wr wr{}, *bad = nullptr;
    sg.addr = w.sge.addr; sg.length = w.sge.length; sg.lkey = w.sge.lkey;
    wr.wr_id = w.wr_id; wr.sg_list = w.num_sge ? &sg : nullptr; wr.num_sge = w.num_sge;
    wr.send_flags = ((w.send_flags & shim::SEND_SIGNALED) ? IBV_SEND_SIGNALED : 0) | ((w.send_flags & shim::SEND_INLINE) ? IBV_SEND_INLINE : 0);
    switch (w.opcode) {
      case WrOp::SEND: wr.opcode = IBV_WR_SEND; break;
      case WrOp::SEND_WITH_IMM: wr.opcode = IBV_WR_SEND_WITH_IMM; wr.imm_data = htonl(w.imm_data); break;
      case WrOp::RDMA_WRITE: wr.opcode = IBV_WR_RDMA_WRITE; break;
      case WrOp::RDMA_WRITE_WITH_IMM: wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; wr.imm_data = htonl(w.imm_data); break;
    }
    if (w.opcode == WrOp::RDMA_WRITE || w.opcode == WrOp::RDMA_WRITE_WITH_IMM) { wr.wr.rdma.remote_addr = w.remote_addr; wr.wr.rdma.rkey = w.rkey; }
    return -ibv_post_send(qp(h), &wr, &bad);          // ibv_post_send returns 0 or an errno value (positive)
  }
  int post_recv(int h, const RecvWr& w) override {
    ibv_sge sg{}; ibv_recv_wr wr{}, *bad = nullptr;
    sg.addr = w.sge.addr; sg.length = w.sge.length; sg.lkey = w.sge.lkey;
    wr.wr_id = w.wr_id; wr.sg_list = w.num_sge ? &sg : nullptr; wr.num_sge = w.num_sge;
    return -ibv_post_recv(qp(h), &wr, &bad);
  }
  int poll_cq(int h, WorkCompletion* out, int n) override {
    std::vector<ibv_wc> wc(n);
    int k = ibv_poll_cq(cq(h), n, wc.data());
    for (int i = 0; i < k; ++i) {
      out[i] = WorkCompletion{};
      out[i].wr_id = wc[i].wr_id; out[i].qp_num = wc[i].qp_num; out[i].byte_len = wc[i].byte_len;
      switch (wc[i].status) {
        case IBV_WC_SUCCESS: out[i].status = WcStatus::SUCCESS; break;
        case IBV_WC_LOC_LEN_ERR: out[i].status = WcStatus::LOC_LEN_ERR; break;
        case IBV_WC_LOC_PROT_ERR: out[i].status = WcStatus::LOC_PROT_ERR; break;
        case IBV_WC_WR_FLUSH_ERR: out[i].status = WcStatus::WR_FLUSH_ERR; break;
        case IBV_WC_REM_INV_REQ_ERR: out[i].status = WcStatus::REM_INV_REQ_ERR; break;
        case IBV_WC_REM_ACCESS_ERR: out[i].status = WcStatus::REM_ACCESS_ERR; break;
        case IBV_WC_RNR_RETRY_EXC_ERR: out[i].status = WcStatus::RNR_RETRY_EXC_ERR; break;
        case IBV_WC_RETRY_EXC_ERR: out[i].status = WcStatus::RETRY_EXC_ERR; break;
        default: out[i].status = WcStatus::GENERAL_ERR; break;
      }
      if (wc[i].status == IBV_WC_SUCCESS) {            // opcode / imm only valid on success
        if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) out[i].opcode = WcOp::RECV_RDMA_WITH_IMM;
        else if (wc[i].opcode == IBV_WC_RECV) out[i].opcode = WcOp::RECV;
        else if (wc[i].opcode == IBV_WC_RDMA_WRITE) out[i].opcode = WcOp::RDMA_WRITE;
        else out[i].opcode = WcOp::SEND;
        if (wc[i].wc_flags & IBV_WC_WITH_IMM) { out[i].imm_data = ntohl(wc[i].imm_data); out[i].wc_flags = shim::WC_WITH_IMM; }
      }
    }
    return k;
  }
  bool wait_cq(int h, int timeout_ms) override {
    Cq c; { std::lock_guard<std::mutex> l(mu_); c = cqs_[h]; }
    ibv_req_notify_cq(c.cq, 0);
    pollfd pf{c.ch->fd, POLLIN, 0};
    if (::poll(&pf, 1, timeout_ms) <= 0) return false;
    ibv_cq* ev_cq = nullptr; void* ev_ctx = nullptr;
    if (ibv_get_cq_event(c.ch, &ev_cq, &ev_ctx) == 0) ibv_ack_cq_events(ev_cq, 1);
    return true;
  }
  void interrupt_cq(int) override {}                  // the CQ threads wake every cq_poll_ms and re-check their stop flag
  bool poll_async(AsyncEvent& ev) override {
    ibv_async_event ae;
    if (ibv_get_async_event(ctx_, &ae) != 0) return false;   // O_NONBLOCK: fails when nothing is pending
    bool fatal = false; uint32_t qn = 0;
    switch (ae.event_type) {
      case IBV_EVENT_QP_FATAL: case IBV_EVENT_QP_REQ_ERR: case IBV_EVENT_QP_ACCESS_ERR: fatal = true; qn = ae.element.qp->qp_num; break;
      default: break;
    }
    ibv_ack_async_event(&ae);
    if (!fatal) return false;
    ev.qpn = qn; ev.peer_initiated = false; return true;
  }
 private:
  struct Cq { ibv_cq* cq = nullptr; ibv_comp_channel* ch = nullptr; };
  ibv_qp* qp(int h) { std::lock_guard<std::mutex> l(mu_); return qps_.at(h); }
  ibv_cq* cq(int h) { std::lock_guard<std::mutex> l(mu_); return cqs_.at(h).cq; }
  ibv_context* ctx_ = nullptr; ibv_pd* pd_ = nullptr; ibv_port_attr pattr_{}; ibv_gid gid_{};
  int port_, gid_index_;
  std::mutex mu_; std::vector<ibv_mr*> mrs_; std::vector<ibv_qp*> qps_; std::vector<Cq> cqs_;
};
}  // namespace kvmd
#endif  // HAVE_IBVERBS
