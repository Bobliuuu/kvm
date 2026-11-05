#pragma once
// verbs_shim.hpp: an in-process SOFTWARE EMULATION of the libibverbs subset the RDMA transport needs.
//
// THIS IS AN EMULATION. It is memcpy between buffers of one process. It has no NIC, no wire, no PCIe, no DMA engine, no
// real timing. It exists so the protocol logic built on verbs semantics (credits, ring slots, write-with-immediate signalling,
// data-before-signal ordering, error/flush handling) can be tested deterministically on a laptop. Passing these tests does NOT
// show that the code works on a real RDMA NIC; see the "not verified" list in transport_rdma.hpp.
//
// Shape (names mirror ibverbs on purpose):
//   ShimFabric  the "network". Owns a lock and one delivery thread (used only for delayed / retried work).
//   ShimDevice  one per "host" (like ibv_context). Creates PD / CQ / QP and holds an async-event queue (ibv_get_async_event).
//   ShimPD      protection domain. MR and QP must share a PD for an access to be legal (as on real hardware).
//   ShimMR      memory region: reg_mr(ptr, len, access) -> lkey/rkey. Remote access needs REMOTE_WRITE and the right rkey.
//   ShimCQ      completion queue: poll() is non-blocking (ibv_poll_cq), wait() is an optional blocking wait.
//   ShimQP      reliable-connected (RC) queue pair, state machine RESET -> INIT -> RTR -> RTS -> ERR (-> RESET).
//
// Work request semantics implemented:
//   SEND, SEND_WITH_IMM            consume a posted receive on the peer; recv completion opcode RECV (imm flagged for _WITH_IMM)
//   RDMA_WRITE                     writes remote memory (rkey + bounds + REMOTE_WRITE checked); NO receive consumed, NO remote CQE
//   RDMA_WRITE_WITH_IMM            write, then consume a posted receive and generate a recv completion (RECV_RDMA_WITH_IMM)
//                                  carrying imm_data and byte_len; the write is visible BEFORE that completion is.
//   send_flags SIGNALED / INLINE   unsignalled sends complete silently (errors and flushes always complete)
// Errors (always produce a completion on the requester, then the QP goes to ERR):
//   LOC_PROT_ERR (bad lkey / out of bounds / not local-writable receive buffer), REM_ACCESS_ERR (bad rkey / bounds /
//   permission / wrong PD), RNR_RETRY_EXC_ERR (no receive posted and the RNR retry counter ran out; retry = 7 means infinite,
//   as in ibverbs), REM_INV_REQ_ERR (message longer than the posted receive), RETRY_EXC_ERR (peer QP gone).
// QP error handling: every outstanding WR completes with WR_FLUSH_ERR (signalled or not), the peer QP is moved to ERR too
// (its WRs flush, and both devices get an async QP_FATAL event). That is how a transport sees PEER_DOWN.
// Ordering: each QP executes its send queue strictly in order; the receiver's completion for write-with-imm is generated after
// the data landed. Test knobs (ShimFabric): delay_us (wire latency, order preserved), reorder_writes_us (DELIBERATELY BREAKS
// ordering: plain RDMA_WRITE data lands later than a following write-with-imm; used only by negative tests),
// ShimQP::force_rnr(n), ShimFabric::kill_qp(qpn), rnr_delay_us.
// Threading: every call may come from any thread. Delivery runs inline in post_send (holding the fabric lock) unless a work
// item is not yet due, in which case the fabric thread runs it. TSAN-clean: all shared state is under the fabric / CQ locks, and
// stores of <= 8 aligned bytes into registered memory are relaxed atomics (so a CQ thread may read a landing word without a race).
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "log.hpp"

namespace kvmd {
namespace shim {

using SClock = std::chrono::steady_clock;

enum Access : int { ACCESS_LOCAL_WRITE = 1, ACCESS_REMOTE_WRITE = 2, ACCESS_REMOTE_READ = 4 };
enum class WrOp : int { SEND, SEND_WITH_IMM, RDMA_WRITE, RDMA_WRITE_WITH_IMM };
enum SendFlags : int { SEND_SIGNALED = 1, SEND_INLINE = 2 };
enum class WcStatus : int { SUCCESS, LOC_LEN_ERR, LOC_PROT_ERR, WR_FLUSH_ERR, REM_INV_REQ_ERR, REM_ACCESS_ERR, RNR_RETRY_EXC_ERR, RETRY_EXC_ERR, GENERAL_ERR };
enum class WcOp : int { SEND, RDMA_WRITE, RECV, RECV_RDMA_WITH_IMM };
enum WcFlags : int { WC_WITH_IMM = 1 };
enum class QpState : int { RESET, INIT, RTR, RTS, ERR };
enum QpAttrMask : int { QP_STATE = 1, QP_ACCESS = 2, QP_DEST_QPN = 4, QP_RQ_PSN = 8, QP_SQ_PSN = 16, QP_RNR_RETRY = 32, QP_RETRY_CNT = 64 };

inline const char* to_string(WcStatus s) {
  static const char* n[] = {"SUCCESS", "LOC_LEN_ERR", "LOC_PROT_ERR", "WR_FLUSH_ERR", "REM_INV_REQ_ERR", "REM_ACCESS_ERR", "RNR_RETRY_EXC_ERR", "RETRY_EXC_ERR", "GENERAL_ERR"};
  return n[(int)s];
}
inline const char* to_string(QpState s) {
  static const char* n[] = {"RESET", "INIT", "RTR", "RTS", "ERR"};
  return n[(int)s];
}

struct Sge { uint64_t addr = 0; uint32_t length = 0, lkey = 0; };
struct SendWr {
  uint64_t wr_id = 0;
  WrOp opcode = WrOp::SEND;
  int send_flags = 0;
  uint32_t imm_data = 0;
  Sge sge;               // num_sge == 0: zero-length (imm-only writes, credit messages). INLINE: sge.addr is a host pointer, no lkey.
  int num_sge = 0;
  uint64_t remote_addr = 0;
  uint32_t rkey = 0;
};
struct RecvWr { uint64_t wr_id = 0; Sge sge; int num_sge = 0; };
struct WorkCompletion {
  uint64_t wr_id = 0;
  WcStatus status = WcStatus::SUCCESS;
  WcOp opcode = WcOp::SEND;
  uint32_t byte_len = 0, imm_data = 0, qp_num = 0;
  int wc_flags = 0;
};
struct QpAttr {
  QpState state = QpState::RESET;
  int access_flags = 0;
  uint32_t dest_qpn = 0, rq_psn = 0, sq_psn = 0;
  int rnr_retry = 7, retry_cnt = 7;
};
struct QpCaps { int max_send_wr = 256, max_recv_wr = 256, max_inline = 256; };
struct AsyncEvent { uint32_t qpn = 0; bool peer_initiated = false; };    // IBV_EVENT_QP_FATAL

class ShimFabric;
class ShimDevice;
class ShimPD;
class ShimQP;

class ShimMR {
 public:
  uint64_t addr = 0; size_t length = 0; int access = 0; uint32_t lkey = 0, rkey = 0;
  ShimPD* pd = nullptr;
  ~ShimMR();
  ShimFabric* fabric = nullptr;
};

class ShimCQ {
 public:
  explicit ShimCQ(int cap) : cap_(cap) {}
  // ibv_poll_cq: up to n completions, never blocks.
  int poll(int n, WorkCompletion* wc) {
    std::lock_guard<std::mutex> l(m_);
    int k = 0;
    while (k < n && !q_.empty()) { wc[k++] = q_.front(); q_.pop_front(); }
    return k;
  }
  // Optional blocking wait (stands in for the completion channel): true when at least one completion is queued.
  bool wait(int timeout_ms) {
    std::unique_lock<std::mutex> l(m_);
    cv_.wait_for(l, std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms), [&] { return !q_.empty() || intr_; });
    intr_ = false;
    return !q_.empty();
  }
  void interrupt() { { std::lock_guard<std::mutex> l(m_); intr_ = true; } cv_.notify_all(); }
  size_t size() { std::lock_guard<std::mutex> l(m_); return q_.size(); }
  std::atomic<long> overruns{0};
  void push(const WorkCompletion& w) {          // internal
    { std::lock_guard<std::mutex> l(m_); if ((int)q_.size() >= cap_) { ++overruns; return; } q_.push_back(w); }
    cv_.notify_one();
  }
 private:
  int cap_; std::mutex m_; std::condition_variable cv_; std::deque<WorkCompletion> q_; bool intr_ = false;
};

class ShimPD {
 public:
  ShimPD(ShimDevice& d, uint32_t id) : dev(d), id(id) {}
  // ibv_reg_mr. The memory must stay valid until the MR object is destroyed (ibv_dereg_mr).
  std::unique_ptr<ShimMR> reg_mr(void* p, size_t n, int access);
  ShimDevice& dev; uint32_t id;
};

class ShimQP {
 public:
  ~ShimQP();
  uint32_t qpn() const { return qpn_; }
  QpState state();
  int modify(const QpAttr& a, int mask);                       // 0 or -EINVAL
  int post_send(const SendWr& wr);                             // 0, -EINVAL (wrong state / bad wr), -ENOMEM (queue full)
  int post_recv(const RecvWr& wr);                             // 0, -EINVAL (state RESET or ERR), -ENOMEM
  void force_rnr(int n);                                       // TEST: the next n deliveries to this QP find "no receive posted"
  // ---- internal (fabric) ----
  struct Item { SendWr wr; std::vector<uint8_t> inl; bool inlined = false; SClock::time_point due; int rnr_tries = 0; };
  ShimFabric* fab = nullptr; ShimDevice* dev = nullptr; ShimPD* pd = nullptr; ShimCQ *scq = nullptr, *rcq = nullptr;
  uint32_t qpn_ = 0, dest_qpn = 0, rq_psn = 0, sq_psn = 0;
  QpState st = QpState::RESET;
  int rnr_retry = 7, retry_cnt = 7, access = 0, force_rnr_left = 0;
  QpCaps caps;
  std::deque<Item> sq; std::deque<RecvWr> rq;
};

class ShimDevice {
 public:
  explicit ShimDevice(ShimFabric& f, const char* name = "shim0") : fab(f), name(name) {}
  std::unique_ptr<ShimPD> alloc_pd();
  std::unique_ptr<ShimCQ> create_cq(int cap = 4096) { return std::make_unique<ShimCQ>(cap); }
  std::unique_ptr<ShimQP> create_qp(ShimPD& pd, ShimCQ& send_cq, ShimCQ& recv_cq, QpCaps caps = {});
  bool poll_async(AsyncEvent& ev);                             // ibv_get_async_event, non-blocking
  ShimFabric& fab; std::string name;
  std::deque<AsyncEvent> async;                                // internal, under the fabric lock
};

class ShimFabric {
 public:
  ShimFabric() : thr_([this] { loop(); }) {}
  ~ShimFabric() { { std::lock_guard<std::mutex> l(m_); stop_ = true; } cv_.notify_all(); thr_.join(); }
  ShimFabric(const ShimFabric&) = delete;
  // ---- knobs (tests) ----
  void set_delay_us(int us) { std::lock_guard<std::mutex> l(m_); delay_us_ = us; }
  void set_reorder_writes_us(int us) { std::lock_guard<std::mutex> l(m_); reorder_us_ = us; }   // breaks write ordering ON PURPOSE
  void set_rnr_delay_us(int us) { std::lock_guard<std::mutex> l(m_); rnr_delay_us_ = us; }
  void kill_qp(uint32_t qpn) {                                 // the QP's host "crashes": error + flush, peer notified
    std::lock_guard<std::mutex> l(m_);
    ShimQP* q = find_qp(qpn);
    if (q) to_error(*q, false, false);
    wake_locked();
  }
  std::atomic<long> st_sends{0}, st_writes{0}, st_imm_recv{0}, st_rnr{0}, st_errors{0};

  // ---- internal API used by the classes above (fabric lock held unless noted) ----
  std::mutex& mutex() { return m_; }
  uint32_t next_id() { return ++id_; }
  void add_qp(ShimQP* q) { qps_[q->qpn_] = q; }
  void del_qp(ShimQP* q) { qps_.erase(q->qpn_); }
  ShimQP* find_qp(uint32_t n) { auto it = qps_.find(n); return it == qps_.end() ? nullptr : it->second; }
  void add_mr(ShimMR* m) { lk_[m->lkey] = m; rk_[m->rkey] = m; }
  void del_mr(ShimMR* m) { lk_.erase(m->lkey); rk_.erase(m->rkey); }
  ShimMR* by_lkey(uint32_t k) { auto it = lk_.find(k); return it == lk_.end() ? nullptr : it->second; }
  ShimMR* by_rkey(uint32_t k) { auto it = rk_.find(k); return it == rk_.end() ? nullptr : it->second; }
  void wake_locked() { process_locked(); cv_.notify_all(); }

  static bool in_bounds(const ShimMR& m, uint64_t addr, uint64_t len) { return addr >= m.addr && addr + len >= addr && addr + len <= m.addr + m.length; }

  // Copy into registered memory. Small aligned stores are relaxed atomics so a CQ thread can read them race-free.
  static void copy_into(void* dst, const void* src, size_t n) {
    if (n == 4 && ((uintptr_t)dst & 3) == 0) { uint32_t v; memcpy(&v, src, 4); __atomic_store_n((uint32_t*)dst, v, __ATOMIC_RELAXED); return; }
    if (n == 8 && ((uintptr_t)dst & 7) == 0) { uint64_t v; memcpy(&v, src, 8); __atomic_store_n((uint64_t*)dst, v, __ATOMIC_RELAXED); return; }
    if (n && (((uintptr_t)dst | n) & 3) == 0 && ((uintptr_t)src & 3) == 0 && n <= 64) {
      for (size_t i = 0; i < n; i += 4) { uint32_t v; memcpy(&v, (const char*)src + i, 4); __atomic_store_n((uint32_t*)((char*)dst + i), v, __ATOMIC_RELAXED); }
      return;
    }
    if (n) memcpy(dst, src, n);
  }
  // Word-wise relaxed copy (used for deferred writes, which race with a concurrent reader BY DESIGN in the negative test).
  static void copy_into_atomic(void* dst, const void* src, size_t n) {
    size_t i = 0;
    if ((((uintptr_t)dst | (uintptr_t)src) & 3) == 0)
      for (; i + 4 <= n; i += 4) { uint32_t v; memcpy(&v, (const char*)src + i, 4); __atomic_store_n((uint32_t*)((char*)dst + i), v, __ATOMIC_RELAXED); }
    for (; i < n; ++i) __atomic_store_n((uint8_t*)dst + i, ((const uint8_t*)src)[i], __ATOMIC_RELAXED);
  }

  void to_error(ShimQP& q, bool from_peer, bool user) {
    if (q.st == QpState::ERR) return;
    q.st = QpState::ERR;
    while (!q.sq.empty()) {
      auto& w = q.sq.front().wr;
      WorkCompletion c; c.wr_id = w.wr_id; c.status = WcStatus::WR_FLUSH_ERR; c.qp_num = q.qpn_; c.opcode = (w.opcode == WrOp::RDMA_WRITE || w.opcode == WrOp::RDMA_WRITE_WITH_IMM) ? WcOp::RDMA_WRITE : WcOp::SEND;
      q.scq->push(c); q.sq.pop_front();
    }
    while (!q.rq.empty()) {
      WorkCompletion c; c.wr_id = q.rq.front().wr_id; c.status = WcStatus::WR_FLUSH_ERR; c.qp_num = q.qpn_; c.opcode = WcOp::RECV;
      q.rcq->push(c); q.rq.pop_front();
    }
    if (!user) { q.dev->async.push_back({q.qpn_, from_peer}); ++st_errors; }
    if (!from_peer) {
      ShimQP* p = find_qp(q.dest_qpn);
      if (p && p->dest_qpn == q.qpn_ && p->st != QpState::ERR && p->st != QpState::RESET && p->st != QpState::INIT) to_error(*p, true, false);
    }
  }

  void process_locked() {
    auto now = SClock::now();
    for (auto& kv : qps_) {
      ShimQP& q = *kv.second;
      while (q.st == QpState::RTS && !q.sq.empty() && q.sq.front().due <= now) if (exec_head(q) == Step::WAIT) break;
    }
    for (size_t i = 0; i < deferred_.size();) {
      if (deferred_[i].due > now) { ++i; continue; }
      Deferred d = std::move(deferred_[i]); deferred_.erase(deferred_.begin() + i);
      ShimMR* m = by_rkey(d.rkey);
      if (m && in_bounds(*m, d.addr, d.data.size())) copy_into_atomic((void*)d.addr, d.data.data(), d.data.size());
    }
  }

 private:
  enum class Step { DONE, WAIT, FAILED };
  struct Deferred { SClock::time_point due; uint64_t addr; uint32_t rkey; std::vector<uint8_t> data; };

  static WcOp send_op(WrOp o) { return (o == WrOp::RDMA_WRITE || o == WrOp::RDMA_WRITE_WITH_IMM) ? WcOp::RDMA_WRITE : WcOp::SEND; }

  Step fail(ShimQP& q, WcStatus s) {
    auto& w = q.sq.front().wr;
    WorkCompletion c; c.wr_id = w.wr_id; c.status = s; c.qp_num = q.qpn_; c.opcode = send_op(w.opcode);
    q.scq->push(c); q.sq.pop_front();
    KD_LOG(3, "shim", "qp %u: %s -> ERR", q.qpn_, to_string(s));
    to_error(q, false, false);
    return Step::FAILED;
  }
  Step rnr_wait(ShimQP& q, ShimQP::Item& it) {
    ++st_rnr;
    if (q.rnr_retry != 7 && ++it.rnr_tries > q.rnr_retry) return fail(q, WcStatus::RNR_RETRY_EXC_ERR);
    it.due = SClock::now() + std::chrono::microseconds(rnr_delay_us_);
    return Step::WAIT;
  }

  Step exec_head(ShimQP& q) {
    ShimQP::Item& it = q.sq.front();
    const SendWr& wr = it.wr;
    ShimQP* p = find_qp(q.dest_qpn);
    if (!p || p->st == QpState::ERR || p->st == QpState::RESET || p->dest_qpn != q.qpn_) return fail(q, WcStatus::RETRY_EXC_ERR);
    if (p->st != QpState::RTR && p->st != QpState::RTS) {          // peer not ready yet (still INIT): behaves like RNR
      if (q.retry_cnt != 7 && ++it.rnr_tries > q.retry_cnt) return fail(q, WcStatus::RETRY_EXC_ERR);
      it.due = SClock::now() + std::chrono::microseconds(rnr_delay_us_);
      return Step::WAIT;
    }
    const uint8_t* src = nullptr; size_t len = 0;
    if (it.inlined) { src = it.inl.data(); len = it.inl.size(); }
    else if (wr.num_sge) {
      ShimMR* m = by_lkey(wr.sge.lkey);
      if (!m || m->pd != q.pd || !in_bounds(*m, wr.sge.addr, wr.sge.length)) return fail(q, WcStatus::LOC_PROT_ERR);
      src = (const uint8_t*)wr.sge.addr; len = wr.sge.length;
    }
    bool write = wr.opcode == WrOp::RDMA_WRITE || wr.opcode == WrOp::RDMA_WRITE_WITH_IMM;
    bool imm = wr.opcode == WrOp::SEND_WITH_IMM || wr.opcode == WrOp::RDMA_WRITE_WITH_IMM;
    bool need_rq = wr.opcode != WrOp::RDMA_WRITE;
    ShimMR* rm = nullptr;
    if (write) {
      rm = by_rkey(wr.rkey);
      if (!rm || rm->pd != p->pd || !(rm->access & ACCESS_REMOTE_WRITE) || !in_bounds(*rm, wr.remote_addr, len)) return fail(q, WcStatus::REM_ACCESS_ERR);
    }
    ShimMR* lm = nullptr;
    if (need_rq) {
      bool forced = p->force_rnr_left > 0;
      if (forced) --p->force_rnr_left;
      if (forced || p->rq.empty()) return rnr_wait(q, it);
      if (!write) {                                                  // SEND: the receive buffer must fit and be locally writable
        const RecvWr& r = p->rq.front();
        WcStatus rs = WcStatus::SUCCESS;
        if (len > 0 && (r.num_sge == 0 || len > r.sge.length)) rs = WcStatus::LOC_LEN_ERR;
        else if (len > 0) { lm = by_lkey(r.sge.lkey); if (!lm || lm->pd != p->pd || !(lm->access & ACCESS_LOCAL_WRITE) || !in_bounds(*lm, r.sge.addr, len)) rs = WcStatus::LOC_PROT_ERR; }
        if (rs != WcStatus::SUCCESS) {
          WorkCompletion c; c.wr_id = r.wr_id; c.status = rs; c.qp_num = p->qpn_; c.opcode = WcOp::RECV; p->rcq->push(c); p->rq.pop_front();
          return fail(q, rs == WcStatus::LOC_LEN_ERR ? WcStatus::REM_INV_REQ_ERR : WcStatus::REM_ACCESS_ERR);
        }
      }
    }
    // ---- delivery: data first, then the receive completion ----
    if (write) {
      if (wr.opcode == WrOp::RDMA_WRITE && reorder_us_ > 0 && len > 0)
        deferred_.push_back({SClock::now() + std::chrono::microseconds(reorder_us_), wr.remote_addr, wr.rkey, std::vector<uint8_t>(src, src + len)});
      else copy_into((void*)wr.remote_addr, src, len);
      ++st_writes;
    }
    if (need_rq) {
      RecvWr r = p->rq.front(); p->rq.pop_front();
      if (!write) { if (len) copy_into((void*)r.sge.addr, src, len); ++st_sends; }
      WorkCompletion c; c.wr_id = r.wr_id; c.status = WcStatus::SUCCESS; c.qp_num = p->qpn_; c.byte_len = (uint32_t)len;
      c.opcode = write ? WcOp::RECV_RDMA_WITH_IMM : WcOp::RECV;
      if (imm) { c.imm_data = wr.imm_data; c.wc_flags = WC_WITH_IMM; }
      if (write) ++st_imm_recv;
      p->rcq->push(c);
    }
    if (wr.send_flags & SEND_SIGNALED) {
      WorkCompletion c; c.wr_id = wr.wr_id; c.status = WcStatus::SUCCESS; c.qp_num = q.qpn_; c.byte_len = (uint32_t)len; c.opcode = send_op(wr.opcode);
      q.scq->push(c);
    }
    q.sq.pop_front();
    return Step::DONE;
  }

  void loop() {
    std::unique_lock<std::mutex> l(m_);
    while (!stop_) {
      process_locked();
      SClock::time_point next = SClock::time_point::max();
      for (auto& kv : qps_) { ShimQP& q = *kv.second; if (q.st == QpState::RTS && !q.sq.empty()) next = std::min(next, q.sq.front().due); }
      for (auto& d : deferred_) next = std::min(next, d.due);
      if (next == SClock::time_point::max()) cv_.wait(l);
      else cv_.wait_until(l, next);
    }
  }

  friend class ShimQP;
  std::mutex m_; std::condition_variable cv_;
  std::unordered_map<uint32_t, ShimQP*> qps_;
  std::unordered_map<uint32_t, ShimMR*> lk_, rk_;
  std::vector<Deferred> deferred_;
  int delay_us_ = 0, reorder_us_ = 0, rnr_delay_us_ = 100;
  uint32_t id_ = 0x100;
  bool stop_ = false;
  std::thread thr_;
};

// ---------------------------------------------------------------------------------------------------- implementations
inline ShimMR::~ShimMR() {
  if (!fabric) return;
  std::lock_guard<std::mutex> l(fabric->mutex());
  fabric->del_mr(this);
}
inline std::unique_ptr<ShimMR> ShimPD::reg_mr(void* p, size_t n, int access) {
  auto m = std::make_unique<ShimMR>();
  std::lock_guard<std::mutex> l(dev.fab.mutex());
  uint32_t k = dev.fab.next_id();
  m->addr = (uint64_t)(uintptr_t)p; m->length = n; m->access = access; m->pd = this; m->fabric = &dev.fab;
  m->lkey = (k << 8) | 0xA5; m->rkey = (k << 8) | 0x5A;
  dev.fab.add_mr(m.get());
  return m;
}
inline std::unique_ptr<ShimPD> ShimDevice::alloc_pd() { std::lock_guard<std::mutex> l(fab.mutex()); return std::make_unique<ShimPD>(*this, fab.next_id()); }
inline std::unique_ptr<ShimQP> ShimDevice::create_qp(ShimPD& pd, ShimCQ& scq, ShimCQ& rcq, QpCaps caps) {
  auto q = std::make_unique<ShimQP>();
  std::lock_guard<std::mutex> l(fab.mutex());
  q->fab = &fab; q->dev = this; q->pd = &pd; q->scq = &scq; q->rcq = &rcq; q->caps = caps; q->qpn_ = fab.next_id();
  fab.add_qp(q.get());
  return q;
}
inline bool ShimDevice::poll_async(AsyncEvent& ev) {
  std::lock_guard<std::mutex> l(fab.mutex());
  if (async.empty()) return false;
  ev = async.front(); async.pop_front();
  return true;
}
inline ShimQP::~ShimQP() {
  if (!fab) return;
  std::lock_guard<std::mutex> l(fab->mutex());
  if (st != QpState::RESET) fab->to_error(*this, false, true);   // closing a connection: the peer finds out
  fab->del_qp(this);
}
inline QpState ShimQP::state() { std::lock_guard<std::mutex> l(fab->mutex()); return st; }
inline int ShimQP::modify(const QpAttr& a, int mask) {
  std::lock_guard<std::mutex> l(fab->mutex());
  if (mask & QP_ACCESS) access = a.access_flags;
  if (mask & QP_RNR_RETRY) rnr_retry = a.rnr_retry;
  if (mask & QP_RETRY_CNT) retry_cnt = a.retry_cnt;
  if (mask & QP_DEST_QPN) dest_qpn = a.dest_qpn;
  if (mask & QP_RQ_PSN) rq_psn = a.rq_psn;
  if (mask & QP_SQ_PSN) sq_psn = a.sq_psn;
  if (!(mask & QP_STATE)) return 0;
  QpState to = a.state;
  if (to == QpState::ERR) { fab->to_error(*this, false, true); fab->wake_locked(); return 0; }
  if (to == QpState::RESET) { st = QpState::RESET; sq.clear(); rq.clear(); dest_qpn = 0; return 0; }
  bool ok = (st == QpState::RESET && to == QpState::INIT) ||
            (st == QpState::INIT && to == QpState::RTR && (mask & QP_DEST_QPN) && (mask & QP_RQ_PSN)) ||
            (st == QpState::RTR && to == QpState::RTS && (mask & QP_SQ_PSN));
  if (!ok) return -EINVAL;
  st = to;
  fab->wake_locked();
  return 0;
}
inline int ShimQP::post_send(const SendWr& wr) {
  std::lock_guard<std::mutex> l(fab->mutex());
  if (st != QpState::RTS) return -EINVAL;
  if ((int)sq.size() >= caps.max_send_wr) return -ENOMEM;
  Item it; it.wr = wr;
  if ((wr.send_flags & SEND_INLINE) && wr.num_sge) {
    if (wr.sge.length > (uint32_t)caps.max_inline) return -EINVAL;
    it.inl.assign((const uint8_t*)(uintptr_t)wr.sge.addr, (const uint8_t*)(uintptr_t)wr.sge.addr + wr.sge.length);
    it.inlined = true;
  }
  it.due = SClock::now() + std::chrono::microseconds(fab->delay_us_);
  sq.push_back(std::move(it));
  fab->wake_locked();
  return 0;
}
inline int ShimQP::post_recv(const RecvWr& wr) {
  std::lock_guard<std::mutex> l(fab->mutex());
  if (st == QpState::RESET || st == QpState::ERR) return -EINVAL;
  if ((int)rq.size() >= caps.max_recv_wr) return -ENOMEM;
  rq.push_back(wr);
  for (auto& kv : fab->qps_) {                                       // a sender parked on RNR retries as soon as a receive appears
    ShimQP& s = *kv.second;
    if (s.dest_qpn == qpn_ && !s.sq.empty() && s.sq.front().rnr_tries > 0) s.sq.front().due = SClock::now();
  }
  fab->wake_locked();
  return 0;
}
inline void ShimQP::force_rnr(int n) { std::lock_guard<std::mutex> l(fab->mutex()); force_rnr_left = n; }

}  // namespace shim
}  // namespace kvmd
