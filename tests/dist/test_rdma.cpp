// RDMA transport tests. Pure C++17, no CUDA, no libibverbs: everything runs on the in-process SOFTWARE EMULATION of verbs
// (dist/verbs_shim.hpp). These tests check protocol logic; they do not (and cannot) show that a real NIC behaves the same.
//   g++ -O2 -std=c++17 -pthread -I dist tests/dist/test_rdma.cpp -o test_rdma      (-fsanitize=thread / address,undefined also clean)
#include <atomic>
#include <cmath>
#include <functional>

#include "../../dist/proxy.hpp"
#include "../../dist/transport_rdma.hpp"

using namespace kvmd;
using namespace kvmd::shim;

static std::atomic<int> g_fail{0};
static std::atomic<int> g_checks{0};
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; fprintf(stderr, "  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #c); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

// ------------------------------------------------------------------------------------------------ helpers (as in test_transport.cpp)
static std::vector<uint8_t> msg_of(uint32_t tag, size_t n) {
  std::vector<uint8_t> v(n);
  uint32_t x = tag * 2654435761u + 1;
  for (size_t i = 0; i < n; ++i) { x = x * 1664525u + 1013904223u; v[i] = (uint8_t)(x >> 24); }
  if (n >= 4) memcpy(v.data(), &tag, 4);
  return v;
}

// Two reliable endpoints exchange `count` messages each way; checks exactly-once in-order delivery of exact bytes.
// (Copied from tests/dist/test_transport.cpp, where it is the contract test for Loopback and TCP.)
static void exchange(const char* label, Transport& a, Transport& b, int count, size_t size, ReliableConfig rc, bool expect_retx = false) {
  Reliable ra(a, rc), rb(b, rc);
  std::atomic<int> finished{0};
  auto pump = [&](Reliable& r, int me, int n, std::vector<uint32_t>& got, std::atomic<bool>& stop, bool& ok) {
    int sent = 0; bool mine = false;
    while (!stop) {
      if (mine && finished == 2) return;
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
  if (ms_since(t0) > 50000) fprintf(stderr, "  HANG: A %s | B %s | got %zu/%zu\n", ra.debug(1).c_str(), rb.debug(0).c_str(), ga.size(), gb.size());
  bool inorder = true;
  for (size_t i = 0; i < gb.size(); ++i) inorder &= gb[i] == 0 * 1000000u + (uint32_t)i;
  for (size_t i = 0; i < ga.size(); ++i) inorder &= ga[i] == 1 * 1000000u + (uint32_t)i;
  CHECK(oka && okb, "%s: payload corrupted or peer declared dead", label);
  CHECK((int)ga.size() == count && (int)gb.size() == count, "%s: delivered %zu/%zu of %d", label, ga.size(), gb.size(), count);
  CHECK(inorder, "%s: not exactly-once in-order", label);
  if (expect_retx) CHECK(ra.st.retx + rb.st.retx > 0, "%s: faults injected but no retransmission", label);
  fprintf(stderr, "  %-38s %d x %zu B: retx %ld/%ld dups %ld/%ld reordered %ld/%ld  %d ms\n", label, count, size, ra.st.retx, rb.st.retx, ra.st.dups,
          rb.st.dups, ra.st.reordered, rb.st.reordered, ms_since(t0));
}
// 200 KiB messages: a 20 ms RTO is below the processing time of a loaded (TSAN) machine and floods the ring with retransmissions.
static ReliableConfig big_cfg() { ReliableConfig c; c.window_msgs = 16; c.rto_ms = 300; c.rto_max_ms = 1500; c.max_retries = 30; c.dead_ms = 60000; return c; }
static ReliableConfig fast_cfg() { ReliableConfig c; c.rto_ms = 20; c.rto_max_ms = 200; c.max_retries = 30; c.dead_ms = 20000; return c; }

// ================================================================================================== shim unit tests
// Two devices, one RC QP each, connected, one send CQ and one recv CQ per side.
struct QPair {
  ShimFabric fab;
  ShimDevice d0{fab, "a"}, d1{fab, "b"};
  std::unique_ptr<ShimPD> pd0, pd1;
  std::unique_ptr<ShimCQ> scq0, rcq0, scq1, rcq1;
  std::unique_ptr<ShimQP> q0, q1;
  explicit QPair(int rnr_retry = 7, QpCaps caps = {}, bool connect = true) {
    pd0 = d0.alloc_pd(); pd1 = d1.alloc_pd();
    scq0 = d0.create_cq(); rcq0 = d0.create_cq(); scq1 = d1.create_cq(); rcq1 = d1.create_cq();
    q0 = d0.create_qp(*pd0, *scq0, *rcq0, caps); q1 = d1.create_qp(*pd1, *scq1, *rcq1, caps);
    if (connect) {
      go(*q0, *q1, rnr_retry); go(*q1, *q0, rnr_retry);
    }
  }
  static void go(ShimQP& q, ShimQP& peer, int rnr_retry) {
    QpAttr a; a.state = QpState::INIT; a.access_flags = ACCESS_REMOTE_WRITE;
    q.modify(a, QP_STATE | QP_ACCESS);
    a.state = QpState::RTR; a.dest_qpn = peer.qpn(); a.rq_psn = 1;
    q.modify(a, QP_STATE | QP_DEST_QPN | QP_RQ_PSN);
    a.state = QpState::RTS; a.sq_psn = 1; a.rnr_retry = rnr_retry;
    q.modify(a, QP_STATE | QP_SQ_PSN | QP_RNR_RETRY);
  }
};

static bool poll1(ShimCQ& cq, WorkCompletion& wc, int timeout_ms = 1000) {
  if (cq.poll(1, &wc) == 1) return true;
  cq.wait(timeout_ms);
  return cq.poll(1, &wc) == 1;
}
static SendWr wr_write(uint64_t id, const ShimMR* src, size_t n, uint64_t raddr, uint32_t rkey, bool imm = false, uint32_t immv = 0, bool sig = true) {
  SendWr w; w.wr_id = id; w.opcode = imm ? WrOp::RDMA_WRITE_WITH_IMM : WrOp::RDMA_WRITE; w.imm_data = immv; w.send_flags = sig ? SEND_SIGNALED : 0;
  if (src) { w.sge = {src->addr, (uint32_t)n, src->lkey}; w.num_sge = 1; }
  w.remote_addr = raddr; w.rkey = rkey; return w;
}
static RecvWr wr_recv(uint64_t id, const ShimMR* m = nullptr, size_t n = 0) { RecvWr r; r.wr_id = id; if (m) { r.sge = {m->addr, (uint32_t)n, m->lkey}; r.num_sge = 1; } return r; }

static void test_shim_mr_errors() {
  fprintf(stderr, "== shim: memory regions, bounds, keys, permissions\n");
  auto fresh = [] { return std::make_unique<QPair>(); };
  std::vector<uint8_t> a(256, 0xAB), b(256, 0), ro(256, 0);
  auto expect_err = [&](const char* what, std::function<SendWr(QPair&, ShimMR&, ShimMR&, ShimMR&)> mk, WcStatus want, bool post_recv_first = false) {
    auto p = fresh();
    auto ma = p->pd0->reg_mr(a.data(), a.size(), ACCESS_LOCAL_WRITE);
    auto mb = p->pd1->reg_mr(b.data(), b.size(), ACCESS_LOCAL_WRITE | ACCESS_REMOTE_WRITE);
    auto mr = p->pd1->reg_mr(ro.data(), ro.size(), ACCESS_LOCAL_WRITE);                 // no REMOTE_WRITE
    if (post_recv_first) p->q1->post_recv(wr_recv(7));
    CHECK(p->q0->post_send(mk(*p, *ma, *mb, *mr)) == 0, "%s: post", what);
    WorkCompletion wc;
    bool got = poll1(*p->scq0, wc);
    CHECK(got && wc.status == want, "%s: wc status %s, want %s", what, got ? to_string(wc.status) : "none", to_string(want));
    CHECK(p->q0->state() == QpState::ERR && p->q1->state() == QpState::ERR, "%s: both QPs must be in ERR (%s/%s)", what, to_string(p->q0->state()), to_string(p->q1->state()));
    CHECK(p->q0->post_send(wr_write(2, nullptr, 0, 0, 0)) == -EINVAL, "%s: post in ERR must be rejected", what);
    AsyncEvent ev; CHECK(p->d1.poll_async(ev) && ev.peer_initiated, "%s: peer must get an async QP_FATAL event", what);
  };
  expect_err("write past the end of the MR", [&](QPair& p, ShimMR& ma, ShimMR& mb, ShimMR&) { (void)p; return wr_write(1, &ma, 128, mb.addr + 200, mb.rkey); }, WcStatus::REM_ACCESS_ERR);
  expect_err("write before the MR", [&](QPair&, ShimMR& ma, ShimMR& mb, ShimMR&) { return wr_write(1, &ma, 16, mb.addr - 8, mb.rkey); }, WcStatus::REM_ACCESS_ERR);
  expect_err("wrong rkey", [&](QPair&, ShimMR& ma, ShimMR& mb, ShimMR&) { return wr_write(1, &ma, 16, mb.addr, mb.rkey ^ 0x100); }, WcStatus::REM_ACCESS_ERR);
  expect_err("lkey used as rkey", [&](QPair&, ShimMR& ma, ShimMR& mb, ShimMR&) { return wr_write(1, &ma, 16, mb.addr, mb.lkey); }, WcStatus::REM_ACCESS_ERR);
  expect_err("no REMOTE_WRITE permission", [&](QPair&, ShimMR& ma, ShimMR&, ShimMR& mr) { return wr_write(1, &ma, 16, mr.addr, mr.rkey); }, WcStatus::REM_ACCESS_ERR);
  expect_err("local gather out of bounds", [&](QPair&, ShimMR& ma, ShimMR& mb, ShimMR&) { SendWr w = wr_write(1, &ma, 16, mb.addr, mb.rkey); w.sge.length = 512; return w; }, WcStatus::LOC_PROT_ERR);
  expect_err("bad lkey", [&](QPair&, ShimMR& ma, ShimMR& mb, ShimMR&) { SendWr w = wr_write(1, &ma, 16, mb.addr, mb.rkey); w.sge.lkey ^= 0x200; return w; }, WcStatus::LOC_PROT_ERR);
  {   // an rkey that exists but belongs to another protection domain
    auto p = fresh(); auto ma = p->pd0->reg_mr(a.data(), 64, 0);
    std::vector<uint8_t> other(64); auto pd2 = p->d1.alloc_pd(); auto m2 = pd2->reg_mr(other.data(), other.size(), ACCESS_REMOTE_WRITE | ACCESS_LOCAL_WRITE);
    p->q0->post_send(wr_write(1, ma.get(), 16, m2->addr, m2->rkey));
    WorkCompletion wc; CHECK(poll1(*p->scq0, wc) && wc.status == WcStatus::REM_ACCESS_ERR, "rkey of another PD must be refused");
  }
  {   // deregistered MR: its keys die
    auto p = fresh(); auto ma = p->pd0->reg_mr(a.data(), 64, 0); auto mb = p->pd1->reg_mr(b.data(), 64, ACCESS_REMOTE_WRITE);
    uint32_t rk = mb->rkey; uint64_t ad = mb->addr; mb.reset();
    p->q0->post_send(wr_write(1, ma.get(), 16, ad, rk));
    WorkCompletion wc; CHECK(poll1(*p->scq0, wc) && wc.status == WcStatus::REM_ACCESS_ERR, "use after dereg must fail");
  }
  {   // SEND into a receive buffer that is not LOCAL_WRITE: responder LOC_PROT_ERR, requester REM_ACCESS_ERR
    auto p = fresh(); auto ma = p->pd0->reg_mr(a.data(), 64, 0); auto mb = p->pd1->reg_mr(b.data(), 64, ACCESS_REMOTE_WRITE);
    p->q1->post_recv(wr_recv(5, mb.get(), 64));
    SendWr w; w.wr_id = 1; w.opcode = WrOp::SEND; w.send_flags = SEND_SIGNALED; w.sge = {ma->addr, 16, ma->lkey}; w.num_sge = 1;
    p->q0->post_send(w);
    WorkCompletion wc;
    CHECK(poll1(*p->rcq1, wc) && wc.status == WcStatus::LOC_PROT_ERR && wc.wr_id == 5, "receive buffer without LOCAL_WRITE");
    CHECK(poll1(*p->scq0, wc) && wc.status == WcStatus::REM_ACCESS_ERR, "requester status");
  }
  {   // SEND longer than the posted receive
    auto p = fresh(); auto ma = p->pd0->reg_mr(a.data(), 64, 0); auto mb = p->pd1->reg_mr(b.data(), 64, ACCESS_LOCAL_WRITE);
    p->q1->post_recv(wr_recv(5, mb.get(), 8));
    SendWr w; w.wr_id = 1; w.opcode = WrOp::SEND; w.sge = {ma->addr, 16, ma->lkey}; w.num_sge = 1;
    p->q0->post_send(w);
    WorkCompletion wc;
    CHECK(poll1(*p->rcq1, wc) && wc.status == WcStatus::LOC_LEN_ERR, "recv too small: responder");
    CHECK(poll1(*p->scq0, wc) && wc.status == WcStatus::REM_INV_REQ_ERR, "recv too small: requester (even though unsignalled)");
  }
}

static void test_shim_qp_state() {
  fprintf(stderr, "== shim: QP state machine\n");
  QPair p(7, {}, false);
  ShimQP& q = *p.q0;
  auto mod = [&](QpState s, int mask) { QpAttr a; a.state = s; a.dest_qpn = p.q1->qpn(); return q.modify(a, mask | QP_STATE); };
  CHECK(q.state() == QpState::RESET, "initial state");
  CHECK(mod(QpState::RTR, QP_DEST_QPN | QP_RQ_PSN) == -EINVAL, "RESET->RTR illegal");
  CHECK(mod(QpState::RTS, QP_SQ_PSN) == -EINVAL, "RESET->RTS illegal");
  CHECK(q.post_send(wr_write(1, nullptr, 0, 0, 0)) == -EINVAL, "post_send in RESET");
  CHECK(q.post_recv(wr_recv(1)) == -EINVAL, "post_recv in RESET");
  CHECK(mod(QpState::INIT, 0) == 0, "RESET->INIT");
  CHECK(mod(QpState::INIT, 0) == -EINVAL, "INIT->INIT illegal");
  CHECK(mod(QpState::RTS, QP_SQ_PSN) == -EINVAL, "INIT->RTS illegal");
  CHECK(mod(QpState::RTR, 0) == -EINVAL, "INIT->RTR without dest/psn illegal");
  CHECK(q.post_send(wr_write(1, nullptr, 0, 0, 0)) == -EINVAL, "post_send in INIT");
  CHECK(q.post_recv(wr_recv(1)) == 0, "post_recv in INIT is legal");
  CHECK(mod(QpState::RTR, QP_DEST_QPN | QP_RQ_PSN) == 0, "INIT->RTR");
  CHECK(q.post_send(wr_write(1, nullptr, 0, 0, 0)) == -EINVAL, "post_send in RTR");
  CHECK(mod(QpState::RTS, 0) == -EINVAL, "RTR->RTS without sq_psn illegal");
  CHECK(mod(QpState::RTS, QP_SQ_PSN) == 0, "RTR->RTS");
  CHECK(mod(QpState::RTR, QP_DEST_QPN | QP_RQ_PSN) == -EINVAL, "RTS->RTR illegal");
  CHECK(q.post_recv(wr_recv(2)) == 0, "post_recv in RTS");
  CHECK(mod(QpState::ERR, 0) == 0 && q.state() == QpState::ERR, "->ERR from any state");
  WorkCompletion wc;
  CHECK(poll1(*p.rcq0, wc) && wc.status == WcStatus::WR_FLUSH_ERR, "posted receives flush on ERR");
  CHECK(poll1(*p.rcq0, wc) && wc.status == WcStatus::WR_FLUSH_ERR, "second receive flushes too");
  CHECK(q.post_send(wr_write(1, nullptr, 0, 0, 0)) == -EINVAL && q.post_recv(wr_recv(3)) == -EINVAL, "posting in ERR rejected");
  CHECK(mod(QpState::RTS, QP_SQ_PSN) == -EINVAL, "ERR->RTS illegal");
  CHECK(mod(QpState::RESET, 0) == 0 && mod(QpState::INIT, 0) == 0, "ERR->RESET->INIT recovers");
  {   // queue depth
    QpCaps c; c.max_send_wr = 2; c.max_recv_wr = 2; QPair q2(7, c); q2.fab.set_delay_us(200000);
    CHECK(q2.q0->post_send(wr_write(1, nullptr, 0, 0, 0)) == 0 && q2.q0->post_send(wr_write(2, nullptr, 0, 0, 0)) == 0, "two fit");
    CHECK(q2.q0->post_send(wr_write(3, nullptr, 0, 0, 0)) == -ENOMEM, "third exceeds max_send_wr");
    CHECK(q2.q0->post_recv(wr_recv(1)) == 0 && q2.q0->post_recv(wr_recv(2)) == 0 && q2.q0->post_recv(wr_recv(3)) == -ENOMEM, "max_recv_wr");
  }
}

static void test_shim_completions() {
  fprintf(stderr, "== shim: opcodes, immediate data, signalled / unsignalled, SEND\n");
  QPair p;
  std::vector<uint8_t> src(256), dst(256, 0), rbuf(64, 0);
  for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i * 7 + 1);
  auto ms = p.pd0->reg_mr(src.data(), src.size(), ACCESS_LOCAL_WRITE);
  auto md = p.pd1->reg_mr(dst.data(), dst.size(), ACCESS_REMOTE_WRITE | ACCESS_LOCAL_WRITE);
  auto mrx = p.pd1->reg_mr(rbuf.data(), rbuf.size(), ACCESS_LOCAL_WRITE);
  WorkCompletion wc;
  // plain RDMA_WRITE: data lands, no receive consumed, no remote completion; unsignalled => no local completion either
  p.q1->post_recv(wr_recv(100));
  CHECK(p.q0->post_send(wr_write(1, ms.get(), 64, md->addr, md->rkey, false, 0, false)) == 0, "post");
  CHECK(memcmp(dst.data(), src.data(), 64) == 0, "plain write placed the data");
  CHECK(p.rcq1->size() == 0, "plain RDMA_WRITE must not generate a receive completion");
  CHECK(p.scq0->size() == 0, "unsignalled send must not complete");
  CHECK(p.q1->state() == QpState::RTS, "still fine");
  // signalled plain write
  p.q0->post_send(wr_write(2, ms.get(), 64, md->addr + 64, md->rkey, false, 0, true));
  CHECK(poll1(*p.scq0, wc) && wc.wr_id == 2 && wc.status == WcStatus::SUCCESS && wc.opcode == WcOp::RDMA_WRITE && wc.byte_len == 64, "signalled write completion");
  CHECK(p.rcq1->size() == 0, "still no receive completion");
  // write with immediate: consumes the receive, imm + byte_len reported
  p.q0->post_send(wr_write(3, ms.get(), 100, md->addr + 128, md->rkey, true, 0xCAFEBABE, false));
  CHECK(poll1(*p.rcq1, wc) && wc.wr_id == 100 && wc.status == WcStatus::SUCCESS && wc.opcode == WcOp::RECV_RDMA_WITH_IMM && wc.imm_data == 0xCAFEBABEu &&
        wc.byte_len == 100 && (wc.wc_flags & WC_WITH_IMM), "write-with-imm receive completion");
  CHECK(memcmp(dst.data() + 128, src.data(), 100) == 0, "data placed before the completion");
  CHECK(p.scq0->size() == 0, "unsignalled write-with-imm: no sender completion");
  // zero-length write with immediate (credit style)
  p.q1->post_recv(wr_recv(101));
  p.q0->post_send(wr_write(4, nullptr, 0, md->addr, md->rkey, true, 7, false));
  CHECK(poll1(*p.rcq1, wc) && wc.wr_id == 101 && wc.imm_data == 7 && wc.byte_len == 0, "zero-length imm");
  // SEND / SEND_WITH_IMM into a posted buffer
  p.q1->post_recv(wr_recv(102, mrx.get(), 64));
  SendWr s; s.wr_id = 5; s.opcode = WrOp::SEND_WITH_IMM; s.imm_data = 99; s.send_flags = SEND_SIGNALED; s.sge = {ms->addr, 40, ms->lkey}; s.num_sge = 1;
  p.q0->post_send(s);
  CHECK(poll1(*p.rcq1, wc) && wc.wr_id == 102 && wc.opcode == WcOp::RECV && wc.byte_len == 40 && wc.imm_data == 99, "SEND_WITH_IMM receive");
  CHECK(memcmp(rbuf.data(), src.data(), 40) == 0, "SEND payload in the receive buffer");
  CHECK(poll1(*p.scq0, wc) && wc.opcode == WcOp::SEND && wc.wr_id == 5, "SEND completion");
  // INLINE: the source may be reused / unregistered immediately
  { uint32_t v = 0x11223344; SendWr w = wr_write(6, nullptr, 0, md->addr + 200, md->rkey, false, 0, true);
    w.send_flags |= SEND_INLINE; w.sge = {(uint64_t)(uintptr_t)&v, 4, 0}; w.num_sge = 1;
    CHECK(p.q0->post_send(w) == 0, "inline post"); v = 0;
    CHECK(poll1(*p.scq0, wc) && wc.status == WcStatus::SUCCESS, "inline completion");
    uint32_t got; memcpy(&got, dst.data() + 200, 4); CHECK(got == 0x11223344u, "inline data copied at post time"); }
  // CQ: non-blocking poll and blocking wait
  CHECK(p.rcq1->poll(1, &wc) == 0, "empty poll returns 0");
  auto t0 = Clock::now(); CHECK(!p.rcq1->wait(30), "wait on an empty CQ times out"); CHECK(ms_since(t0) >= 25, "wait slept");
  std::thread th([&] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); p.q1->post_recv(wr_recv(103)); p.q0->post_send(wr_write(8, nullptr, 0, md->addr, md->rkey, true, 5, false)); });
  CHECK(p.rcq1->wait(2000), "blocking wait wakes when a completion arrives"); th.join();
  CHECK(p.rcq1->poll(1, &wc) == 1 && wc.imm_data == 5, "completion after wake");
}

static void test_shim_rnr() {
  fprintf(stderr, "== shim: receiver-not-ready\n");
  std::vector<uint8_t> a(64, 1), b(64, 0);
  {   // finite retry count: no receive ever posted => RNR_RETRY_EXC_ERR, both QPs error
    QPair p(2); p.fab.set_rnr_delay_us(200);
    auto ma = p.pd0->reg_mr(a.data(), 64, 0); auto mb = p.pd1->reg_mr(b.data(), 64, ACCESS_REMOTE_WRITE);
    p.q0->post_send(wr_write(1, ma.get(), 16, mb->addr, mb->rkey, true, 3, false));
    WorkCompletion wc; CHECK(poll1(*p.scq0, wc) && wc.status == WcStatus::RNR_RETRY_EXC_ERR, "RNR retry exceeded: %s", to_string(wc.status));
    CHECK(p.fab.st_rnr >= 3, "retried: %ld RNR events", p.fab.st_rnr.load());
    CHECK(p.q0->state() == QpState::ERR && p.q1->state() == QpState::ERR, "QPs in ERR");
  }
  {   // infinite retry (7): a late receive is enough
    QPair p(7); p.fab.set_rnr_delay_us(200);
    auto ma = p.pd0->reg_mr(a.data(), 64, 0); auto mb = p.pd1->reg_mr(b.data(), 64, ACCESS_REMOTE_WRITE);
    p.q0->post_send(wr_write(1, ma.get(), 16, mb->addr, mb->rkey, true, 3, false));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    WorkCompletion wc; CHECK(p.rcq1->poll(1, &wc) == 0 && p.q0->state() == QpState::RTS, "still waiting, QP alive");
    p.q1->post_recv(wr_recv(9));
    CHECK(poll1(*p.rcq1, wc) && wc.status == WcStatus::SUCCESS && wc.imm_data == 3, "delivered after the receive appeared");
    CHECK(p.fab.st_rnr > 0, "RNR was exercised");
  }
  {   // forced RNR knob: n artificial misses, then success with a receive already posted
    QPair p(7); p.fab.set_rnr_delay_us(100);
    auto ma = p.pd0->reg_mr(a.data(), 64, 0); auto mb = p.pd1->reg_mr(b.data(), 64, ACCESS_REMOTE_WRITE);
    p.q1->post_recv(wr_recv(9)); p.q1->force_rnr(3);
    p.q0->post_send(wr_write(1, ma.get(), 16, mb->addr, mb->rkey, true, 4, false));
    WorkCompletion wc; CHECK(poll1(*p.rcq1, wc) && wc.imm_data == 4, "delivered after forced RNRs");
    CHECK(p.fab.st_rnr == 3, "exactly 3 forced misses, got %ld", p.fab.st_rnr.load());
  }
  {   // in-order: a message stuck on RNR holds back the ones behind it (RC semantics)
    QPair p(7); p.fab.set_rnr_delay_us(100);
    auto ma = p.pd0->reg_mr(a.data(), 64, 0); auto mb = p.pd1->reg_mr(b.data(), 64, ACCESS_REMOTE_WRITE);
    p.q0->post_send(wr_write(1, ma.get(), 8, mb->addr, mb->rkey, true, 1, false));
    p.q0->post_send(wr_write(2, ma.get(), 8, mb->addr, mb->rkey, false, 0, true));   // plain write behind it
    WorkCompletion wc; std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(p.scq0->poll(1, &wc) == 0, "the write behind a stalled write-with-imm must not overtake it");
    p.q1->post_recv(wr_recv(9));
    CHECK(poll1(*p.rcq1, wc) && wc.imm_data == 1, "first");
    CHECK(poll1(*p.scq0, wc) && wc.wr_id == 2, "second completes after the first");
  }
}

static void test_shim_flush_and_peer_down() {
  fprintf(stderr, "== shim: QP error flushes outstanding WRs and takes the peer down\n");
  std::vector<uint8_t> a(64, 1), b(64, 0);
  {   // delayed wire: three unsignalled sends are outstanding when the QP is killed => three FLUSH_ERR completions anyway
    QPair p; p.fab.set_delay_us(300000);
    auto ma = p.pd0->reg_mr(a.data(), 64, 0); auto mb = p.pd1->reg_mr(b.data(), 64, ACCESS_REMOTE_WRITE);
    for (int i = 0; i < 3; ++i) p.q0->post_send(wr_write(10 + i, ma.get(), 8, mb->addr, mb->rkey, false, 0, false));
    for (int i = 0; i < 4; ++i) p.q1->post_recv(wr_recv(20 + i));
    CHECK(p.scq0->size() == 0, "nothing delivered yet");
    p.fab.kill_qp(p.q0->qpn());
    WorkCompletion wc; int fl = 0;
    for (int i = 0; i < 3 && poll1(*p.scq0, wc); ++i) fl += wc.status == WcStatus::WR_FLUSH_ERR && wc.wr_id >= 10 && wc.wr_id < 13;
    CHECK(fl == 3, "outstanding sends flushed (%d)", fl);
    int rfl = 0;
    for (int i = 0; i < 4 && poll1(*p.rcq1, wc); ++i) rfl += wc.status == WcStatus::WR_FLUSH_ERR && wc.wr_id >= 20;
    CHECK(rfl == 4, "the PEER's posted receives flush too (%d)", rfl);
    CHECK(p.q1->state() == QpState::ERR, "peer QP in ERR");
    AsyncEvent ev; CHECK(p.d1.poll_async(ev) && ev.qpn == p.q1->qpn() && ev.peer_initiated, "peer async event"); CHECK(p.d0.poll_async(ev) && !ev.peer_initiated, "local async event");
    CHECK(memcmp(b.data(), a.data(), 8) != 0 || b[0] == 0, "flushed writes were never delivered");
  }
  {   // destroying a QP (closing the connection) is seen by the peer
    QPair p; p.q1->post_recv(wr_recv(1));
    p.q0.reset();
    WorkCompletion wc; CHECK(poll1(*p.rcq1, wc) && wc.status == WcStatus::WR_FLUSH_ERR, "peer sees the close as a flush");
    CHECK(p.q1->state() == QpState::ERR, "peer QP ERR after the other side was destroyed");
    auto ma = p.pd1->reg_mr(a.data(), 8, 0);
    CHECK(p.q1->post_send(wr_write(1, ma.get(), 8, 0, 0)) == -EINVAL, "no sends on a dead QP");
  }
}

// Reader thread checks data-before-immediate. Tile i is 64 B; the writer posts the data write (unsignalled) and then a zero-length
// write-with-imm(i) on the same QP. When the reader sees imm i, tile i must be complete.
static long ordering_run(int reorder_us, int tiles) {
  QpCaps caps; caps.max_send_wr = 64; caps.max_recv_wr = tiles + 8;
  QPair p(7, caps); p.fab.set_reorder_writes_us(reorder_us);
  const size_t T = 64;
  std::vector<uint8_t> src(tiles * T), dst(tiles * T, 0);
  for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(1 + (i / T) % 250 + i % 3);
  auto ms = p.pd0->reg_mr(src.data(), src.size(), 0); auto md = p.pd1->reg_mr(dst.data(), dst.size(), ACCESS_REMOTE_WRITE);
  for (int i = 0; i < tiles; ++i) p.q1->post_recv(wr_recv(i));
  std::atomic<long> bad{0}; std::atomic<int> seen{0};
  std::thread reader([&] {
    WorkCompletion wc;
    while (seen < tiles) {
      if (p.rcq1->poll(1, &wc) != 1) { p.rcq1->wait(5); continue; }
      uint32_t i = wc.imm_data;
      bool ok = wc.status == WcStatus::SUCCESS && i < (uint32_t)tiles;
      for (size_t k = 0; ok && k < T; ++k) ok &= __atomic_load_n(&dst[i * T + k], __ATOMIC_RELAXED) == src[i * T + k];
      if (!ok) ++bad;
      ++seen;
    }
  });
  for (int i = 0; i < tiles; ++i) {
    SendWr dw = wr_write(i, ms.get(), T, md->addr + i * T, md->rkey, false, 0, false); dw.sge.addr = ms->addr + i * T;
    p.q0->post_send(dw);
    while (p.q0->post_send(wr_write(i, nullptr, 0, md->addr + i * T, md->rkey, true, i, false)) == -ENOMEM) std::this_thread::yield();
  }
  reader.join();
  return bad;
}

static void test_shim_ordering() {
  fprintf(stderr, "== shim: data is visible before the write-with-immediate completion (in-order delivery)\n");
  long bad = ordering_run(0, 3000);
  CHECK(bad == 0, "%ld tiles incomplete when their immediate arrived", bad);
  long bad2 = ordering_run(3000, 300);
  CHECK(bad2 > 0, "negative test: with the reorder knob (data lands late) the checker must see incomplete tiles, saw %ld", bad2);
  fprintf(stderr, "  ordered: %ld violations in 3000 tiles; reorder knob: %ld violations in 300 (the test can fail)\n", bad, bad2);
  {   // wire delay preserves order and delays completions
    QPair p; p.fab.set_delay_us(3000);
    std::vector<uint8_t> a(8, 1), b(8, 0);
    auto ma = p.pd0->reg_mr(a.data(), 8, 0); auto mb = p.pd1->reg_mr(b.data(), 8, ACCESS_REMOTE_WRITE);
    for (int i = 0; i < 5; ++i) p.q1->post_recv(wr_recv(i));
    auto t0 = Clock::now();
    for (int i = 0; i < 5; ++i) p.q0->post_send(wr_write(i, ma.get(), 8, mb->addr, mb->rkey, true, i, false));
    WorkCompletion wc; bool inorder = true;
    for (int i = 0; i < 5; ++i) { bool g = poll1(*p.rcq1, wc); inorder &= g && wc.imm_data == (uint32_t)i; }
    CHECK(inorder, "delayed completions in order"); CHECK(ms_since(t0) >= 2, "delay applied");
  }
}

// ================================================================================================== RdmaTransport
struct RdmaPair {
  ShimFabric fab; InProcRendezvous rdz;
  std::vector<std::unique_ptr<ShimVerbs>> verbs; std::vector<std::unique_ptr<RdmaTransport>> t;
  bool ok = true;
  explicit RdmaPair(RdmaConfig c = {}, int world = 2) {
    for (int r = 0; r < world; ++r) verbs.push_back(std::make_unique<ShimVerbs>(fab, ("dev" + std::to_string(r)).c_str()));
    t.resize(world);
    for (int r = 0; r < world; ++r) t[r] = std::make_unique<RdmaTransport>(*verbs[r], r, world, rdz, c);
    std::vector<std::thread> th; std::atomic<int> bad{0};
    for (int r = 0; r < world; ++r) th.emplace_back([&, r] { if (!t[r]->connect()) ++bad; });
    for (auto& x : th) x.join();
    ok = bad == 0;
  }
  ~RdmaPair() { for (auto& x : t) x->close(); }
};

static RdmaConfig small_cfg(int slots = 16) { RdmaConfig c; c.slots = slots; c.max_msg = 8192; c.max_pending_msgs = 4096; return c; }

static void suite_transport_basic() {
  fprintf(stderr, "== RdmaTransport (shim): contract\n");
  RdmaPair p(small_cfg());
  CHECK(p.ok, "connect");
  Transport &a = *p.t[0], &b = *p.t[1];
  CHECK(a.rank() == 0 && b.rank() == 1 && a.world() == 2 && std::string(a.name()) == "rdma", "identity");
  Event ev;
  CHECK(!a.poll(ev, 20), "poll times out when idle");
  auto m = msg_of(1, 100);
  CHECK(a.send(1, m.data(), m.size()), "send");
  CHECK(b.poll(ev, 1000) && ev.type == EvType::RECV && ev.peer == 0 && ev.data == m, "recv exact bytes");
  CHECK(!a.send(0, m.data(), m.size()), "send to self");
  CHECK(!a.send(5, m.data(), m.size()) && !a.send(-1, m.data(), m.size()), "send to a bad rank");
  std::vector<uint8_t> big(9000, 1);
  CHECK(!a.send(1, big.data(), big.size()), "message larger than max_msg is refused");
  CHECK(a.send(1, nullptr, 0), "empty message"); CHECK(b.poll(ev, 1000) && ev.data.empty(), "empty message delivered");
  for (int i = 0; i < 500; ++i) { auto x = msg_of(i, 65 + i % 300); a.send(1, x.data(), x.size()); }
  bool ordered = true;
  for (int i = 0; i < 500; ++i) { if (!b.poll(ev, 2000)) { ordered = false; break; } ordered &= ev.data == msg_of(i, 65 + i % 300); }
  CHECK(ordered, "500 messages, exact bytes, in order");
  CHECK(p.fab.st_rnr == 0, "no receiver-not-ready events with credit flow control (%ld)", p.fab.st_rnr.load());
  // bidirectional at the same time
  for (int i = 0; i < 200; ++i) { auto x = msg_of(i, 50); a.send(1, x.data(), 50); b.send(0, x.data(), 50); }
  int ga = 0, gb = 0;
  while (ga < 200 || gb < 200) { if (a.poll(ev, 500)) ++ga; else break; if (b.poll(ev, 500)) ++gb; }
  CHECK(ga == 200 && gb == 200, "bidirectional %d/%d", ga, gb);
}

static void suite_transport_credits() {
  fprintf(stderr, "== RdmaTransport (shim): credit flow control / back-pressure with a small ring\n");
  {   // 4 slots; the receiver does not poll: the sender must stop after 4 messages, queue the rest, and never overrun the ring
    RdmaConfig c = small_cfg(4); c.max_pending_msgs = 1000;
    RdmaPair p(c);
    for (int i = 0; i < 100; ++i) { auto m = msg_of(i, 64); CHECK(p.t[0]->send(1, m.data(), m.size()), "send %d must be accepted (queued)", i); }
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    CHECK(p.t[0]->msgs_out == 4, "only the ring size may be in flight while the receiver is idle: %ld posted", p.t[0]->msgs_out.load());
    CHECK(p.t[0]->credit_stalls > 0, "sender stalled on credits");
    Event ev; bool ordered = true; int n = 0;
    while (n < 100 && p.t[1]->poll(ev, 1000)) { ordered &= ev.type == EvType::RECV && ev.data == msg_of(n, 64); ++n; }
    CHECK(n == 100 && ordered, "all 100 delivered in order after the receiver drained (%d)", n);
    CHECK(p.t[0]->max_inflight <= 4, "in-flight never exceeded the %d receive slots: %ld", 4, p.t[0]->max_inflight.load());
    CHECK(p.fab.st_rnr == 0, "never an RNR: the ring was never overrun (%ld)", p.fab.st_rnr.load());
    CHECK(p.t[1]->credit_msgs + p.t[0]->credit_msgs > 0, "explicit credit messages were used (one-way traffic has no piggyback path)");
    CHECK(p.t[0]->overflow_dropped == 0, "nothing dropped");
  }
  {   // bounded queue: beyond max_pending the transport drops (lossy contract) and still says true
    RdmaConfig c = small_cfg(4); c.max_pending_msgs = 10;
    RdmaPair p(c);
    bool all_true = true;
    for (int i = 0; i < 100; ++i) { auto m = msg_of(i, 64); all_true &= p.t[0]->send(1, m.data(), m.size()); }
    CHECK(all_true, "overflow is a drop, not a failure");
    CHECK(p.t[0]->overflow_dropped == 100 - 4 - 10, "dropped %ld, expected %d", p.t[0]->overflow_dropped.load(), 100 - 4 - 10);
    Event ev; int n = 0; while (p.t[1]->poll(ev, 300)) ++n;
    CHECK(n == 14, "delivered %d of the 14 accepted", n);
  }
  {   // Reliable on top of a 4-slot ring: its window (256 msgs) is far larger than the ring; credits + queue absorb it
    RdmaConfig c = small_cfg(4); RdmaPair p(c);
    ReliableConfig rc = fast_cfg(); rc.window_msgs = 64;
    exchange("reliable over rdma, 4 slots", *p.t[0], *p.t[1], 1500, 100, rc);
    CHECK(p.t[0]->max_inflight <= 4 && p.t[1]->max_inflight <= 4, "ring overrun");
    fprintf(stderr, "  (control-message receive tokens ran dry %ld times; the shim's infinite RNR retry absorbs it)\n", p.fab.st_rnr.load());
  }
}

static void suite_transport_reliable() {
  fprintf(stderr, "== Reliable over RdmaTransport (and over FaultyTransport over RdmaTransport)\n");
  { RdmaPair p(small_cfg(64)); exchange("rdma clean", *p.t[0], *p.t[1], 3000, 200, fast_cfg()); }
  { RdmaConfig c; c.slots = 16; RdmaPair p(c); exchange("rdma 200 KiB messages (default max_msg)", *p.t[0], *p.t[1], 40, 200 << 10, big_cfg());
    if (getenv("RDMA_DEBUG")) fprintf(stderr, "    %s | %s\n", p.t[0]->debug(1).c_str(), p.t[1]->debug(0).c_str()); }
  {
    RdmaPair p(small_cfg(8)); FaultSpec f; f.drop = 0.2; f.dup = 0.2; f.delay = 0.3; f.delay_ms_max = 20; f.seed = 31;
    FaultyTransport a(*p.t[0], f), b(*p.t[1], f);
    exchange("faulty(rdma) drop+dup+delay", a, b, 600, 128, fast_cfg(), true);
  }
  {   // WAN emulation on top (latency), small ring
    RdmaPair p(small_cfg(32)); FaultSpec f; f.latency_ms = 5; f.seed = 5;
    FaultyTransport a(*p.t[0], f), b(*p.t[1], f);
    ReliableConfig rc = fast_cfg(); rc.rto_ms = 100; rc.rto_max_ms = 400;
    exchange("faulty(rdma) 5 ms latency", a, b, 300, 512, rc);
  }
  {   // bootstrap over another Transport (what a real deployment does with TcpTransport)
    LoopbackHub hub(2); LoopbackTransport l0(hub, 0), l1(hub, 1);
    TransportRendezvous r0(l0), r1(l1);
    ShimFabric fab; ShimVerbs v0(fab, "x"), v1(fab, "y");
    RdmaTransport t0(v0, 0, 2, r0, small_cfg()), t1(v1, 1, 2, r1, small_cfg());
    bool ok0 = false, ok1 = false;
    std::thread th([&] { ok1 = t1.connect(); }); ok0 = t0.connect(); th.join();
    CHECK(ok0 && ok1, "connect with TransportRendezvous");
    if (ok0 && ok1) exchange("rdma bootstrapped via TransportRendezvous", t0, t1, 500, 100, fast_cfg());
  }
  {   // 3 ranks all to all
    RdmaPair p(small_cfg(8), 3);
    CHECK(p.ok, "3-rank connect");
    std::vector<std::unique_ptr<Reliable>> rel; for (int r = 0; r < 3; ++r) rel.push_back(std::make_unique<Reliable>(*p.t[r], fast_cfg()));
    std::vector<std::vector<uint32_t>> got(3); std::atomic<int> done{0}; std::vector<std::thread> w;
    for (int r = 0; r < 3; ++r) w.emplace_back([&, r] {
      int sent = 0; const int N = 200; auto t0 = Clock::now();
      while (ms_since(t0) < 30000) {
        if (sent < 2 * N) { int dst = (r + 1 + sent % 2) % 3; auto m = msg_of(r * 100000 + sent, 40); if (rel[r]->send(dst, m.data(), m.size()) == Reliable::OK) ++sent; }
        Reliable::Delivery d; if (rel[r]->poll(d, 0)) { uint32_t tag; memcpy(&tag, d.data.data(), 4); got[r].push_back(tag); }
        if (sent == 2 * N && (int)got[r].size() == 2 * N && rel[r]->unacked_total() == 0) { ++done; break; }
      }
      while (done < 3 && ms_since(t0) < 30000) { Reliable::Delivery d; rel[r]->poll(d, 2); }
    });
    for (auto& x : w) x.join();
    CHECK(done == 3, "only %d/3 ranks completed", done.load());
    for (int r = 0; r < 3; ++r) { std::map<int, uint32_t> last; bool ok = true; for (uint32_t tag : got[r]) { int src = tag / 100000; ok &= !last.count(src) || tag > last[src]; last[src] = tag; } CHECK(ok, "rank %d per-source order", r); }
  }
}

static void suite_transport_failure() {
  fprintf(stderr, "== RdmaTransport (shim): peer failure => PEER_DOWN => Reliable DEAD\n");
  {   // QP killed in the fabric (crash): both sides are told
    RdmaPair p(small_cfg());
    ReliableConfig c = fast_cfg(); c.max_retries = 4;
    Reliable ra(*p.t[0], c), rb(*p.t[1], c);
    auto m = msg_of(1, 10); Reliable::Delivery d;
    CHECK(ra.send(1, m.data(), m.size()) == Reliable::OK, "first send");
    CHECK(rb.poll(d, 500), "delivered before the crash"); ra.poll(d, 50);
    p.fab.kill_qp(p.t[1]->peer_qpn(0));
    auto t0 = Clock::now(); int dead = -1, dead_b = -1;
    while ((dead < 0 || dead_b < 0) && ms_since(t0) < 3000) { ra.poll(d, 2); rb.poll(d, 2); if (dead < 0) dead = ra.dead_peer(); if (dead_b < 0) dead_b = rb.dead_peer(); }
    CHECK(dead == 1 && dead_b == 0, "PEER_DOWN must make Reliable declare DEAD on both sides (%d, %d)", dead, dead_b);
    CHECK(ra.send(1, m.data(), m.size()) == Reliable::DEAD, "send to a dead peer says DEAD");
    CHECK(!p.t[0]->send(1, m.data(), m.size()), "transport send to a down peer is false");
    CHECK(ms_since(t0) < 1000, "detection took %d ms", ms_since(t0));
  }
  {   // orderly close: messages sent before the close are delivered, THEN PEER_DOWN
    RdmaPair p(small_cfg());
    for (int i = 0; i < 5; ++i) { auto m = msg_of(i, 20); p.t[0]->send(1, m.data(), m.size()); }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    p.t[0]->close();
    Event ev; int recv = 0; bool down = false, order = true;
    while (p.t[1]->poll(ev, 1000)) { if (ev.type == EvType::RECV) { order &= !down; ++recv; } else { down = true; CHECK(ev.peer == 0, "peer id"); break; } }
    CHECK(recv == 5 && down && order, "5 messages then PEER_DOWN (got %d, down %d)", recv, (int)down);
    Event e2; CHECK(!p.t[1]->poll(e2, 50), "PEER_DOWN is reported once");
    auto m = msg_of(1, 4); CHECK(!p.t[1]->send(0, m.data(), m.size()), "send to the closed peer fails");
  }
  {   // a peer that is silent but not dead at the verbs level: Reliable's own retransmit limit still ends the wait
    RdmaPair p(small_cfg()); FaultSpec f; f.drop = 1.0; FaultyTransport silent(*p.t[1], f);
    ReliableConfig c = fast_cfg(); c.max_retries = 5; c.rto_max_ms = 40;
    Reliable ra(*p.t[0], c), rb(silent, c);
    auto m = msg_of(2, 10); ra.send(1, m.data(), m.size());
    auto t0 = Clock::now(); int dead = -1; Reliable::Delivery d;
    while (dead < 0 && ms_since(t0) < 5000) { ra.poll(d, 5); rb.poll(d, 0); dead = ra.dead_peer(); }
    CHECK(dead == 1, "silent peer not declared dead");
  }
  {   // proxy end to end over RDMA: peer dies -> the device sink is told
    RdmaPair p(small_cfg());
    std::vector<float> mem(64, 0); std::vector<std::atomic<unsigned>> inbox(4); for (auto& x : inbox) x = 0;
    std::vector<uint8_t> qmem(kvm_sq_bytes(8, 1024) + 64); unsigned long long tail = 0;
    SendQ::init(qmem.data(), 8, 1024, 5ull * 1000000000ull, &tail);
    ReliableConfig c = fast_cfg(); c.max_retries = 3;
    Reliable ra(*p.t[0], c);
    HostSink sink({{mem.data(), 8, 8}}, inbox.data(), 4);
    Proxy px(0, 2, SendQ(qmem.data()), ra, sink);
    px.start();
    p.t[1]->close();                                                   // rank 1 vanishes
    KvmSqHdr* h = (KvmSqHdr*)qmem.data();
    unsigned long long t = __atomic_fetch_add(h->tail_dev, 1, __ATOMIC_RELAXED);
    uint8_t* slot = qmem.data() + sizeof(KvmSqHdr) + (t % h->slots) * kvm_sq_stride(h->slot_bytes);
    float d8[8] = {1, 2, 3, 4, 5, 6, 7, 8}; memcpy(slot + KVM_SQ_SLOT_HDR, d8, 32);
    KvmSqSlot* s = (KvmSqSlot*)slot; s->dst_rank = 1; s->dst_buf = 0; s->dst_off = 0; s->dst_ld = 8; s->rows = 1; s->cols = 8; s->var = 0; s->ver = 1; s->nbytes = 32;
    __atomic_store_n(&s->seq, t + 1, __ATOMIC_RELEASE);
    auto t0 = Clock::now();
    while (sink.fail_code == 0 && ms_since(t0) < 5000) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(sink.fail_code == KVM_ERR_DIST_PEER_DOWN && sink.fail_arg == 1, "device not told about the dead peer (code %d)", sink.fail_code.load());
    px.kernel_done(); CHECK(px.wait_finished(5000), "proxy must finish with a dead peer"); px.stop();
  }
}

// ================================================================================================== RdmaVersionInbox
struct InboxRig {
  ShimFabric fab; InProcRendezvous rdz;
  std::vector<std::unique_ptr<ShimVerbs>> v; std::vector<std::vector<uint32_t>> inbox; std::vector<std::vector<uint8_t>> land;
  std::vector<std::unique_ptr<RdmaVersionInbox>> ib; bool ok = true;
  InboxRig(int nvars, size_t land_bytes, int nland, RdmaInboxConfig c = {}) : inbox(2, std::vector<uint32_t>(nvars, 0)), land(2 * nland, std::vector<uint8_t>(land_bytes, 0)) {
    for (int r = 0; r < 2; ++r) v.push_back(std::make_unique<ShimVerbs>(fab, r ? "i1" : "i0"));
    for (int r = 0; r < 2; ++r) {
      std::vector<RdmaVersionInbox::Landing> l; for (int k = 0; k < nland; ++k) l.push_back({land[r * nland + k].data(), land_bytes});
      ib.push_back(std::make_unique<RdmaVersionInbox>(*v[r], r, 2, inbox[r].data(), nvars, l, rdz, c));
    }
    std::vector<std::thread> th; std::atomic<int> bad{0};
    for (int r = 0; r < 2; ++r) th.emplace_back([&, r] { if (!ib[r]->connect()) ++bad; });
    for (auto& x : th) x.join();
    ok = bad == 0;
  }
  ~InboxRig() { for (auto& x : ib) x->close(); }
};

static void suite_inbox_basic() {
  fprintf(stderr, "== RdmaVersionInbox (shim): put_signal, publish, monotone max\n");
  InboxRig r(8, 4096, 2);
  CHECK(r.ok, "connect");
  auto data = msg_of(5, 512);
  auto rl = r.ib[0]->remote_landing(1, 1);
  CHECK(rl.len == 4096 && rl.rkey != 0, "remote landing described");
  CHECK(r.ib[0]->put_signal(1, 1, 100, data.data(), data.size(), 3, 1), "put_signal by landing index");
  CHECK(r.ib[1]->wait_ge(3, 1, 2000), "signal arrives");
  CHECK(memcmp(r.land[2 + 1].data() + 100, data.data(), data.size()) == 0, "the data was in place when the signal was observed");
  CHECK(r.ib[1]->load(3) == 1 && r.ib[1]->load(2) == 0, "only var 3 moved");
  CHECK(r.ib[0]->publish_remote(1, 3, 7), "signal only"); CHECK(r.ib[1]->wait_ge(3, 7, 2000), "v7");
  CHECK(r.ib[0]->publish_remote(1, 3, 5), "stale signal posted"); std::this_thread::sleep_for(std::chrono::milliseconds(30));
  CHECK(r.ib[1]->load(3) == 7, "inbox is max(): a stale version never regresses (got %u)", r.ib[1]->load(3));
  CHECK(r.ib[0]->publish_remote_imm(1, 4, 9) && r.ib[1]->wait_ge(4, 9, 2000), "imm-only publish");
  CHECK(!r.ib[0]->publish_remote_imm(1, 4, 70000) && !r.ib[0]->publish_remote_imm(1, 40000, 1), "imm-only range check");
  CHECK(!r.ib[0]->put_signal(1, 0, 4000, data.data(), data.size(), 1, 1), "write past the landing buffer is refused locally");
  CHECK(!r.ib[0]->put_signal(1, 0, 0, data.data(), data.size(), 99, 1), "var out of range");
  CHECK(!r.ib[1]->wait_ge(5, 1, 20), "wait_ge times out when nothing is published");
  // reverse direction
  CHECK(r.ib[1]->put_signal(0, 0, 0, data.data(), data.size(), 2, 4) && r.ib[0]->wait_ge(2, 4, 2000) && memcmp(r.land[0].data(), data.data(), data.size()) == 0, "rank 1 -> rank 0");
  // bounce exhaustion: 64 bounce buffers, 2000 puts => completion-driven reuse
  bool all = true; for (uint32_t v = 1; v <= 2000; ++v) all &= r.ib[0]->put_signal(1, 0, 0, data.data(), data.size(), 6, v);
  CHECK(all && r.ib[1]->wait_ge(6, 2000, 5000), "2000 back-to-back puts through 64 bounce buffers");
  // a wait wakes promptly
  std::thread th([&] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); r.ib[0]->publish_remote(1, 7, 1); });
  auto t0 = Clock::now(); CHECK(r.ib[1]->wait_ge(7, 1, 3000) && ms_since(t0) < 1000, "waiter wakes"); th.join();
}

// Each var has ONE writer thread on rank 0 (single-writer rule); tile `ver` of var v is written to landing[v] at offset ver*TILE.
// Readers on rank 1 poll the inbox; when they see version `ver` they must find the complete tile `ver` (and every earlier one).
static long inbox_stress(int reorder_us, int vars, int tiles, long* seen_out) {
  const size_t TILE = 128;
  InboxRig r(vars, (size_t)(tiles + 2) * TILE, vars);
  r.fab.set_reorder_writes_us(reorder_us);
  auto pat = [&](int v, int ver, size_t k) { return (uint8_t)(1 + (v * 31 + ver * 7 + k) % 251); };
  std::atomic<long> bad{0}, seen{0}; std::atomic<bool> stop{false};
  std::vector<std::thread> rd, wr;
  for (int v = 0; v < vars; ++v) rd.emplace_back([&, v] {
    uint32_t last = 0;
    while (!stop) {
      uint32_t ver = r.ib[1]->load(v);                                // ACQUIRE
      if (ver == last) { std::this_thread::yield(); continue; }
      for (uint32_t k = std::max(1u, last); k <= ver; k += std::max(1u, (ver - last) / 4 + 1)) {   // a few tiles <= ver, always including ver
        uint32_t t = (k == last) ? ver : k;
        for (size_t i = 0; i < TILE; ++i) if (__atomic_load_n(&r.land[vars + v][t * TILE + i], __ATOMIC_RELAXED) != pat(v, (int)t, i)) { ++bad; break; }
      }
      for (size_t i = 0; i < TILE; ++i) if (__atomic_load_n(&r.land[vars + v][ver * TILE + i], __ATOMIC_RELAXED) != pat(v, (int)ver, i)) { ++bad; break; }
      ++seen; last = ver;
      if (ver == (uint32_t)tiles) return;
    }
  });
  for (int v = 0; v < vars; ++v) wr.emplace_back([&, v] {
    std::vector<uint8_t> t(TILE);
    for (int ver = 1; ver <= tiles; ++ver) {
      for (size_t i = 0; i < TILE; ++i) t[i] = pat(v, ver, i);
      if (!r.ib[0]->put_signal(1, (size_t)v, (size_t)ver * TILE, t.data(), TILE, (uint32_t)v, (uint32_t)ver)) { ++bad; return; }
    }
  });
  for (auto& x : wr) x.join();
  for (int v = 0; v < vars; ++v) r.ib[1]->wait_ge(v, tiles, 5000);
  std::this_thread::sleep_for(std::chrono::milliseconds(reorder_us ? 20 : 0));
  stop = true; for (auto& x : rd) x.join();
  if (seen_out) *seen_out = seen;
  return bad;
}

static void suite_inbox_ordering() {
  fprintf(stderr, "== RdmaVersionInbox (shim): a reader that sees inbox[var] == ver always sees the whole tile\n");
  long seen = 0;
  long bad = inbox_stress(0, 4, 1500, &seen);
  CHECK(bad == 0, "%ld torn reads in the ordered shim", bad);
  fprintf(stderr, "  ordered shim: 4 vars x 1500 tiles, %ld observed versions, %ld torn reads\n", seen, bad);
  long bad_neg = inbox_stress(2500, 4, 150, &seen);
  CHECK(bad_neg > 0, "negative test: with the REORDER knob the same reader must observe torn tiles, saw %ld", bad_neg);
  fprintf(stderr, "  reorder knob (data lands 2.5 ms late): %ld torn reads -> the test detects a broken ordering\n", bad_neg);
}

static void suite_inbox_failure() {
  fprintf(stderr, "== RdmaVersionInbox (shim): peer down\n");
  InboxRig r(4, 1024, 1);
  std::atomic<int> cb{-1}; r.ib[0]->on_peer_down([&](int p) { cb = p; });
  auto d = msg_of(1, 64);
  CHECK(r.ib[0]->put_signal(1, 0, 0, d.data(), 64, 0, 1), "ok before");
  r.fab.kill_qp(r.ib[0]->peer_qpn(1));
  auto t0 = Clock::now(); while (cb < 0 && ms_since(t0) < 2000) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  CHECK(cb == 1 && r.ib[0]->peer_down(1), "peer-down callback");
  CHECK(!r.ib[0]->put_signal(1, 0, 0, d.data(), 64, 0, 2) && !r.ib[0]->publish_remote(1, 0, 3), "no puts to a down peer");
}

// ./test_rdma [substring]   runs only the suites whose name contains the substring (default: all)
int main(int argc, char** argv) {
  auto t0 = Clock::now();
  const char* only = argc > 1 ? argv[1] : nullptr;
  struct S { const char* name; void (*fn)(); };
  const S suites[] = {
    {"shim_mr", test_shim_mr_errors}, {"shim_qp", test_shim_qp_state}, {"shim_completions", test_shim_completions}, {"shim_rnr", test_shim_rnr},
    {"shim_flush", test_shim_flush_and_peer_down}, {"shim_ordering", test_shim_ordering}, {"transport_basic", suite_transport_basic},
    {"transport_credits", suite_transport_credits}, {"transport_reliable", suite_transport_reliable}, {"transport_failure", suite_transport_failure},
    {"inbox_basic", suite_inbox_basic}, {"inbox_ordering", suite_inbox_ordering}, {"inbox_failure", suite_inbox_failure}};
  for (auto& s : suites) if (!only || strstr(s.name, only)) s.fn();
  fprintf(stderr, "wall time %d ms\n", ms_since(t0));
  printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks.load(), g_fail.load());
  return g_fail ? 1 : 0;
}
