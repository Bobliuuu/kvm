#pragma once
// RDMA transport: NOT IMPLEMENTED. This is the documented slot for ibverbs / GPUDirect RDMA; nothing here pretends to work.
//
// TODO (what a real implementation needs, in order):
//  1. Queue pairs: RC (reliable connected) QP per peer, or one UD QP + the ARQ in reliable.hpp. Connection setup exchanges
//     {lid/gid, qpn, psn, rkey, remote buffer addr} over the existing TcpTransport (it already authenticates the job).
//  2. send(): ibv_post_send of a registered bounce buffer (IBV_WR_SEND_WITH_IMM), signalled every k sends; poll(): ibv_poll_cq
//     mapped to Event::RECV / PEER_DOWN (IBV_WC_RETRY_EXC_ERR, RNR, QP error state).
//  3. GPUDirect RDMA: register device memory (ibv_reg_mr on a cudaMalloc'd range with nvidia-peermem loaded) so the NIC reads
//     the send-queue payload / writes the landing buffer directly. Then the host proxy shrinks to doorbells: the device
//     writes a work request into a GPU-mapped send queue and rings the NIC doorbell itself (DOCA GPUNetIO / "IBGDA"),
//     and the inbox signal becomes an RDMA write with immediate into the receiver's inbox array. This removes the staging copy
//     through pinned host memory that the TCP path needs. Ordering requirement: the data write must complete before the
//     inbox write (same QP, in-order) or use IBV_SEND_FENCE.
//  4. Memory ordering on the device side stays exactly kvm_dist.cuh's argument (fence.sc.sys + st.release.sys), because the NIC
//     observes system-scope-released stores.
//  5. Tests: tests/dist/test_transport.cpp already runs the Transport contract suite against any Transport; add RdmaTransport
//     to its factory list and the TCP fault-injection matrix applies unchanged.
// Needs hardware (an RDMA NIC pair) to test; the project owner decided not to spend on that yet.
#include <stdexcept>

#include "transport.hpp"

namespace kvmd {
class RdmaTransport : public Transport {
 public:
  RdmaTransport(int, int) { throw std::runtime_error("RdmaTransport is not implemented (see the TODO in transport_rdma.hpp)"); }
  int rank() const override { return -1; }
  int world() const override { return 0; }
  const char* name() const override { return "rdma(unimplemented)"; }
  bool send(int, const void*, size_t) override { return false; }
  bool poll(Event&, int) override { return false; }
};
}  // namespace kvmd
