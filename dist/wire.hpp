#pragma once
// Layouts shared by the device (include/kvm_dist.cuh), the host proxy and the tests. Pure C++: no CUDA.
// Little-endian only (x86-64, aarch64 Linux): the static_asserts below pin sizes, and the wire format is not byte-swapped.
//
// SEND QUEUE (device -> host proxy), in mapped pinned host memory:
//   [KvmSqHdr 384 B][slot 0][slot 1]...   slot = KvmSqSlot (64 B) + payload capacity `slot_bytes`.
//   Multi-producer (any worker block), single consumer (the proxy). Ticket t = fetch_add(*tail_dev); slot t % slots.
//   Producer: wait until t - head < slots; write payload and slot fields; fence.sc.sys; store.release.sys seq = t + 1.
//   Consumer: for head = 0,1,2,...: wait acquire seq(head % slots) == head + 1; read; store.release head = head + 1.
//   Strict ticket order, so a producer that took a ticket always completes it (no gaps), and head-of-line blocking is bounded
//   by one block's copy time.
#include <cstdint>

#define KVM_SQ_MAGIC 0x4b565351u  // "KVSQ"
#define KVM_SQ_SLOT_HDR 64

struct alignas(128) KvmSqHdr {
  unsigned int magic, slots, slot_bytes, version;   // slot_bytes: payload capacity, multiple of 64
  unsigned long long timeout_ns;                    // a producer that cannot get a slot for this long fails the device
  unsigned long long* tail_dev;                     // ticket counter in DEVICE memory: atomics on pinned host memory need PCIe
                                                    // atomics (cudaDevAttrHostNativeAtomicSupported is 0 on x86 hosts); the host
                                                    // reads it only after the kernel ended, or from a host-emulated device in tests
  alignas(128) unsigned long long head;             // host-written (release), device-read (acquire.sys)
};
static_assert(sizeof(KvmSqHdr) == 256, "KvmSqHdr size");

struct KvmSqSlot {
  unsigned long long seq;
  int32_t dst_rank, dst_buf, dst_off, dst_ld, rows, cols, var, ver, nbytes, pad[5];
};
static_assert(sizeof(KvmSqSlot) == KVM_SQ_SLOT_HDR, "KvmSqSlot size");

inline unsigned long long kvm_sq_stride(uint32_t slot_bytes) { return KVM_SQ_SLOT_HDR + (unsigned long long)slot_bytes; }
inline unsigned long long kvm_sq_bytes(uint32_t slots, uint32_t slot_bytes) { return sizeof(KvmSqHdr) + (unsigned long long)slots * kvm_sq_stride(slot_bytes); }

// Message body carried by the reliable layer: header then `nbytes` of payload (rows x cols floats, packed).
struct KvmMsgHdr { int32_t dst_buf, dst_off, dst_ld, rows, cols, var, ver, nbytes; };
static_assert(sizeof(KvmMsgHdr) == 32, "KvmMsgHdr size");

// Device error codes owned by the distributed layer (the executor's own codes are 0..17; these never collide).
#define KVM_DIST_ERRORS(E) \
  E(DIST_PEER_DOWN, 64, "peer unreachable or silent past its deadline (a = peer rank)") \
  E(DIST_SENDQ_TIMEOUT, 65, "send queue full past its timeout (proxy stalled or remote too slow)") \
  E(DIST_BAD_QUEUE, 66, "q buffer is not an initialised send queue") \
  E(DIST_MSG_TOO_BIG, 67, "tile payload larger than the send queue's slot capacity") \
  E(DIST_BAD_SIGNAL, 68, "signal variable outside the target mailbox") \
  E(DIST_BAD_MSG, 69, "proxy rejected a message (destination buffer/bounds/rank)") \
  E(DIST_PROXY_DEAD, 70, "host proxy stopped heartbeating (a = ms since the last beat)")
enum KvmDistErr {
#define E(n, c, s) KVM_ERR_##n = c,
  KVM_DIST_ERRORS(E)
#undef E
};
