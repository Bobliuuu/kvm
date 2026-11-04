#pragma once
// CUDA side of the distributed runtime (host code). Include AFTER kvm_exec.cu (it uses KvmExec's CK/KLOG helpers).
//
//   DistDevice  per-rank device resources: inbox (u32 per version variable, in DEVICE memory so peers can store into it over
//               NVLink/PCIe), the in_vars list, the control block and the send queue (both mapped pinned HOST memory), and the
//               KvmDistDev descriptor handed to the executor (KvmExec::set_dist).
//   CudaSink    applies messages received by the proxy to this device: cudaMemcpyAsync of the tile into the destination
//               buffer, THEN (same stream) a 4-byte copy of the version into the inbox. Stream order completes the data copy
//               before the inbox write begins, and the executor's pump reads the inbox with ld.acquire.sys.
//   enable_peer_access  cudaDeviceEnablePeerAccess both ways for every device pair that supports it.
#include <map>

#include "../include/kvm_dist.cuh"
#include "proxy.hpp"

namespace kvmd {

#define KD_CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  char m_[512]; snprintf(m_, sizeof m_, "%s:%d: %s -> %s", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); throw std::runtime_error(m_); } } while (0)

inline int device_count() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0; }

// Enables P2P between all pairs of the given devices that can reach each other. Returns the number of directed links enabled;
// pairs that cannot (e.g. PCIe across root complexes without P2P) are reported and left to the proxy path.
inline int enable_peer_access(const std::vector<int>& devices) {
  int enabled = 0;
  for (int a : devices) for (int b : devices) {
    if (a == b) continue;
    int can = 0;
    KD_CK(cudaDeviceCanAccessPeer(&can, a, b));
    if (!can) { KD_LOG(1, "cuda", "device %d cannot access device %d peer-to-peer", a, b); continue; }
    KD_CK(cudaSetDevice(a));
    cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); ++enabled; continue; }
    KD_CK(e);
    ++enabled;
    KD_LOG(2, "cuda", "peer access %d -> %d enabled", a, b);
  }
  return enabled;
}

class DistDevice {
 public:
  DistDevice(int rank, int world, int device, int num_vars, uint32_t sq_slots = 64, uint32_t sq_slot_bytes = 256 * 1024,
             unsigned long long sq_timeout_ns = 10ull * 1000000000ull, unsigned long long dead_ns = 0)
      : rank_(rank), world_(world), device_(device), num_vars_(num_vars) {
    KD_CK(cudaSetDevice(device));
    KD_CK(cudaMalloc(&inbox_, (size_t)num_vars * 4)); KD_CK(cudaMemset(inbox_, 0, (size_t)num_vars * 4));
    KD_CK(cudaMalloc(&in_vars_, (size_t)std::max(num_vars, 1) * 4));
    KD_CK(cudaHostAlloc((void**)&ctl_, sizeof(KvmDistCtl), cudaHostAllocMapped));
    memset((void*)ctl_, 0, sizeof(KvmDistCtl));
    ctl_->dead_ns = dead_ns;
    sq_bytes_ = kvm_sq_bytes(sq_slots, sq_slot_bytes);
    KD_CK(cudaHostAlloc(&sq_, sq_bytes_, cudaHostAllocMapped));
    KD_CK(cudaMalloc(&tail_, 8)); KD_CK(cudaMemset(tail_, 0, 8));
    SendQ::init(sq_, sq_slots, sq_slot_bytes, sq_timeout_ns, tail_);
    void* d;
    KD_CK(cudaHostGetDevicePointer(&d, sq_, 0)); sq_dev_ = (float*)d;
    KD_CK(cudaHostGetDevicePointer(&d, (void*)ctl_, 0)); KvmDistCtl* ctl_dev = (KvmDistCtl*)d;
    KD_CK(cudaMalloc(&dev_, sizeof(KvmDistDev)));
    host_ = KvmDistDev{rank, world, 0, 0, in_vars_, inbox_, ctl_dev};
    KD_CK(cudaMemcpy(dev_, &host_, sizeof host_, cudaMemcpyHostToDevice));
  }
  ~DistDevice() {
    cudaSetDevice(device_);
    cudaFree(inbox_); cudaFree(in_vars_); cudaFree(dev_); cudaFree(tail_); cudaFreeHost((void*)ctl_); cudaFreeHost(sq_);
  }
  DistDevice(const DistDevice&) = delete; DistDevice& operator=(const DistDevice&) = delete;

  void set_in_vars(const std::vector<int>& v) {
    KD_CK(cudaSetDevice(device_));
    host_.n_in = (int)v.size();
    if (!v.empty()) KD_CK(cudaMemcpy(in_vars_, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    KD_CK(cudaMemcpy(dev_, &host_, sizeof host_, cudaMemcpyHostToDevice));
  }
  const KvmDistDev* dev() const { return dev_; }
  float* inbox_buf() const { return (float*)inbox_; }       // as a KvmBuf: rows 1, cols num_vars
  int num_vars() const { return num_vars_; }
  float* sendq_buf() const { return sq_dev_; }             // as a KvmBuf: rows 1, cols sq_bytes/4
  size_t sendq_floats() const { return sq_bytes_ / 4; }
  void* sendq_host() const { return sq_; }
  // The proxy's SendQ reads the device-resident ticket counter through this (only after the kernel ended; cudaMemcpy is synchronous).
  SendQ make_sendq() const {
    SendQ q(sq_);
    q.set_tickets_reader([this]() { unsigned long long t = 0; cudaSetDevice(device_); cudaMemcpy(&t, tail_, 8, cudaMemcpyDeviceToHost); return (uint64_t)t; });
    return q;
  }
  KvmDistCtl* ctl() const { return ctl_; }
  int device() const { return device_; }
  std::vector<unsigned> read_inbox() const {
    cudaSetDevice(device_);
    std::vector<unsigned> v(num_vars_); KD_CK(cudaMemcpy(v.data(), inbox_, v.size() * 4, cudaMemcpyDeviceToHost)); return v;
  }

 private:
  int rank_, world_, device_, num_vars_;
  unsigned* inbox_ = nullptr; int* in_vars_ = nullptr;
  KvmDistCtl* ctl_ = nullptr;
  void* sq_ = nullptr; size_t sq_bytes_ = 0; float* sq_dev_ = nullptr; unsigned long long* tail_ = nullptr;
  KvmDistDev host_{}, *dev_ = nullptr;
};

struct DevBuf { float* p; int rows, cols; };

class CudaSink : public Sink {
 public:
  CudaSink(int device, std::vector<DevBuf> bufs, DistDevice& dd) : device_(device), bufs_(std::move(bufs)), dd_(dd), last_(dd.num_vars(), 0) {
    KD_CK(cudaSetDevice(device));
    KD_CK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    KD_CK(cudaHostAlloc((void**)&vals_, sizeof(unsigned) * kSlab, cudaHostAllocDefault));
  }
  ~CudaSink() override { cudaSetDevice(device_); cudaStreamDestroy(st_); cudaFreeHost(vals_); }
  int deliver(int, const KvmMsgHdr& h, const uint8_t* payload) override {
    if (h.dst_buf < 0 || h.dst_buf >= (int)bufs_.size() || h.rows < 1 || h.cols < 1 || h.dst_ld < h.cols || h.dst_off < 0) return KVM_ERR_DIST_BAD_MSG;
    const DevBuf& b = bufs_[h.dst_buf];
    if (!b.p || (long long)h.dst_off + (long long)(h.rows - 1) * h.dst_ld + h.cols > (long long)b.rows * b.cols) return KVM_ERR_DIST_BAD_MSG;
    if (h.var >= dd_.num_vars()) return KVM_ERR_DIST_BAD_SIGNAL;
    KD_CK(cudaSetDevice(device_));
    float* dst = b.p + h.dst_off;
    if (h.dst_ld == h.cols) KD_CK(cudaMemcpyAsync(dst, payload, (size_t)h.rows * h.cols * 4, cudaMemcpyHostToDevice, st_));
    else KD_CK(cudaMemcpy2DAsync(dst, (size_t)h.dst_ld * 4, payload, (size_t)h.cols * 4, (size_t)h.cols * 4, h.rows, cudaMemcpyHostToDevice, st_));
    if (h.var >= 0) { unsigned& p = pend_[h.var]; p = std::max(p, (unsigned)h.ver); }
    return 0;
  }
  void flush() override {
    if (bufs_.empty() && pend_.empty()) return;
    KD_CK(cudaSetDevice(device_));
    size_t k = 0;
    for (auto& kv : pend_) {
      if (kv.second <= last_[kv.first]) continue;
      if (k == kSlab) { KD_CK(cudaStreamSynchronize(st_)); k = 0; }
      vals_[k] = last_[kv.first] = kv.second;
      KD_CK(cudaMemcpyAsync((unsigned*)dd_.inbox_buf() + kv.first, &vals_[k], 4, cudaMemcpyHostToDevice, st_));
      ++k;
    }
    pend_.clear();
    KD_CK(cudaStreamSynchronize(st_));      // data + versions are on the device; the staging slab may be reused
  }
  void fail(int code, int arg) override {
    KvmDistCtl* c = dd_.ctl();
    if (c->fail_code) return;
    c->fail_arg = arg;
    __atomic_store_n((int*)&c->fail_code, code, __ATOMIC_RELEASE);
  }
  void heartbeat() override { __atomic_add_fetch((unsigned*)&dd_.ctl()->heartbeat, 1u, __ATOMIC_RELEASE); }

 private:
  static constexpr size_t kSlab = 4096;
  int device_;
  std::vector<DevBuf> bufs_;
  DistDevice& dd_;
  cudaStream_t st_{};
  unsigned* vals_ = nullptr;
  std::map<int, unsigned> pend_;
  std::vector<unsigned> last_;
};

}  // namespace kvmd
