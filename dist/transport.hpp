#pragma once
// Pluggable inter-node transport. Pure C++17 + POSIX sockets: no CUDA, no MPI, no vendor library.
//
//   Transport        abstract message transport between ranks. send() takes a copy of the bytes (so the caller's buffer is free
//                    on return, like a bounce-buffered RDMA send); poll() yields received messages and peer-down notices.
//                    It is allowed to be UNRELIABLE (drop/duplicate/reorder): reliability, ordering, flow control and failure
//                    detection live one layer up (reliable.hpp), so a lossy fabric (UD verbs, UDP, a WAN VPN) can drop in.
//   LoopbackTransport  in-process queues between ranks (tests, and single-process multi-rank runs).
//   TcpTransport     one TCP connection per pair; framing = magic, length, CRC-32 of the payload. A frame that fails its
//                    check is DROPPED (counted), never delivered. Works across hosts and datacenters.
//   FaultyTransport  decorator injecting drop / duplicate / delay(+reorder) on send, deterministic from a seed.
//   RdmaTransport    see transport_rdma.hpp (stub: ibverbs / GPUDirect RDMA drop-in, NOT implemented).
//
// Threading: send() may be called from any thread; poll() from one thread at a time.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "log.hpp"

namespace kvmd {

using Clock = std::chrono::steady_clock;
inline int ms_since(Clock::time_point t) { return (int)std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t).count(); }

inline uint32_t crc32(const void* data, size_t n, uint32_t crc = 0) {
  static const std::vector<uint32_t> tab = [] {
    std::vector<uint32_t> t(256);
    for (uint32_t i = 0; i < 256; ++i) { uint32_t c = i; for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; t[i] = c; }
    return t;
  }();
  crc = ~crc;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < n; ++i) crc = tab[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
  return ~crc;
}

enum class EvType { RECV, PEER_DOWN };
struct Event { EvType type = EvType::RECV; int peer = -1; std::vector<uint8_t> data; };

class Transport {
 public:
  virtual ~Transport() = default;
  virtual int rank() const = 0;
  virtual int world() const = 0;
  virtual const char* name() const = 0;
  // Best effort. false: the peer is known to be unreachable (never "slow").
  virtual bool send(int dst, const void* p, size_t n) = 0;
  // Waits up to timeout_ms for one event. false: timeout.
  virtual bool poll(Event& ev, int timeout_ms) = 0;
  virtual void close() {}
};

// A blocking queue of events (used by Loopback and the TCP reader threads).
class EventQueue {
 public:
  void push(Event e) { { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(e)); } cv_.notify_one(); }
  bool pop(Event& e, int timeout_ms) {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms), [&] { return !q_.empty(); })) return false;
    e = std::move(q_.front()); q_.pop_front(); return true;
  }
  size_t size() { std::lock_guard<std::mutex> l(m_); return q_.size(); }
 private:
  std::mutex m_; std::condition_variable cv_; std::deque<Event> q_;
};

// ---------------------------------------------------------------------------------------------------------- loopback
class LoopbackHub {
 public:
  explicit LoopbackHub(int world) : q_(world), down_(world) { for (auto& d : down_) d = false; }
  int world() const { return (int)q_.size(); }
  EventQueue& queue(int r) { return q_[r]; }
  bool down(int r) const { return down_[r]; }
  void kill(int r) {                            // simulate a crashed rank: nobody can reach it, everyone else is told
    down_[r] = true;
    for (int p = 0; p < world(); ++p) if (p != r) { Event e; e.type = EvType::PEER_DOWN; e.peer = r; q_[p].push(std::move(e)); }
  }
 private:
  std::vector<EventQueue> q_;
  std::vector<std::atomic<bool>> down_;
};

class LoopbackTransport : public Transport {
 public:
  LoopbackTransport(LoopbackHub& hub, int rank) : hub_(hub), rank_(rank) {}
  int rank() const override { return rank_; }
  int world() const override { return hub_.world(); }
  const char* name() const override { return "loopback"; }
  bool send(int dst, const void* p, size_t n) override {
    if (dst < 0 || dst >= world() || hub_.down(dst) || hub_.down(rank_)) return false;
    Event e; e.peer = rank_; e.data.assign((const uint8_t*)p, (const uint8_t*)p + n);
    hub_.queue(dst).push(std::move(e));
    return true;
  }
  bool poll(Event& ev, int timeout_ms) override { return hub_.queue(rank_).pop(ev, timeout_ms); }
 private:
  LoopbackHub& hub_;
  int rank_;
};

// ---------------------------------------------------------------------------------------------------------- fault injection
struct FaultSpec {
  double drop = 0, dup = 0, delay = 0;   // probabilities per send
  int delay_ms_max = 20;                  // a delayed message is held 1..delay_ms_max ms (later sends overtake it: reordering)
  int latency_ms = 0;                     // EVERY message is held this long (order preserved): a long-fat-pipe / WAN emulation
  uint64_t seed = 1;
};

class FaultyTransport : public Transport {
 public:
  FaultyTransport(Transport& inner, FaultSpec f) : in_(inner), f_(f), rng_(f.seed ^ (0x9e3779b97f4a7c15ull * (inner.rank() + 1))) {}
  int rank() const override { return in_.rank(); }
  int world() const override { return in_.world(); }
  const char* name() const override { return "faulty"; }
  bool send(int dst, const void* p, size_t n) override {
    std::lock_guard<std::mutex> l(m_);
    std::uniform_real_distribution<double> u(0, 1);
    if (u(rng_) < f_.drop) { ++dropped; return true; }
    int copies = u(rng_) < f_.dup ? 2 : 1;
    if (copies == 2) ++duplicated;
    for (int c = 0; c < copies; ++c) {
      int hold_ms = 0;                                   // 0 = send now
      if (u(rng_) < f_.delay) { ++delayed; hold_ms = 1 + (int)(rng_() % (uint64_t)std::max(1, f_.delay_ms_max)); }   // overtaken: reordering
      else hold_ms = f_.latency_ms;
      if (hold_ms > 0) held_.push_back({Clock::now() + std::chrono::milliseconds(hold_ms), dst, std::vector<uint8_t>((const uint8_t*)p, (const uint8_t*)p + n)});
      else if (!in_.send(dst, p, n)) return false;
    }
    return true;
  }
  bool poll(Event& ev, int timeout_ms) override {
    auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
      flush();
      int slice = held_empty() ? timeout_ms : 1;
      auto left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
      if (in_.poll(ev, std::max(0, std::min(slice, left)))) return true;
      if (Clock::now() >= end) return false;
    }
  }
  void close() override { in_.close(); }
  std::atomic<long> dropped{0}, duplicated{0}, delayed{0};
 private:
  struct Held { Clock::time_point due; int dst; std::vector<uint8_t> data; };
  bool held_empty() { std::lock_guard<std::mutex> l(m_); return held_.empty(); }
  void flush() {
    std::lock_guard<std::mutex> l(m_);
    for (size_t i = 0; i < held_.size();)
      if (held_[i].due <= Clock::now()) { in_.send(held_[i].dst, held_[i].data.data(), held_[i].data.size()); held_.erase(held_.begin() + i); }
      else ++i;
  }
  Transport& in_;
  FaultSpec f_;
  std::mt19937_64 rng_;
  std::mutex m_;
  std::vector<Held> held_;
};

// ---------------------------------------------------------------------------------------------------------- TCP
struct TcpConfig {
  int rank = 0;
  std::vector<std::string> hosts;        // one entry per rank
  std::vector<int> ports;                // one per rank
  uint32_t cluster_id = 0;               // refuses connections from another job
  int connect_timeout_ms = 30000;
  int sndbuf = 0, rcvbuf = 0;            // 0 = OS default; set to the bandwidth-delay product on long fat links
  size_t max_frame = 256u << 20;
  unsigned corrupt_ppm = 0;              // TEST ONLY: flip one payload bit after the CRC was computed
};

class TcpTransport : public Transport {
  struct FrameHdr { uint32_t magic, len, crc; };
  struct Hello { uint32_t magic, rank, world, cluster; };
  static constexpr uint32_t kMagic = 0x4b4d5446, kHello = 0x4b4d4831;
  struct Conn {
    int fd = -1;
    std::mutex wm;
    std::atomic<bool> up{false}, down{false};
    std::thread reader;
  };

 public:
  explicit TcpTransport(TcpConfig c) : c_(std::move(c)), conn_(c_.hosts.size()) {
    signal(SIGPIPE, SIG_IGN);
    if (c_.hosts.size() != c_.ports.size() || c_.rank < 0 || c_.rank >= (int)c_.hosts.size()) throw std::invalid_argument("tcp: hosts/ports/rank mismatch");
    for (auto& cn : conn_) cn = std::make_unique<Conn>();
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons((uint16_t)c_.ports[c_.rank]);
    if (::bind(listen_fd_, (sockaddr*)&a, sizeof a) != 0 || ::listen(listen_fd_, 64) != 0)
      throw std::runtime_error("tcp: cannot listen on port " + std::to_string(c_.ports[c_.rank]) + ": " + strerror(errno));
    KD_LOG(2, "tcp", "rank %d listening on %d (world %zu)", c_.rank, c_.ports[c_.rank], c_.hosts.size());
    acceptor_ = std::thread([this] { accept_loop(); });
    for (int p = c_.rank + 1; p < (int)c_.hosts.size(); ++p) dialers_.emplace_back([this, p] { dial(p); });
  }
  ~TcpTransport() override { close(); }
  int rank() const override { return c_.rank; }
  int world() const override { return (int)c_.hosts.size(); }
  const char* name() const override { return "tcp"; }

  // Blocks until the connection to dst is up (connect_timeout), then writes one frame.
  bool send(int dst, const void* p, size_t n) override {
    if (dst < 0 || dst >= world() || dst == c_.rank || n > c_.max_frame) return false;
    Conn& cn = *conn_[dst];
    if (!wait_up(dst)) return false;
    FrameHdr h{kMagic, (uint32_t)n, crc32(p, n)};
    std::vector<uint8_t> tmp;
    const void* body = p;
    if (c_.corrupt_ppm && n && (uint32_t)(corrupt_rng_() % 1000000u) < c_.corrupt_ppm) {
      tmp.assign((const uint8_t*)p, (const uint8_t*)p + n); tmp[corrupt_rng_() % n] ^= 1u << (corrupt_rng_() % 8); body = tmp.data();
    }
    std::lock_guard<std::mutex> l(cn.wm);
    if (cn.down) return false;
    iovec iov[2] = {{&h, sizeof h}, {const_cast<void*>(body), n}};
    size_t total = sizeof h + n, done = 0;
    while (done < total) {
      iovec v[2]; int cnt = 0; size_t skip = done;
      for (auto& x : iov) { if (skip >= x.iov_len) { skip -= x.iov_len; continue; } v[cnt++] = {(char*)x.iov_base + skip, x.iov_len - skip}; skip = 0; }
      ssize_t w = ::writev(cn.fd, v, cnt);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) { mark_down(dst, "write failed"); return false; }
      done += (size_t)w;
    }
    bytes_sent += n;
    return true;
  }
  bool poll(Event& ev, int timeout_ms) override { return events_.pop(ev, timeout_ms); }
  void close() override {
    if (closed_.exchange(true)) return;
    if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); }
    for (auto& cn : conn_) if (cn->fd >= 0) ::shutdown(cn->fd, SHUT_RDWR);
    cv_.notify_all();
    if (acceptor_.joinable()) acceptor_.join();
    for (auto& t : dialers_) if (t.joinable()) t.join();
    for (auto& cn : conn_) { if (cn->reader.joinable()) cn->reader.join(); if (cn->fd >= 0) ::close(cn->fd); cn->fd = -1; }
  }
  std::atomic<long> crc_errors{0}, frames_in{0}, bytes_sent{0};

 private:
  static bool read_full(int fd, void* p, size_t n) {
    char* c = (char*)p;
    while (n) { ssize_t r = ::recv(fd, c, n, 0); if (r < 0 && errno == EINTR) continue; if (r <= 0) return false; c += r; n -= (size_t)r; }
    return true;
  }
  static bool write_full(int fd, const void* p, size_t n) {
    const char* c = (const char*)p;
    while (n) { ssize_t w = ::send(fd, c, n, 0); if (w < 0 && errno == EINTR) continue; if (w <= 0) return false; c += w; n -= (size_t)w; }
    return true;
  }
  void tune(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    if (c_.sndbuf) setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &c_.sndbuf, sizeof c_.sndbuf);
    if (c_.rcvbuf) setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &c_.rcvbuf, sizeof c_.rcvbuf);
  }
  void adopt(int peer, int fd) {
    Conn& cn = *conn_[peer];
    { std::lock_guard<std::mutex> l(mu_); if (cn.up) { ::close(fd); return; } cn.fd = fd; cn.up = true; }
    cn.reader = std::thread([this, peer] { read_loop(peer); });
    KD_LOG(2, "tcp", "rank %d <-> %d connected", c_.rank, peer);
    cv_.notify_all();
  }
  void accept_loop() {
    while (!closed_) {
      int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) { if (closed_) return; continue; }
      tune(fd);
      Hello h{};
      if (!read_full(fd, &h, sizeof h) || h.magic != kHello || h.world != (uint32_t)world() || h.cluster != c_.cluster_id || h.rank >= h.world || (int)h.rank >= c_.rank) {
        KD_LOG(1, "tcp", "rank %d rejected a connection (bad hello)", c_.rank); ::close(fd); continue;
      }
      Hello me{kHello, (uint32_t)c_.rank, (uint32_t)world(), c_.cluster_id};
      if (!write_full(fd, &me, sizeof me)) { ::close(fd); continue; }
      adopt((int)h.rank, fd);
    }
  }
  void dial(int peer) {
    auto t0 = Clock::now();
    int delay = 20;
    while (!closed_ && ms_since(t0) < c_.connect_timeout_ms) {
      addrinfo hints{}, *res = nullptr;
      hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
      if (getaddrinfo(c_.hosts[peer].c_str(), std::to_string(c_.ports[peer]).c_str(), &hints, &res) == 0 && res) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
          freeaddrinfo(res); tune(fd);
          Hello me{kHello, (uint32_t)c_.rank, (uint32_t)world(), c_.cluster_id}, h{};
          if (write_full(fd, &me, sizeof me) && read_full(fd, &h, sizeof h) && h.magic == kHello && (int)h.rank == peer && h.cluster == c_.cluster_id) { adopt(peer, fd); return; }
          ::close(fd);
        } else { ::close(fd); freeaddrinfo(res); }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      delay = std::min(delay * 2, 500);
    }
    if (!closed_) { KD_LOG(1, "tcp", "rank %d could not reach %d within %d ms", c_.rank, peer, c_.connect_timeout_ms); mark_down(peer, "connect timeout"); }
  }
  bool wait_up(int peer) {
    Conn& cn = *conn_[peer];
    std::unique_lock<std::mutex> l(mu_);
    cv_.wait_for(l, std::chrono::milliseconds(c_.connect_timeout_ms), [&] { return cn.up.load() || cn.down.load() || closed_.load(); });
    return cn.up && !cn.down;
  }
  void read_loop(int peer) {
    Conn& cn = *conn_[peer];
    std::vector<uint8_t> buf;
    for (;;) {
      FrameHdr h;
      if (!read_full(cn.fd, &h, sizeof h)) break;
      if (h.magic != kMagic || h.len > c_.max_frame) { KD_LOG(0, "tcp", "rank %d: bad frame header from %d; closing", c_.rank, peer); break; }
      buf.resize(h.len);
      if (h.len && !read_full(cn.fd, buf.data(), h.len)) break;
      if (crc32(buf.data(), h.len) != h.crc) { ++crc_errors; KD_LOG(1, "tcp", "rank %d: CRC mismatch on a %u-byte frame from %d: dropped", c_.rank, h.len, peer); continue; }
      ++frames_in;
      Event e; e.peer = peer; e.data = buf;
      events_.push(std::move(e));
    }
    if (!closed_) mark_down(peer, "connection lost");
  }
  void mark_down(int peer, const char* why) {
    Conn& cn = *conn_[peer];
    if (cn.down.exchange(true)) return;
    if (cn.fd >= 0) ::shutdown(cn.fd, SHUT_RDWR);
    KD_LOG(1, "tcp", "rank %d: peer %d down (%s)", c_.rank, peer, why);
    Event e; e.type = EvType::PEER_DOWN; e.peer = peer;
    events_.push(std::move(e));
    cv_.notify_all();
  }

  TcpConfig c_;
  std::vector<std::unique_ptr<Conn>> conn_;
  int listen_fd_ = -1;
  std::thread acceptor_;
  std::vector<std::thread> dialers_;
  EventQueue events_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> closed_{false};
  std::mt19937 corrupt_rng_{12345};
};

// Free TCP ports on this host (tests): bind to port 0, read the number, release.
inline std::vector<int> pick_free_ports(int n) {
  std::vector<int> ports; std::vector<int> fds;
  for (int i = 0; i < n; ++i) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    socklen_t len = sizeof a;
    if (::bind(fd, (sockaddr*)&a, sizeof a) != 0 || getsockname(fd, (sockaddr*)&a, &len) != 0) throw std::runtime_error("pick_free_ports");
    ports.push_back(ntohs(a.sin_port)); fds.push_back(fd);
  }
  for (int fd : fds) ::close(fd);
  return ports;
}

}  // namespace kvmd
