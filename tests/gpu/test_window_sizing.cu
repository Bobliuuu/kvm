// Host-only unit test of KvmConfig::size_for_one_record, the window sizing kvm_exec's CLI and demo/kvm_run share. It touches no GPU.
//   nvcc -arch=compute_80 -code=compute_80 -std=c++17 -Iinclude -DKVM_LEAN tests/gpu/test_window_sizing.cu -o test_window_sizing && ./test_window_sizing
// (compute_80 only: the device code is kept as PTX, so the build runs no ptxas.)
#define KVM_EXEC_NO_MAIN
#include "../../kvm_exec.cu"
#include <cstdio>

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// kvm_exec's CLI rule before it moved into KvmConfig, verbatim but for the log line: the method must give the same capacities.
static bool old_cli_rule(KvmConfig& cfg, long long nt, long long ne, long long nw) {
  const long long words = (KVM_REC_HDR + nt * KVM_TASK_WORDS + ne + nw + 7) & ~7ll;
  auto p2 = [](long long v) { long long r = 2; while (r < v) r <<= 1; return r; };
  if (ne >= 0 && nw >= 0 && (!cfg.fits_one_record(nt, words) || ne + 2 > (long long)cfg.succ_cap) && nt <= (1 << 20) && words < (1ll << 26)) {
    cfg.live_cap = (int)std::max<long long>(cfg.live_cap, p2(nt)); cfg.max_tasks = std::max(cfg.max_tasks, cfg.live_cap);
    cfg.ring_words = (int)std::max<long long>(cfg.ring_words, p2(2 * words + 128));
    cfg.succ_cap = std::max<unsigned>(cfg.succ_cap, (unsigned)p2(ne + 2));
    return true;
  }
  return false;
}

static bool same(const KvmConfig& a, const KvmConfig& b) {
  return a.live_cap == b.live_cap && a.max_tasks == b.max_tasks && a.ring_words == b.ring_words && a.succ_cap == b.succ_cap;
}

int main() {
  // 1. The CLI's sizing is unchanged: every bundle shape of a grid, from the defaults.
  const long long ts[] = {1, 2, 100, 4095, 4096, 16384, 16385, 65536, 100000, 280000, 830000, 1 << 20, (1 << 20) + 1, 3000000};
  const long long es[] = {0, 1, 1000, 1 << 21, (1 << 21) - 1, 5000000, 40000000};
  const long long ws[] = {0, 1, 12345, 3000000, 30000000};
  for (long long t : ts) for (long long e : es) for (long long w : ws) {
    KvmConfig a, b;
    const bool ga = a.size_for_one_record(t, e, w), gb = old_cli_rule(b, t, e, w);
    CHECK(ga == gb && same(a, b), "tasks %lld edges %lld args %lld: method grew %d (live %d slots %d ring %d succ %u), old rule %d (live %d slots %d ring %d succ %u)",
          t, e, w, ga, a.live_cap, a.max_tasks, a.ring_words, a.succ_cap, gb, b.live_cap, b.max_tasks, b.ring_words, b.succ_cap);
    if (ga) CHECK(a.fits_one_record(t, KvmConfig::record_words(t, e, w)) && e + 2 <= (long long)a.succ_cap, "tasks %lld: sized but still cut", t);
  }
  // 2. A missing or negative count changes nothing.
  { KvmConfig a; const KvmConfig d; CHECK(!a.size_for_one_record(-1, 10, 10) && same(a, d), "task_count -1 changed the config"); }
  { KvmConfig a; const KvmConfig d; CHECK(!a.size_for_one_record(500000, -1, 10) && same(a, d), "edge_count -1 changed the config"); }
  // 3. demo/kvm_run's bound: live_cap x classes stays within 2^20 (the ready rings of the CLI's largest one-class record).
  for (long long classes = 1; classes <= 6; ++classes) {
    long long max_live = 1 << 20;
    while (max_live * classes > (1 << 20)) max_live >>= 1;
    CHECK(max_live * classes <= (1 << 20) && max_live * 2 * classes > (1 << 20), "classes %lld: max_live %lld is not the largest power of two", classes, max_live);
    for (long long t : ts) {
      KvmConfig a; const KvmConfig d;
      const bool g = a.size_for_one_record(t, 4 * t, 20 * t, max_live);
      CHECK((long long)a.live_cap * classes <= (1 << 20) || same(a, d), "classes %lld tasks %lld: live_cap %d over the ring budget", classes, t, a.live_cap);
      if (t > max_live) CHECK(!g && same(a, d), "classes %lld tasks %lld above max_live %lld: the config changed", classes, t, max_live);
    }
  }
  printf("%d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
