// Host-only unit test of KvmConfig::size_for_one_record, the window sizing kvm_exec's CLI and demo/kvm_run share, and of the successor ring's floor
// (include/kvm_window_size.h) that KvmExec applies to both. It touches no GPU.
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
  // 4. The successor ring's floor (KvmExec raises succ_cap to kvm_succ_ring_words): for every admission ring the CLI accepts (a multiple of 8, 64 to
  //    2^28 words) and any starting successor ring, it is the smallest power of two at least the ring and the start, so it holds any one record's edges
  //    (submit() refuses a record above the ring) and the constructor's power-of-two check still passes, for a ring like --ring-words 3000 too.
  auto pow2 = [](unsigned long long v) { return v && !(v & (v - 1)); };
  for (long long ring = 64; ring <= (1ll << 28); ring = ring * 2 + 8 * (ring % 3 == 0)) {
    const int rw = (int)(ring & ~7ll);
    for (unsigned succ : {2u, 1u << 15, 1u << 21, 1u << 28}) {
      const unsigned f = kvm_succ_ring_words(succ, rw);
      CHECK(pow2(f) && f >= succ && f >= (unsigned)rw && (f == succ || f / 2 < (unsigned)rw),
            "ring %d start %u: floor %u is not the smallest power of two at least both", rw, succ, f);
      CHECK((long long)f >= rw - KVM_REC_HDR - KVM_TASK_WORDS, "ring %d: floor %u is under one record's edges", rw, f);
    }
  }
  for (int rw : {3000, 4194312, 33554432}) {   // explicit rings, two of them not powers of two
    const unsigned f = kvm_succ_ring_words(1u << 21, rw);
    CHECK(rw % 8 == 0 && pow2(f) && f >= (unsigned)rw, "--ring-words %d: floor %u", rw, f);
  }
  // 5. h640-d4-r16 (16 replicated runs of h640-d4): 307,200 tasks, 4,132,064 edges and 3,261,600 argument words as written. The dynamic path's noop splice
  //    gave 301,584 tasks and 13,200,000 edges (A100, main 4bbdaab's binary): fewer tasks, three times the edges.
  {
    const long long T = 307200, E = 4132064, A = 3261600, Ts = 301584, Es = 13200000;
    KvmConfig w; w.size_for_one_record(T, E, A);   // sized from the bundle as written, before this fix: the successor ring p2(E + 2) = 2^22 words
    CHECK(w.succ_cap == (1u << 22), "as written: successor ring %u", w.succ_cap);
    CHECK(!w.fits_one_record(Ts, KvmConfig::record_words(Ts, Es, A)) && (long long)w.ring_words / 2 > (long long)w.succ_cap,
          "as written: the spliced bundle should be cut into windows of up to %d words, more than the %u-word successor ring", w.ring_words / 2, w.succ_cap);
    KvmConfig a; const bool grew = a.size_for_one_record(Ts, Es, A);   // sized from the bundle as admitted
    const unsigned f = kvm_succ_ring_words(a.succ_cap, a.ring_words);
    CHECK(grew && a.fits_one_record(Ts, KvmConfig::record_words(Ts, Es, A)) && (long long)f >= Es + 2, "as admitted: not one record (ring %d, floor %u)", a.ring_words, f);
    KvmConfig b = a; CHECK(!b.size_for_one_record(Ts, Es, A) && same(a, b), "sizing the admitted bundle twice changed it");
    KvmConfig m; m.max_tasks = 524288; m.live_cap = 524288; m.ring_words = 33554432;   // run_mlp.py's flags: no sizing, the window as the flags set it
    const unsigned fm = kvm_succ_ring_words(m.succ_cap, m.ring_words);
    CHECK((long long)fm >= m.ring_words - KVM_REC_HDR - KVM_TASK_WORDS && (long long)fm > Es, "run_mlp.py's flags: floor %u", fm);
  }
  // 6. The litmus fuzzer's small window (--max-tasks 4096 --live-cap 256 --ring-words 4096) keeps its successor ring: the default already holds any record.
  CHECK(kvm_succ_ring_words(1u << 21, 4096) == (1u << 21), "the litmus window's successor ring moved");
  // 7. Random bundles sized without flags: when sizing makes one a record, the executor's successor ring holds that record's edges.
  unsigned long long x = 20261111;
  auto rnd = [&x](unsigned long long n) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (long long)(x % n); };
  for (int i = 0; i < 100000; ++i) {
    const long long t = 1 + rnd(1 << 20), e = rnd(1ll << 24), w = rnd(1ll << 23);
    KvmConfig a; a.size_for_one_record(t, e, w);
    const long long words = KvmConfig::record_words(t, e, w);
    if (a.fits_one_record(t, words)) CHECK((long long)kvm_succ_ring_words(a.succ_cap, a.ring_words) >= e + 2, "tasks %lld edges %lld: one record over its successor ring", t, e);
    else CHECK(words >= (1ll << 26) || t > (1 << 20) || words > a.ring_words / 2, "tasks %lld edges %lld args %lld: not one record, yet small enough", t, e, w);
  }
  printf("%d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
