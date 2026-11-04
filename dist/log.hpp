#pragma once
// Leveled logging for the distributed runtime: 0 error, 1 warn, 2 info, 3 debug, 4 trace. Verbose by default (debug);
// KVM_DIST_LOG=<0..4> overrides. Per-message events are trace (off by default); lifecycle and faults are debug/info/warn.
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace kvmd {
inline int& log_level() {
  static int lvl = [] { const char* e = getenv("KVM_DIST_LOG"); return e ? atoi(e) : 3; }();
  return lvl;
}
inline void logf(int lvl, const char* who, const char* fmt, ...) {
  if (lvl > log_level()) return;
  static const char* nm[] = {"E", "W", "I", "D", "T"};
  static const auto t0 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  char buf[1024];
  va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
  fprintf(stderr, "[kvm_dist][%s][%9.3f][%s] %s\n", nm[lvl < 0 ? 0 : lvl > 4 ? 4 : lvl], ms, who, buf);
}
}  // namespace kvmd
#define KD_LOG(lvl, who, ...) ::kvmd::logf(lvl, who, __VA_ARGS__)
