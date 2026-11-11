// Which GPT shapes of the hold-out draw fuse on an A100 (41,216 floats a member: 163 KB of opt-in shared memory less the static share; 108
// workers). The draw (referee/host/holdout.py): head dim 16, 32 or 64 times 2, 3, 4, 6 or 8 heads, d from 48 to 512 except 32, 64 and
// 768; 1-4 layers; T 32, 64, 96 or 128; B 1, 2 or 4; V 64, 128, 256 or 512; ffn 4 d. Every shape counts once.
// BEFORE: built against the tree before row chunks (kvm_fgpt_gang); else the tree after (kvm_fgpt_fit).
#include <cstdio>
#include <map>
#include "kvm_fused_gpt.h"
static int fits(int N, int T, int d, int H, int V) {
  const int smem = 41216, workers = 108;
#ifdef BEFORE
  return kvm_fgpt_gang(N, T, d, H, 4 * d, V, smem, workers) > 0;
#else
  return kvm_fgpt_fit(N, T, d, H, 4 * d, V, smem, workers).G > 0;
#endif
}
int main() {
  std::map<int, std::pair<int, int>> by_d;   // d -> (fused, all)
  for (int hd : {16, 32, 64}) for (int H : {2, 3, 4, 6, 8}) {
    const int d = hd * H;
    if (d < 48 || d > 512 || d == 32 || d == 64 || d == 768) continue;
    for (int L = 1; L <= 4; ++L) for (int T : {32, 64, 96, 128}) for (int B : {1, 2, 4}) for (int V : {64, 128, 256, 512}) {
      auto& c = by_d[d]; c.second++; c.first += fits(B * T, T, d, H, V);
    }
  }
  int f = 0, n = 0;
  for (auto& [d, c] : by_d) { f += c.first; n += c.second; }
#ifdef BEFORE
  const char* tag = "before row chunks";
#else
  const char* tag = "row chunks";
#endif
  printf("%s: %d of %d hold-out GPT shapes fuse on an A100 (%.0f%%)\n", tag, f, n, 100.0 * f / n);
  for (auto& [d, c] : by_d) printf("  d %3d: %3d of %3d (%.1f%%)\n", d, c.first, c.second, 100.0 * c.first / c.second);
#ifndef BEFORE
  const KvmFgptFit h = kvm_fgpt_fit(192, 96, 128, 4, 512, 256, 41216, 108);   // the hold-out's d128 draw: N 192 (B 2), T 96, 4 heads, V 256
  printf("the hold-out's d128 draw (N 192, T 96, 4 heads, V 256): G %d, %d query blocks, %d-row chunks, %lld floats a member of 41,216\n", h.G, h.qb,
         h.rows, kvm_fgpt_smem_floats(192, 96, 128, 4, 512, 256, h.G, h.qb, h.rows));
#endif
}
