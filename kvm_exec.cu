// KVM GPU executor, host side: bundle building/validation, the admission ring producer, launch, result decoding, and the
// CLI (compiled out with -DKVM_EXEC_NO_MAIN so tests/gpu can include this file).
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude kvm_exec.cu -o kvm_exec
//   KVM_HOST=... ./run_exec_tests.sh        # tests on a GPU box
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "kvm_exec.cuh"
#include "kvm_static.h"
#include "kvm_fuse.h"

// ---------------- logging (levels 0 error .. 3 debug; everything is shown by default) ----------------
static int g_kvm_log = 3;
static void kvm_log(int lvl, const char* fmt, ...) {
  if (lvl > g_kvm_log) return;
  static const char* nm[] = {"E", "W", "I", "D"};
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[kvm_exec][%s] ", nm[lvl]); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
  va_end(ap);
}
#define KLOG(lvl, ...) kvm_log(lvl, __VA_ARGS__)
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  char m_[512]; snprintf(m_, sizeof m_, "%s:%d: %s -> %s", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); throw std::runtime_error(m_); } } while (0)

static const char* kvm_dist_err(int c, bool text) {  // KVM_DIST_HOOK: codes 64.. come from dist/wire.hpp
  switch (c) {
#define E(n, code, str) case code: return text ? str : #n;
    KVM_DIST_ERRORS(E)
#undef E
  }
  return nullptr;
}
// The GPT pack's device errors (kvm_ops_gpt.cuh's one list), or nullptr when the code is not one of them.
static const char* kvm_gpt_err(int c, bool text) {
#ifdef KVM_GPT_ERR_BASE
  switch (c) {
#define KVM_GPT_ERR_CASE(n, off, str) case KVM_GPT_ERR_BASE + off: return text ? str : #n;
    KVM_GPT_ERRORS(KVM_GPT_ERR_CASE)
#undef KVM_GPT_ERR_CASE
  }
#else
  (void)c; (void)text;
#endif
  return nullptr;
}
static const char* kvm_err_name(int c) {
  if (const char* d = kvm_dist_err(c, false)) return d;
  if (const char* g = kvm_gpt_err(c, false)) return g;
  static const char* nm[] = {
#define E(n, s) #n,
      KVM_ERRORS(E)
#undef E
  };
  return c >= 0 && c < (int)(sizeof nm / sizeof nm[0]) ? nm[c] : "UNKNOWN";
}
static const char* kvm_err_text(int c) {
  if (const char* d = kvm_dist_err(c, true)) return d;
  if (const char* g = kvm_gpt_err(c, true)) return g;
  static const char* tx[] = {
#define E(n, s) s,
      KVM_ERRORS(E)
#undef E
  };
  return c >= 0 && c < (int)(sizeof tx / sizeof tx[0]) ? tx[c] : "unknown error code";
}
static int kvm_op_by_name(const std::string& n) {
  for (const KvmOpInfo& o : kKvmOps) if (n == o.name) return o.id;
  return -1;
}

// MFLOPs of one GEMM tile (tile extent clipped to the GEMM's edge); 0 for every other op. The work term of the static schedule compiler's cost.
static double kvm_tile_mflop(int op, const int32_t* a) {
  const KvmOpInfo* o = kvm_op(op);
  if (!o || strcmp(o->name, "matmul_tile")) return 0.0;
  const double m0 = (unsigned)a[6], n0 = (unsigned)a[7], M = (unsigned)a[8], N = (unsigned)a[9], K = (unsigned)a[10];
  return std::min(64.0, M - m0) * std::min(64.0, N - n0) * K * 2.0 / 1e6;
}

// Scheduler weights (microseconds, relative only) of the static schedule compiler: per-op means of the Nov 9 executor traces (gpt_tiny) with the
// dynamic path's fixed overhead taken out, plus the FLOPs of a GEMM tile at an SM's TF32 rate (kvm_static_cost). sgd_range and softmax_ce_fused sit
// at the flat 4 us an unlisted op used to get: the a100 preset's bundles carry them (the split SGD, the fused cross-entropy), and the path rule was
// read with them at that price. Any other op has no measured cost here, so the path rule decides only bundles whose every op is listed (main()).
static const std::map<std::string, double>& kvm_static_costs() {
  static const std::map<std::string, double> base = {{"noop", 0.1}, {"matmul_tile", 4.0}, {"adamw", 5.0}, {"gradnorm_partial", 2.0}, {"gradnorm_finish", 2.0}, {"colsum", 2.0},
    {"layernorm_fwd", 7.0}, {"layernorm_bwd_dx", 8.0}, {"norm_bwd_params", 5.0}, {"attn_fwd", 8.0}, {"attn_bwd_dq", 9.0}, {"attn_bwd_dkv", 10.0}, {"attn_bwd_delta", 6.0},
    {"rope_fwd", 4.0}, {"rope_bwd", 4.0}, {"embedding_fwd", 3.0}, {"embedding_bwd", 3.0}, {"softmax_ce_fwd", 3.0}, {"softmax_ce_bwd", 3.0}, {"softmax_ce_reduce", 2.0}, {"gelu_bwd", 3.0}, {"gelu_fwd", 3.0}, {"select_window", 2.0}, {"counter_inc", 1.0}, {"fmlp_slice", 60.0}, {"fmlp_sgd", 8.0}, {"fmlp_split", 40.0},
    {"sgd_range", 4.0}, {"softmax_ce_fused", 4.0}};
  return base;
}
static double kvm_static_cost(int op, const int32_t* a) {
  const std::map<std::string, double>& base = kvm_static_costs();
  const KvmOpInfo* o = kvm_op(op);
  auto it = o ? base.find(o->name) : base.end();
  double us = it != base.end() ? it->second : 4.0;
  return us + kvm_tile_mflop(op, a) / 1.4;
}

// A bundle whose ops the cost model all prices and whose mean estimated task cost (kvm_static_cost) is at most this runs on the static path by default; above it, on the dynamic path (main()). The crossover measured on
// frontend GPTs outside the panel (Nov 10, A100, fusion off, static against dynamic on one binary): mean 4.7-5.6 us static x1.26-1.28, 5.7 us even, 6.6-8.3 us static x0.62-0.71.
#ifndef KVM_STATIC_MAX_MEAN_US
#define KVM_STATIC_MAX_MEAN_US 6.0
#endif

// ---------------- bundles ----------------
struct KvmArg {
  int32_t w; int type;
  static KvmArg buf(int idx) { return {idx, KVM_A_BUF}; }
  static KvmArg none() { return {-1, KVM_A_BUF}; }   // absent optional buffer (a `buf?` argument)
  static KvmArg i32(int v) { return {v, KVM_A_I32}; }
  static KvmArg f32(float v) { int32_t w; memcpy(&w, &v, 4); return {w, KVM_A_F32}; }
};
struct KvmTaskOpts { int priority = 0, deadline_us = -1, wait_var = -1, wait_ver = 0, pub_var = -1, pub_ver = 0, workload = 0; };

struct KvmBundle {
  std::vector<KvmTaskDesc> tasks;
  std::vector<int32_t> succ, args;
  // "" if well formed, else the first problem. nbufs/nvars < 0 skip those range checks.
  std::string validate(int nbufs, int nvars, int live_cap) const {
    std::ostringstream e;
    const int nt = (int)tasks.size();
    if (nt == 0) return "empty bundle";
    if (live_cap > 0 && nt > live_cap) { e << nt << " tasks > live_cap " << live_cap; return e.str(); }
    std::vector<int> in(nt, 0);
    for (int i = 0; i < nt; ++i) {
      const KvmTaskDesc& t = tasks[i];
      const KvmOpInfo* op = kvm_op(t.op_id);
      if (!op) { e << "task " << i << ": bad op_id " << t.op_id; return e.str(); }
      if (t.args_words != op->arity) { e << "task " << i << ": " << op->name << " takes " << op->arity << " args, got " << t.args_words; return e.str(); }
      if (t.args_off < 0 || (long long)t.args_off + t.args_words > (long long)args.size()) { e << "task " << i << ": args range"; return e.str(); }
      if (t.succ_off < 0 || t.succ_cnt < 0 || (long long)t.succ_off + t.succ_cnt > (long long)succ.size()) { e << "task " << i << ": successor range"; return e.str(); }
      if (t.priority < 0 || t.priority >= KVM_NBUCKETS) { e << "task " << i << ": priority " << t.priority << " outside [0," << KVM_NBUCKETS - 1 << "]"; return e.str(); }
      if (t.deadline_us < -1) { e << "task " << i << ": deadline_us"; return e.str(); }
      for (int v : {t.wait_var, t.pub_var}) if (v < -1 || (nvars >= 0 && v >= nvars)) { e << "task " << i << ": version variable " << v << " out of range"; return e.str(); }
      if ((t.wait_var >= 0 && t.wait_ver < 0) || (t.pub_var >= 0 && t.pub_ver < 1)) { e << "task " << i << ": wait_ver >= 0 and pub_ver >= 1 (0 is the initial state)"; return e.str(); }
      for (int j = 0; j < t.args_words; ++j) {
        const int ty = (int)((op->types >> (3 * j)) & 7u);
        if (ty == KVM_A_BUF || ty == KVM_A_BUF_OPT) {
          int b = args[t.args_off + j];
          if ((b == -1 && ty == KVM_A_BUF) || b < -1 || (nbufs >= 0 && b >= nbufs)) { e << "task " << i << ": " << op->name << " arg " << j << " buffer index " << b << " out of range"; return e.str(); }
        }
      }
      for (int k = 0; k < t.succ_cnt; ++k) {
        int s = succ[t.succ_off + k];
        if (s < 0 || s >= nt || s == i) { e << "task " << i << ": successor " << s << " out of range"; return e.str(); }
        ++in[s];
      }
    }
    for (int i = 0; i < nt; ++i) if (in[i] != tasks[i].indegree) { e << "task " << i << ": indegree " << tasks[i].indegree << " != in-edges " << in[i]; return e.str(); }
    std::vector<int> q, rem = in;  // Kahn: any cycle would deadlock the device
    for (int i = 0; i < nt; ++i) if (!rem[i]) q.push_back(i);
    for (size_t h = 0; h < q.size(); ++h) {
      const KvmTaskDesc& t = tasks[q[h]];
      for (int k = 0; k < t.succ_cnt; ++k) if (--rem[succ[t.succ_off + k]] == 0) q.push_back(succ[t.succ_off + k]);
    }
    if ((int)q.size() != nt) return "dependency cycle";
    // Version waits decidable inside one bundle: a task that waits for version n of v needs every in-bundle publication of v up to n
    // first. Edges publisher -> waiter added to the dependency graph must stay acyclic, otherwise nothing can ever run.
    {
      std::map<int, std::vector<int>> by_var;   // per variable: tasks that publish it
      for (int i = 0; i < nt; ++i) if (tasks[i].pub_var >= 0) by_var[tasks[i].pub_var].push_back(i);
      std::vector<std::pair<int, int>> extra;   // publisher -> waiter edges (few); the successor lists are walked in place, not copied
      for (int i = 0; i < nt; ++i) if (tasks[i].wait_var >= 0) {
        auto it = by_var.find(tasks[i].wait_var); if (it == by_var.end()) continue;
        for (int p : it->second) if (p != i && tasks[p].pub_ver <= tasks[i].wait_ver) extra.push_back({p, i});
        for (int p : it->second) if (p == i && tasks[p].pub_ver <= tasks[i].wait_ver) { e << "task " << i << ": waits for version " << tasks[i].wait_ver << " of variable " << tasks[i].wait_var << " that it publishes itself"; return e.str(); }
      }
      int seen = 0;
      if (extra.empty()) seen = nt;             // the dependency graph was just shown acyclic and nothing is added to it
      else {
        std::sort(extra.begin(), extra.end());
        std::vector<int> deg(nt, 0), xs(nt + 1, 0);
        for (int i = 0; i < nt; ++i) for (int k = 0; k < tasks[i].succ_cnt; ++k) ++deg[succ[tasks[i].succ_off + k]];
        for (auto& pe : extra) { ++deg[pe.second]; ++xs[pe.first + 1]; }
        for (int i = 0; i < nt; ++i) xs[i + 1] += xs[i];
        std::vector<int> st; for (int i = 0; i < nt; ++i) if (!deg[i]) st.push_back(i);
        while (!st.empty()) {
          int u = st.back(); st.pop_back(); ++seen;
          for (int k = 0; k < tasks[u].succ_cnt; ++k) { int v = succ[tasks[u].succ_off + k]; if (--deg[v] == 0) st.push_back(v); }
          for (int x = xs[u]; x < xs[u + 1]; ++x) { int v = extra[x].second; if (--deg[v] == 0) st.push_back(v); }
        }
      }
      if (seen != nt) return "version-wait cycle: a task waits for a version that only a task depending on it can publish";
    }
    return "";
  }
  size_t words() const { return (KVM_REC_HDR + tasks.size() * KVM_TASK_WORDS + succ.size() + args.size() + 7) & ~(size_t)7; }
};

// ---------------- dissolve noops that only order tasks (on a longest path) ----------------
// A noop has no work; its effect is the ordering through it and, when it publishes, a version. When no task of the bundle waits on a version, that version has no reader, so a noop is spliced out: every
// predecessor points straight at every successor, and the noop costs no task, level or dependency hop. A noop whose fan-in times fan-out exceeds max_fan stays (a join: splicing it would explode the edge count).
// Returns the number dissolved. Runs on a plain single-bundle run, before the first task is admitted.
static int kvm_dissolve_noops(KvmBundle& b, long long max_fan = 65536) {
  const int nt = (int)b.tasks.size();
  for (const KvmTaskDesc& t : b.tasks) if (t.wait_var >= 0) return 0;
  // Only a noop on a longest path (counted in tasks) costs the run a hop; anywhere else splicing it buys nothing and multiplies edges (a GPT step's barrier, 20-45 in by
  // 21-81 out, lies off the path: spliced, gpt-tiny and gpt-small lost 0.5-0.9%; the MLP rows' barrier lies on it and gained 4-9%).
  std::vector<int> depth(nt, 0), height(nt, 0), pending(nt), order;
  order.reserve(nt);
  for (int i = 0; i < nt; ++i) { pending[i] = b.tasks[i].indegree; if (pending[i] == 0) order.push_back(i); }
  for (size_t k = 0; k < order.size(); ++k) {
    const KvmTaskDesc& t = b.tasks[order[k]];
    for (int e = 0; e < t.succ_cnt; ++e) { const int s = b.succ[t.succ_off + e]; depth[s] = std::max(depth[s], depth[order[k]] + 1); if (--pending[s] == 0) order.push_back(s); }
  }
  if ((int)order.size() != nt) return 0;                      // not a DAG by its indegrees: leave the bundle to validation
  int longest = 0;
  for (int k = nt - 1; k >= 0; --k) {
    const int i = order[k]; const KvmTaskDesc& t = b.tasks[i];
    for (int e = 0; e < t.succ_cnt; ++e) height[i] = std::max(height[i], height[b.succ[t.succ_off + e]] + 1);
    longest = std::max(longest, depth[i] + height[i]);
  }
  std::vector<char> gone(nt, 0); int ngone = 0;
  for (int i = 0; i < nt; ++i) {
    const KvmTaskDesc& t = b.tasks[i];
    if (t.op_id == KVM_OP_noop && depth[i] + height[i] == longest && (long long)t.indegree * t.succ_cnt <= max_fan) { gone[i] = 1; ++ngone; }
  }
  if (!ngone) return 0;
  std::vector<int> new_id(nt, -1); int m = 0;
  for (int i = 0; i < nt; ++i) if (!gone[i]) new_id[i] = m++;
  // A version variable whose publishers are only partly spliced out keeps the rest: they must still publish in their old order and count up by one. So each remaining
  // publisher gets the next remaining one of its variable as an explicit successor and a renumbered version. No task waits on a version here (checked above), so the
  // numbers carry no reader's meaning, only the executor's contiguity check.
  std::map<int, int> first_ver;                                // a variable's first published version as written (its base + 1)
  std::map<int, std::vector<int>> kept;                        // a variable's remaining publishers, old indices
  for (int i = 0; i < nt; ++i) {
    const KvmTaskDesc& t = b.tasks[i];
    if (t.pub_var < 0) continue;
    auto f = first_ver.find(t.pub_var);
    if (f == first_ver.end() || t.pub_ver < f->second) first_ver[t.pub_var] = t.pub_ver;
    if (!gone[i]) kept[t.pub_var].push_back(i);
  }
  std::vector<int> next_pub(nt, -1), new_ver(nt, 0);
  for (auto& kv : kept) {
    std::vector<int>& ix = kv.second;
    std::sort(ix.begin(), ix.end(), [&](int x, int y) { return b.tasks[x].pub_ver < b.tasks[y].pub_ver; });
    for (size_t r = 0; r < ix.size(); ++r) { new_ver[ix[r]] = first_ver[kv.first] + (int)r; if (r + 1 < ix.size()) next_pub[ix[r]] = ix[r + 1]; }
  }
  KvmBundle out; out.tasks.reserve(m);
  std::vector<int> indeg(m, 0), stamp(nt, -1), stack;
  for (int i = 0; i < nt; ++i) {
    if (gone[i]) continue;
    KvmTaskDesc t = b.tasks[i];
    std::set<int> succs;                                      // direct successors, and those reached through spliced noops (each noop visited once for this task)
    stack.clear();
    for (int k = 0; k < t.succ_cnt; ++k) stack.push_back(b.succ[t.succ_off + k]);
    while (!stack.empty()) {
      const int v = stack.back(); stack.pop_back();
      if (!gone[v]) { succs.insert(new_id[v]); continue; }
      if (stamp[v] == i) continue;
      stamp[v] = i;
      for (int k = 0; k < b.tasks[v].succ_cnt; ++k) {
        const int w = b.succ[b.tasks[v].succ_off + k];
        if (gone[w] && b.tasks[v].pub_var >= 0 && b.tasks[w].pub_var == b.tasks[v].pub_var) continue;   // the token chain between two publishers of one variable only orders the publications (nobody reads them): following it would add every later step's successors
        stack.push_back(w);
      }
    }
    if (next_pub[i] >= 0) succs.insert(new_id[next_pub[i]]);
    if (t.pub_var >= 0) t.pub_ver = new_ver[i];
    const int old_args = t.args_off;
    t.succ_off = (int)out.succ.size(); t.succ_cnt = (int)succs.size();
    for (int s : succs) { out.succ.push_back(s); ++indeg[s]; }
    t.args_off = (int)out.args.size();
    out.args.insert(out.args.end(), b.args.begin() + old_args, b.args.begin() + old_args + t.args_words);
    out.tasks.push_back(t);
  }
  for (int i = 0; i < m; ++i) out.tasks[i].indegree = indeg[i];
  b = std::move(out);
  return ngone;
}

// ---------------- the tape in dependency order ----------------
// The admitter reads a record in task order and a task can be released only once it has passed it, so a task's position in the tape sets the earliest moment it can start. Replicated
// runs concatenate R tapes, which puts run k's first task behind every task of runs 0..k-1 (R=32: run 31 became ready 18.7 ms into a 76 ms run, at 62 ns a task). Any
// topological order is a valid tape; this one is by dependency level, in the old order within a level (a GEMM's tiles stay consecutive).
static bool kvm_sort_tape(KvmBundle& b) {
  const int n = (int)b.tasks.size();
  std::vector<int> indeg(n), order, cur, next;
  order.reserve(n);
  for (int i = 0; i < n; ++i) { indeg[i] = b.tasks[i].indegree; if (indeg[i] == 0) cur.push_back(i); }
  while (!cur.empty()) {
    for (int i : cur) {
      order.push_back(i);
      for (int k = 0; k < b.tasks[i].succ_cnt; ++k) { const int s = b.succ[b.tasks[i].succ_off + k]; if (--indeg[s] == 0) next.push_back(s); }
    }
    std::sort(next.begin(), next.end());
    cur.swap(next); next.clear();
  }
  if ((int)order.size() != n) return false;                    // not a DAG by its indegrees: leave the tape alone (validation reports it)
  std::vector<int> pos(n);
  for (int k = 0; k < n; ++k) pos[order[k]] = k;
  std::vector<KvmTaskDesc> tasks(n);
  std::vector<int32_t> succ; succ.reserve(b.succ.size());
  for (int k = 0; k < n; ++k) {
    KvmTaskDesc t = b.tasks[order[k]];
    const int off = (int)succ.size();
    for (int j = 0; j < t.succ_cnt; ++j) succ.push_back(pos[b.succ[t.succ_off + j]]);
    t.succ_off = off;
    tasks[k] = t;
  }
  b.tasks.swap(tasks); b.succ.swap(succ);
  return true;
}

class KvmBundleBuilder {
 public:
  int add(const char* op, std::vector<KvmArg> args, KvmTaskOpts o = {}) {
    int id = kvm_op_by_name(op);
    if (id < 0) throw std::invalid_argument(std::string("unknown operator ") + op);
    const KvmOpInfo* info = kvm_op(id);
    if ((int)args.size() != info->arity) throw std::invalid_argument(std::string(op) + ": wrong argument count");
    for (size_t j = 0; j < args.size(); ++j)
    {
      const int want = (int)((info->types >> (3 * j)) & 7u);
      const bool buf_ok = args[j].type == KVM_A_BUF && (want == KVM_A_BUF || want == KVM_A_BUF_OPT) && (args[j].w >= 0 || want == KVM_A_BUF_OPT);
      const bool type_ok = args[j].type == KVM_A_BUF ? buf_ok : args[j].type == want;
      if (!type_ok) throw std::invalid_argument(std::string(op) + ": argument " + std::to_string(j) + " has the wrong type (or an absent buffer for a required buf)");
    }
    KvmTaskDesc t{};
    t.op_id = id; t.priority = o.priority; t.deadline_us = o.deadline_us; t.wait_var = o.wait_var; t.wait_ver = o.wait_ver;
    t.pub_var = o.pub_var; t.pub_ver = o.pub_ver; t.workload_id = o.workload; t.args_words = (int)args.size();
    t.args_off = (int)args_.size();
    for (auto& a : args) args_.push_back(a.w);
    tasks_.push_back(t); out_.emplace_back();
    return (int)tasks_.size() - 1;
  }
  void edge(int from, int to) {  // duplicate edges are removed here (the device counts every in-edge)
    if (from < 0 || to < 0 || from >= (int)tasks_.size() || to >= (int)tasks_.size() || from == to) throw std::invalid_argument("bad edge");
    out_[from].insert(to);
  }
  KvmBundle build() const {
    KvmBundle b; b.tasks = tasks_; b.args = args_;
    for (size_t i = 0; i < tasks_.size(); ++i) {
      b.tasks[i].succ_off = (int)b.succ.size();
      b.tasks[i].succ_cnt = (int)out_[i].size();
      for (int s : out_[i]) { b.succ.push_back(s); ++b.tasks[s].indegree; }
    }
    return b;
  }
 private:
  std::vector<KvmTaskDesc> tasks_; std::vector<int32_t> args_; std::vector<std::set<int>> out_;
};

// ---------------- executor ----------------
struct KvmConfig {
  int blocks = 0;               // 0: occupancy-sized (SMs * resident blocks); >= 2 (block 0 admits, the rest work)
  int max_tasks = 1 << 18;      // task slots, power of two; slots are recycled when a task completes
  int live_cap = 1 << 14;       // max admitted-but-unfinished tasks (and max tasks per bundle record); ready rings hold this many each
  unsigned succ_cap = 1u << 21; // successor ring (words), power of two; a window's words return when all its tasks completed
  int win_cap = 1 << 16;        // bundle records in flight, power of two
  int num_vars = 64;
  int ring_words = 1 << 16;     // admission ring, multiple of 8
  int aging_us = 0;             // 0: strict priority; else one priority level per aging_us of waiting
  // A bundle of `tasks` tasks and `words` record words (KvmBundle::words()) goes in as one record: else it is cut into windows that drain.
  bool fits_one_record(long long tasks, long long words) const { return tasks <= live_cap && tasks <= max_tasks && words <= ring_words / 2; }
#ifdef KVM_PROF_GAP
  int noop_ops = 0;             // profiling only: run the graph with every op body skipped (--noop-ops)
#endif
  int cont = 1;                 // 1: continuation (a worker runs one released successor directly, when the ready queue is empty); 0: every release goes through the queue
  int fifo_only = 0;            // 1: one FIFO ring with ticket waits (every task priority 0, no deadline, scheduler off); the CLI sets it when its bundles qualify
  int no_recycle = 0;           // 1: the whole run is one record and slots are not reused (the CLI, when a single bundle fits as one record)
  int no_waits = 0;             // 1: no task waits on a version
  int trace_on = 1;             // 1: per-task timeline and witness in trace[] (0: --trace-csv/--golden not asked for and the scheduler off: the CLI's timing runs)
  unsigned long long timeout_us = 60ull * 1000000;   // STALL timeout: no task finished or admitted for this long while work is pending (not a lifetime cap); must exceed the longest task
  unsigned long long task_limit_us = 30ull * 1000000;  // a task that ran longer than this ends the run with TASK_LIMIT (0 = off); tasks are not interruptible, this reports
  size_t stack_bytes = 0;       // per-thread stack (cudaLimitStackSize); ops with deep device calls (game ops) need a large one
  int warm_ms = 0;              // >0: keep the GPU busy this long in start(), right before the timed launch, so a short timed kernel runs at the clock of sustained work (the A100's SM clock falls to 1095 MHz under sparse work)
  KvmSchedCfg sched;            // SCHED: scheduling policy (off by default: the original single-queue executor)
};
enum class KvmSubmit { OK, BACKPRESSURE_TIMEOUT, TOO_BIG, DEVICE_ERROR, CLOSED, INVALID };
static const char* kvm_submit_name(KvmSubmit s) {
  static const char* n[] = {"OK", "BACKPRESSURE_TIMEOUT", "TOO_BIG", "DEVICE_ERROR", "CLOSED", "INVALID"};
  return n[(int)s];
}
struct KvmSchedStats {  // end-of-run copy of the device accounting (SCHED)
  bool valid = false;
  KvmSchedDev dev;
};
struct KvmResult {
  KvmSchedStats sched;
  std::vector<KvmSlotInfo> info;
  int err_code = 0, err_slot = -1, err_a = 0, err_b = 0, admitted = 0, done = 0, grid = 0;
  float kernel_ms = 0;
  std::vector<KvmTaskDesc> tasks;   // per slot: the latest occupant (== admission order while nothing was recycled)
  std::vector<KvmTrace> trace;
  std::vector<int> version;
  bool ok() const { return err_code == 0; }
};
struct KvmStats { int bundles = 0, backpressure_waits = 0, pads = 0, windows = 0, window_waits = 0; };

// How a buffer is initialised (device memory is written directly; no giant host copies).
struct KvmInit {
  enum Kind { ZERO, FILL, VALUES, SEEDED, FILE } kind = ZERO;
  float value = 0;                  // FILL
  std::vector<float> values;        // VALUES (small data)
  unsigned long long seed = 0; int draws = 1; float bias = 0, coef = 1;   // SEEDED (draws/bias/coef as in metadata)
  std::string path;                 // FILE: absolute or cwd-relative path, little-endian 4-byte elements
};
// Host reference of the seeded generator (bit-identical to the device kernel and runtime/initgen.py).
static void kvm_seeded_reference(float* out, size_t n, unsigned long long seed, int draws, float bias, float coef) {
  for (size_t i = 0; i < n; ++i) out[i] = kvm_seeded_value(seed, i, draws, bias, coef);
}

// Float arithmetic on every thread of every block for a few milliseconds per launch: what holds the SM clock up before a timed launch (--warm-ms).
__global__ void kvm_clock_warm_kernel(float* sink) {
  float x = 1.0f + static_cast<float>(threadIdx.x) / 512.0f;
  for (int i = 0; i < 300000; ++i) x = x * 1.0000001f + 1.0e-7f;
  if (threadIdx.x == 0) atomicAdd(sink, x);         // (keeps the loop's result live)
}

#ifdef KVM_GPT_SMEM_BYTES
// The position of argument `arg` in a registry signature ("name:type,..."), at compile time; -1 when absent. Arguments are read by name
// here because the registry is the ABI the frontend writes them by.
constexpr int kvm_sig_index(const char* sig, const char* arg) {
  for (int idx = 0; *sig; ++idx) {
    const char* q = arg; const char* p = sig;
    while (*q && *p == *q) { ++p; ++q; }
    if (!*q && *p == ':') return idx;
    while (*sig && *sig != ',') ++sig;
    if (*sig == ',') ++sig;
  }
  return -1;
}
constexpr const char* kvm_op_sig(int id) {
  switch (id) {
#define X(n, i, sig) case i: return sig;
#include "kvm_ops_list.inc"
#undef X
  }
  return "";
}
constexpr int KVM_ARG_TILE_FLAGS = kvm_sig_index(kvm_op_sig(KVM_OP_matmul_tile), "flags"), KVM_ARG_ATTN_FWD_HD = kvm_sig_index(kvm_op_sig(KVM_OP_attn_fwd), "hd"),
              KVM_ARG_ATTN_DQ_HD = kvm_sig_index(kvm_op_sig(KVM_OP_attn_bwd_dq), "hd"), KVM_ARG_ATTN_DKV_HD = kvm_sig_index(kvm_op_sig(KVM_OP_attn_bwd_dkv), "hd");
static_assert(KVM_ARG_TILE_FLAGS == KVM_GPT_ARG_TILE_FLAGS && KVM_ARG_ATTN_FWD_HD == KVM_GPT_ARG_ATTN_FWD_HD && KVM_ARG_ATTN_DQ_HD == KVM_GPT_ARG_ATTN_BWD_HD &&
              KVM_ARG_ATTN_DKV_HD == KVM_GPT_ARG_ATTN_BWD_HD, "kvm_gpt_task_smem reads the arguments it sizes by where the registry puts them");
// The dynamic shared memory one task needs. A GPT-pack op's is kvm_gpt_task_smem's, the function the op tests launch with too; the core pack
// takes the small ops' size; an op of any other pack (extra, comm, game, fused steps, test packs) gets the build's maximum, so a bundle
// holding one keeps that launch.
static size_t kvm_task_smem(const KvmTaskDesc& d, const int32_t* args) {
  const int gpt = kvm_gpt_task_smem(d.op_id, args + d.args_off);
  if (gpt >= 0) return (size_t)gpt;
  switch (d.op_id) {
#define X(n, i, sig) case i:
#include "kvm_ops.def"
#undef X
      return KVM_GPT_SMEM_SMALL;
    default: return KVM_DYN_SMEM_BYTES;
  }
}
#endif

class KvmExec {
 public:
  explicit KvmExec(KvmConfig c) : cfg_(c) {
    auto pow2 = [](unsigned long long v) { return v >= 2 && !(v & (v - 1)); };
    if (cfg_.aging_us < 0) throw std::invalid_argument("aging_us must be >= 0");
    if (cfg_.timeout_us == 0 || cfg_.timeout_us > 86400ull * 1000000) throw std::invalid_argument("timeout_us must be in [1, 86400 s]");
    if (cfg_.ring_words % 8 || cfg_.ring_words < 64) throw std::invalid_argument("ring_words must be a multiple of 8 and >= 64");
    if (!pow2(cfg_.live_cap) || !pow2(cfg_.max_tasks) || !pow2(cfg_.succ_cap) || !pow2(cfg_.win_cap)) throw std::invalid_argument("live_cap, max_tasks, succ_cap and win_cap must be powers of two");
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    sms_ = p.multiProcessorCount; name_ = p.name;
    if (cfg_.stack_bytes) CK(cudaDeviceSetLimit(cudaLimitStackSize, cfg_.stack_bytes));
    dyn_smem_ = KVM_DYN_SMEM_BYTES;
#ifdef KVM_TMA_MAXB   // GPT op pack built: arm the sm_90a TMA + wgmma tiles (no-op elsewhere; KVM_NO_HOPPER=1 disables)
    if (kvm_hopper_host_init()) { hopper_ = true; dyn_smem_ = kvm_hopper_dyn_smem(dyn_smem_); KLOG(2, "hopper: TMA + wgmma tiles armed, dynamic smem %zu B", dyn_smem_); }
#endif
    if (dyn_smem_ > 0) CK(cudaFuncSetAttribute(kvm_exec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kvm_exec_kernel, KVM_THREADS, dyn_smem_));
    if (occ < 1) throw std::runtime_error("kernel does not fit on an SM with " + std::to_string(dyn_smem_) + " bytes of dynamic shared memory");
    grid_ = cfg_.blocks ? cfg_.blocks : sms_ * occ;
    const KvmSchedCfg& sc = cfg_.sched;
    const int nsys = 1 + (sc.on ? 1 : 0);  // admitter (+ policy block)
    if (sc.on && (sc.num_classes < 1 || sc.num_classes > KVM_MAX_CLS)) throw std::invalid_argument("sched.num_classes must be 1..8");
    if (grid_ < nsys + 1) throw std::invalid_argument("need an admitter" + std::string(sc.on ? ", a policy block" : "") + " and at least one worker block");
    nc_ = sc.on ? sc.num_classes : 1;
    KLOG(2, "device %s, %d SMs, %d resident blocks/SM with %zu B dynamic smem -> grid %d (1 admitter%s + %d workers)", name_.c_str(), sms_, occ, dyn_smem_, grid_, sc.on ? " + 1 policy" : "", grid_ - nsys);
    const size_t T = cfg_.max_tasks;
    CK(cudaMalloc(&S_.tasks, T * sizeof(KvmTaskDesc)));
    CK(cudaMalloc(&S_.indeg, T * 4)); CK(cudaMalloc(&S_.state, T * 4)); CK(cudaMalloc(&S_.park_next, T * 4));
    CK(cudaMemset(S_.indeg, 0, T * 4));   // the admitter ADDS a task's indegree to what its predecessors already subtracted (kvm_admit_record), so a free slot's count is 0
    CK(cudaMalloc(&S_.trace, T * sizeof(KvmTrace)));
    CK(cudaMalloc(&S_.args, T * KVM_ARG_STRIDE * 4)); CK(cudaMalloc(&S_.adm_slots, (size_t)cfg_.live_cap * 4)); CK(cudaMalloc(&S_.stage, (size_t)cfg_.ring_words * 4));
    CK(cudaMalloc(&S_.succ, (size_t)cfg_.succ_cap * 4));
    CK(cudaMalloc(&S_.version, cfg_.num_vars * 4)); CK(cudaMalloc(&S_.park_head, cfg_.num_vars * 4));
    // S_.cells is allocated in prepare(), once the run's facts are known (a FIFO run needs one ring, not one per priority bucket per class: 1/32 of the memory, 4 GB on the largest record)
    CK(cudaMalloc(&S_.fcells, T * sizeof(KvmCell)));
    CK(cudaMalloc(&S_.qctl, nc_ * KVM_NBUCKETS * sizeof(KvmQCtl)));
    CK(cudaMalloc(&S_.wins, (size_t)cfg_.win_cap * sizeof(KvmWin)));
    CK(cudaMalloc(&S_.cnt, sizeof(KvmCounters)));
    CK(cudaMemset(S_.tasks, 0, T * sizeof(KvmTaskDesc))); CK(cudaMemset(S_.trace, 0, T * sizeof(KvmTrace)));
    CK(cudaMemset(S_.version, 0, cfg_.num_vars * 4)); CK(cudaMemset(S_.qctl, 0, nc_ * KVM_NBUCKETS * sizeof(KvmQCtl)));
    CK(cudaMemset(S_.wins, 0, (size_t)cfg_.win_cap * sizeof(KvmWin))); CK(cudaMemset(S_.cnt, 0, sizeof(KvmCounters)));
    // host-shared control block: ring + doorbell/ring_head/closed on separate lines + error mirror + per-window done marks
    CK(cudaHostAlloc((void**)&ctl_, 4096, cudaHostAllocMapped)); memset(ctl_, 0, 4096);
    init_sched();
    CK(cudaHostAlloc((void**)&ring_, (size_t)cfg_.ring_words * 4, cudaHostAllocMapped)); memset(ring_, 0, (size_t)cfg_.ring_words * 4);
    CK(cudaHostAlloc((void**)&win_done_, (size_t)cfg_.win_cap * 4, cudaHostAllocMapped)); memset((void*)win_done_, 0, (size_t)cfg_.win_cap * 4);
    doorbell_ = (unsigned long long*)(ctl_ + 0); ring_head_ = (unsigned long long*)(ctl_ + 128);
    closed_ = (unsigned*)(ctl_ + 256); herr_ = (KvmHostErr*)(ctl_ + 384);
    void* d;
    CK(cudaHostGetDevicePointer(&d, ring_, 0)); S_.ring = (int*)d;
    CK(cudaHostGetDevicePointer(&d, doorbell_, 0)); S_.doorbell = (unsigned long long*)d;
    CK(cudaHostGetDevicePointer(&d, ring_head_, 0)); S_.ring_head = (unsigned long long*)d;
    CK(cudaHostGetDevicePointer(&d, closed_, 0)); S_.closed = (unsigned*)d;
    CK(cudaHostGetDevicePointer(&d, herr_, 0)); S_.herr = (KvmHostErr*)d;
    CK(cudaHostGetDevicePointer(&d, (void*)win_done_, 0)); S_.win_done = (unsigned*)d;
    S_.nslots = cfg_.max_tasks; S_.live_cap = cfg_.live_cap; S_.succ_cap = cfg_.succ_cap; S_.win_cap = cfg_.win_cap;
    S_.num_vars = cfg_.num_vars; S_.ring_words = cfg_.ring_words;
    S_.aging_ns = cfg_.aging_us ? (long long)cfg_.aging_us * 1000 : (1ll << 50);
    S_.cont_on = cfg_.cont;
#ifdef KVM_PROF_GAP
    S_.noop_ops = cfg_.noop_ops;
#endif
    set_fifo(cfg_.fifo_only);
    S_.no_recycle = cfg_.no_recycle; S_.no_waits = cfg_.no_waits; S_.trace_on = cfg_.trace_on;
    S_.prestaged = 0; S_.warm_only = 0;
    S_.timeout_ns = cfg_.timeout_us * 1000ull; S_.task_limit_ns = cfg_.task_limit_us * 1000ull;
    S_.rcap = 4 * cfg_.live_cap;
    CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CK(cudaEventCreate(&e0_)); CK(cudaEventCreate(&e1_));
  }
  ~KvmExec() {
    if (started_ && !finished_) { try { close(); finish(); } catch (...) {} }
    for (size_t i = 0; i < bufs_.size(); ++i) if (!std::count(views_.begin(), views_.end(), i)) cudaFree(bufs_[i].p);
    for (float* p : arenas_) cudaFree(p);
    for (void* p : {(void*)S_.tasks, (void*)S_.indeg, (void*)S_.state, (void*)S_.park_next, (void*)S_.trace, (void*)S_.succ, (void*)S_.args, (void*)S_.adm_slots, (void*)S_.stage,
                    (void*)S_.version, (void*)S_.park_head, (void*)S_.cells, (void*)S_.fcells, (void*)S_.qctl, (void*)S_.wins, (void*)S_.cnt, (void*)S_.bufs, (void*)S_.sp.dev, (void*)S_.sp.info, (void*)S_.sp.pv, (void*)S_.sp.pubcls}) cudaFree(p);
    cudaFreeHost(ctl_); cudaFreeHost(ring_); cudaFreeHost((void*)win_done_);
    cudaStreamDestroy(stream_); cudaEventDestroy(e0_); cudaEventDestroy(e1_);
  }
  KvmExec(const KvmExec&) = delete; KvmExec& operator=(const KvmExec&) = delete;

  // Allocates a rows x cols buffer of dtype 0 (f32), 1 (i32) or 2 (bf16: raw 16-bit storage, 2 bytes per element) and initialises it on the device.
  int add_buffer(int rows, int cols, const KvmInit& init = {}, int dtype = 0) {
    if (started_ || prepared_) throw std::logic_error("buffers are fixed at start()");
    if (rows < 1 || cols < 1 || (dtype < 0 || dtype > 2)) throw std::invalid_argument("bad buffer shape or dtype");
    const size_t n = (size_t)rows * cols;
    KvmBuf b{}; CK(cudaMalloc(&b.p, n * (dtype == 2 ? 2 : 4)));
    b.rows = rows; b.cols = cols; b.ld = cols; b.dtype = dtype;
    init_buffer((float*)b.p, n, init, dtype);
    bufs_.push_back(b);
    return (int)bufs_.size() - 1;
  }
  int add_buffer(int rows, int cols, const std::vector<float>& values) { KvmInit i; if (!values.empty()) { i.kind = KvmInit::VALUES; i.values = values; } return add_buffer(rows, cols, i); }
  // Pooled device memory for many small buffers: arena() then add_view(). Zero-initialised, freed with the executor.
  float* arena(size_t floats) {
    float* p; CK(cudaMalloc(&p, floats * 4)); CK(cudaMemset(p, 0, floats * 4)); arenas_.push_back(p); return p;
  }
  int add_view(float* p, int rows, int cols, int dtype = 0) {
    if (started_ || prepared_) throw std::logic_error("buffers are fixed at start()");
    KvmBuf b{}; b.p = p; b.rows = rows; b.cols = cols; b.ld = cols; b.dtype = dtype; views_.push_back(bufs_.size());
    bufs_.push_back(b); return (int)bufs_.size() - 1;
  }
  void set_version(int var, int v) {
    if (started_) throw std::logic_error("versions are fixed at start()");
    if (var < 0 || var >= cfg_.num_vars) throw std::invalid_argument("version variable out of range");
    CK(cudaMemcpy(S_.version + var, &v, 4, cudaMemcpyHostToDevice));
  }
  // KVM_DIST_HOOK: attach the distributed-execution descriptor (inbox, in_vars, ctl). Device-visible memory, outlives the run.
  void set_dist(const KvmDistDev* d) { if (started_) throw std::logic_error("dist is fixed at start()"); S_.dist = d; }
  int nbufs() const { return (int)bufs_.size(); }
  int grid() const { return grid_; }
  const KvmConfig& config() const { return cfg_; }
  const KvmStats& stats() const { return stats_; }

  // ---- SCHED: runtime policy control through the mapped control block (no relaunch) ----
  KvmSchedParams* sched_params() { return hp_; }                // controllers write with KvmController::write_params
  void set_sched_params(const KvmSchedParams& p) { if (!cfg_.sched.on) throw std::logic_error("scheduler is off"); validate_sched(p); KvmController::write_params(hp_, p); }
  const KvmSchedTele* telemetry() const { return ht_; }
  int num_classes() const { return nc_; }
  int workers() const { return grid_ - 1 - (cfg_.sched.on ? 1 : 0); }
  // The FIFO ring (KvmDevState::fifo_only) never waits for a free cell: kvm_publish_cell stores a slot and its generation in one release store,
  // and each idle worker watches the cell of its own ticket. Its 4 x live_cap cells must exceed every position in flight: the ready tasks
  // (at most 2 x live_cap, the header's budgets) plus one ticket per worker. workers < live_cap keeps them under 3 x live_cap, so a worker
  // that stalls between taking its ticket and reading its cell is overrun only after more than live_cap further enqueues; past the bound
  // waiting workers share cells. Either way the worker finds a later generation and stops the run with QUEUE_OVERFLOW (the litmus probe
  // `queuefifo_cap` overruns it at live_cap 2 with 200 CTAs). The cell holds a slot in its low KVM_FIFO_SHIFT bits and the generation above
  // them, so the slots must fit those bits, and a run has 2^(64 - KVM_FIFO_SHIFT) enqueues before the generation wraps (the stall timeout
  // reports that). The bucket rings wait for a free cell, so they serve any live_cap. CLI runs sit far inside the bound (live_cap >= 16,384
  // against a few hundred workers).
  bool fifo_fits() const { return workers() < cfg_.live_cap && cfg_.max_tasks <= (1 << KVM_FIFO_SHIFT); }
  bool fifo_only() const { return S_.fifo_only != 0; }   // the ring in use; config().fifo_only stays the caller's request, which submit() holds to
  void static_split(KvmSchedParams& p, const std::vector<double>& shares) const { kvm_static_split(p, shares, sms_); }

  // KVM_DIST_HOOK: the allocation half of start(). cudaMalloc and cudaHostAlloc may synchronise with EVERY kernel running on the
  // device, so when several executors share one GPU (tests, small clusters) all of them must prepare() before the first
  // start(): a start() that allocates while a peer's persistent kernel waits for this very executor deadlocks until the peer's
  // timeout. A lone executor never needs to call it.
  // Launch with the dynamic shared memory the run's tasks need, not the build's maximum (KVM_DYN_SMEM_BYTES): the SM's 192 KB serve shared
  // memory first, so a 147,456 B launch leaves the A100 28 KB of L1 for every op's cached loads and spill reloads (measured on an A100: the same tree's
  // gpt-tiny ran x1.12 faster at 90 KB). Call once every task of the run is known and before any launch (the CLI's bundles); a run that may
  // be handed any op later keeps the maximum. The grid stays the one the constructor sized at the maximum (one block an SM either way).
  void size_launch(const std::vector<KvmBundle>& bundles) {
    if (started_) throw std::logic_error("size_launch after start");
#ifdef KVM_GPT_SMEM_BYTES
    if (hopper_) { KLOG(2, "launch: %zu B of dynamic shared memory, kept for the Hopper tiles", dyn_smem_); return; }
    size_t need = KVM_GPT_SMEM_SMALL; int by = -1; long tasks = 0;
    for (const KvmBundle& b : bundles)
      for (const KvmTaskDesc& d : b.tasks) { const size_t n = kvm_task_smem(d, b.args.data()); ++tasks; if (n > need) { need = n; by = d.op_id; } }
    need = std::min(need, dyn_smem_);
    const KvmOpInfo* op = by >= 0 ? kvm_op(by) : nullptr;
    KLOG(2, "launch: %zu B of dynamic shared memory, sized to the largest of %ld tasks (%s), against the build's %zu B", need, tasks, op ? op->name : "small ops only", dyn_smem_);
    if (need != dyn_smem_) { dyn_smem_ = need; CK(cudaFuncSetAttribute(kvm_exec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_)); }
#else
    (void)bundles;
#endif
  }
  void prepare() {
    if (prepared_) return;
    prepared_ = true;
    CK(cudaMalloc(&S_.cells, (size_t)(S_.fifo_only ? 1 : nc_ * KVM_NBUCKETS) * 4 * (size_t)cfg_.live_cap * sizeof(KvmCell)));   // ready rings (see the constructor)
    KvmBuf* d; size_t n = std::max<size_t>(bufs_.size(), 1);
    CK(cudaMalloc(&d, n * sizeof(KvmBuf)));
    if (!bufs_.empty()) CK(cudaMemcpy(d, bufs_.data(), bufs_.size() * sizeof(KvmBuf), cudaMemcpyHostToDevice));
    S_.bufs = d; S_.nbufs = (int)bufs_.size();
    if (cfg_.sched.on) {  // the policy mirror starts as the launch parameters, so workers never see a zero weight
      KvmSchedDev init{}; for (int& b : init.op_blk) b = -1; const KvmSchedParams& p = cfg_.sched.p;
      init.flags = p.flags; init.affinity_ns = p.affinity_us * 1000; init.bw_limit = std::max(p.bw_limit_min, std::min(p.bw_limit_max, p.bw_limit_init));
      for (int c = 0; c < nc_; ++c) {
        KvmClsDev& k = init.cls[c]; const KvmClassParams& q = p.cls[c];
        k.weight = q.weight; k.min_cta = q.min_cta; k.max_cta = q.max_cta; k.bw_cta = q.bw_cta; k.urgent_us = q.urgent_us;
        k.protect = q.protect; k.target_wait_us = q.target_wait_us; k.sm_hi = q.sm_hi; k.vt_mul = (1 << 20) / std::max(q.weight, 1);
      }
      CK(cudaMemcpy(S_.sp.dev, &init, sizeof init, cudaMemcpyHostToDevice));
    }
    // The constructor's cudaMemsets and the upload above run in the LEGACY default stream (a pageable cudaMemcpy may even return
    // before its DMA lands), while stream_ is non-blocking, so nothing orders them before the kernel: with several executors
    // starting at once the kernel could see an unzeroed counter or a half-copied buffer table (random TIMEOUT / BAD_BUF).
    // cudaStreamSynchronize(0) waits for the legacy stream only, never for another executor's kernel.
    CK(cudaStreamSynchronize(0));
  }
  // Per-run facts the CLI learns from its bundles, set before start(): see KvmDevState::fifo_only, no_recycle and no_waits.
  void set_run_facts(bool fifo_only, bool no_recycle, bool no_waits, bool trace_on) {
    if (started_ || prepared_) throw std::logic_error("set_run_facts before start()");
    set_fifo(fifo_only); S_.no_recycle = cfg_.no_recycle = no_recycle; S_.no_waits = cfg_.no_waits = no_waits;
    S_.trace_on = cfg_.trace_on = trace_on || cfg_.sched.on;   // the scheduler's statistics read the trace
  }
  // Write records into the ring BEFORE the kernel starts (submit after prestart(), then close() and start()): the host's validation and copy then sit outside the
  // kernel's timed window, and the admitter finds the record the moment it runs. For a run whose whole input is known up front (the CLI's single bundle).
  void prestart() { if (started_) throw std::logic_error("prestart after start()"); prepare(); prestart_ = true; }
  // Copy the record already in the mapped ring (one record, at ring offset 0) into device memory before the kernel starts: the admitter then skips its staging pass, and the PCIe copy leaves the kernel's timed window.
  void prestage() {
    if (started_ || !prestart_) throw std::logic_error("prestage after prestart(), before start()");
    if (wpos_) CK(cudaMemcpy(S_.stage, ring_, wpos_ * 4, cudaMemcpyHostToDevice));
    S_.prestaged = wpos_ != 0;
  }
  // --warm-ms: sustained GPU work between the setup and the timed launch. The SM clock runs at 1095 MHz under light or sparse work and at 1410 under sustained work (A100, Nov 9);
  // it reaches 1410 within about 0.3 s of sustained load and holds it about 1.7 s after. A process's setup (bundle load, allocations) can outlast that hold, so the warm-up runs here, after setup.
  void clock_warmup() {
    if (cfg_.warm_ms <= 0) return;
    float* sink = nullptr;
    CK(cudaMalloc(&sink, sizeof(float)));
    const auto t0 = std::chrono::steady_clock::now();
    int launches = 0;
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < cfg_.warm_ms) {
      kvm_clock_warm_kernel<<<grid_, KVM_THREADS, 0, stream_>>>(sink);
      CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream_));
      ++launches;
    }
    CK(cudaFree(sink));
    KLOG(2, "clock warm-up: %d launches, %d ms requested", launches, cfg_.warm_ms);
  }
  void start() {
    if (started_) throw std::logic_error("already started");
    prepare();
    { KvmDevState warm = S_; warm.warm_only = 1; kvm_exec_kernel<<<1, KVM_THREADS, dyn_smem_, stream_>>>(warm); CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream_)); }   // upload the kernel's code outside the timed window
    clock_warmup();
    kvm_init_kernel<<<64, KVM_THREADS, 0, stream_>>>(S_);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e0_, stream_));
    kvm_exec_kernel<<<grid_, KVM_THREADS, dyn_smem_, stream_>>>(S_);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1_, stream_));
    started_ = true;
    KLOG(2, "kernel launched: grid %d x %d threads, live_cap %d, slots %d, succ ring %u words, ring %d words, aging %d us, sched %s", grid_, KVM_THREADS, cfg_.live_cap, cfg_.max_tasks, cfg_.succ_cap, cfg_.ring_words, cfg_.aging_us,
         !cfg_.sched.on ? "off" : cfg_.sched.mode == KVM_SCHED_GLOBAL ? "global" : cfg_.sched.mode == KVM_SCHED_STATIC ? "static" : "dynamic");
  }

  // Thread-safe producer. Blocks (up to timeout_ms) while the ring is full: that is the backpressure; it never drops.
  // *wid_out: the bundle record's window id (see wait_window).
  KvmSubmit submit(const KvmBundle& b, int timeout_ms = 10000, bool validate = true, unsigned* wid_out = nullptr) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_ && !prestart_) throw std::logic_error("submit before start()");
    if (closed_flag_) { KLOG(1, "submit after close()"); return KvmSubmit::CLOSED; }
    if (herr_->code) { last_error_ = std::string("device error ") + kvm_err_name(herr_->code); return KvmSubmit::DEVICE_ERROR; }   // the device is dead: accepted bundles would never run
    std::string why = validate ? b.validate(nbufs(), cfg_.num_vars, cfg_.live_cap) : std::string();
    if (!why.empty()) { KLOG(0, "bundle rejected: %s", why.c_str()); last_error_ = why; return why.find("live_cap") != std::string::npos ? KvmSubmit::TOO_BIG : KvmSubmit::INVALID; }
    if (cfg_.fifo_only) for (const KvmTaskDesc& t : b.tasks) if (t.priority != 0 || t.deadline_us != -1) {   // the FIFO ring serves everything in release order: a priority or deadline it cannot honor is a caller error, not a silent reorder
      last_error_ = "the FIFO ring takes only tasks of priority 0 with no deadline"; KLOG(0, "bundle rejected: %s (task priority %d, deadline %d us)", last_error_.c_str(), t.priority, t.deadline_us); return KvmSubmit::INVALID;
    }
    const size_t words = b.words();
    if (words > (size_t)cfg_.ring_words) { last_error_ = "bundle record larger than the admission ring"; KLOG(0, "%s (%zu > %d words)", last_error_.c_str(), words, cfg_.ring_words); return KvmSubmit::TOO_BIG; }
    auto t0 = std::chrono::steady_clock::now();
    bool waited = false;
    // A record is contiguous. If it does not fit before the ring end, a pad record fills the tail and is published ON ITS OWN
    // (waiting for pad + record at once could need more than the whole ring); then the record waits for space at lap start.
    for (;;) {
      unsigned long long head = __atomic_load_n(ring_head_, __ATOMIC_ACQUIRE);
      const size_t rem = cfg_.ring_words - wpos_ % cfg_.ring_words, need = words > rem ? rem : words;
      if (cfg_.ring_words - (wpos_ - head) >= need) {
        if (need == words) break;
        write_header(rem, 0, 0, 0, KVM_REC_PAD); wpos_ += rem; ++stats_.pads;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(doorbell_, wpos_, __ATOMIC_RELEASE);
        KLOG(3, "ring wrap: %zu pad words", rem);
        continue;
      }
      if (herr_->code) { last_error_ = "device error"; return KvmSubmit::DEVICE_ERROR; }
      if (!waited) { waited = true; ++stats_.backpressure_waits; KLOG(3, "ring full (%llu/%d words used): backpressure", (unsigned long long)(wpos_ - head), cfg_.ring_words); }
      if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
        last_error_ = "admission ring full for " + std::to_string(timeout_ms) + " ms";
        KLOG(1, "%s", last_error_.c_str()); return KvmSubmit::BACKPRESSURE_TIMEOUT;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    int* r = ring_ + wpos_ % cfg_.ring_words;
    write_header(words, (int)b.tasks.size(), (int)b.succ.size(), (int)b.args.size(), KVM_REC_BUNDLE);
    r += KVM_REC_HDR;
    memcpy(r, b.tasks.data(), b.tasks.size() * sizeof(KvmTaskDesc)); r += b.tasks.size() * KVM_TASK_WORDS;
    if (!b.succ.empty()) memcpy(r, b.succ.data(), b.succ.size() * 4);
    r += b.succ.size();
    if (!b.args.empty()) memcpy(r, b.args.data(), b.args.size() * 4);
    wpos_ += words;
    // Publication: record bytes first (seq_cst fence also orders non-temporal stores), then the doorbell with release.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(doorbell_, wpos_, __ATOMIC_RELEASE);
    if (wid_out) *wid_out = (unsigned)stats_.bundles;
    rec_tasks_.push_back((int)b.tasks.size());
    ++stats_.bundles;
    return KvmSubmit::OK;
  }
  const std::string& last_error() const { return last_error_; }
  // Tasks per submitted record, indexed by window id: the admitter hands out slots in record order, so while nothing was recycled
  // (total <= max_tasks) record w's tasks occupy slots [sum of earlier records, +size).
  std::vector<int> record_tasks() { std::lock_guard<std::mutex> lk(mu_); return rec_tasks_; }
  int device_error() const { return (int)herr_->code; }   // 0 while healthy
  // Words of the admission ring the device has not finished copying yet (what a record submitted now would wait behind).
  size_t ring_backlog_words() {
    std::lock_guard<std::mutex> lk(mu_);
    return (size_t)(wpos_ - __atomic_load_n(ring_head_, __ATOMIC_ACQUIRE));
  }
  // Live versions of all variables, read while the kernel runs (one small DMA from the non-blocking stream's device memory).
  std::vector<int> live_versions() const {
    std::vector<int> v(cfg_.num_vars); CK(cudaMemcpy(v.data(), S_.version, v.size() * 4, cudaMemcpyDeviceToHost)); return v;
  }

  // True once every task of window wid (a submitted bundle record) has completed.
  // win_done_[slot] holds the id+1 of the newest finished window in that slot. A slot is reused only after its old window finished, so
  // a value at or past wid+1 (wrap-safe signed distance) means wid finished, even after the table wrapped over it.
#ifdef KVM_BREAK_WINDOW_DONE   // negative build: the pre-fix equality test (false for a finished window once the table wrapped over it: the `recycle` flake)
  bool window_done(unsigned wid) const { return __atomic_load_n(&win_done_[wid & (cfg_.win_cap - 1)], __ATOMIC_ACQUIRE) == wid + 1; }
#else
  bool window_done(unsigned wid) const { return (int)(__atomic_load_n(&win_done_[wid & (cfg_.win_cap - 1)], __ATOMIC_ACQUIRE) - (wid + 1)) >= 0; }
#endif
  // Diagnostics for a window that never reads as done: the raw mark in wid's table slot (id+1 of the newest finished window there, 0 =
  // none yet). A mark past wid+1 means the table wrapped over wid, which window_done counts as finished; below wid+1 means it really has not.
  unsigned window_mark(unsigned wid) const { return __atomic_load_n(&win_done_[wid & (cfg_.win_cap - 1)], __ATOMIC_ACQUIRE); }
  std::string window_why_not_done(unsigned wid) const {
    char b[200]; snprintf(b, sizeof b, "window %u (slot %u of %d): mark %u, needs >= %u (%d windows behind)", wid, wid & (cfg_.win_cap - 1), cfg_.win_cap, window_mark(wid), wid + 1, (int)(wid + 1 - window_mark(wid)));
    return b;
  }

  // One huge bundle through bounded slots: the bundle is cut into windows of consecutive tasks (task order must be
  // topological: every edge points to a higher index). Edges inside a window go to the device as usual; an edge that crosses
  // windows is enforced HERE: the later window is submitted only after the windows holding its predecessors completed (the
  // host reads the per-window completion marks the device mirrors into pinned memory). Windows of other producers interleave
  // freely. Cost: a window with a cross-window dependency starts after its producers drain, so choose big windows.
  KvmSubmit submit_stream(const KvmBundle& big, int max_window_tasks = 0, int timeout_ms = 60000, std::vector<unsigned>* wids_out = nullptr) {
    const auto tp0 = std::chrono::steady_clock::now();   // host prep time (validate + plan) is time the device idles: logged below
    std::string why = big.validate(nbufs(), cfg_.num_vars, 0);
    if (!why.empty()) { KLOG(0, "stream rejected: %s", why.c_str()); last_error_ = why; return KvmSubmit::INVALID; }
    const int nt = (int)big.tasks.size();
    for (int i = 0; i < nt; ++i) for (int k = 0; k < big.tasks[i].succ_cnt; ++k)
      if (big.succ[big.tasks[i].succ_off + k] <= i) { last_error_ = "stream needs topologically ordered tasks (edge " + std::to_string(i) + " -> " + std::to_string(big.succ[big.tasks[i].succ_off + k]) + ")"; KLOG(0, "%s", last_error_.c_str()); return KvmSubmit::INVALID; }
    // window boundaries: bounded by tasks, and by record words (half the ring) counting every outgoing edge
    const int cap = std::min({max_window_tasks > 0 ? max_window_tasks : cfg_.live_cap, cfg_.live_cap, cfg_.max_tasks});
    std::vector<int> lo{0};
    { size_t words = KVM_REC_HDR; int cnt = 0;
      for (int i = 0; i < nt; ++i) {
        size_t w = KVM_TASK_WORDS + big.tasks[i].succ_cnt + big.tasks[i].args_words;
        if (cnt > 0 && (cnt >= cap || words + w > (size_t)cfg_.ring_words / 2)) { lo.push_back(i); words = KVM_REC_HDR; cnt = 0; }
        words += w; ++cnt;
      }
      lo.push_back(nt); }
    const int nw = (int)lo.size() - 1;
    std::vector<int> win_of(nt);
    for (int k = 0; k < nw; ++k) for (int i = lo[k]; i < lo[k + 1]; ++i) win_of[i] = k;
    // Windows each window must wait for. Sources are visited in task (= window) order, so the producers of one window arrive in
    // non-decreasing order and a "last producer added" stamp de-duplicates exactly (a set insert per edge cost 350 ms on 6.9M edges).
    std::vector<std::vector<int>> needs(nw);
    { std::vector<int> last(nw, -1);
      for (int i = 0; i < nt; ++i) for (int k = 0; k < big.tasks[i].succ_cnt; ++k) {
        const int t = big.succ[big.tasks[i].succ_off + k], wt = win_of[t], wi = win_of[i];
        if (wt != wi && last[wt] != wi) { last[wt] = wi; needs[wt].push_back(wi); }
      } }
    KLOG(2, "stream: host prep (validate, order check, %d windows, cross-window needs) %.1f ms", nw, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count());
    std::vector<unsigned> wid(nw, 0);
    for (int k = 0; k < nw; ++k) {
      for (int j : needs[k]) {                                  // cross-window predecessors must have completed
        auto t0 = std::chrono::steady_clock::now(); bool waited = false;
        while (!window_done(wid[j])) {
          if (herr_->code) { last_error_ = "device error"; return KvmSubmit::DEVICE_ERROR; }
          if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) { last_error_ = "window " + std::to_string(j) + " did not complete: " + window_why_not_done(wid[j]); KLOG(1, "%s", last_error_.c_str()); return KvmSubmit::BACKPRESSURE_TIMEOUT; }
          if (!waited) { waited = true; ++stats_.window_waits; }
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
      }
      KvmBundle w; const int a = lo[k], z = lo[k + 1];
      w.tasks.reserve(z - a);
      for (int i = a; i < z; ++i) {
        KvmTaskDesc t = big.tasks[i];
        const int off = (int)w.succ.size(); t.succ_off = off;
        for (int e = 0; e < t.succ_cnt; ++e) { int s = big.succ[big.tasks[i].succ_off + e]; if (s < z) w.succ.push_back(s - a); }
        t.succ_cnt = (int)w.succ.size() - off; t.indegree = 0;
        t.args_off = (int)w.args.size();
        w.args.insert(w.args.end(), big.args.begin() + big.tasks[i].args_off, big.args.begin() + big.tasks[i].args_off + t.args_words);
        w.tasks.push_back(t);
      }
      for (int s : w.succ) ++w.tasks[s].indegree;
      KvmSubmit st = submit(w, timeout_ms, false, &wid[k]);
      if (st != KvmSubmit::OK) return st;
      if (wids_out) wids_out->push_back(wid[k]);
      ++stats_.windows;
    }
    KLOG(3, "stream: %d tasks in %d windows", nt, nw);
    return KvmSubmit::OK;
  }

  // ---- fused MLP steps (kvm_fuse.h): the rewritten bundle's runs, their device state
  // Runs the fusion pass over a closed bundle: what it fuses, and why it left each other run alone, goes to the log. The caller swaps the rewritten bundle in when something fused.
  // The shared memory a fused block may use on this device: the static kernel's opt-in maximum less its own static shared memory, in floats (16-float multiples).
  int fused_smem_floats() const {
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kvm_static_kernel));
    const long long bytes = (long long)p.sharedMemPerBlockOptin - (long long)fa.sharedSizeBytes;
    return bytes > 0 ? (int)(bytes / 4) & ~15 : 0;
  }
  KvmFuseResult fuse_bundle(const KvmBundle& b, int workers, const std::vector<int>& keep) const {
    KvmFuseOps ops;
    ops.noop = kvm_op_by_name("noop"); ops.select_window = kvm_op_by_name("select_window"); ops.embedding_fwd = kvm_op_by_name("embedding_fwd"); ops.matmul_tile = kvm_op_by_name("matmul_tile");
    ops.softmax_ce_fwd = kvm_op_by_name("softmax_ce_fwd"); ops.softmax_ce_reduce = kvm_op_by_name("softmax_ce_reduce"); ops.softmax_ce_bwd = kvm_op_by_name("softmax_ce_bwd"); ops.softmax_ce_fused = kvm_op_by_name("softmax_ce_fused");
    ops.gelu_bwd = kvm_op_by_name("gelu_bwd"); ops.colsum = kvm_op_by_name("colsum"); ops.embedding_bwd = kvm_op_by_name("embedding_bwd"); ops.sgd = kvm_op_by_name("sgd"); ops.sgd_range = kvm_op_by_name("sgd_range");
    ops.fmlp_slice = kvm_op_by_name("fmlp_slice"); ops.fmlp_sgd = kvm_op_by_name("fmlp_sgd"); ops.fmlp_split = kvm_op_by_name("fmlp_split");
    std::vector<KvmFuseBuf> infos(bufs_.size());
    for (size_t i = 0; i < bufs_.size(); ++i) infos[i] = KvmFuseBuf{bufs_[i].rows, bufs_[i].cols, bufs_[i].ld, bufs_[i].dtype};
    const KvmSOpLookup op_of = [](int id) { const KvmOpInfo* i = kvm_op(id); KvmSOpInfo v; if (i) { v.arity = i->arity; v.types = i->types; kvm_static_gang_args(i->sig, v); } return v; };
    KvmFuseOptions opts; opts.workers = workers; opts.keep = keep; opts.smem_floats = fused_smem_floats();
    KvmFuseResult r = kvm_fuse_mlp(b.tasks, b.succ, b.args, infos, ops, op_of, opts);
    for (const std::string& line : r.report) KLOG(2, "fuse: %s", line.c_str());
    if (r.fused_runs) KLOG(2, "fuse: shared-memory budget %d floats (%d bytes) a fused block", opts.smem_floats, opts.smem_floats * 4);
    KLOG(2, "fuse: %d run(s) fused, %d unchanged; %d tasks replaced by %d", r.fused_runs, r.unchanged_runs, r.replaced_tasks, r.new_tasks);
    return r;
  }

  struct FusedState { KvmFusedDesc* desc = nullptr; KvmSplitDesc* split = nullptr; std::vector<void*> allocs; int smem_bytes = 0; };
  // One descriptor per fused run, with its partial-gradient scratch (zeroed), and the shared memory the largest run's slice (or split member) needs. A run's index is its descriptor's in both
  // tables, the one its form uses (kvm_fuse.h's ops carry it as `desc`); the other table's entry is unused. Buffers are already device-allocated (after prepare()).
  FusedState make_fused_state(const std::vector<KvmFuseRun>& runs) {
    FusedState st;
    std::vector<KvmFusedDesc> descs(runs.size());
    std::vector<KvmSplitDesc> splits(runs.size());
    bool any_split = false;
    const auto scratch = [&](size_t floats) { float* p; CK(cudaMalloc(&p, floats * sizeof(float))); CK(cudaMemset(p, 0, floats * sizeof(float))); st.allocs.push_back(p); return p; };
    const auto param = [&](int idx) { return static_cast<float*>(bufs_[idx].p); };
    for (size_t i = 0; i < runs.size(); ++i) {
      const KvmFuseRun& r = runs[i];
      if (r.split) {   // the split form: Q x S members, their partials, the exchanges, the gang's barrier words (kvm_fused_split.h)
        KvmSplitDesc& d = splits[i]; any_split = true;
        d.V = r.V; d.E = r.E; d.L = r.L; d.H = r.H[0]; d.S = r.S; d.Q = r.Q; d.lr = r.lr;
        const int floats = kvm_split_layout(d);
        if (floats < 0 || floats > r.smem_floats) throw std::runtime_error("fuse: a split run's layout does not fit what the pass sized it for");
        st.smem_bytes = std::max(st.smem_bytes, floats * (int)sizeof(float));
        const int nslot = d.Q * d.S, R = KVM_SPLIT_R;
        d.we = param(r.wte); d.wh = param(r.head);
        for (int l = 0; l < r.L; ++l) {
          d.w[l] = param(r.w[l]); d.b[l] = param(r.b[l]);
          d.dW[l] = scratch((size_t)nslot * (l ? d.HR : d.E) * d.HP); d.db[l] = scratch((size_t)nslot * d.HP);
          if (l + 1 < r.L) d.XH[l] = scratch((size_t)d.Q * R * d.HR);   // zeroed: the pad columns past H are never written
        }
        for (int k = 0; k < 2; ++k) d.XD[k] = r.L > 1 ? scratch((size_t)nslot * R * d.HR) : nullptr;
        d.P = scratch((size_t)nslot * R * d.VP); d.dWh = scratch((size_t)nslot * d.HP * d.VP); d.dX0 = scratch((size_t)nslot * R * d.E);
        d.loss_part = scratch(d.Q);
        d.ctl = reinterpret_cast<unsigned*>(scratch(kvm_gang::line_words(nslot)));   // the gang's flag lines, zeroed once (cudaMalloc: 256-byte aligned)
        continue;
      }
      KvmFusedDesc& d = descs[i];
      d.R = r.slice_rows; d.V = r.V; d.E = r.E; d.L = r.L; d.nsl = r.nsl; d.T = r.T; d.lr = r.lr; d.groups = r.groups;
      for (int l = 0; l < r.L; ++l) d.H[l] = r.H[l];
      st.smem_bytes = std::max(st.smem_bytes, kvm_fmlp_layout(d, r.smem_floats) * (int)sizeof(float));
      d.we = param(r.wte); d.wh = param(r.head);
      d.ge = scratch((size_t)r.nsl * r.V * r.E);
      d.loss_part = scratch(r.nsl);
      for (int l = 0; l < r.L; ++l) {
        const int K = l ? r.H[l - 1] : r.E;
        d.w[l] = param(r.w[l]); d.b[l] = param(r.b[l]);
        d.gwld[l] = kvm_fmlp_pad16(r.H[l]);
        d.gw[l] = scratch((size_t)r.nsl * kvm_fmlp_pad16(K) * d.gwld[l]);
        d.gb[l] = scratch((size_t)r.nsl * r.H[l]);
      }
      d.ghld = kvm_fmlp_pad16(r.V);
      d.gh = scratch((size_t)r.nsl * kvm_fmlp_pad16(r.H[r.L - 1]) * d.ghld);
    }
    CK(cudaMalloc(&st.desc, descs.size() * sizeof(KvmFusedDesc)));
    CK(cudaMemcpy(st.desc, descs.data(), descs.size() * sizeof(KvmFusedDesc), cudaMemcpyHostToDevice));
    st.allocs.push_back(st.desc);
    if (any_split) {
      CK(cudaMalloc(&st.split, splits.size() * sizeof(KvmSplitDesc)));
      CK(cudaMemcpy(st.split, splits.data(), splits.size() * sizeof(KvmSplitDesc), cudaMemcpyHostToDevice));
      st.allocs.push_back(st.split);
    }
    return st;
  }

  // STATIC SCHEDULE: compile `b` (kvm_static.h) into one program per worker and run them all in one launch (kvm_static.cuh), instead of start()/submit():
  // no admission, no ready queue, no per-edge atomics. Fills the same KvmResult (no per-task trace; final versions from the plan).
  struct StaticOptions { int workers = 0; double comm_us = 1.5; const char* trace_csv = nullptr; bool spin_idle = false, partition = true, fuse_phases = false; const std::vector<KvmFuseRun>* fused = nullptr; };
  const KvmResult& run_static(const KvmBundle& b, const StaticOptions& o) {
    if (started_ || finished_) throw std::logic_error("run_static replaces start()");
    std::string why = b.validate(nbufs(), cfg_.num_vars, 0);
    if (!why.empty()) { res_.err_code = -1; last_error_ = why; throw std::runtime_error("bundle rejected: " + why); }
    prepare();   // buffers and the host mirrors are in place before anything launches
    KvmSConfig sc; sc.workers = o.workers ? std::min(o.workers, grid_) : grid_; sc.comm_us = o.comm_us; sc.partition = o.partition;
    const KvmSOpLookup op_of = [](int id) { const KvmOpInfo* i = kvm_op(id); KvmSOpInfo v; if (i) { v.arity = i->arity; v.types = i->types; kvm_static_gang_args(i->sig, v); } return v; };
    const auto t0 = std::chrono::steady_clock::now();
    const KvmSCostFn cost_fn = kvm_static_cost;
    const KvmSPlan plan = kvm_static_plan(b.tasks, b.succ, b.args, sc, op_of, cost_fn);
    const KvmSLayout layout = kvm_static_layout(plan);
    int nrec = 0, used = 0;
    std::vector<int> offs(sc.workers + 1, 0);
    for (int w = 0; w < sc.workers; ++w) { offs[w] = nrec; nrec += layout.nrecs[w]; used += layout.nrecs[w] > 0; }
    offs[sc.workers] = nrec;
    std::vector<KvmSRec> host(nrec);
    for (int w = 0; w < sc.workers; ++w) if (layout.nrecs[w]) kvm_static_emit(plan, layout, w, b.tasks, b.args, bufs_.data(), nbufs(), op_of, host.data() + offs[w]);
    KLOG(2, "static schedule: %d tasks -> %d records on %d/%d workers, %d of %d edges cross workers, %d records publish; runs partitioned: %s; plan %.1f us, work %.1f us (compiled in %.3f s)", plan.nt, nrec, used, sc.workers, plan.cross_edges, plan.edges, plan.publishing, plan.partitioned ? "yes" : "no", plan.makespan_us, plan.total_work_us,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    FusedState fused;
    if (o.fused && !o.fused->empty()) fused = make_fused_state(*o.fused);   // the buffers' device pointers exist since prepare()
    const size_t static_smem = std::max(dyn_smem_, (size_t)fused.smem_bytes);
    long long* dphases = nullptr;
    if (o.fuse_phases && fused.desc) { const size_t bytes = (size_t)sc.workers * KVM_FMLP_NPHASES * sizeof(long long); CK(cudaMalloc(&dphases, bytes)); CK(cudaMemset(dphases, 0, bytes)); }
    KvmStaticDev S{};
    S.fuse_phases = dphases;
    KvmSRec* drecs; int* doffs; int* dprog; KvmCounters* dcnt; unsigned long long* dtrace = nullptr; int* dfinished;
    CK(cudaMalloc(&drecs, (size_t)nrec * sizeof(KvmSRec))); CK(cudaMemcpy(drecs, host.data(), (size_t)nrec * sizeof(KvmSRec), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&doffs, offs.size() * sizeof(int))); CK(cudaMemcpy(doffs, offs.data(), offs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dprog, (size_t)sc.workers * KVM_S_LINE_WORDS * sizeof(int))); CK(cudaMemset(dprog, 0, (size_t)sc.workers * KVM_S_LINE_WORDS * sizeof(int)));
    CK(cudaMalloc(&dcnt, sizeof(KvmCounters))); CK(cudaMemset(dcnt, 0, sizeof(KvmCounters)));
    if (o.trace_csv) { CK(cudaMalloc(&dtrace, (size_t)nrec * KVM_S_TRACE_WORDS * 8)); CK(cudaMemset(dtrace, 0, (size_t)nrec * KVM_S_TRACE_WORDS * 8)); }
    CK(cudaMalloc(&dfinished, sizeof(int))); CK(cudaMemset(dfinished, 0, sizeof(int)));
    S.recs = drecs; S.offs = doffs; S.progress = dprog; S.nworkers = sc.workers; S.nactive = used; S.spin_idle = o.spin_idle; S.finished = dfinished; S.timeout_ns = cfg_.timeout_us * 1000ull; S.cnt = dcnt; S.herr = S_.herr; S.fused = fused.desc; S.split = fused.split; S.trace = dtrace; S.trace_on = dtrace != nullptr;
    CK(cudaStreamSynchronize(0));
    if (static_smem > 0) CK(cudaFuncSetAttribute(kvm_static_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)static_smem));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kvm_static_kernel, KVM_THREADS, static_smem));
    if (occ < 1) throw std::runtime_error("static kernel does not fit on an SM");
    KLOG(2, "static kernel: %d resident blocks/SM, %zu bytes of dynamic shared memory, launching %d workers", occ, static_smem, sc.workers);
    // Warm-up launch with nworkers = 0: every block returns at once, but the union kernel's code is uploaded and started outside the timed window (like a graph captured
    // before timing): the very first launch of this kernel costs ~1.5 ms before any record runs.
    { KvmStaticDev warm = S; warm.nworkers = 0; kvm_static_kernel<<<1, KVM_THREADS, static_smem, stream_>>>(warm); CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream_)); }
    clock_warmup();
    CK(cudaEventRecord(e0_, stream_));
    kvm_static_kernel<<<sc.workers, KVM_THREADS, static_smem, stream_>>>(S);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1_, stream_));
    started_ = true;
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
    KvmCounters c; CK(cudaMemcpy(&c, dcnt, sizeof c, cudaMemcpyDeviceToHost));
    res_.err_code = c.err_code; res_.err_slot = c.err_slot; res_.err_a = c.err_a; res_.err_b = c.err_b;
    res_.admitted = plan.nt; res_.done = c.err_code ? 0 : plan.nt; res_.grid = sc.workers;
    CK(cudaEventElapsedTime(&res_.kernel_ms, e0_, e1_));
    res_.version.assign(cfg_.num_vars, 0);
    CK(cudaMemcpy(res_.version.data(), S_.version, cfg_.num_vars * 4, cudaMemcpyDeviceToHost));
    for (const KvmTaskDesc& d : b.tasks) if (d.pub_var >= 0) res_.version[d.pub_var] = std::max(res_.version[d.pub_var], d.pub_ver);   // versions are ordering only here: the plan's final values
    finished_ = true;
    if (o.trace_csv) {
      std::vector<unsigned long long> tr((size_t)nrec * KVM_S_TRACE_WORDS);
      CK(cudaMemcpy(tr.data(), dtrace, tr.size() * 8, cudaMemcpyDeviceToHost));
      FILE* f = fopen(o.trace_csv, "w");
      if (!f) throw std::runtime_error(std::string("cannot open ") + o.trace_csv);
      fprintf(f, "worker,pc,task,op,start_ns,ready_ns,done_ns,published_ns,nwait,flags\n");
      unsigned long long t00 = ~0ull; for (size_t i = 0; i < tr.size(); i += KVM_S_TRACE_WORDS) if (tr[i]) t00 = std::min(t00, tr[i]);
      for (int w = 0; w < sc.workers; ++w) for (int k = offs[w]; k < offs[w + 1]; ++k) {
        const KvmSRec& r = host[k]; const unsigned long long* x = tr.data() + (size_t)k * KVM_S_TRACE_WORDS;
        fprintf(f, "%d,%d,%d,%s,%llu,%llu,%llu,%llu,%d,%d\n", w, k - offs[w], r.task, (r.flags & KVM_S_WAITONLY) ? "(wait)" : kvm_op(r.op)->name, x[0] - t00, x[1] - t00, x[2] - t00, x[3] - t00, r.nwait, r.flags);
      }
      fclose(f); KLOG(2, "static trace: %d records -> %s", nrec, o.trace_csv);
    }
    CK(cudaFree(drecs)); CK(cudaFree(doffs)); CK(cudaFree(dprog)); CK(cudaFree(dcnt)); CK(cudaFree(dfinished)); if (dtrace) CK(cudaFree(dtrace));
    for (void* p : fused.allocs) CK(cudaFree(p));
    if (dphases) {   // per-phase cycles of the slice bodies, summed over the workers that ran them, per slice task
      static const char* names[KVM_FMLP_NPHASES] = {"load weights+ids", "embedding gather", "forward GEMMs (+gelu)", "head GEMM", "softmax CE", "head backward GEMMs", "dZ gelu' (+sync)", "bias colsum", "gelu_f again (l>0)", "weight gradient GEMM", "activation gradient GEMM"};
      std::vector<long long> tot((size_t)sc.workers * KVM_FMLP_NPHASES);
      CK(cudaMemcpy(tot.data(), dphases, tot.size() * sizeof(long long), cudaMemcpyDeviceToHost)); CK(cudaFree(dphases));
      long long nslice = 0; for (const KvmTaskDesc& d : b.tasks) nslice += d.op_id == kvm_op_by_name("fmlp_slice");
      long long total = 0;
      for (int i = 0; i < KVM_FMLP_NPHASES; ++i) { long long sum = 0; for (int w = 0; w < sc.workers; ++w) sum += tot[(size_t)w * KVM_FMLP_NPHASES + i]; total += sum; if (sum && names[i]) KLOG(2, "fuse phase %-34s %8.0f cycles per slice", names[i], (double)sum / nslice); }
      KLOG(2, "fuse phases: %.0f cycles per slice in total, over %lld slices", (double)total / nslice, nslice);
    }
    if (res_.ok()) KLOG(2, "static run done: %d tasks in %.3f ms on %d workers", plan.nt, res_.kernel_ms, sc.workers);
    else KLOG(0, "DEVICE ERROR %s (%s, code %d): task %d, a=%d, b=%d", kvm_err_name(res_.err_code), kvm_err_text(res_.err_code), res_.err_code, res_.err_slot, res_.err_a, res_.err_b);
    return res_;
  }

  // No more bundles: the kernel exits once everything admitted has finished.
  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_flag_) return;
    closed_flag_ = true;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(closed_, 1u, __ATOMIC_RELEASE);
  }

  const KvmResult& finish() {
    if (!started_) throw std::logic_error("finish before start()");
    if (finished_) return res_;
    close();
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
    KvmCounters c; CK(cudaMemcpy(&c, S_.cnt, sizeof c, cudaMemcpyDeviceToHost));
    res_.err_code = c.err_code; res_.err_slot = c.err_slot; res_.err_a = c.err_a; res_.err_b = c.err_b;
    res_.admitted = c.admitted; res_.done = c.done; res_.grid = grid_;
    CK(cudaEventElapsedTime(&res_.kernel_ms, e0_, e1_));
    const int ns = std::min(c.admitted, cfg_.max_tasks);
    res_.tasks.resize(ns); res_.trace.resize(ns); res_.version.resize(cfg_.num_vars);
    if (ns) {
      CK(cudaMemcpy(res_.tasks.data(), S_.tasks, ns * sizeof(KvmTaskDesc), cudaMemcpyDeviceToHost));
      if (cfg_.trace_on) CK(cudaMemcpy(res_.trace.data(), S_.trace, ns * sizeof(KvmTrace), cudaMemcpyDeviceToHost));   // (else the zeroed vector stands)
    }
    CK(cudaMemcpy(res_.version.data(), S_.version, cfg_.num_vars * 4, cudaMemcpyDeviceToHost));
    if (cfg_.sched.on) {
      CK(cudaMemcpy(&res_.sched.dev, S_.sp.dev, sizeof(KvmSchedDev), cudaMemcpyDeviceToHost)); res_.sched.valid = true;
      res_.info.resize(ns);
      if (ns) CK(cudaMemcpy(res_.info.data(), S_.sp.info, ns * sizeof(KvmSlotInfo), cudaMemcpyDeviceToHost));
    }
    finished_ = true;
    if (res_.ok()) KLOG(2, "kernel done: %d/%d tasks in %.3f ms, %d records, %d ring wraps, %d backpressure waits", res_.done, res_.admitted, res_.kernel_ms, stats_.bundles, stats_.pads, stats_.backpressure_waits);
    else KLOG(0, "DEVICE ERROR %s (%s, code %d): slot %d, a=%d, b=%d; done %d/%d", kvm_err_name(res_.err_code), kvm_err_text(res_.err_code), res_.err_code, res_.err_slot, res_.err_a, res_.err_b, res_.done, res_.admitted);
    return res_;
  }

  // Raw 4-byte elements as floats (an i32 buffer's bits: reinterpret); a bf16 buffer widened exactly to fp32.
  std::vector<float> read_buffer(int idx) const {
    if (idx < 0 || idx >= nbufs()) throw std::out_of_range("buffer index");
    std::vector<float> v((size_t)bufs_[idx].rows * bufs_[idx].cols);
    if (bufs_[idx].dtype == 2) {   // bf16 storage: widened exactly to fp32
      std::vector<uint16_t> h(v.size());
      CK(cudaMemcpy(h.data(), bufs_[idx].p, h.size() * 2, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < h.size(); ++i) { uint32_t u = (uint32_t)h[i] << 16; memcpy(&v[i], &u, 4); }
      return v;
    }
    CK(cudaMemcpy(v.data(), bufs_[idx].p, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
  }
  int buffer_dtype(int idx) const { return bufs_.at(idx).dtype; }
  int sms() const { return sms_; }
  const std::string& device_name() const { return name_; }

 private:
  // The FIFO ring only where it fits (fifo_fits); refused, the run takes the 32 bucket rings. The request stays in cfg_.fifo_only: it is the
  // caller's contract (priority 0 and no deadlines, which submit() enforces), the same on every grid whichever ring serves it.
  void set_fifo(bool want) {
    cfg_.fifo_only = want;
    S_.fifo_only = want && fifo_fits();
    if (want && !S_.fifo_only)
      KLOG(1, "ready queue: the FIFO ring does not fit (%d workers, live_cap %d, max_tasks %d; it needs workers < live_cap and max_tasks <= %d): using the 32 bucket rings",
           workers(), cfg_.live_cap, cfg_.max_tasks, 1 << KVM_FIFO_SHIFT);
  }
  void validate_sched(const KvmSchedParams& p) const {
    for (int c = 0; c < nc_; ++c) {
      const KvmClassParams& q = p.cls[c];
      if (q.weight < 1 || q.min_cta < 0 || q.max_cta < 0 || q.bw_cta < 0 || q.urgent_us < 0) throw std::invalid_argument("class " + std::to_string(c) + ": weight >= 1, min/max/bw caps and urgent_us >= 0");
      if (q.max_cta < q.min_cta) throw std::invalid_argument("class " + std::to_string(c) + ": max_cta < min_cta");
    }
    if (p.bw_limit_min < 0 || p.bw_limit_max < p.bw_limit_min) throw std::invalid_argument("bw_limit_min <= bw_limit_max");
  }
  void init_sched() {
    const KvmSchedCfg& sc = cfg_.sched;
    hp_ = (KvmSchedParams*)(ctl_ + 512); ht_ = (KvmSchedTele*)(ctl_ + 1024);
    S_.sp = KvmSchedPtrs{}; S_.sp.nc = nc_; S_.sp.on = sc.on ? 1 : 0; S_.sp.mode = sc.on ? sc.mode : KVM_SCHED_GLOBAL;
    if (!sc.on) return;
    validate_sched(sc.p);
    void* d;
    CK(cudaHostGetDevicePointer(&d, hp_, 0)); S_.sp.hp = (KvmSchedParams*)d;
    CK(cudaHostGetDevicePointer(&d, ht_, 0)); S_.sp.ht = (KvmSchedTele*)d;
    CK(cudaMalloc(&S_.sp.dev, sizeof(KvmSchedDev))); CK(cudaMemset(S_.sp.dev, 0, sizeof(KvmSchedDev)));
    CK(cudaMalloc(&S_.sp.info, (size_t)cfg_.max_tasks * sizeof(KvmSlotInfo))); CK(cudaMemset(S_.sp.info, 0, (size_t)cfg_.max_tasks * sizeof(KvmSlotInfo)));
    CK(cudaMalloc(&S_.sp.pv, (size_t)cfg_.num_vars * KVM_MAX_CLS * 4)); CK(cudaMemset(S_.sp.pv, 0, (size_t)cfg_.num_vars * KVM_MAX_CLS * 4));
    CK(cudaMalloc(&S_.sp.pubcls, (size_t)cfg_.num_vars * 4)); CK(cudaMemset(S_.sp.pubcls, 0, (size_t)cfg_.num_vars * 4));
    KvmController::write_params(hp_, sc.p);   // generation 1: the policy block's first tick re-reads the launch parameters
  }
  static uint16_t bf16_rne(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x40u : 0u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
  // dtype 2 (bf16): fill / values are rounded to bf16 on the host; a file holds raw little-endian bf16 (n * 2 bytes).
  void init_buffer_bf16(void* p, size_t n, const KvmInit& in) {
    switch (in.kind) {
      case KvmInit::ZERO: CK(cudaMemset(p, 0, n * 2)); break;
      case KvmInit::FILL: {
        std::vector<uint16_t> v(std::min<size_t>(n, 1 << 20), bf16_rne(in.value));
        for (size_t o = 0; o < n; o += v.size()) CK(cudaMemcpy((uint16_t*)p + o, v.data(), std::min(v.size(), n - o) * 2, cudaMemcpyHostToDevice));
        break;
      }
      case KvmInit::VALUES: {
        if (in.values.size() != n) throw std::invalid_argument("buffer init size");
        std::vector<uint16_t> v(n); for (size_t i = 0; i < n; ++i) v[i] = bf16_rne(in.values[i]);
        CK(cudaMemcpy(p, v.data(), n * 2, cudaMemcpyHostToDevice)); break;
      }
      case KvmInit::FILE: {
        FILE* f = fopen(in.path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open init file " + in.path);
        std::vector<uint16_t> v(n);
        size_t got = fread(v.data(), 2, n, f); fclose(f);
        if (got != n) throw std::runtime_error(in.path + ": expected " + std::to_string(n * 2) + " bytes");
        CK(cudaMemcpy(p, v.data(), n * 2, cudaMemcpyHostToDevice)); break;
      }
      case KvmInit::SEEDED: throw std::invalid_argument("seeded init needs an f32 buffer");
    }
  }
  void init_buffer(float* p, size_t n, const KvmInit& in, int dtype) {
    if (dtype == 2) { init_buffer_bf16(p, n, in); return; }
    switch (in.kind) {
      case KvmInit::ZERO: CK(cudaMemset(p, 0, n * 4)); break;
      case KvmInit::FILL: {
        if (dtype == 1) { std::vector<int> v(std::min<size_t>(n, 1 << 20), (int)in.value); for (size_t o = 0; o < n; o += v.size()) CK(cudaMemcpy((int*)p + o, v.data(), std::min(v.size(), n - o) * 4, cudaMemcpyHostToDevice)); }
        else { std::vector<float> v(std::min<size_t>(n, 1 << 20), in.value); for (size_t o = 0; o < n; o += v.size()) CK(cudaMemcpy(p + o, v.data(), std::min(v.size(), n - o) * 4, cudaMemcpyHostToDevice)); }
        break;
      }
      case KvmInit::VALUES:
        if (in.values.size() != n) throw std::invalid_argument("buffer init size");
        CK(cudaMemcpy(p, in.values.data(), n * 4, cudaMemcpyHostToDevice)); break;
      case KvmInit::SEEDED:
        if (dtype != 0) throw std::invalid_argument("seeded init needs an f32 buffer");
        kvm_seeded_fill_kernel<<<(unsigned)std::min<size_t>((n + 255) / 256, 4096), 256>>>(p, n, in.seed, in.draws, in.bias, in.coef);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); break;
      case KvmInit::FILE: {   // streamed through one 8 MiB staging buffer
        FILE* f = fopen(in.path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open init file " + in.path);
        fseek(f, 0, SEEK_END); long long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if ((unsigned long long)sz != n * 4) { fclose(f); throw std::runtime_error(in.path + ": " + std::to_string(sz) + " bytes, expected " + std::to_string(n * 4)); }
        std::vector<char> stage(8u << 20);
        for (size_t off = 0; off < n * 4;) {
          size_t got = fread(stage.data(), 1, std::min(stage.size(), n * 4 - off), f);
          if (got == 0) { fclose(f); throw std::runtime_error(in.path + ": short read"); }
          CK(cudaMemcpy((char*)p + off, stage.data(), got, cudaMemcpyHostToDevice)); off += got;
        }
        fclose(f); break;
      }
    }
  }
  void write_header(size_t total, int nt, int ne, int na, int kind) {
    int* h = ring_ + wpos_ % cfg_.ring_words;
    h[0] = KVM_REC_MAGIC; h[1] = (int)total; h[2] = nt; h[3] = ne; h[4] = na; h[5] = stats_.bundles; h[6] = kind; h[7] = 0;
  }
  KvmConfig cfg_;
  KvmDevState S_{};
  std::vector<KvmBuf> bufs_;
  std::vector<size_t> views_;
  std::vector<float*> arenas_;
  char* ctl_ = nullptr; int* ring_ = nullptr; volatile unsigned* win_done_ = nullptr;
  unsigned long long *doorbell_ = nullptr, *ring_head_ = nullptr; unsigned* closed_ = nullptr; KvmHostErr* herr_ = nullptr;
  unsigned long long wpos_ = 0;
  cudaStream_t stream_{}; cudaEvent_t e0_{}, e1_{};
  int sms_ = 0, grid_ = 0, nc_ = 1; size_t dyn_smem_ = 0; bool hopper_ = false;
  KvmSchedParams* hp_ = nullptr; KvmSchedTele* ht_ = nullptr;
  std::string name_, last_error_;
  std::mutex mu_;
  bool prestart_ = false;   // prestart(): the ring may be written before the kernel runs
  bool started_ = false, finished_ = false, closed_flag_ = false, prepared_ = false;
  KvmResult res_;
  KvmStats stats_;
  std::vector<int> rec_tasks_;
};

// ---------------- admission front end: chunked, class-aware submission ----------------
// The device admits records strictly in ring order, so a latency-sensitive record submitted behind a big one waits for all of it. The front
// end keeps the ring SHALLOW and decides what enters it next: a submitted bundle is cut at its connected components into chunks of about
// `chunk_tasks` tasks (a component is never split, so every edge stays inside one record; version waits already work across records),
// chunks wait in one FIFO per class, and a pump thread moves the next chunk into the ring whenever less than `ring_budget_tasks` worth of
// records is still in the ring. The next chunk is the earliest-deadline one (task deadline_us counted from submission); with no deadline
// pending, the class that has been served the fewest tasks per unit of weight. A late urgent bundle therefore waits for at most the
// ring budget (about ring_budget_tasks of admission work), not for the backlog. No SM partition is involved: the classes only share the ring.
struct KvmFrontCfg { int chunk_tasks = 2048, ring_budget_tasks = 4096, classes = 8; std::vector<int> weights; };
class KvmFront {
 public:
  using Cfg = KvmFrontCfg;
  using Ticket = unsigned long long;
  explicit KvmFront(KvmExec& ex, Cfg cfg = Cfg()) : ex_(ex), cfg_(cfg) {
    if (cfg_.chunk_tasks < 1 || cfg_.ring_budget_tasks < cfg_.chunk_tasks || cfg_.classes < 1) throw std::invalid_argument("KvmFront: chunk_tasks >= 1, ring_budget_tasks >= chunk_tasks, classes >= 1");
    cfg_.weights.resize(cfg_.classes, 1);
    q_.resize(cfg_.classes); served_.assign(cfg_.classes, 0.0);
    pump_ = std::thread([this] { loop(); });
  }
  ~KvmFront() { stop(); }
  void stop() {
    { std::lock_guard<std::mutex> lk(mu_); if (stop_) return; stop_ = true; }
    cv_.notify_all(); if (pump_.joinable()) pump_.join();
  }
  // Queue a bundle of class `cls`; returns at once. The ticket completes when every chunk's window is done.
  Ticket submit(const KvmBundle& b, int cls) {
    if (cls < 0 || cls >= cfg_.classes) throw std::invalid_argument("KvmFront: class out of range");
    std::vector<KvmBundle> parts = split(b, cfg_.chunk_tasks);
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu_);
    const Ticket t = next_++;
    tickets_[t].pending = (int)parts.size();
    for (KvmBundle& p : parts) {
      int dl = -1; for (const KvmTaskDesc& d : p.tasks) if (d.deadline_us >= 0) dl = dl < 0 ? d.deadline_us : std::min(dl, d.deadline_us);
      Chunk c; c.ticket = t; c.tasks = (int)p.tasks.size(); c.deadline = dl >= 0 ? now + std::chrono::microseconds(dl) : std::chrono::steady_clock::time_point::max(); c.b = std::move(p);
      q_[cls].push_back(std::move(c));
    }
    cv_.notify_all();
    return t;
  }
  // True once every task of the ticket's bundle has completed.
  bool done(Ticket t) {
    std::vector<unsigned> w;
    { std::lock_guard<std::mutex> lk(mu_); auto it = tickets_.find(t); if (it == tickets_.end()) return true; if (it->second.pending > 0) return false; w = it->second.wids; }
    for (unsigned x : w) if (!ex_.window_done(x)) return false;
    return true;
  }
  bool wait(Ticket t, int timeout_ms) {
    auto t0 = std::chrono::steady_clock::now();
    while (!done(t)) { if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) return false; std::this_thread::sleep_for(std::chrono::microseconds(20)); }
    return true;
  }
  // Wait until every queued chunk has entered the ring (then ex.close() is safe).
  bool flush(int timeout_ms) {
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      { std::lock_guard<std::mutex> lk(mu_); bool empty = true; for (auto& q : q_) empty = empty && q.empty(); if (empty && !inflight_) return true; }
      if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
  int chunks_submitted() const { return submitted_.load(); }

  // Cut a bundle at its connected components into chunks of about `limit` tasks (components stay whole; an oversized one is its own chunk).
  static std::vector<KvmBundle> split(const KvmBundle& b, int limit) {
    const int nt = (int)b.tasks.size();
    if (nt <= limit) return {b};
    std::vector<int> uf(nt); for (int i = 0; i < nt; ++i) uf[i] = i;
    std::function<int(int)> find = [&](int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
    for (int i = 0; i < nt; ++i) for (int k = 0; k < b.tasks[i].succ_cnt; ++k) { int v = b.succ[b.tasks[i].succ_off + k]; if (v >= 0 && v < nt) uf[find(i)] = find(v); }
    std::vector<int> comp_of(nt), comp_size, comp_first;   // components numbered by their first task
    std::map<int, int> id;
    for (int i = 0; i < nt; ++i) { int r = find(i); auto it = id.find(r); if (it == id.end()) { it = id.emplace(r, (int)comp_size.size()).first; comp_size.push_back(0); comp_first.push_back(i); } comp_of[i] = it->second; ++comp_size[it->second]; }
    std::vector<int> chunk_of_comp(comp_size.size()); int chunk = 0, fill = 0;
    for (size_t c = 0; c < comp_size.size(); ++c) { if (fill > 0 && fill + comp_size[c] > limit) { ++chunk; fill = 0; } chunk_of_comp[c] = chunk; fill += comp_size[c]; }
    std::vector<KvmBundle> out(chunk + 1); std::vector<int> newidx(nt);
    for (int i = 0; i < nt; ++i) { KvmBundle& o = out[chunk_of_comp[comp_of[i]]]; newidx[i] = (int)o.tasks.size(); o.tasks.push_back(b.tasks[i]); }
    for (int i = 0; i < nt; ++i) {
      KvmBundle& o = out[chunk_of_comp[comp_of[i]]]; KvmTaskDesc& d = o.tasks[newidx[i]];
      const KvmTaskDesc& s = b.tasks[i];
      d.succ_off = (int)o.succ.size(); for (int k = 0; k < s.succ_cnt; ++k) o.succ.push_back(newidx[b.succ[s.succ_off + k]]);
      d.args_off = (int)o.args.size(); for (int k = 0; k < s.args_words; ++k) o.args.push_back(b.args[s.args_off + k]);
    }
    return out;
  }

 private:
  struct Chunk { Ticket ticket = 0; int tasks = 0; std::chrono::steady_clock::time_point deadline; KvmBundle b; };
  struct TicketState { int pending = 0; std::vector<unsigned> wids; };
  // Next chunk: the earliest absolute deadline among the queue heads, else the class with the least weighted service. Caller holds mu_.
  int pick() const {
    int best = -1;
    for (int c = 0; c < cfg_.classes; ++c) if (!q_[c].empty() && q_[c].front().deadline != std::chrono::steady_clock::time_point::max() &&
                                              (best < 0 || q_[c].front().deadline < q_[best].front().deadline)) best = c;
    if (best >= 0) return best;
    for (int c = 0; c < cfg_.classes; ++c) if (!q_[c].empty() && (best < 0 || served_[c] / cfg_.weights[c] < served_[best] / cfg_.weights[best])) best = c;
    return best;
  }
  void loop() {
    const size_t budget_words = (size_t)cfg_.ring_budget_tasks * KVM_TASK_WORDS;
    for (;;) {
      Chunk c; int cls;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { if (stop_) return true; for (auto& q : q_) if (!q.empty()) return true; return false; });
        if (stop_) return;
        cls = pick(); c = std::move(q_[cls].front()); q_[cls].pop_front(); inflight_ = true;
      }
      // keep the ring shallow: wait until what is in it fits the budget (a record bigger than the budget goes in when the ring is empty)
      while (ex_.ring_backlog_words() > 0 && ex_.ring_backlog_words() + c.b.words() > budget_words) {
        { std::lock_guard<std::mutex> lk(mu_); if (stop_) return; }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
        // a more urgent chunk may have arrived while this one waited: swap it in (the queue head is re-picked, this chunk goes back to its class's front)
        std::lock_guard<std::mutex> lk(mu_);
        int better = pick();
        if (better >= 0 && q_[better].front().deadline < c.deadline) { q_[cls].push_front(std::move(c)); cls = better; c = std::move(q_[cls].front()); q_[cls].pop_front(); }
      }
      unsigned wid = 0; KvmSubmit st = ex_.submit(c.b, 60000, true, &wid);
      std::lock_guard<std::mutex> lk(mu_);
      if (st == KvmSubmit::OK) { auto& t = tickets_[c.ticket]; t.wids.push_back(wid); --t.pending; served_[cls] += c.tasks; ++submitted_; }
      else { KLOG(0, "KvmFront: chunk submit failed (%s): dropping the ticket's remaining chunks", kvm_submit_name(st)); tickets_[c.ticket].pending = 1 << 30; }
      inflight_ = false;
    }
  }
  KvmExec& ex_; Cfg cfg_;
  std::mutex mu_; std::condition_variable cv_;
  std::vector<std::deque<Chunk>> q_; std::vector<double> served_;
  std::map<Ticket, TicketState> tickets_; Ticket next_ = 1;
  bool stop_ = false, inflight_ = false; std::atomic<int> submitted_{0};
  std::thread pump_;
};

// Timeline-ready CSV: times in microseconds since the first admission.
static void kvm_write_trace_csv(const KvmResult& r, const char* path) {
  FILE* f = fopen(path, "w");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  unsigned long long t0 = ~0ull;
  for (auto& t : r.trace) if (t.admit_ns) t0 = std::min(t0, t.admit_ns);
  fprintf(f, "slot,op,workload,priority,deadline_us,admit_us,ready_us,start_us,end_us,queue_wait_us,run_us,smid,block,ver_start,ver_end,missed,exec_count,bytes,res,resv,boost,urgent,cont"
#ifdef KVM_PROF_GAP
    ",gp0,gp1,gp2,gp3,gp4,gp5,gp6,gp7,gp8,gp_valid,gp_cycle,sub_succ_id,sub_atomic,sub_enqueue,sub_desc,sub_post,sub_errv"
#endif
    "\n");
  for (size_t i = 0; i < r.trace.size(); ++i) {
    const KvmTrace& t = r.trace[i]; const KvmTaskDesc& d = r.tasks[i];
    auto us = [&](unsigned long long x) { return x ? (double)(x - t0) / 1e3 : -1.0; };
    const KvmSlotInfo si = i < r.info.size() ? r.info[i] : KvmSlotInfo{0, 0, 0};
    fprintf(f, "%zu,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%llu,%d,%d,%d,%d,%d", i, kvm_op(d.op_id)->name, d.workload_id, d.priority, d.deadline_us,
            us(t.admit_ns), us(t.ready_ns), us(t.start_ns), us(t.end_ns), t.start_ns ? (double)(t.start_ns - t.ready_ns) / 1e3 : -1.0,
            t.end_ns ? (double)(t.end_ns - t.start_ns) / 1e3 : -1.0, t.smid, t.block, t.ver0, t.ver1, (t.flags & KVM_TRACE_MISSED) ? 1 : 0, t.exec_count,
            si.bytes, si.res, (t.flags & KVM_TRACE_RESV) ? 1 : 0, (t.flags & KVM_TRACE_BOOST) ? 1 : 0, (t.flags & KVM_TRACE_URGENT) ? 1 : 0, (t.flags & KVM_TRACE_CONT) ? 1 : 0);
#ifdef KVM_PROF_GAP
    for (int k = 0; k < 9; ++k) fprintf(f, ",%d", t.gp[k]);
    fprintf(f, ",%d,%u", t.gp[15], (unsigned)t.gp[16]);
    for (int k = 0; k < KVM_PG_SUBS; ++k) fprintf(f, ",%d", t.gp[9 + k]);
#endif
    fputc('\n', f);
  }
  fclose(f);
  KLOG(2, "trace: %zu tasks -> %s", r.trace.size(), path);
}

// Per-workload telemetry from the trace: tasks, CTA-seconds, queue-wait percentiles, deadline misses, bytes moved and bandwidth.
static void kvm_print_class_report(const KvmResult& r) {
  if (r.trace.empty() || !r.trace[0].admit_ns) return;   // a timing run keeps no per-task trace
  int maxc = 0; for (auto& d : r.tasks) maxc = std::max(maxc, d.workload_id);
  unsigned long long t0 = ~0ull, t1 = 0; for (auto& t : r.trace) if (t.admit_ns) { t0 = std::min(t0, t.admit_ns); t1 = std::max(t1, t.end_ns); }
  const double span_us = t1 > t0 ? (t1 - t0) / 1e3 : 1.0;
  for (int c = 0; c <= maxc; ++c) {
    std::vector<double> w; double cta_us = 0, bytes = 0; int n = 0, miss = 0;
    for (size_t i = 0; i < r.trace.size(); ++i) if (r.tasks[i].workload_id == c) {
      const KvmTrace& t = r.trace[i]; ++n; w.push_back((t.start_ns - t.ready_ns) / 1e3); cta_us += (t.end_ns - t.start_ns) / 1e3;
      miss += (t.flags & KVM_TRACE_MISSED) != 0; if (i < r.info.size()) bytes += (double)r.info[i].bytes;
    }
    if (!n) continue;
    std::sort(w.begin(), w.end());
    auto q = [&](double p) { return w[std::min(w.size() - 1, (size_t)(p * w.size()))]; };
    KLOG(2, "workload %d: %d tasks, %.3f CTA-s, queue wait p50 %.1f p99 %.1f max %.1f us, %d deadline misses, %.2f GB moved (%.1f GB/s over the run)",
         c, n, cta_us / 1e6, q(0.5), q(0.99), w.back(), miss, bytes / 1e9, bytes / (span_us * 1e3));
  }
}

// ======================= CLI =======================
#ifndef KVM_EXEC_NO_MAIN
// Minimal JSON reader (objects, arrays, numbers, strings, true/false/null): enough for bundle metadata.json.
struct Json {
  enum T { Z, B, N, S, A, O } t = Z;
  double n = 0; std::string s; std::vector<Json> a; std::vector<std::pair<std::string, Json>> o;
  const Json* get(const char* k) const { for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
  const Json& need(const char* k) const { const Json* j = get(k); if (!j) throw std::runtime_error(std::string("metadata/golden: missing key '") + k + "'"); return *j; }
  double num(const char* k, double dflt) const { auto* j = get(k); return j && j->t == N ? j->n : dflt; }
};
struct JsonParser {
  const std::string& s; size_t i = 0;
  explicit JsonParser(const std::string& str) : s(str) {}
  void ws() { while (i < s.size() && isspace((unsigned char)s[i])) ++i; }
  [[noreturn]] void bad(const char* m) { throw std::runtime_error(std::string("json: ") + m + " at byte " + std::to_string(i)); }
  Json parse() { Json j = value(); ws(); if (i != s.size()) bad("trailing data"); return j; }
  int depth = 0;
  Json value() {
    ws(); if (i >= s.size()) bad("eof");
    struct D { int& d; explicit D(int& x) : d(x) { ++d; } ~D() { --d; } } guard(depth);
    if (depth > 64) bad("nesting too deep");
    Json j; char c = s[i];
    if (c == '{') { j.t = Json::O; ++i; ws(); if (s[i] == '}') { ++i; return j; }
      for (;;) { ws(); std::string k = str(); ws(); if (s[i++] != ':') bad("expected ':'"); j.o.emplace_back(k, value()); ws();
        if (s[i] == ',') { ++i; continue; } if (s[i] == '}') { ++i; return j; } bad("expected ',' or '}'"); } }
    if (c == '[') { j.t = Json::A; ++i; ws(); if (s[i] == ']') { ++i; return j; }
      for (;;) { j.a.push_back(value()); ws(); if (s[i] == ',') { ++i; continue; } if (s[i] == ']') { ++i; return j; } bad("expected ',' or ']'"); } }
    if (c == '"') { j.t = Json::S; j.s = str(); return j; }
    if (!s.compare(i, 4, "true")) { i += 4; j.t = Json::B; j.n = 1; return j; }
    if (!s.compare(i, 5, "false")) { i += 5; j.t = Json::B; return j; }
    if (!s.compare(i, 4, "null")) { i += 4; return j; }
    char* e; j.n = strtod(s.c_str() + i, &e); if (e == s.c_str() + i) bad("bad value");
    j.s = s.substr(i, e - (s.c_str() + i));   // raw text: 64-bit seeds do not survive a double
    i = e - s.c_str(); j.t = Json::N; return j;
  }
  std::string str() {
    if (s[i] != '"') bad("expected string");
    std::string r; ++i;
    while (i < s.size() && s[i] != '"') { if (s[i] == '\\') { ++i; char c = s[i]; r += c == 'n' ? '\n' : c == 't' ? '\t' : c; } else r += s[i]; ++i; }
    if (i >= s.size()) bad("unterminated string");
    ++i; return r;
  }
};
static std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error("cannot read " + p);
  std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
template <class T> static std::vector<T> read_bin(const std::string& p, size_t count) {
  std::string raw = read_file(p);
  if (raw.size() != count * sizeof(T)) throw std::runtime_error(p + ": expected " + std::to_string(count * sizeof(T)) + " bytes, got " + std::to_string(raw.size()));
  std::vector<T> v(count); if (count) memcpy(v.data(), raw.data(), raw.size()); return v;
}

// A buffer's `init` object from bundle metadata.json (kinds: fill, values, seeded, file; file paths are relative to `dir`).
static KvmInit kvm_parse_init(const Json& jb, const std::string& dir, int dtype) {
  KvmInit init;
  const Json* in = jb.get("init");
  if (!in) return init;
  const Json* kind = in->get("kind"); const std::string k = kind ? kind->s : "";
  if (k == "fill") { init.kind = KvmInit::FILL; init.value = (float)in->num("value", 0); }
  else if (k == "values") { init.kind = KvmInit::VALUES; for (auto& v : in->need("values").a) init.values.push_back(dtype == 1 ? [&] { int iv = (int)v.n; float f; memcpy(&f, &iv, 4); return f; }() : (float)v.n); }
  else if (k == "seeded") {
    init.kind = KvmInit::SEEDED; init.seed = in->get("seed") ? strtoull(in->get("seed")->s.c_str(), nullptr, 10) : 0;
    init.draws = (int)in->num("draws", 1); init.bias = (float)in->num("bias", 0); init.coef = (float)in->num("coef", 1);
  }
  else if (k == "file") { init.kind = KvmInit::FILE; init.path = dir + "/" + in->need("path").s; }
  else throw std::runtime_error(dir + ": unknown buffer init kind '" + k + "'");
  return init;
}

// Scheduler options shared by the CLIs (kvm_exec, doom_exec). Returns 1 if argv[i] was a scheduler option (i advanced past its value), 0 if not, -1 on a missing value.
static int kvm_cli_sched_option(KvmConfig& cfg, int& i, int argc, char** argv) {
  const std::string a = argv[i];
  auto val = [&](long long lo, long long hi) -> long long {
    if (i + 1 >= argc) return lo - 1;
    char* e; errno = 0; long long v = strtoll(argv[++i], &e, 10);
    if (e == argv[i] || *e || errno || v < lo || v > hi) { fprintf(stderr, "%s: integer in [%lld,%lld] expected, got '%s'\n", a.c_str(), lo, hi, argv[i]); exit(2); }
    return v; };
  auto list = [&](auto set) { if (i + 1 >= argc) return false; std::stringstream ss(argv[++i]); std::string tok; int c = 0; while (std::getline(ss, tok, ',') && c < KVM_MAX_CLS) set(cfg.sched.p.cls[c++], atoi(tok.c_str())); return true; };
  if (a == "--sched") { if (i + 1 >= argc) return -1; std::string m = argv[++i]; cfg.sched.on = true; cfg.sched.mode = m == "global" ? KVM_SCHED_GLOBAL : m == "static" ? KVM_SCHED_STATIC : KVM_SCHED_DYNAMIC; }
  else if (a == "--classes") { long long v = val(1, KVM_MAX_CLS); if (v < 1) return -1; cfg.sched.num_classes = (int)v; }
  else if (a == "--weights") { if (!list([](KvmClassParams& q, int v) { q.weight = v; })) return -1; }
  else if (a == "--min") { if (!list([](KvmClassParams& q, int v) { q.min_cta = v; })) return -1; }
  else if (a == "--max") { if (!list([](KvmClassParams& q, int v) { q.max_cta = v; })) return -1; }
  else if (a == "--bw-cta") { if (!list([](KvmClassParams& q, int v) { q.bw_cta = v; })) return -1; }
  else if (a == "--urgent-us") { if (!list([](KvmClassParams& q, int v) { q.urgent_us = v; })) return -1; }
  else if (a == "--op-affinity") { if (!list([](KvmClassParams& q, int v) { if (v) q.protect |= KVM_CF_OPAFF; })) return -1; }   // e.g. 1,0: class 0 prefers the block that last ran the same op
  else if (a == "--shares") {   // STATIC mode: SM share per class, e.g. 1,3
    if (i + 1 >= argc) return -1;
    std::stringstream ss(argv[++i]); std::string tok; std::vector<double> sh; while (std::getline(ss, tok, ',')) sh.push_back(atof(tok.c_str()));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); kvm_static_split(cfg.sched.p, sh, pr.multiProcessorCount);
  }
  else if (a == "--arbiter") cfg.sched.p.flags |= KVM_SF_ARBITER;
  else if (a == "--bw-limit") { long long v = val(0, 1 << 24); if (v < 0) return -1; cfg.sched.p.bw_limit_init = (int)v; }
  else if (a == "--affinity-us") { long long v = val(0, 1000000); if (v < 0) return -1; cfg.sched.p.affinity_us = (int)v; }
  else if (a == "--no-guarantee") cfg.sched.p.flags &= ~KVM_SF_GUARANTEE;
  else if (a == "--no-inherit") cfg.sched.p.flags &= ~KVM_SF_INHERIT;
  else return 0;
  return 1;
}

static void usage() {
  fprintf(stderr, "usage: kvm_exec [options] bundle_dir...\n"
    "  bundle_dir: tasks.bin successors.bin args.bin metadata.json (ABI v2)\n"
    "  --blocks N --live-cap N --max-tasks N --ring-words N --aging-us N --timeout-s N (stall timeout) --task-limit-s N\n"
    "  --sched MODE        global|static|dynamic: per-workload scheduling (workload_id = class, up to 8)  --classes N\n"
    "  --weights a,b,..    --min a,b,.. (reserved CTAs)  --max a,b,.. (CTA caps)  --bw-cta a,b,..  --urgent-us a,b,..\n"
    "  --op-affinity a,b (1 = class prefers the block that last ran the same op)  --shares a,b (static SM shares)  --arbiter  --bw-limit N   adaptive concurrent bandwidth-bound CTA limit   --affinity-us N   --no-guarantee  --no-inherit\n"

    "  --golden [--golden-tol R]  compare with <bundle_dir>/golden.json (single bundle): final buffers, versions, observed versions\n"
    "  --trace-csv FILE   per-task timeline  --dump-buffers   print buffers   --quiet   errors only   --check-ops   print registry\n"
    "  --warm-ms N   keep the GPU busy N ms right before the timed launch (the SM clock runs low under sparse work; short timed runs inherit it)\n"
    "  --full-smem   launch with the build's maximum dynamic shared memory instead of the bundles' largest task's (less L1 for every op)\n"
    "  --no-dissolve      keep the noops that only order tasks (default: spliced out of a plain single-bundle run on the dynamic path)\n"
    "  --no-tape-sort     admit the tape as the bundle wrote it (default on the dynamic path: dependency order, which starts every replicated run at once)\n"
    "  --dump-bin DIR i,j,..  write the listed buffers (by metadata index, widened to little-endian f32 / raw i32) to DIR/<index>.bin after the run\n"
    "  --static           (default for one bundle) run the bundle as per-worker programs compiled at load: no ready queue, no admission  --workers N  --comm-us X (scheduler's cross-worker cost)  --no-partition (independent runs share all workers)  --no-fuse (no fused MLP steps; by default a run whose steps match is fused on the static path, --dump-buffers included)\n"
    "  --dynamic          use the admission ring and ready queue even for one bundle\n");
}

int main(int argc, char** argv) {
  KvmConfig cfg; std::vector<std::string> dirs; const char* csv = nullptr; bool dump = false, golden = false, dissolve = true, tape_sort = true; float golden_tol = 1e-6f;
  std::string dump_dir; std::vector<int> dump_idx;
  bool sized = false;   // the caller gave a sizing flag: keep its live_cap/ring/slots
  bool no_fifo = false;  // --no-fifo: keep the 32-bucket ready rings even for a bundle set that qualifies for the FIFO ring
  bool opt_static = false, opt_dynamic = false, static_spin = false, static_partition = true, opt_fuse = true, fuse_phases = false, full_smem = false; int static_workers = 0; double static_comm = 1.5;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&](long long lo = 0, long long hi = 1ll << 30) -> long long {   // strict integer in [lo, hi]
      if (i + 1 >= argc) { usage(); exit(2); }
      char* e; errno = 0; long long v = strtoll(argv[++i], &e, 10);
      if (e == argv[i] || *e || errno || v < lo || v > hi) { fprintf(stderr, "%s: integer in [%lld,%lld] expected, got '%s'\n", a.c_str(), lo, hi, argv[i]); exit(2); }
      return v; };
    if (a == "--blocks") cfg.blocks = (int)val(0, 1 << 20);
    else if (a == "--live-cap") { cfg.live_cap = (int)val(2, 1 << 24); sized = true; }
    else if (a == "--max-tasks") { cfg.max_tasks = (int)val(2, 1 << 26); sized = true; }
    else if (a == "--ring-words") { cfg.ring_words = (int)val(64, 1 << 28); sized = true; }
    else if (a == "--task-limit-s") cfg.task_limit_us = (unsigned long long)val(0, 86400) * 1000000ull;
    else if (a == "--warm-ms") cfg.warm_ms = (int)val(0, 60000);
    else if (a == "--full-smem") full_smem = true;
    else if (a == "--aging-us") cfg.aging_us = (int)val(0, 1000000000);
    else if (a == "--no-cont") cfg.cont = 0;
#ifdef KVM_PROF_GAP
    else if (a == "--noop-ops") cfg.noop_ops = 1;
#endif
    else if (a == "--no-fifo") no_fifo = true;
    else if (a == "--timeout-s") cfg.timeout_us = (unsigned long long)val(1, 86400) * 1000000ull;
    else if (int r = kvm_cli_sched_option(cfg, i, argc, argv)) { if (r < 0) { usage(); return 2; } }
    else if (a == "--trace-csv") { if (i + 1 >= argc) { usage(); return 2; } csv = argv[++i]; }
    else if (a == "--static") opt_static = true;
    else if (a == "--dynamic") opt_dynamic = true;
    else if (a == "--workers") static_workers = (int)val(1, 1 << 20);
    else if (a == "--static-spin") static_spin = true;
    else if (a == "--no-partition") static_partition = false;
    else if (a == "--fuse") opt_fuse = true;
    else if (a == "--no-fuse") opt_fuse = false;
    else if (a == "--fuse-phases") fuse_phases = true;
    else if (a == "--comm-us") { if (i + 1 >= argc) { usage(); return 2; } static_comm = atof(argv[++i]); }
    else if (a == "--dump-buffers") dump = true;
    else if (a == "--dump-bin") {
      if (i + 2 >= argc) { usage(); return 2; }
      dump_dir = argv[++i];
      for (char* p = argv[++i]; *p;) { char* e; long v = strtol(p, &e, 10); if (e == p || v < 0) { usage(); return 2; } dump_idx.push_back((int)v); p = *e == ',' ? e + 1 : e; }
    }
    else if (a == "--golden") golden = true;
    else if (a == "--golden-tol") { if (i + 1 >= argc) { usage(); return 2; } golden_tol = (float)atof(argv[++i]); }
    else if (a == "--quiet") g_kvm_log = 0;
    else if (a == "--no-dissolve") dissolve = false;
    else if (a == "--no-tape-sort") tape_sort = false;
    else if (a == "--check-ops") { for (auto& o : kKvmOps) printf("%2d %-9s arity %d  (%s)\n", o.id, o.name, o.arity, o.sig); return 0; }
    else if (a[0] == '-') { usage(); return 2; }
    else dirs.push_back(a);
  }
  if (dirs.empty()) { usage(); return 2; }
  try {
    // Window sizing: the default live_cap / ring cut a large bundle into windows, and the host submits a window only after the windows holding its producers
    // completed (a drain per window). When no sizing flag was given and a single bundle would be cut by any capacity (its tasks, or its words: a 9,800-task
    // bundle of 4 replicated runs ran its runs one after another in 10 windows, x4 slower), size them so the bundle is one record.
    if (!sized && dirs.size() == 1) {
      Json m = JsonParser(read_file(dirs[0] + "/metadata.json")).parse();
      const long long nt = (long long)m.num("task_count", -1), ne = (long long)m.num("edge_count", -1), nw = (long long)m.num("args_words", -1);
      const long long words = (KVM_REC_HDR + nt * KVM_TASK_WORDS + ne + nw + 7) & ~7ll;   // KvmBundle::words() of the bundle once read
      auto p2 = [](long long v) { long long r = 2; while (r < v) r <<= 1; return r; };
      if (ne >= 0 && nw >= 0 && (!cfg.fits_one_record(nt, words) || ne + 2 > (long long)cfg.succ_cap) && nt <= (1 << 20) && words < (1ll << 26)) {
        cfg.live_cap = (int)std::max<long long>(cfg.live_cap, p2(nt)); cfg.max_tasks = std::max(cfg.max_tasks, cfg.live_cap);
        cfg.ring_words = (int)std::max<long long>(cfg.ring_words, p2(2 * words + 128));
        cfg.succ_cap = std::max<unsigned>(cfg.succ_cap, (unsigned)p2(ne + 2));
        KLOG(2, "window sizing: %lld tasks, %lld words as one record (live_cap %d, ring %d words, successor ring %u words)", nt, words, cfg.live_cap, cfg.ring_words, cfg.succ_cap);
      }
    }
    KvmExec ex(cfg);
    std::vector<KvmBundle> bundles;
    std::vector<std::pair<int, int>> shapes;
    std::vector<std::string> buf_names, var_names; std::map<int, std::string> var_by_id;
    std::map<int, int> version_base;   // per variable: its base version (metadata), the static path's feasibility check
    Json meta0;
    for (auto& dir : dirs) {
      Json m = JsonParser(read_file(dir + "/metadata.json")).parse();
      if ((int)m.num("abi_version", -1) != KVM_ABI_VERSION) throw std::runtime_error(dir + ": abi_version " + std::to_string((int)m.num("abi_version", -1)) + " != " + std::to_string(KVM_ABI_VERSION));
      int nt = (int)m.num("task_count", -1), ne = (int)m.num("edge_count", -1), nw = (int)m.num("args_words", -1);
      if (nt < 1 || ne < 0 || nw < 0) throw std::runtime_error(dir + ": task_count/edge_count/args_words missing");
      KvmBundle b;
      b.tasks = read_bin<KvmTaskDesc>(dir + "/tasks.bin", nt);
      b.succ = read_bin<int32_t>(dir + "/successors.bin", ne);
      b.args = read_bin<int32_t>(dir + "/args.bin", nw);
      if (dirs.size() == 1) meta0 = m;
      if (auto* bs = m.get("buffers")) {
        for (auto& jb : bs->a) {
          if (const Json* nm = jb.get("name")) buf_names.push_back(nm->s);
          int idx = (int)jb.num("index", -1), rows = (int)jb.num("rows", 0), cols = (int)jb.num("cols", 0);
          if (idx < 0 || rows < 1 || cols < 1) throw std::runtime_error(dir + ": bad buffer entry");
          if (idx < (int)shapes.size()) { if (shapes[idx] != std::make_pair(rows, cols)) throw std::runtime_error(dir + ": buffer " + std::to_string(idx) + " shape differs from an earlier bundle"); continue; }
          if (idx != (int)shapes.size()) throw std::runtime_error(dir + ": buffer indices must be dense and ascending");
          const int dtype = (int)jb.num("dtype", 0);
          KvmInit init = kvm_parse_init(jb, dir, dtype);
          if (ex.add_buffer(rows, cols, init, dtype) != idx) throw std::logic_error("buffer index mismatch");
          shapes.push_back({rows, cols});
        }
      }
      if (auto* vs = m.get("variables")) for (auto& jv : vs->a) { int id = (int)jv.num("id", -1), base = (int)jv.num("base_version", 0); ex.set_version(id, base); version_base[id] = base; }
      KLOG(2, "%s: %d tasks, %d edges, %d arg words", dir.c_str(), nt, ne, nw);
      bundles.push_back(std::move(b));
    }
    // The static schedule is the default for one closed bundle (a compiled DAG, no streaming, no scheduling classes): the ring path remains for several bundles, the per-class scheduler, or an explicit --dynamic.
    if (opt_static && opt_dynamic) throw std::runtime_error("--static and --dynamic exclude each other");
    // A bundle can run on the static path only if everything it waits for is ordered inside it: a wait on a version no task of the bundle publishes
    // (and its base version does not cover) needs the dynamic path's parking and an external publisher.
    auto static_feasible = [&](const KvmBundle& b) {
      for (const KvmTaskDesc& d : b.tasks) {   // every buffer a task names must fit its record (the planner would refuse otherwise)
        const KvmOpInfo* o = kvm_op(d.op_id);
        if (!o || d.args_words > KVM_S_ARGS) return false;
        std::set<int> seen;
        for (int j = 0; j < d.args_words; ++j) { const int ty = (int)((o->types >> (3 * j)) & 7u); if ((ty == KVM_A_BUF || ty == KVM_A_BUF_OPT) && b.args[d.args_off + j] != -1) seen.insert(b.args[d.args_off + j]); }
        if (seen.size() > KVM_S_BUFS) return false;
      }
      std::map<int, int> newest;
      for (const KvmTaskDesc& d : b.tasks) if (d.pub_var >= 0) newest[d.pub_var] = std::max(newest[d.pub_var], d.pub_ver);
      for (const KvmTaskDesc& d : b.tasks) if (d.wait_var >= 0) {
        auto it = newest.find(d.wait_var);
        if (it == newest.end() ? d.wait_ver > version_base[d.wait_var] : d.wait_ver > it->second) return false;
      }
      return true;
    };
    // The static path pays where tasks are short. A precompiled order saves the dynamic path's per-task scheduling (claim, admission, successor counts: about 1-2 us of an SM per task) and
    // gives up its ready queue's balancing, which a plan from the cost model cannot replace when tasks are long and the model is far off (measured: gpt-162m's ~80 us tiles ran x0.70 static).
    // Decided from the bundle alone: the planner's own mean estimated cost per task against KVM_STATIC_MAX_MEAN_US, the crossover measured on bundles outside the panel.
    auto static_mean_us = [&](const KvmBundle& b) {
      double work = 0.0;
      for (const KvmTaskDesc& d : b.tasks) work += kvm_static_cost(d.op_id, b.args.data() + d.args_off);
      return b.tasks.empty() ? 0.0 : work / (double)b.tasks.size();
    };
    // The rule decides only what its cost model prices: an op outside kvm_static_costs() would enter the mean at a flat guess no measurement backs (a bf16 GEMM of
    // ~130 us priced at 4), so such a bundle runs dynamic, the path that needs no cost model and the one every single bundle took before the static path existed.
    auto unpriced_ops = [&](const KvmBundle& b) {
      std::set<std::string> names;
      for (const KvmTaskDesc& d : b.tasks) {
        const KvmOpInfo* o = kvm_op(d.op_id);
        if (!o) names.insert("op " + std::to_string(d.op_id));
        else if (!kvm_static_costs().count(o->name)) names.insert(o->name);
      }
      std::string list;
      for (const std::string& n : names) list += (list.empty() ? "" : ", ") + n;
      return list;
    };
    // A gang op's members meet at barriers inside the op, so they must run at once: only the static plan places them so (kvm_static_plan).
    auto gang_task = [&](const KvmBundle& b) {
      for (size_t t = 0; t < b.tasks.size(); ++t) {
        const KvmOpInfo* o = kvm_op(b.tasks[t].op_id);
        KvmSOpInfo g;
        if (o) kvm_static_gang_args(o->sig, g);
        if (g.gang_arg >= 0) return (int)t;
      }
      return -1;
    };
    if (!opt_dynamic && !opt_static && bundles.size() == 1 && !cfg.sched.on && static_feasible(bundles[0])) {
      const std::string unpriced = unpriced_ops(bundles[0]);
      if (gang_task(bundles[0]) >= 0) {
        opt_static = true;
        KLOG(2, "path: static (the bundle has gang ops, which only the static plan runs)");
      } else if (!unpriced.empty()) {
        KLOG(2, "path: dynamic (the static cost model does not price %s)", unpriced.c_str());
      } else {
        const double mean = static_mean_us(bundles[0]);
        opt_static = mean <= KVM_STATIC_MAX_MEAN_US;
        KLOG(2, "path: %s (mean estimated task %.2f us %s %.1f)", opt_static ? "static" : "dynamic", mean, opt_static ? "<=" : ">", (double)KVM_STATIC_MAX_MEAN_US);
      }
    }
    if (opt_static && !static_feasible(bundles[0])) throw std::runtime_error("--static: a task waits for a version no task of this bundle publishes");
    if (!full_smem) ex.size_launch(bundles);   // every task of the run is known here, before any launch
    const KvmResult* resp = nullptr;
    KvmFuseResult fused;            // outside the branch: the golden check below skips what the fused runs no longer write
    if (opt_static) {
      if (bundles.size() != 1) throw std::runtime_error("--static takes exactly one bundle directory");
      KvmExec::StaticOptions so; so.workers = static_workers ? static_workers : cfg.blocks; so.comm_us = static_comm; so.trace_csv = csv; csv = nullptr; so.spin_idle = static_spin; so.partition = static_partition; so.fuse_phases = fuse_phases; so.fuse_phases = fuse_phases;
      if (opt_fuse) {
        // the pass sizes slices and owners for the workers the schedule will have (run_static's own count: --workers, at most the grid; the grid when unset). Measured: given 0, it took the
        // widest slice that fits, so a single run chose 32-row slices and the same run inside a merged bundle 16, and their rounding differed.
        const int workers = so.workers ? std::min(so.workers, ex.grid()) : ex.grid();
        fused = ex.fuse_bundle(bundles[0], workers, dump_idx);   // --dump-buffers keeps the fused path: a gate must read the path that is timed; the buffers no fused run writes dump as they started
        if (fused.fused_runs) { bundles[0].tasks = fused.tasks; bundles[0].succ = fused.succ; bundles[0].args = fused.args; so.fused = &fused.runs; }
      }
      resp = &ex.run_static(bundles[0], so);
    } else {
      for (const KvmBundle& b : bundles) {
        const int t = gang_task(b);
        if (t >= 0) throw std::runtime_error("task " + std::to_string(t) + ": " + kvm_op(b.tasks[t].op_id)->name + " is a gang op, whose members must run at once; only the static path plans that (one bundle, no --dynamic, no scheduling classes)");
      }
      if (dissolve && !golden && !csv && dirs.size() == 1 && !cfg.sched.on) {   // the dynamic path only (the static planner and the fused-step matcher read the bundle as written); a trace's slots are the bundle's task indices, so a traced run keeps its noops
        const int nd = kvm_dissolve_noops(bundles[0]);
        if (nd) KLOG(2, "%d noops dissolved: %zu tasks, %zu edges", nd, bundles[0].tasks.size(), bundles[0].succ.size());
      }
      if (tape_sort && !golden && !csv && bundles.size() == 1 && kvm_sort_tape(bundles[0])) KLOG(2, "tape in dependency order");
      bool one_record = false;
      {
        bool fifo = !no_fifo && !cfg.sched.on, no_waits = true;
        for (const KvmBundle& b : bundles) for (const KvmTaskDesc& t : b.tasks) { if (t.priority != 0 || t.deadline_us != -1) fifo = false; if (t.wait_var >= 0) no_waits = false; }
        one_record = bundles.size() == 1 && cfg.fits_one_record((long long)bundles[0].tasks.size(), (long long)bundles[0].words());
        const bool trace_on = csv != nullptr || golden || getenv("KVM_TRACE") != nullptr;
        ex.set_run_facts(fifo, one_record, no_waits, trace_on);
        KLOG(2, "ready queue: %s; slots %s; version waits %s; task trace %s", ex.fifo_only() ? "one FIFO ring, ticket waits" : "32 priority buckets", one_record ? "not recycled (one record)" : "recycled", no_waits ? "none" : "present", (trace_on || cfg.sched.on) ? "on" : "off");
      }
      if (one_record) {                   // the whole input is one record: written and validated before the kernel starts, so no host work sits inside the kernel's timed window
        ex.prestart();
        const KvmSubmit st = ex.submit(bundles[0]);
        if (st != KvmSubmit::OK) { KLOG(0, "bundle 0: %s (%s)", kvm_submit_name(st), ex.last_error().c_str()); return 1; }
        ex.close();
        ex.prestage();
        ex.start();
      } else {
        ex.start();
        for (size_t i = 0; i < bundles.size(); ++i) {
          const KvmBundle& bi = bundles[i];
          KvmSubmit st = (bi.tasks.size() > (size_t)cfg.live_cap || bi.words() > (size_t)cfg.ring_words / 2) ? ex.submit_stream(bi) : ex.submit(bi);
          if (st != KvmSubmit::OK) { KLOG(0, "bundle %zu: %s (%s)", i, kvm_submit_name(st), ex.last_error().c_str()); ex.close(); ex.finish(); return 1; }
        }
      }
      resp = &ex.finish();
    }
    const KvmResult& r = *resp;
    if (r.ok() && !opt_static) kvm_print_class_report(r);
    if (csv) kvm_write_trace_csv(r, csv);
    int golden_bad = 0;
    if (golden) {
      if (dirs.size() != 1) throw std::runtime_error("--golden needs exactly one bundle directory");
      Json g = JsonParser(read_file(dirs[0] + "/golden.json")).parse();
      if (!r.ok()) { KLOG(0, "golden: skipped, the run failed"); return 1; }
      int nb = 0, exact = 0, golden_interior = 0; float worst = 0;
      if (const Json* gb = g.get("buffers")) for (auto& kv : gb->o) {
        auto it = std::find(buf_names.begin(), buf_names.end(), kv.first);
        if (it == buf_names.end()) { KLOG(0, "golden: unknown buffer %s", kv.first.c_str()); ++golden_bad; continue; }
        if (fused.interior.count((int)(it - buf_names.begin()))) { ++golden_interior; continue; }   // an activation or gradient a fused step keeps in shared memory: the run's numerics are the weights and losses
        std::vector<float> got = ex.read_buffer((int)(it - buf_names.begin()));
        if (got.size() != kv.second.a.size()) { KLOG(0, "golden: buffer %s has %zu values, golden %zu", kv.first.c_str(), got.size(), kv.second.a.size()); ++golden_bad; continue; }
        const bool is_int = ex.buffer_dtype((int)(it - buf_names.begin())) == 1;
        double num = 0, den = 0; size_t ndiff = 0;
        for (size_t i = 0; i < got.size(); ++i) {
          float want = (float)kv.second.a[i].n; ++nb;
          if (is_int) { int iv = (int)kv.second.a[i].n; memcpy(&want, &iv, 4); }
          if (memcmp(&got[i], &want, 4) == 0) { ++exact; den += is_int ? 0 : (double)want * want; continue; }
          ++ndiff;
          if (is_int) { if (ndiff <= 3) KLOG(0, "golden: %s[%zu] differs (i32)", kv.first.c_str(), i); ++golden_bad; continue; }
          num += ((double)got[i] - want) * ((double)got[i] - want); den += (double)want * want;
        }
        // f32 buffers: relative L2 error of the whole buffer (elementwise relative error is meaningless near zero)
        float rel = (float)(std::sqrt(num) / std::max(1e-30, std::sqrt(den))); worst = std::max(worst, rel);
        if (rel > golden_tol) { KLOG(0, "golden: buffer %s: relative L2 error %.3g > %.3g (%zu of %zu elements differ)", kv.first.c_str(), rel, golden_tol, ndiff, got.size()); ++golden_bad; }
      }
      KLOG(2, "golden: %d buffer values compared, %d bitwise equal, %d buffers interior to fused steps not compared, worst buffer relative L2 error %.3g (tolerance %.3g)", nb, exact, golden_interior, worst, golden_tol);
      std::map<std::string, int> var_id;
      if (const Json* vs = meta0.get("variables")) for (auto& jv : vs->a) var_id[jv.need("name").s] = (int)jv.num("id", -1);
      if (const Json* gv = g.get("versions")) for (auto& kv : gv->o) {
        int want = (int)kv.second.n, got = var_id.count(kv.first) ? r.version[var_id[kv.first]] : -1;
        if (got != want) { KLOG(0, "golden: version %s = %d, golden %d", kv.first.c_str(), got, want); ++golden_bad; }
      }
      int nreads = 0;
      if (const Json* rs = meta0.get("reads")) if (!opt_static) for (auto& jr : rs->a) {   // the per-read version checks need the per-task trace (the dynamic path's)
        const Json* gate = jr.get("gate");
        int slot = (int)jr.num("task", -1); if (gate && gate->t == Json::N) slot = (int)gate->n;  // the task carrying the wait
        const Json* pol = jr.get("policy"); int need = (int)jr.num("version", 0);
        if (slot < 0 || slot >= (int)r.trace.size() || !pol) throw std::runtime_error("metadata: malformed reads[] entry (task index out of range or no policy)");
        const KvmTrace& t = r.trace[slot]; ++nreads;
        bool ok = t.ver0 >= need;
        if (pol && pol->s == "exact") ok = ok && t.ver0 == need && t.ver1 == need;
        KLOG(3, "golden: read %s (%s v%d): observed ver_start %d ver_end %d -> %s", jr.need("name").s.c_str(), pol->s.c_str(), need, t.ver0, t.ver1, ok ? "ok" : "VIOLATION");
        golden_bad += !ok;
      }
      KLOG(2, "golden: %d reads checked, final versions compared; %d mismatches", nreads, golden_bad);
    }
    if (dump) for (int i = 0; i < ex.nbufs(); ++i) { auto v = ex.read_buffer(i); printf("buffer %d:", i); for (float x : v) printf(" %.9g", x); printf("\n"); }
    for (int idx : dump_idx) {   // raw dump of selected buffers: the text dump above is ~11 bytes per element, too slow for a 162M-parameter model
      if (idx >= ex.nbufs()) { fprintf(stderr, "--dump-bin: buffer %d out of range (%d buffers)\n", idx, ex.nbufs()); return 2; }
      auto v = ex.read_buffer(idx);
      std::string path = dump_dir + "/" + std::to_string(idx) + ".bin";
      FILE* f = fopen(path.c_str(), "wb");
      if (!f || fwrite(v.data(), sizeof(float), v.size(), f) != v.size()) { fprintf(stderr, "--dump-bin: cannot write %s\n", path.c_str()); return 2; }
      fclose(f);
    }
    bool pass = r.ok() && golden_bad == 0;
    printf("%s: %d/%d tasks, %.3f ms%s\n", pass ? "OK" : "FAIL", r.done, r.admitted, r.kernel_ms, golden ? (golden_bad ? ", golden MISMATCH" : ", golden match") : "");
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    KLOG(0, "%s", e.what());
    return 1;
  }
}
#endif
