// Magni VM GPU executor, host side: bundle building/validation, the admission ring producer, launch, result decoding, and the
// CLI (compiled out with -DMVM_EXEC_NO_MAIN so tests/gpu can include this file).
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude mvm_exec.cu -o mvm_exec
//   MVM_HOST=... ./run_exec_tests.sh        # tests on a GPU box
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
#include "mvm_exec.cuh"
#include "mvm_window_size.h"   // the record count and the successor ring's floor, host only
#include "mvm_static.h"
#include "mvm_fuse.h"
#include "mvm_fuse_gpt.h"
#include "mvm_fgpt_kernel.cuh"
#include "mvm_f2d_kernel.cuh"

// ---------------- logging (levels 0 error .. 3 debug; everything is shown by default) ----------------
static int g_mvm_log = 3;
static void mvm_log(int lvl, const char* fmt, ...) {
  if (lvl > g_mvm_log) return;
  static const char* nm[] = {"E", "W", "I", "D"};
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[mvm_exec][%s] ", nm[lvl]); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
  va_end(ap);
}
#define KLOG(lvl, ...) mvm_log(lvl, __VA_ARGS__)
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  char m_[512]; snprintf(m_, sizeof m_, "%s:%d: %s -> %s", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); throw std::runtime_error(m_); } } while (0)

static const char* mvm_dist_err(int c, bool text) {  // MVM_DIST_HOOK: codes 64.. come from dist/wire.hpp
  switch (c) {
#define E(n, code, str) case code: return text ? str : #n;
    MVM_DIST_ERRORS(E)
#undef E
  }
  return nullptr;
}
// The GPT pack's device errors (mvm_ops_gpt.cuh's one list), or nullptr when the code is not one of them.
static const char* mvm_gpt_err(int c, bool text) {
#ifdef MVM_GPT_ERR_BASE
  switch (c) {
#define MVM_GPT_ERR_CASE(n, off, str) case MVM_GPT_ERR_BASE + off: return text ? str : #n;
    MVM_GPT_ERRORS(MVM_GPT_ERR_CASE)
#undef MVM_GPT_ERR_CASE
  }
#else
  (void)c; (void)text;
#endif
  return nullptr;
}
static const char* mvm_err_name(int c) {
  if (const char* d = mvm_dist_err(c, false)) return d;
  if (const char* g = mvm_gpt_err(c, false)) return g;
  static const char* nm[] = {
#define E(n, s) #n,
      MVM_ERRORS(E)
#undef E
  };
  return c >= 0 && c < (int)(sizeof nm / sizeof nm[0]) ? nm[c] : "UNKNOWN";
}
static const char* mvm_err_text(int c) {
  if (const char* d = mvm_dist_err(c, true)) return d;
  if (const char* g = mvm_gpt_err(c, true)) return g;
  static const char* tx[] = {
#define E(n, s) s,
      MVM_ERRORS(E)
#undef E
  };
  return c >= 0 && c < (int)(sizeof tx / sizeof tx[0]) ? tx[c] : "unknown error code";
}
static int mvm_op_by_name(const std::string& n) {
  for (const MvmOpInfo& o : kMvmOps) if (n == o.name) return o.id;
  return -1;
}

// MFLOPs of one GEMM tile (tile extent clipped to the GEMM's edge); 0 for every other op. The work term of the static schedule compiler's cost.
static double mvm_tile_mflop(int op, const int32_t* a) {
  const MvmOpInfo* o = mvm_op(op);
  if (!o || strcmp(o->name, "matmul_tile")) return 0.0;
  const double m0 = (unsigned)a[6], n0 = (unsigned)a[7], M = (unsigned)a[8], N = (unsigned)a[9], K = (unsigned)a[10];
  return std::min(64.0, M - m0) * std::min(64.0, N - n0) * K * 2.0 / 1e6;
}

// Scheduler weights (microseconds, relative only) of the static schedule compiler: per-op means of the Nov 9 executor traces (gpt_tiny) with the
// dynamic path's fixed overhead taken out, plus the FLOPs of a GEMM tile at an SM's TF32 rate (mvm_static_cost). sgd_range and softmax_ce_fused sit
// at the flat 4 us an unlisted op used to get: the a100 preset's bundles carry them (the split SGD, the fused cross-entropy), and the path rule was
// read with them at that price. Any other op has no measured cost here, so the path rule decides only bundles whose every op is listed (main()).
static const std::map<std::string, double>& mvm_static_costs() {
  static const std::map<std::string, double> base = {{"noop", 0.1}, {"matmul_tile", 4.0}, {"adamw", 5.0}, {"gradnorm_partial", 2.0}, {"gradnorm_finish", 2.0}, {"colsum", 2.0},
    {"layernorm_fwd", 7.0}, {"layernorm_bwd_dx", 8.0}, {"norm_bwd_params", 5.0}, {"attn_fwd", 8.0}, {"attn_bwd_dq", 9.0}, {"attn_bwd_dkv", 10.0}, {"attn_bwd_delta", 6.0},
    {"rope_fwd", 4.0}, {"rope_bwd", 4.0}, {"embedding_fwd", 3.0}, {"embedding_bwd", 3.0}, {"softmax_ce_fwd", 3.0}, {"softmax_ce_bwd", 3.0}, {"softmax_ce_reduce", 2.0}, {"gelu_bwd", 3.0}, {"gelu_fwd", 3.0}, {"select_window", 2.0}, {"counter_inc", 1.0}, {"fmlp_slice", 60.0}, {"fmlp_sgd", 8.0}, {"fmlp_split", 40.0}, {"fmlp_grid", 60.0}, {"fmlp_batch", 80.0},
    {"sgd_range", 4.0}, {"softmax_ce_fused", 4.0}};
  return base;
}
static double mvm_static_cost(int op, const int32_t* a) {
  const std::map<std::string, double>& base = mvm_static_costs();
  const MvmOpInfo* o = mvm_op(op);
  auto it = o ? base.find(o->name) : base.end();
  double us = it != base.end() ? it->second : 4.0;
  return us + mvm_tile_mflop(op, a) / 1.4;
}

// A bundle whose ops the cost model all prices and whose mean estimated task cost (mvm_static_cost) is at most this runs on the static path by default; above it, on the dynamic path (main()). The crossover measured on
// frontend GPTs outside the panel (Nov 11, A100-SXM4-80GB, 24 single bundles with means from 4.54 to 10.67 us, both paths on one binary, weights bitwise between them):
// static faster up to 6.67 us (x1.015-1.119), dynamic faster from 7.04 us (x1.08-1.63). 6.85 is the middle of that gap, where no bundle read takes its slower path.
#ifndef MVM_STATIC_MAX_MEAN_US
#define MVM_STATIC_MAX_MEAN_US 6.85
#endif

// ---------------- bundles ----------------
struct MvmArg {
  int32_t w; int type;
  static MvmArg buf(int idx) { return {idx, MVM_A_BUF}; }
  static MvmArg none() { return {-1, MVM_A_BUF}; }   // absent optional buffer (a `buf?` argument)
  static MvmArg i32(int v) { return {v, MVM_A_I32}; }
  static MvmArg f32(float v) { int32_t w; memcpy(&w, &v, 4); return {w, MVM_A_F32}; }
};
struct MvmTaskOpts { int priority = 0, deadline_us = -1, wait_var = -1, wait_ver = 0, pub_var = -1, pub_ver = 0, workload = 0; };

struct MvmBundle {
  std::vector<MvmTaskDesc> tasks;
  std::vector<int32_t> succ, args;
  // "" if well formed, else the first problem. nbufs/nvars < 0 skip those range checks.
  std::string validate(int nbufs, int nvars, int live_cap) const {
    std::ostringstream e;
    const int nt = (int)tasks.size();
    if (nt == 0) return "empty bundle";
    if (live_cap > 0 && nt > live_cap) { e << nt << " tasks > live_cap " << live_cap; return e.str(); }
    std::vector<int> in(nt, 0);
    for (int i = 0; i < nt; ++i) {
      const MvmTaskDesc& t = tasks[i];
      const MvmOpInfo* op = mvm_op(t.op_id);
      if (!op) { e << "task " << i << ": bad op_id " << t.op_id; return e.str(); }
      if (t.args_words != op->arity) { e << "task " << i << ": " << op->name << " takes " << op->arity << " args, got " << t.args_words; return e.str(); }
      if (t.args_off < 0 || (long long)t.args_off + t.args_words > (long long)args.size()) { e << "task " << i << ": args range"; return e.str(); }
      if (t.succ_off < 0 || t.succ_cnt < 0 || (long long)t.succ_off + t.succ_cnt > (long long)succ.size()) { e << "task " << i << ": successor range"; return e.str(); }
      if (t.priority < 0 || t.priority >= MVM_NBUCKETS) { e << "task " << i << ": priority " << t.priority << " outside [0," << MVM_NBUCKETS - 1 << "]"; return e.str(); }
      if (t.deadline_us < -1) { e << "task " << i << ": deadline_us"; return e.str(); }
      for (int v : {t.wait_var, t.pub_var}) if (v < -1 || (nvars >= 0 && v >= nvars)) { e << "task " << i << ": version variable " << v << " out of range"; return e.str(); }
      if ((t.wait_var >= 0 && t.wait_ver < 0) || (t.pub_var >= 0 && t.pub_ver < 1)) { e << "task " << i << ": wait_ver >= 0 and pub_ver >= 1 (0 is the initial state)"; return e.str(); }
      for (int j = 0; j < t.args_words; ++j) {
        const int ty = (int)((op->types >> (3 * j)) & 7u);
        if (ty == MVM_A_BUF || ty == MVM_A_BUF_OPT) {
          int b = args[t.args_off + j];
          if ((b == -1 && ty == MVM_A_BUF) || b < -1 || (nbufs >= 0 && b >= nbufs)) { e << "task " << i << ": " << op->name << " arg " << j << " buffer index " << b << " out of range"; return e.str(); }
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
      const MvmTaskDesc& t = tasks[q[h]];
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
  size_t words() const { return (size_t)mvm_record_words((long long)tasks.size(), (long long)succ.size(), (long long)args.size(), MVM_REC_HDR, MVM_TASK_WORDS); }
};

// ---------------- dissolve noops that only order tasks (on a longest path) ----------------
// A noop has no work; its effect is the ordering through it and, when it publishes, a version. When no task of the bundle waits on a version, that version has no reader, so a noop is spliced out: every
// predecessor points straight at every successor, and the noop costs no task, level or dependency hop. A noop whose fan-in times fan-out exceeds max_fan stays (a join: splicing it would explode the edge count).
// Returns the number dissolved. Runs on a plain single-bundle run, before the first task is admitted.
static int mvm_dissolve_noops(MvmBundle& b, long long max_fan = 65536) {
  const int nt = (int)b.tasks.size();
  for (const MvmTaskDesc& t : b.tasks) if (t.wait_var >= 0) return 0;
  // Only a noop on a longest path (counted in tasks) costs the run a hop; anywhere else splicing it buys nothing and multiplies edges (a GPT step's barrier, 20-45 in by
  // 21-81 out, lies off the path: spliced, gpt-tiny and gpt-small lost 0.5-0.9%; the MLP rows' barrier lies on it and gained 4-9%).
  std::vector<int> depth(nt, 0), height(nt, 0), pending(nt), order;
  order.reserve(nt);
  for (int i = 0; i < nt; ++i) { pending[i] = b.tasks[i].indegree; if (pending[i] == 0) order.push_back(i); }
  for (size_t k = 0; k < order.size(); ++k) {
    const MvmTaskDesc& t = b.tasks[order[k]];
    for (int e = 0; e < t.succ_cnt; ++e) { const int s = b.succ[t.succ_off + e]; depth[s] = std::max(depth[s], depth[order[k]] + 1); if (--pending[s] == 0) order.push_back(s); }
  }
  if ((int)order.size() != nt) return 0;                      // not a DAG by its indegrees: leave the bundle to validation
  int longest = 0;
  for (int k = nt - 1; k >= 0; --k) {
    const int i = order[k]; const MvmTaskDesc& t = b.tasks[i];
    for (int e = 0; e < t.succ_cnt; ++e) height[i] = std::max(height[i], height[b.succ[t.succ_off + e]] + 1);
    longest = std::max(longest, depth[i] + height[i]);
  }
  std::vector<char> gone(nt, 0); int ngone = 0;
  for (int i = 0; i < nt; ++i) {
    const MvmTaskDesc& t = b.tasks[i];
    if (t.op_id == MVM_OP_noop && depth[i] + height[i] == longest && (long long)t.indegree * t.succ_cnt <= max_fan) { gone[i] = 1; ++ngone; }
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
    const MvmTaskDesc& t = b.tasks[i];
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
  MvmBundle out; out.tasks.reserve(m);
  std::vector<int> indeg(m, 0), stamp(nt, -1), stack;
  for (int i = 0; i < nt; ++i) {
    if (gone[i]) continue;
    MvmTaskDesc t = b.tasks[i];
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
static bool mvm_sort_tape(MvmBundle& b) {
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
  std::vector<MvmTaskDesc> tasks(n);
  std::vector<int32_t> succ; succ.reserve(b.succ.size());
  for (int k = 0; k < n; ++k) {
    MvmTaskDesc t = b.tasks[order[k]];
    const int off = (int)succ.size();
    for (int j = 0; j < t.succ_cnt; ++j) succ.push_back(pos[b.succ[t.succ_off + j]]);
    t.succ_off = off;
    tasks[k] = t;
  }
  b.tasks.swap(tasks); b.succ.swap(succ);
  return true;
}

class MvmBundleBuilder {
 public:
  int add(const char* op, std::vector<MvmArg> args, MvmTaskOpts o = {}) {
    int id = mvm_op_by_name(op);
    if (id < 0) throw std::invalid_argument(std::string("unknown operator ") + op);
    const MvmOpInfo* info = mvm_op(id);
    if ((int)args.size() != info->arity) throw std::invalid_argument(std::string(op) + ": wrong argument count");
    for (size_t j = 0; j < args.size(); ++j)
    {
      const int want = (int)((info->types >> (3 * j)) & 7u);
      const bool buf_ok = args[j].type == MVM_A_BUF && (want == MVM_A_BUF || want == MVM_A_BUF_OPT) && (args[j].w >= 0 || want == MVM_A_BUF_OPT);
      const bool type_ok = args[j].type == MVM_A_BUF ? buf_ok : args[j].type == want;
      if (!type_ok) throw std::invalid_argument(std::string(op) + ": argument " + std::to_string(j) + " has the wrong type (or an absent buffer for a required buf)");
    }
    MvmTaskDesc t{};
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
  MvmBundle build() const {
    MvmBundle b; b.tasks = tasks_; b.args = args_;
    for (size_t i = 0; i < tasks_.size(); ++i) {
      b.tasks[i].succ_off = (int)b.succ.size();
      b.tasks[i].succ_cnt = (int)out_[i].size();
      for (int s : out_[i]) { b.succ.push_back(s); ++b.tasks[s].indegree; }
    }
    return b;
  }
 private:
  std::vector<MvmTaskDesc> tasks_; std::vector<int32_t> args_; std::vector<std::set<int>> out_;
};

// ---------------- executor ----------------
struct MvmConfig {
  int blocks = 0;               // 0: occupancy-sized (SMs * resident blocks); >= 2 (block 0 admits, the rest work)
  int max_tasks = 1 << 18;      // task slots, power of two; slots are recycled when a task completes
  int live_cap = 1 << 14;       // max admitted-but-unfinished tasks (and max tasks per bundle record); ready rings hold this many each
  unsigned succ_cap = 1u << 21; // successor ring (words), power of two; a window's words return when all its tasks completed; never below ring_words (MvmExec raises it)
  int win_cap = 1 << 16;        // bundle records in flight, power of two
  int num_vars = 64;
  int ring_words = 1 << 16;     // admission ring, multiple of 8
  int aging_us = 0;             // 0: strict priority; else one priority level per aging_us of waiting
  // A bundle of `tasks` tasks and `words` record words (MvmBundle::words()) goes in as one record: else it is cut into windows that drain.
  bool fits_one_record(long long tasks, long long words) const { return tasks <= live_cap && tasks <= max_tasks && words <= ring_words / 2; }
  // Record words of a bundle with `tasks` tasks, `edges` edges and `args` argument words, as MvmBundle::words() counts them once it is read.
  static long long record_words(long long tasks, long long edges, long long args) { return mvm_record_words(tasks, edges, args, MVM_REC_HDR, MVM_TASK_WORDS); }
  // Grow the capacities so such a bundle goes in as one record when it would otherwise be cut: a cut bundle runs window by window, each drained
  // before the next (a 9,800-task bundle of 4 replicated runs ran its runs one after another in 10 windows, x4 slower). Above `max_live` tasks
  // (a power of two, at most 2^20) or 2^26 words the capacities stay and the bundle streams. The ready rings cost 4 KiB of device memory per
  // live_cap slot per scheduler class (MVM_NBUCKETS rings of 4 x live_cap 32-byte cells; a FIFO run needs one), so a caller with several classes
  // passes a smaller max_live. Returns whether anything grew. mvm_exec's CLI and demo/mvm_run size their runs with it.
  bool size_for_one_record(long long tasks, long long edges, long long args, long long max_live = 1 << 20) {
    const long long words = record_words(tasks, edges, args);
    if (tasks < 1 || edges < 0 || args < 0 || tasks > std::min<long long>(max_live, 1 << 20) || words >= (1ll << 26)) return false;
    if (fits_one_record(tasks, words) && edges + 2 <= (long long)succ_cap) return false;
    auto p2 = [](long long v) { long long r = 2; while (r < v) r <<= 1; return r; };
    live_cap = (int)std::max<long long>(live_cap, p2(tasks));
    max_tasks = std::max(max_tasks, live_cap);
    ring_words = (int)std::max<long long>(ring_words, p2(2 * words + 128));
    succ_cap = std::max<unsigned>(succ_cap, (unsigned)p2(edges + 2));
    return true;
  }
#ifdef MVM_PROF_GAP
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
  MvmSchedCfg sched;            // SCHED: scheduling policy (off by default: the original single-queue executor)
};
enum class MvmSubmit { OK, BACKPRESSURE_TIMEOUT, TOO_BIG, DEVICE_ERROR, CLOSED, INVALID };
static const char* mvm_submit_name(MvmSubmit s) {
  static const char* n[] = {"OK", "BACKPRESSURE_TIMEOUT", "TOO_BIG", "DEVICE_ERROR", "CLOSED", "INVALID"};
  return n[(int)s];
}
struct MvmSchedStats {  // end-of-run copy of the device accounting (SCHED)
  bool valid = false;
  MvmSchedDev dev;
};
struct MvmResult {
  MvmSchedStats sched;
  std::vector<MvmSlotInfo> info;
  int err_code = 0, err_slot = -1, err_a = 0, err_b = 0, admitted = 0, done = 0, grid = 0;
  float kernel_ms = 0;
  std::vector<MvmTaskDesc> tasks;   // per slot: the latest occupant (== admission order while nothing was recycled)
  std::vector<MvmTrace> trace;
  std::vector<int> version;
  bool ok() const { return err_code == 0; }
};
struct MvmStats { int bundles = 0, backpressure_waits = 0, pads = 0, windows = 0, window_waits = 0; };

// How a buffer is initialised (device memory is written directly; no giant host copies).
struct MvmInit {
  enum Kind { ZERO, FILL, VALUES, SEEDED, FILE } kind = ZERO;
  float value = 0;                  // FILL
  std::vector<float> values;        // VALUES (small data)
  unsigned long long seed = 0; int draws = 1; float bias = 0, coef = 1;   // SEEDED (draws/bias/coef as in metadata)
  std::string path;                 // FILE: absolute or cwd-relative path, little-endian 4-byte elements
};
// Host reference of the seeded generator (bit-identical to the device kernel and runtime/initgen.py).
static void mvm_seeded_reference(float* out, size_t n, unsigned long long seed, int draws, float bias, float coef) {
  for (size_t i = 0; i < n; ++i) out[i] = mvm_seeded_value(seed, i, draws, bias, coef);
}

// Float arithmetic on every thread of every block for a few milliseconds per launch: what holds the SM clock up before a timed launch (--warm-ms).
__global__ void mvm_clock_warm_kernel(float* sink) {
  float x = 1.0f + static_cast<float>(threadIdx.x) / 512.0f;
  for (int i = 0; i < 300000; ++i) x = x * 1.0000001f + 1.0e-7f;
  if (threadIdx.x == 0) atomicAdd(sink, x);         // (keeps the loop's result live)
}

#ifdef MVM_GPT_SMEM_BYTES
// The position of argument `arg` in a registry signature ("name:type,..."), at compile time; -1 when absent. Arguments are read by name
// here because the registry is the ABI the frontend writes them by.
constexpr int mvm_sig_index(const char* sig, const char* arg) {
  for (int idx = 0; *sig; ++idx) {
    const char* q = arg; const char* p = sig;
    while (*q && *p == *q) { ++p; ++q; }
    if (!*q && *p == ':') return idx;
    while (*sig && *sig != ',') ++sig;
    if (*sig == ',') ++sig;
  }
  return -1;
}
constexpr const char* mvm_op_sig(int id) {
  switch (id) {
#define X(n, i, sig) case i: return sig;
#include "mvm_ops_list.inc"
#undef X
  }
  return "";
}
constexpr int MVM_ARG_TILE_FLAGS = mvm_sig_index(mvm_op_sig(MVM_OP_matmul_tile), "flags"), MVM_ARG_ATTN_FWD_HD = mvm_sig_index(mvm_op_sig(MVM_OP_attn_fwd), "hd"),
              MVM_ARG_ATTN_DQ_HD = mvm_sig_index(mvm_op_sig(MVM_OP_attn_bwd_dq), "hd"), MVM_ARG_ATTN_DKV_HD = mvm_sig_index(mvm_op_sig(MVM_OP_attn_bwd_dkv), "hd");
static_assert(MVM_ARG_TILE_FLAGS == MVM_GPT_ARG_TILE_FLAGS && MVM_ARG_ATTN_FWD_HD == MVM_GPT_ARG_ATTN_FWD_HD && MVM_ARG_ATTN_DQ_HD == MVM_GPT_ARG_ATTN_BWD_HD &&
              MVM_ARG_ATTN_DKV_HD == MVM_GPT_ARG_ATTN_BWD_HD, "mvm_gpt_task_smem reads the arguments it sizes by where the registry puts them");
// The dynamic shared memory one task needs. A GPT-pack op's is mvm_gpt_task_smem's, the function the op tests launch with too; the core pack
// takes the small ops' size; an op of any other pack (extra, comm, game, fused steps, test packs) gets the build's maximum, so a bundle
// holding one keeps that launch.
static size_t mvm_task_smem(const MvmTaskDesc& d, const int32_t* args) {
  const int gpt = mvm_gpt_task_smem(d.op_id, args + d.args_off);
  if (gpt >= 0) return (size_t)gpt;
  switch (d.op_id) {
#define X(n, i, sig) case i:
#include "mvm_ops.def"
#undef X
      return MVM_GPT_SMEM_SMALL;
    default: return MVM_DYN_SMEM_BYTES;
  }
}
#endif

class MvmExec {
 public:
  explicit MvmExec(MvmConfig c) : cfg_(c) {
    auto pow2 = [](unsigned long long v) { return v >= 2 && !(v & (v - 1)); };
    if (cfg_.aging_us < 0) throw std::invalid_argument("aging_us must be >= 0");
    if (cfg_.timeout_us == 0 || cfg_.timeout_us > 86400ull * 1000000) throw std::invalid_argument("timeout_us must be in [1, 86400 s]");
    if (cfg_.ring_words % 8 || cfg_.ring_words < 64) throw std::invalid_argument("ring_words must be a multiple of 8 and >= 64");
    if (!pow2(cfg_.live_cap) || !pow2(cfg_.max_tasks) || !pow2(cfg_.succ_cap) || !pow2(cfg_.win_cap)) throw std::invalid_argument("live_cap, max_tasks, succ_cap and win_cap must be powers of two");
    // The successor ring fits every record the admission ring can carry (mvm_window_size.h), whatever produced the record: a sizing flag, or a transform that
    // adds edges after the window was sized (Nov 11: the dynamic path's noop splice took h640-d4-r16 past a ring sized from the bundle as written, and
    // admission stopped with DEVICE ERROR POOL). Every runner builds its executor here, mvm_exec's CLI and demo/mvm_run alike.
    const unsigned succ_floor = mvm_succ_ring_words(cfg_.succ_cap, cfg_.ring_words);   // a power of two: it doubles the power of two checked above
    if (succ_floor != cfg_.succ_cap)
      KLOG(2, "successor ring raised from %u to %u words: the floor of a %d-word admission ring", cfg_.succ_cap, succ_floor, cfg_.ring_words);
    cfg_.succ_cap = succ_floor;
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    sms_ = p.multiProcessorCount; name_ = p.name;
    if (cfg_.stack_bytes) CK(cudaDeviceSetLimit(cudaLimitStackSize, cfg_.stack_bytes));
    dyn_smem_ = MVM_DYN_SMEM_BYTES;
#ifdef MVM_TMA_MAXB   // GPT op pack built: arm the sm_90a TMA + wgmma tiles (no-op elsewhere; MVM_NO_HOPPER=1 disables)
    if (mvm_hopper_host_init()) { hopper_ = true; dyn_smem_ = mvm_hopper_dyn_smem(dyn_smem_); KLOG(2, "hopper: TMA + wgmma tiles armed, dynamic smem %zu B", dyn_smem_); }
#endif
    if (dyn_smem_ > 0) CK(cudaFuncSetAttribute(mvm_exec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mvm_exec_kernel, MVM_THREADS, dyn_smem_));
    if (occ < 1) throw std::runtime_error("kernel does not fit on an SM with " + std::to_string(dyn_smem_) + " bytes of dynamic shared memory");
    grid_ = cfg_.blocks ? cfg_.blocks : sms_ * occ;
    const MvmSchedCfg& sc = cfg_.sched;
    const int nsys = 1 + (sc.on ? 1 : 0);  // admitter (+ policy block)
    if (sc.on && (sc.num_classes < 1 || sc.num_classes > MVM_MAX_CLS)) throw std::invalid_argument("sched.num_classes must be 1..8");
    if (grid_ < nsys + 1) throw std::invalid_argument("need an admitter" + std::string(sc.on ? ", a policy block" : "") + " and at least one worker block");
    nc_ = sc.on ? sc.num_classes : 1;
    KLOG(2, "device %s, %d SMs, %d resident blocks/SM with %zu B dynamic smem -> grid %d (1 admitter%s + %d workers)", name_.c_str(), sms_, occ, dyn_smem_, grid_, sc.on ? " + 1 policy" : "", grid_ - nsys);
    const size_t T = cfg_.max_tasks;
    CK(cudaMalloc(&S_.tasks, T * sizeof(MvmTaskDesc)));
    CK(cudaMalloc(&S_.indeg, T * 4)); CK(cudaMalloc(&S_.state, T * 4)); CK(cudaMalloc(&S_.park_next, T * 4));
    CK(cudaMemset(S_.indeg, 0, T * 4));   // the admitter ADDS a task's indegree to what its predecessors already subtracted (mvm_admit_record), so a free slot's count is 0
    CK(cudaMalloc(&S_.trace, T * sizeof(MvmTrace)));
    CK(cudaMalloc(&S_.args, T * MVM_ARG_STRIDE * 4)); CK(cudaMalloc(&S_.adm_slots, (size_t)cfg_.live_cap * 4)); CK(cudaMalloc(&S_.stage, (size_t)cfg_.ring_words * 4));
    CK(cudaMalloc(&S_.succ, (size_t)cfg_.succ_cap * 4)); succ_ring_ = S_.succ;
    CK(cudaMalloc(&S_.version, cfg_.num_vars * 4)); CK(cudaMalloc(&S_.park_head, cfg_.num_vars * 4));
    // S_.cells is allocated in prepare(), once the run's facts are known (a FIFO run needs one ring, not one per priority bucket per class: 1/32 of the memory, 4 GB on the largest record)
    CK(cudaMalloc(&S_.fcells, T * sizeof(MvmCell)));
    CK(cudaMalloc(&S_.qctl, nc_ * MVM_NBUCKETS * sizeof(MvmQCtl)));
    CK(cudaMalloc(&S_.wins, (size_t)cfg_.win_cap * sizeof(MvmWin)));
    CK(cudaMalloc(&S_.cnt, sizeof(MvmCounters)));
    CK(cudaMemset(S_.tasks, 0, T * sizeof(MvmTaskDesc))); CK(cudaMemset(S_.trace, 0, T * sizeof(MvmTrace)));
    CK(cudaMemset(S_.version, 0, cfg_.num_vars * 4)); CK(cudaMemset(S_.qctl, 0, nc_ * MVM_NBUCKETS * sizeof(MvmQCtl)));
    CK(cudaMemset(S_.wins, 0, (size_t)cfg_.win_cap * sizeof(MvmWin))); CK(cudaMemset(S_.cnt, 0, sizeof(MvmCounters)));
    // host-shared control block: ring + doorbell/ring_head/closed on separate lines + error mirror + per-window done marks
    CK(cudaHostAlloc((void**)&ctl_, 4096, cudaHostAllocMapped)); memset(ctl_, 0, 4096);
    init_sched();
    CK(cudaHostAlloc((void**)&ring_, (size_t)cfg_.ring_words * 4, cudaHostAllocMapped)); memset(ring_, 0, (size_t)cfg_.ring_words * 4);
    CK(cudaHostAlloc((void**)&win_done_, (size_t)cfg_.win_cap * 4, cudaHostAllocMapped)); memset((void*)win_done_, 0, (size_t)cfg_.win_cap * 4);
    doorbell_ = (unsigned long long*)(ctl_ + 0); ring_head_ = (unsigned long long*)(ctl_ + 128);
    closed_ = (unsigned*)(ctl_ + 256); herr_ = (MvmHostErr*)(ctl_ + 384);
    void* d;
    CK(cudaHostGetDevicePointer(&d, ring_, 0)); S_.ring = (int*)d;
    CK(cudaHostGetDevicePointer(&d, doorbell_, 0)); S_.doorbell = (unsigned long long*)d;
    CK(cudaHostGetDevicePointer(&d, ring_head_, 0)); S_.ring_head = (unsigned long long*)d;
    CK(cudaHostGetDevicePointer(&d, closed_, 0)); S_.closed = (unsigned*)d;
    CK(cudaHostGetDevicePointer(&d, herr_, 0)); S_.herr = (MvmHostErr*)d;
    CK(cudaHostGetDevicePointer(&d, (void*)win_done_, 0)); S_.win_done = (unsigned*)d;
    S_.nslots = cfg_.max_tasks; S_.live_cap = cfg_.live_cap; S_.succ_cap = cfg_.succ_cap; S_.win_cap = cfg_.win_cap;
    S_.num_vars = cfg_.num_vars; S_.ring_words = cfg_.ring_words;
    S_.aging_ns = cfg_.aging_us ? (long long)cfg_.aging_us * 1000 : (1ll << 50);
    S_.cont_on = cfg_.cont;
#ifdef MVM_PROF_GAP
    S_.noop_ops = cfg_.noop_ops;
#endif
    set_fifo(cfg_.fifo_only);
    S_.no_recycle = cfg_.no_recycle; S_.no_waits = cfg_.no_waits; S_.trace_on = cfg_.trace_on;
    S_.prestaged = 0; S_.succ_alias = 0; S_.warm_only = 0;
    S_.timeout_ns = cfg_.timeout_us * 1000ull; S_.task_limit_ns = cfg_.task_limit_us * 1000ull;
    S_.rcap = 4 * cfg_.live_cap;
    CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CK(cudaEventCreate(&e0_)); CK(cudaEventCreate(&e1_));
  }
  ~MvmExec() {
    if (started_ && !finished_) { try { close(); finish(); } catch (...) {} }
    for (size_t i = 0; i < bufs_.size(); ++i) if (!std::count(views_.begin(), views_.end(), i)) cudaFree(bufs_[i].p);
    for (float* p : arenas_) cudaFree(p);
    for (void* p : {(void*)S_.tasks, (void*)S_.indeg, (void*)S_.state, (void*)S_.park_next, (void*)S_.trace, (void*)succ_ring_, (void*)S_.args, (void*)S_.adm_slots, (void*)S_.stage,
                    (void*)S_.version, (void*)S_.park_head, (void*)S_.cells, (void*)S_.fcells, (void*)S_.qctl, (void*)S_.wins, (void*)S_.cnt, (void*)S_.bufs, (void*)S_.sp.dev, (void*)S_.sp.info, (void*)S_.sp.pv, (void*)S_.sp.pubcls}) cudaFree(p);
    cudaFreeHost(ctl_); cudaFreeHost(ring_); cudaFreeHost((void*)win_done_);
    cudaStreamDestroy(stream_); cudaEventDestroy(e0_); cudaEventDestroy(e1_);
  }
  MvmExec(const MvmExec&) = delete; MvmExec& operator=(const MvmExec&) = delete;

  // Allocates a rows x cols buffer of dtype 0 (f32), 1 (i32) or 2 (bf16: raw 16-bit storage, 2 bytes per element) and initialises it on the device.
  int add_buffer(int rows, int cols, const MvmInit& init = {}, int dtype = 0) {
    if (started_ || prepared_) throw std::logic_error("buffers are fixed at start()");
    if (rows < 1 || cols < 1 || (dtype < 0 || dtype > 2)) throw std::invalid_argument("bad buffer shape or dtype");
    const size_t n = (size_t)rows * cols;
    MvmBuf b{}; CK(cudaMalloc(&b.p, n * (dtype == 2 ? 2 : 4)));
    b.rows = rows; b.cols = cols; b.ld = cols; b.dtype = dtype;
    init_buffer((float*)b.p, n, init, dtype);
    bufs_.push_back(b);
    return (int)bufs_.size() - 1;
  }
  int add_buffer(int rows, int cols, const std::vector<float>& values) { MvmInit i; if (!values.empty()) { i.kind = MvmInit::VALUES; i.values = values; } return add_buffer(rows, cols, i); }
  // Pooled device memory for many small buffers: arena() then add_view(). Zero-initialised, freed with the executor.
  float* arena(size_t floats) {
    float* p; CK(cudaMalloc(&p, floats * 4)); CK(cudaMemset(p, 0, floats * 4)); arenas_.push_back(p); return p;
  }
  int add_view(float* p, int rows, int cols, int dtype = 0) {
    if (started_ || prepared_) throw std::logic_error("buffers are fixed at start()");
    MvmBuf b{}; b.p = p; b.rows = rows; b.cols = cols; b.ld = cols; b.dtype = dtype; views_.push_back(bufs_.size());
    bufs_.push_back(b); return (int)bufs_.size() - 1;
  }
  void set_version(int var, int v) {
    if (started_) throw std::logic_error("versions are fixed at start()");
    if (var < 0 || var >= cfg_.num_vars) throw std::invalid_argument("version variable out of range");
    CK(cudaMemcpy(S_.version + var, &v, 4, cudaMemcpyHostToDevice));
  }
  // MVM_DIST_HOOK: attach the distributed-execution descriptor (inbox, in_vars, ctl). Device-visible memory, outlives the run.
  void set_dist(const MvmDistDev* d) { if (started_) throw std::logic_error("dist is fixed at start()"); S_.dist = d; }
  int nbufs() const { return (int)bufs_.size(); }
  int grid() const { return grid_; }
  const MvmConfig& config() const { return cfg_; }
  const MvmStats& stats() const { return stats_; }

  // ---- SCHED: runtime policy control through the mapped control block (no relaunch) ----
  MvmSchedParams* sched_params() { return hp_; }                // controllers write with MvmController::write_params
  void set_sched_params(const MvmSchedParams& p) { if (!cfg_.sched.on) throw std::logic_error("scheduler is off"); validate_sched(p); MvmController::write_params(hp_, p); }
  const MvmSchedTele* telemetry() const { return ht_; }
  int num_classes() const { return nc_; }
  int workers() const { return grid_ - 1 - (cfg_.sched.on ? 1 : 0); }
  // The FIFO ring (MvmDevState::fifo_only) never waits for a free cell: mvm_publish_cell stores a slot and its generation in one release store,
  // and each idle worker watches the cell of its own ticket. Its 4 x live_cap cells must exceed every position in flight: the ready tasks
  // (at most 2 x live_cap, the header's budgets) plus one ticket per worker. workers < live_cap keeps them under 3 x live_cap, so a worker
  // that stalls between taking its ticket and reading its cell is overrun only after more than live_cap further enqueues; past the bound
  // waiting workers share cells. Either way the worker finds a later generation and stops the run with QUEUE_OVERFLOW (the litmus probe
  // `queuefifo_cap` overruns it at live_cap 2 with 200 CTAs). The cell holds a slot in its low MVM_FIFO_SHIFT bits and the generation above
  // them, so the slots must fit those bits, and a run has 2^(64 - MVM_FIFO_SHIFT) enqueues before the generation wraps (the stall timeout
  // reports that). The bucket rings wait for a free cell, so they serve any live_cap. CLI runs sit far inside the bound (live_cap >= 16,384
  // against a few hundred workers).
  bool fifo_fits() const { return workers() < cfg_.live_cap && cfg_.max_tasks <= (1 << MVM_FIFO_SHIFT); }
  bool fifo_only() const { return S_.fifo_only != 0; }   // the ring in use; config().fifo_only stays the caller's request, which submit() holds to
  void static_split(MvmSchedParams& p, const std::vector<double>& shares) const { mvm_static_split(p, shares, sms_); }

  // MVM_DIST_HOOK: the allocation half of start(). cudaMalloc and cudaHostAlloc may synchronise with EVERY kernel running on the
  // device, so when several executors share one GPU (tests, small clusters) all of them must prepare() before the first
  // start(): a start() that allocates while a peer's persistent kernel waits for this very executor deadlocks until the peer's
  // timeout. A lone executor never needs to call it.
  // Launch with the dynamic shared memory the run's tasks need, not the build's maximum (MVM_DYN_SMEM_BYTES): the SM's 192 KB serve shared
  // memory first, so a 147,456 B launch leaves the A100 28 KB of L1 for every op's cached loads and spill reloads (measured on an A100: the same tree's
  // gpt-tiny ran x1.12 faster at 90 KB). Call once every task of the run is known and before any launch (the CLI's bundles); a run that may
  // be handed any op later keeps the maximum. The grid stays the one the constructor sized at the maximum (one block an SM either way).
  void size_launch(const std::vector<MvmBundle>& bundles) {
    if (started_) throw std::logic_error("size_launch after start");
#ifdef MVM_GPT_SMEM_BYTES
    if (hopper_) { KLOG(2, "launch: %zu B of dynamic shared memory, kept for the Hopper tiles", dyn_smem_); return; }
    size_t need = MVM_GPT_SMEM_SMALL; int by = -1; long tasks = 0;
    for (const MvmBundle& b : bundles)
      for (const MvmTaskDesc& d : b.tasks) { const size_t n = mvm_task_smem(d, b.args.data()); ++tasks; if (n > need) { need = n; by = d.op_id; } }
    need = std::min(need, dyn_smem_);
    const MvmOpInfo* op = by >= 0 ? mvm_op(by) : nullptr;
    KLOG(2, "launch: %zu B of dynamic shared memory, sized to the largest of %ld tasks (%s), against the build's %zu B", need, tasks, op ? op->name : "small ops only", dyn_smem_);
    if (need != dyn_smem_) { dyn_smem_ = need; CK(cudaFuncSetAttribute(mvm_exec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_)); }
#else
    (void)bundles;
#endif
  }
  void prepare() {
    if (prepared_) return;
    prepared_ = true;
    CK(cudaMalloc(&S_.cells, (size_t)(S_.fifo_only ? 1 : nc_ * MVM_NBUCKETS) * 4 * (size_t)cfg_.live_cap * sizeof(MvmCell)));   // ready rings (see the constructor)
    MvmBuf* d; size_t n = std::max<size_t>(bufs_.size(), 1);
    CK(cudaMalloc(&d, n * sizeof(MvmBuf)));
    if (!bufs_.empty()) CK(cudaMemcpy(d, bufs_.data(), bufs_.size() * sizeof(MvmBuf), cudaMemcpyHostToDevice));
    S_.bufs = d; S_.nbufs = (int)bufs_.size();
    if (cfg_.sched.on) {  // the policy mirror starts as the launch parameters, so workers never see a zero weight
      MvmSchedDev init{}; for (int& b : init.op_blk) b = -1; const MvmSchedParams& p = cfg_.sched.p;
      init.flags = p.flags; init.affinity_ns = p.affinity_us * 1000; init.bw_limit = std::max(p.bw_limit_min, std::min(p.bw_limit_max, p.bw_limit_init));
      for (int c = 0; c < nc_; ++c) {
        MvmClsDev& k = init.cls[c]; const MvmClassParams& q = p.cls[c];
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
  // Per-run facts the CLI learns from its bundles, set before start(): see MvmDevState::fifo_only, no_recycle and no_waits.
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
    // The record is the ring's only one (offset 0, no pad), nothing is recycled and no scheduler classes it, so the admitter's successor copy is an identity
    // (slot = record index, ring offset 0): the workers read the successor words where they already are. That copy ran on the admitter block alone, before the
    // first task and inside the timed window (gpt-162m: 8.5M words, 5.06 ms of a 203 ms kernel on an A100). submit()'s validation is then the only range check
    // on those words (the admitter's per-word check is skipped), so a record submitted with validate = false keeps the copy.
    if (S_.prestaged && S_.no_recycle && !cfg_.sched.on && !unvalidated_ && ring_[0] == MVM_REC_MAGIC && ring_[6] == MVM_REC_BUNDLE &&
        (unsigned long long)ring_[1] == wpos_ && ring_[3] >= 0) {
      // the workers index the ring modulo succ_cap: below the record's edge count they would wrap into other words (the CLI raises succ_cap past edges + 2)
      if ((unsigned)ring_[3] > S_.succ_cap) throw std::logic_error("successor ring (" + std::to_string(S_.succ_cap) + " words) below the record's " + std::to_string(ring_[3]) + " edges");
      S_.succ = S_.stage + MVM_REC_HDR + (size_t)ring_[2] * MVM_TASK_WORDS;
      S_.succ_alias = 1;
      CK(cudaFree(succ_ring_)); succ_ring_ = nullptr;   // never written or read under the alias
      KLOG(2, "successors read in place: %d words of the prestaged record, no copy; the %u-word successor ring freed", ring_[3], S_.succ_cap);
    }
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
      mvm_clock_warm_kernel<<<grid_, MVM_THREADS, 0, stream_>>>(sink);
      CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream_));
      ++launches;
    }
    CK(cudaFree(sink));
    KLOG(2, "clock warm-up: %d launches, %d ms requested", launches, cfg_.warm_ms);
  }
  void start() {
    if (started_) throw std::logic_error("already started");
    prepare();
    { MvmDevState warm = S_; warm.warm_only = 1; mvm_exec_kernel<<<1, MVM_THREADS, dyn_smem_, stream_>>>(warm); CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream_)); }   // upload the kernel's code outside the timed window
    clock_warmup();
    mvm_init_kernel<<<64, MVM_THREADS, 0, stream_>>>(S_);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e0_, stream_));
    mvm_exec_kernel<<<grid_, MVM_THREADS, dyn_smem_, stream_>>>(S_);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1_, stream_));
    started_ = true;
    KLOG(2, "kernel launched: grid %d x %d threads, live_cap %d, slots %d, succ ring %u words, ring %d words, aging %d us, sched %s", grid_, MVM_THREADS, cfg_.live_cap, cfg_.max_tasks, cfg_.succ_cap, cfg_.ring_words, cfg_.aging_us,
         !cfg_.sched.on ? "off" : cfg_.sched.mode == MVM_SCHED_GLOBAL ? "global" : cfg_.sched.mode == MVM_SCHED_STATIC ? "static" : "dynamic");
  }

  // Thread-safe producer. Blocks (up to timeout_ms) while the ring is full: that is the backpressure; it never drops.
  // *wid_out: the bundle record's window id (see wait_window).
  MvmSubmit submit(const MvmBundle& b, int timeout_ms = 10000, bool validate = true, unsigned* wid_out = nullptr) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_ && !prestart_) throw std::logic_error("submit before start()");
    if (closed_flag_) { KLOG(1, "submit after close()"); return MvmSubmit::CLOSED; }
    if (herr_->code) { last_error_ = std::string("device error ") + mvm_err_name(herr_->code); return MvmSubmit::DEVICE_ERROR; }   // the device is dead: accepted bundles would never run
    if (!validate) unvalidated_ = true;   // prestage() reads successors in place only when the host checked every one
    std::string why = validate ? b.validate(nbufs(), cfg_.num_vars, cfg_.live_cap) : std::string();
    if (!why.empty()) { KLOG(0, "bundle rejected: %s", why.c_str()); last_error_ = why; return why.find("live_cap") != std::string::npos ? MvmSubmit::TOO_BIG : MvmSubmit::INVALID; }
    if (cfg_.fifo_only) for (const MvmTaskDesc& t : b.tasks) if (t.priority != 0 || t.deadline_us != -1) {   // the FIFO ring serves everything in release order: a priority or deadline it cannot honor is a caller error, not a silent reorder
      last_error_ = "the FIFO ring takes only tasks of priority 0 with no deadline"; KLOG(0, "bundle rejected: %s (task priority %d, deadline %d us)", last_error_.c_str(), t.priority, t.deadline_us); return MvmSubmit::INVALID;
    }
    const size_t words = b.words();
    if (words > (size_t)cfg_.ring_words) { last_error_ = "bundle record larger than the admission ring"; KLOG(0, "%s (%zu > %d words)", last_error_.c_str(), words, cfg_.ring_words); return MvmSubmit::TOO_BIG; }
    auto t0 = std::chrono::steady_clock::now();
    bool waited = false;
    // A record is contiguous. If it does not fit before the ring end, a pad record fills the tail and is published ON ITS OWN
    // (waiting for pad + record at once could need more than the whole ring); then the record waits for space at lap start.
    for (;;) {
      unsigned long long head = __atomic_load_n(ring_head_, __ATOMIC_ACQUIRE);
      const size_t rem = cfg_.ring_words - wpos_ % cfg_.ring_words, need = words > rem ? rem : words;
      if (cfg_.ring_words - (wpos_ - head) >= need) {
        if (need == words) break;
        write_header(rem, 0, 0, 0, MVM_REC_PAD); wpos_ += rem; ++stats_.pads;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(doorbell_, wpos_, __ATOMIC_RELEASE);
        KLOG(3, "ring wrap: %zu pad words", rem);
        continue;
      }
      if (herr_->code) { last_error_ = "device error"; return MvmSubmit::DEVICE_ERROR; }
      if (!waited) { waited = true; ++stats_.backpressure_waits; KLOG(3, "ring full (%llu/%d words used): backpressure", (unsigned long long)(wpos_ - head), cfg_.ring_words); }
      if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
        last_error_ = "admission ring full for " + std::to_string(timeout_ms) + " ms";
        KLOG(1, "%s", last_error_.c_str()); return MvmSubmit::BACKPRESSURE_TIMEOUT;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    int* r = ring_ + wpos_ % cfg_.ring_words;
    write_header(words, (int)b.tasks.size(), (int)b.succ.size(), (int)b.args.size(), MVM_REC_BUNDLE);
    r += MVM_REC_HDR;
    memcpy(r, b.tasks.data(), b.tasks.size() * sizeof(MvmTaskDesc)); r += b.tasks.size() * MVM_TASK_WORDS;
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
    return MvmSubmit::OK;
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
#ifdef MVM_BREAK_WINDOW_DONE   // negative build: the pre-fix equality test (false for a finished window once the table wrapped over it: the `recycle` flake)
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
  MvmSubmit submit_stream(const MvmBundle& big, int max_window_tasks = 0, int timeout_ms = 60000, std::vector<unsigned>* wids_out = nullptr) {
    const auto tp0 = std::chrono::steady_clock::now();   // host prep time (validate + plan) is time the device idles: logged below
    std::string why = big.validate(nbufs(), cfg_.num_vars, 0);
    if (!why.empty()) { KLOG(0, "stream rejected: %s", why.c_str()); last_error_ = why; return MvmSubmit::INVALID; }
    const int nt = (int)big.tasks.size();
    for (int i = 0; i < nt; ++i) for (int k = 0; k < big.tasks[i].succ_cnt; ++k)
      if (big.succ[big.tasks[i].succ_off + k] <= i) { last_error_ = "stream needs topologically ordered tasks (edge " + std::to_string(i) + " -> " + std::to_string(big.succ[big.tasks[i].succ_off + k]) + ")"; KLOG(0, "%s", last_error_.c_str()); return MvmSubmit::INVALID; }
    // window boundaries: bounded by tasks, and by record words (half the ring) counting every outgoing edge
    const int cap = std::min({max_window_tasks > 0 ? max_window_tasks : cfg_.live_cap, cfg_.live_cap, cfg_.max_tasks});
    std::vector<int> lo{0};
    { size_t words = MVM_REC_HDR; int cnt = 0;
      for (int i = 0; i < nt; ++i) {
        size_t w = MVM_TASK_WORDS + big.tasks[i].succ_cnt + big.tasks[i].args_words;
        if (cnt > 0 && (cnt >= cap || words + w > (size_t)cfg_.ring_words / 2)) { lo.push_back(i); words = MVM_REC_HDR; cnt = 0; }
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
          if (herr_->code) { last_error_ = "device error"; return MvmSubmit::DEVICE_ERROR; }
          if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) { last_error_ = "window " + std::to_string(j) + " did not complete: " + window_why_not_done(wid[j]); KLOG(1, "%s", last_error_.c_str()); return MvmSubmit::BACKPRESSURE_TIMEOUT; }
          if (!waited) { waited = true; ++stats_.window_waits; }
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
      }
      MvmBundle w; const int a = lo[k], z = lo[k + 1];
      w.tasks.reserve(z - a);
      for (int i = a; i < z; ++i) {
        MvmTaskDesc t = big.tasks[i];
        const int off = (int)w.succ.size(); t.succ_off = off;
        for (int e = 0; e < t.succ_cnt; ++e) { int s = big.succ[big.tasks[i].succ_off + e]; if (s < z) w.succ.push_back(s - a); }
        t.succ_cnt = (int)w.succ.size() - off; t.indegree = 0;
        t.args_off = (int)w.args.size();
        w.args.insert(w.args.end(), big.args.begin() + big.tasks[i].args_off, big.args.begin() + big.tasks[i].args_off + t.args_words);
        w.tasks.push_back(t);
      }
      for (int s : w.succ) ++w.tasks[s].indegree;
      MvmSubmit st = submit(w, timeout_ms, false, &wid[k]);
      if (st != MvmSubmit::OK) return st;
      if (wids_out) wids_out->push_back(wid[k]);
      ++stats_.windows;
    }
    KLOG(3, "stream: %d tasks in %d windows", nt, nw);
    return MvmSubmit::OK;
  }

  // ---- fused MLP steps (mvm_fuse.h): the rewritten bundle's runs, their device state
  // Runs the fusion pass over a closed bundle: what it fuses, and why it left each other run alone, goes to the log. The caller swaps the rewritten bundle in when something fused.
  // The shared memory a fused block may use on this device: the static kernel's opt-in maximum less its own static shared memory, in floats (16-float multiples).
  int fused_smem_floats() const { return smem_floats_of(reinterpret_cast<const void*>(mvm_static_kernel)); }
  static int fgpt_smem_floats() {   // every instance's (both crossing forms, with ZBW's code and without): the smallest budget, so any run fits its own
    return std::min(std::min(smem_floats_of(reinterpret_cast<const void*>(mvm_fgpt_kernel<false, false>)), smem_floats_of(reinterpret_cast<const void*>(mvm_fgpt_kernel<true, false>))),
                    std::min(smem_floats_of(reinterpret_cast<const void*>(mvm_fgpt_kernel<false, true>)), smem_floats_of(reinterpret_cast<const void*>(mvm_fgpt_kernel<true, true>))));
  }
  // The one-block form (mvm_fgpt1.h) for a GPT run whose whole state fits one block: on by default; off, every run takes its gang (a read of
  // the form against its gang, mvm_exec --no-fgpt1).
  void set_fgpt1(bool on) { fgpt1_ = on; }
  static int f2d_smem_floats() {    // F's kernel's, the same way (its own static shared memory)
    return std::min(smem_floats_of(reinterpret_cast<const void*>(mvm_f2d_kernel<false>)), smem_floats_of(reinterpret_cast<const void*>(mvm_f2d_kernel<true>)));
  }
  // A kernel's dynamic shared memory a block, in floats (a multiple of 16): the device's opt-in limit less the kernel's static share.
  static int smem_floats_of(const void* kernel) {
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kernel));
    const long long bytes = (long long)p.sharedMemPerBlockOptin - (long long)fa.sharedSizeBytes;
    return bytes > 0 ? (int)(bytes / 4) & ~15 : 0;
  }
  // batch_min_runs: the fewest grid runs of one shape the batched form takes (MvmFuseOptions); 0 turns the batched form off (--no-batch).
  // virtual_over_batch: the most grid runs the virtual form takes ahead of the batched form (MvmFuseOptions::grid_virtual_over_batch; 0: the
  // batched form first). Neither count has a default, so a call names both.
  MvmFuseResult fuse_bundle(const MvmBundle& b, int workers, const std::vector<int>& keep, int batch_min_runs, int virtual_over_batch) const {
    MvmFuseOps ops;
    ops.noop = mvm_op_by_name("noop"); ops.select_window = mvm_op_by_name("select_window"); ops.embedding_fwd = mvm_op_by_name("embedding_fwd"); ops.matmul_tile = mvm_op_by_name("matmul_tile");
    ops.softmax_ce_fwd = mvm_op_by_name("softmax_ce_fwd"); ops.softmax_ce_reduce = mvm_op_by_name("softmax_ce_reduce"); ops.softmax_ce_bwd = mvm_op_by_name("softmax_ce_bwd"); ops.softmax_ce_fused = mvm_op_by_name("softmax_ce_fused");
    ops.gelu_bwd = mvm_op_by_name("gelu_bwd"); ops.colsum = mvm_op_by_name("colsum"); ops.embedding_bwd = mvm_op_by_name("embedding_bwd"); ops.sgd = mvm_op_by_name("sgd"); ops.sgd_range = mvm_op_by_name("sgd_range");
    ops.fmlp_slice = mvm_op_by_name("fmlp_slice"); ops.fmlp_sgd = mvm_op_by_name("fmlp_sgd"); ops.fmlp_split = mvm_op_by_name("fmlp_split"); ops.fmlp_grid = mvm_op_by_name("fmlp_grid");
    ops.fmlp_batch = batch_min_runs > 0 ? mvm_op_by_name("fmlp_batch") : -1;
    const std::vector<MvmFuseBuf> infos = fuse_bufs();
    const MvmSOpLookup op_of = [](int id) { const MvmOpInfo* i = mvm_op(id); MvmSOpInfo v; if (i) { v.arity = i->arity; v.types = i->types; mvm_static_gang_args(i->sig, v); } return v; };
    MvmFuseOptions opts; opts.workers = workers; opts.keep = keep; opts.smem_floats = fused_smem_floats(); opts.batch_min_runs = std::max(batch_min_runs, 1);
    opts.grid_virtual_over_batch = virtual_over_batch;
    MvmFuseResult r = mvm_fuse_mlp(b.tasks, b.succ, b.args, infos, ops, op_of, opts);
    for (const std::string& line : r.report) KLOG(2, "fuse: %s", line.c_str());
    for (const std::string& line : r.warnings) KLOG(1, "fuse: %s", line.c_str());
    if (batch_min_runs <= 0) KLOG(2, "fuse: the batched form is off (--no-batch)");
    else KLOG(2, "fuse: the batched form takes %d to %d grid runs of one shape (--batch-min-runs), the virtual form ahead of it up to %d (--virtual-over-batch)",
              opts.batch_min_runs, MVM_BATCH_MAXR, virtual_over_batch);
    if (!r.dropped.empty()) KLOG(1, "fuse: nothing fused: %s, and each fused form runs on a static-kernel entry of its own", r.dropped.c_str());
    if (r.fused_runs) KLOG(2, "fuse: shared-memory budget %d floats (%d bytes) a fused block", opts.smem_floats, opts.smem_floats * 4);
    KLOG(2, "fuse: %d run(s) fused, %d unchanged; %d tasks replaced by %d; static kernel entry: %s", r.fused_runs, r.unchanged_runs, r.replaced_tasks, r.new_tasks, mvm_fuse_entry_name(r.entry).c_str());
    return r;
  }

  // The fusion passes' view of this executor's buffers: shape, row pitch and type, as add_buffer and add_view laid them out.
  std::vector<MvmFuseBuf> fuse_bufs() const {
    std::vector<MvmFuseBuf> v(bufs_.size());
    for (size_t i = 0; i < bufs_.size(); ++i) v[i] = MvmFuseBuf{bufs_[i].rows, bufs_[i].cols, bufs_[i].ld, bufs_[i].dtype};
    return v;
  }
  // The device's SM count, as the executor reads it, for a caller that must decide before an executor exists.
  static int device_sms() {
    int dev, n = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
  }
  // The fused GPT step's pass (mvm_fuse_gpt.h): each GPT training run whose gang fits the fused kernel's shared memory on this device and
  // `workers` resident blocks becomes one fgpt_step gang (run_fgpt runs a bundle that is such gangs alone). Static, on buffers described by
  // `bufs`, so main() can run it before the executor is built: the path it decides sets how the bundle is admitted (only the dynamic path
  // dissolves noops and sorts the tape), and the admission pools are sized from the bundle as admitted when the executor is built.
  static MvmFuseGptPlan plan_fgpt(const MvmBundle& b, const std::vector<MvmFuseBuf>& bufs, int workers, const std::vector<int>& keep, bool log, bool fgpt1) {
    MvmFuseGptOps ops;
    const auto id = [](const char* n) { return mvm_op_by_name(n); };
    ops.select_window = id("select_window"); ops.embedding_fwd = id("embedding_fwd"); ops.layernorm_fwd = id("layernorm_fwd"); ops.matmul_tile = id("matmul_tile");
    ops.rope_fwd = id("rope_fwd"); ops.attn_fwd = id("attn_fwd"); ops.softmax_ce_fused = id("softmax_ce_fused"); ops.softmax_ce_reduce = id("softmax_ce_reduce");
    ops.layernorm_bwd_dx = id("layernorm_bwd_dx"); ops.norm_bwd_params = id("norm_bwd_params"); ops.colsum = id("colsum"); ops.attn_bwd_delta = id("attn_bwd_delta");
    ops.attn_bwd_dq = id("attn_bwd_dq"); ops.attn_bwd_dkv = id("attn_bwd_dkv"); ops.rope_bwd = id("rope_bwd"); ops.embedding_bwd = id("embedding_bwd");
    ops.gradnorm_partial = id("gradnorm_partial"); ops.gradnorm_finish = id("gradnorm_finish"); ops.counter_inc = id("counter_inc"); ops.adamw = id("adamw"); ops.noop = id("noop");
    MvmFuseGptOptions opts; opts.workers = workers; opts.keep = keep; opts.smem_floats = fgpt_smem_floats(); opts.f2d_smem_floats = f2d_smem_floats();
    // The one-block kernel's shared memory, asked only of a bundle with attention (every run the form takes has it): the query loads that
    // kernel, before any buffer is allocated, and the load alone moved an MLP bundle's buffers (mlp1024-r1 x0.966 on unchanged kernels).
    const bool attn = ops.attn_fwd >= 0 && std::any_of(b.tasks.begin(), b.tasks.end(), [&](const MvmTaskDesc& t) { return t.op_id == ops.attn_fwd; });
    opts.fgpt1_smem_floats = fgpt1 && attn ? smem_floats_of(reinterpret_cast<const void*>(mvm_fgpt1_kernel)) : 0;
    MvmFuseGptPlan r = mvm_fuse_gpt(b.tasks, b.succ, b.args, bufs, ops, id("fgpt_step"), opts);
    if (log) {
      for (const std::string& line : r.report) KLOG(2, "fuse gpt: %s", line.c_str());
      KLOG(2, "fuse gpt: %d run(s) fused, %d unchanged; %d tasks replaced by %d", r.fused_runs, r.unchanged_runs, r.replaced_tasks, r.new_tasks);
    }
    return r;
  }
  MvmFuseGptPlan fuse_gpt(const MvmBundle& b, int workers, const std::vector<int>& keep, bool log = true) const { return plan_fgpt(b, fuse_bufs(), workers, keep, log, fgpt1_); }

  struct FusedState { MvmFusedDesc* desc = nullptr; MvmSplitDesc* split = nullptr; MvmGridDesc* grid = nullptr; MvmBatchDesc* batch = nullptr; std::vector<void*> allocs; int smem_bytes = 0; };
  // One descriptor per fused run, with its partial-gradient scratch (zeroed), and the shared memory the largest run's slice (or split member) needs. A run's index is its descriptor's in both
  // tables, the one its form uses (mvm_fuse.h's ops carry it as `desc`); the other table's entry is unused. Buffers are already device-allocated (after prepare()).
  FusedState make_fused_state(const std::vector<MvmFuseRun>& runs, const std::vector<MvmFuseBatch>& batches) {
    FusedState st;
    std::vector<MvmFusedDesc> descs(runs.size());
    std::vector<MvmSplitDesc> splits(runs.size());
    std::vector<MvmGridDesc> grids(runs.size());
    bool any_split = false, any_grid = false;
    const auto scratch = [&](size_t floats) { float* p; CK(cudaMalloc(&p, floats * sizeof(float))); CK(cudaMemset(p, 0, floats * sizeof(float))); st.allocs.push_back(p); return p; };
    const auto param = [&](int idx) { return static_cast<float*>(bufs_[idx].p); };
    for (size_t i = 0; i < runs.size(); ++i) {
      const MvmFuseRun& r = runs[i];
      if (r.batch) continue;   // its batch's descriptor below holds it
      if (r.grid) {   // the grid form: one copy of the model over P members, its partials and exchanges, the gang's barrier words (mvm_fused_grid.h);
                      // on virtual members (r.phys), phys blocks carry the P members and gelu'(Z) goes to L2 (gq)
        MvmGridDesc& d = grids[i]; any_grid = true;
        d.V = r.V; d.E = r.E; d.L = r.L; d.H = r.H[0]; d.T = r.T; d.P = r.P; d.lr = r.lr; d.phys = r.phys;
        const int floats = r.phys ? mvm_grid_layout_chunked(d, r.smem_floats) : mvm_grid_layout(d);
        if (floats < 0 || floats > r.smem_floats) throw std::runtime_error("fuse: a grid run's layout does not fit what the pass sized it for");
        st.smem_bytes = std::max(st.smem_bytes, floats * (int)sizeof(float));
        const size_t TH = (size_t)d.T * d.Hp;   // the step's activations and partials, at the padded width
        d.emb = param(r.wte); d.wh = param(r.head);
        for (int l = 0; l < r.L; ++l) { d.w[l] = param(r.w[l]); d.b[l] = param(r.b[l]); d.Y[l] = scratch(TH); }
        d.part = scratch(d.L > 1 ? (size_t)d.ka * TH : 1); d.hpart = scratch((size_t)d.kah * d.T * d.VP); d.dlog = scratch((size_t)d.T * d.VP);
        d.dyp = scratch((size_t)std::max(d.nb, 1) * TH); d.dz = scratch(TH); d.dx0 = scratch((size_t)d.nb0 * d.T * d.E); d.rloss = scratch(d.T);
        if (d.phys) d.gq = scratch((size_t)d.phys * d.gqs);   // each block's gelu'(Z) of its chunk, written in the forward, read in the backward
        d.ctl = reinterpret_cast<unsigned*>(scratch(mvm_gang::line_words(d.phys ? d.phys : d.P)));   // the gang's control words (mvm_gang::cross: one a block), zeroed once
        continue;
      }
      if (r.split) {   // the split form: Q x S members, their partials, the exchanges, the gang's barrier words (mvm_fused_split.h)
        MvmSplitDesc& d = splits[i]; any_split = true;
        d.V = r.V; d.E = r.E; d.L = r.L; d.H = r.H[0]; d.S = r.S; d.Sv = r.Sv; d.Q = r.Q; d.lr = r.lr;
        const int floats = mvm_split_layout(d, r.kc);   // r.kc > 0: the streamed member's layout (its staging tiles)
        if (floats < 0 || floats > r.smem_floats) throw std::runtime_error("fuse: a split run's layout does not fit what the pass sized it for");
        st.smem_bytes = std::max(st.smem_bytes, floats * (int)sizeof(float));
        const int nslot = d.Q * d.S, vslot = d.Q * d.Sv, R = MVM_SPLIT_R;   // a member's partials, and a virtual share's
        d.we = param(r.wte); d.wh = param(r.head);
        for (int l = 0; l < r.L; ++l) {
          d.w[l] = param(r.w[l]); d.b[l] = param(r.b[l]);
          d.dW[l] = scratch((size_t)nslot * (l ? d.HR : d.E) * d.HP); d.db[l] = scratch((size_t)nslot * d.HP);
          if (l + 1 < r.L) d.XH[l] = scratch((size_t)d.Q * R * d.HR);   // zeroed: the pad columns past H are never written
        }
        for (int k = 0; k < 2; ++k) d.XD[k] = r.L > 1 ? scratch((size_t)vslot * R * d.HR) : nullptr;
        d.P = scratch((size_t)vslot * R * d.VP); d.dWh = scratch((size_t)nslot * d.HP * d.VP); d.dX0 = scratch((size_t)vslot * R * d.E);
        d.loss_part = scratch(d.Q);
        d.ctl = reinterpret_cast<unsigned*>(scratch(mvm_gang::line_words(nslot)));   // the gang's control words (mvm_gang::cross), zeroed once (cudaMalloc: 256-byte aligned)
        continue;
      }
      MvmFusedDesc& d = descs[i];
      d.R = r.slice_rows; d.V = r.V; d.E = r.E; d.L = r.L; d.nsl = r.nsl; d.T = r.T; d.lr = r.lr; d.groups = r.groups;
      for (int l = 0; l < r.L; ++l) d.H[l] = r.H[l];
      st.smem_bytes = std::max(st.smem_bytes, mvm_fmlp_layout(d, r.smem_floats) * (int)sizeof(float));
      d.we = param(r.wte); d.wh = param(r.head);
      const int ngr = r.T / MVM_FMLP_GRANULE;   // the step's gradient granules: one partial each (mvm_fused_mlp.h)
      d.ge = scratch((size_t)ngr * r.V * r.E);
      d.loss_part = scratch(ngr);
      for (int l = 0; l < r.L; ++l) {
        const int K = l ? r.H[l - 1] : r.E;
        d.w[l] = param(r.w[l]); d.b[l] = param(r.b[l]);
        d.gwld[l] = mvm_fmlp_pad16(r.H[l]);
        d.gw[l] = scratch((size_t)ngr * mvm_fmlp_pad16(K) * d.gwld[l]);
        d.gb[l] = scratch((size_t)ngr * r.H[l]);
      }
      d.ghld = mvm_fmlp_pad16(r.V);
      d.gh = scratch((size_t)ngr * mvm_fmlp_pad16(r.H[r.L - 1]) * d.ghld);
    }
    CK(cudaMalloc(&st.desc, descs.size() * sizeof(MvmFusedDesc)));
    CK(cudaMemcpy(st.desc, descs.data(), descs.size() * sizeof(MvmFusedDesc), cudaMemcpyHostToDevice));
    st.allocs.push_back(st.desc);
    if (any_split) {
      CK(cudaMalloc(&st.split, splits.size() * sizeof(MvmSplitDesc)));
      CK(cudaMemcpy(st.split, splits.data(), splits.size() * sizeof(MvmSplitDesc), cudaMemcpyHostToDevice));
      st.allocs.push_back(st.split);
    }
    if (any_grid) {
      CK(cudaMalloc(&st.grid, grids.size() * sizeof(MvmGridDesc)));
      CK(cudaMemcpy(st.grid, grids.data(), grids.size() * sizeof(MvmGridDesc), cudaMemcpyHostToDevice));
      st.allocs.push_back(st.grid);
    }
    // The batched form (mvm_fused_batch.h): a descriptor a batch, holding each run's model (the bundle's buffers), its steps' tokens, targets
    // and loss (a device pointer a step), its scratch, and the gang's control words. The pass batches only runs of one shape, Sv, step count
    // and learning rate (the descriptor holds one of each); checked again here, since another run's shape would compute garbage, not fail.
    const auto upload = [&](const auto& v) {
      using E = typename std::decay_t<decltype(v)>::value_type;
      E* p; CK(cudaMalloc(&p, std::max<size_t>(v.size(), 1) * sizeof(E))); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(E), cudaMemcpyHostToDevice));
      st.allocs.push_back((void*)p); return p;
    };
    const auto stream = [&](int b, int dtype, long long n, const char* what) {   // a step's tokens or targets (i32), or its loss pair (f32)
      if (b < 0 || b >= (int)bufs_.size() || bufs_[b].dtype != dtype || (bufs_[b].rows > 1 && bufs_[b].ld != bufs_[b].cols) || (long long)bufs_[b].rows * bufs_[b].cols < n)
        throw std::runtime_error(std::string("fuse: batched ") + what + " buffer " + std::to_string(b) + " is not " + std::to_string(n) + " contiguous " + (dtype ? "i32" : "f32"));
      return bufs_[b].p;
    };
    std::vector<MvmBatchDesc> bdescs(batches.size());
    for (size_t k = 0; k < batches.size(); ++k) {
      const MvmFuseBatch& bt = batches[k];
      if (bt.runs.empty() || bt.steps.size() != bt.runs.size()) throw std::logic_error("fuse: batch " + std::to_string(k) + " has no runs, or step buffers for another run count");
      const MvmFuseRun& r0 = runs.at(bt.runs[0]);
      MvmBatchDesc& d = bdescs[k];
      d.R = (int)bt.runs.size(); d.E = r0.E; d.V = r0.V; d.H = r0.H[0]; d.L = r0.L; d.T = r0.T; d.members = r0.B; d.steps = r0.steps; d.lr = r0.lr;
      if (!mvm_fuse_batch_arith(r0, d)) throw std::logic_error("fuse: batch " + std::to_string(k) + "'s runs carry neither a split nor a grid arithmetic");
      const int floats = mvm_batch_layout(d);
      if (floats < 0 || floats > r0.smem_floats) throw std::runtime_error("fuse: a batch's layout does not fit what the pass sized it for");
      st.smem_bytes = std::max(st.smem_bytes, floats * (int)sizeof(float));
      const size_t TH = (size_t)d.T * d.H;
      std::vector<MvmBatchRun> hr(d.R);
      for (int i = 0; i < d.R; ++i) {
        const MvmFuseRun& r = runs.at(bt.runs[i]);
        bool same = r.batch && r.B == d.members && r.L == d.L && r.E == d.E && r.V == d.V && r.T == d.T && r.Sv == r0.Sv && r.P == r0.P && r.steps == d.steps && r.lr == d.lr &&
                    (int)bt.steps[i].size() == d.steps;
        for (int l = 0; l < r.L && same; ++l) same = r.H[l] == d.H;
        if (!same) throw std::logic_error("fuse: batch " + std::to_string(k) + ": run " + std::to_string(bt.runs[i]) + " is not of the batch's shape, arithmetic, step count and learning rate");
        MvmBatchRun& x = hr[i];
        x.emb = param(r.wte); x.wh = param(r.head);
        for (int l = 0; l < r.L; ++l) { x.w[l] = param(r.w[l]); x.b[l] = param(r.b[l]); x.y[l] = scratch(TH); x.g[l] = scratch(TH); }
        x.x0 = scratch((size_t)d.T * d.E); x.dlog = scratch((size_t)d.T * d.VP); x.dx0 = scratch((size_t)d.T * d.E); x.rloss = scratch(d.T);
        for (float*& z : x.dz) z = scratch(TH);
        std::vector<const int32_t*> ids(d.steps), tgt(d.steps); std::vector<float*> loss(d.steps);
        for (int s = 0; s < d.steps; ++s) {
          ids[s] = static_cast<const int32_t*>(stream(bt.steps[i][s][0], 1, d.T, "tokens"));
          tgt[s] = static_cast<const int32_t*>(stream(bt.steps[i][s][1], 1, d.T, "targets"));
          loss[s] = static_cast<float*>(stream(bt.steps[i][s][2], 0, 2, "loss"));
        }
        x.ids = upload(ids); x.tgt = upload(tgt); x.loss = upload(loss);
      }
      d.runs = upload(hr);
      d.ctl = reinterpret_cast<unsigned*>(scratch(mvm_gang::line_words(d.members)));   // the gang's control words (mvm_gang::cross), zeroed once (cudaMalloc: 256-byte aligned)
      if (d.gHp > 0) {   // the grid sums' walks, one word a piece (mvm_batch_grid_table)
        std::vector<unsigned> tab((size_t)mvm_batch_grid_words(d));
        tab.resize((size_t)mvm_batch_grid_table(d, tab.data()));
        d.gtab = upload(tab);
      }
      KLOG(2, "fuse: batch %zu: %d runs in a gang of %d, %d steps, arithmetic %s, forward tiles %d x %d, backward blocks of %d rows (%s), %.2f MB of scratch, %d bytes of shared memory a member", k, d.R,
           d.members, d.steps, (d.Sv ? std::to_string(d.Sv) + " virtual shares (split)" : "grid at " + std::to_string(r0.P) + " members").c_str(), 16 * d.fm, d.nw, d.kw, d.bsplit ? "split" : "fused", (double)d.R * ((2.0 * d.L + 3) * TH + 2.0 * d.T * d.E + (double)d.T * d.VP + d.T) * 4 / 1e6, floats * 4);
    }
    if (!bdescs.empty()) st.batch = upload(bdescs);
    return st;
  }

  // One descriptor per fused GPT run (mvm_fused_gpt.h; a run's index is its members' `desc` argument), with the run's scratch, its gang's
  // barrier words and its steps' loss pointers, and the shared memory its members need. Its buffers are device-allocated already (prepare()):
  // a parameter, gradient or moment that is not contiguous f32, or a stream that is not i32, is refused here, loudly. A run in F's 2-D form
  // (plan.f2d[i].G > 0) also gets F's descriptor (mvm_f2d::build over the same buffers), which its kernel takes by value. zbw: a fused-step
  // run on ZBW's instance.
  // upd_ctl: each one-block run's helper-update control words (mvm_fgpt1.cuh's upd), reset before the timed launches; helpers, the update's blocks
  // beside each one-block run (MVM_FGPT1_HELPERS, 16 unless set: on the A100 bed 16 read 1% under 8; 0 runs the update on the run's own block)
  struct FgptState {
    MvmFgptDesc* desc = nullptr; std::vector<mvm_f2d::SDesc> f2d; std::vector<void*> allocs; int smem_bytes = 0, smem1_bytes = 0; std::vector<char> zbw;
    std::vector<unsigned*> upd_ctl; int helpers = 0;
  };
  FgptState make_fgpt_state(const MvmFuseGptPlan& plan) {
    FgptState st;
    std::vector<MvmFgptDesc> descs(plan.runs.size());
    st.upd_ctl.assign(plan.runs.size(), nullptr);
    {   // the helper update (mvm_fgpt1.cuh's upd): each one-block run's update on helper blocks
      const char* hv = std::getenv("MVM_FGPT1_HELPERS");
      st.helpers = hv ? std::max(0, std::atoi(hv)) : 16;
    }
    st.f2d.resize(plan.runs.size());
    const auto alloc = [&](size_t bytes) { void* p; CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 0, bytes)); st.allocs.push_back(p); return p; };
    const auto need = [&](int b, int dtype, const char* what) {
      if (b < 0 || b >= (int)bufs_.size() || bufs_[b].dtype != dtype || (bufs_[b].rows > 1 && bufs_[b].ld != bufs_[b].cols))
        throw std::runtime_error(std::string("fuse gpt: ") + what + " buffer " + std::to_string(b) + " is not contiguous " + (dtype ? "i32" : "f32"));
      return bufs_[b].p;
    };
    for (size_t i = 0; i < plan.runs.size(); ++i) {
      const MvmFuseGptRun& run = plan.runs[i];
      const MvmGptStep& s = run.shape;
      MvmFgptDesc& d = descs[i];
      d.N = s.N; d.T = s.T; d.B = s.B; d.d = s.d; d.H = s.H; d.hd = s.hd; d.F = s.F; d.V = s.V; d.L = s.L; d.G = plan.gang[i]; d.steps = run.steps;
      d.mm_numerics = s.mm_numerics; d.qb = mvm_fgpt_qblocks(s.B, s.H, s.T, d.G); d.rows = plan.rows[i];
      d.ln_eps = s.ln_eps; d.theta = s.theta; d.scale = s.scale; d.lr = s.lr; d.beta1 = s.beta1; d.beta2 = s.beta2; d.adam_eps = s.adam_eps; d.max_norm = s.max_norm;
      // the parameters in slot order (mvm_fused_gpt.h), each with its gradient, moments and weight decay: the run's AdamW updates
      std::vector<int> slot(mvm_fgpt_nparams(s.L), -1);
      slot[0] = s.wte;
      for (int l = 0; l < s.L; ++l) {
        const MvmGptLayer& P = s.layers[l];
        const int role[MVM_FGPT_PER_LAYER] = {P.n1g, P.n1b, P.wq, P.bq, P.wk, P.bk, P.wv, P.bv, P.wo, P.bo, P.n2g, P.n2b, P.w1, P.b1, P.w2, P.b2};
        for (int r = 0; r < MVM_FGPT_PER_LAYER; ++r) slot[mvm_fgpt_slot_layer(l, r)] = role[r];
      }
      slot[mvm_fgpt_slot_nfg(s.L)] = s.nfg; slot[mvm_fgpt_slot_nfb(s.L)] = s.nfb; slot[mvm_fgpt_slot_head(s.L)] = s.head;
      for (size_t k = 0; k < slot.size(); ++k) {
        const MvmGptUpdate* u = nullptr;
        for (const MvmGptUpdate& x : s.updates) if (x.p == slot[k]) u = &x;
        if (!u) throw std::runtime_error("fuse gpt: parameter buffer " + std::to_string(slot[k]) + " of a fused run has no AdamW update");
        d.p[k] = static_cast<float*>(need(u->p, 0, "a parameter")); d.g[k] = static_cast<float*>(need(u->g, 0, "a gradient"));
        d.m[k] = static_cast<float*>(need(u->m, 0, "a first moment")); d.v[k] = static_cast<float*>(need(u->v, 0, "a second moment"));
        d.wd[k] = u->wd; d.prows[k] = bufs_[u->p].rows; d.pcols[k] = bufs_[u->p].cols;
      }
      d.ids_src = static_cast<const int32_t*>(need(s.ids_src, 1, "the token stream")); d.tgt_src = static_cast<const int32_t*>(need(s.tgt_src, 1, "the target stream"));
      d.ids_stride = s.ids_stride; d.ids_off = s.ids_off; d.tgt_stride = s.tgt_stride; d.tgt_off = s.tgt_off;
      d.ids_len = (long long)bufs_[s.ids_src].rows * bufs_[s.ids_src].cols; d.tgt_len = (long long)bufs_[s.tgt_src].rows * bufs_[s.tgt_src].cols;
      d.step = static_cast<int32_t*>(need(s.step_buf, 1, "the step counter"));
      std::vector<float*> lp(run.steps);
      for (int k = 0; k < run.steps; ++k) lp[k] = static_cast<float*>(need(run.losses[k], 0, "a loss"));
      float** dl = static_cast<float**>(alloc(lp.size() * sizeof(float*)));
      CK(cudaMemcpy(dl, lp.data(), lp.size() * sizeof(float*), cudaMemcpyHostToDevice));
      d.loss = dl;
      if (plan.one_block[i]) {   // the one-block form: no gang; its bias corrections, the helper update's control words, and one block's shared memory
        d.bc = static_cast<float*>(alloc(2 * (size_t)run.steps * sizeof(float)));   // filled by its block at the run's start
        d.ctl = static_cast<unsigned*>(alloc(32 * sizeof(unsigned)));                   // the helper update's mode, go, done and coef
        st.upd_ctl[i] = d.ctl;
        const long long floats = mvm_fgpt1_layout(d.N, d.T, d.d, d.H, d.F, d.V, d.L, MVM_FGPT1_EXEC_FAST).total;
        st.smem1_bytes = std::max(st.smem1_bytes, (int)(floats * (long long)sizeof(float)));
        KLOG(2, "fuse gpt: run %zu: the one-block form (relaxed numerics mask %d), %d steps, its whole state in %lld floats (%lld bytes) of one block's "
                "shared memory", i, MVM_FGPT1_EXEC_FAST, d.steps, floats, floats * (long long)sizeof(float));
        continue;
      }
      if (plan.f2d[i].G) {   // F's 2-D form: its descriptor and buffers from the run's (the fused step's scratch, gang and chunk fields stay unused)
        const mvm_f2d::Fit& f = plan.f2d[i];
        size_t bytes = 0;
        const auto alloc_floats = [&](size_t n) { bytes += std::max<size_t>(n, 1) * sizeof(float); return static_cast<float*>(alloc(std::max<size_t>(n, 1) * sizeof(float))); };
        if (const char* why = mvm_f2d::build(d, f, alloc_floats, st.f2d[i])) throw std::runtime_error(std::string("fuse gpt: run ") + std::to_string(i) + " in F's form: " + why);
        st.smem_bytes = std::max(st.smem_bytes, (int)(f.smem_floats * sizeof(float)));
        KLOG(2, "fuse gpt: run %zu: F's 2-D form, a gang of %d (attention %d x %d, MLP %d x %d), %d steps, %d query block(s) a unit, %d vocabulary shards, %.2f MB of buffers, %lld bytes of shared memory a member",
             i, d.G, f.gta, f.gca, f.gtm, f.gcm, d.steps, f.qb, f.nv, bytes / 1e6, f.smem_floats * (long long)sizeof(float));
        continue;
      }
      const MvmFgptScratch sc = mvm_fgpt_scratch(d.N, d.T, d.d, d.H, d.F, d.V, d.L, d.G, d.qb);
      d.xch = static_cast<float*>(alloc((size_t)sc.total * sizeof(float))); d.xch_floats = (size_t)sc.total;
      d.bc = static_cast<float*>(alloc(2 * (size_t)run.steps * sizeof(float)));     // filled by member 0 at the run's start
      d.ctl = static_cast<unsigned*>(alloc(mvm_gang::line_words(MVM_FGPT_MAX_GANG) * sizeof(unsigned)));   // the gang's control words (mvm_gang::cross), zeroed once (cudaMalloc: 256-byte aligned)
      const int smem = (int)(mvm_fgpt_smem_floats(d.N, d.T, d.d, d.H, d.F, d.V, d.G, d.qb, d.rows) * sizeof(float));
      st.smem_bytes = std::max(st.smem_bytes, smem);
      KLOG(2, "fuse gpt: run %zu: a gang of %d, %d steps, attention in %d query block(s) a unit, rows in chunks of %d of %d, %.2f MB of scratch, %d bytes of shared memory a member", i, d.G, d.steps, d.qb, d.rows, d.N, sc.total * 4.0 / 1e6, smem);
    }
    // ZBW's instance (mvm_fgpt_zbw_of, which that instance plans with) for a run it engages, once the launch's shared memory is known: the
    // instance without ZBW's code serves every other run, so a run ZBW leaves alone runs the step as it was.
    st.zbw.assign(descs.size(), 0);
    for (size_t i = 0; i < descs.size(); ++i) {
      if (plan.f2d[i].G) { KLOG(2, "fuse gpt: run %zu: F's 2-D form (its own kernel, no ZBW)", i); continue; }
      if (plan.one_block[i]) { KLOG(2, "fuse gpt: run %zu: the one-block form (its own kernel, no ZBW)", i); continue; }
      const MvmFgptDesc& d = descs[i];
      const MvmFgptZbwRun z = mvm_fgpt_zbw_of(d.N, d.T, d.d, d.H, d.F, d.G, d.qb, d.rows, st.smem_bytes / (int)sizeof(float));
      st.zbw[i] = z.any() ? 1 : 0;
      KLOG(2, "fuse gpt: run %zu: ZBW %s (the MLP's doers: %s; the attention's: %s)", i, z.any() ? "engaged, on its instance" : "off, on the step's instance without its code",
           z.mlp.zg > 0 ? (std::to_string(z.mlp.doers) + " for " + std::to_string(z.mlp.zg) + " owner(s) a pass").c_str() : "none", z.attn ? "yes" : "no");
    }
    if (!descs.empty()) {
      st.desc = static_cast<MvmFgptDesc*>(alloc(descs.size() * sizeof(MvmFgptDesc)));
      CK(cudaMemcpy(st.desc, descs.data(), descs.size() * sizeof(MvmFgptDesc), cudaMemcpyHostToDevice));
    }
    return st;
  }

  // FUSED GPT RUNS: a bundle that the GPT pass rewrote into gangs and nothing else (mvm_fuse_gpt.h) runs as one launch of the fused kernel
  // per run (mvm_fgpt_kernel.cuh), in place of start()/submit() or run_static. Runs are separate components, so their order is free. A
  // gang's members wait on each other at barriers, so all G of its blocks must be resident at once: refused here when the device cannot hold
  // them. Fills the same MvmResult as run_static (no per-task trace; final versions from the plan).
  const MvmResult& run_fgpt(const MvmBundle& b, const MvmFuseGptPlan& plan) {
    if (started_ || finished_) throw std::logic_error("run_fgpt replaces start()");
    std::string why = b.validate(nbufs(), cfg_.num_vars, 0);
    if (!why.empty()) { res_.err_code = -1; last_error_ = why; throw std::runtime_error("bundle rejected: " + why); }
    if (plan.runs.empty()) throw std::logic_error("run_fgpt: the plan fused no run");
    const int member_op = mvm_op_by_name("fgpt_step");
    int members = 0, gmax = 0;
    for (int g : plan.gang) { members += g; gmax = std::max(gmax, g); }
    for (size_t t = 0; t < b.tasks.size(); ++t)   // the slots below are the members' task indices: the pass appends each run's members in order
      if (b.tasks[t].op_id != member_op) throw std::logic_error("run_fgpt: task " + std::to_string(t) + " is not a fused GPT member (the bundle must be its gangs alone)");
    if ((int)b.tasks.size() != members) throw std::logic_error("run_fgpt: " + std::to_string(b.tasks.size()) + " tasks for gangs of " + std::to_string(members) + " members");
    prepare();   // buffers and the host mirrors are in place before anything launches
    FgptState st = make_fgpt_state(plan);
    // Each run's gang crosses in the form its size takes (mvm_gang::counted_form), on its program's instance for that form: the fused step's,
    // with ZBW's code only where ZBW's plan engages (FgptState::zbw), or F's for a run in F's form. Each crossing site then carries one form's
    // code, and a run ZBW leaves alone runs none of its code. The instances share the arena, so each one the plan uses gets the same attribute
    // and check. Instance k: the fused step's (counted) + 2 (ZBW) for k < 4; F's 4 + (counted).
    const auto instance = [&](size_t r) {
      if (plan.one_block[r]) return 6;   // the one-block form's own kernel
      const int c = mvm_gang::counted_form(plan.gang[r]) ? 1 : 0;
      return plan.f2d[r].G ? 4 + c : c + (st.zbw[r] ? 2 : 0);
    };
    bool form_used[7] = {false, false, false, false, false, false, false};
    for (size_t r = 0; r < plan.runs.size(); ++r) form_used[instance(r)] = true;
    const void* const form_kernel[6] = {reinterpret_cast<const void*>(mvm_fgpt_kernel<false, false>), reinterpret_cast<const void*>(mvm_fgpt_kernel<true, false>),
                                        reinterpret_cast<const void*>(mvm_fgpt_kernel<false, true>), reinterpret_cast<const void*>(mvm_fgpt_kernel<true, true>),
                                        reinterpret_cast<const void*>(mvm_f2d_kernel<false>), reinterpret_cast<const void*>(mvm_f2d_kernel<true>)};
    const int form_threads[6] = {MVM_FGPT_NT, MVM_FGPT_NT, MVM_FGPT_NT, MVM_FGPT_NT, MVM_F2D_NT, MVM_F2D_NT};
    int occ = -1;
    for (int f = 0; f < 6; ++f) {
      if (!form_used[f]) continue;
      CK(cudaFuncSetAttribute(form_kernel[f], cudaFuncAttributeMaxDynamicSharedMemorySize, st.smem_bytes));
      int o = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, form_kernel[f], form_threads[f], st.smem_bytes));
      occ = occ < 0 ? o : std::min(occ, o);
    }
    int gang_max = 0;   // the gangs' residency: a one-block run is one block, checked on its own below
    for (size_t r = 0; r < plan.runs.size(); ++r) if (!plan.one_block[r]) gang_max = std::max(gang_max, plan.gang[r]);
    if (gang_max > 0 && (long long)occ * sms_ < gang_max)
      throw std::runtime_error("fused GPT kernel: a gang of " + std::to_string(gang_max) + " needs that many resident blocks; " + std::to_string(sms_) + " SMs hold " +
                               std::to_string(occ) + " each with " + std::to_string(st.smem_bytes) + " bytes of shared memory");
    if (form_used[6]) {
      CK(cudaFuncSetAttribute(reinterpret_cast<const void*>(mvm_fgpt1_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, st.smem1_bytes));
      int o = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, reinterpret_cast<const void*>(mvm_fgpt1_kernel), MVM_FGPT_NT, st.smem1_bytes));
      if (o < 1) throw std::runtime_error("fused GPT kernel: the one-block form cannot be resident with " + std::to_string(st.smem1_bytes) + " bytes of shared memory");
      if (st.helpers > o * sms_ - 1) st.helpers = std::max(0, o * sms_ - 1);   // every block of a run resident at once (the helpers wait on block 0)
      KLOG(2, "fused GPT kernel: one-block runs with %d update helper block(s)%s", st.helpers, st.helpers ? ", the update on them" : " (the update on the run's own block)");
    }
    MvmCounters* dcnt; CK(cudaMalloc(&dcnt, sizeof(MvmCounters))); CK(cudaMemset(dcnt, 0, sizeof(MvmCounters)));
    CK(cudaStreamSynchronize(0));
    const bool lines = form_used[0] || form_used[2] || form_used[4], counted = form_used[1] || form_used[3] || form_used[5];
    KLOG(2, "fused GPT kernel: %zu run(s), %d members, %d resident blocks/SM, %d bytes of dynamic shared memory a member; %s; crossings on %s; %s",
         plan.runs.size(), members, occ, st.smem_bytes,
         (form_used[0] || form_used[1] || form_used[2] || form_used[3]) && (form_used[4] || form_used[5]) ? "the fused step's form and F's" : form_used[4] || form_used[5] ? "F's 2-D form" : "the fused step's form",
         lines && counted ? "both forms (each gang's own)" : counted ? "the counted barrier" : "the flag lines",
         (form_used[2] || form_used[3]) && (form_used[0] || form_used[1]) ? "ZBW's instance for the runs it engages, the plain step for the rest"
                                                                           : form_used[2] || form_used[3] ? "every fused-step run on ZBW's instance" : "no run on ZBW's instance");
    // A gang's members (the fused step's and F's 2-D form's) and a one-block run's helpers wait on one another, so their launches are cooperative: a GPU that cannot hold every
    // block at once refuses the launch (cudaErrorCooperativeLaunchTooLarge), where a plain launch could hang beside a co-tenant or under MPS.
    // The residency checks above hold either way. MVM_FGPT_COOP=0 launches them plainly, for the A/B of the attribute's cost.
    const char* const coop_env = std::getenv("MVM_FGPT_COOP");
    const bool coop = !(coop_env && std::atoi(coop_env) == 0);
    KLOG(2, "fused GPT kernel: gangs (the fused step's, F's) and one-block runs with helpers launch %s", coop ? "cooperatively" : "plainly (MVM_FGPT_COOP=0)");
    const auto launch_waiting = [&](auto kern, int blocks, int threads, int smem, auto... args) {   // args as the kernel takes them
      if (coop) {
        cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(blocks); lc.blockDim = dim3(threads); lc.dynamicSmemBytes = smem; lc.stream = stream_;
        cudaLaunchAttribute at = {}; at.id = cudaLaunchAttributeCooperative; at.val.cooperative = 1; lc.attrs = &at; lc.numAttrs = 1;
        CK(cudaLaunchKernelEx(&lc, kern, args...));
      } else {
        kern<<<blocks, threads, smem, stream_>>>(args...);
      }
    };
    // Every run's launch as it is timed. `warm` passes the warm-up descriptor instead (null, or F's gangless one), on which every block returns
    // at once: the same kernel, grid, shared memory and launch attribute, and no work.
    const auto launch_runs = [&](bool warm) {
      for (size_t r = 0, slot0 = 0; r < plan.runs.size(); slot0 += plan.gang[r], ++r) {
        const MvmFgptDesc* const dr = warm ? nullptr : st.desc + r;
        switch (instance(r)) {
          case 0: launch_waiting(mvm_fgpt_kernel<false, false>, plan.gang[r], MVM_FGPT_NT, st.smem_bytes, dr, dcnt, S_.herr, (int)slot0); break;
          case 1: launch_waiting(mvm_fgpt_kernel<true, false>, plan.gang[r], MVM_FGPT_NT, st.smem_bytes, dr, dcnt, S_.herr, (int)slot0); break;
          case 2: launch_waiting(mvm_fgpt_kernel<false, true>, plan.gang[r], MVM_FGPT_NT, st.smem_bytes, dr, dcnt, S_.herr, (int)slot0); break;
          case 3: launch_waiting(mvm_fgpt_kernel<true, true>, plan.gang[r], MVM_FGPT_NT, st.smem_bytes, dr, dcnt, S_.herr, (int)slot0); break;
          case 6:
            if (st.helpers) launch_waiting(mvm_fgpt1_kernel, 1 + st.helpers, MVM_FGPT_NT, st.smem1_bytes, dr, dcnt, S_.herr, (int)slot0);
            else mvm_fgpt1_kernel<<<1, MVM_FGPT_NT, st.smem1_bytes, stream_>>>(dr, dcnt, S_.herr, (int)slot0);   // one block waits on nothing
            break;
          default: {   // F's form: its descriptor by value, with the run's error word, which every crossing reads
            mvm_f2d::SDesc fd = warm ? mvm_f2d::SDesc{} : st.f2d[r];
            if (!warm) mvm_f2d::set_abort(fd, &dcnt->err_code);
            if (instance(r) == 5) launch_waiting(mvm_f2d_kernel<true>, plan.gang[r], MVM_F2D_NT, st.smem_bytes, fd, dcnt, S_.herr, (int)slot0);
            else launch_waiting(mvm_f2d_kernel<false>, plan.gang[r], MVM_F2D_NT, st.smem_bytes, fd, dcnt, S_.herr, (int)slot0);
          }
        }
      }
    };
    // The warm-ups, twice, so the timed window holds no launch's one-time cost. The first, before the clock warm-up, takes each kernel's first
    // launch (its code load) and each launch path's first use, a cooperative one's included. The second, after it, makes each timed launch follow
    // a launch of its own kind, not the clock warm-up's plain ones.
    launch_runs(true);
    CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream_));
    for (unsigned* c : st.upd_ctl)   // the helper update's words at zero before each launch: go and done count this launch's steps from 0
      if (c) CK(cudaMemsetAsync(c, 0, 32 * sizeof(unsigned), stream_));
    CK(cudaStreamSynchronize(stream_));
    clock_warmup();
    launch_runs(true);
    CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream_));
    CK(cudaEventRecord(e0_, stream_));
    launch_runs(false);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1_, stream_));
    started_ = true;
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
    MvmCounters c; CK(cudaMemcpy(&c, dcnt, sizeof c, cudaMemcpyDeviceToHost));
    res_.err_code = c.err_code; res_.err_slot = c.err_slot; res_.err_a = c.err_a; res_.err_b = c.err_b;
    res_.admitted = members; res_.done = c.err_code ? 0 : members; res_.grid = gmax;
    CK(cudaEventElapsedTime(&res_.kernel_ms, e0_, e1_));
    res_.version.assign(cfg_.num_vars, 0);
    CK(cudaMemcpy(res_.version.data(), S_.version, cfg_.num_vars * 4, cudaMemcpyDeviceToHost));
    for (const MvmTaskDesc& d : b.tasks) if (d.pub_var >= 0) res_.version[d.pub_var] = std::max(res_.version[d.pub_var], d.pub_ver);   // member 0 carries its run's last version
    finished_ = true;
    for (size_t i = 0; i < st.upd_ctl.size(); ++i) {   // the helper update: the path each one-block run took (its block 0 chooses: mvm_fgpt1.cuh's upd)
      if (!st.upd_ctl[i] || !st.helpers) continue;
      unsigned mode = 0; CK(cudaMemcpy(&mode, st.upd_ctl[i] + mvm_fgpt1::upd::MODE, sizeof mode, cudaMemcpyDeviceToHost));
      if (mode == mvm_fgpt1::upd::kOn)
        KLOG(2, "fused GPT kernel: run %zu: its update on its %d helper blocks%s", i, st.helpers, res_.err_code ? ", until the run faulted (its error word)" : "");
      else if (mode == mvm_fgpt1::upd::kOff)
        KLOG(1, "fused GPT kernel: run %zu: its update on its own block, not its %d helpers: declined (parameters not 16-byte vectors, over 12 float4 "
                "groups a thread, a moment ring under its size, or slices past a helper's shared memory)", i, st.helpers);
      else KLOG(1, "fused GPT kernel: run %zu: the run faulted before its block chose where its update runs (mode word %u)", i, mode);
    }
    CK(cudaFree(dcnt));
    for (void* p : st.allocs) CK(cudaFree(p));
    if (res_.ok()) KLOG(2, "fused GPT run done: %d members in %.3f ms", members, res_.kernel_ms);
    return res_;
  }

  // STATIC SCHEDULE: compile `b` (mvm_static.h) into one program per worker and run them all in one launch (mvm_static.cuh), instead of start()/submit():
  // no admission, no ready queue, no per-edge atomics. Fills the same MvmResult (no per-task trace; final versions from the plan).
  struct StaticOptions { int workers = 0; double comm_us = 1.5; const char* trace_csv = nullptr; bool spin_idle = false, partition = true, fuse_phases = false; const std::vector<MvmFuseRun>* fused = nullptr; const std::vector<MvmFuseBatch>* batches = nullptr; MvmFuseEntry entry; };
  const MvmResult& run_static(const MvmBundle& b, const StaticOptions& o) {
    if (started_ || finished_) throw std::logic_error("run_static replaces start()");
    std::string why = b.validate(nbufs(), cfg_.num_vars, 0);
    if (!why.empty()) { res_.err_code = -1; last_error_ = why; throw std::runtime_error("bundle rejected: " + why); }
    prepare();   // buffers and the host mirrors are in place before anything launches
    MvmSConfig sc; sc.workers = o.workers ? std::min(o.workers, grid_) : grid_; sc.comm_us = o.comm_us; sc.partition = o.partition;
    const MvmSOpLookup op_of = [](int id) { const MvmOpInfo* i = mvm_op(id); MvmSOpInfo v; if (i) { v.arity = i->arity; v.types = i->types; mvm_static_gang_args(i->sig, v); } return v; };
    const auto t0 = std::chrono::steady_clock::now();
    const MvmSCostFn cost_fn = mvm_static_cost;
    const MvmSPlan plan = mvm_static_plan(b.tasks, b.succ, b.args, sc, op_of, cost_fn);
    const MvmSLayout layout = mvm_static_layout(plan);
    int nrec = 0, used = 0;
    std::vector<int> offs(sc.workers + 1, 0);
    for (int w = 0; w < sc.workers; ++w) { offs[w] = nrec; nrec += layout.nrecs[w]; used += layout.nrecs[w] > 0; }
    offs[sc.workers] = nrec;
    std::vector<MvmSRec> host(nrec);
    for (int w = 0; w < sc.workers; ++w) if (layout.nrecs[w]) mvm_static_emit(plan, layout, w, b.tasks, b.args, bufs_.data(), nbufs(), op_of, host.data() + offs[w]);
    KLOG(2, "static schedule: %d tasks -> %d records on %d/%d workers, %d of %d edges cross workers, %d records publish; runs partitioned: %s; plan %.1f us, work %.1f us (compiled in %.3f s)", plan.nt, nrec, used, sc.workers, plan.cross_edges, plan.edges, plan.publishing, plan.partitioned ? "yes" : "no", plan.makespan_us, plan.total_work_us,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    FusedState fused;
    if (o.fused && !o.fused->empty()) fused = make_fused_state(*o.fused, o.batches ? *o.batches : std::vector<MvmFuseBatch>{});   // the buffers' device pointers exist since prepare()
    for (size_t t = 0; t < b.tasks.size(); ++t) {   // a batched member names its batch, its gang's size and its step: checked here, before the device reads past a table
      if (b.tasks[t].op_id != MVM_OP_fmlp_batch) continue;
      const int32_t* a = b.args.data() + b.tasks[t].args_off;
      const int nb = o.batches ? (int)o.batches->size() : 0;
      if (!fused.batch || a[0] < 0 || a[0] >= nb) throw std::logic_error("static kernel: task " + std::to_string(t) + " names batch " + std::to_string(a[0]) + " of the " + std::to_string(nb) + " built");
      const MvmFuseRun& r0 = o.fused->at((*o.batches)[a[0]].runs.at(0));
      if (a[3] != r0.B || a[4] < 0 || a[4] >= r0.steps)
        throw std::logic_error("static kernel: task " + std::to_string(t) + " is a member of " + std::to_string(a[3]) + " at step " + std::to_string(a[4]) + "; its batch's gang has " + std::to_string(r0.B) + " members and " + std::to_string(r0.steps) + " steps");
    }
    const size_t static_smem = std::max(dyn_smem_, (size_t)fused.smem_bytes);
    long long* dphases = nullptr;
    if (o.fuse_phases && fused.desc) { const size_t bytes = (size_t)sc.workers * MVM_FMLP_NPHASES * sizeof(long long); CK(cudaMalloc(&dphases, bytes)); CK(cudaMemset(dphases, 0, bytes)); }
    MvmStaticDev S{};
    S.fuse_phases = dphases;
    MvmSRec* drecs; int* doffs; int* dprog; MvmCounters* dcnt; unsigned long long* dtrace = nullptr; int* dfinished;
    CK(cudaMalloc(&drecs, (size_t)nrec * sizeof(MvmSRec))); CK(cudaMemcpy(drecs, host.data(), (size_t)nrec * sizeof(MvmSRec), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&doffs, offs.size() * sizeof(int))); CK(cudaMemcpy(doffs, offs.data(), offs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dprog, (size_t)sc.workers * MVM_S_LINE_WORDS * sizeof(int))); CK(cudaMemset(dprog, 0, (size_t)sc.workers * MVM_S_LINE_WORDS * sizeof(int)));
    CK(cudaMalloc(&dcnt, sizeof(MvmCounters))); CK(cudaMemset(dcnt, 0, sizeof(MvmCounters)));
    if (o.trace_csv) { CK(cudaMalloc(&dtrace, (size_t)nrec * MVM_S_TRACE_WORDS * 8)); CK(cudaMemset(dtrace, 0, (size_t)nrec * MVM_S_TRACE_WORDS * 8)); }
    CK(cudaMalloc(&dfinished, sizeof(int))); CK(cudaMemset(dfinished, 0, sizeof(int)));
    S.recs = drecs; S.offs = doffs; S.progress = dprog; S.nworkers = sc.workers; S.nactive = used; S.spin_idle = o.spin_idle; S.finished = dfinished; S.timeout_ns = cfg_.timeout_us * 1000ull; S.cnt = dcnt; S.herr = S_.herr; S.fused = fused.desc; S.split = fused.split; S.grid = fused.grid; S.batch = fused.batch; S.trace = dtrace; S.trace_on = dtrace != nullptr;
    // The entry the pass chose (mvm_fuse_entry): every fused form runs on an entry of its own, and the generic one refuses the fused ops, so
    // a fused task bound for it is a host bug, caught here before any launch. The fused budget subtracted the generic entry's static shared
    // memory (fused_smem_floats), so the chosen entry may hold no more than that.
    const MvmStaticEntryFn entry = mvm_static_entry_fn(o.entry);
    if (!entry) throw std::logic_error("static kernel: no entry for " + mvm_fuse_entry_name(o.entry));
    if (o.entry.kind == MvmFuseEntry::generic)
      for (size_t t = 0; t < b.tasks.size(); ++t)
        if (b.tasks[t].op_id == MVM_OP_fmlp_slice || b.tasks[t].op_id == MVM_OP_fmlp_sgd || b.tasks[t].op_id == MVM_OP_fmlp_split || b.tasks[t].op_id == MVM_OP_fmlp_grid || b.tasks[t].op_id == MVM_OP_fmlp_batch)
          throw std::logic_error("static kernel: task " + std::to_string(t) + " is a fused op (" + mvm_op(b.tasks[t].op_id)->name + "), which the generic entry refuses");
    if (o.entry.kind == MvmFuseEntry::split && o.fused)   // the split entry runs one member, resident or streamed: each run's descriptor is laid out for it
      for (size_t i = 0; i < o.fused->size(); ++i)
        if ((*o.fused)[i].split && ((*o.fused)[i].kc > 0) != o.entry.stream)
          throw std::logic_error("static kernel: split run " + std::to_string(i) + (o.entry.stream ? " holds its weights" : " streams its weights") + ", entry " + mvm_fuse_entry_name(o.entry));
    for (size_t t = 0; t < b.tasks.size(); ++t)   // a fused GPT member runs only as its gang's block of the fused GPT kernel (run_fgpt): no entry's dispatch runs it
      if (b.tasks[t].op_id == MVM_OP_fgpt_step)
        throw std::logic_error("static kernel: task " + std::to_string(t) + " is a fused GPT member (fgpt_step), which only the fused GPT kernel runs (run_fgpt)");
    {
      cudaFuncAttributes fe, fg; CK(cudaFuncGetAttributes(&fe, entry)); CK(cudaFuncGetAttributes(&fg, mvm_static_kernel));
      if (fe.sharedSizeBytes > fg.sharedSizeBytes) throw std::logic_error("static kernel: entry " + mvm_fuse_entry_name(o.entry) + " holds " + std::to_string(fe.sharedSizeBytes) + " bytes of static shared memory, the generic one " + std::to_string(fg.sharedSizeBytes) + ": the fused budget assumed at most the generic one's");
    }
    CK(cudaStreamSynchronize(0));
    if (static_smem > 0) CK(cudaFuncSetAttribute(entry, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)static_smem));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, entry, MVM_THREADS, static_smem));
    if (occ < 1) throw std::runtime_error("static kernel does not fit on an SM");
    // A static schedule assumes every worker resident at once: a worker that never starts would hang a gang at its barrier (which has no timeout).
    if ((long long)occ * sms_ < sc.workers)
      throw std::runtime_error("static kernel: entry " + mvm_fuse_entry_name(o.entry) + " holds " + std::to_string(occ) + " block(s) an SM at " + std::to_string(static_smem) + " bytes of dynamic shared memory, " +
                               std::to_string((long long)occ * sms_) + " resident in all, fewer than the plan's " + std::to_string(sc.workers) + " workers");
    KLOG(2, "static kernel: entry %s, %d resident blocks/SM, %zu bytes of dynamic shared memory, launching %d workers", mvm_fuse_entry_name(o.entry).c_str(), occ, static_smem, sc.workers);
    // Every worker of a static schedule waits on others (a gang's crossings, the schedule's task flags), so the launch is cooperative: a GPU that
    // cannot hold every worker at once refuses it (cudaErrorCooperativeLaunchTooLarge), where a plain launch could hang beside a co-tenant or
    // under MPS. The residency check above holds either way. MVM_FGPT_COOP=0 launches plainly, the fused GPT kernel's switch, so one A/B prices
    // both kinds of gang (a row runs one or the other).
    const char* const coop_env = std::getenv("MVM_FGPT_COOP");
    const bool coop = !(coop_env && std::atoi(coop_env) == 0);
    KLOG(2, "static kernel: workers launch %s", coop ? "cooperatively" : "plainly (MVM_FGPT_COOP=0)");
    const auto launch_workers = [&](const MvmStaticDev& d) {   // the schedule's one launch, as it is timed
      if (coop) {
        cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(sc.workers); lc.blockDim = dim3(MVM_THREADS); lc.dynamicSmemBytes = static_smem; lc.stream = stream_;
        cudaLaunchAttribute at = {}; at.id = cudaLaunchAttributeCooperative; at.val.cooperative = 1; lc.attrs = &at; lc.numAttrs = 1;
        CK(cudaLaunchKernelEx(&lc, entry, d));
      } else {
        entry<<<sc.workers, MVM_THREADS, static_smem, stream_>>>(d);
      }
      CK(cudaGetLastError());
    };
    // The warm-ups: the timed launch itself with nworkers = 0, on which every block returns at once (the same entry, grid, shared memory and
    // launch attribute, and no work), twice, so the timed window holds no launch's one-time cost. The first, before the clock warm-up, takes the
    // entry's first launch (its code load: ~1.5 ms before any record runs) and the launch path's first use, a cooperative one's included. The
    // second, after it, makes the timed launch follow a launch of its own kind, not the clock warm-up's plain ones.
    MvmStaticDev warm = S; warm.nworkers = 0;
    launch_workers(warm); CK(cudaStreamSynchronize(stream_));
    clock_warmup();
    launch_workers(warm); CK(cudaStreamSynchronize(stream_));
    CK(cudaEventRecord(e0_, stream_));
    launch_workers(S);
    CK(cudaEventRecord(e1_, stream_));
    started_ = true;
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
    MvmCounters c; CK(cudaMemcpy(&c, dcnt, sizeof c, cudaMemcpyDeviceToHost));
    res_.err_code = c.err_code; res_.err_slot = c.err_slot; res_.err_a = c.err_a; res_.err_b = c.err_b;
    res_.admitted = plan.nt; res_.done = c.err_code ? 0 : plan.nt; res_.grid = sc.workers;
    CK(cudaEventElapsedTime(&res_.kernel_ms, e0_, e1_));
    res_.version.assign(cfg_.num_vars, 0);
    CK(cudaMemcpy(res_.version.data(), S_.version, cfg_.num_vars * 4, cudaMemcpyDeviceToHost));
    for (const MvmTaskDesc& d : b.tasks) if (d.pub_var >= 0) res_.version[d.pub_var] = std::max(res_.version[d.pub_var], d.pub_ver);   // versions are ordering only here: the plan's final values
    finished_ = true;
    if (o.trace_csv) {
      std::vector<unsigned long long> tr((size_t)nrec * MVM_S_TRACE_WORDS);
      CK(cudaMemcpy(tr.data(), dtrace, tr.size() * 8, cudaMemcpyDeviceToHost));
      FILE* f = fopen(o.trace_csv, "w");
      if (!f) throw std::runtime_error(std::string("cannot open ") + o.trace_csv);
      fprintf(f, "worker,pc,task,op,start_ns,ready_ns,done_ns,published_ns,nwait,flags\n");
      unsigned long long t00 = ~0ull; for (size_t i = 0; i < tr.size(); i += MVM_S_TRACE_WORDS) if (tr[i]) t00 = std::min(t00, tr[i]);
      for (int w = 0; w < sc.workers; ++w) for (int k = offs[w]; k < offs[w + 1]; ++k) {
        const MvmSRec& r = host[k]; const unsigned long long* x = tr.data() + (size_t)k * MVM_S_TRACE_WORDS;
        fprintf(f, "%d,%d,%d,%s,%llu,%llu,%llu,%llu,%d,%d\n", w, k - offs[w], r.task, (r.flags & MVM_S_WAITONLY) ? "(wait)" : mvm_op(r.op)->name, x[0] - t00, x[1] - t00, x[2] - t00, x[3] - t00, r.nwait, r.flags);
      }
      fclose(f); KLOG(2, "static trace: %d records -> %s", nrec, o.trace_csv);
    }
    CK(cudaFree(drecs)); CK(cudaFree(doffs)); CK(cudaFree(dprog)); CK(cudaFree(dcnt)); CK(cudaFree(dfinished)); if (dtrace) CK(cudaFree(dtrace));
    for (void* p : fused.allocs) CK(cudaFree(p));
    if (dphases) {   // per-phase cycles of the slice bodies, summed over the workers that ran them, per slice task
      static const char* names[MVM_FMLP_NPHASES] = {"load weights+ids", "embedding gather", "forward GEMMs (+gelu)", "head GEMM", "softmax CE", "head backward GEMMs", "dZ gelu' (+sync)", "bias colsum", "gelu_f again (l>0)", "weight gradient GEMM", "activation gradient GEMM"};
      std::vector<long long> tot((size_t)sc.workers * MVM_FMLP_NPHASES);
      CK(cudaMemcpy(tot.data(), dphases, tot.size() * sizeof(long long), cudaMemcpyDeviceToHost)); CK(cudaFree(dphases));
      long long nslice = 0; for (const MvmTaskDesc& d : b.tasks) nslice += d.op_id == mvm_op_by_name("fmlp_slice");
      long long total = 0;
      for (int i = 0; i < MVM_FMLP_NPHASES; ++i) { long long sum = 0; for (int w = 0; w < sc.workers; ++w) sum += tot[(size_t)w * MVM_FMLP_NPHASES + i]; total += sum; if (sum && names[i]) KLOG(2, "fuse phase %-34s %8.0f cycles per slice", names[i], (double)sum / nslice); }
      KLOG(2, "fuse phases: %.0f cycles per slice in total, over %lld slices", (double)total / nslice, nslice);
    }
    if (res_.ok()) KLOG(2, "static run done: %d tasks in %.3f ms on %d workers", plan.nt, res_.kernel_ms, sc.workers);
    else KLOG(0, "DEVICE ERROR %s (%s, code %d): task %d, a=%d, b=%d", mvm_err_name(res_.err_code), mvm_err_text(res_.err_code), res_.err_code, res_.err_slot, res_.err_a, res_.err_b);
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

  const MvmResult& finish() {
    if (!started_) throw std::logic_error("finish before start()");
    if (finished_) return res_;
    close();
    CK(cudaStreamSynchronize(stream_));
    CK(cudaGetLastError());
    MvmCounters c; CK(cudaMemcpy(&c, S_.cnt, sizeof c, cudaMemcpyDeviceToHost));
    res_.err_code = c.err_code; res_.err_slot = c.err_slot; res_.err_a = c.err_a; res_.err_b = c.err_b;
    res_.admitted = c.admitted; res_.done = c.done; res_.grid = grid_;
    CK(cudaEventElapsedTime(&res_.kernel_ms, e0_, e1_));
    const int ns = std::min(c.admitted, cfg_.max_tasks);
    res_.tasks.resize(ns); res_.trace.resize(ns); res_.version.resize(cfg_.num_vars);
    if (ns) {
      CK(cudaMemcpy(res_.tasks.data(), S_.tasks, ns * sizeof(MvmTaskDesc), cudaMemcpyDeviceToHost));
      if (cfg_.trace_on) CK(cudaMemcpy(res_.trace.data(), S_.trace, ns * sizeof(MvmTrace), cudaMemcpyDeviceToHost));   // (else the zeroed vector stands)
    }
    CK(cudaMemcpy(res_.version.data(), S_.version, cfg_.num_vars * 4, cudaMemcpyDeviceToHost));
    if (cfg_.sched.on) {
      CK(cudaMemcpy(&res_.sched.dev, S_.sp.dev, sizeof(MvmSchedDev), cudaMemcpyDeviceToHost)); res_.sched.valid = true;
      res_.info.resize(ns);
      if (ns) CK(cudaMemcpy(res_.info.data(), S_.sp.info, ns * sizeof(MvmSlotInfo), cudaMemcpyDeviceToHost));
    }
    finished_ = true;
    if (res_.ok()) KLOG(2, "kernel done: %d/%d tasks in %.3f ms, %d records, %d ring wraps, %d backpressure waits", res_.done, res_.admitted, res_.kernel_ms, stats_.bundles, stats_.pads, stats_.backpressure_waits);
    else KLOG(0, "DEVICE ERROR %s (%s, code %d): slot %d, a=%d, b=%d; done %d/%d", mvm_err_name(res_.err_code), mvm_err_text(res_.err_code), res_.err_code, res_.err_slot, res_.err_a, res_.err_b, res_.done, res_.admitted);
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
           workers(), cfg_.live_cap, cfg_.max_tasks, 1 << MVM_FIFO_SHIFT);
  }
  void validate_sched(const MvmSchedParams& p) const {
    for (int c = 0; c < nc_; ++c) {
      const MvmClassParams& q = p.cls[c];
      if (q.weight < 1 || q.min_cta < 0 || q.max_cta < 0 || q.bw_cta < 0 || q.urgent_us < 0) throw std::invalid_argument("class " + std::to_string(c) + ": weight >= 1, min/max/bw caps and urgent_us >= 0");
      if (q.max_cta < q.min_cta) throw std::invalid_argument("class " + std::to_string(c) + ": max_cta < min_cta");
    }
    if (p.bw_limit_min < 0 || p.bw_limit_max < p.bw_limit_min) throw std::invalid_argument("bw_limit_min <= bw_limit_max");
  }
  void init_sched() {
    const MvmSchedCfg& sc = cfg_.sched;
    hp_ = (MvmSchedParams*)(ctl_ + 512); ht_ = (MvmSchedTele*)(ctl_ + 1024);
    S_.sp = MvmSchedPtrs{}; S_.sp.nc = nc_; S_.sp.on = sc.on ? 1 : 0; S_.sp.mode = sc.on ? sc.mode : MVM_SCHED_GLOBAL;
    if (!sc.on) return;
    validate_sched(sc.p);
    void* d;
    CK(cudaHostGetDevicePointer(&d, hp_, 0)); S_.sp.hp = (MvmSchedParams*)d;
    CK(cudaHostGetDevicePointer(&d, ht_, 0)); S_.sp.ht = (MvmSchedTele*)d;
    CK(cudaMalloc(&S_.sp.dev, sizeof(MvmSchedDev))); CK(cudaMemset(S_.sp.dev, 0, sizeof(MvmSchedDev)));
    CK(cudaMalloc(&S_.sp.info, (size_t)cfg_.max_tasks * sizeof(MvmSlotInfo))); CK(cudaMemset(S_.sp.info, 0, (size_t)cfg_.max_tasks * sizeof(MvmSlotInfo)));
    CK(cudaMalloc(&S_.sp.pv, (size_t)cfg_.num_vars * MVM_MAX_CLS * 4)); CK(cudaMemset(S_.sp.pv, 0, (size_t)cfg_.num_vars * MVM_MAX_CLS * 4));
    CK(cudaMalloc(&S_.sp.pubcls, (size_t)cfg_.num_vars * 4)); CK(cudaMemset(S_.sp.pubcls, 0, (size_t)cfg_.num_vars * 4));
    MvmController::write_params(hp_, sc.p);   // generation 1: the policy block's first tick re-reads the launch parameters
  }
  static uint16_t bf16_rne(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x40u : 0u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
  // dtype 2 (bf16): fill / values are rounded to bf16 on the host; a file holds raw little-endian bf16 (n * 2 bytes).
  void init_buffer_bf16(void* p, size_t n, const MvmInit& in) {
    switch (in.kind) {
      case MvmInit::ZERO: CK(cudaMemset(p, 0, n * 2)); break;
      case MvmInit::FILL: {
        std::vector<uint16_t> v(std::min<size_t>(n, 1 << 20), bf16_rne(in.value));
        for (size_t o = 0; o < n; o += v.size()) CK(cudaMemcpy((uint16_t*)p + o, v.data(), std::min(v.size(), n - o) * 2, cudaMemcpyHostToDevice));
        break;
      }
      case MvmInit::VALUES: {
        if (in.values.size() != n) throw std::invalid_argument("buffer init size");
        std::vector<uint16_t> v(n); for (size_t i = 0; i < n; ++i) v[i] = bf16_rne(in.values[i]);
        CK(cudaMemcpy(p, v.data(), n * 2, cudaMemcpyHostToDevice)); break;
      }
      case MvmInit::FILE: {
        FILE* f = fopen(in.path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open init file " + in.path);
        std::vector<uint16_t> v(n);
        size_t got = fread(v.data(), 2, n, f); fclose(f);
        if (got != n) throw std::runtime_error(in.path + ": expected " + std::to_string(n * 2) + " bytes");
        CK(cudaMemcpy(p, v.data(), n * 2, cudaMemcpyHostToDevice)); break;
      }
      case MvmInit::SEEDED: throw std::invalid_argument("seeded init needs an f32 buffer");
    }
  }
  void init_buffer(float* p, size_t n, const MvmInit& in, int dtype) {
    if (dtype == 2) { init_buffer_bf16(p, n, in); return; }
    switch (in.kind) {
      case MvmInit::ZERO: CK(cudaMemset(p, 0, n * 4)); break;
      case MvmInit::FILL: {
        if (dtype == 1) { std::vector<int> v(std::min<size_t>(n, 1 << 20), (int)in.value); for (size_t o = 0; o < n; o += v.size()) CK(cudaMemcpy((int*)p + o, v.data(), std::min(v.size(), n - o) * 4, cudaMemcpyHostToDevice)); }
        else { std::vector<float> v(std::min<size_t>(n, 1 << 20), in.value); for (size_t o = 0; o < n; o += v.size()) CK(cudaMemcpy(p + o, v.data(), std::min(v.size(), n - o) * 4, cudaMemcpyHostToDevice)); }
        break;
      }
      case MvmInit::VALUES:
        if (in.values.size() != n) throw std::invalid_argument("buffer init size");
        CK(cudaMemcpy(p, in.values.data(), n * 4, cudaMemcpyHostToDevice)); break;
      case MvmInit::SEEDED:
        if (dtype != 0) throw std::invalid_argument("seeded init needs an f32 buffer");
        mvm_seeded_fill_kernel<<<(unsigned)std::min<size_t>((n + 255) / 256, 4096), 256>>>(p, n, in.seed, in.draws, in.bias, in.coef);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); break;
      case MvmInit::FILE: {   // streamed through one 8 MiB staging buffer
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
    h[0] = MVM_REC_MAGIC; h[1] = (int)total; h[2] = nt; h[3] = ne; h[4] = na; h[5] = stats_.bundles; h[6] = kind; h[7] = 0;
  }
  MvmConfig cfg_;
  bool fgpt1_ = true;   // the GPT pass may pick the one-block form (set_fgpt1)
  MvmDevState S_{};
  std::vector<MvmBuf> bufs_;
  std::vector<size_t> views_;
  std::vector<float*> arenas_;
  char* ctl_ = nullptr; int* ring_ = nullptr; volatile unsigned* win_done_ = nullptr;
  unsigned long long *doorbell_ = nullptr, *ring_head_ = nullptr; unsigned* closed_ = nullptr; MvmHostErr* herr_ = nullptr;
  unsigned long long wpos_ = 0;
  cudaStream_t stream_{}; cudaEvent_t e0_{}, e1_{};
  int sms_ = 0, grid_ = 0, nc_ = 1; size_t dyn_smem_ = 0; bool hopper_ = false;
  MvmSchedParams* hp_ = nullptr; MvmSchedTele* ht_ = nullptr;
  std::string name_, last_error_;
  std::mutex mu_;
  bool prestart_ = false;   // prestart(): the ring may be written before the kernel runs
  bool unvalidated_ = false;   // a record was submitted without validation: prestage() keeps the successor copy
  int* succ_ring_ = nullptr;   // the successor ring's allocation (S_.succ points into S_.stage when aliased)
  bool started_ = false, finished_ = false, closed_flag_ = false, prepared_ = false;
  MvmResult res_;
  MvmStats stats_;
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
struct MvmFrontCfg { int chunk_tasks = 2048, ring_budget_tasks = 4096, classes = 8; std::vector<int> weights; };
class MvmFront {
 public:
  using Cfg = MvmFrontCfg;
  using Ticket = unsigned long long;
  explicit MvmFront(MvmExec& ex, Cfg cfg = Cfg()) : ex_(ex), cfg_(cfg) {
    if (cfg_.chunk_tasks < 1 || cfg_.ring_budget_tasks < cfg_.chunk_tasks || cfg_.classes < 1) throw std::invalid_argument("MvmFront: chunk_tasks >= 1, ring_budget_tasks >= chunk_tasks, classes >= 1");
    cfg_.weights.resize(cfg_.classes, 1);
    q_.resize(cfg_.classes); served_.assign(cfg_.classes, 0.0);
    pump_ = std::thread([this] { loop(); });
  }
  ~MvmFront() { stop(); }
  void stop() {
    { std::lock_guard<std::mutex> lk(mu_); if (stop_) return; stop_ = true; }
    cv_.notify_all(); if (pump_.joinable()) pump_.join();
  }
  // Queue a bundle of class `cls`; returns at once. The ticket completes when every chunk's window is done.
  Ticket submit(const MvmBundle& b, int cls) {
    if (cls < 0 || cls >= cfg_.classes) throw std::invalid_argument("MvmFront: class out of range");
    std::vector<MvmBundle> parts = split(b, cfg_.chunk_tasks);
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu_);
    const Ticket t = next_++;
    tickets_[t].pending = (int)parts.size();
    for (MvmBundle& p : parts) {
      int dl = -1; for (const MvmTaskDesc& d : p.tasks) if (d.deadline_us >= 0) dl = dl < 0 ? d.deadline_us : std::min(dl, d.deadline_us);
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
  static std::vector<MvmBundle> split(const MvmBundle& b, int limit) {
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
    std::vector<MvmBundle> out(chunk + 1); std::vector<int> newidx(nt);
    for (int i = 0; i < nt; ++i) { MvmBundle& o = out[chunk_of_comp[comp_of[i]]]; newidx[i] = (int)o.tasks.size(); o.tasks.push_back(b.tasks[i]); }
    for (int i = 0; i < nt; ++i) {
      MvmBundle& o = out[chunk_of_comp[comp_of[i]]]; MvmTaskDesc& d = o.tasks[newidx[i]];
      const MvmTaskDesc& s = b.tasks[i];
      d.succ_off = (int)o.succ.size(); for (int k = 0; k < s.succ_cnt; ++k) o.succ.push_back(newidx[b.succ[s.succ_off + k]]);
      d.args_off = (int)o.args.size(); for (int k = 0; k < s.args_words; ++k) o.args.push_back(b.args[s.args_off + k]);
    }
    return out;
  }

 private:
  struct Chunk { Ticket ticket = 0; int tasks = 0; std::chrono::steady_clock::time_point deadline; MvmBundle b; };
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
    const size_t budget_words = (size_t)cfg_.ring_budget_tasks * MVM_TASK_WORDS;
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
      unsigned wid = 0; MvmSubmit st = ex_.submit(c.b, 60000, true, &wid);
      std::lock_guard<std::mutex> lk(mu_);
      if (st == MvmSubmit::OK) { auto& t = tickets_[c.ticket]; t.wids.push_back(wid); --t.pending; served_[cls] += c.tasks; ++submitted_; }
      else { KLOG(0, "MvmFront: chunk submit failed (%s): dropping the ticket's remaining chunks", mvm_submit_name(st)); tickets_[c.ticket].pending = 1 << 30; }
      inflight_ = false;
    }
  }
  MvmExec& ex_; Cfg cfg_;
  std::mutex mu_; std::condition_variable cv_;
  std::vector<std::deque<Chunk>> q_; std::vector<double> served_;
  std::map<Ticket, TicketState> tickets_; Ticket next_ = 1;
  bool stop_ = false, inflight_ = false; std::atomic<int> submitted_{0};
  std::thread pump_;
};

// Timeline-ready CSV: times in microseconds since the first admission.
static void mvm_write_trace_csv(const MvmResult& r, const char* path) {
  FILE* f = fopen(path, "w");
  if (!f) throw std::runtime_error(std::string("cannot open ") + path);
  unsigned long long t0 = ~0ull;
  for (auto& t : r.trace) if (t.admit_ns) t0 = std::min(t0, t.admit_ns);
  fprintf(f, "slot,op,workload,priority,deadline_us,admit_us,ready_us,start_us,end_us,queue_wait_us,run_us,smid,block,ver_start,ver_end,missed,exec_count,bytes,res,resv,boost,urgent,cont"
#ifdef MVM_PROF_GAP
    ",gp0,gp1,gp2,gp3,gp4,gp5,gp6,gp7,gp8,gp_valid,gp_cycle,sub_succ_id,sub_atomic,sub_enqueue,sub_desc,sub_post,sub_errv"
#endif
    "\n");
  for (size_t i = 0; i < r.trace.size(); ++i) {
    const MvmTrace& t = r.trace[i]; const MvmTaskDesc& d = r.tasks[i];
    auto us = [&](unsigned long long x) { return x ? (double)(x - t0) / 1e3 : -1.0; };
    const MvmSlotInfo si = i < r.info.size() ? r.info[i] : MvmSlotInfo{0, 0, 0};
    fprintf(f, "%zu,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%llu,%d,%d,%d,%d,%d", i, mvm_op(d.op_id)->name, d.workload_id, d.priority, d.deadline_us,
            us(t.admit_ns), us(t.ready_ns), us(t.start_ns), us(t.end_ns), t.start_ns ? (double)(t.start_ns - t.ready_ns) / 1e3 : -1.0,
            t.end_ns ? (double)(t.end_ns - t.start_ns) / 1e3 : -1.0, t.smid, t.block, t.ver0, t.ver1, (t.flags & MVM_TRACE_MISSED) ? 1 : 0, t.exec_count,
            si.bytes, si.res, (t.flags & MVM_TRACE_RESV) ? 1 : 0, (t.flags & MVM_TRACE_BOOST) ? 1 : 0, (t.flags & MVM_TRACE_URGENT) ? 1 : 0, (t.flags & MVM_TRACE_CONT) ? 1 : 0);
#ifdef MVM_PROF_GAP
    for (int k = 0; k < 9; ++k) fprintf(f, ",%d", t.gp[k]);
    fprintf(f, ",%d,%u", t.gp[15], (unsigned)t.gp[16]);
    for (int k = 0; k < MVM_PG_SUBS; ++k) fprintf(f, ",%d", t.gp[9 + k]);
#endif
    fputc('\n', f);
  }
  fclose(f);
  KLOG(2, "trace: %zu tasks -> %s", r.trace.size(), path);
}

// Per-workload telemetry from the trace: tasks, CTA-seconds, queue-wait percentiles, deadline misses, bytes moved and bandwidth.
static void mvm_print_class_report(const MvmResult& r) {
  if (r.trace.empty() || !r.trace[0].admit_ns) return;   // a timing run keeps no per-task trace
  int maxc = 0; for (auto& d : r.tasks) maxc = std::max(maxc, d.workload_id);
  unsigned long long t0 = ~0ull, t1 = 0; for (auto& t : r.trace) if (t.admit_ns) { t0 = std::min(t0, t.admit_ns); t1 = std::max(t1, t.end_ns); }
  const double span_us = t1 > t0 ? (t1 - t0) / 1e3 : 1.0;
  for (int c = 0; c <= maxc; ++c) {
    std::vector<double> w; double cta_us = 0, bytes = 0; int n = 0, miss = 0;
    for (size_t i = 0; i < r.trace.size(); ++i) if (r.tasks[i].workload_id == c) {
      const MvmTrace& t = r.trace[i]; ++n; w.push_back((t.start_ns - t.ready_ns) / 1e3); cta_us += (t.end_ns - t.start_ns) / 1e3;
      miss += (t.flags & MVM_TRACE_MISSED) != 0; if (i < r.info.size()) bytes += (double)r.info[i].bytes;
    }
    if (!n) continue;
    std::sort(w.begin(), w.end());
    auto q = [&](double p) { return w[std::min(w.size() - 1, (size_t)(p * w.size()))]; };
    KLOG(2, "workload %d: %d tasks, %.3f CTA-s, queue wait p50 %.1f p99 %.1f max %.1f us, %d deadline misses, %.2f GB moved (%.1f GB/s over the run)",
         c, n, cta_us / 1e6, q(0.5), q(0.99), w.back(), miss, bytes / 1e9, bytes / (span_us * 1e3));
  }
}

// ======================= CLI =======================
#ifndef MVM_EXEC_NO_MAIN
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
static MvmInit mvm_parse_init(const Json& jb, const std::string& dir, int dtype) {
  MvmInit init;
  const Json* in = jb.get("init");
  if (!in) return init;
  const Json* kind = in->get("kind"); const std::string k = kind ? kind->s : "";
  if (k == "fill") { init.kind = MvmInit::FILL; init.value = (float)in->num("value", 0); }
  else if (k == "values") { init.kind = MvmInit::VALUES; for (auto& v : in->need("values").a) init.values.push_back(dtype == 1 ? [&] { int iv = (int)v.n; float f; memcpy(&f, &iv, 4); return f; }() : (float)v.n); }
  else if (k == "seeded") {
    init.kind = MvmInit::SEEDED; init.seed = in->get("seed") ? strtoull(in->get("seed")->s.c_str(), nullptr, 10) : 0;
    init.draws = (int)in->num("draws", 1); init.bias = (float)in->num("bias", 0); init.coef = (float)in->num("coef", 1);
  }
  else if (k == "file") { init.kind = MvmInit::FILE; init.path = dir + "/" + in->need("path").s; }
  else throw std::runtime_error(dir + ": unknown buffer init kind '" + k + "'");
  return init;
}

// Scheduler options shared by the CLIs (mvm_exec, doom_exec). Returns 1 if argv[i] was a scheduler option (i advanced past its value), 0 if not, -1 on a missing value.
static int mvm_cli_sched_option(MvmConfig& cfg, int& i, int argc, char** argv) {
  const std::string a = argv[i];
  auto val = [&](long long lo, long long hi) -> long long {
    if (i + 1 >= argc) return lo - 1;
    char* e; errno = 0; long long v = strtoll(argv[++i], &e, 10);
    if (e == argv[i] || *e || errno || v < lo || v > hi) { fprintf(stderr, "%s: integer in [%lld,%lld] expected, got '%s'\n", a.c_str(), lo, hi, argv[i]); exit(2); }
    return v; };
  auto list = [&](auto set) { if (i + 1 >= argc) return false; std::stringstream ss(argv[++i]); std::string tok; int c = 0; while (std::getline(ss, tok, ',') && c < MVM_MAX_CLS) set(cfg.sched.p.cls[c++], atoi(tok.c_str())); return true; };
  if (a == "--sched") { if (i + 1 >= argc) return -1; std::string m = argv[++i]; cfg.sched.on = true; cfg.sched.mode = m == "global" ? MVM_SCHED_GLOBAL : m == "static" ? MVM_SCHED_STATIC : MVM_SCHED_DYNAMIC; }
  else if (a == "--classes") { long long v = val(1, MVM_MAX_CLS); if (v < 1) return -1; cfg.sched.num_classes = (int)v; }
  else if (a == "--weights") { if (!list([](MvmClassParams& q, int v) { q.weight = v; })) return -1; }
  else if (a == "--min") { if (!list([](MvmClassParams& q, int v) { q.min_cta = v; })) return -1; }
  else if (a == "--max") { if (!list([](MvmClassParams& q, int v) { q.max_cta = v; })) return -1; }
  else if (a == "--bw-cta") { if (!list([](MvmClassParams& q, int v) { q.bw_cta = v; })) return -1; }
  else if (a == "--urgent-us") { if (!list([](MvmClassParams& q, int v) { q.urgent_us = v; })) return -1; }
  else if (a == "--op-affinity") { if (!list([](MvmClassParams& q, int v) { if (v) q.protect |= MVM_CF_OPAFF; })) return -1; }   // e.g. 1,0: class 0 prefers the block that last ran the same op
  else if (a == "--shares") {   // STATIC mode: SM share per class, e.g. 1,3
    if (i + 1 >= argc) return -1;
    std::stringstream ss(argv[++i]); std::string tok; std::vector<double> sh; while (std::getline(ss, tok, ',')) sh.push_back(atof(tok.c_str()));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); mvm_static_split(cfg.sched.p, sh, pr.multiProcessorCount);
  }
  else if (a == "--arbiter") cfg.sched.p.flags |= MVM_SF_ARBITER;
  else if (a == "--bw-limit") { long long v = val(0, 1 << 24); if (v < 0) return -1; cfg.sched.p.bw_limit_init = (int)v; }
  else if (a == "--affinity-us") { long long v = val(0, 1000000); if (v < 0) return -1; cfg.sched.p.affinity_us = (int)v; }
  else if (a == "--no-guarantee") cfg.sched.p.flags &= ~MVM_SF_GUARANTEE;
  else if (a == "--no-inherit") cfg.sched.p.flags &= ~MVM_SF_INHERIT;
  else return 0;
  return 1;
}

static void usage() {
  fprintf(stderr, "usage: mvm_exec [options] bundle_dir...\n"
    "  bundle_dir: tasks.bin successors.bin args.bin metadata.json (ABI v2)\n"
    "  --blocks N --live-cap N --max-tasks N --ring-words N --aging-us N --timeout-s N (stall timeout) --task-limit-s N\n"
    "  --sched MODE        global|static|dynamic: per-workload scheduling (workload_id = class, up to 8)  --classes N\n"
    "  --weights a,b,..    --min a,b,.. (reserved CTAs)  --max a,b,.. (CTA caps)  --bw-cta a,b,..  --urgent-us a,b,..\n"
    "  --op-affinity a,b (1 = class prefers the block that last ran the same op)  --shares a,b (static SM shares)  --arbiter  --bw-limit N   adaptive concurrent bandwidth-bound CTA limit   --affinity-us N   --no-guarantee  --no-inherit\n"

    "  --golden [--golden-tol R]  compare with <bundle_dir>/golden.json (single bundle): final buffers, versions, observed versions\n"
    "  --trace-csv FILE   per-task timeline  --dump-buffers   print buffers   --quiet   errors only   --check-ops   print registry\n"
    "  --warm-ms N   keep the GPU busy N ms right before the timed launch (the SM clock runs low under sparse work; short timed runs inherit it)\n"
    "  --full-smem   launch with the build's maximum dynamic shared memory instead of the bundles' largest task's (less L1 for every op)\n"
    "  --batch-min-runs N (the fewest grid runs of one shape that run as one batched gang a step; default 8)\n"
    "  --no-batch (no batched form: grid runs take the virtual form, else their own gangs a wave each)\n"
    "  --virtual-over-batch N (the most grid runs of one gang the virtual form takes ahead of the batched form; default 16; 0: the batched form first)\n"
    "  --no-dissolve      keep the noops that only order tasks (default: spliced out of a plain single-bundle run on the dynamic path)\n"
    "  --no-tape-sort     admit the tape as the bundle wrote it (default on the dynamic path: dependency order, which starts every replicated run at once)\n"
    "  --dump-bin DIR i,j,..  write the listed buffers (by metadata index, widened to little-endian f32 / raw i32) to DIR/<index>.bin after the run\n"
    "  --static           (default for one bundle) run the bundle as per-worker programs compiled at load: no ready queue, no admission  --workers N  --comm-us X (scheduler's cross-worker cost)  --no-partition (independent runs share all workers)  --no-fuse (no fused training steps; by default every run whose steps match is fused, --dump-buffers included: MLP steps on the static path, GPT runs on the fused GPT kernel)  --no-fuse-gpt (the GPT runs only)  --no-fgpt1 (no GPT run in the one-block form: each takes its gang)\n"
    "  --dynamic          use the admission ring and ready queue even for one bundle\n");
}

int main(int argc, char** argv) {
  MvmConfig cfg; std::vector<std::string> dirs; const char* csv = nullptr; bool dump = false, golden = false, dissolve = true, tape_sort = true; float golden_tol = 1e-6f;
  std::string dump_dir; std::vector<int> dump_idx;
  bool sized = false;   // the caller gave a sizing flag: keep its live_cap/ring/slots
  bool no_fifo = false;  // --no-fifo: keep the 32-bucket ready rings even for a bundle set that qualifies for the FIFO ring
  bool opt_static = false, opt_dynamic = false, static_spin = false, static_partition = true, opt_fuse = true, fuse_phases = false, full_smem = false; int static_workers = 0; double static_comm = 1.5;
  int batch_min_runs = 8;     // --batch-min-runs N, --no-batch (0): the batched form's threshold (two arms of one binary in a paired read)
  int virtual_over_batch = MvmFuseOptions{}.grid_virtual_over_batch;   // --virtual-over-batch N (0: the batched form first): the pass's default
  bool opt_fuse_gpt = true;   // --no-fuse-gpt: the GPT pass off, the MLP passes on (a read of an MLP lever that must leave the GPT rows on their old path)
  bool opt_fgpt1 = true;      // --no-fgpt1: the one-block form off, every fused GPT run on its gang (a read of the form against its gang)
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
#ifdef MVM_PROF_GAP
    else if (a == "--noop-ops") cfg.noop_ops = 1;
#endif
    else if (a == "--no-fifo") no_fifo = true;
    else if (a == "--timeout-s") cfg.timeout_us = (unsigned long long)val(1, 86400) * 1000000ull;
    else if (int r = mvm_cli_sched_option(cfg, i, argc, argv)) { if (r < 0) { usage(); return 2; } }
    else if (a == "--trace-csv") { if (i + 1 >= argc) { usage(); return 2; } csv = argv[++i]; }
    else if (a == "--static") opt_static = true;
    else if (a == "--dynamic") opt_dynamic = true;
    else if (a == "--workers") static_workers = (int)val(1, 1 << 20);
    else if (a == "--static-spin") static_spin = true;
    else if (a == "--no-partition") static_partition = false;
    else if (a == "--fuse") opt_fuse = true;
    else if (a == "--no-fuse") opt_fuse = false;
    else if (a == "--no-batch") batch_min_runs = 0;
    else if (a == "--batch-min-runs") batch_min_runs = (int)val(1, MVM_BATCH_MAXR);
    else if (a == "--virtual-over-batch") virtual_over_batch = (int)val(0, MVM_BATCH_MAXR);
    else if (a == "--no-fuse-gpt") opt_fuse_gpt = false;
    else if (a == "--no-fgpt1") opt_fgpt1 = false;
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
    else if (a == "--quiet") g_mvm_log = 0;
    else if (a == "--no-dissolve") dissolve = false;
    else if (a == "--no-tape-sort") tape_sort = false;
    else if (a == "--check-ops") { for (auto& o : kMvmOps) printf("%2d %-9s arity %d  (%s)\n", o.id, o.name, o.arity, o.sig); return 0; }
    else if (a[0] == '-') { usage(); return 2; }
    else dirs.push_back(a);
  }
  if (dirs.empty()) { usage(); return 2; }
  try {
    std::vector<MvmBundle> bundles;
    std::vector<std::pair<int, int>> shapes;
    struct BufSpec { int rows, cols, dtype; MvmInit init; };
    std::vector<BufSpec> buf_specs;   // added once the executor exists: its pools are sized from the bundles as they will be admitted (below)
    std::vector<std::string> buf_names, var_names; std::map<int, std::string> var_by_id;
    std::map<int, int> version_base;   // per variable: its base version (metadata), the static path's feasibility check
    Json meta0;
    for (auto& dir : dirs) {
      Json m = JsonParser(read_file(dir + "/metadata.json")).parse();
      if ((int)m.num("abi_version", -1) != MVM_ABI_VERSION) throw std::runtime_error(dir + ": abi_version " + std::to_string((int)m.num("abi_version", -1)) + " != " + std::to_string(MVM_ABI_VERSION));
      int nt = (int)m.num("task_count", -1), ne = (int)m.num("edge_count", -1), nw = (int)m.num("args_words", -1);
      if (nt < 1 || ne < 0 || nw < 0) throw std::runtime_error(dir + ": task_count/edge_count/args_words missing");
      MvmBundle b;
      b.tasks = read_bin<MvmTaskDesc>(dir + "/tasks.bin", nt);
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
          buf_specs.push_back({rows, cols, dtype, mvm_parse_init(jb, dir, dtype)});
          shapes.push_back({rows, cols});
        }
      }
      if (auto* vs = m.get("variables")) for (auto& jv : vs->a) { int id = (int)jv.num("id", -1), base = (int)jv.num("base_version", 0); version_base[id] = base; }
      KLOG(2, "%s: %d tasks, %d edges, %d arg words", dir.c_str(), nt, ne, nw);
      for (const auto& t : b.tasks) {   // a -DMVM_OPS_ONLY build holds only its listed ops: refuse the bundle here, not with a device error mid-run
        const MvmOpInfo* o = mvm_op(t.op_id);
        if (o && !mvm_op_built(o->name)) throw std::runtime_error(dir + ": op " + o->name + " is not in this build's MVM_OPS_ONLY list");
      }
      bundles.push_back(std::move(b));
    }
    // The static schedule is the default for one closed bundle (a compiled DAG, no streaming, no scheduling classes): the ring path remains for several bundles, the per-class scheduler, or an explicit --dynamic.
    if (opt_static && opt_dynamic) throw std::runtime_error("--static and --dynamic exclude each other");
    // A bundle can run on the static path only if everything it waits for is ordered inside it: a wait on a version no task of the bundle publishes
    // (and its base version does not cover) needs the dynamic path's parking and an external publisher.
    auto static_feasible = [&](const MvmBundle& b) {
      for (const MvmTaskDesc& d : b.tasks) {   // every buffer a task names must fit its record (the planner would refuse otherwise)
        const MvmOpInfo* o = mvm_op(d.op_id);
        if (!o || d.args_words > MVM_S_ARGS) return false;
        std::set<int> seen;
        for (int j = 0; j < d.args_words; ++j) { const int ty = (int)((o->types >> (3 * j)) & 7u); if ((ty == MVM_A_BUF || ty == MVM_A_BUF_OPT) && b.args[d.args_off + j] != -1) seen.insert(b.args[d.args_off + j]); }
        if (seen.size() > MVM_S_BUFS) return false;
      }
      std::map<int, int> newest;
      for (const MvmTaskDesc& d : b.tasks) if (d.pub_var >= 0) newest[d.pub_var] = std::max(newest[d.pub_var], d.pub_ver);
      for (const MvmTaskDesc& d : b.tasks) if (d.wait_var >= 0) {
        auto it = newest.find(d.wait_var);
        if (it == newest.end() ? d.wait_ver > version_base[d.wait_var] : d.wait_ver > it->second) return false;
      }
      return true;
    };
    // The static path pays where tasks are short. A precompiled order saves the dynamic path's per-task scheduling (claim, admission, successor counts: about 1-2 us of an SM per task) and
    // gives up its ready queue's balancing, which a plan from the cost model cannot replace when tasks are long and the model is far off (measured: gpt-162m's ~80 us tiles ran x0.70 static).
    // Decided from the bundle alone: the planner's own mean estimated cost per task against MVM_STATIC_MAX_MEAN_US, the crossover measured on bundles outside the panel.
    auto static_mean_us = [&](const MvmBundle& b) {
      double work = 0.0;
      for (const MvmTaskDesc& d : b.tasks) work += mvm_static_cost(d.op_id, b.args.data() + d.args_off);
      return b.tasks.empty() ? 0.0 : work / (double)b.tasks.size();
    };
    // The rule decides only what its cost model prices: an op outside mvm_static_costs() would enter the mean at a flat guess no measurement backs (a bf16 GEMM of
    // ~130 us priced at 4), so such a bundle runs dynamic, the path that needs no cost model and the one every single bundle took before the static path existed.
    auto unpriced_ops = [&](const MvmBundle& b) {
      std::set<std::string> names;
      for (const MvmTaskDesc& d : b.tasks) {
        const MvmOpInfo* o = mvm_op(d.op_id);
        if (!o) names.insert("op " + std::to_string(d.op_id));
        else if (!mvm_static_costs().count(o->name)) names.insert(o->name);
      }
      std::string list;
      for (const std::string& n : names) list += (list.empty() ? "" : ", ") + n;
      return list;
    };
    // A gang op's members meet at barriers inside the op, so they must run at once: only the static plan places them so (mvm_static_plan).
    auto gang_task = [&](const MvmBundle& b) {
      for (size_t t = 0; t < b.tasks.size(); ++t) {
        const MvmOpInfo* o = mvm_op(b.tasks[t].op_id);
        MvmSOpInfo g;
        if (o) mvm_static_gang_args(o->sig, g);
        if (g.gang_arg >= 0) return (int)t;
      }
      return -1;
    };
    // The fused GPT path (MvmExec::run_fgpt): a bundle whose every task belongs to a GPT training run that fuses into one gang (mvm_fuse_gpt.h).
    // A gang's members must be resident at once, one block an SM at the shared memory they take: at most the SM count, or --workers.
    // Decided before the executor is built (it is built once the bundle is admitted and its pools sized, below), so on the device's SM count
    // and on the buffers as add_buffer will lay them out (ld = cols); the pass that applies the decision runs on the executor's own buffers.
    bool opt_fgpt = false;
    const int sms = MvmExec::device_sms();
    const int fgpt_workers = static_workers ? std::min(static_workers, sms) : sms;
    std::vector<MvmFuseBuf> spec_bufs;
    for (const BufSpec& b : buf_specs) spec_bufs.push_back(MvmFuseBuf{b.rows, b.cols, b.cols, b.dtype});
    const auto fgpt_whole = [](const MvmFuseGptPlan& g) { return g.fused_runs > 0 && g.unchanged_runs == 0; };
    if (!opt_dynamic && !opt_static && bundles.size() == 1 && !cfg.sched.on && static_feasible(bundles[0])) {
      const std::string unpriced = unpriced_ops(bundles[0]);
      const MvmFuseGptPlan gpt_dry = opt_fuse && opt_fuse_gpt ? MvmExec::plan_fgpt(bundles[0], spec_bufs, fgpt_workers, dump_idx, false, opt_fgpt1) : MvmFuseGptPlan{};   // a dry run of the pass the branch applies
      if (gpt_dry.fused_runs > 0 && !fgpt_whole(gpt_dry))
        KLOG(1, "fuse gpt: %d GPT run(s) would fuse, but %d other component(s) do not; the fused GPT kernel runs a bundle of gangs alone, so this bundle runs unfused", gpt_dry.fused_runs, gpt_dry.unchanged_runs);
      if (gang_task(bundles[0]) >= 0) {
        opt_static = true;
        KLOG(2, "path: static (the bundle has gang ops, which only the static plan runs)");
      } else if (fgpt_whole(gpt_dry)) {
        opt_fgpt = true;
        KLOG(2, "path: fused GPT kernel (every task belongs to a GPT training run that fuses into one gang)");
      } else if (!unpriced.empty()) {
        KLOG(2, "path: dynamic (the static cost model does not price %s)", unpriced.c_str());
      } else {
        const double mean = static_mean_us(bundles[0]);
        opt_static = mean <= MVM_STATIC_MAX_MEAN_US;
        KLOG(2, "path: %s (mean estimated task %.2f us %s %.2f)", opt_static ? "static" : "dynamic", mean, opt_static ? "<=" : ">", (double)MVM_STATIC_MAX_MEAN_US);
      }
    }
    if (opt_static && !static_feasible(bundles[0])) throw std::runtime_error("--static: a task waits for a version no task of this bundle publishes");
    // The dynamic path's host transforms come before the window is sized: the splice adds up to fan-in x fan-out edges a noop, so a window sized from the bundle as
    // written can be too small for the bundle as admitted (Nov 11: h640-d4-r16, 307,200 tasks, ran as windows and overran its successor ring). The dynamic path only:
    // the static planner and the fused GPT pass read the bundle as written (the pass below must match its dry run above).
    if (!opt_static && !opt_fgpt) {
      if (dissolve && !golden && !csv && dirs.size() == 1 && !cfg.sched.on) {   // a trace's slots are the bundle's task indices, so a traced run keeps its noops
        const int nd = mvm_dissolve_noops(bundles[0]);
        if (nd) KLOG(2, "%d noops dissolved: %zu tasks, %zu edges", nd, bundles[0].tasks.size(), bundles[0].succ.size());
      }
      if (tape_sort && !golden && !csv && bundles.size() == 1 && mvm_sort_tape(bundles[0])) KLOG(2, "tape in dependency order");
    }
    // Window sizing: the default live_cap / ring cut a large bundle into windows, and the host submits a window only after the windows holding its producers
    // completed (a drain per window). When no sizing flag was given and a single bundle would be cut by any capacity (its tasks, or its words: a 9,800-task
    // bundle of 4 replicated runs ran its runs one after another in 10 windows, x4 slower), size them so the bundle, as it will be admitted, is one record.
    // A sizing flag keeps its window (the litmus fuzzer and the sanitizer jobs set small ones to exercise the windowed path); the successor ring's floor
    // holds either way (MvmExec).
    if (!sized && bundles.size() == 1) {
      const long long nt = (long long)bundles[0].tasks.size(), ne = (long long)bundles[0].succ.size(), nw = (long long)bundles[0].args.size();
      if (cfg.size_for_one_record(nt, ne, nw))
        KLOG(2, "window sizing: %lld tasks, %lld words as one record (live_cap %d, ring %d words, successor ring %u words)", nt, MvmConfig::record_words(nt, ne, nw),
             cfg.live_cap, cfg.ring_words, cfg.succ_cap);
    }
    MvmExec ex(cfg);
    ex.set_fgpt1(opt_fgpt1);
    for (size_t i = 0; i < buf_specs.size(); ++i)
      if (ex.add_buffer(buf_specs[i].rows, buf_specs[i].cols, buf_specs[i].init, buf_specs[i].dtype) != (int)i) throw std::logic_error("buffer index mismatch");
    for (const auto& vb : version_base) ex.set_version(vb.first, vb.second);
    if (!full_smem) ex.size_launch(bundles);   // every task of the run is known here, before any launch
    const MvmResult* resp = nullptr;
    MvmFuseResult fused;            // outside the branch: the golden check below skips what the fused runs no longer write
    if (opt_fgpt) {
      const MvmFuseGptPlan gpt = ex.fuse_gpt(bundles[0], fgpt_workers, dump_idx);
      if (!fgpt_whole(gpt)) throw std::logic_error("fuse gpt: the pass fused " + std::to_string(gpt.fused_runs) + " run(s) and left " + std::to_string(gpt.unchanged_runs) + ", unlike its dry run");
      bundles[0].tasks = gpt.tasks; bundles[0].succ = gpt.succ; bundles[0].args = gpt.args;
      for (const MvmFuseGptRun& g : gpt.runs) {   // what the gang no longer writes: the step's activations and gradients, and the windows' copies
        fused.interior.insert(g.shape.interior.begin(), g.shape.interior.end());
        fused.interior.insert({g.shape.ids, g.shape.tgt, g.shape.coef});
      }
      if (csv) { KLOG(1, "trace: the fused GPT kernel keeps no per-task trace, so %s is not written", csv); csv = nullptr; }
      resp = &ex.run_fgpt(bundles[0], gpt);
    } else if (opt_static) {
      if (bundles.size() != 1) throw std::runtime_error("--static takes exactly one bundle directory");
      MvmExec::StaticOptions so; so.workers = static_workers ? static_workers : cfg.blocks; so.comm_us = static_comm; so.trace_csv = csv; csv = nullptr; so.spin_idle = static_spin; so.partition = static_partition; so.fuse_phases = fuse_phases; so.fuse_phases = fuse_phases;
      if (opt_fuse) {
        // the pass sizes slices and owners for the workers the schedule will have (run_static's own count: --workers, at most the grid; the grid when unset). Measured: given 0, it took the
        // widest slice that fits, so a single run chose 32-row slices and the same run inside a merged bundle 16, and their rounding differed.
        const int workers = so.workers ? std::min(so.workers, ex.grid()) : ex.grid();
        fused = ex.fuse_bundle(bundles[0], workers, dump_idx, batch_min_runs, virtual_over_batch);   // --dump-buffers keeps the fused path: a gate must read the path that is timed; the buffers no fused run writes dump as they started
        if (fused.fused_runs) { bundles[0].tasks = fused.tasks; bundles[0].succ = fused.succ; bundles[0].args = fused.args; so.fused = &fused.runs; so.batches = &fused.batches; }
        so.entry = fused.entry;
      }
      resp = &ex.run_static(bundles[0], so);
    } else {
      for (const MvmBundle& b : bundles) {
        const int t = gang_task(b);
        if (t >= 0) throw std::runtime_error("task " + std::to_string(t) + ": " + mvm_op(b.tasks[t].op_id)->name + " is a gang op, whose members must run at once; only the static path plans that (one bundle, no --dynamic, no scheduling classes)");
      }
      bool one_record = false;
      {
        bool fifo = !no_fifo && !cfg.sched.on, no_waits = true;
        for (const MvmBundle& b : bundles) for (const MvmTaskDesc& t : b.tasks) { if (t.priority != 0 || t.deadline_us != -1) fifo = false; if (t.wait_var >= 0) no_waits = false; }
        one_record = bundles.size() == 1 && cfg.fits_one_record((long long)bundles[0].tasks.size(), (long long)bundles[0].words());
        const bool trace_on = csv != nullptr || golden || getenv("MVM_TRACE") != nullptr;
        ex.set_run_facts(fifo, one_record, no_waits, trace_on);
        KLOG(2, "ready queue: %s; slots %s; version waits %s; task trace %s", ex.fifo_only() ? "one FIFO ring, ticket waits" : "32 priority buckets", one_record ? "not recycled (one record)" : "recycled", no_waits ? "none" : "present", (trace_on || cfg.sched.on) ? "on" : "off");
      }
      if (one_record) {                   // the whole input is one record: written and validated before the kernel starts, so no host work sits inside the kernel's timed window
        ex.prestart();
        const MvmSubmit st = ex.submit(bundles[0]);
        if (st != MvmSubmit::OK) { KLOG(0, "bundle 0: %s (%s)", mvm_submit_name(st), ex.last_error().c_str()); return 1; }
        ex.close();
        ex.prestage();
        ex.start();
      } else {
        ex.start();
        for (size_t i = 0; i < bundles.size(); ++i) {
          const MvmBundle& bi = bundles[i];
          MvmSubmit st = (bi.tasks.size() > (size_t)cfg.live_cap || bi.words() > (size_t)cfg.ring_words / 2) ? ex.submit_stream(bi) : ex.submit(bi);
          if (st != MvmSubmit::OK) { KLOG(0, "bundle %zu: %s (%s)", i, mvm_submit_name(st), ex.last_error().c_str()); ex.close(); ex.finish(); return 1; }
        }
      }
      resp = &ex.finish();
    }
    const MvmResult& r = *resp;
    if (r.ok() && !opt_static && !opt_fgpt) mvm_print_class_report(r);
    if (csv) mvm_write_trace_csv(r, csv);
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
      if (const Json* rs = meta0.get("reads")) if (!opt_static && !opt_fgpt) for (auto& jr : rs->a) {   // the per-read version checks need the per-task trace (the dynamic path's)
        const Json* gate = jr.get("gate");
        int slot = (int)jr.num("task", -1); if (gate && gate->t == Json::N) slot = (int)gate->n;  // the task carrying the wait
        const Json* pol = jr.get("policy"); int need = (int)jr.num("version", 0);
        if (slot < 0 || slot >= (int)r.trace.size() || !pol) throw std::runtime_error("metadata: malformed reads[] entry (task index out of range or no policy)");
        const MvmTrace& t = r.trace[slot]; ++nreads;
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
