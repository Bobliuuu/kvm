#pragma once
// KVM static schedules, host side: compile a bundle whose task DAG is known before launch into per-worker programs.
//
// ===== Why =====
// The dynamic executor pays for dependencies at run time: an atomic indegree per edge, a ready-queue enqueue and pop per task, and every
// hop between tasks goes through global memory. A DAG known before launch needs none of that. Here each worker (one persistent block of
// KVM_THREADS threads) runs a program computed ahead of time, in the spirit of Hazy's Megakernels (per-SM instruction queues built by a
// scheduler before launch):
//   * the program of a worker is a list of self-contained records (op, typed args, resolved buffer views, the cross-worker waits);
//   * a dependency between two tasks of ONE worker is its program order: a __syncthreads() between two records, nothing else;
//   * a dependency across workers is a wait on the other worker's completion watermark: `progress[w] >= n` means the first n records of w
//     have finished and their stores are visible GPU-wide (published by a fence and one release store, no atomic). Only the record that
//     another worker actually waits for publishes.
// The scheduler (HEFT list scheduling) keeps long chains on one worker, so that the critical path pays barriers instead of hops.
//
// This header is host-only (no CUDA): tests/static exercises it with plain g++. Device side: kvm_static.cuh. Interfaces with the rest of the
// tree only through the bundle ABI (kvm_task_abi.h) and an operator signature view.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "kvm_task_abi.h"

#define KVM_S_WAITS 16  // cross-worker waits a record holds inline (polled in parallel by the lanes of warp 0: a join of 16 costs one poll); more go to wait-only records in front of it
#define KVM_S_ARGS 20   // argument words per record (the registry's KVM_MAX_ARGS)
#define KVM_S_BUFS 8    // buffer views per record (the largest op, attn_bwd_dkv, has 8 buffer arguments)
#define KVM_S_REC_BYTES 512
static_assert(KVM_S_REC_BYTES % 16 == 0, "records are fetched in 16-byte chunks");

// Record flags.
#define KVM_S_PUBLISH 1   // after the op: fence, then progress[this worker] = record index + 1
#define KVM_S_WAITONLY 2  // no op: only the waits (the first waits of a task with more than KVM_S_WAITS)

struct KvmSWait { int32_t worker, need; };   // progress[worker] >= need

// One instruction. Self-contained, so a worker reads nothing else from global memory for it: the descriptor, the argument words and
// the buffer views it needs are all here (fetched together, one cp.async per lane of warp 0).
struct alignas(16) KvmSRec {
    int32_t op, flags, nwait, pos1;      // pos1 = this record's index in its worker's program + 1 (the value it publishes)
    int32_t task, nbufs, pad0, pad1;     // task id (errors, trace); buffer views filled below
    KvmSWait waits[KVM_S_WAITS];
    int32_t args[KVM_S_ARGS];            // buffer arguments are rewritten to indices into bufs[] below (-1 = absent stays -1)
    KvmBuf bufs[KVM_S_BUFS];
    unsigned char pad[KVM_S_REC_BYTES - 8 * 4 - KVM_S_WAITS * 8 - KVM_S_ARGS * 4 - KVM_S_BUFS * sizeof(KvmBuf)];
};
static_assert(sizeof(KvmSRec) == KVM_S_REC_BYTES, "KvmSRec layout");

// What the scheduler needs to know about an operator: its registry view (arity, 3 bits per argument type) and a cost estimate.
struct KvmSOpInfo { int arity = -1; uint64_t types = 0; };   // arity < 0: unknown op
using KvmSOpLookup = std::function<KvmSOpInfo(int op_id)>;
using KvmSCostFn = std::function<double(int op_id, const int32_t* args)>;   // microseconds (only relative weights matter)

struct KvmSConfig {
    int workers = 1;
    double comm_us = 1.5;            // the scheduler's cost of a cross-worker edge (fence, release, polled acquire)
    double min_cost_us = 0.05;       // every task costs at least this (keeps ranks strictly topological)
    bool partition = true;           // independent components (replicated runs) get disjoint workers
};

// The per-worker programs: task ids in execution order, plus the reduced cross-worker waits of every task.
struct KvmSPlan {
    int workers = 0, nt = 0;
    bool partitioned = false;                            // independent runs were given disjoint workers (the rule is in kvm_static_plan)
    std::vector<std::vector<int>> prog;                  // per worker: task ids in order
    std::vector<int> worker_of, order_of;                // per task: worker, position among that worker's tasks
    std::vector<std::vector<KvmSWait>> waits;            // per task: (worker, need in *task count*): the record indices are assigned at emission
    std::vector<char> publishes;                         // per task: some record of another worker waits for it
    // statistics
    int cross_edges = 0, edges = 0, publishing = 0;
    double makespan_us = 0;                              // predicted by the list scheduler
    double total_work_us = 0;
};

namespace kvm_static {

// Edges of the full dependency DAG: the successor CSR plus the version-variable orderings (a waiter runs after the in-bundle publishers of
// the versions it needs; publishers of one variable are totally ordered by version number, as the device enforces at run time).
inline std::vector<std::vector<int>> build_dag(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& succ) {
    const int nt = (int)tasks.size();
    std::vector<std::vector<int>> out(nt);
    for (int i = 0; i < nt; ++i)
        for (int k = 0; k < tasks[i].succ_cnt; ++k) {
            const int s = succ[tasks[i].succ_off + k];
            if (s < 0 || s >= nt || s == i) throw std::invalid_argument("task " + std::to_string(i) + ": successor out of range");
            out[i].push_back(s);
        }
    std::map<int, std::vector<std::pair<int, int>>> pubs;   // var -> (version, task)
    for (int i = 0; i < nt; ++i)
        if (tasks[i].pub_var >= 0) pubs[tasks[i].pub_var].push_back({tasks[i].pub_ver, i});
    for (auto& kv : pubs) {
        auto& v = kv.second;
        std::sort(v.begin(), v.end());
        for (size_t j = 1; j < v.size(); ++j) {
            if (v[j].first == v[j - 1].first) throw std::invalid_argument("two publishers of one version of variable " + std::to_string(kv.first));
            out[v[j - 1].second].push_back(v[j].second);    // publisher of N-1 before publisher of N
        }
    }
    for (int i = 0; i < nt; ++i) {
        if (tasks[i].wait_var < 0) continue;
        auto it = pubs.find(tasks[i].wait_var);
        if (it == pubs.end()) continue;   // a version from before this bundle (base_version) or another workload: nothing inside the bundle orders it
        // the newest in-bundle publisher this waiter needs: its version <= wait_ver; older ones precede it transitively
        const auto& v = it->second;
        auto u = std::upper_bound(v.begin(), v.end(), std::make_pair(tasks[i].wait_ver, std::numeric_limits<int>::max()));
        if (u == v.begin()) continue;
        --u;
        if (u->second == i) throw std::invalid_argument("task " + std::to_string(i) + " waits for a version it publishes itself");
        out[u->second].push_back(i);
    }
    for (auto& e : out) { std::sort(e.begin(), e.end()); e.erase(std::unique(e.begin(), e.end()), e.end()); }
    return out;
}

}  // namespace kvm_static

// List scheduling (HEFT): tasks in decreasing upward rank (longest remaining path, cross-worker edges costed); each goes to the worker
// where it finishes first; a predecessor on the same worker costs no communication. Worker programs are appended in rank order, so a
// worker's order is consistent with the DAG (a later record never has to precede an earlier one) and no cycle of waits can arise.
inline KvmSPlan kvm_static_plan(const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& succ, const std::vector<int32_t>& args,
                                const KvmSConfig& cfg, const KvmSOpLookup& op_of, const KvmSCostFn& cost_of) {
    const int nt = (int)tasks.size();
    if (nt < 1 || cfg.workers < 1) throw std::invalid_argument("empty bundle or no workers");
    std::vector<std::vector<int>> out = kvm_static::build_dag(tasks, succ);
    std::vector<int> indeg(nt, 0);
    for (int i = 0; i < nt; ++i) for (int j : out[i]) ++indeg[j];
    std::vector<int> topo; topo.reserve(nt);
    { std::vector<int> st; for (int i = 0; i < nt; ++i) if (!indeg[i]) st.push_back(i);
      std::vector<int> deg = indeg;
      while (!st.empty()) { int u = st.back(); st.pop_back(); topo.push_back(u); for (int v : out[u]) if (--deg[v] == 0) st.push_back(v); } }
    if ((int)topo.size() != nt) throw std::invalid_argument("dependency cycle");
    std::vector<double> cost(nt);
    for (int i = 0; i < nt; ++i) {
        const KvmSOpInfo info = op_of(tasks[i].op_id);
        if (info.arity < 0 || tasks[i].args_words != info.arity) throw std::invalid_argument("task " + std::to_string(i) + ": unknown op or wrong arity");
        cost[i] = std::max(cfg.min_cost_us, cost_of(tasks[i].op_id, args.data() + tasks[i].args_off));
    }
    std::vector<double> rank(nt, 0.0);
    for (int k = nt - 1; k >= 0; --k) {   // reverse topological order
        const int u = topo[k];
        double m = 0;
        for (int v : out[u]) m = std::max(m, cfg.comm_us + rank[v]);
        rank[u] = cost[u] + m;
    }
    std::vector<int> order(nt);
    for (int i = 0; i < nt; ++i) order[i] = i;
    // ties broken by task id: reproducible schedules (and ids follow the bundle's program order)
    std::sort(order.begin(), order.end(), [&](int a, int b) { return rank[a] != rank[b] ? rank[a] > rank[b] : a < b; });

    // Independent runs of a replicated bundle (connected components of the DAG): each gets its own workers, in proportion to its work, so no run's chain queues behind
    // another run's tasks on a shared worker (R=32 on the panel). A single component uses every worker, exactly as before.
    std::vector<std::vector<int>> group_workers;   // per component: the workers it may use
    std::vector<int> comp_of(nt, -1);
    bool partitioned = false;
    {
        std::vector<int> uf(nt); for (int i = 0; i < nt; ++i) uf[i] = i;
        std::function<int(int)> find = [&](int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
        for (int i = 0; i < nt; ++i) for (int j : out[i]) uf[find(i)] = find(j);
        std::map<int, int> ids; std::vector<double> work;
        for (int i = 0; i < nt; ++i) { auto it = ids.emplace(find(i), (int)work.size()); if (it.second) work.push_back(0.0); comp_of[i] = it.first->second; work[comp_of[i]] += cost[i]; }
        const int nc = (int)work.size();
        group_workers.assign(nc, {});
        if (nc == 1 || !cfg.partition) { group_workers.assign(nc, {}); for (int c = 0; c < nc; ++c) for (int w = 0; w < cfg.workers; ++w) group_workers[c].push_back(w); }
        else {
            // largest-remainder share of the workers per component (every component gets at least one; beyond W components, workers are shared)
            double tot = 0; for (double x : work) tot += x;
            std::vector<int> share(nc, 0); std::vector<std::pair<double, int>> rem;
            int given = 0;
            for (int c = 0; c < nc; ++c) { const double ideal = work[c] / tot * cfg.workers; share[c] = std::max(1, (int)ideal); given += share[c]; rem.push_back({ideal - (int)ideal, c}); }
            std::sort(rem.rbegin(), rem.rend());
            for (size_t k = 0; given < cfg.workers; ++k, ++given) share[rem[k % rem.size()].second]++;
            // A run given fewer workers than it can keep busy queues behind its own tasks, so the partition holds only where each run has at least twice its average parallelism
            // (work / longest path; measured: 8 runs with 13 workers each gained 10%, 32 runs with 3 each lost 7% to one shared pool).
            std::vector<double> span(nc, 0.0);
            for (int i = 0; i < nt; ++i) span[comp_of[i]] = std::max(span[comp_of[i]], rank[i]);
            partitioned = true;
            for (int c = 0; c < nc; ++c) if (share[c] < 2.0 * work[c] / span[c]) partitioned = false;
            if (partitioned) { int w = 0; for (int c = 0; c < nc; ++c) for (int k = 0; k < share[c]; ++k) group_workers[c].push_back(w++ % cfg.workers); }
            else for (int c = 0; c < nc; ++c) for (int w = 0; w < cfg.workers; ++w) group_workers[c].push_back(w);
        }
    }

    KvmSPlan p;
    p.workers = cfg.workers; p.nt = nt; p.partitioned = partitioned;
    p.prog.assign(cfg.workers, {});
    p.worker_of.assign(nt, -1); p.order_of.assign(nt, -1);
    p.waits.assign(nt, {}); p.publishes.assign(nt, 0);
    std::vector<std::vector<int>> in(nt);
    for (int i = 0; i < nt; ++i) for (int j : out[i]) in[j].push_back(i);
    std::vector<double> avail(cfg.workers, 0.0), finish(nt, 0.0), local(cfg.workers, 0.0);
    std::vector<int> touched;
    for (int t : order) {
        // arrival of every predecessor's data if t ran on some other worker than that predecessor's: best and second best, per distinct source worker
        double first = 0, second = 0; int first_w = -1;
        touched.clear();
        for (int q : in[t]) {
            const double arrive = finish[q] + cfg.comm_us;
            const int w = p.worker_of[q];
            if (w == first_w) first = std::max(first, arrive);
            else if (arrive > first) { second = std::max(second, first); first = arrive; first_w = w; }
            else second = std::max(second, arrive);
            if (local[w] == 0.0) touched.push_back(w);
            local[w] = std::max(local[w], finish[q]);
        }
        double best = 0; int bw = -1;
        for (int w : group_workers[comp_of[t]]) {
            const double ready = std::max({avail[w], local[w], (w == first_w ? second : first)});
            const double end = ready + cost[t];
            if (bw < 0 || end < best) { best = end; bw = w; }
        }
        for (int w : touched) local[w] = 0.0;
        finish[t] = best; avail[bw] = best;
        p.worker_of[t] = bw; p.order_of[t] = (int)p.prog[bw].size();
        p.prog[bw].push_back(t);
        p.makespan_us = std::max(p.makespan_us, best);
    }
    for (int i = 0; i < nt; ++i) p.total_work_us += cost[i];
    // Reduced cross-worker waits: one per (task, source worker), on the last predecessor in that worker's order; the need is in task counts here and
    // becomes a record count at emission.
    std::vector<int> stamp(cfg.workers, -1), last(cfg.workers, -1);
    for (int v = 0; v < nt; ++v) {
        const int wv = p.worker_of[v];
        touched.clear();
        for (int q : in[v]) {
            ++p.edges;
            const int wq = p.worker_of[q];
            if (wq == wv) continue;
            ++p.cross_edges;
            if (stamp[wq] != v) { stamp[wq] = v; last[wq] = p.order_of[q]; touched.push_back(wq); }
            else last[wq] = std::max(last[wq], p.order_of[q]);
        }
        for (int w : touched) p.waits[v].push_back({w, last[w]});   // task count position of the last predecessor on w (rewritten at emission)
    }
    // A wait an earlier record of the same worker already satisfied is redundant (the watermark only grows): walk each worker's program in order, keep only
    // the waits that raise what it has already waited for on that source. Only the waited-for tasks publish. On wide levels this is the difference between waiting once per worker and once per task.
    std::fill(p.publishes.begin(), p.publishes.end(), 0);
    for (int w = 0; w < cfg.workers; ++w) {
        std::vector<int> seen(cfg.workers, -1);
        for (int t : p.prog[w]) {
            auto& wt = p.waits[t];
            size_t out = 0;
            for (const KvmSWait& x : wt) if (x.need > seen[x.worker]) wt[out++] = x;
            wt.resize(out);
            for (const KvmSWait& x : wt) { seen[x.worker] = std::max(seen[x.worker], x.need); p.publishes[p.prog[x.worker][x.need]] = 1; }
        }
    }
    for (int i = 0; i < nt; ++i) p.publishing += p.publishes[i];
    return p;
}

// The record count of each task's program point on its worker: wait-only records (for waits past the KVM_S_WAITS that fit in the task's own
// record) sit in front of the task, so a record index differs from a task's position. Returns per worker the task -> record-index mapping and each worker's record count.
struct KvmSLayout { std::vector<int> rec_of; std::vector<int> nrecs; };   // per task, per worker
inline KvmSLayout kvm_static_layout(const KvmSPlan& p) {
    KvmSLayout l; l.rec_of.assign(p.nt, -1); l.nrecs.assign(p.workers, 0);
    for (int w = 0; w < p.workers; ++w) {
        int r = 0;
        for (int t : p.prog[w]) {
            const int nw = (int)p.waits[t].size();
            r += nw > KVM_S_WAITS ? (nw - 1) / KVM_S_WAITS : 0;   // wait-only records in front of this task
            l.rec_of[t] = r++;
        }
        l.nrecs[w] = r;
    }
    return l;
}

// Fill the record stream of one worker. `bufs` maps a buffer index to its device view. Records of different workers are independent: each worker's stream is contiguous.
inline void kvm_static_emit(const KvmSPlan& p, const KvmSLayout& l, int w, const std::vector<KvmTaskDesc>& tasks, const std::vector<int32_t>& args,
                            const KvmBuf* bufs, int nbufs, const KvmSOpLookup& op_of, KvmSRec* out) {
    int r = 0;
    for (int t : p.prog[w]) {
        const KvmTaskDesc& d = tasks[t];
        // needs in record counts: the source task's record index + 1 on its worker
        std::vector<KvmSWait> waits;
        for (const KvmSWait& x : p.waits[t]) waits.push_back({x.worker, l.rec_of[p.prog[x.worker][x.need]] + 1});
        size_t wi = 0;
        // waits beyond what fits in the task's own record go to wait-only records ahead of it
        while (waits.size() - wi > KVM_S_WAITS) {
            KvmSRec& wo = out[r];
            memset((void*)&wo, 0, sizeof wo);
            wo.op = 0; wo.flags = KVM_S_WAITONLY; wo.task = t; wo.pos1 = r + 1; wo.nbufs = 0;
            for (int j = 0; j < KVM_S_WAITS; ++j) wo.waits[j] = waits[wi + j];
            wo.nwait = KVM_S_WAITS; wi += KVM_S_WAITS; ++r;
        }
        KvmSRec& rec = out[r];
        memset((void*)&rec, 0, sizeof rec);
        rec.op = d.op_id; rec.task = t; rec.pos1 = r + 1;
        rec.flags = p.publishes[t] ? KVM_S_PUBLISH : 0;
        rec.nwait = (int32_t)(waits.size() - wi);
        for (int j = 0; j < rec.nwait; ++j) rec.waits[j] = waits[wi + j];
        if (d.args_words > KVM_S_ARGS) throw std::invalid_argument("task " + std::to_string(t) + ": more argument words than a record holds");
        const KvmSOpInfo info = op_of(d.op_id);
        int nb = 0, gidx[KVM_S_BUFS];
        for (int j = 0; j < d.args_words; ++j) {
            int32_t v = args[d.args_off + j];
            const int ty = (int)((info.types >> (3 * j)) & 7u);
            if ((ty == 0 /*BUF*/ || ty == 3 /*BUF_OPT*/) && v != -1) {
                if (v < 0 || v >= nbufs) throw std::invalid_argument("task " + std::to_string(t) + ": buffer index out of range");
                int slot = -1;
                for (int k = 0; k < nb; ++k) if (gidx[k] == v) { slot = k; break; }
                if (slot < 0) {
                    if (nb >= KVM_S_BUFS) throw std::invalid_argument("task " + std::to_string(t) + ": more buffers than a record holds");
                    gidx[nb] = v; rec.bufs[nb] = bufs[v]; slot = nb++;
                }
                v = slot;
            }
            rec.args[j] = v;
        }
        rec.nbufs = nb;
        if (r + 1 != l.rec_of[t] + 1) throw std::logic_error("record numbering");
        ++r;
    }
}
